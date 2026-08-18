/* GPAP for C++ — one header, two tiers.
 *
 * ============================ TIER 1: just sample a formula ============================
 *
 *     #include <gpap/gpap.hpp>
 *
 *     gpap::Sampler sampler("formula.cnf");
 *     gpap::Solutions sols = sampler.sample(100000);      // draw 100k satisfying assignments
 *
 *     printf("%zu solutions at %.0f/s\n", sols.size(), sols.rate());
 *     for (std::size_t i = 0; i < sols.size(); ++i)
 *         for (std::size_t j = 0; j < sols.columns(); ++j)
 *             bool value = sols(i, j);                    // of variable sols.variables()[j]
 *
 * That is the whole API for the flagship use case. The sampler sizes the GPU batch to the free
 * VRAM, picks the faster kernel variant, orders decisions so propagation does the work, runs as
 * many chunks as it takes, and checks on the device that every row it returns is a real model.
 * Variables and literals are DIMACS (1-based, +v true / -v false) — the same numbers as the file.
 *
 * The other entry point, testing many assumption sets at once:
 *
 *     gpap::AssumptionChecker checker("formula.cnf");
 *     auto r = checker.check({{1, -2, 3}, {-1, 2}});      // K sets, ONE kernel launch
 *     if (!r.conflict(0)) for (int lit : r.literals(0)) ...
 *
 * ============================ TIER 2: drive the engine yourself ========================
 *
 * `Formula` + `Engine` below are a thin RAII/exception wrapper over the C ABI in <gpap/gpap.h>,
 * for callers who want to own the chunk loop, the decision order or the readout. Everything in
 * tier 1 is written in terms of it — there is no private back door. Tier 2 is where the concepts
 * live (K, packed splits, selection slots, pair masks, internal 2N literal indices); tier 1
 * exists so that using GPAP does not require any of them.
 */
#ifndef GPAP_HPP
#define GPAP_HPP

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gpap/gpap.h"

namespace gpap {

/** Any failed libgpap call. `status()` is the underlying `gpap_status`. */
class Error : public std::runtime_error {
  public:
    Error(gpap_status st, const std::string &what)
        : std::runtime_error(what + ": " + (gpap_last_error() ? gpap_last_error() : "")),
          status_(st) {}
    gpap_status status() const noexcept { return status_; }

  private:
    gpap_status status_;
};

inline void check(gpap_status st, const char *what) {
    if (st != GPAP_OK) throw Error(st, what);
}

/** Propagate kernel variant. `Auto` picks per batch size — leave it alone unless benchmarking.
 *  Both variants are bit-identical in their results; only the speed differs. */
enum class Mode { Auto = -1, Scalar = GPAP_MODE_SCALAR, Wave = GPAP_MODE_WAVE };

/** Engine construction options; every default is the sensible one.
 *  NOTE: `mode` defaults to Auto here, where the raw C ABI defaults to scalar — C++ callers get
 *  the dispatch for free, C callers pick a mode themselves. The choice never changes results. */
struct Options {
    int device = 0;
    Mode mode = Mode::Auto;
    /** Pin the packed split (8 or 16) instead of deriving it from the formula. Testing only. */
    std::int32_t force_shift = 0;
    /** Warp-wide decision scan in the wave kernel: 0 = default (on), 1 = on, -1 = off. */
    std::int32_t vscan = 0;

    gpap_engine_options raw() const {
        gpap_engine_options o;
        o.device = device;
        /* Auto is resolved by Engine::reset once K is known; start from the C default. */
        o.mode = mode == Mode::Wave ? GPAP_MODE_WAVE : GPAP_MODE_SCALAR;
        o.force_shift = force_shift;
        o.vscan = vscan;
        return o;
    }
};

/** A CNF formula: occurrence lists and per-clause counters, built once and shared by engines. */
class Formula {
  public:
    Formula() = default;
    Formula(Formula &&o) noexcept : h_(o.h_) { o.h_ = nullptr; }
    Formula &operator=(Formula &&o) noexcept {
        if (this != &o) { reset(); h_ = o.h_; o.h_ = nullptr; }
        return *this;
    }
    Formula(const Formula &) = delete;
    Formula &operator=(const Formula &) = delete;
    ~Formula() { reset(); }

    static Formula from_dimacs(const std::string &path) {
        gpap_formula *h = nullptr;
        check(gpap_formula_from_dimacs(path.c_str(), &h), "gpap_formula_from_dimacs");
        return Formula(h);
    }

    /** `clauses` in DIMACS literals (+v / -v, 1-based). */
    static Formula from_clauses(std::int32_t num_vars,
                                const std::vector<std::vector<std::int32_t>> &clauses) {
        std::vector<std::int32_t> flat, lens;
        lens.reserve(clauses.size());
        for (const auto &c : clauses) {
            lens.push_back(static_cast<std::int32_t>(c.size()));
            flat.insert(flat.end(), c.begin(), c.end());
        }
        gpap_formula *h = nullptr;
        check(gpap_formula_from_clauses(num_vars, flat.data(), lens.data(),
                                        static_cast<std::int32_t>(clauses.size()), &h),
              "gpap_formula_from_clauses");
        return Formula(h);
    }

    gpap_formula_info info() const {
        gpap_formula_info i{};
        gpap_formula_info_get(h_, &i);
        return i;
    }
    std::int32_t num_vars() const { return info().num_vars; }
    std::int32_t num_clauses() const { return info().num_clauses; }
    /** A `p cnf` file containing an empty clause is UNSAT outright; it has no solutions. */
    bool trivially_unsat() const { return info().has_empty_clause != 0; }

    /** Independent support from the file's `c ind` lines, as 0-BASED variable indices (tier 2
     *  speaks the ABI's indices throughout, so this drops straight into pair_mask/var_bits);
     *  empty if the file declared none. Decide THESE first and let propagation determine the
     *  rest — that phase structure is what makes projected sampling succeed. */
    std::vector<std::int32_t> support() const {
        std::int32_t n = 0;
        check(gpap_formula_support(h_, nullptr, &n), "gpap_formula_support");
        std::vector<std::int32_t> out(static_cast<size_t>(n));
        if (n) check(gpap_formula_support(h_, out.data(), nullptr), "gpap_formula_support");
        return out;
    }

    const gpap_formula *handle() const noexcept { return h_; }
    explicit operator bool() const noexcept { return h_ != nullptr; }

  private:
    explicit Formula(gpap_formula *h) : h_(h) {}
    void reset() {
        if (h_) gpap_formula_free(h_);
        h_ = nullptr;
    }
    gpap_formula *h_ = nullptr;
};

/** The K-parallel packed BCP engine. Not thread-safe: use one engine per thread. */
class Engine {
  public:
    explicit Engine(const Formula &f, const Options &opts = {}) : want_(opts.mode) {
        const gpap_engine_options raw = opts.raw();
        check(gpap_engine_create(f.handle(), &raw, &h_), "gpap_engine_create");
        device_ = opts.device;
    }
    Engine(Engine &&o) noexcept
        : h_(o.h_), want_(o.want_), device_(o.device_) {
        o.h_ = nullptr;
    }
    Engine &operator=(Engine &&o) noexcept {
        if (this != &o) {
            reset_handle();
            h_ = o.h_; want_ = o.want_; device_ = o.device_;
            o.h_ = nullptr;
        }
        return *this;
    }
    Engine(const Engine &) = delete;
    Engine &operator=(const Engine &) = delete;
    ~Engine() { reset_handle(); }

    // ---------------------------------------------------------------- configuration
    /** Pin the propagate variant, disabling the Mode::Auto dispatch. */
    void set_mode(Mode m) {
        want_ = m;
        if (m != Mode::Auto)
            check(gpap_engine_set_mode(h_, static_cast<gpap_mode>(m)), "gpap_engine_set_mode");
    }
    Mode mode() const { return static_cast<Mode>(gpap_engine_mode(h_)); }
    std::int32_t shift() const { return gpap_engine_shift(h_); }
    std::int32_t k() const { return gpap_engine_k(h_); }
    std::int32_t num_vars() const { return gpap_engine_num_vars(h_); }
    std::int32_t num_clauses() const { return gpap_engine_num_clauses(h_); }
    bool flush_conflict() const { return gpap_engine_flush_conflict(h_) != 0; }

    // ---------------------------------------------------------------- entry points
    /** Under Mode::Auto, choose the propagate variant for a batch of `K` rows: WAVE once the
     *  formula has enough clauses for a warp to have work to spread (and for a batch too small to
     *  fill the device either way), SCALAR on very small formulas at large batches. The rule and
     *  its calibration live in gpap_dispatch_wave. `reset` calls this itself; call it directly only
     *  to make `mode()` truthful before the first reset. A no-op when a mode was pinned. */
    void plan_for(std::int32_t K) {
        if (want_ != Mode::Auto) return;
        const gpap_mode m = gpap_dispatch_wave(num_clauses(), K, device_) ? GPAP_MODE_WAVE
                                                                         : GPAP_MODE_SCALAR;
        if (m != gpap_engine_mode(h_))
            check(gpap_engine_set_mode(h_, m), "gpap_engine_set_mode");
    }

    /** Restore K rows to the cached post-units-flush state (and resolve Mode::Auto for that K). */
    void reset(std::int32_t K) {
        plan_for(K);
        check(gpap_engine_reset(h_, K), "gpap_engine_reset");
    }

    /** Entry point 1: K x L internal literal indices (-1 = padding), BCP to fixpoint. */
    void apply(const std::vector<std::int64_t> &decisions, std::int32_t L) {
        check(gpap_engine_apply(h_, decisions.data(), L), "gpap_engine_apply");
    }

    /** Entry point 2: the whole randomized decide+propagate trajectory per row, one launch.
     *  `order` is the decision priority as 0-based variable indices, support variables FIRST;
     *  `n_support` is where phase 1 ends. Returns the max decision rounds any row used. */
    std::int32_t trajectory(const std::vector<std::int32_t> &order, std::int32_t n_support,
                            std::int32_t B, std::int64_t seed, bool soft = false) {
        std::int32_t rounds = 0;
        check(gpap_engine_trajectory(h_, order.data(), static_cast<std::int32_t>(order.size()),
                                     n_support, B, seed, soft ? 1 : 0, &rounds),
              "gpap_engine_trajectory");
        return rounds;
    }

    void recache_flush() {
        check(gpap_engine_recache_flush(h_), "gpap_engine_recache_flush");
    }

    // ---------------------------------------------------------------- selections + readout
    /** Register the variables a completion test considers (0-based); empty = all of them.
     *  Call this ONCE per variable set and keep the id: masks live as long as the engine and
     *  cannot be released, so creating one per chunk leaks ceil(2N/32) int32 each time. */
    std::int32_t pair_mask(const std::vector<std::int32_t> &vars = {}) {
        std::int32_t id = 0;
        check(gpap_pair_mask_create(h_, vars.empty() ? nullptr : vars.data(),
                                    static_cast<std::int32_t>(vars.size()), &id),
              "gpap_pair_mask_create");
        return id;
    }

    /** Rows that are active AND fully assigned under `mask_id` — the models in this chunk. */
    std::int32_t select_complete(std::int32_t mask_id, std::int32_t slot = 0) {
        std::int32_t n = 0;
        check(gpap_select_complete(h_, mask_id, slot, &n), "gpap_select_complete");
        return n;
    }

    std::vector<std::int32_t> selection_rows(std::int32_t slot, std::int32_t count) {
        std::vector<std::int32_t> rows(static_cast<size_t>(count));
        if (count) check(gpap_selection_rows(h_, slot, rows.data(), count), "gpap_selection_rows");
        return rows;
    }

    void selection_set(std::int32_t slot, const std::vector<std::int32_t> &rows) {
        check(gpap_selection_set(h_, slot, rows.data(), static_cast<std::int32_t>(rows.size())),
              "gpap_selection_set");
    }

    /** How many selected rows have a clause with sat_count == 0. Must be 0. */
    std::int32_t check_sound(std::int32_t slot = 0, std::int32_t max_rows = 4096) {
        std::int32_t n = 0;
        check(gpap_check_sound(h_, slot, max_rows, &n), "gpap_check_sound");
        return n;
    }

    /** (count x cols.size()) row-major 0/1: is variable cols[j] (0-based) TRUE in the i-th
     *  selected row? Gathers min(count, selection count) rows; the buffer is sized to `count`. */
    std::vector<std::uint8_t> var_bits(const std::vector<std::int32_t> &cols, std::int32_t count,
                                       std::int32_t slot = 0) {
        std::vector<std::uint8_t> out(static_cast<size_t>(count) * cols.size());
        if (count)
            check(gpap_var_bits(h_, slot, cols.data(), static_cast<std::int32_t>(cols.size()),
                                reinterpret_cast<std::int8_t *>(out.data()), count),
                  "gpap_var_bits");
        return out;
    }

    /** (count x 2N) row-major: 1 where that internal literal index is TRUE. */
    std::vector<std::int8_t> assignment_rows(std::int32_t count, std::int32_t slot = 0) {
        std::vector<std::int8_t> out(static_cast<size_t>(count) * 2 *
                                     static_cast<size_t>(num_vars()));
        if (count) check(gpap_assignment_rows(h_, slot, out.data(), count), "gpap_assignment_rows");
        return out;
    }

    /** (K,) active and conflict flags. */
    std::pair<std::vector<std::int8_t>, std::vector<std::int8_t>> flags() {
        std::vector<std::int8_t> a(static_cast<size_t>(k())), c(static_cast<size_t>(k()));
        check(gpap_flags(h_, a.data(), c.data()), "gpap_flags");
        return {std::move(a), std::move(c)};
    }

    /** (K,) unit-propagation implications of the last launch (assigned minus decisions). */
    std::vector<std::int32_t> num_propagated() {
        std::vector<std::int32_t> out(static_cast<size_t>(k()));
        check(gpap_num_propagated(h_, out.data()), "gpap_num_propagated");
        return out;
    }

    // ---------------------------------------------------------------- counters
    gpap_counters counters() const {
        gpap_counters c{};
        check(gpap_counters_get(const_cast<gpap_engine *>(h_), &c), "gpap_counters_get");
        return c;
    }
    void counters_reset() { gpap_counters_reset(h_); }

    gpap_engine *handle() noexcept { return h_; }
    int device() const noexcept { return device_; }

  private:
    void reset_handle() {
        if (h_) gpap_engine_free(h_);
        h_ = nullptr;
    }
    gpap_engine *h_ = nullptr;
    Mode want_ = Mode::Auto;
    int device_ = 0;
};

/** Bytes available for new allocations on `device` (free VRAM minus a driver reserve). */
inline std::int64_t mem_available(int device = 0, double fraction = 0.75) {
    return gpap_mem_available(device, fraction);
}

/** Largest K fitting `bytes` for a formula shape. See cpp/src/calibrate.cpp for the model. */
inline std::int32_t compute_k_safe(std::int32_t M, std::int32_t N, std::int64_t bytes,
                                   std::int32_t cap, std::int32_t decisions_per_round = 8) {
    return gpap_compute_k_safe(M, N, bytes, cap, decisions_per_round);
}

/** Device bytes one row of engine state costs (the exact allocation list). */
inline std::int64_t row_bytes(std::int32_t M, std::int32_t N, std::int32_t trail_cap = 0) {
    return gpap_row_bytes(M, N, trail_cap);
}

/** Batch cap sized to this device's total VRAM — the `cap` above, chosen for you. */
inline std::int32_t default_k_cap(int device = 0) { return gpap_default_k_cap(device); }

inline std::string version() { return gpap_version(); }

/** Can this engine run that formula at all? The packed clause word has two field splits (16|16
 *  and 8|24) and a formula fits one of them or neither; constructing an Engine on a formula that
 *  fits neither throws. Sweeping a benchmark suite, call this to SKIP rather than to catch. */
inline bool is_supported(const Formula &f) {
    const gpap_formula_info i = f.info();
    std::int32_t shift = 0;
    return gpap_packed_envelope(i.num_vars, i.max_clause_len, &shift) == GPAP_OK;
}

/* ======================================================================================
 *                       TIER 1 — the high-level API
 * ====================================================================================== */

/** How the sampler should run. Every field has a working default; `seed` is the only one most
 *  callers set. Sizes and kernel choices are derived from the GPU and the formula. */
struct SampleOptions {
    /** Reproducibility: a run is a deterministic function of this seed. (Same seed, same
     *  binary, same solutions — `std::shuffle`'s draw is implementation-defined, so identity
     *  across different standard libraries is not promised.) */
    std::int64_t seed = 0;
    /** Stop after this many seconds of sampling even if the target is not met (0 = no limit). */
    double time_limit_s = 0.0;
    /** How many solutions to materialize on the host. 0 = count only, keep nothing. */
    std::size_t max_keep = 100000;
    /** Which variables to report, as DIMACS variables. Empty = the formula's `c ind` independent
     *  support if it declares one, else every variable. */
    std::vector<std::int32_t> variables{};
    /** CUDA device ordinal. */
    int device = 0;
    /** Variables decided per round before propagating. 8 is a good default; 1 lets propagation
     *  do more of the work and wins on tightly constrained formulas. */
    std::int32_t batch = 8;
    /** In-kernel risk-aware polarity. Helps on some circuit instances, hurts on others. */
    bool lookahead = false;
    /** If the configuration above finds NOTHING in the first few chunks, try once more with
     *  `lookahead` flipped before giving up. This matters: on b12 unrolled 50 cycles, B=8 and
     *  B=1 both return 0 while the same engine with the risk polarity delivers ~1100 models/s —
     *  reporting 0 there would be actively misleading. It is one retry, not a search; the Python
     *  pipeline (`gpap.sampling`) runs a proper adaptive portfolio. */
    bool auto_retry = true;
    /** Trajectories per chunk. 0 = size it to the free VRAM (recommended). */
    std::int32_t k = 0;
    /** Upper bound on the auto-sized K. **0 (default) = size the cap to this GPU's total VRAM**
     *  (`gpap_default_k_cap`); a positive value is that cap verbatim — what a cross-GPU A/B needs,
     *  since the auto value differs per card; negative means no cap at all, leaving the analytic
     *  memory model as the only bound. This used to default to a hardcoded 49,152, which handed
     *  an 80 GB A100 the 12 GB laptop's batch. */
    std::int32_t k_cap = 0;
    /** Verify on the device that emitted rows really satisfy every clause. Costs ~nothing;
     *  turning it off is only for measuring the kernel in isolation. */
    bool verify = true;
};

/** The result of `Sampler::sample`: a bit matrix of solutions plus what it cost to get them.
 *  Row i, column j is the value of variable `variables()[j]` in the i-th solution. */
class Solutions {
  public:
    /** Solutions materialized here (<= found(), capped by SampleOptions::max_keep). */
    std::size_t size() const noexcept { return n_kept_; }
    std::size_t columns() const noexcept { return vars_.size(); }
    bool empty() const noexcept { return n_kept_ == 0; }

    /** Value of `variables()[col]` in solution `row`. */
    bool operator()(std::size_t row, std::size_t col) const {
        return bits_[row * vars_.size() + col] != 0;
    }
    /** The DIMACS variables the columns correspond to. */
    const std::vector<std::int32_t> &variables() const noexcept { return vars_; }
    /** Solution `row` as DIMACS literals (+v where true, -v where false). */
    std::vector<std::int32_t> literals(std::size_t row) const {
        std::vector<std::int32_t> out(vars_.size());
        for (std::size_t j = 0; j < vars_.size(); ++j) out[j] = (*this)(row, j) ? vars_[j] : -vars_[j];
        return out;
    }
    /** Raw row-major 0/1 buffer, size() x columns(). */
    const std::vector<std::uint8_t> &bits() const noexcept { return bits_; }

    /** Total solutions found, including those past the keep cap. */
    std::int64_t found() const noexcept { return found_; }
    /** Trajectories run to find them. */
    std::int64_t drawn() const noexcept { return drawn_; }
    /** Chunks (kernel launches) used. */
    std::int64_t chunks() const noexcept { return chunks_; }
    double seconds() const noexcept { return seconds_; }
    /** Solutions per second. */
    double rate() const noexcept {
        return seconds_ > 0 ? static_cast<double>(found_) / seconds_ : 0.0;
    }
    /** Fraction of trajectories that reached a solution. */
    double success_rate() const noexcept { return drawn_ > 0 ? double(found_) / double(drawn_) : 0.0; }
    /** True when the run stopped on the time limit rather than reaching the target. */
    bool timed_out() const noexcept { return timed_out_; }
    /** Whether SampleOptions::lookahead ended up flipped by the auto-retry, and its final value —
     *  worth carrying over to the next run on the same formula. */
    bool retried() const noexcept { return retried_; }
    bool lookahead_used() const noexcept { return lookahead_; }

  private:
    friend class Sampler;
    std::vector<std::uint8_t> bits_;
    std::vector<std::int32_t> vars_;
    std::size_t n_kept_ = 0;
    std::int64_t found_ = 0, drawn_ = 0, chunks_ = 0;
    double seconds_ = 0.0;
    bool timed_out_ = false, retried_ = false, lookahead_ = false;
};

/** Draws satisfying assignments of a CNF formula on the GPU.
 *
 *      gpap::Sampler sampler("formula.cnf");
 *      auto sols = sampler.sample(100000);
 *
 *  Each of the K rows in a chunk walks its own randomized decide-and-propagate trajectory; a row
 *  that ends with every variable assigned and no conflict IS a satisfying assignment — there is
 *  no repair step and nothing to round. Rows that conflict are dropped, which is why
 *  `Solutions::success_rate()` is worth looking at: it is the fraction that made it.
 *
 *  Construction picks the batch size K from free VRAM, the propagate kernel from K, and the
 *  decision order from the formula's independent support. Read them back with k(), mode() and
 *  variables() if you care; override them through SampleOptions if you must.
 *
 *  This is a SAMPLER, not a decision procedure: `found() == 0` means these trajectories did not
 *  reach a solution, never that the formula is unsatisfiable. Use a CDCL solver for that.
 *
 *  Not thread-safe (it owns an Engine): one Sampler per thread. */
class Sampler {
  public:
    explicit Sampler(const std::string &cnf_path, SampleOptions opts = SampleOptions())
        : Sampler(Formula::from_dimacs(cnf_path), std::move(opts)) {}

    explicit Sampler(Formula formula, SampleOptions opts = SampleOptions())
        : formula_(std::move(formula)), opts_(std::move(opts)),
          engine_(formula_, engine_options(opts_)), rng_(mix_seed(opts_.seed)) {
        const gpap_formula_info info = formula_.info();
        n_vars_ = info.num_vars;

        // --- what to report ------------------------------------------------------------
        // Default: the declared independent support (projected sampling), else all variables.
        // `support` is 0-based (tier 2); everything a caller sees below is DIMACS. A `c ind`
        // line naming a variable the header never declared is dropped rather than trusted —
        // it would otherwise become an out-of-range gather column.
        std::vector<std::int32_t> support;
        for (std::int32_t v : formula_.support())
            if (v >= 0 && v < n_vars_) support.push_back(v);
        vars_ = opts_.variables;
        if (vars_.empty()) {
            if (support.empty()) {
                vars_ = all_vars(n_vars_);
            } else {
                vars_.reserve(support.size());
                for (std::int32_t v : support) vars_.push_back(v + 1);
            }
        } else {
            for (std::int32_t v : vars_)
                if (v < 1 || v > n_vars_)
                    throw Error(GPAP_ERR_ARG, "SampleOptions::variables: variable out of range");
        }
        cols_.reserve(vars_.size());
        for (std::int32_t v : vars_) cols_.push_back(v - 1);          // -> 0-based for the ABI

        // --- decision order ------------------------------------------------------------
        // Phase 1 decides the support and propagates to a fixpoint; phase 2 finishes whatever
        // is still free. The kernel never straddles the boundary. Deciding a variable that
        // propagation would have implied is a wasted decision AND a chance to conflict, so this
        // split is what makes success rates on structured formulas survivable.
        std::vector<char> in_support(static_cast<size_t>(n_vars_), 0);
        for (std::int32_t v : support) in_support[static_cast<size_t>(v)] = 1;
        for (std::int32_t v = 0; v < n_vars_; ++v)
            (in_support[static_cast<size_t>(v)] ? head_ : tail_).push_back(v);
        if (head_.empty()) { head_.swap(tail_); }                    // no `c ind`: a single phase
        n_support_ = static_cast<std::int32_t>(head_.size());
        order_.reserve(static_cast<size_t>(n_vars_));

        // --- batch size ----------------------------------------------------------------
        K_ = opts_.k > 0 ? opts_.k : auto_k(info, opts_);
        engine_.plan_for(K_);          // resolve wave/scalar now, so mode() is true before sample()
        // A model must have EVERY variable assigned, so completion is tested over all of them.
        mask_all_ = engine_.pair_mask();
    }

    /** Draw solutions until `want` have been found, the time limit expires, or the formula
     *  proves hopeless (see below). `want` is a target, not a promise — check `found()`. */
    Solutions sample(std::int64_t want) {
        Solutions out;
        out.vars_ = vars_;
        if (want <= 0 || formula_.trivially_unsat()) return out;      // empty clause => no models

        const std::size_t keep_cap = opts_.max_keep;
        const auto t0 = std::chrono::steady_clock::now();
        out.lookahead_ = opts_.lookahead;
        std::int64_t window_start = 0;                 // first chunk of the current config

        while (out.found_ < want) {
            const std::int32_t n = run_chunk(out.lookahead_);
            out.drawn_ += K_;
            out.chunks_ += 1;
            out.found_ += n;

            if (n > 0 && out.n_kept_ < keep_cap) {
                const std::int32_t take = static_cast<std::int32_t>(
                    std::min<std::int64_t>(n, static_cast<std::int64_t>(keep_cap - out.n_kept_)));
                gather(take, n, out);
            }
            out.seconds_ = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - t0).count();
            // Give up only when NOTHING has been found after a grace period. "This chunk was
            // empty" is not a signal: at a 0.01% success rate most chunks are empty and the run
            // is still productive — an eager stop would report 0 on instances that do deliver.
            if (out.found_ == 0 && out.chunks_ - window_start >= kGraceChunks) {
                if (!opts_.auto_retry || out.retried_) break;
                out.retried_ = true;                   // one flip of the polarity, then stop
                out.lookahead_ = !out.lookahead_;
                window_start = out.chunks_;
            }
            if (opts_.time_limit_s > 0 && out.seconds_ >= opts_.time_limit_s) {
                out.timed_out_ = out.found_ < want;
                break;
            }
        }
        out.seconds_ =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return out;
    }

    // ------------------------------------------------------------------ what it chose
    /** Trajectories per chunk. */
    std::int32_t k() const noexcept { return K_; }
    /** The propagate variant in use. */
    Mode mode() const { return engine_.mode(); }
    /** Variables reported in the solution columns (DIMACS). */
    const std::vector<std::int32_t> &variables() const noexcept { return vars_; }
    /** Size of decision phase 1 — the independent support, or all variables if none is declared. */
    std::int32_t num_support() const noexcept { return n_support_; }
    /** Work done so far: literals assigned, decisions, occurrence entries, launches. */
    gpap_counters counters() const { return engine_.counters(); }

    /** Escape hatches to tier 2 — the same objects the sampler is driving. */
    Engine &engine() noexcept { return engine_; }
    const Formula &formula() const noexcept { return formula_; }

  private:
    /** Chunks to run before concluding that a formula yields nothing at all. */
    static constexpr std::int64_t kGraceChunks = 4;
    /** Share of free VRAM a batch may occupy. The row model is exact (see calibrate.cpp), so
     *  this is the ONLY headroom in the sizing path — it covers readout transients and whatever
     *  else shares the device. It used to be one of three stacked factors (0.75 x 0.60 against a
     *  1.57x-inflated model = 0.29 of free memory actually used). */
    static constexpr double kMemFraction = 0.80;

    static Options engine_options(const SampleOptions &o) {
        Options e;
        e.device = o.device;
        return e;                                     // Mode::Auto — resolved on the first reset
    }
    static std::uint64_t mix_seed(std::int64_t s) {
        return static_cast<std::uint64_t>(s) * 6364136223846793005ull + 1442695040888963407ull;
    }
    static std::vector<std::int32_t> all_vars(std::int32_t n) {
        std::vector<std::int32_t> v(static_cast<size_t>(n));
        std::iota(v.begin(), v.end(), 1);
        return v;
    }

    /** K from the free VRAM, matching the Python sampler's kill-switch-validated model: the
     *  conservative analytic bound, then the fused-trajectory boost (a fused chunk builds no
     *  per-round decision masks, so its real footprint is one (K,M) packed word plus a small
     *  per-variable allowance). This hardware spills to host RAM rather than failing, so the
     *  model deliberately over-reserves — an optimistic K silently halves throughput. */
    static std::int32_t auto_k(const gpap_formula_info &info, const SampleOptions &o) {
        /* SampleOptions::k_cap: 0 = auto (this GPU's VRAM), >0 = verbatim, <0 = uncapped.
         * gpap_compute_k_safe reads cap<=0 as "no cap", which is what the negative case wants. */
        const std::int32_t cap = o.k_cap == 0 ? gpap_default_k_cap(o.device) : o.k_cap;
        const std::int64_t free_bytes = gpap_mem_available(o.device, kMemFraction);
        return gpap_compute_k_safe(info.num_clauses, info.num_vars, free_bytes, cap, o.batch);
    }

    /** One chunk: reshuffle within each phase, run K trajectories, count the models. */
    std::int32_t run_chunk(bool soft) {
        std::shuffle(head_.begin(), head_.end(), rng_);
        std::shuffle(tail_.begin(), tail_.end(), rng_);
        order_.clear();
        order_.insert(order_.end(), head_.begin(), head_.end());
        order_.insert(order_.end(), tail_.begin(), tail_.end());

        engine_.reset(K_);
        engine_.trajectory(order_, n_support_, opts_.batch,
                           static_cast<std::int64_t>(rng_() >> 2), soft);
        const std::int32_t n = engine_.select_complete(mask_all_, /*slot=*/0);
        if (opts_.verify && n > 0 && engine_.check_sound(/*slot=*/0) != 0)
            throw Error(GPAP_ERR_KERNEL, "a returned row does not satisfy the formula");
        return n;
    }

    /** Copy `take` of this chunk's `n` models into `out` (device-side gather of the columns). */
    void gather(std::int32_t take, std::int32_t n, Solutions &out) {
        std::int32_t slot = 0;
        if (take < n) {                            // keep-cap partial: narrow a scratch selection
            std::vector<std::int32_t> rows = engine_.selection_rows(0, n);
            rows.resize(static_cast<size_t>(take));
            slot = 2;
            engine_.selection_set(slot, rows);
        }
        const std::vector<std::uint8_t> chunk = engine_.var_bits(cols_, take, slot);
        out.bits_.insert(out.bits_.end(), chunk.begin(), chunk.end());
        out.n_kept_ += static_cast<std::size_t>(take);
    }

    Formula formula_;
    SampleOptions opts_;
    Engine engine_;
    std::mt19937_64 rng_;
    std::vector<std::int32_t> vars_, cols_;          // DIMACS ids / 0-based columns
    std::vector<std::int32_t> head_, tail_, order_;  // 0-based decision order, phase 1 then 2
    std::int32_t n_vars_ = 0, n_support_ = 0, K_ = 0, mask_all_ = 0;
};

/** Value of a variable after propagation. */
enum class Value : std::int8_t { False = 0, True = 1, Free = 2 };

/** What BCP made of each assumption set. Row order matches the input. */
class CheckResults {
  public:
    std::size_t size() const noexcept { return n_; }
    /** True when the assumptions (plus everything they imply) falsify a clause. */
    bool conflict(std::size_t row) const { return conflict_[row] != 0; }
    std::size_t num_conflicts() const {
        std::size_t c = 0;
        for (auto f : conflict_) c += (f != 0);
        return c;
    }
    /** Literals BCP derived beyond the assumptions themselves. */
    std::int32_t num_propagated(std::size_t row) const { return propagated_[row]; }

    /** Value of DIMACS variable `var` in `row` — True, False, or Free (undetermined). */
    Value value(std::size_t row, std::int32_t var) const {
        if (assign_.empty()) throw Error(GPAP_ERR_ARG, "assignments were not requested");
        const std::size_t base = row * 2 * static_cast<std::size_t>(n_vars_);
        const std::size_t l = base + 2 * static_cast<std::size_t>(var - 1);
        if (assign_[l]) return Value::True;
        if (assign_[l + 1]) return Value::False;
        return Value::Free;
    }
    /** Every literal true in `row`, as DIMACS literals (assumptions and implications alike). */
    std::vector<std::int32_t> literals(std::size_t row) const {
        std::vector<std::int32_t> out;
        for (std::int32_t v = 1; v <= n_vars_; ++v) {
            const Value x = value(row, v);
            if (x == Value::True) out.push_back(v);
            else if (x == Value::False) out.push_back(-v);
        }
        return out;
    }

  private:
    friend class AssumptionChecker;
    std::vector<std::int8_t> conflict_, assign_;
    std::vector<std::int32_t> propagated_;
    std::size_t n_ = 0;
    std::int32_t n_vars_ = 0;
};

/** Tests many assumption sets against one formula in a single kernel launch — the engine's
 *  core primitive.
 *
 *      gpap::AssumptionChecker checker("formula.cnf");
 *      auto r = checker.check({{1, -2}, {-1, 3}, {2}});
 *      r.conflict(1);        // did assuming x1=false, x3=true falsify a clause?
 *      r.value(0, 5);        // what did BCP make of x5 under the first set?
 *
 *  The sets are independent: nothing learned in one affects another. */
class AssumptionChecker {
  public:
    explicit AssumptionChecker(const std::string &cnf_path, int device = 0)
        : AssumptionChecker(Formula::from_dimacs(cnf_path), device) {}

    explicit AssumptionChecker(Formula formula, int device = 0)
        : formula_(std::move(formula)), engine_(formula_, options(device)) {
        n_vars_ = formula_.num_vars();
        device_ = device;
    }

    /** Propagate every set to a fixpoint. `sets` hold DIMACS literals (+v / -v).
     *
     *  ANY NUMBER of sets: they are split into GPU-sized chunks internally and the results come
     *  back in input order, so how many launches it took is not observable. `chunk_size()` says
     *  what one launch holds if you want to know, but nothing here requires you to.
     *
     *  `with_assignments` materializes the full per-row assignment (n x 2N bytes) so that
     *  `value()` and `literals()` work; drop it when only the conflict flags matter — it is the
     *  term that makes a large call expensive, and chunking cannot shrink it. */
    CheckResults check(const std::vector<std::vector<std::int32_t>> &sets,
                       bool with_assignments = true) {
        CheckResults r;
        r.n_ = sets.size();
        r.n_vars_ = n_vars_;
        if (sets.empty()) return r;

        std::size_t L = 0;
        for (const auto &s : sets) L = std::max(L, s.size());
        L = std::max<std::size_t>(L, 1);

        const std::size_t n = sets.size();
        const std::size_t row_lits = 2 * static_cast<std::size_t>(n_vars_);
        r.conflict_.assign(n, 0);
        r.propagated_.assign(n, 0);
        if (with_assignments) r.assign_.assign(n * row_lits, 0);

        const std::size_t step = static_cast<std::size_t>(chunk_size(n, L));
        std::vector<std::int64_t> decisions;
        for (std::size_t a = 0; a < n; a += step) {
            const std::size_t b = std::min(a + step, n);
            const std::int32_t K = static_cast<std::int32_t>(b - a);
            // Padded (K, L) matrix of INTERNAL literal indices: +v -> 2(v-1), -v -> 2(v-1)+1.
            decisions.assign((b - a) * L, -1);
            for (std::size_t k = a; k < b; ++k)
                for (std::size_t j = 0; j < sets[k].size(); ++j) {
                    const std::int32_t lit = sets[k][j];
                    if (lit == 0 || std::abs(lit) > n_vars_)
                        throw Error(GPAP_ERR_ARG, "assumption literal out of range");
                    decisions[(k - a) * L + j] = 2 * (std::abs(lit) - 1) + (lit < 0 ? 1 : 0);
                }

            engine_.reset(K);          /* the LAST chunk is short: reset to its own size */
            engine_.apply(decisions, static_cast<std::int32_t>(L));

            auto fl = engine_.flags();
            std::copy_n(fl.second.begin(), b - a, r.conflict_.begin() + a);
            const std::vector<std::int32_t> prop = engine_.num_propagated();
            std::copy_n(prop.begin(), b - a, r.propagated_.begin() + a);
            if (with_assignments) {
                std::vector<std::int32_t> rows(static_cast<std::size_t>(K));
                std::iota(rows.begin(), rows.end(), 0);
                engine_.selection_set(/*slot=*/2, rows);
                const std::vector<std::int8_t> rowsa = engine_.assignment_rows(K, /*slot=*/2);
                std::copy_n(rowsa.begin(), (b - a) * row_lits, r.assign_.begin() + a * row_lits);
            }
        }
        return r;
    }

    /** Assumption sets one launch holds. The engine's own per-row state (the conservative,
     *  kill-switch-validated model) plus the (K, L) int64 decision matrix this entry point
     *  allocates and the sampler's path does not — 8 bytes per literal slot per row, which on a
     *  small formula with long assumption sets is the dominant term. Capped like the sampler is,
     *  and never larger than the input. */
    std::size_t chunk_size(std::size_t n_sets, std::size_t max_lits) const {
        const gpap_formula_info i = formula_.info();
        const std::int64_t free_bytes = gpap_mem_available(device_, 0.80);
        const std::int64_t per_row =
            gpap_row_bytes(i.num_clauses, i.num_vars, 0) +
            8ll * static_cast<std::int64_t>(std::max<std::size_t>(max_lits, 1));
        std::int64_t k = free_bytes / std::max<std::int64_t>(per_row, 1);
        k = std::min<std::int64_t>(k, gpap_default_k_cap(device_));
        k = std::max<std::int64_t>(k, 1);
        return std::min<std::size_t>(static_cast<std::size_t>(k), n_sets);
    }

    Engine &engine() noexcept { return engine_; }
    const Formula &formula() const noexcept { return formula_; }

  private:
    static Options options(int device) {
        Options o;
        o.device = device;
        return o;
    }
    Formula formula_;
    Engine engine_;
    std::int32_t n_vars_ = 0;
    int device_ = 0;
};

}  // namespace gpap

#endif /* GPAP_HPP */
