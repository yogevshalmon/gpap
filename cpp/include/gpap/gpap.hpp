/* GPAP (GPU-Parallel (partial) Assignment Propagation) for C++ — one header, two tiers.
 *
 * Tier 1: gpap::Sampler("f.cnf").sample(100000) draws satisfying assignments, sizing the GPU
 * batch to free VRAM and running as many chunks as it takes; gpap::AssumptionChecker propagates
 * many assumption sets in one launch; gpap::Prober BCP-probes every k-subset of a literal pool.
 * Tier 2: `Formula` + `Engine`, a RAII/exception wrapper over the C ABI in <gpap/gpap.h>, for
 * callers who want the chunk loop, the decision order or the readout themselves. Tier 1 is
 * written entirely in terms of it and owns the concepts (K, packed splits, selection slots, pair
 * masks, internal 2N literal indices). Literals are DIMACS (1-based, +v / -v) on both. */
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

/** Propagate kernel variant; `Auto` picks per formula/batch. All agree bit for bit. */
enum class Mode { Auto = GPAP_MODE_AUTO, Scalar = GPAP_MODE_SCALAR, Wave = GPAP_MODE_WAVE };

/** Engine construction options; every field's zero is its default, `mode` Auto on both faces. */
struct Options {
    int device = 0;
    Mode mode = Mode::Auto;
    /** Pin the packed split (8 or 16) instead of deriving it from the formula. Testing only. */
    std::int32_t force_shift = 0;
    /** Warp-wide decision scan in the wave kernel: 0 = default (on), 1 = on, -1 = off. */
    std::int32_t vscan = 0;

    gpap_engine_options raw() const {
        /* Value-initialized; Auto passes through and the core re-resolves it on each reset(K). */
        gpap_engine_options o{};
        o.device = device;
        o.mode = static_cast<gpap_mode>(mode);
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

    /** Independent support from the file's `c ind` lines, as 0-based variable indices (what
     *  pair_mask/var_bits take); empty if none. Decide these first, then let propagation finish. */
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
    explicit Engine(const Formula &f, const Options &opts = {}) {
        const gpap_engine_options raw = opts.raw();
        check(gpap_engine_create(f.handle(), &raw, &h_), "gpap_engine_create");
        device_ = opts.device;
    }
    Engine(Engine &&o) noexcept
        : h_(o.h_), device_(o.device_) {
        o.h_ = nullptr;
    }
    Engine &operator=(Engine &&o) noexcept {
        if (this != &o) {
            reset_handle();
            h_ = o.h_; device_ = o.device_;
            o.h_ = nullptr;
        }
        return *this;
    }
    Engine(const Engine &) = delete;
    Engine &operator=(const Engine &) = delete;
    ~Engine() { reset_handle(); }

    /** Pin the propagate variant, or hand the choice back with Mode::Auto (auto_dispatch() asks). */
    void set_mode(Mode m) {
        check(gpap_engine_set_mode(h_, static_cast<gpap_mode>(m)), "gpap_engine_set_mode");
    }
    Mode mode() const { return static_cast<Mode>(gpap_engine_mode(h_)); }
    bool auto_dispatch() const { return gpap_engine_auto(h_) != 0; }
    std::int32_t shift() const { return gpap_engine_shift(h_); }
    std::int32_t k() const { return gpap_engine_k(h_); }
    std::int32_t num_vars() const { return gpap_engine_num_vars(h_); }
    std::int32_t num_clauses() const { return gpap_engine_num_clauses(h_); }
    bool flush_conflict() const { return gpap_engine_flush_conflict(h_) != 0; }

    /** Which kernel a batch of `K` rows runs: a pure query; reset() resolves Auto on its own. */
    Mode plan_for(std::int32_t K) const {
        if (!gpap_engine_auto(h_)) return mode();
        return gpap_dispatch_wave(gpap_engine_num_clauses(h_), K, device_)
                   ? Mode::Wave : Mode::Scalar;
    }

    /** Restore K rows to the cached post-units-flush state (and resolve Mode::Auto for that K). */
    void reset(std::int32_t K) { check(gpap_engine_reset(h_, K), "gpap_engine_reset"); }

    /** Entry point 1: K x L internal literal indices (-1 = padding), BCP to fixpoint. */
    void apply(const std::vector<std::int64_t> &decisions, std::int32_t L) {
        require_matrix(decisions, L, "apply");
        check(gpap_engine_apply(h_, decisions.data(), L), "gpap_engine_apply");
    }

    /** apply() in CLEAN mode: identical flags and counters, but the launch restores its own packed
     *  rows to the baseline so the next reset(K) skips the K*M broadcast — so it is only for
     *  callers that read flags/counters. See gpap_engine_apply_clean in gpap.h. */
    void apply_clean(const std::vector<std::int64_t> &decisions, std::int32_t L) {
        require_matrix(decisions, L, "apply_clean");
        check(gpap_engine_apply_clean(h_, decisions.data(), L), "gpap_engine_apply_clean");
    }

    /** Entry point 2: the whole randomized decide+propagate trajectory per row, one launch.
     *  `order` is decision priority (0-based variables, support first) and `n_support` where
     *  phase 1 ends; returns the max decision rounds any row used. */
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

    /** {full K*M broadcasts, broadcasts skipped after a clean one}: the lite count proves CLEAN. */
    std::pair<std::int64_t, std::int64_t> reset_stats() const {
        std::int64_t full = 0, lite = 0;
        gpap_engine_reset_stats(h_, &full, &lite);
        return {full, lite};
    }

    /** Register the variables a completion test considers (0-based); empty = all. Call once per
     *  variable set and keep the id: a mask lives as long as the engine, so one per chunk leaks. */
    std::int32_t pair_mask(const std::vector<std::int32_t> &vars = {}) {
        std::int32_t id = 0;
        check(gpap_pair_mask_create(h_, vars.empty() ? nullptr : vars.data(),
                                    static_cast<std::int32_t>(vars.size()), &id),
              "gpap_pair_mask_create");
        return id;
    }

    /** Rows that are active and fully assigned under `mask_id` — the models in this chunk. */
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

    /** Selected rows with an emptied clause; must be 0. `max_rows` bounds it to a prefix. */
    std::int32_t check_sound(std::int32_t slot = 0, std::int32_t max_rows = 0) {
        std::int32_t n = 0;
        check(gpap_check_sound(h_, slot, max_rows, &n), "gpap_check_sound");
        return n;
    }

    /** (count x cols.size()) row-major 0/1: is 0-based variable cols[j] true in selected row i? */
    std::vector<std::uint8_t> var_bits(const std::vector<std::int32_t> &cols, std::int32_t count,
                                       std::int32_t slot = 0) {
        std::vector<std::uint8_t> out(static_cast<size_t>(count) * cols.size());
        if (count)
            check(gpap_var_bits(h_, slot, cols.data(), static_cast<std::int32_t>(cols.size()),
                                reinterpret_cast<std::int8_t *>(out.data()), count),
                  "gpap_var_bits");
        return out;
    }

    /** (count x 2N) row-major: 1 where that internal literal index is true. */
    std::vector<std::int8_t> assignment_rows(std::int32_t count, std::int32_t slot = 0) {
        std::vector<std::int8_t> out(static_cast<size_t>(count) * 2 *
                                     static_cast<size_t>(num_vars()));
        if (count) check(gpap_assignment_rows(h_, slot, out.data(), count), "gpap_assignment_rows");
        return out;
    }

    /** (K,) active and conflict flags; num_propagated() is the implication count per row. */
    std::pair<std::vector<std::int8_t>, std::vector<std::int8_t>> flags() {
        std::vector<std::int8_t> a(static_cast<size_t>(k())), c(static_cast<size_t>(k()));
        check(gpap_flags(h_, a.data(), c.data(), k()), "gpap_flags");
        return {std::move(a), std::move(c)};
    }

    std::vector<std::int32_t> num_propagated() {
        std::vector<std::int32_t> out(static_cast<size_t>(k()));
        check(gpap_num_propagated(h_, out.data(), k()), "gpap_num_propagated");
        return out;
    }

    gpap_counters counters() const {
        gpap_counters c{};
        check(gpap_counters_get(const_cast<gpap_engine *>(h_), &c), "gpap_counters_get");
        return c;
    }
    void counters_reset() { gpap_counters_reset(h_); }

    gpap_engine *handle() noexcept { return h_; }
    int device() const noexcept { return device_; }

  private:
    /** Memory safety: the ABI reads K*L int64s unchecked, and an out-of-range literal index
     *  writes into a later row's bitmap, undetectably. */
    void require_matrix(const std::vector<std::int64_t> &decisions, std::int32_t L,
                        const char *what) const {
        const std::size_t want = static_cast<std::size_t>(k()) * static_cast<std::size_t>(L);
        /* `<`, not `!=`: a larger buffer is safe. `L < 0` is not redundant — it can wrap small. */
        if (L < 0 || decisions.size() < want)
            throw Error(GPAP_ERR_ARG, std::string("Engine::") + what + ": decisions holds " +
                                          std::to_string(decisions.size()) + " literals but K=" +
                                          std::to_string(k()) + " x L=" + std::to_string(L) +
                                          " needs " + std::to_string(want));
    }

    void reset_handle() {
        if (h_) gpap_engine_free(h_);
        h_ = nullptr;
    }
    gpap_engine *h_ = nullptr;
    int device_ = 0;
};

/** The batch-sizing model (cpp/src/calibrate.cpp). mem_available() is free VRAM minus a driver
 *  reserve, and `fraction` — 0.80 on every face — is the only headroom left in the sizing path.
 *  compute_k_safe() is the largest K fitting `bytes` and row_bytes() one row's device cost, both
 *  over `gpap_formula_info::num_long`, NOT num_clauses. default_k_cap() is the `cap` itself. */
inline std::int64_t mem_available(int device = 0, double fraction = 0.80) {
    return gpap_mem_available(device, fraction);
}

inline std::int32_t compute_k_safe(std::int32_t num_long, std::int32_t N, std::int64_t bytes,
                                   std::int32_t cap = 0) {
    return gpap_compute_k_safe(num_long, N, bytes, cap);
}
inline std::int64_t row_bytes(std::int32_t num_long, std::int32_t N,
                              std::int32_t trail_cap = 0) {
    return gpap_row_bytes(num_long, N, trail_cap);
}

inline std::int32_t default_k_cap(int device = 0) { return gpap_default_k_cap(device); }

inline std::string version() { return gpap_version(); }
/** Identity of this exact build; key a cache of measurements on this, never on version(). */
inline std::string build_id() { return gpap_build_id(); }

/** Can this engine run that formula? The packed clause word has two field splits (16|16, 8|24)
 *  and an Engine over a formula that fits neither throws; sweep a suite with this, not a catch. */
inline bool is_supported(const Formula &f) {
    const gpap_formula_info i = f.info();
    std::int32_t shift = 0;
    return gpap_packed_envelope(i.num_vars, i.max_clause_len, &shift) == GPAP_OK;
}

/* ================================ TIER 1 — the high-level API ======================== */

/** Which unassigned variable each decision picks; int-valued to match the ABI's config. */
enum class VarOrder : std::int32_t {
    Structured = 0,  ///< first free variable in index order; wins search-heavy instances
    Random = 1       ///< uniform-random free variable (CMSGen's Algorithm 1); default
};

/** How the sampler should run; `seed` is the only field most callers set. A default run is
 *  deterministic — same seed, build and batch, same solutions — except that a `portfolio` run is
 *  not machine-reproducible by design and `k`/`k_cap` at 0 size the batch from live free VRAM. */
struct SampleOptions {
    /** A single-config run at a fixed batch is a pure function of this seed (not of `k`). */
    std::int64_t seed = 0;
    /** Stop short of the target after this much sampling wall, or this many chunks (0 = off). */
    double time_limit_s = 0.0;
    std::int64_t max_chunks = 0;
    /** Solutions to materialize on the host (0 = count only) and the DIMACS variables to report
     *  (empty = the `c ind` support, else all). Kept rows are billed outside the rate. */
    std::size_t max_keep = 100000;
    std::vector<std::int32_t> variables{};
    int device = 0;
    /** When non-empty, each chunk runs one of these configs at full K, raced as a throughput
     *  bandit in the core, and only the winner's kept rows are returned. Empty (default) = the
     *  single config below, the deterministic pinned-measurement mode. */
    std::vector<gpap_sample_config> portfolio{};
    /** Single-config mode: how decisions pick a variable, and how many per round before
     *  propagating. B = 1 lets propagation do more of the work and wins on tight formulas;
     *  B >= |support| makes the surviving rows exact rejection sampling, hence exactly uniform. */
    VarOrder var_order = VarOrder::Random;
    std::int32_t batch = 8;
    /** Single-config mode: in-kernel risk-aware ("soft") polarity, preferring the value less
     *  likely to conflict. A large success-rate win on tight/deep circuits and NOT uniform — it
     *  biases the distribution by construction and Barbarik rejects it on nearly every circuit
     *  instance. Never for a distribution claim. docs/findings/UNIFORMITY_FINDINGS.md 4b. */
    bool lookahead = false;
    /** If the run finds nothing in the first few chunks, try once more with `lookahead` flipped
     *  before giving up. Off by default because the flip enables that non-uniform soft polarity;
     *  check Solutions::lookahead_used() when it is set. (Renamed from `auto_retry`, not
     *  reversed, so an out-of-tree caller gets a compile error.) */
    bool retry_with_lookahead = false;
    /** Trajectories per chunk, and the ceiling when they are auto-sized. k = 0 sizes the batch
     *  to free VRAM, making it a function of live conditions rather than of the seed — pin it
     *  whenever two runs must be comparable. k_cap: 0 = the library's device-independent batch
     *  cap, positive = that cap verbatim, negative = no cap. */
    std::int32_t k = 0;
    std::int32_t k_cap = 0;
    /** Re-check on the device that every emitted row really satisfies every clause. Off by
     *  default: a self-test, not part of producing samples. When on it covers every completed
     *  row, is billed to Solutions::verify_seconds (outside the rate), and makes sample() throw
     *  if any row fails. Matches gpap.sampling's `verify_sound`. */
    bool verify = false;
    /** Chunks a config must accumulate before the target may stop a portfolio run (0 = 1). */
    std::int32_t min_measure_chunks = 0;
};

/** The portfolio the Python pipeline races by default. Uniformity-safe: no soft member. */
inline std::vector<gpap_sample_config> default_portfolio() {
    return {{1, 8, 0}, {1, 1, 0}, {0, 1, 0}};
}

/** `Sampler::sample`'s result: row i, column j is `variables()[j]` in solution i. */
class Solutions {
  public:
    /** Solutions materialized here (<= found(), capped by max_keep), all from the winner. */
    std::size_t size() const noexcept { return n_kept_; }
    std::size_t columns() const noexcept { return vars_.size(); }
    bool empty() const noexcept { return n_kept_ == 0; }

    bool operator()(std::size_t row, std::size_t col) const {
        return bits_[row * vars_.size() + col] != 0;
    }
    /** The DIMACS variables the columns correspond to; literals(row)/bits() read them out. */
    const std::vector<std::int32_t> &variables() const noexcept { return vars_; }
    std::vector<std::int32_t> literals(std::size_t row) const {
        std::vector<std::int32_t> out(vars_.size());
        for (std::size_t j = 0; j < vars_.size(); ++j) out[j] = (*this)(row, j) ? vars_[j] : -vars_[j];
        return out;
    }
    const std::vector<std::uint8_t> &bits() const noexcept { return bits_; }

    /** Totals: solutions found (keep cap included), trajectories run, chunks launched. */
    std::int64_t found() const noexcept { return stats_.n_models; }
    std::int64_t drawn() const noexcept { return stats_.attempts; }
    std::int64_t chunks() const noexcept { return stats_.chunks; }

    /** Clocks. seconds() is the chunk loop alone, wall_seconds() all three: copy-out and the
     *  verify re-check are billed apart so that neither can move rate(). */
    double seconds() const noexcept { return stats_.sampling_s; }
    double deliver_seconds() const noexcept { return stats_.deliver_s; }
    double verify_seconds() const noexcept { return stats_.verify_s; }
    double wall_seconds() const noexcept {
        return stats_.sampling_s + stats_.deliver_s + stats_.verify_s;
    }
    double rate() const noexcept {
        return stats_.sampling_s > 0 ? double(stats_.n_models) / stats_.sampling_s : 0.0;
    }
    /** Fraction of trajectories that reached a solution. */
    double success_rate() const noexcept {
        return stats_.attempts > 0 ? double(stats_.n_models) / double(stats_.attempts) : 0.0;
    }

    /** Why the run ended: STOP_UNSAT is a proof (construction conflicted), ZERO_YIELD is not. */
    gpap_sample_stop stop() const noexcept { return static_cast<gpap_sample_stop>(stats_.stop); }
    bool timed_out() const noexcept { return stats_.timed_out != 0; }
    /** Whether the retry flip fired, and the soft flag the winning config ran with. */
    bool retried() const noexcept { return stats_.retried != 0; }
    bool lookahead_used() const noexcept { return stats_.lookahead_used != 0; }

    /** Rows the verify re-check covered; 0 with verify on means it never ran — no verdict. */
    std::int64_t sound_rows_checked() const noexcept { return stats_.sound_rows_checked; }

    /** Portfolio breakdown, one entry per config; `best_config()`'s rows fill bits(). */
    std::size_t n_configs() const noexcept { return cstats_.size(); }
    const gpap_config_stats &config_stats(std::size_t i) const { return cstats_.at(i); }
    std::int32_t best_config() const noexcept { return stats_.best_config; }
    std::int32_t dominant_config() const noexcept { return stats_.dominant_config; }
    const gpap_sample_stats &stats() const noexcept { return stats_; }

  private:
    friend class Sampler;
    std::vector<std::uint8_t> bits_;
    std::vector<std::int32_t> vars_;
    std::size_t n_kept_ = 0;
    gpap_sample_stats stats_{};
    std::vector<gpap_config_stats> cstats_;
};

/** Draws satisfying assignments of a CNF formula on the GPU: `Sampler("f.cnf").sample(100000)`.
 *  Each of the K rows in a chunk walks its own randomized decide-and-propagate trajectory; a row
 *  that ends fully assigned with no conflict is a satisfying assignment (no repair step, nothing
 *  to round) and conflicted rows are dropped, so `Solutions::success_rate()` is the fraction that
 *  made it. The chunk loop lives in the core (`gpap_sampler_*`, cpp/src/sample.cpp), shared with
 *  the Python face; this class is marshalling, and construction picks K, the kernel and the
 *  decision order for you. A sampler, not a decision procedure: `found() == 0` with stop() !=
 *  UNSAT means these trajectories reached no solution, not that the formula is unsatisfiable.
 *  Not thread-safe (it owns an Engine): one Sampler per thread. */
class Sampler {
  public:
    explicit Sampler(const std::string &cnf_path, SampleOptions opts = SampleOptions())
        : Sampler(Formula::from_dimacs(cnf_path), std::move(opts)) {}

    explicit Sampler(Formula formula, SampleOptions opts = SampleOptions())
        : formula_(std::move(formula)), opts_(std::move(opts)),
          engine_(formula_, engine_options(opts_)) {
        const gpap_formula_info info = formula_.info();
        n_vars_ = info.num_vars;

        run_opts_ = gpap_sampler_options{};          /* {0} is every default, by the ground rule */
        run_opts_.max_chunks = opts_.max_chunks;
        run_opts_.keep_cap = static_cast<std::int64_t>(opts_.max_keep);
        run_opts_.time_limit_s = opts_.time_limit_s;
        run_opts_.min_measure_chunks = opts_.min_measure_chunks;
        run_opts_.verify = opts_.verify ? 1 : 0;
        run_opts_.retry_with_lookahead = opts_.retry_with_lookahead ? 1 : 0;
        check(gpap_sampler_create(engine_.handle(), opts_.seed, opts_.k_cap, &h_),
              "gpap_sampler_create");
        /* A ctor that throws never runs ~Sampler(): without this the handle leaks. */
        try {
            init_from_options();
        } catch (...) {
            reset_handle();
            throw;
        }
    }


    Sampler(const Sampler &) = delete;
    Sampler &operator=(const Sampler &) = delete;
    ~Sampler() { reset_handle(); }

    /** Draw until `want` solutions are found or a valve fires; `want` is a target, not a promise. */
    Solutions sample(std::int64_t want) {
        Solutions out;
        out.vars_ = vars_;
        const std::vector<gpap_sample_config> cfgs = run_configs();
        check(gpap_sampler_run(h_, want, cfgs.data(),
                               static_cast<std::int32_t>(cfgs.size()), &run_opts_, &out.stats_),
              "gpap_sampler_run");
        out.cstats_.resize(cfgs.size());
        for (std::size_t i = 0; i < cfgs.size(); ++i)
            check(gpap_sampler_config_stats(h_, static_cast<std::int32_t>(i), &out.cstats_[i]),
                  "gpap_sampler_config_stats");
        const std::int64_t kept =
            out.cstats_[static_cast<std::size_t>(out.stats_.best_config)].kept_rows;
        if (kept > 0 && !vars_.empty()) {
            out.bits_.resize(static_cast<std::size_t>(kept) * vars_.size());
            check(gpap_sampler_bits(h_, -1, out.bits_.data(), kept), "gpap_sampler_bits");
            out.n_kept_ = static_cast<std::size_t>(kept);
        }
        if (opts_.verify && out.stats_.unsound_rows > 0)
            throw Error(GPAP_ERR_KERNEL, "a returned row does not satisfy the formula (" +
                                             std::to_string(out.stats_.unsound_rows) + " of " +
                                             std::to_string(out.stats_.sound_rows_checked) +
                                             " checked)");
        return out;
    }

    /** What construction chose: K, propagate variant, solution columns, support size, work. */
    std::int32_t k() const noexcept { return gpap_sampler_k(h_); }
    Mode mode() const { return engine_.mode(); }
    const std::vector<std::int32_t> &variables() const noexcept { return vars_; }
    std::int32_t num_support() const noexcept { return n_support_; }
    gpap_counters counters() const { return engine_.counters(); }
    /** Restore the RNG to a known point; set_k pins the batch (k <= 0 = the calibrated one). */
    void reseed(std::int64_t seed) { check(gpap_sampler_reseed(h_, seed), "gpap_sampler_reseed"); }
    void set_k(std::int32_t k) { check(gpap_sampler_set_k(h_, k), "gpap_sampler_set_k"); }

    /** Escape hatches to tier 2 — the same objects the sampler is driving. */
    Engine &engine() noexcept { return engine_; }
    const Formula &formula() const noexcept { return formula_; }

  private:
    void init_from_options() {
        if (opts_.k > 0) check(gpap_sampler_set_k(h_, opts_.k), "gpap_sampler_set_k");

        /* Decision phase 1, 0-based; a `c ind` variable the header never declared is dropped. */
        std::vector<std::int32_t> support;
        for (std::int32_t v : formula_.support())
            if (v >= 0 && v < n_vars_) support.push_back(v);
        if (!support.empty())
            check(gpap_sampler_set_support(h_, support.data(),
                                           static_cast<std::int32_t>(support.size())),
                  "gpap_sampler_set_support");
        n_support_ = support.empty() ? n_vars_ : static_cast<std::int32_t>(support.size());

        vars_ = opts_.variables;
        if (vars_.empty()) {
            if (support.empty()) {
                vars_.resize(static_cast<std::size_t>(n_vars_));
                std::iota(vars_.begin(), vars_.end(), 1);
            } else {
                vars_.reserve(support.size());
                for (std::int32_t v : support) vars_.push_back(v + 1);
            }
        } else {
            for (std::int32_t v : vars_)
                if (v < 1 || v > n_vars_)
                    throw Error(GPAP_ERR_ARG, "SampleOptions::variables: variable out of range");
        }
        std::vector<std::int32_t> cols;
        cols.reserve(vars_.size());
        for (std::int32_t v : vars_) cols.push_back(v - 1);
        check(gpap_sampler_set_columns(h_, cols.data(),
                                       static_cast<std::int32_t>(cols.size())),
              "gpap_sampler_set_columns");
    }

    static Options engine_options(const SampleOptions &o) {
        Options e;
        e.device = o.device;
        return e;                                     // Mode::Auto — the core resolves and pins
    }
    std::vector<gpap_sample_config> run_configs() const {
        if (!opts_.portfolio.empty()) return opts_.portfolio;
        return {gpap_sample_config{static_cast<std::int32_t>(opts_.var_order), opts_.batch,
                                   opts_.lookahead ? 1 : 0}};
    }
    void reset_handle() {
        if (h_) {
            gpap_sampler_free(h_);
            h_ = nullptr;
        }
    }

    /* ~Sampler() frees the handle in its body, while the borrowed engine is still alive. */
    Formula formula_;
    SampleOptions opts_;
    gpap_sampler_options run_opts_{};            // opts_, translated once for the ABI
    Engine engine_;
    std::vector<std::int32_t> vars_;             // DIMACS ids of the reported columns
    std::int32_t n_vars_ = 0, n_support_ = 0;
    gpap_sampler *h_ = nullptr;
};

/** Value of a variable after propagation. */
enum class Value : std::int8_t { False = 0, True = 1, Free = 2 };

/** What BCP made of each assumption set. Row order matches the input. */
class CheckResults {
  public:
    std::size_t size() const noexcept { return n_; }
    /** True when the assumptions, plus what they imply, falsify a clause. */
    bool conflict(std::size_t row) const { return conflict_[row] != 0; }
    std::size_t num_conflicts() const {
        std::size_t c = 0;
        for (auto f : conflict_) c += (f != 0);
        return c;
    }
    std::int32_t num_propagated(std::size_t row) const { return propagated_[row]; }

    /** Value of DIMACS variable `var` in `row` — True, False, or Free (undetermined);
     *  `literals(row)` lists every literal true there, assumptions and implications alike. */
    Value value(std::size_t row, std::int32_t var) const {
        if (assign_.empty()) throw Error(GPAP_ERR_ARG, "assignments were not requested");
        const std::size_t base = row * 2 * static_cast<std::size_t>(n_vars_);
        const std::size_t l = base + 2 * static_cast<std::size_t>(var - 1);
        if (assign_[l]) return Value::True;
        if (assign_[l + 1]) return Value::False;
        return Value::Free;
    }
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

/** Tests many assumption sets against one formula in a single kernel launch — the engine's core
 *  primitive. The sets are independent: nothing learned in one affects another.
 *      auto r = gpap::AssumptionChecker("f.cnf").check({{1, -2}, {-1, 3}, {2}});
 *      r.conflict(1);   // did assuming x1=false, x3=true falsify a clause? */
class AssumptionChecker {
  public:
    explicit AssumptionChecker(const std::string &cnf_path, int device = 0)
        : AssumptionChecker(Formula::from_dimacs(cnf_path), device) {}

    explicit AssumptionChecker(Formula formula, int device = 0)
        : formula_(std::move(formula)), engine_(formula_, options(device)) {
        n_vars_ = formula_.num_vars();
    }

    /** Propagate every set to a fixpoint; `sets` hold DIMACS literals. Any number of sets: they
     *  are split into GPU-sized chunks by `gpap_check_sets` (cpp/src/check.cpp — the one home of
     *  that loop, shared with the Python face) and come back in input order. `with_assignments`
     *  materializes the per-row assignment (n x 2N bytes) so `value()` and `literals()` work;
     *  drop it when only the flags matter, since chunking cannot shrink that term. */
    CheckResults check(const std::vector<std::vector<std::int32_t>> &sets,
                       bool with_assignments = true) {
        CheckResults r;
        r.n_ = sets.size();
        r.n_vars_ = n_vars_;
        if (sets.empty()) return r;

        const std::size_t n = sets.size();
        std::vector<std::int32_t> lits;
        std::vector<std::int64_t> offsets(n + 1, 0);
        std::size_t total = 0;
        for (const auto &s : sets) total += s.size();
        lits.reserve(total);
        for (std::size_t i = 0; i < n; ++i) {
            lits.insert(lits.end(), sets[i].begin(), sets[i].end());
            offsets[i + 1] = static_cast<std::int64_t>(lits.size());
        }

        r.conflict_.assign(n, 0);
        r.propagated_.assign(n, 0);
        if (with_assignments)
            r.assign_.assign(n * 2 * static_cast<std::size_t>(n_vars_), 0);
        gpap::check(gpap_check_sets(engine_.handle(), lits.data(), offsets.data(),
                                    static_cast<std::int64_t>(n), /*chunk=*/0,
                                    r.conflict_.data(), r.propagated_.data(),
                                    with_assignments ? r.assign_.data() : nullptr,
                                    /*sat_out=*/nullptr),
                    "gpap_check_sets");
        return r;
    }

    /** Assumption sets one launch holds — `gpap_check_chunk_size`, the core's one memory model. */
    std::size_t chunk_size(std::size_t n_sets, std::size_t max_lits) {
        return static_cast<std::size_t>(
            gpap_check_chunk_size(engine_.handle(), static_cast<std::int64_t>(n_sets),
                                  static_cast<std::int64_t>(max_lits)));
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
};

/** When a probe sweep should stop. Every field is optional; the first to bind wins. */
struct ProbeBudget {
    /** Wall-clock ceiling, checked between chunks (the smallest unit of GPU work). */
    double seconds = 0.0;
    /** A work budget, checked against an exact count before anything is generated: the sweep
     *  probes this many and reports `exhausted() == false`, and every conflict it found is still
     *  a genuine entailed clause. `refuse_over_max` refuses such a sweep outright instead. */
    std::int64_t max_tuples = 0;
    bool refuse_over_max = false;
    /** Tuples per launch; 0 sizes it from free VRAM with the library's one memory model. */
    std::int32_t batch = 0;
    /** Drop tuples holding two literals of one variable: they conflict for reasons that have
     *  nothing to do with the formula. */
    bool distinct_vars = true;
};

/** One sweep's result: which assumption tuples conflicted, and what the sweep cost. */
class Probe {
  public:
    Probe() = default;
    Probe(const Probe &) = delete;
    Probe &operator=(const Probe &) = delete;
    Probe(Probe &&o) noexcept { swap(o); }
    Probe &operator=(Probe &&o) noexcept { swap(o); return *this; }
    ~Probe() { if (h_) gpap_probe_free(h_); }

    std::int64_t size() const { return stats_.conflicts; }
    std::int32_t width() const { return k_; }
    /** Conflicting tuple `i` as DIMACS literals: assuming these drove BCP to a conflict, so the
     *  formula entails the negation of their conjunction, which `clause(i)` returns. */
    std::vector<std::int32_t> operator[](std::int64_t i) const {
        const std::size_t off = static_cast<std::size_t>(i) * static_cast<std::size_t>(k_);
        return {tuples_.begin() + static_cast<std::ptrdiff_t>(off),
                tuples_.begin() + static_cast<std::ptrdiff_t>(off + static_cast<std::size_t>(k_))};
    }
    std::vector<std::int32_t> clause(std::int64_t i) const {
        std::vector<std::int32_t> c = (*this)[i];
        for (auto &l : c) l = -l;
        return c;
    }

    std::int64_t total() const { return stats_.total; }
    std::int64_t probed() const { return stats_.probed; }
    double seconds() const { return stats_.seconds; }
    /** True iff every candidate was probed — what makes a negative result meaningful: an
     *  exhausted sweep that found nothing proves no entailed clause of this width exists (that
     *  BCP can see), where a budget-truncated sweep proves nothing. */
    bool exhausted() const { return stats_.exhausted != 0; }
    gpap_probe_stop stop() const { return static_cast<gpap_probe_stop>(stats_.stop); }

  private:
    friend class Prober;
    void adopt(gpap_probe *h, std::int32_t k) {
        if (h_) gpap_probe_free(h_);
        h_ = h;
        k_ = k;
        gpap_probe_stats_get(h_, &stats_);
        tuples_.assign(static_cast<std::size_t>(stats_.conflicts) * static_cast<std::size_t>(k_), 0);
        if (stats_.conflicts > 0)
            check(gpap_probe_conflicts(h_, tuples_.data(), stats_.conflicts),
                  "gpap_probe_conflicts");
    }
    void swap(Probe &o) noexcept {
        std::swap(h_, o.h_); std::swap(k_, o.k_);
        std::swap(stats_, o.stats_); tuples_.swap(o.tuples_);
    }
    gpap_probe *h_ = nullptr;
    std::int32_t k_ = 0;
    gpap_probe_stats stats_{};
    std::vector<std::int32_t> tuples_;
};

/** BCP-probes every k-subset of a literal set — the engine's core primitive in its combinatorial
 *  shape. Literals are DIMACS throughout.
 *      auto failed = prober.probe(1);   // failed literals; each clause(i) is an entailed unit
 *      gpap::ProbeBudget b; b.seconds = 5; b.max_tuples = 200000;
 *      auto pairs = prober.probe(support_literals, 2, b);
 *  `probe(k)` with no pool sweeps every literal the formula's own unit propagation left
 *  undetermined. This is the primitive, not a mining strategy: which widths to sweep, over which
 *  pool, whether to fold proven units back in, and which clauses to keep are an application's
 *  decisions (`gpap.mining` in Python is one such).
 *  Not thread-safe (it owns an Engine): one Prober per thread. */
class Prober {
  public:
    explicit Prober(const std::string &cnf_path, int device = 0)
        : Prober(Formula::from_dimacs(cnf_path), device) {}

    explicit Prober(Formula formula, int device = 0)
        : formula_(std::move(formula)), engine_(formula_, options(device)) {}

    /** True when the formula's own units already conflict — nothing is worth probing. */
    bool unsat() const { return engine_.flush_conflict() || formula_.trivially_unsat(); }

    /** DIMACS literals the unit flush left undetermined. */
    std::vector<std::int32_t> free_literals() {
        std::int32_t n = 0;
        check(gpap_free_literals(engine_.handle(), nullptr, &n), "gpap_free_literals");
        std::vector<std::int32_t> out(static_cast<std::size_t>(std::max(n, 0)));
        if (n > 0) check(gpap_free_literals(engine_.handle(), out.data(), &n), "gpap_free_literals");
        return out;
    }

    Probe probe(std::int32_t k, const ProbeBudget &budget = ProbeBudget()) {
        return sweep(nullptr, 0, k, budget);
    }
    /** Probe every `k`-subset of `pool` (DIMACS literals). An empty pool is rejected, not swept:
     *  `data()` is nullptr when empty and nullptr is the ABI's sentinel for every free literal,
     *  so a caller whose filter left nothing would get the largest possible sweep. */
    Probe probe(const std::vector<std::int32_t> &pool, std::int32_t k,
                const ProbeBudget &budget = ProbeBudget()) {
        if (pool.empty()) throw Error(GPAP_ERR_ARG, "gpap::Prober::probe: the pool is empty — "
                                      "call probe(k) if you meant every free literal");
        return sweep(pool.data(), static_cast<std::int32_t>(pool.size()), k, budget);
    }

    /** Exact candidate count for a sweep without generating it — check a ceiling against this,
     *  never against a materialized enumeration. Saturates at INT64_MAX instead of wrapping. */
    std::int64_t count(const std::vector<std::int32_t> &pool, std::int32_t k,
                       const ProbeBudget &budget = ProbeBudget()) {
        /* Load-bearing, as in probe(): without it an empty pool counts as the largest sweep. */
        if (pool.empty()) return 0;
        const gpap_probe_options o = to_options(budget);
        return gpap_probe_count(engine_.handle(), pool.data(),
                                static_cast<std::int32_t>(pool.size()), k, &o);
    }
    std::int64_t count(std::int32_t k, const ProbeBudget &budget = ProbeBudget()) {
        const gpap_probe_options o = to_options(budget);
        return gpap_probe_count(engine_.handle(), nullptr, 0, k, &o);
    }

    /** Probe an explicit (n, k) row-major matrix of DIMACS literals (caller-built candidates). */
    Probe probe_tuples(const std::vector<std::int32_t> &tuples, std::int32_t k,
                       const ProbeBudget &budget = ProbeBudget()) {
        const gpap_probe_options o = to_options(budget);
        gpap_probe *h = nullptr;
        check(gpap_probe_tuples(engine_.handle(), tuples.data(),
                                static_cast<std::int64_t>(tuples.size()) / std::max(k, 1), k, &o,
                                &h), "gpap_probe_tuples");
        Probe p;
        p.adopt(h, k);
        return p;
    }

    /** Fold proven units into the cached flush state, so later sweeps see a narrowed formula
     *  without a rebuild. Sound only if the literals really are entailed (probe at k = 1). */
    void assume(const std::vector<std::int32_t> &units) {
        if (units.empty()) return;
        check(gpap_assume_units(engine_.handle(), units.data(),
                                static_cast<std::int32_t>(units.size())), "gpap_assume_units");
    }

    const Formula &formula() const { return formula_; }
    Engine &engine() { return engine_; }

  private:
    static Options options(int device) {
        Options o;
        o.device = device;
        return o;
    }
    static gpap_probe_options to_options(const ProbeBudget &b) {
        gpap_probe_options o{};
        o.seconds = b.seconds;
        o.max_tuples = b.max_tuples;
        o.batch = b.batch;
        o.same_var = b.distinct_vars ? 0 : 1;
        o.refuse_over_max = b.refuse_over_max ? 1 : 0;
        return o;
    }
    Probe sweep(const std::int32_t *pool, std::int32_t n, std::int32_t k,
                const ProbeBudget &budget) {
        const gpap_probe_options o = to_options(budget);
        gpap_probe *h = nullptr;
        check(gpap_probe_combinations(engine_.handle(), pool, n, k, &o, &h),
              "gpap_probe_combinations");
        Probe p;
        p.adopt(h, k);
        return p;
    }

    Formula formula_;
    Engine engine_;
};

}  // namespace gpap

#endif /* GPAP_HPP */
