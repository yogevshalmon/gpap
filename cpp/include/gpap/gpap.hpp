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
 * many chunks as it takes, and — when SampleOptions::verify is set — re-checks on the device
 * that every row it returns is a real model.
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
enum class Mode { Auto = GPAP_MODE_AUTO, Scalar = GPAP_MODE_SCALAR, Wave = GPAP_MODE_WAVE };

/** Engine construction options; every default is the sensible one — the same defaults the raw C
 *  ABI gives a NULL options pointer, which is the point: `mode` is Auto on BOTH faces, so no
 *  consumer has to know to ask for the dispatch. (The C ABI's zero value was once Scalar, and
 *  this note used to tell C callers to pick a mode themselves; both are fixed.)
 *  The choice never changes results, only speed. */
struct Options {
    int device = 0;
    Mode mode = Mode::Auto;
    /** Pin the packed split (8 or 16) instead of deriving it from the formula. Testing only. */
    std::int32_t force_shift = 0;
    /** Warp-wide decision scan in the wave kernel: 0 = default (on), 1 = on, -1 = off. */
    std::int32_t vscan = 0;

    gpap_engine_options raw() const {
        /* VALUE-INITIALIZED, so a field added to gpap_engine_options later reaches the core as
         * its documented zero default rather than as stack garbage from this face. Same rule as
         * "`{0}` must equal NULL on every options struct" — the C caller gets it from the
         * struct's own zero, and this is how the C++ caller gets it. */
        gpap_engine_options o{};
        o.device = device;
        /* Auto goes THROUGH to the core, which re-resolves it on every reset(K). */
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

    // ---------------------------------------------------------------- configuration
    /** Pin the propagate variant, or hand the choice back with Mode::Auto. */
    void set_mode(Mode m) {
        check(gpap_engine_set_mode(h_, static_cast<gpap_mode>(m)), "gpap_engine_set_mode");
    }
    Mode mode() const { return static_cast<Mode>(gpap_engine_mode(h_)); }
    /** Is the engine still choosing its kernel per reset(K)? (Nothing pinned one — not
     *  options, not set_mode, not $GPAP_KERNEL.) */
    bool auto_dispatch() const { return gpap_engine_auto(h_) != 0; }
    std::int32_t shift() const { return gpap_engine_shift(h_); }
    std::int32_t k() const { return gpap_engine_k(h_); }
    std::int32_t num_vars() const { return gpap_engine_num_vars(h_); }
    std::int32_t num_clauses() const { return gpap_engine_num_clauses(h_); }
    bool flush_conflict() const { return gpap_engine_flush_conflict(h_) != 0; }

    // ---------------------------------------------------------------- entry points
    /** Which kernel a batch of `K` rows will run — a PURE query, unlike the Python twin's
     *  namesake it does not touch engine state. Nothing needs to call it before sampling:
     *  gpap_engine_reset resolves Mode::Auto itself, for every consumer of the ABI rather than
     *  for the wrappers that remember to ask (this USED to be where the rule was applied, and a
     *  core-side consumer, gpap_probe_combinations, duly ran SCALAR forever). Returns the pinned
     *  mode when one is pinned — by options, by set_mode, or by $GPAP_KERNEL.
     *  ⚠️ It used to be a second spelling of reset(): a caller who read "pure query" and used it
     *  mid-run silently discarded every pending readout slot and paid a full K*M broadcast. */
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

    /** apply() in CLEAN mode: identical flags and counters, but the launch restores its own
     *  packed rows to the baseline so the NEXT reset(K) skips the K*M broadcast. Only for
     *  callers that read flags/counters and nothing else — see gpap_engine_apply_clean in
     *  gpap.h for exactly which readouts see the baseline, which see the fixpoint, and why
     *  recache_flush() refuses afterwards. */
    void apply_clean(const std::vector<std::int64_t> &decisions, std::int32_t L) {
        require_matrix(decisions, L, "apply_clean");
        check(gpap_engine_apply_clean(h_, decisions.data(), L), "gpap_engine_apply_clean");
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

    /** How the resets executed: {full K*M broadcasts, broadcasts SKIPPED after a clean launch}.
     *  The lite count is the proof CLEAN mode engaged — a silent fallback measures as 1.00x. */
    std::pair<std::int64_t, std::int64_t> reset_stats() const {
        std::int64_t full = 0, lite = 0;
        gpap_engine_reset_stats(h_, &full, &lite);
        return {full, lite};
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

    /** How many selected rows have a clause with sat_count == 0. Must be 0.
     *  `max_rows = 0` (the default) checks EVERY selected row; a positive bound checks a prefix,
     *  and then a 0 answer only speaks for that prefix. */
    std::int32_t check_sound(std::int32_t slot = 0, std::int32_t max_rows = 0) {
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
        check(gpap_flags(h_, a.data(), c.data(), k()), "gpap_flags");
        return {std::move(a), std::move(c)};
    }

    /** (K,) unit-propagation implications of the last launch (assigned minus decisions). */
    std::vector<std::int32_t> num_propagated() {
        std::vector<std::int32_t> out(static_cast<size_t>(k()));
        check(gpap_num_propagated(h_, out.data(), k()), "gpap_num_propagated");
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
    /** MEMORY SAFETY, not politeness: the ABI stages K*L int64s from `decisions.data()` and has
     *  no size to check against, so a short vector is read past its end and the garbage becomes
     *  literal indices — and an index past 2N makes the kernel write into a LATER ROW's bitmap,
     *  which nothing downstream can detect (see gpap_engine_apply in gpap.h). The size is free
     *  here, so this face pays for it; the Python face already did. K is read from the CORE, so
     *  it is right even if the caller never called reset(). */
    void require_matrix(const std::vector<std::int64_t> &decisions, std::int32_t L,
                        const char *what) const {
        const std::size_t want = static_cast<std::size_t>(k()) * static_cast<std::size_t>(L);
        /* `<`, not `!=`: the ABI reads exactly K*L entries, so a LARGER buffer is safe, and
         * rejecting it breaks reusing one scratch vector sized for K_max across chunks. `L < 0`
         * is not redundant — `k() * size_t(L)` can WRAP small (k()=2, L=-2^63 gives want 0). */
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

/** Bytes available for new allocations on `device` (free VRAM minus a driver reserve).
 *  `fraction` is the ONLY headroom left in the sizing path now that gpap_row_bytes is the exact
 *  allocation list, and 0.80 is what EVERY other face uses — the sampler's own auto-K (cpp/src/sample.cpp),
 *  gpap_check_chunk_size, the probe batch, and Python's gpu_mem_available. This default used to
 *  be 0.75, so a tier-2 C++ caller writing the natural
 *  `compute_k_safe(info.num_long, info.num_vars, mem_available())` silently got a 6% smaller
 *  batch than every packaged path. One model means one number. */
inline std::int64_t mem_available(int device = 0, double fraction = 0.80) {
    return gpap_mem_available(device, fraction);
}

/** Largest K fitting `bytes` for a formula shape. See cpp/src/calibrate.cpp for the model.
 *  ⚠️ `num_long` is `gpap_formula_info::num_long`, NOT num_clauses — only length>=3 clauses
 *  carry per-row state. The total clause count is the natural misreading and it is SILENT: a
 *  1.78-2.36x smaller batch on circuit and sk_ CNFs, no error. */
inline std::int32_t compute_k_safe(std::int32_t num_long, std::int32_t N, std::int64_t bytes,
                                   std::int32_t cap = 0) {
    return gpap_compute_k_safe(num_long, N, bytes, cap);
}
/** Device bytes one row of engine state costs. `num_long` as in compute_k_safe above. */
inline std::int64_t row_bytes(std::int32_t num_long, std::int32_t N,
                              std::int32_t trail_cap = 0) {
    return gpap_row_bytes(num_long, N, trail_cap);
}

/** The default batch cap — the `cap` above, chosen for you. Device-independent. */
inline std::int32_t default_k_cap(int device = 0) { return gpap_default_k_cap(device); }

inline std::string version() { return gpap_version(); }
/** Identity of this exact BUILD ("<version>+<hex of the library's bytes>"). Key a cache
 *  of measurements on THIS, never on version() — see the C header. */
inline std::string build_id() { return gpap_build_id(); }

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
/** Which unassigned variable each decision picks. int-valued to match the ABI's
 *  gpap_sample_config.random_order (Random = 1, Structured = 0). */
enum class VarOrder : std::int32_t {
    Structured = 0,  ///< first free variable in index order — the gate-DAG order on a Tseitin
                     ///< CNF; wins search-heavy instances where a random order keeps picking
                     ///< variables propagation would have implied anyway
    Random = 1       ///< uniform-random free variable (CMSGen's Algorithm 1) — diverse; default
};

/** How the sampler should run. Every field has a working default; `seed` is the only one most
 *  callers set. Sizes and kernel choices are derived from the GPU and the formula.
 *
 *  A default run is DETERMINISTIC: same seed, same library build, same solutions, on any
 *  standard library (the RNG and shuffle are libgpap's own — see cpp/src/sample.cpp). A
 *  `portfolio` run is not machine-reproducible by design: the bandit selects configs on
 *  measured wall-clock throughput. */
struct SampleOptions {
    /** Reproducibility: a single-config run is a pure function of this seed. */
    std::int64_t seed = 0;
    /** Stop after this many seconds of SAMPLING wall even if the target is not met (0 = no
     *  limit). Delivery and verification are billed outside this clock. */
    double time_limit_s = 0.0;
    /** Stop after this many chunks (0 = unbounded) — the expert knob behind fixed-work runs. */
    std::int64_t max_chunks = 0;
    /** How many solutions to materialize on the host. 0 = count only, keep nothing (kept rows
     *  are gathered per chunk and billed to Solutions::deliver_seconds, never to the rate). */
    std::size_t max_keep = 100000;
    /** Which variables to report, as DIMACS variables. Empty = the formula's `c ind` independent
     *  support if it declares one, else every variable. */
    std::vector<std::int32_t> variables{};
    /** CUDA device ordinal. */
    int device = 0;
    /** THE PORTFOLIO: when non-empty, each chunk runs ONE of these configs at full K, raced as
     *  a throughput bandit in the core (cold-start each once, then decaying-epsilon-greedy on
     *  measured models/wall; kept rows are bucketed per config and only the winner's are
     *  returned). Empty (default) = the single config the three fields below describe — the
     *  deterministic, pinned-measurement mode. `gpap::default_portfolio()` is the same
     *  portfolio the Python pipeline runs by default; reach for it when a formula yields
     *  nothing under the default config (it is what samples the search-heavy instances the
     *  single config cannot). */
    std::vector<gpap_sample_config> portfolio{};
    /** Single-config mode: how decisions pick a variable. */
    VarOrder var_order = VarOrder::Random;
    /** Single-config mode: variables decided per round before propagating. 8 is a good default;
     *  1 lets propagation do more of the work and wins on tightly constrained formulas.
     *  B >= |support| makes the surviving rows exact rejection sampling — exactly uniform. */
    std::int32_t batch = 8;
    /** Single-config mode: in-kernel risk-aware polarity (prefer the value less likely to
     *  conflict). A big success-rate win on tight/deep circuits (b03-c25: 10% -> 88%) and NOT
     *  UNIFORM — it biases the distribution by construction, and Barbarik rejects it on 64 of
     *  the 65 circuit instances it can sample. Use it for "many solutions fast, distribution
     *  does not matter", never for a distribution claim. The Python face spells this opt-in
     *  `--allow-soft` and prints a warning. docs/findings/UNIFORMITY_FINDINGS.md 4b. */
    bool lookahead = false;
    /** If the run finds NOTHING in the first few chunks, try once more with `lookahead`
     *  FLIPPED before giving up. Rescues instances that yield only under the risk polarity
     *  (b12 unrolled 50 cycles: 0 either way, ~1100 models/s with it) — but that is exactly
     *  why it is OFF by default and named for what it opts into: the flip turns the
     *  NON-UNIFORM polarity on, silently, on the hard instances whose rows a caller is most
     *  likely to publish. Check Solutions::lookahead_used() whenever this is set. (It was
     *  `auto_retry = true`; renamed rather than reversed so an out-of-tree caller gets a
     *  compile error, not a silent default change — the library's own ground rule.) */
    bool retry_with_lookahead = false;
    /** Trajectories per chunk. 0 = size it to the free VRAM (recommended). */
    std::int32_t k = 0;
    /** Upper bound on the auto-sized K. **0 (default) = the library's batch cap**
     *  (`gpap_default_k_cap`, the same number on every GPU); a positive value is that cap
     *  verbatim; negative means no cap at all, leaving the analytic memory model as the only
     *  bound. */
    std::int32_t k_cap = 0;
    /** Re-check on the device that every emitted row really satisfies every clause.
     *  OFF by default: it is a self-test, not part of producing samples — a row is only counted
     *  when it is active and fully assigned, which already means no clause was emptied. Turn it
     *  on in tests, or in a run whose output will be published. When on it covers EVERY
     *  completed row, its wall is billed to Solutions::verify_seconds (outside the rate), and
     *  `Sampler::sample` throws if any row fails — after the run, with the evidence in
     *  Solutions::sound_rows_checked. Matches gpap.sampling's `verify_sound`. */
    bool verify = false;
    /** Some config must accumulate this many chunks before the target may stop a portfolio run
     *  (a single chunk's rate is ±1.9x run-to-run noise). 0 = the core default (1). */
    std::int32_t min_measure_chunks = 0;
};

/** The portfolio the Python pipeline races by default: (random, B=8) — the classic niche
 *  config; (random, B=1) — the success-leaning deep-circuit winner; (structured, B=1) — the
 *  search-heavy "service" family. Uniformity-safe: no soft-polarity member. */
inline std::vector<gpap_sample_config> default_portfolio() {
    return {{1, 8, 0}, {1, 1, 0}, {0, 1, 0}};
}

/** The result of `Sampler::sample`: a bit matrix of solutions plus what it cost to get them.
 *  Row i, column j is the value of variable `variables()[j]` in the i-th solution. */
class Solutions {
  public:
    /** Solutions materialized here (<= found(), capped by SampleOptions::max_keep). All from
     *  ONE config — the winner's bucket, never a mixture of the bandit's arms. */
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
    std::int64_t found() const noexcept { return stats_.n_models; }
    /** Trajectories run to find them. */
    std::int64_t drawn() const noexcept { return stats_.attempts; }
    /** Chunks (kernel launches) used. */
    std::int64_t chunks() const noexcept { return stats_.chunks; }

    /** SAMPLING wall: the chunk loop only. Copying solutions to the host and the optional
     *  soundness re-check are billed separately below and EXCLUDED here, so neither asking for
     *  rows nor asking for verification can move the rate — the same contract gpap.sampling
     *  keeps as `out_project_s` / `verify_sound_s`. */
    double seconds() const noexcept { return stats_.sampling_s; }
    /** Host copy-out of the kept rows — delivery, not sampling. Grows with max_keep and with
     *  the number of reported variables. */
    double deliver_seconds() const noexcept { return stats_.deliver_s; }
    /** The SampleOptions::verify re-check — a self-test, not sampling. 0 when verify is off. */
    double verify_seconds() const noexcept { return stats_.verify_s; }
    /** Everything the call took, including delivery and verification. */
    double wall_seconds() const noexcept {
        return stats_.sampling_s + stats_.deliver_s + stats_.verify_s;
    }
    /** Solutions per second, over seconds(). */
    double rate() const noexcept {
        return stats_.sampling_s > 0 ? double(stats_.n_models) / stats_.sampling_s : 0.0;
    }
    /** Fraction of trajectories that reached a solution. */
    double success_rate() const noexcept {
        return stats_.attempts > 0 ? double(stats_.n_models) / double(stats_.attempts) : 0.0;
    }

    /** Why the run ended — which safety valve fired, if one did. GPAP_SAMPLE_STOP_UNSAT is a
     *  PROOF (the construction flush conflicted); ZERO_YIELD is only an absence of evidence. */
    gpap_sample_stop stop() const noexcept { return static_cast<gpap_sample_stop>(stats_.stop); }
    /** True when the run stopped on the time limit rather than reaching the target. */
    bool timed_out() const noexcept { return stats_.timed_out != 0; }
    /** Whether the retry_with_lookahead flip fired, and the soft flag the winning config (or
     *  the final retry window) actually ran with — a distribution claim needs it false. */
    bool retried() const noexcept { return stats_.retried != 0; }
    bool lookahead_used() const noexcept { return stats_.lookahead_used != 0; }

    /** Rows the verify re-check covered — the EVIDENCE behind a soundness claim, and what
     *  distinguishes "verified every one of a million rows" from "verify was on but nothing
     *  ever completed, so it never ran". 0 with verify on is NOT a passing verdict. */
    std::int64_t sound_rows_checked() const noexcept { return stats_.sound_rows_checked; }

    /** Portfolio breakdown: one entry per config passed to the run (a single entry for the
     *  default single-config mode). `best_config()` is the winner whose rows fill bits(). */
    std::size_t n_configs() const noexcept { return cstats_.size(); }
    const gpap_config_stats &config_stats(std::size_t i) const { return cstats_.at(i); }
    std::int32_t best_config() const noexcept { return stats_.best_config; }
    std::int32_t dominant_config() const noexcept { return stats_.dominant_config; }
    /** The whole core stats record, verbatim. */
    const gpap_sample_stats &stats() const noexcept { return stats_; }

  private:
    friend class Sampler;
    std::vector<std::uint8_t> bits_;
    std::vector<std::int32_t> vars_;
    std::size_t n_kept_ = 0;
    gpap_sample_stats stats_{};
    std::vector<gpap_config_stats> cstats_;
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
 *  THE LOOP LIVES IN THE CORE (`gpap_sampler_*`, cpp/src/sample.cpp), shared verbatim with the
 *  Python face — this class is marshalling: DIMACS ids, RAII, exceptions. Construction picks
 *  the batch size K from free VRAM, the propagate kernel from the formula, and the decision
 *  order from the formula's independent support. Read them back with k(), mode() and
 *  variables(); override them through SampleOptions if you must. If a formula yields nothing
 *  under the default config, run `SampleOptions::portfolio = gpap::default_portfolio()` — the
 *  adaptive portfolio the Python pipeline uses, which also samples the search-heavy instances
 *  a single random-order config cannot.
 *
 *  This is a SAMPLER, not a decision procedure: `found() == 0` with stop() != UNSAT means these
 *  trajectories did not reach a solution, never that the formula is unsatisfiable (UNSAT is
 *  reported only when construction PROVED it). Use a CDCL solver for the question.
 *
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

        /* --- the core handle --------------------------------------------------------- */
        run_opts_ = gpap_sampler_options{};          /* {0} is every default, by the ground rule */
        run_opts_.max_chunks = opts_.max_chunks;
        run_opts_.keep_cap = static_cast<std::int64_t>(opts_.max_keep);
        run_opts_.time_limit_s = opts_.time_limit_s;
        run_opts_.min_measure_chunks = opts_.min_measure_chunks;
        run_opts_.verify = opts_.verify ? 1 : 0;
        run_opts_.retry_with_lookahead = opts_.retry_with_lookahead ? 1 : 0;
        check(gpap_sampler_create(engine_.handle(), opts_.seed, opts_.k_cap, &h_),
              "gpap_sampler_create");
        /* Everything past the create can throw -- an out-of-range SampleOptions::variables is
         * plain user error, and the support's pair mask is a device allocation. A constructor
         * that throws never runs ~Sampler(), so without this the handle leaks AND the borrowed
         * engine keeps the kernel this sampler pinned. */
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

    /** Draw solutions until `want` have been found, a safety valve fires, or the formula is
     *  proved hopeless — `Solutions::stop()` says which. `want` is a target, not a promise. */
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

    // ------------------------------------------------------------------ what it chose
    /** Trajectories per chunk. */
    std::int32_t k() const noexcept { return gpap_sampler_k(h_); }
    /** The propagate variant in use. */
    Mode mode() const { return engine_.mode(); }
    /** Variables reported in the solution columns (DIMACS). */
    const std::vector<std::int32_t> &variables() const noexcept { return vars_; }
    /** Size of decision phase 1 — the independent support, or all variables if none declared. */
    std::int32_t num_support() const noexcept { return n_support_; }
    /** Work done so far: literals assigned, decisions, occurrence entries, launches. */
    gpap_counters counters() const { return engine_.counters(); }
    /** Restore the RNG to a known point — a run is then a function of the seed alone again. */
    void reseed(std::int64_t seed) { check(gpap_sampler_reseed(h_, seed), "gpap_sampler_reseed"); }
    /** Pin the batch (k <= 0 restores the calibrated value) — the cross-run A/B lever. */
    void set_k(std::int32_t k) { check(gpap_sampler_set_k(h_, k), "gpap_sampler_set_k"); }

    /** Escape hatches to tier 2 — the same objects the sampler is driving. */
    Engine &engine() noexcept { return engine_; }
    const Formula &formula() const noexcept { return formula_; }

  private:
    /* The fallible half of construction; see the try/catch in the ctor. */
    void init_from_options() {
        if (opts_.k > 0) check(gpap_sampler_set_k(h_, opts_.k), "gpap_sampler_set_k");

        /* --- decision phase 1: the declared independent support ---------------------- */
        /* `support` is 0-based (tier 2). A `c ind` line naming a variable the header never
         * declared is dropped rather than trusted — it would otherwise become an out-of-range
         * column. The head/tail split itself is the core's. */
        std::vector<std::int32_t> support;
        for (std::int32_t v : formula_.support())
            if (v >= 0 && v < n_vars_) support.push_back(v);
        if (!support.empty())
            check(gpap_sampler_set_support(h_, support.data(),
                                           static_cast<std::int32_t>(support.size())),
                  "gpap_sampler_set_support");
        n_support_ = support.empty() ? n_vars_ : static_cast<std::int32_t>(support.size());

        /* --- what to report ---------------------------------------------------------- */
        /* Default: the support (projected sampling), else all variables — DIMACS out here,
         * 0-based columns into the core. */
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

    /* Declaration order is the destruction contract: the handle is declared LAST so it is freed
     * FIRST — gpap_sampler_free touches the engine it borrowed. */
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
    }

    /** Propagate every set to a fixpoint. `sets` hold DIMACS literals (+v / -v).
     *
     *  ANY NUMBER of sets: they are split into GPU-sized chunks internally and the results come
     *  back in input order, so how many launches it took is not observable. `chunk_size()` says
     *  what one launch holds if you want to know, but nothing here requires you to.
     *
     *  The chunk loop itself is `gpap_check_sets` in the CORE (cpp/src/check.cpp) — this method
     *  only flattens the sets and shapes the results, so a Python caller and a C++ caller get
     *  the same validation, the same chunking and the same padding.
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

    /** Assumption sets one launch holds — `gpap_check_chunk_size`, the core's one model
     *  (the engine's exact per-row state plus the decision matrix this entry point adds). */
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

// ---------------------------------------------------------------------------- probing

/** When a probe sweep should stop. Every field is optional; the first to bind wins. */
struct ProbeBudget {
    /** Wall-clock ceiling, checked BETWEEN chunks (a chunk is the smallest unit of GPU work). */
    double seconds = 0.0;
    /** How many candidates to probe at most, checked against an EXACT count before anything is
     *  generated. A WORK BUDGET: the sweep probes this many and reports `exhausted() == false`.
     *  Every conflict found is still a genuine entailed clause. */
    std::int64_t max_tuples = 0;
    /** Refuse the sweep outright when it exceeds `max_tuples`, instead of probing that many.
     *  For a caller that wants an exhaustive answer or none at all. */
    bool refuse_over_max = false;
    /** Tuples per launch; 0 sizes it from free VRAM with the library's one memory model. */
    std::int32_t batch = 0;
    /** Drop tuples holding two literals of the SAME variable — they conflict for reasons that
     *  have nothing to do with the formula. */
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

    /** Number of conflicting tuples. */
    std::int64_t size() const { return stats_.conflicts; }
    /** Tuple width. */
    std::int32_t width() const { return k_; }
    /** Conflicting tuple `i` as DIMACS literals — assuming these drove BCP to a conflict, so the
     *  formula entails the negation of their conjunction. */
    std::vector<std::int32_t> operator[](std::int64_t i) const {
        const std::size_t off = static_cast<std::size_t>(i) * static_cast<std::size_t>(k_);
        return {tuples_.begin() + static_cast<std::ptrdiff_t>(off),
                tuples_.begin() + static_cast<std::ptrdiff_t>(off + static_cast<std::size_t>(k_))};
    }
    /** The CLAUSE tuple `i` proves: the negation of its literals. Mining is this, plus a
     *  strategy for which pools and widths to sweep. */
    std::vector<std::int32_t> clause(std::int64_t i) const {
        std::vector<std::int32_t> c = (*this)[i];
        for (auto &l : c) l = -l;
        return c;
    }

    /** Candidates the full enumeration has. */
    std::int64_t total() const { return stats_.total; }
    /** Candidates actually probed — a prefix of the enumeration. */
    std::int64_t probed() const { return stats_.probed; }
    double seconds() const { return stats_.seconds; }
    /** TRUE iff every candidate was probed. This is what makes a NEGATIVE result meaningful: an
     *  exhausted sweep that found nothing PROVES no entailed clause of this width exists (that
     *  BCP can see), where a sweep cut short by a budget proves nothing at all. */
    bool exhausted() const { return stats_.exhausted != 0; }
    /** Why it stopped. */
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
 *  shape.
 *
 *      gpap::Prober prober("formula.cnf");
 *      auto failed = prober.probe(1);                  // failed literals (the BCP backbone)
 *      for (std::int64_t i = 0; i < failed.size(); ++i)
 *          use(failed.clause(i));                      // each is an entailed unit
 *
 *      gpap::ProbeBudget b; b.seconds = 5; b.max_tuples = 200000;
 *      auto pairs = prober.probe(support_literals, 2, b);
 *
 *  `probe(k)` with no pool sweeps every literal the formula's own unit propagation left
 *  undetermined — probing a settled literal tells you nothing. Literals are DIMACS throughout.
 *
 *  This is the PRIMITIVE, not a mining strategy: which widths to sweep, over which pool, whether
 *  to fold proven units back in and re-sweep, and which of the resulting clauses to keep
 *  are decisions an application makes. `gpap.mining` (Python) is one such application; a C++
 *  consumer builds its own on top of `probe` and `assume`.
 *
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
        /* `n` is now the CAPACITY on the way in and the count on the way out — the same value
           here, since the vector was sized from the query. */
        if (n > 0) check(gpap_free_literals(engine_.handle(), out.data(), &n), "gpap_free_literals");
        return out;
    }

    /** Probe every `k`-subset of every free literal. */
    Probe probe(std::int32_t k, const ProbeBudget &budget = ProbeBudget()) {
        return sweep(nullptr, 0, k, budget);
    }
    /** Probe every `k`-subset of `pool` (DIMACS literals). An EMPTY pool is rejected rather than
     *  silently swept: `std::vector::data()` is nullptr when empty, and nullptr is the ABI's
     *  sentinel for "every free literal" — so a caller whose filter left nothing would have got
     *  the LARGEST possible sweep (~1e11 tuples at k=3 on an unrolled circuit) instead of none. */
    Probe probe(const std::vector<std::int32_t> &pool, std::int32_t k,
                const ProbeBudget &budget = ProbeBudget()) {
        if (pool.empty()) throw Error(GPAP_ERR_ARG, "gpap::Prober::probe: the pool is empty — "
                                      "call probe(k) if you meant every free literal");
        return sweep(pool.data(), static_cast<std::int32_t>(pool.size()), k, budget);
    }

    /** Exact candidate count for a sweep, WITHOUT generating it — check a ceiling against this
     *  rather than against a materialized enumeration, which for a wide pool at k=3 is terabytes.
     *  Saturates at INT64_MAX instead of wrapping. */
    std::int64_t count(const std::vector<std::int32_t> &pool, std::int32_t k,
                       const ProbeBudget &budget = ProbeBudget()) {
        /* LOAD-BEARING, for the same reason probe() throws on an empty pool: `std::vector::data()`
         * is nullptr when empty, and nullptr is the ABI's sentinel for EVERY FREE LITERAL — so
         * without this an empty pool would be counted as the largest possible sweep instead of
         * the smallest. Do not "simplify" it into the call below. The pair is deliberately
         * asymmetric: counting nothing is 0, probing nothing is a caller mistake worth an error. */
        if (pool.empty()) return 0;
        const gpap_probe_options o = to_options(budget);
        return gpap_probe_count(engine_.handle(), pool.data(),
                                static_cast<std::int32_t>(pool.size()), k, &o);
    }
    std::int64_t count(std::int32_t k, const ProbeBudget &budget = ProbeBudget()) {
        const gpap_probe_options o = to_options(budget);
        return gpap_probe_count(engine_.handle(), nullptr, 0, k, &o);
    }

    /** Probe an explicit (n, k) row-major matrix of DIMACS literals — for a caller that builds
     *  its own candidates (filtered, weighted, or from somewhere else entirely). */
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

    /** Fold PROVEN units into the cached flush state, so later sweeps see a narrowed formula
     *  without it being rebuilt. Sound only if the literals really are entailed — probing at
     *  k = 1 is one way to have earned that. */
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
