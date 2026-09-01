/* Solution sampling: THE chunk loop and the adaptive config bandit behind both sampler faces.
 *
 * Entry point 2 is reset(K) + trajectory — one launch, K randomized decide-and-propagate
 * trajectories. What every caller then wants is the same packaging: chunks until a target or a
 * valve, a portfolio of decision configs raced by measured throughput, completion counted on
 * device, kept rows gathered per config, and delivery/self-test wall billed OUTSIDE the
 * sampling clock. That loop used to be written twice, once per face (gpap::Sampler in the
 * header and Python's GPUSampler.sample), and the two had drifted much further than the
 * checking loop ever did — the C++ face had no structured order and no portfolio (instances
 * the Python face sampled fine returned NOTHING), charged delivery and verification to the
 * sampling clock, and defaulted into the non-uniform soft retry. Same lesson as check.cpp and
 * probe.cpp: a rule callers can observe must have one home.
 *
 * Everything here goes through the public ABI calls the faces themselves used (reset /
 * trajectory / select_complete / var_bits / check_sound / counters), so this file adds
 * packaging, not a second engine path. The sampler BORROWS its engine, exactly as
 * gpap_check_sets does.
 *
 * DETERMINISM IS PART OF THE CONTRACT: the RNG is the library's own (splitmix64-seeded
 * xoshiro256**), the shuffle is an in-house Fisher-Yates, and uniform draws use masked
 * rejection — never std::mt19937 / std::shuffle / uniform_int_distribution, whose outputs are
 * implementation-defined and would make "same seed, same solutions" true on one stdlib and
 * false on another. A single-config run is a pure function of (formula, options, seed).
 * A PORTFOLIO run is not cross-machine reproducible by design: the bandit selects on measured
 * wall-clock throughput. Two streams, as the Python face had: `rng` drives sampling (the
 * per-chunk shuffle and trajectory seed), `bandit_rng` drives the epsilon-greedy draws — so
 * WHICH config the bandit explores cannot shift the sampling stream.
 */
#include "gpap_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

using gpap::set_error;

namespace {

/* ------------------------------------------------------------------ RNG (deterministic) */

uint64_t splitmix64(uint64_t &x) {
    x += 0x9E3779B97F4A7C15ull;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

struct Rng {                                   /* xoshiro256** (public domain reference) */
    uint64_t s[4] = {1, 2, 3, 4};

    void seed(uint64_t v) {
        uint64_t x = v;
        for (auto &w : s) w = splitmix64(x);
    }
    static uint64_t rotl(uint64_t v, int k) { return (v << k) | (v >> (64 - k)); }
    uint64_t next() {
        const uint64_t out = rotl(s[1] * 5, 7) * 9;
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t;    s[3] = rotl(s[3], 45);
        return out;
    }
    /* Uniform in [0, n): masked rejection — unbiased, portable, deterministic. */
    uint64_t below(uint64_t n) {
        if (n <= 1) return 0;
        uint64_t mask = n - 1;
        mask |= mask >> 1;  mask |= mask >> 2;  mask |= mask >> 4;
        mask |= mask >> 8;  mask |= mask >> 16; mask |= mask >> 32;
        for (;;) {
            const uint64_t v = next() & mask;
            if (v < n) return v;
        }
    }
    double unit() { return (double)(next() >> 11) * 0x1.0p-53; }   /* [0, 1) */

    void shuffle(int32_t *a, size_t n) {       /* Fisher-Yates, high-to-low */
        for (size_t i = n; i > 1; --i) std::swap(a[i - 1], a[(size_t)below(i)]);
    }
};

/* The two seeds one options.seed expands into. The bandit stream is offset so the two cannot
 * collide at small seeds (the Python face kept the same split for the same reason). */
constexpr uint64_t kBanditSeedXor = 0xB0BA5EED0117ull;

/* Share of free VRAM a batch may occupy — the ONLY headroom in the sizing path (the row model
 * is exact; see calibrate.cpp). Same constant on every face. */
constexpr double kMemFraction = 0.80;

double seconds_between(std::chrono::steady_clock::time_point a,
                       std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

}  // namespace

/* One sampler: borrowed engine + decision phases + per-config buckets. Opaque to callers. */
struct gpap_sampler {
    gpap_engine *e = nullptr;
    gpap_sampler_options opt{};          /* the CURRENT run's knobs (set per gpap_sampler_run) */
    int32_t k_cap = 0;

    int32_t K_auto = 0;                  /* calibrated at create */
    int32_t K = 0;                       /* effective (set_k overrides) */
    int we_pinned_mode = 0;              /* we resolved AUTO -> hand it back on free */

    Rng rng, bandit_rng;

    /* Decision phases, 0-based, ascending (the STRUCTURED order); a chunk copies and, for a
     * random-order config, shuffles each phase in place. */
    std::vector<int32_t> head0, tail0, order;
    int32_t n_support = 0;               /* head0.size(); == num_vars when no support is set */
    int has_support = 0;

    int32_t mask_all = -1;               /* completion over every variable */
    int32_t mask_sup = -1;               /* completion over phase 1 (== mask_all, no support) */

    std::vector<int32_t> cols;           /* kept-row columns; empty = every variable */

    /* Last run. `run_cols_n` snapshots the column width the buckets were gathered with, so a
     * set_columns between run and bits cannot misshape the readout. */
    std::vector<gpap_config_stats> cstats;
    std::vector<std::vector<uint8_t>> buckets;
    std::vector<gpap_sample_config> run_cfgs;
    gpap_sample_stats stats{};
    int32_t run_cols_n = 0;
    int has_run = 0;
};

namespace {

int32_t auto_k(gpap_engine *e, int32_t k_cap) {
    const int32_t cap = k_cap == 0 ? gpap_default_k_cap(e->device_ordinal) : k_cap;
    const int64_t bytes = gpap_mem_available(e->device_ordinal, kMemFraction);
    /* num_long, never num_clauses: only length>=3 clauses carry per-row state. */
    return gpap_compute_k_safe(e->num_long, e->num_vars, bytes, cap);
}

/* Pin the propagate kernel to the batch the chunks will run, so a short reset elsewhere cannot
 * switch kernels mid-measurement — the same rule the Python face applied. Only when the choice
 * is ours to make: an explicit pin ($GPAP_KERNEL, gpap_engine_set_mode) is never overridden,
 * which is what gpap_engine_auto reports. Once WE pinned, engine_auto goes 0, so re-pins after
 * set_k key off we_pinned_mode instead. */
gpap_status pin_mode(gpap_sampler *s) {
    if (!gpap_engine_auto(s->e) && !s->we_pinned_mode) return GPAP_OK;
    s->we_pinned_mode = 1;
    const int wave = gpap_dispatch_wave(s->e->num_clauses, s->K, s->e->device_ordinal);
    return gpap_engine_set_mode(s->e, wave ? GPAP_MODE_WAVE : GPAP_MODE_SCALAR);
}

void rebuild_phases(gpap_sampler *s) {
    const int32_t N = s->e->num_vars;
    s->head0.clear();
    s->tail0.clear();
    if (!s->has_support) {
        /* No support: ONE phase over every variable (phase 2 empty), completion == support
         * completion by definition. */
        s->head0.resize((size_t)N);
        for (int32_t v = 0; v < N; ++v) s->head0[(size_t)v] = v;
    }
    s->n_support = (int32_t)s->head0.size();
    s->order.reserve((size_t)N);
}

double cfg_rate(const gpap_config_stats &c) {
    return c.wall_s > 0 ? (double)c.models / c.wall_s : 0.0;
}

int32_t best_by_rate(const std::vector<gpap_config_stats> &cs) {
    int32_t best = 0;
    for (int32_t i = 1; i < (int32_t)cs.size(); ++i)
        if (cfg_rate(cs[(size_t)i]) > cfg_rate(cs[(size_t)best])) best = i;
    return best;
}

}  // namespace

extern "C" gpap_status gpap_sampler_create(gpap_engine *e, int64_t seed, int32_t k_cap,
                                           gpap_sampler **out) {
    if (!e || !out) return GPAP_ERR_ARG;
    *out = nullptr;
    return gpap::guard_entry("gpap_sampler_create", [&]() -> gpap_status {
        gpap_sampler *s = new gpap_sampler();
        s->e = e;
        s->k_cap = k_cap;
        s->rng.seed((uint64_t)seed);
        s->bandit_rng.seed((uint64_t)seed ^ kBanditSeedXor);
        s->K_auto = auto_k(e, k_cap);
        s->K = s->K_auto;
        rebuild_phases(s);
        gpap_status st = gpap_pair_mask_create(e, nullptr, 0, &s->mask_all);
        if (st == GPAP_OK) {
            s->mask_sup = s->mask_all;              /* no support yet: the two tests coincide */
            st = pin_mode(s);
        }
        if (st != GPAP_OK) {
            delete s;
            return st;
        }
        *out = s;
        return GPAP_OK;
    });
}

extern "C" void gpap_sampler_free(gpap_sampler *s) {
    if (!s) return;
    /* Borrower hygiene: if WE resolved AUTO into a concrete kernel, hand the choice back —
     * set_mode(AUTO) re-enables per-reset dispatch (and keeps any $GPAP_KERNEL pin, see
     * gpap_engine_set_mode). A pin the CALLER made is left exactly as found. */
    if (s->we_pinned_mode) (void)gpap_engine_set_mode(s->e, GPAP_MODE_AUTO);
    delete s;
}

extern "C" int32_t gpap_sampler_k(const gpap_sampler *s) { return s ? s->K : 0; }

extern "C" gpap_status gpap_sampler_set_k(gpap_sampler *s, int32_t k) {
    if (!s) return GPAP_ERR_ARG;
    s->K = k > 0 ? k : s->K_auto;
    return pin_mode(s);
}

extern "C" gpap_status gpap_sampler_reseed(gpap_sampler *s, int64_t seed) {
    if (!s) return GPAP_ERR_ARG;
    s->rng.seed((uint64_t)seed);
    s->bandit_rng.seed((uint64_t)seed ^ kBanditSeedXor);
    return GPAP_OK;
}

extern "C" gpap_status gpap_sampler_set_support(gpap_sampler *s, const int32_t *vars,
                                                int32_t n) {
    if (!s || n < 0 || (n > 0 && !vars)) return GPAP_ERR_ARG;
    return gpap::guard_entry("gpap_sampler_set_support", [&]() -> gpap_status {
        const int32_t N = s->e->num_vars;
        if (n == 0) {                               /* back to "no support" */
            s->has_support = 0;
            s->mask_sup = s->mask_all;
            rebuild_phases(s);
            return GPAP_OK;
        }
        /* Deduplicate + validate through one bitmap; keep phases ASCENDING (the structured
         * order), a random-order chunk shuffles its copy. */
        std::vector<char> in((size_t)N, 0);
        for (int32_t i = 0; i < n; ++i) {
            const int32_t v = vars[i];
            if (v < 0 || v >= N) {
                set_error("gpap_sampler_set_support: variable %d out of range [0, %d) — "
                          "indices are 0-based here (tier 2), DIMACS minus one", v, N);
                return GPAP_ERR_ARG;
            }
            in[(size_t)v] = 1;
        }
        s->head0.clear();
        s->tail0.clear();
        for (int32_t v = 0; v < N; ++v) (in[(size_t)v] ? s->head0 : s->tail0).push_back(v);
        s->has_support = 1;
        s->n_support = (int32_t)s->head0.size();
        /* One mask per support set — pair masks live until the engine dies (see gpap.h), so a
         * caller that re-sets the support in a loop would leak ceil(2N/32) words each time.
         * Every in-tree face sets it once, right after create. */
        GPAP_TRY(gpap_pair_mask_create(s->e, s->head0.data(), s->n_support, &s->mask_sup));
        return GPAP_OK;
    });
}

extern "C" gpap_status gpap_sampler_set_columns(gpap_sampler *s, const int32_t *cols,
                                                int32_t n) {
    if (!s || n < 0 || (n > 0 && !cols)) return GPAP_ERR_ARG;
    return gpap::guard_entry("gpap_sampler_set_columns", [&]() -> gpap_status {
        for (int32_t i = 0; i < n; ++i)
            if (cols[i] < 0 || cols[i] >= s->e->num_vars) {
                set_error("gpap_sampler_set_columns: variable %d out of range [0, %d)",
                          cols[i], s->e->num_vars);
                return GPAP_ERR_ARG;
            }
        s->cols.assign(cols, cols + n);             /* takes effect at the next run */
        return GPAP_OK;
    });
}

extern "C" gpap_status gpap_sampler_run(gpap_sampler *s, int64_t n_target,
                                        const gpap_sample_config *configs, int32_t n_configs,
                                        const gpap_sampler_options *opts,
                                        gpap_sample_stats *stats_out) {
    if (!s || !configs || n_configs < 1) return GPAP_ERR_ARG;
    s->opt = opts ? *opts : gpap_sampler_options{};   /* {0} == NULL == the defaults */
    if (n_configs > 4096) {                        /* a portfolio, not a candidate sweep */
        set_error("gpap_sampler_run: %d configs — the portfolio races configs by measured "
                  "throughput, it is not a probe primitive (see gpap_probe_combinations)",
                  n_configs);
        return GPAP_ERR_ARG;
    }
    return gpap::guard_entry("gpap_sampler_run", [&]() -> gpap_status {
        using clock = std::chrono::steady_clock;

        /* Normalize once: batch clamps to >= 1, flags to 0/1. */
        s->run_cfgs.assign(configs, configs + n_configs);
        for (auto &c : s->run_cfgs) {
            c.batch = std::max<int32_t>(1, c.batch);
            c.random_order = c.random_order ? 1 : 0;
            c.soft = c.soft ? 1 : 0;
        }
        s->cstats.assign((size_t)n_configs, gpap_config_stats{});
        s->buckets.assign((size_t)n_configs, {});
        s->stats = gpap_sample_stats{};
        s->has_run = 1;

        /* Kept-row columns, snapshotted for this run. Empty = every variable. */
        const int32_t N = s->e->num_vars;
        std::vector<int32_t> all_cols;
        const int32_t *cols_p = s->cols.data();
        int32_t n_cols = (int32_t)s->cols.size();
        if (n_cols == 0) {
            all_cols.resize((size_t)N);
            for (int32_t v = 0; v < N; ++v) all_cols[(size_t)v] = v;
            cols_p = all_cols.data();
            n_cols = N;
        }
        s->run_cols_n = n_cols;

        gpap_sample_stats &st = s->stats;
        st.best_config = st.dominant_config = 0;
        st.stop = GPAP_SAMPLE_STOP_TARGET;

        if (gpap_engine_flush_conflict(s->e)) {
            /* The construction units flush already conflicted: PROVABLY no models. Distinct
             * from "none found" — the faces report it as UNSAT, not as a sampler failure. */
            st.stop = GPAP_SAMPLE_STOP_UNSAT;
            if (stats_out) *stats_out = st;
            return GPAP_OK;
        }

        const bool adaptive = n_configs > 1;
        const int32_t mmc = std::max<int32_t>(1, s->opt.min_measure_chunks);
        const int32_t grace = s->opt.grace_chunks > 0 ? s->opt.grace_chunks : 2;
        const int64_t cold_size = adaptive ? n_configs : 1;
        const int64_t keep_cap = s->opt.keep_cap;

        int64_t rounds_sum = 0;
        int64_t window_start = 0;               /* first chunk of the current retry window */
        int retry_soft = 0;                     /* single-config retry state (see below) */

        const auto t0 = clock::now();
        auto sampling_s = [&] {
            return seconds_between(t0, clock::now()) - st.deliver_s - st.verify_s;
        };

        for (;;) {
            /* ---- stop valves (between chunks) -------------------------------------- */
            bool cold = false;
            if (adaptive)
                for (const auto &c : s->cstats) cold |= (c.chunks == 0);
            int64_t max_measured = 0;
            for (const auto &c : s->cstats) max_measured = std::max(max_measured, c.chunks);
            /* Target met — but the bandit must first measure every config once (a big-K GPU
             * meets the target in the cold start, and an unmeasured config is undiscoverable)
             * and pile min_measure_chunks onto SOME config (a single chunk's rate is ±1.9x
             * run-to-run noise; the headline would rest on it). */
            const bool enough = !adaptive || max_measured >= mmc;
            if (!cold && enough && st.n_models >= n_target) break;               /* TARGET */
            if (s->opt.max_chunks > 0 && st.chunks >= s->opt.max_chunks) {
                st.stop = GPAP_SAMPLE_STOP_CHUNKS;
                break;
            }
            if (s->opt.time_limit_s > 0 && sampling_s() >= s->opt.time_limit_s) {
                st.stop = GPAP_SAMPLE_STOP_TIME;
                st.timed_out = st.n_models < n_target ? 1 : 0;
                break;
            }
            /* Zero yield: NOTHING after the cold start plus a grace window — the target is
             * unreachable ("this chunk was empty" is not a signal at a 0.01% success rate).
             * The lookahead retry (opt-in, SINGLE config only — a portfolio says which configs
             * may run, and soft belongs in it or not at all) flips the risk polarity once. */
            if (st.n_models == 0 && !cold && st.chunks - window_start >= cold_size + grace) {
                if (s->opt.retry_with_lookahead && !adaptive && !st.retried) {
                    st.retried = 1;
                    retry_soft = 1;
                    window_start = st.chunks;
                } else {
                    st.stop = GPAP_SAMPLE_STOP_ZERO_YIELD;
                    break;
                }
            }

            /* ---- pick the chunk's config ------------------------------------------- */
            int32_t ci = 0;
            if (adaptive) {
                if (cold) {
                    for (int32_t i = 0; i < n_configs; ++i)
                        if (s->cstats[(size_t)i].chunks == 0) { ci = i; break; }
                } else if (st.n_models >= n_target) {
                    /* Past the target, short of the window: EXPLOIT, so the winner's rate is
                     * measured on back-to-back chunks rather than diluted by exploration. */
                    ci = best_by_rate(s->cstats);
                } else {
                    /* Decaying epsilon-greedy, exploration PLAUSIBILITY-FILTERED to configs
                     * within 20x of the best measured rate: run-to-run noise is ~1.9x, so a
                     * 20x-slower config is dead on this formula, not noisy — it keeps exactly
                     * its one cold-start chunk. */
                    const int32_t best = best_by_rate(s->cstats);
                    ci = best;
                    const double eps =
                        std::max(0.05, 1.0 / std::sqrt((double)(st.chunks + 1)));
                    if (s->bandit_rng.unit() < eps) {
                        int32_t cands[4096];
                        int32_t n_cand = 0;
                        const double floor_rate = cfg_rate(s->cstats[(size_t)best]) / 20.0;
                        for (int32_t i = 0; i < n_configs; ++i)
                            if (i != best && cfg_rate(s->cstats[(size_t)i]) >= floor_rate)
                                cands[n_cand++] = i;
                        if (n_cand > 0)
                            ci = cands[s->bandit_rng.below((uint64_t)n_cand)];
                    }
                }
            }
            const gpap_sample_config &cfg = s->run_cfgs[(size_t)ci];
            const int soft = cfg.soft || retry_soft;

            /* ---- one chunk --------------------------------------------------------- */
            /* Order: phase 1 (support) then phase 2, each phase shuffled for a random-order
             * config and left ascending (the gate-DAG order) for a structured one. The kernel
             * never straddles the boundary; deciding a variable propagation would have implied
             * is a wasted decision AND a chance to conflict, which is why the split exists. */
            s->order.assign(s->head0.begin(), s->head0.end());
            const int32_t P_all = (int32_t)(s->head0.size() + s->tail0.size());
            if (!s->opt.support_only)
                s->order.insert(s->order.end(), s->tail0.begin(), s->tail0.end());
            if (cfg.random_order) {
                s->rng.shuffle(s->order.data(), (size_t)s->n_support);
                if (!s->opt.support_only && (int32_t)s->order.size() > s->n_support)
                    s->rng.shuffle(s->order.data() + s->n_support,
                                   s->order.size() - (size_t)s->n_support);
            }
            const int32_t P = s->opt.support_only ? s->n_support : P_all;
            const int64_t traj_seed = (int64_t)(s->rng.next() >> 2);

            gpap_counters c0{};
            GPAP_TRY(gpap_counters_get(s->e, &c0));   /* before the wall stamp, like the faces */
            const auto tc = clock::now();

            GPAP_TRY(gpap_engine_reset(s->e, s->K));
            int32_t rounds = 0;
            /* A formula whose variables unit propagation already determined has an EMPTY order
             * (P == 0): every active row is a complete model of it. std::vector::data() is null
             * then, and the host entry point refuses null unconditionally — hand it a dummy the
             * P == 0 early-return never reads. */
            static const int32_t kNoOrder = 0;
            const int32_t *order_p = s->order.empty() ? &kNoOrder : s->order.data();
            GPAP_TRY(gpap_engine_trajectory(s->e, order_p, P, s->n_support, cfg.batch,
                                            traj_seed, soft, &rounds));
            int32_t n = 0;
            GPAP_TRY(gpap_select_complete(s->e, s->mask_all, /*slot=*/0, &n));
            int32_t n_sup = n;
            if (s->has_support)
                GPAP_TRY(gpap_select_complete(s->e, s->mask_sup, /*slot=*/1, &n_sup));
            /* select_complete reads the count back, so the launch has drained: the wall stamp
             * below covers real GPU work, not an async enqueue. */
            const double wall = seconds_between(tc, clock::now());

            gpap_counters c1{};
            GPAP_TRY(gpap_counters_get(s->e, &c1));

            gpap_config_stats &cs = s->cstats[(size_t)ci];
            cs.models += n;
            cs.attempts += s->K;
            cs.chunks += 1;
            cs.wall_s += wall;
            cs.decs += c1.decs - c0.decs;
            cs.props += c1.props - c0.props;
            st.n_models += n;
            st.attempts += s->K;
            st.chunks += 1;
            st.n_support_complete += n_sup;
            rounds_sum += rounds;
            st.rounds_max = std::max(st.rounds_max, rounds);

            /* Self-test (opt-in): EVERY completed row, billed to verify_s so it cannot move a
             * rate. The C ABI reports; whether a bad row aborts is the face's policy. */
            if (s->opt.verify && n > 0) {
                const auto tv = clock::now();
                int32_t bad = 0;
                GPAP_TRY(gpap_check_sound(s->e, /*slot=*/0, /*max_rows=*/0, &bad));
                st.sound_rows_checked += n;
                st.unsound_rows += bad;
                st.verify_s += seconds_between(tv, clock::now());
            }

            /* Delivery (opt-in): keep up to keep_cap rows PER CONFIG, gathered off the chunk
             * that produced them (a prefix of the selection — rows are i.i.d. trajectories, so
             * a prefix carries no information). Billed to deliver_s. */
            if (keep_cap != 0 && n > 0) {
                const int64_t have = cs.kept_rows;
                const int64_t room = keep_cap < 0 ? n : keep_cap - have;
                const int32_t take = (int32_t)std::min<int64_t>(n, std::max<int64_t>(room, 0));
                if (take > 0) {
                    const auto td = clock::now();
                    auto &bucket = s->buckets[(size_t)ci];
                    const size_t off = bucket.size();
                    bucket.resize(off + (size_t)take * (size_t)n_cols);
                    GPAP_TRY(gpap_var_bits(s->e, /*slot=*/0, cols_p, n_cols,
                                           (int8_t *)(bucket.data() + off), take));
                    cs.kept_rows += take;
                    st.deliver_s += seconds_between(td, clock::now());
                }
            }
        }

        /* ---- verdicts of the run ------------------------------------------------------ */
        st.sampling_s = sampling_s();
        st.rounds_mean = st.chunks > 0 ? (double)rounds_sum / (double)st.chunks : 0.0;
        int32_t dom = 0, best = 0;
        for (int32_t i = 1; i < n_configs; ++i)
            if (s->cstats[(size_t)i].models > s->cstats[(size_t)dom].models) dom = i;
        /* Best-THROUGHPUT config, preferring one that is WELL-MEASURED (>= mmc chunks) over a
         * lucky single cold-start chunk — the headline rests on the config the top-up phase
         * committed to, and its kept rows are what bits(-1) delivers. */
        for (int32_t i = 1; i < n_configs; ++i) {
            const auto &a = s->cstats[(size_t)best], &b = s->cstats[(size_t)i];
            const bool am = a.chunks >= mmc, bm = b.chunks >= mmc;
            if ((bm && !am) || (bm == am && cfg_rate(b) > cfg_rate(a))) best = i;
        }
        st.dominant_config = dom;
        st.best_config = best;
        st.lookahead_used = adaptive ? s->run_cfgs[(size_t)best].soft
                                     : (s->run_cfgs[0].soft || retry_soft);
        if (stats_out) *stats_out = st;
        return GPAP_OK;
    });
}

extern "C" gpap_status gpap_sampler_config_stats(const gpap_sampler *s, int32_t i,
                                                 gpap_config_stats *out) {
    if (!s || !out || !s->has_run || i < 0 || i >= (int32_t)s->cstats.size())
        return GPAP_ERR_ARG;
    *out = s->cstats[(size_t)i];
    return GPAP_OK;
}

extern "C" gpap_status gpap_sampler_bits(const gpap_sampler *s, int32_t i, uint8_t *out,
                                         int64_t capacity) {
    if (!s || !s->has_run || capacity < 0 || (capacity > 0 && !out)) return GPAP_ERR_ARG;
    const int32_t idx = i < 0 ? s->stats.best_config : i;
    if (idx < 0 || idx >= (int32_t)s->buckets.size()) return GPAP_ERR_ARG;
    const auto &bucket = s->buckets[(size_t)idx];
    const size_t row = (size_t)std::max<int32_t>(s->run_cols_n, 1);
    const size_t rows = std::min((size_t)capacity, bucket.size() / row);
    if (rows) std::memcpy(out, bucket.data(), rows * row);
    return GPAP_OK;
}
