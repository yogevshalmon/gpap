/* GPAP — GPU-parallel (partial) assignment propagation: the C ABI of the engine core.
 *
 * BCP over K independent assumption sets at once, with no clause scanning: negated occurrence
 * lists (CSR) plus one packed int32 word per (row, long clause) — low SHIFT bits count the
 * non-falsified literals, high bits hold the modular sum of the surviving literal indices, so
 * at count 1 the field is the forced literal (docs/ENGINE.md). Two entry points: (1) batched
 * assumption checking, gpap_engine_reset + gpap_engine_apply; (2) fused trajectory sampling,
 * gpap_engine_reset + gpap_engine_trajectory.
 *
 * The library owns every device allocation (driver API) and links against libcuda only. An
 * engine is not thread-safe (one per thread); calls run on the legacy default stream of the
 * device's primary context, so they interoperate with any CUDA runtime user in the process.
 * Every call returns gpap_status; gpap_last_error() gives a thread-local message. */
#ifndef GPAP_H
#define GPAP_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GPAP_OK = 0,
    GPAP_ERR_ARG = 1,       /* bad argument */
    GPAP_ERR_CUDA = 2,      /* a driver call failed (see gpap_last_error) */
    GPAP_ERR_ENVELOPE = 3,  /* formula outside both packed splits */
    GPAP_ERR_IO = 4,        /* file could not be read / parsed */
    GPAP_ERR_KERNEL = 5,    /* no cubin for the requested specialization */
    GPAP_ERR_MEMORY = 6     /* an allocation failed — device OR host; the message says which */
} gpap_status;

typedef enum {
    /* Let the engine choose per reset(K) with gpap_dispatch_wave. Input only: gpap_engine_mode
     * reports the resolved kernel. Zero on purpose, so `{0}` and NULL options both dispatch. */
    GPAP_MODE_AUTO = 0,
    GPAP_MODE_SCALAR = 1,   /* serial-trail propagate (best below ~2k clauses) */
    GPAP_MODE_WAVE = 2      /* warp-wide wavefront propagate (best above it) */
} gpap_mode;

typedef struct gpap_formula gpap_formula;
typedef struct gpap_engine gpap_engine;
typedef struct gpap_sampler gpap_sampler;

/* ------------------------------------------------------------------ formula */

typedef struct {
    int32_t num_vars;
    int32_t num_clauses;      /* empty clauses excluded (see has_empty_clause) */
    int32_t max_clause_len;
    int32_t shift;            /* packed field split: 16 (16|16) or 8 (8|24 "wide") */
    int32_t has_empty_clause; /* 1 => formula is trivially UNSAT */
    int64_t total_lits;
    /* num_clauses = units + num_bin + num_long; only LONG (length >= 3) clauses carry a packed
     * word and per-row state — length 2 are occ implication edges, length 1 live in the flush. */
    int32_t num_bin;          /* length-2 clauses */
    int32_t num_long;         /* length >= 3 clauses (the packed-word list) */
} gpap_formula_info;

/* Literal convention on this ABI: DIMACS signed literals (+v / -v, 1-based). The 2N internal
 * index scheme (+v -> 2(v-1), -v -> 2(v-1)+1) is used only where explicitly documented. */
gpap_status gpap_formula_from_dimacs(const char *path, gpap_formula **out);
gpap_status gpap_formula_from_clauses(int32_t num_vars, const int32_t *flat_lits,
                                      const int32_t *clause_lens, int32_t num_clauses,
                                      gpap_formula **out);
void gpap_formula_info_get(const gpap_formula *f, gpap_formula_info *out);
void gpap_formula_free(gpap_formula *f);

/* Sign bit set on BINARY occurrence payloads, so a reader classifies an entry from the value it
 * already loaded. Unambiguous by construction: a long payload is a clause index <= INT32_MAX
 * (enforced at build), a binary payload a 2N index <= 0xFFFFFF. Mask with ~GPAP_OCC_BIN_TAG. */
#define GPAP_OCC_BIN_TAG ((int32_t)0x80000000)

/* Read-back of the built structures; NULL out pointers query sizes. Entries of occ[l] below
 * mid_out[l] are BINARY (payload = the other literal's 2N index | GPAP_OCC_BIN_TAG), the rest
 * LONG (payload = index in 0..num_long-1). Units have no entry. mid_out has n_offsets - 1 entries. */
gpap_status gpap_formula_occ(const gpap_formula *f, int32_t *offsets_out, int32_t *mid_out,
                             int32_t *clauses_out, int32_t *n_offsets, int64_t *n_clauses);
gpap_status gpap_formula_clause_arrays(const gpap_formula *f, int32_t *sizes_out,
                                       int32_t *lit_sum_out);

/* The clauses as FLAT DIMACS literals (clause c = the next clause_sizes[c] entries, sizes from
 * gpap_formula_clause_arrays) in the original clause order; within a clause, ascending by
 * variable. `n_lits` (may be NULL) receives total_lits; lits_out NULL = query only. */
gpap_status gpap_formula_clauses(const gpap_formula *f, int32_t *lits_out, int64_t *n_lits);

/* Independent support from `c ind <v> ... 0` header lines (`cr <v> ...` accepted too), as
 * 0-based variable indices, ascending. out=NULL queries the count; 0 = none declared. */
gpap_status gpap_formula_support(const gpap_formula *f, int32_t *out, int32_t *count);

/* Which packed split fits a formula shape, without building it: GPAP_OK + *shift_out, or
 * GPAP_ERR_ENVELOPE. Suite drivers use this to SKIP oversized instances. */
gpap_status gpap_packed_envelope(int32_t num_vars, int32_t max_clause_len, int32_t *shift_out);

/* ------------------------------------------------------------------ engine */

/* Engine options. NULL and `{0}` are the same struct and both mean every default (device 0,
 * auto kernel, auto split): each field's zero is its default. */
typedef struct {
    int device;             /* CUDA device ordinal (0) */
    gpap_mode mode;         /* GPAP_MODE_AUTO (0) = pick per reset(K); pin a kernel only to A/B */
    int32_t force_shift;    /* 0 = from the formula shape; 8 or 16 pins a split (A/B only) */
    int32_t vscan;          /* wave-kernel decision scan: 0 = default (on), 1 = on, -1 = off */
} gpap_engine_options;

gpap_status gpap_engine_create(const gpap_formula *f, const gpap_engine_options *opts,
                               gpap_engine **out);
void gpap_engine_free(gpap_engine *e);
/* Pin a kernel, or pass GPAP_MODE_AUTO to hand the choice back to gpap_engine_reset. */
gpap_status gpap_engine_set_mode(gpap_engine *e, gpap_mode mode);
/* The RESOLVED kernel (never GPAP_MODE_AUTO) -- what the next launch will actually run. */
gpap_mode gpap_engine_mode(const gpap_engine *e);
/* 1 while the engine still chooses its kernel per reset(K) — nothing pinned one (options,
 * gpap_engine_set_mode, or $GPAP_KERNEL). Ask this before pinning a kernel of your own. */
int32_t gpap_engine_auto(const gpap_engine *e);
int32_t gpap_engine_shift(const gpap_engine *e);
int32_t gpap_engine_k(const gpap_engine *e);
int32_t gpap_engine_num_vars(const gpap_engine *e);
int32_t gpap_engine_num_clauses(const gpap_engine *e);   /* TOTAL clauses (unit + bin + long) */
int32_t gpap_engine_num_long(const gpap_engine *e);      /* packed-word clauses: the row width */

/* Restore K rows to the cached post-units-flush state (allocations reused when K matches), and
 * resolve GPAP_MODE_AUTO for this K — every entry point resets, so every entry point dispatches. */
gpap_status gpap_engine_reset(gpap_engine *e, int32_t K);

/* Entry point 1 — apply a (K, L) matrix of INTERNAL literal indices (-1 = padding) and BCP to
 * fixpoint in one launch. Duplicates are skipped; l/~l pairs conflict the row.
 * PRECONDITION, unchecked and the caller's: every entry is -1 or in [0, 2N). An index past the
 * row's bitmap writes into a LATER row's assignment, the call returns GPAP_OK, and no readout
 * can tell. gpap_check_sets and gpap_probe_* validate before any launch — prefer them.
 * The `_dev` variants take a DEVICE pointer that must be 16-byte aligned (the AOT cubins carry
 * a tt.divisibility(16) attribute on it); cuMemAlloc and framework allocators satisfy it. */
gpap_status gpap_engine_apply(gpap_engine *e, const int64_t *decisions_host, int32_t L);
gpap_status gpap_engine_apply_dev(gpap_engine *e, uintptr_t decisions_dev, int32_t L);

/* CLEAN-mode apply: the same fixpoint launch — identical conflict flags and (K,) counters —
 * that also restores its own packed rows to the post-flush baseline on exit, so the next reset
 * skips the K*M broadcast ("reset-lite", docs/findings/CLEAN_MODE_FINDINGS.md). reset() first.
 * Only the PACKED words are restored: flags + counters stay valid and bitmap readouts
 * (gpap_assignment_rows, gpap_var_bits, gpap_select_complete) see the fixpoint, but packed
 * readouts (gpap_sat_rows, gpap_check_sound), gpap_engine_recache_flush and any further launch
 * are REFUSED — they would run against the baseline (gpap_packed_rows alone is not refused: the
 * CLEAN identity gate reads it). GPAP_CLEAN=0 in the environment (read at create) makes this a
 * plain apply and every reset a broadcast. */
gpap_status gpap_engine_apply_clean(gpap_engine *e, const int64_t *decisions_host, int32_t L);

/* Entry point 2 — the whole randomized decide+propagate trajectory per row, one launch.
 * `order` is P variable indices (0-based), support variables first; a decision group never
 * straddles the n_support boundary. Polarity is a stateless hash of (seed, row, var), so a run
 * is reproducible from `seed` alone. Returns the max decision rounds any row used.
 * `n_support` must lie in [0, P] (checked on both variants); order entries must be in
 * [0, num_vars) — the host variant checks, the `_dev` one cannot (cross-row corruption as above). */
gpap_status gpap_engine_trajectory(gpap_engine *e, const int32_t *order_host, int32_t P,
                                   int32_t n_support, int32_t B, int64_t seed, int soft,
                                   int32_t *max_rounds_out);
gpap_status gpap_engine_trajectory_dev(gpap_engine *e, uintptr_t order_dev, int32_t P,
                                       int32_t n_support, int32_t B, int64_t seed, int soft,
                                       int32_t *max_rounds_out);

/* Incremental-units hook (the entailed-clause miner): reset(1) + apply(units) + recache. */
gpap_status gpap_engine_recache_flush(gpap_engine *e);

/* How this engine's resets executed: *full_out = K*M broadcasts run, *lite_out = broadcasts
 * skipped after a CLEAN launch. A non-zero lite count is the proof the optimization engaged. */
void gpap_engine_reset_stats(const gpap_engine *e, int64_t *full_out, int64_t *lite_out);
int gpap_engine_flush_conflict(const gpap_engine *e);
gpap_status gpap_engine_flushed_assignment(gpap_engine *e, int8_t *out_host); /* (2N,) */

/* ------------------------------------------------------------------ batched checking
 * Entry point 1, packaged: any number of assumption sets to their BCP fixpoints, chunked
 * internally, results in input order — THE chunk loop of both faces (cpp/src/check.cpp). */

/* Sets are FLATTENED DIMACS literals: set i is lits[offsets[i] .. offsets[i+1]), offsets has
 * n+1 entries with offsets[0] == 0 (an empty set propagates to the units flush). The whole
 * input is validated before any launch. `chunk` sets per launch; <= 0 sizes it with
 * gpap_check_chunk_size. Outputs, host buffers:
 *   conflict_out    (n,)     int8, required — the set (with its implications) falsifies a clause
 *   propagated_out  (n,)     int32, required — implications BCP derived beyond the assumptions
 *   assign_out      (n, 2N)  int8, or NULL — 1 where that literal index is TRUE
 *   sat_out         (n, M)   int32, or NULL — surviving literals per clause
 * `conflict_out` is exact for every set, and so is everything else on a non-conflicted set
 * (confluence). On a conflicted set the other fields are a snapshot of where BCP stopped, which
 * the chunk size can move (it feeds the kernel dispatch): compare those on `conflict_out`. */
gpap_status gpap_check_sets(gpap_engine *e, const int32_t *lits, const int64_t *offsets,
                            int64_t n, int32_t chunk, int8_t *conflict_out,
                            int32_t *propagated_out, int8_t *assign_out, int32_t *sat_out);

/* Assumption sets one launch holds: the engine's per-row state (gpap_row_bytes) plus the
 * (chunk, max_lits) int64 decision matrix this entry point allocates, read against LIVE free
 * memory, capped by gpap_default_k_cap and never larger than n_sets. */
int64_t gpap_check_chunk_size(gpap_engine *e, int64_t n_sets, int64_t max_lits);

/* ------------------------------------------------------------------ probing
 * The combinatorial form of entry point 1 — "which k-subsets of this literal set conflict" —
 * behind failed-literal detection, entailed-clause mining, backbone extraction and lookahead.
 * If assuming (a, b) drives BCP to a conflict the formula entails (~a | ~b); the ABI returns
 * the ASSUMPTIONS that conflicted and the application negates. Candidates are counted EXACTLY
 * and checked against the ceiling BEFORE anything is generated, then generated one chunk at a
 * time — a materialized enumeration is terabytes at width 3. */

/** One sweep's result: conflicting tuples + what it cost. Free with gpap_probe_free. */
typedef struct gpap_probe gpap_probe;

/** Why a sweep stopped. */
typedef enum {
    GPAP_PROBE_COMPLETE = 0,  /* the whole enumeration was probed */
    GPAP_PROBE_CEILING  = 1,  /* candidates exceeded max_tuples */
    GPAP_PROBE_DEADLINE = 2,  /* the wall-clock budget expired between chunks */
    GPAP_PROBE_UNSAT    = 3   /* the formula's own units already conflict; nothing to probe */
} gpap_probe_stop;

/** How to sweep, and when to stop. Every field is optional and 0 is its default, so `{0}` and a
 *  NULL options pointer are the same sweep. */
typedef struct {
    double  seconds;       /* wall-clock budget, checked BETWEEN chunks (a chunk is atomic) */
    int64_t max_tuples;    /* probe at most this many candidates (EXACT count, before generation) */
    int32_t batch;         /* tuples per launch; 0 = size it from free VRAM */
    int32_t same_var;      /* 1 = keep tuples with two literals of one variable (they conflict
                            * regardless of the formula); 0 = drop them. Named for the opt-out. */
    int32_t refuse_over_max; /* 0 (default) = max_tuples is a WORK BUDGET: probe that many, report
                            * exhausted = 0 (sound, not complete). 1 = refuse the sweep outright. */
} gpap_probe_options;

/** What a sweep did. `exhausted` is the one that licenses a negative result: it means every
 *  candidate was probed, so "no conflict of this width" is a PROOF, not an absence of evidence. */
typedef struct {
    int64_t total;      /* candidates the full enumeration has (after distinct_vars) */
    int64_t probed;     /* candidates actually probed — a prefix of the enumeration */
    int64_t conflicts;  /* how many of those conflicted AND were reported */
    int32_t exhausted;  /* 1 iff probed == total and nothing cut the sweep short */
    int32_t stop;       /* gpap_probe_stop */
    double  seconds;
} gpap_probe_stats;

/** Probe every `k`-subset of `pool` (DIMACS literals). `pool = NULL` means EVERY literal the
 *  formula's own unit propagation left undetermined. Pass NULL options for the defaults. */
gpap_status gpap_probe_combinations(gpap_engine *e, const int32_t *pool, int32_t pool_n,
                                    int32_t k, const gpap_probe_options *opts,
                                    gpap_probe **out);
/** Probe an explicit (n, k) row-major matrix of DIMACS literals — for callers that build their
 *  own candidates. `same_var` is IGNORED here and `total` is the matrix's own row count. */
gpap_status gpap_probe_tuples(gpap_engine *e, const int32_t *tuples, int64_t n, int32_t k,
                              const gpap_probe_options *opts, gpap_probe **out);
/** How many k-subsets `pool` has, EXACTLY, without generating any — check a ceiling against
 *  this before enumerating. Saturates at INT64_MAX rather than wrapping past the ceiling.
 *  `pool = NULL` counts every free literal. Takes the sweep's options; only `same_var` is read. */
int64_t gpap_probe_count(gpap_engine *e, const int32_t *pool, int32_t pool_n, int32_t k,
                         const gpap_probe_options *opts);
void gpap_probe_free(gpap_probe *p);
void gpap_probe_stats_get(const gpap_probe *p, gpap_probe_stats *out);
/** The conflicting tuples, (conflicts, k) row-major DIMACS literals. Writes
 *  min(conflicts, capacity) TUPLES; `capacity` is in tuples, not ints. */
gpap_status gpap_probe_conflicts(const gpap_probe *p, int32_t *out, int64_t capacity);

/** DIMACS literals whose variable the unit flush left undetermined. `count` is IN/OUT when `out`
 *  is non-NULL (capacity in, written out; too small is GPAP_ERR_ARG); out=NULL queries the count. */
gpap_status gpap_free_literals(gpap_engine *e, int32_t *out, int32_t *count);
/** Fold PROVEN units into the cached flush state in place, so later probes see a narrowed
 *  formula without a rebuild. Sound only if the literals really are entailed — not checked. */
gpap_status gpap_assume_units(gpap_engine *e, const int32_t *lits, int32_t n);

/* ------------------------------------------------------------------ sampling
 * THE solution-sampling chunk loop (reset(K) + trajectory + completion count), the config
 * bandit over it and the accounting around both — cpp/src/sample.cpp, one home for both faces.
 * The sampler BORROWS an engine that must outlive it; nothing else may drive that engine
 * meanwhile. Not thread-safe. A single-config run at a FIXED batch is a pure function of
 * (formula, options, seed): the RNG and shuffle are the library's own and delivery draws nothing
 * from that stream. The batch is NOT part of the seed — K defaults to the analytic bound over
 * LIVE free VRAM, so pin it (gpap_sampler_set_k) before comparing counts (a narrower run's rows
 * are a prefix of the wider one's). A portfolio run is machine-local: the bandit picks on
 * measured wall-clock throughput. */

/* One portfolio entry: how a trajectory decides. */
typedef struct {
    int32_t random_order;  /* 1 = uniform-random free variable (CMSGen's Algorithm 1; diverse).
                            * 0 = first free variable in index order — the gate-DAG order on a
                            *     Tseitin CNF; wins the search-heavy "service" family. */
    int32_t batch;         /* decisions per propagation round (B); < 1 is clamped to 1.
                            * B >= |support| makes the surviving rows exact rejection sampling
                            * (exactly uniform); smaller B weights solutions by their basin. */
    int32_t soft;          /* 1 = in-kernel risk-aware polarity: a large success-rate win on tight
                            * circuits and NOT UNIFORM (Barbarik rejects it) — opt-in on every
                            * face, never for a distribution claim. UNIFORMITY_FINDINGS.md §4b. */
} gpap_sample_config;

/* Why the run stopped. TARGET is the good case; the rest are safety valves, and which one fired
 * matters (a time-capped run and a hopeless run report the same count and opposite facts). */
typedef enum {
    GPAP_SAMPLE_STOP_TARGET = 0,     /* n_target delivered (after the measurement window) */
    GPAP_SAMPLE_STOP_TIME = 1,       /* time_limit_s elapsed */
    GPAP_SAMPLE_STOP_CHUNKS = 2,     /* max_chunks ran */
    GPAP_SAMPLE_STOP_ZERO_YIELD = 3, /* nothing found after every config + grace (and the
                                      * lookahead retry, when enabled) — target unreachable */
    GPAP_SAMPLE_STOP_UNSAT = 4       /* the construction flush already conflicted: the formula
                                      * PROVABLY has no models (not merely none found) */
} gpap_sample_stop;

/* Per-RUN options for gpap_sampler_run (seed and batch cap live on the HANDLE). {0} == NULL ==
 * every default; fields whose useful default is "on" are named for the OPT-OUT.
 *
 * `{0}` IS AN UNBOUNDED REQUEST. max_chunks, time_limit_s and grace_chunks are each "no bound"
 * at zero, so a default-constructed options struct asks the loop to run until the target is
 * met -- which never happens on a formula that yields nothing. That was always true of the
 * first two; grace_chunks defaulting to 2 had been quietly acting as the safety net, at the
 * price of ending every budgeted run early. A caller that needs a guaranteed return sets at
 * least one of the three. */
typedef struct {
    int64_t max_chunks;        /* 0 = unbounded */
    int64_t keep_cap;          /* rows KEPT per config for gpap_sampler_bits: 0 = none (count
                                * only), -1 = unlimited; the gather wall is billed to deliver_s */
    double  time_limit_s;      /* 0 = no limit (checked between chunks) */
    int32_t min_measure_chunks;/* some config must accumulate >= this many chunks before the
                                * target may stop the run (one chunk's rate is noisy); 0 -> 1 */
    int32_t grace_chunks;      /* OPT-IN early give-up: stop once every config has run (cold
                                * start) PLUS this many chunks have found nothing. 0 (the
                                * default, like every field here) = never give up early, run
                                * out max_chunks / time_limit_s / the target. It defaulted to 2
                                * and that made every budget nominal: a 60 s run could end after
                                * ~2 s and report zero on an instance the budget would have
                                * sampled. Whether a formula is hopeless is the caller's call. */
    int32_t verify;            /* 1 = re-check ON DEVICE that every completed row satisfies every
                                * clause: a self-test, off by default on every face, billed to
                                * verify_s. The C ABI reports (unsound_rows), never aborts. */
    int32_t retry_with_lookahead; /* 1 = after a zero-yield window, flip `soft` once and retry.
                                * OFF by default and named for what it opts into: the flip turns
                                * the NON-UNIFORM polarity on. stats.retried says if it fired. */
    int32_t support_only;      /* 1 = decide ONLY phase 1 (the support); 0 (default) = every
                                * variable. n_models / gpap_sampler_bits count FULLY assigned rows
                                * either way; n_support_complete is the support-complete count. */
} gpap_sampler_options;

/* Per-config outcome — the bandit's evidence, and the breakdown both faces report. */
typedef struct {
    int64_t models;     /* completed rows this config produced */
    int64_t attempts;   /* trajectories it ran (chunks x K) */
    int64_t chunks;
    int64_t kept_rows;  /* rows held for gpap_sampler_bits (<= keep_cap) */
    int64_t decs;       /* decision literals, from the engine's own counters */
    int64_t props;      /* literals assigned (decisions + implications) */
    double  wall_s;     /* sampling wall attributed to this config (delivery/verify excluded) */
} gpap_config_stats;

/* What a run did. sampling_s is THE clock every rate divides by; deliver_s (kept-row gathers)
 * and verify_s (the soundness self-test) are billed apart, so neither asking for rows nor
 * asking for verification can move samples/s (gpap.sampling's out_project_s / verify_sound_s). */
typedef struct {
    int64_t n_models;            /* completed rows over all configs */
    int64_t attempts;            /* trajectories over all configs */
    int64_t chunks;
    int64_t n_support_complete;  /* rows whose PHASE-1 variables completed (== n_models when
                                  * there is no support or it is a true independent support) */
    int64_t sound_rows_checked;  /* rows the verify re-check covered — the EVIDENCE behind a
                                  * verdict; 0 with verify on means it never ran, not a pass */
    int64_t unsound_rows;        /* rows that failed it (must be 0) */
    int32_t best_config;         /* index: highest models/wall among configs with
                                  * >= min_measure_chunks; its kept rows are gpap_sampler_bits(-1) */
    int32_t dominant_config;     /* index: most models produced */
    int32_t stop;                /* gpap_sample_stop */
    int32_t timed_out;           /* 1 = stopped on time_limit_s short of n_target */
    int32_t retried;             /* 1 = the zero-yield lookahead retry fired */
    int32_t lookahead_used;      /* the soft flag the FINAL window ran with (retry flips it) */
    int32_t rounds_max;          /* max decision rounds any chunk reported */
    double  rounds_mean;         /* mean over chunks */
    double  sampling_s;          /* the chunk loop alone */
    double  deliver_s;           /* kept-row gathers (keep_cap > 0) */
    double  verify_s;            /* the soundness self-test (verify = 1) */
} gpap_sample_stats;

/* Create a sampler over `e` (which must outlive it; nothing else drives it meanwhile). K =
 * min(gpap_compute_k_safe over live free VRAM, k_cap) unless pinned by set_k; `k_cap` 0 =
 * gpap_default_k_cap, > 0 verbatim, < 0 uncapped. `seed` seeds the RNG once; runs continue it. */
gpap_status gpap_sampler_create(gpap_engine *e, int64_t seed, int32_t k_cap,
                                gpap_sampler **out);
/* Frees the sampler and NOTHING ELSE — it never touches the borrowed engine, so it is safe from
 * a destructor that cannot prove the engine is alive. A kernel this sampler pinned stays pinned
 * (a pure launch flag); gpap_engine_set_mode(e, GPAP_MODE_AUTO) restores per-reset dispatch. */
void gpap_sampler_free(gpap_sampler *s);

/* The resolved batch (rows per chunk). */
int32_t gpap_sampler_k(const gpap_sampler *s);
/* Pin the batch (the Python face's pin_batch). k <= 0 restores the calibrated value. */
gpap_status gpap_sampler_set_k(gpap_sampler *s, int32_t k);
/* Restore the RNG to a known point, as after create with options.seed = seed. */
gpap_status gpap_sampler_reseed(gpap_sampler *s, int64_t seed);

/* Decision phase 1: 0-based variable indices decided (and completion-counted) first — the faces
 * pass the formula's `c ind` support. NULL/0 = one phase over every variable. Out-of-range is
 * rejected; order and duplicates do not matter. SET IT ONCE: each distinct support costs a pair
 * mask that lives until the engine dies; re-passing the same set is detected and free. */
gpap_status gpap_sampler_set_support(gpap_sampler *s, const int32_t *vars, int32_t n);
/* Columns of the kept rows: 0-based variables, in output order; NULL/0 (default) = every
 * variable. Takes effect at the NEXT run; bits() reads with the width its own run used. */
gpap_status gpap_sampler_set_columns(gpap_sampler *s, const int32_t *cols, int32_t n);

/* Run until `n_target` models are found, a valve fires, or — with several configs — the bandit's
 * measurement window completes. `configs` is the PORTFOLIO: each chunk runs ONE config at full
 * K, cold-starting each once, then decaying-epsilon-greedy on measured models/wall, exploiting
 * the winner past the target until min_measure_chunks accumulate. ONE config = no bandit (the
 * pinned-measurement mode). Kept rows are bucketed PER CONFIG (never a mixture). Resets
 * stats/buckets from any previous run; the RNG continues. `opts` NULL = every default. */
gpap_status gpap_sampler_run(gpap_sampler *s, int64_t n_target,
                             const gpap_sample_config *configs, int32_t n_configs,
                             const gpap_sampler_options *opts, gpap_sample_stats *stats_out);

/* Per-config breakdown of the last run; `i` in [0, n_configs). */
gpap_status gpap_sampler_config_stats(const gpap_sampler *s, int32_t i, gpap_config_stats *out);
/* Kept rows of config `i` (-1 = stats.best_config, the delivery rule), as (rows, n_cols)
 * row-major 0/1 uint8 over the set_columns order. Writes min(kept_rows, capacity) ROWS and
 * leaves the rest of `out` untouched. */
gpap_status gpap_sampler_bits(const gpap_sampler *s, int32_t i, uint8_t *out, int64_t capacity);

/* ------------------------------------------------------------------ selections + readout */

/* A pair mask marks the variables a completion test considers. `vars` are 0-based variable
 * indices, or NULL for "all variables". A mask lives until the engine is freed and cannot be
 * released (ceil(2N/32) int32 each): create one per variable set, never per chunk. */
gpap_status gpap_pair_mask_create(gpap_engine *e, const int32_t *vars, int32_t n_vars,
                                  int32_t *mask_id_out);

/* Row selection: slot = active AND no free variable under `mask_id`. GPAP_SLOTS slots exist so
 * a caller can hold "fully assigned" and "support complete" at once (slot 2 is a scratch slot
 * for ad-hoc readout). Returns the row count. */
#define GPAP_SLOTS 3

gpap_status gpap_select_complete(gpap_engine *e, int32_t mask_id, int32_t slot,
                                 int32_t *count_out);
/* Ascending row indices of a selection. Writes min(selection count, capacity) entries and
 * leaves the rest of `rows_out` untouched — there is no out-count, so pass the count
 * gpap_select_complete returned. A slot with no selection since the last reset writes nothing. */
gpap_status gpap_selection_rows(gpap_engine *e, int32_t slot, int32_t *rows_out,
                                int32_t capacity);
/* Put an EXPLICIT ascending row list in a slot, so the readout calls below can serve arbitrary
 * rows rather than only completed ones. */
gpap_status gpap_selection_set(gpap_engine *e, int32_t slot, const int32_t *rows,
                               int32_t n_rows);

/* Soundness: how many selected rows have ANY clause with sat_count == 0 (must be 0 for a
 * fully-assigned row). `max_rows` = 0 checks EVERY selected row; a positive bound checks a
 * prefix, and "0 unsound" then speaks only for that prefix. Cost is per row. */
gpap_status gpap_check_sound(gpap_engine *e, int32_t slot, int32_t max_rows,
                             int32_t *n_unsound_out);

/* (rows, n_cols) int8: 1 iff variable cols[j] (0-based) is TRUE in the i-th selected row, where
 * rows = min(selection count, max_rows). `max_rows` is the row capacity of `out_host`. */
gpap_status gpap_var_bits(gpap_engine *e, int32_t slot, const int32_t *cols, int32_t n_cols,
                          int8_t *out_host, int32_t max_rows);
/* (rows, 2N) int8 assignment rows for a selection (1 = that literal index is TRUE), same
 * min(selection count, max_rows) rule. */
gpap_status gpap_assignment_rows(gpap_engine *e, int32_t slot, int8_t *out_host,
                                 int32_t max_rows);
/* (K,) int8 each; either pointer may be NULL. `capacity` is the row capacity of the non-NULL
 * buffers: these write K entries, so a capacity below gpap_engine_k() is GPAP_ERR_ARG rather
 * than a silent truncation or a heap smash. */
gpap_status gpap_flags(gpap_engine *e, int8_t *active_out, int8_t *conflict_out,
                       int32_t capacity);
/* (K,) int32 unit-propagation implications of the last launch (assigned minus decisions).
 * `capacity` as above. */
gpap_status gpap_num_propagated(gpap_engine *e, int32_t *out_host, int32_t capacity);
/* (rows, M) int32 sat_count (= non-falsified literal count) per ORIGINAL clause, for `n_rows`
 * explicit row indices — inspection/tests. Long clauses read their packed word; binary and unit
 * clauses are synthesized from the row's assignment bitmap. */
gpap_status gpap_sat_rows(gpap_engine *e, const int32_t *rows, int32_t n_rows, int32_t *out_host);

/* (rows, num_long) RAW packed words (sat_count | sum << SHIFT), unmasked, LONG clauses only —
 * binary/unit clauses have no per-row words. Tests/debugging: this is the CLEAN-mode identity
 * gate's readout (gpap_engine_num_long gives the row width). */
gpap_status gpap_packed_rows(gpap_engine *e, const int32_t *rows, int32_t n_rows,
                             int32_t *out_host);

/* ------------------------------------------------------------------ counters + sizing */

typedef struct {
    int64_t ops;       /* occurrence entries processed */
    int64_t props;     /* literals assigned (decisions + implications) */
    int64_t decs;      /* decision literals */
    int64_t launches;  /* kernel launches through the two entry points */
} gpap_counters;

gpap_status gpap_counters_get(gpap_engine *e, gpap_counters *out);
void gpap_counters_reset(gpap_engine *e);

/* ------------------------------------------------------------------ batch sizing
 * THE one implementation of "how big a batch fits" (cpp/src/calibrate.cpp); consumers in any
 * language call these rather than re-deriving the arithmetic. There is deliberately no
 * formula-bytes helper: gpap_mem_available() already reports memory free AFTER the formula was
 * uploaded, so deducting it again would double-count. */

/* Bytes available for new allocations (free VRAM minus a driver reserve, times `fraction`),
 * capped by GPAP_VRAM_BUDGET_MIB when that is set to a positive integer. The cap is for
 * CROSS-CARD REPRODUCTION — a memory-bound K sizes larger on a larger card — and is a ceiling
 * only: a card with less free memory still reports less. Unset = no cap. */
int64_t gpap_mem_available(int device, double fraction);
/* Trail slots one row needs — THE rule (cpp/src/calibrate.cpp), which the engine's allocator
 * calls too. A function of the variable count ALONE: every push is deduplicated through the
 * assignment bitmap, so a variable contributes at most one entry. Hence there is no ceiling on
 * the decision batch B anywhere in the engine. */
int32_t gpap_trail_cap(int32_t num_vars);
/* Device bytes ONE row of engine state costs — the exact allocation list, not an estimate.
 * Pass trail_cap <= 0 for the default. `num_long` is gpap_formula_info.num_long, NOT
 * num_clauses: only clauses of length >= 3 carry a packed word. Passing the total is silent
 * and shrinks the batch (measured; docs/findings/BINARY_OCC_FINDINGS.md). */
int64_t gpap_row_bytes(int32_t num_long, int32_t N, int32_t trail_cap);
/* Largest K fitting `bytes` for a formula shape. `cap` <= 0 means uncapped.
 * `num_long` as above — gpap_formula_info.num_long, not num_clauses. */
int32_t gpap_compute_k_safe(int32_t num_long, int32_t N, int64_t bytes, int32_t cap);
/* THE default batch cap — the `cap` above for a caller who does not want to think about K. One
 * constant, the same on every device (`device` is accepted and ignored). A saturation limit,
 * not a memory bound: gpap_compute_k_safe is what keeps a chunk inside VRAM. */
int32_t gpap_default_k_cap(int device);
/* Resident-warp count of the device (SMs x warps per SM) — the ceiling on rows a SCALAR launch
 * can have in flight at once, and so the small-batch term of the dispatch rule below. */
int32_t gpap_resident_warps(int device);
/* Which propagate kernel a chunk of `batch` rows of an M-clause formula should run: 1 = WAVE,
 * 0 = SCALAR. THE dispatch rule, defined once (cpp/src/calibrate.cpp); it keys off M, not the
 * batch. Confluence makes the choice a pure launch flag: it can only change speed. */
int32_t gpap_dispatch_wave(int32_t num_clauses, int32_t batch, int device);

const char *gpap_last_error(void);

/* The library's semantic version (the CMake project version). It says which RELEASE this is,
 * nothing about which BUILD: use gpap_build_id() for that. */
const char *gpap_version(void);

/* Identity of this exact BUILD: "<version>+<16 hex>", the hex fingerprinting the bytes of the
 * loaded libgpap ("<version>+unknown" if it cannot locate its own file). Key anything that
 * CACHES a measurement on this, never on gpap_version(): a version literal changes only when
 * someone remembers to change it. Hashed once, cached for the process. */
const char *gpap_build_id(void);

#ifdef __cplusplus
}
#endif
#endif /* GPAP_H */
