/* GPAP — GPU-parallel assumption propagation: the C ABI of the engine core.
 *
 * The engine runs Boolean Constraint Propagation over K INDEPENDENT assumption sets at once,
 * with no clause scanning: negated occurrence lists (CSR) plus ONE packed int32 word per
 * (row, clause) whose low SHIFT bits are the count of non-falsified literals and whose high
 * bits are the modular sum of the surviving literal indices — at count 1 that field IS the
 * forced literal. See docs/ENGINE.md.
 *
 * Two entry points, exactly as the Python API:
 *   1. batched assumption checking : gpap_engine_reset + gpap_engine_apply
 *   2. fused trajectory sampling   : gpap_engine_reset + gpap_engine_trajectory
 *
 * Everything the kernels touch is owned by this library (driver-API allocations). Link against
 * libcuda only; no CUDA toolkit, no PyTorch, no Python is required at runtime.
 *
 * Threading: an engine is NOT thread-safe; use one engine per thread. All calls run on the
 * legacy default stream of the device's PRIMARY context, so they interoperate with any CUDA
 * runtime user (PyTorch included) in the same process.
 *
 * Errors: every call returns gpap_status; gpap_last_error() gives a thread-local message.
 */
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
    GPAP_ERR_MEMORY = 6     /* device allocation failed */
} gpap_status;

typedef enum {
    GPAP_MODE_SCALAR = 0,   /* serial-trail propagate (best below ~2k clauses) */
    GPAP_MODE_WAVE = 1,     /* warp-wide wavefront propagate (best above it) */
    /* Let the engine choose, per reset(K), with gpap_dispatch_wave. INPUT ONLY: gpap_engine_mode
     * always reports the RESOLVED kernel, never this. Auto is the right default for anything that
     * is not deliberately A/B-ing the two kernels, and resolving it HERE rather than in a wrapper
     * is what makes that true for every consumer -- see gpap_engine_reset. */
    GPAP_MODE_AUTO = 2
} gpap_mode;

typedef struct gpap_formula gpap_formula;
typedef struct gpap_engine gpap_engine;

/* ------------------------------------------------------------------ formula */

typedef struct {
    int32_t num_vars;
    int32_t num_clauses;      /* empty clauses excluded (see has_empty_clause) */
    int32_t max_clause_len;
    int32_t shift;            /* packed field split: 16 (16|16) or 8 (8|24 "wide") */
    int32_t has_empty_clause; /* 1 => formula is trivially UNSAT */
    int64_t total_lits;
} gpap_formula_info;

/* Literal convention on this ABI: DIMACS signed literals (+v / -v, 1-based). The 2N internal
 * index scheme (+v -> 2(v-1), -v -> 2(v-1)+1) is used only where explicitly documented. */
gpap_status gpap_formula_from_dimacs(const char *path, gpap_formula **out);
gpap_status gpap_formula_from_clauses(int32_t num_vars, const int32_t *flat_lits,
                                      const int32_t *clause_lens, int32_t num_clauses,
                                      gpap_formula **out);
void gpap_formula_info_get(const gpap_formula *f, gpap_formula_info *out);
void gpap_formula_free(gpap_formula *f);

/* Read-back of the built structures (for tests / inspection). Pass NULL to query sizes. */
gpap_status gpap_formula_occ(const gpap_formula *f, int32_t *offsets_out, int32_t *clauses_out,
                             int32_t *n_offsets, int64_t *n_clauses);
gpap_status gpap_formula_clause_arrays(const gpap_formula *f, int32_t *sizes_out,
                                       int32_t *lit_sum_out);

/* Independent support parsed from the file's `c ind <v> ... 0` header lines (the
 * UniGen/QuickSampler convention), as 0-based variable indices, ascending. Pass out=NULL to
 * query the count. 0 means the file declared none — decide every variable. */
gpap_status gpap_formula_support(const gpap_formula *f, int32_t *out, int32_t *count);

/* Which packed split fits a formula shape, without building it: GPAP_OK + *shift_out, or
 * GPAP_ERR_ENVELOPE. Callers sweeping a benchmark suite use this to SKIP oversized instances. */
gpap_status gpap_packed_envelope(int32_t num_vars, int32_t max_clause_len, int32_t *shift_out);

/* ------------------------------------------------------------------ engine */

/* Engine options. Pass NULL to gpap_engine_create for the defaults (device 0, scalar propagate,
 * auto split) — every field is "0 means the sensible default". */
typedef struct {
    int device;             /* CUDA device ordinal (0) */
    gpap_mode mode;         /* GPAP_MODE_SCALAR (so `{0}` stays the historical default);
                             * GPAP_MODE_AUTO to let every reset(K) pick -- what the C++ and
                             * Python faces both pass unless a caller pins a kernel */
    int32_t force_shift;    /* 0 = pick the split from the formula shape; 8 or 16 to pin it
                             * (A/B testing the packed splits on a small formula) */
    int32_t vscan;          /* warp-wide decision scan in the wave kernel:
                             * 0 = default (on), 1 = force on, -1 = force off */
} gpap_engine_options;

gpap_status gpap_engine_create(const gpap_formula *f, const gpap_engine_options *opts,
                               gpap_engine **out);
void gpap_engine_free(gpap_engine *e);
/* Pin a kernel, or pass GPAP_MODE_AUTO to hand the choice back to gpap_engine_reset. */
gpap_status gpap_engine_set_mode(gpap_engine *e, gpap_mode mode);
/* The RESOLVED kernel (never GPAP_MODE_AUTO) -- what the next launch will actually run. */
gpap_mode gpap_engine_mode(const gpap_engine *e);
int32_t gpap_engine_shift(const gpap_engine *e);
int32_t gpap_engine_k(const gpap_engine *e);
int32_t gpap_engine_num_vars(const gpap_engine *e);
int32_t gpap_engine_num_clauses(const gpap_engine *e);

/* Restore K rows to the cached post-units-flush state (allocations reused when K matches).
 * Also RESOLVES GPAP_MODE_AUTO for this K, which is why the choice lives here and not in a
 * wrapper: every entry point resets, so every entry point dispatches. It used to be resolved by
 * gpap::Engine::plan_for and BCPEngine.plan_for instead, and the predictable happened twice --
 * a consumer that reset through the ABI ran SCALAR forever without anyone noticing (the batched
 * assumption path, then gpap_probe_combinations; measured at 1.28-2.20x). */
gpap_status gpap_engine_reset(gpap_engine *e, int32_t K);

/* Entry point 1 — apply a (K, L) matrix of INTERNAL literal indices (-1 = padding) and BCP to
 * fixpoint in one launch. Duplicates are skipped; l/~l pairs conflict the row. */
gpap_status gpap_engine_apply(gpap_engine *e, const int64_t *decisions_host, int32_t L);
gpap_status gpap_engine_apply_dev(gpap_engine *e, uintptr_t decisions_dev, int32_t L);

/* Entry point 2 — the whole randomized decide+propagate trajectory per row, one launch.
 * `order` is P variable indices (0-based), support variables first; a decision group never
 * straddles the n_support boundary. Polarity is a stateless hash of (seed, row, var), so a run
 * is reproducible from `seed` alone. Returns the max decision rounds any row used. */
gpap_status gpap_engine_trajectory(gpap_engine *e, const int32_t *order_host, int32_t P,
                                   int32_t n_support, int32_t B, int64_t seed, int soft,
                                   int32_t *max_rounds_out);
gpap_status gpap_engine_trajectory_dev(gpap_engine *e, uintptr_t order_dev, int32_t P,
                                       int32_t n_support, int32_t B, int64_t seed, int soft,
                                       int32_t *max_rounds_out);

/* Incremental-units hook (the entailed-clause miner): reset(1) + apply(units) + recache. */
gpap_status gpap_engine_recache_flush(gpap_engine *e);
int gpap_engine_flush_conflict(const gpap_engine *e);
gpap_status gpap_engine_flushed_assignment(gpap_engine *e, int8_t *out_host); /* (2N,) */

/* ------------------------------------------------------------------ probing
 * Entry point 1 asks "do THESE assumption sets conflict". Probing asks the combinatorial form of
 * the same question: "which k-subsets of this literal set conflict" — the primitive behind
 * failed-literal detection, entailed-clause mining, backbone extraction and lookahead scoring.
 *
 * If assuming literals (a, b) drives BCP to a conflict then the formula entails (~a | ~b), so a
 * sweep at width k yields every entailed clause of width k that BCP alone can see. This ABI
 * returns the ASSUMPTIONS that conflicted, not their negations: the primitive stays honest and
 * the application (mining) negates.
 *
 * Two things make a combinatorial sweep usable, and both live here rather than in each caller:
 * a batch that fits the GPU, and a budget that stops it. C(P,k) grows fast enough that an
 * unbounded sweep of triples over a wide pool is hours of work — and, if it were materialized
 * first, terabytes of host memory. Candidates are therefore counted EXACTLY (a binomial, or an
 * elementary-symmetric DP when distinct_vars filters same-variable tuples) and checked against
 * the ceiling BEFORE anything is generated, then generated one chunk at a time. */

/** One sweep's result: the conflicting tuples plus what the sweep cost. Free with
 *  gpap_probe_free. */
typedef struct gpap_probe gpap_probe;

/** Why a sweep stopped. */
typedef enum {
    GPAP_PROBE_COMPLETE = 0,  /* the whole enumeration was probed */
    GPAP_PROBE_CEILING  = 1,  /* candidates exceeded max_tuples */
    GPAP_PROBE_DEADLINE = 2,  /* the wall-clock budget expired between chunks */
    GPAP_PROBE_UNSAT    = 3   /* the formula's own units already conflict; nothing to probe */
} gpap_probe_stop;

/** When to stop. Every field is optional; 0 (or negative) means "no limit". */
typedef struct {
    double  seconds;       /* wall-clock budget, checked BETWEEN chunks (a chunk is atomic) */
    int64_t max_tuples;    /* how many candidates to probe at most — checked against an EXACT
                            * count before anything is generated. See refuse_over_max. */
    int32_t batch;         /* tuples per launch; 0 = size it from free VRAM */
    int32_t distinct_vars; /* 1 = drop tuples holding two literals of the SAME variable. Such a
                            * tuple is a repeat or a direct contradiction, so it conflicts for
                            * reasons that have nothing to do with the formula. Default for a
                            * literal pool; pass 0 only if you want them. */
    int32_t refuse_over_max; /* what `max_tuples` MEANS. 0 (default) = a WORK BUDGET: probe the
                            * first max_tuples candidates and report `exhausted = 0`. Every
                            * conflict found is still a genuine entailed clause — a truncated
                            * sweep is SOUND, just not complete. 1 = refuse the sweep outright,
                            * for a caller that only wants an answer it can call a proof. */
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
 *  formula's own unit propagation left undetermined — "all the lits", the common case, and
 *  cheaper than asking for them and passing them back. Pass NULL options for the defaults. */
gpap_status gpap_probe_combinations(gpap_engine *e, const int32_t *pool, int32_t pool_n,
                                    int32_t k, const gpap_probe_options *opts,
                                    gpap_probe **out);
/** Probe an explicit (n, k) row-major matrix of DIMACS literals — the escape hatch for callers
 *  that build candidates themselves (filtered, weighted, or drawn from somewhere else). */
gpap_status gpap_probe_tuples(gpap_engine *e, const int32_t *tuples, int64_t n, int32_t k,
                              const gpap_probe_options *opts, gpap_probe **out);
/** How many k-subsets `pool` has, EXACTLY, without generating any of them — the number to check
 *  a ceiling against BEFORE enumerating. Saturates at INT64_MAX rather than wrapping, because a
 *  wrapped count would compare BELOW a ceiling and let through the very sweep it should refuse.
 *  `pool = NULL` counts over every free literal. Exported because a caller that filters the
 *  RESULTS (gpap.mining does) must still be able to reject an impossible width up front. */
int64_t gpap_probe_count(gpap_engine *e, const int32_t *pool, int32_t pool_n, int32_t k,
                         int32_t distinct_vars);
void gpap_probe_free(gpap_probe *p);
void gpap_probe_stats_get(const gpap_probe *p, gpap_probe_stats *out);
/** The conflicting tuples, (conflicts, k) row-major DIMACS literals. Writes
 *  min(conflicts, capacity) TUPLES; `capacity` is in tuples, not ints. */
gpap_status gpap_probe_conflicts(const gpap_probe *p, int32_t *out, int64_t capacity);

/** DIMACS literals whose variable the formula's unit flush left undetermined — what is worth
 *  probing at all. Pass out=NULL to query the count. */
gpap_status gpap_free_literals(gpap_engine *e, int32_t *out, int32_t *count);
/** Fold PROVEN units into the cached flush state in place, so later probes see a narrowed
 *  formula without it being rebuilt. Sound only if the literals really are entailed — this does
 *  not check, it assumes; that is the caller's proof to have. */
gpap_status gpap_assume_units(gpap_engine *e, const int32_t *lits, int32_t n);

/* ------------------------------------------------------------------ selections + readout */

/* A pair mask marks the variables a completion test considers. `vars` are 0-based variable
 * indices, or NULL for "all variables".
 * LIFETIME: a mask lives until the engine is freed and there is no way to release one — create
 * them ONCE per variable set (typically one or two per engine), never inside the chunk loop.
 * Each costs ceil(2N/32) int32 of device memory, so a per-chunk create would leak steadily. */
gpap_status gpap_pair_mask_create(gpap_engine *e, const int32_t *vars, int32_t n_vars,
                                  int32_t *mask_id_out);

/* Row selection: slot = active AND no free variable under `mask_id`. GPAP_SLOTS slots exist so
 * a caller can hold "fully assigned" and "support complete" at once (slot 2 is a scratch slot
 * for ad-hoc readout). Returns the row count. */
#define GPAP_SLOTS 3
gpap_status gpap_select_complete(gpap_engine *e, int32_t mask_id, int32_t slot,
                                 int32_t *count_out);
/* Ascending row indices of a selection. Writes min(selection count, capacity) entries — pass the
 * real capacity of `rows_out`, which gpap_select_complete told you. */
gpap_status gpap_selection_rows(gpap_engine *e, int32_t slot, int32_t *rows_out,
                                int32_t capacity);
/* Put an EXPLICIT ascending row list in a slot, so the readout calls below can serve arbitrary
 * rows rather than only completed ones. */
gpap_status gpap_selection_set(gpap_engine *e, int32_t slot, const int32_t *rows,
                               int32_t n_rows);

/* Soundness: how many selected rows have ANY clause with sat_count == 0 (must be 0 for a
 * fully-assigned row). max_rows caps the check; pass 0 for "all selected". */
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
/* (K,) int8 each; either pointer may be NULL. */
gpap_status gpap_flags(gpap_engine *e, int8_t *active_out, int8_t *conflict_out);
/* (K,) int32 unit-propagation implications of the last launch (assigned minus decisions). */
gpap_status gpap_num_propagated(gpap_engine *e, int32_t *out_host);
/* (rows, M) int32 sat_count values for `n_rows` explicit row indices — inspection/tests. */
gpap_status gpap_sat_rows(gpap_engine *e, const int32_t *rows, int32_t n_rows, int32_t *out_host);

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
 * THE one implementation of "how big a batch fits" — see cpp/src/calibrate.cpp. Consumers in
 * any language call these rather than re-deriving the arithmetic; two copies of a memory model
 * drift silently, and the symptom is a spill (this hardware degrades to host RAM instead of
 * failing), not an error.
 *
 * There is deliberately no formula-bytes helper: gpap_mem_available() already reports memory
 * that is free AFTER the formula was uploaded, so deducting it again double-counts. */

/* Bytes available for new allocations (free VRAM minus a driver reserve, times `fraction`). */
int64_t gpap_mem_available(int device, double fraction);
/* Trail slots one row can need. Bounded by the VARIABLE count, not by how many literals are
 * assumed or decided: every trail push is deduplicated against the assignment bitmap, so a
 * variable contributes at most one entry before the row conflicts. */
int32_t gpap_trail_cap(int32_t num_vars, int32_t decisions_per_round);
/* Device bytes ONE row of engine state costs — the exact allocation list, not an estimate.
 * Exported so a caller that allocates something else per row (a decision matrix, say) can
 * budget for both against one model. Pass trail_cap <= 0 for the default. */
int64_t gpap_row_bytes(int32_t M, int32_t N, int32_t trail_cap);
/* Largest K fitting `bytes` for a formula shape. `cap` <= 0 means uncapped. */
int32_t gpap_compute_k_safe(int32_t M, int32_t N, int64_t bytes, int32_t cap,
                            int32_t decisions_per_round);
/* Batch cap sized to THIS device's total VRAM — what a caller who does not want to think about
 * K should pass as `cap` above. Falls back to the reference card's cap if the device cannot be
 * queried, so it always returns a usable number. */
int32_t gpap_default_k_cap(int device);
/* Resident-warp count of the device (SMs x warps per SM) — the ceiling on rows a SCALAR launch can
 * have in flight at once, and so the small-batch term of the dispatch rule below. */
int32_t gpap_resident_warps(int device);
/* Which propagate kernel a chunk of `batch` rows of an M-clause formula should run: 1 = WAVE
 * (warp-wide), 0 = SCALAR (serial trail). THE dispatch rule, defined once — gpap::Engine::plan_for
 * and Python's calibrate.dispatch_mode both call this. Keys off M, not batch: see the calibration
 * table and the proxy-collapse history in cpp/src/calibrate.cpp. Confluence makes the choice a pure
 * launch flag, so it can only change speed, never an answer. */
int32_t gpap_dispatch_wave(int32_t num_clauses, int32_t batch, int device);

const char *gpap_last_error(void);
const char *gpap_version(void);

#ifdef __cplusplus
}
#endif
#endif /* GPAP_H */
