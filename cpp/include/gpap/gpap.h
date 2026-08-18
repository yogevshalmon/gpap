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
    GPAP_MODE_SCALAR = 0,   /* serial-trail propagate (best at large K) */
    GPAP_MODE_WAVE = 1      /* warp-wide wavefront propagate (best when K is small) */
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
    gpap_mode mode;         /* GPAP_MODE_SCALAR; GPUSampler-style dispatch picks WAVE at low K */
    int32_t force_shift;    /* 0 = pick the split from the formula shape; 8 or 16 to pin it
                             * (A/B testing the packed splits on a small formula) */
    int32_t vscan;          /* warp-wide decision scan in the wave kernel:
                             * 0 = default (on), 1 = force on, -1 = force off */
} gpap_engine_options;

gpap_status gpap_engine_create(const gpap_formula *f, const gpap_engine_options *opts,
                               gpap_engine **out);
void gpap_engine_free(gpap_engine *e);
gpap_status gpap_engine_set_mode(gpap_engine *e, gpap_mode mode);
gpap_mode gpap_engine_mode(const gpap_engine *e);
int32_t gpap_engine_shift(const gpap_engine *e);
int32_t gpap_engine_k(const gpap_engine *e);
int32_t gpap_engine_num_vars(const gpap_engine *e);
int32_t gpap_engine_num_clauses(const gpap_engine *e);

/* Restore K rows to the cached post-units-flush state (allocations reused when K matches). */
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
