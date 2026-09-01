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
    GPAP_ERR_MEMORY = 6     /* an allocation failed — device OR host; the message says which */
} gpap_status;

typedef enum {
    /* Let the engine choose, per reset(K), with gpap_dispatch_wave. INPUT ONLY: gpap_engine_mode
     * always reports the RESOLVED kernel, never this. Auto is the right default for anything that
     * is not deliberately A/B-ing the two kernels, and resolving it HERE rather than in a wrapper
     * is what makes that true for every consumer -- see gpap_engine_reset.
     * IT IS ZERO on purpose: `gpap_engine_options o = {0}` and a NULL options pointer must mean
     * the same thing, and that thing must be the default a library would pick for you. Scalar
     * held 0 once, which made the raw C face the ONE consumer that did not dispatch -- both
     * wrappers overrode it -- so it ran the slower kernel on large formulas. Same hole
     * gpap_engine_reset's comment describes; the measurements are in docs/findings/INSIGHTS.md. */
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
    /* Clause-length segregation (num_clauses = units + num_bin + num_long). Only the LONG
     * (length >= 3) clauses carry a packed word and per-row engine state; length-2 clauses are
     * implication edges in the occurrence list and length-1 clauses live in the flush cache. */
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

/* Sign bit set on BINARY occurrence payloads, so a reader can classify an occ entry from the
 * value it has ALREADY LOADED instead of looking that segment's split point up again (what the
 * WAVE kernel's hottest loop does). Free and unambiguous BY CONSTRUCTION: a long payload is a
 * clause index in [0, num_long), and num_long <= total_lits <= INT32_MAX is enforced at build
 * time, so its sign bit is always 0; a binary payload is a 2N literal index, which the widest
 * packed envelope caps at 0xFFFFFF. Mask with ~GPAP_OCC_BIN_TAG to recover the literal. */
#define GPAP_OCC_BIN_TAG ((int32_t)0x80000000)

/* Read-back of the built structures (for tests / inspection). Pass NULL to query sizes.
 * occ entry payloads are SPLIT by kind: within occ[l], entries below gpap_formula_occ_mid()[l]
 * are BINARY (payload = the other literal's 2N index, TAGGED with GPAP_OCC_BIN_TAG) and entries
 * at or above it are LONG (payload = the length>=3 clause's index in 0..num_long-1, never
 * tagged). Unit clauses have no occ entry.
 * `mid_out` receives that split point per literal; it is (2N,) = n_offsets - 1 entries, so it
 * needs no size query of its own. Any out pointer may be NULL. */
gpap_status gpap_formula_occ(const gpap_formula *f, int32_t *offsets_out, int32_t *mid_out,
                             int32_t *clauses_out, int32_t *n_offsets, int64_t *n_clauses);
gpap_status gpap_formula_clause_arrays(const gpap_formula *f, int32_t *sizes_out,
                                       int32_t *lit_sum_out);

/* The clauses back out as FLAT DIMACS literals (clause c = the next clause_sizes[c] entries;
 * sizes from gpap_formula_clause_arrays), in the ORIGINAL input clause order. Within-clause
 * order is NOT input order: it is ascending by VARIABLE (long clauses are rebuilt by inverting
 * the occ CSR, which is keyed by the NEGATED literal, so the two polarities of one variable come
 * back negative-first; binary pairs and units come from their side lists, sorted by literal
 * index). The two orders therefore differ only inside a TAUTOLOGICAL clause, and the set of
 * literals is exact in every case. `n_lits` (may be NULL) receives total_lits;
 * lits_out NULL = query only. This is what lets a host-side consumer (simplification, mining,
 * re-emission) get Python/C++ clause lists from THE parser instead of shipping a second one. */
gpap_status gpap_formula_clauses(const gpap_formula *f, int32_t *lits_out, int64_t *n_lits);

/* Independent support parsed from the file's `c ind <v> ... 0` header lines (the
 * UniGen/QuickSampler convention; the `cr <v> ...` variant is accepted too), as 0-based
 * variable indices, ascending. Pass out=NULL to query the count. 0 means the file declared
 * none — decide every variable. */
gpap_status gpap_formula_support(const gpap_formula *f, int32_t *out, int32_t *count);

/* Which packed split fits a formula shape, without building it: GPAP_OK + *shift_out, or
 * GPAP_ERR_ENVELOPE. Callers sweeping a benchmark suite use this to SKIP oversized instances. */
gpap_status gpap_packed_envelope(int32_t num_vars, int32_t max_clause_len, int32_t *shift_out);

/* ------------------------------------------------------------------ engine */

/* Engine options. Pass NULL to gpap_engine_create for the defaults (device 0, auto kernel,
 * auto split) — every field is "0 means the sensible default", and `{0}` is that same struct. */
typedef struct {
    int device;             /* CUDA device ordinal (0) */
    gpap_mode mode;         /* GPAP_MODE_AUTO (0): every reset(K) picks the kernel. Pin
                             * GPAP_MODE_SCALAR / GPAP_MODE_WAVE only to A/B the two. */
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
/* 1 while the engine is still CHOOSING its kernel per reset(K) -- i.e. nothing pinned one, by
 * options, by gpap_engine_set_mode, or by $GPAP_KERNEL. A caller about to pin a kernel of its
 * own (the sampler pins the chunk batch's kernel so a short final chunk cannot switch it
 * mid-measurement) must ask this first, or it will override the pin a user just set. */
int32_t gpap_engine_auto(const gpap_engine *e);
int32_t gpap_engine_shift(const gpap_engine *e);
int32_t gpap_engine_k(const gpap_engine *e);
int32_t gpap_engine_num_vars(const gpap_engine *e);
int32_t gpap_engine_num_clauses(const gpap_engine *e);   /* TOTAL clauses (unit + bin + long) */
int32_t gpap_engine_num_long(const gpap_engine *e);      /* packed-word clauses: the per-row
                                                            state width, gpap_packed_rows' M */

/* Restore K rows to the cached post-units-flush state (allocations reused when K matches).
 * Also RESOLVES GPAP_MODE_AUTO for this K, which is why the choice lives here and not in a
 * wrapper: every entry point resets, so every entry point dispatches. It used to be resolved by
 * gpap::Engine::plan_for and BCPEngine.plan_for instead, and the predictable happened twice --
 * a consumer that reset through the ABI ran SCALAR forever without anyone noticing (the batched
 * assumption path, then gpap_probe_combinations). What it cost each time: INSIGHTS.md. */
gpap_status gpap_engine_reset(gpap_engine *e, int32_t K);

/* Entry point 1 — apply a (K, L) matrix of INTERNAL literal indices (-1 = padding) and BCP to
 * fixpoint in one launch. Duplicates are skipped; l/~l pairs conflict the row.
 *
 * PRECONDITION, AND IT IS THE CALLER'S: every entry is -1 or an internal index in [0, 2N).
 * This is the RAW primitive and it does NOT check, because the check is not free where it would
 * sit: an O(K*L) host pass costs 9.7-13.6% of a probe chunk at small M (measured, b03-c25) and
 * is pure duplication there — gpap_check_sets and gpap_probe_* both validate their whole input
 * BEFORE any launch, which is also what gives them their all-or-nothing property. Prefer those
 * faces; reach for this one only when you are forming internal indices yourself.
 * What a bad index does, if you do: the kernel turns it into `ab_row + (lit >> 5)`, so an index
 * past the row's bitmap writes into a LATER ROW's assignment. Row A's bad literal silently
 * changes row B's answer, the call still returns GPAP_OK, and no readout downstream can tell.
 * (Verified, tests/test_core.py.) Convert with the same rule the library uses — +v -> 2(v-1),
 * -v -> 2(v-1)+1 — after checking |v| <= num_vars.
 *
 * The `_dev` variants take a DEVICE pointer, which must be 16-BYTE ALIGNED: the AOT cubins are
 * compiled with a tt.divisibility(16) attribute on it, so the promise is baked into the binary,
 * not merely assumed. cuMemAlloc and every framework allocator (PyTorch, the CUDA runtime)
 * satisfy it; a pointer you formed by INDEXING INTO a larger buffer may not. Today's kernels
 * happen not to vectorize these loads — the offsets are runtime values — so a misaligned pointer
 * currently produces identical results (measured), but that is a compiler outcome and not a
 * guarantee this ABI makes. Align, or pass the host variant and let the library stage it. */
gpap_status gpap_engine_apply(gpap_engine *e, const int64_t *decisions_host, int32_t L);
gpap_status gpap_engine_apply_dev(gpap_engine *e, uintptr_t decisions_dev, int32_t L);

/* CLEAN-mode apply: the same fixpoint launch — identical conflict flags and (K,) counters —
 * that additionally restores its own packed rows to the post-flush baseline before exiting.
 * The next gpap_engine_reset then skips the K*M state broadcast ("reset-lite"), which is
 * 41-94% of a probe chunk's wall (docs/plans/RESET_RESTORE_PLAN.md). Contract: reset() first,
 * exactly as gpap_engine_apply — and nothing may read post-apply STATE afterwards. What the
 * state readouts do is ASYMMETRIC, because this restores the PACKED words only:
 *   flags + counters (gpap_flags, gpap_num_propagated) — the RESULT, always valid;
 *   ABITS readouts   (gpap_assignment_rows, gpap_var_bits, gpap_select_complete) — see the
 *                     FIXPOINT: the clean tail does not touch the assignment bitmap;
 *   PACKED readouts  (gpap_sat_rows, gpap_check_sound) — REFUSED with an error. They would
 *                     answer from the baseline, i.e. describe no launch that ran, and a
 *                     soundness verdict on the wrong state is worse than no verdict;
 *   gpap_engine_recache_flush — REFUSED for the same reason one step on: that pair is
 *                     inconsistent and caching it would corrupt the flush cache for the life
 *                     of the engine. Re-run with gpap_engine_apply and recache THAT;
 *   another launch (gpap_engine_apply, gpap_engine_trajectory) — also REFUSED: it would
 *                     decrement baseline clause counts against literals already TRUE. Call
 *                     gpap_engine_reset first, as every chunked caller already does;
 *   gpap_packed_rows — deliberately NOT refused. It is the raw/debug readout, and returning
 *                     the baseline is exactly what the CLEAN-mode identity gate checks.
 * Probing/mining and flags-only checking run on this; GPAP_CLEAN=0 in the environment (read at
 * engine create) restores EXACT legacy behaviour — every apply_clean becomes plain apply AND
 * every reset broadcasts — the A/B lever and the escape hatch. */
gpap_status gpap_engine_apply_clean(gpap_engine *e, const int64_t *decisions_host, int32_t L);

/* Entry point 2 — the whole randomized decide+propagate trajectory per row, one launch.
 * `order` is P variable indices (0-based), support variables first; a decision group never
 * straddles the n_support boundary. Polarity is a stateless hash of (seed, row, var), so a run
 * is reproducible from `seed` alone. Returns the max decision rounds any row used.
 *
 * `n_support` must lie in [0, P] and is CHECKED on both variants — it is the end of phase 1 and
 * the kernel walks `order` up to it, so a larger value read past the caller's buffer and still
 * returned GPAP_OK. The order ENTRIES must be variable indices in [0, num_vars); the host
 * variant checks them (it is walking the array to stage it anyway), the `_dev` variant cannot
 * without a device pass and so does not — same cross-row corruption as entry point 1 above, via
 * `ab_row + (v >> 4)`. Build the order from a permutation of range(num_vars), as every in-tree
 * caller does. */
gpap_status gpap_engine_trajectory(gpap_engine *e, const int32_t *order_host, int32_t P,
                                   int32_t n_support, int32_t B, int64_t seed, int soft,
                                   int32_t *max_rounds_out);
gpap_status gpap_engine_trajectory_dev(gpap_engine *e, uintptr_t order_dev, int32_t P,
                                       int32_t n_support, int32_t B, int64_t seed, int soft,
                                       int32_t *max_rounds_out);

/* Incremental-units hook (the entailed-clause miner): reset(1) + apply(units) + recache. */
gpap_status gpap_engine_recache_flush(gpap_engine *e);

/* How this engine's resets executed: *full_out = K*M broadcasts run, *lite_out = broadcasts
 * SKIPPED because the previous CLEAN-mode launch had already restored the baseline. The lite
 * count is how you verify the optimization actually engaged (a silent fallback measures 1.00x). */
void gpap_engine_reset_stats(const gpap_engine *e, int64_t *full_out, int64_t *lite_out);
int gpap_engine_flush_conflict(const gpap_engine *e);
gpap_status gpap_engine_flushed_assignment(gpap_engine *e, int8_t *out_host); /* (2N,) */

/* ------------------------------------------------------------------ batched checking
 * Entry point 1, packaged: propagate ANY number of assumption sets to their BCP fixpoints,
 * chunked to the GPU internally, results in INPUT ORDER — a caller cannot tell how many
 * launches it took. THE chunk loop of both one-call faces (gpap::AssumptionChecker and
 * Python's), living here for the same reason the probe sweep lives in cpp/src/probe.cpp: two
 * wrapper copies of the same loop drift (they already had — one face validated before any
 * launch, the other mid-loop), and the rules a caller observes — validation, chunk sizing,
 * padding — must not depend on which language asked. */

/* Sets are FLATTENED DIMACS literals: set i is lits[offsets[i] .. offsets[i+1]), offsets has
 * n+1 entries with offsets[0] == 0 (a set may be empty: it propagates to the units flush).
 * The WHOLE input is validated before any launch — an out-of-range literal cannot be allowed
 * to reach the kernel (see gpap_probe_tuples for why), and failing after some chunks ran would
 * leave a partially-filled result the caller cannot distinguish from a full one.
 *
 * `chunk` sets are propagated per launch; pass <= 0 to size it with gpap_check_chunk_size
 * (callers pin it only to A/B chunking itself — results are chunk-invariant on every
 * SURVIVING row; see below). Outputs, all host buffers, filled per chunk:
 *   conflict_out    (n,)     int8, required — the set (with its implications) falsifies a clause
 *   propagated_out  (n,)     int32, required — implications BCP derived beyond the assumptions
 *   assign_out      (n, 2N)  int8, or NULL — 1 where that literal index is TRUE
 *   sat_out         (n, M)   int32, or NULL — surviving literals per clause
 *
 * WHICH FIELDS ARE SPECIFIED: `conflict_out` is exact for every set. For a set that does NOT
 * conflict, so is everything else (BCP fixpoints are confluent). For a set that DOES conflict,
 * propagation stopped as soon as some clause emptied, so the other fields on that row are a
 * snapshot of where BCP happened to stop — anything that moves the propagation order, the
 * chunk size included (it feeds the wave/scalar dispatch through gpap_engine_reset), can
 * change them. Compare conflicted rows on `conflict_out` alone. */
gpap_status gpap_check_sets(gpap_engine *e, const int32_t *lits, const int64_t *offsets,
                            int64_t n, int32_t chunk, int8_t *conflict_out,
                            int32_t *propagated_out, int8_t *assign_out, int32_t *sat_out);

/* Assumption sets one launch holds: the engine's own per-row state (the shared exact model,
 * gpap_row_bytes) plus the (chunk, max_lits) int64 decision matrix this entry point allocates
 * and the sampler's path never does — 8 bytes per literal slot per row, the dominant term on a
 * small formula with long sets. Read against LIVE free memory, capped by gpap_default_k_cap,
 * never larger than n_sets. */
int64_t gpap_check_chunk_size(gpap_engine *e, int64_t n_sets, int64_t max_lits);

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

/** How to sweep, and when to stop. Every field is optional and 0 is its default, so `{0}` and a
 *  NULL options pointer are the same sweep. */
typedef struct {
    double  seconds;       /* wall-clock budget, checked BETWEEN chunks (a chunk is atomic) */
    int64_t max_tuples;    /* how many candidates to probe at most — checked against an EXACT
                            * count before anything is generated. See refuse_over_max. */
    int32_t batch;         /* tuples per launch; 0 = size it from free VRAM */
    int32_t same_var;      /* 1 = KEEP tuples holding two literals of the same variable. Such a
                            * tuple is a repeat or a direct contradiction, so it conflicts for
                            * reasons that have nothing to do with the formula — 0 (the default)
                            * drops them, which is what a literal pool wants.
                            * The sense is INVERTED from the `distinct_vars` field it replaced,
                            * precisely so that 0 is the default: with the old spelling `{0}` and
                            * NULL were DIFFERENT sweeps over the same pool, and `exhausted` —
                            * which this header calls a PROOF — was computed against whichever
                            * total the caller accidentally chose. The rename is a compile error
                            * on purpose. docs/findings/INSIGHTS.md. */
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
 *  that build candidates themselves (filtered, weighted, or drawn from somewhere else).
 *  `same_var` is IGNORED here: these are the caller's tuples, so the library does not filter
 *  them, and `total` is the matrix's own row count. */
gpap_status gpap_probe_tuples(gpap_engine *e, const int32_t *tuples, int64_t n, int32_t k,
                              const gpap_probe_options *opts, gpap_probe **out);
/** How many k-subsets `pool` has, EXACTLY, without generating any of them — the number to check
 *  a ceiling against BEFORE enumerating. Saturates at INT64_MAX rather than wrapping, because a
 *  wrapped count would compare BELOW a ceiling and let through the very sweep it should refuse.
 *  `pool = NULL` counts over every free literal. Exported because a caller that filters the
 *  RESULTS (gpap.mining does) must still be able to reject an impossible width up front.
 *  Takes the SWEEP'S OWN options (NULL = defaults) rather than a loose flag, so the count and
 *  the sweep cannot be asked about different enumerations — only `same_var` is read. */
int64_t gpap_probe_count(gpap_engine *e, const int32_t *pool, int32_t pool_n, int32_t k,
                         const gpap_probe_options *opts);
void gpap_probe_free(gpap_probe *p);
void gpap_probe_stats_get(const gpap_probe *p, gpap_probe_stats *out);
/** The conflicting tuples, (conflicts, k) row-major DIMACS literals. Writes
 *  min(conflicts, capacity) TUPLES; `capacity` is in tuples, not ints. */
gpap_status gpap_probe_conflicts(const gpap_probe *p, int32_t *out, int64_t capacity);

/** DIMACS literals whose variable the formula's unit flush left undetermined — what is worth
 *  probing at all. `count` is IN/OUT when `out` is non-NULL: pass the capacity of `out`, receive
 *  how many were written. Pass out=NULL to query the count alone (`*count` is then ignored on
 *  entry). Too small a capacity is GPAP_ERR_ARG — the two-call query-then-fill pattern is the
 *  intended use and cannot overrun, but a caller who guessed a size must hear about it. */
gpap_status gpap_free_literals(gpap_engine *e, int32_t *out, int32_t *count);
/** Fold PROVEN units into the cached flush state in place, so later probes see a narrowed
 *  formula without it being rebuilt. Sound only if the literals really are entailed — this does
 *  not check, it assumes; that is the caller's proof to have. */
gpap_status gpap_assume_units(gpap_engine *e, const int32_t *lits, int32_t n);

/* ------------------------------------------------------------------ sampling
 *
 * THE solution-sampling chunk loop — reset(K) + trajectory + completion count, the adaptive
 * config bandit over it, and the honest accounting around both. It lives HERE for the same
 * reason the checking loop (check.cpp) and the probe sweep (probe.cpp) do: a rule callers can
 * observe must have one home. This loop used to be written twice — gpap::Sampler in the C++
 * header and Python's GPUSampler — and the two had drifted apart in capability (no structured
 * order or portfolio on the C++ face: instances Python sampled fine returned nothing), in
 * accounting (the C++ rate charged delivery and the soundness self-test to the sampling clock),
 * and in defaults (a C++ default run could silently flip the non-uniform soft polarity on).
 *
 * The sampler BORROWS an engine, exactly as gpap_check_sets and gpap_probe_combinations do; it
 * adds packaging, not a second engine path. Not thread-safe (the engine is not); the engine
 * must outlive the sampler, and nothing else may drive that engine between create and free.
 *
 * DETERMINISM: a run is a pure function of (formula, options, configs, seed) — the RNG and the
 * shuffle are the library's own (splitmix64-seeded xoshiro256**, in-house Fisher-Yates), never
 * std::mt19937/std::shuffle, whose draws differ across standard libraries. Delivery draws
 * nothing from that stream, so asking for rows cannot change which models a seed finds. */

/* One portfolio entry: how a trajectory decides. */
typedef struct {
    int32_t random_order;  /* 1 = uniform-random free variable (CMSGen's Algorithm 1; diverse).
                            * 0 = first free variable in index order — the gate-DAG order on a
                            *     Tseitin CNF; wins the search-heavy "service" family, where a
                            *     random order keeps picking variables propagation would have
                            *     implied anyway. */
    int32_t batch;         /* decisions per propagation round (B); < 1 is clamped to 1.
                            * B >= |support| makes the surviving rows EXACT rejection sampling
                            * (exactly uniform); smaller B interleaves propagation and weights
                            * solutions by their attraction basin. */
    int32_t soft;          /* 1 = in-kernel risk-aware polarity (prefer the value less likely to
                            * conflict). A big success-rate win on tight/deep circuits (b03-c25:
                            * 10% -> 88%) and NOT UNIFORM — it biases the distribution by
                            * construction; Barbarik rejects it on 64 of the 65 circuit
                            * instances it can sample. Use for "many solutions fast,
                            * distribution does not matter", never for a distribution claim.
                            * The Python face spells this opt-in `--allow-soft` and prints a
                            * warning. docs/findings/UNIFORMITY_FINDINGS.md par. 4b. */
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

/* Per-RUN options for gpap_sampler_run — the knobs a caller re-decides per call (the seed and
 * the batch cap live on the HANDLE: create/reseed/set_k). {0} == NULL == every default, per the
 * ground rule: each field's zero is its default, and fields whose useful default is "on" are
 * named for the OPT-OUT. */
typedef struct {
    int64_t max_chunks;        /* 0 = unbounded */
    int64_t keep_cap;          /* rows KEPT per config for gpap_sampler_bits: 0 = none (count
                                * only — the timed-run spelling), -1 = unlimited, else the cap.
                                * Kept rows are gathered as they appear; the gather wall is
                                * billed to deliver_s, never to sampling_s. */
    double  time_limit_s;      /* 0 = no limit (checked between chunks) */
    int32_t min_measure_chunks;/* some config must accumulate >= this many chunks before the
                                * target may stop the run (a single chunk's rate is +-1.9x
                                * run-to-run noise); 0 -> 1 */
    int32_t grace_chunks;      /* zero-yield stop: give up only after every config ran once
                                * (cold start) PLUS this many chunks with nothing found.
                                * "This chunk was empty" is not a signal at a 0.01% success
                                * rate; 0 -> 2 */
    int32_t verify;            /* 1 = re-check ON DEVICE that every completed row satisfies
                                * every clause (a self-test, off by default on every face).
                                * Covers EVERY completed row; wall billed to verify_s, counts
                                * in sound_rows_checked / unsound_rows. The C ABI REPORTS,
                                * never aborts — the faces decide whether to throw. */
    int32_t retry_with_lookahead; /* 1 = after a zero-yield window, flip `soft` once and try
                                * again before giving up (rescues b12-class instances that
                                * yield ONLY under the risk polarity). OFF by default and named
                                * for what it opts into: the flip turns the NON-UNIFORM
                                * polarity on — silently, on exactly the hard instances whose
                                * rows a caller is most likely to publish. stats.retried /
                                * lookahead_used say whether it fired. */
    int32_t support_only;      /* 1 = decide ONLY phase 1 (the support) and count rows whose
                                * support is complete; 0 (default) = decide every variable. */
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
 * and verify_s (the soundness self-test) are billed apart so that neither asking for rows nor
 * asking for verification can move samples/s — the contract gpap.sampling documents as
 * out_project_s / verify_sound_s. */
typedef struct {
    int64_t n_models;            /* completed rows over all configs */
    int64_t attempts;            /* trajectories over all configs */
    int64_t chunks;
    int64_t n_support_complete;  /* rows whose PHASE-1 variables completed (== n_models when
                                  * the support is a true independent support; == n_models by
                                  * definition when there is no support) */
    int64_t sound_rows_checked;  /* rows the verify re-check covered — the EVIDENCE behind a
                                  * soundness verdict; 0 with verify on means the check never
                                  * ran (nothing completed), which is NOT a passing verdict */
    int64_t unsound_rows;        /* rows that failed it (must be 0) */
    int32_t best_config;         /* index into configs: highest models/wall among configs with
                                  * >= min_measure_chunks (falling back to the best-measured
                                  * rate) — whose kept rows gpap_sampler_bits(-1) returns */
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

/* Create a sampler over `e`. The engine must outlive the sampler; nothing else may drive it
 * while the sampler exists (chunk loops assume the state they left). Reads the engine's shape
 * for the batch: K = min(gpap_compute_k_safe over live free VRAM, the k_cap below) unless
 * pinned by set_k. `k_cap`: 0 = gpap_default_k_cap, > 0 verbatim, < 0 uncapped.
 * `seed` seeds the handle's RNG once; runs CONTINUE the stream (reseed restarts it). */
gpap_status gpap_sampler_create(gpap_engine *e, int64_t seed, int32_t k_cap,
                                gpap_sampler **out);
void gpap_sampler_free(gpap_sampler *s);

/* The resolved batch (rows per chunk). */
int32_t gpap_sampler_k(const gpap_sampler *s);
/* Pin the batch (the Python face's pin_batch; the warmup runs at K <= 1024 through this).
 * k <= 0 restores the calibrated value. */
gpap_status gpap_sampler_set_k(gpap_sampler *s, int32_t k);
/* Restore the RNG to a known point, as after create with options.seed = seed. */
gpap_status gpap_sampler_reseed(gpap_sampler *s, int64_t seed);

/* Decision phase 1: 0-based variable indices decided (and completion-counted) first. The faces
 * pass the formula's `c ind` support here — the engine does not retain the formula, so the
 * sampler cannot read it itself. NULL/0 = no support: a single phase over every variable.
 * Out-of-range indices are rejected (an index past N corrupts a LATER row's bitmap). */
gpap_status gpap_sampler_set_support(gpap_sampler *s, const int32_t *vars, int32_t n);
/* Columns of the kept rows: 0-based variables, in output order. NULL/0 (default) = every
 * variable. Takes effect at the NEXT run; gpap_sampler_bits reads with the column width its
 * own run was gathered under, so a change here cannot misshape an existing collection. */
gpap_status gpap_sampler_set_columns(gpap_sampler *s, const int32_t *cols, int32_t n);

/* Run the loop until `n_target` models are found, a valve fires, or — with more than one
 * config — the bandit's measurement window completes. `configs` is the PORTFOLIO: each chunk
 * runs ONE config at full K, cold-starting each once (even past n_target: a big-K GPU meets
 * the target in one chunk, and a config never measured is a config never discoverable), then
 * decaying-epsilon-greedy on measured models/wall with a 20x plausibility filter, exploiting
 * the winner once the target is met until min_measure_chunks accumulate. ONE config = no
 * bandit: the pinned-measurement mode. Kept rows are bucketed PER CONFIG so the emitted set is
 * ONE sampler's, never a mixture of the bandit's arms (a mixture is a uniformity question).
 * Resets `stats`/buckets from any previous run on this handle; the RNG continues (reseed to
 * restart it). n_target <= 0 with one config runs nothing; with a portfolio it still cold-
 * starts every config once. `opts` are THIS run's knobs (NULL = every default). */
gpap_status gpap_sampler_run(gpap_sampler *s, int64_t n_target,
                             const gpap_sample_config *configs, int32_t n_configs,
                             const gpap_sampler_options *opts, gpap_sample_stats *stats_out);

/* Per-config breakdown of the last run; `i` in [0, n_configs). */
gpap_status gpap_sampler_config_stats(const gpap_sampler *s, int32_t i, gpap_config_stats *out);
/* Kept rows of config `i` (-1 = stats.best_config, the delivery rule), as (rows, n_cols)
 * row-major 0/1 uint8 over the set_columns order. Writes min(kept_rows, capacity) ROWS and
 * leaves the rest of `out` untouched — pass the capacity you allocated, read back what
 * config_stats.kept_rows says. */
gpap_status gpap_sampler_bits(const gpap_sampler *s, int32_t i, uint8_t *out, int64_t capacity);

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
/* Ascending row indices of a selection. Writes min(selection count, capacity) entries and LEAVES
 * THE REST OF `rows_out` UNTOUCHED — pass the real capacity, which gpap_select_complete told you,
 * and read only that many back. There is no out-count here (unlike gpap_free_literals), so a
 * capacity larger than the live selection cannot be distinguished from a full write: the tail is
 * whatever the caller's buffer already held. A slot that has had NO selection since the last
 * gpap_engine_reset writes NOTHING — it does not compact the mask, whose device bytes are
 * uninitialized until gpap_select_complete fills them. */
gpap_status gpap_selection_rows(gpap_engine *e, int32_t slot, int32_t *rows_out,
                                int32_t capacity);
/* Put an EXPLICIT ascending row list in a slot, so the readout calls below can serve arbitrary
 * rows rather than only completed ones. */
gpap_status gpap_selection_set(gpap_engine *e, int32_t slot, const int32_t *rows,
                               int32_t n_rows);

/* Soundness: how many selected rows have ANY clause with sat_count == 0 (must be 0 for a
 * fully-assigned row). `max_rows` = 0 checks EVERY selected row and is what a caller should
 * pass; a positive bound checks a PREFIX, and then "0 unsound" is a statement about that
 * prefix rather than about the rows the caller received. Cost is per row and scales with the
 * formula — a small fraction of the chunk it rides on, measured in
 * docs/findings/INSIGHTS.md. */
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
 * than a silent truncation OR a heap smash. It is not optional and there is no 0-means-K
 * spelling, because the whole point is that the caller states the size it actually allocated —
 * every other readout call here takes one, and these two did not (this one segfaulted a harness
 * that passed a 1-byte buffer). */
gpap_status gpap_flags(gpap_engine *e, int8_t *active_out, int8_t *conflict_out,
                       int32_t capacity);
/* (K,) int32 unit-propagation implications of the last launch (assigned minus decisions).
 * `capacity` as above. */
gpap_status gpap_num_propagated(gpap_engine *e, int32_t *out_host, int32_t capacity);
/* (rows, M) int32 sat_count (= non-falsified literal count) per ORIGINAL clause, for `n_rows`
 * explicit row indices — inspection/tests. Long clauses read their packed word; binary and unit
 * clauses are synthesized from the row's assignment bitmap (identical values on non-conflicted
 * rows; conflicted rows' long-clause fields stay propagation-order-dependent, as ever). */
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
 * THE one implementation of "how big a batch fits" — see cpp/src/calibrate.cpp. Consumers in
 * any language call these rather than re-deriving the arithmetic; two copies of a memory model
 * drift silently, and the symptom is a spill (this hardware degrades to host RAM instead of
 * failing), not an error.
 *
 * There is deliberately no formula-bytes helper: gpap_mem_available() already reports memory
 * that is free AFTER the formula was uploaded, so deducting it again double-counts. */

/* Bytes available for new allocations (free VRAM minus a driver reserve, times `fraction`). */
int64_t gpap_mem_available(int device, double fraction);
/* Trail slots one row needs — THE rule (cpp/src/calibrate.cpp), which the engine's own
 * allocator calls too, so the model and the allocation cannot disagree.
 *
 * A function of the VARIABLE count ALONE, not of how many literals are assumed or decided at
 * once: every trail push is gated on the assignment bitmap flipping a literal's bit, and l/~l
 * share a word, so a variable contributes at most one entry before the row conflicts. The bound
 * is exactly num_vars (measured tight on both kernels) plus one wavefront of headroom for the
 * vectorised append. There is therefore NO ceiling on the decision batch B anywhere in the
 * engine: deciding the whole support in one round is a supported — and used — configuration. */
int32_t gpap_trail_cap(int32_t num_vars);
/* Device bytes ONE row of engine state costs — the exact allocation list, not an estimate.
 * Exported so a caller that allocates something else per row (a decision matrix, say) can
 * budget for both against one model. Pass trail_cap <= 0 for the default.
 *
 * ⚠️ `num_long` IS `gpap_formula_info.num_long`, NOT num_clauses. Only clauses of length >= 3
 * carry a packed word: binary clauses are implication edges in the shared occurrence list and
 * unit clauses live in the flush bitmap, so neither costs a single per-row byte. Passing the
 * total clause count is the natural misreading and it is SILENT — no error, just a smaller
 * batch: measured 1.78-2.36x fewer rows on circuit CNFs and the sk_ instances (65-82% of their
 * clauses are binary), 1.00x only where the batch cap binds anyway. */
int64_t gpap_row_bytes(int32_t num_long, int32_t N, int32_t trail_cap);
/* Largest K fitting `bytes` for a formula shape. `cap` <= 0 means uncapped.
 * `num_long` as above — gpap_formula_info.num_long, not num_clauses. */
int32_t gpap_compute_k_safe(int32_t num_long, int32_t N, int64_t bytes, int32_t cap);
/* THE default batch cap — what a caller who does not want to think about K passes as `cap`
 * above. One constant, the SAME on every device (`device` is accepted and ignored), so a run's
 * batch does not depend on which GPU it woke up on and a cross-machine comparison needs no
 * pinned override. It is a saturation limit, not a memory bound: gpap_compute_k_safe is what
 * keeps a chunk inside VRAM. */
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

/* The library's semantic version, e.g. "0.1.0" — the CMake project version, not a second
 * literal that has to be bumped in step with it. It says which RELEASE this is, and nothing
 * about which BUILD: use gpap_build_id() for that. */
const char *gpap_version(void);

/* Identity of this exact BUILD: "<version>+<16 hex>", where the hex fingerprints the bytes of
 * the loaded libgpap — kernels, sizing rules, propagate loops and all. "<version>+unknown" if
 * the library cannot locate its own file (static link, stripped dl info).
 *
 * Use this, never gpap_version(), as the key of anything that CACHES a measurement: a version
 * literal changes only when someone remembers to change it, so a cache keyed on it scores a
 * REBUILT engine as a hit — which is how a rival comparison came to pair a current rival against
 * a months-old GPAP (docs/findings/INSIGHTS.md). Cheap: the file is hashed once, on first call,
 * and the result is cached for the process. */
const char *gpap_build_id(void);

#ifdef __cplusplus
}
#endif
#endif /* GPAP_H */
