/* Internal plumbing: error helpers, RAII device buffers and the engine/formula layouts. The
 * device itself is behind gpu_api.hpp. */
#ifndef GPAP_INTERNAL_HPP
#define GPAP_INTERNAL_HPP

#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpap/gpap.h"
#include "gpu_api.hpp"
#include "memory.hpp"

namespace gpap {

void set_error(const char *fmt, ...);
const char *get_error();

/* Share of free VRAM one batch may occupy — the only proportional headroom in the sizing path,
 * defined once so a formula cannot pick a different batch per entry point. */
constexpr double kMemFraction = 0.80;

/* Guard for packed-derived readouts after a CLEAN launch (engine.cpp; used by readout.cpp). */
gpap_status engine_reject_if_packed_is_baseline(gpap_engine *e, const char *what);

/* DIMACS literal -> internal 2N index, or -1 if out of range: the one conversion every
 * DIMACS-speaking entry point shares. The magnitude is taken in int64 because
 * std::abs(INT32_MIN) is undefined and INT32_MIN reaches this from the public API. */
inline int64_t lit_index_from_dimacs(int32_t lit, int32_t num_vars) {
    if (lit == 0) return -1;
    const int64_t v = lit > 0 ? (int64_t)lit : -(int64_t)lit;
    if (v > num_vars) return -1;
    return 2 * (v - 1) + (lit < 0 ? 1 : 0);
}

#define GPAP_TRY(expr)                                                                   \
    do {                                                                                 \
        gpap_status _st = (expr);                                                        \
        if (_st != GPAP_OK) return _st;                                                  \
    } while (0)

/* Run an entry point's body so a C++ exception becomes a gpap_status instead of escaping into
 * std::terminate(). Used where caller- or file-derived sizes reach a host container. */
template <typename F>
gpap_status guard_entry(const char *what, F &&body) {
    try {
        return body();
    } catch (const std::bad_alloc &) {
        set_error("%s: out of host memory", what);
        return GPAP_ERR_MEMORY;
    } catch (const std::exception &ex) {
        set_error("%s: %s", what, ex.what());
        return GPAP_ERR_ARG;
    } catch (...) {
        set_error("%s: unknown host-side failure", what);
        return GPAP_ERR_ARG;
    }
}

/* A device buffer that grows and frees itself: THE way the engine allocates, so every byte it
 * holds is counted (memory.hpp). It remembers the holder that counted it, and through it the
 * plug, so it can be released without the engine at hand. */
struct DevBuf {
    MemHolder *owner = nullptr;
    GpuMem ptr = 0;
    size_t bytes = 0;
    MemKind kind = MemKind::ROWS;            /* the count it goes on (memory.hpp, MemKind) */

    /* Grows only; keeps the allocation if big enough. It frees the old buffer before it takes
     * the new one, so a grow never holds both. GPAP_ERR_MEMORY when the device's limit, or the
     * device, has no room. */
    gpap_status reserve(MemHolder &h, size_t n);
    gpap_status zero();
    void release();
    ~DevBuf() { release(); }

    DevBuf() = default;
    explicit DevBuf(MemKind k) : kind(k) {}
    DevBuf(const DevBuf &) = delete;
    DevBuf &operator=(const DevBuf &) = delete;
};
/* A buffer the engine builds once and trim keeps: the formula's side of the ledger. */
struct FormulaBuf : DevBuf {
    FormulaBuf() : DevBuf(MemKind::FORMULA) {}
};
inline GpuArg buf(const DevBuf &b, size_t offset = 0) { return buf(b.ptr, offset); }

/* The memory budget one batch may take out of `free_bytes` — the reserve, the fraction and the
 * $GPAP_VRAM_BUDGET_MIB cap of gpap_mem_available, for a caller that already holds a device. */
int64_t mem_budget(int64_t free_bytes, double fraction);
/* GPAP_VRAM_BUDGET_MIB in bytes, the ceiling mem_budget applies; 0 = none. */
int64_t mem_budget_cap();
/* Rows of `per_row` bytes that fit `bytes`, at most `cap` (<= 0: no cap), at least one. */
int32_t rows_fitting(int64_t bytes, int64_t per_row, int32_t cap);
/* gpap_dispatch_wave's rule for one GPU library. An engine passes the library it was created
 * on, never $GPAP_BACKEND, which may have changed since. */
int32_t dispatch_wave(const GpuDriver *driver, int32_t num_clauses, int32_t batch, int device);
/* THE automatic kernel choice of an engine for K rows: dispatch_wave on its own library, and
 * SCALAR on a device that cannot run WAVE. Every per-engine choice calls this. */
gpap_mode dispatch_mode(gpap_engine *e, int32_t K);
/* GPAP_ERR_KERNEL, naming the backend, if the engine's plug has no trajectory kernel: a
 * sampler refuses at creation rather than at its first launch. */
gpap_status engine_require_trajectory(gpap_engine *e);
/* THE batch an engine sizes for itself — probing, checking and the sampler all call this, so
 * none can skip a bound: size_rows (memory.cpp) over the engine's device, for rows whose state
 * plus `extra_row_bytes` the caller allocates per row fit beside `reserve_bytes` it allocates
 * once (the queue's launch buffers); at most `cap` (0 = gpap_default_k_cap, < 0 = none). It
 * records the bound that set the batch (e->bound). 0, the message set, when not even one row
 * fits, the device cannot report its memory or the limit is malformed: the caller fails with
 * e->sizing_status rather than run one row. */
int32_t engine_batch(gpap_engine *e, int64_t extra_row_bytes, int32_t cap,
                     int64_t reserve_bytes = 0);
/* Bytes a call's transients may take now on the engine's device: what the limit, or a host's
 * grant, leaves beside everything GPAP holds, with `reuse` bytes the transient frees before it
 * grows counted back. INT64_MAX where neither applies. readout.cpp sizes its slices from it. */
int64_t engine_transient_room(gpap_engine *e, int64_t reuse);
/* gpap_engine_reset(e, K) for a caller that will take `keep` rows again (keep > K): under an
 * exact bound a reset gives back row buffers bigger than its own K (ensure_state's `fit`), and
 * this spares those of up to `keep` rows. A chunk loop passes the rows of its first chunk, which
 * its sizing counted, so its short last chunk does not give them back for the next call to take
 * again. gpap_engine_reset is this with keep = K. */
gpap_status engine_reset_keeping(gpap_engine *e, int32_t K, int32_t keep);

/* ---- The job queue (engine.cpp): entry point 1 for many sets on few rows. A row takes the next
 * set as it finishes one and puts back the words it touched, so a launch does not wait for its
 * deepest set and nothing is reset between sets. The calls that only need a flag and a count
 * per set run on it (probe.cpp, check.cpp); a caller that reads row state afterwards cannot.
 * A queued call is one QueuedCall: begin, then next_block and launch once per block. Until it
 * ends, when the object goes, on every return path, the engine refuses every call that would
 * change the rows or the kernel (gpap_engine::queued_call). */

/* Sets one queued launch may take: bounds the job counter and the per-launch buffers. */
constexpr int64_t kQueueMaxJobs = 1 << 20;
/* Literals one queued launch may take, padding included: bounds the job matrix of wide sets. */
constexpr int64_t kQueueMaxLits = 1 << 24;
/* Device bytes a queued row needs besides its state: its sums of ops, props and decisions. */
constexpr int64_t kQueueRowBytes = 3 * sizeof(int64_t);
/* Device bytes the launches of one call of `n_sets` sets, the widest `width` literals, allocate
 * besides the rows: a launch takes at most min(n_sets, kQueueMaxJobs) jobs, each a flag and an
 * implied count, a job matrix of at most that many rows of `width` and of kQueueMaxLits
 * literals (one set, however wide), and the counter. Not kQueueMaxLits / width jobs: a block of
 * narrower sets is cut by its own width (check_on_queue), so it takes more. QueuedCall::begin
 * takes them off the memory budget, and the destructor frees them, so a call never pays for
 * another's. Keep in step with the reserves in QueuedCall::launch. */
constexpr int64_t queue_call_bytes(int64_t n_sets, int64_t width) {
    const int64_t w = width > 1 ? width : 1;
    int64_t jobs = n_sets < kQueueMaxJobs ? n_sets : kQueueMaxJobs;
    if (jobs < 1) jobs = 1;
    const int64_t cut = kQueueMaxLits > w ? kQueueMaxLits : w;
    const int64_t lits = jobs * w < cut ? jobs * w : cut;
    return lits * 4 + jobs * 5 + 16;
}
/* Rows of a queue on a device whose library cannot say how many it runs at once (OpenCL). */
constexpr int32_t kQueueRowsUnknown = 4096;

class QueuedCall {
  public:
    explicit QueuedCall(gpap_engine *e) : e_(e) {}
    /* Ends the call begin started, whatever happened after it: frees the per-launch buffers, so
     * a later batch is sized as on a fresh engine, and leaves the engine as
     * gpap_engine_reset(e, R) leaves it, its kernel chosen for K = R by the normal rule. */
    ~QueuedCall();
    QueuedCall(const QueuedCall &) = delete;
    QueuedCall &operator=(const QueuedCall &) = delete;

    /* Start a call of `n_sets` sets, or decline it (engaged() stays false, nothing on the engine
     * changed), which sends the caller to
     * its chunk loop (every backend has one): the switch is off ($GPAP_QUEUE, or
     * $GPAP_CLEAN=0, since a queued row restores itself with CLEAN mode's restore), the
     * formula's own units conflict (no baseline to restore to), the engine's plug has no queue
     * kernel, or the call has no more sets than the queue has rows — every row would then take
     * one set, which is a chunk with more bookkeeping. Otherwise the call is engaged on R rows,
     * all at the baseline. `first_block` is the sets the caller's chunk loop gives one launch,
     * `width` the literals of its widest set.
     * ROWS: `requested` if > 0 (a caller's pin), else all that fit: engine_batch() with this
     * call's launch buffers (queue_call_bytes) taken out of the budget, which bounds a pin too;
     * never more than the device runs at once (a row it is not running takes no job and only
     * costs memory; kQueueRowsUnknown where its library cannot say).
     * COPY: the packed words where packed_clean_rows does not cover the R rows; the bitmap
     * always, since no claim covers it (a CLEAN launch leaves it at its fixpoint). That one copy
     * per call is the price of keeping no second claim that every launch path would have to
     * drop.
     * KERNEL: a pin ($GPAP_KERNEL, gpap_engine_set_mode), else WAVE, for every formula: the
     * rows are counted in warps, and only a WAVE row fills one on both backends. Not the
     * dispatch rule, which chooses for a chunk; the call's end gives the engine that rule's
     * kernel back. */
    gpap_status begin(int32_t requested, int64_t n_sets, int64_t width, int64_t first_block);
    /* Whether begin started a call: the caller then launches, else it runs its chunk loop. */
    bool engaged() const { return rows_ > 0; }
    /* Sets the next launch takes, with `seconds_left` of the call's deadline (infinity without
     * one). The first takes `first_block`, so it is no longer than the chunk it replaces. The
     * second takes eight times that, whatever the first took: a launch pays for its deepest
     * set once, so a row needs several sets a launch to amortize it, also where one set a row
     * already takes long. Later blocks grow eightfold while launches take under 50 ms, which
     * hides a launch's fixed cost, and are never smaller than the last launch: growth starts
     * from the sets it took, which a caller's cut can make fewer than its block, never from a
     * block that did not run. A block grows only while eight times the
     * last launch fits in what is left of the deadline, which is checked between launches, so
     * a sweep ends within about one launch of it. So no launch lasts much more than eight
     * chunks or 0.4 s. Results never depend on the sizes. */
    int64_t next_block(double seconds_left);
    /* One launch on the call's rows, timed for next_block: `n` jobs (<= kQueueMaxJobs) of up
     * to `L` literal INDICES each, row-major int32 with -1 padding; a caller keeps n * L within
     * kQueueMaxLits unless one set alone is wider. conflict_out (n,) is exact; implied_out
     * (n,), which may be NULL, is exact on a job that did not conflict. Every row is at the
     * baseline after the launch as before it, so launches follow each other with no reset. */
    gpap_status launch(const int32_t *jobs, int64_t n, int32_t L, int8_t *conflict_out,
                       int32_t *implied_out);

  private:
    gpap_engine *e_;
    int32_t rows_ = 0;
    int64_t first_ = 1, last_ = 0;   /* the first block, and the sets the last launch took */
    double took_ = 0.0;              /* seconds the last launch took */
};

/* A clause carries a packed word (per-row state) iff it has this many literals; shorter ones
 * are implication edges or flush-cache facts. The one definition. */
constexpr int32_t kMinLongLen = 3;
inline bool is_long_clause(int32_t n) { return n >= kMinLongLen; }

/* Packed-word envelopes: (SHIFT, max 2N, max clause length); outside both is rejected. */
struct PackedSplit {
    int32_t shift, max_two_n, max_len;
};
extern const PackedSplit kPackedSplits[2];

}  // namespace gpap

/* Host-side formula: built once, uploaded by the engine. Clauses are SEGREGATED BY LENGTH —
 * only length >= 3 ("long") clauses carry a packed word and per-row state:
 *   len == 0  ->  has_empty_clause (dropped from M)
 *   len == 1  ->  unit_lits only, no occ entry: the construction flush asserts them into the
 *                 baseline bitmap, which detects any later ~u push
 *   len == 2  ->  IMPLICATION EDGES: two occ entries payloading THE OTHER LITERAL, one bitmap
 *                 push and zero per-row state (bin_pairs keeps them for readout and soundness)
 *   len >= 3  ->  re-indexed 0..num_long-1: packed word + occ entries payloading that index
 * Binary entries come FIRST in each occ[l] segment (occ_mid[l] splits them) and both sub-ranges
 * stay clause-ascending; clause_sizes / lit_sum keep all M clauses in input order. */
struct gpap_formula {
    int32_t num_vars = 0;
    /* M = all kept clauses; num_bin = length-2 (edges), num_long = length>=3; total_lits sums
       the kept clause lengths (occ has fewer entries: units contribute none). */
    int32_t num_clauses = 0;
    int32_t num_bin = 0;
    int32_t num_long = 0;
    int32_t max_clause_len = 0;
    int32_t shift = 0;
    int32_t has_empty_clause = 0;
    int64_t total_lits = 0;
    std::vector<int32_t> occ_offsets;  /* (2N+1) CSR: occ[l] = entries where ~l appears;
                                          occ_mid = (2N) end of its binary prefix */
    std::vector<int32_t> occ_mid;
    /* payload, ascending within each kind: below occ_mid[l] the OTHER LITERAL's 2N index ORed
       with GPAP_OCC_BIN_TAG (the sign bit), above it the untagged long clause index. */
    std::vector<int32_t> occ_clauses;
    /* (M) in original order: clause lengths and literal-index sums exact mod 2^32. packed_init
       is (num_long) size | ((sum & mask) << shift); bin_pairs (2*num_bin) sorted pairs. */
    std::vector<int32_t> clause_sizes;
    std::vector<int32_t> lit_sum;
    std::vector<int32_t> packed_init;
    std::vector<int32_t> bin_pairs;
    /* (M) The clause -> per-kind slot map: >= 0 the long/packed list, < 0 the value -(p+1) into
     * the merged pair list [binaries..., units-as-(u,u)...]. Consumers index it, never re-walk. */
    std::vector<int32_t> sat_src;
    std::vector<int32_t> unit_lits;    /* the formula's own units, as literal indices; support
                                          is 0-based vars from `c ind` lines (may be empty) */
    std::vector<int32_t> support;
};

namespace gpap {
/* Build the (num_long,) packed init words for `shift`: size | ((lit_sum & mask) << shift), in
 * the dense re-index the occ payloads use. One definition, shared with the force_shift repack. */
inline void pack_long_init(const gpap_formula *f, int32_t shift, std::vector<int32_t> &out);
}  // namespace gpap

/* One row selection (gpap_select_complete): a device row mask plus, lazily, the ascending
 * selected-row indices readout kernels index through. */
struct Selection {
    gpap::DevBuf mask;                 /* (K,) int8; rows is (count,) int32 ascending and
                                          host_rows the same list host-side */
    gpap::DevBuf rows;
    std::vector<int32_t> host_rows;
    int32_t count = 0;
    bool rows_valid = false;
    /* Has anything actually SELECTED into this slot since the last reset? reserve() does not
     * zero the mask, so an unselected slot would compact garbage into arbitrary rows. */
    bool selected = false;
};

struct gpap_engine {
    int32_t num_vars = 0, num_clauses = 0, nw = 0, shift = 16;
    int32_t num_long = 0;              /* packed-word (state) clauses; num_clauses stays TOTAL,
                                          n_pairs = soundness pairs (binaries + units as (u,u)) */
    int32_t n_pairs = 0;
    int32_t K = 0, trail_cap = 0;
    int64_t total_lits = 0;
    int32_t max_clause_len = 0;
    /* mode is always RESOLVED; env_mode is $GPAP_KERNEL read once (AUTO = unset); auto_mode = 1
       re-resolves mode on every reset(K). */
    gpap_mode mode = GPAP_MODE_SCALAR;
    gpap_mode env_mode = GPAP_MODE_AUTO;
    int auto_mode = 0;
    int vscan = 1;
    int has_empty_clause = 0;
    int flush_conflict = 0;
    int flush_ready = 0;      /* the post-units cache is built (not packed_flushed.ptr) */
    int64_t launches = 0;

    /* CLEAN-mode: leading rows KNOWN equal to packed_flushed. Raised by a full reset or a
     * queued call's copy, preserved by a completed CLEAN apply or queued launch, else zeroed.
     * The only claim: the bitmap has none, and a queued call copies it at its start. */
    int32_t packed_clean_rows = 0;
    /* 1 = packed and abits describe DIFFERENT states (a CLEAN launch restored packed while
     * abits kept the fixpoint); recache_flush refuses on it, reset clears it. */
    int abits_diverged = 0;
    int64_t resets_full = 0, resets_lite = 0;
    int clean_enabled = 1;   /* GPAP_CLEAN=0 kills the lite path (A/B lever + escape hatch) */

    /* The queue. queue_enabled is $GPAP_QUEUE (0 = chunks only); queue_mode is the kernel of
     * the last queued launch (AUTO: none yet), since the engine's own goes back to the dispatch
     * rule's when a queued call ends. */
    int queue_enabled = 1;
    /* Set while a QueuedCall holds the rows (begin to its end): reset, apply, trajectory,
     * recache_flush, set_mode and another queued call are refused until it is cleared. */
    int queued_call = 0;
    int64_t queue_launches = 0, queue_jobs = 0;
    gpap_mode queue_mode = GPAP_MODE_AUTO;
    /* The queued launches' work, summed on the host from each row's sums and added to the
     * device accumulators by gpap_counters_get. */
    int64_t host_ops = 0, host_props = 0, host_decs = 0;

    /* The device. Declared before every DevBuf so it is destroyed after them: a buffer frees
     * itself through this plug. */
    std::unique_ptr<gpap::Gpu> gpu;
    const gpap::GpuDriver *driver = nullptr;    /* the library `gpu` was opened from */
    int device_ordinal = 0;
    /* What this engine's buffers are counted on (memory.hpp). Declared before every DevBuf, as
     * `gpu` is: a buffer uncounts itself here when it frees. */
    gpap::MemHolder mem;
    gpap::GpuMemory::Source mem_source = gpap::GpuMemory::TOTAL;   /* what the plug's `free` is */
    gpap_bound bound = GPAP_BOUND_NONE;         /* what set the last batch (engine_batch) */
    /* Why the last engine_batch returned 0: GPAP_ERR_MEMORY when no row fits, or the status of
     * the memory query or the limit that failed. A caller returns it, so a driver error or a
     * malformed limit is not reported as memory. */
    gpap_status sizing_status = GPAP_OK;

    /* formula-resident (never rebuilt): the FORMULA count, which trim keeps */
    gpap::FormulaBuf occ_off, occ_mid, occ_cls, packed_init, packed_flushed, abits_flushed;
    gpap::FormulaBuf pair_check;       /* (2*n_pairs) int32 — sound_rows' binary/unit pass */

    /* Host side of the original-order readout synthesis (the engine drops the formula). */
    std::vector<int32_t> sat_src_host, pair_host;
    int32_t num_bin = 0;               /* pair indices below this are binary clauses */

    /* per-chunk row state */
    gpap::DevBuf packed, abits, active, conflict, trail;
    gpap::DevBuf nprops, nops, ndecs, nrounds;
    gpap::FormulaBuf acc_ops, acc_props, acc_decs;   /* int64 accumulators, no host sync */
    /* one queued launch: the jobs, the shared counter, a flag and an implied count per job,
       three int64 sums per row; released when a queued call ends (~QueuedCall) */
    gpap::DevBuf queue_dec, queue_next, queue_conf, queue_impl, queue_sums;

    gpap::FormulaBuf scratch_i32;       /* scratch: small results (counts) */
    gpap::DevBuf staging_a, staging_b;  /* staging: uploads and readout buffers */
    std::vector<gpap::FormulaBuf *> pair_masks;
    Selection slots[GPAP_SLOTS];

    ~gpap_engine();
};

namespace gpap {
inline void pack_long_init(const gpap_formula *f, int32_t shift, std::vector<int32_t> &out) {
    out.assign((size_t)f->num_long, 0);
    if (!shift) return;                      /* outside every envelope: no packed representation */
    const uint32_t mask = (uint32_t)((1ull << (32 - shift)) - 1);
    int32_t j = 0;
    for (int32_t c = 0; c < f->num_clauses; ++c) {
        const int32_t n = f->clause_sizes[(size_t)c];
        if (!is_long_clause(n)) continue;
        out[(size_t)j++] = (int32_t)((uint32_t)n |
                                     (((uint32_t)f->lit_sum[(size_t)c] & mask) << shift));
    }
}
}  // namespace gpap

#endif /* GPAP_INTERNAL_HPP */
