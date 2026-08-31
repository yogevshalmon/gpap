/* Internal plumbing shared by the GPAP core translation units: driver-API helpers, the
 * AOT kernel registry, RAII device buffers and the engine/formula layouts. */
#ifndef GPAP_INTERNAL_HPP
#define GPAP_INTERNAL_HPP

#include <cuda.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "gpap/gpap.h"

namespace gpap {

void set_error(const char *fmt, ...);
const char *get_error();

/* Guard for PACKED-derived readouts: errors when a CLEAN launch left the baseline behind
 * (defined in engine.cpp, used by readout.cpp). */
gpap_status engine_reject_if_packed_is_baseline(gpap_engine *e, const char *what);

/* DIMACS literal -> internal 2N index (+v -> 2(v-1), -v -> 2(v-1)+1), or -1 if out of range
 * for an N-variable formula. The one conversion every DIMACS-speaking entry point shares —
 * including the BUILDER, which used to carry its own copy built on std::abs().
 * The magnitude is taken in int64 because -INT32_MIN does not fit in int32: std::abs(INT32_MIN)
 * and -lit are both undefined behaviour, and the range check that would have rejected the
 * literal happened AFTER them. It is reachable from public API (gpap_formula_from_clauses takes
 * caller-supplied int32 literals), so it is a real input, not a theoretical one. */
inline int64_t lit_index_from_dimacs(int32_t lit, int32_t num_vars) {
    if (lit == 0) return -1;
    const int64_t v = lit > 0 ? (int64_t)lit : -(int64_t)lit;
    if (v > num_vars) return -1;
    return 2 * (v - 1) + (lit < 0 ? 1 : 0);
}

/* Every driver call goes through this: it records a readable message and returns. */
#define GPAP_CU(call)                                                                    \
    do {                                                                                 \
        CUresult _rc = (call);                                                           \
        if (_rc != CUDA_SUCCESS) {                                                        \
            const char *_msg = nullptr;                                                  \
            cuGetErrorString(_rc, &_msg);                                                \
            ::gpap::set_error("%s: %s (%s:%d)", #call, _msg ? _msg : "unknown",          \
                              __FILE__, __LINE__);                                       \
            return GPAP_ERR_CUDA;                                                        \
        }                                                                                \
    } while (0)

#define GPAP_TRY(expr)                                                                   \
    do {                                                                                 \
        gpap_status _st = (expr);                                                        \
        if (_st != GPAP_OK) return _st;                                                  \
    } while (0)

/* ------------------------------------------------------------------ kernels */

struct Kernel {
    CUfunction fn = nullptr;
    int shared = 0;
    int threads = 32;
};

/* Loads cubins from the generated table on first use and caches the CUfunctions. */
class KernelCache {
  public:
    gpap_status get(const std::string &key, Kernel *out);
    ~KernelCache();

  private:
    std::unordered_map<std::string, Kernel> cache_;
    std::vector<CUmodule> modules_;
};

/* Launch with a runtime grid. `args` must NOT include triton's two trailing scratch
 * pointers — those are appended here (both are null; the AOT build asserts no kernel
 * requires scratch). */
gpap_status launch(const Kernel &k, unsigned gx, unsigned gy, std::vector<void *> args);

/* ------------------------------------------------------------------ device memory */

struct DevBuf {
    CUdeviceptr ptr = 0;
    size_t bytes = 0;

    gpap_status reserve(size_t n);   /* grows only; keeps the allocation if already big enough */
    gpap_status zero();
    void release();
    ~DevBuf() { release(); }

    DevBuf() = default;
    DevBuf(const DevBuf &) = delete;
    DevBuf &operator=(const DevBuf &) = delete;
};

/* ------------------------------------------------------------------ formula */

/* A clause carries a packed word (and per-row state) iff it has at least this many literals;
 * shorter ones are implication edges / flush-cache facts. THE definition — the builder, the
 * off-split repack and the readout synthesis all test through it. */
constexpr int32_t kMinLongLen = 3;
inline bool is_long_clause(int32_t n) { return n >= kMinLongLen; }

/* Packed-word envelopes: (SHIFT, max 2N, max clause length). The sum field must hold any
 * literal index and the count field any clause length; anything outside both is rejected. */
struct PackedSplit {
    int32_t shift, max_two_n, max_len;
};
extern const PackedSplit kPackedSplits[2];

}  // namespace gpap

/* Host-side formula: built once, uploaded by the engine.
 *
 * Clauses are SEGREGATED BY LENGTH — only length >= 3 ("long") clauses carry a packed word
 * and per-row state:
 *   len == 0  ->  has_empty_clause (dropped from M as before)
 *   len == 1  ->  unit_lits ONLY: no occ entry, no packed word. The construction flush asserts
 *                 them into the baseline bitmap, and the bitmap detects any later ~u push —
 *                 observationally identical to the old per-word path, at zero state.
 *   len == 2  ->  IMPLICATION EDGES: two occ entries whose payload is THE OTHER LITERAL
 *                 (clause (~l | b) appears in occ[l] with payload b). Zero per-row state;
 *                 propagation is one bitmap push. bin_pairs keeps the pairs for readout,
 *                 reconstruction and the soundness check.
 *   len >= 3  ->  re-indexed 0..num_long-1: packed word + occ entries whose payload is the
 *                 LONG clause index.
 * Within each occ[l] segment the binary entries come FIRST — occ_mid[l] is the split point —
 * and both sub-ranges stay clause-ascending. clause_sizes / lit_sum keep ALL M kept clauses in
 * the original input order (accessors reconstruct in that order; the per-kind sub-lists are
 * aligned with it by construction). */
struct gpap_formula {
    int32_t num_vars = 0;
    int32_t num_clauses = 0;           /* M: ALL kept clauses (unit + binary + long) */
    int32_t num_bin = 0;               /* length-2 clauses (implication edges) */
    int32_t num_long = 0;              /* length>=3 clauses (the packed-word list) */
    int32_t max_clause_len = 0;
    int32_t shift = 0;
    int32_t has_empty_clause = 0;
    int64_t total_lits = 0;            /* sum of kept clause lengths (occ has fewer entries:
                                          unit clauses contribute none) */
    std::vector<int32_t> occ_offsets;  /* (2N+1) CSR: occ[l] = entries where ~l appears */
    std::vector<int32_t> occ_mid;      /* (2N) end of the binary prefix of occ[l] */
    /* payload, ascending within each kind: below occ_mid[l] the OTHER LITERAL's 2N index
       ORed with GPAP_OCC_BIN_TAG (the sign bit); at or above it the long clause's index in
       [0, num_long), never tagged. BOTH readers depend on that tag — the scalar propagate
       masks it off, the WAVE propagate classifies with `payload < 0` instead of gathering
       occ_mid a second time — and it is free by construction: num_long <= total_lits <=
       INT32_MAX (checked in build()) so a long payload's sign bit is 0, and a literal index
       is capped at 0xFFFFFF by the widest packed envelope. */
    std::vector<int32_t> occ_clauses;
    std::vector<int32_t> clause_sizes; /* (M) original order, all kept clauses */
    std::vector<int32_t> lit_sum;      /* (M) sum of literal indices, exact mod 2^32 */
    std::vector<int32_t> packed_init;  /* (num_long) size | ((sum & mask) << shift) */
    std::vector<int32_t> bin_pairs;    /* (2*num_bin) sorted literal-index pairs, input order */
    /* (M) THE clause -> per-kind slot map, and the only place the "k-th clause of kind K in input
     * order occupies slot k" rule is written down. >= 0 : index into the long/packed list.
     * < 0 : -(p+1) where p indexes the merged pair list [binaries..., units-as-(u,u)...], so
     * p < num_bin is a binary clause and p >= num_bin a unit. Every consumer INDEXES this;
     * nobody re-walks the clauses with their own cursors. */
    std::vector<int32_t> sat_src;
    std::vector<int32_t> unit_lits;    /* literal indices of the formula's own unit clauses */
    std::vector<int32_t> support;      /* 0-based vars from `c ind` header lines (may be empty) */
};

namespace gpap {
/* Build the (num_long,) packed init words for `shift`: size | ((lit_sum & mask) << shift), long
 * clauses only, in the same dense re-index the occ payloads use. ONE definition of the packed
 * layout — gpap_formula's own build and gpap_engine_create's force_shift repack both call it, so
 * a change to the split or to kMinLongLen cannot land in one and miss the other. */
inline void pack_long_init(const gpap_formula *f, int32_t shift, std::vector<int32_t> &out);
}  // namespace gpap

/* One row selection (see gpap_select_complete): a device row mask plus, lazily, the ascending
 * list of selected row indices that the readout kernels index through. */
struct Selection {
    gpap::DevBuf mask;                 /* (K,) int8 */
    gpap::DevBuf rows;                 /* (count,) int32, ascending */
    std::vector<int32_t> host_rows;    /* same, host side (compaction happens here) */
    int32_t count = 0;
    bool rows_valid = false;
};

struct gpap_engine {
    /* shape */
    int32_t num_vars = 0, num_clauses = 0, nw = 0, shift = 16;
    int32_t num_long = 0;              /* packed-word (state) clauses; num_clauses stays TOTAL */
    int32_t n_pairs = 0;               /* soundness pair list: binaries + units as (u,u) */
    int32_t K = 0, trail_cap = 0;
    int64_t total_lits = 0;
    int32_t max_clause_len = 0;
    gpap_mode mode = GPAP_MODE_SCALAR;   /* always RESOLVED: scalar or wave, never auto */
    gpap_mode env_mode = GPAP_MODE_AUTO; /* $GPAP_KERNEL, read once; AUTO = not set */
    int auto_mode = 0;                   /* 1 = re-resolve `mode` on every reset(K) */
    int vscan = 1;
    int has_empty_clause = 0;
    int flush_conflict = 0;
    int flush_ready = 0;      /* the post-units cache is built (NOT packed_flushed.ptr:
                                 an M == 0 formula has a null one) */
    int64_t launches = 0;

    /* CLEAN-mode bookkeeping (gpap_engine_apply_clean): leading rows whose packed row is KNOWN
     * equal to packed_flushed. Raised by a full reset broadcast; PRESERVED by a completed
     * CLEAN-mode apply (it restores its own dirt); zeroed by anything that dirties or
     * invalidates without a restorable record — a plain apply, a trajectory, recache_flush
     * (the baseline itself moves), or a packed realloc (DevBuf::reserve grows by
     * release+alloc, so the contents are gone). */
    int32_t packed_clean_rows = 0;
    /* 1 = packed and abits describe DIFFERENT states: a CLEAN launch restored packed to the
     * baseline while abits kept the fixpoint. Only recache_flush cares — snapshotting that pair
     * would cache a self-contradictory baseline (abits claiming literals TRUE whose clause
     * decrements packed never recorded), so it refuses. Cleared by reset (re-broadcasts abits)
     * and by any plain apply / trajectory (both leave the two consistent). */
    int abits_diverged = 0;
    int64_t resets_full = 0, resets_lite = 0;
    int clean_enabled = 1;   /* GPAP_CLEAN=0 kills the lite path (A/B lever + escape hatch) */

    /* CUDA */
    CUdevice device = 0;
    CUcontext ctx = nullptr;   /* primary context, retained */
    int device_ordinal = 0;
    gpap::KernelCache kernels;

    /* formula-resident (never rebuilt) */
    gpap::DevBuf occ_off, occ_mid, occ_cls, packed_init, packed_flushed, abits_flushed;
    gpap::DevBuf pair_check;           /* (2*n_pairs) int32 — sound_rows' binary/unit pass */

    /* Host side of the original-order readout synthesis (gpap_sat_rows): the engine does not
     * retain the formula, and binary/unit sat counts come from the BITMAP. `pair_host` is the
     * SAME list staged into pair_check above (moved, not copied); `sat_src_host` is the
     * formula's kind/slot map. */
    std::vector<int32_t> sat_src_host, pair_host;
    int32_t num_bin = 0;               /* pair indices below this are binary clauses */

    /* per-chunk row state */
    gpap::DevBuf packed, abits, active, conflict, trail;
    gpap::DevBuf nprops, nops, ndecs, nrounds;
    gpap::DevBuf acc_ops, acc_props, acc_decs;   /* int64 accumulators, no host sync */

    /* scratch / staging */
    gpap::DevBuf scratch_i32;      /* small results (counts) */
    gpap::DevBuf staging_a, staging_b;  /* uploads (order, decisions, cols) + readout buffers */
    std::vector<gpap::DevBuf *> pair_masks;
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
