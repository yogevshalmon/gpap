/* Internal plumbing: driver-API helpers, the AOT kernel registry, RAII device buffers and the
 * engine/formula layouts. */
#ifndef GPAP_INTERNAL_HPP
#define GPAP_INTERNAL_HPP

#include <cuda.h>

#include <cstdint>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "gpap/gpap.h"

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

/* Launch with a runtime grid; `args` must NOT include triton's two trailing scratch pointers. */
gpap_status launch(const Kernel &k, unsigned gx, unsigned gy, std::vector<void *> args);

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

    /* CLEAN-mode: leading rows KNOWN equal to packed_flushed. Raised by a full reset,
     * preserved by a completed CLEAN apply, else zeroed. */
    int32_t packed_clean_rows = 0;
    /* 1 = packed and abits describe DIFFERENT states (a CLEAN launch restored packed while
     * abits kept the fixpoint); recache_flush refuses on it, reset clears it. */
    int abits_diverged = 0;
    int64_t resets_full = 0, resets_lite = 0;
    int clean_enabled = 1;   /* GPAP_CLEAN=0 kills the lite path (A/B lever + escape hatch) */

    CUdevice device = 0;
    CUcontext ctx = nullptr;   /* primary context, retained */
    int device_ordinal = 0;
    gpap::KernelCache kernels;

    /* formula-resident (never rebuilt) */
    gpap::DevBuf occ_off, occ_mid, occ_cls, packed_init, packed_flushed, abits_flushed;
    gpap::DevBuf pair_check;           /* (2*n_pairs) int32 — sound_rows' binary/unit pass */

    /* Host side of the original-order readout synthesis (the engine drops the formula). */
    std::vector<int32_t> sat_src_host, pair_host;
    int32_t num_bin = 0;               /* pair indices below this are binary clauses */

    /* per-chunk row state */
    gpap::DevBuf packed, abits, active, conflict, trail;
    gpap::DevBuf nprops, nops, ndecs, nrounds;
    gpap::DevBuf acc_ops, acc_props, acc_decs;   /* int64 accumulators, no host sync */

    gpap::DevBuf scratch_i32;           /* scratch: small results (counts) */
    gpap::DevBuf staging_a, staging_b;  /* staging: uploads and readout buffers */
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
