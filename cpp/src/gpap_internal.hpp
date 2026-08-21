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

/* DIMACS literal -> internal 2N index (+v -> 2(v-1), -v -> 2(v-1)+1), or -1 if out of range
 * for an N-variable formula. The one conversion every DIMACS-speaking entry point shares. */
inline int64_t lit_index_from_dimacs(int32_t lit, int32_t num_vars) {
    if (lit == 0) return -1;
    const int32_t v = lit > 0 ? lit : -lit;
    if (v > num_vars) return -1;
    return 2ll * (v - 1) + (lit < 0 ? 1 : 0);
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

/* Packed-word envelopes: (SHIFT, max 2N, max clause length). The sum field must hold any
 * literal index and the count field any clause length; anything outside both is rejected. */
struct PackedSplit {
    int32_t shift, max_two_n, max_len;
};
extern const PackedSplit kPackedSplits[2];

}  // namespace gpap

/* Host-side formula: built once, uploaded by the engine. */
struct gpap_formula {
    int32_t num_vars = 0;
    int32_t num_clauses = 0;
    int32_t max_clause_len = 0;
    int32_t shift = 0;
    int32_t has_empty_clause = 0;
    int64_t total_lits = 0;
    std::vector<int32_t> occ_offsets;  /* (2N+1) CSR: occ[l] = clauses where ~l appears */
    std::vector<int32_t> occ_clauses;  /* (total_lits) ascending clause index per literal */
    std::vector<int32_t> clause_sizes; /* (M) */
    std::vector<int32_t> lit_sum;      /* (M) sum of literal indices, exact mod 2^32 */
    std::vector<int32_t> packed_init;  /* (M) size | ((sum & mask) << shift) */
    std::vector<int32_t> unit_lits;    /* literal indices of the formula's own unit clauses */
    std::vector<int32_t> support;      /* 0-based vars from `c ind` header lines (may be empty) */
};

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
    int32_t K = 0, trail_cap = 0;
    int64_t total_lits = 0;
    int32_t max_clause_len = 0;
    gpap_mode mode = GPAP_MODE_SCALAR;   /* always RESOLVED: scalar or wave, never auto */
    int auto_mode = 0;                   /* 1 = re-resolve `mode` on every reset(K) */
    int vscan = 1;
    int has_empty_clause = 0;
    int flush_conflict = 0;
    int flush_ready = 0;      /* the post-units cache is built (NOT packed_flushed.ptr:
                                 an M == 0 formula has a null one) */
    int64_t launches = 0;

    /* CUDA */
    CUdevice device = 0;
    CUcontext ctx = nullptr;   /* primary context, retained */
    int device_ordinal = 0;
    gpap::KernelCache kernels;

    /* formula-resident (never rebuilt) */
    gpap::DevBuf occ_off, occ_cls, packed_init, packed_flushed, abits_flushed;

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

#endif /* GPAP_INTERNAL_HPP */
