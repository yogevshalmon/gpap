/* Row selections and readout: the completion test, the soundness check and the two gathers
 * the sampler needs. All of it runs on device — the host only ever receives the small result
 * (a count, a bit matrix, or an assignment slice), never a (K, M) or (K, N) transient.
 *
 * Replaces the torch-side `active & ~any_free(...)`, `sat_rows(...) > 0`, `var_true_bits` and
 * `_bits_to_bytes` of the Python engine.
 */
#include "gpap_internal.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

using gpap::DevBuf;
using gpap::Kernel;
using gpap::launch;
using gpap::set_error;

namespace gpap {
gpap_status engine_use_ctx(gpap_engine *e);
gpap_status engine_stage(DevBuf &buf, const void *host, size_t n);
gpap_status engine_ensure_rows(gpap_engine *e, int32_t slot);
size_t engine_readout_slice_bytes();
unsigned engine_ceil_div(int64_t a, int64_t b);
}  // namespace gpap

using gpap::engine_ceil_div;
using gpap::engine_ensure_rows;
using gpap::engine_readout_slice_bytes;
using gpap::engine_stage;
using gpap::engine_use_ctx;

namespace {

/* Reduce an (n,) int8 flag array into e->scratch_i32[0] and read it back (one sync). */
gpap_status reduce_flags(gpap_engine *e, DevBuf &flags, int32_t n, int32_t *out) {
    Kernel k;
    GPAP_TRY(e->kernels.get("sum_i8_to", &k));
    GPAP_TRY(launch(k, 1, 1, {&flags.ptr, &n, &e->scratch_i32.ptr}));
    GPAP_CU(cuMemcpyDtoH(out, e->scratch_i32.ptr, sizeof(int32_t)));
    return GPAP_OK;
}

}  // namespace

extern "C" gpap_status gpap_pair_mask_create(gpap_engine *e, const int32_t *vars, int32_t n_vars,
                                             int32_t *mask_id_out) {
    if (!e || !mask_id_out) return GPAP_ERR_ARG;
    GPAP_TRY(engine_use_ctx(e));
    /* Even bit 2v set for every selected variable; padding pairs beyond N stay 0 (a raw
     * even-bits constant would misread them as free — the pair mask is REQUIRED). */
    std::vector<uint32_t> bits((size_t)e->nw, 0u);
    const int32_t n = vars ? n_vars : e->num_vars;
    for (int32_t i = 0; i < n; ++i) {
        const int32_t v = vars ? vars[i] : i;
        if (v < 0 || v >= e->num_vars) {
            set_error("pair mask: variable %d out of range (N=%d)", v, e->num_vars);
            return GPAP_ERR_ARG;
        }
        bits[(size_t)(v >> 4)] |= 1u << (2 * (v & 15));
    }
    DevBuf *buf = new DevBuf();
    gpap_status st = engine_stage(*buf, bits.data(), bits.size() * sizeof(uint32_t));
    if (st != GPAP_OK) {
        delete buf;
        return st;
    }
    e->pair_masks.push_back(buf);
    *mask_id_out = (int32_t)e->pair_masks.size() - 1;
    return GPAP_OK;
}

extern "C" gpap_status gpap_select_complete(gpap_engine *e, int32_t mask_id, int32_t slot,
                                            int32_t *count_out) {
    if (!e || slot < 0 || slot >= GPAP_SLOTS || mask_id < 0 || mask_id >= (int32_t)e->pair_masks.size())
        return GPAP_ERR_ARG;
    GPAP_TRY(engine_use_ctx(e));
    Selection &s = e->slots[slot];
    GPAP_TRY(s.mask.reserve((size_t)e->K));

    Kernel k;
    GPAP_TRY(e->kernels.get("complete_mask", &k));
    int32_t NW = e->nw;
    GPAP_TRY(launch(k, (unsigned)e->K, 1,
                    {&e->abits.ptr, &e->pair_masks[(size_t)mask_id]->ptr, &e->active.ptr,
                     &s.mask.ptr, &NW}));
    s.rows_valid = false;
    s.selected = true;                   /* the mask now holds a real selection */
    GPAP_TRY(reduce_flags(e, s.mask, e->K, &s.count));
    if (count_out) *count_out = s.count;
    return GPAP_OK;
}

extern "C" gpap_status gpap_selection_set(gpap_engine *e, int32_t slot, const int32_t *rows,
                                          int32_t n_rows) {
    if (!e || slot < 0 || slot >= GPAP_SLOTS || n_rows < 0 || (n_rows && !rows))
        return GPAP_ERR_ARG;
    GPAP_TRY(engine_use_ctx(e));
    Selection &s = e->slots[slot];
    for (int32_t i = 0; i < n_rows; ++i)
        if (rows[i] < 0 || rows[i] >= e->K) {
            set_error("selection row %d out of range (K=%d)", rows[i], e->K);
            return GPAP_ERR_ARG;
        }
    s.host_rows.assign(rows, rows + n_rows);
    s.count = n_rows;
    /* INVALIDATED BEFORE the fallible upload, not just set true after it. The host side has
     * already moved; if the upload then failed, `rows_valid` would keep its PREVIOUS value, and
     * a slot that was valid would pair the NEW count with the OLD device row list — so
     * gpap_var_bits would gather the previous selection's rows while reporting this one's
     * count, with no error anywhere downstream. */
    s.rows_valid = false;
    s.selected = true;
    GPAP_TRY(engine_stage(s.rows, s.host_rows.data(), s.host_rows.size() * sizeof(int32_t)));
    s.rows_valid = true;                 /* explicit rows: no mask compaction needed */
    return GPAP_OK;
}

extern "C" gpap_status gpap_selection_rows(gpap_engine *e, int32_t slot, int32_t *rows_out,
                                           int32_t capacity) {
    if (!e || slot < 0 || slot >= GPAP_SLOTS || !rows_out || capacity < 0) return GPAP_ERR_ARG;
    GPAP_TRY(engine_use_ctx(e));
    GPAP_TRY(engine_ensure_rows(e, slot));
    const Selection &s = e->slots[slot];
    /* Bounded by the caller's capacity: the selection can be recompacted between the
     * gpap_select_complete that produced the count and this call. */
    const size_t n = std::min((size_t)capacity, s.host_rows.size());
    if (n) memcpy(rows_out, s.host_rows.data(), n * sizeof(int32_t));
    return GPAP_OK;
}

extern "C" gpap_status gpap_check_sound(gpap_engine *e, int32_t slot, int32_t max_rows,
                                        int32_t *n_unsound_out) {
    if (!e || slot < 0 || slot >= GPAP_SLOTS || !n_unsound_out) return GPAP_ERR_ARG;
    GPAP_TRY(gpap::engine_reject_if_packed_is_baseline(e, "gpap_check_sound"));
    GPAP_TRY(engine_use_ctx(e));
    GPAP_TRY(engine_ensure_rows(e, slot));
    Selection &s = e->slots[slot];
    int32_t n = s.count;
    if (max_rows > 0 && max_rows < n) n = max_rows;
    *n_unsound_out = 0;
    if (n == 0) return GPAP_OK;

    GPAP_TRY(e->staging_b.reserve((size_t)n));
    Kernel k;
    GPAP_TRY(e->kernels.get("sound_rows.s" + std::to_string(e->shift), &k));
    /* Long clauses are checked against their packed words; binary + unit clauses against the
     * bitmap through the (n_pairs, 2) pair list (see helpers.py::sound_rows). */
    int32_t M = e->num_long, NW = e->nw, NP = e->n_pairs;
    GPAP_TRY(launch(k, (unsigned)n, 1,
                    {&e->packed.ptr, &e->abits.ptr, &s.rows.ptr, &M, &NW, &e->pair_check.ptr,
                     &NP, &e->staging_b.ptr}));
    return reduce_flags(e, e->staging_b, n, n_unsound_out);
}

extern "C" gpap_status gpap_var_bits(gpap_engine *e, int32_t slot, const int32_t *cols,
                                     int32_t n_cols, int8_t *out_host, int32_t max_rows) {
    if (!e || slot < 0 || slot >= GPAP_SLOTS || n_cols < 0 || max_rows < 0 ||
        (n_cols > 0 && (!cols || !out_host)))
        return GPAP_ERR_ARG;
    GPAP_TRY(engine_use_ctx(e));
    GPAP_TRY(engine_ensure_rows(e, slot));
    Selection &s = e->slots[slot];
    const int32_t n_rows = std::min(max_rows, s.count);   /* never write past the caller */
    /* Zero rows or zero columns are legal degenerate requests (a formula whose variables were
     * all determined by unit propagation / ELS has nothing left to gather). */
    if (n_rows == 0 || n_cols == 0) return GPAP_OK;

    GPAP_TRY(engine_stage(e->staging_a, cols, (size_t)n_cols * sizeof(int32_t)));
    /* Slice the rows so the (rows, n_cols) transient stays bounded. */
    const int64_t per_row = n_cols;
    int32_t step = (int32_t)std::max<int64_t>(
        32, (int64_t)(engine_readout_slice_bytes() / (size_t)std::max<int64_t>(per_row, 1)));
    step = (step / 32) * 32;                 /* keep the row-pointer offset 16-byte aligned */
    if (step <= 0) step = 32;
    GPAP_TRY(e->staging_b.reserve((size_t)std::min(step, n_rows) * (size_t)n_cols));

    Kernel k;
    GPAP_TRY(e->kernels.get("var_bits", &k));
    int32_t NW = e->nw, ncols = n_cols;
    for (int32_t r0 = 0; r0 < n_rows; r0 += step) {
        const int32_t rows = std::min(step, n_rows - r0);
        CUdeviceptr rows_ptr = s.rows.ptr + (size_t)r0 * sizeof(int32_t);
        GPAP_TRY(launch(k, (unsigned)rows, engine_ceil_div(n_cols, 256),
                        {&e->abits.ptr, &rows_ptr, &e->staging_a.ptr, &e->staging_b.ptr, &ncols,
                         &NW}));
        GPAP_CU(cuMemcpyDtoH(out_host + (size_t)r0 * n_cols, e->staging_b.ptr,
                             (size_t)rows * n_cols));
    }
    return GPAP_OK;
}

extern "C" gpap_status gpap_assignment_rows(gpap_engine *e, int32_t slot, int8_t *out_host,
                                            int32_t max_rows) {
    if (!e || slot < 0 || slot >= GPAP_SLOTS || max_rows < 0) return GPAP_ERR_ARG;
    GPAP_TRY(engine_use_ctx(e));
    GPAP_TRY(engine_ensure_rows(e, slot));
    Selection &s = e->slots[slot];
    const int32_t two_n = 2 * e->num_vars;
    const int32_t n_rows = std::min(max_rows, s.count);   /* never write past the caller */
    if (n_rows == 0 || two_n == 0) return GPAP_OK;        /* degenerate: nothing to unpack */
    if (!out_host) return GPAP_ERR_ARG;
    int32_t step = (int32_t)std::max<int64_t>(
        32, (int64_t)(engine_readout_slice_bytes() / (size_t)std::max(two_n, 1)));
    step = (step / 32) * 32;
    if (step <= 0) step = 32;
    GPAP_TRY(e->staging_b.reserve((size_t)std::min(step, n_rows) * (size_t)two_n));

    Kernel k;
    GPAP_TRY(e->kernels.get("unpack_rows", &k));
    int32_t NW = e->nw, tn = two_n;
    for (int32_t r0 = 0; r0 < n_rows; r0 += step) {
        const int32_t rows = std::min(step, n_rows - r0);
        CUdeviceptr rows_ptr = s.rows.ptr + (size_t)r0 * sizeof(int32_t);
        GPAP_TRY(launch(k, (unsigned)rows, engine_ceil_div(two_n, 1024),
                        {&e->abits.ptr, &rows_ptr, &e->staging_b.ptr, &tn, &NW}));
        GPAP_CU(cuMemcpyDtoH(out_host + (size_t)r0 * two_n, e->staging_b.ptr,
                             (size_t)rows * two_n));
    }
    return GPAP_OK;
}

extern "C" gpap_status gpap_flags(gpap_engine *e, int8_t *active_out, int8_t *conflict_out,
                                  int32_t capacity) {
    if (!e) return GPAP_ERR_ARG;
    if ((active_out || conflict_out) && capacity < e->K) {
        set_error("gpap_flags: buffers hold %d rows, the engine has K=%d", capacity, e->K);
        return GPAP_ERR_ARG;
    }
    GPAP_TRY(engine_use_ctx(e));
    if (active_out) GPAP_CU(cuMemcpyDtoH(active_out, e->active.ptr, (size_t)e->K));
    if (conflict_out) GPAP_CU(cuMemcpyDtoH(conflict_out, e->conflict.ptr, (size_t)e->K));
    return GPAP_OK;
}

extern "C" gpap_status gpap_num_propagated(gpap_engine *e, int32_t *out_host,
                                           int32_t capacity) {
    if (!e || !out_host) return GPAP_ERR_ARG;
    if (capacity < e->K) {
        set_error("gpap_num_propagated: out holds %d rows, the engine has K=%d", capacity, e->K);
        return GPAP_ERR_ARG;
    }
    GPAP_TRY(engine_use_ctx(e));
    std::vector<int32_t> decs((size_t)e->K);
    GPAP_CU(cuMemcpyDtoH(out_host, e->nprops.ptr, (size_t)e->K * sizeof(int32_t)));
    GPAP_CU(cuMemcpyDtoH(decs.data(), e->ndecs.ptr, (size_t)e->K * sizeof(int32_t)));
    for (int32_t i = 0; i < e->K; ++i) out_host[i] -= decs[(size_t)i];   /* implications */
    return GPAP_OK;
}

extern "C" gpap_status gpap_sat_rows(gpap_engine *e, const int32_t *rows, int32_t n_rows,
                                     int32_t *out_host) {
    /* (n, M_TOTAL) in the ORIGINAL clause order. Long clauses read their packed word's count
     * field; binary and unit clauses have no per-row word, so their non-falsified count is
     * synthesized from the row's assignment bitmap — identical values (a packed word also
     * counts non-falsified literals), with no per-row state behind them. */
    if (!e || !rows || !out_host || n_rows < 0) return GPAP_ERR_ARG;
    GPAP_TRY(gpap::engine_reject_if_packed_is_baseline(e, "gpap_sat_rows"));
    GPAP_TRY(engine_use_ctx(e));
    const int32_t M = e->num_clauses, ML = e->num_long;
    const uint32_t sat_mask = (1u << e->shift) - 1;
    /* The bitmap is only read to synthesize binary/unit counts, so a formula made entirely of
     * length>=3 clauses must not pay for it: that copy is per ROW, and this runs over a whole
     * chunk from gpap_check_sets. */
    const bool need_bits = e->n_pairs > 0;
    std::vector<int32_t> lw((size_t)ML);
    std::vector<uint32_t> aw((size_t)(need_bits ? e->nw : 0));
    for (int32_t i = 0; i < n_rows; ++i) {
        if (rows[i] < 0 || rows[i] >= e->K) return GPAP_ERR_ARG;
        if (ML)
            GPAP_CU(cuMemcpyDtoH(lw.data(), e->packed.ptr + (size_t)rows[i] * ML * sizeof(int32_t),
                                 (size_t)ML * sizeof(int32_t)));
        if (need_bits)
            GPAP_CU(cuMemcpyDtoH(aw.data(), e->abits.ptr + (size_t)rows[i] * e->nw * sizeof(int32_t),
                                 (size_t)e->nw * sizeof(int32_t)));
        auto not_falsified = [&](int32_t li) -> int32_t {
            const int32_t neg = li ^ 1;
            return (int32_t)(1u - ((aw[(size_t)(neg >> 5)] >> (neg & 31)) & 1u));
        };
        int32_t *dst = out_host + (size_t)i * M;
        for (int32_t c = 0; c < M; ++c) {
            const int32_t src = e->sat_src_host[(size_t)c];   /* THE kind/slot map, not a cursor */
            if (src >= 0) {
                dst[c] = (int32_t)((uint32_t)lw[(size_t)src] & sat_mask);
            } else {
                const int32_t p = -src - 1;                   /* merged pair index */
                dst[c] = not_falsified(e->pair_host[(size_t)(2 * p)]);
                if (p < e->num_bin)                           /* a unit's pair is (u, u) */
                    dst[c] += not_falsified(e->pair_host[(size_t)(2 * p + 1)]);
            }
        }
    }
    return GPAP_OK;
}

extern "C" gpap_status gpap_packed_rows(gpap_engine *e, const int32_t *rows, int32_t n_rows,
                                        int32_t *out_host) {
    /* RAW words, sum field included, (n, num_long) — the packed state's true width: only
     * length>=3 clauses have words. gpap_sat_rows above masks the sum off, which would hide
     * sum corruption from the CLEAN-mode identity gate (tests/test_clean_mode.py). */
    if (!e || !rows || !out_host || n_rows < 0) return GPAP_ERR_ARG;
    GPAP_TRY(engine_use_ctx(e));
    const int32_t M = e->num_long;
    for (int32_t i = 0; i < n_rows; ++i) {
        if (rows[i] < 0 || rows[i] >= e->K) return GPAP_ERR_ARG;
        if (M)
            GPAP_CU(cuMemcpyDtoH(out_host + (size_t)i * M,
                                 e->packed.ptr + (size_t)rows[i] * M * sizeof(int32_t),
                                 (size_t)M * sizeof(int32_t)));
    }
    return GPAP_OK;
}

extern "C" gpap_status gpap_engine_flushed_assignment(gpap_engine *e, int8_t *out_host) {
    if (!e || !out_host) return GPAP_ERR_ARG;
    GPAP_TRY(engine_use_ctx(e));
    std::vector<uint32_t> words((size_t)e->nw);
    GPAP_CU(cuMemcpyDtoH(words.data(), e->abits_flushed.ptr, (size_t)e->nw * sizeof(uint32_t)));
    const int32_t two_n = 2 * e->num_vars;
    for (int32_t l = 0; l < two_n; ++l)
        out_host[l] = (int8_t)((words[(size_t)(l >> 5)] >> (l & 31)) & 1u);
    return GPAP_OK;
}
