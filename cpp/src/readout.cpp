/* Row selections and readout: the completion test, the soundness check and the two gathers the
 * sampler needs, all on device — the host receives a count or a small matrix, never (K, M). */
#include "gpap_internal.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

using gpap::buf;
using gpap::DevBuf;
using gpap::i32;
using gpap::set_error;

namespace gpap {
gpap_status engine_use_ctx(gpap_engine *e);
gpap_status engine_stage(gpap_engine *e, DevBuf &dst, const void *host, size_t n);
gpap_status engine_ensure_rows(gpap_engine *e, int32_t slot);
size_t engine_readout_slice_bytes();
}  // namespace gpap

using gpap::engine_ensure_rows;
using gpap::engine_readout_slice_bytes;
using gpap::engine_stage;
using gpap::engine_use_ctx;

namespace {

/* Rows one readout slice takes, for `per_row` bytes a row: the slice's bytes, or what a memory
 * limit or a host's grant leaves beside everything GPAP holds where that is less (staging_b's
 * own bytes count back: a grow frees it first). Never under 32, which keeps the row-pointer
 * offset 16-byte aligned; the exact bounds set that much aside when they size the batch
 * (engine.cpp, exact_terms). */
int32_t slice_rows(gpap_engine *e, int64_t per_row) {
    const int64_t bytes = std::min<int64_t>(
        (int64_t)engine_readout_slice_bytes(),
        gpap::engine_transient_room(e, (int64_t)e->staging_b.bytes));
    int64_t step = std::max<int64_t>(32, bytes / std::max<int64_t>(per_row, 1));
    step = std::min<int64_t>(step, INT32_MAX);
    step = (step / 32) * 32;
    return step > 0 ? (int32_t)step : 32;
}

/* Reduce an (n,) int8 flag array into e->scratch_i32[0] and read it back (one sync). */
gpap_status reduce_flags(gpap_engine *e, DevBuf &flags, int32_t n, int32_t *out) {
    GPAP_TRY(e->gpu->run("sum_i8_to", 1, 1, {buf(flags), i32(n), buf(e->scratch_i32)}));
    return e->gpu->download(out, e->scratch_i32.ptr, 0, sizeof(int32_t));
}

}  // namespace

extern "C" gpap_status gpap_pair_mask_create(gpap_engine *e, const int32_t *vars, int32_t n_vars,
                                             int32_t *mask_id_out) {
    if (!e || !mask_id_out) return GPAP_ERR_ARG;
    GPAP_TRY(engine_use_ctx(e));
    /* Even bit 2v per selected variable; padding pairs beyond N stay 0, so the mask is needed. */
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
    gpap::FormulaBuf *mask = new gpap::FormulaBuf();
    gpap_status st = engine_stage(e, *mask, bits.data(), bits.size() * sizeof(uint32_t));
    if (st != GPAP_OK) {
        delete mask;
        return st;
    }
    e->pair_masks.push_back(mask);
    *mask_id_out = (int32_t)e->pair_masks.size() - 1;
    return GPAP_OK;
}

extern "C" gpap_status gpap_select_complete(gpap_engine *e, int32_t mask_id, int32_t slot,
                                            int32_t *count_out) {
    if (!e || slot < 0 || slot >= GPAP_SLOTS || mask_id < 0 || mask_id >= (int32_t)e->pair_masks.size())
        return GPAP_ERR_ARG;
    GPAP_TRY(engine_use_ctx(e));
    Selection &s = e->slots[slot];
    GPAP_TRY(s.mask.reserve(e->mem, (size_t)e->K));

    GPAP_TRY(e->gpu->run("complete_mask", e->K, 1,
                         {buf(e->abits), buf(*e->pair_masks[(size_t)mask_id]), buf(e->active),
                          buf(s.mask), i32(e->nw)}));
    s.rows_valid = false;
    s.selected = true;
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
    /* Invalidated BEFORE the fallible upload: the host side has already moved, so a failed one
     * would pair the NEW count with the OLD device rows, with no error downstream. */
    s.rows_valid = false;
    s.selected = true;
    GPAP_TRY(engine_stage(e, s.rows, s.host_rows.data(),
                          s.host_rows.size() * sizeof(int32_t)));
    s.rows_valid = true;
    return GPAP_OK;
}

extern "C" gpap_status gpap_selection_rows(gpap_engine *e, int32_t slot, int32_t *rows_out,
                                           int32_t capacity) {
    if (!e || slot < 0 || slot >= GPAP_SLOTS || !rows_out || capacity < 0) return GPAP_ERR_ARG;
    GPAP_TRY(engine_use_ctx(e));
    GPAP_TRY(engine_ensure_rows(e, slot));
    const Selection &s = e->slots[slot];
    /* Bounded by the caller's capacity: the selection can be recompacted since the count. */
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

    GPAP_TRY(e->staging_b.reserve(e->mem, (size_t)n));
    /* Long clauses against their packed words, binary/unit against the bitmap via the
     * (n_pairs, 2) pair list (helpers.py::sound_rows). */
    const std::string key = "sound_rows.s" + std::to_string(e->shift);
    GPAP_TRY(e->gpu->run(key.c_str(), n, 1,
                         {buf(e->packed), buf(e->abits), buf(s.rows), i32(e->num_long), i32(e->nw),
                          buf(e->pair_check), i32(e->n_pairs), buf(e->staging_b)}));
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
    const int32_t n_rows = std::min(max_rows, s.count);
    /* Zero rows or columns are legal: a fully determined formula has nothing to gather. */
    if (n_rows == 0 || n_cols == 0) return GPAP_OK;

    GPAP_TRY(engine_stage(e, e->staging_a, cols, (size_t)n_cols * sizeof(int32_t)));
    /* Slice the rows so the (rows, n_cols) transient stays bounded. */
    const int32_t step = slice_rows(e, n_cols);
    GPAP_TRY(e->staging_b.reserve(e->mem, (size_t)std::min(step, n_rows) * (size_t)n_cols));

    for (int32_t r0 = 0; r0 < n_rows; r0 += step) {
        const int32_t rows = std::min(step, n_rows - r0);
        /* this slice's rows: an input view into the selection's row list */
        GPAP_TRY(e->gpu->run("var_bits", rows, n_cols,
                             {buf(e->abits), buf(s.rows, (size_t)r0 * sizeof(int32_t)),
                              buf(e->staging_a), buf(e->staging_b), i32(n_cols), i32(e->nw)}));
        GPAP_TRY(e->gpu->download(out_host + (size_t)r0 * n_cols, e->staging_b.ptr, 0,
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
    const int32_t n_rows = std::min(max_rows, s.count);
    if (n_rows == 0 || two_n == 0) return GPAP_OK;
    if (!out_host) return GPAP_ERR_ARG;
    const int32_t step = slice_rows(e, two_n);
    /* No single-buffer check needed: these rows are the batch's, and a row's 2N bytes here are
     * fewer than its trail's 4N, a buffer engine_batch already keeps under the device's limit. */
    GPAP_TRY(e->staging_b.reserve(e->mem, (size_t)std::min(step, n_rows) * (size_t)two_n));

    for (int32_t r0 = 0; r0 < n_rows; r0 += step) {
        const int32_t rows = std::min(step, n_rows - r0);
        GPAP_TRY(e->gpu->run("unpack_rows", rows, two_n,
                             {buf(e->abits), buf(s.rows, (size_t)r0 * sizeof(int32_t)),
                              buf(e->staging_b), i32(two_n), i32(e->nw)}));
        GPAP_TRY(e->gpu->download(out_host + (size_t)r0 * two_n, e->staging_b.ptr, 0,
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
    if (active_out) GPAP_TRY(e->gpu->download(active_out, e->active.ptr, 0, (size_t)e->K));
    if (conflict_out) GPAP_TRY(e->gpu->download(conflict_out, e->conflict.ptr, 0, (size_t)e->K));
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
    GPAP_TRY(e->gpu->download(out_host, e->nprops.ptr, 0, (size_t)e->K * sizeof(int32_t)));
    GPAP_TRY(e->gpu->download(decs.data(), e->ndecs.ptr, 0, (size_t)e->K * sizeof(int32_t)));
    for (int32_t i = 0; i < e->K; ++i) out_host[i] -= decs[(size_t)i];   /* implications */
    return GPAP_OK;
}

extern "C" gpap_status gpap_sat_rows(gpap_engine *e, const int32_t *rows, int32_t n_rows,
                                     int32_t *out_host) {
    /* (n, M_TOTAL) in the ORIGINAL clause order. Long clauses read their packed word's count
     * field; binary and unit clauses have none, so theirs is synthesized from the bitmap. */
    if (!e || !rows || !out_host || n_rows < 0) return GPAP_ERR_ARG;
    GPAP_TRY(gpap::engine_reject_if_packed_is_baseline(e, "gpap_sat_rows"));
    GPAP_TRY(engine_use_ctx(e));
    const int32_t M = e->num_clauses, ML = e->num_long;
    const uint32_t sat_mask = (1u << e->shift) - 1;
    /* The bitmap is only read to synthesize binary/unit counts, and that copy is per ROW, so an
     * all-long formula must not pay for it. */
    const bool need_bits = e->n_pairs > 0;
    std::vector<int32_t> lw((size_t)ML);
    std::vector<uint32_t> aw((size_t)(need_bits ? e->nw : 0));
    for (int32_t i = 0; i < n_rows; ++i) {
        if (rows[i] < 0 || rows[i] >= e->K) return GPAP_ERR_ARG;
        if (ML)
            GPAP_TRY(e->gpu->download(lw.data(), e->packed.ptr,
                                      (size_t)rows[i] * ML * sizeof(int32_t),
                                      (size_t)ML * sizeof(int32_t)));
        if (need_bits)
            GPAP_TRY(e->gpu->download(aw.data(), e->abits.ptr,
                                      (size_t)rows[i] * e->nw * sizeof(int32_t),
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
                const int32_t p = -src - 1;
                dst[c] = not_falsified(e->pair_host[(size_t)(2 * p)]);
                if (p < e->num_bin)
                    dst[c] += not_falsified(e->pair_host[(size_t)(2 * p + 1)]);
            }
        }
    }
    return GPAP_OK;
}

extern "C" gpap_status gpap_packed_rows(gpap_engine *e, const int32_t *rows, int32_t n_rows,
                                        int32_t *out_host) {
    /* RAW words, sum field included, (n, num_long): gpap_sat_rows masks the sum off, which would
     * hide sum corruption from the CLEAN-mode gate (tests/engine/test_clean_mode.py). */
    if (!e || !rows || !out_host || n_rows < 0) return GPAP_ERR_ARG;
    GPAP_TRY(engine_use_ctx(e));
    const int32_t M = e->num_long;
    for (int32_t i = 0; i < n_rows; ++i) {
        if (rows[i] < 0 || rows[i] >= e->K) return GPAP_ERR_ARG;
        if (M)
            GPAP_TRY(e->gpu->download(out_host + (size_t)i * M, e->packed.ptr,
                                      (size_t)rows[i] * M * sizeof(int32_t),
                                      (size_t)M * sizeof(int32_t)));
    }
    return GPAP_OK;
}

extern "C" gpap_status gpap_engine_flushed_assignment(gpap_engine *e, int8_t *out_host) {
    if (!e || !out_host) return GPAP_ERR_ARG;
    GPAP_TRY(engine_use_ctx(e));
    std::vector<uint32_t> words((size_t)e->nw);
    GPAP_TRY(e->gpu->download(words.data(), e->abits_flushed.ptr, 0,
                              (size_t)e->nw * sizeof(uint32_t)));
    const int32_t two_n = 2 * e->num_vars;
    for (int32_t l = 0; l < two_n; ++l)
        out_host[l] = (int8_t)((words[(size_t)(l >> 5)] >> (l & 31)) & 1u);
    return GPAP_OK;
}
