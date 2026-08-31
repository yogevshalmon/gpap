/* Batched assumption checking: THE chunk loop behind both AssumptionChecker faces.
 *
 * Entry point 1 is reset(K) + apply — one launch, K sets. What every caller then wants is the
 * same packaging: ANY number of sets, chunked to what the GPU holds, validated before any work,
 * results reassembled in input order. That loop used to be written twice, once per face
 * (gpap::AssumptionChecker::check and Python's AssumptionChecker.check_assumptions), and the
 * two had already drifted — the Python face validated every literal before the first launch
 * while the C++ face threw mid-loop, after some chunks had already run. Same lesson as the
 * probe sweep (cpp/src/probe.cpp): a rule callers can observe must have one home.
 *
 * Everything here goes through the public ABI calls the faces themselves used
 * (reset/apply/flags/readout), so this file adds packaging, not a second engine path.
 */
#include "gpap_internal.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

using gpap::lit_index_from_dimacs;
using gpap::set_error;

extern "C" int64_t gpap_check_chunk_size(gpap_engine *e, int64_t n_sets, int64_t max_lits) {
    if (!e) return 0;
    if (n_sets <= 0) return 1;      /* the documented floor: a launch always holds >= 1 set */
    /* The engine's own per-row state — the shared exact model, with THIS engine's real trail
     * cap — plus the (chunk, L) int64 decision matrix this entry point allocates and the
     * sampler's path never does. Read against LIVE free memory each call. */
    const int64_t per_row = gpap_row_bytes(e->num_long, e->num_vars, e->trail_cap) +
                            8ll * std::max<int64_t>(max_lits, 1);
    const int64_t budget = gpap_mem_available(e->device_ordinal, 0.80);
    int64_t k = per_row > 0 ? budget / per_row : 1;
    k = std::min<int64_t>(k, gpap_default_k_cap(e->device_ordinal));
    k = std::max<int64_t>(k, 1);
    return std::min<int64_t>(k, n_sets);
}

extern "C" gpap_status gpap_check_sets(gpap_engine *e, const int32_t *lits,
                                       const int64_t *offsets, int64_t n, int32_t chunk,
                                       int8_t *conflict_out, int32_t *propagated_out,
                                       int8_t *assign_out, int32_t *sat_out) {
    if (!e || n < 0) return GPAP_ERR_ARG;
    if (n == 0) return GPAP_OK;
    if (!offsets || !conflict_out || !propagated_out) return GPAP_ERR_ARG;
    if (offsets[0] != 0) {
        set_error("gpap_check_sets: offsets[0] must be 0 (got %lld)",
                  static_cast<long long>(offsets[0]));
        return GPAP_ERR_ARG;
    }

    /* Validate the WHOLE input before any launch. An out-of-range literal must never reach the
     * kernel (an internal index past a row's bitmap makes atomic_or corrupt a LATER row — see
     * gpap_probe_tuples), and an error thrown after some chunks ran would leave a partially
     * filled result indistinguishable from a full one. One pass; being wrong is undetectable. */
    int64_t max_len = 1;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t len = offsets[i + 1] - offsets[i];
        if (len < 0) {
            set_error("gpap_check_sets: offsets must be non-decreasing (set %lld)",
                      static_cast<long long>(i));
            return GPAP_ERR_ARG;
        }
        max_len = std::max(max_len, len);
    }
    if (max_len > std::numeric_limits<int32_t>::max()) {
        set_error("gpap_check_sets: a set with %lld literals exceeds the int32 apply width",
                  static_cast<long long>(max_len));
        return GPAP_ERR_ARG;
    }
    const int64_t total = offsets[n];
    if (total > 0 && !lits) return GPAP_ERR_ARG;
    for (int64_t i = 0; i < total; ++i) {
        if (lit_index_from_dimacs(lits[i], e->num_vars) < 0) {
            set_error("assumption literal %d out of range (formula has %d variables; literals "
                      "are DIMACS, 1..%d or -1..-%d, never 0)",
                      lits[i], e->num_vars, e->num_vars, e->num_vars);
            return GPAP_ERR_ARG;
        }
    }

    int64_t step = chunk > 0 ? chunk : gpap_check_chunk_size(e, n, max_len);
    step = std::max<int64_t>(step, 1);
    const size_t row_lits = 2 * static_cast<size_t>(e->num_vars);
    std::vector<int64_t> decisions;
    std::vector<int32_t> rows;
    for (int64_t a = 0; a < n; a += step) {
        const int64_t b = std::min(a + step, n);
        const int32_t K = static_cast<int32_t>(b - a);
        /* Pad to the CHUNK's longest set (never the global one): the (K, L) upload is the
         * per-chunk cost, and -1 padding is skipped by the kernel either way. */
        int64_t L = 1;
        for (int64_t i = a; i < b; ++i) L = std::max(L, offsets[i + 1] - offsets[i]);
        decisions.assign(static_cast<size_t>(K) * static_cast<size_t>(L), -1);
        for (int64_t i = a; i < b; ++i)
            for (int64_t j = offsets[i]; j < offsets[i + 1]; ++j)
                decisions[static_cast<size_t>(i - a) * static_cast<size_t>(L) +
                          static_cast<size_t>(j - offsets[i])] =
                    lit_index_from_dimacs(lits[j], e->num_vars);

        GPAP_TRY(gpap_engine_reset(e, K));   /* the LAST chunk is short: reset to its own size
                                              * (this is also where GPAP_MODE_AUTO resolves) */
        /* Flags-only calls take the CLEAN kernel: the launch restores its own rows, so the
         * next chunk's reset skips the K*M broadcast. Calls that read STATE (assignments /
         * sat_count) need it to SURVIVE the launch — they keep the plain kernel. */
        if (assign_out || sat_out)
            GPAP_TRY(gpap_engine_apply(e, decisions.data(), static_cast<int32_t>(L)));
        else
            GPAP_TRY(gpap_engine_apply_clean(e, decisions.data(), static_cast<int32_t>(L)));
        GPAP_TRY(gpap_flags(e, nullptr, conflict_out + a, K));
        GPAP_TRY(gpap_num_propagated(e, propagated_out + a, K));
        if (assign_out || sat_out) {
            rows.resize(static_cast<size_t>(K));
            std::iota(rows.begin(), rows.end(), 0);
        }
        if (assign_out) {
            GPAP_TRY(gpap_selection_set(e, 2, rows.data(), K));   /* slot 2: the scratch slot */
            GPAP_TRY(gpap_assignment_rows(e, 2, assign_out + static_cast<size_t>(a) * row_lits,
                                          K));
        }
        if (sat_out)
            GPAP_TRY(gpap_sat_rows(e, rows.data(), K,
                                   sat_out + static_cast<size_t>(a) *
                                                 static_cast<size_t>(e->num_clauses)));
    }
    return GPAP_OK;
}
