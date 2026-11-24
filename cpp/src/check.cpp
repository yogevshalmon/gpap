/* Batched assumption checking: the loops behind both AssumptionChecker faces, so chunking,
 * validate-before-any-launch and padding have one home. Entry point 1 is reset(K) + apply, one
 * launch per K sets (check_in_chunks); a flags-only call runs on the job queue instead when it
 * engages (check_on_queue). This is the packaging around them, over the same public ABI. */
#include "gpap_internal.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

using gpap::lit_index_from_dimacs;
using gpap::set_error;

namespace {

/* Validate the WHOLE input before any launch: an index past a row's bitmap makes atomic_or
 * corrupt a LATER row, and an error after some chunks ran leaves a partial-looking result.
 * Sets *max_len to the longest set, at least 1. */
gpap_status validate_sets(gpap_engine *e, const int32_t *lits, const int64_t *offsets, int64_t n,
                          const int8_t *conflict_out, const int32_t *propagated_out,
                          int64_t *max_len) {
    if (!offsets || !conflict_out || !propagated_out) {
        set_error("gpap_check_sets: %s must not be NULL",
                  !offsets ? "offsets" : !conflict_out ? "conflict_out" : "propagated_out");
        return GPAP_ERR_ARG;
    }
    if (offsets[0] != 0) {
        set_error("gpap_check_sets: offsets[0] must be 0 (got %lld)",
                  static_cast<long long>(offsets[0]));
        return GPAP_ERR_ARG;
    }
    *max_len = 1;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t len = offsets[i + 1] - offsets[i];
        if (len < 0) {
            set_error("gpap_check_sets: offsets must be non-decreasing (set %lld)",
                      static_cast<long long>(i));
            return GPAP_ERR_ARG;
        }
        *max_len = std::max(*max_len, len);
    }
    if (*max_len > std::numeric_limits<int32_t>::max()) {
        set_error("gpap_check_sets: a set with %lld literals exceeds the int32 apply width",
                  static_cast<long long>(*max_len));
        return GPAP_ERR_ARG;
    }
    const int64_t total = offsets[n];
    if (total > 0 && !lits) {
        set_error("gpap_check_sets: lits is NULL but the sets hold %lld literals",
                  static_cast<long long>(total));
        return GPAP_ERR_ARG;
    }
    for (int64_t i = 0; i < total; ++i) {
        if (lit_index_from_dimacs(lits[i], e->num_vars) < 0) {
            set_error("assumption literal %d out of range (formula has %d variables; literals "
                      "are DIMACS, 1..%d or -1..-%d, never 0)",
                      lits[i], e->num_vars, e->num_vars, e->num_vars);
            return GPAP_ERR_ARG;
        }
    }
    return GPAP_OK;
}

/* FLAGS ONLY, on the queue: a launch takes a block of sets over the call's few rows; a row
 * takes its next set as it finishes one and puts back the words it touched
 * (QueuedCall::launch). That restore is why this path cannot serve assign_out or sat_out,
 * which read a row after its set. A block is padded to its own longest set and cut where its
 * matrix would pass kQueueMaxLits. No deadline: a check runs every set. */
gpap_status check_on_queue(gpap_engine *e, gpap::QueuedCall &q, const int32_t *lits,
                           const int64_t *offsets, int64_t n, int8_t *conflict_out,
                           int32_t *propagated_out) {
    std::vector<int32_t> jobs;
    for (int64_t a = 0; a < n;) {
        const int64_t block = q.next_block(std::numeric_limits<double>::infinity());
        int64_t b = a, L = 1;
        while (b < n && b - a < block) {
            const int64_t L2 = std::max(L, offsets[b + 1] - offsets[b]);
            if (b > a && (b - a + 1) * L2 > gpap::kQueueMaxLits) break;
            L = L2;
            ++b;
        }
        jobs.assign(static_cast<size_t>(b - a) * static_cast<size_t>(L), -1);
        for (int64_t i = a; i < b; ++i)
            for (int64_t j = offsets[i]; j < offsets[i + 1]; ++j)
                jobs[static_cast<size_t>(i - a) * static_cast<size_t>(L) +
                     static_cast<size_t>(j - offsets[i])] =
                    static_cast<int32_t>(lit_index_from_dimacs(lits[j], e->num_vars));
        GPAP_TRY(q.launch(jobs.data(), b - a, static_cast<int32_t>(L), conflict_out + a,
                          propagated_out + a));
        a = b;
    }
    return GPAP_OK;
}

/* In chunks of `step` sets: a reset and one launch per chunk, whatever the call reads. */
gpap_status check_in_chunks(gpap_engine *e, const int32_t *lits, const int64_t *offsets,
                            int64_t n, int64_t step, int8_t *conflict_out,
                            int32_t *propagated_out, int8_t *assign_out, int32_t *sat_out) {
    const size_t row_lits = 2 * static_cast<size_t>(e->num_vars);
    std::vector<int64_t> decisions;
    std::vector<int32_t> rows;
    for (int64_t a = 0; a < n; a += step) {
        const int64_t b = std::min(a + step, n);
        const int32_t K = static_cast<int32_t>(b - a);
        /* Pad to the CHUNK's longest set, not the global one: (K, L) is the per-chunk upload. */
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
        /* The CLEAN kernel restores the packed words its launch touched, so the next reset skips
         * the broadcast, and keeps the bitmap at the fixpoint: flags, counts and the assignment
         * readout (a bitmap readout) are exact after it. Sat counts read the packed words, so a
         * call that wants them takes the plain kernel. Afterwards a packed readout of these rows
         * is refused, never answered from the baseline (engine_reject_if_packed_is_baseline). */
        if (sat_out)
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

}  // namespace

extern "C" int64_t gpap_check_chunk_size(gpap_engine *e, int64_t n_sets, int64_t max_lits) {
    if (!e) return 0;
    if (n_sets <= 0) return 1;      /* the documented floor: a launch always holds >= 1 set */
    /* The engine's batch, with the (chunk, L) decision matrix only this entry point allocates. */
    const int32_t k = gpap::engine_batch(e, 8ll * std::max<int64_t>(max_lits, 1), 0);
    return std::min<int64_t>(k, n_sets);
}

extern "C" gpap_status gpap_check_sets(gpap_engine *e, const int32_t *lits,
                                       const int64_t *offsets, int64_t n, int32_t chunk,
                                       int8_t *conflict_out, int32_t *propagated_out,
                                       int8_t *assign_out, int32_t *sat_out) {
    /* Every guard names itself: gpap_last_error() is a thread-local nothing clears, so a bare
     * return would leave the caller reading an earlier call's message. */
    if (!e || n < 0) {
        set_error("gpap_check_sets: %s", !e ? "engine must not be NULL" : "n must be >= 0");
        return GPAP_ERR_ARG;
    }
    if (n == 0) return GPAP_OK;
    int64_t max_len = 1;
    GPAP_TRY(validate_sets(e, lits, offsets, n, conflict_out, propagated_out, &max_len));

    /* The sets one chunk takes: the chunk loop's step, and the queue's first block. */
    int64_t step = chunk > 0 ? chunk : gpap_check_chunk_size(e, n, max_len);
    step = std::max<int64_t>(step, 1);
    if (!assign_out && !sat_out) {
        /* Flags only: the queue, when it engages (`chunk` > 0 pins its rows). */
        gpap::QueuedCall q(e);
        GPAP_TRY(q.begin(chunk, n, step));
        if (q.engaged())
            return check_on_queue(e, q, lits, offsets, n, conflict_out, propagated_out);
    }
    return check_in_chunks(e, lits, offsets, n, step, conflict_out, propagated_out, assign_out,
                           sat_out);
}
