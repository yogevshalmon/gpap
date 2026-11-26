/* The formula's side of an engine (gpap_internal.hpp, FormulaState): its upload, and the one way
 * it changes shape, gpap_engine_add_clauses, which builds a new state beside the old one and
 * swaps it in. */
#include "gpap_internal.hpp"

#include <algorithm>
#include <memory>
#include <vector>

namespace {

/* Upload `n` bytes into one of the formula's buffers, counted on the formula's holder. */
gpap_status put(gpap::FormulaState &F, gpap::DevBuf &dst, const void *host, size_t n) {
    GPAP_TRY(dst.reserve(F.mem, n));
    if (n) GPAP_TRY(F.dev->gpu->upload(dst.ptr, host, n));
    return GPAP_OK;
}

}  // namespace

gpap::FormulaState::~FormulaState() {
    if (dev && dev->gpu) dev->gpu->make_current();   /* the buffers free themselves on it */
}

gpap_status gpap::formula_state_upload(const std::shared_ptr<Device> &dev, const gpap_formula *f,
                                       int32_t shift, bool with_init,
                                       std::shared_ptr<FormulaState> *out) {
    /* Freed on every early return: its buffers give their bytes back as it goes. */
    auto F = std::make_shared<FormulaState>();
    F->dev = dev;
    F->mem.gpu = dev->gpu.get();
    F->mem.account = dev->account;
    F->num_vars = f->num_vars;
    F->num_clauses = f->num_clauses;
    F->num_long = f->num_long;
    F->num_bin = f->num_bin;
    F->shift = shift;
    F->total_lits = f->total_lits;
    F->max_clause_len = f->max_clause_len;
    F->has_empty_clause = f->has_empty_clause;
    F->nw = std::max(1, (2 * f->num_vars + 31) / 32);

    GPAP_TRY(put(*F, F->occ_off, f->occ_offsets.data(), f->occ_offsets.size() * sizeof(int32_t)));
    GPAP_TRY(put(*F, F->occ_mid, f->occ_mid.data(), f->occ_mid.size() * sizeof(int32_t)));
    GPAP_TRY(put(*F, F->occ_cls, f->occ_clauses.data(), f->occ_clauses.size() * sizeof(int32_t)));
    if (with_init) {
        if (shift == f->shift) {
            GPAP_TRY(put(*F, F->packed_init, f->packed_init.data(),
                         f->packed_init.size() * sizeof(int32_t)));
        } else {   /* pinned to the other split: re-pack the LONG clauses from sizes + sums */
            std::vector<int32_t> pk;
            pack_long_init(f, shift, pk);
            GPAP_TRY(put(*F, F->packed_init, pk.data(), pk.size() * sizeof(int32_t)));
        }
    }
    /* Soundness pairs: binaries plus every unit u as (u, u), since none carries a word. */
    std::vector<int32_t> pairs;
    pairs.reserve(f->bin_pairs.size() + 2 * f->unit_lits.size());
    pairs.insert(pairs.end(), f->bin_pairs.begin(), f->bin_pairs.end());
    for (int32_t u : f->unit_lits) {
        pairs.push_back(u);
        pairs.push_back(u);
    }
    F->n_pairs = (int32_t)(pairs.size() / 2);
    GPAP_TRY(put(*F, F->pair_check, pairs.data(), pairs.size() * sizeof(int32_t)));
    F->pair_host = std::move(pairs);   /* the SAME list, not a second copy */
    /* Host side of gpap_sat_rows' original-order synthesis (the formula outlives f). */
    F->sat_src_host = f->sat_src;
    *out = std::move(F);
    return GPAP_OK;
}
