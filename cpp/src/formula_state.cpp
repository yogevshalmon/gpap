/* The formula's side of an engine (gpap_internal.hpp, FormulaState): its upload, and the one way
 * it changes shape, gpap_engine_add_clauses, which builds a new state beside the old one and
 * swaps it in. */
#include "gpap_internal.hpp"

#include <algorithm>
#include <climits>
#include <memory>
#include <string>
#include <vector>

using gpap::get_error;
using gpap::set_error;

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

gpap_status gpap::formula_state_keep_host(FormulaState &F, const gpap_formula *f) {
    return guard_entry("gpap_engine_create", [&]() -> gpap_status {
        std::unique_ptr<gpap_formula> copy(new gpap_formula(*f));
        copy->packed_init.clear();            /* only a first flush reads it */
        copy->packed_init.shrink_to_fit();
        F.host = std::move(copy);
        return GPAP_OK;
    });
}

/* ------------------------------------------------------------------ growing a formula */

gpap_status gpap::formula_growth_plan(const FormulaState &old, const int32_t *flat_lits,
                                      const int32_t *clause_lens, int32_t n, FormulaGrowth *g) {
    const gpap_formula *h = old.host.get();
    if (!h || !old.flush_ready) {
        set_error("gpap_engine_add_clauses: the engine has no built formula to add to");
        return GPAP_ERR_ARG;
    }
    /* The new clauses alone through the builder: every literal checked, each clause taken as the
     * set of its literals and an empty one flagged, exactly as the formula's own were. */
    gpap_formula *raw = nullptr;
    gpap_status st = gpap_formula_from_clauses(old.num_vars, flat_lits, clause_lens, n, &raw);
    if (st != GPAP_OK) {
        const std::string why = get_error();
        set_error("gpap_engine_add_clauses: %s", why.c_str());
        return st;
    }
    std::unique_ptr<gpap_formula> add(raw);
    /* N never changes, so neither does the split: a clause past its length bound is refused.
     * The engine's split, which may be a pin, not the one the builder would pick. */
    const PackedSplit *sp = nullptr;
    for (const auto &cand : kPackedSplits)
        if (cand.shift == old.shift) sp = &cand;
    if (!sp || add->max_clause_len > sp->max_len) {
        set_error("gpap_engine_add_clauses: a clause of %d literals does not fit the engine's "
                  "packed split (shift %d holds at most %d)", add->max_clause_len, old.shift,
                  sp ? sp->max_len : 0);
        return GPAP_ERR_ENVELOPE;
    }
    if ((int64_t)h->num_clauses + add->num_clauses > INT32_MAX) {
        set_error("gpap_engine_add_clauses: %d + %d clauses exceed the int32 clause index",
                  h->num_clauses, add->num_clauses);
        return GPAP_ERR_ARG;
    }

    /* The old clauses, then the new, through the builder: a rebuild's formula, so every old
     * clause keeps its index (its sat_rows column, its packed word) and the new ones follow. */
    int64_t n_old = 0, n_add = 0;
    GPAP_TRY(gpap_formula_clauses(h, nullptr, &n_old));
    GPAP_TRY(gpap_formula_clauses(add.get(), nullptr, &n_add));
    std::vector<int32_t> flat((size_t)(n_old + n_add));
    if (n_old) GPAP_TRY(gpap_formula_clauses(h, flat.data(), nullptr));
    if (n_add) GPAP_TRY(gpap_formula_clauses(add.get(), flat.data() + n_old, nullptr));
    std::vector<int32_t> lens(h->clause_sizes);
    lens.insert(lens.end(), add->clause_sizes.begin(), add->clause_sizes.end());
    gpap_formula *built = nullptr;
    st = gpap_formula_from_clauses(old.num_vars, flat.data(), lens.data(), (int32_t)lens.size(),
                                   &built);
    if (st != GPAP_OK) {
        const std::string why = get_error();
        set_error("gpap_engine_add_clauses: %s", why.c_str());
        return st;
    }
    g->built.reset(built);
    /* Empty clauses are dropped by the builder and only flagged: carry both formulas' flags. */
    g->built->has_empty_clause |= h->has_empty_clause | add->has_empty_clause;
    g->built->support = h->support;
    g->built->packed_init.clear();            /* only a first flush reads it (FormulaState) */
    g->built->packed_init.shrink_to_fit();
    if (g->built->num_long != old.num_long + add->num_long ||
        g->built->num_clauses != old.num_clauses + add->num_clauses) {
        set_error("gpap_engine_add_clauses: the rebuilt formula does not extend the old one "
                  "(%d long / %d clauses, expected %d / %d)", g->built->num_long,
                  g->built->num_clauses, old.num_long + add->num_long,
                  old.num_clauses + add->num_clauses);
        return GPAP_ERR_ARG;
    }

    /* The old baseline row, read once: the old clauses' words carry over, and the new clauses'
     * words and the literals they force are read off its assignment. */
    Gpu *gpu = old.dev->gpu.get();
    g->words.assign((size_t)g->built->num_long, 0);
    g->bits.assign((size_t)old.nw, 0);
    if (old.num_long)
        GPAP_TRY(gpu->download(g->words.data(), old.packed_flushed.ptr, 0,
                               (size_t)old.num_long * sizeof(int32_t)));
    GPAP_TRY(gpu->download(g->bits.data(), old.abits_flushed.ptr, 0,
                           (size_t)old.nw * sizeof(int32_t)));
    auto is_true = [&](int32_t l) {
        return (((uint32_t)g->bits[(size_t)(l >> 5)] >> (l & 31)) & 1u) != 0;
    };
    /* A word is the count and the sum of the literals not yet false, as the kernels keep it: each
     * falsified literal takes 1 + (its index << shift) off its clause's word. */
    const uint32_t sum_mask = (uint32_t)((1ull << (32 - old.shift)) - 1);
    int32_t long_id = old.num_long;
    const int32_t *lit = flat.data() + n_old;
    for (int32_t c = 0; c < add->num_clauses; ++c) {
        const int32_t k = add->clause_sizes[(size_t)c];
        uint32_t count = 0, sum = 0;
        bool satisfied = false;
        int32_t open = -1;
        for (int32_t i = 0; i < k; ++i) {
            const int32_t l = (int32_t)lit_index_from_dimacs(lit[i], old.num_vars);
            if (is_true(l)) satisfied = true;
            if (!is_true(l ^ 1)) {
                ++count;
                sum += (uint32_t)l;
                open = l;
            }
        }
        if (is_long_clause(k))
            g->words[(size_t)long_id++] = (int32_t)(count | ((sum & sum_mask) << old.shift));
        if (!satisfied) {
            if (count == 0) g->conflict = true;
            else if (count == 1) g->forced.push_back(open);
        }
        lit += k;
    }
    if (g->built->has_empty_clause) g->conflict = true;
    return GPAP_OK;
}

gpap_status gpap::formula_growth_upload(const FormulaState &old, FormulaGrowth *g) {
    std::shared_ptr<FormulaState> F;
    GPAP_TRY(formula_state_upload(old.dev, g->built.get(), old.shift, /*with_init=*/false, &F));
    if (!g->words.empty())
        GPAP_TRY(put(*F, F->packed_flushed, g->words.data(), g->words.size() * sizeof(int32_t)));
    GPAP_TRY(put(*F, F->abits_flushed, g->bits.data(), g->bits.size() * sizeof(int32_t)));
    F->flush_conflict = (old.flush_conflict || g->conflict) ? 1 : 0;
    F->flush_ready = 1;
    F->host = std::move(g->built);
    g->state = std::move(F);
    return GPAP_OK;
}
