/* CNF formula construction: DIMACS parsing, the negated-occurrence CSR, and the packed
 * clause-word initial state. Port of gpap/engine/formula.py — and now the ONLY implementation
 * (the Python class is a facade over this, so the two can never drift).
 *
 * Literal indexing (the 2N scheme): +v -> 2(v-1), -v -> 2(v-1)+1, negation l ^ 1.
 * occ[l] = the clauses where ~l appears, so setting l TRUE falsifies a literal in exactly
 * those clauses. Within a literal's list, clause indices stay ASCENDING (the Python build used
 * a stable argsort; a counting-sort fill in clause order reproduces it exactly).
 */
#include "gpap_internal.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using gpap::set_error;

/* ------------------------------------------------------------------ helpers */

static inline int32_t lit_index(int32_t ext) {
    return 2 * (std::abs(ext) - 1) + (ext > 0 ? 0 : 1);
}

extern "C" gpap_status gpap_packed_envelope(int32_t num_vars, int32_t max_clause_len,
                                            int32_t *shift_out) {
    const int64_t two_n = 2ll * num_vars;
    for (const auto &s : gpap::kPackedSplits) {
        if (two_n <= s.max_two_n && max_clause_len <= s.max_len) {
            if (shift_out) *shift_out = s.shift;
            return GPAP_OK;
        }
    }
    set_error("formula outside the packed envelopes: 2N=%lld, max clause len=%d "
              "(16|16 needs 2N<=65535 and len<=65535; 8|24 needs 2N<=16777215 and len<=255)",
              (long long)two_n, max_clause_len);
    return GPAP_ERR_ENVELOPE;
}

/* ------------------------------------------------------------------ build */

static gpap_status build(int32_t num_vars, const int32_t *flat, const int32_t *lens, int32_t m,
                         gpap_formula **out) {
    /* num_vars == 0 is legal and reachable: unit-propagation / ELS simplification can
     * determine EVERY variable, leaving an empty formula that is trivially satisfied. */
    *out = nullptr;
    if (num_vars < 0 || m < 0 || (m > 0 && (!flat || !lens))) {
        set_error("gpap_formula_from_clauses: bad arguments (num_vars=%d, num_clauses=%d)",
                  num_vars, m);
        return GPAP_ERR_ARG;
    }
    gpap_formula *f = new gpap_formula();
    f->num_vars = num_vars;

    /* Pass 1: drop empty clauses, measure. */
    std::vector<int32_t> keep_len;
    std::vector<const int32_t *> keep_ptr;
    keep_len.reserve(m);
    keep_ptr.reserve(m);
    const int32_t *cur = flat;
    for (int32_t c = 0; c < m; ++c) {
        /* A negative length would walk `cur` backwards out of the caller's buffer. */
        if (lens[c] < 0) {
            set_error("clause %d has negative length %d", c, lens[c]);
            delete f;
            return GPAP_ERR_ARG;
        }
        if (lens[c] == 0) {
            f->has_empty_clause = 1;
        } else {
            keep_len.push_back(lens[c]);
            keep_ptr.push_back(cur);
            f->total_lits += lens[c];
            if (lens[c] > f->max_clause_len) f->max_clause_len = lens[c];
        }
        cur += lens[c];
    }
    f->num_clauses = (int32_t)keep_len.size();
    /* The occurrence CSR indexes with int32; past 2^31 literal slots its offsets would wrap
     * into silent corruption, so refuse rather than build something subtly wrong. */
    if (f->total_lits > 0x7FFFFFFFll) {
        set_error("formula too large: %lld literal occurrences exceeds the int32 CSR limit",
                  (long long)f->total_lits);
        delete f;
        return GPAP_ERR_ARG;
    }

    /* The packed split is the ENGINE's representation choice, so a formula outside every
     * envelope still BUILDS (shape queries, suite pre-checks) and carries shift == 0;
     * gpap_engine_create is what rejects it. */
    if (gpap_packed_envelope(num_vars, f->max_clause_len, &f->shift) != GPAP_OK) f->shift = 0;

    const int32_t M = f->num_clauses;
    const int64_t num_lits = 2ll * num_vars;
    f->clause_sizes.resize(M);
    f->lit_sum.resize(M);
    f->packed_init.resize(M);
    f->occ_offsets.assign(num_lits + 1, 0);
    f->occ_clauses.resize(f->total_lits);

    /* Pass 2: clause counters + occurrence histogram (of the NEGATED literal). */
    for (int32_t c = 0; c < M; ++c) {
        const int32_t n = keep_len[c];
        const int32_t *lits = keep_ptr[c];
        uint32_t sum = 0;
        for (int32_t i = 0; i < n; ++i) {
            const int32_t li = lit_index(lits[i]);
            if (li < 0 || li >= num_lits) {
                set_error("literal %d out of range for num_vars=%d", lits[i], num_vars);
                delete f;
                return GPAP_ERR_ARG;
            }
            sum += (uint32_t)li;
            f->occ_offsets[(li ^ 1) + 1] += 1;
        }
        f->clause_sizes[c] = n;
        f->lit_sum[c] = (int32_t)sum;             /* exact bit pattern mod 2^32 */
    }
    for (int64_t l = 0; l < num_lits; ++l) f->occ_offsets[l + 1] += f->occ_offsets[l];

    /* Pass 3: counting-sort fill, clause-ascending within each literal's list. */
    std::vector<int32_t> cursor(f->occ_offsets.begin(), f->occ_offsets.end() - 1);
    for (int32_t c = 0; c < M; ++c) {
        const int32_t *lits = keep_ptr[c];
        for (int32_t i = 0; i < keep_len[c]; ++i)
            f->occ_clauses[cursor[lit_index(lits[i]) ^ 1]++] = c;
    }

    /* Packed initial word + the formula's own unit clauses (asserted by the flush). */
    const uint32_t sum_mask =
        f->shift ? (uint32_t)((1ull << (32 - f->shift)) - 1) : 0u;
    for (int32_t c = 0; c < M; ++c) {
        if (f->shift)
            f->packed_init[c] = (int32_t)((uint32_t)f->clause_sizes[c] |
                                          (((uint32_t)f->lit_sum[c] & sum_mask) << f->shift));
        if (f->clause_sizes[c] == 1) f->unit_lits.push_back(f->lit_sum[c]);
    }
    *out = f;
    return GPAP_OK;
}

extern "C" gpap_status gpap_formula_from_clauses(int32_t num_vars, const int32_t *flat_lits,
                                                 const int32_t *clause_lens, int32_t num_clauses,
                                                 gpap_formula **out) {
    if (!out) return GPAP_ERR_ARG;
    return build(num_vars, flat_lits, clause_lens, num_clauses, out);
}

/* ------------------------------------------------------------------ DIMACS */

extern "C" gpap_status gpap_formula_from_dimacs(const char *path, gpap_formula **out) {
    if (!path || !out) return GPAP_ERR_ARG;
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        set_error("cannot open '%s'", path);
        return GPAP_ERR_IO;
    }
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::string buf((size_t)(size > 0 ? size : 0), '\0');
    if (size > 0 && fread(&buf[0], 1, (size_t)size, fp) != (size_t)size) {
        fclose(fp);
        set_error("short read on '%s'", path);
        return GPAP_ERR_IO;
    }
    fclose(fp);

    /* Token stream: clauses are 0-TERMINATED, not line-based (multiple clauses per line and
     * clauses spanning lines are both legal). 'c' comments and the SATLIB '%' end marker are
     * handled line-wise; N = max(header N, highest variable used) so trailing declared
     * variables (support vars that appear only in `c ind` lines) keep their slots. */
    std::vector<int32_t> toks;
    std::vector<int32_t> support;
    int32_t header_n = 0;
    const char *p = buf.c_str(), *end = p + buf.size();
    bool at_line_start = true;
    while (p < end) {
        if (*p == ' ' || *p == '\t' || *p == '\r') { ++p; continue; }
        if (*p == '\n') { ++p; at_line_start = true; continue; }
        if (at_line_start && (*p == 'c' || *p == 'p' || *p == '%')) {
            const char c0 = *p;
            const char *line_end = (const char *)memchr(p, '\n', (size_t)(end - p));
            if (!line_end) line_end = end;
            if (c0 == '%') break;
            if (c0 == 'p') {
                int n = 0, m = 0;
                if (sscanf(p, "p cnf %d %d", &n, &m) >= 1) header_n = n;
            } else if (c0 == 'c' && (size_t)(line_end - p) > 5 &&
                       strncmp(p, "c ind", 5) == 0) {
                /* independent support (UniGen / QuickSampler convention), 0-terminated */
                const char *q = p + 5;
                while (q < line_end) {
                    char *tail = nullptr;
                    long v = strtol(q, &tail, 10);
                    if (tail == q) break;
                    if (v != 0) support.push_back((int32_t)(v < 0 ? -v : v) - 1);
                    q = tail;
                }
            }
            p = line_end;
            continue;
        }
        at_line_start = false;
        if (*p == '-' || (*p >= '0' && *p <= '9')) {
            char *tail = nullptr;
            long v = strtol(p, &tail, 10);
            if (tail == p) { ++p; continue; }
            toks.push_back((int32_t)v);
            p = tail;
        } else {
            ++p;   /* stray character: ignore, as the Python tokenizer does */
        }
    }

    std::vector<int32_t> flat, lens;
    flat.reserve(toks.size());
    int32_t max_var = 0, cur = 0;
    for (int32_t t : toks) {
        if (t == 0) {
            lens.push_back(cur);
            cur = 0;
        } else {
            flat.push_back(t);
            ++cur;
            const int32_t a = t < 0 ? -t : t;
            if (a > max_var) max_var = a;
        }
    }
    if (cur) lens.push_back(cur);      /* unterminated final clause */

    const int32_t nv = header_n > max_var ? header_n : max_var;   /* 0 => empty formula, legal */
    gpap_status st = build(nv, flat.data(), lens.data(), (int32_t)lens.size(), out);
    if (st == GPAP_OK && !support.empty()) {
        std::sort(support.begin(), support.end());
        support.erase(std::unique(support.begin(), support.end()), support.end());
        while (!support.empty() && support.back() >= nv) support.pop_back();
        (*out)->support = std::move(support);
    }
    return st;
}

/* ------------------------------------------------------------------ accessors */

extern "C" void gpap_formula_info_get(const gpap_formula *f, gpap_formula_info *out) {
    if (!f || !out) return;
    out->num_vars = f->num_vars;
    out->num_clauses = f->num_clauses;
    out->max_clause_len = f->max_clause_len;
    out->shift = f->shift;
    out->has_empty_clause = f->has_empty_clause;
    out->total_lits = f->total_lits;
}

extern "C" void gpap_formula_free(gpap_formula *f) { delete f; }

extern "C" gpap_status gpap_formula_occ(const gpap_formula *f, int32_t *offsets_out,
                                        int32_t *clauses_out, int32_t *n_offsets,
                                        int64_t *n_clauses) {
    if (!f) return GPAP_ERR_ARG;
    if (n_offsets) *n_offsets = (int32_t)f->occ_offsets.size();
    if (n_clauses) *n_clauses = (int64_t)f->occ_clauses.size();
    if (offsets_out)
        memcpy(offsets_out, f->occ_offsets.data(), f->occ_offsets.size() * sizeof(int32_t));
    if (clauses_out)
        memcpy(clauses_out, f->occ_clauses.data(), f->occ_clauses.size() * sizeof(int32_t));
    return GPAP_OK;
}

extern "C" gpap_status gpap_formula_clause_arrays(const gpap_formula *f, int32_t *sizes_out,
                                                  int32_t *lit_sum_out) {
    if (!f) return GPAP_ERR_ARG;
    if (sizes_out)
        memcpy(sizes_out, f->clause_sizes.data(), f->clause_sizes.size() * sizeof(int32_t));
    if (lit_sum_out)
        memcpy(lit_sum_out, f->lit_sum.data(), f->lit_sum.size() * sizeof(int32_t));
    return GPAP_OK;
}

extern "C" gpap_status gpap_formula_support(const gpap_formula *f, int32_t *out, int32_t *count) {
    if (!f) return GPAP_ERR_ARG;
    if (count) *count = (int32_t)f->support.size();
    if (out && !f->support.empty())
        memcpy(out, f->support.data(), f->support.size() * sizeof(int32_t));
    return GPAP_OK;
}
