/* CNF formula construction: DIMACS parsing, the negated-occurrence CSR, and the packed
 * clause-word initial state. Port of gpap/engine/formula.py — and now the ONLY implementation
 * (the Python class is a facade over this, so the two can never drift).
 *
 * Literal indexing (the 2N scheme): +v -> 2(v-1), -v -> 2(v-1)+1, negation l ^ 1.
 * occ[l] = the clauses where ~l appears, so setting l TRUE falsifies a literal in exactly
 * those clauses. Clauses are SEGREGATED BY LENGTH (see gpap_internal.hpp): binary entries come
 * first in every occ segment with the OTHER LITERAL as payload, long (>=3) entries follow with
 * the re-indexed long clause id; unit clauses have no occ entry at all. Within each kind's
 * sub-range, order stays clause-ASCENDING (a counting-sort fill in clause order).
 */
#include "gpap_internal.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using gpap::set_error;

/* ------------------------------------------------------------------ helpers */

/* DIMACS -> 2N index for a formula of `nv` variables, or -1 if the literal is 0 or out of
 * range. THE conversion is gpap::lit_index_from_dimacs (gpap_internal.hpp) — this is only the
 * int32 narrowing the builder's arrays want, and it is safe because the caller has already
 * rejected every negative result. The builder used to carry its own std::abs() copy that took
 * the magnitude BEFORE the range check, which is undefined behaviour at INT32_MIN. */
static inline int32_t lit_index(int32_t ext, int32_t nv) {
    return (int32_t)gpap::lit_index_from_dimacs(ext, nv);
}

/* The inverse: 2N literal index -> signed DIMACS literal. */
static inline int32_t to_dimacs(int32_t li) {
    const int32_t v = (li >> 1) + 1;
    return (li & 1) ? -v : v;
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
    /* OWNED for the whole build: every vector operation below can throw (guard_entry turns
     * that into GPAP_ERR_MEMORY rather than terminate()), and a raw pointer would leak the
     * half-built formula on the way out. */
    std::unique_ptr<gpap_formula> f(new gpap_formula());
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
     * into silent corruption, so refuse rather than build something subtly wrong. Checked against
     * total_lits, which now OVER-counts the CSR (unit clauses contribute no occ entry) — the
     * conservative direction, and the bound that also protects the flat clause read-back.
     * It is ALSO what keeps GPAP_OCC_BIN_TAG unambiguous: num_long <= num_clauses <= total_lits,
     * so a long payload never reaches the sign bit the binary entries claim. */
    if (f->total_lits > 0x7FFFFFFFll) {
        set_error("formula too large: %lld literal occurrences exceeds the int32 CSR limit",
                  (long long)f->total_lits);
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
    f->occ_offsets.assign(num_lits + 1, 0);
    f->occ_mid.assign((size_t)num_lits, 0);

    /* Pass 2: clause counters + per-literal occurrence histograms of the NEGATED literal,
     * SPLIT BY CLAUSE KIND (see the struct comment in gpap_internal.hpp). Length-1 clauses get
     * NO occ entry: the construction flush asserts them into the baseline bitmap, and any later
     * contradicting push is caught there — observationally identical to the old per-word path.
     * Length-2 clauses become implication edges (payload = the other literal); only length>=3
     * clauses will carry a packed word. */
    std::vector<int32_t> bin_cnt((size_t)num_lits, 0);
    for (int32_t c = 0; c < M; ++c) {
        const int32_t n = keep_len[c];
        const int32_t *lits = keep_ptr[c];
        uint32_t sum = 0;
        for (int32_t i = 0; i < n; ++i) {
            const int32_t li = lit_index(lits[i], num_vars);
            if (li < 0) {                          /* 0, or a variable past num_vars */
                set_error("literal %d out of range for num_vars=%d", lits[i], num_vars);
                return GPAP_ERR_ARG;
            }
            sum += (uint32_t)li;
            if (n == 2)
                bin_cnt[(size_t)(li ^ 1)] += 1;
            else if (gpap::is_long_clause(n))
                f->occ_offsets[(li ^ 1) + 1] += 1;
        }
        f->clause_sizes[c] = n;
        f->lit_sum[c] = (int32_t)sum;             /* exact bit pattern mod 2^32 */
    }
    /* occ_offsets[l+1] currently holds l's LONG count; fold the binary count in, prefix-sum,
     * and mark each segment's binary/long split point. */
    for (int64_t l = 0; l < num_lits; ++l) f->occ_offsets[l + 1] += bin_cnt[(size_t)l];
    for (int64_t l = 0; l < num_lits; ++l) f->occ_offsets[l + 1] += f->occ_offsets[l];
    for (int64_t l = 0; l < num_lits; ++l)
        f->occ_mid[(size_t)l] = f->occ_offsets[l] + bin_cnt[(size_t)l];
    f->occ_clauses.resize((size_t)f->occ_offsets[num_lits]);

    /* Pass 3: counting-sort fill, clause-ascending within each kind's sub-range, plus the
     * per-kind side lists (all aligned with the original clause order by construction). */
    std::vector<int32_t> bin_cur(f->occ_offsets.begin(), f->occ_offsets.end() - 1);
    std::vector<int32_t> long_cur(f->occ_mid.begin(), f->occ_mid.end());
    int32_t long_id = 0;
    for (int32_t c = 0; c < M; ++c) {
        const int32_t n = keep_len[c];
        const int32_t *lits = keep_ptr[c];
        if (n == 1) {
            f->unit_lits.push_back(lit_index(lits[0], num_vars));
        } else if (n == 2) {
            /* Binary payloads carry GPAP_OCC_BIN_TAG (the sign bit): a literal index needs at
             * most 24 bits (the 8|24 envelope's 2N ceiling), and a long payload is a clause
             * index in [0, num_long) with num_long <= total_lits <= INT32_MAX (refused above),
             * so the sign bit is free and unambiguous for both kinds. It lets the WAVE
             * kernel classify an entry from the value it has already loaded instead of gathering
             * this segment's occ_mid through an extra shuffle gather in its hottest loop.
             * occ_mid still bounds the scalar loop split, the CLEAN restore and the SOFT scan. */
            const int32_t a = lit_index(lits[0], num_vars), b = lit_index(lits[1], num_vars);
            f->occ_clauses[(size_t)bin_cur[(size_t)(a ^ 1)]++] = b | GPAP_OCC_BIN_TAG;
            f->occ_clauses[(size_t)bin_cur[(size_t)(b ^ 1)]++] = a | GPAP_OCC_BIN_TAG;
            f->bin_pairs.push_back(a < b ? a : b);
            f->bin_pairs.push_back(a < b ? b : a);
        } else {
            for (int32_t i = 0; i < n; ++i)
                f->occ_clauses[(size_t)long_cur[(size_t)(lit_index(lits[i], num_vars) ^ 1)]++] =
                    long_id;
            ++long_id;
        }
    }
    f->num_long = long_id;
    f->num_bin = (int32_t)(f->bin_pairs.size() / 2);

    /* THE kind/slot map (see gpap_internal.hpp). One cursor walk, here, once — the readout and
     * the flat clause read-back index it instead of each re-deriving the same ordering rule. */
    f->sat_src.resize((size_t)M);
    for (int32_t c = 0, li = 0, bi = 0, ui = 0; c < M; ++c) {
        const int32_t n = f->clause_sizes[(size_t)c];
        if (gpap::is_long_clause(n)) f->sat_src[(size_t)c] = li++;
        else if (n == 2) f->sat_src[(size_t)c] = -(bi++) - 1;
        else f->sat_src[(size_t)c] = -(f->num_bin + ui++) - 1;
    }

    /* Packed initial word — LONG clauses only (binary/unit clauses have no per-row state). */
    gpap::pack_long_init(f.get(), f->shift, f->packed_init);
    *out = f.release();
    return GPAP_OK;
}

extern "C" gpap_status gpap_formula_from_clauses(int32_t num_vars, const int32_t *flat_lits,
                                                 const int32_t *clause_lens, int32_t num_clauses,
                                                 gpap_formula **out) {
    if (!out) return GPAP_ERR_ARG;
    /* guard_entry: the clause counts and num_vars are the CALLER's, and they size every
       container below. See gpap_internal.hpp for why the boundary is drawn here. */
    return gpap::guard_entry("gpap_formula_from_clauses", [&] {
        return build(num_vars, flat_lits, clause_lens, num_clauses, out);
    });
}

/* ------------------------------------------------------------------ DIMACS */

static gpap_status from_dimacs(const char *path, gpap_formula **out) {
    /* Cleared UP FRONT, not just inside build(): the tokenizer below has a dozen early
       GPAP_ERR_IO returns, and a caller whose cleanup path frees `*out` would otherwise free
       whatever the variable held before the call. */
    *out = nullptr;
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        set_error("cannot open '%s'", path);
        return GPAP_ERR_IO;
    }
    /* READ TO EOF; do not size the buffer from ftell(). Two real inputs broke the sized read,
     * one in each direction:
     *   - a DIRECTORY. fopen("rb") SUCCEEDS on a directory on Linux and ftell then reports
     *     LONG_MAX, so `std::string buf(size)` threw std::length_error — out through this
     *     extern "C" boundary, where there is no handler, so the whole process called
     *     std::terminate(). `gpap sample <a directory>` core-dumped.
     *   - any file ftell CANNOT size: a FIFO, /dev/stdin, a <(...) process substitution
     *     (ftell = -1), or procfs/sysfs (ftell = 0 while the file has content). The sized read
     *     was then skipped entirely and the caller got the EMPTY formula back with GPAP_OK —
     *     zero variables, trivially satisfiable, no warning. That is the same failure the
     *     tokenizer below is strict about: a file silently becoming a DIFFERENT formula.
     * Reading until EOF answers both — a directory fails inside fread (EISDIR, caught by
     * ferror) and a stream parses the bytes it actually carries. ftell survives only as a
     * reserve HINT, where being wrong costs one reallocation and nothing else. */
    std::string buf;
    if (fseek(fp, 0, SEEK_END) == 0) {
        const long hint = ftell(fp);
        /* 1LL, not 1L: `1L << 40` is undefined where long is 32-bit. The cap only
           keeps a nonsense ftell (a directory's LONG_MAX) out of reserve(). */
        if (hint > 0 && (long long)hint < (1LL << 40)) buf.reserve((size_t)hint);
        rewind(fp);
    }
    {
        std::vector<char> chunk(1 << 16);   /* heap: 64 KB overflows a small thread stack */
        for (;;) {
            /* fread returns short ONLY at EOF or on error (it loops internally), so this is a
             * complete read of a pipe, not a first-packet read of one. */
            const size_t n = fread(chunk.data(), 1, chunk.size(), fp);
            if (n) buf.append(chunk.data(), n);
            if (n < chunk.size()) break;
        }
    }
    const bool read_failed = ferror(fp) != 0;
    fclose(fp);
    if (read_failed) {
        set_error("cannot read '%s' — is it a directory?", path);
        return GPAP_ERR_IO;
    }

    /* Token stream: clauses are 0-TERMINATED, not line-based (multiple clauses per line and
     * clauses spanning lines are both legal). 'c' comments and the SATLIB '%' end marker are
     * handled line-wise; N = max(header N, highest variable used) so trailing declared
     * variables (support vars that appear only in `c ind` lines) keep their slots.
     *
     * STRICT on corruption, lenient on structure. Anything that is not a literal, a comment,
     * the header or the '%' trailer is a PARSE ERROR naming the line — a stray token used to be
     * silently dropped, which turned a corrupt file into a DIFFERENT formula with no warning.
     * Deliberately still tolerated (all common in real benchmark files): a missing header
     * (N from the highest variable), a header whose counts disagree with the body, a final
     * clause unterminated at EOF, and empty/tautological/duplicate-literal clauses (the engine
     * is sound on all of them). */
    std::vector<int32_t> toks;
    std::vector<int32_t> support;
    int32_t header_n = 0;
    bool seen_header = false;
    long line = 1;
    const char *p = buf.c_str(), *end = p + buf.size();
    bool at_line_start = true;
    while (p < end) {
        if (*p == ' ' || *p == '\t' || *p == '\r') { ++p; continue; }
        if (*p == '\n') { ++p; ++line; at_line_start = true; continue; }
        if (at_line_start && (*p == 'c' || *p == 'p' || *p == '%')) {
            const char c0 = *p;
            const char *line_end = (const char *)memchr(p, '\n', (size_t)(end - p));
            if (!line_end) line_end = end;
            if (c0 == '%') break;
            if (c0 == 'p') {
                /* sscanf is NOT line-bounded — on "p cnf\n1 2 0" it would walk through the
                 * newline and read the counts off the next clause. Scan a copy of the line. */
                const std::string hline(p, line_end);
                int n = 0, m = 0;
                if (sscanf(hline.c_str(), "p cnf %d %d", &n, &m) != 2 || n < 0 || m < 0) {
                    set_error("'%s' line %ld: malformed header (expected 'p cnf <vars> "
                              "<clauses>')", path, line);
                    return GPAP_ERR_IO;
                }
                /* The QuickSampler/HTS benchmark distributions repeat the IDENTICAL header
                 * line — harmless redundancy, accepted. Two headers that DISAGREE are two
                 * different formulas glued together; that is corruption. */
                if (seen_header && n != header_n) {
                    set_error("'%s' line %ld: conflicting 'p cnf' header (%d vars after a "
                              "header declaring %d)", path, line, n, header_n);
                    return GPAP_ERR_IO;
                }
                seen_header = true;
                header_n = n;
            } else if (c0 == 'c') {
                /* Independent-support headers: `c ind <v...> 0` (the UniGen / QuickSampler
                 * convention) and the `cr <v...>` variant some samplers emit — same meaning,
                 * both accepted. Any other c-line is a comment. */
                const char *q = nullptr;
                if ((size_t)(line_end - p) > 5 && strncmp(p, "c ind", 5) == 0)
                    q = p + 5;
                else if ((size_t)(line_end - p) > 2 && p[1] == 'r' &&
                         (p[2] == ' ' || p[2] == '\t'))
                    q = p + 2;
                while (q && q < line_end) {
                    char *tail = nullptr;
                    errno = 0;
                    const long v = strtol(q, &tail, 10);
                    /* `tail > line_end`: strtol skips leading whitespace INCLUDING \r and \n,
                     * so a cursor left short of the newline — one \r on a CRLF file, or a
                     * trailing space — parsed across it and pulled the NEXT line's first literal
                     * into the support, silently widening the projection set. */
                    if (tail == q || tail > line_end) break;
                    q = tail;
                    /* The MAGNITUDE in int64, range-checked BEFORE it is negated or narrowed —
                     * the same discipline the literal path above uses, and for the same reason:
                     * `-v` at LONG_MIN is undefined, and an int32 cast of an oversized value
                     * WRAPS into a plausible-looking variable. Both were reachable from a file
                     * and both were silent: `c ind 4294967297` became variable 1, and
                     * `c ind 9223372036854775808` became support index -2 — a NEGATIVE "0-based
                     * variable index", which this ABI says it never returns. Dropped rather than
                     * refused, matching the established policy for a support variable the header
                     * never declared (the `>= nv` trim below); `c ind` is a comment line, and
                     * tools do write junk there. */
                    /* RANGE-CHECKED BEFORE NEGATING. `long` and `int64_t` are the same width
                     * here, so there is no wider type to take the magnitude in the way
                     * lit_index_from_dimacs does — and LONG_MIN is exactly representable, so
                     * strtol does NOT set ERANGE for it while `-v` is still undefined. Testing
                     * the range first is the same guard the literal path above uses. */
                    if (errno == ERANGE || v == 0 || v > INT32_MAX || v < -(long)INT32_MAX)
                        continue;
                    support.push_back((int32_t)(v > 0 ? v : -v) - 1);
                }
            }
            p = line_end;
            continue;
        }
        at_line_start = false;
        if (*p == '-' || (*p >= '0' && *p <= '9')) {
            const bool neg = (*p == '-');
            char *tail = nullptr;
            errno = 0;
            long v = strtol(p, &tail, 10);
            if (tail == p) {                       /* '-' followed by a non-digit */
                set_error("'%s' line %ld: stray '-' is not a literal", path, line);
                return GPAP_ERR_IO;
            }
            if (errno == ERANGE || v > INT32_MAX || v < -(long)INT32_MAX) {
                set_error("'%s' line %ld: literal magnitude exceeds %d", path, line, INT32_MAX);
                return GPAP_ERR_IO;
            }
            if (v == 0 && neg) {                   /* '-0' is a typo, not a terminator */
                set_error("'%s' line %ld: '-0' is not a literal (0 alone terminates a clause)",
                          path, line);
                return GPAP_ERR_IO;
            }
            if (tail < end && *tail != ' ' && *tail != '\t' && *tail != '\r' && *tail != '\n') {
                set_error("'%s' line %ld: malformed token — unexpected '%c' after a literal",
                          path, line, *tail);
                return GPAP_ERR_IO;
            }
            toks.push_back((int32_t)v);
            p = tail;
        } else {
            set_error("'%s' line %ld: unexpected character '%c' — not DIMACS (comments start "
                      "with 'c' at the beginning of a line)", path, line, *p);
            return GPAP_ERR_IO;
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

/* The FILE's contents size every container in the tokenizer and the builder, so this is the
 * other side of the guard_entry boundary (gpap_internal.hpp). */
extern "C" gpap_status gpap_formula_from_dimacs(const char *path, gpap_formula **out) {
    if (!path || !out) return GPAP_ERR_ARG;
    return gpap::guard_entry("gpap_formula_from_dimacs", [&] { return from_dimacs(path, out); });
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
    out->num_bin = f->num_bin;
    out->num_long = f->num_long;
}

extern "C" void gpap_formula_free(gpap_formula *f) { delete f; }

extern "C" gpap_status gpap_formula_occ(const gpap_formula *f, int32_t *offsets_out,
                                        int32_t *mid_out, int32_t *clauses_out,
                                        int32_t *n_offsets, int64_t *n_clauses) {
    if (!f) return GPAP_ERR_ARG;
    /* mid is (2N,) = n_offsets - 1, so it needs no size of its own. */
    if (n_offsets) *n_offsets = (int32_t)f->occ_offsets.size();
    if (n_clauses) *n_clauses = (int64_t)f->occ_clauses.size();
    /* EVERY copy is guarded on non-empty, not just some of them. `std::vector::data()` on an
     * empty vector may be nullptr, and `memcpy(dst, nullptr, 0)` is UNDEFINED even at length 0 —
     * UBSan reports "null pointer passed as argument 2, which is declared to never be null".
     * Both are reachable: `occ_clauses` is empty for a units-only formula and `occ_mid` for a
     * 0-variable one (which unit propagation / ELS really can produce). One carried the guard and
     * the other did not, which is the shape of bug that stays hidden until a test happens to
     * build the degenerate formula. `occ_offsets` needs NO guard: build() gives it num_lits + 1
     * entries, so it holds at least one even at N = 0. */
    if (offsets_out)
        memcpy(offsets_out, f->occ_offsets.data(), f->occ_offsets.size() * sizeof(int32_t));
    if (mid_out && !f->occ_mid.empty())
        memcpy(mid_out, f->occ_mid.data(), f->occ_mid.size() * sizeof(int32_t));
    if (clauses_out && !f->occ_clauses.empty())
        memcpy(clauses_out, f->occ_clauses.data(), f->occ_clauses.size() * sizeof(int32_t));
    return GPAP_OK;
}

extern "C" gpap_status gpap_formula_clause_arrays(const gpap_formula *f, int32_t *sizes_out,
                                                  int32_t *lit_sum_out) {
    if (!f) return GPAP_ERR_ARG;
    /* Guarded on non-empty for the same reason as gpap_formula_occ above: both vectors are
     * empty when the formula has no clauses, and a null memcpy source is UB at any length. */
    if (sizes_out && !f->clause_sizes.empty())
        memcpy(sizes_out, f->clause_sizes.data(), f->clause_sizes.size() * sizeof(int32_t));
    if (lit_sum_out && !f->lit_sum.empty())
        memcpy(lit_sum_out, f->lit_sum.data(), f->lit_sum.size() * sizeof(int32_t));
    return GPAP_OK;
}

extern "C" gpap_status gpap_formula_clauses(const gpap_formula *f, int32_t *lits_out,
                                            int64_t *n_lits) {
    if (!f) return GPAP_ERR_ARG;
    if (n_lits) *n_lits = f->total_lits;
    if (!lits_out) return GPAP_OK;
    /* Reconstructed in the ORIGINAL clause order from the per-kind lists, which are aligned
     * with clause_sizes by construction. LONG clauses come back by INVERTING the long occ
     * sub-ranges: every entry (l, j) says long clause j contains the literal l^1, so duplicate
     * literals are preserved and within-clause order is ascending by literal index — exactly
     * the pre-segregation behaviour. Binary pairs are stored sorted and units are single
     * literals, so both properties hold for them trivially. */
    std::vector<int64_t> long_pos(static_cast<size_t>(f->num_long), 0);  /* long id -> write pos */
    int64_t pos = 0;
    for (int32_t c = 0; c < f->num_clauses; ++c) {
        const int32_t n = f->clause_sizes[static_cast<size_t>(c)];
        const int32_t src = f->sat_src[static_cast<size_t>(c)];
        if (src >= 0) {                            /* long: filled by the occ inversion below */
            long_pos[static_cast<size_t>(src)] = pos;
        } else {
            const int32_t p = -src - 1;            /* merged pair index */
            if (p < f->num_bin)
                for (int i = 0; i < 2; ++i)
                    lits_out[pos + i] = to_dimacs(f->bin_pairs[static_cast<size_t>(2 * p + i)]);
            else
                lits_out[pos] = to_dimacs(
                    f->unit_lits[static_cast<size_t>(p - f->num_bin)]);
        }
        pos += n;
    }
    const int64_t num_lits = 2ll * f->num_vars;
    for (int64_t l = 0; l < num_lits; ++l) {
        const int64_t in_clause = l ^ 1;                       /* the literal the clause holds */
        const int32_t dimacs = to_dimacs(static_cast<int32_t>(in_clause));
        for (int32_t k = f->occ_mid[static_cast<size_t>(l)];
             k < f->occ_offsets[static_cast<size_t>(l) + 1]; ++k)
            lits_out[long_pos[static_cast<size_t>(f->occ_clauses[static_cast<size_t>(k)])]++] =
                dimacs;
    }
    return GPAP_OK;
}

extern "C" gpap_status gpap_formula_support(const gpap_formula *f, int32_t *out, int32_t *count) {
    if (!f) return GPAP_ERR_ARG;
    if (count) *count = (int32_t)f->support.size();
    if (out && !f->support.empty())
        memcpy(out, f->support.data(), f->support.size() * sizeof(int32_t));
    return GPAP_OK;
}
