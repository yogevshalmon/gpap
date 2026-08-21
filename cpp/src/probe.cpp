/* Combinatorial probing: BCP every k-subset of a literal set, within a budget.
 *
 * Entry point 1 answers "do THESE assumption sets conflict". This answers the combinatorial form:
 * "which k-subsets of this literal set conflict" — the primitive behind failed-literal detection,
 * entailed-clause mining, backbone extraction and lookahead scoring. It is deliberately the SAME
 * machinery underneath (reset + apply + read the conflict flags); what belongs here, rather than
 * in every caller, is the enumeration, the chunking and the budget.
 *
 * TWO THINGS THIS FILE EXISTS TO GET RIGHT, both of which the Python original got wrong:
 *
 * 1. THE CEILING MUST PRECEDE THE ENUMERATION. `max_tuples` is what keeps a wide pool at k=3 from
 *    being hours of GPU work — but a ceiling checked AFTER materializing the candidate list cannot
 *    protect anything. The Python miner built the whole (C(P,k), k) array and then compared its
 *    length: at pool = "all free literals" of an unrolled circuit that is C(2N,3) ~ 1e11 tuples,
 *    2 TB of host memory, and the process was OOM-killed before the ceiling was ever consulted.
 *    Here the candidate count is computed EXACTLY and compared first (see count_candidates), and
 *    generation is PAGED — one chunk is materialized at a time, so peak host memory is the chunk,
 *    not the enumeration.
 *
 * 2. `exhausted` IS A PROOF, SO `total` MUST BE EXACT. A sweep that probed everything licenses
 *    "no entailed clause of this width exists (that BCP can see)"; a sweep that quietly stopped
 *    short licenses nothing. That distinction is only as good as the candidate count, which is
 *    why distinct_vars is counted with an elementary-symmetric DP rather than estimated.
 *
 * Generation stays on the HOST for now, and the API is shaped so that can change without a caller
 * noticing: candidates are produced one chunk at a time from (pool, k, first), which is exactly
 * the interface a device-side combinatorial-unranking kernel would implement. It is not built
 * because it is not the bottleneck — once the enumeration is vectorized, it costs a small fraction
 * of the probe it feeds, and paging already bounds the memory. Measure before adding a kernel.
 */
#include "gpap_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using gpap::set_error;

/* One sweep's conflicting tuples. Opaque to callers; freed with gpap_probe_free. */
struct gpap_probe {
    std::vector<int32_t> conflicts;   /* (n, k) row-major DIMACS literals */
    int32_t k = 0;
    gpap_probe_stats stats{};
};

namespace {

constexpr int64_t kInfinite = std::numeric_limits<int64_t>::max();

/* Saturating helpers: candidate counts overflow int64 for any pool worth refusing, and a wrapped
 * count would compare BELOW the ceiling and let the sweep through — the exact failure the ceiling
 * exists to prevent. Saturation is sound here because the value is only ever compared. */
int64_t sat_add(int64_t a, int64_t b) {
    if (a >= kInfinite - b) return kInfinite;
    return a + b;
}
int64_t sat_mul(int64_t a, int64_t b) {
    if (a == 0 || b == 0) return 0;
    if (a > kInfinite / b) return kInfinite;
    return a * b;
}

inline int32_t to_dimacs(int64_t idx) {
    const int32_t v = static_cast<int32_t>(idx / 2) + 1;
    return (idx % 2) ? -v : v;
}

/* How many k-subsets of `pool` there are, EXACTLY, without generating any of them.
 *
 * Without distinct_vars that is C(P, k). With it, tuples holding two literals of one variable are
 * excluded, and the count becomes the elementary symmetric polynomial e_k(m_1..m_v) where m_i is
 * how many literals of variable i the pool holds (1 or 2): choosing k distinct variables and one
 * literal from each. The usual pool holds both polarities of every variable, for which this is
 * C(v,k)*2^k — but a caller may pass a one-sided pool, so it is computed rather than assumed.
 * O(v*k), and saturating. */
int64_t count_candidates(const std::vector<int64_t> &pool, int32_t k, bool distinct_vars,
                         int32_t num_vars) {
    const int64_t P = static_cast<int64_t>(pool.size());
    if (k <= 0 || P < k) return 0;
    if (!distinct_vars || k == 1) {
        if (k == 1) return P;
        int64_t c = 1;                                   /* C(P, k), saturating */
        for (int64_t i = 0; i < k; ++i) {
            c = sat_mul(c, P - i);
            if (c == kInfinite) return kInfinite;
            c /= (i + 1);
        }
        return c;
    }
    /* multiplicity per variable, then e_k over the multiplicities */
    std::vector<int32_t> mult(static_cast<size_t>(num_vars), 0);
    for (int64_t l : pool) mult[static_cast<size_t>(l / 2)] += 1;
    std::vector<int64_t> e(static_cast<size_t>(k) + 1, 0);
    e[0] = 1;
    for (int32_t m : mult) {
        if (m == 0) continue;
        for (int32_t j = k; j >= 1; --j)
            e[static_cast<size_t>(j)] = sat_add(e[static_cast<size_t>(j)],
                                                sat_mul(e[static_cast<size_t>(j) - 1], m));
    }
    return e[static_cast<size_t>(k)];
}

/* Lexicographic k-combinations of POOL POSITIONS, filtered and paged.
 *
 * The order matters beyond taste: it is the order the numpy reference enumerates (ascending
 * positions, and distinct_vars applied as a filter that preserves order), so a port can be gated
 * on producing the identical clause SET. `next_chunk` appends up to `want` tuples as internal
 * literal indices and returns how many it wrote; 0 means the enumeration is finished. */
class Combinations {
  public:
    Combinations(const std::vector<int64_t> &pool, int32_t k, bool distinct_vars)
        : pool_(pool), k_(k), distinct_(distinct_vars), idx_(static_cast<size_t>(k), 0) {
        for (int32_t j = 0; j < k_; ++j) idx_[static_cast<size_t>(j)] = j;
        done_ = static_cast<int64_t>(pool_.size()) < k_;
    }

    int64_t next_chunk(std::vector<int64_t> &out, int64_t want) {
        out.clear();
        int64_t n = 0;
        while (n < want && !done_) {
            if (!distinct_ || distinct_vars_ok()) {
                for (int32_t j = 0; j < k_; ++j) out.push_back(pool_[idx_[static_cast<size_t>(j)]]);
                ++n;
            }
            advance();
        }
        return n;
    }

  private:
    bool distinct_vars_ok() const {
        for (int32_t a = 0; a < k_; ++a)
            for (int32_t b = a + 1; b < k_; ++b)
                if (pool_[idx_[static_cast<size_t>(a)]] / 2 == pool_[idx_[static_cast<size_t>(b)]] / 2)
                    return false;
        return true;
    }
    void advance() {
        const int64_t P = static_cast<int64_t>(pool_.size());
        int32_t j = k_ - 1;
        while (j >= 0 && idx_[static_cast<size_t>(j)] == P - k_ + j) --j;
        if (j < 0) { done_ = true; return; }
        ++idx_[static_cast<size_t>(j)];
        for (int32_t t = j + 1; t < k_; ++t)
            idx_[static_cast<size_t>(t)] = idx_[static_cast<size_t>(t) - 1] + 1;
    }

    const std::vector<int64_t> &pool_;
    int32_t k_;
    bool distinct_;
    std::vector<int64_t> idx_;
    bool done_ = false;
};

const gpap_probe_options kDefaults = {0.0, 0, 0, 1, 0};   /* refuse_over_max off */

/* Tuples one launch can hold: the engine's own per-row state plus the (batch, k) int64 decision
 * matrix this entry point allocates and the sampler's path never does. ONE model, the shared one —
 * a second copy of the memory arithmetic is exactly what the sizing collapse removed. */
int32_t probe_batch(gpap_engine *e, int32_t k, int64_t n_tuples, int32_t requested) {
    if (requested > 0) return static_cast<int32_t>(std::min<int64_t>(requested, n_tuples));
    const int64_t per_row =
        gpap_row_bytes(e->num_clauses, e->num_vars, e->trail_cap) + 8ll * std::max(k, 1);
    const int64_t budget = gpap_mem_available(e->device_ordinal, 0.80);
    int64_t b = per_row > 0 ? budget / per_row : 1;
    b = std::min<int64_t>(b, gpap_default_k_cap(e->device_ordinal));
    b = std::min<int64_t>(b, n_tuples);
    return static_cast<int32_t>(std::max<int64_t>(1, b));
}

/* Run one chunk of already-materialized tuples and append the conflicting ones to `out`.
 *
 * NOTE there is no redundancy filter here, deliberately. There was one (`skip_subsumed`: drop a
 * conflict the assumptions alone caused). It was SOUND but worth 0.0-0.3% of the result buffer on
 * every instance measured, while costing an int32 per row and a branch in the innermost loop of
 * both propagate kernels — and the host-side filter in gpap.mining has to be exact anyway, since
 * it is the one that decides what a caller receives. Redundancy is a property of the CLAUSE
 * against a FORMULA, so it belongs where the formula is known, not in the sweep. */
gpap_status probe_chunk(gpap_engine *e, const std::vector<int64_t> &tuples, int32_t k,
                        std::vector<int8_t> &flags, gpap_probe *out) {
    const int32_t rows = static_cast<int32_t>(tuples.size() / static_cast<size_t>(k));
    if (rows == 0) return GPAP_OK;
    GPAP_TRY(gpap_engine_reset(e, rows));           /* broadcast-restore the post-units cache */
    GPAP_TRY(gpap_engine_apply(e, tuples.data(), k));
    flags.resize(static_cast<size_t>(rows));
    GPAP_TRY(gpap_flags(e, nullptr, flags.data()));
    for (int32_t r = 0; r < rows; ++r) {
        if (!flags[static_cast<size_t>(r)]) continue;
        for (int32_t j = 0; j < k; ++j)
            out->conflicts.push_back(
                to_dimacs(tuples[static_cast<size_t>(r) * static_cast<size_t>(k) + j]));
        out->stats.conflicts += 1;
    }
    return GPAP_OK;
}

/* The shared sweep driver. `gen` yields chunks; everything else is budget and bookkeeping. */
template <typename NextChunk>
gpap_status run_sweep(gpap_engine *e, int32_t k, int64_t total, const gpap_probe_options &o,
                      NextChunk next, gpap_probe *out) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto elapsed = [&t0] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };
    out->k = k;
    out->stats.total = total;

    if (e->flush_conflict || e->has_empty_clause) {
        /* The formula's own units already conflict: every assumption set conflicts trivially and
         * none of it says anything about the assumptions. Report it rather than return noise. */
        out->stats.stop = GPAP_PROBE_UNSAT;
        out->stats.seconds = elapsed();
        return GPAP_OK;
    }
    int64_t limit = total;
    if (o.max_tuples > 0 && total > o.max_tuples) {
        /* A WORK BUDGET by default: probe the first max_tuples candidates. Every conflict found
         * is a genuine entailed clause whatever else went unprobed, so returning them beats
         * returning nothing — only `exhausted` is lost, and it is reported honestly. Refusing
         * outright is opt-in, for a caller that wants a proof or nothing. */
        if (o.refuse_over_max) {
            out->stats.stop = GPAP_PROBE_CEILING;
            out->stats.seconds = elapsed();
            return GPAP_OK;
        }
        limit = o.max_tuples;
        out->stats.stop = GPAP_PROBE_CEILING;
    }

    const int32_t batch = probe_batch(e, k, std::max<int64_t>(limit, 1), o.batch);
    std::vector<int64_t> chunk;
    std::vector<int8_t> flags;
    while (out->stats.probed < limit) {
        /* Checked BETWEEN chunks: a chunk is the smallest unit of GPU work and abandoning one
         * mid-flight would leave the engine's row state half-written. */
        if (o.seconds > 0.0 && elapsed() > o.seconds) {
            out->stats.stop = GPAP_PROBE_DEADLINE;
            break;
        }
        const int64_t want = std::min<int64_t>(batch, limit - out->stats.probed);
        const int64_t got = next(chunk, want);
        if (got == 0) break;
        GPAP_TRY(probe_chunk(e, chunk, k, flags, out));
        out->stats.probed += got;
    }
    out->stats.exhausted =
        (out->stats.stop == GPAP_PROBE_COMPLETE && out->stats.probed == total) ? 1 : 0;
    out->stats.seconds = elapsed();
    return GPAP_OK;
}

/* Validate + convert a DIMACS pool, or collect every free literal when the caller passed none. */
gpap_status build_pool(gpap_engine *e, const int32_t *pool, int32_t pool_n,
                       std::vector<int64_t> *out) {
    if (pool == nullptr) {
        int32_t count = 0;
        GPAP_TRY(gpap_free_literals(e, nullptr, &count));
        std::vector<int32_t> lits(static_cast<size_t>(std::max(count, 0)));
        if (count > 0) GPAP_TRY(gpap_free_literals(e, lits.data(), &count));
        out->reserve(lits.size());
        for (int32_t l : lits) out->push_back(gpap::lit_index_from_dimacs(l, e->num_vars));
        return GPAP_OK;
    }
    out->reserve(static_cast<size_t>(std::max(pool_n, 0)));
    for (int32_t i = 0; i < pool_n; ++i) {
        const int64_t idx = gpap::lit_index_from_dimacs(pool[i], e->num_vars);
        if (idx < 0) {
            set_error("probe pool: literal %d out of range (formula has %d variables; literals "
                      "are DIMACS, 1..%d or -1..-%d, never 0)",
                      pool[i], e->num_vars, e->num_vars, e->num_vars);
            return GPAP_ERR_ARG;
        }
        out->push_back(idx);
    }
    return GPAP_OK;
}

}  // namespace

extern "C" gpap_status gpap_free_literals(gpap_engine *e, int32_t *out, int32_t *count) {
    if (!e || !count) return GPAP_ERR_ARG;
    const int32_t n = e->num_vars;
    std::vector<int8_t> assign(static_cast<size_t>(2) * static_cast<size_t>(std::max(n, 0)));
    if (n > 0) GPAP_TRY(gpap_engine_flushed_assignment(e, assign.data()));
    int32_t found = 0;
    for (int32_t v = 0; v < n; ++v) {
        /* Free = the flush settled NEITHER polarity. Probing a settled literal tells you nothing:
         * if it is already TRUE the probe is a no-op, and if it is FALSE the probe conflicts for
         * a reason that is already known. */
        if (assign[static_cast<size_t>(2 * v)] || assign[static_cast<size_t>(2 * v + 1)]) continue;
        if (out) {
            out[found] = v + 1;
            out[found + 1] = -(v + 1);
        }
        found += 2;
    }
    *count = found;
    return GPAP_OK;
}

extern "C" gpap_status gpap_assume_units(gpap_engine *e, const int32_t *lits, int32_t n) {
    if (!e || (n > 0 && !lits)) return GPAP_ERR_ARG;
    if (n <= 0) return GPAP_OK;
    std::vector<int64_t> row(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) {
        row[static_cast<size_t>(i)] = gpap::lit_index_from_dimacs(lits[i], e->num_vars);
        if (row[static_cast<size_t>(i)] < 0) {
            set_error("gpap_assume_units: literal %d out of range (formula has %d variables)",
                      lits[i], e->num_vars);
            return GPAP_ERR_ARG;
        }
    }
    /* reset(1) + apply + recache: propagate the units once on top of the current cache and make
     * the result the new baseline, which is exactly the fixpoint a rebuilt formula's flush would
     * reach. Avoids rebuilding the occurrence lists per round, which dominated mining wall. */
    GPAP_TRY(gpap_engine_reset(e, 1));
    GPAP_TRY(gpap_engine_apply(e, row.data(), n));
    return gpap_engine_recache_flush(e);
}

extern "C" gpap_status gpap_probe_combinations(gpap_engine *e, const int32_t *pool, int32_t pool_n,
                                               int32_t k, const gpap_probe_options *opts,
                                               gpap_probe **out) {
    if (!e || !out || k < 1) return GPAP_ERR_ARG;
    if (pool && pool_n < 0) return GPAP_ERR_ARG;
    const gpap_probe_options o = opts ? *opts : kDefaults;
    std::vector<int64_t> lits;
    GPAP_TRY(build_pool(e, pool, pool_n, &lits));

    auto *p = new gpap_probe();
    const int64_t total = count_candidates(lits, k, o.distinct_vars != 0, e->num_vars);
    Combinations gen(lits, k, o.distinct_vars != 0);
    const gpap_status rc = run_sweep(
        e, k, total, o,
        [&gen](std::vector<int64_t> &c, int64_t want) { return gen.next_chunk(c, want); }, p);
    if (rc != GPAP_OK) { delete p; return rc; }
    *out = p;
    return GPAP_OK;
}

extern "C" gpap_status gpap_probe_tuples(gpap_engine *e, const int32_t *tuples, int64_t n,
                                         int32_t k, const gpap_probe_options *opts,
                                         gpap_probe **out) {
    if (!e || !out || k < 1 || n < 0 || (n > 0 && !tuples)) return GPAP_ERR_ARG;
    const gpap_probe_options o = opts ? *opts : kDefaults;
    /* Validate the WHOLE matrix before any GPU work. An out-of-range literal must not reach the
     * kernel: internal indices past the end of a row's assignment bitmap make `atomic_or` write
     * into a LATER ROW's words, so one bad tuple silently corrupts a different tuple's answer and
     * nothing downstream can detect it. One pass over the caller's own array is cheap; being
     * wrong here is undetectable. (Same reasoning as AssumptionChecker's `_flatten`.) */
    for (int64_t i = 0; i < n * k; ++i) {
        if (gpap::lit_index_from_dimacs(tuples[i], e->num_vars) < 0) {
            set_error("probe tuple: literal %d out of range (formula has %d variables; literals "
                      "are DIMACS, 1..%d or -1..-%d, never 0)",
                      tuples[i], e->num_vars, e->num_vars, e->num_vars);
            return GPAP_ERR_ARG;
        }
    }
    auto *p = new gpap_probe();
    int64_t cursor = 0;
    const gpap_status rc = run_sweep(
        e, k, n, o,
        [&](std::vector<int64_t> &c, int64_t want) -> int64_t {
            c.clear();
            const int64_t take = std::min<int64_t>(want, n - cursor);
            for (int64_t i = 0; i < take; ++i)
                for (int32_t j = 0; j < k; ++j)
                    c.push_back(gpap::lit_index_from_dimacs(tuples[(cursor + i) * k + j], e->num_vars));
            cursor += take;
            return take;
        },
        p);
    if (rc != GPAP_OK) { delete p; return rc; }
    *out = p;
    return GPAP_OK;
}

extern "C" int64_t gpap_probe_count(gpap_engine *e, const int32_t *pool, int32_t pool_n,
                                    int32_t k, int32_t distinct_vars) {
    if (!e || k < 1) return 0;
    std::vector<int64_t> lits;
    if (build_pool(e, pool, pool_n, &lits) != GPAP_OK) return 0;
    return count_candidates(lits, k, distinct_vars != 0, e->num_vars);
}

extern "C" void gpap_probe_free(gpap_probe *p) { delete p; }

extern "C" void gpap_probe_stats_get(const gpap_probe *p, gpap_probe_stats *out) {
    if (!p || !out) return;
    *out = p->stats;
}

extern "C" gpap_status gpap_probe_conflicts(const gpap_probe *p, int32_t *out, int64_t capacity) {
    if (!p || (capacity > 0 && !out)) return GPAP_ERR_ARG;
    const int64_t n = std::min<int64_t>(capacity, p->stats.conflicts);
    if (n > 0)
        std::memcpy(out, p->conflicts.data(),
                    static_cast<size_t>(n) * static_cast<size_t>(p->k) * sizeof(int32_t));
    return GPAP_OK;
}
