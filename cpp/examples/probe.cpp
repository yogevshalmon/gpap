/* gpap::Prober — BCP-probe every k-subset of a literal set.
 *
 *   build/gpap_probe formula.cnf [width]
 *
 * Assuming a set of literals and propagating to a conflict PROVES the formula entails the
 * negation of that set — a clause. Sweeping every k-subset of a pool therefore finds every
 * entailed clause of width k that propagation alone can see: at k=1 the backbone, at k=2 the
 * binary implications, and so on. This is the primitive; deciding which pools and widths are
 * worth sweeping is an application's job (see gpap.mining on the Python side).
 */
#include <gpap/gpap.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s formula.cnf [width]\n", argv[0]);
        return 2;
    }
    const std::int32_t width = argc > 2 ? std::atoi(argv[2]) : 1;

    try {
        gpap::Prober prober(argv[1]);
        if (prober.unsat()) {
            std::printf("formula is UNSAT on its own units — nothing to probe\n");
            return 0;
        }
        const auto free = prober.free_literals();
        std::printf("%d variables, %d clauses, %zu free literals\n",
                    prober.formula().num_vars(), prober.formula().num_clauses(), free.size());

        // Ask what the sweep COSTS before running it: at width 3 over a large formula the
        // enumeration is astronomically large, and this is how you find that out cheaply.
        const std::int64_t candidates = prober.count(width);
        std::printf("width %d: %lld candidate tuples\n", width,
                    static_cast<long long>(candidates));

        gpap::ProbeBudget budget;
        budget.seconds = 10.0;
        budget.max_tuples = 5'000'000;

        const gpap::Probe r = prober.probe(width, budget);
        std::printf("probed %lld of %lld in %.2fs -> %lld entailed clause(s)%s\n",
                    static_cast<long long>(r.probed()), static_cast<long long>(r.total()),
                    r.seconds(), static_cast<long long>(r.size()),
                    r.exhausted() ? "  [exhaustive: a negative result would be a PROOF]" : "");
        // max_tuples is a WORK BUDGET, not a refusal: at the default (ProbeBudget::
        // refuse_over_max == false) the sweep probes that many candidates and KEEPS what it
        // found — every conflict is a genuine entailed clause whatever went unprobed. Only
        // `exhausted` is lost, and only that licenses "no clause of this width exists".
        if (!r.exhausted() && r.stop() == GPAP_PROBE_CEILING)
            std::printf("  (stopped at max_tuples: the clauses above are entailed, but the "
                        "sweep is not a proof — raise the ceiling or narrow the pool)\n");
        if (!r.exhausted() && r.stop() == GPAP_PROBE_DEADLINE)
            std::printf("  (stopped at the %.0fs deadline — same caveat)\n", budget.seconds);

        for (std::int64_t i = 0; i < r.size() && i < 10; ++i) {
            std::printf("  entailed: (");
            const auto c = r.clause(i);
            for (std::size_t j = 0; j < c.size(); ++j)
                std::printf("%s%d", j ? " | " : "", c[j]);
            std::printf(")\n");
        }
        if (r.size() > 10) std::printf("  ... and %lld more\n", static_cast<long long>(r.size() - 10));
    } catch (const gpap::Error &e) {
        std::fprintf(stderr, "gpap error: %s\n", e.what());
        return 1;
    }
    return 0;
}
