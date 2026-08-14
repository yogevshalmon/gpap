// Sampling a CNF's solutions from C++ with libgpap.
//
//   ./gpap_sample formula.cnf [n_solutions] [seed] [out.txt]
//
// Everything that makes GPAP fast — the batch size, the propagate kernel, the decision order,
// the chunk loop — is chosen by gpap::Sampler. There is no K to pick and no tuning to do; the
// only argument most callers set is the seed. Build via the top CMakeLists, or against an
// installed libgpap:
//
//   g++ -std=c++17 sample.cpp -lgpap -o gpap_sample
#include <cstdio>
#include <cstdlib>

#include "gpap/gpap.hpp"

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s formula.cnf [n_solutions] [seed] [out.txt]\n", argv[0]);
        return 2;
    }
    const std::int64_t want = argc > 2 ? std::atoll(argv[2]) : 100000;
    const char *out_path = argc > 4 ? argv[4] : nullptr;

    try {
        gpap::SampleOptions opts;
        opts.seed = argc > 3 ? std::atoll(argv[3]) : 5;
        // Keep only what we are going to look at: 10 to print, or everything if writing a file.
        // Solutions are found on the GPU either way; this caps what is copied back to the host.
        opts.max_keep = out_path ? static_cast<std::size_t>(want) : 10;
        // Other things worth knowing about, all optional:
        //   opts.variables   = {1, 2, 3};   // report only these DIMACS variables (projection).
        //                                   // Defaults to the file's `c ind` independent support
        //                                   // if it declares one, else every variable.
        //   opts.time_limit_s = 30.0;       // give up after 30s even if `want` is not met
        //   opts.batch        = 1;          // decide 1 variable per round instead of 8: slower
        //                                   // per chunk, but propagation does more of the work,
        //                                   // which wins on tightly constrained formulas
        //   opts.k            = 65536;      // pin the batch size; 0 (default) sizes it to VRAM

        gpap::Sampler sampler(argv[1], opts);
        // What it chose for you, for the record — none of this had to be supplied.
        std::printf("%s: %d vars, %d clauses — %d rows/chunk, %s propagate, "
                    "%d variables decided first, %zu reported\n",
                    argv[1], sampler.formula().num_vars(), sampler.formula().num_clauses(),
                    sampler.k(), sampler.mode() == gpap::Mode::Wave ? "wave" : "scalar",
                    sampler.num_support(), sampler.variables().size());

        const gpap::Solutions sols = sampler.sample(want);

        if (sols.retried())
            std::printf("note: nothing found with lookahead=%s, retried with lookahead=%s\n",
                        sols.lookahead_used() ? "off" : "on",
                        sols.lookahead_used() ? "on" : "off");
        std::printf("%lld solutions in %.2fs — %.0f/s, %.0f%% of trajectories succeeded%s\n",
                    (long long)sols.found(), sols.seconds(), sols.rate(),
                    100.0 * sols.success_rate(), sols.timed_out() ? " (timed out)" : "");

        // Each kept row is a satisfying assignment, projected onto sols.variables().
        if (out_path) {
            std::FILE *fp = std::fopen(out_path, "w");
            if (!fp) { std::perror(out_path); return 1; }
            for (std::size_t i = 0; i < sols.size(); ++i) {
                for (std::int32_t lit : sols.literals(i)) std::fprintf(fp, "%d ", lit);
                std::fprintf(fp, "\n");
            }
            std::fclose(fp);
            std::printf("wrote %zu solutions to %s\n", sols.size(), out_path);
        } else {
            for (std::size_t i = 0; i < sols.size() && i < 3; ++i) {
                std::printf("  solution %zu:", i);
                const std::vector<std::int32_t> lits = sols.literals(i);
                for (std::size_t j = 0; j < lits.size() && j < 12; ++j) std::printf(" %d", lits[j]);
                std::printf("%s\n", lits.size() > 12 ? " ..." : "");
            }
        }

        const gpap_counters ctr = sampler.counters();
        std::printf("work: %.2e literals assigned/s (%lld decisions, %lld occurrence entries)\n",
                    sols.seconds() > 0 ? ctr.props / sols.seconds() : 0.0,
                    (long long)ctr.decs, (long long)ctr.ops);
        return sols.found() > 0 ? 0 : 1;
    } catch (const gpap::Error &e) {
        std::fprintf(stderr, "gpap error: %s\n", e.what());
        return 1;
    }
}
