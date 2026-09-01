// Sampling a CNF's solutions from C++ with libgpap.
//
//   ./gpap_sample formula.cnf [n_solutions] [seed] [out.txt] [--portfolio]
//
// --portfolio races the library's default config portfolio (random B=8 / random B=1 /
// structured B=1) as a throughput bandit — what the Python pipeline runs by default, and what
// samples the search-heavy instances the single default config cannot. Without it the run is a
// single config and DETERMINISTIC: same seed, same solutions — as long as the BATCH is the
// same, which --k N pins (the default sizes it to live free VRAM).
//
// Everything that makes GPAP fast — the batch size, the propagate kernel, the decision order,
// the chunk loop — is chosen by gpap::Sampler. There is no K to pick and no tuning to do; the
// only argument most callers set is the seed. Build via the top CMakeLists, or against an
// installed libgpap:
//
//   g++ -std=c++17 sample.cpp -lgpap -o gpap_sample
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "gpap/gpap.hpp"

int main(int argc, char **argv) {
    // Pull the flags out first so the positional arguments stay simple.
    bool portfolio = false;
    std::int32_t pin_k = 0;                     // 0 = size the batch to free VRAM
    std::vector<const char *> pos;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--portfolio") { portfolio = true; continue; }
        if (a != "--k") { pos.push_back(argv[i]); continue; }
        // A flag that takes a value must SAY SO when it does not get one. Falling through to
        // the positional list instead would make `--k` the out.txt path and write a file
        // called "--k".
        if (i + 1 >= argc || std::atoi(argv[i + 1]) < 1) {
            std::fprintf(stderr, "%s: --k needs a positive batch size\n", argv[0]);
            return 2;
        }
        pin_k = std::atoi(argv[++i]);
    }
    if (pos.empty()) {
        std::fprintf(stderr, "usage: %s formula.cnf [n_solutions] [seed] [out.txt] "
                             "[--portfolio] [--k N]\n", argv[0]);
        return 2;
    }
    const char *cnf = pos[0];
    const std::int64_t want = pos.size() > 1 ? std::atoll(pos[1]) : 100000;
    const char *out_path = pos.size() > 3 ? pos[3] : nullptr;

    try {
        gpap::SampleOptions opts;
        opts.seed = pos.size() > 2 ? std::atoll(pos[2]) : 5;
        if (portfolio) opts.portfolio = gpap::default_portfolio();
        // --k pins the batch. The default sizes it to LIVE free VRAM, so it is not part of the
        // seed: two runs of the same formula and seed on a differently loaded card draw
        // different numbers of trajectories. Pin it whenever runs have to be comparable.
        opts.k = pin_k;
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
        //   opts.verify       = true;       // re-check on the GPU that every returned row is a
        //                                   // real model. OFF by default, like every other face:
        //                                   // it is a self-test, not part of producing samples.

        gpap::Sampler sampler(cnf, opts);
        // What it chose for you, for the record — none of this had to be supplied.
        std::printf("%s: %d vars, %d clauses — %d rows/chunk, %s propagate, "
                    "%d variables decided first, %zu reported\n",
                    cnf, sampler.formula().num_vars(), sampler.formula().num_clauses(),
                    sampler.k(), sampler.mode() == gpap::Mode::Wave ? "wave" : "scalar",
                    sampler.num_support(), sampler.variables().size());

        const gpap::Solutions sols = sampler.sample(want);

        if (sols.n_configs() > 1) {                 // the bandit's evidence, one line per arm
            for (std::size_t i = 0; i < sols.n_configs(); ++i) {
                const gpap_config_stats &c = sols.config_stats(i);
                std::printf("  config %zu: %lld models / %lld chunks%s\n", i,
                            (long long)c.models, (long long)c.chunks,
                            (int)i == sols.best_config() ? "  <- winner (rows reported)" : "");
            }
        }
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
