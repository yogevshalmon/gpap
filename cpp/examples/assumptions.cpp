// Checking many assumption sets at once — the engine's core primitive, from C++.
//
//   ./gpap_assumptions [formula.cnf]
//
// Every set is propagated to a BCP fixpoint independently of the others, all in ONE kernel
// launch. That is the operation GPAP parallelizes: not one big propagation, but K of them.
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "gpap/gpap.hpp"

int main(int argc, char **argv) {
    try {
        if (argc < 2) {
            // (x1 v x2) ^ (~x1 v x3) ^ (~x2 v ~x3)
            gpap::AssumptionChecker checker(gpap::Formula::from_clauses(3, {{1, 2}, {-1, 3}, {-2, -3}}));
            const auto r = checker.check({{1}, {-1, -2}, {2, 3}});
            const char *label[] = {"x1=T", "x1=F,x2=F", "x2=T,x3=T"};
            for (std::size_t i = 0; i < r.size(); ++i) {
                std::printf("%-12s -> %s", label[i], r.conflict(i) ? "CONFLICT" : "ok");
                if (!r.conflict(i)) {
                    std::printf("  (%d implied)  ", r.num_propagated(i));
                    for (int lit : r.literals(i)) std::printf("%d ", lit);
                }
                std::printf("\n");
            }
            return 0;
        }

        // On a real formula: probe 4096 random 8-literal assumption sets in one launch and
        // report how many survive propagation.
        gpap::AssumptionChecker checker(argv[1]);
        const std::int32_t n = checker.formula().num_vars();
        std::mt19937_64 rng(12345);
        std::vector<std::vector<std::int32_t>> sets(4096, std::vector<std::int32_t>(8));
        for (auto &s : sets)
            for (auto &lit : s) {
                const std::int32_t v = 1 + std::int32_t(rng() % std::uint64_t(n));
                lit = (rng() & 1) ? v : -v;
            }

        const auto t0 = std::chrono::steady_clock::now();
        const auto r = checker.check(sets, /*with_assignments=*/false);   // flags only: cheap
        const double secs = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t0).count();

        std::printf("%s: %zu of %zu assumption sets survive BCP  (%.1f ms, %.0f sets/s)\n",
                    argv[1], r.size() - r.num_conflicts(), r.size(), secs * 1e3,
                    r.size() / secs);
        return 0;
    } catch (const gpap::Error &e) {
        std::fprintf(stderr, "gpap error: %s\n", e.what());
        return 1;
    }
}
