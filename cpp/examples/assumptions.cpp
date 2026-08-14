// Checking many assumption sets at once — the engine's core primitive, from C++.
//
//   ./gpap_assumptions [formula.cnf]
//
// Every set is propagated to a BCP fixpoint independently of the others. That is the operation
// GPAP parallelizes: not one big propagation, but K of them. Pass as many sets as you like — the
// checker sizes each launch to the GPU and returns results in input order, so the batching is
// invisible.
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "gpap/gpap.hpp"

// A three-variable formula, so every step can be checked by hand:
//   (x1 v x2) ^ (~x1 v x3) ^ (~x2 v ~x3)
static int tutorial() {
    gpap::AssumptionChecker checker(
        gpap::Formula::from_clauses(3, {{1, 2}, {-1, 3}, {-2, -3}}));

    const std::vector<std::vector<std::int32_t>> sets = {{1}, {-1, -2}, {2, 3}, {}};
    const char *label[] = {"x1=T", "x1=F,x2=F", "x2=T,x3=T", "(no assumptions)"};

    const gpap::CheckResults r = checker.check(sets);
    std::printf("%zu assumption sets, %zu conflicted:\n\n", r.size(), r.num_conflicts());
    for (std::size_t i = 0; i < r.size(); ++i) {
        std::printf("  %-16s -> ", label[i]);
        if (r.conflict(i)) {
            std::printf("CONFLICT\n");
            continue;
        }
        // What BCP settled, variable by variable. A variable can come out True or False even
        // when nothing assumed it — that is an implication — or stay Free, meaning this
        // assumption set simply does not determine it.
        std::printf("ok (%d implied):", r.num_propagated(i));
        for (std::int32_t v = 1; v <= 3; ++v) {
            const gpap::Value x = r.value(i, v);
            std::printf("  x%d=%s", v, x == gpap::Value::True    ? "T"
                                       : x == gpap::Value::False ? "F"
                                                                 : "free");
        }
        std::printf("\n");
    }
    std::printf("\nRow 0 shows the point: assuming x1 alone forces x3 (via ~x1 v x3) and then\n"
                "~x2 (via ~x2 v ~x3) — two implications from one assumption, on the GPU.\n");
    return 0;
}

// On a real formula: how many random partial assignments survive propagation? This is the shape
// of most real uses — a large population of candidate assumptions, filtered in bulk.
static int survey(const char *path) {
    gpap::AssumptionChecker checker(path);
    const std::int32_t n_vars = checker.formula().num_vars();
    constexpr std::size_t kSets = 200000;
    constexpr std::size_t kLits = 8;

    std::mt19937_64 rng(12345);
    std::vector<std::vector<std::int32_t>> sets(kSets, std::vector<std::int32_t>(kLits));
    for (auto &s : sets)
        for (auto &lit : s) {
            const std::int32_t v = 1 + std::int32_t(rng() % std::uint64_t(n_vars));
            lit = (rng() & 1) ? v : -v;
        }

    // with_assignments=false: only the conflict flags come back. The per-row assignment is
    // n x 2N bytes and is what makes a large call expensive, so ask for it only if you read it.
    const auto t0 = std::chrono::steady_clock::now();
    const gpap::CheckResults r = checker.check(sets, /*with_assignments=*/false);
    const double secs = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0).count();

    std::printf("%s: %d vars, %d clauses\n", path, n_vars, checker.formula().num_clauses());
    std::printf("%zu random %zu-literal assumption sets, %zu per launch\n",
                sets.size(), kLits, checker.chunk_size(sets.size(), kLits));
    std::printf("  %zu survived BCP, %zu conflicted  (%.1f ms, %.2e sets/s)\n",
                r.size() - r.num_conflicts(), r.num_conflicts(), secs * 1e3, r.size() / secs);
    return 0;
}

int main(int argc, char **argv) {
    try {
        return argc < 2 ? tutorial() : survey(argv[1]);
    } catch (const gpap::Error &e) {
        std::fprintf(stderr, "gpap error: %s\n", e.what());
        return 1;
    }
}
