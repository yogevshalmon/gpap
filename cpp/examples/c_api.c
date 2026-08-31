/* The C ABI, end to end — no C++, no Python, no PyTorch, no CUDA toolkit.
 *
 *   ./gpap_c_api                # both entry points on a 3-variable formula
 *   ./gpap_c_api formula.cnf    # sample that formula's solutions
 *
 * This is the reference for `gpap/gpap.h`. Everything here is the raw ABI: 0-based variable
 * indices, caller-owned buffers, `gpap_status` returns. C++ callers should prefer
 * <gpap/gpap.hpp>, which wraps all of it in RAII and exceptions (see sample.cpp).
 *
 * Build: cmake -S cpp -B build && cmake --build build -j
 */
#include <stdio.h>
#include <stdlib.h>

#include "gpap/gpap.h"

/* Every ABI call returns gpap_status; gpap_last_error() explains a failure in words. Real code
 * would propagate rather than exit — this keeps the example's control flow readable. */
#define CHECK(what, call)                                        \
    do {                                                         \
        if ((call) != GPAP_OK) {                                 \
            fprintf(stderr, "gpap: %s failed: %s\n", (what),     \
                    gpap_last_error());                          \
            goto fail;                                           \
        }                                                        \
    } while (0)

/* DIMACS literal -> internal 2N literal index: +v -> 2(v-1), -v -> 2(v-1)+1, so l^1 negates. */
static int64_t lit_index(int ext) { return 2 * (abs(ext) - 1) + (ext > 0 ? 0 : 1); }

/* ------------------------------------------------------------------ entry point 1
 * K assumption sets, propagated independently, in ONE launch. This is the primitive the whole
 * engine exists for: not one big propagation, but K of them at once. */
static int demo_assumptions(void) {
    /* (x1 v x2) ^ (~x1 v x3) ^ (~x2 v ~x3) ^ (x1 v ~x2), as flat literals + per-clause lengths */
    const int32_t flat[] = {1, 2, -1, 3, -2, -3, 1, -2};
    const int32_t lens[] = {2, 2, 2, 2};
    const int32_t K = 2;
    gpap_formula *f = NULL;
    gpap_engine *e = NULL;
    int rc = 1;

    CHECK("formula_from_clauses", gpap_formula_from_clauses(3, flat, lens, 4, &f));
    CHECK("engine_create", gpap_engine_create(f, NULL, &e));   /* NULL opts = every default */
    CHECK("engine_reset", gpap_engine_reset(e, K));

    /* One row per assumption set, padded with -1. Row 0 assumes x1=TRUE, row 1 assumes x1=FALSE. */
    int64_t decisions[2] = {lit_index(1), lit_index(-1)};
    CHECK("engine_apply", gpap_engine_apply(e, decisions, /*L=*/1));

    int8_t active[2], conflict[2];
    int32_t implied[2];
    CHECK("flags", gpap_flags(e, active, conflict, K));
    CHECK("num_propagated", gpap_num_propagated(e, implied, K));

    /* The full assignment of every row: (K, 2N) int8, 1 where that literal is TRUE. Row-major,
     * literal l of row k at [k * 2N + l] — so variable v is TRUE at 2(v-1) and FALSE at 2(v-1)+1. */
    const int32_t two_n = 2 * 3;
    int8_t assignment[2 * 6];
    int32_t rows[2] = {0, 1};
    CHECK("selection_set", gpap_selection_set(e, /*slot=*/0, rows, K));
    CHECK("assignment_rows", gpap_assignment_rows(e, /*slot=*/0, assignment, K));

    printf("entry point 1 — assumption checking (K=%d):\n", K);
    for (int32_t k = 0; k < K; ++k) {
        printf("  row %d (x1=%s): %s, %d implication(s)", k, k == 0 ? "TRUE " : "FALSE",
               conflict[k] ? "CONFLICT" : "ok      ", implied[k]);
        if (!conflict[k]) {
            printf("  ->");
            for (int32_t v = 1; v <= 3; ++v) {
                const int8_t t = assignment[k * two_n + 2 * (v - 1)];
                const int8_t fl = assignment[k * two_n + 2 * (v - 1) + 1];
                if (t || fl) printf(" %s%d", t ? "" : "-", v);
            }
        }
        printf("\n");
    }

    gpap_counters c;
    CHECK("counters_get", gpap_counters_get(e, &c));
    printf("  work: %lld launch(es), %lld literals assigned, %lld occurrence entries\n",
           (long long)c.launches, (long long)c.props, (long long)c.ops);

    rc = (conflict[0] == 0 && conflict[1] == 1) ? 0 : 1;
    printf("  %s\n", rc == 0 ? "PASS — x1=TRUE propagates to a fixpoint; x1=FALSE conflicts"
                             : "FAIL — unexpected result");
fail:
    gpap_engine_free(e);          /* both are NULL-safe, so one cleanup path serves every exit */
    gpap_formula_free(f);
    return rc;
}

/* ------------------------------------------------------------------ entry point 2
 * The whole randomized decide-and-propagate loop, per row, in ONE launch. */
static int demo_sampling(const char *path, int32_t K, int32_t B, int64_t seed) {
    gpap_formula *f = NULL;
    gpap_engine *e = NULL;
    int32_t *order = NULL;
    int32_t *cols = NULL;
    int8_t *bits = NULL;
    gpap_formula_info info;
    int rc = 1;

    CHECK("formula_from_dimacs", gpap_formula_from_dimacs(path, &f));
    gpap_formula_info_get(f, &info);
    printf("%s: %d vars, %d clauses, longest clause %d, packed split %d|%d\n", path,
           info.num_vars, info.num_clauses, info.max_clause_len, info.shift, 32 - info.shift);

    CHECK("engine_create", gpap_engine_create(f, NULL, &e));
    CHECK("engine_reset", gpap_engine_reset(e, K));

    /* Decision order: identity, single phase. A real sampler puts the independent support FIRST
     * and passes its size as n_support, so phase 1 reaches a BCP fixpoint over the support before
     * any other variable is decided — see gpap::Sampler, which does this for you. */
    order = (int32_t *)malloc((size_t)info.num_vars * sizeof(int32_t));
    cols = (int32_t *)malloc((size_t)info.num_vars * sizeof(int32_t));
    if (!order || !cols) { fprintf(stderr, "out of memory\n"); goto fail; }
    for (int32_t v = 0; v < info.num_vars; ++v) order[v] = cols[v] = v;

    int32_t rounds = 0;
    CHECK("engine_trajectory", gpap_engine_trajectory(e, order, info.num_vars,
                                                      /*n_support=*/info.num_vars, B, seed,
                                                      /*soft=*/0, &rounds));

    /* A row is a model iff it never conflicted AND has no free variable left. That test runs on
     * the device and only the COUNT comes back; the rows themselves stay in slot 0. */
    int32_t mask_id = 0, models = 0, unsound = -1;
    CHECK("pair_mask_create", gpap_pair_mask_create(e, NULL, 0, &mask_id));   /* all variables */
    CHECK("select_complete", gpap_select_complete(e, mask_id, /*slot=*/0, &models));
    /* max_rows = 0 = EVERY selected row. A positive bound checks a PREFIX, so "0 unsound" would
     * then be a claim about that prefix and not about the rows this program just counted. */
    CHECK("check_sound", gpap_check_sound(e, /*slot=*/0, /*max_rows=*/0, &unsound));

    printf("entry point 2 — trajectory sampling (K=%d, B=%d, %d decision rounds):\n", K, B, rounds);
    printf("  %d models (%.1f%% of trajectories), unsound rows: %d\n", models,
           100.0 * (double)models / (double)K, unsound);

    if (models > 0) {
        /* Gather the variables of the selected rows straight out of the assignment bitmap. */
        const int32_t show = models < 3 ? models : 3;
        bits = (int8_t *)malloc((size_t)models * (size_t)info.num_vars);
        if (!bits) { fprintf(stderr, "out of memory\n"); goto fail; }
        CHECK("var_bits", gpap_var_bits(e, /*slot=*/0, cols, info.num_vars, bits, models));
        const int32_t width = info.num_vars < 32 ? info.num_vars : 32;
        for (int32_t i = 0; i < show; ++i) {
            printf("  model %d: ", i);
            for (int32_t v = 0; v < width; ++v)
                printf("%d", bits[(size_t)i * (size_t)info.num_vars + v]);
            printf("%s\n", info.num_vars > width ? "..." : "");
        }
    }
    rc = (unsound == 0) ? 0 : 1;
fail:
    free(bits);
    free(cols);
    free(order);
    gpap_engine_free(e);
    gpap_formula_free(f);
    return rc;
}

int main(int argc, char **argv) {
    printf("libgpap %s\n\n", gpap_version());
    fflush(stdout);          /* so a later error on stderr cannot appear above this banner */
    if (argc > 1)
        return demo_sampling(argv[1], /*K=*/argc > 2 ? atoi(argv[2]) : 4096, /*B=*/8, /*seed=*/12345);
    return demo_assumptions();
}
