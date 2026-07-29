/* Minimal C consumer of libgpap: no Python, no PyTorch, no CUDA toolkit.
 *
 *   cmake -S cpp -B build && cmake --build build -j && ./build/gpap_demo [formula.cnf]
 *
 * With no argument it runs the engine's canonical demo formula through both entry points;
 * with a DIMACS path it samples that formula's solutions with the fused trajectory kernel.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gpap/gpap.h"

#define CHECK(call)                                                                  \
    do {                                                                             \
        gpap_status _st = (call);                                                    \
        if (_st != GPAP_OK) {                                                        \
            fprintf(stderr, "%s -> status %d: %s\n", #call, _st, gpap_last_error()); \
            return 1;                                                                \
        }                                                                            \
    } while (0)

/* DIMACS literal -> internal 2N literal index (+v -> 2(v-1), -v -> 2(v-1)+1). */
static int lit_index(int ext) { return 2 * (abs(ext) - 1) + (ext > 0 ? 0 : 1); }

static int demo_assumptions(void) {
    /* (x1 v x2) ^ (~x1 v x3) ^ (~x2 v ~x3) ^ (x1 v ~x2) */
    const int32_t flat[] = {1, 2, -1, 3, -2, -3, 1, -2};
    const int32_t lens[] = {2, 2, 2, 2};
    gpap_formula *f = NULL;
    gpap_engine *e = NULL;
    CHECK(gpap_formula_from_clauses(3, flat, lens, 4, &f));
    CHECK(gpap_engine_create(f, NULL, &e));

    const int32_t K = 2;
    CHECK(gpap_engine_reset(e, K));
    /* row 0 assumes x1 = TRUE, row 1 assumes x1 = FALSE */
    int64_t dec[2] = {lit_index(1), lit_index(-1)};
    CHECK(gpap_engine_apply(e, dec, 1));

    int8_t conflict[2], active[2];
    int32_t implied[2];
    CHECK(gpap_flags(e, active, conflict));
    CHECK(gpap_num_propagated(e, implied));

    int8_t asg[2 * 3 * 2];
    int32_t all_rows[2] = {0, 1};
    (void)all_rows;
    printf("assumption checking (K=%d):\n", K);
    for (int32_t k = 0; k < K; ++k)
        printf("  row %d: conflict=%d implications=%d\n", k, conflict[k], implied[k]);
    (void)asg;

    gpap_counters c;
    CHECK(gpap_counters_get(e, &c));
    printf("  counters: launches=%lld assigned=%lld decisions=%lld occ-entries=%lld\n",
           (long long)c.launches, (long long)c.props, (long long)c.decs, (long long)c.ops);

    int ok = (conflict[0] == 0 && conflict[1] == 1);
    printf("  %s\n", ok ? "PASS (x1=TRUE propagates; x1=FALSE conflicts)" : "FAIL");
    gpap_engine_free(e);
    gpap_formula_free(f);
    return ok ? 0 : 1;
}

static int demo_sampling(const char *path, int32_t K, int32_t B, int64_t seed) {
    gpap_formula *f = NULL;
    gpap_engine *e = NULL;
    gpap_formula_info info;
    CHECK(gpap_formula_from_dimacs(path, &f));
    gpap_formula_info_get(f, &info);
    printf("%s: N=%d M=%d max_len=%d split=%d|%d\n", path, info.num_vars, info.num_clauses,
           info.max_clause_len, info.shift, 32 - info.shift);

    CHECK(gpap_engine_create(f, NULL, &e));
    CHECK(gpap_engine_reset(e, K));

    /* Decision order: every variable, identity order, no separate support phase. */
    int32_t *order = (int32_t *)malloc((size_t)info.num_vars * sizeof(int32_t));
    for (int32_t v = 0; v < info.num_vars; ++v) order[v] = v;
    int32_t rounds = 0;
    CHECK(gpap_engine_trajectory(e, order, info.num_vars, info.num_vars, B, seed, 0, &rounds));

    int32_t mask_id = 0, models = 0, unsound = -1;
    CHECK(gpap_pair_mask_create(e, NULL, 0, &mask_id));      /* all variables */
    CHECK(gpap_select_complete(e, mask_id, 0, &models));
    CHECK(gpap_check_sound(e, 0, 4096, &unsound));

    gpap_counters c;
    CHECK(gpap_counters_get(e, &c));
    printf("  K=%d B=%d rounds=%d -> %d models (%.1f%% success), unsound=%d\n", K, B, rounds,
           models, 100.0 * models / K, unsound);
    printf("  work: %lld literals assigned (%lld decisions), %lld occ entries\n",
           (long long)c.props, (long long)c.decs, (long long)c.ops);

    if (models > 0) {                                        /* print the first model's support */
        int32_t show = models < 4 ? models : 4;
        int32_t *cols = (int32_t *)malloc((size_t)info.num_vars * sizeof(int32_t));
        for (int32_t v = 0; v < info.num_vars; ++v) cols[v] = v;
        int8_t *bits = (int8_t *)malloc((size_t)models * info.num_vars);
        CHECK(gpap_var_bits(e, 0, cols, info.num_vars, bits, models));   /* models = row capacity */
        for (int32_t i = 0; i < show; ++i) {
            printf("  model %d:", i);
            for (int32_t v = 0; v < (info.num_vars < 24 ? info.num_vars : 24); ++v)
                printf("%d", bits[(size_t)i * info.num_vars + v]);
            printf("%s\n", info.num_vars > 24 ? "..." : "");
        }
        free(bits);
        free(cols);
    }
    free(order);
    gpap_engine_free(e);
    gpap_formula_free(f);
    return unsound == 0 ? 0 : 1;
}

int main(int argc, char **argv) {
    printf("libgpap %s\n\n", gpap_version());
    if (argc > 1) {
        const int32_t K = argc > 2 ? atoi(argv[2]) : 4096;
        return demo_sampling(argv[1], K, 8, 12345);
    }
    return demo_assumptions();
}
