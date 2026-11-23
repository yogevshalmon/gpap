/* The host program of cpp/examples/embed: libgpap built without CUDA has the OpenCL plug only,
 * and it is the default there. Checks that, and that a host can open GPAP's OpenCL plug itself
 * (gpap::open_gpu) and create an engine on it. Exit status 1 at the first failed check. */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "gpap/gpap.h"
#include "gpap/gpu_api.hpp"

#define CHECK(cond, what)                                                                     \
    do {                                                                                      \
        if (!(cond)) {                                                                        \
            std::fprintf(stderr, "FAIL %s\n  %s:%d: %s\n  gpap_last_error: %s\n", what,       \
                         __FILE__, __LINE__, #cond, gpap_last_error());                       \
            std::exit(1);                                                                     \
        }                                                                                     \
        std::printf("ok   %s\n", what);                                                   \
    } while (0)

/* Width 1 on (-1 2) (-1 3) (-2 -3): exactly literal 1 fails. */
static void probe_finds_the_failed_literal(gpap_engine *e) {
    gpap_probe *p = nullptr;
    CHECK(gpap_probe_combinations(e, nullptr, 0, 1, nullptr, &p) == GPAP_OK, "a probe sweep");
    gpap_probe_stats st{};
    gpap_probe_stats_get(p, &st);
    int32_t lit = 0;
    CHECK(st.conflicts == 1 && gpap_probe_conflicts(p, &lit, 1) == GPAP_OK && lit == 1,
          "it finds the one failed literal");
    gpap_probe_free(p);
}

int main() {
    unsetenv("GPAP_BACKEND");
    CHECK(gpap_backend() && std::strcmp(gpap_backend(), "opencl") == 0,
          "without CUDA the default backend is opencl");

    const int32_t lits[] = {-1, 2, -1, 3, -2, -3};
    const int32_t lens[] = {2, 2, 2};
    gpap_formula *f = nullptr;
    CHECK(gpap_formula_from_clauses(3, lits, lens, 3, &f) == GPAP_OK, "build a formula");
    gpap_engine *e = nullptr;
    CHECK(gpap_engine_create(f, nullptr, &e) == GPAP_OK &&
              std::strcmp(gpap_engine_backend(e), "opencl") == 0,
          "gpap_engine_create opens the OpenCL plug");
    probe_finds_the_failed_literal(e);
    gpap_engine_free(e);

    std::unique_ptr<gpap::Gpu> gpu;
    CHECK(gpap::open_gpu("opencl", 0, &gpu) == GPAP_OK && gpu, "open_gpu opens the OpenCL plug");
    CHECK(gpap::engine_create_on(f, nullptr, std::move(gpu), &e) == GPAP_OK,
          "engine_create_on takes it");
    probe_finds_the_failed_literal(e);
    gpap_engine_free(e);

    setenv("GPAP_BACKEND", "cuda", 1);
    CHECK(gpap_backend() == nullptr && std::strstr(gpap_last_error(), "GPAP_WITH_CUDA=OFF"),
          "GPAP_BACKEND=cuda is refused, naming the build option");
    CHECK(gpap_engine_create(f, nullptr, &e) != GPAP_OK && e == nullptr,
          "so is an engine on it");
    CHECK(gpap::open_gpu("cuda", 0, &gpu) == GPAP_ERR_ARG && !gpu &&
              std::strstr(gpap_last_error(), "GPAP_WITH_CUDA=OFF"),
          "and open_gpu(\"cuda\")");
    gpap_formula_free(f);
    std::printf("all checks passed\n");
    return 0;
}
