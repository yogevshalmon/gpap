/* A host's own GPU plug, checked from the engine's side.
 *
 * A program that brings its own GPU layer implements gpap::Gpu over it and creates engines with
 * gpap::engine_create_on (include/gpap/gpu_api.hpp); mallob does, over its MGI layer. The plug
 * here forwards every call to GPAP's own plug for $GPAP_BACKEND (gpap::open_gpu) and records the
 * calls, which checks the contract a host relies on:
 *   1. an engine on the plug mines exactly what gpap_engine_create's engine mines;
 *   2. the engine reports the plug's own backend name;
 *   3. the engine makes the plug current before its first allocation;
 *   4. the automatic kernel choice follows the plug's resident_warps();
 *   5. a refused create (a NULL argument, a failed allocation) returns its status and a NULL
 *      engine, keeps the plug's message in gpap_last_error(), and destroys the plug once.
 * One line per check; exit status 1 at the first that fails. tests/engine/test_host_plug.py
 * runs it.
 *
 *   ./build/gpap_host_plug [formula.cnf]     # default: a 7-variable formula with known answers
 */
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "gpap/gpap.h"
#include "gpap/gpu_api.hpp"

#define CHECK(cond, what)                                                                     \
    do {                                                                                      \
        if (!(cond)) {                                                                        \
            std::fprintf(stderr, "FAIL %s\n  %s:%d: %s\n  gpap_last_error: %s\n", what,       \
                         __FILE__, __LINE__, #cond, gpap_last_error());                       \
            std::exit(1);                                                                     \
        }                                                                                     \
    } while (0)

namespace {

int g_destroyed = 0;            /* Recorder destructors that have run */

/* Forwards to GPAP's own plug and records what the engine asks of it. */
class Recorder final : public gpap::Gpu {
  public:
    std::vector<std::string> calls;     /* "make_current", "alloc", "run <key>" in call order */

    Recorder(std::unique_ptr<gpap::Gpu> inner, int32_t warps, int fail_alloc)
        : inner_(std::move(inner)), warps_(warps), fail_alloc_(fail_alloc) {}
    ~Recorder() override { g_destroyed++; }

    const char *backend() const override { return "host-recorder"; }
    const char *device_name() const override { return inner_->device_name(); }
    gpap_status make_current() override {
        calls.push_back("make_current");
        return inner_->make_current();
    }
    gpap_status mem_info(int64_t *free_bytes, int64_t *total_bytes) override {
        return inner_->mem_info(free_bytes, total_bytes);
    }
    bool free_is_measured() const override { return inner_->free_is_measured(); }
    int64_t max_buffer_bytes() const override { return inner_->max_buffer_bytes(); }

    gpap_status alloc(size_t bytes, gpap::GpuMem *out) override {
        calls.push_back("alloc");
        if (++allocs_ == fail_alloc_) {
            gpap::set_error("host-recorder: allocation %d refused (injected)", allocs_);
            return GPAP_ERR_MEMORY;
        }
        return inner_->alloc(bytes, out);
    }
    void free(gpap::GpuMem m) override { inner_->free(m); }
    gpap_status zero(gpap::GpuMem m, size_t bytes) override { return inner_->zero(m, bytes); }
    gpap_status upload(gpap::GpuMem dst, const void *src, size_t bytes) override {
        return inner_->upload(dst, src, bytes);
    }
    gpap_status download(void *dst, gpap::GpuMem src, size_t offset, size_t bytes) override {
        return inner_->download(dst, src, offset, bytes);
    }
    gpap_status copy(gpap::GpuMem dst, gpap::GpuMem src, size_t bytes) override {
        return inner_->copy(dst, src, bytes);
    }
    gpap_status adopt(uintptr_t foreign, gpap::GpuMem *out) override {
        return inner_->adopt(foreign, out);
    }

    bool has_kernel(const char *key) const override { return inner_->has_kernel(key); }
    gpap_status run(const char *key, int64_t n0, int64_t n1,
                    std::initializer_list<gpap::GpuArg> args) override {
        calls.push_back(std::string("run ") + key);
        return inner_->run(key, n0, n1, args);
    }
    int32_t resident_warps() const override { return warps_; }

  private:
    std::unique_ptr<gpap::Gpu> inner_;
    int32_t warps_;
    int fail_alloc_;                    /* refuse this allocation (1 = the first); 0 = none */
    int allocs_ = 0;
};

/* A Recorder over GPAP's own plug for $GPAP_BACKEND, device 0. */
std::unique_ptr<Recorder> recorder(int32_t warps = 0, int fail_alloc = 0) {
    std::unique_ptr<gpap::Gpu> inner;
    CHECK(gpap::open_gpu(nullptr, 0, &inner) == GPAP_OK && inner, "open GPAP's own plug");
    return std::unique_ptr<Recorder>(new Recorder(std::move(inner), warps, fail_alloc));
}

/* (-1 2) (-1 3) (-2 -3) (-4 -5 6) (-6 7) (-6 -7), 7 variables: literals 1 and 6 fail on their
 * own; with -1 and -6 assumed, exactly the pairs (2,3) and (4,5) conflict. */
gpap_formula *tiny_formula() {
    const int32_t lits[] = {-1, 2, -1, 3, -2, -3, -4, -5, 6, -6, 7, -6, -7};
    const int32_t lens[] = {2, 2, 2, 3, 2, 2};
    gpap_formula *f = nullptr;
    CHECK(gpap_formula_from_clauses(7, lits, lens, 6, &f) == GPAP_OK, "build the formula");
    return f;
}

struct Mined {
    std::vector<int32_t> units;                 /* negated failed literals, sorted */
    std::vector<std::array<int32_t, 2>> pairs;  /* conflicting pairs after the units, sorted */
    bool unsat = false;
    bool operator==(const Mined &o) const {
        return units == o.units && pairs == o.pairs && unsat == o.unsat;
    }
};

/* Every k-subset of the free literals: the conflicting tuples, each and the list sorted. */
std::vector<std::vector<int32_t>> sweep(gpap_engine *e, int32_t k, bool *unsat) {
    gpap_probe *p = nullptr;
    CHECK(gpap_probe_combinations(e, nullptr, 0, k, nullptr, &p) == GPAP_OK, "probe sweep");
    gpap_probe_stats st{};
    gpap_probe_stats_get(p, &st);
    std::vector<std::vector<int32_t>> tuples;
    if (st.stop == GPAP_PROBE_UNSAT) {
        *unsat = true;
    } else {
        std::vector<int32_t> flat((size_t)st.conflicts * k);
        if (st.conflicts > 0)
            CHECK(gpap_probe_conflicts(p, flat.data(), st.conflicts) == GPAP_OK, "conflicts");
        for (int64_t i = 0; i < st.conflicts; i++) {
            std::vector<int32_t> t(flat.begin() + i * k, flat.begin() + (i + 1) * k);
            std::sort(t.begin(), t.end());
            tuples.push_back(t);
        }
        std::sort(tuples.begin(), tuples.end());
    }
    gpap_probe_free(p);
    return tuples;
}

/* Failed literals, assumed as units, then the conflicting pairs. */
Mined mine(gpap_engine *e) {
    Mined m;
    for (const auto &t : sweep(e, 1, &m.unsat)) m.units.push_back(-t[0]);
    std::sort(m.units.begin(), m.units.end());
    if (m.unsat) return m;
    if (!m.units.empty())
        CHECK(gpap_assume_units(e, m.units.data(), (int32_t)m.units.size()) == GPAP_OK,
              "assume the units");
    for (const auto &t : sweep(e, 2, &m.unsat)) m.pairs.push_back({t[0], t[1]});
    return m;
}

int index_of(const std::vector<std::string> &calls, const std::string &call) {
    const auto it = std::find(calls.begin(), calls.end(), call);
    return it == calls.end() ? -1 : (int)(it - calls.begin());
}

/* Fixpoint launches of each kernel width: [0] scalar (.w0), [1] WAVE (.w1). */
std::array<int, 2> fixpoint_runs(const std::vector<std::string> &calls) {
    std::array<int, 2> n{0, 0};
    for (const std::string &c : calls)
        if (c.rfind("run fixpoint", 0) == 0) n[c.find(".w1") != std::string::npos]++;
    return n;
}

void ok(const char *what) { std::printf("ok   %s\n", what); }

void check_same_results(const gpap_formula *f, bool known_answer) {
    gpap_engine *own = nullptr, *hosted = nullptr;
    CHECK(gpap_engine_create(f, nullptr, &own) == GPAP_OK, "an engine on GPAP's own plug");
    std::unique_ptr<Recorder> plug = recorder();
    Recorder *rec = plug.get();         /* owned by the engine from here on */
    CHECK(gpap::engine_create_on(f, nullptr, std::move(plug), &hosted) == GPAP_OK && hosted,
          "an engine on the host plug");
    const Mined a = mine(own), b = mine(hosted);
    CHECK(a == b, "1. the host plug's engine mines what gpap_engine_create's engine mines");
    if (known_answer) {
        const Mined want{{-6, -1}, {{2, 3}, {4, 5}}, false};
        CHECK(b == want, "1. the known answer");
    }
    std::printf("     %zu units, %zu pairs, unsat=%d on %s (%s)\n", b.units.size(),
                b.pairs.size(), (int)b.unsat, gpap_engine_backend(own),
                gpap_engine_device_name(own));
    ok("1. same results as gpap_engine_create");

    CHECK(std::strcmp(gpap_engine_backend(hosted), "host-recorder") == 0,
          "2. the engine reports the plug's backend name");
    ok("2. the engine reports the plug's backend name");

    const int current = index_of(rec->calls, "make_current");
    const int first_alloc = index_of(rec->calls, "alloc");
    CHECK(first_alloc >= 0 && current >= 0 && current < first_alloc,
          "3. make_current comes before the first allocation");
    ok("3. make_current comes before the first allocation");
    gpap_engine_free(own);
    gpap_engine_free(hosted);
}

void check_kernel_choice() {
    gpap_formula *f = tiny_formula();      /* far below the WAVE clause threshold */
    for (const int32_t warps : {0, 1 << 20}) {
        std::unique_ptr<Recorder> plug = recorder(warps);
        Recorder *rec = plug.get();
        gpap_engine *e = nullptr;
        CHECK(gpap::engine_create_on(f, nullptr, std::move(plug), &e) == GPAP_OK, "create");
        const std::string wave_key =
            "fixpoint.s" + std::to_string(gpap_engine_shift(e)) + ".w1";
        if (!gpap_engine_auto(e) || !rec->has_kernel(wave_key.c_str())) {
            std::printf("skip 4. the kernel choice (a pinned kernel, or no WAVE kernel here)\n");
            gpap_engine_free(e);
            gpap_formula_free(f);
            return;
        }
        bool unsat = false;
        sweep(e, 2, &unsat);
        const std::array<int, 2> runs = fixpoint_runs(rec->calls);
        if (warps == 0)
            CHECK(runs[0] > 0 && runs[1] == 0, "4. no resident warps keep a small batch scalar");
        else
            CHECK(runs[1] > 0 && runs[0] == 0, "4. enough resident warps run a small batch WAVE");
        gpap_engine_free(e);
    }
    gpap_formula_free(f);
    ok("4. the kernel choice follows resident_warps()");
}

void check_refusals() {
    gpap_formula *f = tiny_formula();
    gpap_engine *out = nullptr;
    const auto refused = [&](gpap_status want, gpap_status got, int destroyed_before) {
        return got == want && out == nullptr && g_destroyed == destroyed_before + 1;
    };

    int before = g_destroyed;
    out = reinterpret_cast<gpap_engine *>(1);
    gpap_status st = gpap::engine_create_on(nullptr, nullptr, recorder(), &out);
    CHECK(refused(GPAP_ERR_ARG, st, before), "5. a NULL formula: ERR_ARG, no engine, plug freed");

    out = reinterpret_cast<gpap_engine *>(1);
    st = gpap::engine_create_on(f, nullptr, nullptr, &out);
    CHECK(st == GPAP_ERR_ARG && out == nullptr, "5. a NULL plug: ERR_ARG, no engine");

    before = g_destroyed;
    st = gpap::engine_create_on(f, nullptr, recorder(), nullptr);
    CHECK(st == GPAP_ERR_ARG && g_destroyed == before + 1, "5. a NULL out: ERR_ARG, plug freed");

    for (const int nth : {1, 3}) {
        before = g_destroyed;
        out = reinterpret_cast<gpap_engine *>(1);
        st = gpap::engine_create_on(f, nullptr, recorder(0, nth), &out);
        CHECK(refused(GPAP_ERR_MEMORY, st, before),
              "5. a refused allocation: its status, no engine, plug freed once");
        CHECK(std::strstr(gpap_last_error(), "refused (injected)") != nullptr,
              "5. the plug's message reaches gpap_last_error()");
    }

    std::unique_ptr<gpap::Gpu> none;
    CHECK(gpap::open_gpu("no-such-backend", 0, &none) == GPAP_ERR_ARG && !none &&
              std::strstr(gpap_last_error(), "no-such-backend") != nullptr,
          "5. open_gpu refuses an unknown backend by name");
    gpap_formula_free(f);
    ok("5. a refused create returns its status, a NULL engine and frees the plug once");
}

}  // namespace

int main(int argc, char **argv) {
    std::printf("backend %s\n", gpap_backend() ? gpap_backend() : "(none)");
    gpap_formula *f = nullptr;
    if (argc > 1) {
        CHECK(gpap_formula_from_dimacs(argv[1], &f) == GPAP_OK, "read the formula");
    } else {
        f = tiny_formula();
    }
    check_same_results(f, argc <= 1);
    gpap_formula_free(f);
    check_kernel_choice();
    check_refusals();
    std::printf("all checks passed\n");
    return 0;
}
