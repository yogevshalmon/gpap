# libgpap — the GPAP engine as a C/C++ library

The K-parallel packed BCP engine (`docs/ENGINE.md`) built as a shared library with a C ABI, so
it can be linked from C++ as well as driven from Python. The GPU kernels are **not rewritten**:
[`kernels/bcp.py`](kernels/bcp.py) stays the single source of truth and is
compiled ahead of time into cubins that are embedded in the library (verified bit-identical, at
the SASS level, to what the Triton JIT produces).

```
             ┌──────────────────────────────────────────────┐
C/C++  ─────►│ libgpap.so   —  include/gpap/gpap.h          │
             │   formula (DIMACS / CSR / packed init)       │
Python ─────►│   engine state, launches, readout            │──► embedded cubins ──► GPU
 (ctypes)    │   owns its device memory (cuMemAlloc)        │
             └──────────────────────────────────────────────┘
```

## Build

Requirements are deliberately thin — **no CUDA toolkit and no nvcc**:

| | why |
|---|---|
| a C++17 compiler + CMake | the host library |
| `libcuda.so.1` | the CUDA *driver*, which ships with the NVIDIA driver |
| `cuda.h` | driver header — taken from the toolkit if present, else from the triton wheel |
| Python + triton | **build time only**, to compile the kernels ahead of time |

```bash
scripts/build_core.sh                    # both steps below, with diagnostics
# ...which is just:
python cpp/tools/build_kernels.py        # kernels/{bcp,helpers}.py -> src/generated/*.c
cmake -S cpp -B build && cmake --build build -j

./build/gpap_c_api                       # both entry points, plain C
./build/gpap_sample formula.cnf 16384    # sample solutions (C++ wrapper)
```

Consuming it from another CMake project:

```cmake
find_package(gpap REQUIRED)
target_link_libraries(my_solver PRIVATE gpap::gpap)
```

`build_kernels.py` targets the local GPU's architecture by default; `--arch 89` cross-compiles
(no GPU needed). Rerun it whenever a kernel changes — that is the only step that regenerates
GPU code.

## Using it from C++

`<gpap/gpap.hpp>` is header-only and has two tiers. **Tier 1 is the whole API for the normal
case** — it needs no knowledge of how the engine works:

```cpp
#include <gpap/gpap.hpp>

gpap::Sampler sampler("formula.cnf");
gpap::Solutions sols = sampler.sample(100000);       // draw 100k satisfying assignments

printf("%zu solutions at %.0f/s\n", sols.size(), sols.rate());
for (std::size_t i = 0; i < sols.size(); ++i)
    for (std::size_t j = 0; j < sols.columns(); ++j)
        bool value = sols(i, j);                     // of variable sols.variables()[j]
```

The sampler sizes the GPU batch to free VRAM, picks the propagate kernel, orders decisions so
propagation does the work, and runs as many chunks as the target needs. Variables and literals
are **DIMACS** — the same numbers as in the `.cnf` file. `SampleOptions` covers the rest
(`seed`, `time_limit_s`, `max_chunks`, `max_keep`, `variables`, `portfolio`, `var_order`,
`batch`, `lookahead`, `retry_with_lookahead`, `device`, `k`, `k_cap`, `verify`,
`min_measure_chunks`).

**The loop itself lives in the library** (`gpap_sampler_*`, `src/sample.cpp`), shared verbatim
with the Python face — chunks, the adaptive config bandit, the stop valves and the accounting
are ONE implementation, so the two faces cannot drift and, at the same seed and a single
config, they find the *same* models (gated by `tests/test_face_parity.py`).

`verify` re-checks on the device that every returned row really satisfies every clause. It is
**off by default**, on every face — it is a self-test, not part of producing samples: a row is
only counted when it is active and fully assigned, which already means no clause was emptied.
Set it in tests, or in a run whose output will be published; `Sampler::sample` then throws if any
row fails.

Two levers matter when a formula yields nothing under the defaults, and they are different in
kind. `portfolio = gpap::default_portfolio()` races the three uniformity-safe configs the
Python pipeline runs by default (random B=8 / random B=1 / structured B=1) as a throughput
bandit — it is what samples the search-heavy instances a single random-order config cannot,
and only the winning config's rows are returned. `retry_with_lookahead` (**off** by default)
instead flips the in-kernel risk polarity once after a zero-yield window: on b12 unrolled 50
cycles that is the difference between 0 solutions and ~1100/s — but the risk polarity is **not
uniform** (Barbarik rejects it on 64 of the 65 circuit instances it can sample), which is why
it is opt-in here exactly as `--allow-soft` is on the Python side. `Solutions::lookahead_used()`
says whether any returned rows came from it.

The other entry point — testing many assumption sets in one launch, which is the primitive the
whole engine exists for:

```cpp
gpap::AssumptionChecker checker("formula.cnf");
auto r = checker.check(sets);                        // ANY number of sets, propagated independently
r.conflict(1);                                       // did set 1 falsify a clause?
r.value(0, 5);                                       // True / False / Free for x5 under set 0
```

`sets` may be as long as you like: the checker sizes each launch to the free VRAM and the
formula's shape, and results come back in **input order**, so how many launches it took is not
observable (`chunk_size()` tells you if you care). Pass `with_assignments=false` when only the
conflict flags matter — the per-row assignment is `n x 2N` bytes and is the term that makes a
large call expensive; chunking cannot shrink what you asked to be returned.

Examples: [`examples/sample.cpp`](examples/sample.cpp),
[`examples/assumptions.cpp`](examples/assumptions.cpp),
[`examples/probe.cpp`](examples/probe.cpp) — see [`../examples/`](../examples/)
for the Python ones and how to run all of them.

### Tier 2 — driving the engine yourself

`gpap::Formula` and `gpap::Engine` are a thin RAII/exception wrapper over the C ABI, for callers
who want to own the chunk loop, the decision order or the readout. Tier 1 is written entirely in
terms of them, so nothing is hidden behind a private door. Tier 2 speaks the ABI's **0-based**
variable indices, not DIMACS.

```cpp
gpap::Formula formula = gpap::Formula::from_dimacs("formula.cnf");
gpap::Engine  engine(formula);                       // throws gpap::Error on failure

engine.reset(65536);                                 // 65k independent trajectories
engine.trajectory(order, n_support, /*B=*/8, seed);  // order: support variables FIRST
int models = engine.select_complete(engine.pair_mask());
auto bits  = engine.var_bits(formula.support(), models);   // (models x |support|) 0/1
```

**Decision order matters** (tier 1 does this for you). Put the independent support
(`formula.support()`, read from the file's `c ind` lines) first and pass its size as
`n_support`: phase 1 reaches a BCP fixpoint over the support before any completion variable is
decided. Without that split, decisions are spent on variables propagation would have determined
and the success rate collapses — on b03-c25 it is the difference between 9% and 0%.

Or straight from C, with no C++ at all:

```c
#include "gpap/gpap.h"

gpap_formula *f; gpap_engine *e;
gpap_formula_from_dimacs("formula.cnf", &f);
gpap_engine_create(f, NULL, &e);   /* NULL opts = every default: device 0, AUTO kernel
                                      (re-picked per reset), split from the formula */

gpap_engine_reset(e, /*K=*/65536);
int32_t rounds;
gpap_engine_trajectory(e, order, P, n_support, /*B=*/8, /*seed=*/42, /*soft=*/0, &rounds);

int32_t mask, models, unsound;
gpap_pair_mask_create(e, NULL, 0, &mask);          /* all variables */
gpap_select_complete(e, mask, /*slot=*/0, &models);
gpap_check_sound(e, 0, /*max_rows=*/0, &unsound);  /* 0 = EVERY selected row; must be 0.
                                                      A positive bound checks a PREFIX, and
                                                      then "0 unsound" only speaks for it. */
```

Full C example: [`examples/c_api.c`](examples/c_api.c). The two entry points are the same ones the
Python API exposes — batched assumption checking (`gpap_engine_apply`) and fused trajectory
sampling (`gpap_engine_trajectory`) — plus selections and readout. Every readout hands back a
small host buffer; a `(K, M)` or `(K, N)` transient never crosses the bus.

**Getting answers out — [`SELECTION_API.md`](SELECTION_API.md).** `pair_mask` / `select_complete` /
`selection_rows` / `var_bits` are the least obvious part of this ABI, and that document explains
them from the problem they solve, with a worked 6-row example. Read it before driving `gpap::Engine`
directly; `gpap::Sampler` handles all of it for you.

What tier 1 deliberately does *not* do is the Python pipeline's formula-level work:
entailed-clause mining, ELS/unit-closure preprocessing, the uniformity readout and the circuit
front-ends all live in `gpap.sampling` / `gpap.circuits`. The mechanism — the chunk loop, the
config bandit, the batch sizing, the accounting — is the library's and both faces share it;
the policy (what to mine, which portfolio to hand it, what to report) stays where it can
change weekly without an ABI in the way.

## Using it from Python

`gpap.engine._core` binds the same ABI with `ctypes` (no extension module to build, no
interpreter/ABI coupling). The library is found via `$GPAP_LIB`, then `<repo>/build/libgpap.so`,
then `gpap/lib/libgpap.so` (the copy installed alongside the package). The build tree comes
first on purpose: in a development checkout it is the fresh one, and a stale installed copy
would otherwise silently shadow a rebuild.

```python
from gpap.engine import _core
core = _core.Engine(_core.Formula.from_file("formula.cnf"), mode="wave")
core.reset(K)
core.trajectory_dev(order.data_ptr(), order.numel(), n_support, B, seed)   # torch tensor, zero-copy
models = core.select_complete(core.pair_mask(None), 0)
bits = core.var_bits(0, support_cols, models)      # (models, |support|) uint8
```

## Layout

```
include/gpap/gpap.h      the C ABI — the stable boundary, and all a C consumer needs
include/gpap/gpap.hpp    header-only C++: tier 1 (Sampler/AssumptionChecker) over tier 2 (RAII)
src/formula.cpp          DIMACS tokenizer (+ `c ind` support), occurrence CSR, packed word
src/engine.cpp           state ownership, reset/flush cache, the two entry points, counters
src/sample.cpp           THE sampling loop both faces share: chunks + config bandit + valves +
                         kept-row buckets + delivery/verify accounting (gpap_sampler_*)
src/readout.cpp          selections, on-device soundness, bit gathers, assignment rows
src/calibrate.cpp        the memory model, K sizing and THE wave/scalar dispatch rule
src/probe.cpp            combinatorial probing: every k-subset of a literal set, paged + budgeted
src/check.cpp            batched checking: THE chunk loop shared by both AssumptionChecker faces
src/runtime.cpp          driver-API helpers, cubin registry, launches, device buffers
src/gpap_internal.hpp    shared internals (not installed)
src/generated/           GENERATED cubins + lookup table (build_kernels.py output; gitignored)
kernels/helpers.py       support kernels (Triton, so the toolchain stays nvcc-free)
tools/build_kernels.py   the AOT compiler
tools/make_libcuda_stub.py  link-only libcuda stub, so CI can build without a driver
                            (`--print` lists every driver call the core makes)
examples/c_api.c         standalone C consumer (raw ABI) — the header's reference
examples/sample.cpp      solution sampling from C++ (gpap::Sampler)
examples/assumptions.cpp batched assumption checking from C++ (gpap::AssumptionChecker)
examples/probe.cpp       entailed-clause probing from C++ (gpap::Prober)
```

The core's contract is gated by [`tests/test_core.py`](../tests/test_core.py) against two
INDEPENDENT pure-Python oracles — the formula builder in
[`tests/reference_formula.py`](../tests/reference_formula.py) and the BCP in
[`tests/reference_bcp.py`](../tests/reference_bcp.py) — plus direct clause-satisfaction checks on
every emitted model, in both kernel modes and both packed splits. (Not
`gpap/engine/formula.py`: that is a FACADE over `src/formula.cpp`, so gating against it would be
gating the core against itself.)

### Sanitizers

The library and the examples build clean under ASan + UBSan, and the whole Python suite runs
against such a build. One option is REQUIRED — the CUDA driver reserves address space that
collides with ASan's shadow gap, and without it `cuInit` fails with a misleading "out of memory":

```bash
cmake -S cpp -B build-asan -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined" \
      -DCMAKE_C_FLAGS="-fsanitize=address,undefined" \
      -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
      -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address,undefined" && cmake --build build-asan -j
ASAN_OPTIONS=protect_shadow_gap=0 ./build-asan/gpap_sample formula.cnf 20000
# the Python suite against the same library:
LD_PRELOAD=$(gcc -print-file-name=libasan.so) ASAN_OPTIONS=protect_shadow_gap=0:detect_leaks=0 \
  GPAP_LIB=$PWD/build-asan/libgpap.so python -m pytest -q
```

Leak checking needs `leak:libcuda.so` suppressed: the WSL driver shim leaks ~50 KB in `cuInit`
on every process, which is not ours.
