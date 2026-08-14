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
python cpp/tools/build_kernels.py        # kernels.py + helpers.py -> src/generated/*.c
cmake -S cpp -B build && cmake --build build -j

./build/gpap_demo                        # assumption checking on the canonical demo formula
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
propagation does the work, runs as many chunks as the target needs, and verifies on the device
that every row it returns satisfies every clause. Variables and literals are **DIMACS** — the
same numbers as in the `.cnf` file. `SampleOptions` covers the rest (`seed`, `time_limit_s`,
`max_keep`, `variables`, `batch`, `lookahead`, `auto_retry`, `device`, `k`).

`auto_retry` (on by default) is worth knowing about: if the configured settings find nothing in
the first few chunks, it flips the in-kernel risk polarity once before giving up. On b12 unrolled
50 cycles that is the difference between **0 solutions and ~1100/s** — both `batch=8` and
`batch=1` return nothing there while the risk polarity succeeds at 3%. It is one retry, not a
search; the Python pipeline runs a proper adaptive portfolio over the same knobs.

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
[`examples/assumptions.cpp`](examples/assumptions.cpp).

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
gpap_engine_create(f, NULL, &e);   /* NULL opts = device 0, scalar, auto split */

gpap_engine_reset(e, /*K=*/65536);
int32_t rounds;
gpap_engine_trajectory(e, order, P, n_support, /*B=*/8, /*seed=*/42, /*soft=*/0, &rounds);

int32_t mask, models, unsound;
gpap_pair_mask_create(e, NULL, 0, &mask);          /* all variables */
gpap_select_complete(e, mask, /*slot=*/0, &models);
gpap_check_sound(e, 0, 4096, &unsound);            /* must be 0 */
```

Full C example: [`examples/demo.c`](examples/demo.c). The two entry points are the same ones the
Python API exposes — batched assumption checking (`gpap_engine_apply`) and fused trajectory
sampling (`gpap_engine_trajectory`) — plus selections and readout. Every readout hands back a
small host buffer; a `(K, M)` or `(K, N)` transient never crosses the bus.

**Getting answers out — [`SELECTION_API.md`](SELECTION_API.md).** `pair_mask` / `select_complete` /
`selection_rows` / `var_bits` are the least obvious part of this ABI, and that document explains
them from the problem they solve, with a worked 6-row example. Read it before driving `gpap::Engine`
directly; `gpap::Sampler` handles all of it for you.

What tier 1 deliberately does *not* do is the Python pipeline's per-instance work: the adaptive
config portfolio, entailed-clause mining, preprocessing, uniformity readout and the circuit
front-ends all live in `gpap.sampling` / `gpap.circuits`. The C++ side is the engine plus a
straightforward sampling loop over it.

## Using it from Python

`gpap.engine._core` binds the same ABI with `ctypes` (no extension module to build, no
interpreter/ABI coupling). The library is found via `$GPAP_LIB`, then `gpap/lib/libgpap.so`,
then `<repo>/build/libgpap.so`.

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
src/readout.cpp          selections, on-device soundness, bit gathers, assignment rows
src/calibrate.cpp        free-VRAM query + the conservative analytic K bound
src/runtime.cpp          driver-API helpers, cubin registry, launches, device buffers
src/gpap_internal.hpp    shared internals (not installed)
src/generated/           GENERATED cubins + lookup table (build_kernels.py output; gitignored)
kernels/helpers.py       support kernels (Triton, so the toolchain stays nvcc-free)
tools/build_kernels.py   the AOT compiler
tools/make_libcuda_stub.py  link-only libcuda stub, so CI can build without a driver
                            (`--print` lists every driver call the core makes: 18)
examples/demo.c          standalone C consumer (raw ABI)
examples/sample.cpp      solution sampling from C++ (gpap::Sampler)
examples/assumptions.cpp batched assumption checking from C++ (gpap::AssumptionChecker)
```

The core's contract is gated by [`tests/test_core.py`](../tests/test_core.py) against two
pure-Python oracles — the reference formula builder in `gpap/engine/formula.py` and the
reference BCP in `tests/reference_bcp.py` — plus direct clause-satisfaction checks on every
emitted model, in both kernel modes and both packed splits.
