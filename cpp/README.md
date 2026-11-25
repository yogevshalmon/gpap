# libgpap — the GPAP engine as a C/C++ library

> **Reader:** C and C++ users of libgpap — the entry points, the two API tiers, and how to build and link against it.

The K-parallel packed BCP engine (`docs/ENGINE.md`) as a shared library with a C ABI, linkable from
C++ and driven from Python. [`kernels/bcp.py`](kernels/bcp.py) is the reference, AOT-compiled
into embedded cubins (bit-identical at SASS level to the Triton JIT's); the OpenCL backend's
[`kernels/bcp.cl`](kernels/bcp.cl) follows it.

## Build

Requirements are deliberately thin — no CUDA toolkit, no nvcc: a C++17 compiler + CMake for the
host library; `cuda.h`, from the toolkit if present and else from the triton wheel; and Python +
triton at build time only, for the AOT kernel compile. The NVIDIA driver is needed only to run
the `cuda` backend: libgpap opens `libcuda.so.1` at run time, so it builds and loads without it.

```bash
scripts/build_core.sh                    # both steps below, with diagnostics
# ...which is just:
python cpp/tools/build_kernels.py        # kernels/{bcp,helpers}.py -> src/generated/*.c
cmake -S cpp -B build && cmake --build build -j

./build/gpap_c_api                       # both entry points, plain C
./build/gpap_sample formula.cnf 16384    # sample solutions (C++ wrapper)
```

From another CMake project:

```cmake
find_package(gpap REQUIRED)
target_link_libraries(my_solver PRIVATE gpap::gpap)
```

`build_kernels.py` targets the local GPU; `--arch 89` cross-compiles (no GPU needed). Rerun it
whenever a kernel changes — the only step that regenerates GPU code.

The library is 0.4.0 / SOVERSION 1: an ABI break must move the soname, an additive change must not
(0.3.0 added `gpap_sampler_*` and 0.4.0 the backend queries, changing no existing symbol).

## Backends

The engine reaches the GPU through one internal interface, [`src/gpu_api.hpp`](src/gpu_api.hpp),
with a plug per GPU library. `GPAP_BACKEND` chooses when an engine is created:

- `cuda` (the default): the driver API and the Triton kernels above. The reference and the fast
  path; everything in this file.
- `opencl`: [`kernels/bcp.cl`](kernels/bcp.cl), compiled on the device at first use. It covers
  batched checking, probing and the readouts, not sampling. Nothing is needed at build time; at
  run time it opens `libOpenCL.so.1`, on any OpenCL device, with or without the NVIDIA driver.
  See [docs/OPENCL.md](../docs/OPENCL.md).

`gpap_backend()`, `gpap_engine_backend(e)` and `gpap_engine_device_name(e)` say which is in use.

## Tier 1 — the whole API for the normal case

`<gpap/gpap.hpp>` is header-only and has two tiers. Tier 1 needs no knowledge of the engine:

```cpp
#include <gpap/gpap.hpp>

gpap::Sampler sampler("formula.cnf");
gpap::Solutions sols = sampler.sample(100000);       // draw 100k satisfying assignments
bool value = sols(i, j);                             // of variable sols.variables()[j]
```

The sampler sizes the GPU batch to free VRAM, picks the propagate kernel, orders decisions so
propagation does the work, and chunks the target; literals are DIMACS. `SampleOptions` carries the
rest (`seed`, `time_limit_s`, `max_chunks`, `max_keep`,
`variables`, `portfolio`, `var_order`, `batch`, `lookahead`, `retry_with_lookahead`, `device`, `k`,
`k_cap`, `verify`, `min_measure_chunks`).

The loop itself lives in the library (`gpap_sampler_*`, `src/sample.cpp`), shared verbatim with the
Python face, so at one seed and a single config both faces find the *same* models (gated by
`tests/sampling/test_face_parity.py`). `verify` re-checks on device that every returned row satisfies
every clause; it is a self-test, off by default on every face, and `Sampler::sample` throws if a row
fails.

Two levers matter when a formula yields nothing. `portfolio = gpap::default_portfolio()` races the
three uniformity-safe configs (random B=8 / random B=1 / structured B=1) as a throughput bandit and
returns only the winner's rows. `retry_with_lookahead` flips the in-kernel risk polarity after a
zero-yield window (b12 unrolled 50 cycles: 0 solutions against ~1100/s), but is off by default
because that polarity is not uniform: Barbarik rejects it on 64 of the 65 circuit instances it can
sample, so it is opt-in here as `--allow-soft` is in Python.

The other entry point tests many assumption sets in one launch:

```cpp
gpap::AssumptionChecker checker("formula.cnf");
auto r = checker.check(sets);                        // ANY number of sets, propagated independently
r.conflict(1);                                       // did set 1 falsify a clause?
auto a = checker.check(sets, /*assignments=*/true);  // also what each set implies:
a.value(0, 5);                                       // True / False / Free for x5 under set 0
```

`sets` may be any length: launches are sized to free VRAM and the formula's shape, and results come
back in input order. Only the conflict flags and implication counts come back unless you pass
`assignments=true`, as in the Python face: the per-row assignment is `n x 2N` bytes, which
chunking cannot shrink.

To cap the GPU memory GPAP holds, set a limit before the first engine; every face sizes its
batches under it ([docs/ENGINE.md](../docs/ENGINE.md) §7):

```cpp
gpap::set_mem_limit("6G");                           // or "75%", "none"; or GPAP_MEM_LIMIT=6G
gpap::Memory m = gpap::memory();                     // m.limit, m.held, m.peak on device 0
```

A batch that fits no row under it throws `gpap::Error` (`GPAP_ERR_MEMORY`), saying what a row
needs. The C ABI is `gpap_set_mem_limit`, `gpap_mem_limit`, `gpap_mem_held`, `gpap_mem_peak`;
per engine, `gpap_engine_mem_held` by kind, `gpap_engine_batch_bound` and `gpap_engine_trim`.

## Tier 2 — driving the engine yourself

`gpap::Formula` and `gpap::Engine` are a thin RAII/exception wrapper over the C ABI, for callers owning
the chunk loop, decision order or readout; tier 1 is written in terms of them, and
speaks the ABI's 0-based variable indices rather than DIMACS.

```cpp
gpap::Formula formula = gpap::Formula::from_dimacs("formula.cnf");
gpap::Engine  engine(formula);                       // throws gpap::Error on failure

engine.reset(65536);                                 // 65k independent trajectories
engine.trajectory(order, n_support, /*B=*/8, seed);  // order: support variables FIRST
int models = engine.select_complete(engine.pair_mask());
auto bits  = engine.var_bits(formula.support(), models);   // (models x |support|) 0/1
```

Decision order matters (tier 1 does this for you). Put the independent support
(`formula.support()`, from the file's `c ind` lines) first and pass its size as `n_support`, so
phase 1 reaches a BCP fixpoint over the support before any completion variable is decided. Without
that split, decisions go to variables propagation would have determined and success collapses — on
b03-c25, 9% against 0%.

From plain C the same sequence is [`examples/c_api.c`](examples/c_api.c), where
`gpap_engine_create(f, NULL, &e)` with NULL opts means every default: device 0, AUTO kernel, split
from the formula.

Batched checking also takes its sets flat, in DIMACS, as the C ABI and Python's `check_sets` do:
tier 1's `check` runs on it, and a caller that already holds flat buffers skips the copy.

```cpp
// set i is lits[offsets[i] .. offsets[i+1]); offsets holds n + 1 entries
gpap::CheckResults r = engine.check_sets(lits, offsets);   // flags and implication counts
auto a = engine.check_sets(lits, offsets, /*assignments=*/true, /*sat_count=*/true);
a.sat_count(0, 4);                       // surviving literals of clause index 4 (0-based)
```

The first call, which asks for flags and counts only, runs on the engine's queue when it has
more sets than the queue has rows: a few rows, each taking set after set
([`docs/ENGINE.md`](../docs/ENGINE.md) §6b). The second reads each set's row afterwards, so it
runs one row per set. Probing (`gpap::Prober`) runs on the queue too.

## Getting answers out: the selection API

`gpap::Sampler` does all of this for you; read it if you drive `gpap::Engine`. A chunk ends with K
rows, often 100,000+; shipping a K-long yes/no array back per chunk is a PCIe round-trip at ~100
chunks/s, so the host never sees per-row data: a count crosses the bus, then only the bits you asked
for. Variable `v` owns two adjacent bits of a `(K, ceil(2N/32))` int32 array — `2v` = TRUE, `2v+1`
= FALSE, both 0 = free — sharing one word, since 32 is even.

Worked example: N = 5, K = 6, support {x1, x2}, x3–x5 internal. Rows 1 and 4 conflicted; rows 0 and
3 are fully assigned; rows 2 and 5 have an internal variable still free. So *fully assigned* is
{0, 3} and *support complete* is {0, 2, 3, 5}, and the sampler needs both at once: it emits the
first and measures the support's conflict-free rate with the second. Hence slots.

```cpp
std::int32_t mask_all = engine.pair_mask();          // empty = every variable
std::int32_t mask_sup = engine.pair_mask({0, 1});    // 0-based variable indices: x1, x2

std::int32_t n = engine.select_complete(mask_all, /*slot=*/0);  // -> 2, shelf 0 = {0, 3}
engine.select_complete(mask_sup, /*slot=*/1);                   // -> 4, shelf 1 = {0, 2, 3, 5}
engine.check_sound(/*slot=*/0);                                 // -> 0
engine.var_bits(/*cols=*/{0, 1}, n, /*slot=*/0);                // x1, x2 of rows 0 and 3
```

A pair mask sets the even bit of every variable it asks about; `~(w | (w >> 1)) & mask` then has
bit `2v` set iff `v` is free, so `!= 0` means "still incomplete" — row 2 gives 0 against `mask_sup`,
non-zero against `mask_all`. The mask is what makes the test correct rather than configurable:
padding bits belong to variables that do not exist, are always 0, and so read as free. Masks cannot
be freed — make one per variable set, keep the id.

`select_complete` runs that test over all K rows in one kernel, leaves the selection in a slot on
the device and returns only the count. `GPAP_SLOTS` = 3: slots 0 and 1 are yours, for the two
answers above; 2 is scratch. `selection_rows(slot, count)` is the first call that compacts anything,
and its capacity argument is a real bound, writing `min(actual selection, capacity)`.
`selection_set(slot, rows)` puts a row list you chose on a shelf — how `AssumptionChecker` reads all
K rows, conflicted included.

`check_sound(slot, max_rows = 0)` is the independent safety net: at a fully assigned row, "no clause
has zero surviving literals" is equivalent to "every clause has a true literal". It must return 0;
`0` covers every selected row, a positive bound only a prefix, of which alone "0 unsound" then
speaks. `var_bits(cols, count, slot)` gathers the named columns of that shelf's rows out of the
bitmap — 4 bytes here — and `assignment_rows(count, slot)` the full literal assignment, sliced to
stay bounded in N. Harvesting the chunk above costs sixteen bytes.

C names are these prefixed `gpap_`; implementation [src/readout.cpp](src/readout.cpp), over the
helper kernels in [kernels/helpers.py](kernels/helpers.py).

## Layout

`include/gpap/gpap.h` is the C ABI, the stable boundary and all a C consumer needs;
`include/gpap/gpap.hpp` is tier 1 over tier 2. Each rule has one home in `src/`: `formula.cpp`
(DIMACS, occurrence CSR, packed word), `engine.cpp` (state, the two entry points), `sample.cpp`
(the sampling loop), `check.cpp` (the checker's chunk loop), `probe.cpp` (paged k-subset probing),
`calibrate.cpp` (memory model, K sizing, wave/scalar dispatch), `readout.cpp` (selections),
`gpu_api.hpp` (the GPU socket), `gpu_cuda.cpp` (cubins and launches), `gpu_opencl.cpp` (the
OpenCL plug), `runtime.cpp` (errors, the backend choice, the build id). `kernels/helpers.py` holds
the support kernels in Triton, so the toolchain stays nvcc-free; `tools/build_kernels.py` is the
AOT compiler; `kernels/bcp.cl` is the OpenCL C source the OpenCL plug compiles.

Examples: [`examples/sample.cpp`](examples/sample.cpp) (`gpap::Sampler`),
[`examples/assumptions.cpp`](examples/assumptions.cpp) (`gpap::AssumptionChecker`),
[`examples/probe.cpp`](examples/probe.cpp) (`gpap::Prober`),
[`examples/c_api.c`](examples/c_api.c) (plain C, the raw ABI), and the Python ones in
[`../examples/`](../examples/). `gpap.engine._core` binds the same ABI with `ctypes`, taking the
library from `$GPAP_LIB`, then `$GPAP_BUILD_DIR` (what `scripts/build_core.sh` builds into),
then `<repo>/build/`, then `<repo>/cpp/build/`, then the copy installed beside the package.

The core's contract is gated by [`tests/engine/test_core.py`](../tests/engine/test_core.py) against two
independent pure-Python oracles ([`tests/reference/formula.py`](../tests/reference/formula.py),
[`tests/reference/bcp.py`](../tests/reference/bcp.py)) plus clause-satisfaction checks on every
emitted model, in both kernel modes and both packed splits — not `gpap/engine/formula.py`, a
facade over `src/formula.cpp`.

### Sanitizers

The library and examples build clean under ASan + UBSan, and the Python suite runs against such a
build. One option is required: the CUDA driver reserves address space colliding with ASan's shadow
gap, and without it `cuInit` fails with a misleading "out of memory".

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

Leak checking needs `leak:libcuda.so` suppressed: the WSL driver shim leaks ~50 KB in `cuInit` per
process, not ours.
