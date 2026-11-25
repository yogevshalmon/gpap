# Contributing to GPAP

> **Reader:** anyone changing GPAP, people and coding agents alike — how to build and test, where
> each kind of fact belongs, and the rules that keep the engine and its published claims correct.

## Build and test

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[build,dev]"      # triton to build the kernels; pytest, ruff, python-sat
scripts/build_core.sh              # builds libgpap; needs cmake and a C++17 compiler
gpap info                          # confirms the library loads and sees the GPU

pytest -m "not gpu and not slow"   # host-only checks, no GPU needed (this is what CI runs)
pytest                             # the default suite: engine, applications, repository checks
pytest tests/experiments           # the evaluation layer's tests, run separately
ruff check .
```

No CUDA toolkit is needed: the kernels are written in Triton and compiled ahead of time into the
library, which opens the driver's `libcuda` at run time. There is no CPU execution path on CUDA.
A second backend runs part of the engine through OpenCL (`GPAP_BACKEND=opencl`,
[docs/OPENCL.md](docs/OPENCL.md)), on any OpenCL device and without the NVIDIA driver; it needs
nothing at build time, and `scripts/setup_opencl.sh` installs a runtime where the system has
none.
Optional tools build from pinned sources: `scripts/build_abc.sh` (the ABC circuit encoder),
`scripts/build_cmsgen.sh` (CMSGen, for comparisons) and `scripts/fetch_benchmarks.sh`.

## Layout

```
cpp/                     the engine: libgpap, a C ABI and a header-only C++ wrapper
  include/gpap/gpap.h    the C ABI: formula, engine, selections, readout, counters, sizing
  include/gpap/gpap.hpp  the C++ wrapper: Sampler / AssumptionChecker over Formula / Engine
  src/                   formula builder, engine state, readout, batch sizing and kernel
                         dispatch, the sampling loop, probing, batched checking
  src/gpu_api.hpp        the GPU socket: every device operation, one plug per GPU library
                         (gpu_cuda.cpp, gpu_opencl.cpp)
  kernels/bcp.py         the BCP kernels, in Triton (the reference)
  kernels/bcp.cl         the fixpoint and support kernels in OpenCL C; follows bcp.py
  tools/build_kernels.py compiles the Triton kernels into the library ahead of time
gpap/                    the Python package
  engine/                the Python face of the core (ctypes): BCPEngine, AssumptionChecker,
                         LiteralProbe, batch sizing
  sampling/              the sampler: pipeline, portfolios, preprocessing, output writers
  circuits/              .bench / AIGER parsing, simulation, unrolling, encoders
  mining.py              entailed-clause mining, built on LiteralProbe
  verdict.py             the soundness-verdict readers, importable without torch
  cli.py, benchmark.py   the `gpap` command and the `gpap bench` suite driver
examples/                runnable Python examples (C++ examples are in cpp/examples/)
eval/                    the evaluation layer: campaign drivers, rival harnesses, uniformity
                         tests, the paper's table generator and artifact tooling
scripts/                 build and fetch scripts
benchmarks/              small committed instances
docs/                    guides, design, measurement protocol, findings; papers/ holds one
                         folder per paper
tests/                   pytest; tests/experiments covers eval/, tests/reference holds the
                         pure-Python oracles the kernels are checked against
```

Three directories are never committed, and only `gpap/paths.py` and `eval/paths.py` name them
(shell scripts ask those modules):
- `.gpap/` — built tools, fetched sources and benchmark sets, scratch (`$GPAP_HOME` moves it);
- `.experiments/` — a run's output while it is being produced;
- `$GPAP_RESULTS` (default `../gpap-results`) — where finished campaigns are kept.

## Where things go

Each kind of fact has one home; everything else links to it.

| kind of fact | its one home |
|---|---|
| what the code does, and its invariants | the code and its comments |
| why we decided it, with the measurement | `docs/findings/` (dated) |
| how to use it | the guides and references in `docs/`, `cpp/README.md` |
| a paper's claims and numbers | the paper and the files generated from the data; never retyped |
| a paper's evaluation, protocol and artifact guide | its folder, `docs/papers/<venue-yy>/` |
| how to work in the repo | this file |
| open questions and planned work | GitHub issues |

## Conventions

- **Code carries the rule; `docs/` carries the evidence.** A comment says what the code does and
  why it is correct, and names the one refactor that would break it. The measurement that settled
  it (date, GPU, numbers) goes in `docs/findings/`. Source files carry no dated numbers and no
  GPU names.
- **Every doc names its reader** in a `> **Reader:**` line under its title. Guides are timeless:
  they state rules and point to the evidence instead of quoting it.
- **New measurements go in the findings**, dated, or in files generated from data; a guide
  links to them rather than copying a number. (A few older guides still quote illustrative
  figures.)
- **Findings** open with a status block (`Status`, `Measured`, and `Rule lives in` or
  `Record only`). Code cites their sections by number (`UNIFORMITY_FINDINGS.md §4b`), so sections are
  appended, never renumbered. A new measurement annotates the old one and adds an entry to
  `docs/findings/INSIGHTS.md`.
- **A test's docstring says what it proves**, and a new check must be seen to fail before it
  passes. `tests/repo/test_docs_hygiene.py` keeps doc links, section citations and status blocks
  honest.
- **Check comment-only edits mechanically**: compare the AST with docstrings removed (Python) or
  the comment-free token stream (C/C++) against the previous version.

## Invariants that break silently

The engine:
- On every options struct, `{0}` and a NULL pointer mean the same thing: each field's zero is its
  default, and a field whose useful default is "on" is named for the opt-out (`same_var`).
- An ABI break moves the soname (`cpp/CMakeLists.txt`); an additive change does not.
- Key any cache of results on `gpap_build_id()`, never on `gpap_version()`.
- A kernel binary the kernel cannot tell apart is not shipped: `build_kernels.vscan_is_read` and
  `engine.cpp::traj_key` are one rule (`tests/engine/test_kernel_table.py`). Kernel binaries embed
  line info, so compare their executable sections, not their bytes.
- Kernel dispatch is decided once, in the core (`gpap_dispatch_wave`), from the clause count —
  never from the batch alone. The queue is the one exception: it runs WAVE.
- `gpap_trail_cap` is the only definition of the trail bound; keep its slack in step with `WAVE_W`.
- Every device buffer the engine allocates is a `DevBuf`, which counts it on the ledger
  (`cpp/src/memory.hpp`) as formula or rows; a buffer a trim keeps is a `FormulaBuf`. A buffer
  that goes round `DevBuf` is invisible to the memory limit.
- Under a memory limit, a call's sizing counts everything the call allocates afterwards
  (`exact_terms` in `cpp/src/engine.cpp`). A new buffer goes there, or
  `tests/engine/test_mem_limit.py` fails under a tight limit. There a smaller batch refits the
  row state to itself (`ensure_state`'s `fit`), and only a caller that re-initializes every row
  afterwards may ask for that: a refit under `apply` or a trajectory loses the rows.
- The batch comes from the memory model (`gpap_row_bytes`, `cpp/src/calibrate.cpp`) and the batch
  rule (`size_rows`, `cpp/src/memory.cpp`), never from a runtime memory probe. Any change to that model changes the batch: re-validate it as
  [BENCHMARKING.md](docs/BENCHMARKING.md) §2 describes, and re-check dispatch.
- `import gpap` must set `PYTORCH_CUDA_ALLOC_CONF` before torch initializes CUDA.
- A CLEAN-mode launch restores the words it touched, so anything that reads state after the
  launch (assignments, satisfied-clause readout, selections, `recache_flush`) uses plain `apply`.
- A queued launch leaves every row at the baseline, so only a sweep that needs a flag or a count
  per set runs on it. It runs WAVE unless a kernel is pinned. A queued call copies the bitmap at
  its start and trusts one claim, `packed_clean_rows`, for the packed words, so every engine path
  that dirties a row drops it; never add a second claim ([ENGINE.md](docs/ENGINE.md) §6b,
  `tests/engine/test_queue.py`).
- `max_tuples` is a work budget. A per-width budget must give width 1 an explicit `0`, or the
  fallback silently caps the backbone (`pipeline.STAGE1_MAX_TUPLES`).
- Application code reads results through the selection API; the tensor accessors are copies for
  tests.
- The Python and C++ faces share their defaults as well as their loop
  (`tests/cli/test_cli_contracts.py`).
- The engine names no GPU library. Every device operation goes through `cpp/src/gpu_api.hpp`,
  and only `gpu_cuda.cpp` includes `cuda.h`.
- The Triton kernels are the reference and `cpp/kernels/bcp.cl` follows them: a change of
  behaviour in a kernel both implement lands in both, in one change
  ([OPENCL.md](docs/OPENCL.md), `tests/engine/test_opencl_backend.py`). A kernel the OpenCL plug
  does not list is refused by name, never replaced by another.

Sampling and measurement:
- The batch is not part of the seed. Before comparing counts or rates across runs or machines,
  pin the batch and a single configuration (`--portfolio b8|b1|s1`).
- The no-change gate for a refactor: a fixed batch, one configuration and one seed (the CLI
  default is 5) must give bit-identical model counts ([BENCHMARKING.md](docs/BENCHMARKING.md) §3).
  Rates alone are noisy.
- Optional self-checks bill their time to `phases_diag`, never to `phases`, so no rate depends on
  them.
- The `soft` configuration stays opt-in (`--allow-soft`): it fails the uniformity test. Claims
  about uniformity come only from Barbarik with its controls (`eval/uniformity/`).
- Directions that were measured and lost are listed in `docs/findings/INSIGHTS.md`; do not retry
  one without new evidence.

The evaluation:
- The TACAS 2027 submission's evaluation protocol is frozen
  ([PROTOCOL.md](docs/papers/tacas-27/PROTOCOL.md)). A change may fix a bug or the prose, never a
  metric, configuration, screen, budget or set of cells.
  Proposals are logged in [RIVAL_FAIRNESS_FINDINGS.md](docs/findings/RIVAL_FAIRNESS_FINDINGS.md)
  §17.
- The rules that decide which cells a table counts live once, in
  `eval/artifact/paper_tables.py`; every table, figure and gate imports them. One row per
  encoder: never report the better of two encoders.

## Definition of done

- The code, its test and the one doc that owns the rule change together.
- `pytest -m "not gpu and not slow"` passes, and `pytest` on a machine with a GPU. Run
  `pytest tests/experiments` if you touched `eval/`, and `GPAP_BACKEND=opencl pytest` if you
  touched a kernel, the GPU socket or a plug.
- `ruff check .` is clean.
- A performance claim comes with a measurement in `docs/findings/`;
  [BENCHMARKING.md](docs/BENCHMARKING.md) says how to take one.

## Proposing a change

Small fixes can go straight to a pull request. Open an issue first for anything that changes the
C ABI, a default, or the evaluation protocol.
