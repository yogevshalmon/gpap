<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)"  srcset="docs/assets/wordmark/gpap_wordmark_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/wordmark/gpap_wordmark_light.png">
  <img src="https://raw.githubusercontent.com/yogevshalmon/gpap/v0.1.0/docs/assets/wordmark/gpap_wordmark_light.png" width="590"
       alt="GPAP — the wordmark beside the logo, a 3x3 grid of purple cells inside a chip die with gold pins">
</picture>

**GPU-Parallel (partial) Assignment Propagation**

Boolean constraint propagation on the GPU: hand it a CNF formula and **thousands of independent
partial assignments**, and it propagates every one of them to its own fixpoint **in one kernel
launch**. A foundation for SAT on the GPU, and a library to build on.

[![CI](https://github.com/yogevshalmon/gpap/actions/workflows/ci.yml/badge.svg)](https://github.com/yogevshalmon/gpap/actions/workflows/ci.yml)
[![license MIT](https://img.shields.io/badge/license-MIT-8E5EFA?labelColor=2C125A)](LICENSE)
[![python 3.10+](https://img.shields.io/badge/python-3.10%2B-8E5EFA?labelColor=2C125A)](pyproject.toml)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-8E5EFA?labelColor=2C125A)](cpp/)
[![runtime: libcuda only](https://img.shields.io/badge/runtime-libcuda%20only-E4C470?labelColor=2C125A)](#install)

[**Docs**](docs/) · [**Engine internals**](docs/ENGINE.md) · [**Python API**](docs/PYTHON_API.md) ·
[**C++ API**](cpp/README.md) · [**Examples**](examples/)

</div>

---

## What it is

**An engine.** GPAP takes one formula and K partial assignments and propagates each assignment
to its own BCP fixpoint, all in a single launch. For each one it reports whether it conflicted
and, if not, what it implied. The engine also has a second entry point that makes decisions
itself: pick free variables, give them polarities, propagate, and repeat. That runs a whole
randomized decide-and-propagate trajectory per row, again in one launch.

Both entry points live in one compiled library, [`libgpap`](cpp/), with a Python face and a C++
face over the same core. The engine is application-agnostic. Anything that needs "propagate a
large number of assumption sets" can call it, and everything else in this repository does.

**Applications built on it today:**

- **[Solution sampling](#solution-sampling)** (`gpap sample`) draws random solutions of a CNF or a
  circuit. Every sample is a real solution.
- **[Entailed-clause mining](#entailed-clause-mining)** (`gpap mine`) proves clauses the formula
  already implies, so that later propagation becomes stronger.
- **[Batched assumption checking](#batched-assumption-checking)** (`AssumptionChecker`,
  `LiteralProbe`) answers "which of these sets conflict, and what do they force?" for any number
  of sets.

More applications are planned for future work — stay tuned!

**Design at a glance:**

|  |  |
|---|---|
| **No clause scanning** | Negated occurrence lists, one packed 32-bit word per long clause, and the sum trick. The whole per-literal clause update is a single `atomic_add`, and its return value detects units and conflicts. |
| **K private jobs, one shared formula** | Each row owns its assignment, trail and clause words, and rows never communicate. A row that conflicts just stops: there is no conflict analysis to serialize. |
| **It sizes itself** | The batch is an analytic bound over the card's free VRAM. The kernel choice, trail bound and packed layout are all derived from the formula. |
| **One library, two faces** | Python (ctypes) and C++ (`#include <gpap/gpap.hpp>`) are thin marshalling layers over the same C ABI. At one seed, config and batch they draw identical models. |
| **Runs on a driver alone** | The kernels are Triton, compiled ahead of time into the library. At run time `libgpap.so` links `libcuda.so.1` and nothing else: no CUDA toolkit, no nvcc, no Triton, and no torch for the C++ face. |
| **Checked against a reference** | A BCP fixpoint is unique, so on every row that does not conflict both kernels must match a pure-Python reference propagator exactly. The test suite enforces this. |

## How it works

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)"  srcset="docs/assets/animation/gpap_propagate_dark.gif">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/animation/gpap_propagate_light.gif">
  <img src="https://raw.githubusercontent.com/yogevshalmon/gpap/v0.1.0/docs/assets/animation/gpap_propagate_light.gif" width="675"
       alt="Animation of the engine's core idea, in three acts. Throughout, the assignment sits on the left as a green vine with a node per variable: a gold leaf marks a variable decided at random, a green leaf one forced by propagation, a bare node one never touched. The vine runs level over its own nodes, steps down once the assignment ends, and carries on at that lower level to the bunch, which hangs from it by its own stem — so the two are one plant rather than two objects near each other, with the plant's one leaf sitting on the vine just before it. On the right is the formula, a bunch of grapes in which every row is a clause and every berry a literal. In the first act a real propagation trace runs, and the vine moves with it: a decision lights a berry gold and grows a gold leaf in the same frame, the literals that decision falsifies shrink and darken wherever they occur, and a clause left with one survivor forces it — a white flash and a green leaf — which falsifies more. It halts at a fixpoint with four clauses satisfied, one still open with two live literals, and most of the formula never touched. In the second act each clause row collapses onto a single cell, because under one assignment a clause's whole state is one packed word; the five stack into an even column, and that column unfolds into a row beside the vine. In the third act rows multiply downwards until the die holds five, each a vine with its own decisions and its own row of clause states, now plugged into the die's pins, all propagating in the same launch and each stopping at its own fixpoint while the row just watched holds the one it already reached, so no vine fills and no row ever lights up completely.">
</picture>
</div>

**Bytes per clause decide everything.** The formula is built once and every row reads it. What
propagation changes belongs to each row alone, so the per-row footprint decides how many
assignments fit on the card at once. That is why each clause's per-row bookkeeping is a counter
and a key, updated **in place**. Two watched literals per clause would be smaller still, but each
update *moves* which literals are watched. That kind of irregular pointer work is exactly what a
GPU does worst.

Three choices mean a clause is never scanned:

1. **Negated occurrence lists** (CSR). `occ[l]` holds the clauses where `~l` appears. Setting `l`
   TRUE falsifies `~l` in exactly those clauses and in no others. Binary clauses exist only here,
   as implication edges with no per-row state. Unit clauses are applied once, at construction.
2. **One packed word per clause of length ≥ 3**, batched over every row. The low bits count the
   non-falsified literals (`0` means conflict, `1` means unit). The high bits hold the sum of
   those literals' indices.
3. **The sum trick.** At count 1, the sum *is* the index of the surviving literal, so no scan or
   gather is needed. Each clause update is one `atomic_add`, and its return value detects units
   and conflicts without races.

Literal indexing is `+v → 2(v-1)` and `-v → 2(v-1)+1`, with negation `l ^ 1`. The assignment is a
bitmap in which `l` and `~l` share a word, so one `atomic_or` answers both "already set?" and
"contradiction?". Two propagate kernels, scalar and warp-wide, are chosen per formula. The
details are in [`docs/ENGINE.md`](docs/ENGINE.md), and the kernels are in
[`cpp/kernels/bcp.py`](cpp/kernels/bcp.py).

## Install

```bash
pip install -e ".[build]"    # python side + triton (triton ships with torch on Linux/Colab)
scripts/build_core.sh        # the engine core, libgpap.so (cmake + a C++17 compiler; no CUDA toolkit)
pip install -e ".[dev]"      # + pytest, ruff, python-sat
gpap info                    # device, kernels, encoder, power-limit warning
```

`scripts/build_core.sh` compiles the Triton kernels ahead of time and builds the library. It
needs cmake, a C++17 compiler and the NVIDIA driver, plus triton at build time only. The compiled
kernels target one GPU architecture, so rebuild when you move to a different card generation. If
you forget, the loader names the mismatch.

> [!IMPORTANT]
> A CUDA GPU is required. There is no CPU path. This is a design decision, not a gap: the whole
> point is the batch.

## Quickstart: Python

```python
# The engine: propagate many assumption sets at once (DIMACS literals)
from gpap.engine import AssumptionChecker
r = AssumptionChecker("formula.cnf").check_assumptions([[1, -2, 3], [-1, 2, -3], [4]])
r["conflict"]                 # (n,) bool: did propagation refute the set?
r["num_propagated"]           # (n,) how many literals each set implied

# Sampling: solutions of a CNF, projected on its support (`c ind` lines if present)
import gpap
samples, info = gpap.sample("formula.cnf", n=50_000)
samples.shape                 # (50000, S) uint8, one column per support variable
info["cols"]                  # the DIMACS variable each column stands for

# ...or of a sequential circuit: unroll T cycles, pin the outputs, sample the inputs
samples, info = gpap.sample_circuit("b10.bench", cycles=25, n=50_000)
info["wire_map"]              # column -> "input@cycle"
```

Pass any number of assumption sets. They are split into GPU-sized launches internally and returned
in input order. [`docs/PYTHON_API.md`](docs/PYTHON_API.md) maps the API tiers, and
[`examples/`](examples/) has runnable scripts for each application.

## Quickstart: CLI

```bash
gpap sample formula.cnf -n 50000 --out samples.out --stats run.json
gpap sample b10.bench --cycles 25 -n 50000 --out stim.out   # unroll, sample, write input stimuli
gpap sample design.aig --cycles 25 -n 50000                 # AIGER input (.aig/.aag)
gpap mine formula.cnf --width 1 2 --pool all --seconds 10 --out stronger.cnf
gpap circuit2cnf benches/ cnf_out/ --cycles 25 --pin-outputs 1
```

## Quickstart: C++

The same library, as a C ABI (`gpap/gpap.h`) plus a header-only C++ layer (`gpap/gpap.hpp`).

```cpp
#include <gpap/gpap.hpp>

gpap::AssumptionChecker checker("formula.cnf");            // the engine
gpap::CheckResults r = checker.check({{1, -2, 3}, {-1, 2, -3}});
bool refuted = r.conflict(0);
gpap::Value x3 = r.value(1, 3);                            // True / False / Free after BCP

gpap::Sampler sampler("formula.cnf");                      // an application
gpap::Solutions sols = sampler.sample(100000);             // 100k satisfying assignments
```

```cmake
find_package(gpap REQUIRED)
target_link_libraries(my_solver PRIVATE gpap::gpap)
```

`gpap::Engine` exposes both entry points directly, for building a loop of your own. Examples are
in [`cpp/examples/`](cpp/examples/), and the full API is in [`cpp/README.md`](cpp/README.md).

## Applications

### Solution sampling

Each row of a launch runs one randomized decide-and-propagate trajectory. A trajectory either
reaches a full assignment, which is a model by construction, or conflicts and is discarded.
Validity is structural, not checked afterwards, so there is no gradient, no rounding and no repair
pass. If you want the check anyway, `--verify-sound` re-checks every emitted row on the GPU. The
sampler handles plain projected CNF and the primary-input support of unrolled sequential and
combinational circuits.

<div align="center">
<picture>
  <source srcset="docs/assets/demo/gpap_demo.gif">
  <img src="https://raw.githubusercontent.com/yogevshalmon/gpap/v0.1.0/docs/assets/demo/gpap_demo.gif" width="812"
       alt="Terminal recording: gpap sample runs on b10.bench unrolled 25 cycles, encodes it to 7,851 variables and 19,682 clauses, mines 86 units that pay their way, and writes 50,000 input stimuli in under seven seconds at about 20,000 samples per second">
</picture>
</div>

The library makes no uniformity claim. Testing one needs the whole solution set or conditional
sampling, so it lives in the evaluation layer ([below](#evaluation)).

<details>
<summary><b>Circuits: inputs, unrolling and targets</b></summary>

<br>

`.bench` (ISCAS-85/89, ITC'99) and AIGER `.aig`/`.aag` are parsed natively. Sequential circuits are
unrolled `--cycles` times from the reset state. The last-cycle outputs are pinned to a reachable
pattern found by simulation, and the primary inputs across all cycles become the sampling support
(`wire@cycle` columns). `--target rare` pins the rarest output pattern seen in a large simulation
batch instead of a random one. `--sim-check` replays every emitted sample through the original
circuit, which checks soundness independently of the CNF encoding.

`--encoder abc` unrolls and logic-optimizes the circuit with
[berkeley-abc](https://github.com/berkeley-abc/abc), then Tseitin-encodes the resulting AIG. This
gives a smaller CNF with the same projected solution space (asserted by
`tests/circuits/test_abc_encoder.py`). `scripts/build_abc.sh` builds a pinned upstream commit, and
`--abc-opt full|fast|auto` picks the optimization effort. The default encoder stays `tseitin`, so
a result never depends on whether ABC is installed.

</details>

<details>
<summary><b>Output</b></summary>

<br>

`--out FILE` writes support projections, one bit per support variable
(`--out-format bits|npy|dimacs`). The rows are collected during the timed run itself. `--stats
FILE` writes the run's info JSON, with rates, phase timings, mining stats and the soundness
verdict. `--max-samples` (default 100k) caps the number of rows written.

</details>

### Entailed-clause mining

Assume a set of literals and propagate. If BCP hits a conflict, the formula entails the negation
of that set, which is a clause. `gpap mine` sweeps every k-subset of a literal pool under a work
budget and writes the clauses it proves. Width 1 gives the BCP-visible backbone. Every clause
found is entailed, so adding them leaves the solution set exactly unchanged and makes later
propagation stronger. The sampler runs mining as a preprocessing step and keeps the mined clauses
only if they pay for themselves. See [`examples/mine_clauses.py`](examples/mine_clauses.py).

### Batched assumption checking

This is the engine's first entry point, exposed directly. `AssumptionChecker` takes any number of
assumption sets and returns, for each one, whether it conflicts, how many literals it implied
and, on request, its full assignment. `LiteralProbe` checks every k-subset of a literal pool
within a budget; mining is built on it. See
[`examples/check_assumptions.py`](examples/check_assumptions.py).

## Layout

```
cpp/              # the engine core: libgpap, C ABI + C++ wrapper, shared by both faces
gpap/
  engine/         # the Python face of the core (ctypes): checker, probe, engine
  sampling/       # application: the sampler, its pipeline, preprocessing, output
  circuits/       # .bench / AIGER parse, unroll, Tseitin, pin -> projected CNF
  mining.py       # application: entailed-clause mining over the probe primitive
  cli.py          # `gpap` subcommands
examples/         # runnable Python examples, one per application
eval/             # the experiment layer (see Evaluation); nothing in gpap/ imports it
scripts/          # build libgpap, optional third-party tools, fetch benchmarks
docs/             # start at docs/README.md
tests/            # pytest, one directory per area (engine, sampling, mining, ...)
benchmarks/       # small committed .bench, AIGER and CNF instances
.gpap/            # built/fetched tools, benchmark sets and run scratch, gitignored
```

Code carries the rule and `docs/` carries the evidence. A comment says what the code does and why
it is correct. The measurement that settled a design decision lives in
[`docs/findings/`](docs/findings/), not in a source file.

## Tests

```bash
pytest                             # the engine, the applications and repo checks (GPU tests skip off-CUDA)
pytest -m "not gpu and not slow"   # portable smoke suite, no GPU needed
pytest tests/experiments                  # the experiment layer's tests, not part of the default run
```

## Evaluation

The experiments for the paper live outside the library:

- [`eval/`](eval/): campaign drivers, rival harnesses and uniformity tests. Nothing in `gpap/`
  imports it.
- [`ARTIFACT.md`](ARTIFACT.md): reproducing the paper's tables and figures.
- [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md): the measurement protocol.
- `gpap bench`: benchmark a glob of instances, optionally against CMSGen
  (`scripts/build_cmsgen.sh`).

## Citation and license

[`CITATION.cff`](CITATION.cff). MIT — see [`LICENSE`](LICENSE).

<div align="center">
<br>
<picture>
  <source media="(prefers-color-scheme: dark)"  srcset="docs/assets/animation/gpap_idle_wake_dark.gif">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/animation/gpap_idle_wake_light.gif">
  <img src="https://raw.githubusercontent.com/yogevshalmon/gpap/v0.1.0/docs/assets/icon_wake/gpap_icon_wake_256.png" width="96"
       alt="The GPAP chip mark, its nine cells blinking">
</picture>
</div>
