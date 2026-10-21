<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)"  srcset="docs/assets/wordmark/gpap_wordmark_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/wordmark/gpap_wordmark_light.png">
  <img src="https://raw.githubusercontent.com/yogevshalmon/gpap/v0.3.0/docs/assets/wordmark/gpap_wordmark_light.png" width="590"
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

[**Docs**](docs/) · [**Sampling guide**](docs/SAMPLING.md) · [**Engine internals**](docs/ENGINE.md) ·
[**Python API**](docs/PYTHON_API.md) · [**C++ API**](cpp/README.md) · [**Examples**](examples/) ·
[**Contributing**](CONTRIBUTING.md) · [**Papers**](docs/papers/README.md)

</div>

---

## What it is

**An engine.** GPAP takes one formula and K partial assignments and propagates each assignment
to its own BCP fixpoint, all in a single launch, reporting whether it conflicted and, if not, what
it implied. A second entry point makes the decisions itself and runs a whole randomized
decide-and-propagate trajectory per row, again in one launch. The engine is application-agnostic;
everything else in this repository is built on it.

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

- **No clause scanning.** A clause update is one `atomic_add` on a packed word, and its return
  value flags units and conflicts ([how](#how-it-works)).
- **K private jobs, one shared formula.** Rows never communicate. A row that conflicts just stops,
  so there is nothing to serialize and the batch is as wide as the card's memory.
- **One library, two faces.** [`libgpap`](cpp/), with Python and C++ faces over one C ABI that
  draw identical models. It runs on the NVIDIA driver alone.

## How it works

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)"  srcset="docs/assets/animation/gpap_propagate_dark.gif">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/animation/gpap_propagate_light.gif">
  <img src="https://raw.githubusercontent.com/yogevshalmon/gpap/v0.3.0/docs/assets/animation/gpap_propagate_light.gif" width="675"
       alt="Animation of the engine's core idea, in three acts. Throughout, the assignment sits on the left as a green vine with a node per variable, named x1 to x6 above it: a gold leaf marks a variable decided at random, a green leaf one forced by propagation, a bare node one never touched, and a one-line key in the bottom-left corner says so. The vine runs level over its own nodes, steps down once the assignment ends, and carries on at that lower level to the bunch, which hangs from it by its own stem — so the two are one plant rather than two objects near each other, with the plant's one leaf sitting on the vine just before it. On the right is the formula, a bunch of grapes in which every row is a clause and every berry a literal. In the first act a real propagation trace runs, and the vine moves with it: a decision lights a berry gold and grows a gold leaf in the same frame, the literals that decision falsifies shrink and darken wherever they occur, and a clause left with one survivor forces it — a white flash and a green leaf — which falsifies more. It halts at a fixpoint with four clauses satisfied, one still open with two live literals, and two variables never touched. In the second act each clause row collapses onto a single cell, because under one assignment a clause's whole state is one packed word; the five stack into an even column, and that column unfolds into a row beside the vine. In the third act rows multiply downwards until the die holds five, each a vine with its own decisions and its own row of clause states, now plugged into the die's pins, all propagating in the same launch and each stopping at its own fixpoint while the row just watched holds the one it already reached, so no vine fills and no row ever lights up completely.">
</picture>
</div>

Every row reads the same formula; only a few bytes per clause belong to each row, and that
footprint decides how many rows fit on the card. So a clause is never scanned:

1. **Negated occurrence lists.** Setting `l` true visits exactly the clauses where `~l` appears.
   Binary clauses live there as implication edges, with no per-row state.
2. **One packed word per longer clause, per row:** how many of its literals are still alive, and
   the sum of their indices.
3. **The sum trick.** At count 1 the sum *is* the surviving literal, so one `atomic_add` updates
   the clause and its return value reveals a unit or a conflict.

The details are in [`docs/ENGINE.md`](docs/ENGINE.md), the kernels in
[`cpp/kernels/bcp.py`](cpp/kernels/bcp.py).

## Install

You need Linux (WSL2 works), an NVIDIA GPU of compute capability 8.0 or newer (Ampere or later)
with its driver, cmake, a C++17 compiler and Python 3.10 or later.

```bash
git clone https://github.com/yogevshalmon/gpap.git && cd gpap
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[build]"    # python side + triton (triton ships with torch on Linux/Colab)
scripts/build_core.sh        # the engine core, libgpap.so (cmake + a C++17 compiler; no CUDA toolkit)
pip install -e ".[dev]"      # + pytest, ruff, python-sat
gpap info                    # device, kernels, encoder, power-limit warning
```

The kernels are compiled for one GPU architecture: rebuild when you move to another card
generation (the loader names the mismatch). To skip the toolchain, run GPAP in Docker:
[docs/DOCKER.md](docs/DOCKER.md).

A first run, on a circuit that ships with the repository (the demo below):

```bash
gpap sample benchmarks/circuits/bench/seq/b10.bench --cycles 25 -n 50000 --out stim.out
```

> [!IMPORTANT]
> An NVIDIA GPU is required. There is no CPU path. This is a design decision, not a gap: the
> whole point is the batch.

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
  <img src="https://raw.githubusercontent.com/yogevshalmon/gpap/v0.3.0/docs/assets/demo/gpap_demo.gif" width="812"
       alt="Terminal recording: gpap sample runs on b10.bench unrolled 25 cycles, encodes it to 7,851 variables and 19,682 clauses, mines 86 units that pay their way, and writes 50,000 input stimuli in under seven seconds at about 20,000 samples per second">
</picture>
</div>

The library makes no uniformity claim. Testing one needs the whole solution set or conditional
sampling, so it lives in the evaluation layer ([below](#evaluation)). Every option, its default and
how to read a run's statistics: [`docs/SAMPLING.md`](docs/SAMPLING.md).

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

## Contributing

The tests, the repository layout, the conventions and how to propose a change:
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Evaluation

The experiments live outside the library:

- [`eval/`](eval/): campaign drivers, rival harnesses and uniformity tests. Nothing in `gpap/`
  imports it. [`docs/EVAL_RUNBOOK.md`](docs/EVAL_RUNBOOK.md) runs a campaign of your own.
- [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md): how to measure GPAP, and compare it with another
  sampler, fairly.
- [`docs/papers/`](docs/papers/README.md): the papers that use GPAP, each with its evaluation,
  protocol and artifact.
- `gpap bench`: benchmark a glob of instances, optionally against CMSGen
  (`scripts/build_cmsgen.sh`).

## Citation and licence

To cite GPAP, use [`CITATION.cff`](CITATION.cff); GitHub's "Cite this repository" button reads
it. GPAP is under the MIT licence ([`LICENSE`](LICENSE)). Benchmark files and quoted upstream code
keep their own licences: see [`THIRD_PARTY.md`](THIRD_PARTY.md).

<div align="center">
<br>
<picture>
  <source media="(prefers-color-scheme: dark)"  srcset="docs/assets/animation/gpap_idle_wake_dark.gif">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/animation/gpap_idle_wake_light.gif">
  <img src="https://raw.githubusercontent.com/yogevshalmon/gpap/v0.3.0/docs/assets/icon_wake/gpap_icon_wake_256.png" width="96"
       alt="The GPAP chip mark, its nine cells blinking">
</picture>
</div>
