# GPAP — GPU-Parallel (partial) Assignment Propagation & CircuitSAT Sampling

GPAP is a GPU primitive for **Boolean constraint propagation over millions of independent partial
assignments at once**, packaged as a C++ library with Triton kernels compiled in ahead of time
and driven from Python or C++, and a flagship application built on it: **guaranteed-valid SAT
solution sampling** for CircuitSAT and projected sampling.

- **The engine** ([`gpap.engine`](gpap/engine/), [`cpp/`](cpp/)): K independent assumption sets
  are propagated to fixpoint in parallel with no clause scanning, via negated occurrence lists,
  one packed per-clause word and the sum trick (below). A fused kernel runs a whole randomized
  decide-and-propagate trajectory per row in one launch.
- **The sampler** ([`gpap.sampling`](gpap/sampling/)): GPU-parallel CMSGen. K randomized BCP
  completions per chunk; every emitted model satisfies the formula (no gradient, rounding or
  repair). It samples the primary-input support of unrolled sequential and combinational
  circuits, the setting of the recent GPU-sampling papers (DEMOTIC, Recurrent CircuitSAT).

## Install

```bash
pip install -e ".[build]"    # python side + triton (triton ships with torch on Linux/Colab)
scripts/build_core.sh        # the engine core, libgpap.so (cmake + a C++17 compiler; no CUDA toolkit)
pip install -e ".[dev]"      # + pytest, ruff, python-sat
```

`scripts/build_core.sh` compiles the Triton kernels ahead of time and builds the library; it
needs cmake, a C++17 compiler, the NVIDIA driver and, at build time only, triton. At run time
`libgpap.so` links `libcuda.so.1` alone. The cubins target one GPU architecture, so rebuild on a
different card generation (the loader says so by name). A GPU is required; there is no CPU path.
`gpap info` reports what is installed.

## Quickstart: Python

```python
import gpap

# A DIMACS CNF (support read from `c ind` header lines if present)
samples, info = gpap.sample("formula.cnf", n=50_000, seed=5)
samples.shape                 # (50000, S) uint8, one column per support variable
info["cols"]                  # the DIMACS variable each column stands for
print(info["samples_per_s"], info["success_rate"], info["sound_check"])

# A sequential circuit: unroll T cycles, pin outputs to a reachable pattern, sample inputs
samples, info = gpap.sample_circuit("b10.bench", cycles=25, n=50_000)
info["wire_map"]              # column -> "input@cycle"

# In-memory clauses
samples, info = gpap.sample_clauses(3, [[1, 2], [-1, 3], [-2, -3]], n=1_000, support=[0, 1, 2])

# The engine as a primitive: batched assumption checking
from gpap.engine import AssumptionChecker
r = AssumptionChecker("formula.cnf").check_assumptions([[1, -2, 3], [-1, 2, -3]])
print(r["conflict"])          # (n,) bool
```

Nothing above names a batch size: every entry point sizes itself to the GPU and the formula.
`samples` is a `(rows, S)` uint8 matrix of support projections, projected on the GPU and
collected off the timing path; a run cut short by its time limit returns what it produced
(`min(n, max_samples, delivered)` rows). `info` carries throughput (`samples_per_s` steady
state, `sps_e2e_total` including setup), `success_rate`, per-phase timings, mining stats and the
soundness verdict. [`docs/PYTHON_API.md`](docs/PYTHON_API.md) maps the API tiers.

## Quickstart: CLI

```bash
gpap info                                   # device, triton, abc encoder, power-limit warning
gpap sample formula.cnf -n 50000 --out samples.out --stats run.json
gpap sample b10.bench --cycles 25 -n 50000 --out stim.out   # unroll, sample, write input stimuli
gpap sample design.aig --cycles 25 -n 50000                 # AIGER input (.aig/.aag)
gpap sample b10.bench --cycles 25 --encoder abc             # smaller CNF via berkeley-abc
gpap circuit2cnf benches/ cnf_out/ --cycles 25 --pin-outputs 1
gpap bench 'benches/*.bench' --encoders tseitin abc:fast --cycles 25 --target rare --resume
```

**Circuit inputs.** `.bench` (ISCAS-85/89, ITC'99) and AIGER `.aig`/`.aag` are parsed natively.
Sequential circuits are unrolled `--cycles` times from the reset state, the last-cycle outputs
are pinned to a simulated reachable pattern, and the primary inputs across cycles become the
sampling support (`wire@cycle` columns). `--target rare` pins the rarest output pattern seen in
a large simulation batch instead of a random one, which is biased toward easy patterns.
`--sim-baseline` measures the CNF-free rival, random inputs plus bit-parallel simulation, and
reports its acceptance rate beside GPAP's; `--sim-check` simulates every emitted sample through
the original circuit as an encoder-independent soundness check.

**Circuit encoder.** `--encoder abc` unrolls and logic-optimizes the circuit with
[berkeley-abc](https://github.com/berkeley-abc/abc) and Tseitins the resulting AIG. The
projected solution space is identical to the native encoder's (asserted by
`tests/test_abc_encoder.py`); only internal variables change, and the clause count drops by
2-8x on the benches here. `scripts/build_abc.sh` builds a pinned upstream commit into `.gpap/bin`;
`--abc-opt full|fast|auto` picks the effort. The default stays `tseitin` so results never
change with ABC's presence.

**Writing solutions.** `--out FILE` writes support projections, one bit per support variable
(`--out-format bits|npy|dimacs`), collected by the timed run itself; `--stats FILE` writes the
run's info JSON. `--max-samples` (default 100k) is a ceiling.

**Comparing against CMSGen.** Build it with `scripts/build_cmsgen.sh` (pinned upstream plus
`scripts/cmsgen_gpap_eval.patch`, which adds `--nowrite` and `--statsevery`). GPAP's timed run
emits nothing, so `gpap bench` times CMSGen the same way and reports its emission cost apart.
[`docs/BENCHMARKING.md`](docs/BENCHMARKING.md).

**The full experimental sweep**, the rival harnesses and the figures:
[`docs/EVAL_RUNBOOK.md`](docs/EVAL_RUNBOOK.md).

```bash
eval/rivals/rivals_setup.sh     # abc + patched cmsgen + DEMOTIC / Recurrent-CircuitSAT / HTS
eval/run_full_eval.sh campaign  # the paper run; resumable
```

**Uniformity.** A uniformity number is a claim about a distribution, so `gpap sample` does not
make one. The tools that do live in [`eval/uniformity/`](eval/uniformity/): an exact-oracle
fuzzer over enumerable instances, and a [Barbarik](https://github.com/meelgroup/barbarik)
harness with GPAP hooked in-process and SPUR as reference. Run it with both controls (an
exactly uniform sampler under test must Accept; the `soft` config must Reject). Results and the
asymmetry of the guarantee: [`docs/findings/UNIFORMITY_FINDINGS.md`](docs/findings/UNIFORMITY_FINDINGS.md).

## From C++

The same library, a C ABI (`gpap/gpap.h`) plus a header-only C++ layer (`gpap/gpap.hpp`):

```cpp
#include <gpap/gpap.hpp>

gpap::Sampler sampler("formula.cnf");
gpap::Solutions sols = sampler.sample(100000);     // 100k satisfying assignments
bool v = sols(i, j);                               // of variable sols.variables()[j]

gpap::Engine engine(formula);                      // or own the loop yourself
engine.reset(65536);
engine.trajectory(order, n_support, /*B=*/8, seed);
int models = engine.select_complete(engine.pair_mask());
```

```cmake
find_package(gpap REQUIRED)
target_link_libraries(my_solver PRIVATE gpap::gpap)
```

Runtime dependency: `libcuda` only. Examples for both faces in [`cpp/examples/`](cpp/examples/)
and [`examples/`](examples/); details in [`cpp/README.md`](cpp/README.md).

## The core idea: no clause scanning

1. **Negated occurrence lists** (CSR): `occ[l]` = clauses where `~l` appears, so setting `l`
   TRUE falsifies `~l` in exactly those clauses. Binary clauses live only here, as implication
   edges with zero per-row state; unit clauses live in a bitmap.
2. **One packed word per clause of length ≥ 3**, batched over K assignments: a count of
   non-falsified literals (`0` = conflict, `1` = unit) and the sum of their indices.
3. **The sum trick**: at count 1 the surviving literal's index *is* the sum. No scan, no gather.
   The whole clause update is one `atomic_add` whose return value is the unit/conflict
   detection. [`cpp/kernels/bcp.py`](cpp/kernels/bcp.py), [`docs/ENGINE.md`](docs/ENGINE.md).

Literal indexing: `+v → 2(v-1)`, `-v → 2(v-1)+1`, negation `l ^ 1`.

## Layout

```
cpp/              # the engine core: libgpap, C ABI + C++ wrapper, shared by both faces
gpap/
  engine/         # the Python face of the core (ctypes)
  sampling/       # the sampler, the e2e pipeline, preprocessing, output
  circuits/       # .bench / AIGER parse, unroll, Tseitin, pin -> projected-sampling CNF
  mining.py       # entailed-clause mining over the core's probe primitive
  cli.py          # `gpap` subcommands
eval/             # the experiment layer: campaign driver, rival harnesses, uniformity, paper tables
scripts/          # build libgpap, optional third-party tools, fetch benchmarks
docs/             # start at docs/README.md; docs/findings/ is the evidence behind every rule
tests/            # pytest (CPU smoke + `gpu`/`slow` markers); circuit CNFs generated on demand
benchmarks/       # small committed .bench sources (CNFs are generated, never committed)
.gpap/            # built/fetched tools and benchmark sets, gitignored
```

Code carries the rule; `docs/` carries the evidence.

## Tests

```bash
pytest -m "not gpu and not slow"   # portable smoke suite
pytest                             # full suite (GPU tests skip off-CUDA; `slow` = deep unrolls)
```

## Citation and license

[`CITATION.cff`](CITATION.cff). MIT, see [`LICENSE`](LICENSE).
