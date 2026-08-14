# GPAP — GPU-Parallel Assumption Propagation & CircuitSAT Sampling

GPAP is a GPU primitive for **Boolean Constraint Propagation (BCP) over millions of independent
assumption sets at once** (PyTorch + fused Triton kernels), and a flagship application built on it:
**guaranteed-valid SAT solution sampling** for CircuitSAT and projected sampling.

- **The engine** ([`gpap.engine`](gpap/engine/)) — the novelty. K independent assumption sets are
  BCP'd to fixpoint in parallel with no clause scanning, via *negated occurrence lists* + two
  per-clause counters and **the sum trick** (below). A fused Triton kernel runs a whole randomized
  decide+propagate trajectory per GPU thread in one launch.
- **The sampler** ([`gpap.sampling`](gpap/sampling/)) — GPU-parallel CMSGen: K randomized
  BCP-completions per chunk, every emitted model provably satisfies the formula (no
  gradient/rounding/repair). It samples the primary-input support of unrolled sequential/
  combinational circuits, the setup of the recent GPU-sampling papers (DEMOTIC, Recurrent
  CircuitSAT).

> Hardware caveat for the numbers below: they were measured on a laptop RTX 4080 (12 GB, WSL2),
> often under a 40 W power cap. Ratios (GPAP vs CMSGen, config A vs B) hold within one power state;
> absolute rates are not comparable across states or GPUs. See [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md).

## Install

```bash
pip install -e ".[build]"    # python side + triton (triton ships with torch on Linux/Colab)
scripts/build_core.sh        # the engine core: libgpap.so  (cmake + gcc; NO CUDA toolkit needed)
pip install -e ".[dev]"      # + pytest, ruff  (for tests/lint)
```

The engine itself is a C++ library, [`libgpap`](cpp/) — the same one C++ consumers link against
(see [cpp/README.md](cpp/README.md)). `scripts/build_core.sh` compiles the Triton kernels ahead of
time into it and builds it; it needs a C++17 compiler, cmake, the NVIDIA driver and (at build time
only) triton — **no CUDA toolkit and no nvcc**. `gpap info` reports whether it is there.

At RUN time `libgpap.so` links **`libcuda.so.1` alone** — no triton, no CUDA toolkit, no Python.
The cubins inside it are built for one GPU architecture, so moving the library to a different
card generation means re-running `scripts/build_core.sh` there (the loader says so by name if
you forget).

A GPU is required: there is no CPU execution path.

## Quickstart — Python API

```python
import gpap

# Sample a DIMACS CNF (support read from `c ind` header lines if present)
samples, info = gpap.sample("formula.cnf", n=50_000, seed=5, device="cuda")
samples.shape                 # (50000, S) uint8 — one column per support variable
info["cols"]                  # the DIMACS variable each column stands for
print(info["samples_per_s"], info["success_rate"], info["gpu_sound"])

# Sample a sequential circuit: unroll T cycles, pin outputs to a reachable pattern, sample inputs
samples, info = gpap.sample_circuit("b10.bench", cycles=25, n=50_000)
info["wire_map"]              # column variable -> "input@cycle"

# In-memory clauses
samples, info = gpap.sample_clauses(3, [[1, 2], [-1, 3], [-2, -3]], n=1_000, support=[0, 1, 2])

# The engine as a primitive — batched assumption checking (the reusable core):
from gpap.engine import AssumptionChecker
checker = AssumptionChecker("formula.cnf")
r = checker.check_assumptions([[1, -2, 3], [-1, 2, -3]])   # any number of sets, chunked for you
print(r["conflict"])          # (n,) bool
```

Nothing above mentions a batch size: every entry point sizes itself to the GPU and the formula.

`samples` is a numpy `(rows, S)` uint8 bit matrix of support projections — projected on the GPU,
so only those bits cross the bus, and collected OFF the timing path so `samples_per_s` is
unaffected. Rows are capped by `max_samples` (default 100,000) and every one is a guaranteed-valid
projection. Pass `full_models=True` for the historical `list[list[int]]` of full DIMACS models
instead (via the coverage pass — more informative, considerably more expensive).

`info` is a rich dict: throughput (`samples_per_s` steady-state, `sps_e2e_total` incl. setup),
`success_rate`, per-phase timings, mining stats, and soundness flags.

## Quickstart — CLI

```bash
gpap info                                   # device, triton, abc encoder, power-limit warning
gpap sample formula.cnf -n 50000 --out samples.out --stats run.json
gpap sample b10.bench --cycles 25 -n 50000 --out stim.out   # unroll, sample, write input stimuli
gpap sample design.aig --cycles 25 -n 50000                 # AIGER input (.aig/.aag), same flow
gpap sample b10.bench --cycles 25 --encoder abc             # smaller CNF via berkeley-abc (below)
gpap circuit2cnf benches/ cnf_out/ --cycles 25 --pin-outputs 1   # circuit dir -> CNF dir
gpap bench 'benchmarks/circuits/*.cnf' --cmsgen ~/.cache/gpap/bin/cmsgen --n-cov 5000
gpap bench 'benches/*.bench' --encoders tseitin abc:fast --cycles 25 --target rare --resume
```

**Comparing against CMSGen fairly.** Build it with `scripts/build_cmsgen.sh` (upstream at a
pinned commit + the small in-repo patch `scripts/cmsgen_gpap_eval.patch`). GPAP's timed run emits
nothing — models stay on the GPU — while stock CMSGen writes every sample as a full N-variable
DIMACS assignment, which on wide formulas costs several times the sampling itself (measured
`27.sk_3_32`, 50k samples: 0.42 s vs 3.6 s). The patch adds `--nowrite` (generate, emit nothing)
and `--statsevery` (its own propagation/decision counters); `gpap bench` detects both, times the
like-for-like run, and reports the emission cost separately. A stock binary still works — the row
is just marked `cms_mode="write"`. See [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) §4b.

**The full experimental sweep** (all families, both encoders, both target regimes, the GPU rivals,
and the figures) is one script per family — [`docs/EVAL_RUNBOOK.md`](docs/EVAL_RUNBOOK.md):

```bash
eval/rivals_setup.sh            # abc + patched cmsgen + DEMOTIC / Recurrent-CircuitSAT / HTS
eval/run_full_eval.sh seq       # resumable; also comb | sk | epfl | hwmcc | rivals | plots | all
```

**Circuit inputs.** `.bench` (ISCAS-85/89, ITC'99) and AIGER `.aig`/`.aag` (binary and ASCII,
combinational or sequential) are parsed natively — no external tool needed. Sequential circuits
are unrolled `--cycles` times from the reset state (AIGER latch reset values 0/1 are honoured;
`.bench` flip-flops reset to 0; combinational AIGs clamp to one cycle), the last-cycle outputs are
pinned to a simulated reachable pattern, and the primary inputs across cycles become the sampling
support (named `wire@cycle` in the output).

**Rejection baseline & target hardness.** For circuit inputs, `--sim-baseline` measures the
CNF-free rival every reviewer should ask about: draw random input sequences, bit-parallel
simulate the circuit (no CNF, perfectly uniform), keep the ones hitting the pinned outputs. It
reports `p_accept`, the accepted-samples/s that rejection achieves, and GPAP's *constraint
amplification* (success rate ÷ p_accept) — when `p_accept` is high, simple rejection wins and
the instance is not a constrained-sampling benchmark. `--target rare` pins the *rarest* output
pattern observed in a large simulation batch instead of the default random one (which is drawn
from the circuit's own output distribution and therefore biased toward easy patterns) — still
reachable, so the CNF stays SAT, but adversarial for rejection. `--sim-check` simulates every
emitted sample through the original circuit and verifies the pinned outputs: an
encoder-independent, end-to-end soundness check.

**Uniformity.** Opt-in (default `off` — the readout costs an extra sampling pass, so it is not
run unless asked for). `--uniformity fast` prints a one-line **repeat-rate check**: draw ~2k
samples and see whether some solutions repeat far more often than any uniform sampler would
allow (`l2_ratio` ≈ 1 is uniform-consistent, > 1.5 is `SKEWED`; collision-free samples report
the implied solution-space floor — no solver work, catches gross bias only). Two deeper levels:

- `--uniformity exact` — enumerate the projected solution space when small (≤4096 solutions) and
  run a real goodness-of-fit test: chi-square p-value + total-variation distance over all `|S|`
  solutions (e.g. `TV=0.136 chi2(255)=236 p=0.792 [exact: n=2,000 over |S|=256]`). This is the
  sensitive check — subtle instance-dependent bias invisible to the repeat-rate check shows here.
- `eval/barbarik_gpap.py` — a standalone harness running the
  [Barbarik](https://github.com/meelgroup/barbarik) (AAAI'19) conditional-sampling tester with
  GPAP hooked in-process (SPUR as the trusted reference); the only way to *certify* uniformity on
  large spaces. Barbarik **accepts** GPAP (η=0.9, ε=0.1, δ=0.05; ~70k samples per instance) on
  both instances tested so far.

**Writing solutions.** `--out FILE` writes the sampled solutions as compact **support projections**
— one row per sample, one bit per support variable (for a `.bench` input, one primary-input
`wire@cycle` per column, named in the header). The projection is built on the GPU and only the bit
matrix crosses to host RAM, so a large dump is cheap. Formats via `--out-format`: `bits` (0/1 text,
default), `npy` (numpy `uint8` array — `numpy.load`), `dimacs` (signed literals). Output is capped at
`--max-samples` (default 100k; raise for a full dump). `--stats FILE` writes the run's info/stats
JSON (rates, phases, mining, soundness) — separate from the samples. In Python the same matrix is
`info["samples"]` (with `info["cols"]` / `info["wire_map"]`), and `gpap.sampling.write_samples()`
writes any of the formats.

## Circuit encoding: native Tseitin, or the optional ABC pass

Sampling cost is dominated by BCP, which scales with the clause count `M`. `--encoder abc` unrolls
and logic-optimizes the circuit with [berkeley-abc](https://github.com/berkeley-abc/abc)
(structural hashing + `dc2` rewriting + `fraig` SAT-sweeping) and Tseitins the resulting AIG,
instead of encoding gate-by-gate. Measured on this repo's benches (vs the native encoder *after*
its own `unit_closure`): **2.2× fewer clauses on b10-c25, 7.8× on b14-c25** (976k → 125k), which
turns into a geomean **~5.9× end-to-end sampling speedup** across the non-degenerate T=25 suite.

This does **not** change what you sample. `strash`/`dc2`/`fraig` are function-preserving, and
`zero` + `frames -i` reproduces exactly the reset-state-0 unroll, so the primary-input→output
function — hence the projected solution space under the same output pins — is identical; only
internal gate variables are restructured. `tests/test_abc_encoder.py` enumerates the projected
solution set under both encoders and asserts set equality.

Encoding **cost** is visible and tunable: the CLI prints `encoded in N.Ns [...]` (also in the
`--stats` JSON as `encode_s`), and `--abc-opt` picks the effort — `full` optimizes the *unrolled*
AIG (best shrink; its SAT sweep can cost ~100 s on the biggest unrolls), `fast` optimizes the
small sequential AIG *before* unrolling plus one cheap post-pass, and `auto` (default) uses
`full` where it is cheap and `fast` where it would blow up. Since the encoder already removes
the formula's backbone, the pipeline's entailed-clause miner auto-disables on ABC-encoded CNFs
(a `c encoder abc` provenance comment travels with the file; `--mine on` forces it back).

ABC is **optional and never vendored**. Build a pinned upstream commit into the GPAP cache:

```bash
scripts/build_abc.sh          # -> ~/.cache/gpap/bin/abc   (git + make + gcc; a few minutes)
gpap info                     # confirms: "abc circuit encoder: <path>"
```

GPAP finds the binary via `$GPAP_ABC` → `abc` on `PATH` → `$GPAP_ABC_HOME/bin/abc`, or takes
`--abc PATH`. The default encoder stays `tseitin` so results never change silently depending on
whether ABC happens to be installed; `--encoder auto` opts into "abc if available".

## Linking from C++

The engine is the same library both faces use, so anything the Python API can do is available
without Python — a C ABI (`gpap/gpap.h`) plus a header-only C++ layer (`gpap/gpap.hpp`).
**Sampling a formula is three lines**, and needs no knowledge of how the engine works:

```cpp
#include <gpap/gpap.hpp>

gpap::Sampler sampler("formula.cnf");
gpap::Solutions sols = sampler.sample(100000);     // 100k satisfying assignments

for (std::size_t i = 0; i < sols.size(); ++i)
    for (std::size_t j = 0; j < sols.columns(); ++j)
        bool value = sols(i, j);                   // of variable sols.variables()[j] (DIMACS)
```

The sampler sizes the GPU batch to free VRAM, picks the propagate kernel, orders decisions so
propagation does the work, and verifies on the device that every row it returns is a real model.
`gpap::AssumptionChecker` is the same deal for the other entry point — K assumption sets
propagated independently in one launch. Underneath, `gpap::Formula` / `gpap::Engine` expose the
engine directly for callers who want to own the loop:

```cpp
gpap::Engine engine(formula);
engine.reset(65536);                                      // 65k independent trajectories
engine.trajectory(order, n_support, /*B=*/8, seed);       // one fused launch
int models = engine.select_complete(engine.pair_mask());  // guaranteed-valid solutions
```

```cmake
find_package(gpap REQUIRED)
target_link_libraries(my_solver PRIVATE gpap::gpap)
```

Runtime dependencies: `libcuda` only. Working examples in [`cpp/examples/`](cpp/examples/)
(`sample.cpp`, `assumptions.cpp`, and `demo.c` for the raw C ABI); details in
[cpp/README.md](cpp/README.md).

## The core idea (no clause scanning)

Instead of a clause matrix, GPAP keeps:

1. **Negated occurrence lists** (CSR): `occ[l]` = clauses where `~l` appears. Setting `l` TRUE
   falsifies `~l` in exactly those clauses.
2. **Two per-clause counters**, batched over K assumption sets:
   - `sat_count (K, M)` — remaining non-falsified literals (init = clause size). Decrement on
     falsification. `== 0` ⇒ conflict, `== 1` ⇒ unit.
   - `lit_sum (K, M)` — running sum of non-falsified literal *indices*.
3. **The sum trick**: when `sat_count == 1`, the surviving literal's index **is** `lit_sum`. No
   scan, no gather — pure arithmetic. The kernels pack both counters into ONE 32-bit word, which
   makes the whole clause update a single `atomic_add` whose return value *is* the conflict/unit
   detection; see [`cpp/kernels/bcp.py`](cpp/kernels/bcp.py) and [docs/ENGINE.md](docs/ENGINE.md).

Literal indexing (2N scheme): `+v → 2(v-1)`, `-v → 2(v-1)+1`, negation `l ^ 1`.

## Layout

```
cpp/            # THE ENGINE CORE — libgpap: C ABI + C++ wrapper, shared by both faces
gpap/
  engine/       # the Python face of the core (ctypes) + kernels.py, the Triton kernel SOURCE
  sampling/     # THE FLAGSHIP — sampler, pipeline (e2e flow), miner, preprocess, metrics
  circuits/     # bench.py: parse/unroll/Tseitin/pin .bench circuits -> projected-sampling CNF
  cli.py        # `gpap` subcommands
eval/           # reproducible cross-GPU sampling benchmark driver + plots (Colab / A100)
docs/findings/  # settled results: INSIGHTS.md + the deep *_FINDINGS.md write-ups
tests/          # pytest suite (CPU smoke + `gpu`/`slow` markers); circuit CNFs generated on demand
benchmarks/circuits/bench/  # tiny committed .bench sources (CNFs are generated, never committed)
archive/        # concluded experiment lines (not maintained; see archive/README.md)
```

## Tests

```bash
pytest -m "not gpu and not slow"   # portable CPU smoke suite (runs anywhere)
pytest                             # full suite (GPU tests auto-skip off-CUDA; `slow` = deep unrolls)
```

## Citation

A citation stub is in [`CITATION.cff`](CITATION.cff). The engine's defensible novelty: the
K-independent-assumption-set BCP primitive + the incremental occurrence-list/sum-trick propagation
+ the one-trajectory-per-thread fused kernel (prior GPU-SAT work uses watched-literals or
gradient relaxation — orthogonal mechanisms).

## License

MIT — see [`LICENSE`](LICENSE).
