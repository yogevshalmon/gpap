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
pip install -e ".[cuda]"     # CUDA + the fused Triton kernels (ships with torch on Linux/Colab)
pip install -e ".[dev]"      # + pytest, ruff  (for tests/lint)
```

CPU-only works too (the PyTorch engine is the oracle/fallback) — the fused kernels simply switch off.

## Quickstart — Python API

```python
import gpap

# Sample a DIMACS CNF (support read from `c ind` header lines if present)
models, info = gpap.sample("formula.cnf", n=50_000, seed=5, device="cuda", n_cov=5_000)
print(info["samples_per_s"], info["success_rate"], info["gpu_sound"])

# Sample a sequential circuit: unroll T cycles, pin outputs to a reachable pattern, sample inputs
models, info = gpap.sample_circuit("b10.bench", cycles=25, n=50_000)

# In-memory clauses
models, info = gpap.sample_clauses(nv=3, clauses=[[1, 2], [-1, 3], [-2, -3]], n=1_000,
                                   support=[0, 1, 2], device="cuda", n_cov=1_000)

# The engine as a primitive — batched assumption checking (the reusable core):
from gpap.engine import AssumptionChecker
checker = AssumptionChecker("formula.cnf")
r = checker.check_assumptions([[1, -2, 3], [-1, 2, -3]])   # K sets at once
print(r["conflict"])          # (K,) bool
```

`info` is a rich dict: throughput (`samples_per_s` steady-state, `sps_e2e_total` incl. setup),
`success_rate`, per-phase timings, mining stats, and soundness flags. `models` come from the
optional coverage pass (`n_cov > 0`); the timed run itself collects nothing (throughput only).

## Quickstart — CLI

```bash
gpap info                                   # device, triton, abc encoder, power-limit warning
gpap sample formula.cnf -n 50000 --out samples.out --stats run.json
gpap sample b10.bench --cycles 25 -n 50000 --out stim.out   # unroll, sample, write input stimuli
gpap sample design.aig --cycles 25 -n 50000                 # AIGER input (.aig/.aag), same flow
gpap sample b10.bench --cycles 25 --encoder abc             # smaller CNF via berkeley-abc (below)
gpap circuit2cnf benches/ cnf_out/ --cycles 25 --pin-outputs 1   # circuit dir -> CNF dir
gpap bench 'benchmarks/circuits/*.cnf' --cmsgen /path/to/cmsgen --n-cov 5000
```

**Circuit inputs.** `.bench` (ISCAS-85/89, ITC'99) and AIGER `.aig`/`.aag` (binary and ASCII,
combinational or sequential) are parsed natively — no external tool needed. Sequential circuits
are unrolled `--cycles` times from the reset state (AIGER latch reset values 0/1 are honoured;
`.bench` flip-flops reset to 0; combinational AIGs clamp to one cycle), the last-cycle outputs are
pinned to a simulated reachable pattern, and the primary inputs across cycles become the sampling
support (named `wire@cycle` in the output).

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

## The core idea (no clause scanning)

Instead of a clause matrix, GPAP keeps:

1. **Negated occurrence lists** (CSR): `occ[l]` = clauses where `~l` appears. Setting `l` TRUE
   falsifies `~l` in exactly those clauses.
2. **Two per-clause counters**, batched over K assumption sets:
   - `sat_count (K, M)` — remaining non-falsified literals (init = clause size). Decrement on
     falsification. `== 0` ⇒ conflict, `== 1` ⇒ unit.
   - `lit_sum (K, M)` — running sum of non-falsified literal *indices*.
3. **The sum trick**: when `sat_count == 1`, the surviving literal's index **is** `lit_sum`. No
   scan, no gather — pure arithmetic. (The fused kernel packs both counters into one 32-bit word,
   replacing the sum with an XOR that never overflows; see [`gpap/engine/kernels.py`](gpap/engine/kernels.py).)

Literal indexing (2N scheme): `+v → 2(v-1)`, `-v → 2(v-1)+1`, negation `l ^ 1`.

## Layout

```
gpap/
  engine/       # THE PRIMITIVE — formula, bcp_engine (oracle), kernels (fused Triton), calibrate, checker
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
