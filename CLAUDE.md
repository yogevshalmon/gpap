# GPAP — GPU-Parallel Assumption Propagation & CircuitSAT Sampling

GPU-accelerated BCP over K independent assumption sets (PyTorch + fused Triton), and a
CircuitSAT/projected **solution sampler** built on it. The engine is the novelty; sampling is the
flagship app. Deep material lives in [`docs/`](docs/) — keep this file short and current.

## Core design — no clause scanning

- **Negated occurrence lists** (CSR): `occ[l]` = clauses where `~l` appears. Setting `l` TRUE
  falsifies `~l` in exactly those clauses.
- **Two per-clause counters, batched over K sets**: `sat_count (K,M)` (non-falsified literals;
  `==0` conflict, `==1` unit) and `lit_sum (K,M)` (sum of surviving literal indices).
- **The sum trick**: at `sat_count == 1` the surviving literal's index **is** `lit_sum`. The fused
  kernel packs both into one int32, replacing the sum with an XOR (never overflows).
- **Literal indexing (2N)**: `+v → 2(v-1)`, `-v → 2(v-1)+1`, negation `l ^ 1`. Assignment `(K,2N)`
  int8, 0=unassigned / 1=TRUE.

## Layout

```
gpap/
  engine/                  # THE PRIMITIVE (app-agnostic, reusable)
    formula.py             # CnfFormula + OccurrenceList + literal-index helpers (2N scheme, SAT types)
    bcp_engine.py          # PyTorch reference engine: the A/B oracle + CPU fallback (SLIMMED)
    kernels.py             # all Triton kernels: fixpoint + trajectory (unpacked) + packed (SHIFT=16|8)
    calibrate.py           # K sizing, VRAM model, gpu_mem_available
    checker.py             # AssumptionChecker: the batched-assumptions API face
    external_solver.py     # PySAT wrapper (verification)
  sampling/                # THE FLAGSHIP
    sampler.py             # GPUSampler (trajectories, bandit, soft-stop) — engine consumer only
    pipeline.py            # sample_file / sample_clauses: the full e2e flow + the two portfolios
    miner.py               # entailed-clause mining (the lose-set killer)
    preprocess.py          # unit_closure / equivalent_literal_substitution (ELS) primitives
    metrics.py             # distinct_projections / two_wise_coverage (off the timing path)
  circuits/bench.py        # parse/unroll/Tseitin/pin .bench -> projected-sampling CNF
  cli.py                   # `gpap` subcommands (sample / bench / circuit2cnf / info)
eval/                      # cross-GPU sampling benchmark driver + plots (Colab / A100)
scripts/                   # thin CLI wrappers (sample_benchmark, bench_to_cnf) + profile_engine
docs/findings/             # INSIGHTS.md (settled results index) + *_FINDINGS.md deep dives
tests/                     # pytest; circuit CNFs GENERATED from benchmarks/circuits/bench/ at test time
archive/                   # concluded experiments (not maintained; see archive/README.md)
```

## Engine contract — two entry points (see `gpap/engine/__init__.py`)

1. **Batched assumption checking**: `apply_and_propagate(K×L decisions) → conflict / real_conflict /
   assignment` (sum-trick BCP). `AssumptionChecker` is the one-call face.
2. **Fused trajectory sampling**: `reset_fused(K)` + `run_trajectory(order, n_support, B, seed, …)`
   — the whole randomized decide+propagate loop in one launch.

App code must go through these + the accessors (`sat_rows()`, `has_kernel`) — do NOT reach into
`_sat_count`/`_fused_kernel`. BCP fixpoints are confluent, so kernel and PyTorch paths are
bit-identical on non-conflicted rows (the PyTorch path is the oracle + CPU fallback).

## Fused kernel & K sizing (the memory model that keeps biting)

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set in `gpap/__init__.py` **before** torch
  inits CUDA. It removes the allocator fragmentation that inflated reserved memory ~2.3× and forced
  a small K. Do not remove it.
- `GPUSampler._calibrate_k`: **K = min(4 × safe_floor, analytic, cap)**, `safe_floor =
  0.7·free/(70M+40N)`. A runtime memory PROBE is unreliable here — do not resurrect it.
- **`K_fused`** (fused-only chunks skip the (K,N) decision masks): a boosted per-chunk K. The packed
  branch deducts `eng_hold` UNCONDITIONALLY — skipping it for all-fused portfolios spilled b05-c50
  under the kill-switch (the headroom covers the fixed 4096×M soundness temp). Don't re-try without
  an explicit 4096·M reservation + kill-switch re-validation.
- **Packed field split** (per formula, in `TritonBCP._build_flush_cache`): 16|16 for 2N ≤ 65535;
  **8|24 ("wide") for 65535 < 2N ≤ 0xFFFFFF, max clause len ≤ 255** (the big instances — the
  default now covers them, geomean 1.31× on c50 unrolls); else the unpacked fallback. One merged
  `_bcp_trajectory_packed_kernel` with a `SHIFT` constexpr — `mono` (1-byte) lost to wide and was
  deleted (see docs/findings/INSIGHTS.md).

## Running

```bash
pytest -m "not gpu and not slow"      # portable CPU smoke (any platform)
pytest                                # full suite (gpu tests skip off-CUDA; slow = deep unrolls)

gpap info                             # device / triton / power-limit warning
gpap sample formula.cnf -n 50000 --json out.json
gpap sample b10.bench --cycles 25 -n 50000
gpap bench 'path/*.cnf' --cmsgen /path/to/cmsgen --n-cov 5000
```

Benchmark suite driver (thin table over `gpap.sampling.pipeline`):
```bash
PYTHONPATH=. .venv/bin/python scripts/sample_benchmark.py \
    --glob '/path/to/circuits/*.cnf' --use-kernel auto --output .experiments/out.json
```
Cross-GPU (Colab/A100): `eval/colab_sampling_benchmark.py` + `notebooks/gpap_colab_sampling.ipynb`.

## Ground rules (do not re-litigate — see docs/findings/INSIGHTS.md)

- Runtime per-batch memory PROBES are unreliable → fixed 4×-floor K stays.
- This box SPILLS to host RAM (never OOMs) → validate every K change under the nvidia-smi
  kill-switch; check the power state first (a 40 W cap / battery clocks ~halve every rate).
- `mono` kernel (0.93× vs wide), `orac` circuit config (0/64 wins), full/soft global lookahead:
  measured out — soft stays a per-instance bandit option (regime-dependent), never a global default.
- `--n-time` is a delivery contract; safety valves are `--time-limit` + the zero-progress stop.
- Measurement protocol: [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md).

## Tests / methodology

- Circuit CNFs are GENERATED at test time from the committed `.bench` sources (no dependence on any
  user-local benchmark tree). `tests/conftest.py::circuit_cnfs` builds the matrix once per session.
- The no-change gate for a pure refactor: fixed K, `K_fused` pinned to K, single config, seed 5 ⇒
  bit-identical model counts (+ model-set hashes on small instances). Rates alone are draw-noisy on
  low-success instances (±40%) — see docs/BENCHMARKING.md §3.
