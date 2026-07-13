# GPAP — GPU-Parallel Assumption Propagation & CircuitSAT Sampling

GPU-accelerated BCP over K independent assumption sets (packed Triton kernels — the ONLY
execution path; CUDA+triton required, no CPU fallback), and a CircuitSAT/projected **solution
sampler** built on it. The engine is the novelty; sampling is the flagship app. Deep material
lives in [`docs/`](docs/) — keep this file short and current.

## Core design — no clause scanning

- **Negated occurrence lists** (CSR, int32): `occ[l]` = clauses where `~l` appears. Setting `l`
  TRUE falsifies `~l` in exactly those clauses.
- **ONE packed int32 clause word, batched over K rows**: low `SHIFT` bits = `sat_count`
  (non-falsified literals; `==0` conflict, `==1` unit), high bits = **XOR of surviving literal
  indices** — at `sat_count == 1` the XOR **is** the forced literal (xor of a one-element set;
  never overflows). Update per falsified literal: `st = (st - 1) ^ (neg << SHIFT)`.
- **Literal indexing (2N)**: `+v → 2(v-1)`, `-v → 2(v-1)+1`, negation `l ^ 1`. Assignment `(K,2N)`
  int8, 0=unassigned / 1=TRUE.
- Correctness story: BCP fixpoints are confluent ⇒ kernels are A/B'd against a pure-Python
  reference BCP (`tests/reference_bcp.py`) + every emitted model is externally verifiable.

## Layout

```
gpap/
  engine/                  # THE PRIMITIVE (app-agnostic, reusable; CUDA+triton-only)
    formula.py             # CnfFormula + int32 CSR occ + init_lit_xor + DIMACS tokenizer (2N scheme)
    bcp_engine.py          # BCPEngine: packed state + flush cache + kernel launches + counters
    kernels.py             # the 2 Triton kernels (fixpoint + trajectory, SHIFT=16|8) + _propagate helper
    calibrate.py           # K sizing, VRAM model, gpu_mem_available
    checker.py             # AssumptionChecker: the batched-assumptions API face
  sampling/                # THE FLAGSHIP
    sampler.py             # GPUSampler (trajectories, bandit, soft-stop) — engine consumer only
    pipeline.py            # sample_file / sample_clauses: the full e2e flow + the two portfolios
    miner.py               # entailed-clause mining (the lose-set killer)
    preprocess.py          # unit_closure / equivalent_literal_substitution (ELS) primitives
    metrics.py             # distinct/two-wise + UNIFORMITY (exact chi2/TV or collision l2_ratio)
    output.py              # compact sample writers (bits / npy / dimacs) — GPU bit-projection out
  circuits/
    bench.py               # parse/unroll/Tseitin/pin .bench -> projected-sampling CNF
    aiger.py               # AIGER reader: binary aig + ASCII aag, latches + reset values
    aig2cnf.py             # NATIVE .aig/.aag input: simulate/unroll/Tseitin (no external tool)
    abc_backend.py         # OPTIONAL berkeley-abc encoder (unroll + dc2/fraig -> AIG -> Tseitin)
    simulate.py            # bit-parallel sim: REJECTION baseline (p_accept), rare targets, sim-check
  cli.py                   # `gpap` subcommands (sample / bench / circuit2cnf / info)
  benchmark.py             # the `gpap bench` suite driver (in-package so it works installed)
eval/                      # cross-GPU sampling benchmark driver + plots (Colab / A100)
scripts/                   # shims (sample_benchmark -> gpap.benchmark), bench_to_cnf, build_abc.sh,
                           #   external_solver.py (PySAT utility, kept for future use)
docs/findings/             # INSIGHTS.md (settled results index) + *_FINDINGS.md deep dives
tests/                     # pytest; circuit CNFs GENERATED from benchmarks/circuits/bench/ at test time
archive/                   # concluded experiments (not maintained; see archive/README.md)
```

## Engine contract — two entry points (see `gpap/engine/__init__.py`)

1. **Batched assumption checking**: `reset(K)` + `apply_and_propagate(K×L decisions) → conflict /
   assignment` (one fixpoint launch). `AssumptionChecker` is the one-call face; the miner's
   incremental flow is `reset(1) + apply_and_propagate(units) + recache_flush()`.
2. **Fused trajectory sampling**: `reset(K)` + `run_trajectory(order, n_support, B, seed,
   soft=…)` — the whole randomized decide+propagate loop in one launch.

`reset(K)` broadcast-restores the construction-time post-units-flush cache (no per-chunk flush).
App code must go through these + the accessors (`sat_rows()`, work counters) — do NOT reach into
`_packed`/`_assignment`. The engine picks the packed split per formula (16|16 / 8|24) and
REJECTS formulas outside both envelopes; the PyTorch path was deleted 2026-07-13 (packed-XOR
kernels are the only path — the oracle is `tests/reference_bcp.py`).

## Fused kernel & K sizing (the memory model that keeps biting)

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set in `gpap/__init__.py` **before** torch
  inits CUDA. It removes the allocator fragmentation that inflated reserved memory ~2.3× and forced
  a small K. Do not remove it.
- `GPUSampler._calibrate_k`: **K = min(4 × safe_floor, analytic, cap)**, `safe_floor =
  0.7·free/(70M+40N)`. A runtime memory PROBE is unreliable here — do not resurrect it.
- **`K_fused`** (every chunk is fused now): a boosted per-chunk K. The `eng_hold` deduction stays
  UNCONDITIONAL — skipping it spilled b05-c50 under the kill-switch (the headroom covers the fixed
  4096×M soundness temp). Don't re-try without an explicit 4096·M reservation + kill-switch
  re-validation. `compute_k_safe`'s analytic model is knowingly conservative (calibrated on the
  retired PyTorch path) — retuning it = K change = kill-switch re-validation.
- **Packed field split** (per formula, in `BCPEngine.__init__`): 16|16 for 2N ≤ 65535;
  **8|24 ("wide") for 65535 < 2N ≤ 0xFFFFFF, max clause len ≤ 255** (the big instances — covers
  them, geomean 1.31× on c50 unrolls); anything larger is REJECTED with a ValueError (no unpacked
  fallback any more). One kernel source with a `SHIFT` constexpr — `mono` (1-byte) lost to wide
  and was deleted (see docs/findings/INSIGHTS.md).

## Running

```bash
pytest -m "not gpu and not slow"      # host-side logic smoke (parsers/preprocess/circuits/metrics
                                      #   run anywhere; ALL engine/sampler tests need CUDA+triton
                                      #   and self-skip without them)
pytest                                # full suite (gpu tests skip off-CUDA; slow = deep unrolls)

gpap info                             # device / triton / power-limit warning
gpap sample formula.cnf -n 50000 --out samples.out --stats run.json
gpap sample b10.bench --cycles 25 -n 50000 --out stim.out    # .bench: columns = wire@cycle
gpap sample b10.bench --cycles 25 --encoder abc              # ABC-encoded CNF (2-8x fewer clauses)
gpap bench 'path/*.cnf' --cmsgen /path/to/cmsgen --n-cov 5000
scripts/build_abc.sh                                         # optional: pinned abc -> ~/.cache/gpap/bin
```

**Circuit inputs**: `.bench` AND AIGER `.aig`/`.aag` (aig2cnf.py — native parse/simulate/unroll;
latch resets 0/1 honoured, `x` rejected, combinational clamps to T=1; labels from the symbol
table or `i<k>`).

**Rejection baseline + targets** (simulate.py — the "why not just simulate?" answer, measured).
`--sim-baseline` (sample + bench): random inputs + bit-parallel circuit simulation (64
patterns/word, numpy, CPU) = the CNF-free rival; PERFECTLY uniform, rate = p_accept × sim
throughput. Reports p_accept + accepted/s + GPAP amplification (= success/p_accept); bench
columns p_acc / rej/s / amp. `--target random|rare` (all circuit conversions): `random` (default)
pins the outputs of one seed-random sequence — SIZE-BIASED toward high-probability,
rejection-easy patterns; `rare` pins the rarest pattern in a 262k-sequence sim batch (observed ⇒
reachable ⇒ SAT, deterministic per seed). `--sim-check` (sample): simulate every emitted sample
row through the ORIGINAL circuit and verify the pinned outputs — encoder-INDEPENDENT end-to-end
soundness (validates unroll+Tseitin/ABC, not just CNF clauses).

**Circuit encoder** (`--encoder tseitin|abc|auto`, default `tseitin`). `abc` = strash + `zero` +
`frames -F T -i` + dc2/fraig then Tseitin its AIG (only 2-/3-literal clauses — what the
sum-trick BCP wants). Same projected solution space: those passes are function-preserving and
`zero`+`frames -i` IS our reset-state-0 unroll (gated by tests/test_abc_encoder.py, which
enumerates projections under both encoders). `zero` is MANDATORY — without it ABC turns the T=0
register state into free PIs (and it is why reset-1 AIGs are refused on the abc route). Default
stays `tseitin` so results never shift silently with ABC's presence. AIGER (not `write_cnf`) is
the route: `write_cnf` is ~1.5x tighter but exposes no primary-input→CNF-var map; the framed
CI/CO mapping is POSITIONAL (frame-major), name-verified when symbols exist.
`--abc-opt full|fast|auto`: post-unroll fraig is the whole encode cost (~99s on b15-c25) and the
whole cross-frame win (b12's 12-14x NEEDS an equivalence sweep; dc2-only gets 1.4x) — `auto`
runs `full` below ~50k estimated unrolled ANDs, `fast` (pre-unroll opt + cheap post) above.
Encode wall is printed + in stats (`encode_s`) — it is NOT inside the pipeline's e2e rate.
**Miner overlap**: dc2/fraig removes the backbone L1 mines (0-45 units on ABC CNFs vs 59-1007 on
Tseitin, for the same 5s budget) ⇒ `sample_file` auto-disables mining on `c encoder abc` CNFs
(`--mine on` forces).

**Uniformity readout** (`--uniformity off|fast|exact`, default **off** — the readout costs an
extra sampling pass, so it is opt-in; always off the timing path).
`fast` = repeat-rate check only, NO solver work: distribution-free `l2_ratio` = p̂₂·d — >1.5
certifies skew for ANY |S| (uniform ⇒ ||p||₂² = 1/|S| ≤ 1/d); ≤1 prints OK but is a GROSS-skew
check only (it does NOT see b06-c4-class subtle bias, l2 1.14 there); collision-free prints the
95% birthday floor on |S|. `exact` adds
enumeration (≤4096 sols / 2s / M≤200k) + chi2 p (needs n≥5·|S|) + TV over ALL |S| solutions + an
observed⊆enumerated soundness cross-check — the sensitive test (catches TV=0.17 at p=0.000);
when the space/formula defeats enumeration it falls back to the repeat-rate line with an
honest "barbarik for certification" hint (`enum_status` capped/timeout/skipped-large).
Certificates on big spaces = conditional sampling = Barbarik (`eval/barbarik_gpap.py`, GPAP
hooked in-process, SPUR reference; Accept 2/2 so far). chi2 sf is native (`metrics._gammq`) —
no scipy dep. **2-wise coverage** (CMSGen-paper spread metric, NOT uniformity) is opt-in via
`gpap bench --twise` (auto-bumps n_cov to 2000) and vectorised (one matmul; the old O(S²·n)
pair loop dominated the quality pass).
Sample output = compact support projections (1 bit/support var; GPU-projected so only the bit
matrix leaves the device). `--out-format bits|npy|dimacs`, `--max-samples` cap (default 100k),
`--stats` for the info JSON. Sound: every row extends to a full model (test_output.py). Circuit
CNFs embed a `c wire <var> <input@cycle>` map (bench.py) so output columns are named by wire.

Benchmark suite driver (thin table over `gpap.sampling.pipeline`; the glob may mix CNFs with
circuits — .bench/.aig/.aag are converted first with the same --encoder/--abc-opt flags, encode
wall reported per instance as t_enc, never folded into either sampler's rate):
```bash
PYTHONPATH=. .venv/bin/python scripts/sample_benchmark.py \
    --glob '/path/to/circuits/*.cnf' --use-kernel auto --output .experiments/out.json
PYTHONPATH=. .venv/bin/python scripts/sample_benchmark.py \
    --glob 'benchmarks/circuits/bench/seq/*.bench' --encoder abc --cycles 25
```
Cross-GPU (Colab/A100): `eval/colab_sampling_benchmark.py` + `notebooks/gpap_colab_sampling.ipynb`.

## Ground rules (do not re-litigate — see docs/findings/INSIGHTS.md)

- Runtime per-batch memory PROBES are unreliable → fixed 4×-floor K stays.
- This box SPILLS to host RAM (never OOMs) → validate every K change under the nvidia-smi
  kill-switch; check the power state first (a 40 W cap / battery clocks ~halve every rate).
- `mono` kernel (0.93× vs wide), `orac` circuit config (0/64 wins), full/soft global lookahead:
  measured out — soft stays a per-instance bandit option (regime-dependent), never a global default.
- `--n-time` is a delivery contract; safety valves are `--time-limit` + the zero-progress stop.
- **VACUOUS-PIN instances**: at T=25 the last-cycle outputs of `b02`, `b05`, `b07`, `s526` are
  CONSTANT for every input sequence ⇒ pinning them is a tautology and "sampling" is a free
  hypercube over |PI|·T bits (any sampler gets ~100% success; ABC collapses the whole output cone,
  so its "shrink" there is instance degeneracy, not encoder merit). Exclude them from headline
  encoder/sampler comparisons, or report them separately. `b01`/`b09` are near-vacuous (2 patterns).
- Measurement protocol: [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md).

## Tests / methodology

- Circuit CNFs are GENERATED at test time from the committed `.bench` sources (no dependence on any
  user-local benchmark tree). `tests/conftest.py::circuit_cnfs` builds the matrix once per session.
- The no-change gate for a pure refactor: fixed K, `K_fused` pinned to K, single config, seed 5 ⇒
  bit-identical model counts (+ model-set hashes on small instances). Rates alone are draw-noisy on
  low-success instances (±40%) — see docs/BENCHMARKING.md §3.
