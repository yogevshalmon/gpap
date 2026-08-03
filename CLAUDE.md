# GPAP — GPU-Parallel Assumption Propagation & CircuitSAT Sampling

GPU-accelerated BCP over K independent assumption sets, and a CircuitSAT/projected **solution
sampler** built on it. The engine is the novelty; sampling is the flagship app. Deep material
lives in [`docs/`](docs/) — keep this file short and current.

**The engine is a C++ library, [`libgpap`](cpp/)** (since 2026-07-28) — ONE implementation that
Python (`gpap.engine`, via ctypes) and C++ consumers (`#include <gpap/gpap.hpp>`) share. The GPU
kernels are still written in Triton: [`gpap/engine/kernels.py`](gpap/engine/kernels.py) is the
single source of truth, AOT-compiled into the library by `cpp/tools/build_kernels.py` (cubins
verified BIT-IDENTICAL at SASS level to the JIT ones). Build: `scripts/build_core.sh` — needs
cmake + a C++17 compiler + the NVIDIA driver + triton at build time; **no CUDA toolkit, no nvcc**.
Runtime deps of the library: `libcuda` alone. CUDA is required; there is no CPU path.

## Core design — no clause scanning

- **Negated occurrence lists** (CSR, int32): `occ[l]` = clauses where `~l` appears. Setting `l`
  TRUE falsifies `~l` in exactly those clauses.
- **ONE packed int32 clause word, batched over K rows**: low `SHIFT` bits = `sat_count`
  (non-falsified literals; `==0` conflict, `==1` unit), high bits = **SUM of surviving literal
  indices mod 2^(32-SHIFT)** — at `sat_count == 1` the field **is** the forced literal (every
  removal exact mod 2^F, one literal < 2^F). Update per falsified literal is ONE subtraction
  `st -= 1 + (neg << SHIFT)` (count never borrows, sum wraps off the top) — which is also a
  single `atomic_add` whose RETURN value gives race-free unit/conflict detection (the WAVE
  kernel's whole trick; XOR packing was the serial-only variant, replaced 2026-07-13).
- **Literal indexing (2N)**: `+v → 2(v-1)`, `-v → 2(v-1)+1`, negation `l ^ 1`. Assignment =
  BITMAP (bit l of int32 words; l and ~l share a word ⇒ ONE atomic_or answers duplicate +
  contradiction; 8× smaller than the former (K,2N) int8 — accessors materialize copies).
- Correctness story: BCP fixpoints are confluent ⇒ kernels are A/B'd against a pure-Python
  reference BCP (`tests/reference_bcp.py`) + every emitted model is externally verifiable.

## Layout

```
cpp/                       # THE ENGINE CORE — libgpap (C ABI + header-only C++ wrapper)
  include/gpap/gpap.h      # the C ABI: formula, engine, selections, readout, counters, sizing
  include/gpap/gpap.hpp    # C++, TWO TIERS: Sampler/AssumptionChecker (DIMACS, auto-everything)
                           #   over Formula/Engine (RAII + exceptions over the ABI's 0-based ids)
  src/formula.cpp          # DIMACS tokenizer + CSR occ + packed init (THE builder; python defers)
  src/engine.cpp           # state ownership (cuMemAlloc), reset/flush cache, both entry points
  src/readout.cpp          # selections, on-device soundness, bit gathers, assignment rows
  src/calibrate.cpp        # K sizing, VRAM model  |  src/runtime.cpp: cubin registry + launches
  kernels/helpers.py       # support kernels (Triton, so the toolchain stays nvcc-free)
  tools/build_kernels.py   # AOT: kernels.py + helpers.py -> embedded cubins (29 specializations)
  examples/                # sample.cpp + assumptions.cpp (tier 1) + demo.c (raw C ABI)
gpap/
  engine/                  # THE PYTHON FACE of the core (app-agnostic, reusable)
    _core.py               # ctypes binding to libgpap (private)
    formula.py             # CnfFormula: facade over the core's builder (numpy fallback off-GPU)
    bcp_engine.py          # BCPEngine: marshalling + torch-friendly accessors over the core
    _torch_engine.py       # the retired PyTorch/Triton-JIT engine — equivalence-gate reference only
    kernels.py             # the 2 Triton kernels (fixpoint + trajectory, SHIFT=16|8) + scalar/WAVE
    calibrate.py           # K sizing, VRAM model, gpu_mem_available
    checker.py             # AssumptionChecker: the batched-assumptions API face
  sampling/                # THE FLAGSHIP
    sampler.py             # GPUSampler (trajectories, bandit, soft-stop) — engine consumer only
    pipeline.py            # sample_file / sample_clauses: the full e2e flow + the portfolios
                           #   (VarOrder/Lookahead enums + SamplingConfig NamedTuple, so a config
                           #   documents itself; `portfolio(name, allow_soft=)` is the accessor)
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
eval/                      # the experiment layer: run_full_eval.sh (staged/resumable terminal
                           #   sweep) + eval_plots.py + rivals_{demotic,recurrent,hts}.py +
                           #   rivals_setup.sh + barbarik_gpap.py (CM_test, gpap|cmsgen under
                           #   test) + fuzz_uniformity.py (exact-oracle uniformity fuzzing)
                           #   + the cross-GPU Colab driver
scripts/                   # shims (sample_benchmark -> gpap.benchmark), bench_to_cnf,
                           #   build_abc.sh, build_cmsgen.sh + cmsgen_gpap_eval.patch,
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

**Hot-path readout is the SELECTION api** (added with the C core): `make_pair_mask(vars)` once,
then per chunk `select_complete(mask_id, slot) -> count` (the completion test runs on device; only
the count crosses the bus), plus `check_sound(slot)` (per-row flag on device — no (rows, M)
transfer), `var_bits(slot, cols, n)` and `assignment_rows_sel(slot, n)`. The tensor-shaped
accessors (`assignment`, `conflict`, `active`, `sat_rows`) are copies for tests / off-hot-path
consumers. App code must go through these — do NOT reach into the row state. The engine picks the packed split per formula (16|16 / 8|24) and
REJECTS formulas outside both envelopes; the PyTorch path was deleted 2026-07-13 (packed
kernels are the only path — the oracle is `tests/reference_bcp.py`).

## Fused kernel & K sizing (the memory model that keeps biting)

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set in `gpap/__init__.py` **before** torch
  inits CUDA. It removes the allocator fragmentation that inflated reserved memory ~2.3× and forced
  a small K. Do not remove it.
- `GPUSampler._calibrate_k`: **K = min(analytic, cap)** (retuned 2026-07-13, kill-switch
  validated: the old 4×0.7·free/(70M+40N) floor modeled the deleted PyTorch path and bound K at
  ~1/3 of the packed fit; the 4096×M soundness transient is now read in ≤256 MB slices). A
  runtime memory PROBE is unreliable here — do not resurrect it.
- **The `cap` is VRAM-proportional** (`engine.calibrate.default_k_cap`, 2026-08-03; `--k-cap auto`
  is the default of `gpap sample`/`gpap bench`/`pipeline.sample_clauses`). `K_CAP_REF ·
  total_VRAM / K_CAP_VRAM_REF` floored at K_CAP_REF — EXACT no-op on this box and smaller (gated
  by tests/test_k_cap.py + a bit-identical model-count A/B). Keys off **total** VRAM, not free: a
  free-memory cap drifted between runs in one session (49,526 vs 49,671), and a sweep whose K
  depends on what else was resident is not a measurement. Resolved value in every row's
  `engine.k_cap`; **a cross-GPU A/B must pass an explicit `--k-cap N`** since `auto` differs per
  card. ⚠️ **This is HYGIENE, not a throughput lever — measured, Colab A100+L4 2026-08-03**: auto
  vs pinned bought 1.012× (A100) / 0.985× (L4), and 4× `K_fused` bought 2.8%. The cap only binds
  for M ≲ 20k (above that the analytic bound is lower — and it ALREADY scales with free VRAM on
  its own), and those small-M instances are already occupancy-saturated. The cross-GPU win is the
  hardware: **A100 1.71× / L4 0.874× vs the 4080 at matched K** — and 1.71× off 3.6× the
  bandwidth says we are occupancy-bound, not bandwidth-bound. In the FLAGSHIP large-M regime
  (deep tseitin unrolls, T=25, rare) the A100's analytic K really is 3.9-6.4× the laptop's and
  throughput STILL only follows sublinearly: **b12 3.89× K → 1.19×, b14 6.37× K → 2.37×**. Budget
  a big GPU at ~1.2-2.4×, NOT the VRAM ratio. See docs/EVAL_RUNBOOK.md.
- **Read `smp.batch_size`, set it with `smp.pin_batch(n)` / `with smp.pinned_batch(n):`**
  (2026-08-03). K and K_fused are two MEMORY MODELS, not two batch sizes to choose between:
  `batch_size` = `K_fused if K >= K_init else K` is the single number that says what a chunk
  runs, and it is in every row as `engine.batch`. Poking the attributes directly was the bug
  factory — `K_fused = K` and `K = min(K, n)` only bound the batch when they happened to drop K
  BELOW its construction value, so at `n >= K` the boost stayed on and the chunk ran orders of
  magnitude wider than asked. That cost 28 GB of host RAM in one test, left the same hole in the
  pipeline's coverage run, and made the colab K-sweep mismeasure its own x-axis. `pin_batch`
  states a bound that holds in both states. Gated by tests/test_k_cap.py; the refactor was
  no-change-gated (identical model counts AND identical K/K_fused on 4 sk_ instances).
- **`K_fused`** (every chunk is fused now): a boosted per-chunk K. The `eng_hold` deduction stays
  UNCONDITIONAL — skipping it spilled b05-c50 under the kill-switch (the headroom covers the fixed
  4096×M soundness temp). Don't re-try without an explicit 4096·M reservation + kill-switch
  re-validation. `compute_k_safe`'s analytic model is knowingly conservative (calibrated on the
  retired PyTorch path) — retuning it = K change = kill-switch re-validation.
- **Assignment BITMAP** (2026-07-14): assignment = bit l of (K, ⌈2N/32⌉) int32 words (l/~l
  share a word) — 8× smaller than byte-per-literal (small-2N rows go L1-resident), and in
  WAVE a forced-literal push is ONE `tl.atomic_or` whose RETURN deduplicates (old bit l) and
  detects contradiction (old bit l^1) race-free — the old W×W in-warp dedup is deleted.
  Readout goes through materializing accessors (`assignment`, `assignment_rows`,
  `var_true_bits`, `any_free` — the sampler's completion check is one bitwise pass, never a
  (K,N) mask; strided bit-unpack slice-writes cost 16× sector RMW amplification, measured).
- **Kernel variant auto-dispatch** (same state, pure launch flag; gate-validated bit-identical):
  warp-wide **WAVE** propagate (≤W trail lits at once, occ segments flattened, the packed-word
  `atomic_add` return value resolving cross-literal clause races) when calibrated K_fused <
  30× resident warps (retuned post-bitmap: wave b03@49k 1.41×, scalar only holds the huge-K/
  tiny-M extreme b06@129k); scalar serial-trail propagate beyond. Session total vs the
  pre-wave baseline: b14 5.7×, b12 2.8×, b04 1.8×, b03 1.5×. `GPAP_KERNEL=scalar|wave`
  forces; W via `GPAP_WAVE_W` (32 best; W>32 spreads over W/32 real warps via num_warps —
  measured 2-5× WORSE, cross-warp reduction sync dominates). The trajectory kernel's
  **decision scan** is warp-wide in WAVE only (bit-identical picks; +12-19% there; in scalar
  the bitmap made the serial scan L1-cheap and the screen LOSES; `GPAP_VSCAN=0` disables;
  SOFT path stays scalar).
- **Packed field split** (per formula, in `BCPEngine.__init__`): 16|16 for 2N ≤ 65535;
  **8|24 ("wide") for 65535 < 2N ≤ 0xFFFFFF, max clause len ≤ 255** (the big instances — covers
  them, geomean 1.31× on c50 unrolls); anything larger is REJECTED with a ValueError (no unpacked
  fallback any more). One kernel source with a `SHIFT` constexpr — `mono` (1-byte) lost to wide
  and was deleted (see docs/findings/INSIGHTS.md).

## Running

```bash
scripts/build_core.sh                 # build libgpap (kernels AOT + cmake); `gpap info` verifies
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
(`--mine on` forces). **At `--target rare` the encoder is PER-INSTANCE** (2026-07-14, same
engine, 3-seed): ABC wins 1.8-5.7× on most and unlocks b14 (0→24 models/s), but the rare-pin
success can COLLAPSE under dc2/fraig (b13 1.00→0.00 succ = 0.0001×, s386 0.22×, s1196 0.62×)
— don't report suite geomeans across encoders without the split (see INSIGHTS).

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
hooked in-process OR `--sampler cmsgen`, SPUR reference; defaults are the CMSGen paper's
ε=0.3/η=1.8/δ=0.1 ⇒ 1,872 samples/instance, 38× cheaper than ε=0.1/η=0.9). chi2 sf is native
(`metrics._gammq`) — no scipy dep.
**Uniformity FUZZING against an exact oracle** (`eval/fuzz_uniformity.py`, 2026-07-30, full
writeup `docs/findings/UNIFORMITY_FINDINGS.md`): generates small random CNFs / random Tseitin
circuits whose solution set is enumerated EXHAUSTIVELY (solver-free oracle; the circuit generator
proves its own encoding over the whole truth table), then runs GPAP and CMSGen on identical files
with negative + positive **controls** (exact-uniform must not be flagged; a known TV must be —
these calibrate the instrument, sensitivity floor TV≈0.05 at n=100|S|×3 seeds). Two settled
results: **B ≥ |support| ⇒ exactly uniform** (rejection sampling, since GPAP discards conflicted
trajectories) and **`soft` is Barbarik-REJECTED 5/5** while b8/b1/CMSGen are Accepted 5/5 — the
bandit picks configs on models/s with no uniformity term, so `soft` winning was silent ⇒
**`soft` was REMOVED from CIRCUIT_PORTFOLIO 2026-07-30 and is now opt-in via `--allow-soft`**
(`gpap sample`/`gpap bench`; prints a warning; `pipeline.SOFT_CONFIG`).
⚠️ **OPEN DECISION — the throughput price is BIG and is parked, not settled** (user, 2026-07-30:
"that's a big loss, keep it for now, we'll think about it later"). Profiled on all 19 seq benches,
abc:fast, T=25, target rare: **geomean 0.410 ⇒ 2.44× slower**, median 0.788 (loss is CONCENTRATED —
soft won the bandit on 7/15 non-vacuous), and **two CAPABILITY losses: b11 11,269 models → 0,
b12 50,434 @ 11k/s → 789 @ 27/s**. Re-measure with `eval/profile_soft_portfolio.py` (encodes once,
runs both portfolios in one process, alternates arm order — clock drift makes any other method
meaningless). Sketched middle path, NOT implemented: let the bandit fall back to soft only when
every uniformity-safe config yields ~nothing (exactly the b11/b12 signature). Gated by
`tests/test_uniformity_fuzz.py` (<1 s; soundness at every B + the B≥|support| property). **2-wise coverage** (CMSGen-paper spread metric, NOT uniformity) is opt-in via
`gpap bench --twise` (auto-bumps n_cov to 2000) and vectorised (one matmul; the old O(S²·n)
pair loop dominated the quality pass).
Sample output = compact support projections (1 bit/support var; GPU-projected so only the bit
matrix leaves the device). `--out-format bits|npy|dimacs`, `--max-samples` cap (default 100k),
`--stats` for the info JSON. Sound: every row extends to a full model (test_output.py). Circuit
CNFs embed a `c wire <var> <input@cycle>` map (bench.py) so output columns are named by wire.

Benchmark suite driver (`gpap/benchmark.py`, schema-3 JSON; thin table over
`gpap.sampling.pipeline`; the glob may mix CNFs with circuits — .bench/.aig/.aag are converted
first, encode wall reported per instance as t_enc, never folded into either sampler's rate):
```bash
gpap bench 'benchmarks/circuits/bench/seq/*.bench' --encoders tseitin abc:fast --target rare \
    --cmsgen ~/.cache/gpap/bin/cmsgen --output .experiments/seq.json --resume
```
- **`--encoders tseitin abc:fast`** runs EVERY circuit through each encoder back-to-back in ONE
  process (same GPU state) → one row per (instance, encoder) + a paired A/B summary. Single
  `--encoder` still works.
- **Never dies mid-sweep**: a failing cell becomes an `error`/`skipped` row and the run continues;
  rows are written to `--output` after each one and `--resume` skips what is already there
  (the Colab-disconnect story). Out-of-envelope formulas (`gpap.engine.packed_envelope`) and
  `--max-clauses/--max-vars` oversizes are skipped WITH the reason, never a raise.
- **Engine provenance in every row** (`engine`: packed split 16|16 vs 8|24, wave/scalar propagate,
  K/K_fused) + a `krn` column — the evidence a number came from the sum-trick kernel in-envelope.
- **CMSGen fairness**: build it with `scripts/build_cmsgen.sh` (upstream + the in-repo patch
  `scripts/cmsgen_gpap_eval.patch`). GPAP's timed run emits nothing, stock CMSGen writes every
  sample as a full N-var assignment — measured 8.2× of its wall on 27.sk_3_32 (0.42s vs 3.6s /
  373 MB). The driver auto-detects the patch's `--nowrite` and times THAT; a stock binary is
  used as-is but the row is marked `cms_mode="write"` and a warning printed. The patch's
  `--statsevery` also gives CMSGen's own props/decisions ⇒ the `Cdec/s`/`Cimpl/s` columns.
  **Pre-2026-07-27 GPAP/CMSGen ratios include the emission handicap** (docs/BENCHMARKING.md §4b).

**Full experimental sweep** — [`docs/EVAL_RUNBOOK.md`](docs/EVAL_RUNBOOK.md). Terminal-driven
(Colab reaps idle notebook cells), staged and resumable:
```bash
eval/rivals_setup.sh                  # abc + patched cmsgen + DEMOTIC/Recurrent/HTS checkouts
eval/run_full_eval.sh seq             # or comb | sk | epfl | hwmcc | rivals | plots | all
```
Each circuit stage sweeps encoders × targets(rare,random) × depths(25,50) into
`.experiments/full_eval/<gpu>/`; `eval/eval_plots.py` turns those into the encoder-A/B,
vs-CMSGen, work-rate, amplification and rival figures + `summary.md` (which lists every
skipped/errored instance — read it before quoting a geomean).
Rival harnesses: `eval/rivals_demotic.py` (`--mode native` = their ISCAS-85 pins, `--mode miter`
= our rare-target instance as a single-output miter; sanitizes wire names + BUFF→BUF because
DEMOTIC exec's generated Python), `eval/rivals_recurrent.py`, `eval/rivals_hts.py`.
**Same circuit on both sides** (`--circuit-variants raw abc`, eval/rival_common.py): the
circuit-level rivals unroll `.bench` themselves, so an ABC-encoded GPAP arm must not face a rival
holding the raw circuit — `abc_backend.simplify_bench_abc` ABC-optimizes the SEQUENTIAL AIG and
writes it back as a primitive-gate .bench (via AIGER; abc's `write_bench` emits LUT/DFFRSE that
nothing here parses), gated by `bench_equiv_check` (identical stimuli ⇒ bit-identical accept
words). The post-unroll ABC pass has no sequential equivalent — stated, not hidden. Measured:
DEMOTIC gains 15-25% from it on ISCAS-85 and c1355 flips 0.5×→1.5× in our favour.
Cross-GPU K-scaling: `eval/colab_sampling_benchmark.py` + `notebooks/gpap_colab_sampling.ipynb`.

## Ground rules (do not re-litigate — see docs/findings/INSIGHTS.md)

- Runtime per-batch memory PROBES are unreliable → fixed 4×-floor K stays.
- This box SPILLS to host RAM (never OOMs) → validate every K change under the nvidia-smi
  kill-switch; check the power state first (a 40 W cap / battery clocks ~halve every rate).
- `mono` kernel (0.93× vs wide), `orac` circuit config (0/64 wins), full/soft global lookahead:
  measured out. **soft is now OPT-IN (`--allow-soft`), not a default portfolio member** — it is
  Barbarik-rejected 5/5 (2026-07-30), so a throughput bandit must not be able to choose it for a
  user who asked for a sampler. It stays reachable: regime-dependence rule, it still wins
  tight/deep circuits on success rate.
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
