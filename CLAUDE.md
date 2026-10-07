# GPAP — GPU-Parallel (partial) Assignment Propagation & CircuitSAT Sampling

GPU-accelerated BCP over K independent partial assignments, and a CircuitSAT / projected
**solution sampler** built on it. The engine is the novelty; sampling is the flagship app.
Deep material lives in [`docs/`](docs/); keep this file short and current.

**The engine is one C++ library, [`libgpap`](cpp/)**, shared by Python (`gpap.engine`, ctypes)
and C++ (`#include <gpap/gpap.hpp>`). Kernels are Triton (`cpp/kernels/bcp.py`, the single
source), AOT-compiled into the library by `cpp/tools/build_kernels.py`. Build with
`scripts/build_core.sh` (cmake, C++17, the NVIDIA driver, triton at build time; no CUDA toolkit,
no nvcc). Runtime dependency: `libcuda` alone. CUDA is required; there is no CPU path.

**Code carries the rule; `docs/` carries the evidence.** A comment says what the code does and
why it is correct. The measurement that settled it (date, GPU, per-instance table) lives in
`docs/findings/`; never put a dated number or a card name into a source file.

## Core design: no clause scanning

- **Negated occurrence lists** (CSR, int32): `occ[l]` = clauses where `~l` appears; setting `l`
  TRUE falsifies `~l` in exactly those clauses.
- **Clauses are segregated by length; only len ≥ 3 carries per-row state** (`num_long`).
  **Binary clauses are implication edges in occ** (payload = the other literal, the prefix of
  each segment, split at `occ_mid[l]`): their propagation is the one bitmap `atomic_or` of the
  forced push. Units live in the flush bitmap. Fixpoints are unchanged (confluence);
  `sat_rows` keeps (n, M_total) by bitmap synthesis, `packed_rows` is (n, num_long).
  docs/findings/BINARY_OCC_FINDINGS.md.
- **One packed int32 word per long clause, batched over K rows**: low `SHIFT` bits =
  `sat_count` (non-falsified literals; `==0` conflict, `==1` unit), high bits = sum of surviving
  literal indices mod 2^(32-SHIFT). At `sat_count == 1` the sum field *is* the forced literal.
  The update per falsified literal is one subtraction `st -= 1 + (neg << SHIFT)`, i.e. one
  `atomic_add` whose return value gives race-free unit/conflict detection (the WAVE kernel's
  trick).
- **Literal indexing (2N)**: `+v → 2(v-1)`, `-v → 2(v-1)+1`, negation `l ^ 1`. The assignment
  is a bitmap (bit l of int32 words; l and ~l share a word, so one `atomic_or` answers
  duplicate and contradiction). Accessors materialize copies off the hot path.
- Correctness story: BCP fixpoints are confluent, so kernels are A/B'd against a pure-Python
  reference (`tests/reference/bcp.py`), and every emitted model is externally verifiable.

## Layout

```
cpp/                       # THE ENGINE CORE: libgpap (C ABI + header-only C++ wrapper)
  include/gpap/gpap.h      # the C ABI: formula, engine, selections, readout, counters, sizing
  include/gpap/gpap.hpp    # C++, two tiers: Sampler/AssumptionChecker over Formula/Engine
  src/formula.cpp          # DIMACS tokenizer + CSR occ + packed init (THE builder)
  src/engine.cpp           # state ownership, reset/flush cache, both entry points
  src/readout.cpp          # selections, on-device soundness, bit gathers
  src/calibrate.cpp        # K sizing, VRAM model, THE wave/scalar dispatch rule, trail cap
  src/sample.cpp           # THE sampling loop (both faces): chunks, bandit, valves, buckets, RNG
  src/probe.cpp            # combinatorial probing: every k-subset of a literal set, paged + budgeted
  src/check.cpp            # batched checking: THE chunk loop of both AssumptionChecker faces
  src/runtime.cpp          # cubin registry + launches
  kernels/bcp.py           # the 2 BCP kernels (fixpoint + trajectory; SHIFT=16|8; scalar/WAVE)
  kernels/helpers.py       # support kernels
  tools/build_kernels.py   # AOT: kernels -> embedded cubins (`vscan_is_read` trims duplicates)
  examples/                # sample.cpp, assumptions.cpp, probe.cpp, c_api.c
examples/                  # runnable Python examples for both faces
gpap/
  mining.py                # entailed-clause mining: the STRATEGY over the core's probe primitive
  engine/                  # the Python face of the core (app-agnostic)
    _core.py               # ctypes binding (private)
    formula.py             # CnfFormula over the core's builder (reference: tests/reference/formula.py)
    bcp_engine.py          # BCPEngine: marshalling + accessors
    calibrate.py           # K sizing, VRAM model, gpu_mem_available
    checker.py             # AssumptionChecker (marshalling; the loop is cpp/src/check.cpp)
    probe.py               # LiteralProbe (the sweep is cpp/src/probe.cpp)
  verdict.py               # sound_verdict / merge_sound_check: a stdlib-only leaf so the table
                           #   generators read a row without importing torch (pipeline re-exports)
  sampling/                # THE FLAGSHIP
    sampler.py             # GPUSampler: marshalling over gpap_sampler_*, ELS reconstruction, projection fill
    pipeline.py            # sample_file / sample_clauses: the e2e flow, portfolios (SamplingConfig),
                           #   sound_verdict/merge_sound_check (THE tri-state readers)
    preprocess.py          # unit_closure / equivalent-literal substitution (ELS)
    output.py              # compact sample writers (bits / npy / dimacs)
  circuits/                # bench.py, aiger.py, aig2cnf.py (native), abc_backend.py (optional),
                           #   simulate.py (rejection baseline, rare targets, sim-check)
  cli.py, benchmark.py     # `gpap` subcommands; the `gpap bench` suite driver + coverage metrics
eval/                      # the experiment layer (raw data lives OUTSIDE the repo: $GPAP_RESULTS)
  run_full_eval.sh         # staged, resumable campaign driver; eval_plots.py makes the figures
  rivals/                  # rivals_{demotic,recurrent,hts}.py + rival_common.py (THE fairness
                           #   policy: circuit variants, the parallelism grant, the GPAP-arm cache)
                           #   + check_emission_symmetry.py + extract_grants.py + grants/
  uniformity/              # barbarik_gpap.py / _screen.py / _table.py, fuzz_uniformity.py, metrics.py
  artifact/                # paper_tables.py (THE table generator; every rule lives there once),
                           #   verify_artifact.py (publishability gate), slide/soft tables,
                           #   reproduce.sh (THE artifact entry point), estimate_wall.py,
                           #   compare_campaign.py (the cross-card gate), subset.json, make_artifact.sh
  colab/                   # cross-GPU drivers + colab_supervisor.py (survives the ~1 h VM lifetime)
scripts/                   # build_core.sh, build_abc.sh, build_cmsgen.sh (+ cmsgen_gpap_eval.patch),
                           #   fetch_benchmarks.sh, perf_ab_trees.py
docs/findings/             # INSIGHTS.md (settled-results index) + *_FINDINGS.md
tests/                     # pytest; circuit CNFs generated from benchmarks/circuits/bench/ at test time
.gpap/                     # DISPOSABLE MACHINERY, one root for it: built/fetched tools, sources,
                           #   benchmark sets, and `tmp/` — ALL scratch, every layer. Gitignored,
                           #   not $HOME, not /tmp; `$GPAP_HOME` moves it. `gpap/paths.py` owns it
.experiments/              # a run's output while it is produced, then hand-copied to $GPAP_RESULTS
                           #   (the record, outside the repo). Both owned by `eval/paths.py`.
                           #   Nothing else names these three; shell asks. gpap/paths.py says why
```

## Engine contract: two entry points (`gpap/engine/__init__.py`)

1. **Batched assumption checking**: `reset(K)` + `apply_and_propagate(K×L decisions)` →
   conflict / assignment in one fixpoint launch. Two one-call faces: `AssumptionChecker`
   ("check these sets") and `LiteralProbe` ("check every k-subset of this pool within this
   budget"; the primitive behind mining). The sweep lives in the core (`gpap_probe_*`), so both
   faces probe the same candidates in the same order; literals are DIMACS on both.
   - `max_tuples` is a work budget: a width with more candidates probes that many and reports
     `exhausted=False`; partial results are entailed regardless and are kept. It binds at width 1
     too, spent across that width's fixpoint rounds, so width 1's `candidates`/`probed` are sums
     over `rounds[1]` and never comparable to the pool size. Never sum `candidates` across widths.
     A per-width dict must give width 1 an explicit `0` (see `pipeline.STAGE1_MAX_TUPLES`); the
     fallback `DEFAULT_MAX_TUPLES` would cap the backbone silently. The budget is checked against
     `gpap_probe_count` before anything is generated. The cost is the probe, not the enumeration:
     do not "optimize" the enumerator.
   - **CLEAN mode** (`gpap_engine_apply_clean`) runs the fixpoint and restores its own touched
     words from the baseline inside the launch; `reset()` goes lite when rows are provably clean.
     Probe sweeps and flags-only `check_sets` ride it; anything that reads post-apply state
     (assignments, sat readout, `recache_flush`, selections) must use plain `apply`. `GPAP_CLEAN=0`
     is exact legacy. Gated by tests/engine/test_clean_mode.py; docs/findings/CLEAN_MODE_FINDINGS.md.
   - **Redundancy is filtered on the host, at three levels** (`mine_entailed(drop=)`, `gpap mine
     --drop`): `none` / `subsumed` (default) / `derivable`. The default is deliberately not the
     strongest: a derivable clause still shortens propagation and the sampler is propagation-bound.
     A clause does k jobs and all must be covered before it can go; propagation is not
     contrapositive-complete. The subsumption reference is the formula reduced by the engine's
     baseline, not the raw input (else the output is kernel-dependent; gated by
     `test_mining_is_kernel_invariant`). There is no filter in the engine. docs/findings/MINING_FINDINGS.md.
2. **Fused trajectory sampling**: `reset(K)` + `run_trajectory(order, n_support, B, seed, soft=…)`,
   the whole randomized decide+propagate loop in one launch. **The chunk loop over it is also
   core** (`gpap_sampler_*`, `cpp/src/sample.cpp`): chunks, the config bandit, the stop valves,
   per-config kept-row buckets, delivery/verify accounting, and the core's own RNG. Both faces
   are marshalling; at one seed, one config and one batch they draw identical models
   (tests/sampling/test_face_parity.py, tests/sampling/test_sample_core.py).
   - **K is not part of the seed**: it is the analytic bound over live free VRAM, so a narrower
     run's models are a prefix of a wider one's. Pin the batch before comparing counts
     (`pin_batch` / `gpap_sampler_set_k` / `--k`).
   - The faces agree on defaults, not only on the loop: decision batch 8 everywhere (gated by
     tests/cli/test_cli_contracts.py, which reads the C++ default out of the header). The C++
     `retry_with_lookahead` is OFF by default; turning it on runs the Barbarik-rejected soft
     polarity, the same choice `--allow-soft` gates in Python.
   - Rows recorded before the RNG moved into the core replay under torch seeds and are not
     seed-comparable (docs/BENCHMARKING.md §3).

`reset(K)` broadcast-restores the construction-time post-units-flush cache.

**Hot-path readout is the selection API**: `make_pair_mask(vars)` once, then per chunk
`select_complete(mask_id, slot) -> count` (the completion test runs on device; only the count
crosses the bus), `check_sound(slot)` (a per-row device flag) and `var_bits(slot, cols, n)`.
The tensor-shaped accessors (`assignment`, `conflict`, `active`, `sat_rows`) are copies for
tests. App code goes through these; do not reach into row state. The engine picks the packed
split per formula (16|16 for 2N ≤ 65535; 8|24 "wide" for 2N ≤ 0xFFFFFF with max clause length
≤ 255) and rejects formulas outside both envelopes.

## K sizing and kernel dispatch (the memory model that keeps biting)

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set in `gpap/__init__.py` before torch
  inits CUDA. Do not remove it.
- **K = min(analytic, cap)** on the core sampler handle (`GPUSampler.K` reads it; assignment =
  `gpap_sampler_set_k` = the pin). The analytic bound is `compute_k_safe` over `gpap_row_bytes`,
  the engine's exact per-chunk allocation list (4·M_long packed words, the bitmap, the trail,
  flags), verified against measured VRAM. Runtime memory probes are unreliable; do not resurrect
  one.
- **`GPAP_VRAM_BUDGET_MIB` caps the budget, clamped once inside `gpap_mem_available`** so every
  consumer sees it, C++ callers included. THE RIVALS ARE SIZED BY THE SAME CALL
  (`rival_vram_budget` -> `gpu_mem_available`), so one cap covers both sides of a head-to-head
  and there is no second memory model to keep in step. Unset = today's behaviour. It exists because a cross-card reproduction can flatter us in both directions: K is
  VRAM-bound below the cap on large formulas, so a bigger card grows OURS while a rival pinned by
  a grants file stays fixed; and a smaller card walks the rival's rung down. A ceiling, never a
  floor. `eval/artifact/reproduce.sh --vram-cap l4` is the reproduction default;
  docs/findings/ARTIFACT_FINDINGS.md.
- **The cap is one constant on every GPU**: `kBatchCap = 262,144` (`cpp/src/calibrate.cpp`,
  `engine.calibrate.K_CAP`, `--k-cap auto`). It is a saturation limit, not a safety one, and it is
  hygiene, not a throughput lever: success is machine-independent and the work rate tracks SM
  count (docs/findings/CROSS_GPU_FINDINGS.md). Device-independence is gated by tests/engine/test_k_cap.py.
  Two cross-GPU traps: a free bandit picks differently per machine (always pin the config), and
  a `--n-time` that finishes in one chunk measures first-chunk overhead.
- **`gpap_trail_cap(N)` in `cpp/src/calibrate.cpp` is the one definition** of the trail term:
  `N + 32`, i.e. exactly N (every push flips a bitmap bit; l/~l share a word) plus one wavefront,
  because the wave paths append a whole wavefront with no bound test. Keep the slack in step
  with `WAVE_W`. There is no ceiling on B; `B ≥ |support|` is the exactly-uniform configuration.
- **Read `smp.batch_size`; set it with `smp.pin_batch(n)` / `with smp.pinned_batch(n):`.**
  `batch_size == K`, stamped into every row as `engine.batch`. Gated by tests/engine/test_k_cap.py.
- Any retune of `compute_k_safe` is a K change: re-validate under the nvidia-smi kill-switch and
  re-check dispatch, since the batch feeds the dispatch rule's small-batch term.
- Wave and scalar agree exactly on models and decisions; the implication counter differs on
  conflicted rows (which clause empties first depends on propagation order). Gate on models.
- **Kernel dispatch lives in the core** (`GPAP_MODE_AUTO`, which is 0, resolved in
  `gpap_engine_reset`), so every consumer gets it, including raw C ABI callers. The rule is
  `gpap_dispatch_wave(M, batch, device)`: warp-wide WAVE propagate when `M ≥ 2048` or
  `batch ≤ resident_warps`, scalar otherwise. It keys off M, the clause count, never the batch
  alone (the batch is a constant at the cap, so a batch-only rule misclassifies every mid-size
  formula). `GPAP_KERNEL=scalar|wave` forces, applied only where AUTO resolves. `GPUSampler` pins
  the mode to the chunk batch so small resets cannot switch kernel mid-measurement. W is the AOT
  constant `WAVE_W` (32; both ways of raising it measured out). The WAVE lane→segment schedule
  is shuffle-based with zero shared memory (docs/findings/WAVE_SHUFFLE_FINDINGS.md); the
  trajectory kernel's decision scan is warp-wide in WAVE only (`vscan=-1` disables; the soft
  path stays scalar). Calibration and the paper's kernel A/B: docs/findings/KERNEL_AB_FINDINGS.md.

## Running

```bash
scripts/build_core.sh                 # build libgpap; `gpap info` verifies
pytest -m "not gpu and not slow"      # host-side smoke; engine tests need CUDA and self-skip
pytest                                # full suite (slow = deep unrolls)

gpap info
gpap sample formula.cnf -n 50000 --out samples.out --stats run.json
gpap mine formula.cnf --width 1 2 --pool all --seconds 10 --out stronger.cnf
gpap sample b10.bench --cycles 25 -n 50000 --out stim.out    # .bench: columns = wire@cycle
gpap sample b10.bench --cycles 25 --encoder abc              # ABC-encoded CNF (2-8x fewer clauses)
gpap bench 'path/*.cnf' --cmsgen /path/to/cmsgen --n-cov 5000
scripts/build_abc.sh                                         # optional: pinned abc -> .gpap/bin
```

## Sampling defaults

| | default | opt out |
|---|---|---|
| ELS (equivalent-literal substitution) | on | `--no-els` |
| entailed-clause mining | on above `MINE_MIN_CLAUSES`, bounded by `MINE_SECONDS` | `--mine off` |
| mined clauses | kept only if they pay (below) | |
| soundness re-check of emitted models | off everywhere | `--verify-sound` turns it on |

- `gpap sample`/`gpap bench` default to `--portfolio circuit` for every input, including plain
  CNF, where it costs Barbarik certificates on two sk_ instances (the `circuit` portfolio has no
  `s1`). Not changed, because it would move a default every recorded CNF row was produced
  under. docs/findings/UNIFORMITY_FINDINGS.md §4b.
- `MINE_SECONDS` is honoured to chunk granularity and is a wall-clock deadline from mining
  start, so it bounds total pre-sampling overhead. `--mine on` forces past `MINE_MIN_CLAUSES`.
- **Soundness re-check is opt-in and, when on, covers every emitted row** (`verify_sound=`,
  `--verify-sound`, C++ `SampleOptions::verify`; campaigns set `VERIFY_SOUND=1`). Reported as
  the tri-state `sound_check`: `"ok"` | `"UNSOUND"` | `"off"`, plus `unsound_rows` and
  `sound_rows_checked`. `"off"` also covers requested-but-never-reached (a run that completed no
  row never called it); `sound_rows_checked` is the evidence. `pipeline.sound_verdict` is the
  reader (legacy `gpu_sound` included) and `merge_sound_check` the fold, where UNSOUND
  dominates; every consumer imports them. Its wall `verify_sound_s` is billed to `phases_diag`,
  outside every rate, and the pipeline subtracts it on the pinned path too.
- **Keep-if-it-helps**: mined width ≥ 3 clauses enlarge `num_long` and shrink K (width-2 ones
  are K-free); `K x M` is roughly constant so models per chunk = `K x success` is the throughput
  proxy. Clauses that do not beat the pre-mining sampler are dropped; they are kept when the
  un-mined warmup completed nothing. `mined.kept` / `mined.gain` in `--stats`.
- **Delivery**: sample output is compact support projections (1 bit per support var, projected
  on the GPU), `--out-format bits|npy|dimacs`, capped by `--max-samples` (default 100k). Rows are
  collected by the timed run (`sample(out_cap=…)`), bucketed per config with only the winner's
  returned (one sampler, never a mixture), and the file holds `min(-n, --max-samples,
  delivered)` rows. Delivery draws from `_proj_gen`, never from the sampling stream, so `--out`
  cannot change which models a seed finds. `out_project_s` is excluded from every rate.
  `phases` sums to `wall_total_s`; the opt-in readouts (`--n-cov`, `--verify-sound`) are billed
  to `phases_diag`, so `sps_e2e_total` is the same number whether or not a self-test ran.
  docs/findings/CLI_DELIVERY_FINDINGS.md.
- **`--n-time` is a delivery target, not a guarantee**: a run overshoots to the end of a chunk
  and can fall short when a valve fires (`--time-limit`, the zero-progress stop; flagged by
  `chunk_capped`). Quote the delivered count. `--time-limit 0` = no cap on `gpap sample`; `gpap
  bench` refuses 0 because there the flag is also the rival's timeout.
- Preprocessing is inside the honest rate: `samples_per_s` is the winning config's steady-state
  throughput and excludes parse, construction, ELS, warmup and mining; `sps_e2e_total` includes
  them. CMSGen's rate is samples per subprocess wall, which already carries its setup, so Table
  1 prints both clocks plus `pre%`; quote `e2e` for a delivered-samples claim. Circuit encoding
  is in neither (`t_enc`).

## Circuits

- **Inputs**: `.bench` and AIGER `.aig`/`.aag` (native parse, simulate, unroll; latch resets 0/1
  honoured, `x` rejected; combinational clamps to T=1). Circuit CNFs embed a `c wire <var>
  <input@cycle>` map so output columns are named by wire.
- **Rejection baseline and targets** (`simulate.py`): `--sim-baseline` = random inputs +
  bit-parallel simulation, the CNF-free rival (perfectly uniform; rate = p_accept × sim
  throughput; reports amplification = success/p_accept). `--target random` (default) pins the
  outputs of one random sequence and is size-biased toward rejection-easy patterns; `--target
  rare` pins the rarest pattern in a 262k-sequence sim batch (observed ⇒ reachable ⇒ SAT).
  Headline circuit comparisons use `rare`. `--sim-check` simulates every emitted row through
  the original circuit: encoder-independent end-to-end soundness.
- **Encoder** (`--encoder tseitin|abc|auto`, default `tseitin` so results never shift with
  ABC's presence): `abc` = strash + `zero` + `frames -F T -i` + dc2/fraig, then Tseitin of the
  AIG (2-/3-literal clauses). Same projected solution space (gated by tests/circuits/test_abc_encoder.py);
  `zero` is mandatory (without it the T=0 register state becomes free PIs; reset-1 AIGs are
  refused on the abc route). AIGER, not `write_cnf`, is the route because it keeps the PI→var
  map. `--abc-opt full|fast|auto`: post-unroll fraig is the whole encode cost and the whole
  cross-frame win. Encode wall is `encode_s`, outside the e2e rate. Mining is worth more on ABC
  CNFs than on Tseitin ones, so it stays on for both. At `--target rare` the encoder is
  per-instance (ABC can collapse a rare-pin success); never report cross-encoder geomeans
  without the split.
- **Same circuit on both sides** (`--circuit-variants raw abc`, `rival_common.py`): the
  circuit-level rivals unroll `.bench` themselves, so the ABC-optimized sequential AIG is
  written back as a primitive-gate `.bench` (via AIGER) and checked by `bench_equiv_check`. The
  post-unroll ABC pass has no sequential equivalent: stated, not hidden.

## Uniformity lives in `eval/uniformity/`, not in the library

A uniformity number is a claim about a distribution; the honest ones need the whole solution
set (enumeration) or conditional sampling (Barbarik), neither of which `gpap sample` can do on
the way past. `metrics.py` holds the statistics; `fuzz_uniformity.py` is the exact-oracle
campaign; `barbarik_gpap.py` the certificate harness (GPAP in-process or `--sampler cmsgen`,
SPUR reference). The rules (docs/findings/UNIFORMITY_FINDINGS.md §4b):
- Two parameter sets, never pooled (`barbarik_table.py` refuses): the CMSGen paper's
  ε=0.3/η=1.8/δ=0.1 (1,872 samples) and barbarik's own ε=0.1/η=0.9/δ=0.05 (70,596 samples).
- The guarantee is asymmetric: Reject is unconditional; Accept is "not caught", never "uniform".
  CMSGen's own Accept was tuned against this test, so parity is the honest framing.
- A table is evidence only with both controls (`--sampler spur` must Accept, `--config soft` must
  Reject) and the eligibility screen (`barbarik_screen.py`): CM_test rejects its own reference
  where the declared `c ind` is not an independent support. Bucket on the control's verdict.
- Barbarik's kernel is adversarial for a rejection sampler, so an unfilled order is "no verdict",
  not a Reject. `k_cap` is a batch size, not a distribution knob.
- Settled: `B ≥ |support|` is exactly uniform (rejection sampling); the shipped config is
  Accepted 64/66 with zero rejections (CMSGen 66/66); `soft` is rejected on 64 of 65 circuit
  instances and is therefore opt-in (`--allow-soft`), at a large parked throughput price
  (`eval/profile_soft_portfolio.py`; the user: keep it for now). The `l2_ratio` collision check
  is gross-skew only. 2-wise coverage (`gpap bench --twise`) is a spread metric, not uniformity.

## Benchmark suite and CMSGen fairness (`gpap/benchmark.py`, schema 4)

- `--encoders tseitin abc:fast` runs every circuit through each encoder in one process: one row
  per (instance, encoder). A failing cell becomes an `error`/`skipped` row; rows are written
  after each one and `--resume` skips what is there (and refuses to append rows with a
  different arm than the file holds). Engine provenance in every row (`engine`: split, kernel,
  K, cap; `krn` column).
- **CMSGen**: build with `scripts/build_cmsgen.sh` (upstream + `scripts/cmsgen_gpap_eval.patch`).
  GPAP's timed run emits nothing, so CMSGen is timed with the patch's `--nowrite`; a stock binary
  is marked `cms_mode="write"`. `--statsevery` gives its own counters. **`--maxtime` is what
  makes a budgeted count real**: a run killed on the budget reports the last multiple of
  `--statsevery`, and a floored zero scores as our win; `--maxtime` makes it exit normally with
  an exact count (`cms_count_exact`; `verify_artifact` fails a floored zero). It bounds the
  sampling wall, so `cms_sps` pairs with our steady state and `cms_sps_e2e` with `wall_total`.
  docs/findings/RIVAL_FAIRNESS_FINDINGS.md §7.

## The paper protocol: v1.0, frozen 2026-09-06

**Reproducing it is `eval/artifact/reproduce.sh`** (check/setup/tables/smoke/subset/full), which
invokes `run_full_eval.sh` for every measurement and adds only a cell set, an estimate
(`estimate_wall.py`, read off the recorded rows), the cross-card gate `compare_campaign.py`, and
two shortcuts the frozen driver does not take: recorded rival failures pinned rather than
re-derived (`--remeasure-failures` restores) and the memory cap above. ARTIFACT.md is the guide;
docs/findings/ARTIFACT_FINDINGS.md the evidence.

`docs/EVAL_RUNBOOK.md` §0 is the statement; RIVAL_FAIRNESS_FINDINGS.md §17 the record and the
considered-not-adopted list; §17b the presentation revision of 2026-09-07. In one line: the
rivals' own published benchmark sets (the full set is the headline and the only population
printed), each under the rival's own pins and our rare pin, T=25 with T=50 as an appendix, one
config per rival (its shipped script setting), the batch = the largest rung of the ladder it
completes walked down from 30,000,000, one circuit variant (`abc`) everywhere, 1,200 s wall +
1,200 s init budget, and on every cell GPAP (abc:fast CNF of the same circuit + pins, the
shipped `circuit` portfolio) and CMSGen on that very CNF. Three tools, one constraint, one row.
Table 2 has one clock, the rivals' own metric, and no e2e column. The grant is part of the
record (`eval/rivals/extract_grants.py` → `eval/rivals/grants/`; `--grants` pins a re-run).
Rival knobs come from the rival or a stated rule, never from us (`RIVAL_STEPS` from their own
artifacts; a tie goes to the value better for the rival; not swept). **A review after the
freeze fixes bugs and prose only**, never a metric, config, screen, budget or cell set; log
proposals in §17. The family stages (`eval/run_full_eval.sh seq|comb|sk|epfl|hwmcc`) are
supplementary. `eval/eval_plots.py` writes the figures and `summary.md`, which lists every
skipped/errored instance; read it before quoting a geomean.

## Ground rules (settled; do not re-litigate without new evidence — docs/findings/INSIGHTS.md)

- **An ABI break moves the soname; an additive change does not.** libgpap is 0.3.0 / SOVERSION 1.
  Keep this line current.
- **`{0}` must equal NULL on every options struct**: every field's zero is its default, and
  where the useful default is "on" the field is named for the opt-out (`same_var`). A rename is a
  compile error for out-of-tree callers; a flipped meaning under the same name is not.
- **A cubin the kernel cannot tell apart is not shipped**: `build_kernels.vscan_is_read` and
  `engine.cpp::traj_key` are one rule; gated by tests/engine/test_kernel_table.py. Cubin byte-identity
  is not a no-change gate (line info is embedded); compare the executable sections.
- **Key a results cache on `gpap_build_id()`, never `gpap_version()`.**
- Runtime per-batch memory probes are unreliable; this box spills to host RAM rather than
  OOMing, so validate every K change under the nvidia-smi kill-switch and check the power state
  first (a power cap or battery clocks halve every rate, and the clock moves within a run).
- Measured out, do not retry: `mono` kernel, `orac` config, full/soft global lookahead, in-kernel
  retry-on-conflict, clause renumbering, W=64. `soft` is opt-in only.
- **Vacuous-pin instances** (at T=25 `b02`, `b05`, `b07`, `s526`; near-vacuous `b01`, `b09`) and
  **zero-search cells** (`n_models > 0` with zero decisions and implications per second: the
  support is free after propagation) are degenerate; both are screened by `paper_tables` and
  flagged by `verify_artifact`. Screen first, then bucket.
- **The zero ledger is symmetric and must add up.** CMSGen-zero → `ratio=inf`, GPAP-zero →
  `ratio=0.0`, both-zero → `ratio=None` (the last two are falsy; never branch on `ratio` alone).
  Invariants: `inst = n + CMSGen=0 + GPAP=0 + excluded` (Table 1) and `cells = n + rival=0 +
  GPAP=0 + failed + excluded` (Table 2), pinned by tests/eval/test_paper_tables.py. Every reason a
  cell carries no ratio is the one `excluded` column, itemised by name (`excl_names`); there is
  no thin-denominator floor (a tool that scored anything is scored; cells under
  `THIN_DENOMINATOR = 30` events are named in the caveats); no `paper set` column. Table 1 and
  3 read `rival_cms_cells`, Table 2 `rival_cells`, and every figure imports them. Absent is not
  zero: a row that never ran the CMSGen arm or whose subprocess errored is neither a win nor a
  zero (`paper_tables.cms_arm_ran`; `verify_artifact` fails a mixed or errored stage).
- **No best-of-encoders selection: one row per (family, encoder).** Selecting the encoder with
  the highest ratio lets the rival's number pick which of our runs is reported.
  RIVAL_FAIRNESS_FINDINGS.md §8/§9.
- **Propagation rate isolates the engine from the sampler**: the identity `samples/s ratio =
  (impl/s ratio) / (impl-per-sample ratio)` closes only when every column uses the same rows
  (Table 3). The spread in the headline is rejection cost, not engine speed.
- Measurement protocol: docs/BENCHMARKING.md. The no-change gate for a pure refactor is fixed K
  + one config + seed 5 ⇒ bit-identical model counts, within one seed era (§3). Rates alone are
  draw-noisy on low-success instances.

## Working rules for this repo

- Never `git commit` or `git push` unless asked that turn; do the work, report, wait.
- Raw results live in `$GPAP_RESULTS` (`../gpap-results`), never in the repo. The paper folder
  is `../gpap-results/paper/` (data copies, manifest, `regenerate.sh`).
- A campaign launches only on an explicit go; always `colab stop` when done; never
  `pkill -f colab_supervisor.py` (it matches itself); never put a kill-by-pattern and a launch
  in the same shell call.
- No soft-config data in the paper; uniformity claims via Barbarik only. The talk decks live in
  `$GPAP_RESULTS/paper/slides/`, outside the repo, and are out of scope for the paper work.
