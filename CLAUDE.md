# GPAP — GPU-Parallel Assumption Propagation & CircuitSAT Sampling

GPU-accelerated BCP over K independent assumption sets, and a CircuitSAT/projected **solution
sampler** built on it. The engine is the novelty; sampling is the flagship app. Deep material
lives in [`docs/`](docs/) — keep this file short and current.

**The engine is a C++ library, [`libgpap`](cpp/)** — ONE implementation that
Python (`gpap.engine`, via ctypes) and C++ consumers (`#include <gpap/gpap.hpp>`) share. The GPU
kernels are still written in Triton: [`cpp/kernels/bcp.py`](cpp/kernels/bcp.py) is the
single source of truth, AOT-compiled into the library by `cpp/tools/build_kernels.py` (cubins
verified BIT-IDENTICAL at SASS level to the JIT ones). Build: `scripts/build_core.sh` — needs
cmake + a C++17 compiler + the NVIDIA driver + triton at build time; **no CUDA toolkit, no nvcc**.
Runtime deps of the library: `libcuda` alone. CUDA is required; there is no CPU path.

## Core design — no clause scanning

- **Negated occurrence lists** (CSR, int32): `occ[l]` = clauses where `~l` appears. Setting `l`
  TRUE falsifies `~l` in exactly those clauses.
- **Clauses are SEGREGATED BY LENGTH — only len ≥ 3 carries per-row state** (`num_long`, 10-23%
  of M on circuit CNFs). **Binary clauses are IMPLICATION EDGES in occ** (payload = the OTHER
  literal, the prefix of each segment, split at `occ_mid[l]`): their whole propagation is the ONE
  bitmap `atomic_or` of the forced push (64-75% of circuit occ entries take it). Units live in
  the flush bitmap. Fixpoints/models unchanged (confluence); K ×2.2-2.3 where the analytic bound
  binds; reset/CLEAN-restore/probe-init shrink 4-10×; `sound_rows` checks bins+units via a
  device pair list. Measured: same-K kernels 1.42×, e2e 1.36× (4080), b12-c50 `gpap mine` 1.52×;
  docs/findings/BINARY_OCC_FINDINGS.md. `sat_rows` keeps (n, M_total) by bitmap synthesis;
  `packed_rows` is (n, num_long).
- **ONE packed int32 clause word per LONG clause, batched over K rows**: low `SHIFT` bits = `sat_count`
  (non-falsified literals; `==0` conflict, `==1` unit), high bits = **SUM of surviving literal
  indices mod 2^(32-SHIFT)** — at `sat_count == 1` the field **is** the forced literal (every
  removal exact mod 2^F, one literal < 2^F). Update per falsified literal is ONE subtraction
  `st -= 1 + (neg << SHIFT)` (count never borrows, sum wraps off the top) — which is also a
  single `atomic_add` whose RETURN value gives race-free unit/conflict detection (the WAVE
  kernel's whole trick; XOR packing was the serial-only variant, replaced).
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
  src/calibrate.cpp        # K sizing, VRAM model + THE wave/scalar dispatch rule
  src/sample.cpp           # THE SAMPLING LOOP (both faces): chunks + config bandit + stop
                           #   valves + kept-row buckets + delivery/verify accounting, and the
                           #   core's own deterministic RNG (gpap_sampler_* ABI)
  src/probe.cpp            # combinatorial probing: every k-subset of a literal set, paged +
                           #   budgeted (enumeration/chunking/ceiling all live HERE, not per-face)
  src/check.cpp            # batched checking: THE chunk loop of both AssumptionChecker faces
                           #   (validate-before-any-launch, chunk sizing, padding — one home)
  src/runtime.cpp          # cubin registry + launches
  kernels/bcp.py           # THE 2 BCP kernels (fixpoint + trajectory, SHIFT=16|8, scalar/WAVE)
  kernels/helpers.py       # support kernels (Triton, so the toolchain stays nvcc-free)
  tools/build_kernels.py   # AOT: kernels.py + helpers.py -> embedded cubins (27 specializations;
                           #   `vscan_is_read` is why it is not 33 — see Ground rules)
  examples/                # sample.cpp + assumptions.cpp + probe.cpp (tier 1) + c_api.c (raw C ABI)
examples/                  # RUNNABLE examples, both faces: sample_solutions.py +
                           #   check_assumptions.py (python) + a README pointing at cpp/examples/
gpap/
  implications.py          # what each literal IMPLIES — one width-1 sweep, three readers:
                           #   failed literals, hyper-binary resolvents, equivalent literals
                           #   (`equivalence_edges` CAN feed preprocess's ELS; strictly stronger
                           #   than its binary-clause edges, 1.11-1.32x more vars removed)
  mining.py                # entailed-clause mining — the STRATEGY (widths, fixpoint,
                           #   subsumption vs the BASELINE-REDUCED formula, `_Subsumption`)
                           #   over the core's probe primitive; `gpap mine` fronts it
                           #   (docs/findings/MINING_FINDINGS.md)
  engine/                  # THE PYTHON FACE of the core (app-agnostic, reusable)
    _core.py               # ctypes binding to libgpap (private)
    formula.py             # CnfFormula: facade over the core's builder (the numpy REFERENCE
                           #   build is tests/reference_formula.py, the core's oracle)
    bcp_engine.py          # BCPEngine: marshalling + torch-friendly accessors over the core
    _triton.py             # `kernel_available()`: can this host REBUILD kernels? (gpap info only)
    calibrate.py           # K sizing, VRAM model, gpu_mem_available
    checker.py             # AssumptionChecker: the batched-assumptions face (marshalling only;
                           #   the chunk LOOP is cpp/src/check.cpp, shared with the C++ face)
    probe.py               # LiteralProbe: a thin DIMACS face over the core's probe ABI (the
                           #   sweep itself is cpp/src/probe.cpp) — mining/backbone/lookahead
  sampling/                # THE FLAGSHIP
    sampler.py             # GPUSampler — MARSHALLING over the core's gpap_sampler_* loop
                           #   (configs/knobs in, stats/buckets out) + ELS reconstruction and
                           #   the host projection fill; the loop itself is cpp/src/sample.cpp
    pipeline.py            # sample_file / sample_clauses: the full e2e flow + the portfolios
                           #   (VarOrder/Lookahead enums + SamplingConfig NamedTuple, so a config
                           #   documents itself; `portfolio(name, allow_soft=)` is the accessor)
                           #   DEFAULTS: see "Sampling defaults" below
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
                           #   rivals_setup.sh + rival_common.py (THE fairness policy: circuit
                           #   variants, the PARALLELISM GRANT — a rival's unique/s is
                           #   ~proportional to the batch we hand it, so each gets the largest
                           #   that fits 0.80 of free VRAM, the same rule K uses — and the
                           #   on-disk GPAP-arm cache, so a rival-side re-run costs only the
                           #   rival) + check_emission_symmetry.py (RE-DERIVES that
                           #   no tool is timed while DELIVERING solutions: AST-scans what each
                           #   rival's own timer() wraps, our collect=False, cmsgen --nowrite)
                           #   + barbarik_gpap.py (CM_test; gpap|pipe|cmsgen|spur under test,
                           #   spur = the NEGATIVE CONTROL) + barbarik_screen.py (WHICH instances
                           #   the test can speak about) + barbarik_table.py (THE eligibility rule,
                           #   imported by paper_tables) + fuzz_uniformity.py (exact-oracle fuzzing)
                           #   + the cross-GPU Colab driver + colab_supervisor.py (survives the
                           #   ~1h VM lifetime: mirror every cycle, reseed, resume)
                           #   PAPER OUTPUT: paper_tables.py is THE generator (all exclusion
                           #   rules live there ONCE; slide_tables.py IMPORTS them so the deck
                           #   cannot drift) + soft_price_tables.py (safe-vs-soft paired sweeps)
                           #   + verify_artifact.py (publishability gate) + make_artifact.sh
                           #   RAW DATA LIVES OUTSIDE THE REPO: $GPAP_RESULTS (../gpap-results)
scripts/                   # build_core.sh (libgpap), build_abc.sh, build_cmsgen.sh +
                           #   cmsgen_gpap_eval.patch, fetch_benchmarks.sh, perf_ab_trees.py
                           #   (two-checkout throughput A/B)
docs/findings/             # INSIGHTS.md (settled results index) + *_FINDINGS.md deep dives
tests/                     # pytest; circuit CNFs GENERATED from benchmarks/circuits/bench/ at test time
.gpap/                     # built/fetched tools, sources, benchmark sets — gitignored, NOT $HOME
                           #   (gpap/paths.py; override with $GPAP_HOME)
```

**Code carries the RULE; `docs/` carries the EVIDENCE.** A comment says what the code does and why
it is correct. The measurement that settled it — the date, the GPU, the per-instance table — lives
in `docs/findings/`. Do not put a dated measurement or a card name back into a source file.

## Engine contract — two entry points (see `gpap/engine/__init__.py`)

1. **Batched assumption checking**: `reset(K)` + `apply_and_propagate(K×L decisions) → conflict /
   assignment` (one fixpoint launch). TWO one-call faces over it: `AssumptionChecker` ("check
   THESE sets") and `LiteralProbe` ("check every k-subset of this literal pool, within this
   budget" — `engine/probe.py`, the primitive behind mining). `LiteralProbe.assume()` is the
   incremental-units flow, `reset(1) + apply_and_propagate(units) + recache_flush()`.
   **The sweep lives in the CORE** (`gpap_probe_combinations` / `gpap_probe_tuples`, tier-1
   `gpap::Prober`): enumeration order, chunk sizing and the budget are defined once, so the Python
   and C++ faces probe the same candidates in the same order. Literals are DIMACS on both.
   **`max_tuples` is a WORK BUDGET**: a width with more candidates probes that many and reports
   `exhausted=False` — the clauses found are entailed regardless, so partial results are kept
   (`refuse_over_max` opts into the all-or-nothing behaviour). ⚠️ **It binds at width 1 too**, spent
   ACROSS that width's fixpoint rounds — `_mine_units` used to hardcode the ceiling to 0
   (unbounded) with no way to reach it, while the plan line warned it had been exceeded.
   ⚠️ **Width 1's `candidates`/`probed` are SUMS over `rounds[1]` fixpoint rounds** and are NOT
   comparable to the pool size (the pool shrinks each round, and proving convergence always costs
   one final round that finds nothing).
   Report `rounds` with them, and never sum `candidates` ACROSS widths — that divides a fixpoint
   sum by a C(P,2). ⚠️ **A per-width DICT must give width 1 an explicit `0`** (see
   `pipeline.STAGE1_MAX_TUPLES`): the fallback is `DEFAULT_MAX_TUPLES`, which would silently cap
   the backbone on the DEFAULT sampling path. The non-exhaustive reason comes from the probe's own
   `stopped`, never asserted — width 1 is as often cut by the deadline as by the ceiling.
   docs/findings/CLI_DELIVERY_FINDINGS.md. It is checked against
   `gpap_probe_count` BEFORE anything is generated; it used to be compared against an
   already-materialized enumeration, which cannot protect against the case it exists for — at
   width 3 over a circuit's free literals that array is ~1e11 tuples (~2 TB) and the process was
   OOM-killed reaching the comparison. ⚠️ **The cost is the PROBE, not the enumeration**: probing
   n tuples initializes n·M clause words (~1.2 GiB broadcast per chunk), so generation is free
   beside it — measured, a paged sweep of 4M tuples beats probing the same 4M pre-materialized.
   Do not "optimize" the enumerator. ⚠️ **THE RESET WAS THE BOTTLENECK, and CLEAN mode removed it** (the broadcast wrote
   rows·M words per chunk — ~26 ms, 41–94% of probe wall — while a probe modifies 0.02–2.5% of
   them): `gpap_engine_apply_clean` runs the same fixpoint then restores its OWN touched words
   from the baseline inside the launch (trail in registers, words hot in L2), and `reset()` goes
   LITE (skips the K·M broadcast) when rows are provably clean — b12-c50 chunk 20.9→4.7 ms
   (4.4×), b03-c25 sweep 2.60× e2e. Probe sweeps + flags-only `check_sets` ride it; anything
   that READS post-apply state (assignments/sat readout, `recache_flush`, selections) must use
   plain `apply` — the clean launch leaves the BASELINE behind. `GPAP_CLEAN=0` = exact legacy
   (also disables reset-lite, so A/Bs are uncontaminated); `gpap_engine_reset_stats` proves lite
   actually fired. Gated by tests/test_clean_mode.py (state-identity fuzz vs a plain twin + the
   reference BCP, both kernels × both splits; fails against a sabotaged tail). Design +
   lemmas: docs/plans/RESET_RESTORE_PLAN.md §9, findings in docs/findings/MINING_FINDINGS.md.
   ⚠️ **Redundancy is filtered on the HOST, and there are THREE levels** (`mine_entailed(drop=)`,
   `gpap mine --drop`), each stronger than the last: `none` / **`subsumed`** (default — a clause
   some clause already covers; free, `mining._Subsumption`) / `derivable` (ALSO the ones unit
   propagation re-derives on demand — assume all but one literal and the last is forced; one extra
   batched launch, ~40-70% of mining wall). ⚠️ **The default is NOT the strongest, on purpose.**
   `derivable` removes **61% of the wide clauses on b03-c25 where `subsumed` removes 0%** — the two
   notions barely overlap — but a derivable clause still SHORTENS propagation, and GPAP's sampler
   is propagation-bound: keeping them is measured at **+29% (b03-c25) / +43% (s386-c25) / +3%
   (b09-c25)** end-to-end. ⚠️ **A clause does k JOBS** (when the other k-1 literals are false it
   forces the remaining one) and ALL of them must be covered before it can go. Dropping on ONE
   covered job — the first cut, and what the HBR literature's transitive reduction can do because
   the binary implication graph is symmetric — discards **a third of the mined wide clauses with a
   live direction** (b03-c25 34.6%, b09-c25 34.2%), because propagation is NOT
   contrapositive-complete: `(-1|5),(-1|6),(-5|-6|2)` gives `1 ⇒ 2` but not `-2 ⇒ -1`. Drop them for a consumer that wants a minimal formula (a CDCL
   preprocessor); keep them for one that wants speed. ⚠️ The `_Subsumption` reference is the formula
   REDUCED by the engine's baseline, not the input clauses — mining folds proven units into the
   flush, so an input (a|b|c) whose `c` the baseline falsified is a binary constraint during the
   sweep and proves a clause no input clause subsumes (17 of them on b03-c25). Testing against the
   raw input also made the output KERNEL-DEPENDENT (3,096 clauses under scalar vs 3,113 under
   wave); gated by `test_mining_is_kernel_invariant`. ⚠️ There is deliberately **no filter in the
   engine** any more: `skip_subsumed` was sound but worth 0.0-0.3% of the result buffer while
   costing an int32 per row and a branch in both propagate loops — redundancy is a property of a
   CLAUSE against a FORMULA, so it belongs where the formula is known.
2. **Fused trajectory sampling**: `reset(K)` + `run_trajectory(order, n_support, B, seed,
   soft=…)` — the whole randomized decide+propagate loop in one launch. **The SAMPLING CHUNK
   LOOP OVER IT IS ALSO CORE** (`gpap_sampler_*`, `cpp/src/sample.cpp`, 2026-09-01): chunks,
   the adaptive config bandit, the stop valves, per-config kept-row buckets and the
   delivery/verify accounting — one home, the same lesson as `check.cpp`/`probe.cpp`. Both
   faces (`GPUSampler`, `gpap::Sampler`) are marshalling over it; at one seed + single config
   they draw IDENTICAL models (gated by tests/test_face_parity.py; the loop's own contract by
   tests/test_sample_core.py). The RNG is the core's (xoshiro/Fisher-Yates, stdlib-independent);
   ⚠️ recorded rows from before the port replay under torch seeds and are not
   seed-comparable — re-baseline, docs/BENCHMARKING.md §3. The C++ face gained
   `portfolio`/`var_order` (it could sample NOTHING on the service family before — PhaseService:
   0 in 40 s vs 3,768 in 0.35 s with the portfolio) and its `rate()` no longer charges delivery
   or the soundness self-test; `auto_retry` is now `retry_with_lookahead`, **default OFF** (the
   flip turns the Barbarik-rejected soft polarity on — the same choice `--allow-soft` gates in
   Python, applied to the face that never got the memo).

`reset(K)` broadcast-restores the construction-time post-units-flush cache (no per-chunk flush).

**Hot-path readout is the SELECTION api** (added with the C core): `make_pair_mask(vars)` once,
then per chunk `select_complete(mask_id, slot) -> count` (the completion test runs on device; only
the count crosses the bus), plus `check_sound(slot)` (per-row flag on device — no (rows, M)
transfer) and `var_bits(slot, cols, n)`. The tensor-shaped
accessors (`assignment`, `conflict`, `active`, `sat_rows`) are copies for tests / off-hot-path
consumers. App code must go through these — do NOT reach into the row state. The engine picks the packed split per formula (16|16 / 8|24) and
REJECTS formulas outside both envelopes; the PyTorch path is gone (packed
kernels are the only path — the oracle is `tests/reference_bcp.py`).

## Fused kernel & K sizing (the memory model that keeps biting)

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set in `gpap/__init__.py` **before** torch
  inits CUDA. It removes the allocator fragmentation that inflated reserved memory ~2.3× and forced
  a small K. Do not remove it.
- K sizing lives on the CORE sampler handle (`gpap_sampler_create`; `GPUSampler.K` is a
  property over it, assignment = `gpap_sampler_set_k` = the pin): **K = min(analytic, cap)**
  (kill-switch
  validated: the old 4×0.7·free/(70M+40N) floor modeled the deleted PyTorch path and bound K at
  ~1/3 of the packed fit). A runtime memory PROBE is unreliable here — do not resurrect it.
- **The `cap` is ONE constant, the same on every GPU**: `kBatchCap = 262,144`
  (`cpp/src/calibrate.cpp`; `engine.calibrate.K_CAP`; `--k-cap auto` is the default of
  `gpap sample`/`gpap bench`/`pipeline.sample_clauses`). It is a saturation limit (4× batch
  measured +2.8%), not a safety one — `compute_k_safe` is what keeps a chunk inside VRAM.
  ⚠️ **It used to scale with total VRAM** off a reference card's historical cap
  (`K_CAP_REF·total/K_CAP_VRAM_REF`, floored at 262,144). Removed 2026-08-31: `auto` meant a
  different batch per machine — so a cross-GPU A/B had to remember `--k-cap N` — and the
  proportional term had stopped firing anyway (the floor sat 5.3× above the reference cap, so it
  could not win below ~68 GB and every card in reach got the floor). Device-INDEPENDENCE is now
  the gated property (tests/test_k_cap.py). ⚠️ Sweeps recorded before 2026-08-18 ran at the old
  per-card caps (4080 49,152 / L4 90,300) and are not comparable to new ones. Resolved value in
  every row's `engine.k_cap`. ⚠️ **This is HYGIENE, not a throughput lever**: measured across
  GPUs, auto vs pinned bought ~1%, and a 4× batch bought 2.8%. The cap only binds
  for M ≲ 20k (above that the analytic bound is lower — and it ALREADY scales with free VRAM on
  its own), and those small-M instances are already occupancy-saturated. The cross-GPU win is the
  hardware. **Corroborated with the CONFIG PINNED (`--portfolio b1|b8`) + a multi-chunk budget,
  3 GPUs, 8 instances** — success is machine-INDEPENDENT (exactly 1.000 on 13/14 cells) and work
  rate is FLAT in M, tracking the SM-COUNT ratio rather than bandwidth ⇒ occupancy/compute-bound.
  Per-card factors: docs/findings/CROSS_GPU_FINDINGS.md.
  ⚠️ **Two measurement traps, both of which produced wrong numbers first**: (1) a free bandit
  picks differently per machine (the same instance chose B=1 on one card, B=8 on another) —
  always pin;
  (2) `--n-time 20000` against a 262k batch finishes in ONE chunk and measures first-chunk
  overhead — which read ~30% high. ⚠️ **OPEN**: one instance loses 2× of its success between
  K=20k and K=28k on one card only, machine- AND K-dependent, sound throughout; int32 `K*M`
  overflow was tested by K-ladder and FALSIFIED. See docs/EVAL_RUNBOOK.md.
- **Read `smp.batch_size`, set it with `smp.pin_batch(n)` / `with smp.pinned_batch(n):`**.
  `batch_size == K` now (there is ONE model, below) and it is in every row as
  `engine.batch`. Poke the attributes directly and you get the old bug factory back: when there
  were TWO numbers, `K_fused = K` and `K = min(K, n)` only bound the batch when they happened to
  drop K BELOW its construction value, so at `n >= K` the boost stayed on and the chunk ran orders
  of magnitude wider than asked — 28 GB of host RAM in one test, the same hole in the pipeline's
  coverage run, and a colab K-sweep that mismeasured its own x-axis. `pin_batch` states a bound
  that holds regardless. Gated by tests/test_k_cap.py.
- **ONE batch model**: `K` is `compute_k_safe`, which binds `gpap_row_bytes` — the
  engine's EXACT per-chunk allocation list (4·**M_long** packed — the len≥3 STATE clauses only,
  never total M — + 4·ceil(2N/32) bitmap + 4·trail_cap + flags/counters/masks), verified against
  measured VRAM (cuMemGetInfo deltas: 1.00x on b03-c25, 1.04x on b12-c50 — multi-GB alloc
  rounding). On circuit CNFs the TRAIL term now dominates per-row bytes. **`gpap_trail_cap(N)` in
  `cpp/src/calibrate.cpp` is THE definition** — `ensure_state` calls it, so the allocation and
  the model cannot drift — and it is `N + 32`: the bound is EXACTLY N (every push flips a bitmap
  bit and l/~l share a word, so a variable contributes one entry; measured tight on both kernels,
  and well under N on 100%-conflicted rows), plus **one WAVEFRONT** because the wave paths append
  a whole wavefront at `tail + prefix_sum` with no bound test — that is the granularity a broken
  invariant would overshoot by, at 128 B/row. Keep the slack in step with `WAVE_W`.
  **There is no ceiling on B**: the cap holds for any B, and `B >= |support|` (one round over the
  whole support) is exactly the exactly-uniform configuration. The decision batch used to be a
  parameter here — a floor the engine ignored while the sizing model honoured it, charging a
  caller who asked for B>64 for a trail that was never allocated. Gated by tests/test_k_cap.py
  and tests/test_sampler.py::test_wide_decision_rounds_are_accepted_and_sound. `K_fused` and its 0.6 budget are GONE: the boost only ever existed to
  undo a per-row model that was 1.57x over (it charged a phantom sat array, a byte-per-literal
  assignment and deleted PyTorch scatter temporaries, while omitting the trail). ⚠️ **`trail_cap`
  used to be sized by the width of the last apply and only ever grew**, so the construction units
  flush inflated it permanently — 10.sk_1_46 (1,121 units) ran at 2,623 instead of 1,566, +27%
  per-row forever. It is bounded by the VARIABLE count (every trail push is bitmap-deduped), see
  the proof at `cpp/src/calibrate.cpp::gpap_trail_cap`. Net: K is 1.4-2.5x larger; kill-switch 81% of
  VRAM, sound in both kernels. ⚠️ wave and scalar agree exactly on MODELS and DECISIONS but the
  IMPLICATION counter differs ~0.6% — gate on models, not counters. **The cause is CONFLICTED rows,
  not a tally bug** (checked; the earlier "wave tallies pre-dedup pushes" note was
  wrong — wave's trail push IS deduped, `keep = fm & ((oldw_v & bit_v) == 0)`). A surviving row has
  ONE fixpoint (confluence) so all of its fields match; a row that conflicts stopped as soon as
  some clause emptied, and WHICH clause got there first depends on propagation order. Measured on
  5,000 assumption sets: 809/3,282 conflicted rows differ in `num_propagated`, **0/1,718 surviving
  rows differ in anything**, and `conflict` matches on all 5,000. Documented on
  `AssumptionChecker.check_assumptions`, since chunk count can change the kernel and so a caller
  can observe it.
- ⚠️ **Any retune of `compute_k_safe` is a K change**: it needs kill-switch re-validation AND a
  dispatch re-check, since the batch feeds `gpap_dispatch_wave`'s small-batch term. That coupling
  has caused a regression before. The soundness re-check needs no headroom at all: it is one
  device flag per row (`sound_rows`), never a host transient.
- **Assignment BITMAP**: assignment = bit l of (K, ⌈2N/32⌉) int32 words (l/~l
  share a word) — 8× smaller than byte-per-literal (small-2N rows go L1-resident), and in
  WAVE a forced-literal push is ONE `tl.atomic_or` whose RETURN deduplicates (old bit l) and
  detects contradiction (old bit l^1) race-free — the old W×W in-warp dedup is deleted.
  Readout goes through the selection api (`select_complete` — the completion check is one
  bitwise pass on device, never a (K,N) mask — and `var_bits`' direct bit gather; `assignment`
  / `assignment_rows` materialize int8 rows only off the hot path. Strided bit-unpack
  slice-writes cost 16× sector RMW amplification, measured).
- **Kernel variant auto-dispatch lives in the CORE** — `GPAP_MODE_AUTO`, resolved inside
  `gpap_engine_reset`. Every consumer gets it, not just the one that asks, and that
  now includes consumers written against the C ABI. ⚠️ **This is the FOURTH time the same hole
  opened**, and the last one was IN THE DEFAULT ITSELF: `gpap_engine_options.mode` was
  `GPAP_MODE_SCALAR` and scalar was 0, so `{0}` / NULL opts — the spelling the header calls
  "every default" — gave the raw C face the one kernel that never dispatches, worth 1.28-2.20x.
  **`GPAP_MODE_AUTO` is 0 now**, so the zero value and the documented default are the same thing
  by construction rather than by remembering. (The same trap had already been fixed once for
  `vscan`; `gpap_probe_options.same_var` was the third field, below.) The rule used to be applied
  by the two WRAPPERS (`BCPEngine.plan_for`,
  `gpap::Engine::plan_for`), so for a time only GPUSampler dispatched (batched assumptions
  ran scalar, 1.05-1.15x left on the table) and then `gpap_probe_combinations` — which calls the
  ABI directly — ran scalar from its port until this was fixed (measured 1.28x on b03-c25 M=14.5k,
  1.67x on 6s215rb0 M=66k, 1.76x on b09-c25, **2.20x on b12-c50 M=219k**; 1.65x end-to-end on
  `gpap mine` where the sweep dominates, ~1.0x where it does not). Put a default WHERE THE ABI IS,
  not where the convenient wrapper is. `plan_for` survives in both faces as a pure query.
  GPUSampler still PINS the mode to the chunk batch via `set_mode` so its small resets (warmup,
  short final chunk) cannot switch kernel mid-measurement. Same state, pure launch flag
  (confluence ⇒ it can only change speed). It used NOT to be observationally pure, through the
  engine's `skip_subsumed` filter — that filter is deleted (entry point 1), and mining's answer is
  kernel-invariant again, gated by `test_mining_is_kernel_invariant`.
  **THE RULE IS `gpap_dispatch_wave(M, batch, device)` in `cpp/src/calibrate.cpp` — ONE definition,
  and it keys off M, the CLAUSE COUNT, not the batch**: warp-wide **WAVE** propagate
  (≤W trail lits at once, occ segments flattened, the packed-word `atomic_add` return value
  resolving cross-literal clause races) when `M ≥ 2048`, or when `batch ≤ resident_warps` (too few
  rows for scalar to fill the card either way); scalar serial-trail propagate otherwise.
  ⚠️ **It USED to read `batch < 30× resident_warps`, and that was a PROXY for M, not occupancy** —
  K ≈ budget/(4M), so "huge batch" and "small M" were the same rows. The batch-sizing collapse
  (K 1.4-2.5× larger) broke the proxy: mid-size formulas landed ON the batch cap, where `batch` is
  a constant, so every one was classed "scalar". Measured cost, ABC circuits, fixed wall,
  interleaved best-of-2: **0.727× on the rows that flipped, while the rows that kept their kernel
  were 0.993×.** After the M-based rule the same A/B is
  **1.044× overall (worst row 0.938), with 3 scalar→wave promotions worth 1.28×.** Calibration
  (forced-kernel A/B at FIXED K, 25 instances, two GPUs, ABC circuits + sk_ + tiny circuits, at the
  batch cap so M is the only variable): M=32 0.772, M=483 0.918, M=1,982 0.960, M=2,098 1.041,
  M=2,882 1.407, M=14,012 1.572, and 1.27-1.66 for every larger M out to M=107,065; whole-set
  geomean wave/scalar 1.39–1.46× across both cards. Do NOT reintroduce a batch-only rule.
  **The paper's number is the L4 A/B** (`eval/kernel_ab.py`, 141 cells, all families × both
  encoders, validated by identical model counts at pinned K, mining OFF — it is wall-budgeted so
  a faster kernel would mine more and the arms would run different formulas): **1.30× geomean,
  1.01× below 2k clauses → 1.33× (2k–20k) → 1.48× (20k–100k) → 1.64× (>100k)**; the rule picked
  the faster kernel on 77/93, every miss in the M≈0.5–9k tie band. docs/findings/KERNEL_AB_FINDINGS.md.
  Session total vs the pre-wave baseline: b14 5.7×, b12 2.8×, b04 1.8×, b03 1.5×.
  `GPAP_KERNEL=scalar|wave` forces — read in the CORE beside `GPAP_CLEAN`, so it reaches every
  consumer, and applied only where AUTO resolves so it can never contradict an explicit pin
  (`gpap_engine_auto` is how a caller asks whether it may pin one of its own);
  W is the AOT constant `WAVE_W` in `cpp/tools/build_kernels.py`, power-of-two only
  (32 best; BOTH ways of raising it measured out — over W/32 REAL warps, and as ILP within one
  warp). **The wave lane→segment schedule is SHUFFLE-BASED since 2026-08-27**: a `tl.gather`
  lower_bound (`_seg_lookup`) replaced the W×W one-hot reductions, which lowered through SHARED
  memory with barriers — the kernel now uses ZERO shared memory. **Same-K geomean 1.80× on wave
  cells; the ~1e9 impl/s plateau moves to 1.05-2.41e9** ⇒ EVERY pre-2026-08-27 wave rate in this
  file and in the campaigns is superseded, including the propagation-rate ground rule below.
  `_wave_propagate` changed and scalar `_propagate` did not, so the M≈0.5-9k tie band's true
  crossover moved DOWN — re-derive it at the next kernel_ab campaign, do not guess. Builds need
  Triton ≥ 3.2 (`tl.gather`); runtime still `libcuda` alone.
  docs/findings/WAVE_SHUFFLE_FINDINGS.md. The trajectory kernel's
  **decision scan** is warp-wide in WAVE only (bit-identical picks; +12-19% there; in scalar
  the bitmap made the serial scan L1-cheap and the screen LOSES; `vscan=-1` in the engine options disables (the `GPAP_VSCAN` env knob is gone);
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
gpap mine formula.cnf --width 1 2 --pool all --seconds 10 --out stronger.cnf  # entailed clauses
gpap sample b10.bench --cycles 25 -n 50000 --out stim.out    # .bench: columns = wire@cycle
gpap sample b10.bench --cycles 25 --encoder abc              # ABC-encoded CNF (2-8x fewer clauses)
gpap bench 'path/*.cnf' --cmsgen /path/to/cmsgen --n-cov 5000
scripts/build_abc.sh                                         # optional: pinned abc -> .gpap/bin
```

## Sampling defaults

| | default | opt out |
|---|---|---|
| ELS (equivalent-literal substitution) | **on** | `--no-els` |
| entailed-clause mining | **on** above `MINE_MIN_CLAUSES`, bounded by `MINE_SECONDS` | `--mine off` |
| mined clauses | **kept only if they pay** (below) | — |
| soundness re-check of emitted models | **off** everywhere | — it is OPT-IN: `--verify-sound` turns it ON (`eval/run_full_eval.sh` does) |

⚠️ **`gpap sample`/`gpap bench` default to `--portfolio circuit` for EVERY input, including a
plain CNF, and on CNF that costs certificates**: `circuit` has no `s1`, so the bandit falls back to
`b1`, which Barbarik REJECTS on `registerlesSwap` and `UserServiceImpl` where `s1` is Accepted —
and it is not even a throughput trade (`s1` delivers 46,512 models to `b1`'s 4,767 on
registerlesSwap). Delivered-config verdicts on the 21 eligible sk_ instances: `default` 20 Accept /
0 Reject, `circuit` 18 / 2. On circuits the default is right (19/20 Accept, 0 Reject). NOT FIXED —
defaulting the portfolio by input type moves a default every recorded `gpap bench` CNF row was
produced under. docs/findings/UNIFORMITY_FINDINGS.md §4b.

⚠️ **Recorded sweeps predating these defaults are not comparable to new ones** — re-run, or label
with the flags above. Validation and per-instance numbers: docs/findings/MINING_FINDINGS.md.

- **`MINE_SECONDS` is honored to CHUNK granularity** — a chunk is the smallest unit of GPU work and
  is never abandoned half-written, so a large-M formula can overshoot by one. It is a wall-clock
  deadline from mining start, so the between-stage rebuild + recheck are inside it; that bounds
  total pre-sampling overhead, which is the useful thing to bound.
- **`--mine on` FORCES past `MINE_MIN_CLAUSES`**; `auto` respects it. Gated by
  tests/test_mining.py::TestMiningPipelineDefaults.
- **SOUNDNESS RE-CHECK is OPT-IN everywhere and, when on, covers EVERY emitted row**
  (`verify_sound=` / `--verify-sound`, on the library, `gpap sample`, `gpap bench` and the C++
  `SampleOptions::verify` alike; `eval/run_full_eval.sh` sets `VERIFY_SOUND=1` for a campaign).
  It is a SELF-TEST, not part of producing samples — a row is only counted when it is active and
  fully assigned, which already means no clause was emptied — so it is not a tax on every run,
  and `verify_artifact` counts it without requiring it. Reported as **`sound_check`: `"ok"` |
  `"UNSOUND"` | `"off"`** plus `unsound_rows` and `sound_rows_checked` — a tri-state, never a
  bool, because "not run" is neither verdict (the `unique_est_status` lesson). ⚠️ **"off" also
  covers REQUESTED-BUT-NEVER-REACHED**: the check runs per chunk over the rows that chunk
  COMPLETED, so a 0%-success run (or one a `--time-limit` cut before its first completion) never
  calls it — and reporting that as `"ok"` is the same conflation one state along, an un-run check
  published as a passing one. `sound_rows_checked` is the evidence; it is what a verdict is a
  statement about. (It caught a real one: `test_wide_matches_base_random` had been comparing two
  EMPTY model sets on five instances and asserting `"ok"` over them.) ⚠️ **UNSOUND DOMINATES when
  the timed and coverage runs are merged** (`metrics.merge_sound_check`) — preferring the timed
  run's verdict published `sound_check: "ok"` beside a non-zero `unsound_rows`, and every
  consumer branches on the string.
  Its wall is `verify_sound_s`, EXCLUDED from every rate exactly as `out_project_s` is, so
  enabling it cannot move samples/s — ⚠️ **and it has to be subtracted BY THE PIPELINE**, not
  only stamped per config: `best_rate_sps` exists only on the adaptive path, so a PINNED single
  config (the mode every measurement uses) falls back to `n_models / wall` and would carry the
  check. Exactly the hole documented for `out_project_s` one line up, reopened.
  ⚠️ It used to run unconditionally on a **4096-row prefix** of each chunk, a cap that sized a
  (rows, M) host transient deleted long ago — so it bounded only the COVERAGE, leaving
  "0 unsound" standing on as little as **4.8% of what a chunk emitted**. ⚠️ The old key was
  `gpu_sound` (a bool) and recorded rows still carry it. **`gpap.sampling.metrics.sound_verdict`
  is THE reader** (legacy shape included) and `merge_sound_check` THE fold: the bench table, the
  CLI, the Colab driver, the rival harnesses and `paper_tables` all import them. They were six
  open-coded string compares, of which only two handled the legacy key — so a resumed campaign
  reported a fully verified sweep as "not run", and a legacy FAILURE as a non-verdict. ⚠️ A conflicted row is NOT necessarily unsound: on the
  scalar kernel nearly every conflict surfaces as a contradictory PUSH, having emptied no
  clause. Gated by tests/test_core.py::test_check_sound_agrees_with_an_independent_recount.
- **KEEP-IF-IT-HELPS**: mined clauses of width ≥ 3 enlarge `num_long`, and `K ~ budget/4M_long`,
  so a bigger formula buys a SMALLER batch — **but mined width-2 clauses are K-FREE** (they join
  the binary implication list, zero per-row state), so the proxy naturally keeps more of them. `K x M` is ~constant ⇒ chunk wall is ~constant ⇒ **models per chunk
  = `K x success`** is the throughput proxy, and both terms are already in hand (no extra GPU
  work). If it does not beat the pre-mining sampler the clauses are dropped and the original is
  rebuilt. It is a PROXY (warmup pins K<=1024), and it deliberately keeps the clauses when the
  un-mined warmup completed NOTHING — there is no baseline to be worse than, and mining is the only
  thing that has converted those. `mined.kept` / `mined.gain` in `--stats`.

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
**Miner overlap**: dc2/fraig removes most of the backbone width-1 mining finds (0-45 units on ABC
CNFs vs 59-1007 on Tseitin) — but that COUNT is not the effect. ⚠️ Mining is worth MORE on ABC CNFs
than Tseitin (geomean 3.28x vs 2.05x over 24x2), so the provenance gate that keyed off it is
DELETED, along with the `MINE_MIN_CLAUSES` 3000→500 that hid the same small ABC
formulas a second way. **At `--target rare` the encoder is PER-INSTANCE** (same engine, 3-seed):
ABC wins 1.8-5.7× on most and unlocks b14 (0→24 models/s), but the rare-pin
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
Certificates on big spaces = conditional sampling = Barbarik (`eval/uniformity/barbarik_gpap.py`, GPAP
hooked in-process OR `--sampler cmsgen`, SPUR reference; defaults are the CMSGen paper's
ε=0.3/η=1.8/δ=0.1 ⇒ 1,872 samples/instance = 36 rounds × (1+1+50), accept band [0.05, 0.95],
38× cheaper than ε=0.1/η=0.9; `--explain` prints the derivation). chi2 sf is native
(`metrics._gammq`) — no scipy dep.
⚠️ **THE PARAMETERS ARE A CHOICE AND THERE ARE TWO**: the CMSGen paper's ε=0.3/η=1.8/δ=0.1
(1,872 samples, band [0.05, 0.95], rejects TV ≥ 0.90) and barbarik's OWN default ε=0.1/η=0.9/δ=0.05
(70,596 samples, bands ≈±0.21, rejects TV ≥ 0.45) — different tests, never pooled
(`barbarik_table.py` refuses). ⚠️ **The guarantee is ASYMMETRIC**: Reject is unconditional, Accept
holds only under the paper's *non-adversality w.r.t. Kernel* assumption AND only against η, so an
Accept is "not caught", never "uniform". ⚠️ CMSGen's own Accept was FITTED to this test (their
Remark 1: they tuned CMSGen's restart parameters against Barbarik's rejections) — parity is the
honest framing. ⚠️ **A Barbarik table is only evidence with BOTH controls and the eligibility screen**
(docs/findings/UNIFORMITY_FINDINGS.md §4b): `--sampler spur` puts an exactly uniform sampler under
test through the same hook and MUST Accept; `--config soft` MUST Reject. The screen exists because
**CM_test rejects its own trusted reference** on instances whose declared `c ind` is not an
independent support — its kernel pins every indVar, so the two buckets hold (extensions of σ1) vs
(extensions of σ2) MODELS and a model-uniform reference is skewed by exactly that ratio
(ActivityService 4×, IssueServiceImpl 6.6×, PhaseService 21.8×, IterationService 103.7× ⇒ all
Rejected; every spread-1.0× instance Accepts). `barbarik_screen.py` re-derives it (SPUR feasible
within a 20 s/sample cost bound + extension spread); `barbarik_table.py` buckets on the CONTROL's
verdict, never on the screen — the spread estimate is a LOWER bound (4 projections misses what 8
catches). ⚠️ `k_cap` is a batch size, NOT a distribution knob, but a small one makes short chunks
and the collector's zero-yield stop fires early ⇒ "produced 9/50", which reads as a GPAP failure;
the harness defaults to the shipped cap. ⚠️ Barbarik's kernel is adversarial for a REJECTION
sampler by construction (it enlarges the sampling set while pinning the support), so GPAP
sometimes cannot fill an order — that is "no verdict", not a Reject, and it bounds coverage.
**Uniformity FUZZING against an exact oracle** (`eval/uniformity/fuzz_uniformity.py`, full
writeup `docs/findings/UNIFORMITY_FINDINGS.md`): generates small random CNFs / random Tseitin
circuits whose solution set is enumerated EXHAUSTIVELY (solver-free oracle; the circuit generator
proves its own encoding over the whole truth table), then runs GPAP and CMSGen on identical files
with negative + positive **controls** (exact-uniform must not be flagged; a known TV must be —
these calibrate the instrument, sensitivity floor TV≈0.05 at n=100|S|×3 seeds). Two settled
results: **B ≥ |support| ⇒ exactly uniform** (rejection sampling, since GPAP discards conflicted
trajectories) and **`soft` is Barbarik-REJECTED on 64 of the 65 circuit instances it can sample**
while the config the bandit delivers is Accepted 64/66 with ZERO rejections (CMSGen 66/66; and
6/6 vs 6/6 at barbarik's stricter own default) — the
bandit picks configs on models/s with no uniformity term, so `soft` winning was silent ⇒
**`soft` was REMOVED from CIRCUIT_PORTFOLIO and is now opt-in via `--allow-soft`**
(`gpap sample`/`gpap bench`; prints a warning; `pipeline.SOFT_CONFIG`).
⚠️ **OPEN DECISION — the throughput price is BIG and is parked, not settled** (user:
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
⚠️ **The rows are COLLECTED BY THE TIMED RUN** (`sample(out_cap=…)` → `out_bits`), not by a pass
after it: one `_project` per chunk, a fraction of a percent of the chunk it rides on. It used to
be a SECOND full sampling pass, which (a) obeyed no `--time-limit`, (b) sat outside `wall_total_s`
while being stamped into `phases`, and (c) fetched the full `--max-samples`/`-n` request even when
the timed run had been CUT SHORT — so a capped run reported one model count and then wrote a
larger number of samples. A file now holds `min(-n, --max-samples, delivered)` rows and the CLI
says why when it is short. ⚠️ Rows are bucketed **PER CONFIG and only the winner's are returned**,
so the emitted set stays ONE sampler rather than a mixture of the bandit's arms (a mixture is a
uniformity question, and this was a delivery change). ⚠️ **DELIVERY AND SAMPLING MUST NOT SHARE A
RANDOM STREAM**: `_project` fills don't-care columns from `_proj_gen`, NOT the `_gen` that drives
the per-chunk permutation and trajectory seed — the first version of this shared them, and `--out`
silently changed which models a run found at the same `--seed` (invisible to the no-change gate,
which runs without `--out`). Its wall is `out_project_s`, excluded from every rate — the
per-config stamp alone is not enough, since a PINNED single config has no per-config wall.
⚠️ `phases` sums to `wall_total_s`; the opt-in readouts (`--n-cov`, `--uniformity`) are billed to
`phases_diag`, OUTSIDE it. docs/findings/CLI_DELIVERY_FINDINGS.md.

Benchmark suite driver (`gpap/benchmark.py`, schema-3 JSON; thin table over
`gpap.sampling.pipeline`; the glob may mix CNFs with circuits — .bench/.aig/.aag are converted
first, encode wall reported per instance as t_enc, never folded into either sampler's rate):
```bash
gpap bench 'benchmarks/circuits/bench/seq/*.bench' --encoders tseitin abc:fast --target rare \
    --cmsgen .gpap/bin/cmsgen --output .experiments/seq.json --resume
```
- **`--encoders tseitin abc:fast`** runs EVERY circuit through each encoder back-to-back in ONE
  process (same GPU state) → one row per (instance, encoder) + a paired A/B summary. Single
  `--encoder` still works.
- **Never dies mid-sweep**: a failing cell becomes an `error`/`skipped` row and the run continues;
  rows are written to `--output` after each one and `--resume` skips what is already there
  (the Colab-disconnect story). Out-of-envelope formulas (`gpap.engine.packed_envelope`) and
  `--max-clauses/--max-vars` oversizes are skipped WITH the reason, never a raise.
- **Engine provenance in every row** (`engine`: packed split 16|16 vs 8|24, wave/scalar propagate,
  K and the resolved cap) + a `krn` column — the evidence a number came from the sum-trick kernel in-envelope.
- **CMSGen fairness**: build it with `scripts/build_cmsgen.sh` (upstream + the in-repo patch
  `scripts/cmsgen_gpap_eval.patch`). GPAP's timed run emits nothing, stock CMSGen writes every
  sample as a full N-var assignment — measured 8.2× of its wall on 27.sk_3_32 (0.42s vs 3.6s /
  373 MB). The driver auto-detects the patch's `--nowrite` and times THAT; a stock binary is
  used as-is but the row is marked `cms_mode="write"` and a warning printed. The patch's
  `--statsevery` also gives CMSGen's own props/decisions ⇒ the `Cdec/s`/`Cimpl/s` columns.
  ⚠️ **Ratios recorded before the driver timed the patched binary include that handicap**
  (docs/BENCHMARKING.md §4b).
  ⚠️ **`--maxtime` is what makes a budgeted CMSGen count REAL.** `--statsevery` prints every
  N samples, so a run KILLED on the budget reports the last multiple of N: on the 2026-08-23
  campaign all 174 timed-out rows were multiples of 1000 and **81 read as a clean 0 that scored
  as our win** (one re-measured cell delivers 941 samples — a ~10-25× loss published as an
  infinite win). `--maxtime` makes CMSGen stop itself and exit through its normal path, printing
  a `final=1` line with an exact count — and its work counters even where it emits nothing, which
  used to drop those cells out of Table 3 entirely. `cms_count_exact` per row; `verify_artifact`
  FAILS a floored zero. It bounds the SAMPLING wall (what GPAP's `--time-limit` bounds), so
  `cms_sps` (sampling wall) pairs with our steady state and `cms_sps_e2e` (subprocess wall) with
  our `wall_total` — each column charges both sides or neither. RIVAL_FAIRNESS_FINDINGS.md §7.

**Full experimental sweep** — [`docs/EVAL_RUNBOOK.md`](docs/EVAL_RUNBOOK.md). Terminal-driven
(Colab reaps idle notebook cells), staged and resumable:
```bash
eval/rivals/rivals_setup.sh                  # abc + patched cmsgen + DEMOTIC/Recurrent/HTS checkouts
eval/run_full_eval.sh seq             # or comb | sk | epfl | hwmcc | rivals | plots | all
```
Each circuit stage sweeps encoders × targets(rare,random) × depths(25,50) into
`.experiments/full_eval/<gpu>/`; `eval/eval_plots.py` turns those into the encoder-A/B,
vs-CMSGen, work-rate, amplification and rival figures + `summary.md` (which lists every
skipped/errored instance — read it before quoting a geomean).
Rival harnesses: `eval/rivals/rivals_demotic.py` (`--mode native` = their ISCAS-85 pins, `--mode miter`
= our rare-target instance as a single-output miter; sanitizes wire names + BUFF→BUF because
DEMOTIC exec's generated Python), `eval/rivals/rivals_recurrent.py`, `eval/rivals/rivals_hts.py`.
⚠️ **Rival knobs come from the RIVAL, or from a stated rule — never from us.** The batch is
granted (0.80 of free VRAM, `fit_batch`); `--num_steps` is taken from the tool's own artifacts
(`RIVAL_STEPS`: DEMOTIC 1, Recurrent 1 — their run scripts; HTS 10 — its argparse default, since
its script's 5 is far worse for it), and where a tool ships two values the tie-break goes to the
one better FOR THE RIVAL. Pinning 10 for all three, as the drivers did until 2026-08-26, taxed
Recurrent 8.5× and DEMOTIC 2.6× (both saturate their batch on the first step: Recurrent 2,198,776
unique at steps=1 vs 2,134,454 at steps=10, 14.3 s vs 118.0 s) while under-granting HTS, which
genuinely needs the descent. NOT swept-and-best-of, on purpose — that invites "why stop at 20?".
`eval/rivals/steps_calibration.py` re-derives the tie-break on the campaign's own card.
⚠️ **Rival rows recorded on or before 2026-08-26 used the pinned 10 and are superseded.**

**Same circuit on both sides** (`--circuit-variants raw abc`, eval/rivals/rival_common.py): the
circuit-level rivals unroll `.bench` themselves, so an ABC-encoded GPAP arm must not face a rival
holding the raw circuit — `abc_backend.simplify_bench_abc` ABC-optimizes the SEQUENTIAL AIG and
writes it back as a primitive-gate .bench (via AIGER; abc's `write_bench` emits LUT/DFFRSE that
nothing here parses), gated by `bench_equiv_check` (identical stimuli ⇒ bit-identical accept
words). The post-unroll ABC pass has no sequential equivalent — stated, not hidden. Measured:
DEMOTIC gains 15-25% from it on ISCAS-85 and c1355 flips 0.5×→1.5× in our favour.
Cross-GPU K-scaling: `eval/colab/colab_sampling_benchmark.py` + `notebooks/gpap_colab_sampling.ipynb`.

## Ground rules (do not re-litigate — see docs/findings/INSIGHTS.md)

- **An ABI break MUST move the soname.** libgpap is 0.2.0 / SOVERSION 1 as of the defaults
  change; a caller that only RELINKS gets none of the compile errors the renames were chosen for,
  and under SOVERSION 0 it would have read `GPAP_MODE_WAVE` as scalar and passed an int where a
  `gpap_probe_options*` is dereferenced. Bump both again on the next break.
- **`{0}` MUST equal NULL on every options struct.** Three fields have broken it (`vscan`,
  `mode`, `same_var`) and each was a silent wrong default on the raw C face, which is the ONE
  face with no wrapper to fix it up. So: every field's zero is its default, and where the useful
  default is "on", the field is named for the OPT-OUT (`same_var`, not `distinct_vars`). A
  renamed field is a compile error for an out-of-tree caller — a flipped one with the same name
  is not, which is why the rename is the fix and not a comment.
- **A cubin the kernel cannot tell apart is a cubin we do not ship.** A `tl.constexpr` guarded
  by other constants (`if WAVE and VSCAN and not SOFT`) must not be specialized where it is
  unread: 6 of 33 shipped cubins were byte-identical duplicates, 23% of the payload. The rule
  lives in `build_kernels.vscan_is_read` AND `engine.cpp::traj_key` — trimming the matrix and
  normalizing the lookup key are one change, and half of it is a launch-time `GPAP_ERR_KERNEL`.
  Gated by tests/test_kernel_table.py.
- ⚠️ **Cubin BYTE-identity is not a no-change gate.** Inserting 55 blank lines into `bcp.py`,
  with no code change at all, moves 24 of 33 cubins: line info is embedded. Compare the
  **executable sections** (SHF_EXECINSTR) instead — that is what proved the shared-body refactor
  of `_bcp_fixpoint_kernel` / `_bcp_fixpoint_clean_kernel` emits identical SASS for all 27.
  Compilation IS deterministic (33/33 byte-identical across two runs of one source), so a
  same-source A/B is still exact.
- **Key a results cache on `gpap_build_id()`, never `gpap_version()`.** The version has read
  "0.1.0" for the project's whole life while the engine got 1.8x faster under it; the build id is
  a fingerprint of the loaded library's own bytes. It is what `_core_provenance()` stamps into
  every row and what the rival cache keys on.
- Runtime per-batch memory PROBES are unreliable → fixed 4×-floor K stays.
- This box SPILLS to host RAM (never OOMs) → validate every K change under the nvidia-smi
  kill-switch; check the power state first (a 40 W cap / battery clocks ~halve every rate).
- `mono` kernel (0.93× vs wide), `orac` circuit config (0/64 wins), full/soft global lookahead:
  measured out. **soft is now OPT-IN (`--allow-soft`), not a default portfolio member** — it is
  Barbarik-rejected on 64 of 65 circuit instances, so a throughput bandit must not choose it for a
  user who asked for a sampler. It stays reachable: regime-dependence rule, it still wins
  tight/deep circuits on success rate.
- `--n-time` is a delivery TARGET, not a guarantee in either direction: the batch K is the unit of
  work, so a run overshoots to the end of a chunk (a small formula at the 262,144 cap delivers
  ~1e6 models for `-n 100`) and can fall short when a safety valve fires — `--time-limit`, the
  zero-progress stop
  (both flagged by `chunk_capped` / the CLI's `<`). Quote the DELIVERED count, never the request.
  `--time-limit 0` = no cap on `gpap sample`; `gpap bench` refuses 0, since there the same flag is
  also the rival subprocess's timeout and the two readings do not reconcile.
- **VACUOUS-PIN instances**: at T=25 the last-cycle outputs of `b02`, `b05`, `b07`, `s526` are
  CONSTANT for every input sequence ⇒ pinning them is a tautology and "sampling" is a free
  hypercube over |PI|·T bits (any sampler gets ~100% success; ABC collapses the whole output cone,
  so its "shrink" there is instance degeneracy, not encoder merit). Exclude them from headline
  encoder/sampler comparisons, or report them separately. `b01`/`b09` are near-vacuous (2 patterns).
- **ZERO-SEARCH cells are the same trap, found per-row not per-name**: a row with
  `n_models > 0` but `decisions_per_s == 0 AND implications_per_s == 0` samples at 235-240M/s with
  succ 1.0 because the support is FREE after propagation — no search happened. EPFL `ctrl`, `dec`,
  `square` + sk_ `tableBasedAddition`. Worse than vacuous pins in one way: **the best-encoder rule
  PREFERS them** (they carry the family's top ratio — `ctrl` tseitin succ 0.07 @2M/s vs abc:fast
  succ 1.0 @240M/s). Leaving them in: epfl-rare 9.4→29.0×, epfl-random 8.2→19.5×, sk 98.9→113.1×.
  Excluded by `paper_tables.zero_search`; flagged by `verify_artifact.py`.
- **The zero ledger MUST be symmetric.** CMSGen-zero → `ratio=inf`, GPAP-zero → `ratio=0.0`,
  both-zero → `ratio=None`. The last two are FALSY, so branching on `ratio` alone counts their
  zeros as our wins and drops ours entirely (38 of theirs shown as wins, 14 of ours invisible).
  Report
  `CMSGen=0 / GPAP=0 / both=0` as columns that SUM with the geomean's instances to the instance
  count — a reader must be able to add them up. ⚠️ **And they must actually sum**: the degeneracy
  screens (vacuous pins, zero-search) used to be applied only to the geomean, leaving those cells
  inside `inst` and inside `wins`, so the printed invariant was FALSE on 6 of 7 families. Screen
  FIRST, then bucket; the invariant is `inst = n_gm + CMSGen=0 + GPAP=0 + both=0 + no-search +
  vacuous`, pinned by `test_every_instance_is_counted_exactly_once`. A zero-search cell can also
  be a CMSGen-zero cell (EPFL `ctrl` under ABC), so screening after bucketing published it as an
  outright win.
- **NO best-of-encoders selection — one table row per (family, encoder).** `select_cells` used to
  keep the encoder with the highest `ratio`, which makes the published statistic its own argmax
  key and lets the RIVAL's number pick which of our runs is reported (measured: seq-c25-rare
  84.3→66.7×, seq-c25-random 333.8→250.8×, i.e. 1.24-1.33× on the flagship families; a wash
  pooled). The mechanism is that ABC helps CMSGen more than us on some instances, so maximising
  the quotient awards a shared input-preprocessing gain to one side. Reporting both encoders
  removes the question, and keeps the encoder A/B visible — it is not monotone (ABC unlocks b14
  at a rare pin, collapses b13's). `eval_plots.py` selects on `gpap_sps` where it must select at
  all; `slide_tables.py`/`soft_price_tables.py` import the paper's function. §8/§9 of
  docs/findings/RIVAL_FAIRNESS_FINDINGS.md.
- **Propagation rate isolates the engine from the sampler**: GPAP holds ~1e9
  implications/s on EVERY family (geomean 1.07e9, spread 1.41× over the 9 families with n>=3,
  occupancy-bound) vs CMSGen's 1.7-5.4e6 (its
  OWN `--statsevery` counters). The identity **samples/s ratio = (impl/s ratio) / (impl-per-sample
  ratio)** closes to <1% on 9/9 families ⇒ the 3-orders-of-magnitude spread in the headline is
  REJECTION COST (we discard a conflicted trajectory; CMSGen backtracks and keeps it): 1.0× on
  ISCAS-85, 5.6× seq-c25, 14.4× seq-c50, 53.7× EPFL; engine advantage 197-631× (geomean 306×). ⚠️ Every column must use the SAME rows — filtering per column
  lets an instance that propagates hard but emits nothing (EPFL `bar`) into the numerator only,
  breaking the identity by 3×. Table 3 of `eval/artifact/paper_tables.py`; figure `propagation_summary.png`.
  ⚠️ **The impl/s magnitudes here are the PRE-shuffle-schedule engine (campaign 20260827) and are SUPERSEDED**: the 2026-08-27 wave rewrite moved the plateau to 1.05-2.41e9 on the 4080, so the engine-advantage figures are up to ~2.4× LOW. The IDENTITY and the rejection-cost split are structural and still hold — it is the numbers that must be re-measured before they are quoted.
- **Preprocessing is INSIDE the honest rate.** `samples_per_s` is the winning config's
  steady-state throughput and excludes parse + construction + ELS + warmup + MINING (median
  **20.7%** of GPAP's end-to-end wall, mean 27.4%); `sps_e2e_total` includes them. CMSGen's rate
  is samples/subprocess-wall, so it already carries ITS setup — comparing against our
  steady-state number charges one side only, and the L4 suite geomean reads **165.0x that way vs
  96.7x e2e**. Table 1 prints both (`GPAP/CMSGen` and `e2e`) plus `pre%`; quote `e2e` for a
  delivered-samples claim. Circuit ENCODING is in neither — it is `t_enc`, an encoder cost.
- Measurement protocol: [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md).

## Tests / methodology

- Circuit CNFs are GENERATED at test time from the committed `.bench` sources (no dependence on any
  user-local benchmark tree). `tests/conftest.py::circuit_cnfs` builds the matrix once per session.
- The no-change gate for a pure refactor: fixed K (`--k-cap N` + `pin_batch`), one config, seed 5 ⇒
  bit-identical model counts (+ model-set hashes on small instances). Rates alone are draw-noisy on
  low-success instances (±40%) — see docs/BENCHMARKING.md §3. ⚠️ **Seed streams re-baselined
  2026-09-01** (the sampling RNG moved from torch generators into the core): counts recorded
  before that date are not same-seed-comparable to counts after; the gate holds WITHIN each era,
  and cross-FACE (C++ vs Python) within the new one.
