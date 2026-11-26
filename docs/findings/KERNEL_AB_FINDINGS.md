# The WAVE kernel: its A/B against scalar, and its memory ordering

> **Reader:** contributors and researchers — the evidence behind the rule that picks the WAVE or the scalar kernel, and behind the WAVE kernel's relaxed atomics. A dated research log, not a user guide.
>
> **Status:** current. A WAVE row's atomics are relaxed, ordered by barriers ("Relaxed atomics",
> 2026-11-26). The dispatch rule `M ≥ 2048` stands: re-measured on that engine, no clause-count
> threshold below it speeds up the small formulas without slowing some ("The threshold on the
> relaxed engine", 2026-11-26). The L4 tables of 2026-08-23 predate the shuffle schedule
> ([WAVE_SHUFFLE_FINDINGS.md](WAVE_SHUFFLE_FINDINGS.md)) and both changes: quote no number from
> them.
> **Measured:** 2026-08-23, NVIDIA L4 (Colab), the pre-shuffle, pre-binary-edge engine;
> 2026-11-26, Athena A100-SXM4-40GB and Colab T4, main `15f81f6` against the relaxed kernels.
> **Rule lives in:** `cpp/src/calibrate.cpp::gpap_dispatch_wave` (WAVE iff M ≥ 2048) and
> `cpp/kernels/bcp.py` ("Memory ordering under WAVE"), gated by
> `tests/engine/test_kernel_table.py`.

**Question**: how much faster is the warp-per-row WAVE propagate than the serial-trail SCALAR
kernel as a function of instance size — the number the paper states beside the dispatch rule.

**Answer (NVIDIA L4, Colab, torch 2.11/cu12.8, 2026-08-23)**: WAVE / SCALAR = **1.30× geomean
over 93 cells** (median 1.36×, WAVE faster on 81/93), rising monotonically with clause count.
Tables: regenerate with `eval/profiling/kernel_ab_tables.py` over the raw data
(campaign `l4_campaign_20260823`, `kernel_ab_l4/kernel_ab.json`). The generated tables were
removed from the committed docs on 2026-09-02 rather than left stale.

Concretely, what the shuffle schedule did to this A/B: `_wave_propagate` got 1.80× at the same K
while scalar `_propagate` was unchanged, so the M≈0.5–9k tie band under-dispatches WAVE today.

The bins agree with the RTX 4080 calibration (forced-kernel A/B at fixed K, 25 instances: M=2,098
1.04, M=2,882 1.41, M=14,012 1.57, 1.27–1.66 for every larger M, whole-set geomean 1.39–1.46);
the L4 whole-set geomean is lower only because this set holds many more small-M cells (the
ABC-collapsed circuits), where the kernels tie.

## Method (`eval/profiling/kernel_ab.py`, driver `eval/colab/colab_kernel_ab_l4.sh`)

GPAP only. 141 (instance, encoder) cells: ITC'99+ISCAS-89 T=25, ISCAS-85 T=1 one pin, EPFL T=1,
HWMCC'17 T=25, each with `tseitin` and `abc:fast`, `rare` targets, plus the 29 sk_ CNFs. Per cell:

1. Validation before timing: both kernels forced via `$GPAP_KERNEL` at a pinned K (`k_cap=4096`,
   so the cap binds and K is identical), 3 chunks, seed 5. Required: sound on both, the sampler's
   `engine.mode` provenance equal to the forced kernel, and identical model counts (BCP is
   confluent, so the same K/seed/chunks give the same models). 141/141 passed.
2. Timing: same process, same CNF, pinned `b8` (a free bandit picks differently per arm), mining
   off (it is wall-budgeted, so a faster kernel would mine more and the arms would run different
   formulas), 20 s wall × 2 reps with arm order alternated (SM clocks drift within a run), best-of
   per arm, K recorded per arm (identical on 99/99 comparable cells). Ratio = wave/scalar.
3. Exclusions as the paper tables apply them: 8 vacuous-pin cells, 5 zero-search cells, 30 cells
   where neither kernel produced a model in 20 s (17 EPFL, 10 sk, 3 seq), and a ≥100-models floor
   on both arms (6 cells; the max "5.98×" is one of them). All 99 comparable cells 1.33×; with the
   floor 1.30×.

22 cells reached the 2M-model quota in <2 s (8 chunks); re-measured wall-bound (20 s) they
agreed within ±3% on every cell, so the A/B is stable from a 0.3 s to a 20 s window at 8+ chunks.
The wall-bound rows are the ones tabulated (`remeasured_wallbound`).

## Results

| clauses M | cells | WAVE/SCALAR geomean | median | min–max | WAVE faster |
|---|---:|---:|---:|---:|---:|
| < 2k | 20 | 1.01× | 1.05× | 0.63–1.51 | 13/20 |
| 2k–20k | 48 | 1.33× | 1.37× | 0.80–1.69 | 43/48 |
| 20k–100k | 17 | 1.48× | 1.47× | 1.23–1.80 | 17/17 |
| > 100k | 8 | 1.64× | 1.64× | 1.36–1.88 | 8/8 |
| **all** | **93** | **1.30×** | **1.36×** | 0.63–1.88 | **81/93** |

Per family: sequential 1.36× (25/25), HWMCC 1.48×, sk_ 1.44× (17/17), ISCAS-85 1.18×, EPFL
1.16×. Per encoder: tseitin 1.33×, abc:fast 1.22×, CNF 1.44×. The largest gains are the 8|24
giants (b14 tseitin M=1.05M 2.03×, bobsynthor M=1.15M 1.88×); the largest losses are small
EPFL/ISCAS cones (router tseitin M=802 0.63×, c432 abc M=371 0.78×).

**The dispatch rule picked the faster kernel on 77/93 cells.** All 16 disagreements lie at
M = 549–8,726, where the kernels are close; the costliest mis-pick runs at 0.66× of the faster
kernel (epfl/cavlc abc:fast, M=1,851), most are within ±10%. `M ≥ 2048` is about right on the L4
as on the 4080; a finer rule would buy a few percent on a handful of small cones and is not worth
a second calibration constant.

## Caveats

- The implication counters of the two kernels differ ~0.6% on conflicted rows (which clause
  empties first depends on propagation order); models and decisions agree exactly. Gate on models.
- Mining is off here, so absolute rates are not the headline rates; only the ratio is the result.
- Absolute rates on low-success instances are draw-noisy (±40%), hence the model floor.
- Only the L4 numbers are tabulated; the 4080 bins agree.

## Relaxed atomics (2026-11-26, issue 51)

**What changed.** Triton's default atomic is `acq_rel` at GPU scope: a fence before
(`MEMBAR.ALL.GPU`, `ERRBAR`) and an L1 flush after (`CCTL.IVALL`), around every clause update and
every bitmap push. A row belongs to one program, so its atomics need only atomicity and their
return values: they are `sem="relaxed"` now. The fences also ordered the threads of a row
against each other (thread 0's seeds and decisions, the lanes' pushes onto the same bitmap
words, the trail slots the lanes write and read), so `_wave_propagate` holds two barriers
instead: one before every wave and one after the last. The queue's job counter, the one atomic
rows share, keeps its fence. The scalar kernels have no atomics and are unchanged.

**The machine code** (cuobjdump of every cubin, sm_80). The 10 scalar kernels are the same
instructions as main's. Every WAVE kernel lost its `MEMBAR` and `CCTL.IVALL` and gained two
`BAR.SYNC`; the queue keeps its counter's two. The October pilot (branch `xor_count_kernel`,
`1ba39ee`, `ABLATION_FINDINGS.md` §4 there) measured 1.42× at the kernel level with the decision
scan's bitmap reads moved to L2 and no barriers. Here the barriers took that place: after one,
every thread of the program sees every write made before it, atomics included.

**Same answers.** Data and drivers: `$GPAP_RESULTS/relaxed_20261126/`.

| check | A100 | T4 |
|---|---|---|
| suites on the branch | CUDA 833, host 616, experiments 210, NVIDIA OpenCL 687 passed | CUDA 792, host 575, experiments 209 passed |
| the identity tests (propagation, queue, CLEAN, core) | 5 more runs, all passed | 5 more runs, all passed |
| stress: WAVE against scalar on 6 random families and 8 sampling CNFs, both packed splits: apply, CLEAN, trajectories (b1, b8, soft), queued checks, probe widths 1 and 2 | 15 rounds, 3,645 checks, 0 differ | 4 rounds, 972 checks, 0 differ |
| main against the branch, 68 cells, WAVE forced, batch 4,096 for 6 chunks, every model hashed and re-checked | 272 pairs (b8, b1, s1, soft) identical | 136 pairs (b8, b1) identical |
| probe sweeps, flags-only checks and `gpap mine`, 26 SAT Competition 2024 formulas and 10 cells | identical conflicts and mined clauses | — |

55 of the A100 pairs and 27 of the T4's are hard cells where 6 chunks find no model in either
build; their success rates and counts agree too.

**Faster.** The sampler at its default batch, WAVE forced, one configuration pinned, 8 s timed,
main and the branch alternated (A100: 3 runs each; T4: 2). Per cell the ratio of the medians,
branch over main; per row the geometric mean of those ratios (cells counted in brackets):

| | A100 b1 | A100 b8 | T4 b1 | T4 b8 |
|---|---:|---:|---:|---:|
| all cells | **1.52×** (65) | **1.42×** (62) | **1.24×** (22) | **1.17×** (20) |
| range | 0.99–2.01 | 0.98–2.17 | 1.00–1.39 | 0.99–1.36 |
| M < 2k (engine's clauses) | 1.51× | 1.37× | 1.32× | 1.27× |
| 2k–20k | 1.61× | 1.52× | 1.23× | 1.18× |
| 20k–100k | 1.34× | 1.27× | 1.13× | 0.99× |
| > 100k | 1.00× | 1.00× | 1.00× | 1.00× |

No cell is below 0.97. As the pilot found, the formulas whose occurrence lists fit in L1 gain
most, and the memory-bound ones (300k clauses after preprocessing) tie. The A100 gains more
than the T4.

**The queue** (A100 alone on its node, 16 formulas from 14k to 4.4M clauses, 1,048,576 tuples
or 262,144 cubes a call, median of 6 calls): probe widths 1 / 2 / 3 at 1.11 / 1.11 / 1.14× main,
flags-only checks of 8 / 32 literals at 1.07 / 1.09×, 1.10× overall; the calls under 0.1 s vary
by more than the effect. A build without the two barriers ran at 1.10× as well: they cost
nothing measurable.

## The threshold on the relaxed engine (2026-11-26, issue 52)

`eval/profiling/kernel_ab.py` on the relaxed kernels, the 68 cells as CNFs, scalar against WAVE
forced, validation first (identical models at a pinned batch on every cell, both GPUs), 8 s (A100)
or 6 s (T4) timed, 2 runs each. WAVE over scalar on the cells below the threshold (M here is the
CNF's clause count; the rule reads the engine's, after preprocessing, and picks scalar on all of
these):

| cell | M | T4 b1 | A100 b1 | T4 b8 | A100 b8 |
|---|---:|---:|---:|---:|---:|
| epfl ctrl | 549 | 1.59 | 1.36 | 1.94 | 1.54 |
| c432 | 748 | **0.72** | **0.89** | 1.85 | 1.69 |
| epfl int2float | 788 | 1.71 | 1.76 | 2.08 | 1.66 |
| epfl router | 802 | **0.58** | **0.66** | 1.36 | 1.46 |
| tableBasedAddition | 961 | 0.98 | **0.91** | **0.76** | 1.00 |
| polynomial | 1,027 | 1.67 | 1.65 | 2.41 | 2.22 |
| epfl dec | 1,169 | 0.97 | **0.64** | 0.99 | 1.00 |
| c880 | 1,410 | **0.88** | 1.11 | 2.13 | 2.21 |
| registerlesSwap | 1,493 | 1.36 | 1.41 | 2.00 | 1.76 |

At `b8` WAVE wins on 8 of the 9 on both cards. At `b1`, the configuration with one decision a
round and so the thinnest wavefront, it loses on 4 to 5 of them, by up to 0.58×. No feature
read off the formula separates the losers: router and int2float have 2.9 and 3.4 occurrences per
literal and 64% and 66% binary clauses, and run at 0.6× and 1.7×. Above the threshold the rule
already picks WAVE where `b1` loses (epfl adder 0.58–0.64×, priority 0.83–0.86×), as it did
before the change. So any clause-count threshold below 2048 slows some cells: the rule stays,
and issue 52 stays open for a predictor that separates them. One that would capture the `b8`
gain reads the decisions per round as well as M (WAVE for a trajectory of 8 decisions at any M);
the dispatch rule does not do that today (CONTRIBUTING.md, "Kernel dispatch is decided once"),
and tableBasedAddition would still lose at `b8` on the T4.

Chunked checks that read assignments (the other caller of the rule, `check_sets` with
`assignments=True`, CUDA and NVIDIA OpenCL, 8,192 sets of 4 and 16 literals on 55 cells) agree
exactly between the kernels; at 2 to 20 ms a call their timing is noise and decides nothing.

