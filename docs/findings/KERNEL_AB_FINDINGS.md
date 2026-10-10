# WAVE vs SCALAR propagate kernel — the L4 A/B (2026-08-23)

> **Status:** the RULE is current, the NUMBERS are superseded — re-measure before quoting one.
> The shuffle schedule of 2026-08-27 ([WAVE_SHUFFLE_FINDINGS.md](WAVE_SHUFFLE_FINDINGS.md))
> moved the crossover down; the bins' ORDER stands, the threshold and the ratios do not.
> **Measured:** 2026-08-23, NVIDIA L4 (Colab), torch 2.11/cu12.8 — on the pre-shuffle,
> pre-binary-edge engine.
> **Rule lives in:** `cpp/src/calibrate.cpp::gpap_dispatch_wave` (WAVE iff M ≥ 2048).

**Question**: how much faster is the warp-per-row WAVE propagate than the serial-trail SCALAR
kernel as a function of instance size — the number the paper states beside the dispatch rule.

**Answer (NVIDIA L4, Colab, torch 2.11/cu12.8, 2026-08-23)**: WAVE / SCALAR = **1.30× geomean
over 93 cells** (median 1.36×, WAVE faster on 81/93), rising monotonically with clause count.
Tables: regenerate with `eval/profiling/kernel_ab_tables.py` over the raw data
(`$GPAP_RESULTS/l4_campaign_20260823/kernel_ab_l4/kernel_ab.json`). The generated tables were
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
