# GPU Equivalent-Literal Substitution (ELS) + complete vivification — findings

> **Reader:** contributors and researchers — a historical record of GPU equivalent-literal substitution and vivification. A dated research log, not a user guide.
>
> **Status:** HISTORICAL — a RETIRED direction. GPU inprocessing for a CDCL solver was measured
> out on both legs: discovery is 7–15× slower than CaDiCaL's incremental probe, and the reduction
> helps CaDiCaL but not kissat.
> **Measured:** 2026-06/07. The 7–15× gap is a structural asymmetry (a CPU prober propagates
> each probe's delta; GPAP re-propagates the whole formula), not a property of any card.
> **Rule lives in:** nothing here ships. The code (`gpap/solver/inprocess.py`,
> `scripts/preprocess_solve.py`, …) exists only in git history, on branch
> `pysat_cadical_inprocess_intg`. What survived is ELS as a *sampling* preprocess
> (`gpap/sampling/preprocess.py`, ~1.4–1.5× there, on by default); the full-BCP-closure
> implication sweep shipped for a while as `gpap.implications` and was removed 2026-09-01, unused.
> **Record only:** this file is the sole record of a deleted subsystem, and the reason the
> direction is not re-tried. Summary in [INSIGHTS.md](INSIGHTS.md), "Retired application
> directions".

## What was built (all sound; 44 tests, 2 randomized soundness gates)

- `discover_implications()`: assume each of the 2N literals, BCP to fixpoint, read every implied
  literal. Yields the full implication relation, including hyper-binary implications through 3+
  clauses, plus failed-literal units.
- `equivalent_literal_substitution()`: union-find over mutually implied literals, substitute a
  representative, detect UNSAT (a class holding `l` and `¬l`), return a `var_map` for model
  lifting. The rule the shipping sampler ELS inherits: `x≡y` entails `¬x≡¬y` but BCP is not
  closed under contrapositive, so `(¬x,¬y)` must be unioned explicitly or the substitution is
  inconsistent (caught by the randomized gate); a budgeted variant must likewise keep a class and
  its mirror `¬C` together.
- Asymmetric vivification: assume `¬{C\{lⱼ}}`; a conflict proves `lⱼ` is a redundant interior
  literal. One removal per clause.

## Verdict

Both candidate claims of a GPU-inprocessing paper fail (RTX 4080, CaDiCaL 1.9.5, kissat 4.0.4,
`scripts/els_compute_boost.py`):

1. **"GPU speeds up discovery" — false.** Against CaDiCaL's own incremental in-C probing
   (`process(probe,decompose)`) the GPU is 7–15× slower on every velev instance, and CaDiCaL finds
   more simplification (1541 vars on 1.1-05 vs our 201 failed + 37 equivalent). Probing is
   inherently incremental: a CPU prober keeps the base trail and propagates only each probe's
   delta; GPAP re-propagates the whole formula per probe, in parallel, and pays a launch plus a
   2.4M-edge device-to-host transfer.

   | velev instance | GPU discovery | CaDiCaL probe+decompose | GPU/CPU |
   |---|---|---|---|
   | velev-sss-1.0-cl | 222 ms | 15 ms | 0.07 |
   | velev-pipe-uns-1.0-14 | 378 ms | 42 ms | 0.11 |
   | velev-pipe-uns-1.1-03 | 280 ms | 41 ms | 0.15 |
   | velev-pipe-uns-1.1-05 | 803 ms | 109 ms | 0.14 |
   | velev-pipe-o-uns-1.1-6 | 2190 ms | 193 ms | 0.09 |

   An earlier "3–89× faster" reading was a strawman: its CPU baseline was `2N × pysat
   propagate()`, which re-propagates the entire formula on every probe.

2. **"The reduction speeds up solving" — CaDiCaL only.** `els` on 5 velev instances: CaDiCaL
   1.53× (helped 5/5), kissat 0.82× (helped 1/5); `els+viv+U` on 3 small velev: 1.53× vs 0.77×.
   kissat's own ELS/probing/vivify already captures what we hand it, and the up-front reduction
   perturbs its heuristics.

The one fact that stands is reach: the full-BCP-closure ELS removed 4956 equivalent variables
across the 14-instance pool against 1762 for CaDiCaL's binary-graph `decompose` (2.81×, GPU >
CaDiCaL on 11/14). But reach anti-correlates with benefit — corr(elsV, conflict ratio) = −0.50 on
hwverif (n=12); the wins were on tiny-reach instances (velev-pipe elsV=9 → 2.42×) and the largest
reach was neutral-to-harmful (hoons 472 → 0.70) — so it has no value for solving.

## What the intermediate rounds found (all consistent with the verdict)

- Structured pool (14 instances, 25 s budget): no single reduced config robustly beats plain
  (`els+viv+U` 1.139×, 1.053× ex-outliers); a per-instance SAFE portfolio (plain ‖ all) 1.314×,
  never worse than plain, 0 unsound. Hardware-verification-only pool (24 instances): portfolio
  1.216× wall / 1.234× conflicts, 21/24 helped, concentrated on velev-pipe / schup / velev-sss
  (1.5–2.2×). Trajectory chaos, the same shape as every prior inprocessing phase.
- Deterministic heuristics (representative choice, class budget, vivify stacking; hwverif n=12,
  noise-free conflict counts): none beats plain robustly; best is `els` at 1.045× while losing
  7/12. The win/lose sign is instance-intrinsic and config-invariant (velev-sss-1.0-cl wins under
  every config; its near-twin velev-sss-1.0-05 loses under every one).
- Family scan (12 families): velev-pipe/-sss `els` 1.8–1.9× and goldb-CEC vivification
  1.16–1.23× were the publication candidates before the kissat check retired them.
- Scaling wall: discovery on the 536k-clause velev-pipe-uns-1.1-07 effectively hung (dense
  formula, tiny K, thousands of probe chunks).

## Considered and not adopted

- ELS once up front with CaDiCaL's `decompose` disabled: untested; the chaos result predicts no
  sign flip.
- A per-instance predictor instead of a portfolio: wins and losses live on structurally
  near-identical instances and reach anti-predicts, so it would have to run the solve.
- A100 / fused-kernel discovery: would narrow but not close a 7–15× gap against a 15–193 ms
  target.
