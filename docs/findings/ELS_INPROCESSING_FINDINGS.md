# GPU Equivalent-Literal Substitution (ELS) + Complete Vivification — Findings

> **Retired direction — historical record (2026-06/07).** GPU inprocessing for a CDCL solver was
> measured out on both legs (discovery is 7–15× slower than CaDiCaL's incremental probe, and
> the reduction helps CaDiCaL but not kissat) and is not pursued. The code this file describes
> (`gpap/solver/inprocess.py`, `scripts/preprocess_solve.py`, …) exists only in git history, on
> the branch named below. Two things survived into the shipping tree: ELS as a *sampling*
> preprocess (`gpap/sampling/preprocess.py`, a measured ~1.4–1.5× win there, on by default). The
> full-BCP-closure implication sweep shipped for a while as `gpap.implications` and was REMOVED
> 2026-09-01, unused. Summary in
> [INSIGHTS.md](INSIGHTS.md), "Retired application directions".

Branch: `pysat_cadical_inprocess_intg`. Companion to the clause-addition and vivify findings
(see memory `inprocessing-binary-discovery-finding`). This phase added the **untried** reduction
lever — equivalent-literal substitution — plus the missing **asymmetric** half of vivification,
and characterized them on a structured benchmark pool.

## What was built (all sound — 44 tests green, 2 randomized soundness gates)

- **`GPUInprocessor.discover_implications()`** (`gpap/solver/inprocess.py`): assume each of the 2N
  literals, BCP **to fixpoint**, read every implied literal from `engine.assignment`. Yields the
  full implication relation (the binary-implication-graph closure **including hyper-binary
  implications through 3+ clauses**) + failed-literal units as a byproduct.
- **`equivalent_literal_substitution()`** (`gpap/solver/preprocess.py`): union-find over
  *mutually-implied* literals → equivalence classes → substitute a representative, removing
  variables. Detects UNSAT (a class with `l` and `¬l`); returns a `var_map` for model lifting.
  **Soundness bug found & fixed:** `x≡y` entails `¬x≡¬y`, but BCP is **not** closed under
  contrapositive, so the negation pair isn't always discovered — must union `(¬x,¬y)` explicitly
  or substitution becomes inconsistent (caught by the randomized gate; 500 random CNFs now pass).
- **Asymmetric vivification** (`vivify_clauses(asymmetric=True)`): vectorized leave-one-out —
  assume `¬{C\{lⱼ}}`; a conflict proves `F ⊨ C\{lⱼ}`, so `lⱼ` is a redundant *interior* literal
  (the half the old conflict-prefix-only vivify missed). One removal per clause (sound; joint
  removal would need re-derivation). 400 random CNFs pass (equisat + subset).
- **Eval** (`scripts/preprocess_solve.py`): `els`, `els+viv+U` configs; per-family geomean;
  plain-UNK **rescue** probe; **GPU-ELS vs CaDiCaL `decompose`** value-add diagnostic.

## Result: 14-instance structured pool (`benchmarks/suite_structured.json`, RTX 4080, 25s budget)

`.experiments/els_structured.json`. 11 solvable + 3 plain-UNK (2 lec_mult multiplier-CEC, 1 mdp-unsat).

| config | geomean ×no-GPU | ex-2-outliers | improved |
|---|---|---|---|
| plain | 1.000 | — | — |
| els | 0.973 | — | 4/11 |
| viv | 0.933 | — | 3/11 |
| viv+U | 0.923 | — | 3/11 |
| **els+viv+U** | **1.139** | 1.053 | 6/11 |
| full | 0.854 | — | 2/11 |
| **SAFE portfolio (plain ‖ all)** | **1.314** | **1.231** | — |

Per-family SAFE: **hwverif 1.35** (n=5) · planning 3.48 (n=1) · simon 1.06 · cec 1.05.
**0 unsound, 0 rescues, 0 rescue verdict conflicts.**

## The two headline findings

### 🟢 NOVEL (positive): GPU-ELS finds **2.81× more equivalences than CaDiCaL's own `decompose`**
GPU removed **4956** equivalent variables vs CaDiCaL `decompose`'s **1762** across the pool;
**GPU > CaDiCaL on 11/14** instances. The reason is structural and defensible: CaDiCaL's
`decompose` is the SCC of the **binary-clause** implication graph, whereas GPU-ELS is the SCC of
the **full BCP closure** — it sees equivalences that only emerge after propagating through 3+
clauses (hyper-binary). Examples: simon ~1000–1255 vars vs CaDiCaL 352; hoons 472 vs 294. This is
the first GPU-native, exhaustive ELS and it strictly out-reaches the solver's internal pass.

### 🔴 NEGATIVE (the catch): **ELS reach does NOT convert to solve speedup** — same CDCL chaos
The extra equivalences GPU finds do not help CaDiCaL solve faster, and may hurt:
- Where ELS reach is **largest**, the solve effect is **neutral-to-negative**: hoons (elsV=472)
  → `els` 0.67×; simon (elsV ~1000+) → `els` ~1.0× (and simon solves in <1 ms — timing noise).
- Where there are **big speedups**, ELS removed **almost nothing**: velev-pipe-uns-1.0-14
  (elsV=9) → 2.52×; velev-sss-cl (elsV=12) → 1.75×; mdp-28-14-sat (elsV=48) → `els+viv+U` 3.48×
  (while `els` alone *timed out*). These are CDCL trajectory luck from tiny vivify/reduction
  perturbations, **not** ELS reach.

So speedup is **uncorrelated with reduction magnitude** — exactly the trajectory-chaos result
from the clause-addition/vivify phases, now confirmed for the strongest reduction (variable
removal). The "gentler because it removes variables" hypothesis did **not** hold: removing a
variable still reshapes CaDiCaL's search unpredictably.

## Honest bottom line (matches the chosen "characterize + portfolio" bar)
1. **The lever is sound and novel** (GPU out-reaches the solver's ELS 2.8×) — a real, defensible
   GPU-SAT contribution independent of wall-clock.
2. **No single reduced config robustly beats plain.** `els+viv+U` (1.14, or 1.05 ex-outliers) is
   the best single config — a hair above neutral, outlier-driven. ELS stacked on viv+U beats
   either alone (it "rescues" viv+U on several instances), so if you must pick one config, it's
   `els+viv+U`.
3. **The only robust win is the per-instance SAFE portfolio: ~1.31× (1.23× ex-outliers), never
   worse than plain, 0 unsound** — same shape and magnitude as every prior inprocessing phase.
   Best family: **hardware verification (velev/hoons) at 1.35×**.
4. **No rescues**: ELS does not crack CDCL-hard instances (multiplier-CEC `lec_mult`, mdp-unsat).
   Multiplier miters are XOR-deep, not literal-equivalence-rich (elsV only 0–358).
5. `×wGPU` is ~0.02–0.03 on RTX 4080 (discovery ms ≫ small solve ms) — A100/fused-kernel-improvable,
   consistent with prior phases.

## Caveats
- Small pool (11 solvable, 1–5 per family); geomeans outlier-sensitive (the prior 8–24-instance
  pools repeatedly gave misleading optimism — only validated 50-instance pools were stable).
  The `els+viv+U 1.14` is essentially mdp(3.48)+velev-pipe(2.52); ex-those it's 1.05.
- simon (the biggest ELS-reach demo) solves in <1 ms → its speedups are timing noise; it is a
  reach demonstrator, not a timing instance.

## Where the value-add *could* pay off (open)
The 2.8× reach only matters if the extra equivalences land on instances where they prune search.
Candidates not yet tested: instances where CaDiCaL's `decompose` is budget-throttled on a
formula whose hyper-binary equivalences are load-bearing (large structured UNSAT). Also untested:
applying ELS **once up front then disabling CaDiCaL's decompose** (avoid double work) — but the
chaos result suggests this won't flip the sign.

---

# Follow-up: can a HEURISTIC give a single-config win (no portfolio)? — conflict-level study

Goal: on the hardware-verification niche (where ELS did best), find a **deterministic** reduction
heuristic whose single config beats plain — replacing the race-and-keep-winner portfolio. Metric
upgraded to **conflict count** (CaDiCaL is deterministic ⇒ conflicts are noise-free and
reproducible even on sub-second instances, where wall-ms is pure jitter). `scripts/els_heuristic_search.py`,
`benchmarks/suite_structured_big.json` (23 inst), `.experiments/els_heur.json`. Configs: `els`
(min-var rep), `els_deg` (keep highest-**degree** variable, drop peripheral copies), `els_budget`
(substitute only the largest ~half of equivalence classes), `els_deg+vivU` (asym-vivify→units→ELS_deg).
(Also fixed a real **budget soundness bug**: `max_classes` could split a class from its mirror
`¬C` → a half-renamed variable → unsound; now keeps mirror pairs together, 2400-check gate passes.)

## Result (hwverif n=12, conflict-ratio = plain_conflicts / config_conflicts, >1 = fewer):
| config | geomean | win/lose |
|---|---|---|
| `els` (min-var) | **1.045** | 5 / 7 |
| `els_deg` (degree rep) | 1.005 | 6 / 6 |
| `els_budget` (top-½ classes) | 0.968 | 6 / 6 |
| `els_deg+vivU` | 0.962 | 4 / 8 |

## Conclusions (definitive, noise-free)
1. **No reduction heuristic gives a robust single-config win.** Best is plain `els` at 1.045 — but
   it *loses on 7/12*. Representative-choice (`els_deg`) and budget (`els_budget`) make it **worse
   or neutral**, not better. The "better heuristic avoids the portfolio" hypothesis is **not
   supported**.
2. **The win/lose sign is instance-intrinsic and config-invariant.** velev-sss-1.0-**cl** wins under
   every config (2.26/2.08/1.69/1.78); the near-identical velev-sss-1.0-**05** (same family, N 1453
   vs 1516, M 12.5k vs 12.8k, both UNSAT) loses under every config (0.39/0.39/0.50/0.43). Rep-choice
   flips the sign on only 3/12 — random noise, no pattern.
3. **ELS reach NEGATIVELY correlates with benefit: corr(elsV, conf-ratio) = −0.50.** Where GPU finds
   the MOST equivalences (hoons 472, manol 645/420), ELS is neutral-to-**harmful** (0.70, 0.92); the
   wins are on TINY-reach instances (velev-pipe elsV=9 → 2.42×; velev-sss-cl elsV=12 → 2.26×). So the
   2.8× "value-add" reach is not just useless — it actively concentrates the harm. CaDiCaL's
   conservative binary-only `decompose` is conservative for a reason.

**Bottom line for the heuristic direction: dead end.** There is no deterministic reduction-side
heuristic (representative, budget, or vivify-stacking) that robustly beats plain on the niche — the
chaos is intrinsic to the instance, not a knob we control. A single-config net win is not
achievable this way; only a per-instance **predictor** could help, but the wins/losses live on
structurally near-identical instances and reach anti-predicts, so static prediction is near-hopeless
— any predictor would have to *run* the solve (portfolio-adjacent), which is what we set out to avoid.
(simon instances solve in **0 conflicts** — pure preprocessing — so they're a reach demo only, not a
timing/conflict signal.)

---

# Focused result: PORTFOLIO improvement on the hardware-verification niche

Since a single-config win isn't reachable, we measured the one robust lever — the per-instance
**SAFE portfolio** (run plain ‖ reduced configs, keep the winner; never worse than plain) — on a
focused, hardware-verification-ONLY pool of **24 instances** (velev / hoons / manol-pipe / schup,
M 12.5k–109k, all solve in-band). `benchmarks/suite_hwverif.json`, `.experiments/els_hwverif.json`.
Both wall-ms and the noise-free **conflict** portfolio are reported (they agree closely).

## Portfolio improvement over plain CaDiCaL (geomean, never < plain)
| portfolio (cores) | wall-ms | conflicts | reduction wins |
|---|---|---|---|
| plain ‖ els+viv+U (2) | 1.147 | 1.142 | 14/24 |
| plain ‖ els, els+viv+U (3) | 1.188 | 1.197 | 16/24 |
| **plain ‖ all 5 configs (6)** | **1.216** | **1.234** | **21/24** |

Best single config on this niche: **els+viv+U 1.083× (improved 14/24)** — consistently slightly
positive here, unlike the mixed pool where it was outlier-driven. So even without the portfolio,
`els+viv+U` is a defensible default *on hardware verification specifically*.

## Where the gain lives (conflict portfolio per subfamily)
| subfamily | n | portfolio | best instance |
|---|---|---|---|
| **velev-pipe** | 2 | **2.16** | 2.60 |
| **schup** | 2 | **1.79** | 1.79 |
| **velev-sss** | 2 | **1.50** | 2.26 |
| velev-eng/npe | 4 | 1.09 | 1.18 |
| manol-pipe | 10 | 1.09 | 1.32 |
| hoons | 4 | 1.08 | 1.33 |

**The niche-within-the-niche is velev-pipe + schup + velev-sss (~1.5–2.2× portfolio, up to 2.6×).**
manol-pipe (the bulk) and hoons are mostly neutral — the portfolio just falls back to plain there.
Note schup *breaks* the reach anti-correlation: it has the **highest** ELS reach (3722 vars, 20.7%
literal cut) AND benefits (1.79×) — so some structured families really do have load-bearing
hyper-binary equivalences. velev-pipe wins with the *opposite* profile (elsV=9) — confirming again
that no single feature predicts the sign, but the portfolio captures both.

## Honest summary of the portfolio deliverable
- **Hardware verification: ~1.22× wall-clock (1.23× conflicts) portfolio, zero downside, 21/24 helped.**
  A practical 3-core portfolio (plain + els + els+viv+U) already gives **1.19×**.
- Concentrated value (1.5–2.6×) on velev-pipe / velev-sss / schup; neutral on manol / hoons.
- All sound (0 unsound, verdict always agrees). `×wGPU` still poor on RTX 4080 (discovery ≫ solve);
  the portfolio gain is a *solve-side* win that an A100/fused kernel would make wall-clock-real.

---

# Branch-out: which FAMILY gives the best ROI (for publication)?

Broad scan across 12 structured families (`benchmarks/suite_broad.json`, 27 in-band instances;
`.experiments/els_broad.json`). The `viv` config here is the COMPLETE vivification (conflict-prefix
+ asymmetric). Per-family geomean speedup vs plain (wall-ms), sorted by best single-config:

| family | n | els | viv | best single | SAFE (ms) | mechanism |
|---|---|---|---|---|---|---|
| **velev-pipe** | 2 | **1.92** | 1.12 | **1.92 (els)** | 1.92 | **ELS** |
| **velev-sss** | 1 | **1.77** | 1.09 | 1.77 (els) | 1.77 | **ELS** |
| **schup** | 2 | 1.33 | 1.39 | 1.53 (els+viv+U) | 1.70 | ELS + vivify |
| **goldb-cec** | 4 | 0.98 | **1.15** | 1.23 (els+viv+U) | 1.28 | **VIVIFY** (ELS neutral) |
| mrpp | 1 | 0.31 | 1.03 | 1.26 (viv+U) | 1.26 | vivify |
| hoons | 3 | 0.86 | 0.85 | 0.94 | 1.14 | (portfolio only) |
| ibm | 1 | 0.82 | 1.09 | 1.13 | 1.13 | vivify |
| velev-eng | 4 | 0.93 | 0.93 | 1.03 | 1.05 | ~neutral |
| manol-pipe | 5 | 0.98 | 0.94 | 0.98 | 1.03 | ~neutral |
| x9 | 2 | 1.01 | 0.74 | 1.01 | 1.02 | ~neutral |
| vmpc, mdp | 1,1 | <1 | <1 | <1 | 1.00 | hurt (portfolio→plain) |

## The two publication candidates (distinct mechanisms)
1. **velev-pipe / velev-sss (pipelined-microprocessor formal verification) → ELS.** A *single
   deterministic config* (`els` alone, no portfolio) gives **1.8–1.9×** (velev-pipe-uns-1.0-14
   `els` 2.43×, velev-sss-1.0-cl 1.77×). This is the strongest result in the whole project: a
   net single-config win, not a portfolio. Mechanism is exactly GPAP's novelty — full-BCP-closure
   equivalences these miters are rich in.
2. **goldb-CEC (combinational multiplier equivalence checking, n=4) → VIVIFICATION.** ELS is
   neutral (these are XOR-deep, few literal equivalences), but COMPLETE vivification helps
   (`viv+U` 1.22×, `els+viv+U` 1.23×; goldb-dalumul `els+viv+U` 1.52×). A separate, vivify-driven
   story. (This is the "try only vivification" lever paying off where ELS doesn't.)

**Best ROI = velev-pipe/-sss (ELS single-config ~1.8×).** Negative controls behaved as expected:
random/PHP/coloring absent (0-yield, didn't even reach the pool), vmpc/mdp hurt (portfolio→plain).

## CONFIRMED (champion eval, `benchmarks/suite_champions.json`, 11 inst, 90s budget)
Added 3 BIGGER velev-pipe (154k–304k clauses) to test scaling, and all 6 in-band goldb-CEC.

| family | n | els | viv | viv+U | els+viv+U | SAFE | verdict |
|---|---|---|---|---|---|---|---|
| **velev (pipe+sss+big)** | 5 | 1.55 | 1.05 | — | **1.57** | **1.77** | **the publication family** |
| velev-pipe (small) | 2 | 1.92 | — | — | 1.62 | 1.92 | ELS |
| velev-pipe (big 154k/304k) | 2 | 1.16 | — | 1.47 | **1.61** | 1.61 | ELS+vivify |
| velev-sss | 1 | 1.79 | — | — | 1.39 | 1.79 | ELS |
| goldb-CEC | 6 | 1.00 | 1.11 | **1.16** | 1.14 | 1.20 | vivify (ELS neutral) |

### THE RESULT FOR PUBLICATION
**velev pipelined-microprocessor formal verification is the best-ROI family, and the win SCALES
with instance size.** A single deterministic config `els+viv+U` gives **1.57× (n=5)**, the SAFE
portfolio **1.77×**, and the *largest* instance tested (velev-pipe-o-uns-1.1-6, 304k clauses) gives
the *second-biggest* win: **els+viv+U 2.13×, viv+U 2.08×**. Per-instance: velev-pipe-uns-1.0-14
`els` 2.41×, velev-sss-1.0-cl `els` 1.79×, velev-pipe-uns-1.1-03 `els` 1.53×. Mechanism is GPAP's
novel full-BCP-closure ELS on small instances, shifting to ELS+complete-vivification on large ones.

**goldb-CEC is the robust secondary family (n=6): a VIVIFICATION win (`viv+U` 1.16×, SAFE 1.20×),
ELS-neutral** — confirming the "try only vivification" lever pays off precisely where ELS doesn't
(XOR-deep multiplier miters have few literal equivalences but redundant literals vivify away).

### Caveats / next step (A100 is the lever)
- velev n=5: small but now spans 12k–304k clauses (a real size range), and the trend is the
  *opposite* of decay — bigger instances win bigger. The full Velev FVP/SSS suite has 26+ pipe
  miters, but the rest are M=536k–8.8M and need an A100 (RTX 4080 can't size K for them).
- **Scaling wall found:** GPU discovery on the 536k-clause velev-pipe-uns-1.1-07 effectively hung
  (dense formula → tiny K → thousands of probe chunks). The dense BCP engine doesn't scale to
  ~500k+ clauses on 16 GB; the sparse engine (`[[sparse-bcp-engine]]`) / A100 is required to run
  the larger half of the family — which is exactly where the wins are largest.

---

# RIGOROUS VALIDATION: "do we actually boost compute?" + the right-solver check
`scripts/els_compute_boost.py`, `.experiments/boost_velev_els.json`. Every piece timed separately
(GPU implication-BCP vs GPU vivify), CPU baseline = the SAME 2N failed-literal/implication probes
to fixpoint via CaDiCaL's `propagate()`, and the SOLVE speedup re-measured on **CaDiCaL 1.9.5 AND
kissat 4.0.4 (SOTA)** with a verdict-soundness gate.

## ⚠️ RETRACTED: the "GPU discovery is 3–89× faster than CPU" claim was a MEASUREMENT ARTIFACT
The first cut compared GPU discovery to a CPU baseline of `2N × pysat propagate(assumptions=[lit])`.
Verified facts (cross-checks all pass — the code does the *right* work): GPU & CPU find the SAME 201
failed literals and matching edge counts; pure pysat call overhead is negligible (1.1 µs/call). The
problem is the **baseline algorithm**: pysat's `propagate()` is NOT incremental, so it
**re-propagates the entire 154k-clause formula on every one of the 20052 probes** (~1.8 ms each) —
work a real prober never repeats. That naive loop took 36 s; the GPU's 0.8 s looked 44× faster, but
it was beating a strawman.

### The FAIR comparison: GPU vs CaDiCaL's own incremental in-C probing (`process(probe,decompose)`)
| velev instance | GPU discovery | CaDiCaL probe+decompose | GPU/CPU |
|---|---|---|---|
| velev-sss-1.0-cl | 222 ms | **15 ms** | 0.07 |
| velev-pipe-uns-1.0-14 | 378 ms | **42 ms** | 0.11 |
| velev-pipe-uns-1.1-03 | 280 ms | **41 ms** | 0.15 |
| velev-pipe-uns-1.1-05 | 803 ms | **109 ms** | 0.14 |
| velev-pipe-o-uns-1.1-6 | 2190 ms | **193 ms** | 0.09 |

**The GPU discovery is 7–15× SLOWER than CaDiCaL's incremental CPU probe across every velev instance.**
And CaDiCaL's probe even finds *more* total simplification (1541 vars on 1.1-05 vs our 201 failed +
37 equiv). Root cause is structural: failed-literal/ELS probing is **inherently incremental** —
a CPU prober keeps the base trail and propagates only each probe's delta, backtracking cheaply.
GPAP instead re-seeds and re-propagates the whole formula for every probe (just in parallel) and
pays kernel-launch + 2.4M-edge device→host transfer. On these cheap-per-probe formulas the GPU's
brute-force-parallel approach cannot beat the CPU's algorithmic incrementality. **So: GPAP does NOT
boost the inprocessing-discovery compute.** (An A100 would narrow but not obviously close a 7–15×
gap, and CaDiCaL's 15–193 ms is already a hard target.) The honest residual GPAP advantage is only
*exhaustiveness/reach* — it computes the full closure, finding ~2.8× more equivalences than
CaDiCaL's budgeted `decompose` (37 vs 7 equiv vars here) — but earlier results show that reach
ANTI-correlates with solve benefit, so it isn't worth the time cost.

## 🔴 The right-solver check changes the solve story: pre-reduction is solver-dependent
| config | CaDiCaL 1.9.5 | kissat 4.0.4 (SOTA) |
|---|---|---|
| els (5 velev) | **1.53× (helped 5/5)** | **0.82× (helped 1/5)** |
| els+viv+U (3 small velev) | **1.53× (helped 3/3)** | **0.77× (helped 1/3)** |

**The velev solve-speedup is largely a CaDiCaL artifact.** On the SOTA solver (kissat) the
pre-reduced formula is *neutral-to-slower* — kissat's own aggressive ELS/probing/vivify already
captures (and better-integrates) what we hand it, so our up-front reduction is redundant and
perturbs its heuristics (velev-sss kissat 0.61×, velev-pipe-1.1-03 0.53×). Only velev-pipe-1.0-14
is helped on kissat (1.39×). End-to-end on RTX 4080 `×wGPU ≈ 0.26–0.94` (discovery cost ≈ solve
savings on these sizes); `×wCPU` is far worse (0.30) — i.e. if you did the reduction on CPU it
would never pay, so the GPU is strictly the right engine for it.

## Honest publication takeaway (after full validation — both legs fail)
The two candidate claims for a GPAP-inprocessing paper BOTH fail rigorous checks:
1. **"GPU speeds up the discovery compute"** — FALSE. Against a real incremental CPU prober it's
   7–15× slower (the 3–89× was a non-incremental-pysat-loop artifact).
2. **"The reduction speeds up SAT solving"** — only on CaDiCaL (1.5×); FALSE on kissat/SOTA (0.8×),
   which already does this work internally.

**Conclusion: GPU-accelerated inprocessing (ELS/vivify pre-reduction fed to a CDCL solver) is NOT a
viable GPAP application.** Inprocessing is inherently incremental/CPU-friendly; GPAP's batched-BCP
strength doesn't fit it. The only surviving fact is *reach* (exhaustive closure finds ~2.8× more
equivalences than budgeted decompose), but reach anti-correlates with solve benefit, so it has no
value. The strong GPAP results remain in the OTHER directions (`[[cube-and-conquer-finding]]`,
`[[gpu-oneshot-unsat-prover]]`) where the parallelism unit (independent assumption sets / cubes /
lookahead) actually fits the GPU. Inprocessing should be retired as a GPAP application.
