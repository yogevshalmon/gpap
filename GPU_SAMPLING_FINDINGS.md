# GPU BCP-Completion Sampling — Findings

GPAP as a **GPU-parallel SAT solution sampler**. This is the first *produce-a-solution* task GPAP
wins (clause reduction and unprojected sampling were retired — they need search GPAP lacks). Code:
[`gpap/solver/sampler.py`](gpap/solver/sampler.py), harness
[`scripts/sample_benchmark.py`](scripts/sample_benchmark.py), tests
[`tests/test_sampler.py`](tests/test_sampler.py).

## The algorithm: GPAP = GPU-parallel CMSGen

CMSGen (the SOTA fast "uniform-like" sampler; Golia–Soos–Chakraborty–Meel, FMCAD'21) is, per its
Algorithm 1 *UniformLikeWitness*: *pick an unassigned variable at random → assign polarity 0/1
uniformly → unit-propagate → repeat* — "a randomized variation on CDCL, random var + random polarity
heuristics, no restarts." **GPAP runs exactly this, but K trajectories in parallel on the GPU.** The
one difference: CMSGen is built on CryptoMiniSat so it *backtracks* on conflict; GPAP has no search, so
it **discards** conflicted trajectories. Every fully-assigned, non-conflicted GPAP trajectory is a
guaranteed-valid model (checked on GPU: all clause `sat_count > 0` ⟺ satisfied).

**Two phases** (`GPUSampler.sample`): phase 1 fixes the independent support (`c ind`); phase 2
("completion") decides the remaining free vars to reach a full model, then we project onto the support.

## Simplification (the throughput lever) — and why it's legitimate

`simplify_cnf` applies **unit propagation** (assert forced units, drop satisfied clauses / falsified
literals, compact variables). On the sk_ benchmarks (6–51 % unit clauses) this shrinks M up to **7.6×**
(35.sk 10547→1380), so a much larger K fits in VRAM → proportionally higher throughput.

CMSGen *disables* preprocessing because variable/clause **elimination** (VE/BCE) changes the solution
space and breaks uniformity. Our simplification is **only unit propagation** — strictly
**model-preserving** (it just pins variables already forced in every model), a different class than
VE/BCE. We validate this rigorously in `tests/test_sampler.py::test_simplify_preserves_projection_set`:
the exact set of valid support projections is **identical** with and without simplification, and
reconstructed full models satisfy the **original** clauses. Reconstruction pins forced vars to their
value and samples *don't-care* support vars (free, in no remaining clause) uniformly at random — so the
support-projection distribution is preserved.

## Results vs CMSGen (laptop RTX 4080, single-core CMSGen)

Throughput (end-to-end wall incl. construction; 2-wise coverage = combinatorial-testing quality metric):

| benchmark | succ% | GPAP/s | CMSGen/s | **GPAP/CMS** | 2-wise GPAP / CMS |
|---|---|---|---|---|---|
| 35.sk_3_52 | 50 | 35,799 | 4,455 | **8.04×** | 61.8 / 61.8 |
| 10.sk_1_46 | 60 | 43,651 | 13,237 | 3.30× | 78.5 / 78.5 |
| 32.sk_4_38 | 31 | 14,007 | 5,603 | 2.50× | 61.9 / 61.9 |
| 27.sk_3_32 | 50 | 32,892 | 13,625 | 2.41× | 60.5 / 60.5 |
| 55.sk_3_46 | 32 | 9,900 | 6,906 | 1.43× | 64.3 / 64.3 |
| 109.sk_4_36 | 46 | 7,973 | 5,867 | 1.36× | 68.1 / 68.1 |
| 53.sk_4_32 | 38 | 11,042 | 8,256 | 1.34× | 60.5 / 60.5 |
| 111.sk_2_36 | 31 | 10,488 | 9,158 | 1.15× | 60.0 / 60.0 |
| 56.sk_6_38 | 33 | 4,497 | 4,437 | 1.01× | 69.7 / 69.7 |
| 51.sk_4_38 | 30 | 4,488 | 5,690 | 0.79× | 69.4 / 69.4 |

Geomean **~1.8× over CMSGen** at this conservative 20k-incl-setup setting (a dedicated steady-state 50k
run gave **~3.4×**, up to 16×). **2-wise coverage is identical to CMSGen** wherever GPAP samples — the
speed gain is not from lower quality. On a controlled formula, GPAP's distribution is TV=0.167 from
uniform vs CMSGen's 0.200 (GPAP slightly *more* uniform); both cover all solutions. Throughput scales
~linearly with K (VRAM-limited, not compute-saturated) → an A100 stacks a large multiplier on top.

**Barbarik uniformity test (the standard sampler tester, meelgroup/barbarik + bundled SPUR).** We wired
GPAP in as a custom sampler and ran the CM (uniformity) test on `blasted_case110`: GPAP passed **185
rounds with 0 rejects** (every round's bias stayed inside [loThresh 0.29, hiThresh 0.71]). Barbarik is
designed to *reject quickly* on non-uniform samplers (it rejects QuickSampler fast); GPAP runs clean,
like CMSGen — independent confirmation that GPAP is uniform-like, not just fast.

## Cracking the search-heavy lose-cases: structured decision order (auto-tuned)

The two families GPAP used to score ~0 on were diagnosed precisely (they fail for *opposite* reasons):

- **"service" family** (ActivityService/PhaseService/ProcessBean/…): `supOK` > 0 but `modelOK` = 0,
  with `freeAft` ≈ 627–1457. The support gets fixed; the **completion** never succeeds. The residual
  is **individually-free but jointly-constrained**: after failed-literal probing, ActivityService
  still has 663 free vars and **0/60 are truly forced** — huge solution count, yet random DPLL paints
  itself into a corner. Probing, careful B=1 (0.05 %), restart, and chronological backtracking all
  fail — the signature of *needs clause-learning* (mismatched to GPAP).
- **7/71/80.sk**: `supOK` = `modelOK`, both low, `freeAft` = 0 — here the *support itself*
  self-conflicts (a small-search problem, different from the service residual).

**The fix that works (free, sound): structured decision order.** `random_var=False` decides variables
in index order, which follows the program-synthesis formulas' **gate/definition DAG** (inputs → gates)
instead of a random permutation — so propagation determines the gates rather than the search
conflicting on them. With `random_var=False` + **B=1** the service family goes from ~0 to real:
**ActivityService modelOK 0.05 %→12 %, ProcessBean 0→11 %** — at **2-wise coverage identical to CMSGen**
(51.6 = 51.6) with **96 % distinct** samples (polarity stays random, so diversity/uniformity are
preserved) and GPU-sound. It needs B=1 *and* structured order together (B≥4 or random order → ≈0).

Because the right combo is **formula-dependent** (niche wants B=8/any order; service wants
structured+B=1; 7/80.sk want random order), it is chosen automatically. Two mechanisms:

- **`random_var="auto"`** — a cheap small-K probe of the `(order, B)` grid, then commit. Works, but is
  fragile: the small-K probe **under-rates large B** (kernel-launch overhead dominates B=8's ~3 rounds
  at K=2048), so it can mispick (e.g. it chose B=1 for 111.sk where B=8/B=1 are close), and it commits.
- **Adaptive config BANDIT (`configs=[(True,8),(False,1)]`, the harness default)** — the robust answer:
  each chunk runs ONE config **at full K**, chosen by its measured models/wall so far (decaying-epsilon
  greedy), dynamically switching to the winner and re-checking the rest. Because it measures full-K
  throughput on **productive** chunks (not a throwaway probe) and never permanently commits, it is both
  more accurate (it correctly found B=1 is *genuinely faster* on 111.sk) and more robust. It matches/
  beats `auto` on the niche (35.sk ~68k/s, 10.sk ~46k/s) AND cracks the service family
  (ActivityService ~870/s, ProcessBean ~80/s) in **one setting**, all GPU-sound. A `static` portfolio
  (all configs every chunk, K split) is also available but a slow config's K-independent round count
  drags every chunk ~1.5–2.7× — the bandit avoids that by running one config per chunk.

Honest scope: this makes GPAP **robust across the whole sk_ suite** (no family stuck at 0) — it does
**not** beat CMSGen's CDCL on the service family (30–870 samp/s vs CMSGen's thousands), but closes the
gap from *infinite* (0 samples) to a finite factor, with valid uniform-like samples. 7/71/80.sk's
support self-conflict is still only partly addressed (a feasibility/lookahead problem, separately
net-negative). Robustness fix: a formula that fully simplifies to **M=0 clauses** (e.g.
tableBasedAddition) used to crash `BCPEngine.init_state` (`clause_sizes.max()` on empty) — now handled;
it samples instantly (every assignment is a model). Regression-tested.

## The niche (where GPAP wins) — and where it doesn't

GPAP wins **iff both**: (a) `freeAftSup ≈ 0` — fixing the `c ind` support lets BCP *determine* the rest;
and (b) `supOK` high — a random support assignment is usually feasible. The numeric sk_ family meets
both. GPAP **fails** (succ≈0, produces ~0 samples) on:

- **The "service" family** (ActivityService, PhaseService, IterationService, ProcessBean, …):
  `freeAftSup` = 600–1045. Fixing the support leaves hundreds of vars undetermined by unit propagation,
  so completion **needs real search** — exactly what GPAP lacks and CMSGen's CDCL provides.
- **7.sk / 80.sk**: `freeAftSup`=0 but `supOK`≈5–6 % — the support self-conflicts (cascade), so random
  support assignments are almost always infeasible.
- **Baital config family** (busybox/buildroot/adder; zenodo 4022395): **no `c ind`** (sample all vars)
  → no small support → deep → out of niche.

So the niche is narrower than "all sampling benchmarks": ~10/29 of this suite. Within it, GPAP wins
clearly; outside it, use CMSGen.

## Trying to widen the niche — two levers, both fail (honestly)

**Gate / definability propagation (Arjun-style) — cannot help the service family.** Diagnostic: for a
valid model's support projection, fix it and count residual solutions. The service family
(ActivityService/PhaseService/IssueService/ProcessBean) has **≥200 residual solutions** (free-after-
support = 587–1571) — the residual is *genuinely under-determined* (massive free choice), not a unique
gate-defined value. So no gate/definability propagation can determine it; it needs **search** (which is
why CMSGen uses CDCL). Gates are out for these.

**Full-propagation support lookahead — raises success but is NET-NEGATIVE on throughput.** The other
losing class (7/70/71/80.sk) has free-after-support ≈ 0 but low `supOK` (≈5 %): the support self-conflicts
via deep cascades. We implemented full-propagation lookahead (`_decide_phase(lookahead=True)`: snapshot
engine state, try BOTH polarities to fixpoint, commit the feasible one; K halved to fit the snapshot). A
CPU concept test confirmed it raises feasibility 39–95×, and on GPU it raises success exactly as
predicted (7.sk 3.5→22.6 %, 71.sk 20→57.6 %, 10.sk 59.7→91.3 %, **35.sk 64.5→100 %**). **But throughput
drops** because the overhead (≈3× propagations + half K) exceeds the gain: 35.sk 16.3×→3.44×, 10.sk
4.38×→0.61×, 7.sk 0.04×→0.02×. The 1-step variant (`_immediate_conflict`, using the `sat_count==1`
test) is cheaper but doesn't help at all (misses cascades).

**Cheap `sat_count`-risk polarity (soft lookahead) — BACKFIRES, and we now know exactly why.** A
GPAP-native idea: score each polarity by how much it pushes clauses toward conflict (Σ over occ[lit]
of w(sat_count): w(1)=1e6, w(2)=4, w(3)=1) and pick the safer one — one occ-gather, no snapshot, no
K-halving. But it **lowers** success, often to 0 % (27.sk 49→0 %, 7.sk 6→0 %).

The root cause is a **semantic misreading of `sat_count`**: it is the count of *non-falsified*
literals (TRUE *or* unassigned), not a satisfaction/quality signal. The only genuinely-bad transition
when setting `lit` is `sat_count 1→0` (a real conflict). A clause going `2→1` becomes a **unit → it
propagates**, and in our niche (`freeAftSup ≈ 0`) propagation is *exactly how a trajectory completes
into a model*. So `w(2)=4` penalizes the engine's own success mechanism. **A/B confirmed** (B=1,
conflict-only corrected weights w(1)=1, else 0): original-weights soft 27.sk 44→0 %, 10.sk 71→30 %,
71.sk 26→3 %; **corrected weights ≈ baseline everywhere** (27.sk 44→44 %, 10.sk 71→70 %, 71.sk 26→26 %)
— it neither helps nor harms, because conflict-only weighting *is* the 1-step `_immediate_conflict`
test, which can't see the deep cascades that kill the low-supOK family. **No instantaneous counter
encodes the propagation closure**; "will this decision eventually conflict?" is answerable only by
propagating (full lookahead, net-negative). Lesson: **random polarity (CMSGen's Algorithm 1) is
near-optimal — a 1-step proxy can't beat it.** (`lookahead="soft"` retained only as a documented negative.)

**Conclusion: the niche cannot be cheaply widened by per-decision lookahead.** GPAP's advantage *is*
throughput; both lookahead flavours either trade it away (full) or wreck diversity (soft). The
remaining lever is *structural* — shrink the formula before sampling; see ELS below.

## Soundness note: `sat_count > 0` ⟹ satisfied ONLY at a full assignment

The GPU soundness check (`all clause sat_count > 0` over completed rows) is correct **only because it
is gated on `fully`-assigned rows**: at a complete assignment non-falsified ⟺ TRUE, so `sat_count > 0`
⟺ every clause has a TRUE literal ⟺ satisfied. On a *partial* assignment `sat_count > 0` only means
"not yet violated", NOT satisfied — never use it mid-propagation. It is in fact **redundant**: `active`
is cleared the instant any clause hits `sat_count == 0`, so active⊇fully rows already satisfy it. The
load-bearing independent check is the CPU `_satisfies()` in the tests.

## The real engine improvement: ELS (equivalent-literal substitution) — free, sound, ~1.5×

Profiling a chunk (laptop 4080) shows **propagation is 79–86 % of wall time**; setup/reset is 0.5 %,
pick/apply ~15 %. So the lever is *propagation*, whose cost scales with **M** (the `sat_count (K,M)`
traffic) — and K scales with per-trajectory memory (~M). Anything that shrinks the formula hits the
bottleneck **twice** (faster propagate + bigger K). (This also kills a "recycle dead trajectories"
idea — amortizing the 0.5 % setup saves nothing.)

`simplify_cnf_els` adds **equivalent-literal substitution** on top of unit-closure: it collapses
binary-implication SCCs (mutually-implied literals) to one representative, *removing variables*,
reusing the validated `equivalent_literal_substitution`. The binary-graph SCCs alone (a lower bound —
full GPU-BCP ELS à la the inprocessing branch finds ~2.8× more) remove **5–38 % of vars / 3–30 % of
clauses** on the sk_ niche. Measured end-to-end (construction+ELS in the wall, laptop 4080, B=8):

| benchmark | M cut | **ELS / plain** | ELS / CMSGen | (plain / CMSGen) | 2-wise plain=ELS |
|---|---|---|---|---|---|
| 35.sk_3_52 | 30 % | **2.16×** | 31.6× | (14.6×) | 61.8 = 61.8 |
| 10.sk_1_46 | 18 % | 1.85× | 4.9× | (2.65×) | 78.5 = 78.5 |
| 53.sk_4_32 | 12 % | 1.60× | 2.9× | (1.83×) | 60.5 = 60.5 |
| 32.sk_4_38 | 16 % | 1.45× | 5.0× | (3.48×) | 61.9 = 61.9 |
| 109.sk_4_36 | 5 % | 1.45× | 2.2× | (1.50×) | 68.1 = 68.1 |
| 111.sk_2_36 | 3 % | 1.34× | 1.8× | (1.33×) | 60.0 = 60.0 |
| 27.sk_3_32 | 9 % | 1.22× | 3.1× | (2.54×) | 60.5 = 60.5 |
| 55.sk_3_46 | 8 % | 1.06× | 3.3× | (3.12×) | 64.3 = 64.3 |

**Geomean ELS/plain = 1.48×; ELS/CMSGen = 4.16× (was 2.81× plain at this 50k-incl-setup setting).**
2-wise coverage is **identical** (no quality loss) and every run is GPU-sound. ELS is equisat- and
**support-projection-preserving iff reconstruction is consistent**: every removed variable equals its
representative in *every* model, so removed vars — *including don't-care ones* — are reconstructed from
their representative (one value per equivalence class), never randomized independently (else an
equivalent pair could be assigned unequal values = an invalid projection). Validated by
`tests/test_sampler.py::test_els_*` (projection set == CaDiCaL enumeration; equivalent vars stay
equal). On the laptop K is k_cap-capped so the **N-reduction → bigger-K benefit is largely masked** —
on an A100 with uncapped K it stacks directly onto the K-scaling lever, so the win should grow.
Enabled via `els=True` (default on in the colab driver; `--els` in `sample_benchmark.py`).

## The real improvement: B = 8 (free, +60 % throughput)

Sweeping `decisions_per_round` (B) — how many free vars to fix per round — found **B = 8 is the
throughput optimum** on every winner: fewer rounds (3–4 vs 5–7 at B=4) while success barely drops; B≥16
over-constrains and collapses success. No algorithm change, no cost.

| benchmark | B=4 (×CMSGen) | **B=8 (×CMSGen)** |
|---|---|---|
| 35.sk_3_52 | 17.4× | **24.0×** (107k/s) |
| 10.sk_1_46 | 3.5× | **6.0×** |
| 32.sk_4_38 | 3.7× | **4.7×** |
| 27.sk_3_32 | 3.5× | **4.4×** |
| 111.sk_2_36 | 1.4× | **1.8×** |

Geomean over CMSGen rises **~3.4× → ~5.5×** on the niche, with the best case at **24×** — all free, just
by setting B=8 (now the default in `scripts/sample_benchmark.py`). The A100's larger K stacks on top.

## Methodology notes

- Compared against **CMSGen** (built from meelgroup/cmsgen), the right *fast* competitor — not just
  CaDiCaL+blocking. UniGen3 (pyunigen) is the uniformity-guaranteed reference but too slow for a
  throughput comparison (runs ApproxMC counting first).
- Metrics follow the literature: samples/sec + **2-wise coverage %** (Golia et al.).
- Soundness is checked two ways: GPU (`all sat_count > 0` over completed rows) and CPU (reconstructed
  models vs original clauses, in tests).
- WSL2 spill: K is sized by a calibrated empirical per-batch cost (≈70·M + 40·N bytes/batch at 0.7×
  free VRAM); `compute_k_safe` alone is ~20× optimistic for these formulas.

## Next steps

- **A100 run**: measure the real K-scaling multiplier — and the ELS bonus, since uncapped K turns
  ELS's N-reduction into extra parallelism on top of the propagate speedup. Driver:
  `scripts/colab_sampling_benchmark.py` (+ `notebooks/gpap_colab_sampling.ipynb`), els on by default.
- **GPU-BCP ELS**: replace the cheap binary-graph SCCs with the inprocessing branch's
  `discover_implications` (BCP-to-fixpoint closure) — it finds ~2.8× more equivalences, so the M/N
  cut (hence the 1.48×) should grow. Reconstruction is already class-consistent, so it drops in.
- **Niche detection**: cheaply flag in-niche formulas (`freeAftSup≈0 ∧ supOK high`) to auto-route.
- **Barbarik** uniformity testing for a formal Accept/Reject vs CMSGen (already passed 185/0 once).
