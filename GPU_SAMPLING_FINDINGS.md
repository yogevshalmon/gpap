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

## CircuitSAT sampling — the "Recurrent CircuitSAT Sampling" family (2026-07-01)

Focus shifted to the circuit benchmarks the differentiable GPU samplers use (DEMOTIC / High-Throughput
SAT Sampling / Recurrent CircuitSAT Sampling, arXiv 2502.08086 / 2502.08673 / 2502.21226). Pipeline in
`scripts/bench_to_cnf.py`: parse a gate-level `.bench`, unroll `T` cycles (flip-flop output at `t` shares
its D-input's var at `t-1`; reset state 0), Tseitin to CNF, pin the final-cycle outputs to a simulated
reachable target, sample the primary inputs (the `c ind` support).

**Benchmarks added:**
- **ISCAS-89 sequential** `s386/s526/s832/s1196` (already present).
- **ITC'99** `b01`–`b14` (gate-level `.bench` from `cad-polito-it/I99T`; all parse — AND/NAND/OR/NOR/NOT/DFF).
  A **100k-clause guard** in `run_circuits` auto-skips oversized `(instance, cycle)` unrolls (b05/b12 at
  T=25, b14 at T>1); b15+ are excluded (too big even shallow).
- **ISCAS-85 combinational** `c432`…`c7552` (DEMOTIC's 10; T=1, pin one output).

**The circuit config win — soft-lookahead, and why it needs the bandit.** Naive random-polarity BCP-
completion self-conflicts on deep circuit cascades. The cheap `soft` risk-polarity (score each polarity
by the `sat_count` of the clauses it touches, pick the safer; 2 occ-gathers, **no** extra propagation,
**no** K halving — unlike full lookahead) flips this. Measured on the laptop RTX 4080:

| instance | rand,B=8 (default) | rand,B=1,soft |
|---|---|---|
| b03 T=25 | 10% succ, 2.5k samp/s | **88% succ, 13.6k samp/s** |

But soft is **not universal**: it *hurts* `b12` T=10 (50% vs 95% for B=8), and on deep high-support
`s1196` T=25 the B=1 configs time out (>300 rounds) so B=8 wins. So soft is one option in an
adaptive-bandit **`CIRCUIT_PORTFOLIO = [(True,1,"soft"), (True,8,False), (False,1,False)]`**; the bandit
runs each at full K and spends the budget on the per-instance winner (validated end-to-end: b01/b03 pick
soft, b06 T=25 and the deep instances pick B=8). This required extending the sampler's config tuples from
`(random_var, B)` to `(random_var, B, lookahead)`; the sk_ default portfolio stays 2-tuple (no soft —
soft is net-negative *there*, see the killer-app-regime note: config value is regime-dependent).

**Unique-solutions metric (decoupled from timing).** Re-added the distinct-solution readout, this time as
a **count computed off the throughput path**: the timed run stays `collect=False` (pure GPU generation
rate), and a separate coverage run dedups the collected models by support-projection
(`distinct_projections`), reported as `uniqG`/`uniqC` columns and `gpap_distinct_frac` in the headline.
This avoids the earlier mistake of folding the slow host-side reconstruction into a `unique/sec` rate. On
the large sk_ spaces distinct_frac ≈ 1.0 (raw ≈ distinct); on small circuit spaces it drops (e.g. ~66% in
a quick run) and is genuinely informative as a diversity/correctness check.

**Driver:** `scripts/colab_sampling_benchmark.py --circuits-only` (or `run_all(circuits_only=True)`; the
notebook cell now defaults to it) runs only the circuit suite, swept over unroll depth, and K-sweeps the
best circuit for the A100 story (validated slope ≈1.04, throughput ∝ K, on b03 T=5). All circuit outputs
GPU-sound.

### Bandit high-K regression + fix (found on A100)

Adding the slow B=1/soft configs to the portfolio **regressed the A100 throughput** vs the pre-soft run
(s386 T=1: 8.4M/s → 381k/s). Root cause (reproduced on the laptop with a small `n_time`): on a big-K GPU
**one chunk already produces ≫ `n_time` models**, so the timing loop exits after the *first* cold-start
chunk — which is `plan[0] = soft`. The bandit never measured B=8, and soft (B=1, many rounds) is slower
on shallow circuits. The reported samples/s was also the exploration-*blended* rate (dragged by the slow
B=1 chunks and first-chunk CUDA warmup). It was **not** the unique-counting (that's off the timing path,
measured separately at 1.5–10× — kept off precisely so it can't distort samples/s).

Three-part fix:
1. **Always cold-start every config once** before `n_target` can end the timing run (`sampler.py`), so
   B=8 is measured even when a single chunk exceeds `n_time`. Bounded by `max_chunks`; cold-start chunks
   are productive.
2. **Report the best-*throughput* config's steady-state rate** (`best_rate_sps` = max models/wall over a
   config's own chunks), not the end-to-end blend — the rate a real run achieves after committing to the
   winner. (Excludes one-time construction, which amortises over a real run's millions of samples.)
3. A tiny-K **warmup chunk** before the timed run, to kill first-chunk JIT skew.

Verified on the laptop (forcing the "1 chunk ≥ n_time" regime): shallow s386.c1/s832.c1 now pick **B=8**
(935k / 1.57M samp/s, was soft at 168k / 360k); medium s386.c5 and b03.c25 keep **soft** (its genuine
win); deep instances pick B=8. Re-running the full ISCAS-89 set, the bandit picks B=8 almost everywhere
(soft wins mainly s386 T=5 and the b-instances) — so many of the buggy run's "soft" selections were
artifacts of never measuring B=8. K-scaling stays linear (slope 1.06).

## The Triton BCP megakernel — one launch to fixpoint (2026-07-02)

Phase-A profiling showed deep-circuit BCP is **latency-bound by the Python round loop** in
`BCPEngine.propagate()` (per-round: ~15-20 kernel launches + 2 CUDA syncs + a `torch.unique` sort;
12.9× more trajectories cost only 3.9× wall) and that its per-round temporaries **over-reserve memory
5.6×** (s1196.c25 K=4096: 6,646 MB reserved / 1,179 MB allocated). Pure-PyTorch fixes are structurally
impossible (the dedup needs a sort; the loop itself must go), so the sampler's
`apply_assumptions_bulk + propagate` pair was replaced by a **custom Triton kernel**
(`gpap/solver/bcp_triton.py`) that runs decisions + BCP-to-fixpoint in **one launch**.

**Design (why it's simple and provably right):** one Triton program per trajectory; the program owns
its rows of `sat_count`/`lit_sum`/`assignment` exclusively and processes its pending-literal **trail
sequentially** (per literal, the occ list holds distinct clauses) ⇒ **no atomics, no barriers** — a
textbook sequential BCP per trajectory, K-wide parallelism saturates the GPU. BCP fixpoints are
confluent, so non-conflicted rows are **bit-identical** to the PyTorch oracle (tested exactly:
`tests/test_fast_bcp.py`, 30 random CNFs × multi-step + a real circuit + sampler-level same-seed
identical model sets). Opt-in (`GPUSampler(use_kernel="auto")`), PyTorch engine kept as
fallback/oracle; Triton ships with torch (no nvcc), so it works on the laptop and on Colab unchanged.

**Results (RTX 4080, end-to-end sampler, same configs/success — identical semantics):**

| instance | oracle | kernel | gain |
|---|---|---|---|
| s526 T=1 | 4.17M/s | **14.7M/s** | 3.5× |
| b03 T=25 | 17.0k/s | **90.9k/s** | 5.3× |
| b07 T=25 | 9.0k/s | **41.3k/s** | 4.6× |
| s386 T=25 | 2.6k/s | **12.6k/s** | 4.9× |
| s1196 T=25 | 3.1k/s | **9.7k/s** | 3.2× |
| 35.sk (niche) | 137.9k/s | **1.04M/s** | 7.5× |
| 10.sk (niche) | 82.3k/s | **508.9k/s** | 6.2× |

Micro (single deep cascade): 8–38× over the PyTorch propagate; reserved memory tracks allocated
(s1196: 6,646→1,262 MB at K=4096) — and the oracle **spills** at K=7,204 where the kernel runs in
1.6 GB. The sk_ niche was a feared regression and instead gained the most: BCP was ~80% of its wall
too. 62/62 tests green with the kernel auto-on.

**Follow-ups unlocked:** (1) K re-calibration for the kernel path (its true per-batch footprint is
~(sat+4)·M + 6N — much smaller than the old effective footprint ⇒ bigger K on the same VRAM, stacks
on A100); (2) decide-in-kernel v2 (whole chunk = one launch) if the remaining Python decide loop
dominates; (3) bandit portfolio re-tune under the new cost model (B=1's success edge got cheaper).

## v2: the fused-trajectory kernel + K-boost + portfolio re-tune (2026-07-02)

With v1 the Python **decide loop** became the bottleneck (measured 41–79% of chunk wall: per decision
round `_pick_decisions` builds (K,N) masks + topk + a sync). v2 (`_bcp_trajectory_kernel`) runs the
WHOLE trajectory in ONE launch: each program walks a per-chunk permuted variable order, picks the next
B still-free vars (exactly `_pick_decisions`' first-B-free-in-priority-order semantics), assigns a
stateless splitmix-hash polarity, and propagates in-kernel. Two critical details:
- **The phase boundary must not be straddled**: a decision group stops at the support→completion
  boundary so phase 1 reaches fixpoint over the support before completion vars are decided (the
  pre-fix straddle collapsed b07 from 100% to 0% success at large B; regression-tested).
- **K-boost**: a fused chunk never builds the (K,N) decide masks, so its true footprint is just
  engine state + trail (≈ (sat+4)·M + 10·N per batch). Fused chunks run at `K_fused` (up to 262k;
  kill-switch validated ≤5.3 GB reserved on the densest instances) while v1 configs keep the safe K —
  per-chunk, inside the same bandit run.

Exposed as a THIRD config-tuple value: `(rv, B, "fused")` — the adaptive bandit measures it against
the v1-kernel and soft configs per instance and commits to the winner (it loses nowhere by
construction). New measured portfolios: `CIRCUIT_PORTFOLIO = [(T,64,off),(T,1,soft),(T,8,fused),
(T,1,fused)]`, `DEFAULT_PORTFOLIO = [(T,8,off),(T,1,fused),(F,1,off)]`. Notably **B=1's success edge
is now free** (decision rounds cost nothing in-kernel), so B=1-fused wins broadly.

**End-to-end (RTX 4080, production `gpap_sample` path, vs the same-day pre-kernel baseline):**

| instance | pre-kernel | v1 kernel | v2 portfolio | total | winner |
|---|---|---|---|---|---|
| s526 T=1 | 4.17M/s | 14.7M/s | **70.2M/s** | 16.8× | B=1-fused @K=262k |
| b03 T=25 | 17.0k/s | 90.9k/s | 69.3k/s* | 4.1× | B=1-soft |
| b07 T=25 | 9.0k/s | 41.3k/s | **53.3k/s** | 5.9× | B=1-fused |
| s386 T=25 | 2.6k/s | 12.6k/s | **56.2k/s** | 21.7× | B=1-fused |
| s1196 T=25 | 3.1k/s | 9.7k/s | **19.6k/s** | 6.4× | B=8-fused |
| 35.sk | 137.9k/s | 1.04M/s | **1.40M/s** | 10.2× | B=1-fused |
| 10.sk | 82.3k/s | 508.9k/s | **1.65M/s** | 20.1× | B=1-fused |

*run-to-run seed variance vs the v1 row; soft remains b03's winner both times.

Soundness: fused models are valid by construction (fully-assigned + conflict-free) and CPU-verified in
tests (65/65 green); the fused polarity source is a splitmix64 hash (balanced; binomial-clean in
8-groups) rather than the torch stream — watch 2-wise coverage parity on the next full CMSGen run.

## v3: packed 16|16 (sat|XOR) clause state + cached-flush reset (2026-07-03)

The fused kernel's inner loop is **bandwidth-bound on scattered clause-state sectors**: every occ
entry does read-modify-write on `sat_count` (int8/16) AND `lit_sum` (int32) — 4 uncoalesced 32-byte
sectors (each occ entry hits a "random" clause; each scalar access fetches a full sector). v3 packs
both counters into ONE int32 per (trajectory, clause): **low 16 bits = sat_count, high 16 bits = XOR
of the non-falsified literal indices**. XOR replaces SUM as the remaining-literal trick — the xor of
a one-element set IS the element — and unlike the sum it **never overflows**, so 16 bits suffice
whenever literal indices do (2N ≤ 65535, checked on the POST-simplification formula; every current
bench instance qualifies, incl. b12 T=25 after ELS: N 42,051 → 16,344; the unpacked kernel remains
as the fallback). The packed update is one instruction pair:

    st = (tl.load(st_row + c) - 1) ^ (neg << 16)     # -1 can't borrow into the XOR half:
    tl.store(st_row + c, st)                          # sat ≥ 1 at every decrement (sat==0
    sat = st & 0xFFFF                                 # conflicts and halts the row first)
    forced = (st >> 16) & 0xFFFF                      # when sat == 1

→ 2 sectors per occ entry instead of 4. Second win, same design: the packed state is
**kernel-private** (`TritonBCP._packed`), and the post-units-flush state is row-identical every
chunk, so `reset_fused()` broadcast-restores ONE cached (M,)/(2N,) row pair instead of
`reset_state()+propagate()` — no (K,M) sat/lsum rewrite, no per-chunk Python-round units flush
(the XOR half of the cached row is one numpy `bitwise_xor.reduceat` pass over the occ CSR).
`K_fused` re-budgets to 4M+10N per row (was (sat_b+4)M+10N) → **K_fused +10–30% across the table**.

**Validation (the noise-methodology playbook, RTX 4080):**
- Pinned same-seed gate: **27/27 (instance,config) rows bit-identical model counts** vs the
  pre-packed tree, walls 1.03–1.54× (median ~1.22×) at equal pinned K. Packed-vs-unpacked
  model-set identity is also a permanent test (`test_packed_vs_unpacked_identical_models`).
- 24-chunk portfolio, 22 circuit instances: geomean best-rate **1.27×**, no real row <0.9×
  (b14 T=1 read 0.88× once; re-probes 0.99–1.00× — ms-scale walls, noise).
- Peak reserved 7.0 GB vs baseline's 7.6 GB (fast reset kills the temporaries), no kill-switch trips.

**fsoft joins the K-boost**: fsoft chunks are fused-trajectory launches too (no (K,N) masks), so they
now run at `K_fused` as well — b03 T=25 e2e 2.30×, c1355 1.95×, c499 1.62×; the one soft spot
(s832, whose winner is a *fused* config the change doesn't touch) re-probed at 0.90×/0.90× = its
established noise band, while its own e2e sps improved.

**Final vs the pre-packed baseline: geomean 1.31× e2e samples/s, 1.23× best-rate, worst instance
0.93× (noise-band).** Everything is a pure win on top of v2's 4–22×.

**Measured OUT — in-kernel retry-on-conflict** ("dead slots become extra attempts"): 3 in-kernel
restarts (row state restored from the cached flushed rows, attempt index mixed into the polarity
hash) **tripled per-slot success exactly as predicted** (b04 11→34%, b11 3→12%, s386 37→83%) — and
tripled wall with it: models/wall 0.81–0.97× everywhere. Conflicts on the success-bound residuals
die LATE (a failed attempt ≈ a full trajectory), so retry ≈ running more chunks, minus kernel
overhead. Also: wrapping the retry loop around the packed kernel taxed the RETRIES=0 path ~10%
median (loop-carried state / register pressure) — structural lesson: **experimental paths get their
own kernel; never wrap the accepted hot loop.** The experiment was removed; the residuals still
need a learning-class lever, not restarts.

## Entailed-clause mining: GPAP probes the formula against itself (2026-07-03)

The remaining CMSGen losses (the "service" family + a handful of sk/circuit instances at succ
0-12%) die from LATE BCP conflicts, which restarts don't fix (measured above) but CLAUSES do:
`gpap/solver/miner.py` exhaustively BCP-probes (fused kernel, cached-flush rows, one launch per
chunk) **L1 = every literal to fixpoint** (real conflict assuming l ⇒ unit ¬l — the BCP-visible
BACKBONE, hundreds-to-thousands of units on every losing instance), **L2 = all support pairs**,
**L3 = all support triplets**. Each learned clause is ENTAILED ⇒ the model set is EXACTLY
unchanged (pysat-verified in tests) — but the sampler's own propagation now kills doomed
trajectories EARLY, and the units shrink the simplified formula (M drops ⇒ K and speed rise).

Wired into `gpap_sample(mine="auto")`, gated on the (already-present) warmup chunk's success
rate < 15%: healthy flagships skip at ZERO added cost; losing instances pay 0.1-30s ONCE
(60s hard budget). Two integration bugs mattered: (1) mining's reserved VRAM poisoned the
rebuilt sampler's free-memory read → K collapsed to 1 (throughput 899/s instead of 6.5M/s!) —
`torch.cuda.empty_cache()` after mining; (2) giant PyTorch probe temporaries tripped the WSL2
watchdog ("device not ready") — the kernel-based prober fixed it structurally.

**Full 29-instance sk suite (RTX 4080, vs the 2026-07-03 reference table; CMSGen rates frozen
from it): geomean GPAP/CMSGen 4.65× → 27.3×; suite geomean 5.87× over the previous GPAP.**
2-wise coverage IDENTICAL to CMSGen on every mined instance (quality preserved exactly).

| instance | before | after | vs CMSGen |
|---|---|---|---|
| 80.sk / 70.sk / 7.sk | 6.4k / 12k / 6k/s | 6.6M / 6.5M / 2.8M/s | 1.4-2.6× → **880-1,480×** |
| 51.sk | 92.6k/s | 1.36M/s | 16.8× → **246×** |
| UserServiceImpl | 2.9k/s | 62.4k/s | 0.31× → **6.8×** |
| ActivityService | 9.0k/s | 38.6k/s | 1.27× → **5.5×** |
| ProjectService3 | 759/s | 8.1k/s | 0.38× → **4.1×** |
| IterationService / PhaseService | 548 / 495/s | 9.0k / 7.1k/s | 0.12×/0.07× → **2.0× / 1.0×** |
| polynomial | 127k/s | 1.1M/s | 3.4× → **29.5×** |
| b03 T=25 (circuit) | 427k/s | 1.75M/s | — (**4.1×**, 1,007 units) |
| c3540 T=1 (circuit) | 3.7k/s | 29.1k/s | 0.72× → **~10×** |

Healthy instances gate-skip: b07/s1196/c1355/b13 at 0.99-1.10×; all sk flagships ≥0.94×
(within their noise bands). Residuals: NotificationServiceImpl2 (0.13× — joint constraint
beyond ternary reach; deep mining finds 1,400+ clauses and full lookahead lifts success 50×,
but neither reaches CMSGen throughput: genuinely needs in-search learning) and b04/b11-class
circuits (mining finds ~nothing; their residual isn't backbone-shaped).

**The bug that mining flushed out** (would never have been found from the sampler side):
`apply_assumptions_bulk` re-propagated literals ALREADY assigned by a prior units flush —
`_bulk_propagate_lits` decrements occ[lit] unconditionally, so the double-decrement corrupted
sat_count, materialised garbage forced literals (lit_sum of ≥2 remaining literals is a bogus
index), and reported FALSE CONFLICTS. The sampler never triggers it (decisions are always fresh
free vars) and output soundness is unaffected (conflicted rows are discarded) — but any
assumption-checking flow that assumes an implied literal after a flush silently saw false
conflicts. Fixed with an already-assigned filter on the pair list (mirrors the kernel's
skip-duplicates guard); regression test + miner entailment tests added.
