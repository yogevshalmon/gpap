# How uniform is GPAP, really — measured against an exact oracle, and against CMSGen

Investigated 2026-07-29/30 (RTX 4080 laptop, WSL2, libgpap core). The question: GPAP's sampler is
the GPU-parallel form of CMSGen's Algorithm 1, and neither has a uniformity guarantee. Previous
readouts could only say "consistent with uniform" (the `l2_ratio` gross-skew check) or "Accept"
(Barbarik, at parameters coarse enough to be affordable). This is the first measurement against a
**known** target distribution.

Headline, in three parts: **(1) on the flagship projected-circuit regime the shipped pipeline is
already at the exact-uniform noise floor (TV 0.040 vs the control's 0.040, ok 9/12) and beats
CMSGen (0.087, 4/12) — while on random k-CNF over ALL variables it is clearly worse than CMSGen,
so the aggregate must never be quoted without the split; (2) there is an exactness knob nobody had
noticed — at `B >= |support|` the sampler is exact rejection sampling and indistinguishable from
uniform on 24/24 instances, bounded by a yield collapse when `p_accept` is small; (3) the
portfolio's `soft` config was non-uniform badly enough that Barbarik rejects it 5/5, and is now
removed from the defaults and opt-in behind `--allow-soft`.**

---

## 1. The instrument: generated instances with an enumerable solution set

`eval/fuzz_uniformity.py` generates small formulas, enumerates their solution set exhaustively, and
runs a chi-square goodness-of-fit test plus total variation against the true uniform. Two families,
both chosen so "uniform over the support projections" is UNAMBIGUOUS:

| family | support | oracle |
|---|---|---|
| `cnf` | every variable (projection = identity) | exhaustive evaluation over all 2^n assignments |
| `circuit` | the primary inputs of a random Tseitin-encoded gate DAG, outputs pinned | exhaustive simulation over all 2^k input vectors |

The `circuit` support is INDEPENDENT by construction (every gate var is functionally determined),
so each projection has exactly one extension. That matters because CMSGen samples full models and
we project: on a non-independent support the two tools would be targeting different distributions
(projections weighted by extension count) and the comparison would be meaningless. Both families
sidestep it.

The oracle is deliberately **solver-free** — no pysat, no BCP — so a bug in our own machinery
cannot hide a bug in the sampler. `gen_circuit` additionally PROVES its encoding before the
instance is used: it checks that the simulated rows satisfy every Tseitin clause over the whole
truth table and satisfy the pin clauses exactly on the accepted set, which together imply
`models(CNF) projected == accepted set`.

### The controls are the point

A table showing "CMSGen skews on 27/29" is indistinguishable from "our chi-square is broken" unless
the instrument is calibrated, so every campaign runs two synthetic samplers:

- **negative control** — draws exactly uniform from the enumerated set. Flagged **0/30**. Without
  this, nothing else on the page is evidence.
- **positive controls** — a known TV from uniform. At `n = 100|S|` x 3 seeds, TV 0.05 and TV 0.10
  were both detected **30/30**; at the weaker `n = 50|S|` x 2 seeds, TV 0.05 was detected only 2/6.

So the campaign's sensitivity floor is **TV ~ 0.05**, and an "ok" verdict means "no skew detectable
at TV 0.05", never "uniform".

### Verification of the instrument (2026-07-30 audit)

Every layer was re-derived independently rather than trusted:

| what | check | result |
|---|---|---|
| TV, chi2, dof | recomputed from first principles vs `uniformity_stats` | exact to 1e-12 |
| unseen cells | a solution never sampled must still charge TV and chi2 | exact |
| degenerate sampler | always the same solution must read TV = 1 - 1/S | exact for S=10, 100 |
| `chi2_sf` | median of chi2_50 (49.33) must give sf = 0.5 | 0.5002 |
| **the uniform floor** | theory says E[TV] = 0.399*sqrt(S/n) = **0.0399** at n=100\|S\| | control read 0.0392-0.0400; `b16`/`b32` read 0.0392 |
| **generation** | re-derive all 24 solution sets with pysat blocking-clause enumeration | **24/24 exact SET equality** |
| row -> histogram | plant a known TV=0.20 distribution of real solution rows | measured 0.194-0.202, counts == bincount |
| unsound detection | inject 2 non-solutions | detected exactly 2, excluded from n |
| seed independence | Fisher assumes independent runs | 3/3 distinct histograms, GPAP and CMSGen |
| **Fisher calibration** | combined p must be Uniform(0,1) under H0 | quantiles .049/.269/.505/.941 vs .05/.25/.50/.95 |
| **false positives** | exact-uniform through the whole verdict path, 120 campaigns | **0/120** at alpha=1e-3 |
| **power** | same path, planted TV | 0.05 -> **97%**, 0.10 -> 100%, 0.15 -> 100% |

The floor check is the load-bearing one: `b16`/`b32` reading 0.0392 is not "close to our control",
it is the value a *perfect* sampler must produce at this sample size. And the campaign's headline
numbers reproduce on re-run (cnf-000: `pipe:default` 0.4467, `cmsgen` 0.2795, matching the recorded
0.447 / 0.279).

## 2. GPAP vs CMSGen, paired on identical files

**Instances must have |support| > B or the campaign measures nothing** — at B >= |support| the
result is trivially uniform (§3), so a suite with small supports flatters B=8 for a structural
reason rather than a distributional one. The headline campaign therefore generates only instances
with `|support| >= 12`, strictly above the default B=8: **24 instances, supports 12-16, |S| 28-382,
3 seeds**, chi-square p combined across seeds with Fisher's method, flagged at p < 1e-3
(Bonferroni-ish for ~24 instances). Every arm is run on the identical file.

| sampler | ok | skew | TV median | TV max | vs CMSGen |
|---|---|---|---|---|---|
| `gpap:b16`, `gpap:b32` (B >= \|sup\|) | **24** | **0** | **0.0392** | 0.045 | better TV on 23/24 |
| `gpap:b8` | 9 | 15 | 0.220 | 0.652 | never skews where CMSGen doesn't; better TV on 11/24 |
| `gpap:b1` | 9 | 15 | 0.244 | 0.526 | better TV on 10/24 |
| **`pipe:default`** (full pipeline) | 9 | 15 | 0.298 | 0.504 | better TV on 7/24 |
| **`pipe:circuit`** (full pipeline) | 9 | 15 | 0.325 | 0.489 | better TV on 7/24 |
| `pipe:circuit+soft` (`--allow-soft`) | 7 | 17 | 0.335 | 0.776 | better TV on 8/24 |
| `cmsgen` | 4 | 20 | 0.177 | 0.322 | — |
| `ctl:uniform` (negative control) | 24 | 0 | 0.0400 | 0.042 | PASS |
| `ctl:tv0.05` (power probe) | 0 | 24 | 0.058 | 0.060 | detected 24/24 |

**Zero unsound rows** across all 10 arms x 24 instances x 3 seeds — every emitted row was in the
enumerated solution set.

The `pipe:*` rows are `sample_file` exactly as the CLI runs it (parse, unit-closure simplify,
warmup, entailed-clause mining, the adaptive bandit, the GPU bit-projection output), so they are
what a user actually gets.

### DO NOT QUOTE THE AGGREGATE — it averages two opposite regimes

The suite is half random k-CNF and half circuits by construction, and the two families give
*opposite* answers. Splitting the same 24 instances:

| family (12 each) | `gpap:b8` | `pipe:default` | `cmsgen` | GPAP ok | CMSGen ok |
|---|---|---|---|---|---|
| **circuit** (support = primary inputs) | **0.039** | **0.040** | 0.087 | **9/12** | 4/12 |
| **cnf** (support = every variable) | 0.446 | 0.441 | **0.272** | 0.12 -> 0/12 | 0/12 |

On circuits the GPAP pipeline sits at **0.040, which IS the exact-uniform control's floor
(0.0400)** — statistically indistinguishable from uniform — and beats CMSGen (0.087, flagged on
8/12). On random k-CNF it is clearly worse than CMSGen (0.441 vs 0.272), though both are heavily
skewed there and neither passes a single instance. The 0.298-vs-0.177 aggregate is an artifact of
mixing them 50/50; it describes my generator, not either tool.

**The mechanism is the same one behind §3**: skew comes from propagation *among support
variables*. On a Tseitin circuit the support is the primary inputs — propagation runs forward into
the gates and rarely forces one input from another — so the support decisions are near-independent
and the result is near-uniform even at B=8. With support = every variable, a random k-CNF
propagates constantly *within* the support, and every such implication is attraction-basin
weighting. (Consistent with both observations; not isolated by a dedicated experiment.)

That matters for how the result should be read: **GPAP's flagship regime is projected circuit
sampling, and that is exactly the regime where it is uniform and CMSGen is not.** The regime where
it loses — sampling all variables of a dense random CNF — is not what the tool is for.

Two further readings:

- ~~The full pipeline tracks the raw config it wraps.~~ **WRONG — corrected in §3c.** It held on
  these 24 instances only because both arms happened to sit at the floor; at 600 instances the
  pipeline is 2.8x worse than the `b8` config it contains, and the cause is the bandit.
- **At B >= |support| GPAP is exactly uniform and beats CMSGen on 23/24, in both families.** That
  is the only configuration here with a uniformity *argument* rather than a measurement — bounded
  by the yield collapse in §3.

So: **GPAP spans a range CMSGen does not, it wins the regime it is built for, and only the top of
that range is a claim worth making.** Which leads to:

## 3. The finding: B is an exactness knob, and it bottoms out at exactly uniform

`B` is the number of variables decided per propagation round. Sweeping it over 12 instances,
3 seeds, `n = 100|S|` — TV median per config, and per-instance TV against the support size:

| | b1 | b2 | b4 | b8 | b16 | b32 | b64 | b256 | uniform control |
|---|---|---|---|---|---|---|---|---|---|
| TV median | 0.210 | 0.226 | 0.222 | 0.039 | **0.039** | **0.039** | **0.039** | **0.039** | 0.039 |
| skew / 12 | 9 | 9 | 9 | 4 | **0** | **0** | **0** | **0** | 0 |

| instance | \|sup\| | b1 | b2 | b4 | b8 | b16 |
|---|---|---|---|---|---|---|
| cnf-000 | 12 | 0.387 | 0.375 | 0.289 | 0.275 | **0.044** |
| cnf-002 | 16 | 0.390 | 0.434 | 0.457 | 0.427 | **0.042** |
| cnf-004 | 8 | 0.219 | 0.220 | 0.226 | **0.040** | 0.040 |
| cnf-006 | 8 | 0.470 | 0.470 | 0.257 | **0.029** | 0.029 |
| cnf-010 | 14 | 0.419 | 0.453 | 0.534 | 0.318 | **0.040** |
| circuit-001 | 7 | 0.200 | 0.168 | 0.219 | **0.038** | 0.038 |

The transition is at **B >= |support|**, per instance, sharply. At and above it the TV equals the
exactly-uniform control's floor (0.0386) to three decimals — i.e. indistinguishable from uniform at
this instrument's resolution.

**Why.** With `B >= |support|` one decision round assigns the entire support a uniform random
polarity, then propagates. GPAP has no search: it **DISCARDS** conflicted trajectories rather than
backtracking out of them. Keeping the survivors of a uniformly random support assignment is
therefore exact rejection sampling over the projection — uniform by construction. Small B
interleaves propagation between decisions, which weights each solution by its attraction basin;
that basin weighting is the skew, and it is what CMSGen has too.

This is a structural advantage over CMSGen rather than a tuning win: CMSGen is a CDCL solver and
*must* backtrack, so it cannot reach the rejection-sampling limit at all. It also explains the
`b1` < `b8` ordering that first looked backwards — more propagation per decision means more basin
weighting, not less.

**Scope of the guarantee.** Exactness needs propagation to close the formula from the support, so
that a support assignment either yields a model or conflicts. That holds for Tseitin-encoded
circuits (the flagship regime) and trivially when the support is every variable. On a general CNF
whose declared independent support does not BCP-close, completion needs further decisions and
residual skew survives — B still improves it, but "exact" is not claimed there.

**What it costs.** Less than expected, because the knob scales with `|support|`, not `N`. On real
`sk_` instances (|support| = 32, N = 1509 / 2586), 20k samples:

| | B=1 | B=8 | B=32 | B=128 | B=2048 |
|---|---|---|---|---|---|
| 27.sk_3_32 | 82.7k/s | 229.7k/s | 144.9k/s | 165.9k/s | 163.4k/s |
| 53.sk_4_32 | 176.3k/s | 208.7k/s | 196.5k/s | 196.8k/s | 200.5k/s |

So `B >= |support|` costs ~1.4x on one and nothing on the other. That is a very cheap exactness
option on instances like these — but the mechanism IS rejection sampling, so the acceptance rate
is `p_accept` and it collapses when `p_accept` is tiny. That is not hypothetical:

**Measured failure — `B >= |support|` cannot sample Barbarik's kernel formulas at all.** Running
the certifier against the `bsup` config, GPAP produced **0/50** samples on one instance and
**11/50** on another, so the test cannot even start (Barbarik requires exactly the number of
samples it asks for). The kernel CNF's sampling set is much larger than the original instance's
(it carries the chain-formula variables), and one uniform random round over all of them
essentially never lands in the solution set. The same two instances get **Accept:1 (1,872
samples)** at `B=8`.

Three consequences, and they bound the finding:

1. **The exactness claim rests on the exact-oracle layer alone.** It cannot be cross-validated by
   Barbarik, because the sampler-under-test cannot run on the formulas Barbarik constructs. The
   evidence is §3's 24/24 at the control's TV floor plus the rejection-sampling argument — strong,
   but it is one instrument, not two.
2. **A `--uniform` mode pinning `B >= |support|` would be conditional, not general.** It is viable
   only where `p_accept` is high enough to keep the yield usable, which is an instance property
   the tool would have to measure and fall back from.
3. It sharpens what "exact" means here: exactly uniform **over the samples it emits**, at a
   throughput that degrades with `p_accept` exactly like any rejection sampler. On a hard unroll
   that degradation is the whole reason GPAP interleaves propagation in the first place.

## 3b. What happens when |support| >> B — the regime real benchmarks live in

The §2/§3 suites have |support| 12-16 against B=8, i.e. only ~2 decision rounds. A real unroll has
|support| = |PI| x T in the hundreds, so B=8 means 25+ rounds and 25+ chances for attraction-basin
weighting. That gap has to be probed, and it cannot be probed directly: |S| grows exponentially
with T while |support| grows only linearly, so a large-support real instance is exactly the one
enumeration cannot reach (measured: b01 at T=6 gives |support|=12, |S|=480, enumerable in 0.0s;
at T=10 it is |support|=20 and already past a 4,000-solution cap).

Two things can be done instead, and both were.

**(a) Push the synthetic support up.** With the pin count scaled to the input count, circuits at
|support| 18-20 with |S| 64-768 are generatable and enumerable. 12 instances, 3 seeds:

| arm | ok | TV median | TV max |
|---|---|---|---|
| `gpap:b1` / `b2` / `b4` / `b8` | 6-7/12 | **0.0406** | 0.35 |
| `gpap:b16` / `b32` | **12/12** | **0.0397** | 0.043 |
| `pipe:default` (full pipeline) | 7/12 | 0.0412 | 0.41 |
| `cmsgen` | 3/12 | **0.0766** | 0.26 |

**The circuit story gets BETTER at larger support, not worse.** GPAP's median sits at the uniform
floor (0.0397-0.0412 against the control's 0.0399) at every B, CMSGen is ~2x worse (0.0766) and
passes only 3/12, and GPAP-only skew is 0 at every single B — there is no instance where GPAP is
skewed and CMSGen is not.

**(b) Reach real round counts by lowering B instead.** The mechanism depends on the number of
interleaved propagation rounds, `|support|/B`, so B=1 at |support|=13 gives 13 rounds — the same
order as a real unroll's ~25. On the §2 circuit family:

| config | rounds (\|sup\|/B) | TV median | ok |
|---|---|---|---|
| `b8` | 1.6 | 0.0394 | 9/12 |
| **`b1`** | **13.0** | **0.0394** | **9/12** |
| `cmsgen` | — | 0.0869 | 4/12 |

Going from 1.6 to 13 rounds leaves both the median and the pass count **unchanged**. Round count
is not, by itself, what drives circuit skew.

### A prediction that failed, and what it corrects

§3 reported the B sweep's transition as sharp at `B >= |support|`. That suite was dominated by
`cnf` instances. Here the sharp-boundary reading is **falsified**: `B=16` is strictly *below*
|support| on all 12 instances (18, 19 or 20) and all 12 are still at the floor. So:

- `B >= |support|` is **sufficient** (it is provably exact rejection sampling) — that claim stands.
- It is **not the threshold**. On circuits, B a couple below |support| is already indistinguishable
  from uniform, because the residual rounds decide only 2-4 variables and there is little
  within-support propagation to weight them.
- Skew here is **instance-specific rather than monotone in B**: circuit-004 skews at B=1/2/4
  (0.167) but is clean at B=8; circuit-008 skews at every B <= 8 (0.20-0.35) and is clean at 16.
  Five of twelve instances carry essentially all of it.

The honest summary of the whole B axis: large B removes skew and at `B >= |support|` provably so,
small B is fine on circuits and bad on dense CNF, and *which* circuit instances misbehave at small
B is not predicted by |support|/B.

## 3c. The 600-instance circuit campaign, and what it found in the pipeline

A 34-minute circuit-only campaign (5 stages, |support| 12-22, fresh seed per stage, 600 instances
x 3 seeds x 6 arms = **10,800 sampler runs**, |S| 64-986, p_accept 1.5e-5 to 0.23):

| arm | ok | TV median | TV p90 | GPAP-only skew | TV better than CMSGen |
|---|---|---|---|---|---|
| `gpap:b8` (raw config) | **341/600 (57%)** | **0.0411** | 0.172 | **0/600** | 434/600 (72%) |
| `pipe:default` (shipped) | 271/600 (45%) | 0.1133 | 0.290 | 3/600 | 264/600 (44%) |
| `pipe:circuit` (shipped) | 273/600 (46%) | 0.1117 | 0.281 | **0/600** | 266/600 (44%) |
| `cmsgen` | 141/600 (24%) | 0.1015 | 0.211 | — | — |
| `ctl:uniform` | **600/600** | 0.0400 | — | — | negative control PASS |
| `ctl:tv0.05` | 0/600 (all detected) | 0.0581 | — | — | power 600/600 |

**Zero unsound rows in 10,800 runs.** The controls hold at scale: 0/600 false positives and
600/600 detections of a real TV=0.05, so the sensitivity floor is confirmed on this suite.

GPAP passes 2x more often than CMSGen (45-57% vs 24%) at every support size, and `gpap:b8`'s skew
set is a strict subset of CMSGen's (**GPAP-only skew 0/600**, CMSGen-only 200/600).

### The bandit is the pipeline's uniformity cost — measured, not inferred

The shipped pipeline is **2.8x worse in TV median than the `b8` config it contains** (0.113 vs
0.041). Isolating the cause on 30 instances, by running the full pipeline with the portfolio
restricted to a single config:

| arm | TV median | TV > 0.1 |
|---|---|---|
| raw `b8` | 0.0422 | 11/30 |
| **full pipeline, portfolio = [b8] only** | **0.0420** | 7/30 |
| **full pipeline, default 3-config bandit** | **0.1666** | 19/30 |

**Simplify + mining + the projection path cost nothing** (0.0420 vs 0.0422 — they are
model-preserving and it shows). The entire gap is the adaptive bandit, which **chose B=1 on 30/30
instances** because B=1 wins on throughput, and B=1 is measurably more skewed than B=8. This is the
`soft` failure mode one level milder: *the bandit optimises models/second with no uniformity term.*

**Unlike `soft`, this is a real trade-off and should NOT be fixed by deleting the config.** B=1 is
Barbarik-Accepted (5/5, §4) — it is more skewed, not certifiably non-uniform — and it is worth a
large amount of throughput on exactly the flagship instances (interleaved A/B, models/s, B=8
relative to B=1):

| instance | \|sup\| | B=1 | B=8 | B=8 / B=1 |
|---|---|---|---|---|
| b03-c25 | 100 | 21,868/s | 3,211/s | **0.15x** |
| b06-c25 | 50 | 94,304/s | 17,002/s | **0.18x** |
| b10-c25 | 275 | 2,746/s | 3,640/s | 1.33x |
| 27.sk_3_32 | 32 | 152,291/s | 156,115/s | 1.03x |
| 53.sk_4_32 | 32 | 130,760/s | 153,917/s | 1.18x |

Forcing B=8 would cost **5-7x throughput on deep circuit unrolls** while buying ~4x better TV. That
is a user's call, not a default to flip silently. Recommended (not implemented — needs a decision):

1. **Surface it.** The chosen config is already in `tuned_b` / `config_breakdown`; say in the CLI
   output when the winner is B=1 that the distribution is more skewed than B=8.
2. **A `--uniform` mode** restricting the portfolio to B>=8 (and to B>=|support| where the support
   is small enough for §3's exactness), accepting the measured throughput cost.
3. Do **not** remove B=1 from the portfolio — the throughput loss is large and it is
   Barbarik-Accepted.

## 4. Barbarik: the standard certifier agrees, and rejects `soft`

`eval/barbarik_gpap.py` now defaults to the **CMSGen paper's own parameters** (eps 0.3, eta 1.8,
delta 0.1 — "Designing Samplers is Easy", FMCAD'21, Table I) and can put CMSGen itself under test,
not just GPAP. Those parameters cost ~1,872 samples per instance instead of the ~70,000 that
eps 0.1 / eta 0.9 demands — a 38x cheaper certificate, minutes not hours. Measured budget matches
the paper's stated 1.836x10^3 exactly.

5 instances, CM_test with the SPUR reference:

| sampler under test | verdict | samples to verdict |
|---|---|---|
| `gpap:b8` | **Accept 5/5** | 1,772-1,872 |
| `gpap:b1` | **Accept 5/5** | 1,822-1,872 |
| `cmsgen` | **Accept 5/5** | 1,822-1,872 |
| `gpap:soft` | **Reject 5/5** | 156-936 (8-50% of the accept budget) |
| `gpap:bsup` (B >= \|sup\|) | **cannot be tested** | 0/50 and 11/50 samples produced (§3) |

CMSGen Accepting reproduces the paper's own 50/50 result on our instances, which validates the
harness. GPAP at `b8`/`b1` lands in the same certified class — the defensible claim about GPAP's
uniformity, stated at the standard the field uses.

`soft` being rejected 5/5, fast, is the actionable half. Note the two layers do not align as neatly
as they first appeared: the rejection was predicted from circuit-017, where `soft`'s TV of 0.907
means L1 = 1.81, just past eta = 1.8 — but `soft` was also rejected on cnf-028, where its
original-instance TV was only 0.136. Barbarik samples the *kernel* formulas it constructs, and
`soft`'s conflict-risk polarity bias is more extreme there than on the original instance. So the
honest statement is that original-instance TV **understates** how far `soft` is from uniform, and
Barbarik is the more sensitive instrument for this particular defect.

### Consequence for the portfolio

`soft` is a member of `CIRCUIT_PORTFOLIO`, and the adaptive bandit selects among configs on
**models/second alone — there is no uniformity term**. So on any instance where `soft` wins the
bandit (the tight-cascade family; b03-c25 went 10% -> 88% success with it), the run silently emits
samples from a distribution the standard tester rejects. Nothing is unsound — every row is a real
solution — but the *distribution* is not what a user asking for a sampler expects.

This is not an argument to delete `soft`: the "don't globally discard a lever that is net-negative
in one regime" rule stands, and for a user who wants many solutions fast and does not care about
the distribution it is the right config. It is an argument that the choice must be **visible and
user-controlled**, not made by a throughput bandit on the user's behalf.

**Done 2026-07-30.** `soft` was removed from `CIRCUIT_PORTFOLIO` and is now opt-in via
`--allow-soft` on `gpap sample` / `gpap bench` (which prints a warning naming the trade), or
`pipeline.SOFT_CONFIG` / `portfolio(name, allow_soft=True)` from Python. Both default portfolios
are now uniformity-safe, gated by `tests/test_sampler.py::test_soft_is_not_in_any_default_portfolio`.

**The throughput cost is LARGE on the circuit flagship — much larger than a single `sk_` instance
suggested.** Profiled on all 19 `benchmarks/circuits/bench/seq` circuits, `abc:fast`, T=25,
`--target rare`, n_time=50k, 30 s budget, both portfolios back-to-back in one process with the arm
order alternated across 2 reps (this box's clocks drift, so a two-sweep comparison would not be
evidence):

| | ratio new/base | note |
|---|---|---|
| **geomean, 13 comparable non-vacuous** | **0.410** | removing soft is **2.44x slower** |
| median | 0.788 | 1.27x slower — most instances barely move |
| geomean excluding b12 | 0.627 | the loss is concentrated, not spread |

`soft` won the bandit on **7 of 15** non-vacuous instances (b03, b06, b11, b12, b13, s1196, s832)
and 5 were completely unaffected (b01, b04, b08, b10, s386; the 4 vacuous ones all ~1.00, as they
must be). Two are not throughput regressions but **capability losses**:

| instance | with soft | without soft |
|---|---|---|
| **b11** | 11,269 models @ 386/s (succ 0.16%) | **0 models** (succ 0.000%) |
| **b12** | 50,434 models @ 10,997/s (succ 5.77%) | 789 models @ 27/s (succ 0.007%) |

(b15 produced 0 in both arms — a pre-existing limit at this budget, not a soft regression.)

So the honest trade is: the default is now Barbarik-accepted everywhere, at the price of 2.44x
geomean throughput on the circuit flagship and two instances that effectively stop producing. The
earlier "~1.3x" figure in this document came from 27.sk_3_32 alone — an `sk_` projected instance,
not the circuit regime `soft` exists for — and understated it badly. Method lesson worth keeping:
**never size a circuit-regime trade-off from an `sk_` instance.**

> ### ⚠️ OPEN DECISION — parked, not settled
> User, 2026-07-30, on seeing the numbers above: *"that's a big loss, please document it and keep
> it for now, we will need to think about it later."* So the current state stands — `soft` out of
> the defaults, `--allow-soft` restoring the old behaviour exactly — but the price is **not**
> accepted as final. Re-measure with `eval/profile_soft_portfolio.py` before revisiting; it
> encodes once and runs both portfolios in one process with the arm order alternated, which is the
> only method this box's clock drift permits.
>
> A middle path that the *shape* of the loss suggests, sketched but deliberately **not**
> implemented: the damage is concentrated (median 0.788, 5 instances untouched) and the worst two
> are instances where every uniformity-safe config yields ~nothing. A bandit that falls back to
> `soft` only when the safe configs produce essentially zero would recover b11/b12 without
> silently reintroducing a Barbarik-rejected distribution anywhere a safe config already works.
> Cost: the fallback would need to be reported, or it reintroduces the original silent-selection
> problem in a rarer form.

Still open, and deliberately not done here:

1. A `--uniform` mode that also pins `B >= |support|` where the support size allows, for the
   exactness property of §3 rather than merely "Barbarik does not reject".
2. Folding the uniformity readout into the bandit's objective (expensive — needs an enumerable
   space, so it cannot be the general answer).

### The portfolio is now self-describing

Configs were bare `(True, 8, "soft")` triples, which said nothing to a reader. They are now a
`SamplingConfig` NamedTuple over two enums (`VarOrder.RANDOM|STRUCTURED`,
`Lookahead.NONE|SOFT`), each member carrying the measurement that justifies it — including the
warning that `SOFT` is Barbarik-rejected. `SamplingConfig` still IS the
`(random_var, B, lookahead)` tuple every consumer reads (`VarOrder` is an int-enum so
`if random_var:` works, `Lookahead.NONE` is the empty string so it stays falsy and only `SOFT`
matches the sampler's `in ("soft", "fsoft")` test), so the refactor changes no behaviour —
pinned by `test_portfolio_configs_are_tuple_compatible`. One real trap it did surface: the
results-JSON `config_breakdown` keys are built by formatting the config, and the enum repr leaked
in as `rv=VarOrder.RANDOM,B=8`, which would have broken comparison against every row recorded
before the enums existed. Fixed at the source (`bool()`/`str()` in the label) and gated.

## 5. What is NOT shown

- **Scale.** Every exact result here is on |S| <= 400 and n <= 40k. Skew on tiny random CNFs need
  not extrapolate to production unrolls; the exact layer *cannot* run there, which is exactly why
  Barbarik exists. The B-sweep prediction at scale is untested.
- **The b06-c4 class.** Our earlier finding (TV 0.17 for GPAP and 0.139 for CMSGen on the same CNF,
  replicated over seeds) is consistent with the CMSGen numbers here and unchanged by this work.
- **Barbarik3.** The checkout carries `PM_test` (NeurIPS'22, Pote & Meel — query complexity linear
  in n rather than exponential), but it targets arbitrary WEIGHTED distributions and needs a DUAL
  oracle (sampling + probability queries) on the target. For uniform sampling `CM_test` is the
  correct and cheaper instrument; PM_test with uniform weights is an untried cross-check.
- **Whether `soft` can be fixed.** Its bias is deliberate (epsilon-greedy risk polarity). Whether a
  debiasing weight could keep the success-rate win is unexplored.

## 6. Reproducing

```bash
# the headline campaign: |support| > B, raw configs AND the full shipped pipeline, vs CMSGen
PYTHONPATH=. .venv/bin/python eval/fuzz_uniformity.py --count 24 --seeds 3 --n-mult 100 \
    --sup-min 12 --configs b1 b8 b16 b32 --pipelines default circuit circuit+soft \
    --cmsgen ~/.cache/gpap/bin/cmsgen --controls 0.0 0.05 \
    --keep-dir .experiments/fuzz/instances2 --out .experiments/fuzz/round2.json
# the B sweep (the exactness knob)
PYTHONPATH=. .venv/bin/python eval/fuzz_uniformity.py --count 12 --seeds 3 --n-mult 100 \
    --configs b1 b2 b4 b8 b16 b32 b64 b256 --controls 0.0
# the standard certificate, per sampler under test
PYTHONPATH=. .venv/bin/python eval/barbarik_gpap.py .experiments/fuzz/instances/*.cnf \
    --sampler gpap --config b8      # or: --config soft | --sampler cmsgen
```

`tests/test_uniformity_fuzz.py` gates the invariants: oracle self-consistency, the negative control
(the statistics must not manufacture skew), the positive control (they must have power), soundness
at every B, and `B >= |support|` being indistinguishable from uniform on every generated instance.
It deliberately does NOT gate small-B uniformity — that is a documented property of the algorithm
class, not a defect. Whole file runs in <1 s.
