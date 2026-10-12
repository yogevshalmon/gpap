# How uniform is GPAP — against an exact oracle, against CMSGen, and under Barbarik

> **Status:** settled. §4b supersedes §4's verdict counts; §4c supersedes §2's TV reading for
> the shipped pipeline.
> **Measured:** exact-oracle campaigns 2026-07-29/30; Barbarik re-derivation (§4b) and the
> 120/80-instance TV campaign (§4c) 2026-08-28 — all on the RTX 4080 Laptop (WSL2, libgpap core).
> A distributional result does not depend on the card; the wall-clock prices in §4c do.
> **Rule lives in:** `gpap/sampling/pipeline.py` (the portfolios, `--allow-soft`), the warnings
> on `sound_check` in `cpp/include/gpap/gpap.h` and `.hpp`, and `eval/uniformity/` — which makes
> the claims the library does not: `metrics.py` (statistics), `fuzz_uniformity.py` (exact
> oracle), `barbarik_gpap.py` + `barbarik_screen.py` + `barbarik_table.py` (the certificate).
> **Gated by:** `tests/experiments/test_uniformity_fuzz.py`, `tests/experiments/test_barbarik_harness.py`,
> `tests/sampling/test_sampler.py`.
> **SECTION NUMBERS ARE AN API** — cited from `gpap/`, `cpp/`, `eval/`, the tests and the
> generated paper prose. Append; never renumber.

GPAP's sampler is the GPU-parallel form of CMSGen's Algorithm 1 and neither has a uniformity
guarantee. The settled results:

1. `B >= |support|` is exactly uniform (rejection sampling, §3), at every scale measured (§4c),
   bounded by a yield collapse when `p_accept` is small.
2. `soft` is Barbarik-Rejected on 64 of the 65 circuit instances it can sample (2026-08-28; the
   paper campaign runs no `soft` arm) and has been opt-in (`--allow-soft`) since 2026-07-30
   (§4, §4b).
3. The shipped `circuit` portfolio is Barbarik-Accepted on 60 of the paper campaign's 65
   eligible instances, zero Rejects, 5 no-verdict; pinned `b8` 63/65; CMSGen 65/65 (§4b). At
   barbarik's own stricter default only pinned configs were run: `b8` and `b1` 6/6, CMSGen 6/6.
   ⚠️ The 64/66 "as shipped" in §4b is the EARLIER campaign, and no arm there ran the portfolio:
   it is the pinned-config verdicts joined to the config the bandit delivers. The paper campaign
   is the first to put the portfolio itself under test (`pipe:circuit`).
4. Its TV cost is the bandit picking B=1 for throughput: the default reads 1.31-1.35× worse than
   CMSGen in TV median while never skewing where CMSGen does not; `--portfolio b8` holds the
   exact-uniform floor for ~10% throughput (§3c, §4c).
5. The regime split: on projected circuits GPAP's configs sit at or near the floor and CMSGen does
   not; on random k-CNF over all variables GPAP is clearly worse. Never quote an aggregate (§2).

## 1. The instrument: generated instances with an enumerable solution set

`eval/uniformity/fuzz_uniformity.py` generates small formulas, enumerates the solution set
exhaustively, and runs chi-square plus total variation (TV) against uniform. Two families, both
with an unambiguous "uniform over support projections":

| family | support | oracle |
|---|---|---|
| `cnf` | every variable | all 2^n assignments |
| `circuit` | primary inputs of a random Tseitin gate DAG, outputs pinned | all 2^k input vectors |

The `circuit` support is independent by construction (each projection has one extension), which
is what makes projected GPAP comparable to full-model CMSGen. The oracle is solver-free, and
`gen_circuit` proves its own encoding over the whole truth table before an instance is used.

Controls run in every campaign. Negative control (exactly uniform draws): flagged 0/30. Positive
controls (a planted TV): at `n = 100|S|` × 3 seeds, TV 0.05 and 0.10 detected 30/30; at
`n = 50|S|` × 2 seeds TV 0.05 was detected 2/6. The sensitivity floor is therefore TV ≈ 0.05:
"ok" means "no skew detectable at TV 0.05", never "uniform". Instrument audit (2026-07-30):
TV/chi2/dof exact to 1e-12 against first principles; the floor a perfect sampler must read,
`E[TV] = 0.399·sqrt(S/n) = 0.0399` at `n = 100|S|`, is what the control reads (0.0392-0.0400);
all 24 solution sets set-equal to a pysat blocking-clause enumeration; Fisher-combined p
Uniform(0,1) under H0; 0/120 false positives at alpha = 1e-3; power 97% at TV 0.05 and 100% at
0.10.

## 2. GPAP vs CMSGen, paired on identical files

Instances must have `|support| > B`, otherwise the campaign measures §3's trivial case. Headline
campaign: 24 instances, supports 12-16, |S| 28-382, 3 seeds, Fisher-combined chi-square flagged
at p < 1e-3, every arm on the identical file, zero unsound rows over 10 arms × 24 × 3. The
`pipe:*` arms are `sample_file` exactly as the CLI runs it.

| sampler | ok | TV median | better TV than CMSGen |
|---|---|---|---|
| `gpap:b16`, `gpap:b32` (B ≥ \|sup\|) | 24/24 | 0.0392 | 23/24 |
| `gpap:b8` | 9/24 | 0.220 | 11/24 |
| `pipe:default` / `pipe:circuit` (shipped pipeline) | 9/24 | 0.298 / 0.325 | 7/24 |
| `pipe:circuit+soft` (`--allow-soft`) | 7/24 | 0.335 | 8/24 |
| `cmsgen` | 4/24 | 0.177 | — |
| `ctl:uniform` | 24/24 | 0.0400 | — |

Do not quote the aggregate: the suite is half random k-CNF and half circuits, and the two
families answer oppositely.

| family (12 each) | `gpap:b8` | `pipe:default` | `cmsgen` | GPAP ok | CMSGen ok |
|---|---|---|---|---|---|
| circuit (support = primary inputs) | 0.039 | 0.040 | 0.087 | 9/12 | 4/12 |
| cnf (support = every variable) | 0.446 | 0.441 | 0.272 | 0/12 | 0/12 |

Mechanism: skew comes from propagation among support variables. On a Tseitin circuit propagation
runs forward into the gates and rarely forces one input from another, so support decisions are
near-independent; with support = every variable a random k-CNF propagates constantly within the
support, and every such implication is attraction-basin weighting. Regime-limited: these 12
circuits have |S| 8-384 and below |S| ≈ 380 every sampler sits at the floor; §4c re-measures at
|S| ≥ 128, where the shipped default reads 0.165 against CMSGen's 0.122. The floor result holds
for `B >= |support|` at every size and for `--portfolio b8`, not for the default.

## 3. B is an exactness knob, and it bottoms out at exactly uniform

`B` is the number of variables decided per propagation round. Sweep over 12 instances, 3 seeds,
`n = 100|S|`:

| | b1 | b2 | b4 | b8 | b16 | b32 | b64 | b256 | uniform control |
|---|---|---|---|---|---|---|---|---|---|
| TV median | 0.210 | 0.226 | 0.222 | 0.039 | 0.039 | 0.039 | 0.039 | 0.039 | 0.039 |
| skew / 12 | 9 | 9 | 9 | 4 | 0 | 0 | 0 | 0 | 0 |

At `B >= |support|` the TV equals the exact-uniform control's floor to three decimals, per
instance.

Why: with `B >= |support|` one round assigns the whole support a uniform random polarity, then
propagates. GPAP has no search and discards conflicted trajectories instead of backtracking, so
keeping the survivors is exact rejection sampling over the projection. Small B interleaves
propagation between decisions and weights each solution by its attraction basin, which is the
skew CMSGen has too; CMSGen is a CDCL solver and must backtrack, so it cannot reach this limit.
Scope: exactness needs propagation to close the formula from the support (true for Tseitin
circuits, and trivially when the support is every variable); on a CNF whose declared support does
not BCP-close, residual skew survives.

Cost: the knob scales with `|support|`, not N. On 27.sk_3_32 / 53.sk_4_32 (|support| = 32, 20k
samples) B=32 runs at 144.9k/s / 196.5k/s against B=8's 229.7k/s / 208.7k/s — ~1.4× on one,
nothing on the other. But the mechanism is rejection sampling and yield collapses with
`p_accept`: on Barbarik's kernel formulas, whose sampling set carries the chain-formula
variables, `bsup` produced 0/50 and 11/50 samples where `b8` gets Accept at 1,872. So the
exactness claim rests on the exact-oracle layer alone (Barbarik cannot cross-validate it), a
`--uniform` mode pinning `B >= |support|` would be conditional on `p_accept`, and "exact" means
exactly uniform over the samples emitted, at a rejection sampler's throughput.

### 3b. `|support| >> B`, the regime real benchmarks live in

A real unroll has `|support| = |PI| × T` in the hundreds, i.e. 25+ rounds at B=8, and enumeration
cannot reach it (|S| grows exponentially in T). Two proxies. (a) Circuits at |support| 18-20,
|S| 64-768, 12 instances × 3 seeds: `b1`-`b8` 6-7/12 ok at median 0.0406, `b16`/`b32` 12/12 at
0.0397, `pipe:default` 7/12 at 0.0412, `cmsgen` 3/12 at 0.0766; GPAP-only skew 0 at every B.
(b) Real round counts via small B: `b1` at |support| 13 gives 13 rounds and leaves the §2
circuit median (0.0394) and pass count (9/12) unchanged, so round count alone does not drive
circuit skew. Two corrections to §3: `B >= |support|` is sufficient but not the threshold (`B=16`
is below |support| on all 12 of (a) and all sit at the floor), and small-B skew is
instance-specific rather than monotone in B (5 of 12 carry essentially all of it).

### 3c. The 600-instance circuit campaign: the bandit is the pipeline's uniformity cost

600 circuit instances × 3 seeds × 6 arms (|support| 12-22, |S| 64-986, p_accept 1.5e-5 to 0.23),
10,800 runs, zero unsound rows, controls 0/600 false positives and 600/600 detections at TV 0.05:

| arm | ok | TV median | GPAP-only skew | better TV than CMSGen |
|---|---|---|---|---|
| `gpap:b8` | 341/600 | 0.0411 | 0/600 | 434/600 |
| `pipe:default` / `pipe:circuit` | 271 / 273 | 0.1133 / 0.1117 | 3 / 0 | 264 / 266 |
| `cmsgen` | 141/600 | 0.1015 | — | — |

The shipped pipeline is 2.8× worse than the `b8` config it contains. Isolated on 30 instances with
a one-config portfolio: raw `b8` 0.0422, pipeline with `[b8]` only 0.0420, default 3-config bandit
0.1666. Simplify, mining and the projection path cost nothing here (both arms at the floor; §4c
shows the pipeline is not neutral at |S| ≥ 128); the gap is the bandit, which chose B=1 on 30/30
for throughput. Unlike `soft`, this is a trade-off and must not be fixed by deleting the config:
B=1 is Barbarik-Accepted (§4b), more skewed rather than certifiably non-uniform, and worth real
throughput on deep unrolls. The price of pinning B=8 is measured in §4c. Not implemented:
surfacing "winner is B=1, more skewed than B=8" in the CLI, and a `--uniform` mode restricting the
portfolio to B ≥ 8.

## 4. Barbarik, and the `soft` decision

`eval/uniformity/barbarik_gpap.py` defaults to the CMSGen paper's parameters (eps 0.3, eta 1.8,
delta 0.1; FMCAD'21 Table I) and can put CMSGen itself under test. The first 5-instance run
(2026-07-30) is superseded by §4b (no controls, a config missing from the harness, a non-shipped
`k_cap`). Its one surviving verdict: `soft` is Rejected 5/5, fast, and original-instance TV
understates how far it is from uniform (Rejected on cnf-028 at TV 0.136, because the kernel
formulas amplify its conflict-risk polarity bias).

The bandit selects on models/second with no uniformity term, so wherever `soft` won it the run
silently emitted a distribution the standard tester rejects. Done 2026-07-30: `soft` removed from
`CIRCUIT_PORTFOLIO`, opt-in via `--allow-soft` on `gpap sample`/`gpap bench` (with a warning) or
`pipeline.SOFT_CONFIG` / `portfolio(name, allow_soft=True)`; gated by
`tests/sampling/test_sampler.py::test_soft_is_not_in_any_default_portfolio`. Configs are `SamplingConfig`
NamedTuples over `VarOrder`/`Lookahead` enums that carry their justification, tuple-compatible
(`test_portfolio_configs_are_tuple_compatible`).

The throughput price is large on the circuit flagship. Profiled on all 19
`benchmarks/circuits/bench/seq` circuits, `abc:fast`, T=25, `--target rare`, both portfolios in
one process with arm order alternated (this box's clocks drift, so two sweeps are not evidence):
geomean 0.410 over 13 comparable non-vacuous (2.44× slower), median 0.788, 0.627 excluding b12.
`soft` won the bandit on 7 of 15 non-vacuous (b03, b06, b11, b12, b13, s1196, s832); two are
capability losses: b11 11,269 models @ 386/s → 0, b12 50,434 @ 10,997/s → 789 @ 27/s. Do not size
a circuit-regime trade-off from an `sk_` instance: the earlier "~1.3x" came from 27.sk_3_32 alone.

Open decision, parked (user, 2026-07-30: "that's a big loss, please document it and keep it for
now, we will need to think about it later"). Re-measure with `eval/profiling/profile_soft_portfolio.py`
before revisiting. Sketched, not implemented: a bandit that falls back to `soft` only when every
uniformity-safe config yields ~nothing (the b11/b12 signature), with the fallback reported. Also
open: a uniformity term in the bandit objective (needs an enumerable space, so not general).

## 4b. What Barbarik actually measures, how CMSGen certified itself with it, and how we do

Re-derived 2026-08-28, circuit-only (the flagship regime), with the `sk_`/service set kept as
secondary evidence because that is where the only Rejects live. Supersedes §4's counts.

The test: draw one solution from the sampler under test and one from SPUR (exactly uniform);
build a kernel formula whose solutions are only A-like and B-like in a ratio fixed by
construction; hand it back to the sampler, count the split; a lopsided split is bias; Accept if no
round is lopsided. The sample count depends on (eps, eta, delta) only, not on formula size.

The guarantee is asymmetric (FMCAD'21 §I): "if Barbarik Rejects a sampler, the distribution
generated by sampler is indeed (probabilistically) far from uniform, but if Barbarik Accepts a
sampler, the sampler's distribution is close to uniform under the assumption of non-adversality
with respect to Kernel". Reject is unconditional; Accept is conditional, and only against eta.
An Accept is "not caught", never "uniform".

Two parameter settings, two different tests (`barbarik_gpap.py --explain` prints the derivation):

| | eps | eta | rounds | samples for an Accept | per-round accept band | rejects |
|---|---|---|---|---|---|---|
| CMSGen paper (FMCAD'21 §V-A) | 0.3 | 1.8 | 36 | 1,872 (1,836 from the sampler) | [0.05, 0.95] | TV ≥ 0.90 |
| barbarik's own default | 0.1 | 0.9 | 477 (2 loops) | 70,596 | [0.288, 0.712], [0.3125, 0.6875] | TV ≥ 0.45 |

`36 × 51 = 1,836` from the sampler matches the paper's stated 1.836×10^3, the check that the
parameters are theirs; the paper setting is 38× cheaper. The strict band is about ±0.21 against
±0.45, a genuinely more discriminating instrument. Both are reported, never pooled
(`barbarik_table.py` refuses).

How CMSGen certified itself (FMCAD'21 §V-A, Table I): eps/eta/delta = 0.3/1.8/0.1, SPUR as
reference, 50 benchmarks from the QuickSampler/UniGen3 sets "for which Barbarik terminates within
2 hours"; Accept/Reject QuickSampler 0/50, STS 14/36, UniGen3 50/0, CMSGen 50/0. Two things carry
over: they filtered instances on cost (our 20 s-per-reference-sample budget is the same rule,
stated up front), and CMSGen's Accept was fitted to this test (their Remark 1: restart parameters
were tuned against Barbarik's rejections), so passing is parity with a sampler tuned to pass.

How we do it: same `CM_test`, same SPUR, same parameters, GPAP hooked in-process, plus two
controls the paper lacks and an eligibility screen. Five defects fixed first: `s1` (STRUCTURED,
B=1) is in `DEFAULT_PORTFOLIO` but had no harness arm — the bandit emits only the winner's rows,
so "the pipeline is certified" means every member is (gated by
`test_every_default_portfolio_config_is_barbarik_certifiable`); no negative control
(`--sampler spur` puts the reference under test through the same hook; it must Accept);
`k_cap=4096` was hardcoded — short chunks fire the collector's zero-yield stop early and
`polynomial.sk_7_25` went error → Accept at the shipped cap (K is a batch size, not a
distribution knob); one hang killed the sweep (rows are per instance now, errors recorded, bias
values kept); and not every instance is a valid subject.

Sixth, found 2026-10-12: **the GPAP arms had no budget at all** — they were ended by the
sampler's own zero-yield give-up rule, which matters because what this table reports is a
no-verdict count and a no-verdict is a coverage outcome. The five `pipe:circuit` no-verdicts
below spent 0.99–3.87 s, the two `gpap:b8` ones 9.8 s. Neither upstream tool offers a number to
inherit (barbarik times no sampler; stock CMSGen has no wall budget), so `--order-seconds` is
ours: 30 s an order, 3× the ~10 s the slowest *Accepting* instance averaged (`c2670-c1-random`,
695 s over 72 orders) and 1.5× the reference's 20 s. It is the only bound on an order — no chunk
cap — and every row records it.

**It binds the GPAP arms only, and the asymmetry runs against us.** The symmetric version was
tried first and measured: at 30 s an order CMSGen could not produce one sample from the
`c6288-c1-random` kernel, so a number derived from *our* costs took an Accept off the rival that
it had earned. The rival keeps its own budget (`--cmsgen-order-seconds`, 1,800 s) and SPUR stays
unbounded on barbarik's stock path, so the claim here is not "the same budget" but **GPAP was
held to 30 s an order and CMSGen was not held to anything** — every no-verdict is ours, and no
CMSGen Accept owes anything to a number we chose. `barbarik_table.order_budget` reports each
arm's and refuses to pool one ARM across two.

Re-measured locally 2026-10-12 on those five, same instances (`--encoder abc --abc-opt fast`
reproduces their shapes exactly): with the give-up rule off and 30 s an order **all five
Accept**, none Reject, CMSGen unchanged at 5/5. The counts below predate all of this.

The screen: `spur(control)` is Rejected on `ActivityService`, `ActivityService2`,
`IssueServiceImpl`, `IterationService` and `PhaseService`, also through barbarik's own
unreformatted path (`--sampler spur-raw`), so our marshalling is exonerated. The kernel pins every
declared indVar, so the two buckets hold extensions of sigma1 vs sigma2 as full models; SPUR is
uniform over models, so where the declared sampling set is not an independent support the
reference's own bucket share is off by the extension-count ratio. Counted with `spur -count-only`:

| instance | spread | SPUR verdict |
|---|---:|---|
| `27.sk_3_32`, `53.sk_4_32`, `GuidanceService2` | 1.0× | Accept |
| `ActivityService` | 4.0× | Reject |
| `IssueServiceImpl` | 6.6× | Reject |
| `PhaseService` | 21.8× | Reject |
| `IterationService` | 103.7× | Reject |

A Reject there is a statement about the benchmark. `barbarik_screen.py` re-derives the
exclusion, `barbarik_table.py` counts only control-Accepted instances, `paper_tables.py` imports
that rule. The spread is a lower bound and the control is the gate: a 4-projection re-screen lets
`ActivityService2` and `PhaseService` through. Circuits are ideal subjects: a Tseitin unroll's
sampling set is the primary inputs and every projection has one extension. Screened over 92
generated circuit instances (12 sequential circuits × T ∈ {5,10,25} × {rare, random} + 10
ISCAS-85 × {rare, random}): 67 eligible, and all 25 exclusions were the reference failing the
cost budget, none for non-independent support.

Batch delivery vs the i.i.d. premise: the chunk loop draws one variable permutation per chunk
shared across K rows, so samples are i.i.d. only conditional on that order. Against the exact
oracle (8 circuits × 3 seeds, `n = 100|S|`) batch and `--draw iid` are a wash: TV median 0.0400
vs 0.0399, 6/8 ok each, the same two instances skewed. `--draw iid` is in the harness and Accepts.

### Results — 66 circuit instances, the CMSGen paper's setting

⚠️ SUPERSEDED for the shipped portfolio by the paper campaign (68 generated, `spur` Rejects 3,
**65 eligible**; `pipe:circuit` 60/65 Accept + 5 no-verdict, `gpap:b8` 63/65 + 2, `cmsgen` 65/65,
`spur` 65/65 — `$GPAP_RESULTS/paper/data/barbarik/circ_abc_*.jsonl`, in Table 4b of its
`RESULTS.md`; it runs no `soft` arm). This table measures PINNED configs only: its "as shipped"
64/66 below is their verdicts joined to the config the bandit delivers, not a run of the
portfolio. Kept as the earlier reading, and as the record of the `soft` positive control.

92 generated, screened to 67, control Accepts 66 (`spur` was Rejected on `c6288-c1-rare`; at
delta = 0.1 a uniform sampler may be Rejected up to 10% of the time, and the eligibility rule
drops it either way).

| sampler under test | Accept | Reject | no verdict | mean bias | max bias |
|---|---:|---:|---:|---:|---:|
| `spur(control)` — exactly uniform | 66/66 | 0 | 0 | 0.558 | 0.740 |
| `cmsgen` | 66/66 | 0 | 0 | 0.556 | 0.740 |
| `gpap:b8` | 64/66 | 0 | 2 | 0.556 | 0.760 |
| `gpap:b1` | 64/66 | 0 | 2 | 0.556 | 0.780 |
| `gpap:s1` | 63/66 | 0 | 3 | 0.555 | 0.720 |
| `gpap:soft` (positive control) | 1/66 | 64 | 1 | 0.662 | 0.940 |

Zero rejections for every uniformity-safe config; `soft` is the only arm whose mean bias
separates from the reference's. Joined against the config the bandit delivers
(`--portfolio circuit`: `b1` on 58 instances, `b8` on 8): **Accept 64/66, Reject 0, no verdict
2.** The two are `b13-c25-rare` and `c6288-c1-random`, where GPAP returns 0 samples on the
original formula — a capability limit, not a distribution result.

### Results — the same circuits at barbarik's own default

Six circuits (the cheapest by certificate wall) at eps 0.1 / eta 0.9 / delta 0.05, 70,596 samples
each:

| sampler under test | Accept | Reject | mean bias |
|---|---:|---:|---:|
| `spur(control)` | 6/6 | 0 | 0.533 |
| `cmsgen` | 6/6 | 0 | 0.533 |
| `gpap:b8` | 6/6 | 0 | 0.533 |
| `gpap:b1` | 6/6 | 0 | 0.534 |
| `gpap:soft` (positive control) | 0/6 | 6 | — |

`sample_bits`' give-up rule (stop after one empty chunk past the cold-start grace) cost verdicts
before this was measured: barbarik's kernel is rare-yield by construction, and low yield came
back as "GPAP produced 22/125". With an order-fill retry in the harness (`_FILL_ATTEMPTS`;
independent chunks, so concatenation changes no distribution) `c499-c1-rare` goes error → Accept
in 6.1 s, `c1355-c1-rare` at the strict setting error → Accept in 82 s, and the strict table goes
from 4/6 to 6/6. "Produced n/N" is a budget, not a verdict.

### The secondary set: sk_ / service CNFs, where the only Rejects are

22 screened, 21 eligible, CMSGen-paper setting:

| sampler under test | Accept | Reject | no verdict |
|---|---:|---:|---:|
| `spur(control)` | 21/21 | 0 | 0 |
| `cmsgen` | 21/21 | 0 | 0 |
| `gpap:s1` | 20/21 | 0 | 1 |
| `gpap:b1` | 18/21 | 2 | 1 |
| `gpap:b8` | 16/21 | 1 | 4 |

Delivered-config: `--portfolio default` (the CNF portfolio) 20 Accept / 0 Reject / 1 no verdict;
`--portfolio circuit` 18 / 2 / 1. `gpap sample` and `gpap bench` default to `--portfolio circuit`
for every input, including a plain CNF; `circuit` has no `s1`, so the bandit falls back to `b1`,
which is Rejected on `registerlesSwap` and `UserServiceImpl` where `s1` is Accepted — and it is
not a throughput trade (`s1` delivers 46,512 models to `b1`'s 4,767 on registerlesSwap). Not
fixed: defaulting the portfolio by input type moves a default every recorded `gpap bench` CNF row
was produced under. `registerlesSwap.sk_3_10` (|support| 10, 78 solutions) checks the coarse
verdict against the sensitive instrument: `spur` TV 0.063-0.068, `cmsgen` 0.139-0.159, `gpap:b1`
0.211-0.257, `gpap:s1` 0.204-0.220 — the Reject is genuine and counts against us.

### The bottom line for a circuits paper

⚠️ The 2026-08-28 reading; its first row is the join above. The shipped portfolio's measured
verdict is the paper campaign's `pipe:circuit` 60/65 (the note above the 66-instance table).

| | Accept | Reject | no verdict |
|---|---:|---:|---:|
| GPAP as shipped (`--portfolio circuit`, delivered config) | 64/66 | 0 | 2 |
| CMSGen | 66/66 | 0 | 0 |
| `spur` — exactly uniform (control) | 66/66 | 0 | 0 |
| `soft` (positive control) | 1/66 | 64 | 1 |

plus pinned `b8` and `b1` 6/6 with zero rejections at barbarik's own stricter default, where
CMSGen is also 6/6. This
is a parity claim at the bar the field certifies samplers by, and all `CM_test` can support; the
quantitative claim belongs to §4c.

## 4c. The TV number, measured on 120 circuit instances in the regime that discriminates

An earlier reading, "GPAP's TV is at the floor while CMSGen's is ~1.8× larger", was wrong for
the shipped pipeline: in the 24-instance campaign seven of the twelve circuits were small enough
that every sampler sits at the floor, and only the five with |S| ≥ 176 discriminate (there the
pipeline reads 0.176-0.289 against CMSGen's 0.121-0.305). Re-measured circuit-only at |S| ≥ 128:
120 instances, |S| 128-1920, |support| 14-20, 3 seeds, `n = 100|S|`, both controls (0/120 false
positives, 120/120 detections at TV 0.05), zero unsound rows; the theoretical floor is 0.0399 and
the control reads 0.0399.

| arm | TV median | TV p90 | TV max | unflagged | better TV than CMSGen | skews where CMSGen does not |
|---|---:|---:|---:|---:|---:|---:|
| `ctl:uniform` | 0.0399 | 0.0403 | 0.0408 | 120/120 | 94% | 0/120 |
| `gpap:b16` (B ≥ \|support\|) | 0.0398 | 0.0411 | 0.0439 | 119/120 | 90% | 0/120 |
| `gpap:b8` | 0.0571 | 0.1230 | 0.1598 | 47/120 | 92% | 0/120 |
| `gpap:b1` | 0.0910 | 0.1914 | 0.3136 | 44/120 | 73% | 0/120 |
| `pipe:circuit` (as shipped) | 0.1651 | 0.3010 | 0.4345 | 46/120 | 46% | 0/120 |
| `cmsgen` | 0.1219 | 0.2270 | 0.3501 | 23/120 | — | — |

Supported: `B >= |support|` is exactly uniform at scale (0.0398 median / 0.0439 max, 119/120, out
to |S| = 1920); raw `b8` beats CMSGen (0.0571 vs 0.1219, better on 92%); no GPAP config ever
skews where CMSGen does not; GPAP is unflagged ~2× as often (44-47 vs 23). Killed: the shipped
pipeline does not beat CMSGen on TV — 0.1651 vs 0.1219, 1.35× worse, better on 46%. Split by |S|:
at 128-255 every GPAP arm reads 0.040-0.043 against CMSGen's 0.088; at 512-1920 (65 instances)
`b16` 0.040, `b8` 0.069, `b1` 0.097, `pipe:circuit` 0.168, `cmsgen` 0.137 — the pipeline crosses
CMSGen around |S| ≈ 500. A TV claim must name a config.

Where the default's TV comes from: two effects. With one-entry portfolios on 40 instances, `b8`
reads 0.0569 raw and 0.0405 through the pipeline (it helps), `b1` 0.0967 raw and 0.1605 through
it (it hurts), and the default bandit 0.1661 (= `pipe:b1`). The likely mechanism is ELS shrinking
the effective support, which pushes B=8 toward the exact-rejection regime and does nothing for
B=1; §3c's "neutral" was measured where every arm sat at the floor.

Confirmed paired on 80 fresh circuit instances (independent generation seed):

| arm | TV median | TV p90 | TV max | unflagged | better TV than CMSGen | skews where CMSGen does not |
|---|---:|---:|---:|---:|---:|---:|
| `ctl:uniform` | 0.0400 | 0.0403 | 0.0412 | 80/80 | 91% | 0/80 |
| `gpap:b16` (B ≥ \|support\|) | 0.0398 | 0.0414 | 0.0429 | 80/80 | 90% | 0/80 |
| `pipe:b8` (shipped pipeline, B=8 pinned) | 0.0402 | 0.1599 | 0.2016 | 64/80 | 84% | 0/80 |
| `pipe:circuit` (shipped default) | 0.1471 | 0.2902 | 0.3796 | 36/80 | 51% | 0/80 |
| `cmsgen` | 0.1125 | 0.2131 | 0.2643 | 16/80 | — | — |

`pipe:b8` holds the floor at every |S| (0.040 in all three bins); the default holds it to
|S| ≈ 380 (0.039) and then breaks (0.166 and 0.168 against CMSGen's 0.131 and 0.144).

What may be claimed about the release default (`--portfolio circuit` as shipped), all paired:
Barbarik 60/65 Accept, 0 Reject on the paper campaign (CMSGen 65/65), and pinned `b8`/`b1` 6/6
vs 6/6 at the stricter setting (§4b); it
never skews where CMSGen does not (0/120 and 0/80); it is statistically indistinguishable from
uniform ~2× as often as CMSGen (36/80 vs 16/80; 46/120 vs 23/120); zero unsound rows. Not
claimable: that the default has lower TV than CMSGen — it is 0.1471 vs 0.1125 (1.31× worse),
better on 51%, because GPAP is more often exactly right and more often badly wrong (p90 0.290 vs
0.213) while CMSGen is uniformly mediocre; report both or neither. A TV claim needs
`--portfolio b8`: 0.0402 against a 0.0399 floor, 2.8× better than CMSGen, better on 84%,
unflagged 4× as often (64/80 vs 16/80).

The price of pinning B=8 is ~10%. `eval/profiling/profile_soft_portfolio.py --arm-a circuit --arm-b b8`
(each circuit encoded once, both arms in the same process, arm order alternated, best of reps —
this box's SM clock swings 615-1710 MHz, so nothing else resolves 10%), 19 seq circuits,
`abc:fast`, T=25, `--target rare`, 15 non-vacuous / 14 comparable: geomean 0.910 (1.10× slower),
median 0.964, worst non-vacuous 0.516 (`b13`, 28.9M/s → 14.9M/s), zero capability losses. It is
cheap because the bandit already picks B=8 on 7 of the 15 (b03, b04, b09, b10, b12, s1196, s832).
This supersedes the raw-config Tseitin B=1-vs-B=8 table that once suggested 5-7×; quote this one
for a release decision about `--portfolio`, and re-measure if the encoder or the engine moves.
The trade is ~10% throughput for TV 0.040 instead of 0.147 — a release decision, not taken here
because moving the default portfolio moves every recorded `gpap bench` circuit row.

## 5. What is not shown

- Scale: the exact layer runs only where the solution set is enumerable (|S| ≤ 1920 here); it
  cannot run on production unrolls, which is why Barbarik exists.
- The b06-c4 class (TV 0.17 for GPAP and 0.139 for CMSGen on the same CNF, replicated over seeds)
  is consistent with the numbers here and unchanged.
- Barbarik3's `PM_test` targets weighted distributions and needs a dual oracle; `CM_test` is the
  correct and cheaper instrument for uniform sampling. PM_test with uniform weights is an untried
  cross-check.
- Whether `soft` can be debiased while keeping its success-rate win is unexplored.

## 6. Reproducing

```bash
# the headline campaign: |support| > B, raw configs AND the full shipped pipeline, vs CMSGen
PYTHONPATH=. .venv/bin/python eval/uniformity/fuzz_uniformity.py --count 24 --seeds 3 --n-mult 100 \
    --sup-min 12 --configs b1 b8 b16 b32 --pipelines default circuit circuit+soft \
    --cmsgen .gpap/bin/cmsgen --controls 0.0 0.05 \
    --keep-dir .experiments/fuzz/instances2 --out .experiments/fuzz/round2.json
# the B sweep (the exactness knob)
PYTHONPATH=. .venv/bin/python eval/uniformity/fuzz_uniformity.py --count 12 --seeds 3 --n-mult 100 \
    --configs b1 b2 b4 b8 b16 b32 b64 b256 --controls 0.0
# the certificate (§4b), circuit-only: generate, SCREEN, then every arm including both controls
PYTHONPATH=. .venv/bin/python eval/uniformity/barbarik_gpap.py --explain
mkdir -p .gpap/tmp/circ
# --encoder abc --abc-opt fast is NOT the default and IS what the recorded campaign used:
# bare circuit2cnf writes Tseitin (2,711 vars on b08-c10 against 91), and --abc-opt full differs
# again. Recovered 2026-10-12 by matching n_vars/n_clauses/n_ind against the published rows.
for c in b01 b03 b04 b06 b08 b09 b10 b11 b13 s386 s832 s1196; do for T in 5 10 25; do for tg in rare random; do
  gpap circuit2cnf benchmarks/circuits/bench/seq/$c.bench .gpap/tmp/circ/$c-c$T-$tg.cnf --cycles $T --target $tg --seed 1 --encoder abc --abc-opt fast
done; done; done
for c in c432 c499 c880 c1355 c1908 c2670 c3540 c5315 c6288 c7552; do for tg in rare random; do
  gpap circuit2cnf benchmarks/circuits/bench/comb/$c.bench .gpap/tmp/circ/$c-c1-$tg.cnf --cycles 1 --target $tg --seed 1 --encoder abc --abc-opt fast
done; done
PYTHONPATH=. .venv/bin/python eval/uniformity/barbarik_screen.py .gpap/tmp/circ/*.cnf --out .gpap/tmp/eligible.txt
for arm in "--sampler spur" "--sampler cmsgen" "--sampler pipe --portfolio circuit" \
           "--sampler gpap --config b8" "--sampler gpap --config b1" \
           "--sampler gpap --config s1" "--sampler gpap --config soft"; do
  PYTHONPATH=. .venv/bin/python eval/uniformity/barbarik_gpap.py $(cat .gpap/tmp/eligible.txt) \
      $arm --order-seconds 30 --out .experiments/barbarik/circ.jsonl    # CMSGen-paper setting
      # --order-seconds binds the GPAP arms ONLY; the rival keeps --cmsgen-order-seconds (1800)
  PYTHONPATH=. .venv/bin/python eval/uniformity/barbarik_gpap.py $(head -6 .gpap/tmp/eligible.txt) \
      $arm --epsilon 0.1 --eta 0.9 --delta 0.05 --out .experiments/barbarik/circ_strict.jsonl
done
PYTHONPATH=. .venv/bin/python eval/uniformity/barbarik_table.py .experiments/barbarik/circ.jsonl
PYTHONPATH=. .venv/bin/python eval/uniformity/barbarik_table.py .experiments/barbarik/circ_strict.jsonl
# the TV number (§4c): circuit-only, |S| >= 128 so the arms can separate, paired on identical files
PYTHONPATH=. .venv/bin/python eval/uniformity/fuzz_uniformity.py --kinds circuit --count 80 --seeds 3 \
    --n-mult 100 --sup-min 12 --circuit-inputs 14 20 --s-min 128 --s-max 2000 \
    --configs b16 --pipelines circuit b8 --cmsgen .gpap/bin/cmsgen --controls 0.0 0.05
# and its price: both portfolios in ONE process, arm order alternated
PYTHONPATH=. .venv/bin/python eval/profiling/profile_soft_portfolio.py --glob 'benchmarks/circuits/bench/seq/*.bench' \
    --encoder abc --abc-opt fast --cycles 25 --target rare --arm-a circuit --arm-b b8
```

The 2026-08-28 rows are in `.experiments/barbarik_20260828/`: 92 generated instances, 67 pass
the cost screen, 66 pass the control. `--sampler spur` (must Accept) is the control every table
needs; `--config soft` (must Reject) is the positive control, run here and not re-run by the
paper campaign. `--draw iid` re-runs any
GPAP arm with one chunk per sample, the independence premise barbarik's guarantee is stated
under. `tests/experiments/test_uniformity_fuzz.py` gates the invariants (oracle self-consistency, both
controls, soundness at every B, `B >= |support|` indistinguishable from uniform on every generated
instance) in under 1 s; it deliberately does not gate small-B uniformity, a documented property of
the algorithm class rather than a defect.
