# How uniform is GPAP — against an exact oracle, against CMSGen, and under Barbarik

> **Reader:** contributors and researchers — the evidence on how uniformly GPAP samples. A dated research log, not a user guide.
>
> **Status:** settled. The paper's uniformity claim is Barbarik alone (§4b), from the campaign
> `uniformity_20261013`. The TV campaigns (§1–§3, §4c) are design evidence from 2026-07/08, on
> the engine before the sampling loop moved into the core and before the give-up fix; the paper
> quotes none of their numbers. Condensed 2026-10-21; the full earlier text is
> `git show 823eefd:docs/findings/UNIFORMITY_FINDINGS.md`.
> **Measured:** exact-oracle campaigns 2026-07-29/30 and 2026-08-28; Barbarik 2026-08-28 and the
> paper campaign 2026-10-13 (`uniformity_20261013`, e821180); all on the RTX 4080 Laptop. A
> distributional result does not depend on the card.
> **Rule lives in:** `gpap/sampling/pipeline.py` (the portfolios, `--allow-soft`), the warnings
> on `sound_check` in `cpp/include/gpap/gpap.h` and `.hpp`, and `eval/uniformity/` — which makes
> the claims the library does not: `metrics.py` (statistics), `fuzz_uniformity.py` (exact
> oracle), `barbarik_gpap.py` + `barbarik_screen.py` + `barbarik_table.py` (the certificate).
> **Gated by:** `tests/experiments/test_uniformity_fuzz.py`, `tests/experiments/test_barbarik_harness.py`,
> `tests/sampling/test_sampler.py`.
> **SECTION NUMBERS ARE AN API** — cited from `gpap/`, `cpp/`, `eval/`, the tests and the
> generated paper prose. Append; never renumber.

GPAP's sampler is the GPU-parallel form of CMSGen's Algorithm 1, and neither has a uniformity
guarantee. What is settled:

1. **Barbarik, the paper's claim:** the shipped `circuit` portfolio is Accepted on all 65
   eligible circuit instances, with zero Rejects and zero no-verdicts; pinned `b8` and CMSGen
   are 65/65 too (§4b).
2. **`soft` is not uniform:** Barbarik Rejects it on 64 of 65 circuit instances, so it has been
   opt-in (`--allow-soft`) since 2026-07-30 (§4).
3. **`B >= |support|` is exactly uniform:** keeping the survivors is rejection sampling (§3).
4. **The default pays for throughput in TV:** the bandit picks B=1, which reads 1.31–1.35× worse
   than CMSGen in TV median while never skewing where CMSGen does not; `--portfolio b8` holds the
   exact-uniform floor for ~10% throughput (§4c).
5. **The regime splits:** on circuits GPAP sits at or near the floor and CMSGen does not; on
   random k-CNF over all variables GPAP is worse. Never quote an aggregate (§2).

## 1. The instrument: generated instances with an enumerable solution set

`eval/uniformity/fuzz_uniformity.py` generates small formulas, enumerates their solution sets and
runs chi-square and total variation (TV) against uniform. Two families: random k-CNF with the
support every variable, and random Tseitin circuits with the support their primary inputs (one
extension per projection, so projected GPAP compares with full-model CMSGen).

Both controls run in every campaign. Exactly uniform draws read the theoretical floor (TV 0.0399
at `n = 100|S|`) and are never flagged; a planted TV of 0.05 is detected 30/30. "ok" means no
skew detectable at TV 0.05, never "uniform".

## 2. GPAP vs CMSGen, paired on identical files

24 instances (support 12–16, |S| 28–382), 3 seeds, every arm on the same file. The two families
answer oppositely, so an aggregate means nothing:

| family (12 each), TV median | `gpap:b8` | shipped pipeline | `cmsgen` |
|---|---:|---:|---:|
| circuit (support = primary inputs) | 0.039 | 0.040 | 0.087 |
| random k-CNF (support = every variable) | 0.446 | 0.441 | 0.272 |

Skew comes from propagation among support variables. A Tseitin circuit propagates forward into
its gates and rarely forces one input from another; a random k-CNF over all variables propagates
inside the support constantly. These circuits are small enough that every sampler sits at the
floor; §4c re-measures where the arms separate.

## 3. B is an exactness knob, and it bottoms out at exactly uniform

`B` is the number of variables decided per propagation round. Over 12 instances the TV median is
0.21–0.23 at B = 1–4 and 0.039, the uniform control's floor, from B = 8 up; at
`B >= |support|` every instance matches the floor.

Why: at `B >= |support|` one round assigns the whole support a uniform random polarity, then
propagates. GPAP never backtracks — it discards conflicted trajectories — so keeping the
survivors is exact rejection sampling over the projection. Small B interleaves propagation
between decisions and weights each solution by its attraction basin, the skew CMSGen has too;
CMSGen must backtrack, so it cannot reach this limit. Exactness needs propagation to close the
formula from the support (true for Tseitin circuits).

The cost is a rejection sampler's: yield equals `p_accept`. On Barbarik's kernel formulas
`B >= |support|` produced 0/50 and 11/50 samples where `b8` Accepts, so the exactness claim
rests on the exact oracle alone.

### 3b. `|support| >> B`, the regime real benchmarks live in

A real unroll's support is `|PI| × T`, in the hundreds, beyond enumeration. The nearest proxy,
circuits at support 18–20, keeps every GPAP config at the floor (median 0.0406, CMSGen 0.0766),
and B=1 over 13 rounds leaves the circuit median unchanged: the round count alone does not drive
skew. `B >= |support|` is sufficient, not necessary.

### 3c. The 600-instance circuit campaign: the bandit is the pipeline's uniformity cost

600 circuit instances × 3 seeds, zero unsound rows: `b8` TV median 0.0411, the shipped pipeline
0.112, CMSGen 0.102. A one-config portfolio isolates the cause: the pipeline restricted to `[b8]`
reads 0.0420 (raw `b8` 0.0422), the default bandit 0.1666, because it picked B=1 on 30 of 30 for
throughput. B=1 is Barbarik-Accepted (§4b), so this is a trade-off, not a defect: do not delete
the config.

## 4. Barbarik, and the `soft` decision

`soft` (conflict-risk polarity) was Rejected on all five instances of the first Barbarik run
(2026-07-30) and on 64 of 65 later (§4b). The bandit selects on models per second with no
uniformity term, so wherever `soft` won it the run emitted a distribution the standard tester
rejects. Since 2026-07-30 it is in no default portfolio and opt-in via `--allow-soft` (gated by
`test_soft_is_not_in_any_default_portfolio`), and by policy the paper has no `soft` table.

The price, parked by the user on 2026-07-30: on the 19 sequential circuits (`abc:fast`, T=25,
rare) leaving `soft` out costs a geomean 0.410 (2.44× slower) over 13 comparable instances, with
two capability losses (`b11` 11,269 models → 0, `b12` 50,434 → 789). Re-measure with
`eval/profiling/profile_soft_portfolio.py` before revisiting. Sketched, not built: a bandit that
falls back to `soft` only when every uniformity-safe config yields nothing, and says so.

## 4b. What Barbarik actually measures, how CMSGen certified itself with it, and how we do

**The test** (`CM_test`): draw one solution from the sampler under test and one from SPUR
(exactly uniform), build a kernel formula whose solutions split between the two in a known
ratio, hand it back to the sampler and count the split; a lopsided round is bias. The guarantee
is asymmetric (FMCAD'21 §I): a Reject means the sampler is far from uniform; an Accept means it
was not caught, never that it is uniform.

**Two parameter settings, never pooled** (`barbarik_table.py` refuses; `barbarik_gpap.py
--explain` prints the derivation): the CMSGen paper's (eps 0.3, eta 1.8, delta 0.1; 1,872
samples per Accept; it rejects a TV of 0.90 or more) and barbarik's own default (eps 0.1,
eta 0.9, delta 0.05; 70,596 samples; it rejects 0.45 or more).

**How CMSGen certified itself** (FMCAD'21 §V-A): the paper setting on 50 benchmarks filtered on
cost, CMSGen 50/0 — after tuning its restart parameters against Barbarik's rejections (their
Remark 1), so passing is parity with a sampler tuned to pass.

**How we run it:** the same `CM_test`, SPUR and parameters, GPAP hooked in-process, plus:
- **Two controls the paper lacks.** `--sampler spur` puts the reference through our hook and
  must Accept; `--config soft` is biased by construction and must Reject.
- **An eligibility screen** (`barbarik_screen.py`). SPUR itself is Rejected where the declared
  sampling set is not an independent support (five service CNFs, 4–104× extension spread), so a
  Reject there is about the benchmark; tables count only instances whose control Accepts.
  Circuits are ideal subjects: every projection has one extension.
- **Every default-portfolio member has an arm**
  (`test_every_default_portfolio_config_is_barbarik_certifiable`).
- **A budget per order, on the GPAP arms only:** `--order-seconds 30`, three times what the
  slowest Accepting instance averaged. CMSGen keeps 1,800 s and SPUR is unbounded, so every
  no-verdict is ours. An order GPAP cannot fill is a no-verdict, never a Reject. Before
  2026-10-12 the GPAP arms had no budget and were ended by the sampler's own zero-yield give-up
  after 1–10 s; that was every earlier no-verdict (5 `pipe`, 2 `b8`).

**The paper campaign** (`uniformity_20261013`, after the give-up fix; 68 generated circuit
instances, 65 eligible; the CMSGen-paper setting): the shipped `pipe:circuit`, pinned `b8` and
CMSGen each Accept 65/65, with zero Rejects and zero no-verdicts. Its rows ship as
`artifact-data/paper/data/barbarik/circ_abc_*.jsonl` (Table 4b of the generated `RESULTS.md`). It
runs no `soft` arm.

**Kept from the 2026-08-28 campaign** (pinned configs, 66 circuit instances):
- The positive control: `soft` Rejected on 64 of the 65 instances it could sample; its mean bias,
  0.662 against SPUR's 0.558, is the only one that separates.
- At barbarik's stricter setting (6 circuits): `b8`, `b1`, CMSGen and SPUR 6/6, `soft` 0/6.
- On the secondary `sk_`/service CNFs, `b1` is Rejected on `registerlesSwap` and
  `UserServiceImpl` where `s1` Accepts — genuine (the exact oracle agrees on `registerlesSwap`)
  and against us. The `circuit` portfolio, the default for every input, has no `s1`; defaulting
  the portfolio by input type would fix it, and is not done because it moves a default every
  recorded CNF row ran under.
- One variable order per chunk, shared by its rows, against fully i.i.d. draws: a wash against
  the exact oracle, and `--draw iid` Accepts.

## 4c. The TV number, measured on 120 circuit instances in the regime that discriminates

At |S| ≥ 128 (120 instances, |S| 128–1920, 3 seeds, both controls, floor 0.0399):

| arm | TV median | TV p90 | unflagged | skews where CMSGen does not |
|---|---:|---:|---:|---:|
| `gpap:b16` (B ≥ \|support\|) | 0.0398 | 0.0411 | 119/120 | 0 |
| `gpap:b8` | 0.0571 | 0.1230 | 47/120 | 0 |
| `gpap:b1` | 0.0910 | 0.1914 | 44/120 | 0 |
| `pipe:circuit` (shipped default) | 0.1651 | 0.3010 | 46/120 | 0 |
| `cmsgen` | 0.1219 | 0.2270 | 23/120 | — |

Confirmed paired on 80 fresh instances: `pipe:b8` (the shipped pipeline, B=8 pinned) 0.0402, the
default 0.1471, CMSGen 0.1125. The default holds the floor up to |S| ≈ 380 and crosses CMSGen
near 500. Through the pipeline `b8` improves (likely ELS shrinking the effective support) and
`b1` does not.

What may be claimed about the default: Barbarik parity (§4b); it never skews where CMSGen does
not; it is indistinguishable from uniform about twice as often. Not claimable: a lower TV than
CMSGen — it is 1.31× worse, more often exactly right and more often badly wrong. A TV claim needs
`--portfolio b8`: 0.0402 against the 0.0399 floor, 2.8× better than CMSGen.

The price of `--portfolio b8`: a geomean 0.910 (~10% slower) over 14 comparable sequential
circuits, no capability loss (`profile_soft_portfolio.py --arm-a circuit --arm-b b8`, both arms in
one process, order alternated). A release decision, not taken: moving the default moves every
recorded circuit row.

## 5. What is not shown

- The exact oracle stops at enumerable solution sets (|S| ≤ 1920 here); production unrolls rely
  on Barbarik.
- The TV campaigns have not been re-run on the current engine.
- Whether `soft` can be debiased while keeping its success-rate win.

## 6. Reproducing

- **Barbarik** (the paper's campaign): `barbarik_screen.py` for eligibility, `barbarik_gpap.py`
  per arm, `barbarik_table.py` for the counts; the exact command lines are the campaign's
  `$GPAP_RESULTS/uniformity_20261013/run.sh`. The instances are `gpap circuit2cnf … --encoder abc
  --abc-opt fast`, not the default encoder.
- **The exact oracle**, for example §4c:

```bash
PYTHONPATH=. .venv/bin/python eval/uniformity/fuzz_uniformity.py --kinds circuit --count 80 --seeds 3 \
    --n-mult 100 --sup-min 12 --circuit-inputs 14 20 --s-min 128 --s-max 2000 \
    --configs b16 --pipelines circuit b8 --cmsgen .gpap/bin/cmsgen --controls 0.0 0.05
```

- `tests/experiments/test_uniformity_fuzz.py` gates the invariants (the oracle, both controls,
  soundness at every B, `B >= |support|` indistinguishable from uniform) in under a second. It
  does not gate small-B uniformity, a property of the algorithm class.
