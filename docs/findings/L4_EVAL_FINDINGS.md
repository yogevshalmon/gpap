# Running the evaluation on a Colab L4 — results, and what the platform does to you

Measured 2026-08-03/04 on a Colab **NVIDIA L4** (24 GB, 58 SM, sm_89), driven from a terminal via
the Colab CLI. Two sweeps: the GPU-rival head-to-heads and GPAP vs CMSGen.

Five conclusions, and the second is the one that changes how numbers must be reported:

1. **Against the GPU rivals, the verdict is hardware-independent** — identical win counts on both
   cards, because both sides share the GPU (§1).
2. **Against CMSGen it is not.** Our ratio nearly doubled on this box, and *none* of that is GPAP
   getting better: the host CPU is 2.4× slower and CMSGen is single-core (§2).
3. **The engine's propagation rate is flat at ~10⁹ implications/s across every family** (1.41×
   spread) while the sample-rate ratio spans ~2 orders of magnitude — so the variation
   belongs to rejection cost, not to the propagator, and an exact identity separates the two (§3).
4. **`soft` is decisively non-uniform** (median TV 0.63–0.76 against a 0.040 floor), and the
   uniformity-safe default costs **1.30× of the published ratio** — but that average pools
   regimes: it is 1.2% on the easy majority and up to 4.6× on deep sequential unrolls (§4).
5. **The numbers reproduce.** A 23-cell re-measurement on a fresh VM matched every deterministic
   field exactly and every success rate to two decimals (§5).

Auditing this data found **four** real errors in what we had been quoting — vacuous pins inside the
sequential headline, a one-sided zero ledger that hid every instance *we* failed, three instances
that "sample" without doing any search, and a propagation table whose columns were filtered
independently (§2, §3). All four are now enforced in code rather than remembered. Across every
experiment here — 357 benchmark cells and ~2,800 uniformity runs — there is **not one unsound
emission**.

---

## 1. The rivals: the ranking transfers, the magnitude does not

`eval/colab_rivals_l4.sh`, 3-seed GPAP medians, rival timeout 300 s — identical protocol to the
4080 run. Geomeans over the instances **both** cards scored.

All five rival protocols are now complete on the L4 (9 instances each for Recurrent, 10 for
DEMOTIC-native, 9 for DEMOTIC-miter, 18 CNFs for HTS):

| rival | protocol | GPAP/rival | GPAP wins | rival=0 | GPAP=0 | both=0 | rival failed |
|---|---|---:|---:|---:|---:|---:|---:|
| DEMOTIC | their ISCAS-85 pins | 2.5× | 19/20 | 0 | 0 | 0 | 0 |
| DEMOTIC | our rare pins (miter) | 2,427× | 16/16 | 0 | 0 | 2 | 0 |
| Recurrent-CircuitSAT | **their** target table | **4.4×** | 6/6 | 0 | 0 | 0 | 3 |
| Recurrent-CircuitSAT | our rare pins | **634.9×** | 5/6 | 1 | 0 | 0 | 3 |
| HTS | our CNFs, both encoders | 1,584× | 13/14 | 2 | 1 | 1 | 3 |

*Both tools that report these (`eval/paper_tables.py` and `eval/rivals_summary.py`) now count
identically — the second was aligned to the first, because a deck and a paper that disagree in
print are worse than either alone. A cell the **rival** scored zero on is an outright win and
enters both sides of the fraction; a cell **we** scored zero on is an outright loss and enters the
denominator only; a cell **neither** scored on is neither. None has a finite ratio, so none enters
a geomean.*

> ⚠️ Corrected 2026-08-05. The earlier counts read DEMOTIC-miter 18/18 and HTS 14/15 because a
> `both = 0` cell — where *neither* sampler emitted anything — was being credited to us as a win.
> Same defect as the CMSGen zero ledger below, found by making the two tools agree.

**The protocol axis dominates the rival axis.** On Recurrent-CircuitSAT — the closest rival to our
domain — the same sampler on the same hardware reads **4.4× on their pins and 635× on ours**, a
140-fold swing driven purely by target rarity. Their pins are rejection-easy by construction, so a
win there is not a sampling result; that is the whole reason the `--target rare` protocol exists.

**Win counts are hardware-independent** (identical to the 4080 instance by instance). That is the
claim worth making; the geomeans are not.

**Why the geomeans move, and why it is not the GPU.** At rare pins the rival often finds ≈0
solutions, so the ratio has a near-zero denominator and a change of one or two solutions moves it
by orders of magnitude. The rival side is also a **single run** where GPAP is a 3-seed median.
The decisive check: DEMOTIC's batch sizes were **identical on both cards** (b03 300,000, b04
30,000, …), and its speedups appear only where it finds almost nothing (b10 151 u/s, s1196
0.87 u/s) while vanishing where it finds ~1 M (c499 1,372k → 1,312k). That is run variance.

> An earlier draft of this file claimed the margin *widens* on weaker hardware, from a single
> `c432` row. It does not. On that same instance DEMOTIC reads 469k u/s on the `raw` variant and
> 1,366k on the `abc` variant of the *same circuit* — a 3× swing that cannot be a hardware effect.

HTS is new: it produced no rows in the 4080 sweep. It needs `pycryptosat`
(`utils/baseline_sat.py` imports it at module scope), without which every run dies at import.

---

## 2. GPAP vs CMSGen: the ratio is a property of the CPU too

`eval/colab_cmsgen_l4.sh`, **357 cells, 193 instance-slots, 0 skipped, 0 errors, 0 unsound**,
patched CMSGen (`--nowrite`). GPU **NVIDIA L4**, CPU **Intel Xeon @ 2.20 GHz**.

Every geomean below already excludes vacuous pins and zero-search cells. The zero columns sum
with the geomean's instances to `inst`, so the ledger balances.

| stage | inst | GPAP/CMSGen | wins | CMSGen=0 | GPAP=0 | both=0 | no-search |
|---|---:|---:|---:|---:|---:|---:|---:|
| seq-c25-rare | 19 | **39.1×** | 17/18 | 1 | 0 | 1 | 0 |
| seq-c50-rare | 19 | **14.0×** | 15/16 | 3 | 0 | 3 | 0 |
| seq-c25-random | 19 | 183.3× | 18/18 | 1 | 0 | 1 | 0 |
| seq-c50-random | 19 | 147.0× | 19/19 | 7 | 0 | 0 | 0 |
| comb-rare | 10 | 615.4× | 10/10 | 1 | 0 | 0 | 0 |
| comb-random | 10 | 636.1× | 10/10 | 1 | 0 | 0 | 0 |
| hwmcc-c25-rare | 7 | 195.6× ⚠n=2 | 7/7 | 5 | 0 | 0 | 0 |
| hwmcc-c50-rare | 7 | 421.5× ⚠n=1 | 7/7 | 6 | 0 | 0 | 0 |
| hwmcc-c25-random | 7 | 263.8× ⚠n=2 | 7/7 | 5 | 0 | 0 | 0 |
| hwmcc-c50-random | 7 | 520.6× ⚠n=1 | 7/7 | 6 | 0 | 0 | 0 |
| epfl-rare | 20 | **9.4×** | 13/17 | 1 | 1 | 3 | 3 |
| epfl-random | 20 | **8.2×** | 12/17 | 1 | 1 | 3 | 2 |
| sk | 29 | **98.9×** | 27/29 | 0 | 1 | 0 | 1 |

⚠ = the geomean stands on 1–2 instances because CMSGen scored zero on the rest. **Do not quote
those four HWMCC ratios**; quote "GPAP wins 7/7, CMSGen produced nothing on 5–6 of 7". The HWMCC
random stages were completed after the first write-up, so the family is now complete at both
targets and both depths.

### Four corrections the audit forced

**The vacuous pins were inside the headline.** `b02`, `b05`, `b07`, `s526` have constant
last-cycle outputs at T=25, so pinning them is a tautology and any sampler scores ~100%.
They were sitting in the sequential geomeans on *both* machines:

| stage | as quoted | no-vac | inflation |
|---|---:|---:|---:|
| seq-c25-rare (4080) | **47.7×** | **36.5×** | 1.31× |
| seq-c50-rare (4080) | 47.6× | 35.7× | 1.33× |
| seq-c25-rare (L4) | 58.2× | 39.1× | 1.49× |
| seq-c50-rare (L4) | 26.3× | 14.0× | 1.87× |

The inflation lands almost entirely on the **rare** column — the one we tell people to quote —
and is negligible at random pins (1.04×). `eval/eval_plots.py` now emits a `no-vac` column in
every `summary.md`, and `eval/slide_tables.py` applies the same exclusion, so a hand-copied
number cannot reintroduce it. The deck's headline frame was corrected 2026-08-04.

**CMSGen scoring zero is not a win we count.** On 28 cells CMSGen emitted *no* samples inside the
budget (`b14`, `b15`, `voter`, `c6288`, 5 of 7 HWMCC-rare…). Those give ratio = ∞ and are excluded
from every geomean, so the published figures understate the result; the count is now reported in
its own column rather than silently dropped.

**But the zero ledger was one-sided — and that is the worse bug.** The table showed the 38
instances where CMSGen produced nothing and *hid* the 14 where we produced nothing. The cause was
mechanical rather than deliberate, which is exactly why it survived review: a CMSGen-zero row has
`ratio = inf` and was branched on explicitly, while a GPAP-zero row has `ratio = 0.0` and a
both-zero row has `ratio = None` — both falsy, both fell through every branch, both vanished.

| | instances |
|---|---:|
| CMSGen emitted nothing, we did (shown as wins) | 38 |
| **we emitted nothing, CMSGen did** (was invisible) | **3** — `bar` ×2, `79.sk_4_40` |
| **neither sampler emitted anything** (was invisible) | **11** — `b11`, `b14`, `b15`, `hyp`, `log2`, `multiplier` |

Table 1 now carries `CMSGen=0`, `GPAP=0` and `both=0` columns that sum, with the geomean's
instances, to the instance count — a reader can add them up and check. `eval/verify_artifact.py`
fails the same way in reverse: it now warns on every instance we could not sample.

**What those three losses actually are**, since we now publish them. All hit the sampler's
zero-progress valve (`sampler.py`: every config measured, plus a grace chunk, still 0 models →
stop) rather than the time limit, so the walls are 1.1 s / 0.0 s / 1.8 s and not 60 s:

| instance | M | support | trajectories tried | models | CMSGen | p(accept) |
|---|---:|---:|---:|---:|---:|---:|
| `bar.aig` (tseitin) | 10,137 | 135 | ~600k | 0 | 47,000 | **0.0** |
| `bar.aig` (abc:fast) | 8,985 | 135 | ~600k | 0 | 38,000 | **0.0** |
| `79.sk_4_40` | 21,005 | 40 | ~1.3M | 0 | 1 | — |

This is the **lose-set** failure mode, and it is structural rather than a budget artifact: at 0
models in ~600k trajectories the success rate is below 1/600k, so more wall would not obviously
help. GPAP discards conflicted trajectories, so a solution set that is a vanishingly small
fraction of the assignment space is exactly what it cannot reach — while CMSGen's backtracking
search walks to solutions systematically and finds 47k of them. Note `p_accept = 0.0`: naive
rejection sampling fails here too, so this is not "the target was easy and we missed it".

It is the same story §3 tells in continuous form: EPFL carries a **53.7× work-per-sample penalty**,
the largest of any family, and `bar` is where that penalty goes to infinity. One mechanism, two
readings — which is a better paper argument than either alone.

**Three instances "sample" without sampling.** EPFL `ctrl`, `dec`, `square` and the sk_ instance
`tableBasedAddition` emit models at 235–240M/s with **zero decisions and zero implications** and
success exactly 1.0: after unit propagation the support is a free hypercube, so there is no search
to do. That is the vacuous-pin problem in a new costume, and it is worse in one respect — the
best-encoder rule *actively selects* these cells, because they carry the highest ratio in the
family. On `ctrl` the Tseitin encoding samples at 2.0M/s with success 0.07 while ABC collapses the
output cone and yields success 1.0 at 240M/s, so the encoder A/B hands the headline the degenerate
cell.

| stage | with zero-search cells | excluded | inflation |
|---|---:|---:|---:|
| epfl-rare | 29.0× | **9.4×** | 3.09× |
| epfl-random | 19.5× | **8.2×** | 2.38× |
| sk | 113.1× | **98.9×** | 1.14× |

No other family contains one. Detected per row by `paper_tables.zero_search` rather than by a
hard-coded name list, so a new instance cannot smuggle it back in.

### The number that has to travel with those

Paired against the 4080 on the same instances:

| stage | n | GPAP L4/4080 | **CMSGen L4/4080** | ratio 4080 → L4 |
|---|---:|---:|---:|---:|
| sk | 29 | 0.962 | **0.411** | 46.2× → 98.9× |
| seq-c25-rare | 38 | 0.499 | **0.447** | 22.2× → 26.7× |
| comb-rare | 20 | 0.816 | **0.482** | 437.3× → 611.2× |

CMSGen's **own** counters agree exactly: `cms_props_per_s` and `cms_decisions_per_s` both put the
Colab host at **0.413×** the laptop's — two independent counters landing on the same figure.

So the sk_ ratio going 46.2× → 98.9× is **the baseline getting slower, not GPAP getting faster**
(GPAP itself is 0.96× there). Quoting 98.9× without the CPU beside it is a rigged baseline, and a
reviewer would be right to say so. The same caution applies in advance to the Athena/Zeus runs.

**Fixed as a result**: `gpap bench` now records `cpu` (model, core count) in every JSON alongside
`gpu`. The sweeps above predate it, which is why their host is identified only by CMSGen's
measured work rate.

---

## 3. Propagation rate: separating the engine from the sampler

The sample-rate ratios above range from 8× to 636×, which invites the obvious objection that we
are reporting a benchmark-selection effect. The propagation counters answer it. GPAP's
implications/s comes from the packed-word kernel; CMSGen's comes from its **own** `--statsevery`
counters (`scripts/cmsgen_gpap_eval.patch`), so this is its accounting of its own work, not our
estimate of it.

| family (rare) | n | GPAP impl/s | CMSGen impl/s | engine advantage | work/sample | = net |
|---|---:|---:|---:|---:|---:|---:|
| ISCAS-85 comb | 9 | 1.05×10⁹ | 1.67×10⁶ | 627× | 1.0× | **615.4×** |
| ITC'99/ISCAS-89 seq, T=25 | 13 | 1.14×10⁹ | 5.20×10⁶ | 220× | 5.6× | **39.1×** |
| ITC'99/ISCAS-89 seq, T=50 | 10 | 1.07×10⁹ | 5.29×10⁶ | 202× | 14.4× | **14.0×** |
| HWMCC'17, T=25 | 2 | 1.05×10⁹ | 4.58×10⁶ | 230× | 1.2× | **195.6×** |
| EPFL comb | 12 | 1.08×10⁹ | 2.14×10⁶ | 506× | 53.7× | **9.4×** |
| sk_ projected CNF | 27 | 1.00×10⁹ | 4.23×10⁶ | 237× | 2.4× | **98.9×** |

**GPAP holds ~10⁹ implications/s on every family — geomean 1.07×10⁹, spread 1.41× between the
fastest and slowest** (over the nine families carrying ≥3 instances; the four HWMCC rows where
CMSGen scored zero almost everywhere stand on 1–2 instances and are shown but not quoted) — while the sample-rate ratio in the last column varies by nearly two orders of
magnitude across the same rows. The propagator's rate is a property of the GPU (occupancy-bound,
consistent with the cross-GPU work-rate result), not of the benchmark.

`plots/work_rates_<stage>.png` shows this per instance and is the clearest single picture of it:
on a log-log scatter against CMSGen, GPAP's implication points form a **horizontal band** — our
rate stays at ~10⁹ while CMSGen's moves over half a decade across the very same instances. A
propagator whose speed depended on the formula would trend along the diagonal; ours does not.

The two facts are connected by an identity that holds exactly, per instance and per family:

```
samples/s ratio  =  (implications/s ratio)  /  (implications-per-sample ratio)
```

The `= net` column above is literally `engine advantage / work per sample`, and it reproduces
Table 1's published geomean to the printed digit on every row. Verified to within 1% on 9 of 9
families where it is computable — and **re-derived from the raw rows by
`eval/verify_artifact.py`**, so the claim is checked rather than asserted by the tool that makes
it. The figure is `plots/propagation_summary.png`. It says something a single throughput number
cannot:

* the **engine** buys a flat 197–631× (geomean 306×), essentially independent of the instance;
* our **algorithm spends part of it back**, because GPAP discards a conflicted trajectory outright
  (rejection sampling) whereas CMSGen backtracks and keeps the partial work. That cost is 1.0× on
  ISCAS-85 (we are as cheap per sample as CMSGen there), 5.6× on sequential circuits at T=25,
  14.4× at T=50, and **53.7× on EPFL**;
* what remains is the published ratio.

This is also the cleanest available explanation of where we lose. EPFL is both the family with the
largest work-per-sample penalty *and* the only family where GPAP scored zero on an instance CMSGen
sampled (`bar`) — the same mechanism, read twice.

⚠️ Getting this table wrong is easy in a specific way: if each column is filtered independently,
an instance like `bar` (propagates hard, emits nothing) enters the implications ratio but not the
work-per-sample ratio, and the identity breaks by 3× on that family while every per-instance
identity still holds. Every column is computed over the same rows.

---

## 4. Uniformity, and what `soft` actually costs

`eval/colab_uniformity_l4.sh` → `.experiments/uniformity/`. **72 generated instances** (36 random
k-CNF, 36 random Tseitin circuits) whose *entire* solution set is enumerated by brute force, so
the target distribution is known exactly; χ² over 3 seeds, Fisher-combined, α=0.001.
13 samplers. **0 unsound emissions out of every run.**

| sampler | circuit: median TV | flagged | k-CNF: median TV | flagged |
|---|---:|---:|---:|---:|
| `ctl:uniform` (negative control) | 0.0395 | **0/36** | 0.0401 | **0/36** |
| `ctl:tv0.05` (positive control) | 0.0577 | **36/36** | 0.0580 | **36/36** |
| `gpap:b1` | 0.1109 | 18/36 | 0.4368 | 36/36 |
| `gpap:b8` | 0.0401 | 10/36 | 0.4033 | 36/36 |
| **`gpap:soft`** | **0.6255** | 24/36 | **0.7605** | 36/36 |
| `gpap:b32` | 0.0399 | **0/36** | 0.0401 | **0/36** |
| `gpap:b64` | 0.0399 | **0/36** | 0.0401 | **0/36** |
| `cmsgen` | 0.0945 | 25/36 | 0.2576 | 36/36 |

**The controls calibrate the instrument**, which is what makes the rest of the column readable:
the exact-uniform control is flagged 0/72 (no false positives) and the known-TV-0.05 control
72/72 (full power at the sensitivity floor). Without both, "no skew detected" would be
uninterpretable — `eval/verify_artifact.py` fails an artifact that lacks either.

Three results:

* **`soft` is not a uniform sampler.** Median TV **0.63 / 0.76** against a floor of 0.040 — an
  order of magnitude worse than every other arm, on both instance kinds. This is the independent
  confirmation of the Barbarik rejection that made it opt-in.
* **B ≥ |support| is exactly uniform.** `b32`/`b64` sit *at* the control's TV and are flagged
  0/36 on both kinds. That is the theory holding: GPAP discards conflicted trajectories rather
  than backtracking, so deciding the whole support in one round *is* rejection sampling.
* **The regime split is real and must not be aggregated away.** On circuits `gpap:b8` (0.0401)
  is at the floor and better than CMSGen (0.0945); on random k-CNF we are clearly worse
  (0.40 vs 0.26). Never quote a single uniformity number for GPAP.

### The price of excluding `soft`

`eval/profile_soft_portfolio.py`, both arms in one process on the same encoded CNF, arm order
alternated, 19 sequential circuits (4 vacuous excluded):

**Geomean 0.423× — the uniformity-safe portfolio is ~2.4× slower**, and the loss is
concentrated: `soft` won the bandit on only **7 of 15** instances, so where it applies the effect
is far larger than the geomean suggests.

| instance | with `soft` | safe | ratio |
|---|---:|---:|---:|
| `b11` | 276/s | **0/s** | capability loss |
| `b12` | 10,864/s | 23/s | 0.002× |
| `s832` | 1,382/s | 189/s | 0.137× |
| `b13` | 128/s | 32/s | 0.250× |

This replicates the 4080 measurement (0.410× geomean, 7/15, `b11` → 0) on different hardware, so
the trade-off is a property of the method, not of one machine.

### The price measured on the comparison we actually publish

Both figures above are GPAP against itself on 19 circuits. That answers "how much throughput did
we give up", not "how much of our published lead did we give up" — and only the second is visible
to a reader of Table 1. So the **entire GPAP-vs-CMSGen matrix was run a second time with
`--allow-soft`** (`eval/colab_cmsgen_soft_l4.sh`, same L4 instance type, same budgets, same
encoders, one flag changed, injected through `run_full_eval.sh`'s single `bench` call so both arms
traverse identical code). Paired per (stage, instance, encoder) by `eval/soft_price_tables.py`.

**Both arms complete: 357 cells, 13/13 stages, 0 skipped, 0 errors, 0 unsound**; the bandit chose
`soft` on 69 of them.

**Control first** — only GPAP's portfolio changed, so anything else moving would confound the
result: CMSGen's throughput between arms is **1.004×** (n=274) and **0 of 357 paired cells** differ
in clauses `M` or `|support|`. The arms are comparable, and that is checked, not assumed.

| family | targets | safe/soft (all) | where `soft` chosen | published headline |
|---|---|---:|---:|---|
| ITC'99/ISCAS-89 seq, T=50 | rare | **0.454×** | **0.097×** (12) | 14.0× → **64.9×** (4.6×) |
| ITC'99/ISCAS-89 seq, T=25 | rare | **0.601×** | 0.278× (14) | 39.1× → **74.7×** (1.9×) |
| HWMCC'17, T=50 | rare | 0.599× | 0.109× (3) | 421.5× → 395.8× ⚠n=1 |
| HWMCC'17, T=25 | rare | 0.733× | 0.511× (6) | 195.6× → 202.7× |
| EPFL combinational | random | 0.810× | 0.321× (7) | 8.2× → 11.8× (1.4×) |
| EPFL combinational | rare | 0.849× | 0.624× (10) | 9.4× → 11.1× |
| ITC'99/ISCAS-89 seq, T=50 | random | 0.876× | 0.248× (5) | 147.0× → **253.8×** (1.7×) |
| ITC'99/ISCAS-89 seq, T=25 | random | 0.952× | 0.767× (9) | 183.3× → 199.1× |
| HWMCC'17, T=25/50 | random | 0.959 / 0.974× | — (0) | ≈unchanged |
| ISCAS-85 combinational | rare | 0.969× | 0.970× (1) | 615.4× → 632.4× |
| Sketch / sk_ CNF | — | 0.971× | 1.011× (1) | 98.9× → 101.8× |
| ISCAS-85 combinational | random | **1.016×** | 0.996× (1) | 636.1× → 611.8× |

**The single geomean is the least useful number here.** Over the whole matrix the safe portfolio
runs at **0.796×** and the published headline moves **1.30×**, but that average pools regimes that
behave completely differently. The cost is a **sequential-circuit and EPFL** phenomenon — up to
4.6× of the published ratio at T=50 rare — and is **≈1.0× on ISCAS-85, sk_ and HWMCC-random**, where
`soft` is essentially never chosen. Depth amplifies it at both target types (T=50 costs more than
T=25 at rare *and* at random), which is `soft`'s lookahead earning its keep on deep unrolls,
exactly as predicted.

The sharpest cut is by instance difficulty — the safe arm's own success rate, which is knowable
before a portfolio is chosen:

| safe-arm success rate | cells | safe/soft | `soft` won the bandit |
|---|---:|---:|---:|
| < 10⁻⁴ | 13 | **0.119×** | 5/13 |
| 10⁻⁴ – 10⁻² | 22 | **0.313×** | 12/22 |
| 10⁻² – 0.5 | 86 | 0.822× | 24/86 |
| > 0.5 | 196 | **0.988×** | 25/196 |

Monotone across 317 paired cells. **The uniformity guarantee is nearly free on the instances the
sampler already handles (1.2% on the 196-cell majority) and expensive only where it does not** —
and that is precisely the regime `soft` wins. This puts a number on the middle path parked since
2026-07-30: a bandit falling back to `soft` only in the bottom two buckets would address 35 of 317
cells, i.e. recover most of the loss while leaving ~89% of cells uniform. It is still not
implemented, and it would still need a uniformity argument of its own.

⚠️ **Capability differences below success 10⁻³ are draws, not evidence.** The zero-progress valve
stops a run once every config plus a grace chunk has missed, which makes very-low-success
instances bimodal — and it fires in both directions. On EPFL `div` (succ 2.1×10⁻⁴) the *safe* arm
ran 173 chunks over 60 s and found 441 models while the soft arm stopped at 1.72 s with zero,
having chosen a **non-soft** config. 6 of the 7 capability differences here sit below that
threshold and are flagged automatically (6 lost / 1 gained — and the one *gain* is ours, which
is what shows the valve fires both ways). The exception worth keeping: **`b12` is unambiguous**
(success 0.072 with `soft` vs 7.1×10⁻⁵ without — three orders of magnitude, far above any valve
effect), and **`b11` replicates** across two independent campaigns at ~1.5×10⁻³ vs 0.

**The decision stands** — a throughput bandit must not hand a rejected distribution to someone who
asked for a sampler — but the price is now measured three ways, and the honest form of the claim
is the difficulty table, not a single geomean.

## 5. Consistency re-check: are these numbers reproducible?

The sweeps above were produced across a dozen VMs over ~8 h. That is enough moving parts to ask
whether any result depended on one machine's state, so 23 cells spanning four families were
re-measured on a **fresh L4** with the protocol pinned (`eval/colab_recheck_l4.sh`, into its own
directory so it can never be mistaken for the original), and compared field by field
(`eval/recheck_compare.py`).

What counts as agreement is stated rather than assumed:

| class | fields | requirement | result |
|---|---|---|---|
| deterministic | `M`, `support`, packed split, propagate variant | must match **exactly** | **23/23 exact** |
| stable | `succ` | within 0.05 | **within 0.01 everywhere** |
| noisy by nature | throughput | reported, not asserted | geomean **1.032×** (0.96–1.12) |

Clause counts after simplify+mine are identical to the digit on every cell (`b12` tseitin
109,432 both times; `s1196` abc 31,221 both times), success rates reproduce to two decimals
including the degenerate ones (`adder` 0.00, `i2c` tseitin 0.00), and throughput sits within
±12% — well inside the ±40% draw noise this hardware shows on low-success instances.

**So the measurements are reproducible, and the deterministic half is exactly reproducible.** The
re-check is cheap (~15 min) and is the right thing to run after any multi-VM sweep.

## 6. The platform: a Colab VM lives about an hour

Session history, unambiguous: created 13:45:29 UTC, keep-alive daemon started normally,
**terminated 14:49:18** — 1 h 04 m, no error logged. Not idle pruning; that is the assignment's
lifetime. It recurred across every session of both sweeps.

**The rule this forces: results only exist once they are on the local disk.** Per-row
checkpointing to the VM's disk buys nothing, because that disk dies with the VM. The first rivals
run was lost exactly this way — an hour of work, fully checkpointed, all of it on a machine that
went away.

`eval/colab_supervisor.py` encodes the working loop:

```
ensure session -> push cached binaries -> bootstrap -> push partials -> launch -> poll + PULL
     ^                                                                              |
     +------------------------- VM reclaimed, work unfinished ----------------------+
```

with the pieces that make it survive: every harness takes `--resume`; each cycle downloads the
result files and only replaces a local copy once the new one **parses** (a download of a file
mid-write silently truncates); each fresh VM is seeded with the mirrored partials before
relaunch; and built binaries (CMSGen ≈ 10 min) are cached locally and pushed, since rebuilding
them costs a sixth of every machine.

### Traps, each of which cost real time here

* **Never grep a log for a status string over the console.** The console echoes the command it
  runs, so `grep -c BOOT_FAIL …` matches its own text; and `set -x` prints a script's
  `|| { echo BOOT_FAIL…; }` branches on success. Both read as failure the moment you look. Use
  **marker files** counted with `ls … | wc -l`, which answers with a bare digit.
* **Resume must retry error rows, and must replace them, not append.** Treating an error row as
  finished froze 18 HTS rows behind a missing dependency forever; appending the retry instead of
  replacing produced 25 rows in an 18-instance file.
* **Never wait on a grep against an append-only mirror** — it matches a previous run's output and
  fires early.
* **One poller per session.** Two concurrent console users corrupt each other's commands.
* Editing a running bash script rewrites it under a process that reads by byte offset.

---

## 7. Coverage, honestly

**Complete**: every CMSGen family at `rare` (seq T=25 and T=50, comb, hwmcc T=25 and T=50, epfl,
sk); `random` for seq, comb and epfl; DEMOTIC on both protocols; HTS.

**Nothing is incomplete.** Every CMSGen family ran at both targets and both depths (13 stages),
and all five rival protocols ran to their full instance list. The only qualifications are the
ones the data itself carries: four HWMCC stages whose geomean rests on 1-2 instances because
CMSGen scored zero on the rest, and the rival runs that timed out or found no gradient path —
both counted and printed, never silently dropped.

Nothing was skipped or errored inside the stages that did run: **332 cells, 0 skipped, 0 errors,
0 unsound**.

Raw data and how to regenerate every table here:

```bash
.experiments/cmsgen_l4_2026-08-03/     # GPAP vs CMSGen  (+ plots/summary.md)
.experiments/l4_rivals/                # GPAP vs the 3 GPU rivals
.experiments/recheck/                  # the consistency re-run, deliberately separate

eval/eval_plots.py     <dir>                    # summary.md + figures (has the no-vac column)
eval/rivals_summary.py <4080-dir> <l4-dir>      # rival tables, paired across machines
eval/slide_tables.py   --cmsgen … --rivals …    # the deck's appendix, generated
eval/recheck_compare.py <original> <recheck>    # the consistency verdict
```
