# Running the evaluation on a Colab L4 — results, and what the platform does to you

Measured 2026-08-03/04 on a Colab **NVIDIA L4** (24 GB, 58 SM, sm_89), driven from a terminal via
the Colab CLI. Two sweeps: the GPU-rival head-to-heads and GPAP vs CMSGen.

Four conclusions, and the second is the one that changes how numbers must be reported:

1. **Against the GPU rivals, the verdict is hardware-independent** — identical win counts on both
   cards, because both sides share the GPU (§1).
2. **Against CMSGen it is not.** Our ratio nearly doubled on this box, and *none* of that is GPAP
   getting better: the host CPU is 2.4× slower and CMSGen is single-core (§2).
3. **`soft` is decisively non-uniform** (median TV 0.63–0.76 against a 0.040 floor), and the
   uniformity-safe default costs **2.4× throughput** — measured twice, on two machines (§3).
4. **The numbers reproduce.** A 23-cell re-measurement on a fresh VM matched every deterministic
   field exactly and every success rate to two decimals (§4).

Auditing this data also found a real error in what we had been quoting: the sequential headline
included four instances whose pins are tautologies, inflating the rare column by up to 1.9×
(§2). Across every experiment here — 332 benchmark cells and ~2,800 uniformity runs — there is
**not one unsound emission**.

---

## 1. The rivals: the ranking transfers, the magnitude does not

`eval/colab_rivals_l4.sh`, 3-seed GPAP medians, rival timeout 300 s — identical protocol to the
4080 run. Geomeans over the instances **both** cards scored.

All five rival protocols are now complete on the L4 (9 instances each for Recurrent, 10 for
DEMOTIC-native, 9 for DEMOTIC-miter, 18 CNFs for HTS):

| rival | protocol | GPAP/rival | GPAP wins | rival failed |
|---|---|---:|---:|---:|
| DEMOTIC | their ISCAS-85 pins | 2.5× | 19/20 | 0 |
| DEMOTIC | our rare pins (miter) | 2,427× | 18/18 | 0 |
| Recurrent-CircuitSAT | **their** target table | **4.4×** | 6/6 | 3 |
| Recurrent-CircuitSAT | our rare pins | **634.9×** | 5/6 | 3 |
| HTS | our CNFs, both encoders | 1,584× | 14/15 | 3 |

*Win counts are `eval/paper_tables.py`'s: a cell where the rival scored **zero** counts as a win
in both numerator and denominator, but is excluded from the geomean (no finite ratio exists).
`eval/rivals_summary.py` prints the stricter count that omits those cells entirely — same data,
narrower question, which is why the two tools differ by exactly the zero-scoring cells.*

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

`eval/colab_cmsgen_l4.sh`, **332 cells, 181 instance-slots, 0 skipped, 0 errors, 0 unsound**,
patched CMSGen (`--nowrite`). GPU **NVIDIA L4**, CPU **Intel Xeon @ 2.20 GHz**.

| stage | inst | GPAP/CMSGen | **no-vac** | wins | CMSGen scored 0 |
|---|---:|---:|---:|---:|---:|
| seq-c25-rare | 19 | 58.2× | **39.1×** | 17/18 | 1 |
| seq-c50-rare | 19 | 26.3× | **14.0×** | 15/16 | 3 |
| seq-c25-random | 19 | 191.4× | **183.3×** | 18/18 | 1 |
| seq-c50-random | 19 | 164.6× | **147.0×** | 19/19 | 7 |
| comb-rare | 10 | 615.4× | — | 10/10 | 1 |
| comb-random | 10 | 636.1× | — | 10/10 | 1 |
| hwmcc-c25-rare | 7 | 195.6× | — | 7/7 | 5 |
| hwmcc-c50-rare | 7 | 421.5× | — | 7/7 | 6 |
| hwmcc-c25-random | 2 † | — | — | 2/2 | 2 |
| epfl-rare | 20 | 29.0× | — | 13/16 | 1 |
| epfl-random | 20 | 19.5× | — | 12/16 | 1 |
| sk | 29 | 113.1× | — | 27/28 | 0 |

† incomplete stage (budget). `hwmcc-c50-random` was not run. Every other family is complete at
both targets.

### Two corrections the audit forced

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

### The number that has to travel with those

Paired against the 4080 on the same instances:

| stage | n | GPAP L4/4080 | **CMSGen L4/4080** | ratio 4080 → L4 |
|---|---:|---:|---:|---:|
| sk | 29 | 0.962 | **0.411** | 48.5× → 113.1× |
| seq-c25-rare | 38 | 0.499 | **0.447** | 22.2× → 26.7× |
| comb-rare | 20 | 0.816 | **0.482** | 437.3× → 611.2× |

CMSGen's **own** counters agree exactly: `cms_props_per_s` and `cms_decisions_per_s` both put the
Colab host at **0.413×** the laptop's — two independent counters landing on the same figure.

So the sk_ ratio going 48.5× → 113.1× is **the baseline getting slower, not GPAP getting faster**
(GPAP itself is 0.96× there). Quoting 113× without the CPU beside it is a rigged baseline, and a
reviewer would be right to say so. The same caution applies in advance to the Athena/Zeus runs.

**Fixed as a result**: `gpap bench` now records `cpu` (model, core count) in every JSON alongside
`gpu`. The sweeps above predate it, which is why their host is identified only by CMSGen's
measured work rate.

---

## 3. Uniformity, and what `soft` actually costs

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
the trade-off is a property of the method, not of one machine. **The decision stands** — a
throughput bandit must not hand a rejected distribution to someone who asked for a sampler — but
the price is now measured twice and should be stated whenever the default is defended.

## 4. Consistency re-check: are these numbers reproducible?

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

## 5. The platform: a Colab VM lives about an hour

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

## 6. Coverage, honestly

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
