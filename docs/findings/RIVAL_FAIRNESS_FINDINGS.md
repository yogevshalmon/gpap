# Rival-comparison fairness: what we were measuring, and what we should have been

2026-08-24/25, RTX 4080 Laptop (local calibration) + NVIDIA L4 (the recorded campaign).
Three audits of the GPU head-to-heads, run because a comparison you have not attacked yourself is
one a reviewer will attack for you. All three found something, and **all three had been flattering
GPAP** — which is the expected direction: every unexamined default in a harness we wrote is a
default we chose.

The rules these settled live in `docs/EVAL_RUNBOOK.md` §6 and `docs/BENCHMARKING.md` §4b/§4c;
the numbers live here.

---

## 1. Emission symmetry — is anyone timed while DELIVERING solutions?

GPAP's timed run emits nothing (`collect=False` leaves `keep_rows` False, so the `var_bits`
gather never fires) and CMSGen is timed under the eval patch's `--nowrite` for exactly that
reason — a stock binary writing every sample cost 8.2x of its wall on 27.sk_3_32. The same has to
hold for the GPU rivals, or the ratio is partly a measurement of their PCIe traffic.

Re-derive with `python eval/rivals/check_emission_symmetry.py --results <campaign>`: it AST-scans
the function each rival's own `timer(...)` wraps for host transfers and IO, checks every
`.sample()` in our timed window passes `collect=False`, and audits `cms_mode` over the rows.

| tool | inside its timed region | verdict |
|---|---|---|
| DEMOTIC | `run_back_prop_comb/seq` — pure GPU descent | clean |
| HTS | `run_back_prop` — pure GPU descent | clean |
| CMSGen | `--nowrite` on 737/737 recorded rows | clean |
| GPAP | `collect=False` ⇒ `keep_rows` False | clean |
| Recurrent | `.cpu()` of per-depth candidates, `run.py:186` | **exception, below** |

DEMOTIC and HTS count and dedup their solutions *after* their own timer stops, so their published
metric excludes it entirely. Note the asymmetry that runs against US and is likewise left alone:
our timed loop pays a per-chunk on-device soundness check no rival performs at all.

**The exception, measured** (4080, `--num_steps 10`): Recurrent copies its per-depth candidate
assignments to the host inside its timed loop — `|PI| x sum(d+1 for d in start_point..T-1)`
strided copies, 0.5-4.0 GiB per run.

| instance | batch | metered egress | % of `model_runtime` | unique with copy on/off |
|---|---|---|---|---|
| b03 | 100k | 0.46 s | 0.70% | identical (1,995,117) |
| s386 | 100k | 0.84-0.91 s | 1.32-1.42% | identical (2,134,454) |
| s1196 | 30k | 0.80 s | 0.88% | identical (662,936) |

Removing the copy outright (snapshots kept on device — their `_check_solution` already does
`par.to(self.device)` on each) leaves `num_unique_solutions` bit-identical. On s386, 4 interleaved
repetitions: removing it saves **0.89 s (1.41%)** where a meter around the block reads **0.84 s
(1.32%)** — agreement to 0.05 s inside a ~2 s spread.

⚠️ **One un-repeated on/off pair is worthless at this scale.** The first pairs read -0.23 s,
+3.11 s and +0.06 s against meters of 0.46/0.91/0.80 s, because run-to-run spread IS the effect
size. The meter is the instrument; the A/B calibrates it.

**Decision: not corrected.** One rival, ~1%, and deducting it would RAISE our ratio — so an
uncorrected row under-reports GPAP rather than flattering it, and the rival's rate stays exactly
the one it publishes. The instrumentation was built, validated and reverted; only the bound is
documented.

---

## 2. The parallelism grant — the batch handed to a rival IS its score

Measured on the 4080. `model_runtime` is nearly flat across these, i.e. the GPU is nowhere near
saturated at the small end:

| rival | instance | batch | unique/s | factor |
|---|---|---|---|---|
| HTS | b03-c25-rare | 100 -> 100,000 | 1 -> 294 | **295x** |
| DEMOTIC | c1355 | 1k -> 1M | 1.2k -> 662k | **564x** |
| DEMOTIC | c432 | 1k -> 300k | 2.5k -> 654k | **258x** |
| Recurrent | s386 T=25 | 10k -> 300k | 3.8k -> 69.9k | **18.6x** |

So the batch is the same knob `K` is for us, and GPAP sizes `K` at 0.80 of free VRAM. Two things
were wrong before 2026-08-25, both in our favour: the circuit rivals' budget was a **fixed 5 GiB
whatever the card** (a fifth of a 24 GB L4 while GPAP took ~19 GB of it), and **HTS ran at batch
100/1000**.

**Their own defaults cannot serve as the rule.** All three argparse defaults are `--batch_size
10`, a placeholder; the published demo scripts are per-instance one-liners with a 1000x spread on
comparably tiny circuits — DEMOTIC 1,000 on c17's six gates, Recurrent 1,000,000 on s27's ten,
HTS 100 on a 75-variable CNF. Transplanted onto our instances those hand DEMOTIC ~1/500 and HTS
~1/295 of what their tool actually reaches, inflating our ratios by the same factor. A share of
the machine is the only quantity that means the same thing for both sides.

### Why a static memory model is not enough

Peak VRAM per gate-row is not a constant — measured for DEMOTIC at batch 300k:

| instance | gates | peak GiB | B/gate-row |
|---|---|---|---|
| c432 | 160 | 1.03 | 23.0 |
| c1355 | 546 | 3.05 | 20.0 |
| c2670 | 1193 | 4.88 | 14.6 |
| c3540 | 1669 | 5.40 | 11.6 |

Part of the cost is per-ROW, not per-gate, so any single factor over-charges one end. At 20 B the
model rejects c2670 at batch 1M, which really needs ~16 GiB and fits an L4 fine. The naive
`n_gates * T * batch * 4` estimate undercounts real peak by 2-6x, tool-dependent (Recurrent
s386@300k: 4.44 GiB estimated, 8.36 measured; HTS: 8.4 B per clause-row).

### The rule

Largest batch that fits **0.80 of free VRAM** — **ONE measured run** (2026-09-03; the user's
rule: "1 config" for the rival). The only fallback is a DRIVER REFUSAL, which is not a run at all:
"device not ready" reproduces at a fixed size on some models and passes one rung lower, so the
machine gets the last word over our estimate and the next rung down is tried; the row still
holds exactly one measurement. A TIMEOUT is a result and is recorded as one — never retried at
a smaller batch (a smaller rung might come in under the cap, but that is a second configuration).

⚠️ **Superseded:** until 2026-09-03 the ladder also probed ONE rung up after a successful run and
kept it only if it was also faster (the spill guard: Recurrent on s386 runs at 1M and is *slower*
than at 300k), and it stepped a timeout down the ladder. On the L4 campaign the branches were
load-bearing — 92 up-probes recovered an under-grant, 82 refusals, 18 spills caught — but the
up-probe is a best-of-two, and best-of-N over a knob is the same trap §8 forbids for our own
encoder choice. The static grant now stands on its own; where `fit_batch`'s model under-grants,
the rival is under-granted and the row says what it got (`batch_sweep`).

A timeout must not walk the ladder either way: HTS's three tseitin cells once stepped
1M -> 1,000 at the full 300 s cap on every rung, ~40 min per cell to record an error they had
after the first run.

### What it cost us

Full L4 re-run, 2026-08-25/26, 74 rows, 0 unsound, 0 zero-search, pinned commits, 17.5 GiB
budget on every row (0.80 of the L4's free VRAM):

| stage | old grant | new grant | shift |
|---|---|---|---|
| DEMOTIC, their ISCAS-85 pins | 4.9x | **4.6x** | 1.1x lower |
| DEMOTIC, our rare pins (miter) | 6,176.4x | **4,344.1x** | 1.4x lower |
| Recurrent-CircuitSAT, their table | 8.4x | **6.1x** | 1.4x lower |
| Recurrent-CircuitSAT, our rare pins | 1,507.2x | **660.3x** | 2.3x lower |
| HTS, our CNFs | 6,369.5x | **602.9x** | **10.6x lower** |

The correction is concentrated on HTS exactly as the local calibration predicted, because it was
the only rival being run far below its share: per cell its rate rose 10-274x (b10-c25-abc
4 -> 985 unique/s at batch 1,000 -> 300,000). **Three HTS cells also left the zero column**
(b13-abc, s386-abc, s386-tseitin): at batch 1,000 it found no solutions at all and we scored
them as outright wins; at its proper share it finds some, so the geomean now stands on 13 cells
instead of 10. DEMOTIC barely moved — the old fixed budget happened to serve it about right.

⚠️ **Only the rival side moved, and that was checked rather than assumed.** GPAP's own arm
reproduces across the two campaigns to within draw noise — median new/old 0.94-1.00x per stage,
worst single cell 1.29x — so the shift is attributable to the grant, not to drift on our side.
Where our arm does differ it is slightly LOWER, which makes the new ratios marginally
conservative.

---

## 3. Preprocessing — GPAP's was outside the headline rate, CMSGen's was inside its own

`ratio` divides GPAP's steady-state sampling rate by CMSGen's samples-per-subprocess-wall — but
CMSGen's wall already contains ITS parse and setup. Across the 329 timed L4 rows, GPAP's
preparation (parse + construct + ELS + warmup + mining + rebuild) is a **median 20.7% / mean
27.4% of its end-to-end wall** (max 99%):

| phase | median | mean |
|---|---|---|
| construct | 0.15 s | 1.45 s |
| mine | 0.18 s | 0.92 s |
| rebuild | 0.14 s | 2.35 s |
| parse / warmup / recheck | ~0.01 s each | — |

Suite geomean **165.0x steady-state vs 96.7x e2e — the headline was 1.71x the like-for-like
number** (EPFL 2.4-2.8x, sk 2.2x). `ratio_e2e` was already recorded; only the table omitted it.
Table 1 now prints `e2e` and `pre%` beside the steady-state column. Circuit ENCODING is in
neither — it is `t_enc`, an encoder cost, reported per instance.

---

## 4. Making a re-run affordable

Of the 2026-08-23 campaign's ~3.3 h rival stage, the three rivals' own subprocesses were **64
min**. The rest was our side — the GPAP arm (3 seeds x 18 cells for Recurrent alone), the
`start_point` probes, `sim_baseline` — and all of it was re-paid whenever a stage re-ran for a
RIVAL-side reason, which is exactly what fixing §2 required.

Both are cached on disk under `<output dir>/.gpap_cache`: the **GPAP arm** keyed on the sampled
file's BYTES + the task + the engine build + the GPU name, and Recurrent's **start_point probe**
keyed on circuit + depth (it is pure, yet each probe pays a full process launch and model build).
A probe that TIMED OUT is deliberately not cached — that is about the machine, not the circuit.

Two traps found while building it:

- `cached_gpap(..., lambda: None)` as a read-only lookup POISONS the cache: it stores `None`, and
  every later lookup is then a false hit that silently skips the work. Hence `cache_peek` /
  `cache_store` with a `CACHE_MISS` sentinel.
- `colab_supervisor.pull_results` validated EVERY mirrored file as JSON, so the cache tarball was
  downloaded and deleted on every cycle — any non-JSON artifact was silently discarded. It now
  validates by type and still rejects a download caught mid-write, which is what the check was for.

---

## 5. The artifact packager could not ship the campaign (2026-08-26)

Found while reviewing the branch: `eval/artifact/make_artifact.sh` — the script that produces the
archive a reviewer actually receives — was broken in two independent ways, and the second one
failed *quietly*, which is the dangerous kind.

**(a) `REPO_ROOT` was one level short.** The file lives in `eval/artifact/`, so the root is two
levels up; it computed `..`. That is the same off-by-one the eval/ reorganisation left in eleven
other files — missed here because nothing had run the packager since. With it wrong, `$RESULTS`
pointed *inside* the repo, every dataset resolved to "no source found", and `cd "$REPO_ROOT"`
landed in `eval/` so the table copies and the verifier path were wrong too. The packager could
not build an archive at all.

**(b) `$PRIOR` outranked every fresh run directory.** `copy` takes the first candidate that
exists and `"$PRIOR/<name>"` was always listed first, so with an August archive on disk it
packaged **2026-08-04/05 data for every arm** — including rival rows measured under the old fixed
5 GiB batch grant and the pre-seed-fix HTS arm — while printing `[ok]` for each one. Verified by
hand before the fix: all four principal datasets resolved to the 20260806 archive.

Fixed: the packager is CAMPAIGN-aware (`GPAP_CAMPAIGN`, else the newest `l4_campaign_*`), a
campaign directory outranks whatever the caller listed, every source prints its age, and a source
older than `GPAP_ARTIFACT_MAX_AGE_DAYS` (14) **refuses to package** unless it is named in
`ARCHIVAL` — the two datasets whose age is the point (`uniformity_prior_campaign`, the 4080
reference). `kernel_ab_l4` is now packaged too: it had a paper table but was never in the archive,
so the archive could not regenerate that table.

Verified end-to-end: the rebuilt archive resolves all seven fresh arms to
`l4_campaign_20260823`, carries HTS batches up to 1,000,000 (i.e. the fair grant, not the old
1,000), and reports the correct verdict.

**The general lesson, and it is the same one twice**: a path or a preference that silently selects
the wrong data is worse than one that crashes. Both of these had been sitting in a script nobody
had executed since the directory move — the artifact pipeline needs the same smoke-before-you-need-it
treatment as the sweep drivers.

---

## 6. `num_steps` — the batch problem again, on the knob nobody looked at (2026-08-26)

Section 2 built a documented policy for the rivals' `batch` and left their OTHER first-order
knob pinned at a value **we** chose: `--num_steps 10`, hard-coded in both sweep drivers
(`eval/run_full_eval.sh`, `eval/colab/colab_rivals_l4.sh`), with HTS falling to the same 10 through
its argparse default. Nobody had checked what it cost them.

Measured on an RTX 4080 at each rival's granted batch:

| rival | instance | at our pinned 10 | at their own value | factor |
|---|---|---|---|---|
| Recurrent-CircuitSAT | s386, T=25, batch 100k | 18,081 u/s | steps=1 → **153,726** | **8.50x** |
| DEMOTIC | c432, batch 300k | 650,006 u/s | steps=1 → **1,698,654** | **2.61x** |
| HTS | b03-c25-rare, batch 30k | 183 u/s | steps=5 → 5 | 0.03x (their script's value is WORSE) |

The mechanism is in the counts, not the rates. **Recurrent's solution count is saturated at the
first step**: 2,198,776 unique at steps=1 against 2,134,454 at steps=10, for 14.3 s of runtime
against 118.0 s. Every step after the first is pure cost on our instances, because the batch —
which section 2 made large — already covers the reachable set. DEMOTIC is nearly the same
(272,212 → 299,867 unique for 2.9x the wall). HTS is the opposite: it finds 0 / 33 / 1,850 / 6,664
unique at steps 1 / 5 / 10 / 20, so on our CNFs the descent is doing real work and cutting it
cripples the tool.

### The rule

**Each rival runs at the `num_steps` its own artifacts specify.** Where a tool ships two values —
an argparse default in `src/pytorch/flags.py` and a different one in its demo script — take the
one that is better FOR THE RIVAL. That tie-break can only move a published ratio against us, and
it never reaches for a setting the authors do not themselves use. Resolved once, in one place
(`eval/rivals/rival_common.py::RIVAL_STEPS`), recorded per row as `steps` / `steps_source`, and
re-derived on the campaign's own card by `eval/rivals/steps_calibration.py`:

| rival | their argparse default | their script | **used** |
|---|---|---|---|
| DEMOTIC | 10 | `run_sampler_comb.sh`: **1** | 1 |
| Recurrent-CircuitSAT | 10 | `run_sampler_seq.sh`: **1** | 1 |
| HTS | **10** | `run_sampler_75.sh`: 5 | 10 |

We deliberately do NOT sweep it and keep the best, the way the batch ladder does. A batch is a
share of the machine and has to be granted; a step count is a tool setting with an author-chosen
value, and tuning it past what the authors publish invites the unanswerable question of why we
stopped at 20 rather than 100. HTS's optimum on our instances is ≥20 and we do not know where it
turns over — we run it at 10, which is its own default and NOT its best.

### What it costs us

The `demotic_native` row moves from **4.6x to ~1.1x on c432** in local re-measurement — at
steps=1 DEMOTIC finishes fast enough that the (since-removed, §2) batch up-probe also took it
from 1M to 3M rows, and the two effects compounded. The audit's projection for `recurrent_table` is 6.1x → **~0.75x, an
outright loss**. Those are projections until the L4 stage is re-run; the rival rows recorded
2026-08-26 and earlier are superseded by construction.

⚠️ **The general lesson is the one section 2 already taught and we did not generalise**: any knob
a harness pins for a tool it did not write is a number we chose on that tool's behalf, and it
must be either granted by a stated rule (the batch) or taken from the tool's own artifacts (the
steps). "It was the same for everyone" is not fairness when the tools are not the same.

---

## 7. CMSGen's sample count was FLOORED to the `--statsevery` cadence (2026-08-26)

The eval patch prints `c gpap-stats samples=N ...` every 1000 samples and the harness keeps the
last such line, so a CMSGen killed on its time budget reported the last MULTIPLE OF 1000 it had
passed. On the 2026-08-23 L4 campaign that is **174 of 357 rows, every one of them an exact
multiple of 1000, 81 of them recorded as `0`** — and a `cms_n_samples: 0` is scored as an outright
GPAP win. It cannot be chance; it is the flooring.

Re-measured on one of them, `seq-c25-random / b14 / abc:fast`, published as `ratio = inf` inside a
`wins 19/19`: at `--statsevery 1` under the same 62 s budget CMSGen delivers **941 samples**. GPAP
produces 38 models at 0.673 samples/s there, so the honest ratio is ~0.04-0.10x — **a 10-25x LOSS
published as an infinite win.** The bucket is a MIX, not a uniform correction: of ten timed-out
CNFs re-measured, four are genuine zeros (b14-c25 tseitin, b15-c25, bj08amba2g4f3, bobsynthor) and
four delivered 1-15 samples. Only a re-run gives the real numbers.

### The fix — CMSGen enforces its own deadline

`--maxtime N` (added to `scripts/cmsgen_gpap_eval.patch`): the sampling loop re-arms
`SATSolver::set_max_time` before every `solve()` call, so at the deadline the solver returns
`l_Undef`, the loop falls out through its NORMAL exit, and the post-loop stats line is printed
with an exact count and a new `final=1` marker. The harness passes `--maxtime` equal to the
sampling budget (the same quantity GPAP's `--time-limit` bounds) and keeps the subprocess kill
only as a backstop for the phases the deadline does not cover, with a grace margin for parsing.

Verified locally on the three cases that matter:

| case | before | after |
|---|---|---|
| deadline mid-run (27.sk_3_32, 2 s) | 218,000 (floored) | **218,725**, `final=1`, exits 0 |
| a single `solve()` longer than the budget (b14-c25-rare, 1.05M clauses, 20 s) | no stats line at all | **samples=0 `final=1`**, plus 451.9M props / 1.32M decisions / 25,784 conflicts |
| completed run | exact | exact |

The second row is the bigger win and was not what we set out to fix: on a cell where CMSGen never
reaches its first cadence line, the OLD harness recorded no work counters either, so those
instances silently vanished from Table 3's propagation identity. They now report CMSGen's
implication rate even when it delivers nothing.

`cms_count_exact` is recorded per row, and `verify_artifact.py` FAILS a stage that publishes a
floored zero. It fails the 2026-08-23 campaign on 12 of 13 stages, which is the correct verdict.

### While in there: both rate columns now charge both sides the same way

`cms_sps` used to be samples per SUBPROCESS wall — CMSGen's parse inside it — and was divided into
GPAP's steady-state rate, which excludes parse. The patch's stats line carries CMSGen's own
sampling wall, so the pairing is now steady-state/steady-state (`ratio`) and
end-to-end/end-to-end (`ratio_e2e`). Both move AGAINST us relative to the old definition, which is
the direction an unexamined asymmetry always turns out to run.

---

## 8. The published cell was chosen by `argmax(ratio)` (2026-08-26)

`paper_tables.select_cells` kept, per instance, the encoder with the highest `ratio` — making the
published statistic its own argmax key, so the RIVAL's number voted on which of our runs was
reported. Measured over the campaign:

| family | argmax ratio (published) | argmax `gpap_sps` (rival-blind) |
|---|---|---|
| seq-c25-rare | 84.3x | 66.7x |
| seq-c25-random | 333.8x | 250.8x |
| seq-c50-rare | 42.1x | 34.0x |
| seq-c50-random | 305.7x | 232.1x |
| all families pooled | 161.9x | 161.1x |

A wash in aggregate, **1.24-1.33x on the flagship sequential families**. The mechanism: ABC helps
CMSGen more than it helps us on some instances (seq-c25-rare b04: GPAP x1.2, CMSGen x4.0), so
maximising the quotient awards a shared input-preprocessing gain to one side only.

**Fixed by removing the selection, not by re-keying it**: Table 1 and Table 3 now emit ONE ROW PER
(family, encoder). Both pipelines a user could ship are shown; nothing chooses between them. That
also stops hiding the encoder A/B, which is not monotone — ABC unlocks `b14` at a rare pin and
collapses `b13`'s. `eval/eval_plots.py`'s summary, which had the ratio-keyed rule while its own
figure used `gpap_sps`, is unified on `gpap_sps`; `slide_tables.py` and `soft_price_tables.py`
import the paper's function so the deck cannot drift.

## 9. `wins` and `n` were counted over a different row set than the geomean

The degeneracy screens (vacuous pins, zero-search cells) were applied only when building the
geomean. `inst` and `wins` were computed BEFORE them, so the invariant printed under Table 1 —
`inst` = geomean instances + CMSGen=0 + GPAP=0 + both=0 — was **false on 6 of 7 families**
(seq-c25-rare 13+1+0+1 = 15 against `inst` 19, beside a printed `wins 17/18`). A referee can check
that with a calculator.

Fixed by screening FIRST: a degenerate cell now leaves through its own column (`no-search`,
`vacuous`) and takes no part in the geomean, the zero ledger or the win count. `n_gm` is printed.
The invariant is `inst = n_gm + CMSGen=0 + GPAP=0 + both=0 + no-search + vacuous`, and
`tests/test_paper_tables.py::test_every_instance_is_counted_exactly_once` pins it.

There was a second hole in the old order: `zero_search` was evaluated only over cells where BOTH
samplers emitted, so a zero-search cell on which CMSGen scored nothing (EPFL `ctrl` under ABC:
succ 1.0 at 240M samples/s) was published as an outright win rather than screened out.

---

## 11. The rival was declared incapable on OUR stopwatch (2026-09-03)

`rivals_recurrent.py` finds the rival's usable `--start_point` by probing upward: their
`backward()` raises while the pinned output has no gradient path to any input (Moore register
latency), so the harness walks the depth until a 1-step run survives. Each probe was capped at
`PROBE_TIMEOUT = 60.0` — against the measured run's `DEFAULT_TIMEOUT = 300.0`, with a comment on
the same line reading "(a 1-step run; same budget)". It was not the same budget.

Worse, all three ways the search can fail were reported as one string, `"no gradient path at any
start_point<=10"` — a statement about THE RIVAL — including the case where the string was
produced by our own cap firing.

**Measured on b14, T=25 (RTX 4080):**

| start_point | 60 s cap (what the campaign ran) | 900 s cap |
|---|---|---|
| 0 | no gradient path (3 s) | no gradient path (3 s) |
| 1 | **TIMEOUT at 60 s** → reported "no gradient path at any start_point<=10" | **RUNS: 743 s wall, 636 s by their own clock, 4,657 unique solutions** |

So the rival is not incapable on b14 — it is *slow*. b14 sits in Recurrent's own Table II.

⚠️ **The ratio must come from the GRANTED batch, never from the probe.** The probe runs at
`batch=256` by construction (it asks "does backward() work here?", not "how fast is it"), and
§2 of this document is the rule that a rival's batch IS its score. Re-measured through
`run_recurrent_ladder` at the batch `fit_batch` grants it:

| | batch | wall | their `runtime_s` | unique | **unique/s** |
|---|---|---|---|---|---|
| probe (NOT a measurement) | 256 | 743 s | 636 s | 4,657 | 7.3 |
| **granted** (10,000 refused → 3,000) | 3,000 | 1,025 s | 655 s | 55,458 | **84.7** |

The grant is worth **11.6x** to them. Against GPAP's 3,420 u/s on the same circuit and pin that
makes b14 a **~40x** cell — a real comparison the campaign recorded as a rival failure, not the
~470x a probe-derived ratio would have claimed. Quoting the probe would have understated the
rival by an order of magnitude, which is the same error §2 catalogues, committed while fixing a
different one.

`b17` is the opposite case and the reason this needs measuring rather than assuming. Walked all
the way to `start_point=12` at a 1200 s cap: **0 and 1 fail in 2 s and 6 s** with the genuine
`element 0 of tensors does not require grad`, and **2 through 12 every one time out at 1200 s**
(1132-1134 s each — the model build alone). So b17 is out of the rival's reach at T=25 at any
depth and at four times its granted budget: a real limit, not our cap. The distinction is the
whole point — b14 and b17 produced the identical error string and only one of them was true.

That walk also validates the rule that a **timeout stops the search**: no start_point above the
first timing-out one was reachable, so continuing cost 3.5 h of GPU and changed nothing. Under
the shipped rule b17 costs `2 s + 6 s + one 300 s timeout` and reports the accurate reason.

### The rule (settled 2026-09-03)

- **The start_point search is the rival's INITIALIZATION and gets ONE budget: the run's timeout
  (`--timeout`, default 300 s).** Each probe runs with whatever is LEFT of it, so neither a single
  probe nor the sum of them can exceed 300 s. The user's call: "1 config with max of 300s if it
  takes to initialize the sampling" — generous beside the 60 s the rival is measured under. (An
  intermediate version gave the whole search 4x the run timeout = 1,200 s; that spent 4 x 1,134 s
  on b17 to learn it cannot start.)
- **Then ONE measured run** at the granted batch, `num_steps=1` (their script), the probed
  `start_point` — `run_at_batch`, §2.
- `probe_start_point` returns `(start_point, reason)` and the three failures are **named
  separately**: a gradient-path refusal at every depth reached (a finding about the rival); a
  probe that did not complete within the initialization budget; and a budget used up by the
  probes before one completed. The last two read `did not initialize: ... {budget}s
  initialization budget` — the rival's own outcome under a stated, uniform budget. `paper_tables`
  prints the failed count in the caveat; they are excluded from the geomean, never zeros.
- A timing-out probe stops the walk: a larger `start_point` pays the same model build, so it
  cannot be faster (b17 above: 2-12 all 1,132-1,134 s).

### What the 300 s rule costs — stated, not hidden

**300 s does not buy b14 a cell.** Its granted-batch run needs 1,025 s of subprocess wall (655 s
of it their own clock, the rest model build, which their metric excludes and ours does not charge
them for); its probe alone is 743 s. Under the rule b14 is a "did not initialize" error row, and so
are b17 (every start_point 2-12 exceeds 1,134 s of model build) and, by size, b15/b19/b22/s35932
most likely. The campaign reports "N of 16 Recurrent runs did not initialize within 300 s; b14
completes in 1,025 s when given it" beside the geomean over the rest. The cap is OURS and is the
same for every instance; what is NOT defensible is a cap chosen after seeing which one gives the
better ratio.

## 12. What pinning GPAP's config would cost — MEASURED, not adopted (2026-09-03)

§10 lists an asymmetry: every rival runs one fixed configuration, and so does CMSGen, while
GPAP's `CIRCUIT_PORTFOLIO` is a per-instance **bandit** that measures configs against each other
inside the timed wall and reports the winner's rate. No table says so.

This is the price of closing it, so the decision can be made on a number rather than a guess.
**Not adopted** — the default stays the shipped bandit, because changing it moves every recorded
rival ratio; `--portfolio b8` (`rivals_recurrent.GPAP_PORTFOLIO`, shared by the DEMOTIC harness)
is the knob. MEASURED on the rival path — T=25, their pins, Tseitin, seed 5, 20 s budget, same
circuit and same CNF:

| instance | bandit samples/s | pinned `b8` | pinned `b1` | bandit picked |
|---|---:|---:|---:|---|
| b13 | 560,478 | 534,674 (−4.6%) | 422,483 (−24.6%) | B=8 |
| b12 | 124,570 | 119,017 (−4.5%) | 107,594 (−13.6%) | B=1 |
| b04 | 100,227 | 91,479 (−8.7%) | 68,910 (−31.2%) | B=8 |
| b14 | 3,603 | 3,411 (−5.3%) | 3,469 (−3.7%) | B=8 |

So the pin would cost **4.5-8.7%** and buy a claim the table can actually make. `b8` rather than
`b1` because the bandit selects it on 3 of 4, it is the library default everywhere
(`decisions_per_round=8`), and it is the uniformity-best of the three (Barbarik TV 0.040, the
instrument floor — UNIFORMITY_FINDINGS.md).

⚠️ Note the bandit beats BOTH pinned configs on every instance above, which is the point: its
advantage is real, it is not available to the tools it is being compared against, and it was
being reported without a label. The row now carries `portfolio` and `adaptive` so the claim is
in the data, not in a caption.

## 13. Table 2's GPAP arm encodes with `abc:fast`, like Table 1 (settled 2026-09-03)

**Status: SETTLED.** `rivals_recurrent.run_gpap` (shared by the DEMOTIC harness) builds the GPAP
arm's CNF with `abc_backend.build_cnf_abc(..., opt="fast")` — the encoding every other table in
the paper runs — and records it per row as `gpap.encoder`. There is no Tseitin fallback: without
an `abc` binary the arm is an error row. Rows recorded before this date encoded with the native
Tseitin and are superseded.

### What the Tseitin arm was costing — MEASURED (T=25, their target pins, seed 5, 20 s, bandit)

| instance | Tseitin nv | abc:fast nv | Tseitin samples/s | abc:fast samples/s | ratio |
|---|---:|---:|---:|---:|---:|
| b13 | 12,926 | 913 | 556,438 | 1,220,983 | **2.19x** |
| b04 | 28,626 | 7,809 | 94,418 | 118,194 | **1.25x** |
| b12 | 42,051 | 12,287 | 115,915 | 120,846 | **1.04x** |
| b14 | 413,451 | 41,937 | 3,564 | 17,121 | **4.80x** |

Table 2 was running GPAP **1.04-4.8x slower than Table 1 did**, on the same circuits and the
same pins, and it was unlabelled: the harness called `bench.build_cnf` with no encoder parameter.

### Why it was believed to "have to" be Tseitin, and why that was wrong

The argument was: the rivals consume `.bench` and unroll it themselves, so both sides must be
handed the same circuit, and `abc:fast`'s win lives in a pass that runs *after* unrolling. The
premise is right and the conclusion does not follow. Fairness is **same circuit, same task**:
both sides receive the identical `.bench` and the identical pins, and each compiles it into its
own representation — the rival into its differentiable model (with whatever its own compiler
does), we into CNF. `abc:fast` is a function-preserving encoding of the same unrolled circuit
with the same projected solution space (gated by `tests/test_abc_encoder.py`, which enumerates
projections under both encoders), and its encode wall is `encode_s`, outside every rate exactly
as it is in Table 1. The `--circuit-variants abc` axis remains a different thing — it hands
BOTH sides an ABC-simplified *sequential* circuit — and the campaign runs `raw` only.

### The oversize guard is a Tseitin guard

`rival_common.oversize_reason` models the native Tseitin build (its host-RAM fit is Tseitin's
peak RSS), so it now takes `encoder=` and passes everything under `abc:fast`: every seq
instance in the suite — b19 and s35932 at T=50 included — built and sampled on the 13 GB L4 VM
in the CMSGen stage of the 2026-09-02 campaign (b18: 19,859 vars at T=25 against 2.86M
unrolled gates). A Tseitin-calibrated refusal would have omitted instances the campaign had
already run.

### The coverage run is gone from the rival arm

The 61 GB OOM (§2 of this note's first draft, `pipeline.cov_batch`) was the coverage run:
`n_cov` collects models and reconstructs them on the host at `rows x nv x 72 B`, and `abc:fast`
alone would only have moved the failure (b17 21.9 GB at abc:fast nv, 194.5 GB at Tseitin nv).
`run_gpap` no longer runs it. The distinct fraction behind `unique_per_s_est` is measured on the
timed run's OWN kept output rows — `out_cap=UNIQUE_ROWS` (1,000) GPU-projected support-bit rows
of the winning config, exactly what `gpap sample --out` keeps, at rows x |support| bits of host
memory — with the projection wall subtracted from the timed wall (`out_project_s`;
`check_emission_symmetry.py` re-derives the subtraction). `unique_est_status` is `ok` |
`no-sample-rows`; the pre-2026-09-03 `no-coverage-sample` is still read by
`paper_tables.unique_unmeasured`. `cov_batch` stays for the opt-in `--n-cov` path.

## 14. The converged protocol: three tools on one constraint, and Table 1 read off Table 2's rows (2026-09-03)

**What was wrong.** The CMSGen comparison (Table 1) and the rival comparison (Table 2) came from
two different instance sets. Table 1 was a separate `gpap bench` sweep that pinned the circuits
OUR way — one output on ISCAS-85 (where the DEMOTIC-rare arm pins all of them), the outputs
directly rather than through a miter on the unrolls, plus `random` targets, T=50 and three
families (EPFL, HWMCC, `sk_`) no rival ever ran — and it never saw a rival's own pins at all. A
reader could not line a Table 1 row up against a Table 2 row, and the headline families were
ones we chose.

**The rule now.** The instance set is the rivals' own published sets, under their pins and our
rare pin (§11, §13 unchanged), and EVERY cell is run by three tools on one constraint: the rival
on its circuit, GPAP on the `abc:fast` CNF of that circuit and those pins, CMSGen on that very
CNF file. Mechanically: the harness encodes once (`rivals_recurrent.build_arm_cnf`), the GPAP
arm samples that path (`run_gpap(cnf=...)`), then `rival_common.cmsgen_arm` runs the patched
binary on it (`--nowrite`, `--maxtime` = the GPAP budget, same `n`) through the SAME runner
`gpap bench` uses (`gpap.benchmark.cmsgen_run`), and the row records the bench driver's own
`cms_*` keys plus the host `cpu`/`gpu` (a CMSGen ratio is host-dependent, docs/BENCHMARKING.md).
`paper_tables.rival_cms_family` then reads Table 1 off the rival stages — Table 2's screens
first (vacuous / no-search / tiny), Table 1's ledger after (finite / CMSGen=0 / GPAP=0 / both=0,
absent-is-not-zero, crashed-is-not-zero), the rival paper's own subset beside the whole set —
and `rival_prop_rates` reads Table 3 off the same cells. `verify_artifact.py --rivals` applies
the bench-stage gates to those rows (mixed provenance, crashed subprocess, floored zero,
unpatched binary, unsound GPAP arm, two engine builds in one stage), and
`check_emission_symmetry` audits their `cms_mode` like any other.

**What it costs and what it removes.** +60 s of CMSGen per row (~80 rows). It removes the
whole family matrix from the campaign: `CMS_FAMILIES` stages are supplementary now, printed by
`paper_tables` only when their JSONs exist, never the headline. The rows of every earlier
campaign are superseded for Table 1 (different instances) and unchanged in meaning for Table 2.
Gated by `tests/test_rival_targets.py` (the arm is the bench runner, emission-free, on the very
CNF the GPAP arm sampled, in both harnesses) and `tests/test_paper_tables.py` (the ledger
identity on rival rows; the verifier's gates with no bench directory at all).

**Review of the protocol text (2026-09-03, second pass) closed three more gaps in the ROWS
rather than in the wording.** (1) The rival-facing ratio was described as "GPAP's rate
includes construction"; it is the winning config's STEADY-STATE rate (`samples_per_s`), which
excludes parsing, construction, ELS, mining and warm-up exactly as the rival's `model_runtime`
excludes its model build. Table 2 now prints the e2e ratio (`gm_e2e`: GPAP's `sps_e2e`, the
same distinct fraction, the rival's unchanged timer) beside the steady-state one, and the
caption says which is which. (2) Constraint alignment was asserted per row by the DEMOTIC
harness only: `native_pin_check` moved to `rival_common` and the Recurrent harness runs it in
table mode, and `rivals_recurrent.miter_self_check` (P[MITER=1] = p_accept, the check
`write_unrolled_miter` always ran) flags the rare rows — `pin_check_ok` on every row of every
stage, so the `misaligned` bucket can fire on the headline sequential rows. (3) The
encoder-independent check (`gpap sample --sim-check`) never ran on the campaign path: the arm
now simulates its 1,000 kept rows through the ORIGINAL circuit (`sim_check` ok | BAD | off,
`sim_check_rows`, `sim_check_bad`); `verify_artifact` FAILS a BAD row and counts the ok ones.
Gated by tests/test_rival_targets.py (an inverted miter is caught; both mains call the pin
check; the simulate check passes a satisfying row and fails the all-zero one) and
tests/test_paper_tables.py (the e2e column; the verifier's sim-check gate).

**Third pass: the SOLUTION space, not the input hypercube.** `tiny_space` read `2^|support|`,
so `c1355` under the rare pin (41 input bits, 2^41 points) passed the screen while its kept
rows were 0.56 distinct — ~700 solutions, which both tools enumerate — and the 1,000-row window
let `unique_per_s_est` read ~10^5x the whole-run bound (800k models delivered from ~700
solutions). Now `rival_common.space_from_distinct` turns a repeat rate into a size,
`paper_tables.solution_space_estimate` takes the minimum of the hypercube, p_accept (or its
upper bound) x hypercube and that size, `tiny_space` screens on it in BOTH tables, and
`run_gpap` caps unique/s at |S| per timed wall (`unique_capped`, `space_est` per row — it can
only lower ours). The DEMOTIC-rare miter self-check no longer raises: a failure is a flagged
row like every other alignment failure. Gated by tests in both test files.

## 10. Still open after §6-§9 — things a referee can still ask

Carried over from the 2026-08-26 audit; none is a correctness bug, all are statements the text
owes the reader. Kept here because a scratch note does not survive the session that wrote it.

**Must state in the text (no re-measurement needed):**

- **Table 2 has no `e2e` column** although every rival row carries `sps_e2e`. It divides by GPAP's
  steady-state rate only, while Table 1 now prints both. Measured: DEMOTIC native 4.6x -> 3.3x
  e2e, Recurrent table 6.1x -> 3.7x, HTS 603x -> 404x. It cuts both ways — the rivals' own metric
  excludes their ~30 s model build too — so say WHICH pair of clocks Table 2 uses.
- **The whole 357-cell campaign is n=1** (`seeds: null, seed: 5`). Table 6 is a SAME-SEED replay,
  so it bounds determinism, not seed variance. Rival rows are one rival run against GPAP's
  median-of-3. Say n=1, or add seeds to the headline families. The repo's own protocol note puts
  draw noise at +-40% on low-success instances.
- **Table 5's "0.480x soft price" is over 14 instances, not the stated 15**, and is carried by
  two; the median is 0.979. b11 goes 309 -> 1 samples/s — a capability loss, while the table
  prints "capability losses: 0".
- **Table 2 applies no `p_accept` screen.** Recurrent's `b13` "their target table" cell has
  `p_accept = 1.0` — a tautological pin — and its 12.9x sits above the row geomean (5.27x without
  it). The caption says "p_accept is 0.03-0.5 there"; the real span is 0.0218 to 1.0.
- **GPAP runs an adaptive per-instance config bandit; CMSGen and all three rivals run one fixed
  configuration.** The headline is the bandit winner's rate. No table says so. STILL OPEN — §12
  prices it (4.5-8.7% to pin `b8`) and `--portfolio` is the knob, but the default is unchanged;
  every row now records `portfolio`/`adaptive` so at least the table can state which it was.
- **`paper_tables.py` says "Nothing here is typed by hand" and then hardcodes numbers** in three
  places ("up to 1.9x" is 1.42x/1.58x on this campaign).
- **Table 1's `n` is instances; Table 2's `cells` is instance x variant** (DEMOTIC native's
  20-cell geomean counts each of 10 instances twice). Label the two differently.
- **`pre%` under-explains itself**: 1.07-1.34x of the steady-state-vs-e2e gap is the BANDIT's
  exploration inside the timed wall, not preparation (over 329 rows the reported rate is a median
  1.08x / mean 1.18x / p90 1.34x above what the timed run delivered).

**Nice to have:**

- `verify_artifact.py` accepts `--recheck` and never reads it; the recheck, soft and kernel-A/B
  datasets ship with no verdict. Nothing reads `pin_check_ok` or `equiv_hits` although the Table 2
  caption asserts both are verified per row. No rival JSON records a GPU, so Table 2's "same L4"
  is inferable (from `vram_budget_bytes`) but not verifiable.
- `bench_equiv_check` compares only the accept-word conjunction, so it passes vacuously when no
  stimulus hits the target (12 of 19 abc rows have `equiv_hits=[0,0,32768]`). Verified separately
  that no wrong circuit got through; still, compare all output words.
- HTS caches its generated model at `models/{basename}.json` and the harness clears only
  `src/results/`, so a LOCAL re-run under the same filename silently measures the old formula.
  The recorded campaigns are safe (fresh VMs). One line: unlink the model JSON too.
- `ARTIFACT.md` is still a self-declared placeholder: no claim->file map, no expected verifier
  verdict (`docs/EVAL_RUNBOOK.md` §8 now points at the artifact tooling; the map itself is still
  owed).

**Checked and clean — do not re-litigate:** emission symmetry; the batch grant itself; zero-ledger
symmetry (the buckets are exhaustive); the propagation identity (re-derived from raw rows, closes
within 1%); soundness (357 cells, 0 unsound); the `unique_per_s_est` estimator window (bias
bounded at 7e-4 on the worst published cell, and it runs AGAINST us); DEMOTIC native pin alignment
(`pin_check_ok` true on all 20); ABC variants bit-exact on every output at the measured depth;
seed unification at 5; engine provenance in every row; the GPAP-arm cache key.

Two audit claims were REFUTED and should not be revived: "the unique/s ratio divides two different
estimators" (bounded at 7e-4, and in the wrong direction) and "`ratio=inf` wins the argmax so the
CMSGen=0 win column is populated by encoder choice" (the mechanism is real, the causation is not —
it was a wording problem, and §8 removed the argmax anyway).

## 15. The campaign under the protocol: Colab L4, 2026-09-03/04 — and what the rows taught

`$GPAP_RESULTS/l4_protocol_20260903` (its README is the record). 80 cells, 0 errors, 0 unsound,
`verify_artifact --rivals` 0 failures, emission audit PASS; 6 VM lifetimes, no row lost. Run on
the L4 rather than the laptop because the laptop's 4080 sat at a 40 W power cap (default 80 W,
~1.1 GHz under load, un-liftable from WSL) — check `nvidia-smi -q -d POWER,PERFORMANCE` before
any local measurement.

Headline (screens applied): GPAP/CMSGen 1,262x on DEMOTIC's set under its pins (10), 244x under
our rare pin (6, five cells tiny); 1,150x on Recurrent's set under its table (16), 75x under our
rare pin (17, 17/19 wins); GPAP/rival 4.7x vs DEMOTIC on its pins (9/10), 6.2x vs Recurrent on
its table (12/12), 338x vs Recurrent under our rare pin (15/15); engine 1.71e9 impl/s, 548x
CMSGen's counters. Losses by name: `c499` to DEMOTIC (0.50x); `b11`, `b18` to CMSGen under the
rare pin (the rejection collapse, success 4e-6 / 7e-4); `b01` under Recurrent's target is a
GPAP zero against its depth-pooled count (0 at T=25 for everyone). Recurrent did not initialize
within 300 s on 10/11 of 29 — every large circuit, so its geomeans are over its easier half.

What the rows exposed, fixed in `paper_tables.py` and gated by tests:
- **Screens before buckets, in Table 2 as in Table 1.** `b19` is vacuous at T=25 (p_accept
  1.0) AND Recurrent fails on it; with the error bucket first, Table 1 said vacuous and Table 2
  said rival failure — one stage, two ledgers. The vacuous list is per PIN, not per circuit:
  `b13` is a tautology under Recurrent's own target and not under our rare pin.
- **A run with zero models is a GPAP zero, not "unmeasured".** `unique_est_status`
  "no-sample-rows" with `n_models` 0 was booked as no-verdict (`b01`).
- **The Recurrent egress bound is per card.** 0.7–1.4% of `model_runtime` on the 4080 became
  0.1–15% (30% pessimistic on s208.1) on the L4; the typed number left the source files, the
  audit's section 4 is the figure to quote. Uncorrected either way, in the rival's favour.
- **Two cells the screens do not catch, named instead**: `c432` under the rare pin propagates
  to 4 effective variables (3 decisions and 1 implication per model, 1.1e8 unique/s vs
  DEMOTIC's 1,201 — the only DEMOTIC-rare cell with a ratio), and one repeat in 1,000 kept rows
  implies ~5e5 solutions (`c1908` its pins, `c432` rare, one seed each) — weak evidence that the
  3-seed median absorbed. Widening the free screen or flooring the space estimate at a rival's
  observed unique count are protocol decisions, not taken here.


## §16 — The referee pass on the l4_protocol_20260903 campaign (2026-09-04)

Five independent adversarial reviews of the completed campaign, each re-deriving numbers from the
raw rows rather than reading the prose. Every published figure in `RESULTS.md` reproduced exactly
— the generator is honest — and the defects were in the harness, in two metric definitions, and
in what the LaTeX emitter printed. **The campaign is superseded**; what follows is what changed
and why, so the re-run can be read against it.

**Four defects that moved published numbers, all in our favour.** That direction is the finding:
a result that looks good does not get investigated.

1. **DEMOTIC was under-granted on 4 of 11 ISCAS-85 cells.** `run_at_batch` started at our VRAM
   model's rung and only stepped *down*; the one-rung-up probe that had been masking the model's
   error was removed on 2026-09-03, the campaign's own launch day, for being a best-of-two. The
   model was measured at `num_steps=10` and we run 1, so it predicts low — and a gradient
   sampler's unique/s is near-proportional to its batch. Our own L4 calibration of the previous
   day names the four cells (c880 1.53x, c2670 2.12x, c3540 1.85x, c7552 3.87x). Rescored at the
   largest rung that runs, **DEMOTIC-native 4.73x → 2.99x**.
   FIX: walk the ladder DOWN from the top; reject a rung only on feasibility (OOM, driver
   refusal, or not finishing inside the wall budget), never on throughput. Still one measured
   configuration, so the best-of-N objection does not apply. `probe_budget_s` bounds the search.
2. **`s208.1` was our bug, published as one of the rival's 11 failures.** Their `run.py` strips
   dots from the dataset name when it writes its results CSV (`sat_s2081_1_...`); we globbed the
   key verbatim. It ran at all six rungs and every result was discarded. It is the ONLY one of
   the 11 with `rungs=6, "no results csv"` — the other ten are genuine `start_point` failures.
   `s15850.1` and `s38584.1` are the same shape.
3. **`c1355` ate the session's cold start.** The first DEMOTIC run of a process is ~3.3x slow
   (measured, and the calibration probes have always discarded a warm-up run); the campaign
   harness fired none, so the alphabetically-first cell was published at 0.25x of its warm rate.
   FIX: one discarded run per (tool, process) inside `run_at_batch` — per PROCESS, because the
   supervisor resumes in a fresh interpreter after every VM reclaim.
4. **`s35932` is not their circuit.** Their Table II's 8th ISCAS-89 row reads 12 PI / 278 PO /
   19,253 gates / 1,452 DFF, which is `s38584` exactly; we read the row's name as "s35844" and
   substituted `s35932` on the weaker evidence that it is the only `s359xx` branch in their code.
   Their code covers more circuits than their paper reports. `s35932` also carried the sequential
   headline: dropping it moves the full-set geomean **1,150.2x → 759.0x** and the paper-set
   **1,704.5x → 821.3x**, and its ratio is 586,616x because CMSGen produced ONE sample in 60 s.
   FIX: `s38584` is in `PAPER_INSTANCES` and committed; `s35932` stays as a circuit we add.

**Two metric definitions that were the wrong shape.**

* **Unique/s was an estimate where the rivals measure.** We took the distinct fraction of a
  1,000-row window and applied it to runs delivering up to 1,048,576 models; both rivals
  deduplicate their whole output exactly, off their own clock (`torch.unique`, after the timer).
  The "bias bounded at 7e-4" claim had no derivation in the repo and bounded the collision rate
  *inside the window*, not the run's distinct fraction. FIX: `exact_distinct` over every
  delivered row (worst recorded cell: 262 MB gathered, 32 MB packed, both off the timed clock).
  The unique/s CAP goes with it — with an exact dedup there is nothing to correct, and the cap
  was in any case discarded by the median that followed it (on both cells where it fired, the
  capped seed was the slowest).
* **A published row was a composite no execution produced.** `sps`, `n_models`, `success` and the
  counters were each medianed independently across seeds: **138 field/cell combinations** took
  their value from a different seed than the row's own rate, so Table 3 divided one seed's
  implications by another seed's models. FIX: `median_run` picks the run whose rate is the median
  and publishes all of ITS fields; the safety verdicts (`sound_check`, `sim_check`) still fold
  over every seed, and `sps_min`/`sps_max` record the spread — p90 of max/min was 1.51 on this
  campaign and six published wins sat inside their own seed spread.

**Three screens, and an artifact that contradicted its own caption.**

* `zero_search` was a knife-edge at exactly zero counters. `c432` under the rare pin leaves 4
  effective variables and delivers a model per 1.09 implications — against a next-lowest
  surviving cell of 44.1 — and was the ENTIRE DEMOTIC-rare geomean at 89,786x. The screen is now
  continuous (fewer than two implications per delivered model), with the threshold placed inside
  that two-order gap rather than fitted. DEMOTIC-rare then honestly has no geomean at all:
  5 rival zeros, 5 tiny, 1 free.
* A cell **proved UNSAT at its depth** gets its own column. `b01` under Recurrent's own target is
  unreachable at exactly T=25 (their sampler runs to 50 and pools every depth), and we booked a
  GPAP=0 loss against a solution that does not exist.
* The **LaTeX emitter dropped `tiny` and `vacuous`** from Table 1 and `n`, `tiny` and `vacuous`
  from Table 2, so neither table's columns summed to the instance count its caption promised, and
  `89,786.0x` printed with nothing saying it was one cell. The Markdown was correct throughout.
  Gated now by a test that parses the emitted `.tex`.
* The inflation caveat reported `max()` over stages — the one row where screening was
  conservative (1.42x down) — while hiding that the screens RAISE the other three headlines by
  1.19–1.24x. Now reported per stage, in both directions.

**Two protocol choices that were ours, not theirs, and are now the rival's.**

* **Depth.** T=25 was described as "the rival's own default depth". Their run script says
  `--num_clock_cycles 50` and their paper sets η=50. Both depths are now run and reported.
* **The circuit.** We sampled an ABC-optimized CNF while handing the rivals the raw netlist and
  disclosing it. The capability to hand them the optimized circuit existed and was used before
  (worth geomean 1.34x to DEMOTIC). Both variants are now run and the table scores, per instance,
  whichever is **better for the rival** — the same tie-break `num_steps` uses. `--lr` joins
  `--num_steps` under that rule: both rivals' run scripts say 15, Recurrent's paper text says 50,
  so the tie-break is re-derived on the campaign's card (`steps_calibration.py --axis lr`).

**Left standing, verified by more than one reviewer**: soundness (two independent checks, 80/80,
no verdict over an empty set), emission symmetry with Recurrent's in-timer copy quantified and
left uncorrected in its favour, absent-is-not-zero, `cms_count_exact` on 80/80 with no floored
zeros, one engine build id across 6 VM rebuilds, `pin_check_ok` 80/80, and the 60 s vs 300 s
budget asymmetry (no rival run timed out; max observed wall 259 s).

**Still open, deferred by the user**: the uniformity claim reports Barbarik pass counts only, and
our own exact-oracle instrument reads the shipped configuration 1.31x worse than CMSGen on TV.
Nothing false is published, and the data exists; converge on it before submission.

### §16b — Second referee pass, on the fixed tree (2026-09-04)

Five more reviewers, told exactly what §16 changed and asked to verify the fixes rather than
re-report them. The fixes held. What they found was a further layer, mostly of the same kind —
a rule stated in one place and not carried through to another.

* **`s38584` had no pin transcription.** It was added to `PAPER_INSTANCES` and committed as a
  benchmark, but `rivals_recurrent.TARGETS` had no entry, and table mode drops an untranscribed
  name with `continue`. The re-run would have silently skipped the very circuit §16 added — the
  same silent skip that previously cost `b15` and `s208.1`. The existing transcription test could
  not see it, because it only checks circuits with a NAMED branch in the rival's source and
  `s38584` takes their catch-all. Now transcribed, and gated by "every paper-set instance is
  scoreable".
* **The GPAP-arm cache key stopped one directory short, again.** §16 added `rival_common.py` to
  it; the harness files where `run_gpap` actually lives, the circuit encoders, the miner and the
  Python K-sizing were all still outside it, as was the `abc` BINARY whose output the arm samples.
  A logic fix in `run_gpap` changed no key at all. All now hashed.
* **The field-wise median came back in a derived column.** `_gpap_e2e` took its own median across
  seeds while every other field came from the picked run, so Table 1's e2e ratio and Table 2's
  e2e column mixed two seed selections inside one quotient. It now reads the picked run's own
  `sps_e2e`.
* **The LaTeX ledger was fixed and the MARKDOWN one was not** — `unsat` was missing from both
  Markdown tables, and Table 2's `n` printed the WIN denominator under the header Table 1 uses
  for the geomean's count, so its columns summed to 30 against 29 cells. Both now sum.
* **The both-directions screen caveat was gated on the wrong thing** — on whether a stage had a
  vacuous cell, which silently dropped both ISCAS-85 stages, the two the screens HELP. It is now
  gated on whether the screens moved the geomean at all, and reports all four stages.
* **The generated methodology paragraph still described the superseded harness** ("a timeout
  recorded as the result", "a FIXED 1,000 ... rows"). It is regenerated into the paper, so a
  refreshed table would have shipped a methods section describing a harness that no longer
  exists. Fixed and gated by a drift test.
* **Evidence the tables could not see.** `probe_truncated` (the ladder fell back to the modelled
  rung) and a new `budget_bound` (every rung timed out, so the cell is bounded by OUR wall rather
  than the rival's capability) were buried in `batch_sweep` with no consumer; both are now
  row-level and the gate counts them. The gate also warns when the proved-UNSAT screen never ran
  on a campaign — "not asked" is not the same as zero — and when a distinct fraction was taken
  over a prefix of the delivered rows.
* **Two claims of ours were wrong.** The miter control in `protocol.tex` said GPAP's rate moves
  "by at most 1%" on the tautology cells; `b05` moves 4.3x, where the adaptive portfolio switches
  configuration on a formula propagation has already solved. And `docs/paper/README.md`'s own
  "never quote a number from this file" paragraph went stale within the hour, on the next
  regeneration; it now carries no numbers at all.
* **The failure regime is now a boundary, generated from the rows**: every loss to CMSGen is a
  cell where amplification (success / p_accept) is at most 1.94x AND success is at most 6.9e-4,
  with zero winning cells meeting both conditions. Cells whose baseline delivered fewer than 30
  samples in its whole budget are named separately, so a ratio like `s35932`'s — one CMSGen
  sample in 60 s — reads as "CMSGen could not sample this" rather than as a measured factor.

**The three open items were decided (2026-09-04) and are now protocol, not gaps.**

* **One rival seed, its own default.** `RIVAL_SEED = 0` is passed explicitly on both rivals'
  command lines and recorded per row with its source. The rival is measured in exactly one
  configuration; the asymmetry against GPAP's median-of-three is stated in the protocol rather
  than closed, because what the tables need from the rival is one honest measurement at its own
  settings, not a narrower interval. (Three seeds would have cost ~1.7 h per campaign, and until
  now would have measured nothing new at all: without `--seed` on the command line every "repeat"
  re-ran the same initialization.)
* **The 300 s wall stays.** It is ours, not theirs, and it is now falsifiable per cell rather
  than by inference: a cell that exhausted the budget at every batch rung is flagged
  `budget_bound`, counted apart from a capability failure, and warned about by the gate. The
  T=50 stages will therefore say for themselves whether the budget binds at the deeper unroll.
* **`python-sat` is pinned** (`>=0.1.8.dev0,<1.10`) in `pyproject.toml` and `rivals_setup.sh`,
  because it ADJUDICATES the proved-UNSAT screen: which version is resolved, and which solver
  classes it ships, decides whether a cell lands in `unsat` or `both0`. The verdict also records
  the solver, its version and the conflict budget per row, so the pin and the evidence agree.

The uniformity question from §16 is unchanged and remains deferred.

## 16c. Third review round, against the corrected rows (2026-09-05)

Five reviewers again, this time against `l4_protocol_20260904` with the §16/§16b fixes in
place. Every number-changing claim was re-derived from the raw rows before it was acted on; two
reviewer diagnoses were rejected (below). What was found, what moved, what is left.

**Fixed in this round (code, all gated):**

1. **The methods paragraph asserted a fairness procedure that did not run.** "Each rival is
   handed an ABC-simplified copy of the circuit as well as the raw one and is scored on
   whichever suits it" — a fixed string, over a campaign whose 82 rows are all `variant: abc`.
   The sentence is now read off the rows (one variant names it and says the axis was not run;
   two variants states the tie-break); `test_the_circuit_variant_sentence_is_read_off_the_rows`.
2. **Table 2's `e2e` was not end-to-end.** Table 1 divides GPAP's whole wall by CMSGen's whole
   subprocess wall; Table 2 divided GPAP's whole wall by the rival's UNCHANGED steady-state
   timer — one side charged for getting ready. It printed DEMOTIC on its own pins as a **0.8x
   loss**; symmetric (rival `unique / wall_s`) it reads **7.6x**; Recurrent 6.3x → 23.4x (table)
   and 28.5x → 128.3x (rare). The rival's wall holds interpreter and CUDA start-up that GPAP's
   in-process wall does not, so the symmetric column is the conservative one for us.
   `test_table_2_e2e_charges_the_rival_its_whole_wall_too`.
3. **Table 3 had no gate under the protocol.** `verify_artifact.py`'s identity loop read only the
   `gpap bench` schema; under `--rivals` it never ran, and a 5000x sabotage of one row's impl/s
   moved the headline 527x → 897x while the report stayed byte-identical. The identity is
   algebra — `(a/b) / ((a/s)/(b/t)) ≡ s/t` — so all it can catch is a ROW-SET mismatch, and that
   is now checked against the generator's printed numbers (Table 3's `n` and closure vs Table
   1's `gm`), plus a physical band on the counters (1e6–5e10 GPAP, 1e3–1e9 CMSGen), which is what
   the sabotage actually violates. `rival_prop_rates` also gained the two Table 1 screens it
   lacked (proved-UNSAT, CMSGen<30), so the row sets agree by construction.
4. **One thin-denominator floor for both sides of both tables.** `RIVAL_MIN_SOLUTIONS = 30` was
   enforced on the rival in Table 2 and only narrated on the CMSGen side of Table 1, where `b14`
   (13 CMSGen samples in 60 s) sat in the rare-pin geomean at 212x. `MIN_SOLUTIONS` screens both;
   seq-rare **77.4x → 72.9x** (conservative — it removes a win). New column `CMSGen<30`, in both
   ledgers and both prose formulas (the Markdown one had also lost `+ unsat` — the third
   Markdown/LaTeX drift).
5. **The dotted-name glob, fourth recurrence.** `rivals_recurrent.py`'s pre-run clean-up globbed
   `sat_{key}_*` with the dot intact, one function above the read-back that strips it (with the
   comment explaining why). For `s208.1`/`s15850.1` no stale file was ever removed and `hits[0]`
   was an arbitrary pick among rungs. Both strip now; a multi-hit read-back selects this call's
   batch or errors. Latent this campaign (one rung completed per instance).
6. **"Instances the rival cannot process" → a budget statement.** All 20 Recurrent failures (10
   per task) are one cause — the 300 s initialization budget expiring on the start_point probe —
   no crash among them, and §11 had already measured `b14` initializing at 900 s and delivering
   55,458 unique at 84.7 u/s. Five of the ten are inside their own 16 (`b15`, `b17`, `b19`, `b22`,
   `s38584`). The caveat now says what it is.
7. **Seed-band ties.** A caveat lists cells whose ratio at our slowest and fastest seed straddles
   1.0 against a rival run once: `c1355` (0.85x, 0.69–1.03), `s386`-table (1.20x, 0.67–1.29),
   `s832`-table (1.15x, 0.61–1.16). Ties, not wins.
8. **Barbarik provenance and gate.** Table 4b comes from `l4_abc_20260902/barbarik_abc` — a
   separate campaign, run locally on the RTX 4080, on our pins over the circuit families — under a
   header naming the L4. The table now states its source; `verify_artifact --barbarik` checks one
   parameter setting, a control that Accepts every eligible instance, and one arm set.
9. **Stale prose.** `docs/paper/README.md` still named `l4_protocol_20260903` as the source of
   the checked-in tables (five places, including a regenerate recipe that would have rebuilt the
   retracted numbers over the corrected ones), "80 rows / 29 circuits", and the old loss set
   (`b11`/`b18` — `b12` is a loss now). `INSIGHTS.md` quoted the superseded headline as settled
   and claimed T=50 and raw-vs-abc "are now run both ways". The campaign README said "9 ok / 3
   warnings" and "no row lost" beside a `b19` encoder error. All corrected; the HTS exclusion
   sentence in `protocol.tex` ("evaluates no circuit at all") now says what is true (pre-encoded
   CNFs, two of them circuit-derived). `check_emission_symmetry` widened a field that fused
   `3.46104.23%` on the one row that most needed reading; a vacuous cache-key test now inspects
   the harness's real task dict.

**Found and NOT fixed — protocol decisions, with the evidence:**

* **The 16.3x Recurrent-table headline is four cells.** `s208.1` 1,158x, `b09` 634.8x, `s1196`
  172.8x, `s15850.1` 75.4x; over the other ten it is **5.6x**, below the superseded 6.2x. `b09`'s
  own rate fell **57.8x** under the ABC-simplified netlist (169,265 → 2,930 u/s at identical batch
  and steps, `equiv_ok` True) while the same netlist HELPED it 1.6x under the rare pin — the
  variant we chose without running the alternative is the largest lever in both Recurrent rows.
* **`s208.1`'s rival timer is inconsistent by 5.65x for identical work.** Same circuit (52 ANDs,
  10/1/8), batch 1e6, steps 1, walls 150.1 vs 153.2 s — `model_runtime` 3.32 s under their pins
  vs 18.77 s under ours; every other dual-mode instance is 0.93–1.08. Their `timer` wraps
  `run_back_prop_seq` with `time.time()` and no `cuda.synchronize`. Direction unknown: if 3.32 s is
  wrong the table cell is understated; if 18.77 s is, the rare cell is overstated.
* **The rivals' PAPER configurations were never run.** Recurrent's paper: `lr=50`, **5
  iterations** (η=50; their Fig. 5 shows unique count rising over 5 iterations at that depth).
  DEMOTIC's paper: `lr=15`, batch 500,000, **10 iterations**. `RIVAL_STEPS` only ever compared
  each script's value against the argparse default (1 vs 10); the paired paper configs were never
  measured. Our T=25 evidence (§6) says steps=1 is better for them at this depth (8.5x at 10);
  steps=5 was not run and the paper's Figure 5 is at η=50.
* **`b06`'s top rung (3,000,000) was refused by our 300 s timeout**, the only non-OOM rejection
  in 60 Recurrent rows; it is one of their 16 and counts as a win in both protocols.
* **The raw→abc netlist change on OUR arm** (campaign 20260903 → 20260904, Recurrent stages,
  everything else equal): rare pin — `b12` **0.042x** (success 0.087 → 0.0041, three seeds within
  2.6%; a 16.1x win became a 0.69x loss), `b18` 2.13x, `s832` 2.20x, all else within 2x; their
  table — nothing beyond 2x. The fairness fix cost us one cell outright, deterministically.

**Reviewer claims rejected on re-derivation.** "The loss boundary moved 30x between runs, so it
measures noise" — `b12`'s move is the variant change, not noise (seeds agree to 2.6%). "Table 4b
is from a different GPU than stated" — true (RTX 4080, per that campaign's README), but a CM_test
verdict is hardware-independent; the defect was the missing provenance line, now printed.

## 16d. The three converged decisions, and finishing T=25 under them (2026-09-05)

The user closed the three protocol questions §16c left open ("we spent way too much time";
"ONE config ... for all the circuits everywhere for everyone"). Decisions, each with why it is the
defensible one and what it cost:

1. **Recurrent's `start_point` initialization gets 1,200 s per instance (bounding the search; its wall is NOT added to the published e2e — §16e corrects the earlier "counted in e2e" wording; the omission only raises our ratio).**
   All 20 of the 300 s campaign's "cannot process" cells were that budget expiring, and §11 had
   measured `b14` initializing at ~743 s. The steady-state metric is untouched (it never
   included initialization); the initialization budget bounds the search but its wall is not
   added to the rival's published e2e (§16e — the harness records no `init_s`; the omission only
   raises our ratio). Measured under it: `b14` initializes and delivers 2 unique (thin,
   screened), `b15`/`b20`/`b21`/`s35932`/`s38584` initialize and deliver 0 (both=0 with GPAP 0
   or a GPAP-only win, per row), `b17`/`b18`/`b19`/`b22` still do not initialize at 1,200 s and
   are reported with that budget named. Four of the ten "cannot process" cells were budget
   artifacts. The per-run timeout is 1,200 s too.
2. **Each rival at its SHIPPED run-script configuration** (`num_steps` 1, `lr` 15, both tools,
   both depths) — the artifact they ship to reproduce their results, measured BETTER for them
   than their papers' values at T=25 (§6: Recurrent 8.5x at 10 steps, DEMOTIC 2.6x), so it cannot
   be called a handicap, and it keeps every completed cell. The papers' values (Recurrent 5 @
   `lr` 50, DEMOTIC 10 @ `lr` 15 — `rival_common.RIVAL_PAPER_CONFIG`) are MEASURED, singly and
   paired, by `steps_calibration.py` and now PRINTED by `paper_tables.rival_calibration` (the
   generator used to name the file and never read it, so RESULTS.md could not say whether the
   setting we used was the better one for the rival). Not run as configuration, not swept.
3. **One circuit variant, `abc`, everywhere**: every tool receives the sequential-ABC-simplified
   netlist GPAP encodes from. It is nobody's friend (§16c: it turned `b12` from a 16x win into a
   0.69x loss for us; cost Recurrent 57.8x on `b09` under one pin and helped it 1.6x under the
   other), and the alternative was never run under the final protocol, so the methods sentence is
   read off the rows (`variant_sentence`) rather than asserted.

**Finishing the campaign under them cost three harness fixes, none of which touches a
measurement.** (a) `--retry-errors` re-ran EVERY error row on every resume, and the probe cache
was packed only at stage end — so each ~1 h Colab VM re-paid the previous VM's 1,200 s
initialization failures and was reclaimed before reaching a new instance; `settled_error` keeps
an error row whose reason names the CURRENT budget (the JSON row is the durable record).
(b) `b21`'s walk — a 1,200 s probe, seven doomed OOM rungs each a model build on a 20M-node
circuit, then the run — exceeded a VM lifetime; `known_ladder`/`remember_oom_ceiling` cache the
smallest OOM rung per (file, T) and the driver packs the cache every 4 min. The grant rule is
unchanged (largest rung that completes); only known failures are not re-measured, and timeouts
are never cached. (c) The last 12 table cells were split across TWO concurrent Colab sessions
(disjoint `SEQ_INSTS`, separate mirror dirs, `merge_splits.py` asserting the 18 measured rows are
byte-identical in both before unioning). The probe cache is keyed on the rival FILE's bytes, so
nothing carries from the rare (miter) file to the table (plain) file — the table stage re-pays
its probes, ~4–5 h serial, by design.

**Disclosure the grant rule owes: the largest completing rung is not always the rival's fastest
rung.** `b06` rare, identical circuit, steps and seeds: the 300 s campaign refused the 3,000,000
rung on timeout and measured the 1,000,000 rung at **746,560 u/s**; the 1,200 s campaign
completes the 3,000,000 rung (346 s wall) at **543,768 u/s** — 0.73x. Unique/s stops tracking the
batch once the model saturates the card, and the walk stops at the first rung that completes, so
the smaller rung is never measured in the same run. It is a share-of-the-machine rule, stated as
such in the methods paragraph, not a search for the rival's optimum; on `b06` it under-states
Recurrent by ~1.4x relative to its best observed rung, and GPAP's b06 rare margin is 1.68M vs 0.54M
unique/s (Table 2's quantity; ~3.1x, or ~2.3x against the 1M rung). The table-stage `b06` row, re-measured in the
same completion run, is compared the same way in the campaign README.

**The completion, measured (2026-09-06 01:35, two concurrent L4 sessions, 5 h 15 min wall).**
All 30 table cells are final under the decisions above; the campaign is 82 rows (81 measured +
`b19`'s GPAP encoder error under their pin, as before).

* **`s208.1`'s "inconsistent timer" (§16c) was the dotted-name CSV read-back bug, and the two
  protocols are ONE task on this circuit.** Re-measured, the table row reads 23,114,436 unique
  in 18.8 s — identical to its rare row, because for this single-output circuit their `[-1]=1`
  pin IS the rare pattern (p_accept 0.375). The superseded row (6,309 unique, 3.32 s, 1,158x)
  had read back a stale CSV through the `sat_s208.1_*` glob that the fix in §16c closed. The
  cell is **1.7x** in both protocols, and the two rows are the same measurement counted once per
  protocol — a reader summing "30 + 30 cells" should know one pair coincides.
* **The Recurrent-table headline is 21.5x over 20 cells (was 16.3x over 14), and it moved UP
  while its largest cell collapsed**: six big instances that did not initialize at 300 s now
  score, all as GPAP wins at the rival's granted batch — `s38584` 371x, `b20` 339x, `b21` 216x,
  `s35932` 70x, `b14` 58x, `b15` 22x (geomean 116x over the six; 10.4x over the other 14).
  Structure: without `b09` 18.0x, without the top four (`b09`, `s38584`, `b20`, `b21`) 10.6x
  over 16, without the top six 7.5x over 14; the bottom five (`s386`, `b10`, `s832`, `b06`,
  `s208.1`) are 1.1–1.7x, three of them inside GPAP's seed band. Quote the structure, not the
  geomean alone.
* **Why the six are large: the rival's grant on a 24 GB L4 is 3,000 (`b20`, `b21`, `s35932`,
  `s38584`; the 10,000 rung OOMs) or 10,000 (`b14`, `b15`; 30,000 OOMs), and one gradient step
  over that batch of a 6–20M-node unrolled circuit takes 5–11 minutes** (`runtime_s` 328–682 s
  for 8,831–213,722 unique ⇒ 13–552 u/s). That is the tool at the largest batch this card runs
  — the rule the paper states — but the authors' 8×V100 would grant more, and unique/s is
  near-proportional to batch (§8), so read these cells as "on a 24 GB L4", not as a constant.
  GPAP's side on the same cells is 4,412–32,047 u/s at its own VRAM-bound K.
* **`b06`'s grant rule disclosure, table row**: the 3,000,000 rung completes (331 s wall) at
  1,117,488 u/s; the superseded 300 s row had measured the 1,000,000 rung at 1,447,059 u/s —
  0.77x, the same shape as the rare row (0.73x). Both `b06` cells are 1.7x/3.1x for GPAP;
  against the rival's best observed rung they would be 1.3x/2.3x — still wins, smaller.
* **Rare headline 93.8x over 12 (was 91.0x)**: `s38584` initializes and delivers 0 (both=0
  column, with `b15` `b20` `b21` `s35932`); `b14` delivers 2 (rival<30, screened). Rival
  failures under the 1,200 s budget: `b17` `b18` `b22` in both protocols, `b19` in the table
  protocol (its rare row is screened as vacuous first). Table 1 (vs CMSGen) barely moves:
  818.3x / 73.1x on the Recurrent set (was 806.6x / 77.4x), ISCAS-85 unchanged.
* **Calibration (`rivals_steps_calibration.json`, T=25, printed in RESULTS.md now)**: DEMOTIC
  8,247,863 u/s at its script's 1 step vs 1,294,723 at its paper's 10 @ `lr` 15 (0.16x);
  Recurrent 631,373 at 1 step vs 163,503 at 5, 85,121 at 10, and 162,875 at the paper's paired
  5 @ `lr` 50 (0.26x). The shipped script setting the campaign ran is the better one for both
  tools by 3.9–6.4x; `lr` 15 vs 50 at 1 step is 1.02x (unchanged).
* **The emission audit read the campaign's HISTORY as rows**: its worst-case bound (208%) came
  from `rerun_20260905/…table.before.json`, the superseded `s208.1` row. `campaign_jsons` now
  reads the top level only when it holds result files; the live worst case is `s208.1` at 36.9%
  of its 18.8 s `runtime_s` (a bound on the per-depth host snapshot, left uncorrected — it can
  only under-report GPAP).

## 16e. Fourth review round — five reviewers against the completed T=25 tables (2026-09-06)

Five adversarial reviewers re-derived every published number from the rows; **all four tables
regenerate byte-identically and no headline number was wrong.** The defects were in the
accounting the numbers rest on, and in the prose around them. Fixed:

* **Table 2's `e2e` was asymmetric a third way, and the generator's own comment called the
  asymmetry "conservative".** It divided GPAP's whole in-process wall by the rival's whole
  *subprocess* wall — which carries ~4.4–4.9 s of interpreter + CUDA-context start-up per
  DEMOTIC cell that GPAP's already-warm process never pays, on cells whose entire wall is under
  a second — while GPAP's side had its delivery subtracted and carried no encoding. Charging the
  rival a fixed cost we exclude is the *generous* direction. Now ONE definition
  (`rival_family`): each side's whole wall on the instance with the tool's fixed process
  start-up excluded on both — GPAP charged encoding + preparation + timed run + delivery, the
  rival charged its timed loop plus its non-timed wall *above* its smallest non-timed wall on
  the row set (a floor that still holds the smallest model build, so it takes too much off the
  rival, never too little). The no-subtraction figure is kept as `gm_e2e_nosub` and printed as
  the upper bound in the caveat. Effect on the printed e2e: DEMOTIC 7.6x → **0.8x** (its whole
  wall is start-up), Recurrent-table 20.9x → **11.8x**, Recurrent-rare 132.0x → **68.8x**. The
  steady-state headlines (2.2x / 21.5x / 93.8x) are untouched.
* **"Recurrent's initialization is counted in its e2e" (five documents) is not what the code
  does.** `probe_start_point` runs before the measured subprocess and its wall is recorded in no
  field, so the published e2e does NOT include it. That is conservative — adding it would lower
  the rival's e2e rate and raise our ratio — so the fix is to the CLAIM, not the number: every
  document now says the 1,200 s budget BOUNDS the search and its wall is not added to the
  published e2e. Recording `init_s` and including it is deferred (it needs the harness
  instrumented and a re-run); it can only move the rivals' e2e down.
* **The batch ladder tops at 3,000,000, and on cells where the top rung completes first try
  nothing above it was offered** — a fourth grant-rejection reason ("the ladder ran out") the
  "rejected only on feasibility" rule does not state. It fires on 5 DEMOTIC cells (`c1355`,
  `c1908`, `c432`, `c499`, `c880`) and `b06`. The rival's rate is near-proportional to its batch
  (§8), and its own earlier campaigns show `c432` at 1.39M u/s @1M → 8.16M @3M, with `fit_batch`
  at the recorded VRAM budget admitting 5.86M rows — so a taller ladder could move `c432` from a
  published 1.21x GPAP win toward a loss. The generator now emits a per-stage caveat naming
  these cells and the risk; a taller ladder is the re-run that would settle it, out of this
  round's scope. The rule as coded and as stated coincided on every OTHER cell (no row carries
  `probe_truncated`, `budget_bound`, or a throughput-based rejection).
* **`b01` is proved UNSAT at T=25 and screened, yet Recurrent scores 306,882 u/s on it** — its
  depth-POOLED count admits solutions at shallower unroll depths the fixed-depth task has none
  of; GPAP delivers 0. It is the one cell in 82 where the pooled superset yields a rival-only
  result, and it sat silently in the `unsat` count column while both tables printed `GPAP=0 = 0`
  and "instances GPAP could not sample: none". The generator now names it with the rival's rate.
* **`verify_artifact`'s soundness gate was vacuous under `--rivals`**: it only ever FAILED on an
  explicit `UNSOUND` and printed nothing on `ok`, so a campaign run without `--verify-sound`
  produced a byte-identical "Publishable" verdict. It now FAILS a stage whose emitting rows
  carry no soundness verdict and prints the checked-row count (the data is clean: 57 ok / 6 off,
  every "off" a zero-model row).
* **The rivals' failure reason named a budget one probe did not get**: `b17`/`b19`/`b22` failed
  at `sp≥1`, so the timing-out probe got only what its predecessors left, while the string said
  "did not complete within the 1200s budget". The search as a whole did have the budget, so the
  conclusion stands, but the phrasing now states it at the search level with the time that
  remained. (Rows already written carry the old phrasing; the conclusion is unaffected.)
* **The steps/lr calibration is now PRINTED in RESULTS.md** (`rival_calibration`), not just
  named — DEMOTIC's paper config 0.16x, Recurrent's paired paper config 0.26x of the shipped
  settings — with the depth the probe ran at, not the CLI's `-T` (DEMOTIC's combinational probe
  ignores it, which had mislabelled it "T=25").
* **The 21.5x / 93.8x structure is now in the paper-facing caveats**: median, the geomean's own
  seed band, the tie count, the top cells, the geomean without them, and the six cells measured
  at the card's 3,000/10,000 grant (116x, against 10x over the rest) with "quote it as on an
  L4". The DEMOTIC-rare row (no measured ratio, `wins 5/5` over five rival-zeros) is named as a
  capability contrast, not a rate.
* **Prose re-derived from the rows**: the acceptance ranges (their pins 0.020–0.97 with two
  bounded cells; rare 0.5 down to 3.9e-8 with 12 bounded), the tautology-cell moves (b05 59x
  for the rival, others under 7%), the distinct-fraction sentence (57 of 74 at 1.000, the rest
  tiny or near-vacuous), the `pre%` share, and the ledger. The emission audit's stale "fixed
  1,000 rows" (it is `UNIQUE_ROWS = 4,000,000`, one row per delivered sample) was corrected in
  both its docstring and its printed verdict.

**Deferred (a re-run, out of this round's scope, each conservative for us):** the rival's e2e
with `init_s` and the rejected rungs' walls added; a taller batch ladder for the six ladder-topped
cells; `b22`-rare re-measured on a clean device (its GPAP arm ran at 7.3 of 23.7 GB free). None
can lower a published GPAP ratio.

## 17. Protocol v1.0 — the freeze (2026-09-06)

Four review rounds (§16–§16e) each found real defects, and each also moved a published column:
Table 2's end-to-end ratio was defined four different ways in three rounds (0.8x, 7.6x,
20.9x/132.0x, 11.8x/68.8x on the same rows); Recurrent's initialization budget went 60 s →
1,200 s → 300 s → 1,200 s; the circuit handed to the rivals went raw → raw+abc (best for the
rival) → abc; the unroll depth 25 → 25+50 → 25; the rival's step count from our 10 to their
script's 1 with a tie-break, to the paper-config question, to the script with the paper values
measured beside it. Every one of those changes was defensible on its own. Together they are the
reason the user called a halt ("each time we are going from one point to another, back and
forth, with no end") and asked for a protocol that does not change. This section is that
protocol's record: what was decided, what was removed, what was considered and not adopted,
and the rule that stops a fifth round.

**The diagnosis.** Three causes, and only one of them is a fixable class of bug. (1) Reviews were
allowed to change metric DEFINITIONS, not only fix errors — and a redefinition always has a
next, fairer version. (2) A few quantities have no right answer because they compare things the
two tools define differently: an end-to-end wall across an in-process sampler and a subprocess
gradient tool is our invention, and on DEMOTIC's cells the rival's timed loop is 0.3–0.7 s
inside a ~5 s process while GPAP's real work is also under a second, so whatever that column read
was process start-up. (3) Scope grew sideways (T=50, raw+abc, paper configs, HTS, seeds), and
every widening created new cells to get wrong.

**The principle.** Publish only quantities each tool defines for itself, under one stated rule
for what we hand it; remove the columns that are our invention across tools; every tool at the
configuration its authors ship, and the batch by one rule.

**Decided (the user, 2026-09-06), in one place — docs/EVAL_RUNBOOK.md §0 is the frozen text:**

1. **The headline is the full set** (30 sequential circuits; 11 ISCAS-85). The rival paper's
   16 are printed beside it on every row, never instead of it. "We don't want to present a
   partial result as the headline."
2. **T=25 is the paper's depth. T=50 is run AFTER the freeze, as an additive appendix** under
   the identical protocol, reported beside the T=25 rows and never in their place.
3. **One cleanup run before the freeze**, then nothing: the cells whose grant the old
   3,000,000 ladder top had bound (six DEMOTIC circuits under both tasks, `b02`/`b06` under
   both Recurrent tasks — every cell whose sweep was the single top rung) re-walked from a
   30,000,000 top, and `b22`-rare's GPAP arm re-measured on a clean device (it had run with
   7.3 of 23.7 GB free). Rows removed to `rerun_20260906/`, re-measured in two concurrent Colab
   sessions (`launch_cleanup_{A,B}.sh`, `merge_cleanup.py`), everything else untouched.
4. **GPAP keeps its shipped `circuit` portfolio.** Every tool at its authors' shipped default
   is the one principle; the bandit asymmetry stays disclosed with its measured price (§12).
5. **Governance.** From the freeze, a review may fix a bug in a number (fix, regenerate, say
   so) and fix prose. It may not change a metric definition, a configuration, a screen, a
   budget or the cell set. Such a proposal is logged below, with its reason, and not acted on.

**Removed by rule: Table 2's `e2e` column.** The rivals' metric is unique solutions per second
of their own `model_runtime`, which excludes their model build; GPAP's comparable figure excludes
construction the same way, and that is the one comparison the rivals themselves define. The
column, its four definitions, `_gpap_e2e_full`, the start-up floor and the `gm_e2e_nosub` upper
bound are gone from `paper_tables.rival_family`; the generator's Table 2 text and the LaTeX
caption say why; `test_table_2_has_one_clock_the_rivals_own_and_no_end_to_end_column` pins the
absence (the rows in that test carry every wall field the old column read, and nothing may read
them). Table 1 keeps both clocks: there sampling wall meets sampling wall and whole process
meets whole process, and that column never moved. Removing Table 2's e2e also dissolves two
deferred items of §16e — recording `init_s` in the rival's e2e and adding the rejected rungs'
walls — since neither touches the rivals' metric.

**Two harness pieces added for the freeze, neither a protocol change.**

* **The batch ladder's top is 30,000,000** (`rival_common.BATCH_LADDER`, rungs 10M and 30M
  added). §16e found that on six cells the 3M top completed first try, so the ladder and not
  the card set the grant — a fourth rejection reason the "feasibility only" rule did not state.
  On the campaign's 24 GB L4 no scored cell completes 30M (c17 would, and is screened as a tiny
  space), so the card bounds every grant; `paper_tables` still names any cell whose top rung
  completes first try, which is the mechanical check that a larger card cannot re-open the
  hole. The two extra rungs cost a big circuit two more doomed model builds on a FRESH card
  only (the OOM ceiling is cached, and a recorded grant skips the walk).
* **The grant is part of the record.** `eval/rivals/extract_grants.py` reads every rival row's
  granted rung and, for Recurrent, its settled `start_point` into
  `eval/rivals/grants/<campaign>.json` (committed); `rivals_{demotic,recurrent}.py --grants
  FILE` pins them (`GRANTS=` for the Colab driver; `run_full_eval.sh` defaults to the committed
  file) — no ladder walk, no start_point probe, one run at the recorded rung, `grant_source`
  and `start_point_source` on the row. A pinned rung this card cannot finish on is walked down
  from, so the rule "largest rung that completes" holds on any card; a cell the record holds
  as a rival failure has no grant and is re-measured, never copied. The walk and the probe were
  the bulk of a campaign's wall on the large circuits (whole VM lifetimes on `b21`) and neither
  is a measurement, so this is what makes a re-run reproducible and cheap. Gated by five tests
  in tests/test_rival_targets.py (ladder top and start, pinned walk and descent, the harness
  pinning both halves, the extractor's keys and the never-pin-a-failure rule, the committed
  file's consistency with the ladder).

**Considered, not adopted — the standing list.** A future review that proposes one of these
gets pointed here, not a re-run.

| proposal | why not |
|---|---|
| a fifth definition of Table 2's e2e (init_s, rejected rungs, any start-up rule) | the column is removed; the rivals define one clock and it is printed |
| pin GPAP to `b8` so every side runs one fixed config | every tool at its shipped default is the principle; the price is measured (§12, 4.5–8.7%) and disclosed per row (`portfolio`, `adaptive`) |
| run the rivals' PAPER configurations as configuration | decided 2026-09-05 (§16d); measured beside the script setting and printed (0.16x / 0.26x of it) |
| the raw circuit variant, or raw+abc best-for-rival | decided 2026-09-05 (§16d): one variant everywhere; the axis stays implemented, not run |
| more rival seeds | decided 2026-09-04 (§16b): one run at its own default, the asymmetry stated |
| HTS | its paper evaluates no circuit (§0 of the runbook) |
| a TV / exact-oracle uniformity number in the paper tables | uniformity is Barbarik CM_test with both controls, only (the user's rule, 2026-09-01) |
| widening the free / tiny / vacuous screens | mechanical, per row, applied before the ledger in every table; a threshold moved after seeing the rows is a fitted screen (the THIN floor was removed by the user's presentation revision, §17b — its disclosure kept) |
| a taller ladder still, or a per-rung optimum for the rival | the 30M top is above the card's reach on every scored cell and the generator checks it; the grant is a share-of-the-machine rule, not a search for the rival's fastest rung (§16d) |
| the paper set as the headline | the user's rule: the full set is the headline — and since §17b the ONLY set printed (no paper-set column) |

**What still changes numbers after the freeze, legitimately:** the cleanup run's rows (item 3,
recorded below when merged), the T=50 appendix rows (additive), and a bug in a number — fixed,
regenerated and recorded here, with the protocol untouched.

**The cleanup run, measured (2026-09-06, 09:13–09:52 local, two concurrent Colab L4 sessions,
one VM lifetime each; the campaign README's last section is the record).** 17 cells re-walked
from the 30,000,000 top. **One cell moved: `c432` under DEMOTIC's pins completes the 10,000,000
rung** (30,000,000 OOMs) and DEMOTIC runs 8,952,982 u/s there against 8,157,612 at 3,000,000 —
GPAP 9,925,487 u/s, **1.11x with a seed band of 0.98–1.13x: a tie, where the ladder-bound row
had read a 1.21x win** (DEMOTIC-native `wins` 8/10 → 7/10 + 1 tie; the 2.2x geomean is
unchanged). §16e's suspicion about this cell was right, and it is exactly the direction a
ladder-bound grant hides. Every other re-walked cell OOMs at 10,000,000 and is granted
3,000,000 again at a rate within noise of its earlier row, so the card and not the ladder
bounds them; `c17` completes the 30,000,000 top (89 / 16 u/s) and is screened as a tiny space
on both tables — DEMOTIC's own exception — so it carries no number. `b22`-rare's GPAP arm on a
clean device (23.5 of 23.7 GB free) still delivers 0 models: the residency was not the cause,
the cell is a genuine limit of the comparison, and its caveat is gone. The GPAP and CMSGen arms
of the 17 cells were re-measured with them (the GPAP-arm cache keys on the harness source, so
the ladder change was a miss by design): all within seed noise. Headlines after the run — vs
CMSGen 1,243.7x / 230.4x / 820.4x / 73.2x; vs DEMOTIC 2.2x (7/10, 1 tie); vs Recurrent 21.5x
on its table (paper set 10.2x) and 93.9x under our rare pin (49.0x) — `verify_artifact` 19 ok /
2 warnings / 0 failures, emission audit PASS. The grants file was regenerated off the merged
rows (74 pinned cells, 8 recorded failures). **The protocol is frozen from here.**

**The T=50 appendix, measured (2026-09-06/07, two concurrent Colab L4 sessions).** The additive
appendix of item 2 — the rival's own script depth (`--num_clock_cycles 50`), the identical
protocol otherwise — run after the freeze on the user's go, printed beside the T=25 rows and
replacing none. Both Recurrent stages 30/30 (`recurrent-{rare,table}-t50`), 20 measured / 10
init-budget failures. **Headlines beside T=25:** vs CMSGen 1,031.7x their pins / 34.5x rare
(steady state; 338.6x / 15.9x e2e); vs Recurrent 8.5x their table (paper set 7.1x) and 77.8x our
rare pins (paper set 439.1x); engine advantage 512x / 590x. The `steps-calib-t50` tie-break holds
at the deeper unroll — steps 1 best for the rival (312,421 vs 75,546 at 5, 39,402 at 10),
`agrees: true`. **The finding the depth carries:** Recurrent's init-budget failures rise from 4
at T=25 (`b17 b18 b19 b22`) to 10 at T=50 (adds `b14 b15 b20 b21 s35932 s38584`) — six more large
ITC'99/ISCAS-89 circuits cannot initialise their gradient in 1,200 s at 50 cycles, `s35932` among
them though it measured at T=25 — so the T=50 geomeans stand on fewer, harder cells (which is why
the table-set ratio is 8.5x against T=25's 21.5x while the rare-set stays large). Every failure
carries both arms and is recorded as a rival failure, never a zero. `verify_artifact` 27 ok / 4
warnings / 0 failures (the warnings are the failure counts, stated); emission audit PASS. The
grants file was regenerated to 120 Recurrent cells (60 @ T=25 + 60 @ T=50) + 22 DEMOTIC, 28
recorded failures, so a re-run pins T=50 too.

**Four harness bugs, fixed during the T=50 run — bug-not-protocol, per item 5's governance**
(uncommitted, cumulative diff in the campaign dir; tests + VM bootstrap checks added, host suite
562 + rival/paper 116 green). None changed a metric, config, screen, budget or cell set; each was
a defect that only appears at T=50 scale (a run minutes long, outlasting a ~1 h Colab VM):
1. **A rival killed with empty stdout+stderr crashed the stage** (`[-1]` of an empty list) — now
   a recorded `exit SIGKILL with no output` refusal the ladder walks past (`rivals_recurrent`).
2. **The OOM ladder re-walked from the top on every VM reclaim** because the ceiling was cached
   only when the walk ENDED, and on `s15850.1` the walk outlasts the VM — now each refused rung
   is cached AS SEEN (OOM/kill at any clock, a timeout at its clock) and the search budget carries
   across VMs (`remember_refusal`, `known_spent`, `run_at_batch(spent=)`).
3+4. **The discarded per-process warmup ran a full O(circuit x T) graph build** — ~1,030 s on
   `s15850.1` at T=50, re-paid every reclaim, which kept the heavy cells from ever landing. The
   rival's per-run cost is its unroll+graph-build (gates x cycles), not its batch, so shrinking
   the warmup batch (10,000 -> 256) did nothing; the fix is to KILL the discarded run after
   `WARMUP_TIMEOUT_S = 60`, which is all the cold-start (CUDA context, clocks, torch import) needs.
   A warmup measures nothing, so neither the batch nor the cap moves a number. LESSON: any per-cell
   rival work longer than a VM lifetime must checkpoint per step, and a warmup must be time-capped,
   never sized by the input.

### 17b. Presentation revision v1.1 — the tables, not the protocol (2026-09-07, the user's direction)

With T=50 complete the user turned to the tables: "everything we are doing now is only to
modify a bit and polish how the tables are produced — only how we analyze and present the
data, nothing about how we ran; that's frozen." Item 5's governance says a *review* may not
change a screen; this is not a review, it is the owner revising the presentation, and it is
recorded here with every number it moved so that nothing can look like a silent re-cut.
**Nothing was re-run.** Every row is the frozen campaign's (the paper folder's
`data/MANIFEST.sha256` pins them); only `eval/artifact/paper_tables.py`, `eval/eval_plots.py`
and their tests changed.

**What changed, and why (the user's four points, in order):**

1. **One `excluded` column instead of seven.** Table 1 printed `both=0 | no-search |
   tiny-space | vacuous | unsat | CMSGen<30 | refused`, Table 2 `both=0 | u/s unmeasured |
   rival<30 | no-search | tiny-space | vacuous | unsat | misaligned | out-of-envelope` — "too
   much data" for a main table. Every cell that carries no ratio for a reason other than one
   tool scoring zero is now ONE column, `excluded`, and the reasons are itemised **by name**
   under each table in RESULTS.md and as `%` comments beside each table in `tables.tex`. The
   two one-sided zero columns (`CMSGen=0`/`GPAP=0`, `rival=0`/`GPAP=0`) stay, as the user
   asked, and Table 2 keeps `rival failed` as its own column: a rival that produced no
   measurement inside the budget is a one-sided outcome (GPAP measured those cells) and the
   T=50 finding lives in it. The per-reason counts and name lists remain in the aggregates
   (`excl_names`), so the caveats, the gate and the tests still see each reason apart, and the
   identities a reader adds up are now `inst = n + CMSGen=0 + GPAP=0 + excluded` (Table 1) and
   `cells = n + rival=0 + GPAP=0 + failed + excluded` (Table 2). `inst` in Table 1 now counts
   every instance of the set (the refused `b19` is inside `excluded`, no longer outside the
   identity), so the four Recurrent rows read 30 alike.
2. **`paper set` is gone.** "Drop the paper set column, it's redundant, no need to even
   mention it; we take the rival benchmarks, that's it, we test on the whole set." One
   population per row; `PAPER_INSTANCES` stays in `rival_common` for the harnesses' `paper`
   keyword and nothing in the tables reads it. The considered-not-adopted row "the paper set
   as the headline" is closed by this: neither as headline nor beside it.
3. **No thin-denominator floor.** "I don't like the rival<30 column; we should include that in
   the result — the rival paper includes rows with 0.1 u/s, so we should consider something
   > 0." The 30-event floor (§16c item 4, one floor for both sides) is removed on BOTH sides —
   removing it on the rival side alone would have left the asymmetric ledger the floor was
   introduced to close — and what survives is the disclosure: `THIN_DENOMINATOR = 30` names
   every geomean cell whose CMSGen count or rival solution count is under it, with the count
   and the ratio, in the caveats; a row whose geomean stands on one or two cells is flagged
   ‡ and its caveat says to report the win count, not the ratio. The floor was conservative
   by construction (every cell it removed was a large win of ours), so dropping it moves
   numbers UP — listed below, every one.
4. **The encoder column is dropped where every row shares one** (every row samples the
   `abc:fast` CNF; the preamble and caption say so). It reappears by itself if a campaign ever
   mixes encoders.

**Figures, brought in line with the tables.** The rival dumbbells (`rivals_<stage>.png`) used
to draw every row without a rival error — vacuous pins included — so their subtitle geomean
disagreed with Table 2's; they now classify through `rival_cells` (Table 2's row, drawn), name
what they do not draw, and carry the task in the title. Two figures were missing: GPAP vs
CMSGen on the converged protocol's cells (`vs_cmsgen_<stage>.png`, six of them — Table 1's
rows, drawn) and every Table 1 cell at once (`sps_scatter_cmsgen.png`), both through the new
`rival_cms_cells` classifier that Table 1 and Table 3 now share (Table 3 used to restate the
screens and could drift; it cannot now). T=50 cells are hollow on both scatters. One figure bug:
the rival scatter chose its marker by `proto.startswith("their")`, so "their ISCAS-85, our
rare pin" — our task — was drawn as the rival's own task; the marker is now the task's.

**One folder.** `../gpap-results/paper/` holds copies of the nine stage/calibration JSONs and
the four Barbarik JSONL files with a sha256 manifest, `regenerate.sh` (tables, gate, audit,
figures, then the copy into `docs/paper/`), the generated outputs and `plots/`. The campaign
directory stays the protocol record.

**What moved** (before → after; every move is a thin cell entering a geomean, and each is named
in the caveats):

| table, row | before | after | cause |
|---|---|---|---|
| T1 Recurrent's set, T=25, rare | 73.2x (n 17), e2e 26.1x, wins 15/18 | **78.0x** (n 18), e2e 29.2x, wins 16/19 | `b14` enters: 12 CMSGen samples in 60 s |
| T1 Recurrent's set, T=50, their pins | 1,031.7x (n 13), e2e 338.6x, wins 23/23 | **1,616.3x** (n 14), e2e 552.8x, wins 24/24 | `b04` enters: 9 CMSGen samples |
| T1 Recurrent's set, T=50, rare | 34.5x (n 13), e2e 15.9x, wins 16/18 | **71.7x** (n 14), e2e 32.7x, wins 17/19 | `b04` enters: 8 CMSGen samples |
| T2 DEMOTIC, our rare pin | — (n 0; 5 rival=0, 2 rival<30), wins 5/5 | **782,616.8x‡** (n 2), wins 7/7 | `c880` (1 solution, 7.1M x) and `c3540` (4, 86k x) enter; flagged ‡, caveat: quote 7/7 |
| T2 Recurrent, our rare pins, T=25 | 93.9x (n 12), wins 15/15 | **173.2x** (n 15), wins 18/18 | `b11` (2 solutions, 40x), `b14` (2, 7,395x), `s4863` (2, 27,722x) enter |
| T2 Recurrent, our rare pins, T=50 | 77.8x (n 12), wins 14/15 | **119.3x** (n 13), wins 15/16 | `s4863` (6 solutions, 20,099x) enters |
| T3 (same rows as T1) | 534x / 512x / 590x engine advantage on those three rows | 513x / 508x / 585x | the same cells entering |

**What did not move:** Table 1's ISCAS-85 rows (1,243.7x / 230.4x) and Recurrent's-set T=25
their-pins row (820.4x); Table 2's DEMOTIC-native 2.2x (7/10, 1 tie) and both Recurrent-table
rows (21.5x at T=25, 8.5x at T=50); the failure counts (4 → 10 at T=50); the gate (27 ok /
4 warnings / 0 failures, the propagation identity closing on every row's new `n`); the
emission audit (PASS, worst case `s208.1` 36.9%).

**How to read the rows that moved.** The T=50 their-pins 1,616x is 1,031x plus ONE cell —
`b04`, where CMSGen delivered 9 samples in 60 s against GPAP's ~80,000/s — and the caveat
under Table 1 names it; the T=25 and T=50 rare rows likewise carry one thin cell each. The
DEMOTIC rare-pin 782,617x stands on two cells with one and four rival solutions and is printed
with the ‡ and a caveat that says to report 7/7 and the five rival zeros, not the ratio. The
rival papers publish exactly such cells (0.1 unique/s rows), which is the user's ground for
scoring them; the ratios on them are orders of magnitude, and the text must say so where it
quotes them.

**Governance note.** This revision is the owner's, not a reviewer's, and the protocol (what ran,
at what configuration, on which cells, with which budgets) is untouched. The considered-
not-adopted rows "widening the free / tiny / vacuous / thin screens" and "the paper set as the
headline" are amended by it: the thin floor and the paper-set column are gone by the user's
direction; the free / tiny / vacuous / unsat screens, and their mechanical, per-row, before-the-
ledger application, are unchanged. Any further proposal to move a screen goes to that table, as
before.
