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
