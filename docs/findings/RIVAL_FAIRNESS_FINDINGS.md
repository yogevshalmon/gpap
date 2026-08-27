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

Largest batch that fits **0.80 of free VRAM**, then two bounded checks: step DOWN if the driver
refuses the size, and probe ONE rung up, kept only if it is also FASTER. The speed condition is
the spill guard — too big does not always error (Recurrent on s386 runs at 1M and is *slower*
than at 300k). On the L4 campaign all three branches were load-bearing:

| outcome | count |
|---|---|
| up-probe recovered an under-grant | 92 |
| driver refused a rung | 82 |
| bigger rung ran but was SLOWER (spilling) | 18 |

A timeout is NOT a memory refusal and must not walk the ladder: HTS's three tseitin cells stepped
1M -> 1,000 at the full 300 s cap on every rung, ~40 min per cell to record an error they had
after the first run. Two rungs, then record it.

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
steps=1 DEMOTIC finishes fast enough that the batch up-probe also takes it from 1M to 3M rows,
and the two effects compound. The audit's projection for `recurrent_table` is 6.1x → **~0.75x, an
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
  configuration.** The headline is the bandit winner's rate. No table says so.
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
- `ARTIFACT.md` is still a self-declared placeholder and `docs/EVAL_RUNBOOK.md` never mentions the
  artifact tooling: no claim->file map, no expected verifier verdict.

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
