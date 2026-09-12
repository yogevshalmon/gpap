# Rival-comparison fairness: what we were measuring, and what we should have been

> **Status:** settled record. §17 is protocol v1.0, FROZEN 2026-09-06 (`docs/EVAL_RUNBOOK.md` §0
> is the frozen text); §17b is the presentation revision v1.1 of 2026-09-07. Campaign headlines
> in §15–§16e are superseded by §17 and §17b.
> **Measured:** 2026-08-24/25 through 2026-09-07 — RTX 4080 Laptop for local calibration, NVIDIA
> L4 (Colab) for every recorded campaign.
> **Rule lives in:** `eval/rivals/rival_common.py`, `eval/artifact/paper_tables.py`,
> `eval/run_full_eval.sh`, `scripts/build_cmsgen.sh`; restated in `docs/EVAL_RUNBOOK.md` §6 and
> `docs/BENCHMARKING.md` §4b/§4c.
> **Gated by:** `tests/test_rival_targets.py`, `tests/test_paper_tables.py`.
> **SECTION NUMBERS ARE AN API.** Code, tests and the GENERATED paper prose cite this file by
> section (`paper_tables.py` emits "§17" into `docs/paper/RESULTS.md`). Sections are in writing
> order, not numeric order — §10 follows §14 — and are append-only: never renumber one.

Every audit here (§1–§16e) found a default that had flattered GPAP — the expected direction,
since every unexamined default in a harness we wrote is a default we chose. The numbers live
here; the rules also live in the code.

---

## 1. Emission symmetry — is anyone timed while delivering solutions?

GPAP's timed run emits nothing (`collect=False` leaves `keep_rows` False) and CMSGen is timed
under the eval patch's `--nowrite` (a stock binary writing every sample cost 8.2x of its wall on
27.sk_3_32). Re-derive with `python eval/rivals/check_emission_symmetry.py --results <campaign>`:
it AST-scans the function each rival's own `timer(...)` wraps, checks every timed `.sample()`
passes `collect=False`, and audits `cms_mode` over the rows. Verdicts: DEMOTIC and HTS run pure
GPU descent inside their timer and count/dedup after it; CMSGen `--nowrite` on 737/737 rows; GPAP
clean. The exception is Recurrent, which copies per-depth candidate assignments to the host
inside its timed loop (`run.py:186`, 0.5–4.0 GiB per run): measured on the 4080 at
`--num_steps 10`, 0.70–1.42% of `model_runtime` (b03/s386/s1196), `num_unique_solutions`
bit-identical with the copy removed, the meter agreeing with a 4-repetition A/B to 0.05 s. Do not
trust one on/off pair: run-to-run spread is the effect size. The bound is per card (0.1–15% on the
L4; worst case `s208.1` 36.9%, §16d). Decision: not corrected — deducting it would raise our
ratio, so the rival's rate stays the one it publishes. Our timed loop pays a per-chunk soundness
check no rival performs; likewise left alone.

---

## 2. The parallelism grant — the batch handed to a rival is its score

Measured on the 4080, a rival's unique/s is near-proportional to its batch over the whole range
the card runs: HTS b03-c25-rare 100 → 100,000 rows 295x, DEMOTIC c1355 1k → 1M 564x, c432
1k → 300k 258x, Recurrent s386 T=25 10k → 300k 18.6x. Before 2026-08-25 the circuit rivals had
a fixed 5 GiB budget whatever the card (a fifth of a 24 GB L4 while GPAP took ~19 GB) and HTS ran
at batch 100/1000. Their own defaults cannot be the rule: every argparse default is
`--batch_size 10` and the demo scripts span 1,000 to 1,000,000 on tiny circuits. A static memory
model is not enough either: DEMOTIC's peak is 11.6–23.0 B per gate-row across c432–c3540, and
the naive `n_gates * T * batch * 4` undercounts real peak 2–6x.

Rule (`rival_common.run_at_batch`, `fit_batch`, `BATCH_LADDER`): the largest rung of the ladder
the rival completes, walked down from the top (30,000,000 since §17), rejected only on
feasibility — OOM, driver refusal, or not finishing inside the wall budget — never on throughput;
one measured configuration per cell; a timeout is a result and is never retried smaller. The
one-rung-up probe (a best-of-two, the trap §8 forbids) was removed 2026-09-03; the walk starts at
the top, not at our model's rung (§16). Rungs tried are recorded as `batch_sweep`, the budget as
`vram_budget_bytes`.

Cost, L4 re-run 2026-08-25/26 (74 rows, 0.80 of free VRAM = 17.5 GiB on every row): DEMOTIC
their pins 4.9x → 4.6x, our rare pins 6,176.4x → 4,344.1x; Recurrent their table 8.4x → 6.1x,
our rare pins 1,507.2x → 660.3x; HTS 6,369.5x → 602.9x (10.6x lower — it alone had run far below
its share, and three HTS cells left the zero column). GPAP's own arm reproduced within draw noise
(median new/old 0.94–1.00x), so the shift is the grant.

---

## 3. Preprocessing — GPAP's was outside the headline rate, CMSGen's was inside its own

`ratio` divided GPAP's steady-state rate by CMSGen's samples per subprocess wall, which contains
CMSGen's parse and setup. Over the 329 timed L4 rows GPAP's preparation (parse + construct + ELS
+ warmup + mining + rebuild) is a median 20.7% / mean 27.4% of its end-to-end wall, and the suite
geomean read 165.0x steady-state against 96.7x e2e — the headline was 1.71x the like-for-like
number (EPFL 2.4–2.8x, sk 2.2x). Rule: Table 1 prints `e2e` and `pre%` beside the steady-state
column (`eval/artifact/paper_tables.py`); circuit encoding is in neither and is reported as `t_enc`.

---

## 4. Making a re-run affordable

Of the 2026-08-23 campaign's ~3.3 h rival stage the rivals' own subprocesses were 64 min; the
rest was our side (the GPAP arm, Recurrent's `start_point` probes, `sim_baseline`) and was re-paid
on every rival-side re-run. Both are cached under `<output dir>/.gpap_cache` (`rival_common`):
the GPAP arm keyed on the sampled file's bytes + task + engine build + GPU name (+ the harness
sources and the `abc` binary since §16b), the start_point probe on circuit + depth; a timed-out
probe is never cached. Two traps: a read-only lookup through `cached_gpap(..., lambda: None)`
poisoned the cache with `None` (hence `cache_peek`/`cache_store` with a `CACHE_MISS` sentinel),
and `colab_supervisor.pull_results` validated every mirrored file as JSON and so deleted the cache
tarball every cycle (it validates by type now).

---

## 5. The artifact packager could not ship the campaign (2026-08-26)

`eval/artifact/make_artifact.sh` had `REPO_ROOT` one level short (the eval/ reorg off-by-one
that eleven other files had), so it could not build an archive at all; and `$PRIOR` outranked
every fresh run directory, so with an August archive on disk it packaged 2026-08-04/05 data for
every arm — old 5 GiB grants, pre-seed-fix HTS — while printing `[ok]`. Rule: the packager is
campaign-aware (`GPAP_CAMPAIGN`, else the newest `l4_campaign_*`), prints every source's age,
refuses a source older than `GPAP_ARTIFACT_MAX_AGE_DAYS` (14) unless it is named in `ARCHIVAL`,
and packages `kernel_ab_l4`. A path that silently selects the wrong data is worse than one that
crashes.

---

## 6. `num_steps` — the batch problem again, on the knob nobody looked at (2026-08-26)

Both sweep drivers hard-coded `--num_steps 10` for every rival. Measured on the 4080 at each
rival's granted batch:

| rival | instance | at our pinned 10 | at their own value | factor |
|---|---|---|---|---|
| Recurrent-CircuitSAT | s386, T=25, batch 100k | 18,081 u/s | steps=1 → 153,726 | 8.50x |
| DEMOTIC | c432, batch 300k | 650,006 u/s | steps=1 → 1,698,654 | 2.61x |
| HTS | b03-c25-rare, batch 30k | 183 u/s | steps=5 → 5 | 0.03x (its script's value is worse) |

Recurrent's count saturates at the first step (2,198,776 unique at steps=1 vs 2,134,454 at 10,
14.3 s vs 118.0 s); HTS needs the descent (0 / 33 / 1,850 / 6,664 unique at 1 / 5 / 10 / 20).
Rule (`eval/rivals/rival_common.py::RIVAL_STEPS`, recorded per row as `steps`/`steps_source`,
re-derived on the campaign's card by `eval/rivals/steps_calibration.py`): each rival runs at the
`num_steps` its own artifacts specify, and where the argparse default and the demo script differ,
the value better for the rival — DEMOTIC 1, Recurrent 1 (their `run_sampler_*.sh`), HTS 10 (its
argparse default; its script's 5 is worse for it). Not swept: a tuned step count invites "why
stop at 20?". Rival rows recorded on or before 2026-08-26 are superseded. Any knob a harness pins
for a tool it did not write is either granted by a stated rule (the batch) or taken from the
tool's own artifacts (the steps).

---

## 7. CMSGen's sample count was floored to the `--statsevery` cadence (2026-08-26)

The eval patch prints `c gpap-stats samples=N ...` every 1000 samples and the harness kept the
last line, so a CMSGen killed on its budget reported the last multiple of 1000. On the 2026-08-23
L4 campaign that is 174 of 357 rows, every one an exact multiple of 1000, 81 of them recorded as
`0` — and `cms_n_samples: 0` scored as an outright GPAP win. Re-measured on
`seq-c25-random / b14 / abc:fast` (published `ratio = inf` inside `wins 19/19`): at
`--statsevery 1` under the same 62 s budget CMSGen delivers 941 samples against GPAP's 38 models
at 0.673 samples/s — an honest ~0.04–0.10x, a 10–25x loss published as an infinite win. The
bucket is a mix: of ten timed-out CNFs re-measured, four are genuine zeros (b14-c25 tseitin,
b15-c25, bj08amba2g4f3, bobsynthor) and four delivered 1–15 samples.

Fix: `--maxtime N` in `scripts/cmsgen_gpap_eval.patch` re-arms `SATSolver::set_max_time` before
every `solve()`, so at the deadline the loop exits normally and prints an exact count with
`final=1`. The harness passes `--maxtime` equal to the sampling budget (what GPAP's
`--time-limit` bounds) and keeps the kill only as a backstop. Verified: a deadline mid-run
(27.sk_3_32, 2 s) 218,000 floored → 218,725 exact, exit 0; a single `solve()` longer than the
budget (b14-c25-rare, 1.05M clauses, 20 s) no stats line → `samples=0 final=1` with 451.9M props
/ 1.32M decisions / 25,784 conflicts, so a cell that never reaches a cadence line now reports
CMSGen's implication rate for Table 3; a completed run exact either way. `cms_count_exact` is
recorded per row and `verify_artifact.py` fails a stage that publishes a floored zero (12 of 13
stages of the 2026-08-23 campaign, correctly).

Both rate columns now charge both sides alike: the stats line carries CMSGen's own sampling wall,
so `ratio` is steady-state/steady-state and `ratio_e2e` end-to-end/end-to-end. Both move against
us relative to the old definition.

---

## 8. The published cell was chosen by `argmax(ratio)` (2026-08-26)

`paper_tables.select_cells` kept, per instance, the encoder with the highest `ratio`, so the
rival's number voted on which of our runs was reported:

| family | argmax ratio (published) | argmax `gpap_sps` (rival-blind) |
|---|---|---|
| seq-c25-rare | 84.3x | 66.7x |
| seq-c25-random | 333.8x | 250.8x |
| seq-c50-rare | 42.1x | 34.0x |
| seq-c50-random | 305.7x | 232.1x |
| all families pooled | 161.9x | 161.1x |

A wash pooled, 1.24–1.33x on the flagship sequential families: ABC helps CMSGen more than us on
some instances (seq-c25-rare b04: GPAP x1.2, CMSGen x4.0), so maximising the quotient awards a
shared preprocessing gain to one side. Rule: no selection — Table 1 and Table 3 emit one row per
(family, encoder), which also keeps the non-monotone encoder A/B visible (ABC unlocks `b14` at a
rare pin and collapses `b13`'s). `eval/eval_plots.py` selects on `gpap_sps` where it must;
`slide_tables.py` and `soft_price_tables.py` import the paper's function.

## 9. `wins` and `n` were counted over a different row set than the geomean

The degeneracy screens (vacuous pins, zero-search) were applied only when building the geomean;
`inst` and `wins` were computed before them, so the invariant printed under Table 1 was false on
6 of 7 families (seq-c25-rare 13+1+0+1 = 15 against `inst` 19, beside `wins 17/18`). A second
hole: `zero_search` was evaluated only over cells where both samplers emitted, so a zero-search
cell with a CMSGen zero (EPFL `ctrl` under ABC, succ 1.0 at 240M samples/s) was published as an
outright win. Rule: screen first, then bucket; a degenerate cell leaves through its own column and
takes no part in the geomean, the zero ledger or the win count; the invariant is
`inst = n_gm + CMSGen=0 + GPAP=0 + both=0 + no-search + vacuous` (folded into one `excluded`
column by §17b), pinned by `tests/test_paper_tables.py::test_every_instance_is_counted_exactly_once`.

---

## 11. The rival was declared incapable on our stopwatch (2026-09-03)

`rivals_recurrent.py` probes upward for a usable `--start_point` (their `backward()` raises
while the pinned output has no gradient path to any input). Each probe was capped at
`PROBE_TIMEOUT = 60.0` against the measured run's `DEFAULT_TIMEOUT = 300.0`, and all three ways
the search can fail reported one string, "no gradient path at any start_point<=10" — a claim
about the rival, produced by our cap. Measured on b14, T=25 (4080): at the 60 s cap
start_point=1 times out and is reported incapable; at 900 s it runs — 743 s wall, 636 s by their
clock, 4,657 unique. The ratio must come from the granted batch, never the probe (batch 256 by
construction): through `run_recurrent_ladder` at the granted 3,000 (10,000 refused) it delivers
55,458 unique at 84.7 u/s, 11.6x the probe's 7.3 — against GPAP's 3,420 u/s a ~40x cell, not the
~470x a probe-derived ratio would claim. `b17` is the opposite case: start_point 0 and 1 fail in
2 s and 6 s with a genuine `element 0 of tensors does not require grad`, and 2 through 12 every
one time out at 1200 s (1132–1134 s each, the model build alone) — a real limit. The two produced
the identical error string, and only one was true.

Rule (`probe_start_point`): the start_point search is the rival's initialization and gets one
budget (300 s here; 1,200 s since §16d); each probe runs with what is left; a timing-out probe
stops the walk (a larger start_point pays the same model build); the three failures are named
separately (a gradient-path refusal at every depth reached; a probe that did not complete within
the budget; the budget used up before one completed), the budget-bound ones reading
`did not initialize: ... {budget}s initialization budget`; `paper_tables` prints the failed count
and excludes them from the geomean, never as zeros. The cap is ours and the same for every
instance; what is not defensible is a cap chosen after seeing which one gives the better ratio.

## 12. What pinning GPAP's config would cost — measured, not adopted (2026-09-03)

Every rival and CMSGen run one fixed configuration; GPAP's `CIRCUIT_PORTFOLIO` is a per-instance
bandit that measures configs inside the timed wall and reports the winner's rate. Measured on the
rival path (T=25, their pins, Tseitin, seed 5, 20 s): pinning `b8` costs 4.5–8.7% (b13 −4.6%,
b12 −4.5%, b04 −8.7%, b14 −5.3%), pinning `b1` 3.7–31.2%, and the bandit beats both pins on
every instance. Not adopted (§17 item 4): the default stays the shipped bandit; `--portfolio b8`
(`rivals_recurrent.GPAP_PORTFOLIO`, shared by the DEMOTIC harness) is the knob, and every row
records `portfolio` and `adaptive` so the asymmetry is in the data, not a caption.

## 13. Table 2's GPAP arm encodes with `abc:fast`, like Table 1 (settled 2026-09-03)

`rivals_recurrent.run_gpap` (shared by the DEMOTIC harness) builds the arm's CNF with
`abc_backend.build_cnf_abc(..., opt="fast")` and records `gpap.encoder` per row; without an
`abc` binary the arm is an error row. The harness had called `bench.build_cnf` with no encoder,
so Table 2 ran GPAP 1.04–4.8x slower than Table 1 on the same circuits and pins, unlabelled
(T=25, their pins, seed 5, 20 s, bandit: b13 2.19x, b04 1.25x, b12 1.04x, b14 4.80x — 413,451
Tseitin variables against 41,937 under abc:fast). The "must be Tseitin because the rivals unroll
the raw circuit themselves" argument was wrong: fairness is same circuit, same task, each tool
compiling into its own representation; `abc:fast` is function-preserving with the same projected
solution space (`tests/test_abc_encoder.py`) and its encode wall is `encode_s`, outside every
rate. `rival_common.oversize_reason` takes `encoder=` (its host-RAM model was Tseitin's), so every
seq instance — b19 and s35932 at T=50 included — builds on the 13 GB L4 VM. The coverage run is
gone from the rival arm (its host reconstruction was the 61 GB OOM); the distinct fraction is
measured on the timed run's own kept rows with the projection wall subtracted (`out_project_s`,
re-derived by `check_emission_symmetry.py`). Rows before this date are superseded.

## 14. The converged protocol: three tools on one constraint, and Table 1 read off Table 2's rows (2026-09-03)

Table 1 (vs CMSGen) had been a separate `gpap bench` sweep on instances pinned our way, on
families no rival ran; a reader could not line a Table 1 row up against a Table 2 row. Rule: the
instance set is the rivals' own published sets under their pins and our rare pin, and every cell
is run by three tools on one constraint — the rival on its circuit, GPAP on the `abc:fast` CNF of
that circuit and pins, CMSGen on that very CNF file. The harness encodes once
(`rivals_recurrent.build_arm_cnf`), the GPAP arm samples it (`run_gpap(cnf=...)`),
`rival_common.cmsgen_arm` runs the patched binary (`--nowrite`, `--maxtime` = the GPAP budget,
same `n`) through the runner `gpap bench` uses (`gpap.benchmark.cmsgen_run`), recording the bench
driver's `cms_*` keys plus host `cpu`/`gpu`. `paper_tables.rival_cms_family` reads Table 1 off
the rival stages (Table 2's screens first, Table 1's ledger after; absent and crashed are not
zero) and `rival_prop_rates` Table 3 off the same cells; `verify_artifact.py --rivals` applies
the bench-stage gates to them. Cost: +60 s of CMSGen per row. The family matrix (`CMS_FAMILIES`)
is supplementary, never the headline. Gated by `tests/test_rival_targets.py` and
`tests/test_paper_tables.py`.

Three gaps closed in the rows the same day: the rival-facing ratio is the steady-state
`samples_per_s` (excluding preparation as the rival's `model_runtime` excludes its build) — the
e2e column this added was removed by §17; constraint alignment is asserted on every row of every
stage (`rival_common.native_pin_check`, `rivals_recurrent.miter_self_check`, `pin_check_ok`); the
arm simulates its kept rows through the original circuit (`sim_check` ok | BAD | off, failed by
`verify_artifact`). And the tiny-space screen reads the solution space, not the input hypercube:
`c1355` under the rare pin (2^41 points, ~700 solutions, kept rows 0.56 distinct) had passed it;
`rival_common.space_from_distinct` and `paper_tables.solution_space_estimate` take the minimum of
the hypercube, p_accept x hypercube and the repeat-rate size, in both tables.

## 10. Still open after §6-§9 — things a referee can still ask

Carried over from the 2026-08-26 audit; later resolution in brackets.

Must state in the text:
- Table 2 had no `e2e` column while every row carried `sps_e2e` (DEMOTIC native 4.6x → 3.3x e2e,
  Recurrent table 6.1x → 3.7x, HTS 603x → 404x) [added §14, redefined §16c/§16e, removed by rule
  in §17: one clock].
- The whole 357-cell campaign was n=1 (`seeds: null, seed: 5`); Table 6 is a same-seed replay
  and bounds determinism, not seed variance [GPAP median-of-3, rival one run, §16b].
- Table 5's "0.480x soft price" is over 14 instances, not 15, carried by two (median 0.979);
  b11 goes 309 → 1 samples/s while the table prints "capability losses: 0".
- Table 2 applied no `p_accept` screen: Recurrent's `b13` their-target cell has `p_accept = 1.0`
  and its 12.9x sits above the row geomean (5.27x without it); the real span is 0.0218 to 1.0
  [vacuous screen per pin, §15].
- GPAP runs an adaptive bandit; CMSGen and all three rivals run one fixed configuration [priced
  in §12, disclosed per row, kept by §17].
- `paper_tables.py` says "Nothing here is typed by hand" and hardcodes numbers in three places
  ("up to 1.9x" is 1.42x/1.58x) [prose re-derived from the rows, §16e].
- Table 1's `n` is instances, Table 2's `cells` is instance x variant; label them differently.
- `pre%` under-explains itself: 1.07–1.34x of the steady-state-vs-e2e gap is the bandit's
  exploration inside the timed wall (median 1.08x / mean 1.18x / p90 1.34x over 329 rows).

Nice to have:
- `verify_artifact.py` accepted `--recheck` and never read it; the recheck, soft and kernel-A/B
  datasets shipped with no verdict; nothing read `pin_check_ok` or `equiv_hits`; no rival JSON
  recorded a GPU.
- `bench_equiv_check` compares only the accept-word conjunction, passing vacuously when no
  stimulus hits the target (12 of 19 abc rows `equiv_hits=[0,0,32768]`); compare all output words.
- HTS caches its model at `models/{basename}.json` and the harness cleared only `src/results/`,
  so a local re-run under the same filename measures the old formula (recorded campaigns ran on
  fresh VMs).
- `ARTIFACT.md` was a placeholder: no claim → file map, no expected verifier verdict.

Checked and clean — do not re-litigate: emission symmetry; the batch grant; zero-ledger symmetry;
the propagation identity (closes within 1%); soundness (357 cells, 0 unsound); the
`unique_per_s_est` window (bias 7e-4, against us; replaced by exact dedup in §16); DEMOTIC native
pin alignment (`pin_check_ok` on all 20); ABC variants bit-exact at the measured depth; seed
unification at 5; engine provenance per row; the GPAP-arm cache key.

Refuted, do not revive: "the unique/s ratio divides two different estimators" (bounded at 7e-4,
wrong direction); "`ratio=inf` wins the argmax so the CMSGen=0 column is populated by encoder
choice" (a wording problem; §8 removed the argmax anyway).

## 15. The campaign under the protocol: Colab L4, 2026-09-03/04 — and what the rows taught

`$GPAP_RESULTS/l4_protocol_20260903`: 80 cells, 0 errors, 0 unsound, `verify_artifact --rivals`
0 failures, emission audit PASS, 6 VM lifetimes, no row lost. Run on the L4 because the laptop's
4080 sat at a 40 W power cap (~1.1 GHz under load, un-liftable from WSL) — check
`nvidia-smi -q -d POWER,PERFORMANCE` before any local measurement. Its headlines are superseded
(§16, §17). What the rows fixed in `paper_tables.py`, gated by tests: screens before buckets in
Table 2 as in Table 1, with the vacuous list per pin (`b13` is a tautology under Recurrent's
target and not under our rare pin; `b19` had read vacuous in one table and rival failure in the
other); a run with zero models is a GPAP zero, not "unmeasured" (`b01`); the Recurrent egress
bound is per card (§1); two cells the screens missed were named — `c432` under the rare pin
propagating to 4 effective variables (the continuous zero-search screen of §16 followed) and one
repeat in 1,000 kept rows implying ~5e5 solutions (`c1908` their pins, `c432` rare).

## §16 — The referee pass on the l4_protocol_20260903 campaign (2026-09-04)

Five adversarial reviews re-derived every number from the rows; every published figure
reproduced, and the defects were in the harness and two metric definitions — all four
number-moving ones in our favour. The campaign is superseded.
- DEMOTIC was under-granted on 4 of 11 ISCAS-85 cells (c880 1.53x, c2670 2.12x, c3540 1.85x,
  c7552 3.87x): `run_at_batch` started at our VRAM model's rung (calibrated at `num_steps=10`
  while we run 1) and only stepped down. DEMOTIC-native 4.73x → 2.99x. Rule: walk down from the
  top, reject on feasibility only (§2).
- `s208.1` was our bug published as a rival failure: their `run.py` strips dots from the results
  CSV name (`sat_s2081_1_...`) and we globbed verbatim; `s15850.1` and `s38584.1` share the shape.
- `c1355` ate the session's cold start (the first DEMOTIC run of a process is ~3.3x slow): one
  discarded run per (tool, process) inside `run_at_batch`.
- `s35932` is not their circuit — their Table II's 8th ISCAS-89 row (12 PI / 278 PO / 19,253
  gates / 1,452 DFF) is `s38584`. Dropping `s35932` moved the full-set geomean 1,150.2x → 759.0x
  (its 586,616x was one CMSGen sample in 60 s). `s38584` is in `PAPER_INSTANCES`; `s35932` stays
  as a circuit we add.
- Unique/s was a 1,000-row estimate where the rivals dedup exactly: now `exact_distinct` over
  every delivered row, off the timed clock; the unique/s cap goes with it.
- Fields were medianed independently across seeds (138 field/cell combinations from a different
  seed than the row's rate): `median_run` publishes every field of the run whose rate is the
  median; safety verdicts fold over every seed; `sps_min`/`sps_max` record the spread (p90 of
  max/min 1.51; six wins inside their own seed spread).
- `zero_search` was a knife-edge at zero counters: `c432` rare (a model per 1.09 implications,
  next-lowest 44.1) was the entire DEMOTIC-rare geomean at 89,786x. Now continuous: fewer than
  two implications per delivered model.
- A cell proved UNSAT at its depth gets its own column (`b01` under Recurrent's target is
  unreachable at exactly T=25; their sampler pools every depth).
- The LaTeX emitter dropped `tiny`/`vacuous`, so neither table summed; gated by a test that
  parses the emitted `.tex`. The screen caveat reports per stage in both directions (the screens
  raise three headlines 1.19–1.24x and lower one 1.42x).
- Depth and circuit were ours, not theirs: T=25 was called their default while their script says
  `--num_clock_cycles 50`; we sampled an ABC-optimized CNF while handing rivals the raw netlist.
  Both were widened here and settled in §16d/§17 (T=25 headline + T=50 appendix; one variant,
  `abc`). `--lr` joins `num_steps` under the shipped-artifact rule (both scripts say 15;
  `steps_calibration.py --axis lr`).
- Left standing: soundness 80/80; emission symmetry; absent-is-not-zero; `cms_count_exact`
  80/80; one build id across 6 VM rebuilds; `pin_check_ok` 80/80; no rival run timed out (max
  wall 259 s). Deferred by the user: the uniformity claim (Barbarik counts only; our exact-oracle
  instrument reads the shipped configuration 1.31x worse than CMSGen on TV).

### §16b — Second referee pass, on the fixed tree (2026-09-04)

Five more reviewers verified the fixes and found a further layer, each a rule stated in one place
and not carried to another:
- `s38584` had no entry in `rivals_recurrent.TARGETS`, and table mode drops an untranscribed
  name with `continue` (the skip that had cost `b15` and `s208.1`); gated by "every paper-set
  instance is scoreable".
- The GPAP-arm cache key omitted the harness files, the encoders, the miner, the K-sizing and the
  `abc` binary; all hashed now.
- `_gpap_e2e` took its own median across seeds; it reads the picked run's `sps_e2e`.
- The Markdown ledger lacked `unsat` and Table 2's `n` printed the win denominator (30 against 29
  cells); both sum now.
- The both-directions screen caveat was gated on a vacuous cell's presence, dropping the two
  ISCAS-85 stages; it is gated on whether the screens moved the geomean.
- The generated methodology paragraph described the superseded harness; gated by a drift test.
- `probe_truncated` and `budget_bound` are row-level and counted by the gate, which also warns
  when the proved-UNSAT screen never ran.
- Two claims of ours were wrong: the miter control's "at most 1%" (`b05` moves 4.3x) and
  `docs/paper/README.md`'s numbers (it carries none now).
- The failure regime is generated from the rows: every loss to CMSGen has amplification at most
  1.94x and success at most 6.9e-4, and no winning cell meets both; cells whose baseline delivered
  fewer than 30 samples are named separately.

Decided 2026-09-04, protocol since: one rival seed at its own default (`RIVAL_SEED = 0`, recorded
per row with its source; three seeds would have cost ~1.7 h and, without `--seed` on the command
line, re-ran the same initialization); the 300 s wall stays but is falsifiable per cell
(`budget_bound`) [raised to 1,200 s in §16d]; `python-sat` pinned (`>=0.1.8.dev0,<1.10`) in
`pyproject.toml` and `rivals_setup.sh` because it adjudicates the proved-UNSAT screen, with
solver, version and conflict budget recorded per row.

## 16c. Third review round, against the corrected rows (2026-09-05)

Against `l4_protocol_20260904`. Fixed, all gated:
1. The methods paragraph asserted a raw+abc procedure that did not run (82 rows, all
   `variant: abc`); the sentence is read off the rows
   (`test_the_circuit_variant_sentence_is_read_off_the_rows`).
2. Table 2's `e2e` charged only GPAP for getting ready (DEMOTIC printed a 0.8x loss; symmetric
   7.6x; Recurrent 6.3x → 23.4x table, 28.5x → 128.3x rare) [column removed in §17].
3. Table 3 had no gate under `--rivals` (a 5000x sabotage moved 527x → 897x unseen): row-set
   agreement with Table 1 plus a physical band on the counters (1e6–5e10 GPAP, 1e3–1e9 CMSGen);
   `rival_prop_rates` gained Table 1's screens.
4. One thin-denominator floor for both sides (`MIN_SOLUTIONS = 30`; `b14`'s 13 CMSGen samples had
   sat in the rare geomean at 212x; seq-rare 77.4x → 72.9x) [removed on both sides by §17b, the
   disclosure kept].
5. The dotted-name glob, fourth recurrence, in the pre-run clean-up; both sites strip now.
6. All 20 Recurrent failures were the 300 s initialization budget, none a crash; the caveat says
   so.
7. Seed-band ties are listed: `c1355` (0.85x, 0.69–1.03), `s386`-table (1.20x, 0.67–1.29),
   `s832`-table (1.15x, 0.61–1.16).
8. Table 4b's provenance (`l4_abc_20260902/barbarik_abc`, RTX 4080, our pins) is printed;
   `verify_artifact --barbarik` checks one parameter setting, an Accepting control, one arm set.
9. Stale prose in `docs/paper/README.md`, `INSIGHTS.md`, the campaign README and `protocol.tex`
   (HTS "evaluates no circuit at all" → pre-encoded CNFs, two circuit-derived) corrected.

Found, not fixed — protocol evidence: the 16.3x Recurrent-table headline was four cells
(`s208.1` 1,158x, `b09` 634.8x, `s1196` 172.8x, `s15850.1` 75.4x; 5.6x over the other ten), and
`b09`'s rate fell 57.8x under the ABC-simplified netlist while the same netlist helped it 1.6x
under the rare pin; `s208.1`'s timer read 5.65x apart for identical work (a stale CSV, resolved
in §16d); the rivals' paper configurations (Recurrent 5 iterations @ `lr` 50, DEMOTIC 10 @ `lr`
15) had never been run (measured beside the script setting in §16d); `b06`'s 3,000,000 rung was
refused by our 300 s timeout; the raw → abc change on our arm turned `b12` rare from a 16.1x win
into a 0.69x loss (success 0.087 → 0.0041, seeds within 2.6%), `b18` 2.13x, `s832` 2.20x.

Rejected on re-derivation: "the loss boundary moved 30x, so it measures noise" (`b12`'s move is
the variant change, not noise); "Table 4b is from a different GPU than stated" (true, but a
CM_test verdict is hardware-independent; the missing provenance line was the defect).

## 16d. The three converged decisions, and finishing T=25 under them (2026-09-05)

The user closed the open questions ("ONE config ... for all the circuits everywhere for everyone"):
1. Recurrent's `start_point` initialization gets 1,200 s per instance (the per-run timeout is
   1,200 s too). It bounds the search; its wall is not added to any published rate (§16e). Under
   it `b14` initializes and delivers 2 unique, `b15`/`b20`/`b21`/`s35932`/`s38584` initialize and
   deliver 0, `b17`/`b18`/`b19`/`b22` still do not; four of the ten 300 s "cannot process" cells
   were budget artifacts.
2. Each rival at its shipped run-script configuration (`num_steps` 1, `lr` 15, both tools, both
   depths), measured better for them than their papers' values (`rival_common.RIVAL_PAPER_CONFIG`:
   Recurrent 5 @ `lr` 50, DEMOTIC 10 @ `lr` 15). `rivals_steps_calibration.json`, T=25: DEMOTIC
   8,247,863 u/s at 1 step vs 1,294,723 at its paper config (0.16x); Recurrent 631,373 at 1 vs
   163,503 at 5, 85,121 at 10 and 162,875 at the paired paper config (0.26x); `lr` 15 vs 50 at
   1 step 1.02x. Printed by `paper_tables.rival_calibration`; not run as configuration, not swept.
3. One circuit variant, `abc`, everywhere — nobody's friend (§16c) — with the methods sentence
   read off the rows (`variant_sentence`).

Harness fixes that touch no measurement: `settled_error` keeps an error row whose reason names
the current budget (`--retry-errors` had re-paid every 1,200 s failure on each VM);
`known_ladder`/`remember_oom_ceiling` cache the smallest OOM rung per (file, T), packed every
4 min, timeouts never cached; `merge_splits.py` unions two concurrent sessions after asserting the
shared rows are byte-identical.

Grant-rule disclosure: the largest completing rung is not always the rival's fastest. `b06` rare,
identical circuit, steps and seed: the 1,000,000 rung (300 s campaign) 746,560 u/s, the 3,000,000
rung (1,200 s campaign, 346 s wall) 543,768 u/s — 0.73x; the table row 1,447,059 → 1,117,488
(0.77x). Unique/s stops tracking the batch once the model saturates the card and the walk stops
at the first rung that completes; it is a share-of-the-machine rule, stated as such in the methods
paragraph. GPAP's `b06` cells are 1.7x/3.1x, or 1.3x/2.3x against the rival's best observed rung.

The completion (2026-09-06 01:35, two concurrent L4 sessions, 5 h 15 min): 82 rows (81 measured
+ `b19`'s GPAP encoder error). `s208.1`'s "inconsistent timer" was the dotted-name CSV bug:
re-measured, 23,114,436 unique in 18.8 s in both protocols (its `[-1]=1` pin is the rare pattern,
p_accept 0.375), 1.7x — one measurement counted once per protocol. The six large instances that
did not initialize at 300 s score as GPAP wins at the rival's 3,000 / 10,000 grant on a 24 GB L4
(`s38584` 371x, `b20` 339x, `b21` 216x, `s35932` 70x, `b14` 58x, `b15` 22x; one gradient step
over a 6–20M-node unroll takes 5–11 minutes, 13–552 u/s; GPAP 4,412–32,047 u/s there) — read
them as "on a 24 GB L4", and quote the structure of the Recurrent-table geomean (21.5x over 20;
18.0x without `b09`; 10.6x over 16 without the top four; the bottom five 1.1–1.7x, three inside
GPAP's seed band), not the geomean alone. The emission audit had read history files as rows (a
208% bound from a superseded `s208.1` row); `campaign_jsons` reads the top level only; the live
worst case is `s208.1` at 36.9% of its 18.8 s `runtime_s`, left uncorrected.

## 16e. Fourth review round — five reviewers against the completed T=25 tables (2026-09-06)

All four tables regenerated byte-identically and no headline number was wrong. Fixed:
- Table 2's `e2e` was asymmetric a third way (the rival's subprocess wall carries ~4.4–4.9 s of
  start-up GPAP's warm process never pays): one definition with fixed start-up excluded on both
  sides — DEMOTIC 7.6x → 0.8x, Recurrent-table 20.9x → 11.8x, rare 132.0x → 68.8x [the column
  was then removed by §17].
- "Recurrent's initialization is counted in its e2e" (five documents) was false:
  `probe_start_point`'s wall is recorded in no field. The claim was fixed, not the number; the
  omission only raises our ratio.
- The 3,000,000 ladder top completed first try on `c1355`, `c1908`, `c432`, `c499`, `c880` and
  `b06` — a fourth rejection reason ("the ladder ran out") the rule did not state; earlier
  campaigns had `c432` at 1.39M u/s @1M → 8.16M @3M [30,000,000 top and the cleanup run, §17].
  The generator names any cell whose top rung completes first try.
- `b01` is proved UNSAT at T=25 yet Recurrent scores 306,882 u/s on it through its depth-pooled
  count (GPAP 0); the generator names it with the rival's rate.
- `verify_artifact`'s soundness gate was vacuous under `--rivals` (a campaign without
  `--verify-sound` read "Publishable"); it fails a stage whose emitting rows carry no verdict
  (57 ok / 6 off, every "off" a zero-model row).
- The failure string named a budget one probe did not get; it is phrased at the search level.
- The steps/lr calibration is printed in RESULTS.md with the depth the probe ran at (DEMOTIC's
  combinational probe ignores `-T`).
- The 21.5x / 93.8x structure, seed band, ties, top cells and the six L4-grant cells are in the
  paper-facing caveats; DEMOTIC-rare (`wins 5/5` over five rival zeros) is a capability contrast,
  not a rate.
- Prose re-derived from the rows: acceptance ranges (their pins 0.020–0.97; rare 0.5 down to
  3.9e-8, 12 bounded), tautology-cell moves (b05 59x for the rival), distinct fractions (57 of 74
  at 1.000), the emission audit's "fixed 1,000 rows" (it is `UNIQUE_ROWS = 4,000,000`).

Deferred then, each conservative for us: the rival's e2e with `init_s` and rejected rungs
(dissolved by §17's column removal); a taller ladder (the cleanup run, §17); `b22`-rare on a
clean device (§17: still 0 models).

## 17. Protocol v1.0 — the freeze (2026-09-06)

Four review rounds (§16–§16e) each found real defects, and each moved a published column:
Table 2's end-to-end ratio was defined four ways in three rounds (0.8x, 7.6x, 20.9x/132.0x,
11.8x/68.8x on the same rows); Recurrent's initialization budget went 60 s → 1,200 s → 300 s →
1,200 s; the circuit handed to the rivals raw → raw+abc (best for the rival) → abc; the depth
25 → 25+50 → 25; the step count from our 10 to their script's 1, to the paper-config question, to
the script with the paper values measured beside it. Each change was defensible alone; together
they are why the user called a halt ("each time we are going from one point to another, back and
forth, with no end") and asked for a protocol that does not change.

Diagnosis: (1) reviews were allowed to change metric definitions, not only fix errors, and a
redefinition always has a fairer next version; (2) some quantities have no right answer — an
end-to-end wall across an in-process sampler and a subprocess gradient tool is our invention, and
on DEMOTIC's cells the rival's timed loop is 0.3–0.7 s inside a ~5 s process while GPAP's real
work is also under a second, so that column measured process start-up; (3) scope grew sideways
(T=50, raw+abc, paper configs, HTS, seeds), and every widening created new cells to get wrong.

Principle: publish only quantities each tool defines for itself, under one stated rule for what we
hand it; remove the columns that are our invention across tools; every tool at the configuration
its authors ship, and the batch by one rule.

Decided (the user, 2026-09-06); `docs/EVAL_RUNBOOK.md` §0 is the frozen text:
1. The headline is the full set (30 sequential circuits; 11 ISCAS-85): "We don't want to present
   a partial result as the headline." The rival paper's 16 were printed beside it until §17b
   dropped that column.
2. T=25 is the paper's depth. T=50 is run after the freeze as an additive appendix under the
   identical protocol, reported beside the T=25 rows and never in their place.
3. One cleanup run before the freeze, then nothing: every cell whose sweep was the single
   3,000,000 top rung (six DEMOTIC circuits under both tasks, `b02`/`b06` under both Recurrent
   tasks) re-walked from a 30,000,000 top, and `b22`-rare's GPAP arm re-measured on a clean device
   (it had run with 7.3 of 23.7 GB free). Rows moved to `rerun_20260906/`, re-measured in two
   concurrent Colab sessions (`launch_cleanup_{A,B}.sh`, `merge_cleanup.py`), all else untouched.
4. GPAP keeps its shipped `circuit` portfolio: every tool at its authors' shipped default is the
   one principle; the bandit asymmetry stays disclosed with its measured price (§12).
5. Governance: from the freeze, a review may fix a bug in a number (fix, regenerate, say so) and
   fix prose. It may not change a metric definition, a configuration, a screen, a budget or the
   cell set. Such a proposal is logged in the list below, with its reason, and not acted on.

Removed by rule: Table 2's `e2e` column. The rivals' metric is unique solutions per second of
their own `model_runtime`, which excludes their model build; GPAP's comparable figure excludes
construction the same way, and that is the one comparison the rivals themselves define. The
column, its four definitions, `_gpap_e2e_full`, the start-up floor and the `gm_e2e_nosub` bound
are gone from `paper_tables.rival_family`; the generator's Table 2 text and the LaTeX caption say
why; `test_table_2_has_one_clock_the_rivals_own_and_no_end_to_end_column` pins the absence (its
rows carry every wall field the old column read, and nothing may read them). Table 1 keeps both
clocks — sampling wall meets sampling wall, whole process meets whole process, and that column
never moved. This also dissolves §16e's deferred `init_s` and rejected-rung items.

Two harness pieces added for the freeze, neither a protocol change:
- The batch ladder tops at 30,000,000 (`rival_common.BATCH_LADDER`, rungs 10M and 30M added).
  On the campaign's 24 GB L4 no scored cell completes 30M (c17 would, and is screened as a tiny
  space), so the card bounds every grant; `paper_tables` still names any cell whose top rung
  completes first try, the mechanical check that a larger card cannot re-open the hole. The two
  extra rungs cost a big circuit two more doomed model builds on a fresh card only (the OOM
  ceiling is cached, and a recorded grant skips the walk).
- The grant is part of the record. `eval/rivals/extract_grants.py` reads every rival row's
  granted rung and, for Recurrent, its settled `start_point` into
  `eval/rivals/grants/<campaign>.json` (committed); `rivals_{demotic,recurrent}.py --grants FILE`
  pins them (`GRANTS=` for the Colab driver; `run_full_eval.sh` defaults to the committed file) —
  no ladder walk, no start_point probe, one run at the recorded rung, `grant_source` and
  `start_point_source` on the row. A pinned rung this card cannot finish is walked down from, so
  "largest rung that completes" holds on any card; a recorded rival failure has no grant and is
  re-measured, never copied. The walk and the probe were the bulk of a campaign's wall on the
  large circuits and neither is a measurement, so this is what makes a re-run reproducible and
  cheap. Gated by five tests in tests/test_rival_targets.py (ladder top and start, pinned walk
  and descent, the harness pinning both halves, the extractor's keys and the never-pin-a-failure
  rule, the committed file's consistency with the ladder).

Considered, not adopted — the standing list. A future review that proposes one of these gets
pointed here, not a re-run.

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

What still changes numbers after the freeze, legitimately: the cleanup run's rows (item 3,
below), the T=50 appendix rows (additive), and a bug in a number — fixed, regenerated and
recorded here, with the protocol untouched.

The cleanup run, measured (2026-09-06, 09:13–09:52 local, two concurrent Colab L4 sessions, one
VM lifetime each; the campaign README's last section is the record): 17 cells re-walked from the
30,000,000 top. One cell moved: `c432` under DEMOTIC's pins completes the 10,000,000 rung
(30,000,000 OOMs) and DEMOTIC runs 8,952,982 u/s there against 8,157,612 at 3,000,000 — GPAP
9,925,487 u/s, 1.11x with a seed band of 0.98–1.13x: a tie, where the ladder-bound row had read a
1.21x win (DEMOTIC-native `wins` 8/10 → 7/10 + 1 tie; the 2.2x geomean unchanged). Every other
re-walked cell OOMs at 10,000,000 and is granted 3,000,000 again within noise of its earlier row,
so the card and not the ladder bounds them; `c17` completes the 30,000,000 top (89 / 16 u/s) and
is screened as a tiny space on both tables — DEMOTIC's own exception — so it carries no number.
`b22`-rare's GPAP arm on a clean device (23.5 of 23.7 GB free) still delivers 0 models: the
residency was not the cause, the cell is a genuine limit of the comparison, and its caveat is
gone. The GPAP and CMSGen arms of the 17 cells were re-measured with them (the GPAP-arm cache
keys on the harness source, so the ladder change was a miss by design): all within seed noise.
Headlines after the run — vs CMSGen 1,243.7x / 230.4x / 820.4x / 73.2x; vs DEMOTIC 2.2x (7/10,
1 tie); vs Recurrent 21.5x on its table (paper set 10.2x) and 93.9x under our rare pin (49.0x) —
`verify_artifact` 19 ok / 2 warnings / 0 failures, emission audit PASS. The grants file was
regenerated off the merged rows (74 pinned cells, 8 recorded failures). The protocol is frozen
from here.

The T=50 appendix, measured (2026-09-06/07, two concurrent Colab L4 sessions): the rival's own
script depth (`--num_clock_cycles 50`), the identical protocol otherwise, run after the freeze on
the user's go, printed beside the T=25 rows and replacing none. Both Recurrent stages 30/30
(`recurrent-{rare,table}-t50`), 20 measured / 10 init-budget failures. Headlines beside T=25: vs
CMSGen 1,031.7x their pins / 34.5x rare (steady state; 338.6x / 15.9x e2e); vs Recurrent 8.5x
their table (paper set 7.1x) and 77.8x our rare pins (paper set 439.1x); engine advantage 512x /
590x. The `steps-calib-t50` tie-break holds at the deeper unroll — steps 1 best for the rival
(312,421 vs 75,546 at 5, 39,402 at 10), `agrees: true`. The finding the depth carries:
Recurrent's init-budget failures rise from 4 at T=25 (`b17 b18 b19 b22`) to 10 at T=50 (adds
`b14 b15 b20 b21 s35932 s38584`) — six more large ITC'99/ISCAS-89 circuits cannot initialise
their gradient in 1,200 s at 50 cycles, `s35932` among them though it measured at T=25 — so the
T=50 geomeans stand on fewer, harder cells (the table-set ratio 8.5x against T=25's 21.5x while
the rare set stays large). Every failure carries both arms and is recorded as a rival failure,
never a zero. `verify_artifact` 27 ok / 4 warnings / 0 failures (the warnings are the failure
counts, stated); emission audit PASS. The grants file was regenerated to 120 Recurrent cells
(60 @ T=25 + 60 @ T=50) + 22 DEMOTIC, 28 recorded failures, so a re-run pins T=50 too. Several
of these headlines move under §17b; its table is the current one.

Four harness bugs fixed during the T=50 run — bug-not-protocol, per item 5 (tests + VM bootstrap
checks added, host suite 562 + rival/paper 116 green); none changed a metric, config, screen,
budget or cell set, and each appears only when a run outlasts a ~1 h Colab VM:
1. A rival killed with empty stdout+stderr crashed the stage (`[-1]` of an empty list) — now a
   recorded `exit SIGKILL with no output` refusal the ladder walks past (`rivals_recurrent`).
2. The OOM ladder re-walked from the top on every VM reclaim because the ceiling was cached only
   when the walk ended, and on `s15850.1` the walk outlasts the VM — now each refused rung is
   cached as seen and the search budget carries across VMs (`remember_refusal`, `known_spent`,
   `run_at_batch(spent=)`).
3+4. The discarded per-process warmup ran a full O(circuit x T) graph build — ~1,030 s on
   `s15850.1` at T=50, re-paid every reclaim. The rival's per-run cost is its unroll + graph build,
   not its batch, so shrinking the warmup batch (10,000 → 256) did nothing; the fix kills the
   discarded run after `WARMUP_TIMEOUT_S = 60`, all a cold start needs. A warmup measures nothing,
   so neither its batch nor its cap moves a number. Per-cell rival work longer than a VM lifetime
   must checkpoint per step, and a warmup must be time-capped, never sized by the input.

### 17b. Presentation revision v1.1 — the tables, not the protocol (2026-09-07, the user's direction)

With T=50 complete the user turned to the tables: "everything we are doing now is only to modify
a bit and polish how the tables are produced — only how we analyze and present the data, nothing
about how we ran; that's frozen." Item 5's governance binds a review; this is the owner revising
the presentation, recorded here with every number it moved so nothing can look like a silent
re-cut. Nothing was re-run: every row is the frozen campaign's (the paper folder's
`data/MANIFEST.sha256` pins them); only `eval/artifact/paper_tables.py`, `eval/eval_plots.py` and
their tests changed.

What changed, and why (the user's four points, in order):

1. One `excluded` column instead of seven. Table 1 printed `both=0 | no-search | tiny-space |
   vacuous | unsat | CMSGen<30 | refused`, Table 2 `both=0 | u/s unmeasured | rival<30 |
   no-search | tiny-space | vacuous | unsat | misaligned | out-of-envelope` — "too much data" for
   a main table. Every cell that carries no ratio for a reason other than one tool scoring zero is
   now one column, `excluded`, itemised by name under each table in RESULTS.md and as `%` comments
   beside each table in `tables.tex`. The one-sided zero columns (`CMSGen=0`/`GPAP=0`,
   `rival=0`/`GPAP=0`) stay, and Table 2 keeps `rival failed` as its own column: a rival that
   produced no measurement inside the budget is a one-sided outcome (GPAP measured those cells)
   and the T=50 finding lives in it. The per-reason counts and name lists remain in the aggregates
   (`excl_names`), so the caveats, the gate and the tests still see each reason apart. The
   identities a reader adds up are `inst = n + CMSGen=0 + GPAP=0 + excluded` (Table 1) and
   `cells = n + rival=0 + GPAP=0 + failed + excluded` (Table 2). `inst` in Table 1 now counts every
   instance of the set (the refused `b19` is inside `excluded`), so the four Recurrent rows read
   30 alike.
2. `paper set` is gone. "Drop the paper set column, it's redundant, no need to even mention it; we
   take the rival benchmarks, that's it, we test on the whole set." One population per row;
   `PAPER_INSTANCES` stays in `rival_common` for the harnesses' `paper` keyword and nothing in the
   tables reads it. This closes the considered-not-adopted row "the paper set as the headline":
   neither as headline nor beside it.
3. No thin-denominator floor. "I don't like the rival<30 column; we should include that in the
   result — the rival paper includes rows with 0.1 u/s, so we should consider something > 0." The
   30-event floor (§16c item 4, one floor for both sides) is removed on both sides — removing it on
   the rival side alone would have restored the asymmetric ledger it was introduced to close — and
   the disclosure survives: `THIN_DENOMINATOR = 30` names every geomean cell whose CMSGen count or
   rival solution count is under it, with the count and the ratio, in the caveats; a row whose
   geomean stands on one or two cells is flagged ‡ and its caveat says to report the win count,
   not the ratio. The floor was conservative by construction (every cell it removed was a large
   win of ours), so dropping it moves numbers up — every one listed below.
4. The encoder column is dropped where every row shares one (every row samples the `abc:fast`
   CNF; the preamble and caption say so). It reappears by itself if a campaign ever mixes encoders.

Figures, brought in line with the tables. The rival dumbbells (`rivals_<stage>.png`) drew every
row without a rival error — vacuous pins included — so their subtitle geomean disagreed with
Table 2's; they now classify through `rival_cells` (Table 2's row, drawn), name what they do not
draw, and carry the task in the title. Two figures were missing: GPAP vs CMSGen on the converged
protocol's cells (`vs_cmsgen_<stage>.png`, six of them — Table 1's rows, drawn) and every Table 1
cell at once (`sps_scatter_cmsgen.png`), both through the `rival_cms_cells` classifier that
Table 1 and Table 3 now share (Table 3 used to restate the screens and could drift). T=50 cells
are hollow on both scatters. One figure bug: the rival scatter chose its marker by
`proto.startswith("their")`, so "their ISCAS-85, our rare pin" — our task — was drawn as the
rival's; the marker is now the task's.

One folder. `../gpap-results/paper/` holds copies of the nine stage/calibration JSONs and the four
Barbarik JSONL files with a sha256 manifest, `regenerate.sh` (tables, gate, audit, figures, then
the copy into `docs/paper/`), the generated outputs and `plots/`. The campaign directory stays the
protocol record.

What moved (before → after; every move is a thin cell entering a geomean, each named in the
caveats):

| table, row | before | after | cause |
|---|---|---|---|
| T1 Recurrent's set, T=25, rare | 73.2x (n 17), e2e 26.1x, wins 15/18 | **78.0x** (n 18), e2e 29.2x, wins 16/19 | `b14` enters: 12 CMSGen samples in 60 s |
| T1 Recurrent's set, T=50, their pins | 1,031.7x (n 13), e2e 338.6x, wins 23/23 | **1,616.3x** (n 14), e2e 552.8x, wins 24/24 | `b04` enters: 9 CMSGen samples |
| T1 Recurrent's set, T=50, rare | 34.5x (n 13), e2e 15.9x, wins 16/18 | **71.7x** (n 14), e2e 32.7x, wins 17/19 | `b04` enters: 8 CMSGen samples |
| T2 DEMOTIC, our rare pin | — (n 0; 5 rival=0, 2 rival<30), wins 5/5 | **782,616.8x‡** (n 2), wins 7/7 | `c880` (1 solution, 7.1M x) and `c3540` (4, 86k x) enter; flagged ‡, caveat: quote 7/7 |
| T2 Recurrent, our rare pins, T=25 | 93.9x (n 12), wins 15/15 | **173.2x** (n 15), wins 18/18 | `b11` (2 solutions, 40x), `b14` (2, 7,395x), `s4863` (2, 27,722x) enter |
| T2 Recurrent, our rare pins, T=50 | 77.8x (n 12), wins 14/15 | **119.3x** (n 13), wins 15/16 | `s4863` (6 solutions, 20,099x) enters |
| T3 (same rows as T1) | 534x / 512x / 590x engine advantage on those three rows | 513x / 508x / 585x | the same cells entering |

What did not move: Table 1's ISCAS-85 rows (1,243.7x / 230.4x) and Recurrent's-set T=25
their-pins row (820.4x); Table 2's DEMOTIC-native 2.2x (7/10, 1 tie) and both Recurrent-table
rows (21.5x at T=25, 8.5x at T=50); the failure counts (4 → 10 at T=50); the gate (27 ok /
4 warnings / 0 failures, the propagation identity closing on every row's new `n`); the emission
audit (PASS, worst case `s208.1` 36.9%).

How to read the rows that moved. The T=50 their-pins 1,616x is 1,031x plus one cell — `b04`,
where CMSGen delivered 9 samples in 60 s against GPAP's ~80,000/s — and the caveat under Table 1
names it; the T=25 and T=50 rare rows likewise carry one thin cell each. The DEMOTIC rare-pin
782,617x stands on two cells with one and four rival solutions and is printed with the ‡ and a
caveat that says to report 7/7 and the five rival zeros, not the ratio. The rival papers publish
exactly such cells (0.1 unique/s rows), which is the user's ground for scoring them; the ratios on
them are orders of magnitude, and the text must say so where it quotes them.

Governance note. This revision is the owner's, not a reviewer's, and the protocol (what ran, at
what configuration, on which cells, with which budgets) is untouched. The considered-not-adopted
rows "widening the free / tiny / vacuous / thin screens" and "the paper set as the headline" are
amended by it: the thin floor and the paper-set column are gone by the user's direction; the
free / tiny / vacuous / unsat screens, and their mechanical, per-row, before-the-ledger
application, are unchanged. Any further proposal to move a screen goes to that table, as before.
