# Rival-comparison fairness: what we were measuring, and what we should have been

> **Reader:** contributors and researchers — how the comparison with other samplers was made fair, and the record of the paper's evaluation protocol. A dated research log, not a user guide.
>
> **Status:** settled. §17 is protocol v1.0, frozen 2026-09-06 (`docs/papers/tacas-27/PROTOCOL.md`
> is the frozen text). The paper reports its 2026-10-12 re-run (§17e); every earlier headline is
> superseded. Condensed 2026-10-21; the full earlier text is
> `git show 823eefd:docs/findings/RIVAL_FAIRNESS_FINDINGS.md`.
> **Measured:** 2026-08-24 to 2026-10-13; RTX 4080 Laptop for calibration, NVIDIA L4 (Colab) for
> every campaign.
> **Rule lives in:** `eval/rivals/rival_common.py`, `eval/artifact/paper_tables.py`,
> `eval/run_full_eval.sh`, `scripts/build_cmsgen.sh`.
> **Gated by:** `tests/experiments/test_rival_targets.py`, `tests/experiments/test_paper_tables.py`.
> Code and the generated paper text cite sections by number, so they are never renumbered; a
> missing number is a removed section.

Every audit here found a default that had flattered GPAP.

## 1. Emission symmetry

No tool's rate includes delivering its samples: GPAP runs with `collect=False`, CMSGen with
`--nowrite`, DEMOTIC and HTS dedup after their timer stops, and Recurrent's in-loop host copy is
measured and deducted (§17c). `eval/rivals/check_emission_symmetry.py` re-derives this from the
rivals' source and the rows.

## 2. The batch handed to a rival is its score

A rival's unique/s is near-proportional to its batch (HTS 295× from 100 to 100,000 rows). Rule
(`rival_common.run_at_batch`): the largest rung of the ladder the rival completes, walked down
from the top (30,000,000), rejected on feasibility only (OOM, a driver refusal, the wall budget),
never on throughput.

## 4. A re-run pays only for the rival

Our side of a cell (the GPAP arm, Recurrent's `start_point` probe, `sim_baseline`) is cached
under `<output dir>/.gpap_cache`, keyed on the sampled file, the task, the engine build, the GPU,
the harness sources and the `abc` binary. A timed-out probe is never cached.

## 6. `num_steps` comes from the rival's own artifacts

Our pinned `--num_steps 10` had taxed Recurrent 8.50× and DEMOTIC 2.61×. Rule
(`rival_common.RIVAL_STEPS`): the value the rival's own scripts ship, and where its argparse
default differs, the one better for it (DEMOTIC 1, Recurrent 1, HTS 10); re-derived on each
campaign's card by `steps_calibration.py`, never swept.

## 7. CMSGen's count must be exact

A CMSGen killed on its budget reported its last `--statsevery` multiple: 81 of 357 rows read `0`
and scored as GPAP wins. `--maxtime` in `scripts/cmsgen_gpap_eval.patch` ends it at the deadline
with an exact count; `cms_count_exact` is recorded per row, and `verify_artifact.py` fails a
floored zero.

## 8. No encoder is chosen by the ratio

Picking, per instance, the encoder with the best ratio let the rival's number choose which of
our runs was reported (1.24–1.33× on the sequential families). Rule: one row per (family,
encoder).

## 9. Screen first, then bucket

A degenerate cell leaves through its own column and takes no part in the geomean, the zero
ledger or the win count, so `inst = n + CMSGen=0 + GPAP=0 + excluded` (§17b;
`test_every_instance_is_counted_exactly_once`).

## 11. A rival failure at our budget is about our budget

Recurrent's `start_point` probe was capped at 60 s, and every way it can fail read "no gradient
path": a claim about the rival, produced by our cap (given time, `b14` runs). Rule
(`probe_start_point`): the search is the rival's initialization and gets one budget, 1,200 s
(§16d); its three failure modes are named apart; a failed cell is counted and kept out of the
geomean, never scored as a zero.

## 12. GPAP's bandit, priced and kept

The rivals and CMSGen run one configuration; GPAP's portfolio is a bandit that reports its
winner's rate. Pinning `b8` would cost 4.5–8.7%. Kept (§17, item 4) and disclosed in every row
(`portfolio`, `adaptive`); `--portfolio b8` is the knob.

## 13. The GPAP arm encodes with `abc:fast`

As in Table 1, so Table 2 does not run GPAP on a slower Tseitin CNF (it had, 1.04–4.8× slower).
Each tool compiles the same circuit into its own representation, and every row records
`gpap.encoder`. The arm runs no separate coverage pass (its host reconstruction ran out of
memory): its distinct fraction comes from the timed run's own rows, with the projection wall
(`out_project_s`) subtracted.

## 14. Three tools on one constraint

The instances are the rivals' own published sets, under their pins and under our rare pin. Each
cell runs the rival on its circuit, GPAP on that circuit's `abc:fast` CNF, and CMSGen on the same
file (`rival_common.cmsgen_arm`), so Tables 1 and 3 are read off Table 2's rows. Every row
asserts the constraint matches (`native_pin_check`, `miter_self_check`) and replays GPAP's
samples through the original circuit (`sim_check`).

## §16 — Four review rounds (2026-09-04/06)

Each round re-derived every number from the rows, and every defect that moved a number had
flattered GPAP: an under-granted rival (the walk started at our VRAM model's rung; it starts at
the top now, §2), a glob that published our bug as a rival failure, a cold start inside a timed
run, a wrong substitute circuit. Their rules are §16d, §16e and §17.

### 16d. The converged decisions (2026-09-05)

1. Recurrent's initialization gets 1,200 s per instance, counted in no published rate.
2. Each rival runs its shipped run-script configuration (`num_steps` 1, `lr` 15), measured better
   for it than its paper's (`RIVAL_PAPER_CONFIG`, printed, not run).
3. One circuit variant, `abc`, for every tool: the ABC-simplified netlist helps and hurts each
   side on different cells, so the choice favours no one.

### 16e. The ladder top (2026-09-06)

A top rung that completes first try means the ladder, not the card, bounded the grant. The top
is 30,000,000 since §17, and `paper_tables` names any cell that completes it.

## 17. Protocol v1.0 — the freeze (2026-09-06)

Four review rounds each redefined a published column, and a redefinition always has a fairer
next version, so the protocol froze. Principle: publish only quantities each tool defines for
itself, every tool at its authors' shipped configuration, the batch by one rule (§2). Decided
(`docs/papers/tacas-27/PROTOCOL.md` is the frozen text):
1. The headline is the full set: 30 sequential circuits and 11 ISCAS-85.
2. T=25 is the paper's depth; T=50 is an additive appendix.
3. One cleanup run before the freeze, then nothing.
4. GPAP keeps its shipped portfolio (§12).
5. A review may fix a bug in a number or the prose, never a metric, configuration, screen,
   budget or cell set.

Table 2 has one clock, the rivals' own `model_runtime`; its end-to-end column was removed by
rule. The grant is part of the record: `eval/rivals/extract_grants.py` writes each rival row's
rung and `start_point` to `eval/rivals/grants/<campaign>.json`, and `--grants FILE` pins them. A
recorded rival failure is re-measured unless a run opts in to pinning it (§17e).

Considered, not adopted; a review that proposes one of these is pointed here:

| proposal | why not |
|---|---|
| another end-to-end definition for Table 2 | the column is removed; the rivals define one clock |
| pin GPAP to `b8` | every tool at its shipped default; the price is measured (§12) |
| the rivals' paper configurations | measured beside the script setting and printed (§16d) |
| the raw circuit, or the better of raw and `abc` for the rival | one variant everywhere (§16d) |
| more rival seeds | one run at its own default (`RIVAL_SEED = 0`), the asymmetry stated |
| HTS | its paper evaluates no circuit |
| a TV uniformity number in the tables | uniformity is Barbarik with both controls, only |
| moving a screen after seeing the rows | the screens are mechanical, per row, before the ledger |
| a taller ladder, or the rival's fastest rung | the top is above the card's reach; the grant is a share of the machine |

Checked and clean, do not re-litigate: emission symmetry, the batch grant, zero-ledger symmetry,
the propagation identity, soundness, pin alignment, ABC equivalence, per-row provenance, the
GPAP-arm cache key.

### 17b. Presentation revision v1.1 (2026-09-07)

The tables only; nothing was re-run. One `excluded` column, itemised under each table, instead
of seven; no paper-set column; no thin-denominator floor, with `THIN_DENOMINATOR = 30` still
naming every thin cell and ‡ flagging a row that stands on one or two cells; the encoder column
dropped where every row shares one.

### 17c. Recurrent's host copy is deducted (2026-10-12)

Recurrent copies its candidates to the host inside the loop its own timer wraps, which lowered
its rate. `eval/rivals/recurrent_gpap_eval.patch` times the copy as `egress_runtime`, and the
harness rates unique / (`model_runtime` − `egress_runtime`). On `s386` the deduction matches a
copy-free run within 0.3%, with identical solutions. The harness refuses an unpatched checkout,
and `compare_campaign.py` keeps both sides on one clock.

### 17d. The row decides vacuity (2026-10-12)

`vacuous_pin` read its fallback name list before the row's `p_accept`, so `b02` and `b07` at T=50
(0.375 and 0.4985) were screened as tautologies. The row decides now
(`test_a_named_pin_that_measures_samplable_is_not_screened`).

Open, for the next campaign: `p_accept >= 1.0` is an exact test on an estimate, and `b05` has
measured 0.9999998. A tolerance of ~1e-6 sits in a 90× gap in the measured values and catches
only `b05`. Tightening a screen is a metric change, so it is logged here, not taken.

### 17e. The re-run the paper reports: l4_protocol_20261012 (2026-10-12/13)

Protocol v1.0 on the fixed tree (`paper_campaign_v2_run`, e821180) after §17c, §17d and the
give-up fix (GPAP's zero-yield early stop off). Every rival rung and `start_point` was pinned
from `eval/rivals/grants/l4_protocol_20260904.json`, and the 28 recorded rival failures were
carried over (`not_remeasured`). Every measured rival count matched the 2026-09-04 rows.
Recurrent's rate rose by its copy share, so its Table 2 ratios fell (21.5→18.7× on its own table
at T=25), and GPAP now loses four small circuits there. The rows ship as
`artifact-data/paper/data/`.

### 17f. Presentation revision v1.2 (2026-10-21)

The tables only; nothing was re-run, and `paper_numbers.json` is unchanged but for its date.
`RESULTS.md` prints the paper's own columns and, under each table, every row's identity
(`n + C=0 + G=0 + Fail + not counted`) with each cell not counted named; §17b's `excluded` is
`not counted`. The caveats, the rival calibration and the top-rung check moved to the authors'
notes, `paper_tables.py --out-notes`, which the artifact does not run.
