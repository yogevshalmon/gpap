# The evaluation protocol of the TACAS 2027 submission (v1.0, frozen)

> **Reader:** readers and reviewers of the TACAS 2027 submission who want the exact rules its
> evaluation was run under. [README.md](README.md) explains them in plainer terms.

This is the frozen text, moved here on 2026-10-17 from the general evaluation guide
([EVAL_RUNBOOK.md](../../EVAL_RUNBOOK.md)), unchanged except that one citation became a link. It
was amended once after the freeze, on 2026-10-12, when Recurrent-CircuitSAT's in-loop copy to the
host was deducted from its rate (§17c). A bare section number (§12, §16d, §17b)
refers to [RIVAL_FAIRNESS_FINDINGS.md](../../findings/RIVAL_FAIRNESS_FINDINGS.md).

One departure, recorded in every row it affects: the campaign whose rows the paper prints carried
over the 28 recorded rival initialization failures instead of re-measuring them, against the last
sentence below. [README.md](README.md) describes that campaign and its three bug fixes.

Final; it does not change. Converged over four adversarial review rounds
([RIVAL_FAIRNESS_FINDINGS.md](../../findings/RIVAL_FAIRNESS_FINDINGS.md) §14–§17), frozen on 2026-09-06. A review may fix a
bug in a number (fix, regenerate, say so) and fix prose; it may not change a metric definition, a
tool configuration, a screen, a budget or the cell set — such a proposal is logged under
"considered, not adopted" in §17 and not acted on. Three steps, one stage
(`eval/run_full_eval.sh campaign`, or `eval/colab/colab_rivals_l4.sh` on a rented GPU), and every
published table is read off its rows.

**1. The benchmarks are the rivals' own published sets.** DEMOTIC (ASP-DAC'25): its 11 ISCAS-85
circuits, T=1. Recurrent-CircuitSAT (MLCAD'25): every ITC'99 `b*` / ISCAS-89 `s*` it evaluates that
we can process — all of `benchmarks/circuits/bench/seq/` (b17-b22 via
`scripts/fetch_benchmarks.sh itc99`; 30 circuits), unrolled at T=25. Their Table II's "s35844" is
no ISCAS-89 circuit; its figures (12 PI / 278 PO / 1,452 DFF) identify `s38584`, which their
paper's 16 hold (`rival_common.PAPER_INSTANCES`, the harnesses' `paper` keyword); `s35932` we add.
The full set is the headline — a partial set is not — and since 2026-09-07 the only population the
tables print, with no paper-subset column (§17b). T=50 is the rival's own script depth: an additive
appendix under the identical protocol (`RIVAL_T2=50`, the `*-t50` stages), printed beside the T=25
rows, never in their place. Nothing of ours is added: no EPFL, no HWMCC, no `sk_` CNFs, no `random`
targets, no second encoder.

**2. Each rival runs on its own set, under two tasks** — the pins its own table names, and our rare
pin (the rarest output pattern of a 262k-sequence simulation, all outputs, as a miter) — so across
a pair of rows only the task changes. Then:

- *Configuration is theirs*: its run script's `num_steps` (1 for both) and `lr` (15 for both). The
  papers' values (DEMOTIC 10 iterations, Recurrent 5 at `lr` 50) are measured beside it by
  `steps_calibration.py` and printed under Table 2 as evidence, never run as configuration.
- *Batch*: the largest rung it runs to completion, walking the decade-and-triple ladder down from
  30,000,000. A rung is rejected only when the tool cannot finish on it (OOM, driver refusal, past
  the wall), never on how fast it was, and `paper_tables` names any cell whose top rung completed
  first try, so a larger card cannot silently re-open the hole.
- *One measured run*, at the tool's own default seed (`RIVAL_SEED = 0`, passed explicitly and
  recorded), the asymmetry against our median-of-3 stated; each process discards its first run
  (cold CUDA context, ~3.3x).
- *Budgets*: 1,200 s wall per run, plus a separate 1,200 s initialization budget for Recurrent's
  `start_point` search. The budget bounds the search and is in no published rate; a cell that
  misses it is reported with the budget named, apart from a capability failure, never a zero.
- *One circuit variant everywhere*: the same sequential-ABC-simplified netlist GPAP encodes from,
  equivalence-checked (§16d; the residual — GPAP's post-unroll pass, which has no sequential
  equivalent — is stated).
- *No measurement is an error row*, omitted with the count and the reason, never a zero.

**3. GPAP and CMSGen then run every one of those cells on the same constraint.** GPAP: the
`abc:fast` CNF of that circuit and those pins, the shipped default `circuit` portfolio (the one
asymmetry — a bandit measuring two uniformity-safe configs inside the wall against rivals that run
one, priced in §12), 60 s, the median-rate run of seeds 5/6/7, an on-device soundness re-check of
every emitted model, kept rows simulated through the original netlist, and an exact dedup of every
delivered row for unique/s, as both rivals compute theirs. CMSGen: patched, `--nowrite --maxtime
60`, on that very CNF file (`rival_common.cmsgen_arm`). Three tools, one constraint, one row.

**What is published, and on which clock.** Table 1 (GPAP vs CMSGen, the same cells): two clocks,
each charging both sides alike — steady state (sampling rate vs CMSGen's samples per sampling wall)
and end to end (our whole wall vs its whole subprocess); quote `e2e` for a delivered-samples claim.
Table 2 (GPAP vs each rival): one clock, the rival's own metric — unique solutions per second of
its `model_runtime`, its model build excluded as our construction is, and Recurrent's in-loop copy
of its candidates to the host excluded as our delivery is (on rows from 2026-10-12;
RIVAL_FAIRNESS_FINDINGS.md §17c) — and
no end-to-end column, by rule (§17). Table 3: the implication rate and the identity splitting engine from rejection cost,
over Table 1's rows. Table 4b: Barbarik CM_test with the `spur` control, the only uniformity claim (the `soft`
positive control is not re-run; it is UNIFORMITY_FINDINGS.md §4b's).
Screens, applied before any ledger and in every table: vacuous pin, free cell, tiny space,
proved-UNSAT, and a tie for any cell whose sign lies inside GPAP's own seed band. Every ledger
column sums to the cell count, and every headline is quoted with its structure (median, seed band,
top cells, the geomean without them), which `paper_tables` prints.

**Presentation revision v1.1** (2026-09-07; §17b) changes what a table
prints, not the protocol: the two one-sided zero columns and `rival failed` stay, every other
reason a cell carries no ratio goes into ONE `excluded` column itemised by name under it; no
`paper set` column; no thin-denominator floor — a tool that scored anything is scored, as the rival
papers print 0.1 u/s rows, and cells under 30 events are named in the caveats.

**Reproducing a campaign pays no search.** Each rival's grant — its batch rung and, for Recurrent,
its `start_point` — is part of the record: `eval/rivals/extract_grants.py` writes
`eval/rivals/grants/<campaign>.json` and both drivers pin it (`GRANTS=…`; `run_full_eval.sh`
defaults to the committed file, `RIVAL_GRANTS=none` walks and probes from scratch). A pinned rung
this card cannot finish on is walked down from, and the row says which happened (`grant_source`).
A cell recorded as a rival failure is re-measured, never copied.
