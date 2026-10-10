# Full-evaluation runbook

How to reproduce every experiment behind the paper on any CUDA machine.

## 0. The protocol — v1.0, FROZEN 2026-09-06

Final; it does not change. Converged over four adversarial review rounds
(findings/RIVAL_FAIRNESS_FINDINGS.md §14–§17), frozen by the user on 2026-09-06. A review may fix a
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
its `model_runtime`, its model build excluded as our construction is — and no end-to-end column, by
rule (§17). Table 3: the implication rate and the identity splitting engine from rejection cost,
over Table 1's rows. Table 4b: Barbarik CM_test with both controls, the only uniformity claim.
Screens, applied before any ledger and in every table: vacuous pin, free cell, tiny space,
proved-UNSAT, and a tie for any cell whose sign lies inside GPAP's own seed band. Every ledger
column sums to the cell count, and every headline is quoted with its structure (median, seed band,
top cells, the geomean without them), which `paper_tables` prints.

**Presentation revision v1.1** (2026-09-07, the user's direction; §17b) changes what a table
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

## 1. Setup

```bash
git clone <repo> gpap && cd gpap
apt-get install -y build-essential cmake zlib1g-dev   # compilers for libgpap / cmsgen / abc
pip install -e ".[dev]"                               # torch + triton come with the Colab image
eval/rivals/rivals_setup.sh      # libgpap FIRST, then abc, patched CMSGen, the rivals — all pinned
eval/run_full_eval.sh check      # readiness table — read it before starting
eval/run_full_eval.sh smoke      # ~5 min: every stage on 1–2 instances, in its own directory
gpap info                        # device + power state (BENCHMARKING.md §1)
```

The engine is the compiled library `libgpap` ([cpp/README.md](../cpp/README.md)) with no Python
fallback, so every stage refuses to start without it; CMSGen is built from upstream with
`scripts/cmsgen_gpap_eval.patch`, which is what makes the CPU comparison honest (§4). A cross-GPU
A/B pins `--k-cap N` and one portfolio on every arm and runs several chunks per arm: a free bandit
picks differently per machine, and a one-chunk run measures warmup
([findings/CROSS_GPU_FINDINGS.md](findings/CROSS_GPU_FINDINGS.md)).

## 2. Run a family

```bash
eval/run_full_eval.sh campaign   # THE PAPER RUN (§0): rivals + GPAP + CMSGen per row, then plots
eval/run_full_eval.sh rivals     # the same without the plots
eval/run_full_eval.sh plots      # figures + summary.md from whatever has finished

# supplementary sweeps (`gpap bench` vs CMSGen, pinned our way; not the paper's set)
eval/run_full_eval.sh seq        # ITC'99 b* + ISCAS-89 s* unrolls
eval/run_full_eval.sh comb       # ISCAS-85, T=1, one pinned output
eval/run_full_eval.sh sk         # benchmarks/sampling/*.cnf
eval/run_full_eval.sh epfl       # EPFL combinational AIGs
eval/run_full_eval.sh hwmcc      # HWMCC'17 property AIGs
eval/run_full_eval.sh all        # everything, in order

nohup eval/run_full_eval.sh campaign > campaign.log 2>&1 &     # detached
```

Results go to `.experiments/full_eval/<gpu-slug>/<stage>.json` (+ `logs/`, `plots/`), and
re-running a stage resumes it — finished `(instance, encoder, depth, target)` cells are skipped, so
an interruption costs at most the in-flight instance. Run stages one at a time (they contend for
VRAM) and never edit `run_full_eval.sh` while it runs: bash reads a script by byte offset.

| axis | default | override |
|---|---|---|
| encoders | `abc:fast` | `ENCODERS="tseitin abc:fast"` — the A/B pair in ONE process |
| targets | `rare`, `random` | `TARGETS=rare` |
| unroll depth | 25 and 50 | `CYCLES=25` |
| budget | 50k samples, 60 s | `N_TIME=`, `TIME_LIMIT=` |
| seeds | seed 5 | `SEEDS="5 6 7"` → median-of-3 (BENCHMARKING.md §3) |
| quality pass | off | `N_COV=5000` (2-wise coverage + distinct) |

`rare` is the headline regime: at `random` pins CPU rejection sampling beats every CNF sampler,
ours and the rivals'. A Colab VM is reclaimed after ~1 h whatever it is doing, so
`eval/colab/colab_supervisor.py` mirrors every cycle and reseeds a fresh one
([findings/L4_EVAL_FINDINGS.md](findings/L4_EVAL_FINDINGS.md) §6).

## 3. What comes out

`<stage>.json` — schema 3, one row per `(instance, encoder)`, written after every row: the sampler
and CMSGen numbers (both emission modes, `cms_count_exact`), the rejection baseline, the host `cpu`
and engine provenance (packed split, propagate kernel, `engine.K`, `engine.batch`, `engine.k_cap`).
`plots/summary.md` carries the geomeans with degenerate cells excluded, the win counts and every
instance skipped or errored with its reason — read it before quoting.

## 4. The CMSGen protocol

GPAP's timed run emits nothing; stock CMSGen writes every sample as a full N-variable DIMACS line.
The in-repo patch adds three off-by-default flags, each auto-detected: `--nowrite` (emit nothing —
the timed run; without it the row is marked `cms_mode="write"` and the ratio is inflated),
`--statsevery N` (CryptoMiniSat's own propagation/decision counters) and `--maxtime N` (it stops
itself and prints an exact `final=1` count; a killed run reports the last multiple of N, and such
floored zeros score as our wins). `--maxtime` bounds CMSGen's sampling wall, the quantity GPAP's
`--time-limit` bounds, so `cms_sps` pairs with our steady-state rate and `cms_sps_e2e` with
`wall_total`. Numbers from before 2026-08-26 carry one or both handicaps
([findings/RIVAL_FAIRNESS_FINDINGS.md](findings/RIVAL_FAIRNESS_FINDINGS.md) §7).

## 5. What the engine refuses

Every number comes from the packed sum-trick kernels; there is no other execution path. Two
envelopes keep the one int32 word per (row, clause of length >= 3) exact: `16|16` for 2N <= 65,535
and clause length <= 65,535, `8|24` for 2N <= 16,777,215 and clause length <= 255. Outside them
`gpap.engine.packed_envelope` rejects the formula, as do the `--max-clauses` / `--max-vars` guards
(default 4M clauses) — in a sweep a `skipped` row naming the reason, never a crash.

## 6. The GPU rivals

| harness (in `eval/rivals/`) | rival | what it runs |
|---|---|---|
| `rivals_demotic.py --mode native` | DEMOTIC (ASP-DAC'25) | their ISCAS-85 set, their own pins |
| `rivals_demotic.py --mode miter` | DEMOTIC | our rare-target unroll as a miter (`unroll_to_bench.py`) |
| `rivals_demotic.py --mode miter -T 1` | DEMOTIC | their ISCAS-85 set under our rare pin |
| `rivals_recurrent.py [--rare]` | Recurrent-CircuitSAT (MLCAD'25) | their target table, then our rare pins |
| ~~`rivals_hts.py`~~ | HTS | off since 2026-09-02; `RIVAL_HTS=1` restores the stage, `paper_tables.RIVALS` its row |

The drivers sweep the whole sequential family by listing `benchmarks/circuits/bench/seq/`, so both
see the same set and no hand-written list can drift. HTS is off because its paper (arXiv 2502.08673)
evaluates 60 instances and not one is a netlist, so it is the only rival that cannot be run on its
own published ground two ways.

The invariants that make §0's rules real are in
[findings/RIVAL_FAIRNESS_FINDINGS.md](findings/RIVAL_FAIRNESS_FINDINGS.md), each a wrong number
once: emission symmetry (§1, re-derived by `check_emission_symmetry.py`), the parallelism grant
(§2, `rival_common.fit_batch` sizing the batch to 0.80 of free VRAM), the step count from
`RIVAL_STEPS` (§6), Recurrent's single configuration and init budget (§11), the `abc:fast` GPAP arm
with per-row `unique_per_s_est` instead of a coverage pass (§13), and the CMSGen arm on every rival
row (§14). Two hold per row: seed 5 everywhere, since a different seed pins a different rare
pattern; and constraint alignment (`pin_check_ok`), whose failure lands in the `misaligned` bucket
and fails `verify_artifact`. Kept GPAP rows are also simulated through the original circuit
(`sim_check`), and the GPAP arm and its probes are cached under `<output>/.gpap_cache`.

Quirks handled: DEMOTIC `exec`s generated Python, so wire names are sanitized and `BUFF` becomes
`BUF` in its copy; Recurrent's `backward()` needs `start_point` >= the input-to-output register
distance (probed).

## 7. After the sweep, and the other campaigns

Raw results live outside the repo (`$GPAP_RESULTS`, default `../gpap-results`) and every published
table is generated from them; [papers/tacas-27/README.md](papers/tacas-27/README.md) has the exact commands for
`eval/artifact/paper_tables.py` (the one place the exclusion rules live), `slide_tables.py` and
`soft_price_tables.py` (which import it), `verify_artifact.py` (the publishability gate) and
`make_artifact.sh`. The map of claims to files is [../ARTIFACT.md](../ARTIFACT.md).

**Reproducing this campaign is `eval/artifact/reproduce.sh`, not this driver.** It is the
artifact's single entry point — `check` / `setup` / `tables` / `smoke` / `subset` / `full` — and
it invokes this script for every measurement, so a tier cannot drift from the protocol. What it
adds is a cell set, an estimate of what that costs (`estimate_wall.py`, read off the recorded
rows), two documented shortcuts that this driver does not take (the recorded rival failures are
pinned rather than re-derived, and the engine's memory budget is capped to the reference card's)
and the cross-card gate `compare_campaign.py`. Evidence:
[findings/ARTIFACT_FINDINGS.md](findings/ARTIFACT_FINDINGS.md).

The other campaigns: the kernel A/B (`eval/kernel_ab.py`,
[findings/KERNEL_AB_FINDINGS.md](findings/KERNEL_AB_FINDINGS.md)); the Barbarik certificate and the
exact-oracle TV fuzz (`eval/uniformity/`,
[findings/UNIFORMITY_FINDINGS.md](findings/UNIFORMITY_FINDINGS.md) §4b, §4c, §6); the price of the
uniformity-safe default (`eval/profile_soft_portfolio.py`) and the fresh-machine re-check
(`eval/colab/colab_recheck_l4.sh` + `eval/recheck_compare.py`), both in
[findings/L4_EVAL_FINDINGS.md](findings/L4_EVAL_FINDINGS.md) §4-§5; and the rivals' step-count
tie-break (`eval/rivals/steps_calibration.py`).
