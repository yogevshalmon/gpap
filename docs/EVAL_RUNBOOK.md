# Running an evaluation campaign

> **Reader:** anyone comparing GPAP with other samplers on their own GPU using `eval/` — setup,
> the stages, what comes out, and what the harnesses guarantee.

`eval/run_full_eval.sh` runs every campaign: it drives GPAP, CMSGen and the GPU samplers over a
benchmark family and writes one row per cell. This guide is for running campaigns of your own. To
reproduce a paper's numbers, use that paper's artifact instead ([papers/](papers/README.md)); to
measure without fooling yourself, read [BENCHMARKING.md](BENCHMARKING.md) first.

## 0. The TACAS 2027 submission's protocol

The `campaign` stage runs the protocol of the TACAS 2027 submission, frozen on 2026-09-06: its
benchmarks, targets, tool settings, batch sizes and budgets, and what its tables publish. The
protocol lives with the paper, in [papers/tacas-27/PROTOCOL.md](papers/tacas-27/PROTOCOL.md). A
campaign of your own can change most of it through the `campaign` stage's settings (§2); the
paper's numbers come only from the frozen version.

## 1. Setup

```bash
git clone https://github.com/yogevshalmon/gpap.git && cd gpap
sudo apt-get install -y build-essential cmake zlib1g-dev   # compilers for libgpap, CMSGen and ABC
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[build,dev,eval]"
eval/rivals/rivals_setup.sh      # fetches the GPU samplers, builds libgpap, ABC and patched CMSGen; all pinned
eval/run_full_eval.sh check      # what is ready; read it before starting
eval/run_full_eval.sh smoke      # about 5 minutes: every stage on one or two instances
gpap info                        # the device and its power state (BENCHMARKING.md §1)
```

The engine is the compiled library `libgpap` ([cpp/README.md](../cpp/README.md)) and has no
Python fallback, so every stage refuses to start without it. CMSGen is built from upstream with
`scripts/cmsgen_gpap_eval.patch`, which makes the CPU comparison fair (§4). To compare GPUs, pin
`--k-cap N` and one portfolio on every arm and run several chunks per arm: a free bandit picks
differently on each machine, and a one-chunk run measures warmup
([findings/CROSS_GPU_FINDINGS.md](findings/CROSS_GPU_FINDINGS.md)).

## 2. Run a family

```bash
eval/run_full_eval.sh campaign   # the paper's protocol (§0): the GPU samplers, GPAP and CMSGen per row, then plots
eval/run_full_eval.sh rivals     # the same without the plots
eval/run_full_eval.sh plots      # figures and summary.md from whatever has finished

# other families: `gpap bench` against CMSGen, on GPAP's own targets
eval/run_full_eval.sh seq        # ITC'99 b* and ISCAS-89 s* unrollings
eval/run_full_eval.sh comb       # ISCAS-85, T=1, one pinned output
eval/run_full_eval.sh sk         # benchmarks/sampling/*.cnf
eval/run_full_eval.sh epfl       # EPFL combinational AIGs
eval/run_full_eval.sh hwmcc      # HWMCC'17 property AIGs
eval/run_full_eval.sh all        # everything, in order

nohup eval/run_full_eval.sh campaign > campaign.log 2>&1 &     # detached
```

Results go to `.experiments/full_eval/<gpu-slug>/<stage>.json`, with `logs/` and `plots/` beside
them. Running a stage again resumes it: finished `(instance, encoder, depth, target)` cells are
skipped, so an interruption costs at most the instance in flight. Run stages one at a time, since
they compete for GPU memory, and never edit `run_full_eval.sh` while it runs: bash reads a script
by byte offset.

The other families (`seq` to `hwmcc`) read these settings:

| setting | default | override |
|---|---|---|
| encoders | `abc:fast` | `ENCODERS="tseitin abc:fast"`, both in one process |
| targets | `rare`, `random` | `TARGETS=rare` |
| unrolling depth | 25 and 50 | `CYCLES=25` |
| budget | 50k samples, 60 s | `N_TIME=`, `TIME_LIMIT=` |
| seeds | seed 5 | `SEEDS="5 6 7"`, the median of three (BENCHMARKING.md §3) |
| quality pass | off | `N_COV=5000` (2-wise coverage and distinct count) |

The `campaign` and `rivals` stages read their own:

| setting | default | what it sets |
|---|---|---|
| `RIVAL_SEQ`, `RIVAL_COMB` | every circuit in `benchmarks/circuits/bench/seq/`; DEMOTIC's 11 | the instance sets |
| `RIVAL_T`, `RIVAL_T2` | `25`, `none` | the unrolling depth; `RIVAL_T2=50` adds the T=50 stages |
| `RIVAL_SEEDS` | `5 6 7` | GPAP's seeds; the median is reported, while a GPU sampler runs once |
| `RIVAL_STEPS` | `auto` | each GPU sampler's step count, from its own scripts |
| `RIVAL_TIMEOUT`, `RIVAL_INIT_BUDGET` | `1200`, `1200` | seconds per GPU-sampler run, and for Recurrent's start-point search |
| `RIVAL_GRANTS` | `eval/rivals/grants/l4_protocol_20260904.json` | the batch sizes to pin; `none` searches them on this card |
| `RIVAL_HONOUR_FAILURES` | `0` | `1` carries over the failures the grants file records instead of re-measuring them |
| `RIVAL_VARIANTS`, `RIVAL_DEMOTIC_MITER`, `RIVAL_HTS` | `abc`, `0`, `0` | the circuit variant, and two optional stages |

The default grants are the batch sizes measured on a 24 GB NVIDIA L4, and a pinned size is only
ever walked down from. On a larger card the GPU samplers would stay at the L4's sizes while GPAP
uses the whole card, so search them again there (`RIVAL_GRANTS=none`). Re-measuring the recorded
failures costs their full initialization budget each, about half of a full campaign;
`RIVAL_HONOUR_FAILURES=1` skips that, and every row it affects says `not_remeasured`. The GPU
samplers' configuration calibration (`eval/rivals/steps_calibration.py`) is not part of the
stage; `eval/colab/colab_rivals_l4.sh` runs it beside the campaign on Colab.

Rare targets are the regime that tests a sampler: at random targets, CPU rejection sampling beats
every CNF sampler, ours and the GPU samplers alike. A Colab VM is reclaimed after about an hour
whatever it is doing, so `eval/colab/colab_supervisor.py` mirrors every cycle and starts a fresh
one ([findings/L4_EVAL_FINDINGS.md](findings/L4_EVAL_FINDINGS.md) §6).

## 3. What comes out

`<stage>.json` holds one row per `(instance, encoder)`, written after every row: the sampler's and
CMSGen's numbers (both emission modes, `cms_count_exact`), the rejection baseline, the host `cpu`
and the engine's provenance (packed split, propagate kernel, `engine.K`, `engine.batch`,
`engine.k_cap`). `plots/summary.md` carries the geometric means with degenerate cells excluded,
the win counts, and every instance skipped or failed with its reason; read it before quoting a
number.

## 4. The CMSGen patch

GPAP's timed run emits nothing, while stock CMSGen writes every sample as a full DIMACS line. The
patch adds three flags, off by default and each detected automatically:

- `--nowrite` generates samples without writing them: the timed run. Without it the row is marked
  `cms_mode="write"`, and the ratio is inflated in GPAP's favour.
- `--statsevery N` prints CryptoMiniSat's own propagation and decision counters every N samples.
- `--maxtime N` stops CMSGen itself and prints an exact `final=1` count. A run killed from outside
  reports only the last multiple of the `--statsevery` cadence, and its floored zeros would score
  as wins for GPAP.

`--maxtime` bounds CMSGen's sampling time, the quantity GPAP's `--time-limit` bounds, so
`cms_sps` pairs with GPAP's steady-state rate and `cms_sps_e2e` with `wall_total`. Why each flag
matters: [findings/RIVAL_FAIRNESS_FINDINGS.md](findings/RIVAL_FAIRNESS_FINDINGS.md) §7.

## 5. What the engine refuses

Every number comes from the packed sum-trick kernels; there is no other execution path. Two
layouts keep the one 32-bit word per (row, clause of length ≥ 3) exact: `16|16` for 2N ≤ 65,535
and clause length ≤ 65,535, and `8|24` for 2N ≤ 16,777,215 and clause length ≤ 255. Outside them
`gpap.engine.packed_envelope` rejects the formula, as do the `--max-clauses` and `--max-vars`
guards (4M clauses by default). In a sweep such an instance becomes a `skipped` row that names
the reason, never a crash.

## 6. The GPU samplers

| harness (in `eval/rivals/`) | sampler | what it runs |
|---|---|---|
| `rivals_demotic.py --mode native` | DEMOTIC (ASP-DAC'25) | its ISCAS-85 set, its own targets |
| `rivals_demotic.py --mode miter` | DEMOTIC | our rare-target unrolling as a miter (`unroll_to_bench.py`) |
| `rivals_demotic.py --mode miter -T 1` | DEMOTIC | its ISCAS-85 set under our rare target |
| `rivals_recurrent.py [--rare]` | Recurrent-CircuitSAT (MLCAD'25) | its target table, then our rare targets |
| `rivals_hts.py` | HTS | off by default: `RIVAL_HTS=1` adds its stage, and `paper_tables.RIVALS` its table row |

The drivers sweep the whole sequential family by listing `benchmarks/circuits/bench/seq/`, so both
see the same set and no hand-written list can drift. HTS is off because its paper (arXiv
2502.08673) evaluates 60 instances and not one is a circuit netlist, so it is the only GPU sampler
with no published benchmarks of the kind the other two define.

The rules that keep the comparison fair, each of which once fixed a wrong number, are in
[findings/RIVAL_FAIRNESS_FINDINGS.md](findings/RIVAL_FAIRNESS_FINDINGS.md). Every row uses seed 5
(a different seed pins a different rare pattern), checks that the constraint matches
(`pin_check_ok`; a failure puts the row in the `misaligned` bucket and fails `verify_artifact`),
and replays GPAP's samples through the original circuit (`sim_check`). The GPAP arm and its
probes are cached under `<output>/.gpap_cache`.

Quirks handled: DEMOTIC `exec`s generated Python, so wire names are sanitized and `BUFF` becomes
`BUF` in its copy; Recurrent's `backward()` needs a `start_point` at least the register distance
from inputs to outputs, which the harness probes.

## 7. Results, tables, and the other campaigns

A run writes under `.experiments/`; copy a finished campaign to `$GPAP_RESULTS` (by default
`../gpap-results`, outside the repository) to keep it. Every table is generated from those rows.
`eval/artifact/paper_tables.py` generates the tables and is the one place the exclusion rules
live; `verify_artifact.py` checks a campaign before its numbers are published; and
`make_artifact.sh` packages a campaign's rows with the code for archiving. The first two print
their usage with `--help`; `make_artifact.sh` describes its own in its header. These scripts were
written for the TACAS 2027 submission: its artifact guide,
[papers/tacas-27/AE_README.md](papers/tacas-27/AE_README.md), runs them through one entry point,
`eval/artifact/reproduce.sh`, which calls this driver for every measurement. The evidence behind
that layer is in [findings/ARTIFACT_FINDINGS.md](findings/ARTIFACT_FINDINGS.md).

The other campaigns:
- the kernel A/B (`eval/profiling/kernel_ab.py`,
  [findings/KERNEL_AB_FINDINGS.md](findings/KERNEL_AB_FINDINGS.md));
- the Barbarik uniformity test and the exact-oracle distance fuzz (`eval/uniformity/`,
  [findings/UNIFORMITY_FINDINGS.md](findings/UNIFORMITY_FINDINGS.md) §4b, §4c, §6);
- the price of the uniformity-safe default (`eval/profiling/profile_soft_portfolio.py`) and the
  fresh-machine re-check (`eval/colab/colab_recheck_l4.sh`, compared by
  `verify_artifact.py --recheck`), both in [findings/L4_EVAL_FINDINGS.md](findings/L4_EVAL_FINDINGS.md)
  §4-§5;
- the GPU samplers' step-count calibration (`eval/rivals/steps_calibration.py`).
