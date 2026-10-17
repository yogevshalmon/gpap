# Full-evaluation runbook

> **Reader:** people reproducing the TACAS 2027 evaluation, or running a campaign of their own — the frozen protocol and how to run it.

How to reproduce every experiment behind the paper on any CUDA machine.

## 0. The TACAS 2027 paper's protocol

The protocol of the TACAS 2027 paper, frozen on 2026-09-06, lives with the paper, in
[papers/tacas-27/PROTOCOL.md](papers/tacas-27/PROTOCOL.md). The `campaign` stage (§2) runs it.

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
`make_artifact.sh`. The map of claims to files is [papers/tacas-27/ARTIFACT.md](papers/tacas-27/ARTIFACT.md).

**Reproducing this campaign is `eval/artifact/reproduce.sh`, not this driver.** It is the
artifact's single entry point — `check` / `setup` / `tables` / `smoke` / `subset` / `full` — and
it invokes this script for every measurement, so a tier cannot drift from the protocol. What it
adds is a cell set, an estimate of what that costs (`estimate_wall.py`, read off the recorded
rows), two documented shortcuts that this driver does not take (the recorded rival failures are
pinned rather than re-derived, and the engine's memory budget is capped to the reference card's)
and the cross-card gate `compare_campaign.py`. Evidence:
[findings/ARTIFACT_FINDINGS.md](findings/ARTIFACT_FINDINGS.md).

The other campaigns: the kernel A/B (`eval/profiling/kernel_ab.py`,
[findings/KERNEL_AB_FINDINGS.md](findings/KERNEL_AB_FINDINGS.md)); the Barbarik certificate and the
exact-oracle TV fuzz (`eval/uniformity/`,
[findings/UNIFORMITY_FINDINGS.md](findings/UNIFORMITY_FINDINGS.md) §4b, §4c, §6); the price of the
uniformity-safe default (`eval/profiling/profile_soft_portfolio.py`) and the fresh-machine re-check
(`eval/colab/colab_recheck_l4.sh`, compared by `verify_artifact.py --recheck`), both in
[findings/L4_EVAL_FINDINGS.md](findings/L4_EVAL_FINDINGS.md) §4-§5; and the rivals' step-count
tie-break (`eval/rivals/steps_calibration.py`).
