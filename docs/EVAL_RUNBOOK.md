# Full-evaluation runbook

How to reproduce every experiment behind the paper on any CUDA machine: the benchmark families,
the `abc:fast` encoder, both target regimes, GPAP vs CMSGen, GPAP vs the GPU gradient-descent
rivals, the kernel A/B, the uniformity certificates, and the figures and tables.

## 0. The protocol (converged 2026-09-03)

Three steps, one stage (`eval/run_full_eval.sh campaign`, or `eval/colab/colab_rivals_l4.sh`
on a rented GPU), and every published table is read off its rows:

1. **The benchmarks are the GPU rivals' own published sets.** DEMOTIC (ASP-DAC'25): its 11
   ISCAS-85 combinational circuits, T=1. Recurrent-CircuitSAT (MLCAD'25): the ITC'99 `b*` and
   ISCAS-89 `s*` sequential circuits it evaluates — every one we can process (all of
   `benchmarks/circuits/bench/seq/`, b17-b22 fetched by `scripts/fetch_benchmarks.sh itc99`),
   unrolled T=25, with its paper's 16 (`rival_common.PAPER_INSTANCES`) scored beside the whole
   set. Nothing of ours is added: no EPFL, no HWMCC, no `sk_` CNFs, no `random` targets, no T=50.
2. **Each rival runs on its own set, under two tasks: the pins its own table names, and our
   rare pin** (the rarest output pattern observed in a 262k-sequence simulation, all outputs,
   as a miter) on the same circuits — so across a pair of rows only the task changes. One
   configuration per rival, and it is theirs: its own script's `num_steps` (1 for both), the
   largest batch that fits 0.80 of free VRAM (the share GPAP's K takes), a 300 s budget for
   Recurrent's `start_point` initialization. What a tool cannot process is an error row,
   omitted from the table with the count stated — never a zero.
3. **Every one of those cells is then run by GPAP and by CMSGen on the same constraint**: GPAP
   on the `abc:fast` CNF of that circuit and those pins (the default `circuit` portfolio, 60 s,
   median of seeds 5/6/7, on-device soundness re-check of every emitted model), and CMSGen
   (patched: `--nowrite`, `--maxtime 60`) on that very CNF file (`rivals_*.py --cmsgen`,
   `rival_common.cmsgen_arm`). Three tools, one constraint, one row.

Why this shape: every number is measured on ground a rival chose, the CMSGen comparison and the
rival comparison are the same cells (Table 1 and Table 2 line up instance by instance), and
nothing is selected — one encoder, one config per tool, no best-of. HTS is off (its paper
evaluates no circuit). Uniformity is a separate claim (Barbarik, §7). The older family matrix
(seq/comb/sk/epfl/hwmcc × targets × depths, `gpap bench` on instances pinned our way) still
runs as supplementary stages and `paper_tables` prints its rows only when their JSONs exist.

**Run it from a shell, not a notebook cell.** Every stage checkpoints after each row and resumes;
a Colab cell that looks idle gets reaped, a `nohup`'d process does not.

## 1. Setup

```bash
git clone <repo> gpap && cd gpap
apt-get install -y build-essential cmake zlib1g-dev   # compilers for libgpap / cmsgen / abc
pip install -e ".[dev]"                               # torch + triton come with the Colab image
eval/rivals/rivals_setup.sh      # libgpap FIRST, then abc, patched CMSGen, the 3 rivals — all pinned
eval/run_full_eval.sh check      # readiness table — read it before starting
eval/run_full_eval.sh smoke      # ~5 min: every stage on 1–2 instances, into a separate directory
gpap info                        # device + power state (BENCHMARKING.md §1)
```

The engine is the compiled library `libgpap` ([cpp/README.md](../cpp/README.md)) with no Python
fallback: `rivals_setup.sh` builds it first (`scripts/build_core.sh` — cmake + a C++17 compiler +
the driver + triton at build time; no CUDA toolkit), and every stage refuses to start without it
rather than burn hours producing error rows. CMSGen is built from upstream **with
`scripts/cmsgen_gpap_eval.patch`**, which is what makes the CPU comparison honest (§4). Everything
lands in `.gpap/{bin,src}` beside the repo; nothing is vendored.

### How long it takes

Measured end to end on an RTX 4080 laptop (2026-07-28; 357 rows, 13 stages, 0 errors):

| stage | wall | matrix |
|---|---|---|
| `seq` | ~2 h 10 m | 19 instances × 2 encoders × 2 targets × 2 depths (now 1 encoder ⇒ ~half; 29 instances with b15-b22) |
| `hwmcc` | ~1 h 30 m | 7 × the same matrix |
| `epfl` | ~45 m | 20 × 2 encoders × 2 targets |
| `comb` | ~15 m | 10 × 2 encoders × 2 targets |
| `sk` | ~15 m | 29 CNFs |
| `rivals` / `campaign` | ~3–4 h | four protocols (§6) × ~80 rows; the CMSGen arm adds ~1 min per row |
| `plots` | < 1 m | |

The wall is **budget-bound, not GPU-bound**: most of it is instances sitting at the 60 s
`--time-limit`, plus CMSGen, which is single-core CPU either way. A faster GPU changes the
numbers, not the runtime — budget ~5 h for the five benchmark stages plus the rivals on any card.
To shorten it, cut the matrix (`TARGETS=rare CYCLES=25`), not the per-instance budget.

### A bigger GPU

Expect roughly the SM-count ratio (A100 ~1.8×, L4 ~0.85× of the reference laptop) and not more:
the engine is occupancy-bound and the batch is not a lever (4× the batch bought +2.8%).
`--k-cap auto` resolves per card — a 262,144 floor on every card up to ~64 GB — so **a cross-GPU
A/B pins `--k-cap N` and `--portfolio b8|b1|s1` on every arm** and gives each run several chunks:
a free bandit picks differently per machine, and a one-chunk run measures warmup. One open
anomaly to know about before booking cluster time: b12-c25 loses half its success rate on an
A100 above K≈28k, sound throughout. [findings/CROSS_GPU_FINDINGS.md](findings/CROSS_GPU_FINDINGS.md).

## 2. Run a family

```bash
eval/run_full_eval.sh campaign   # THE PAPER RUN (§0): rivals + GPAP + CMSGen per row, then plots
eval/run_full_eval.sh rivals     # the same without the plots
eval/run_full_eval.sh plots      # figures + summary.md from whatever has finished

# supplementary family sweeps (`gpap bench` vs CMSGen, instances pinned our way; not the paper's set)
eval/run_full_eval.sh seq        # ITC'99 b* + ISCAS-89 s* unrolls
eval/run_full_eval.sh comb       # ISCAS-85, T=1, one pinned output
eval/run_full_eval.sh sk         # benchmarks/sampling/*.cnf vs CMSGen
eval/run_full_eval.sh epfl       # EPFL combinational AIGs
eval/run_full_eval.sh hwmcc      # HWMCC'17 property AIGs
eval/run_full_eval.sh all        # everything, in order

nohup eval/run_full_eval.sh campaign > campaign.log 2>&1 &     # detached, with a log to tail
```

Results go to `.experiments/full_eval/<gpu-slug>/<stage>.json` (+ `logs/`, `plots/`).
**Re-running a stage resumes it** — finished `(instance, encoder, depth, target)` cells are
skipped, so an interruption costs at most the in-flight instance. Run stages **one at a time**:
two concurrent stages contend for VRAM and distort every rate (this hardware spills to host RAM
rather than OOM-ing). Never edit `run_full_eval.sh` while it runs — bash reads a script by byte
offset, and a rewritten file resumes mid-token.

| axis | default | override |
|---|---|---|
| encoders | `abc:fast` only | `ENCODERS="tseitin abc:fast"` runs the A/B pair in ONE process |
| targets | `rare` and `random` | `TARGETS=rare` |
| unroll depth | 25 and 50 | `CYCLES=25` |
| budget | 50k samples, 60 s cap per instance | `N_TIME=`, `TIME_LIMIT=` |
| seeds | one (seed 5) | `SEEDS="5 6 7"` → median-of-3 (BENCHMARKING.md §3) |
| quality pass | off | `N_COV=5000` (2-wise coverage + distinct counts) |

With two encoders both run inside one `gpap bench` invocation, so the A/B pair is measured in
the same process — the paired comparison the encoder plots consume; the campaign default is
`abc:fast` alone, halving the circuit stages. `rare` is the headline regime: at `random`
pins CPU rejection sampling beats every CNF sampler, ours and the rivals', so both are run and the
split is visible per instance (`p_acc` / `rej/s` / `amp`).

### On Colab

The same commands, from Colab's terminal, one family at a time. Two things the platform does: the
VM is reclaimed after ~1 h regardless of activity, and `.experiments/` dies with it — results exist
only once mirrored off the VM. `eval/colab/colab_supervisor.py` runs the loop that survives this
(ensure session → push cached binaries → bootstrap → push partials → launch → poll and PULL every
cycle → reseed a fresh VM), and the `eval/colab/colab_*_l4.sh` drivers are the per-campaign stage
scripts it launches. The rules that came out of it — marker files, not grepped logs; one poller
per session; resume must replace error rows — are in
[findings/L4_EVAL_FINDINGS.md](findings/L4_EVAL_FINDINGS.md) §6 and
[findings/CROSS_GPU_FINDINGS.md](findings/CROSS_GPU_FINDINGS.md) §6.

## 3. What comes out

- `<stage>.json` — schema 3, one row per `(instance, encoder)`, written after every row: the
  sampler numbers, the CMSGen numbers (both emission modes, `cms_count_exact`), the rejection
  baseline, the host `cpu`, and **engine provenance** — the packed split (`16|16` / `8|24`), the
  propagate kernel (`wave` / `scalar`), `engine.K`, `engine.batch`, `engine.k_cap`.
- `plots/` — `encoder_ab_*`, `encoder_size_*`, `vs_cmsgen_*`, `work_rates_*`, `amplification_*`,
  `cmsgen_emission`, `rivals_*`.
- `plots/summary.md` — per-stage geomeans with the degenerate cells excluded, win counts, **and
  every instance that was skipped or errored, with the reason**. Read it before quoting anything.

## 4. The CMSGen protocol

GPAP's timed run emits nothing; stock CMSGen writes every sample as a full N-variable DIMACS
line, which on `27.sk_3_32` turns 0.42 s of sampling into a 3.6 s run. The in-repo patch adds
three off-by-default flags, and the driver auto-detects each:

| flag | what it does | if the binary lacks it |
|---|---|---|
| `--nowrite` | generate, emit nothing — the timed run | row marked `cms_mode="write"`, warning printed, ratio inflated |
| `--statsevery N` | a flushed line with CryptoMiniSat's own propagation/decision counters (`Cdec/s`, `Cimpl/s`) | no work counters |
| `--maxtime N` | CMSGen stops itself at the budget and prints an exact `final=1` count | a killed run reports the last multiple of N — **floored zeros scored as our wins** (81 cells on one campaign) |

`--maxtime` bounds CMSGen's *sampling* wall, the quantity GPAP's `--time-limit` bounds, so
`cms_sps` pairs with GPAP's steady-state rate and `cms_sps_e2e` with `wall_total`. When the
quality pass needs real samples a separate write-mode run provides them (`cms_sps_write` — the
emission-cost appendix, never the headline). Numbers recorded before 2026-08-26 carry one or both
handicaps ([findings/RIVAL_FAIRNESS_FINDINGS.md](findings/RIVAL_FAIRNESS_FINDINGS.md) §7).

## 5. What the engine refuses

Every number comes from the packed sum-trick kernels; there is no other execution path. One int32
word per (row, clause of length ≥ 3) — the low `SHIFT` bits count non-falsified literals, the rest
hold the sum of their indices — and two envelopes keep that exact:

| split | needs | covers |
|---|---|---|
| `16\|16` | 2N ≤ 65,535 and clause length ≤ 65,535 | most instances |
| `8\|24` | 2N ≤ 16,777,215 and clause length ≤ 255 | the big unrolls |

Outside both, `gpap.engine.packed_envelope` rejects the formula; in a sweep that is a `skipped`
row with the reason, never a crash. `--max-clauses` / `--max-vars` add a wall-time guard on top
(the runner defaults to 4M clauses — a 50-cycle b15 parses for minutes). Both appear in
`summary.md`'s "instances not measured" table.

## 6. The GPU rivals

| harness | rival | what it runs |
|---|---|---|
| `eval/rivals/rivals_demotic.py --mode native` | DEMOTIC (ASP-DAC'25) | their ISCAS-85 benchmarks with their own pins; GPAP pins the same bits |
| `eval/rivals/rivals_demotic.py --mode miter` | DEMOTIC | our rare-target unroll as a single-output miter (`eval/rivals/unroll_to_bench.py`) |
| `eval/rivals/rivals_recurrent.py [--rare]` | Recurrent-CircuitSAT (MLCAD'25) | their filename-keyed target table, then our rare pins |
| `eval/rivals/rivals_demotic.py --mode miter -T 1` | DEMOTIC | their ISCAS-85 circuits under OUR rare pin — the same circuits as the native row, so only the task changes |
| ~~`eval/rivals/rivals_hts.py`~~ | HTS | **off by default since 2026-09-02** (`RIVAL_HTS=1`); see below |

**Everything is MEASURED; each rival's own published set is what gets SCORED.** The drivers run
the whole sequential family — all 29 `b*`/`s*` circuits in `benchmarks/circuits/bench/seq/`,
derived by listing that directory, so both drivers sweep the same set and there is no hand-written
list to drift (it used to be 9, of which 6 appear in NEITHER paper). `scripts/fetch_benchmarks.sh`
installs ITC'99 b17-b22 into that same directory; without that step they are silently absent.

Table 2 *scores* every cell that ran, and prints beside it the geomean over the subset each
rival's OWN paper evaluates (`paper set`, from `paper_tables.PAPER_INSTANCES` — one home, shared
with the harnesses' `paper` keyword): DEMOTIC (arXiv 2502.08086 Table II) on all 11 ISCAS-85
circuits, Recurrent-CircuitSAT (arXiv 2502.21226 Table II) on its 8 ISCAS-89 + 8 ITC'99. ⚠️ The
two sets give different headlines and the wider one flatters us: on the 2026-09-02 rows,
Recurrent's rare-pin geomean is **62.1x over their 16 and 97.7x over all b*/s*** (their target
table: 3.0x vs 4.5x). Both are printed; quote one and say which.

**HTS is off by default.** Its paper (arXiv 2502.08673) evaluates 60 instances and not one is a
circuit — random q-family, OR-configuration, `prod-*` and `s15850a_*` CNFs — so in a
circuits-only evaluation it is the only rival that can never be run on its own published ground,
and every cell would be our task on instances we chose. The harness is unchanged and `RIVAL_HTS=1`
restores the stage; restore the row in `paper_tables.RIVALS` too.

Both remaining rivals are PyTorch programs run as subprocesses on the same GPU and scored with **their own**
metric (unique solutions ÷ their `model_runtime`, which excludes their model build). Their pins are
rejection-easy by construction, so `p_acc` is printed beside every row. The protocol invariants —
each was a wrong number once; the measurements are in
[findings/RIVAL_FAIRNESS_FINDINGS.md](findings/RIVAL_FAIRNESS_FINDINGS.md):

- **Seed 5 everywhere** — the sweep, both circuit harnesses' `--target-seed`, the HTS CNF builds
  (`circuit2cnf --seed 5`). A different seed pins a different rare pattern.
- **The parallelism grant** (§2 there): every rival gets the largest batch that fits **0.80 of
  free VRAM** — the share K takes — sized from measured peak memory (`rival_common.fit_batch`,
  `GRAPH_BYTES`), stepping down a rung on a driver refusal; every rung is recorded as
  `batch_sweep`, the budget as `vram_budget_bytes`. Their `--batch_size` defaults are placeholders
  that would hand them ~1/300 of what they reach. A timeout is not a memory refusal and does not
  walk the ladder.
- **The step count is theirs** (§6 there): `--steps auto` runs each rival at the `num_steps` its
  own artifacts specify (`RIVAL_STEPS`: DEMOTIC 1, Recurrent 1), tie-broken in the rival's
  favour, recorded as `steps` / `steps_source`, re-derivable on the campaign's card with
  `eval/rivals/steps_calibration.py`. Never swept — a tool setting is its author's to choose.
- **Emission symmetry** (§1 there): no tool is timed while delivering solutions.
  `eval/rivals/check_emission_symmetry.py --results <campaign>` re-derives it from the rivals' own
  timer scopes, our `collect=False`, and every row's `cms_mode`.
- **The CMSGen arm rides on every rival row** (§14 there, the converged protocol): the harness
  encodes the row's circuit + pins ONCE (`rivals_recurrent.build_arm_cnf`), the GPAP arm samples
  that file and CMSGen (`--cmsgen`, patched, `--nowrite --maxtime`) runs on the same file at the
  same n and budget; the row records `cms_*` under the bench driver's own keys plus the host
  `cpu`/`gpu`. Table 1 is read off these rows (`paper_tables.rival_cms_family`), so it is the
  same cells as Table 2; `verify_artifact.py --rivals` applies the bench-stage gates to them
  (mixed provenance, a crashed subprocess, a floored zero, an unpatched binary all FAIL).
- **Table 2's GPAP arm encodes with `abc:fast`, like Table 1** (§13 there, settled 2026-09-03;
  `gpap.encoder` per row). Fairness is same circuit + same pins: both sides get the identical
  `.bench`, and each compiles it its own way — `abc:fast` is function-preserving over the same
  projected solution space. Rows before that date used Tseitin, **1.04-4.8x slower** for us
  (b12 1.04x, b04 1.25x, b13 2.19x, b14 4.80x), and are superseded. The arm also runs NO
  coverage pass any more: `unique_per_s_est`'s distinct fraction is measured on the timed run's
  own 1,000 kept output rows (`rivals_recurrent.UNIQUE_ROWS`), projection wall subtracted.
- **Recurrent-CircuitSAT is scored on a depth-POOLED superset** — its loop applies the target at
  every depth `start_point..T` and pools unique solutions, an easier task than the fixed-depth
  CNF every other row measures. Generous, kept, stated; `unique_at_T` is recorded per row.
- **Recurrent gets ONE configuration** (§11 there, settled 2026-09-03): the `start_point` search
  is its INITIALIZATION and gets one 300 s budget (= `--timeout`, each probe on what is left),
  then one measured run at the granted batch with its own script's `num_steps=1`. No up-probe of
  the batch, no retry of a timeout at a smaller batch. The failures are named (`no gradient path`
  / `did not initialize: ... 300s initialization budget`), excluded from the geomean, never
  zeros; `paper_tables` prints the count. ⚠️ Measured: b14 at T=25 needs 1,025 s at its granted
  batch (probe alone 743 s) and b17 1,134 s of model build per probe — both are error rows under
  the rule, and the cap is OURS: state it and how many rows it excludes.
- **Constraint alignment is verified**: miter mode asserts p(MITER=1) equals the instance's
  `p_accept`; native mode asserts bit-identical acceptance words (`pin_check_ok`).
- **The same circuit on both sides** (`--circuit-variants raw abc`; the campaign runs `raw` ONLY,
  because `paper_tables` pools every variant in a stage into one geomean): GPAP's ABC arm
  must not face a rival holding the raw circuit, so `rival_common` ABC-optimizes the sequential
  AIG and writes it back as a primitive-gate `.bench`, gated by a bit-exact simulation check.
  Only the sequential pass has a rival-side equivalent — GPAP's post-unroll pass does not, and
  that is stated rather than hidden.
- **The rival stage is mostly our side** (the GPAP arm, `start_point` probes, `sim_baseline`), so
  both are cached under `<output>/.gpap_cache`, keyed on file bytes + task + engine build + GPU
  (`GPAP_RIVAL_CACHE=0` disables); a rival-side re-run then costs only the rival.

Integration quirks already handled: DEMOTIC `exec`s generated Python, so wire names are sanitized
and `BUFF` becomes `BUF` in its copy only; Recurrent's `backward()` needs `start_point` ≥ the
input→output register distance (probed); both step their batch down a ladder on WSL2's "device
not ready".

## 7. The other campaigns

| question | driver | tables / evidence |
|---|---|---|
| WAVE vs scalar kernel, by formula size | `eval/kernel_ab.py` (`eval/colab/colab_kernel_ab_l4.sh`) | `eval/artifact/kernel_ab_tables.py` → `paper/KERNEL_AB.md`; [findings/KERNEL_AB_FINDINGS.md](findings/KERNEL_AB_FINDINGS.md) |
| uniformity certificate (Barbarik, with both controls) | `eval/uniformity/barbarik_screen.py` → `barbarik_gpap.py` → `barbarik_table.py` | [findings/UNIFORMITY_FINDINGS.md](findings/UNIFORMITY_FINDINGS.md) §4b, §6 |
| uniformity against an exact oracle (TV) | `eval/uniformity/fuzz_uniformity.py` (`eval/colab/colab_uniformity_l4.sh`) | UNIFORMITY_FINDINGS.md §4c |
| what the uniformity-safe default costs | `eval/profile_soft_portfolio.py` (any two portfolios, one process); the whole matrix with `--allow-soft` via `eval/colab/colab_cmsgen_soft_l4.sh` | `eval/artifact/soft_price_tables.py` → `paper/SOFT_PRICE.md`; [findings/L4_EVAL_FINDINGS.md](findings/L4_EVAL_FINDINGS.md) §4 |
| do the numbers reproduce on a fresh machine | `eval/colab/colab_recheck_l4.sh` + `eval/recheck_compare.py` | L4_EVAL_FINDINGS.md §5 |
| the rivals' step-count tie-break on this card | `eval/rivals/steps_calibration.py` (`colab_rivals_l4.sh` stage `steps-calib`) | RIVAL_FAIRNESS_FINDINGS.md §6 |

## 8. After the sweep: tables, verification, artifact

Raw results live outside the repo (`$GPAP_RESULTS`, default `../gpap-results`). Every published
table is generated from them — [paper/README.md](paper/README.md) has the exact commands:
`eval/artifact/paper_tables.py` (the ONE place the exclusion rules live), `slide_tables.py` and
`soft_price_tables.py` (which import it), `verify_artifact.py` (the publishability gate — it
re-derives the headline and the propagation identity from the raw rows and fails on a floored
zero or a missing uniformity control), and `make_artifact.sh` (packages a campaign with a manifest
and checksums). The reviewer-facing map of claims to files is [../ARTIFACT.md](../ARTIFACT.md).

## 9. Troubleshooting

| symptom | cause / fix |
|---|---|
| `libgpap ... is not available` / every GPAP row errors | the engine is not built — `scripts/build_core.sh`; verify with `eval/rivals/rivals_setup.sh check` |
| `GPAP_LIB is set but does not exist` | the variable points at a typo; it is refused rather than silently falling back |
| every rate ~halved | power cap or battery clocks — `gpap info` (BENCHMARKING.md §1); and clocks move 2.8× within a run, so never compare rates across runs (§1b) |
| throughput collapses, no OOM | the batch spilled to host RAM — never OOM-probe; use the kill-switch (BENCHMARKING.md §2) |
| `cmsgen=yes [STOCK: ...]` | the binary lacks the patch — `scripts/build_cmsgen.sh` |
| a stage exits instantly | it resumed and everything was done; delete the JSON to force a re-run |
| `SKIPPED: formula outside the packed envelopes` | expected on giant unrolls (§5) |
| rival "device not ready" | a WSL2/driver hiccup between CUDA processes; the harnesses retry and step the batch down |
