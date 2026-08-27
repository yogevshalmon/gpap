# Full-evaluation runbook (Colab / any CUDA box, from a terminal)

The complete experimental sweep: every benchmark family, both circuit encoders, both target
regimes, GPAP vs CMSGen vs the GPU gradient-descent rivals, plus the figures.

**Run it from a shell, not a notebook cell.** Colab reaps a runtime whose notebook UI looks idle
and a long-running cell dies with it; a process started in the terminal (or under `nohup`)
survives, and every stage here checkpoints after each row anyway.

---

## 1. One-time setup on the machine

```bash
git clone <repo> gpap && cd gpap
apt-get install -y build-essential cmake zlib1g-dev   # compilers for libgpap/cmsgen/abc
pip install -e ".[dev]"                # torch+triton come with the Colab image
eval/rivals/rivals_setup.sh                   # libgpap FIRST, then abc, patched CMSGen, the 3 rivals
eval/run_full_eval.sh check            # readiness table — read it before starting
eval/run_full_eval.sh smoke            # ~5 min: every stage, 1-2 instances. Do this FIRST.
gpap info                              # device + POWER STATE (docs/BENCHMARKING.md §1)
```

**The engine is a compiled library now** (`libgpap`, see [cpp/README.md](../cpp/README.md)) and
has NO Python fallback, so `rivals_setup.sh` builds it first via `scripts/build_core.sh` and
aborts if that fails — without it every GPAP row of every stage errors out. `run_full_eval.sh`
re-checks before each stage and refuses to start rather than burn hours producing error rows;
`check` reports it as `libgpap (THE engine)`. If you ever build it by hand, `scripts/build_core.sh`
is the one command (cmake + a C++17 compiler + the driver + triton; no CUDA toolkit).

`rivals_setup.sh` builds CMSGen from upstream **with `scripts/cmsgen_gpap_eval.patch`** applied.
That patch is what makes the CPU comparison honest — see §4. Everything lands in
`.gpap/{bin,src}` beside the repo; nothing is vendored into the repo. `smoke` writes to a separate
`<gpu>-smoke/` directory, so it can never be mistaken for (or resumed as) real results.

### How long it takes, and why the GPU barely matters

Measured end-to-end on an RTX 4080 laptop (2026-07-28, 357 rows, 13 stages, 0 errors):

| stage | wall | note |
|---|---|---|
| `seq` | **2 h 10 m** | 19 instances × 2 encoders × 2 targets × 2 depths |
| `hwmcc` | 1 h 26 m | 7 instances × the same matrix |
| `epfl` | 46 m | 20 instances × 2 encoders × 2 targets |
| `comb` | 14 m | 10 instances × 2 encoders × 2 targets |
| `sk` | 13 m | 29 CNFs, CMSGen timed twice (`--cmsgen-both`) |
| `rivals` | ~2–3 h | with the settings below; was 7 h before they were tuned |
| `plots` | < 1 m | |

**The wall is cap-bound, not GPU-bound**: most of it is instances sitting at the 60 s
`--time-limit` (and CMSGen, which is single-core CPU either way). A faster GPU changes the
*numbers*, not the runtime — do not expect an A100 to finish this in an hour. Budget **~5 h for
the five benchmark stages** plus the rivals, on any machine. If you need it shorter, cut the
matrix (`TARGETS=rare CYCLES=25`) rather than the per-instance budget.

### Batch size on a bigger GPU (read before booking cluster time)

**Measured on Colab 2026-08-03 (A100-40GB + L4), not reasoned about.** The short version: the
`--k-cap` ceiling is NOT where cross-GPU scaling comes from, and an earlier draft of this section
claimed it was. It is wrong and the numbers below are why.

`--k-cap` defaults to `auto` (VRAM-proportional, `gpap.engine.calibrate.default_k_cap`; exact
no-op at or below the 12 GB reference card). Treat it as hygiene — removing an arbitrary
constant — **not as a throughput lever**:

| | 4080 laptop | L4 24 GB | A100 40 GB |
|---|---:|---:|---:|
| auto k_cap | 49,152 | 90,300 | 161,851 |
| models/s vs 4080 at **matched** K | 1.00 | **0.874** | **1.710** |
| auto-K ÷ pinned-K (same GPU, geomean) | — | 0.985 | **1.012** |

Raising the cap bought **nothing** (±1.5%), and pushing `K_fused` to 4× its default moved
throughput 2.8% at 8.8 GB of 41 GB used. Two reasons, and they compose:

1. **The cap only binds for small formulas.** K is `min(analytic, cap)`, and the analytic VRAM
   bound falls below the cap once M ≳ 20k (M=100k: 9,696 on the 4080, 37,440 on the A100). Every
   real circuit unroll is far past that, so the cap is irrelevant there — and the analytic bound
   **already scales with free VRAM by itself**, needing no fix.
2. **Where the cap does bind, the GPU is already saturated.** Those are exactly the small-M
   instances, and they hit occupancy limits well below 262,144 rows.

So the cross-GPU win is the hardware, not the batch.

#### Why more K does not mean more throughput

**The clean experiment is the same-GPU one above**: one A100, one formula, `K_fused` 262,144 →
1,048,576, everything else fixed → **+2.8%**. At 262,144 rows over the A100's ~6,912 resident
warps we are already ~38× oversubscribed, so extra rows queue rather than run. That is the whole
K result, and it is the only measurement here in which K is the only thing that moved.

#### The corroborated cross-GPU numbers (pinned config)

**Always pin the config for a cross-GPU or before/after run** — `--portfolio b1|b8|s1` are
single-config portfolios that leave the bandit nothing to select. The bandit optimises models/wall
with no stability term, so it picks differently per machine (measured: b12-c25 chose B=1 on the
4080 but B=8 on an A100) and a comparison then silently A/Bs the bandit.

Also **give the run enough budget to span several chunks.** At `--n-time 20000` against a 262,144-
row batch the small instances finish in ONE chunk, so the "rate" is first-chunk overhead; that
measurement read A100 2.42×/L4 1.30×, and both moved by ~30% once multiple chunks ran. The numbers
below use `--n-time 2000000 --time-limit 20`, `--k-cap 49152` pinned on all arms, both configs,
two reps with the order alternated, 4 sk_ CNFs + b03/s1196/b12/b14 at T=25 rare:

| vs RTX 4080 laptop | **models/s** | work rate (assigns/s) | success |
|---|---:|---:|---:|
| **L4 24 GB** | **0.838×** | 0.838× | 1.006 |
| **A100 40 GB** | **1.822×** | 1.905× | 0.956 |

Two things this settles:

* **Success is now machine-INDEPENDENT — exactly 1.000 on 13 of 14 cells.** That was the whole
  confound; with it gone, `models/s ≈ work rate` and the per-instance spread collapses to
  1.58–2.32× on the A100 (it was 1.19–2.37× with the bandit free). b12 in particular goes from a
  misleading 1.19× to a normal 1.85×.
* **The work rate is roughly FLAT in M** (sk_ 1.58–2.02, b03 1.90, s1196 1.71–2.31, b12 1.91–2.05),
  not rising as an earlier draft of this section claimed from the confounded data. Flat, and close
  to the A100's 1.86× SM-count advantage rather than its ~3.6× bandwidth advantage — so
  **occupancy/compute-bound, not bandwidth-bound**, and the "it depends on M" correction in that
  draft was itself an artifact. The L4 is genuinely *slower than the laptop* (same 58 SMs, lower
  clock), confirming it is not an L40S stand-in.

**Budget a big GPU at ~1.8×, an L4 at ~0.85×.** b14 (M_eff 917k) is below measurement resolution
on all three (0/2/5 models in 20 s) and is excluded from the geomeans.

> ⚠️ **OPEN, unexplained**: on b12 the A100 shows a reproducible ~2× success drop (0.530 → 0.265)
> that appears between K=20,000 and K=28,000 and then stays flat to K=49,152. It is not noise
> (0.2649–0.2653 across seven K values), not the config (pinned), not the miner (the L4 learned an
> almost identical unit count at 0.530), and **not an int32 `K*M` overflow** — that was the obvious
> hypothesis and a K-ladder falsified it: the drop happens at `K*M ≈ 0.9 × 2^31` and does not
> worsen at 1.55×. Soundness is unaffected (`gpu_sound=True` throughout) — we get half as many
> models, not wrong ones. The L4 at K=27,760 shows 0.530 while the A100 at K=28,000 shows 0.265,
> so it is machine- AND K-dependent. Worth chasing before a cluster run, since it costs 2× on
> exactly the large-M instances a big GPU is bought for.

The measurement rule still holds: **an A/B across two GPUs must pin `--k-cap N` on both arms**,
because `auto` deliberately differs per card and would confound "faster GPU" with "larger batch".
Read the resolved ceiling back from each row's `engine.k_cap`.

## 2. Run a family

```bash
eval/run_full_eval.sh seq        # ITC'99 b* + ISCAS-89 s* unrolls  (the flagship family)
eval/run_full_eval.sh comb       # ISCAS-85, T=1, one pinned output (DEMOTIC's setup)
eval/run_full_eval.sh sk         # benchmarks/sampling/*.cnf vs CMSGen (CNF reference point)
eval/run_full_eval.sh epfl       # EPFL combinational AIGs
eval/run_full_eval.sh hwmcc      # HWMCC'17 property AIGs
eval/run_full_eval.sh rivals     # DEMOTIC / Recurrent-CircuitSAT / HTS head-to-heads
eval/run_full_eval.sh plots      # figures + summary.md from whatever has finished
eval/run_full_eval.sh all        # all of the above, in order
```

Detached, with a log to watch:

```bash
nohup eval/run_full_eval.sh seq > seq.log 2>&1 &
tail -f seq.log
```

Results go to `.experiments/full_eval/<gpu-slug>/<stage>.json` (+ `logs/`, `plots/`).
**Re-running a stage resumes it** — finished `(instance, encoder, depth, target)` cells are
skipped, so an interruption costs at most the in-flight instance.

> **Never edit `eval/run_full_eval.sh` while it is running.** bash reads a script incrementally
> by byte offset; rewriting it under a live process makes it resume mid-token and die with a
> syntax error. Edit, then restart the stage — `--resume` makes that cheap. (This is how the
> first full local sweep lost its last stage.)

### The matrix each circuit stage runs

| axis | default | override |
|---|---|---|
| encoders | `tseitin` and `abc:fast`, back-to-back in ONE process | `ENCODERS="tseitin abc:full"` |
| targets | `rare` and `random` | `TARGETS=rare` |
| unroll depth | `25` and `50` | `CYCLES=25` |
| budget | 50k samples, 60 s cap per instance | `N_TIME=`, `TIME_LIMIT=` |
| seeds | one (seed 5) | `SEEDS="5 6 7"` → median-of-3 (BENCHMARKING §3) |
| quality pass | off | `N_COV=5000` (adds 2-wise coverage + distinct counts) |

Both encoders run inside a single `gpap bench` invocation so the A/B pair is measured on the same
GPU in the same process — the paired comparison the plots consume.

`--target rare` is the headline regime: at `random` pins, CPU rejection sampling beats every CNF
sampler (ours and the rivals'), so a random-target win is not a sampling result. Both are run so
the split is visible; `p_acc` / `rej/s` / `amp` columns quantify it per instance.

## 2b. Colab, concretely

Colab's own terminal (Pro) or a `!`-cell that backgrounds the process; either way the job must
outlive the browser tab.

```bash
# --- terminal, once -------------------------------------------------------------
cd /content && git clone <repo> gpap && cd gpap
apt-get -qq install -y build-essential cmake zlib1g-dev >/dev/null   # libgpap/cmsgen/abc deps
pip install -q -e ".[dev]"
eval/rivals/rivals_setup.sh && eval/run_full_eval.sh check   # builds libgpap first; check must show it
eval/run_full_eval.sh smoke                       # ~5 min sanity pass, separate output dir

# --- the sweep, one family per command, detached --------------------------------
nohup eval/run_full_eval.sh seq    > seq.log    2>&1 &   # ~2 h  (start here: the flagship)
nohup eval/run_full_eval.sh sk     > sk.log     2>&1 &   # ~15 m (the CMSGen reference point)
nohup eval/run_full_eval.sh comb   > comb.log   2>&1 &   # ~15 m
nohup eval/run_full_eval.sh hwmcc  > hwmcc.log  2>&1 &   # ~1.5 h
nohup eval/run_full_eval.sh epfl   > epfl.log   2>&1 &   # ~45 m
nohup eval/run_full_eval.sh rivals > rivals.log 2>&1 &   # ~2-3 h
eval/run_full_eval.sh plots                              # after any subset
```

Run them **one at a time** — they share the GPU, and two concurrent stages would contend for
VRAM and silently distort every rate (this box spills to host RAM rather than OOM-ing).

Colab-specific gotchas:

- **Persist the results before the runtime dies.** `.experiments/` lives on the ephemeral disk:
  `cp -r .experiments/full_eval /content/drive/MyDrive/gpap_eval/` after each stage (or mount
  Drive and pass `OUT=/content/drive/MyDrive/gpap_eval/<gpu>`).
- **Disk**: converted CNFs are deleted after each row by default (a full circuit matrix is tens
  of GB). Keep it that way unless you are debugging (`--keep-cnf`).
- **Check the readiness table**: without the patched CMSGen, `check` prints
  `[STOCK: rates include writing samples]` and every ratio will be inflated (§4).
- **A disconnect costs at most one instance** — re-issue the same command and it resumes.

## 3. What comes out

- `<stage>.json` — schema 3, one row per `(instance, encoder)`, written after every row.
  Each row carries the sampler numbers, the CMSGen numbers (both emission modes), the rejection
  baseline, and **engine provenance**: which packed split (`16|16` or `8|24`) and which propagate
  variant (`wave`/`scalar`) produced it, plus the calibrated `K`/`K_fused`.
- `plots/` — `encoder_ab_*`, `encoder_size_*`, `vs_cmsgen_*`, `work_rates_*`,
  `amplification_*`, `cmsgen_emission`, `rivals_*`, and `summary.md`.
- `summary.md` — per-stage geomeans, win counts, **and every instance that was skipped or
  errored, with the reason**. A sweep that silently dropped instances is not a result; read this
  table before quoting anything.

## 4. The CMSGen protocol (why the ratios changed)

GPAP's timed run **emits nothing**: models stay on the GPU, and delivery is a separate,
off-the-timing-path pass. Stock CMSGen formats and writes every sample as a full N-variable
DIMACS assignment. On `27.sk_3_32`, 50k samples: **0.42 s generating vs 3.6 s with
`--samplefile`** — 373 MB of output, ~88% of its wall. Comparing GPAP-without-emission to
CMSGen-with-emission measures file I/O, not sampling.

The in-repo patch adds three off-by-default flags to upstream CMSGen:

- `--nowrite` — generate, emit nothing. This is the timed run the ratio uses.
- `--statsevery N` — a flushed, parseable line with CryptoMiniSat's own propagation/decision/
  conflict counters, so the `Cdec/s` and `Cimpl/s` columns are the CPU sampler's real work rates
  and survive a run killed on the time budget.
- `--maxtime N` — **CMSGen stops itself at the budget and exits normally**, printing a final
  `... final=1` stats line with an EXACT count. Without it the harness has to kill the process,
  and what survives is the last `--statsevery` multiple: on the 2026-08-23 campaign that floored
  all 174 timed-out rows to a multiple of 1000, **81 of them to a clean `0` that was then scored
  as an outright GPAP win** (one re-measured cell really delivers 941 samples in the budget — a
  ~10-25x loss published as an infinite win). It also recovers CMSGen's work counters on cells
  where it never reaches its first cadence line, which used to drop those instances out of the
  propagation table entirely. Every row records `cms_count_exact`; `verify_artifact.py` FAILS a
  stage that publishes a floored zero. See docs/findings/RIVAL_FAIRNESS_FINDINGS.md §7.

`--maxtime` bounds the SAMPLING wall — the same quantity GPAP's `--time-limit` bounds — so the
subprocess kill remains only as a backstop for parsing, with a grace margin. Both rate columns
then pair like with like: `cms_sps` (samples per CMSGen's own sampling wall) against GPAP's
steady-state rate, and `cms_sps_e2e` (per subprocess wall, parse included) against GPAP's
`wall_total`.

The driver auto-detects all three (`--maxtime` separately, since a binary patched before
2026-08-26 has the other two without it). With a **stock** binary it falls back to the samplefile run and
marks the row `cms_mode="write"` (and prints a warning) rather than folding the handicap into the
ratio silently. When the quality pass needs actual samples, a *separate* write-mode run provides
them and its rate is reported as `cms_sps_write` — the emission-cost appendix, never the headline.

**Numbers published before this change compared GPAP against CMSGen-with-emission and are
inflated by roughly that factor.** Re-run before quoting.

## 5. The engine, and what it refuses

Every number comes from the packed sum-trick Triton kernel — there is no other execution path.
One int32 word per (row, clause): low `SHIFT` bits are the count of non-falsified literals, the
high bits the sum of their literal indices mod 2^(32-SHIFT). Two conditions keep that exact, and
they are the entire overflow story:

| split | needs | covers |
|---|---|---|
| `16\|16` | 2N ≤ 65,535 and max clause length ≤ 65,535 | most instances |
| `8\|24` ("wide") | 2N ≤ 16,777,215 and max clause length ≤ 255 | the big unrolls |

A formula outside both is **rejected** (`gpap.engine.packed_envelope`). In a sweep that is a
`skipped` row with the reason, never a crash: `gpap bench` pre-checks the DIMACS header and also
catches the engine's own rejection. `--max-clauses` / `--max-vars` add a wall-time guard on top
(the runner defaults to 4M clauses) — a 50-cycle b15 unroll parses for minutes before it samples.
Both appear in `summary.md`'s "instances not measured" table.

## 6. Rivals

| harness | rival | what it runs |
|---|---|---|
| `eval/rivals/rivals_demotic.py --mode native` | DEMOTIC (ASP-DAC'25) | their ISCAS-85 benchmarks with their own hard-coded output pins; GPAP pins exactly the same bits |
| `eval/rivals/rivals_demotic.py --mode miter` | DEMOTIC | OUR rare-target unrolled instance, handed to them as a single-output miter (`eval/rivals/unroll_to_bench.py`, which self-checks p(MITER=1) == p_accept) |
| `eval/rivals/rivals_recurrent.py [--rare]` | Recurrent-CircuitSAT (MLCAD'25) | their filename-keyed target table, then our rare pins |
| `eval/rivals/rivals_hts.py '<glob>.cnf'` | HTS | the very CNFs the `seq` stage generated |

All three are PyTorch programs run as subprocesses on the same GPU, scored with **their own**
metric (unique solutions ÷ their `model_runtime`, which excludes their model build). Their pins
are rejection-easy by construction, so `p_acc` is printed alongside every row.

Protocol invariants (each one was a wrong number once):

- **Every target/conversion seed is 5** — the sweep's, both circuit harnesses' `--target-seed`,
  and the HTS CNF builds (`circuit2cnf --seed 5`, now also its default). The HTS arm once built
  its CNFs at the old `circuit2cnf` default of 1: a different seed pins a different rare
  pattern, and seed-1 b13 happened to be a ZERO-SEARCH instance whose two cells carried ratios
  up to 4.7e9x (re-measured 2026-08-23; the seed-1 rows are kept as `_superseded/`).
- **Recurrent-CircuitSAT is scored on a depth-POOLED superset**: their loop applies the target
  at every unroll depth `start_point..T` and pools unique solutions, so their count covers an
  easier task than the fixed-depth CNF every other row measures. Generous, kept, and stated;
  the fixed-depth-only count is recorded per row as `unique_at_T`.
- **Constraint alignment is verified, not assumed**: miter mode asserts p(MITER=1) equals the
  instance's p_accept; native mode asserts bit-identical acceptance words between their
  index-table pins on the original file and our reordered pins (`pin_check_ok`; all 10
  recorded ISCAS-85 cells re-verified 2026-08-23).
- **THE PARALLELISM GRANT — the batch handed to a rival IS its score, so it cannot be a
  constant.** Measured on an RTX 4080 (2026-08-25), every rival's unique/s is near-proportional
  to its batch across the whole range the card will run, with `model_runtime` almost flat:

  | rival | instance | batch | unique/s | factor |
  |---|---|---|---|---|
  | HTS | b03-c25-rare | 100 → 100,000 | 1 → 294 | **295x** |
  | Recurrent | s386 T=25 | 10k → 300k | 3.8k → 69.9k | **18.6x** |
  | DEMOTIC | c432 | 1k → 300k | 2.5k → 654k | **258x** |
  | DEMOTIC | c1355 | 1k → 1M | 1.2k → 662k | **564x** |

  The batch is the same knob `K` is for us, and GPAP sizes `K` at **0.80 of free VRAM**.
  **THE RULE (2026-08-25): every rival gets the largest batch that fits in 0.80 of free VRAM** —
  the same share of the same card, by the same rule, that we grant ourselves
  (`rival_common.fit_batch`). One rule, one run per cell; if the driver refuses the size
  ('device not ready' reproduces at a fixed size on some models) `run_at_batch` steps down a rung.
  Every rung tried is recorded as `batch_sweep`, the resolved budget as `vram_budget_bytes`.
  ⚠️ **Their own "defaults" cannot serve as that rule, and using them would flatter US.** All
  three argparse defaults are `--batch_size 10`, a placeholder; their published demo scripts are
  per-instance one-liners with a 1000x spread on comparably tiny circuits — DEMOTIC 1,000 on
  c17's six gates, Recurrent 1,000,000 on s27's ten, HTS 100 on a 75-variable CNF. Transplanted
  onto our instances those hand DEMOTIC ~1/500 and HTS ~1/295 of what their tool actually
  reaches on the card, inflating our ratios by the same factor.
  ⚠️ **What was wrong before 2026-08-25**: the circuit rivals' budget was a fixed 5 GiB whatever
  the card (a fifth of the L4 while GPAP took ~19 GB of it), and HTS ran at batch 100/1000. The
  new rule reproduces roughly what DEMOTIC and Recurrent already got, so **the correction is
  concentrated on HTS** — whose recorded geomean is inflated by about the batch factor. Rival
  rows recorded before that date are NOT comparable to rows from the current harness.
  **Where our responsibility ends.** None of the three rivals auto-tunes its batch to the card —
  GPAP does (`compute_k_safe`), they take whatever `--batch_size` they are given. So the fair
  thing is to hand them the same share of the machine we take, sized from MEASURED peak VRAM so
  the grant does not push them into a spill, and to step down once if the driver refuses the
  size (a step-down can only help them). Past that, memory management is the tool's own
  concern: we are not obliged to hand-tune a rival past what its own share of the card allows,
  and a refused rung is recorded as what it is. Being strict here is what makes the number
  reproducible — the alternative is a per-instance knob nobody can re-derive.
- **THE STEP COUNT IS THEIRS, NOT OURS.** `--num_steps` is the rivals' other first-order knob,
  and the sweep drivers used to pin it at 10 for all three. Measured (RTX 4080, at each rival's
  granted batch), that pin taxed **Recurrent 8.50x** (s386: 18.1k -> 153.7k unique/s at steps=1)
  and **DEMOTIC 2.61x** (c432: 650k -> 1.70M), because both SATURATE their batch on the very
  first step — Recurrent finds 2,198,776 unique at steps=1 against 2,134,454 at steps=10, for
  14.3 s of runtime against 118.0 s. HTS is the opposite: it finds 0 / 33 / 1,850 / 6,664 unique
  at steps 1 / 5 / 10 / 20, so a low value cripples it.
  **THE RULE (2026-08-26): each rival runs at the `num_steps` its OWN artifacts specify**
  (`rival_common.RIVAL_STEPS`, `--steps auto`, the default). Where a tool ships two — an argparse
  default and a different value in its demo script — take the one better FOR THE RIVAL: DEMOTIC 1
  and Recurrent 1 (their `script/run_sampler_*.sh`), HTS 10 (its `flags.py` default, since its own
  script's 5 is far worse for it). Recorded per row as `steps` / `steps_source`, and re-derived on
  the campaign's card by `eval/rivals/steps_calibration.py` (the `steps-calib` stage).
  ⚠️ We deliberately do NOT sweep it and keep the best, the way the batch ladder does: a batch is
  a share of the machine and must be granted, a step count is an author-chosen tool setting, and
  tuning past what the authors publish invites "why did you stop at 20?". HTS's optimum on our
  instances is ≥20 and we run it at 10 — its own default, and NOT its best.
  ⚠️ **Rival rows recorded on or before 2026-08-26 used the pinned 10 and are superseded.**
  Local re-measurement moves DEMOTIC's c432 cell from 4.6x to ~1.1x and turns Recurrent's s386
  "their table" cell into a 0.2x LOSS. See docs/findings/RIVAL_FAIRNESS_FINDINGS.md §6.
  ⚠️ Size from MEASURED peak VRAM, not the graph formula: `n_gates * T * batch * 4` undercounts
  by 2-6x, tool-dependent (Recurrent s386@300k: 4.44 GiB estimated, 8.36 measured; DEMOTIC
  c432@1M: 0.60 vs 2.97; HTS 8.4 B per clause-row). `rival_common.GRAPH_BYTES` carries the
  measured factors, and they are used AS MEASURED — rounding 1.88 up to 2.0 costs Recurrent a
  whole ladder rung (300k -> 100k, 3x of its throughput) for nothing.
- **The rival stage is mostly NOT the rival**: of the 2026-08-23 campaign's ~3.3 h, the three
  rivals' own subprocesses were 64 min. The rest is our side — the GPAP arm (3 seeds x 18 cells
  for Recurrent alone), the `start_point` probes, `sim_baseline` — and all of it used to be
  re-paid whenever a stage re-ran for a RIVAL-side reason. Both are cached now, on disk under
  `<output dir>/.gpap_cache`: the **GPAP arm** keyed on the sampled file's BYTES + the task + the
  engine build + the GPU name (anything that could move the number is a miss), and Recurrent's
  **start_point probe** keyed on the circuit + depth (it is pure, yet each probe pays a full
  process launch and model build). `GPAP_RIVAL_CACHE=0` disables both. A probe that TIMED OUT is
  deliberately not cached — that is about the machine, not the circuit. The cache is carried
  BETWEEN VMs as `gpap_cache.tgz` (`colab_rivals_l4.sh` re-packs it after every stage).
  ⚠️ `colab_supervisor.pull_results` used to validate EVERY mirrored file as JSON, which silently
  discarded any non-JSON artifact — the cache tarball was downloaded and deleted on every cycle.
  It now validates by type (JSON must parse, a tarball must open, anything else must be
  non-empty) and still rejects a download caught mid-write, which is what the check was for.
- **EMISSION SYMMETRY — no tool is timed while delivering its solutions.** Ours is the reason
  CMSGen is timed with the patch's `--nowrite` (a stock binary writing every sample cost 8.2x of
  its wall on 27.sk_3_32), so the same has to hold for the GPU rivals or the ratio is partly a
  measurement of their PCIe traffic. Re-derive it, don't assume it:
  `python eval/rivals/check_emission_symmetry.py --results .experiments/<campaign>` AST-scans the
  function each rival's own `timer(...)` wraps for host transfers and IO, checks that every
  `.sample()` in our timed window passes `collect=False`, and audits `cms_mode` over the recorded
  rows (its GPAP half is also a runtime gate:
  `test_timed_run_never_gathers_a_model_to_the_host` disarms `var_bits` and re-runs). Verdict on
  the pinned commits: DEMOTIC and HTS time the descent loop alone and count/dedup solutions
  *after* the clock stops; all 737 recorded CMSGen rows are `nowrite`; our timed run gathers
  nothing (`keep_rows` is False). Note the asymmetry that runs against US, and is likewise left
  alone: our timed loop pays a per-chunk on-device soundness check no rival performs at all.
  **One exception — Recurrent-CircuitSAT**, which snapshots its per-depth candidate assignments
  to the host inside its timed loop (`run.py:186`, `|PI| x sum(d+1 for d in start_point..T-1)`
  strided copies = 0.5-4.0 GiB per run). MEASURED, not modelled (RTX 4080, `--num_steps 10`, by
  timing that block directly): **0.70% (b03@100k), 1.32-1.42% (s386@100k), 0.88% (s1196@30k) of
  `model_runtime`; bound <=~2.2% on the recorded cells.** Removing the copy — snapshots kept on
  the device, which is semantically identical since their `_check_solution` does
  `par.to(self.device)` on each — leaves `num_unique_solutions` **bit-identical** (1,995,117 /
  2,134,454 / 662,936) and saves **1.41% on s386 over 4 interleaved repetitions**, against a
  1.32% meter reading. ⚠️ One un-repeated on/off pair cannot resolve this: run-to-run spread is
  the size of the effect (the first pairs read -0.23 s, +3.11 s, +0.06 s).
  **DELIBERATELY NOT CORRECTED.** It is one rival, worth ~1%, and it moves the ratio in OUR
  favour — deducting it would raise our number, so leaving it keeps the rival's rate exactly as
  published and the harness simple. Instrumenting it (meter + on/off switch + a rival-only
  re-measure path) was built, validated, and reverted for that reason; if it is ever wanted
  again, the mechanism is a timed block around `run.py:186` reported on stdout.

**Both sides get the same circuit.** GPAP's circuit numbers are reported on two encodings (native
Tseitin and ABC), and DEMOTIC/Recurrent consume `.bench` and unroll it themselves — so comparing
our ABC arm against a rival that only ever saw the raw circuit would credit the encoder to the
sampler. `--circuit-variants raw abc` runs each instance twice, handing the rival the
ABC-optimized circuit too (`strash; zero; dc2; dc2; fraig`, written back as a primitive-gate
`.bench` via AIGER — abc's own `write_bench` emits `LUT`/`DFFRSE`, which no parser here reads).
The swap is gated by a **bit-exact** check: both circuits are simulated on identical stimuli and
must accept exactly the same input sequences for the pinned target. HTS, a CNF consumer, gets
the tseitin and ABC CNFs instead. The runner passes `raw abc` by default (`RIVAL_VARIANTS=`).

Honest limit: the rivals can only be given the SEQUENTIAL optimization — GPAP's ABC encoder also
runs a pass *after* unrolling (where the cross-frame collapse lives), and that has no sequential
equivalent. Measured, the sequential-only pass is small (b03 129→127 ANDs, s386 166→113), and it
moves the comparison in both directions: on ISCAS-85, DEMOTIC gains 15–25% from it, while c1355
flips from a GPAP loss (0.5×) to a win (1.5×).

Integration quirks already handled: DEMOTIC exec's generated Python, so wire names must be valid
identifiers (the classic ISCAS-85 files name wires `1`, `223`, …) and `BUFF` must be spelled
`BUF` — `sanitize_bench()` alpha-renames their copy only. Recurrent's `backward()` needs
`start_point` ≥ the input→output register distance (probed). Both need a batch that fits the
unrolled autograd graph; this box spills to host RAM instead of OOMing, so the batch is computed,
then stepped down a ladder on "device not ready".

## 7. Troubleshooting

| symptom | cause / fix |
|---|---|
| `libgpap ... is not available` / every GPAP row errors | the engine was not built — `scripts/build_core.sh` (needs cmake + a C++17 compiler + triton). Verify with `eval/rivals/rivals_setup.sh check`. |
| `GPAP_LIB is set but does not exist` | the env var points at a typo; unset it or fix the path (it is refused rather than silently falling back to another build) |
| every rate ~halved | GPU power cap or battery clocks — `gpap info` (BENCHMARKING §1); and note SM clocks move 2.8x WITHIN a run, so do not compare rates across runs (BENCHMARKING §1b) |
| throughput collapses, no OOM | K too large → host-RAM spill. Never OOM-probe; use the nvidia-smi kill-switch (§2) |
| `cmsgen=yes [STOCK: ...]` warning | binary lacks the patch — `scripts/build_cmsgen.sh` |
| a stage exits instantly | it resumed and everything was already done; delete the JSON to force a re-run |
| `SKIPPED: formula outside the packed envelopes` | expected on giant unrolls; see §5 |
| rival "device not ready" | WSL2/driver hiccup between CUDA processes; the harnesses retry and step the batch down |
