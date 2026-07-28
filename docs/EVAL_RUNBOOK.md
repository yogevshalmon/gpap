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
pip install -e ".[dev]"                # torch+triton come with the Colab image
eval/rivals_setup.sh                   # abc, patched CMSGen, DEMOTIC, Recurrent-CircuitSAT, HTS
eval/run_full_eval.sh check            # readiness table — read it before starting
eval/run_full_eval.sh smoke            # ~5 min: every stage, 1-2 instances. Do this FIRST.
gpap info                              # device + POWER STATE (docs/BENCHMARKING.md §1)
```

`rivals_setup.sh` builds CMSGen from upstream **with `scripts/cmsgen_gpap_eval.patch`** applied.
That patch is what makes the CPU comparison honest — see §4. Everything lands in
`~/.cache/gpap/{bin,src}`; nothing is vendored into the repo. `smoke` writes to a separate
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
pip install -q -e ".[dev]"
apt-get -qq install -y build-essential cmake zlib1g-dev >/dev/null   # cmsgen/abc build deps
eval/rivals_setup.sh && eval/run_full_eval.sh check
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

The in-repo patch adds two off-by-default flags to upstream CMSGen:

- `--nowrite` — generate, emit nothing. This is the timed run the ratio uses.
- `--statsevery N` — a flushed, parseable line with CryptoMiniSat's own propagation/decision/
  conflict counters, so the `Cdec/s` and `Cimpl/s` columns are the CPU sampler's real work rates
  and survive a run killed on the time budget.

The driver auto-detects both. With a **stock** binary it falls back to the samplefile run and
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
| `eval/rivals_demotic.py --mode native` | DEMOTIC (ASP-DAC'25) | their ISCAS-85 benchmarks with their own hard-coded output pins; GPAP pins exactly the same bits |
| `eval/rivals_demotic.py --mode miter` | DEMOTIC | OUR rare-target unrolled instance, handed to them as a single-output miter (`eval/unroll_to_bench.py`, which self-checks p(MITER=1) == p_accept) |
| `eval/rivals_recurrent.py [--rare]` | Recurrent-CircuitSAT (MLCAD'25) | their filename-keyed target table, then our rare pins |
| `eval/rivals_hts.py '<glob>.cnf'` | HTS | the very CNFs the `seq` stage generated |

All three are PyTorch programs run as subprocesses on the same GPU, scored with **their own**
metric (unique solutions ÷ their `model_runtime`, which excludes their model build). Their pins
are rejection-easy by construction, so `p_acc` is printed alongside every row.

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
| every rate ~halved | GPU power cap or battery clocks — `gpap info` (BENCHMARKING §1) |
| throughput collapses, no OOM | K too large → host-RAM spill. Never OOM-probe; use the nvidia-smi kill-switch (§2) |
| `cmsgen=yes [STOCK: ...]` warning | binary lacks the patch — `scripts/build_cmsgen.sh` |
| a stage exits instantly | it resumed and everything was already done; delete the JSON to force a re-run |
| `SKIPPED: formula outside the packed envelopes` | expected on giant unrolls; see §5 |
| rival "device not ready" | WSL2/driver hiccup between CUDA processes; the harnesses retry and step the batch down |
