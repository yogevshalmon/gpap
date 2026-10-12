# Benchmarking GPAP — the measurement protocol

Every rule below was once a wrong number in this project. The measurements that established
them are in [`findings/INSIGHTS.md`](findings/INSIGHTS.md) ("Measurement methodology") and in
the findings files each section links.

## 1. Check the power state first

`gpap info` reports the device and warns about a power cap. Laptop GPUs run under power caps and
battery clock profiles that uniformly halve every rate (the reference box has been measured at
40 W against an 80 W default). Read SM clocks during a run, not idle. Ratios measured in one
power state are valid; absolute numbers across states or GPUs are not.

### 1b. Clocks move during a run

Sampled every 3 s through a full sweep on the reference laptop, the SM clock while >90% busy
ranged 615–1710 MHz on mains power, and not monotonically with draw: the instance mix ahead of a
row changes the clock that row is measured at. A few-percent difference between runs on
different days is not resolvable, and a GPAP-only sweep is not comparable to a GPAP+CMSGen sweep
(CMSGen's CPU stretch is a GPU idle stretch). Valid: interleaved A/Bs in one session with the arm
order alternated (`scripts/perf_ab_trees.py` does this across two checkouts); model counts,
which are clock-independent (§3); ratios within one sweep.

## 2. The box spills, it does not OOM

On WSL2/WDDM an oversized batch migrates to host RAM over PCIe and throughput collapses silently.
Never probe for the largest K that "works"; validate every change to batch sizing under an
`nvidia-smi` kill-switch (`used > ~11 GB of 12 → pkill`). If it trips, the sizing is wrong:
revert.

## 3. The no-change gate, and what counts as a verdict

The sampler is seeded, so same-seed repeats replay identical trajectories and measure timing
noise only. Any change to portfolio membership or chunk count shifts RNG consumption, and on
low-success instances per-instance rates then swing ±20–40% either way; a "consistent 0.81×" can
be pure draw luck.

**Seed era boundary, 2026-09-01**: the sampling RNG moved from Python's torch generators into
the core sampler (`gpap_sampler_*` — one deterministic stream shared by the C++ and Python
faces). Same-seed model counts are comparable within an era, never across it; the gate below
applies unchanged inside the new era, and it gains a cross-face variant
(tests/sampling/test_face_parity.py: same seed + single config ⇒ identical rows on both faces).

**A pure refactor is gated on identical model counts**, not on rates:

- pin the batch below both arms' analytic bounds (`--k-cap 20000` on the reference box; the
  shipped cap does not pin K once the memory model differs between arms) and `pin_batch`;
- one config, no bandit (`--portfolio b8|b1|s1`), fixed seed;
- count-terminated (`-n`), never wall-terminated — a cell that stops on `--time-limit` compares
  speed, not determinism;
- mining off, or like-for-like — mining is wall-budgeted, so a faster arm mines a different
  formula;
- ⇒ identical model counts and success rates, plus model-set hashes on small instances.

**A throughput verdict** needs an interleaved same-session A/B (§1b) at pinned config,
best-of-N per arm, or a multi-seed (5+) fixed-config shootout comparing medians
(`gpap bench --seeds 5 6 7`). Never compare two builds on one free-bandit run: at a 30 s budget
the config bandit is a coin flip (1–2 mis-picks in 8 seeds), so the arms may simply have chosen
different configs. `implications_per_s` is bandit-independent engine work and the right
cross-build number when pinning is impossible.

## 4. Throughput is measured with nothing leaving the GPU

The timed run is `collect=False`: models stay on the device, and the per-chunk soundness check
is the only readout inside the window. Everything that delivers samples — the bit projection
`--out` writes, `distinct_projections`, the coverage pass and the soundness self-test — runs off
the timing path and is billed separately (`out_project_s`, `phases_diag`). Rivals report unique
solutions/s; `unique_sps_est = raw × distinct_frac` is computed the same way, off the path.
`distinct_frac ≈ 1.0` on every non-tiny space; on tiny circuit spaces the count is the honest
number (b06-c5 has 64 solutions in total).

### 4b. Charge both samplers for the same deliverable

Stock CMSGen writes every sample as a full N-variable assignment, and on wide formulas that costs
more than the sampling (27.sk_3_32, 50k samples: 0.42 s of generation becomes 3.6 s with the
file). Comparing GPAP-without-emission against CMSGen-with-emission measures file I/O. Build
CMSGen with `scripts/build_cmsgen.sh`, which applies `scripts/cmsgen_gpap_eval.patch`:

| flag | why it exists |
|---|---|
| `--nowrite` | generate, emit nothing — the timed run |
| `--statsevery N` | CMSGen's own propagation/decision counters, flushed (the `Cdec/s` / `Cimpl/s` columns) |
| `--maxtime N` | CMSGen stops itself at the budget and prints an exact `final=1` count. Without it a killed run reports the last `--statsevery` multiple: 81 cells of one campaign read a floored `0` that scored as our win |

`gpap bench` detects the flags, times the `--nowrite` run, records `cms_count_exact`, and marks
a stock binary's row `cms_mode="write"` rather than hiding the handicap. The rate columns pair
like with like: `cms_sps` (per CMSGen's own sampling wall) against GPAP's steady-state rate,
`cms_sps_e2e` (per subprocess wall) against `wall_total`. The GPU rivals are held to the same
rule, re-derived rather than assumed: `eval/rivals/check_emission_symmetry.py` AST-scans what
each rival's own timer wraps. One rival, Recurrent-CircuitSAT, copies its candidates to the host
inside its timed loop; a patch times that copy on its own and the harness deducts it, as GPAP's
own delivery is deducted. Left in, it would lower the rival's rate and flatter us. Evidence:
[`findings/RIVAL_FAIRNESS_FINDINGS.md`](findings/RIVAL_FAIRNESS_FINDINGS.md) §1, §7, §17c.

### 4c. Grant a rival the same share of the machine you take

A GPU rival's batch is the knob K is for us, and its unique/s is near-proportional to it (HTS
295× from batch 100 to 100,000; DEMOTIC 564× from 1k to 1M). The rule: the largest batch that
fits 0.80 of free VRAM — the share GPAP takes — sized from measured peak memory, stepping down a
rung if the driver refuses (`eval/rivals/rival_common.fit_batch`). Their argparse defaults
(`--batch_size 10`) would hand a rival 1/300–1/500 of what it reaches. `num_steps` is taken from
the rival's own artifacts (`RIVAL_STEPS`: DEMOTIC 1, Recurrent 1, HTS 10), tie-broken in the
rival's favour and never swept. RIVAL_FAIRNESS_FINDINGS.md §2, §6.

## 5. Compare against the competitor's own optimized operation

CMSGen's C++ sampler, a solver's in-C `process()` — never a naive pysat re-implementation, which
once inflated a GPU "speedup" 50–100×. Against a paper, quote both the paper's CMSGen number
(footnoted: their CNFs, their CPU) and local CMSGen on our CNFs (ours run ~2–3× faster, because
our unrolls share flip-flop variables).

## 6. `--n-time` is a delivery target — quote the delivered count

The batch is the unit of work, so a run overshoots to the end of a chunk (a small formula at the
262,144 cap returns ~10⁶ models for `-n 100`) and can fall short when `--time-limit` or the
zero-progress stop fires (marked `<` after the count, `chunk_capped` in the JSON).
`samples_per_s` is the best config's steady-state rate; `sps_e2e_total` includes setup. `--out`
never extends the run — the file holds `min(-n, --max-samples, delivered)` rows
([`findings/CLI_DELIVERY_FINDINGS.md`](findings/CLI_DELIVERY_FINDINGS.md)).

## 7. Across GPUs

Pin `--k-cap N` and the config on every arm, and give the run several chunks
(`--n-time 2000000 --time-limit 20`; a run that ends in one chunk measures first-chunk overhead,
~30% off). Read `engine.batch`, `engine.k_cap` and `succ` back from every row: success is a
property of the formula and the config, so a success rate that differs across machines is a
confound, not a result. The miner is wall-budgeted, so a faster GPU also learns a different
clause set. Expect roughly the SM-count ratio (A100 ~1.8×, L4 ~0.85× of the reference laptop);
K is hygiene, not a lever — 4× the batch bought +2.8%.
[`findings/CROSS_GPU_FINDINGS.md`](findings/CROSS_GPU_FINDINGS.md).

## 8. Which number to quote

The reporting rules are enforced once, in `eval/artifact/paper_tables.py`, and every other table
generator imports them ([`papers/tacas-27/README.md`](papers/tacas-27/README.md)).

- **Steady-state vs end-to-end.** `samples_per_s` excludes parse, construction, ELS, warmup and
  mining (a median 21% of GPAP's wall); CMSGen's rate carries its setup. Print both; quote the
  `e2e` column for a delivered-samples claim.
- **Against the GPU rivals, one clock — theirs.** Table 2 is unique solutions per second of the
  rival's own `model_runtime` (Recurrent's less the host copy inside it, which no other clock
  carries) against GPAP's steady-state unique/s, and has no end-to-end column
  by rule (protocol v1.0, [`EVAL_RUNBOOK.md`](EVAL_RUNBOOK.md) §0). The headline is the full
  circuit set.
- **Rare targets.** At random output pins CPU rejection sampling beats every CNF sampler, so a
  win there is not a sampling result. Quote `--target rare` and report `p_acc` / `amp` beside it.
- **Screen degenerate cells first, then bucket**: vacuous pins (constant outputs — `b02`, `b05`,
  `b07`, `s526` at T=25) and zero-search cells (models emitted with zero decisions and zero
  implications).
- **The zero ledger is symmetric**: `CMSGen=0`, `GPAP=0` and `both=0` are columns that sum, with
  the geomean's instances, to the instance count. A geomean over fewer than three instances is
  flagged, not quoted.
- **No best-of-encoders**: one row per (family, encoder). Selecting on the published ratio lets
  the rival's number choose which of our runs is shown.
- **A CMSGen ratio travels with its CPU** (`cpu` is in every row): the same suite reads 46× on
  one host and 99× on a host whose CPU is 0.41× as fast.
- **Uniformity claims name a config and carry their control** — an exactly uniform sampler that
  must pass on the same instances; a known-biased one that must fail shows the test can Reject
  at all, and is cited from where it was measured rather than re-run per campaign — and an
  Accept is "not caught", never "uniform" ([`findings/UNIFORMITY_FINDINGS.md`](findings/UNIFORMITY_FINDINGS.md) §4b).

## 9. Result JSON is versioned

`gpap bench` writes `"schema": 4`. Readers take columns with `.get()`, so old and new rows can
sit in one file; the version tells a reader which columns a row could possibly have. A field
that changes meaning is a version bump, never a silent rename. Schema 4 dropped the `uniformity`
column: the readout moved out of the library into `eval/uniformity/`.
