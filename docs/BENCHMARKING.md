# Benchmarking GPAP — the measurement protocol

These are the measurement traps that have actually bitten this project. Follow them before quoting
any number, and cite [`docs/findings/INSIGHTS.md`](findings/INSIGHTS.md) for the underlying findings.

## 1. Check the power state FIRST

`gpap info` (or `nvidia-smi -q -d POWER` + SM clocks under load). This box class has shown a **40 W
cap** (default 80 W) and a **battery clock profile** (~1.6 GHz vs 3.1 GHz max), each of which
uniformly ~halves every rate. Idle clocks read ~210 MHz and are meaningless — read clocks *during* a
run. **Ratios in one power state are valid; absolute numbers across states or GPUs are not.**

## 1b. Clocks move DURING a run — never compare rates across runs

The power state is not a constant you check once. Sampled every 3 s through a full `seq` sweep on
the 4080 laptop, the SM clock **while the GPU was >90% busy** ranged **615–1710 MHz** (mean 1373),
and between rows it sat at 2190 MHz. Same box, same run, same power source: a **2.8× spread**.
Clock and draw are not even monotonic together (1245 MHz at 96 W, 2190 MHz at 39 W) — occupancy
and the CPU/GPU shared power budget both move it, so the instance mix ahead of a row changes the
clock that row is measured at.

Consequence: **a few-percent difference between two runs on different days, or between two
differently-composed sweeps, is not resolvable on this hardware.** A GPAP-only sweep and a
GPAP+CMSGen sweep are not comparable at that resolution either — CMSGen's CPU-only stretch is
also a GPU idle stretch.

What IS valid:
- **Interleaved A/B in the same session**, alternating the arm order between rounds so neither
  arm systematically gets the cooler slot.
- **Model counts**, which are clock-independent (see §3) — the port gate uses these, not rates.
- Ratios *within* one sweep, where both sides met the same clock.

This is why the C++-core port was gated on model counts and on interleaved A/Bs (which showed
1.013× at the kernel level and 1.028× at the sampler level), not on a before/after sweep.

## 2. The box SPILLS, it does not OOM — use the kill-switch, never an OOM probe

On WSL2/WDDM, oversizing K makes tensors migrate to shared host RAM over PCIe (~12 GB/s vs
~200 GB/s VRAM) and throughput collapses silently — no `OutOfMemoryError`. So:

- Never OOM-probe for max K; it won't crash, it'll thrash.
- Validate **every** K-sizing change under an nvidia-smi kill-switch:
  `used > ~11 GB (of 12) → pkill`. If it trips, the K is bad — **revert**.
- (This is exactly how the Phase-4 K_fused eng_hold "fix" was caught spilling b05-c50 and reverted.)

## 3. Same-seed repeats measure TIMING noise only — not DRAW noise

The sampler pins its seed, so N repeats replay identical trajectories: a "consistent 0.81×" can be
pure draw luck. Worse, changing portfolio membership shifts RNG consumption, so every chunk's
trajectories change and per-instance rates swing ±20–40% on low-success instances.

**Arbiter for a real A/B verdict** — one of:
- a **multi-seed (5+) fixed-config shootout**, comparing MEDIANS; or
- a **deterministic same-seed single-config A/B** where the model COUNTS (and, on small instances,
  the model-set hash) must match bit-identically.

This repo's no-change gate uses the second form (see the pinned-gate script pattern in the refactor
experiments): fixed K, `K_fused` pinned to K, single config, seed 5 ⇒ bit-identical counts.
Like-for-like only: if the compared row **mined**, compare against a mined baseline.

## 4. Report unique-solutions/s OFF the timing path

Competitors (DEMOTIC / HTS / Recurrent CircuitSAT) report **unique** solutions/s. Folding
reconstruction + dedup into the timed run understates GPU throughput up to 10× (Python delivery,
not sampling, is the bottleneck). So:

- The timed run stays `collect=False` (pure GPU generation rate).
- `distinct_projections()` runs on a **separate** coverage sample; `unique_sps_est = raw × distinct_frac`.
- `distinct_frac ≈ 1.0` on all non-tiny spaces; the count matters on tiny circuit spaces (e.g.
  b06-c5 has only 64 distinct solutions total — a raw rate overstates distinct delivery ~2500×).

## 4b. Charge both samplers for the SAME deliverable — CMSGen's emission is not free

GPAP's timed run emits nothing (§4). Stock CMSGen formats and writes every sample as a full
N-variable DIMACS assignment, and on wide formulas that costs more than the sampling:
**`27.sk_3_32`, 50k samples — 0.42 s generating vs 3.6 s with `--samplefile`** (373 MB written;
`--samplefile /dev/null` still costs 3.3 s, so it is ostream formatting, not disk). A
GPAP-without-emission vs CMSGen-with-emission ratio measures file I/O.

So: build CMSGen with `scripts/build_cmsgen.sh` (upstream + `scripts/cmsgen_gpap_eval.patch`) and
let `gpap bench` use its `--nowrite` for the timed run — it detects the flag automatically, and
marks any row it had to measure the old way as `cms_mode="write"`. The patch's `--statsevery`
also exports CryptoMiniSat's own propagation/decision counters, which is where the `Cdec/s` /
`Cimpl/s` columns come from. **Ratios published before 2026-07-27 include the emission handicap.**

The patch also carries **`--maxtime N`**, and it must be used for any budgeted run: without it
the harness kills CMSGen at the deadline and the surviving count is the last `--statsevery`
multiple. On the 2026-08-23 campaign that floored all 174 timed-out rows to a multiple of 1000
and **81 of them to a clean `0`, each scored as an outright GPAP win** — one re-measured cell
really delivers 941 samples in the budget. With `--maxtime` CMSGen stops itself and prints an
exact `final=1` count (and its work counters even on cells where it emits nothing, which used to
drop those instances out of the propagation table). Rows carry `cms_count_exact`;
`verify_artifact.py` FAILS a stage that publishes a floored zero. **Any CMSGen number recorded
before 2026-08-26 is a lower bound, not a measurement.**

`--maxtime` bounds CMSGen's SAMPLING wall, the same quantity GPAP's `--time-limit` bounds, so the
two rate columns pair like with like: `cms_sps` (per sampling wall) against GPAP's steady-state
rate, `cms_sps_e2e` (per subprocess wall, parse included) against GPAP's `wall_total`. The older
definition divided GPAP's steady state by CMSGen's subprocess wall — one side charged for setup,
the other not.

The same rule binds the GPU rivals, and it is CHECKED rather than assumed:
`python eval/rivals/check_emission_symmetry.py --results <campaign>` AST-scans the function each
rival's own `timer(...)` wraps for host transfers and IO, asserts every `.sample()` in our timed
window passes `collect=False`, and audits `cms_mode` over the recorded rows. On the pinned
commits DEMOTIC and HTS time the descent loop alone (they count and dedup *after* the timer
stops). The GPAP half is also a runtime gate:
`test_timed_run_never_gathers_a_model_to_the_host` disarms `var_bits` and re-runs the timed path.

One exception, measured and left uncorrected: Recurrent-CircuitSAT copies its per-depth candidate
assignments to the host inside its own timed loop. Timing that block directly on an RTX 4080 puts
it at **0.7–1.4% of `model_runtime`** (bound ≤2.2%), and removing it — snapshots kept on device,
semantically identical for their solution check — leaves the unique-solution count bit-identical
while saving 1.41% on s386 over 4 interleaved repetitions. ⚠️ A single on/off pair is worthless
here: run-to-run spread is the size of the effect. We do not deduct it, because it is one rival,
worth ~1%, and the correction points **our** way — an uncorrected row under-reports GPAP rather
than flattering it, which is the right side to err on.

## 4c. Grant the rival the same share of the machine you grant yourself

GPAP sizes its batch `K` at **0.80 of free VRAM**. A GPU rival's batch is the same knob — and
measured on an RTX 4080 (2026-08-25), every one of the three scores in near-proportion to it,
with `model_runtime` almost flat, across the whole range the card will run: **HTS 295x from batch
100 to 100,000; DEMOTIC 564x from 1k to 1M; Recurrent 18.6x from 10k to 300k**. A batch is
therefore never "their configuration detail"; it is how much of the machine you decided to let
them use.

So size it by the same rule: **the largest batch that fits in 0.80 of free VRAM**
(`rival_common.fit_batch`), one run per cell, stepping down a rung if the driver refuses the size.
Record every rung as `batch_sweep` and the budget as `vram_budget_bytes`.

Their own defaults cannot be that rule, and reaching for them would flatter you: all three
argparse defaults are `--batch_size 10`, and the published demo scripts are per-instance
one-liners with a 1000x spread on comparably tiny circuits (DEMOTIC 1,000 on c17; Recurrent
1,000,000 on s27; HTS 100 on a 75-variable CNF). On our instances those would hand DEMOTIC ~1/500
and HTS ~1/295 of what their tool actually reaches, inflating our ratios by the same factor. A
share of the machine is the only quantity that means the same thing for both sides.

**Where your responsibility ends.** A rival that does not auto-tune its batch to the card takes
whatever it is given, so hand it the same share you take, sized from measured peak VRAM so the
grant itself does not cause a spill, and step down once if the driver refuses. Past that, memory
management is the tool's own concern — hand-tuning a rival beyond its share of the card produces
a number nobody can re-derive, which is worse than being strict.

Two failure modes this fixes, both of which flattered us: a **fixed 5 GiB** budget regardless of
card (a fifth of a 24 GB L4 while GPAP took ~19 GB of it), and running HTS at **batch 100/1000**,
~1/295 of the rate it reaches when it gets its share. Size from MEASURED peak VRAM, not the
graph formula: `n_gates * T * batch * 4` undercounts by **2-6x**, tool-dependent.

## 5. Benchmark against the competitor's OWN optimized operation

Compare GPU work against CMSGen's actual C++ sampler (or a solver's in-C `process()`), never a naive
pysat re-implementation — the latter inflated a bogus GPU "speedup" 50–100×. When quoting vs the
papers, quote **both** paper-CMSGen (footnoted: their CNFs/CPU) and **local** CMSGen on our CNFs
(ours are ~2–3× faster than the papers' numbers due to DFF variable sharing).

## 6. `--n-time` is a delivery TARGET — quote the delivered count, never the request

`--n-time N` asks for N models, and the run does not stop until it has them. It is not a guarantee
in either direction, so a table must report what was DELIVERED:

- **it overshoots.** The batch K is the unit of work and a chunk is never abandoned half-done, so
  a small formula at the 262,144 cap returns ~1e6 models for `-n 100`.
- **it can fall short.** `--time-limit` and the zero-progress stop both end a run early; both are
  marked `<` after the sols count and set `chunk_capped` in the JSON.

A delivered-blend rate below the winner's steady-state rate is bandit exploration drag, not a bug
— the headline `samples_per_s` is the best-measured config's steady-state rate.

`--out` does not extend the run: the file holds `min(-n, --max-samples, models delivered)` rows,
collected off the timed run's own chunks (docs/findings/CLI_DELIVERY_FINDINGS.md).

## 7. Result JSON is versioned

Benchmark result JSONs should carry a `"schema"` field so `eval/` plots and comparisons stay stable
as the schema evolves.
