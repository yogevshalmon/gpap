# Benchmarking GPAP — the measurement protocol

These are the measurement traps that have actually bitten this project. Follow them before quoting
any number, and cite [`docs/findings/INSIGHTS.md`](findings/INSIGHTS.md) for the underlying findings.

## 1. Check the power state FIRST

`gpap info` (or `nvidia-smi -q -d POWER` + SM clocks under load). This box class has shown a **40 W
cap** (default 80 W) and a **battery clock profile** (~1.6 GHz vs 3.1 GHz max), each of which
uniformly ~halves every rate. Idle clocks read ~210 MHz and are meaningless — read clocks *during* a
run. **Ratios in one power state are valid; absolute numbers across states or GPUs are not.**

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

## 5. Benchmark against the competitor's OWN optimized operation

Compare GPU work against CMSGen's actual C++ sampler (or a solver's in-C `process()`), never a naive
pysat re-implementation — the latter inflated a bogus GPU "speedup" 50–100×. When quoting vs the
papers, quote **both** paper-CMSGen (footnoted: their CNFs/CPU) and **local** CMSGen on our CNFs
(ours are ~2–3× faster than the papers' numbers due to DFF variable sharing).

## 6. `--n-time` is a delivery contract

`--n-time N` means deliver N models. The only early exits are `--time-limit` and the zero-progress
stop, both marked with `<` after the sols count. A delivered-blend rate below the winner's
steady-state rate is bandit exploration drag, not a bug — the headline `samples_per_s` is the
best-measured config's steady-state rate.

## 7. Result JSON is versioned

Benchmark result JSONs should carry a `"schema"` field so `eval/` plots and comparisons stay stable
as the schema evolves.
