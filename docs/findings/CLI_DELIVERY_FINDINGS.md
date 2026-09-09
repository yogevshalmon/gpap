# CLI delivery — the rules for `--out`, counters and accounting

2026-08-27, RTX 4080 Laptop (12 GB, WSL2), engine at `65623ed`. Nothing here implicated the
engine; the defects were in the print/plumbing layer, where nothing crashes and every number
looks plausible. This file keeps the rules that came out of it.

## 1. Rows are collected by the timed run

`--out` rows come from one `_project` per chunk of the timed run, never from a second sampling
pass. The second pass obeyed no `--time-limit`, fetched `min(-n, --max-samples)` regardless of
what the timed run delivered, and sat outside `wall_total_s` while being stamped into `phases`:
a `--time-limit 60` run of b15.bench (abc, T=25) reported 29,037 models and then wrote 50,000 in
~103 s of unreported wall. Cost of the in-run projection on b15-c25: 3.7 ms against a 1,033 ms
chunk (0.4%); the same command went 108.9 s → 74.8 s, interleaved.

Three invariants of that design:

- A file never claims more than the run produced: `min(-n, --max-samples, delivered)` rows, and
  the CLI says why when it is short. Quote the delivered count, never the request.
- Rows come from one config: bucketed per bandit arm, only the winner's returned. A mixture of
  arms is a uniformity question, not a delivery one.
- Delivery and sampling must not share a random stream. `_project` fills don't-care columns from
  `_proj_gen`, not the sampler's generator: the first version shared them and `--out` silently
  changed which models a run found at the same seed (4-var CNF, seed 5, `b8`, K=64: 3,010 models
  at `out_cap=0` vs 3,007 at 500), invisible to the no-change gate, which runs without `--out`.
  After the split, counts and success are identical at `out_cap` ∈ {0, 500, 2000} on `b8` and
  `b1`.

The projection wall is `out_project_s`, excluded from every rate. Interleaved best-of-5, `--out`
vs not: b15-c25 1.007, a tiny CNF 1.049, both inside run-to-run spread.

## 2. Mining counters

- `--max-tuples` binds at width 1 (it was hardcoded to 0 there while the CLI warned about the
  ceiling anyway), and a per-width dict must give width 1 an explicit `0`: `STAGE1_MAX_TUPLES`
  had no key `1`, so the default sampling path fell through to `DEFAULT_MAX_TUPLES` = 200,000,
  less than one round on a large unroll, with no flag.
- "candidates" at width 1 is a sum over fixpoint rounds (s1196-c25: 35,298 + 35,216 = 70,514;
  b15-c25 2,091,780 over ~8 rounds against a 271,548-literal pool) and is not comparable to the
  pool size; proving convergence costs one final empty round. Never sum candidates across widths
  (`summary()` once printed a width-1 fixpoint sum over a C(P,2)).
- The stop reason comes from the probe's own `stopped`, never asserted: a width cut by the
  deadline was printed as `(max_tuples)`.
- Do not build a throwaway `LiteralProbe` to print a plan: 0.71 s on b15-c25 for counts that are
  free once an engine exists.

## 3. Writers

50,000 × 900 support projections: `bits` 0.12 s / 45 MB, `npy` 0.03 s / 45 MB, `dimacs` 3.70 s →
0.60 s (197 → 225 MB; the +14% is fixed-width padding, growing with the spread of variable ids).
`dimacs` renders each column's two tokens once and gathers; values are tested with `!= 0` (a
0/255 mask stays legal) and blocking is by bytes, not rows (peak ~110 MB at S=20,000 instead of
2.4 GB).

## 4. Accounting

`phases` sums to `wall_total_s`. Opt-in readouts (`--n-cov`, `--verify-sound`) are billed to
`phases_diag`, outside it, so `sps_e2e_total` is not dragged down by a diagnostic; the invariant
is gated by a test parametrized over the readouts that exist, not a specific flag (the
`uniformity` readout that first exposed it, at 1.73× on `--uniformity exact`, was removed from
the library 2026-09-01). A pinned single config has no per-config wall, so `out_project_s` and
`verify_sound_s` must be subtracted by the pipeline, not only stamped per config.

## 5. Argument validation

`-n 0`, `-n -5`, `--time-limit 0|-1`, `--max-tuples -1` and `--cycles 0` all ran unvalidated, and
`--time-limit nan|inf` passed every bound (`nan < lo` is False) — on `gpap bench` that value is
also the rival subprocess's timeout. Validators now live on both parsers, `gpap/cli.py` and
`python -m gpap.benchmark`, because `eval/run_full_eval.sh` invokes the latter directly.
`--pin-outputs 0` (pin nothing, the free-hypercube baseline) stays legal. `gpap bench
--allow-soft` was declared and never passed through; it is wired now.

## 6. What was not wrong

The C++ core: its parse diagnostics are precise (the CLI was dropping them under a ctypes
traceback); `gpap_probe_count` and the per-sweep totals are exact; `max_tuples <= 0` meaning
unbounded is the documented ABI contract. One core property worth recording:
`gpap_engine_trajectory` takes one `order` array per launch, so all K rows in a chunk share a
variable order (only polarity is per-row), which is why `--out` rows from few chunks are the most
exposed to the B=1 order-dependence (`docs/findings/UNIFORMITY_FINDINGS.md`).
