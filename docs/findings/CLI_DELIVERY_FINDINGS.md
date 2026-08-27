# CLI delivery: what `--out` cost, and what the counters were saying

**Date**: 2026-08-27 · **GPU**: RTX 4080 Laptop (12 GB, WSL2) · engine at `65623ed` (wave
shuffle-gather schedule) · every number below is from that box on that build.

The engine was not implicated in any of this. The defects were in the print/plumbing layer —
which is precisely where they are hardest to notice, because nothing crashes and every individual
number looks plausible.

## 1. `--out` was a SECOND full sampling pass

`sample_clauses` collected output rows by re-running the sampler after the timed run. Three
independent problems, and they compounded:

1. it obeyed no `--time-limit` (`sample_bits` had no `max_seconds`);
2. it asked for `min(-n, --max-samples)` rows **regardless of what the timed run delivered**;
3. its wall was stamped into `phases` but excluded from `wall_total_s`.

Reported by the user on `b15.bench --encoder abc` (T=25, defaults):

```
29,037< models in 74.9s e2e (60.6s sampling)      <-- the timed run, capped at --time-limit 60
483 samples/s (steady-state), 388/s end-to-end
wrote 50,000 samples (900 cols)                   <-- a SECOND run, ~103 s, in no reported wall
real 3m8.684s
```

`29,037` and `50,000` are both true and describe different runs. The `388/s` "end-to-end" is
~1.5× the delivered rate (50,000 / 188.7 s = 265/s).

### The fix, and its price

Rows are now collected by the timed run itself — one `_project` per chunk. Measured cost of that
projection against the chunk it rides on, b15-c25:

| | per chunk |
|---|---|
| chunk (`_run_chunk`, K=7,682) | 1,033 ms |
| `_project` | 3.7 ms (**0.4%**) |

End-to-end on the user's exact command, same box, HEAD worktree vs the fix, interleaved:

| arm | models | written | wall |
|---|---|---|---|
| HEAD | 50,922 | 50,000 | **108.9 s** |
| fixed | 50,216 | 50,000 | **74.8 s** |

(The user's original 188.7 s was on a slower engine state — 483 samples/s vs ~1,150 today — where
the removed pass was ~103 s of the wall.)

### Two things the rewrite had to preserve, and one it broke first

- **Rows come from ONE config.** They are bucketed per bandit arm and only the winner's are
  returned. A mixture of arms is a different distribution, which is a uniformity question, not a
  delivery one.
- **A file never claims more than the run produced.** The cap is `min(-n, --max-samples,
  models delivered)`, and any top-up is bounded by what is left of the sampling budget.
- ⚠️ **The first version of this change silently altered every run that used `--out`.**
  `_project` fills don't-care columns from a `torch.Generator`, and it was the *sampler's* — the
  one driving the per-chunk variable permutation and the trajectory seed. Moving it inside the
  loop shifted that stream. Measured on a 4-var CNF, seed 5, pinned `b8`, K=64:

  | `out_cap` | models | success |
  |---|---|---|
  | 0 | 3,010 | 0.50033245 |
  | 500 | 3,007 | 0.48942057 |

  Deterministic, reproducible, and invisible to the no-change gate (which runs without `--out`).
  Fixed by giving delivery its own generator (`_proj_gen`). After: identical model counts and
  success at `out_cap` ∈ {0, 500, 2000} on both `b8` and `b1`.

  The projection wall is also excluded from every rate (`out_project_s`). Interleaved best-of-5,
  `--out` vs not: b15-c25 ratio **1.007** (6% run-to-run spread), tiny CNF 1.049 with 24% spread
  — i.e. inside the noise on both.

## 2. Mining counters

`--max-tuples` never bound width 1: `_mine_units` hardcoded the ceiling to 0, which the core reads
as unbounded. The CLI warned about the ceiling anyway.

```
--max-tuples 100  ->  width 1: up to 35,298 candidates  > --max-tuples 100
                      probed 70,514 of 70,514 candidates (100%) [exhaustive]
```

Two further confusions in those three lines:

- **"candidates" meant two things.** The plan line counted ONE round; the result summed the
  width-1 fixpoint's rounds. s1196-c25: round 1 = 35,298 (exactly `probe.count`), round 2 =
  35,216, total 70,514. b15-c25 reads 2,091,780 over ~8 rounds against a 271,548-literal pool.
  Proving convergence always costs one final round that finds nothing.
- **Cross-width totals.** `summary()` printed `probed 270,514 of 620,118,626` — a width-1 fixpoint
  sum divided by a C(P,2).

⚠️ **Applying the ceiling to width 1 is what a `gpap mine --max-tuples` user asks for, and it
nearly caused a silent regression on the DEFAULT sampling path.** `mine_entailed` looks the
ceiling up per width, and `STAGE1_MAX_TUPLES = {2: …, 3: …}` had no key `1`, so it fell through to
`DEFAULT_MAX_TUPLES` = 200,000 — less than one round on a large unroll, with no flag and no
warning. The stage dicts now carry an explicit `1: 0`.

The stop reason was also asserted rather than reported: any non-exhaustive width printed
`(max_tuples)`, including one cut by the wall-clock deadline — sending a user to raise a budget
they may have set to unbounded. It now comes from the probe's own `stopped`.

Cost of the throwaway `LiteralProbe` the CLI built just to print the plan, b15-c25: **0.71 s**,
for counts that take 0.00 s once an engine exists.

## 3. Writers

50,000 × 900 support projections:

| format | before | after |
|---|---|---|
| `bits` | 0.12 s / 45 MB | unchanged |
| `npy` | 0.03 s / 45 MB | unchanged |
| `dimacs` | 3.70 s / 197 MB | **0.60 s** / 225 MB |

`dimacs` was the only per-row Python loop. Vectorised by rendering each column's two possible
tokens once and gathering; the +14% size is fixed-width space padding, and that share grows with
the spread of the variable ids (a support mixing 3- and 6-digit ids pays more).

Two things the vectorisation broke and that are now gated:

- indexing a length-2 table made any value outside {0,1} an `IndexError`, where the previous
  `row.astype(bool)` accepted a 0/255 mask. Restored via `!= 0`.
- blocking by a fixed 4,096 rows made the peak scale with the support width — 2.4 GB at S=20,000.
  Blocked by BYTES now: peak ~110 MB at S=20,000.

## 4. Accounting

`phases` and `wall_total_s` disagreed by exactly the phases billed outside it — first `output`,
then (after that fix) `uniformity`, measured at **1.73×** on `--uniformity exact`. The coverage
run (`--n-cov`) was in neither.

Delivery phases now sum to `wall_total_s`; the opt-in readouts are billed to `phases_diag`,
outside it, so `sps_e2e_total` is not dragged down by a diagnostic the user asked for on the side.
Gated by a parametrized test over `{off, fast, exact, n_cov}`.

## 5. Silent no-ops and unvalidated arguments

- `gpap bench --allow-soft` was declared on the parser, accepted by the driver, and never passed
  between them — no warning either, unlike `gpap sample`.
- `-n 0`, `-n -5`, `--time-limit 0|-1`, `--max-tuples -1`, `--cycles 0` all ran.
- `--time-limit nan` and `inf` passed every bound (`nan < lo` is False) and would have disabled
  the cap they were setting — on `gpap bench` that value is also the rival subprocess's timeout.
- `--pin-outputs 0` means "pin nothing" (the free-hypercube baseline), so it must stay legal —
  the first pass at validation rejected it.
- The validators were added to `gpap/cli.py` only, while `eval/run_full_eval.sh` invokes
  `python -m gpap.benchmark` directly. They now live on that parser too.

## 6. What was NOT wrong

The C++ core. Its parse diagnostics are precise (`line 3: unexpected character 'x' — not
DIMACS`); the CLI was dropping them under a ctypes traceback. `gpap_probe_count` and the per-sweep
`total`/`probed` are exact — the confusing aggregate was a host-side sum. `max_tuples <= 0` meaning
unbounded is the documented ABI contract.

The one core-level property worth recording: `gpap_engine_trajectory` takes ONE `order` array per
launch, so all K rows in a chunk share a variable order (only polarity is per-row). That is why
`--out` rows, which come from few chunks, are the most exposed to the B=1 order-dependence — see
`docs/findings/UNIFORMITY_FINDINGS.md`.
