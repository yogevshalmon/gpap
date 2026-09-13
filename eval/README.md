# eval/ — the experiment layer

Everything here produces *numbers*, not library behaviour. Nothing in `gpap/` imports from here,
so it may depend on third-party tools and result files outside the repo.

- `run_full_eval.sh` — staged and resumable, one stage per family (`seq`, `comb`, `sk`, `epfl`,
  `hwmcc`, `rivals`, `plots`, `all`); a failing cell becomes an error row, not a stop.
- `eval_plots.py` — staged JSON into figures and `summary.md`.
- `recheck_compare.py` — re-runs recorded cells, diffing them; non-zero exit only on a
  *deterministic* disagreement.

- `profile_soft_portfolio.py` — paired safe-vs-soft sweep, one process.
- `artifact/` — the paper and artifact build: `paper_tables.py` (the single generator),
  `slide_tables.py` (imports it, so the deck cannot drift), `verify_artifact.py` (the gate),
  `make_artifact.sh`.
- `rivals/` — harnesses for the GPU-sampler rivals, `rival_common.py` (the
  same-circuit-on-both-sides rule) and `rivals_setup.sh`.
- `uniformity/` — `barbarik_gpap.py` (conditional-sampling certification) and `fuzz_uniformity.py`
  (exact-oracle fuzzing).
- `colab/` — drivers for ephemeral GPU VMs: mirror, resume, survive reclamation.

## Running

Run from the repo root: scripts import `gpap.*`, and each other as `eval.*`.

```bash
PYTHONPATH=. python eval/artifact/paper_tables.py ...
eval/run_full_eval.sh seq
```

Third-party tools and fetched benchmarks install under `.gpap/` beside the repo (`gpap/paths.py`),
never `$HOME`; `$GPAP_HOME` overrides it. **Raw results live outside the repo**, under
`$GPAP_RESULTS`: inputs, not artifacts to commit.

Scratch is the one thing here that does **not** follow `.gpap/`: each harness writes its CNFs to
the system temp directory. That reads like drift and is not safe to tidy — the DIMACS parse of a
harness's CNF falls inside `wall_total_s`, which Table 1's `e2e` column divides
(`paper_tables.gpap_e2e`), so which filesystem the file sits on is inside a published number.
Moving it is a re-measurement, not a cleanup.

## Reading a number before quoting it

`eval_plots.py` writes a `summary.md` listing every skipped and errored instance — read it before
quoting a geomean. The exclusion rules (vacuous pins, zero-search cells) live in
`artifact/paper_tables.py` once; `verify_artifact.py` re-derives the headline from the raw rows.
Methodology: [`docs/BENCHMARKING.md`](../docs/BENCHMARKING.md).
