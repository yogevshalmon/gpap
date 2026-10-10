# eval/ — the experiment layer

Everything here produces *numbers*, not library behaviour. Nothing in `gpap/` imports from here,
so it may depend on third-party tools and result files outside the repo.

- `run_full_eval.sh` — staged and resumable, one stage per family (`seq`, `comb`, `sk`, `epfl`,
  `hwmcc`, `rivals`, `plots`, `all`); a failing cell becomes an error row, not a stop.
- `eval_plots.py` — staged JSON into figures and `summary.md`, an inventory of what ran.
- `artifact/` — the paper and artifact build, and nothing else: every file here is one the
  artifact drivers run. `reproduce.sh` (the entry point), `paper_tables.py` (the single
  generator), `check_paper_numbers.py` and `verify_artifact.py` (the gates),
  `compare_campaign.py`, `estimate_wall.py`, `make_artifact.sh`. `slide_tables.py` sits here
  too, importing the generator so the deck cannot drift.
- `profiling/` — the GPAP-vs-GPAP sweeps that measure a choice rather than a rival, with their
  own table generators: `kernel_ab.py` (WAVE vs SCALAR) and `profile_soft_portfolio.py` (what
  one portfolio costs another). Nothing in the artifact runs them.
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

Three roots, and which one a file belongs in follows from how long it lives:

| root | holds | owner |
|---|---|---|
| `.gpap/` | built tools, fetched sources and benchmark sets, **and all scratch** — delete it and you lose only time (`$GPAP_HOME` moves it) | `gpap/paths.py` |
| `.experiments/` | a run's rows, logs and plots while it produces them, then hand-copied into the record — nothing automates that step (`$GPAP_EXPERIMENTS` moves the root) | `eval/paths.py` |
| `$GPAP_RESULTS` | the record, outside the checkout so deleting either of the above cannot touch evidence | `eval/paths.py` |

Nothing else names one of those directories: a shell script asks (`python3 -m eval.paths results`)
rather than recomputing, which is what stopped four derivations of the record root from drifting
apart. Gated by tests/repo/test_suite_hygiene.py.

## Tests

`pytest tests/experiments` runs this layer's tests. They are not part of the default `pytest` run,
which covers the library; CI runs them as a separate step.

## Reading a number before quoting it

`eval_plots.py` writes a `summary.md` listing every skipped and errored instance — read it before
quoting a geomean. The exclusion rules (vacuous pins, zero-search cells) live in
`artifact/paper_tables.py` once; `verify_artifact.py` re-derives the headline from the raw rows.
Methodology: [`docs/BENCHMARKING.md`](../docs/BENCHMARKING.md).
