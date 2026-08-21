# eval/ — the experiment layer

Everything here produces *numbers*, not library behaviour. Nothing in `gpap/` imports from this
directory; it is free to depend on the package, on optional third-party tools, and on result
files that live outside the repo.

| | what it is |
|---|---|
| `run_full_eval.sh` | the staged, resumable driver. One stage per family (`seq`, `comb`, `sk`, `epfl`, `hwmcc`, `rivals`, `plots`, `all`). A failing cell becomes an error row and the sweep continues. |
| `eval_plots.py` | turns the staged JSON into the figures and `summary.md`. |
| `recheck_compare.py` | re-runs recorded cells and diffs them; non-zero exit only when a *deterministic* field disagrees. |
| `profile_soft_portfolio.py` | paired safe-vs-soft portfolio sweep in one process. |
| `artifact/` | what the paper and the artifact are built from — `paper_tables.py` is the single generator, `slide_tables.py` imports it so the deck cannot drift, `verify_artifact.py` is the publishability gate, `make_artifact.sh` packages it. |
| `rivals/` | head-to-head harnesses for the GPU-sampler rivals, plus `rival_common.py` (the same-circuit-on-both-sides rule) and `rivals_setup.sh` to install them. |
| `uniformity/` | `barbarik_gpap.py` (conditional-sampling certification) and `fuzz_uniformity.py` (exact-oracle fuzzing with calibrated controls). |
| `colab/` | drivers for rented, ephemeral GPU VMs — mirror every cycle, resume, survive reclamation. |

## Running

Scripts import the package as `gpap.*` and each other as `eval.*`, so run them from the repo root:

```bash
PYTHONPATH=. python eval/artifact/paper_tables.py ...
eval/run_full_eval.sh seq
```

Third-party tools and fetched benchmarks install under `.gpap/` beside the repo (see
`gpap/paths.py`), never in `$HOME`. Override with `$GPAP_HOME`.

**Raw results live OUTSIDE the repo**, under `$GPAP_RESULTS`. Measurements are inputs to the
tables here, not artifacts to be committed.

## Reading a number before quoting it

`eval_plots.py` writes a `summary.md` listing every skipped and errored instance — read it before
quoting a geomean. The exclusion rules (vacuous pins, zero-search cells) live in
`artifact/paper_tables.py` exactly once, and `artifact/verify_artifact.py` re-derives the headline
from the raw rows rather than trusting the table. Methodology: [`docs/BENCHMARKING.md`](../docs/BENCHMARKING.md).
