# GPAP — artifact evaluation guide (SKETCH — not final)

> **STATUS: placeholder.** The full evaluation is being re-run on the release commit and every
> table regenerated; numbers, the data-archive name and the tag below will change. Structure is
> settled, contents are TODO until that re-run lands.

Maps the paper's claims to the data, the tables, and the commands that regenerate both.
The two tiers are deliberate: Tier 1 needs **no GPU and no dependencies beyond Python 3.10+**;
Tier 2 re-measures on any CUDA machine.

- **TODO** release tag / commit: `vX.Y.Z` (`<sha>`) — tag once the re-run data is final
- **TODO** data archive: `gpap_experiments_<gpu>_<date>.tar.gz` (DOI: TBD) — regenerate with
  `eval/artifact/make_artifact.sh` after the re-run
- **TODO** CITATION.cff: fill DOI / repository URL / paper reference (remove its TODO line)

## Tier 1 — regenerate every table from the shipped data (minutes, CPU-only)

The archive carries the raw per-instance JSONs, a MANIFEST (commit, host, SHA-256 per file) and
the captured verification verdict. `eval/artifact/paper_tables.py` and
`eval/artifact/verify_artifact.py` are stdlib-only.

```bash
tar xf gpap_experiments_<...>.tar.gz && cd gpap_experiments_<...>
# TODO: paste the exact regeneration block from the archive's MANIFEST.md here
eval/artifact/verify_artifact.py <...>       # the publishability gate: re-derives every caveat
```

- **TODO** claims table: paper claim → table/figure → generator flag → JSON file(s)
- **TODO** state the expected verifier verdict verbatim (N ok / N warnings / 0 failures) and
  that every warning appears in the paper text

## Tier 2 — re-measure (any CUDA box; budget ~5–8 h for the full sweep)

Requirements: NVIDIA GPU + driver, cmake + C++17 compiler, Python ≥ 3.10. **No CUDA toolkit,
no nvcc** — the Triton kernels are AOT-compiled by `scripts/build_core.sh`.

```bash
pip install -e ".[dev]"
scripts/build_core.sh                # the engine (libgpap)
eval/rivals/rivals_setup.sh          # abc + patched CMSGen + the 3 GPU rivals, all PINNED
eval/run_full_eval.sh smoke          # ~5 min kick-the-tires: every stage, 1–2 instances
eval/run_full_eval.sh all            # the full sweep (staged, resumable)
```

Full runbook (stages, time table, Colab notes, troubleshooting): `docs/EVAL_RUNBOOK.md`.

- **TODO** expected-variance statement: absolute rates are GPU- and power-state-dependent
  (`docs/BENCHMARKING.md` §1, `docs/findings/CROSS_GPU_FINDINGS.md`); which quantities are
  expected to reproduce exactly (soundness, win/loss structure, model counts at fixed K/seed)
  vs within a factor (rates, ratios)
- **TODO** known-issues disclosure list (b12/A100 success cliff; soft-portfolio removal and its
  measured price; vacuous-pin / zero-search exclusions) — one place, linking the findings docs

## What is third-party, and how it is pinned

| component | fetched by | pin |
|---|---|---|
| berkeley-abc | `scripts/build_abc.sh` | commit in script (`GPAP_ABC_COMMIT`) |
| CMSGen (+ eval patch) | `scripts/build_cmsgen.sh` | commit in script (`GPAP_CMSGEN_COMMIT`) |
| DEMOTIC / Recurrent-CircuitSAT / HTS | `eval/rivals/rivals_setup.sh` | commits in script; each result JSON records the checkout's actual HEAD (`rival_commit`). Setup also installs their python deps and applies a **function-preserving py3.13 shim** (their `exec`+`locals()` model loader breaks under PEP 667; verified bit-identical output on 3.12) |
| barbarik + SPUR | manual, see `eval/uniformity/barbarik_gpap.py` docstring | commit in docstring |
| benchmark suites | committed subsets (`benchmarks/*/README.md` = provenance/licenses); full sets via `scripts/fetch_benchmarks.sh` | **TODO**: pin the raw-URL fetches to commit SHAs |
