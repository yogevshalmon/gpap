# GPAP — artifact evaluation guide (SKETCH)

> **STATUS: placeholder.** The evaluation is re-running on the release commit; numbers and tag
> will change.

Claims → data → tables, with commands. Tier 1 needs **no GPU and nothing beyond Python
3.10+**; Tier 2 re-measures on CUDA.

- **TODO** release tag / commit `vX.Y.Z` (`<sha>`)
- **TODO** data archive `gpap_experiments_<gpu>_<date>.tar.gz` (DOI: TBD) from
  `eval/artifact/make_artifact.sh`
- **TODO** CITATION.cff: DOI, repository URL, paper reference

## Tier 1 — regenerate every table from the data (CPU-only)

The archive carries the per-instance JSONs, a MANIFEST (commit, host, SHA-256 per file) and the
captured verdict. `paper_tables.py` and `verify_artifact.py` are stdlib-only.

```bash
tar xf gpap_experiments_<...>.tar.gz && cd gpap_experiments_<...>
# TODO: paste the exact regeneration block from the archive's MANIFEST.md here
eval/artifact/verify_artifact.py <...>       # the publishability gate: re-derives every caveat
```

- **TODO** claims table: claim → table/figure → generator flag → JSON file(s)
- **TODO** the verdict verbatim (N ok / N warnings / 0 failures), every warning in the paper

## Tier 2 — re-measure (any CUDA box; ~5–8 h full sweep)

NVIDIA GPU + driver, cmake + C++17 compiler, Python ≥ 3.10. **No CUDA toolkit, no nvcc**:
`scripts/build_core.sh` AOT-compiles the kernels.

```bash
pip install -e ".[dev]"
scripts/build_core.sh                # the engine (libgpap)
eval/rivals/rivals_setup.sh          # abc + patched CMSGen + the 3 GPU rivals, all PINNED
eval/run_full_eval.sh smoke          # ~5 min kick-the-tires: every stage, 1–2 instances
eval/run_full_eval.sh all            # the full sweep (staged, resumable)
```

Runbook: `docs/EVAL_RUNBOOK.md`.

- **TODO** expected variance (`docs/BENCHMARKING.md` §1, `docs/findings/CROSS_GPU_FINDINGS.md`):
  what reproduces exactly (soundness, win/loss, counts at fixed K/seed), what within a factor
- **TODO** known issues in one place: b12/A100 cliff, soft portfolio, degeneracy exclusions

## Third-party components and their pins

- berkeley-abc (`scripts/build_abc.sh`), CMSGen + eval patch (`scripts/build_cmsgen.sh`): commit
  pinned in each script, `GPAP_ABC_COMMIT` / `GPAP_CMSGEN_COMMIT`.
- DEMOTIC / Recurrent-CircuitSAT / HTS (`eval/rivals/rivals_setup.sh`): commits in the script, each
  result JSON records the checkout's HEAD (`rival_commit`); setup adds a function-preserving
  py3.13 shim, verified bit-identical on 3.12.
- barbarik + SPUR: manual, commit in `eval/uniformity/barbarik_gpap.py`'s docstring.
- benchmark suites: committed subsets (`benchmarks/*/README.md`), full sets via
  `scripts/fetch_benchmarks.sh`; **TODO** pin the raw-URL fetches to SHAs.
