# archive/ — concluded experiment lines (kept for reference, not maintained)

Code here is **not imported by the package, the tests, or the eval scripts** and is not kept
working. Every settled insight these experiments produced is summarized in
[docs/findings/INSIGHTS.md](../docs/findings/INSIGHTS.md); the deep write-ups live in
`docs/findings/`. Git history preserves everything regardless.

| area | what it was | verdict |
|---|---|---|
| `scripts/inprocess_*.py`, `els_*.py`, `preprocess_solve.py`, `solver/inprocess.py`, `solver/cadical_hook.py`, `solver/preprocess_pipeline.py`, `tests/test_inprocess.py` | GPU inprocessing / preprocessing-for-solve (vivify, failed-literal probing, binary discovery, ELS-for-solve) | Retired: portfolio-only ~1.2–1.3×, no robust single-config win; CPU incremental probing is 7–15× faster at reductions. ELS itself lives on in `gpap/sampling/` (a measured 1.48× **sampling** win). See ELS_INPROCESSING_FINDINGS.md. |
| `scripts/clause_learn_solve.py`, `solver/clause_learner.py` | GPAP conflict → learned clause → kissat augmented solve | Worked (3.9× on mdp) but not robust across instances; superseded by the sampling flagship. Entailed-clause **mining** (a descendant) lives in `gpap/sampling/miner.py`. |
| `scripts/run_benchmark_suite.py`, `build_benchmark_suite.py`, `benchmarks/suite*.json`, `benchmarks/cnf/`, `rand_500v_2000c.cnf` | pre-sampling-era BCP-throughput suites (sat2006/sat24) | Superseded by the circuit-sampling benchmark flow (`eval/`). |
| `scripts/colab_benchmark.py`, `colab_plots.py`, `benchmarks/colab_results/`, `notebooks/gpap_colab_benchmark.ipynb` | old-engine cross-GPU benchmark (pairs anchor + random-L grid) | Superseded by the sampling Colab driver (`eval/colab_sampling_benchmark.py` + `notebooks/gpap_colab_sampling.ipynb`). |
| `profile_engine.py` | per-step/per-round profiling of the retired PYTORCH BCP path (propagate_profiled, soft_stop_frac, PySAT A/B timings) | Retired with the PyTorch engine path (2026-07-13): the engine is packed-Triton-kernel-only; this script's engine APIs no longer exist. The PySAT utility it used lives on as `scripts/external_solver.py`. |
