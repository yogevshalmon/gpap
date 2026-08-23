# CNF sampling benchmarks — provenance, license, citations

The 29 committed `*.cnf` files are the **`sk_` Sketch/service family** of the standard
CNF-sampling benchmark suite: constraints blasted from [Sketch](https://people.csail.mit.edu/asolar/)
synthesis problems (`<name>.sk_<x>_<y>` is the upstream naming, kept verbatim), used to evaluate
UniGen, QuickSampler, CMSGen and Barbarik. They are the CNF-native reference point of GPAP's
evaluation (`eval/run_full_eval.sh sk`), the default input of the runnable examples
(`examples/*.py` use `10.sk_1_46.cnf`), and part of the Barbarik certification runs.

| what | detail |
|---|---|
| source | the meelgroup sampling-benchmark distribution — [zenodo record 4022395](https://zenodo.org/record/4022395) (the same files ship with [QuickSampler](https://github.com/RafaelTupynamba/quicksampler) and [Barbarik](https://github.com/meelgroup/barbarik)) |
| license / terms | distributed by those projects for research use; no explicit license file accompanies the CNFs themselves |
| cite (suite origin) | Chakraborty, Meel & Vardi, *A Scalable and Nearly Uniform Generator of SAT Witnesses*, CAV 2013 (the UniGen benchmark set) |
| cite (sampling use) | Dutra, Laeufer, Bachrach & Sen, *Efficient Sampling of SAT Solutions for Testing*, ICSE 2018 (QuickSampler); Golia, Soos, Chakraborty & Meel, *Designing Samplers is Easy: The Boon of Testers*, FMCAD 2021 (CMSGen) |

Notes for the sampling use case:

- **`c ind <v...> 0` header lines** carry each instance's independent support (the
  UniGen/QuickSampler convention). GPAP samples exactly that projection; the engine's DIMACS
  parser keeps the lines (`gpap.engine.read_dimacs`), and `gpap sample` reads them by default.
- **Instance hygiene for headline numbers** (see `docs/findings/INSIGHTS.md` and
  `eval/artifact/paper_tables.py`): `tableBasedAddition.sk_240_1024` is a **zero-search**
  instance — after propagation its support is a free hypercube (zero decisions, zero
  implications, success 1.0), so it is excluded from geomeans like the vacuous-pin circuits.
  `79.sk_4_40` is the one instance of the family GPAP cannot sample (CMSGen can); it is a loss
  and appears in the zero ledger, never silently dropped.
- The Java **service-class instances** (`ActivityService`, `ProcessBean`, ...) are the
  individually-free/jointly-constrained regime — the honest hard cases where a CDCL-based
  sampler's learning helps and GPAP's rejection-only search is at its weakest. They are kept
  for exactly that reason.
- The full suite the papers use is larger (blasted/or/tire/modexp/hash families, same zenodo
  record); only this small curated subset is committed. Nothing here is generated — the files
  are upstream bytes, unmodified.
