# CNF sampling benchmarks — provenance, license, citations

The 29 committed `*.cnf` files are the **`sk_` Sketch/service family** of the standard sampling
suite — constraints blasted from [Sketch](https://people.csail.mit.edu/asolar/) synthesis problems,
`<name>.sk_<x>_<y>` being the upstream naming — used to evaluate UniGen, QuickSampler, CMSGen and
Barbarik. They are GPAP's CNF-native reference point (`eval/run_full_eval.sh sk`), the examples'
default input (`10.sk_1_46.cnf`), and Barbarik material.

- Source: the meelgroup sampling-benchmark distribution,
  [zenodo record 4022395](https://zenodo.org/record/4022395) — the same files ship with
  [QuickSampler](https://github.com/RafaelTupynamba/quicksampler) and
  [Barbarik](https://github.com/meelgroup/barbarik). Research use; no license file accompanies
  them.
- Suite: Chakraborty, Meel & Vardi, *A Scalable and Nearly Uniform Generator of SAT
  Witnesses*, CAV 2013. Sampling use: Dutra, Laeufer, Bachrach & Sen, *Efficient Sampling
  of SAT Solutions for Testing*, ICSE 2018 (QuickSampler); Golia, Soos, Chakraborty & Meel,
  *Designing Samplers is Easy: The Boon of Testers*, FMCAD 2021 (CMSGen).

Notes:

- **`c ind <v...> 0` lines** carry each instance's independent support (the UniGen/QuickSampler
  convention). GPAP samples exactly that projection; the parser keeps them
  (`gpap.engine.read_dimacs`) and `gpap sample` reads them by default.
- **Instance hygiene** (docs/findings/INSIGHTS.md,
  `eval/artifact/paper_tables.py`): `tableBasedAddition.sk_240_1024` is zero-search (its
  support is a free hypercube after propagation), so it is excluded from geomeans. `79.sk_4_40` is
  the one instance GPAP cannot sample (CMSGen can): a loss in the zero ledger, never dropped.
- The Java **service-class instances** (`ActivityService`, `ProcessBean`, ...) are the
  individually-free/jointly-constrained regime, where CDCL learning helps and GPAP's
  rejection-only search is weakest; kept for that.
- The full suite is larger (blasted/or/tire/modexp/hash, same record); only this subset is
  committed, unmodified upstream bytes.
