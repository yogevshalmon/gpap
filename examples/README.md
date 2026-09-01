# Examples

Runnable programs for both faces of GPAP. They need CUDA and a built `libgpap`
(`scripts/build_core.sh`; `gpap info` says whether it is there).

## Python

```bash
python examples/sample_solutions.py [formula.cnf]    # sample a CNF's solutions
python examples/check_assumptions.py [formula.cnf]   # propagate many assumption sets at once
python examples/mine_clauses.py [formula.cnf]        # prove clauses the formula already entails
```

All default to `benchmarks/sampling/10.sk_1_46.cnf`, so they run with no arguments.

- [`sample_solutions.py`](sample_solutions.py) — `gpap.sample()`: what comes back, what the
  columns mean, and how to ask distribution questions of the result.
- [`check_assumptions.py`](check_assumptions.py) — `AssumptionChecker`: a hand-checkable
  three-variable walkthrough, then 200,000 sets filtered in bulk.
- [`mine_clauses.py`](mine_clauses.py) — `LiteralProbe` (the k-subset probe primitive) and
  `mine_entailed` (the strategy over it). `gpap mine` is the command-line front for the same
  thing.

## C and C++

Built by the same CMake as the library, into `build/`:

```bash
./build/gpap_sample formula.cnf [n] [seed] [out.txt]   # C++, gpap::Sampler
./build/gpap_assumptions [formula.cnf]                 # C++, gpap::AssumptionChecker
./build/gpap_probe formula.cnf [width]                  # C++, gpap::Prober
./build/gpap_c_api [formula.cnf]                       # plain C, the raw ABI
```

Sources are in [`../cpp/examples/`](../cpp/examples/). Start with the C++ ones: `gpap.hpp`'s
tier 1 (`Sampler`, `AssumptionChecker`) needs no knowledge of how the engine works.
[`c_api.c`](../cpp/examples/c_api.c) is the reference for `gpap/gpap.h` itself — 0-based indices,
caller-owned buffers, `gpap_status` returns — and is what you want if you are binding GPAP from
another language.

## What none of them do

Pick a batch size. Every entry point sizes the GPU batch to the card and the formula, chooses the
propagate kernel, and splits the work into launches on its own. `K` is tunable
(`SampleOptions::k`, `GPUSampler(k_cap=...)`) but you should not need it unless you are
benchmarking — see [`docs/BENCHMARKING.md`](../docs/BENCHMARKING.md).
