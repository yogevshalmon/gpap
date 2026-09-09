# Examples

Runnable programs for both faces of GPAP; they need CUDA and a built `libgpap`
(`scripts/build_core.sh`; `gpap info` checks).

## Python

```bash
python examples/sample_solutions.py [formula.cnf]    # sample a CNF's solutions
python examples/check_assumptions.py [formula.cnf]   # propagate many assumption sets at once
python examples/mine_clauses.py [formula.cnf]        # prove clauses the formula already entails
```

All default to `benchmarks/sampling/10.sk_1_46.cnf`, so they run bare.
[`sample_solutions.py`](sample_solutions.py) reads `gpap.sample()`'s result;
[`check_assumptions.py`](check_assumptions.py) walks three variables by hand, then filters
200,000; [`mine_clauses.py`](mine_clauses.py) pairs `LiteralProbe` with `mine_entailed`, which
`gpap mine` fronts.

## C and C++

Built into `build/` by the library's CMake:

```bash
./build/gpap_sample formula.cnf [n] [seed] [out.txt]   # C++, gpap::Sampler
./build/gpap_assumptions [formula.cnf]                 # C++, gpap::AssumptionChecker
./build/gpap_probe formula.cnf [width]                  # C++, gpap::Prober
./build/gpap_c_api [formula.cnf]                       # plain C, the raw ABI
```

Sources: [`../cpp/examples/`](../cpp/examples/); start with the C++ ones — `gpap.hpp`'s tier 1
needs no knowledge of the engine. [`c_api.c`](../cpp/examples/c_api.c) is the reference for
`gpap/gpap.h` (0-based indices, caller-owned buffers, `gpap_status` returns) and for binding from
another language.

## What none of them do

Pick a batch size: every entry point sizes the batch to the card and the formula, picks the
propagate kernel and splits the work into launches. `K` is tunable (`SampleOptions::k`,
`GPUSampler(k_cap=...)`) but not needed unless benchmarking:
[`docs/BENCHMARKING.md`](../docs/BENCHMARKING.md).
