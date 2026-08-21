# The GPAP Python API — a map

One page: what the tiers are, which one to use, and the conventions that hold everywhere.
The reference documentation is the docstrings themselves (every public module, class and method
carries one); this page is the map that tells you where to look. C++ consumers get the same
map in [`cpp/README.md`](../cpp/README.md) — the two APIs are faces of one engine, `libgpap`.

## The tiers

```
gpap.sample(path, n)                       tier 0 — one call, samples out
│
├── gpap.sampling                          tier 1 — the one-call faces
│     pipeline.sample_file / sample_clauses    the full e2e sampling flow (ELS, mining,
│     GPUSampler                               warmup, portfolios, contract-timed run)
├── gpap.engine.AssumptionChecker              "propagate THESE assumption sets"
├── gpap.engine.LiteralProbe                   "probe every k-subset of this literal pool"
│
├── gpap.engine.BCPEngine / CnfFormula     tier 2 — the engine primitive (torch)
│     reset(K) / apply_and_propagate / check_sets / run_trajectory + selection readout
│
└── gpap.engine._core                      tier 3 — private ctypes binding (numpy)
      └── libgpap (cpp/) — the ONE implementation; C++ callers enter at <gpap/gpap.hpp>
```

**Use the lowest-numbered tier that answers your question.** Sampling: `gpap.sample()` or the
pipeline. Checking assumption sets: `AssumptionChecker`. Combinatorial probing (failed
literals, entailed clauses, backbones, lookahead): `LiteralProbe` — `gpap.mining` and
`gpap.implications` are worked examples of applications built on it. Building a NEW application
(your own decide/propagate loop, a different readout): `BCPEngine`, the tier the one-call faces
are themselves written on. Tier 3 is private: its surface tracks the C ABI, not a compatibility
promise.

## Conventions that hold on every public surface

- **Literals are DIMACS**: `+v` TRUE, `-v` FALSE, 1-based, exactly the numbers in the file.
  The internal 2N index scheme (`+v -> 2(v-1)`, `-v -> 2(v-1)+1`) appears only on `BCPEngine`'s
  raw `apply_and_propagate`, which is documented where it happens.
- **Nobody names K.** Every face sizes its own batches from the formula and the GPU
  (`gpap.engine.calibrate`, binding the model in `cpp/src/calibrate.cpp`). Overrides exist for
  measurement (`k_cap`, `pin_batch`, `chunk`), not for normal use.
- **CUDA + a built `libgpap` are required** (`scripts/build_core.sh`; verify with `gpap info`).
  There is no CPU execution path. `gpap.engine.core_available()` is the programmatic check.
- **Return types**: the sampling stack and `AssumptionChecker` speak torch (tensors on the
  formula's device); `LiteralProbe` and everything below `BCPEngine`'s accessors speak numpy.
  A face states its choice in its docstring.
- **Errors**: out-of-envelope formulas raise `ValueError` at construction; a missing library
  raises with the build instructions; out-of-range literals raise before any GPU work.
- **`__all__` is the contract.** What `gpap`, `gpap.engine` and `gpap.sampling` export is the
  supported API; anything reached another way may change without notice.

## The engine contract (what tier 2 guarantees)

Two entry points, defined once in [`gpap/engine/__init__.py`](../gpap/engine/__init__.py) and
mirrored by the C++ tiers in [`cpp/include/gpap/gpap.hpp`](../cpp/include/gpap/gpap.hpp):

1. **Batched assumption checking** — `reset(K)` + `apply_and_propagate(...)`, or the packaged
   `check_sets(flat_lits, offsets)` whose chunk loop runs in the core (`cpp/src/check.cpp`) so
   every face validates, chunks and pads identically.
2. **Fused trajectory sampling** — `run_trajectory(...)`: the whole randomized
   decide+propagate loop in one launch per chunk.

Everything else is readout. Hot paths use the selection API (`make_pair_mask` +
`select_complete` + `var_bits` / `check_sound` — counts and small matrices cross the bus, never
row state); the tensor-shaped accessors (`assignment`, `conflict`, `active`) are copies for
tests and off-hot-path consumers.

Correctness story: BCP fixpoints are confluent, so kernel/chunking choices can change speed but
never answers on surviving rows (a conflicted row's non-`conflict` fields are snapshots — see
`AssumptionChecker.check_assumptions`). The kernels are A/B'd against a pure-Python reference
BCP (`tests/reference_bcp.py`), the formula builder against `tests/reference_formula.py`, and
every emitted model is externally verifiable.

## Runnable starting points

| you want | start from |
|---|---|
| samples from a CNF / circuit | [`examples/sample_solutions.py`](../examples/sample_solutions.py), `gpap sample` |
| assumption checking | [`examples/check_assumptions.py`](../examples/check_assumptions.py) |
| entailed-clause mining | [`examples/mine_clauses.py`](../examples/mine_clauses.py), `gpap mine` |
| the implication structure of a formula | [`examples/literal_implications.py`](../examples/literal_implications.py) |
| C++ | [`cpp/examples/`](../cpp/examples/) (tier 1 + raw C ABI) |

## Deep material

Design and state layout: [`docs/ENGINE.md`](ENGINE.md). Measurement protocol:
[`docs/BENCHMARKING.md`](BENCHMARKING.md). Settled experimental results:
[`docs/findings/INSIGHTS.md`](findings/INSIGHTS.md).
