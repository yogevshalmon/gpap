"""The GPAP engine — K-parallel GPU BCP over independent assumption sets (the novelty).

This package is the PYTHON FACE of the engine. The engine itself is the C++ library `libgpap`
(see cpp/), which C++ consumers link directly; `_core.py` binds its C ABI with ctypes, and
`CnfFormula` / `BCPEngine` are thin facades over it. One implementation, two faces.

PACKED STATE: propagation runs in Triton kernels (cpp/kernels/bcp.py — the kernel SOURCE,
compiled ahead of time into the library) over a packed int32 clause state — low SHIFT bits = sat_count
(non-falsified literals), high bits = SUM of the surviving literal indices mod the field width
(every removal exact mod 2^F, so at sat_count==1 the field IS the forced literal; the sum form
makes the update a single atomic_add in the warp-parallel WAVE kernel).
Field split per formula: 16|16 for 2N <= 65535, 8|24 ("wide") for 2N <= 0xFFFFFF with clause
len <= 255; anything larger is rejected. Setting literal l TRUE updates exactly the entries in
occ[l] (CSR occurrence lists of the NEGATED literal). No clause scanning, ever.

ONLY length >= 3 clauses carry that packed word: binary clauses are implication EDGES inside occ
(payload = the other literal — one bitmap push, zero per-row state) and unit clauses live in the
construction flush's bitmap. Same fixpoints (BCP confluence), ~4-10x less per-row state on
circuit CNFs.

REQUIREMENTS: CUDA + libgpap (`core_available()`; build it with scripts/build_core.sh). There
is NO CPU execution path. Triton is needed only to BUILD the library, never to run it;
`kernel_available()` reports whether a triton JIT is usable on this host, which is how the GPU
test suite decides to run.

## The contract: exactly two entry points

Applications consume the engine ONLY through these:

1. **Batched assumption checking** — K independent assumption sets, one BCP fixpoint each:

       eng = BCPEngine(formula)                  # occ + packed init + post-units-flush cache
       eng.reset(K)                              # broadcast-restore the flush cache
       eng.apply_and_propagate(dec)              # (K,L) int64 lit-index tensor, -1 padding;
                                                 #   ONE kernel launch to fixpoint
       eng.conflict / eng.assignment             # (K,) bool / (K,2N) int8
       eng.check_sets(flat, offsets)             # the BATCHED form: any number of DIMACS sets,
                                                 #   chunked by the core (cpp/src/check.cpp)

   Two one-call faces sit over this, and which you want depends on where the sets come from:

     AssumptionChecker  — "propagate THESE sets", any number of them, chunked internally.
     LiteralProbe       — "propagate every k-SUBSET of this literal pool, within this budget",
                          which is the shape of failed-literal detection, entailed-clause
                          mining (gpap.mining), backbone extraction and lookahead scoring.
                          It also owns the incremental flow, `assume()` =
                          reset(1) + apply_and_propagate(units) + recache_flush().

2. **Fused trajectory sampling** — the WHOLE randomized decide+propagate loop in one launch
   per chunk (one Triton program per trajectory, sequential trail, no atomics):

       eng.reset(K)
       eng.run_trajectory(order, n_support, B, seed, soft=...)
       eng.active / eng.assignment               # fully-assigned active rows are models

Anything else on BCPEngine is accessors (`sat_rows`, work counters) — do not reach into the
private `_core` handle from application code. (This line used to name `_packed`/`_abits`; those
tensors went with the PyTorch path, and the row state now lives inside the C++ core, reachable
only through the selection API.) Correctness: BCP fixpoints are confluent; the kernels are
validated against a pure-Python reference BCP (tests/reference_bcp.py) and every emitted model
is externally verifiable (clause satisfaction; tests do both).
"""
from gpap.engine.formula import (
    CnfFormula,
    ext_lit_to_lit_index,
    lit_index_to_ext_lit,
    lit_index_negation,
    read_dimacs,
)
from gpap.engine.bcp_engine import (
    BCPEngine,
    clauses_envelope,
    core_available,
    make_assumption_tensor,
    packed_envelope,
)
from gpap.engine.checker import AssumptionChecker
from gpap.engine.probe import LiteralProbe, ProbeBudget, ProbeResult
from gpap.engine._triton import kernel_available

# The GPU memory model (compute_k_safe / gpu_mem_available / default_k_cap / resolve_k_cap) is
# DELIBERATELY not re-exported here. Batch sizing is the engine's job, not the caller's: every
# entry point sizes itself to the card, so a user never has to name K. It stays fully available
# for benchmarking and for anyone who wants to override the sizing —
# `from gpap.engine.calibrate import compute_k_safe` — just not on the front page.
__all__ = [
    "CnfFormula", "read_dimacs", "BCPEngine", "make_assumption_tensor", "core_available",
    "AssumptionChecker", "LiteralProbe", "ProbeBudget", "ProbeResult",
    "kernel_available",
    "packed_envelope", "clauses_envelope",
    "ext_lit_to_lit_index", "lit_index_to_ext_lit", "lit_index_negation",
]
