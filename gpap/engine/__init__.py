"""The GPAP engine — K-parallel GPU BCP over independent assumption sets (the novelty).

This package is the PYTHON FACE of the engine. The engine itself is the C++ library `libgpap`
(see cpp/), which C++ consumers link directly; `_core.py` binds its C ABI with ctypes, and
`CnfFormula` / `BCPEngine` are thin facades over it. One implementation, two faces.

PACKED STATE: propagation runs in Triton kernels (kernels.py — the kernel SOURCE, compiled
ahead of time into the library) over a packed int32 clause state — low SHIFT bits = sat_count
(non-falsified literals), high bits = SUM of the surviving literal indices mod the field width
(every removal exact mod 2^F, so at sat_count==1 the field IS the forced literal; the sum form
makes the update a single atomic_add in the warp-parallel WAVE kernel).
Field split per formula: 16|16 for 2N <= 65535, 8|24 ("wide") for 2N <= 0xFFFFFF with clause
len <= 255; anything larger is rejected. Setting literal l TRUE updates exactly the clauses in
occ[l] (CSR occurrence lists of the NEGATED literal). No clause scanning, ever.

REQUIREMENTS: CUDA + libgpap (`core_available()`; build it with scripts/build_core.sh). There
is NO CPU execution path. Triton is needed only to BUILD the library, never to run it —
`kernel_available()` reports whether the JIT reference path is usable, which the equivalence
tests need but production does not.

## The contract: exactly two entry points

Applications consume the engine ONLY through these:

1. **Batched assumption checking** — K independent assumption sets, one BCP fixpoint each:

       eng = BCPEngine(formula)                  # occ + packed init + post-units-flush cache
       eng.reset(K)                              # broadcast-restore the flush cache
       eng.apply_and_propagate(dec)              # (K,L) int64 lit-index tensor, -1 padding;
                                                 #   ONE kernel launch to fixpoint
       eng.conflict / eng.assignment             # (K,) bool / (K,2N) int8

   `AssumptionChecker` is the one-call convenience face. The miner's incremental flow is
   reset(1) + apply_and_propagate(units) + recache_flush().

2. **Fused trajectory sampling** — the WHOLE randomized decide+propagate loop in one launch
   per chunk (one Triton program per trajectory, sequential trail, no atomics):

       eng.reset(K)
       eng.run_trajectory(order, n_support, B, seed, soft=...)
       eng.active / eng.assignment               # fully-assigned active rows are models

Anything else on BCPEngine is accessors (`sat_rows`, work counters) — do not reach into
`_packed`/`_abits` from application code. Correctness: BCP fixpoints are confluent; the
kernels are validated against a pure-Python reference BCP (tests/reference_bcp.py) and every
emitted model is externally verifiable (clause satisfaction; tests do both).
"""
from gpap.engine.formula import (
    CnfFormula,
    OccurrenceList,
    ext_lit_to_lit_index,
    lit_index_to_ext_lit,
    lit_index_negation,
    lit_index_to_var,
)
from gpap.engine.bcp_engine import (
    BCPEngine,
    clauses_envelope,
    core_available,
    make_assumption_tensor,
    packed_envelope,
)
from gpap.engine.checker import AssumptionChecker
from gpap.engine.calibrate import (
    compute_k_safe,
    default_k_cap,
    gpu_mem_available,
    resolve_k_cap,
)
from gpap.engine.kernels import kernel_available

__all__ = [
    "CnfFormula", "OccurrenceList", "BCPEngine", "make_assumption_tensor", "core_available",
    "AssumptionChecker", "compute_k_safe", "gpu_mem_available", "kernel_available",
    "default_k_cap", "resolve_k_cap",
    "packed_envelope", "clauses_envelope",
    "ext_lit_to_lit_index", "lit_index_to_ext_lit", "lit_index_negation", "lit_index_to_var",
]
