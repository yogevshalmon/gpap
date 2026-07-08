"""The GPAP engine — K-parallel GPU BCP over independent assumption sets (the novelty).

State per batch row k (of K independent rows): `sat_count (K,M)` = non-falsified literals per
clause, `lit_sum (K,M)` = sum of their literal indices, `assignment (K,2N)`. Setting literal l
TRUE decrements both counters over occ[l] (CSR occurrence lists of the NEGATED literal); a
clause at sat_count==1 forces the literal whose index IS lit_sum (the sum trick), sat_count==0
is a conflict. No clause scanning, ever.

## The contract: exactly two entry points

Applications consume the engine ONLY through these (cube-and-conquer, UNSAT proving,
inprocessing and sampling all did):

1. **Batched assumption checking** — K independent assumption sets, one BCP fixpoint each:

       eng = BCPEngine(formula); eng.init_state(K)              # or reset_state(K) to reuse
       eng.apply_and_propagate(dec)                             # (K,L) int64 lit-index tensor,
                                                                #   -1 padding; fused kernel if
                                                                #   attached, else bulk + propagate
       eng.conflict / eng.real_conflict / eng.assignment        # (K,) bool / (K,) bool / (K,2N) int8

   `real_conflict` marks BCP-derived conflicts only (propagation happened) — the signal for
   clause learning / entailed-clause mining. `AssumptionChecker` is the one-call convenience
   face over this path.

2. **Fused trajectory sampling** — the WHOLE randomized decide+propagate loop in one GPU launch
   per chunk (one Triton program per trajectory, sequential trail, no atomics):

       eng.attach_kernel(TritonBCP())
       packed = eng.reset_fused(K)                  # broadcast-restore cached post-flush state
       eng.run_trajectory(order, n_support, B, seed, soft=..., packed=packed)
       eng.active / eng.assignment                  # fully-assigned active rows are models

Anything else on BCPEngine is bookkeeping (work counters) or internals — do not reach into
`_sat_count`/`_lit_sum` from application code; use the accessors (`sat_count`, `sat_rows`).
Semantics guarantee: BCP fixpoints are confluent, so kernel and PyTorch paths produce
bit-identical non-conflicted rows — the PyTorch path is the A/B oracle and the CPU fallback.
"""
from gpap.engine.formula import (
    CnfFormula,
    OccurrenceList,
    ext_lit_to_lit_index,
    lit_index_to_ext_lit,
    lit_index_negation,
    lit_index_to_var,
)
from gpap.engine.bcp_engine import BCPEngine, make_assumption_tensor
from gpap.engine.checker import AssumptionChecker
from gpap.engine.calibrate import compute_k_safe, gpu_mem_available
from gpap.engine.kernels import TritonBCP

__all__ = [
    "CnfFormula", "OccurrenceList", "BCPEngine", "make_assumption_tensor",
    "AssumptionChecker", "compute_k_safe", "gpu_mem_available", "TritonBCP",
    "ext_lit_to_lit_index", "lit_index_to_ext_lit", "lit_index_negation", "lit_index_to_var",
]
