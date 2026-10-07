"""The GPAP engine — K-parallel GPU BCP over independent assumption sets (the novelty).

The PYTHON FACE of the engine; the engine itself is the C++ library `libgpap` (cpp/), which C++
consumers link directly. `_core.py` binds its C ABI with ctypes and `CnfFormula` / `BCPEngine` are
thin facades over it: one implementation, two faces. Requires CUDA + libgpap (`core_available()`,
built by scripts/build_core.sh); there is NO CPU path, and Triton is needed only to BUILD the
library (`kernel_available()` reports whether this host could).

PACKED STATE: propagation runs in Triton kernels (cpp/kernels/bcp.py) over a packed int32 clause
word — low SHIFT bits = sat_count, high bits = the SUM of surviving literal indices mod the field
width, so at sat_count == 1 that field IS the forced literal and the update is a single
atomic_add. The split is per formula (16|16 for 2N <= 65535, 8|24 for 2N <= 0xFFFFFF with clause
len <= 255; larger is rejected), and setting literal l TRUE updates exactly the entries in occ[l],
the CSR occurrence list of the NEGATED literal — no clause scanning, ever. ONLY length >= 3 clauses
carry that word: binaries are implication EDGES inside occ and units live in the construction
flush's bitmap, the same fixpoints by confluence at a fraction of the per-row state.

Applications consume the engine through exactly two entry points. Batched assumption checking is
K independent sets, one BCP fixpoint each — `reset(K)` then `apply_and_propagate(dec)`, a (K,L)
int64 literal-index tensor with -1 padding, one launch to fixpoint, read back through `conflict`
/ `assignment`; `check_sets(flat, offsets)` is the batched form the core chunks itself, and two
one-call faces sit over it, `AssumptionChecker` ("propagate THESE sets") and `LiteralProbe`
("propagate every k-SUBSET of this pool, within this budget"), which also owns the incremental
flow `assume()` = reset(1) + apply_and_propagate(units) + recache_flush(). Fused trajectory
sampling is the second: `reset(K)` then `run_trajectory(order, n_support, B, seed, soft=...)`
runs the whole randomized decide+propagate loop in one launch per chunk, and fully-assigned
active rows are models.

Everything else on BCPEngine is an accessor; do not reach into the private `_core` handle.
Correctness: BCP fixpoints are confluent, the kernels are validated against a pure-Python
reference BCP (tests/reference/bcp.py), and every emitted model is externally verifiable.
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
# deliberately not re-exported: batch sizing is the engine's job, and every entry point sizes
# itself to the card. It stays available as `from gpap.engine.calibrate import compute_k_safe`.
__all__ = [
    "CnfFormula", "read_dimacs", "BCPEngine", "make_assumption_tensor", "core_available",
    "AssumptionChecker", "LiteralProbe", "ProbeBudget", "ProbeResult",
    "kernel_available",
    "packed_envelope", "clauses_envelope",
    "ext_lit_to_lit_index", "lit_index_to_ext_lit", "lit_index_negation",
]
