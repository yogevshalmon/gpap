"""BCPEngine — the K-parallel packed BCP engine, the Python face of the C++ core (libgpap):
marshalling and torch-friendly accessors over `gpap.engine._core`.

State layout (identical to the C core, docs/ENGINE.md): one packed clause word (K, num_long) int32
for LENGTH >= 3 clauses only, low SHIFT bits = sat_count and high bits = the modular SUM of the
surviving literal indices (at sat_count == 1 that field IS the forced literal); an assignment
BITMAP (K, ceil(2N/32)) int32 where l and ~l share a word, which is what makes the kernels'
one-atomic dedup/contradiction work; and the post-units-flush cache, broadcast-restored by
`reset(K)`. Two entry points (`apply_and_propagate`, `run_trajectory`) plus accessors; hot paths
use the SELECTION api, which keeps completion and soundness tests on the GPU.
"""
import numpy as np
import torch

from gpap.engine import _core
from gpap.engine.calibrate import dispatch_mode
from gpap.engine.formula import CnfFormula

# Packed envelopes. Two conditions make the sum-trick word exact and are the whole overflow story:
# the sum field must hold any single literal index (2N <= 2^(32-SHIFT)) and the count field any
# clause's length (max len <= 2^SHIFT - 1). A formula outside BOTH splits is REJECTED, so a suite
# sweep should pre-check with packed_envelope().
PACKED_SPLITS = ((16, 65535, 65535), (8, 0xFFFFFF, 255))   # (SHIFT, max 2N, max clause len)


def packed_envelope(num_vars: int, max_clause_len: int) -> "tuple[int | None, str]":
    """Pick the packed field split: (SHIFT, "") when it fits, else (None, reason). 16|16 whenever
    2N fits, else the 8|24 "wide" split for the big unrolls."""
    n2 = 2 * num_vars
    for shift, max_2n, max_len in PACKED_SPLITS:
        if n2 <= max_2n and max_clause_len <= max_len:
            return shift, ""
    return None, (f"outside the packed envelopes: 2N={n2}, max clause len={max_clause_len} "
                  f"(16|16 needs 2N<=65535 and len<=65535; 8|24 needs 2N<={0xFFFFFF} and "
                  f"len<=255)")


def clauses_envelope(num_vars: int, clauses: list) -> "tuple[int | None, str]":
    return packed_envelope(num_vars, max((len(c) for c in clauses), default=0))


def make_assumption_tensor(assumptions_lit_indices: list[list[int]],
                           device: torch.device) -> torch.Tensor:
    """list[list[int]] -> a (K, L) int64 padded tensor for apply_and_propagate, L = the max
    assumption count and -1 the pad. Build it once, before any timed path."""
    K = len(assumptions_lit_indices)
    L = max((len(lits) for lits in assumptions_lit_indices), default=0)
    t = torch.full((K, L), -1, dtype=torch.long)
    for k, lits in enumerate(assumptions_lit_indices):
        if lits:
            t[k, :len(lits)] = torch.tensor(lits, dtype=torch.long)
    return t.to(device)


def core_available() -> bool:
    return _core.available()


class BCPEngine:
    """Packed-state BCP over K independent rows, on the C++ core.

        eng = BCPEngine(formula); eng.reset(K)   # occ + packed init + flush cache, then K rows
        eng.apply_and_propagate(dec)             # (K,L) int64 lit indices, -1 pad; ONE launch
        eng.run_trajectory(order, n_support, B, seed, soft=...)   # sampling loop, ONE launch
        mid = eng.make_pair_mask(support_mask); n = eng.select_complete(mid, slot=0)

    Setting literal l TRUE decrements the packed word of every clause in occ[l] (CSR lists of the
    NEGATED literal) and subtracts ~l from its remaining-literal sum field; sat==1 forces the
    literal the sum field holds, sat==0 conflicts the row. No clause scanning, ever.
    """

    def __init__(self, formula: CnfFormula, *, force_wide: bool = False,
                 mode: "str | None" = None):
        """`mode`: None means "auto", which picks the propagate kernel per `reset(K)` (see
        `plan_for`); "scalar" is the serial-trail kernel, "wave" the warp-wide one, and naming
        either PINS it, as does $GPAP_KERNEL. Auto is resolved IN THE CORE on every
        `gpap_engine_reset`, so the dispatch reaches every consumer; same fixpoints either way."""
        self._formula = formula
        self._device = formula.device
        if self._device.type != "cuda":
            raise RuntimeError("BCPEngine requires CUDA (the engine is GPU-only; there is no "
                               "CPU path). Check `gpap info`.")
        self._core = _core.Engine(formula.core, device=self._device.index or 0, mode=mode,
                                  force_shift=8 if force_wide else 0)
        self._mode = self._core.requested_mode


    def set_mode(self, mode: str) -> None:
        """PIN the propagate kernel variant ("scalar" | "wave"), disabling auto-dispatch; both
        reach the same fixpoints, so it is a pure launch-flag change. Pin one for a MEASUREMENT."""
        self._core.set_mode(mode)
        self._mode = mode

    @property
    def auto_dispatch(self) -> bool:
        """True while no kernel is pinned — by the constructor, `$GPAP_KERNEL` or `set_mode`. A
        caller about to pin one of its own checks this first, so it never overrides an explicit
        pin; answered by the CORE, where the env override is read."""
        return self._core.auto_dispatch

    def plan_for(self, K: int) -> str:
        """The propagate kernel a chunk of K rows will run — PURE, since `reset(K)` applies it.
        Returns a pinned mode verbatim, $GPAP_KERNEL included."""
        if self.auto_dispatch:
            return dispatch_mode(self.num_clauses, int(K), self._device)
        return self._core.mode

    @property
    def mode(self) -> str:
        return self._core.mode

    @property
    def shift(self) -> int:
        return self._core.shift

    @property
    def num_vars(self) -> int:
        return self._core.num_vars

    @property
    def num_clauses(self) -> int:
        return self._core.num_clauses

    @property
    def num_long(self) -> int:
        """STATE (length >= 3) clauses — what one row's packed words cost. Feed THIS to the K
        model, not num_clauses: binaries and units carry no per-row state."""
        return self._core.num_long

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def K(self) -> int:
        """Rows the engine holds state for. Read from the CORE, never mirrored: the sampling
        loop resets inside libgpap, so a local mirror would read 0 after a run."""
        return self._core.K


    def reset(self, K: int) -> None:
        """Reset K rows to the cached post-units-flush state (allocations reused when K matches),
        resolving the propagate kernel for that batch when the mode is "auto" — in the core, so
        the dispatch reaches every consumer of the ABI."""
        self._core.reset(K)

    def apply_and_propagate(self, decisions) -> None:
        """Apply a (K, L) int64 decision tensor/array (-1 = padding) and BCP to fixpoint in ONE
        launch (entry point 1). Duplicates are skipped; l/~l pairs and decisions contradicting
        the assignment conflict the row."""
        if isinstance(decisions, torch.Tensor):
            if decisions.dim() == 1:
                decisions = decisions.unsqueeze(1)
            if decisions.numel() == 0:
                return
            if decisions.is_cuda:
                # Memory safety, as in _core._decisions: apply_dev hands the core a raw pointer
                # it reads K rows from, so a short tensor is an out-of-bounds device read.
                k = self._core.K
                if decisions.size(0) != k:
                    raise ValueError(
                        f"apply_and_propagate: decisions has {decisions.size(0)} rows but the "
                        f"engine is sized for K={k} — pass one row per assumption set, or "
                        f"reset({decisions.size(0)}) first")
                d = decisions.to(torch.int64).contiguous()
                self._core.apply_dev(d.data_ptr(), d.size(1))
                return
            decisions = decisions.cpu().numpy()
        d = np.ascontiguousarray(decisions, dtype=np.int64)
        if d.ndim == 1:
            d = d.reshape(1, -1)
        if d.size == 0:
            return
        self._core.apply(d)

    def check_sets(self, lits, offsets, *, chunk: int = 0, assignments: bool = False,
                   sat_count: bool = False) -> tuple:
        """Entry point 1, batched: any number of assumption sets (flattened DIMACS literals +
        (n+1,) offsets), chunked to the GPU by the CORE (cpp/src/check.cpp, one loop for every
        face) and returned in input order as numpy arrays; `chunk` <= 0 sizes launches."""
        return self._core.check_sets(lits, offsets, chunk=chunk, assignments=assignments,
                                     sat_count=sat_count)

    def check_chunk_size(self, n_sets: int, max_lits: int = 1) -> int:
        return self._core.check_chunk_size(n_sets, max_lits)

    def run_trajectory(self, order, n_support: int, B: int, seed: int,
                       *, soft: bool = False) -> int:
        """Whole randomized decide+propagate trajectories for every row, in ONE launch (entry
        point 2). `order` is (P,) var indices with support vars first; the kernel picks the next B
        still-free vars with a stateless per-(chunk,k,v) hash polarity, so runs are reproducible
        from `seed` alone. Returns the max decision rounds any row used."""
        if isinstance(order, torch.Tensor) and order.is_cuda:
            o = order.to(torch.int32).contiguous()
            return self._core.trajectory_dev(o.data_ptr(), o.numel(), n_support, B, seed, soft)
        if isinstance(order, torch.Tensor):
            order = order.cpu().numpy()
        return self._core.trajectory(np.ascontiguousarray(order, dtype=np.int32),
                                     n_support, B, seed, soft)

    def recache_flush(self) -> None:
        """Re-cache the flush state from row 0 of the CURRENT state — the incremental-units hook
        (LiteralProbe.assume), folding entailed units into the flush without a rebuild."""
        self._core.recache_flush()


    def make_pair_mask(self, var_mask: "torch.Tensor | None") -> int:
        """Register the variable set a completion test considers; returns a mask id (cached).
        `var_mask` is a (N,) bool tensor or None for "all variables"; padding pairs beyond N are
        excluded by construction, where an even-bits constant would read them as free."""
        if var_mask is None:
            return self._core.pair_mask(None)
        vs = torch.nonzero(var_mask.reshape(-1)).reshape(-1).cpu().numpy().astype(np.int32)
        return self._core.pair_mask(vs)

    def select_complete(self, mask_id: int, slot: int = 0) -> int:
        """Rows that are active AND have no free variable under `mask_id` — the count of
        guaranteed-valid models in this chunk. One kernel + one 4-byte sync; two slots, so "fully
        assigned" and "support complete" can coexist."""
        return self._core.select_complete(mask_id, slot)

    def check_sound(self, slot: int = 0, max_rows: int = 0) -> int:
        """How many selected rows have ANY clause with sat_count == 0 — must be 0. At a fully
        assigned row, non-falsified <=> TRUE, so `all sat_count > 0` <=> satisfied; evaluated on
        device. `max_rows = 0` (the default) checks EVERY selected row, a positive bound only
        that PREFIX."""
        return self._core.check_sound(slot, max_rows)

    def var_bits(self, slot: int, cols, count: int) -> np.ndarray:
        """(count, len(cols)) uint8 — 1 iff variable cols[j] is TRUE in the i-th selected row."""
        if isinstance(cols, torch.Tensor):
            cols = cols.cpu().numpy()
        return self._core.var_bits(slot, np.asarray(cols, dtype=np.int32), count)


    @property
    def active(self) -> torch.Tensor:
        return torch.from_numpy(self._core.flags()[0]).to(self._device)

    @property
    def conflict(self) -> torch.Tensor:
        return torch.from_numpy(self._core.flags()[1]).to(self._device)

    @property
    def assignment(self) -> torch.Tensor:
        """(K, 2N) int8 (1 = TRUE), materialized from the bitmap on every call: a copy, not the
        live state. Prefer the selection accessors on hot paths."""
        return self.assignment_rows(np.arange(self.K, dtype=np.int32))

    def assignment_rows(self, row_idx) -> torch.Tensor:
        return torch.from_numpy(self._core.rows_assignment(_row_indices(row_idx))).to(self._device)

    def sat_rows(self, row_idx) -> torch.Tensor:
        """sat_count values for the given rows — the supported accessor for app-level soundness
        checks (`check_sound` is the on-device version)."""
        return torch.from_numpy(self._core.sat_rows(_row_indices(row_idx))).to(self._device)

    @property
    def num_propagated(self) -> torch.Tensor:
        return torch.from_numpy(self._core.num_propagated()).to(self._device)

    @property
    def flush_conflict(self) -> bool:
        """True iff the formula's own units already conflict (formula UNSAT)."""
        return self._core.flush_conflict


    @property
    def launches(self) -> int:
        return int(self._core.counters().launches)

    def ops_total(self) -> int:
        return int(self._core.counters().ops)

    def props_total(self) -> int:
        return int(self._core.counters().props)

    def decs_total(self) -> int:
        return int(self._core.counters().decs)


def _row_indices(row_idx) -> np.ndarray:
    if isinstance(row_idx, torch.Tensor):
        if row_idx.dtype == torch.bool:
            row_idx = torch.nonzero(row_idx).reshape(-1)
        row_idx = row_idx.cpu().numpy()
    idx = np.asarray(row_idx)
    if idx.dtype == bool:
        idx = np.nonzero(idx)[0]
    return np.ascontiguousarray(idx, dtype=np.int32)
