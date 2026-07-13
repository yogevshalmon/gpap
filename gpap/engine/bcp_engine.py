"""BCPEngine — the K-parallel packed BCP engine (state management + kernel launches).

All propagation executes in the Triton kernels (gpap.engine.kernels); this class owns the
state and the launch plumbing:

  * packed clause state (K, M) int32 — sat_count | XOR-of-remaining-literals, field split
    chosen per formula (SHIFT = 16 or 8; formulas outside both envelopes are rejected);
  * assignment (K, 2N) int8 (1 = TRUE), active / conflict (K,) bool;
  * the post-units-flush cache: the state after asserting the formula's own unit clauses is
    row-identical for every row of every chunk, so it is computed ONCE at construction (one
    fixpoint launch at K=1) and `reset(K)` broadcast-restores it — no per-chunk flush;
  * lifetime work counters accumulated ON GPU (no per-launch sync; the *_total() reads sync).

Requires CUDA + a working triton; construction raises otherwise. There is no CPU fallback —
correctness is validated against a pure-Python reference BCP (tests/reference_bcp.py) and by
external-solver model verification, not by a second production path.
"""
import torch

from gpap.engine.formula import CnfFormula


def make_assumption_tensor(
    assumptions_lit_indices: list[list[int]],
    device: torch.device,
) -> torch.Tensor:
    """Convert list[list[int]] to a (K, L) int64 padded tensor for BCPEngine.apply_and_propagate.

    L = max assumption count across all batches; empty slots filled with -1.
    Build this once before any timed / hot path.
    """
    K = len(assumptions_lit_indices)
    L = max((len(lits) for lits in assumptions_lit_indices), default=0)
    t = torch.full((K, L), -1, dtype=torch.long)
    for k, lits in enumerate(assumptions_lit_indices):
        if lits:
            t[k, :len(lits)] = torch.tensor(lits, dtype=torch.long)
    return t.to(device)


class BCPEngine:
    """Packed-state BCP over K independent rows. Two entry points (see gpap.engine.__init__):

        eng = BCPEngine(formula)                 # builds occ + packed init + the flush cache
        eng.reset(K)                             # broadcast-restore the post-flush state
        eng.apply_and_propagate(dec)             # (K,L) int64 lit indices, -1 pad; ONE launch
        eng.run_trajectory(order, n_support, B, seed, soft=...)   # whole sampling loop, ONE launch
        eng.active / eng.conflict / eng.assignment / eng.sat_rows(idx)

    Setting literal l TRUE decrements the packed word of every clause in occ[l] (CSR lists of
    the NEGATED literal) and XORs ~l out of its remaining-literal field; sat==1 forces the
    literal the XOR field holds, sat==0 conflicts the row. No clause scanning, ever.
    """

    _WIDE_MAX_2N = 0xFFFFFF          # 8|24 split: 24-bit literal index ceiling
    _WIDE_MAX_LEN = 255              # 8|24 split: 8-bit sat_count ceiling

    def __init__(self, formula: CnfFormula, *, force_wide: bool = False):
        from gpap.engine.kernels import kernel_available
        self._formula = formula
        self._device = formula.device
        self._num_vars = formula.num_vars
        self._num_clauses = formula.num_clauses
        if self._device.type != "cuda" or not kernel_available():
            raise RuntimeError(
                "BCPEngine requires CUDA + a working triton kernel (the engine is packed-"
                "kernel-only; there is no CPU path). Check `gpap info`.")

        M, n2 = self._num_clauses, 2 * self._num_vars
        max_len = int(formula.clause_sizes.max().item()) if M else 0
        wide_ok = n2 <= self._WIDE_MAX_2N and max_len <= self._WIDE_MAX_LEN
        if n2 <= 65535 and not force_wide:
            self._shift = 16
        elif wide_ok:
            self._shift = 8              # the "wide" split (or the force_wide test hook)
        else:
            raise ValueError(
                f"formula outside the packed envelopes: 2N={n2}, max clause len={max_len} "
                f"(16|16 needs 2N<=65535; 8|24 needs 2N<={self._WIDE_MAX_2N} and "
                f"len<={self._WIDE_MAX_LEN}). Split long clauses or reduce variables.")

        # int32 CSR (formula already stores int32 — just pin contiguity for the kernels)
        occ = formula.occ_list
        self._occ_off = occ.occ_offsets.contiguous()
        self._occ_cls = occ.occ_clauses.contiguous()

        # packed init row: sat_count | (xor << SHIFT). Packed in uint32 (the shifted XOR may
        # legitimately reach bit 31) then reinterpreted as the kernel's int32 — a numpy view is
        # exact where an integer cast could saturate.
        import numpy as np
        xor_mask = (1 << (32 - self._shift)) - 1
        sizes = formula.clause_sizes.cpu().numpy().astype(np.uint32)
        xors = formula.init_lit_xor.cpu().numpy().astype(np.uint32)
        packed = (sizes | ((xors & xor_mask) << self._shift)).view(np.int32)
        self._packed_init = torch.from_numpy(packed).to(self._device)

        # row-state tensors, sized on first reset()
        self._packed = None              # (K, M) int32
        self._assignment = None          # (K, 2N) int8
        self._active = None              # (K,) bool
        self._conflict = None            # (K,) bool
        # kernel workspace / per-launch outputs (sized with K / B in _ensure)
        self._trail = None               # (K, cap) int32
        self._nprops = None              # (K,) int32: literals set TRUE last launch
        self._nops = None                # (K,) int32: occ entries processed last launch
        self._ndecs = None               # (K,) int32: decision literals last launch
        self._nrounds = None             # (K,) int32: trajectory decision rounds last launch
        # lifetime counters (GPU 0-d accumulators; *_total() syncs lazily)
        self._ops_acc = torch.zeros((), dtype=torch.int64, device=self._device)
        self._props_acc = torch.zeros((), dtype=torch.int64, device=self._device)
        self._decs_acc = torch.zeros((), dtype=torch.int64, device=self._device)
        self.launches = 0

        self._build_flush_cache()

    # ------------------------------------------------------------------ construction

    @torch.inference_mode()
    def _build_flush_cache(self) -> None:
        """One fixpoint launch at K=1 seeded with the formula's unit-clause literals; the
        resulting state is cached and broadcast-restored by every reset(K). An empty clause
        (or contradictory/conflicting units) makes the flush itself conflict — reset() then
        marks every row conflicted (the formula is UNSAT; 0 models either way)."""
        f = self._formula
        M = self._num_clauses
        self.reset(1, _from_init=True)
        if f.has_empty_clause:
            self._conflict.fill_(True)
            self._active.fill_(False)
        elif M:
            unit_mask = f.clause_sizes == 1
            if bool(unit_mask.any()):
                units = f.init_lit_xor[unit_mask].to(torch.int64).view(1, -1)
                self.apply_and_propagate(units)
        self._packed_flushed = self._packed[0].clone()
        self._asg_flushed = self._assignment[0].clone()
        self._flush_conflict = bool(self._conflict[0].item())
        # the flush is construction, not sampled work — reset the lifetime counters
        self._ops_acc.zero_(); self._props_acc.zero_(); self._decs_acc.zero_()
        self.launches = 0

    @torch.inference_mode()
    def recache_flush(self) -> None:
        """Re-cache the flush state from row 0 of the CURRENT state — the incremental-units
        hook (miner): reset(1) + apply_and_propagate(new_units) + recache_flush() folds
        entailed units into the cached flush without rebuilding the formula."""
        self._packed_flushed = self._packed[0].clone()
        self._asg_flushed = self._assignment[0].clone()
        self._flush_conflict = bool(self._conflict[0].item())

    def _ensure(self, K: int, B: int) -> None:
        cap = self._num_vars + max(B, 64) + 8
        if self._trail is None or self._trail.size(0) != K or self._trail.size(1) < cap:
            dev = self._device
            self._trail = torch.empty((K, cap), dtype=torch.int32, device=dev)
            self._nprops = torch.zeros(K, dtype=torch.int32, device=dev)
            self._nops = torch.zeros(K, dtype=torch.int32, device=dev)
            self._ndecs = torch.zeros(K, dtype=torch.int32, device=dev)
            self._nrounds = torch.zeros(K, dtype=torch.int32, device=dev)

    # ------------------------------------------------------------------ entry points

    @torch.inference_mode()
    def reset(self, K: int, _from_init: bool = False) -> None:
        """Reset K rows to the cached post-flush state (allocations reused when K matches)."""
        dev = self._device
        M, n2 = self._num_clauses, 2 * self._num_vars
        if self._assignment is None or self._assignment.size(0) != K:
            self._packed = torch.empty((K, M), dtype=torch.int32, device=dev)
            self._assignment = torch.empty((K, n2), dtype=torch.int8, device=dev)
            self._active = torch.empty(K, dtype=torch.bool, device=dev)
            self._conflict = torch.empty(K, dtype=torch.bool, device=dev)
        if _from_init:                       # construction path: pre-flush init state
            self._packed.copy_(self._packed_init.unsqueeze(0).expand(K, -1))
            self._assignment.zero_()
            self._active.fill_(True)
            self._conflict.fill_(False)
            return
        self._packed.copy_(self._packed_flushed.unsqueeze(0).expand(K, -1))
        self._assignment.copy_(self._asg_flushed.unsqueeze(0).expand(K, -1))
        self._active.fill_(not self._flush_conflict)
        self._conflict.fill_(self._flush_conflict)

    @torch.inference_mode()
    def apply_and_propagate(self, decisions: torch.Tensor) -> None:
        """Apply a (K, L) int64 decision tensor (-1 = padding) and BCP to fixpoint — ONE launch.
        Duplicate literals are skipped; l/~l pairs and decisions contradicting the existing
        assignment conflict the row. Entry point 1 (batched assumption checking)."""
        from gpap.engine.kernels import _bcp_fixpoint_kernel
        K = self._assignment.size(0)
        decisions = decisions.to(self._device)
        if decisions.dim() == 1:
            decisions = decisions.unsqueeze(1)
        B = decisions.size(1)
        if decisions.numel() == 0:
            return
        self._ensure(K, B)
        # Zero the per-row outputs: INACTIVE rows are left unwritten by the kernel, and
        # summing stale values would corrupt the accumulators.
        self._nprops.zero_(); self._nops.zero_(); self._ndecs.zero_()
        dec = decisions.to(torch.int64).contiguous()
        _bcp_fixpoint_kernel[(K,)](
            self._packed, self._assignment,
            self._active.view(torch.int8), self._conflict.view(torch.int8),
            dec, self._occ_off, self._occ_cls, self._trail,
            self._nprops, self._nops, self._ndecs,
            self._num_clauses, 2 * self._num_vars, B, self._trail.size(1),
            SHIFT=self._shift, num_warps=1,
        )
        self.launches += 1
        self._ops_acc += self._nops.sum()               # GPU-side, no sync
        self._props_acc += self._nprops.sum()
        self._decs_acc += self._ndecs.sum()

    @torch.inference_mode()
    def run_trajectory(self, order: torch.Tensor, n_support: int, B: int, seed: int,
                       *, soft: bool = False) -> int:
        """Whole randomized decide+propagate trajectories for every row — ONE launch. `order` =
        (P,) var indices, support vars first; picking the next B still-free vars in `order` with
        a stateless per-(chunk,k,v) hash polarity (the torch RNG stream is NOT consumed, so runs
        are reproducible via `seed` alone). `soft` enables the in-kernel risk-polarity. Entry
        point 2 (trajectory sampling). Returns the max decision rounds any row used (one sync)."""
        from gpap.engine.kernels import _bcp_trajectory_kernel
        K = self._assignment.size(0)
        self._ensure(K, 64)
        self._nprops.zero_(); self._nops.zero_(); self._ndecs.zero_(); self._nrounds.zero_()
        order = order.to(torch.int32).contiguous()
        _bcp_trajectory_kernel[(K,)](
            self._packed, self._assignment,
            self._active.view(torch.int8), self._conflict.view(torch.int8),
            order, self._occ_off, self._occ_cls,
            self._trail, self._nrounds, self._nops, self._nprops, self._ndecs,
            self._num_clauses, 2 * self._num_vars, order.numel(), n_support, B,
            self._trail.size(1), 1 << 30, seed,
            SOFT=(1 if soft else 0), SHIFT=self._shift, num_warps=1,
        )
        self.launches += 1
        self._ops_acc += self._nops.sum()
        self._props_acc += self._nprops.sum()
        self._decs_acc += self._ndecs.sum()
        return int(self._nrounds.max().item())

    # ------------------------------------------------------------------ accessors

    def sat_rows(self, row_idx: torch.Tensor) -> torch.Tensor:
        """sat_count values for the given rows (extracted from the packed state). The supported
        accessor for app-level soundness checks ('every clause > 0' on a fully-assigned row)."""
        return self._packed[row_idx] & ((1 << self._shift) - 1)

    @property
    def assignment(self) -> torch.Tensor:
        return self._assignment

    @property
    def active(self) -> torch.Tensor:
        return self._active

    @property
    def conflict(self) -> torch.Tensor:
        return self._conflict

    @property
    def num_propagated(self) -> torch.Tensor:
        """(K,) int32 — unit-propagation IMPLICATIONS of the last launch (assigned minus
        decisions)."""
        return self._nprops - self._ndecs

    @property
    def flush_conflict(self) -> bool:
        """True iff the formula's own units already conflict (formula UNSAT)."""
        return self._flush_conflict

    @property
    def flushed_assignment(self) -> torch.Tensor:
        """(2N,) int8 — the cached post-flush assignment row (formula units + implications)."""
        return self._asg_flushed

    @property
    def shift(self) -> int:
        """Packed field split: 16 (16|16) or 8 (8|24 'wide')."""
        return self._shift

    @property
    def num_vars(self) -> int:
        return self._num_vars

    @property
    def num_clauses(self) -> int:
        return self._num_clauses

    @property
    def device(self) -> torch.device:
        return self._device

    def ops_total(self) -> int:
        """Occ entries processed so far (syncs; read off the hot path)."""
        return int(self._ops_acc.item())

    def props_total(self) -> int:
        """Literals assigned so far — decisions + implications (syncs; off the hot path)."""
        return int(self._props_acc.item())

    def decs_total(self) -> int:
        """Decision literals assigned so far (syncs; off the hot path).
        Implications = props_total() - decs_total()."""
        return int(self._decs_acc.item())


if __name__ == "__main__":
    from gpap.engine.formula import ext_lit_to_lit_index

    if not torch.cuda.is_available():
        raise SystemExit("demo requires CUDA (the engine is kernel-only)")
    device = torch.device("cuda")
    formula = CnfFormula(device)
    # (x1 OR x2), (~x1 OR x3), (~x2 OR ~x3), (x1 OR ~x2)
    formula.init_formula(3, [[1, 2], [-1, 3], [-2, -3], [1, -2]])

    engine = BCPEngine(formula)
    engine.reset(2)
    # batch 0: assume x1=TRUE, batch 1: assume x1=FALSE
    dec = make_assumption_tensor([[ext_lit_to_lit_index(1)],
                                  [ext_lit_to_lit_index(-1)]], device)
    engine.apply_and_propagate(dec)

    print("Conflicts:", engine.conflict.tolist())
    print("Assignment batch 0:", engine.assignment[0].tolist())
    print("Assignment batch 1:", engine.assignment[1].tolist())
    print("Implications:", engine.num_propagated.tolist())
