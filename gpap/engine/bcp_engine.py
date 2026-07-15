"""BCPEngine — the K-parallel packed BCP engine (state management + kernel launches).

All propagation executes in the Triton kernels (gpap.engine.kernels); this class owns the
state and the launch plumbing:

  * packed clause state (K, M) int32 — sat_count | modular-SUM-of-remaining-literals, field
    split chosen per formula (SHIFT = 16 or 8; formulas outside both envelopes are rejected);
  * assignment BITMAP (K, ceil(2N/32)) int32 — bit l set = literal l TRUE (l and ~l share a
    word, which is what makes the kernels' one-atomic dedup/contradiction work; 8x smaller
    than a byte per literal). Consumers read it through the materializing accessors
    (`assignment`, `assignment_rows`, `var_true_bits`, `any_free`, `flushed_assignment`) —
    copies/derived tensors, never the live words;
  * active / conflict (K,) bool;
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
    the NEGATED literal) and subtracts ~l from its remaining-literal sum field; sat==1 forces
    the literal the sum field holds, sat==0 conflicts the row. No clause scanning, ever.
    """

    _WIDE_MAX_2N = 0xFFFFFF          # 8|24 split: 24-bit literal index ceiling
    _WIDE_MAX_LEN = 255              # 8|24 split: 8-bit sat_count ceiling

    def __init__(self, formula: CnfFormula, *, force_wide: bool = False,
                 mode: "str | None" = None):
        """`mode`: "scalar" (default) = the serial-trail kernels; "wave" (EXPERIMENTAL) = the
        warp-wide wavefront propagate (see kernels._wave_propagate) — same state layout, same
        fixpoints (confluence), different propagation order. Env override: GPAP_KERNEL=wave
        (+ GPAP_WAVE_W for the lane width, default 32; GPAP_VSCAN=0 disables the warp-wide
        decision scan in the trajectory kernel)."""
        import os
        from gpap.engine.kernels import kernel_available
        self._mode = mode or os.environ.get("GPAP_KERNEL", "scalar")
        if self._mode not in ("scalar", "wave"):
            raise ValueError(f"unknown kernel mode {self._mode!r} (scalar|wave)")
        self._wave_w = int(os.environ.get("GPAP_WAVE_W", "32"))
        self._vscan = 0 if os.environ.get("GPAP_VSCAN") == "0" else 1
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

        # packed init row: sat_count | (lit_sum << SHIFT), sum taken mod the field width.
        # Packed in uint32 (the shifted sum may legitimately reach bit 31) then reinterpreted
        # as the kernel's int32 — a numpy view is exact where an integer cast could saturate.
        import numpy as np
        sum_mask = (1 << (32 - self._shift)) - 1
        sizes = formula.clause_sizes.cpu().numpy().astype(np.uint32)
        sums = formula.init_lit_sum.cpu().numpy().astype(np.uint32)
        packed = (sizes | ((sums & sum_mask) << self._shift)).view(np.int32)
        self._packed_init = torch.from_numpy(packed).to(self._device)

        # row-state tensors, sized on first reset()
        self._nw = max(1, (n2 + 31) // 32)   # assignment-bitmap words per row
        self._packed = None              # (K, M) int32
        self._abits = None               # (K, NW) int32 assignment bitmap (bit l = lit l TRUE)
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
                units = f.init_lit_sum[unit_mask].to(torch.int64).view(1, -1)
                self.apply_and_propagate(units)
        self._packed_flushed = self._packed[0].clone()
        self._abits_flushed = self._abits[0].clone()
        self._flush_conflict = bool(self._conflict[0].item())
        # the flush is construction, not sampled work — reset the lifetime counters
        self._ops_acc.zero_(); self._props_acc.zero_(); self._decs_acc.zero_()
        self.launches = 0

    def set_mode(self, mode: str) -> None:
        """Switch the propagate kernel variant ("scalar" | "wave"). The two variants share the
        SAME state layout and reach the SAME fixpoints (confluence — gate-validated
        bit-identical), so this is purely a launch-flag change and is safe at any point.
        Measured (RTX 4080, 2026-07-13): wave wins where K is below warp saturation
        (b14 K=280: 3.06x; b12 K=3k: 1.31x; b04 K=4k: 1.14x) and loses ~0.9x where DRAM is
        already saturated (b03 K=30k) — GPUSampler auto-dispatches on K."""
        if mode not in ("scalar", "wave"):
            raise ValueError(f"unknown kernel mode {mode!r} (scalar|wave)")
        self._mode = mode

    @property
    def mode(self) -> str:
        return self._mode

    @torch.inference_mode()
    def recache_flush(self) -> None:
        """Re-cache the flush state from row 0 of the CURRENT state — the incremental-units
        hook (miner): reset(1) + apply_and_propagate(new_units) + recache_flush() folds
        entailed units into the cached flush without rebuilding the formula."""
        self._packed_flushed = self._packed[0].clone()
        self._abits_flushed = self._abits[0].clone()
        self._flush_conflict = bool(self._conflict[0].item())

    def _num_warps(self) -> int:
        """Warps per program: in wave mode the W-wide tensors are spread over W/32 REAL warps
        (one element per lane), multiplying scattered requests in flight per row — the old
        W=64 test ran on one warp (2 elements/lane = serialization, measured worse) and is
        superseded. Scalar mode stays single-warp (W is only the decision-scan width there)."""
        return max(1, self._wave_w // 32) if self._mode == "wave" else 1

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
        M = self._num_clauses
        if self._abits is None or self._abits.size(0) != K:
            self._packed = torch.empty((K, M), dtype=torch.int32, device=dev)
            self._abits = torch.empty((K, self._nw), dtype=torch.int32, device=dev)
            self._active = torch.empty(K, dtype=torch.bool, device=dev)
            self._conflict = torch.empty(K, dtype=torch.bool, device=dev)
        if _from_init:                       # construction path: pre-flush init state
            self._packed.copy_(self._packed_init.unsqueeze(0).expand(K, -1))
            self._abits.zero_()
            self._active.fill_(True)
            self._conflict.fill_(False)
            return
        self._packed.copy_(self._packed_flushed.unsqueeze(0).expand(K, -1))
        self._abits.copy_(self._abits_flushed.unsqueeze(0).expand(K, -1))
        self._active.fill_(not self._flush_conflict)
        self._conflict.fill_(self._flush_conflict)

    @torch.inference_mode()
    def apply_and_propagate(self, decisions: torch.Tensor) -> None:
        """Apply a (K, L) int64 decision tensor (-1 = padding) and BCP to fixpoint — ONE launch.
        Duplicate literals are skipped; l/~l pairs and decisions contradicting the existing
        assignment conflict the row. Entry point 1 (batched assumption checking)."""
        from gpap.engine.kernels import _bcp_fixpoint_kernel
        K = self._abits.size(0)
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
            self._packed, self._abits,
            self._active.view(torch.int8), self._conflict.view(torch.int8),
            dec, self._occ_off, self._occ_cls, self._trail,
            self._nprops, self._nops, self._ndecs,
            self._num_clauses, self._nw, B, self._trail.size(1),
            SHIFT=self._shift, WAVE=(1 if self._mode == "wave" else 0), W=self._wave_w,
            num_warps=self._num_warps(),
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
        K = self._abits.size(0)
        self._ensure(K, 64)
        self._nprops.zero_(); self._nops.zero_(); self._ndecs.zero_(); self._nrounds.zero_()
        order = order.to(torch.int32).contiguous()
        _bcp_trajectory_kernel[(K,)](
            self._packed, self._abits,
            self._active.view(torch.int8), self._conflict.view(torch.int8),
            order, self._occ_off, self._occ_cls,
            self._trail, self._nrounds, self._nops, self._nprops, self._ndecs,
            self._num_clauses, self._nw, order.numel(), n_support, B,
            self._trail.size(1), 1 << 30, seed,
            SOFT=(1 if soft else 0), SHIFT=self._shift,
            WAVE=(1 if self._mode == "wave" else 0), W=self._wave_w,
            VSCAN=self._vscan, num_warps=self._num_warps(),
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

    @torch.inference_mode()
    def _bits_to_bytes(self, words: torch.Tensor) -> torch.Tensor:
        """(R, NW) int32 bitmap rows -> (R, 2N) int8, the historical byte-per-literal layout.
        A COPY. Contiguous unpack (strided slice-writes measured 16x sector amplification);
        row-sliced so the (rows, NW, 32) int32 transient stays <= ~256 MB."""
        R = words.size(0)
        shifts = torch.arange(32, dtype=torch.int32, device=words.device)
        out = torch.empty((R, self._nw * 32), dtype=torch.int8, device=words.device)
        step = max(1, int(256e6 // max(self._nw * 32 * 4, 1)))
        for r0 in range(0, R, step):
            w = words[r0:r0 + step]
            out[r0:r0 + step] = ((w.unsqueeze(-1) >> shifts) & 1).to(torch.int8).view(
                w.size(0), -1)
        return out[:, :2 * self._num_vars]

    @property
    def assignment(self) -> torch.Tensor:
        """(K, 2N) int8 (1 = TRUE) — MATERIALIZED from the bitmap on every call; a copy, not
        the live state. Prefer assignment_rows / var_true_bits / any_free on hot paths."""
        return self._bits_to_bytes(self._abits)

    def assignment_rows(self, row_idx: torch.Tensor) -> torch.Tensor:
        """(n, 2N) int8 assignments for the selected rows (bool mask or index tensor) —
        materialized copy of just those rows."""
        return self._bits_to_bytes(self._abits[row_idx])

    @torch.inference_mode()
    def var_true_bits(self, row_idx: torch.Tensor, var_cols: torch.Tensor) -> torch.Tensor:
        """(n, V) uint8 — 1 iff var v is assigned TRUE (positive literal bit 2v set), for the
        selected rows x selected var columns. Direct bit gather; no N-wide materialization."""
        w = self._abits[row_idx][:, (var_cols >> 4).long()]
        return ((w >> ((var_cols & 15).to(torch.int32) * 2)) & 1).to(torch.uint8)

    @torch.inference_mode()
    def make_pair_mask(self, var_mask: "torch.Tensor | None") -> torch.Tensor:
        """(NW,) int32 with EVEN bit 2v set for every selected var (all vars when None) —
        the required `pair_mask` for any_free(). Padding pairs beyond N stay 0 (a raw
        even-bits constant would misread them as free)."""
        import numpy as np
        vs = (np.arange(self._num_vars, dtype=np.int64) if var_mask is None
              else np.nonzero(var_mask.cpu().numpy())[0])
        bits = np.zeros(self._nw, dtype=np.uint32)
        np.bitwise_or.at(bits, vs >> 4, np.uint32(1) << (2 * (vs & 15)).astype(np.uint32))
        return torch.from_numpy(bits.view(np.int32)).to(self._device)

    @torch.inference_mode()
    def any_free(self, pair_mask: torch.Tensor) -> torch.Tensor:
        """(K,) bool — row has at least one free var among pair_mask (make_pair_mask output).
        One bitwise pass over the bitmap words: pair v free iff bits 2v/2v+1 both 0, i.e.
        even bit of ~(w | w>>1); no (K, N) mask is ever materialized."""
        w = self._abits
        return (((~(w | (w >> 1))) & pair_mask) != 0).any(dim=1)

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
        """(2N,) int8 — the cached post-flush assignment row (formula units + implications).
        Materialized from the cached bitmap row (a copy)."""
        return self._bits_to_bytes(self._abits_flushed.unsqueeze(0))[0]

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
