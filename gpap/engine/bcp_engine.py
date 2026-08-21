"""BCPEngine — the K-parallel packed BCP engine, backed by the C++ core (libgpap).

The engine *implementation* lives in `cpp/` and is shared, byte for byte, with C++ consumers:
one formula builder, one state layout, one set of AOT-compiled Triton kernels. This module is
the Python face of it — argument marshalling and the torch-friendly accessors — over
`gpap.engine._core` (a ctypes binding to the C ABI in `cpp/include/gpap/gpap.h`).

State layout (identical to the C core, see docs/ENGINE.md):
  * packed clause word (K, M) int32 — low SHIFT bits = sat_count, high bits = the modular SUM
    of the surviving literal indices; at sat_count == 1 that field IS the forced literal;
  * assignment BITMAP (K, ceil(2N/32)) int32 — bit l set = literal l TRUE (l and ~l share a
    word, which is what makes the kernels' one-atomic dedup/contradiction work);
  * the post-units-flush cache, computed ONCE at construction and broadcast-restored by
    `reset(K)` — no per-chunk unit flush.

Two entry points (`apply_and_propagate`, `run_trajectory`) plus accessors; application code
never touches the row state directly. Hot paths should use the SELECTION api
(`make_pair_mask` + `select_complete` + `var_bits` / `assignment_rows` / `check_sound`), which
keeps the completion test and the soundness check on the GPU and returns only counts and small
matrices. The tensor-shaped accessors (`assignment`, `conflict`, `active`, `sat_rows`) are
convenience copies for tests and for off-hot-path consumers.
"""
import numpy as np
import torch

from gpap.engine import _core
from gpap.engine.calibrate import dispatch_mode
from gpap.engine.formula import CnfFormula

# ---------------------------------------------------------------------- packed envelopes
# The sum-trick word is ONE int32: low SHIFT bits = sat_count, high (32-SHIFT) bits = the sum of
# the surviving literal indices mod 2^(32-SHIFT). Two conditions make it exact, and they are the
# WHOLE overflow story:
#   (a) the sum field must hold any single literal index  -> 2N <= 2^(32-SHIFT)
#   (b) the count field must hold any clause's length     -> max clause len <= 2^SHIFT - 1
# Both are checked at construction and a formula outside BOTH splits is REJECTED (no unpacked
# fallback exists) — callers sweeping a benchmark suite should pre-check with packed_envelope()
# and skip, rather than take the exception.
PACKED_SPLITS = ((16, 65535, 65535), (8, 0xFFFFFF, 255))   # (SHIFT, max 2N, max clause len)


def packed_envelope(num_vars: int, max_clause_len: int) -> "tuple[int | None, str]":
    """Pick the packed field split for a formula shape: (SHIFT, "") when it fits, or
    (None, reason) when it is outside every envelope. Preference order is the measured one:
    16|16 whenever 2N fits, else the 8|24 "wide" split for the big unrolls."""
    n2 = 2 * num_vars
    for shift, max_2n, max_len in PACKED_SPLITS:
        if n2 <= max_2n and max_clause_len <= max_len:
            return shift, ""
    return None, (f"outside the packed envelopes: 2N={n2}, max clause len={max_clause_len} "
                  f"(16|16 needs 2N<=65535 and len<=65535; 8|24 needs 2N<={0xFFFFFF} and "
                  f"len<=255)")


def clauses_envelope(num_vars: int, clauses: list) -> "tuple[int | None, str]":
    """packed_envelope() straight off a clause list (computes the max clause length)."""
    return packed_envelope(num_vars, max((len(c) for c in clauses), default=0))


def make_assumption_tensor(assumptions_lit_indices: list[list[int]],
                           device: torch.device) -> torch.Tensor:
    """list[list[int]] -> a (K, L) int64 padded tensor for apply_and_propagate.

    L = max assumption count across all batches; empty slots filled with -1.
    Build this once before any timed / hot path."""
    K = len(assumptions_lit_indices)
    L = max((len(lits) for lits in assumptions_lit_indices), default=0)
    t = torch.full((K, L), -1, dtype=torch.long)
    for k, lits in enumerate(assumptions_lit_indices):
        if lits:
            t[k, :len(lits)] = torch.tensor(lits, dtype=torch.long)
    return t.to(device)


def core_available() -> bool:
    """True iff libgpap can be loaded (see cpp/README.md for the two build commands)."""
    return _core.available()


class BCPEngine:
    """Packed-state BCP over K independent rows, on the C++ core.

        eng = BCPEngine(formula)                 # occ + packed init + the flush cache
        eng.reset(K)                             # broadcast-restore the post-flush state
        eng.apply_and_propagate(dec)             # (K,L) int64 lit indices, -1 pad; ONE launch
        eng.run_trajectory(order, n_support, B, seed, soft=...)   # whole sampling loop, ONE launch

        mid = eng.make_pair_mask(support_mask)   # register a variable set once
        n   = eng.select_complete(mid, slot=0)   # active AND fully assigned -> count (hot path)
        eng.check_sound(0) / eng.var_bits(0, cols, n)

    Setting literal l TRUE decrements the packed word of every clause in occ[l] (CSR lists of
    the NEGATED literal) and subtracts ~l from its remaining-literal sum field; sat==1 forces
    the literal the sum field holds, sat==0 conflicts the row. No clause scanning, ever.
    """

    def __init__(self, formula: CnfFormula, *, force_wide: bool = False,
                 mode: "str | None" = None):
        """`mode`: None resolves to $GPAP_KERNEL when set, else "auto" (resolved by the
        `_core.Engine` binding, ONCE, so every face built on it honours the override). "auto"
        picks the propagate kernel per `reset(K)` — see `plan_for`. "scalar" = the serial-trail
        kernels, "wave" = the warp-wide wavefront propagate; naming either PINS it. Same state,
        same fixpoints (confluence), different propagation order.

        Auto is the default so that every consumer of the engine gets the dispatch, not just the
        one that remembered to ask — and it is RESOLVED IN THE CORE, on every `gpap_engine_reset`,
        so that promise covers consumers this class knows nothing about (the C++ face, the probe
        sweep in cpp/src/probe.cpp)."""
        # `force_wide` pins the 8|24 split on a formula that would otherwise take 16|16 — the
        # A/B hook that lets the wide kernel be exercised on small, fast formulas.
        self._formula = formula
        self._device = formula.device
        if self._device.type != "cuda":
            raise RuntimeError("BCPEngine requires CUDA (the engine is GPU-only; there is no "
                               "CPU path). Check `gpap info`.")
        self._core = _core.Engine(formula.core, device=self._device.index or 0, mode=mode,
                                  force_shift=8 if force_wide else 0)
        self._mode = self._core.requested_mode
        self._auto_mode = self._mode == "auto"
        self._K = 0
        self._counter_base = (0, 0, 0, 0)

    # ------------------------------------------------------------------ configuration

    def set_mode(self, mode: str) -> None:
        """PIN the propagate kernel variant ("scalar" | "wave"), disabling auto-dispatch.

        Both share the SAME state and reach the SAME fixpoints (confluence — gate-validated
        bit-identical), so this is a pure launch-flag change, safe at any point. Pin one when the
        run is a MEASUREMENT and the kernel must not be a free variable; otherwise leave the
        engine on "auto" and let `plan_for` choose."""
        self._core.set_mode(mode)
        self._mode = mode
        self._auto_mode = mode == "auto"

    @property
    def auto_dispatch(self) -> bool:
        """True while no kernel is pinned — neither by the constructor/`$GPAP_KERNEL` nor by a
        later `set_mode`. A caller about to pin a mode of its own (the sampler does, to keep its
        warmup and short final chunk on the measured kernel) should check this first, so it
        never overrides an explicit pin."""
        return self._auto_mode

    def plan_for(self, K: int) -> str:
        """The propagate kernel a chunk of K rows will run. PURE — `reset(K)` is what applies it.

        Kept because a caller sizing a chunk may want to know the kernel before committing to it;
        it returns the pinned mode verbatim when one is pinned."""
        if self._auto_mode:
            return dispatch_mode(self.num_clauses, int(K), self._device)
        return self._core.mode

    @property
    def mode(self) -> str:
        return self._core.mode

    @property
    def shift(self) -> int:
        """Packed field split: 16 (16|16) or 8 (8|24 'wide')."""
        return self._core.shift

    @property
    def num_vars(self) -> int:
        return self._core.num_vars

    @property
    def num_clauses(self) -> int:
        return self._core.num_clauses

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def K(self) -> int:
        return self._K

    # ------------------------------------------------------------------ entry points

    def reset(self, K: int) -> None:
        """Reset K rows to the cached post-units-flush state (allocations reused when K matches),
        and resolve the propagate kernel for that batch if the mode is "auto" (done by the core,
        which is what makes the dispatch reach every consumer of the ABI and not just this one)."""
        self._core.reset(K)
        self._K = int(K)

    def apply_and_propagate(self, decisions) -> None:
        """Apply a (K, L) int64 decision tensor/array (-1 = padding) and BCP to fixpoint — ONE
        launch. Duplicate literals are skipped; l/~l pairs and decisions contradicting the
        existing assignment conflict the row. Entry point 1 (batched assumption checking)."""
        if isinstance(decisions, torch.Tensor):
            if decisions.dim() == 1:
                decisions = decisions.unsqueeze(1)
            if decisions.numel() == 0:
                return
            if decisions.is_cuda:
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
        """Entry point 1, batched: ANY number of assumption sets (flattened DIMACS literals +
        (n+1,) offsets), chunked to the GPU by the CORE (cpp/src/check.cpp — one loop for every
        face) and returned in input order as numpy arrays. `AssumptionChecker` is the
        list-of-lists / torch face over this; call it directly when the flat form is already in
        hand. `chunk` <= 0 sizes launches with `check_chunk_size`."""
        return self._core.check_sets(lits, offsets, chunk=chunk, assignments=assignments,
                                     sat_count=sat_count)

    def check_chunk_size(self, n_sets: int, max_lits: int = 1) -> int:
        """Assumption sets one `check_sets` launch holds (the core's model: `row_bytes` plus
        the decision matrix this entry point allocates)."""
        return self._core.check_chunk_size(n_sets, max_lits)

    def run_trajectory(self, order, n_support: int, B: int, seed: int,
                       *, soft: bool = False) -> int:
        """Whole randomized decide+propagate trajectories for every row — ONE launch. `order` =
        (P,) var indices, support vars first; the kernel picks the next B still-free vars in
        `order` with a stateless per-(chunk,k,v) hash polarity (the torch RNG stream is NOT
        consumed, so runs are reproducible via `seed` alone). Entry point 2. Returns the max
        decision rounds any row used (one sync).

        A CUDA tensor is passed by pointer (zero copy) — which is what keeps results identical
        to a torch-built decision order."""
        if isinstance(order, torch.Tensor) and order.is_cuda:
            o = order.to(torch.int32).contiguous()
            return self._core.trajectory_dev(o.data_ptr(), o.numel(), n_support, B, seed, soft)
        if isinstance(order, torch.Tensor):
            order = order.cpu().numpy()
        return self._core.trajectory(np.ascontiguousarray(order, dtype=np.int32),
                                     n_support, B, seed, soft)

    def recache_flush(self) -> None:
        """Re-cache the flush state from row 0 of the CURRENT state — the incremental-units
        hook (LiteralProbe.assume): reset(1) + apply_and_propagate(new_units) + recache_flush() folds
        entailed units into the cached flush without rebuilding the formula."""
        self._core.recache_flush()

    # ------------------------------------------------------------------ selections (hot path)

    def make_pair_mask(self, var_mask: "torch.Tensor | None") -> int:
        """Register the variable set a completion test considers; returns a mask id (cached).

        `var_mask` is a (N,) bool tensor or None for "all variables". Padding pairs beyond N are
        excluded by construction — a raw even-bits constant would misread them as free."""
        if var_mask is None:
            return self._core.pair_mask(None)
        vs = torch.nonzero(var_mask.reshape(-1)).reshape(-1).cpu().numpy().astype(np.int32)
        return self._core.pair_mask(vs)

    def select_complete(self, mask_id: int, slot: int = 0) -> int:
        """Rows that are active AND have no free variable under `mask_id` — the count of
        guaranteed-valid models in this chunk. One kernel + one 4-byte sync; the (K,) mask never
        crosses the bus. Two slots exist so a caller can hold "fully assigned" and "support
        complete" at the same time."""
        return self._core.select_complete(mask_id, slot)

    def check_sound(self, slot: int = 0, max_rows: int = 4096) -> int:
        """How many selected rows have ANY clause with sat_count == 0 — must be 0.

        At a FULLY-assigned row, non-falsified <=> TRUE, so `all sat_count > 0` <=> satisfied.
        Evaluated on device: no (rows, M) transfer, no slice budget."""
        return self._core.check_sound(slot, max_rows)

    def var_bits(self, slot: int, cols, count: int) -> np.ndarray:
        """(count, len(cols)) uint8 — 1 iff variable cols[j] is TRUE in the i-th selected row.
        Direct bit gather; no (K, N) materialization."""
        if isinstance(cols, torch.Tensor):
            cols = cols.cpu().numpy()
        return self._core.var_bits(slot, np.asarray(cols, dtype=np.int32), count)

    # ------------------------------------------------------------------ accessors (copies)

    @property
    def active(self) -> torch.Tensor:
        return torch.from_numpy(self._core.flags()[0]).to(self._device)

    @property
    def conflict(self) -> torch.Tensor:
        return torch.from_numpy(self._core.flags()[1]).to(self._device)

    @property
    def assignment(self) -> torch.Tensor:
        """(K, 2N) int8 (1 = TRUE) — MATERIALIZED from the bitmap on every call; a copy, not the
        live state. Prefer the selection accessors on hot paths."""
        return self.assignment_rows(np.arange(self._K, dtype=np.int32))

    def assignment_rows(self, row_idx) -> torch.Tensor:
        """(n, 2N) int8 assignments for the selected rows (bool mask or index tensor)."""
        return torch.from_numpy(self._core.rows_assignment(_row_indices(row_idx))).to(self._device)

    def sat_rows(self, row_idx) -> torch.Tensor:
        """sat_count values for the given rows (extracted from the packed state). The supported
        accessor for app-level soundness checks; `check_sound` is the on-device version."""
        return torch.from_numpy(self._core.sat_rows(_row_indices(row_idx))).to(self._device)

    @property
    def num_propagated(self) -> torch.Tensor:
        """(K,) int32 — unit-propagation IMPLICATIONS of the last launch (assigned minus
        decisions)."""
        return torch.from_numpy(self._core.num_propagated()).to(self._device)

    @property
    def flush_conflict(self) -> bool:
        """True iff the formula's own units already conflict (formula UNSAT)."""
        return self._core.flush_conflict

    # ------------------------------------------------------------------ work counters

    @property
    def launches(self) -> int:
        return int(self._core.counters().launches)

    def ops_total(self) -> int:
        """Occ entries processed so far (syncs; read off the hot path)."""
        return int(self._core.counters().ops)

    def props_total(self) -> int:
        """Literals assigned so far — decisions + implications (syncs; off the hot path)."""
        return int(self._core.counters().props)

    def decs_total(self) -> int:
        """Decision literals assigned so far. Implications = props_total() - decs_total()."""
        return int(self._core.counters().decs)


def _row_indices(row_idx) -> np.ndarray:
    """Accept a bool mask / index tensor / array and return ascending int32 row indices."""
    if isinstance(row_idx, torch.Tensor):
        if row_idx.dtype == torch.bool:
            row_idx = torch.nonzero(row_idx).reshape(-1)
        row_idx = row_idx.cpu().numpy()
    idx = np.asarray(row_idx)
    if idx.dtype == bool:
        idx = np.nonzero(idx)[0]
    return np.ascontiguousarray(idx, dtype=np.int32)
