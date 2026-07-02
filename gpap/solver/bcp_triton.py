"""Triton BCP kernel for the SAMPLER path: apply decisions + propagate to fixpoint in ONE launch.

Why: the PyTorch `BCPEngine.propagate()` is a Python round loop — every BCP round pays ~15-20 kernel
launches, 2 CUDA syncs and a `torch.unique` sort, and allocates fresh variable-sized temporaries.
Measured (RTX 4080, deep unrolled circuits): the loop is LATENCY-bound (12.9x more trajectories cost
only 3.9x wall; per-round overhead dominates at 68%+) and the temporaries over-reserve memory 5.6x.
This kernel replaces the whole `apply_assumptions_bulk(decisions); propagate()` sequence with a single
launch, eliminating the round loop, the sort, the syncs, and the temporaries.

Design — one program (~warp) per trajectory, sequential trail, NO atomics / NO barriers:
  * Each Triton program owns trajectory k exclusively: its rows of sat_count/lit_sum/assignment are
    program-private, so there are no cross-program races by construction.
  * The program keeps a "trail" queue of newly-TRUE literals. It pops one literal at a time and walks
    its occurrence list (clauses where ~lit appears), decrementing the sum-trick counters — one
    literal's occ list holds DISTINCT clauses, so even those updates need no atomics.
  * sat_count hits 1 -> the remaining literal IS lit_sum (the sum trick), claim it via the assignment
    row (dedup + l/~l contradiction check) and push it on the trail. Hits 0 -> conflict, stop the row.
  * GPU-wide parallelism comes from the K programs (thousands to hundreds of thousands), which
    saturates the SMs; per-trajectory work is uniform on the target formulas, so warps stay balanced.

Semantics contract vs the PyTorch engine (the A/B oracle; BCP fixpoints are CONFLUENT):
  * conflict/active flags: identical everywhere.
  * NON-conflicted rows: assignment / sat_count / lit_sum bit-identical to the oracle's fixpoint.
  * Conflicted rows: partial counter state is propagation-order-dependent and intentionally
    unspecified (the sampler only reads rows that are active & fully-assigned).
  * Decisions must be fresh distinct variables per row (what `_pick_decisions` produces); l/~l pairs
    and decisions contradicting the existing assignment conflict the row, mirroring
    `apply_assumptions_bulk`. `real_conflict` / `num_propagated` / soft-stop are NOT maintained
    (unused by the sampler).

Requires CUDA + triton (ships with torch on Linux/Colab — no nvcc). `TritonBCP.available()` probes
once; everything degrades gracefully to the PyTorch path when unavailable.
"""
import torch

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:  # noqa: BLE001 — any import/JIT env failure just disables the fast path
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _bcp_fixpoint_kernel(
        sat_ptr,       # (K, M)  int8/int16/int32 — engine's sat_count (row-private per program)
        lsum_ptr,      # (K, M)  int32            — engine's lit_sum
        asg_ptr,       # (K, 2N) int8             — engine's assignment (1 = TRUE)
        active_ptr,    # (K,)    int8 view of bool
        conflict_ptr,  # (K,)    int8 view of bool
        timeout_ptr,   # (K,)    int8 view of bool (defensive only)
        dec_ptr,       # (K, B)  int64 literal indices, -1 = padding
        occ_off_ptr,   # (2N+1,) int32 CSR offsets  (occ[l] = clauses where ~l appears)
        occ_cls_ptr,   # (total,) int32 CSR clause indices
        trail_ptr,     # (K, TRAIL_CAP) int32 workspace
        nprops_ptr,    # (K,) int32 out: literals set TRUE this call (seeds + forced)
        nops_ptr,      # (K,) int32 out: occ entries processed (≈ engine bcp_ops)
        M, N2, B, TRAIL_CAP,
    ):
        k = tl.program_id(0).to(tl.int64)
        if tl.load(active_ptr + k) != 0:
            sat_row = sat_ptr + k * M
            lsum_row = lsum_ptr + k * M
            asg_row = asg_ptr + k * N2
            trail_row = trail_ptr + k * TRAIL_CAP

            conflict = B * 0            # runtime-typed int32 scalars (loop-carried)
            tail = B * 0
            head = B * 0
            nops = B * 0

            # --- seed: assign each decision literal + queue it (occ decrements happen below) ---
            for j in range(0, B):
                lit64 = tl.load(dec_ptr + k * B + j)
                if (lit64 >= 0) & (conflict == 0):
                    lit = lit64.to(tl.int32)
                    if tl.load(asg_row + (lit ^ 1)) != 0:   # contradicts existing / earlier seed
                        conflict = 1
                    elif tl.load(asg_row + lit) == 0:       # skip duplicates
                        tl.store(asg_row + lit, 1)
                        tl.store(trail_row + tail, lit)
                        tail += 1

            # --- propagate to fixpoint: pop one literal, walk occ[lit] (distinct clauses) ---
            while (head < tail) & (conflict == 0):
                lit = tl.load(trail_row + head)
                head += 1
                neg = lit ^ 1
                start = tl.load(occ_off_ptr + lit)
                end = tl.load(occ_off_ptr + lit + 1)
                nops += end - start
                o = start
                while (o < end) & (conflict == 0):
                    c = tl.load(occ_cls_ptr + o).to(tl.int64)
                    sat = tl.load(sat_row + c) - 1          # sum-trick decrement
                    tl.store(sat_row + c, sat)
                    ls = tl.load(lsum_row + c) - neg
                    tl.store(lsum_row + c, ls)
                    if sat == 0:                            # all literals falsified
                        conflict = 1
                    elif sat == 1:                          # unit: remaining literal IS lit_sum
                        forced = ls
                        if tl.load(asg_row + forced) == 0:
                            if tl.load(asg_row + (forced ^ 1)) != 0:
                                conflict = 1                # forced both polarities
                            elif tail < TRAIL_CAP:
                                tl.store(asg_row + forced, 1)
                                tl.store(trail_row + tail, forced)
                                tail += 1
                            else:                           # structurally unreachable (cap = N+B)
                                tl.store(timeout_ptr + k, 1)
                                conflict = 1
                    o += 1

            if conflict != 0:
                tl.store(conflict_ptr + k, 1)
                tl.store(active_ptr + k, 0)
            tl.store(nprops_ptr + k, tail)
            tl.store(nops_ptr + k, nops)


class TritonBCP:
    """Per-engine wrapper: buffers, availability, and the launch. Attach via
    `engine.attach_kernel(TritonBCP())`; `engine.apply_and_propagate(decisions)` then dispatches here.
    Reuses the trail/output buffers across calls (reallocating only when K or B outgrow them) and
    accumulates work counters ON GPU (no per-call sync; `ops_total()` syncs lazily when read)."""

    _avail_cache: "bool | None" = None

    def __init__(self):
        self._occ_key = None            # id(occ_list) the int32 CSR copies were built from
        self._occ_off32 = None
        self._occ_cls32 = None
        self._trail = None              # (K, cap) int32
        self._nprops = None             # (K,) int32
        self._nops = None               # (K,) int32
        self._ops_acc = None            # 0-d int64 GPU accumulator (lazy-synced)
        self.launches = 0

    @classmethod
    def available(cls) -> bool:
        """True iff triton is importable AND a trivial kernel compiles+runs on this GPU (cached)."""
        if cls._avail_cache is not None:
            return cls._avail_cache
        ok = False
        if _HAVE_TRITON and torch.cuda.is_available():
            try:
                dev = torch.device("cuda")
                z = torch.zeros(4, dtype=torch.int32, device=dev)
                _probe_kernel[(4,)](z)
                torch.cuda.synchronize()
                ok = bool((z == 1).all().item())
            except Exception:  # noqa: BLE001 — any compile/launch failure = unavailable
                ok = False
        cls._avail_cache = ok
        return ok

    def _ensure(self, engine, K: int, B: int):
        dev = engine._device
        occ = engine._formula.occ_list
        if self._occ_key != id(occ):
            self._occ_off32 = occ.occ_offsets.to(torch.int32).contiguous()
            self._occ_cls32 = occ.occ_clauses.to(torch.int32).contiguous()
            self._occ_key = id(occ)
        cap = engine._num_vars + max(B, 64) + 8
        if (self._trail is None or self._trail.size(0) != K or self._trail.size(1) < cap):
            self._trail = torch.empty((K, cap), dtype=torch.int32, device=dev)
            self._nprops = torch.empty(K, dtype=torch.int32, device=dev)
            self._nops = torch.empty(K, dtype=torch.int32, device=dev)
        if self._ops_acc is None:
            self._ops_acc = torch.zeros((), dtype=torch.int64, device=dev)

    @torch.inference_mode()
    def run(self, engine, decisions: torch.Tensor):
        """Fused apply_assumptions_bulk + propagate-to-fixpoint (one launch)."""
        K = engine._sat_count.size(0)
        decisions = decisions.to(engine._device)
        if decisions.dim() == 1:
            decisions = decisions.unsqueeze(1)
        B = decisions.size(1)
        if decisions.numel() == 0:
            return
        self._ensure(engine, K, B)
        dec = decisions.to(torch.int64).contiguous()
        _bcp_fixpoint_kernel[(K,)](
            engine._sat_count, engine._lit_sum, engine._assignment,
            engine._active.view(torch.int8), engine._conflict.view(torch.int8),
            engine._timed_out.view(torch.int8),
            dec, self._occ_off32, self._occ_cls32,
            self._trail, self._nprops, self._nops,
            engine._num_clauses, 2 * engine._num_vars, B, self._trail.size(1),
            num_warps=1,
        )
        self.launches += 1
        engine.total_bcp_rounds += 1                    # one fused "round" per launch
        self._ops_acc += self._nops.sum()               # GPU-side, no sync

    def ops_total(self) -> int:
        """Occ entries processed by the kernel so far (syncs; call off the hot path)."""
        return int(self._ops_acc.item()) if self._ops_acc is not None else 0


if _HAVE_TRITON:

    @triton.jit
    def _probe_kernel(x_ptr):
        pid = tl.program_id(0)
        tl.store(x_ptr + pid, 1)
