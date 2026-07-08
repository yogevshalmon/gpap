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

Three JIT kernels live here:
  * `_bcp_fixpoint_kernel`  — batched assumption checking: apply a (K,L) decision tensor and BCP to
    fixpoint (the engine's `apply_and_propagate` fast path; the other apps consume this).
  * `_bcp_trajectory_kernel` — the whole randomized decide+propagate loop in one launch, on the
    engine's own int8/16 sat_count + int32 lit_sum tensors (the UNPACKED fallback).
  * `_bcp_trajectory_packed_kernel` — the same, on a PACKED int32 clause state (sat_count + XOR of
    remaining literals fused into one word), with a compile-time `SHIFT` selecting 16|16 or 8|24
    (see its docstring). This is the default fused path; ~half the scattered-state traffic.

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
        ndecs_ptr,     # (K,) int32 out: DECISION literals this call (seeds only; implications = nprops - ndecs)
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
            ndecs = tail                # seeds assigned so far ARE the decisions (before propagation)

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
            tl.store(ndecs_ptr + k, ndecs)

    @triton.jit
    def _bcp_trajectory_kernel(
        sat_ptr, lsum_ptr, asg_ptr,          # engine state (row-private per program), as in v1
        active_ptr, conflict_ptr,
        order_ptr,                            # (P,) int32 var indices: the decision order (support
                                              #   first, then completion vars; a per-chunk permutation)
        occ_off_ptr, occ_cls_ptr,             # int32 CSR occ lists
        trail_ptr, nrounds_ptr, nops_ptr, nprops_ptr, ndecs_ptr,   # (K,·) workspace / outputs
        M, N2, P, NSUP, B, TRAIL_CAP, MAX_ROUNDS, seed,
        SOFT: tl.constexpr,                   # 1 = "fsoft": in-kernel soft risk-polarity (see below)
    ):
        """v2 — the WHOLE trajectory in one launch: repeatedly pick the next B still-free variables
        in `order` (identical semantics to _pick_decisions' first-B-free-in-priority-order), assign
        each a hash-random polarity, and propagate to fixpoint (the v1 inner loop). One launch
        replaces the sampler's whole Python decide loop (measured 41-79% of chunk wall under v1).

        NSUP = number of leading `order` entries that are PHASE-1 (support) vars. A decision group
        must never straddle the phase boundary: phase 1 must reach fixpoint over the support before
        any completion var is decided (else the group decides variables that propagation would have
        determined, and conflicts explode — measured 100%→0% on b07 when straddled)."""
        k = tl.program_id(0).to(tl.int64)
        if tl.load(active_ptr + k) != 0:
            sat_row = sat_ptr + k * M
            lsum_row = lsum_ptr + k * M
            asg_row = asg_ptr + k * N2
            trail_row = trail_ptr + k * TRAIL_CAP

            conflict = B * 0
            head = B * 0
            tail = B * 0
            nops = B * 0
            rounds = B * 0
            decs = B * 0                        # DECISION literals (picks); implications = tail - decs
            ptr = B * 0
            while (ptr < P) & (conflict == 0) & (rounds < MAX_ROUNDS):
                # --- decision group: next B free vars in order, hash-random polarity.
                #     The group ends at the support/completion boundary (see docstring). ---
                phase_end = NSUP if ptr < NSUP else P
                picked = B * 0
                while (picked < B) & (ptr < phase_end):
                    v = tl.load(order_ptr + ptr)
                    ptr += 1
                    if (tl.load(asg_row + 2 * v) == 0) & (tl.load(asg_row + 2 * v + 1) == 0):
                        # stateless per-(chunk,k,v) polarity bit (splitmix-style 64-bit mix;
                        # constants are the standard splitmix64 ones as SIGNED int64 literals)
                        h = seed + k * -7046029254386353131 + v.to(tl.int64) * -4658895280553007687
                        h = (h ^ (h >> 30)) * -4658895280553007687
                        h = (h ^ (h >> 27)) * -7723592293110705685
                        pol = ((h >> 13) & 1).to(tl.int32)
                        if SOFT:
                            # "fsoft": soft conflict-avoidance polarity AT FUSED SPEED — mirror
                            # _polarity_risk: risk of setting lit TRUE = Σ w(sat_count) over occ[lit]
                            # (clauses lit falsifies): w(1)=1e6 (would empty), w(2)=4 (forces a
                            # unit), w(3)=1. Pick the lower-risk polarity; hash bit breaks ties.
                            r0 = nops * 0                       # risk of lit = 2v (v TRUE)
                            r1 = nops * 0                       # risk of lit = 2v+1 (v FALSE)
                            s0 = tl.load(occ_off_ptr + 2 * v)
                            e0 = tl.load(occ_off_ptr + 2 * v + 1)
                            o0 = s0
                            while o0 < e0:
                                c0 = tl.load(occ_cls_ptr + o0).to(tl.int64)
                                sc0 = tl.load(sat_row + c0)
                                r0 += tl.where(sc0 == 1, 1000000,
                                               tl.where(sc0 == 2, 4, tl.where(sc0 == 3, 1, 0)))
                                o0 += 1
                            e1 = tl.load(occ_off_ptr + 2 * v + 2)
                            o1 = e0
                            while o1 < e1:
                                c1 = tl.load(occ_cls_ptr + o1).to(tl.int64)
                                sc1 = tl.load(sat_row + c1)
                                r1 += tl.where(sc1 == 1, 1000000,
                                               tl.where(sc1 == 2, 4, tl.where(sc1 == 3, 1, 0)))
                                o1 += 1
                            # ε-greedy (25% hash-forced random): pure risk-argmin is
                            # row-DETERMINISTIC when risks never tie — all K trajectories
                            # replicate ONE path (measured: c1355 collapsed to 70 distinct
                            # solutions in 6,144 samples while CMSGen's were 100% distinct).
                            # ε keeps most conflict-avoidance and guarantees divergence.
                            risky = tl.where(r0 < r1, 0, tl.where(r1 < r0, 1, pol))
                            pol = tl.where(((h >> 17) & 7) == 0, pol, risky)
                        lit = 2 * v + pol
                        tl.store(asg_row + lit, 1)
                        tl.store(trail_row + tail, lit)
                        tail += 1
                        picked += 1
                if picked > 0:
                    rounds += 1
                    decs += picked
                # --- propagate to fixpoint (identical to the v1 loop) ---
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
                        sat = tl.load(sat_row + c) - 1
                        tl.store(sat_row + c, sat)
                        ls = tl.load(lsum_row + c) - neg
                        tl.store(lsum_row + c, ls)
                        if sat == 0:
                            conflict = 1
                        elif sat == 1:
                            forced = ls
                            if tl.load(asg_row + forced) == 0:
                                if tl.load(asg_row + (forced ^ 1)) != 0:
                                    conflict = 1
                                elif tail < TRAIL_CAP:
                                    tl.store(asg_row + forced, 1)
                                    tl.store(trail_row + tail, forced)
                                    tail += 1
                                else:
                                    conflict = 1
                        o += 1

            if conflict != 0:
                tl.store(conflict_ptr + k, 1)
                tl.store(active_ptr + k, 0)
            tl.store(nrounds_ptr + k, rounds)
            tl.store(nops_ptr + k, nops)
            tl.store(nprops_ptr + k, tail)    # literals assigned (decisions + implications)
            tl.store(ndecs_ptr + k, decs)     # decisions only (implications = nprops - ndecs)

    @triton.jit
    def _bcp_trajectory_packed_kernel(
        st_ptr,                               # (K, M) int32 PACKED clause state, kernel-private:
                                              #   low SHIFT bits    = sat_count (non-falsified count)
                                              #   high 32-SHIFT bits = XOR of non-falsified lit idx
        asg_ptr, active_ptr, conflict_ptr,
        order_ptr, occ_off_ptr, occ_cls_ptr,
        trail_ptr, nrounds_ptr, nops_ptr, nprops_ptr, ndecs_ptr,
        M, N2, P, NSUP, B, TRAIL_CAP, MAX_ROUNDS, seed,
        SOFT: tl.constexpr,
        SHIFT: tl.constexpr,                  # packed split (16 -> 16|16, 8 -> 8|24)
    ):
        """PACKED variant of _bcp_trajectory_kernel — same trajectory semantics, half the state
        traffic. The inner BCP loop is memory-bound on SCATTERED clause-state sectors (uncoalesced
        by construction: each occ entry is a random clause), so fusing both counters into ONE int32
        turns 2 loads + 2 stores (int8/16 sat + int32 sum, 4 random 32B sectors) into 1 load +
        1 store (2 sectors). XOR replaces SUM as the remaining-literal trick (xor of a one-element
        set IS the element) because it never overflows.

        `SHIFT` selects the field split (a compile-time constexpr, so each value specialises to its
        own binary — this is one source for two kernels, not a runtime branch):
          SHIFT=16 -> 16|16, covers 2N <= 65535       (sat half holds any realistic clause size)
          SHIFT=8  -> 8|24,  covers 2N <= 0xFFFFFF and max clause len <= 255 (the "wide" instances)
        The wrapper falls back to the UNPACKED kernel when neither fits.

        Packed update per falsified literal:  st = (st - 1) ^ (neg << SHIFT)
          * the -1 hits only the sat field: sat >= 1 at every decrement (sat==0 conflicts and halts
            the row before any further update), so the subtraction never borrows into the XOR field;
          * (neg << SHIFT) hits only the XOR field. For 2N near the 24-bit ceiling the shift may set
            bit 31 (int32 sign) — harmless: the arithmetic is bitwise and extraction re-masks.
        sat==1 -> forced literal = (st >> SHIFT) & XOR_MASK."""
        SAT_MASK: tl.constexpr = (1 << SHIFT) - 1
        XOR_MASK: tl.constexpr = (1 << (32 - SHIFT)) - 1
        k = tl.program_id(0).to(tl.int64)
        if tl.load(active_ptr + k) != 0:
            st_row = st_ptr + k * M
            asg_row = asg_ptr + k * N2
            trail_row = trail_ptr + k * TRAIL_CAP

            conflict = B * 0
            head = B * 0
            tail = B * 0
            nops = B * 0
            rounds = B * 0
            decs = B * 0
            ptr = B * 0
            while (ptr < P) & (conflict == 0) & (rounds < MAX_ROUNDS):
                phase_end = NSUP if ptr < NSUP else P
                picked = B * 0
                while (picked < B) & (ptr < phase_end):
                    v = tl.load(order_ptr + ptr)
                    ptr += 1
                    if (tl.load(asg_row + 2 * v) == 0) & (tl.load(asg_row + 2 * v + 1) == 0):
                        h = seed + k * -7046029254386353131 + v.to(tl.int64) * -4658895280553007687
                        h = (h ^ (h >> 30)) * -4658895280553007687
                        h = (h ^ (h >> 27)) * -7723592293110705685
                        pol = ((h >> 13) & 1).to(tl.int32)
                        if SOFT:
                            r0 = nops * 0
                            r1 = nops * 0
                            s0 = tl.load(occ_off_ptr + 2 * v)
                            e0 = tl.load(occ_off_ptr + 2 * v + 1)
                            o0 = s0
                            while o0 < e0:
                                c0 = tl.load(occ_cls_ptr + o0).to(tl.int64)
                                sc0 = tl.load(st_row + c0) & SAT_MASK
                                r0 += tl.where(sc0 == 1, 1000000,
                                               tl.where(sc0 == 2, 4, tl.where(sc0 == 3, 1, 0)))
                                o0 += 1
                            e1 = tl.load(occ_off_ptr + 2 * v + 2)
                            o1 = e0
                            while o1 < e1:
                                c1 = tl.load(occ_cls_ptr + o1).to(tl.int64)
                                sc1 = tl.load(st_row + c1) & SAT_MASK
                                r1 += tl.where(sc1 == 1, 1000000,
                                               tl.where(sc1 == 2, 4, tl.where(sc1 == 3, 1, 0)))
                                o1 += 1
                            # ε-greedy (25% hash-forced random): pure risk-argmin is
                            # row-DETERMINISTIC when risks never tie — all K trajectories
                            # replicate ONE path (measured: c1355 collapsed to 70 distinct
                            # solutions in 6,144 samples while CMSGen's were 100% distinct).
                            # ε keeps most conflict-avoidance and guarantees divergence.
                            risky = tl.where(r0 < r1, 0, tl.where(r1 < r0, 1, pol))
                            pol = tl.where(((h >> 17) & 7) == 0, pol, risky)
                        lit = 2 * v + pol
                        tl.store(asg_row + lit, 1)
                        tl.store(trail_row + tail, lit)
                        tail += 1
                        picked += 1
                if picked > 0:
                    rounds += 1
                    decs += picked
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
                        st = (tl.load(st_row + c) - 1) ^ (neg << SHIFT)
                        tl.store(st_row + c, st)
                        sat = st & SAT_MASK
                        if sat == 0:
                            conflict = 1
                        elif sat == 1:
                            forced = (st >> SHIFT) & XOR_MASK
                            if tl.load(asg_row + forced) == 0:
                                if tl.load(asg_row + (forced ^ 1)) != 0:
                                    conflict = 1
                                elif tail < TRAIL_CAP:
                                    tl.store(asg_row + forced, 1)
                                    tl.store(trail_row + tail, forced)
                                    tail += 1
                                else:
                                    conflict = 1
                        o += 1

            if conflict != 0:
                tl.store(conflict_ptr + k, 1)
                tl.store(active_ptr + k, 0)
            tl.store(nrounds_ptr + k, rounds)
            tl.store(nops_ptr + k, nops)
            tl.store(nprops_ptr + k, tail)
            tl.store(ndecs_ptr + k, decs)



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
        self._ndecs = None              # (K,) int32 (decision literals this launch)
        self._nrounds = None            # (K,) int32 (v2 fused-trajectory decision rounds)
        self._ops_acc = None            # 0-d int64 GPU accumulator (lazy-synced)
        self._props_acc = None          # 0-d int64 GPU accumulator: literals assigned (lazy-synced)
        self._decs_acc = None           # 0-d int64 GPU accumulator: decisions (implications = props-decs)
        # PACKED trajectory state (see _bcp_trajectory_packed_kernel): kernel-private (K, M) int32
        # replacing the engine's sat_count+lit_sum for fused chunks, plus the cached post-units-flush
        # row that reset_fused() broadcast-restores per chunk (the flush is row-identical every chunk).
        self._packed = None             # (K, M) int32 packed clause state
        self._packed_flushed = None     # (M,) int32 packed row AFTER the formula-units flush
        self._asg_flushed = None        # (2N,) int8 assignment row AFTER the flush
        self._packed_mode = None        # None = undecided; True/False once probed for this formula
        # Packed field split, decided per formula in _build_flush_cache: 16 -> 16|16 (2N <= 65535),
        # 8 -> 8|24 (the "wide" instances, 65535 < 2N <= 0xFFFFFF, max clause len <= 255).
        self._shift = 16
        self._force_wide = False        # test hook: pack 8|24 even where 16|16 would fit (A/B tests)
        self.launches = 0

    @property
    def _wide_mode(self) -> bool:
        """True when this formula packs 8|24 (the wide split). Diagnostics / tests."""
        return bool(self._packed_mode) and self._shift == 8

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

    def _ensure_occ(self, engine):
        occ = engine._formula.occ_list
        if self._occ_key != id(occ):
            self._occ_off32 = occ.occ_offsets.to(torch.int32).contiguous()
            self._occ_cls32 = occ.occ_clauses.to(torch.int32).contiguous()
            self._occ_key = id(occ)

    def _ensure(self, engine, K: int, B: int):
        dev = engine._device
        self._ensure_occ(engine)
        cap = engine._num_vars + max(B, 64) + 8
        if (self._trail is None or self._trail.size(0) != K or self._trail.size(1) < cap):
            self._trail = torch.empty((K, cap), dtype=torch.int32, device=dev)
            self._nprops = torch.empty(K, dtype=torch.int32, device=dev)
            self._nops = torch.empty(K, dtype=torch.int32, device=dev)
            self._ndecs = torch.empty(K, dtype=torch.int32, device=dev)
        if self._ops_acc is None:
            self._ops_acc = torch.zeros((), dtype=torch.int64, device=dev)
            self._props_acc = torch.zeros((), dtype=torch.int64, device=dev)
            self._decs_acc = torch.zeros((), dtype=torch.int64, device=dev)

    def _build_flush_cache(self, engine) -> None:
        """One-time (per formula): pick the packed field split and cache the post-units-flush state.

        Split selection (self._shift):
          2N <= 65535                          -> 16|16 (SHIFT=16; the sat half holds any realistic
                                                  clause size, so no max-length constraint here).
          65535 < 2N <= 0xFFFFFF, max len<=255 -> 8|24 (SHIFT=8; the "wide" instances — b12-c25,
                                                  c50 unrolls — that the 16-bit XOR half can't index).
          otherwise                            -> `_packed_mode = False`: fused chunks fall back to
                                                  the UNPACKED kernel on the engine's own tensors.
        `_force_wide` (test hook) forces 8|24 wherever it fits, so A/B tests can compare it against
        16|16 on small formulas.

        The cache: after reset_state()+propagate() the engine state is IDENTICAL for every row of
        every chunk (the formula-units flush is deterministic), so one (M,)/(2N,) row pair suffices
        and reset_fused() broadcast-restores it — replacing both reset_state's (K,M) sat/lsum
        writes and the per-chunk Python-round-loop flush. The packed row's XOR half is computed
        with one numpy pass over the occ CSR (occ[l] ∋ c ⟺ literal l^1 ∈ clause c; falsified
        literals map to 0, the XOR identity)."""
        import numpy as np
        n2 = 2 * engine._num_vars
        M = engine._num_clauses
        max_len = int(engine._formula.clause_sizes.max().item()) if M else 0
        if M == 0 or n2 > 0xFFFFFF or (n2 > 65535 and max_len > 255):
            self._packed_mode = False
            return
        self._shift = 8 if (n2 > 65535 or self._force_wide) else 16
        sat_mask = (1 << self._shift) - 1
        xor_mask = (1 << (32 - self._shift)) - 1
        self._ensure_occ(engine)
        engine.reset_state(1)
        engine.propagate()
        if bool(engine._conflict[0].item()):    # formula units conflict -> whole formula UNSAT;
            self._packed_mode = False           # keep the ordinary path (0 models either way)
            return
        self._asg_flushed = engine._assignment[0].clone()
        sat0 = engine._sat_count[0].to(torch.int32).cpu().numpy()
        off = self._occ_off32.cpu().numpy().astype(np.int64)
        cls = self._occ_cls32.cpu().numpy().astype(np.int64)
        lits = np.repeat(np.arange(n2, dtype=np.int64) ^ 1, off[1:] - off[:-1])
        asg = self._asg_flushed.cpu().numpy()
        vals = np.where(asg[lits ^ 1] == 1, 0, lits)        # falsified -> 0 (XOR identity)
        order = np.argsort(cls, kind="stable")
        starts = np.searchsorted(cls[order], np.arange(M, dtype=np.int64))
        xor = np.bitwise_xor.reduceat(vals[order], starts)  # per-clause XOR of non-falsified lits
        # Pack in uint32 (the shifted XOR may legitimately reach bit 31) then reinterpret as the
        # kernel's int32 — torch integer casts would truncate but a view is exact.
        packed = ((sat0.astype(np.uint32) & sat_mask)
                  | ((xor.astype(np.uint32) & xor_mask) << self._shift)).view(np.int32)
        self._packed_flushed = torch.from_numpy(packed).to(engine._device)
        self._packed_mode = True

    @torch.inference_mode()
    def reset_fused(self, engine, K: int) -> bool:
        """Fast per-chunk reset for PACKED fused chunks: broadcast-restore the cached post-flush
        state instead of reset_state()+propagate(). Engine sat_count/lit_sum are left UNTOUCHED
        (the packed tensor is the kernel's clause state); assignment/active/conflict stay on the
        engine so the sampler's readout code is unchanged. Returns False when packing is
        unavailable (caller must run the ordinary reset_state()+propagate() path)."""
        if self._packed_mode is None:
            self._build_flush_cache(engine)
        if not self._packed_mode:
            return False
        dev = engine._device
        n2 = 2 * engine._num_vars
        if getattr(engine, "_assignment", None) is None or engine._assignment.size(0) != K:
            engine._assignment = torch.empty((K, n2), dtype=torch.int8, device=dev)
            engine._active = torch.empty(K, dtype=torch.bool, device=dev)
            engine._conflict = torch.empty(K, dtype=torch.bool, device=dev)
            engine._timed_out = torch.empty(K, dtype=torch.bool, device=dev)
        engine._assignment.copy_(self._asg_flushed.unsqueeze(0).expand(K, -1))
        engine._active.fill_(True)
        engine._conflict.fill_(False)
        engine._timed_out.fill_(False)
        if self._packed is None or self._packed.size(0) != K:
            self._packed = torch.empty((K, engine._num_clauses), dtype=torch.int32, device=dev)
        self._packed.copy_(self._packed_flushed.unsqueeze(0).expand(K, -1))
        return True

    def packed_sat_rows(self, row_idx: torch.Tensor) -> torch.Tensor:
        """sat_count values for the given rows of the PACKED state (diagnostics/soundness only)."""
        return self._packed[row_idx] & ((1 << self._shift) - 1)

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
        # Zero the per-row outputs: this launch may leave INACTIVE rows (already-conflicted in a
        # prior decide round) unwritten, and summing stale values would double-count. Zeroed rows
        # contribute 0, so the accumulators are exact across the multi-round decide loop.
        self._nprops.zero_(); self._nops.zero_(); self._ndecs.zero_()
        dec = decisions.to(torch.int64).contiguous()
        _bcp_fixpoint_kernel[(K,)](
            engine._sat_count, engine._lit_sum, engine._assignment,
            engine._active.view(torch.int8), engine._conflict.view(torch.int8),
            engine._timed_out.view(torch.int8),
            dec, self._occ_off32, self._occ_cls32,
            self._trail, self._nprops, self._nops, self._ndecs,
            engine._num_clauses, 2 * engine._num_vars, B, self._trail.size(1),
            num_warps=1,
        )
        self.launches += 1
        engine.total_bcp_rounds += 1                    # one fused "round" per launch
        self._ops_acc += self._nops.sum()               # GPU-side, no sync
        self._props_acc += self._nprops.sum()
        self._decs_acc += self._ndecs.sum()

    @torch.inference_mode()
    def run_trajectory(self, engine, order: torch.Tensor, n_support: int, B: int, seed: int,
                       max_rounds: "int | None", soft: bool = False, packed: bool = False) -> int:
        """v2 fused chunk: run WHOLE trajectories (decide + propagate loops) in one launch.
        `order` = (P,) int32 decision order (support vars first, then completion vars). Picking the
        next B still-free vars in `order` is exactly _pick_decisions' first-B-free-in-priority-order
        semantics; polarity is a stateless per-(chunk,k,v) hash bit (distributionally uniform — the
        torch RNG stream is NOT consumed, so fused runs are reproducible via `seed` alone).
        `packed=True` (caller must have prepared state via reset_fused) runs the packed-state
        kernel at this formula's field split (self._shift: 16|16 or 8|24) — identical trajectory
        semantics, half the clause-state memory traffic.
        (In-kernel retry-on-conflict was built and MEASURED OUT here: 3 retries tripled per-slot
        success on the low-success instances exactly as predicted, but wall tripled with it —
        conflicts on those instances die LATE, so a retry costs a full trajectory and models/wall
        stayed 0.8-1.0x. Wrapping the retry loop around this kernel also taxed the no-retry path
        ~10% median, so the experiment was removed rather than kept as a config.)
        Returns the max decision rounds any trajectory used (diagnostic; one small sync)."""
        K = engine._assignment.size(0)          # NOT sat_count: stale-sized on packed chunks
        self._ensure(engine, K, 64)
        if self._nrounds is None or self._nrounds.size(0) != K:
            self._nrounds = torch.empty(K, dtype=torch.int32, device=engine._device)
        order = order.to(torch.int32).contiguous()
        bufs = (order, self._occ_off32, self._occ_cls32,
                self._trail, self._nrounds, self._nops, self._nprops, self._ndecs)
        dims = (engine._num_clauses, 2 * engine._num_vars, order.numel(), n_support, B,
                self._trail.size(1), (max_rounds if max_rounds is not None else 1 << 30), seed)
        if packed:
            _bcp_trajectory_packed_kernel[(K,)](
                self._packed, engine._assignment,
                engine._active.view(torch.int8), engine._conflict.view(torch.int8),
                *bufs, *dims,
                SOFT=(1 if soft else 0), SHIFT=self._shift, num_warps=1,
            )
        else:
            _bcp_trajectory_kernel[(K,)](
                engine._sat_count, engine._lit_sum, engine._assignment,
                engine._active.view(torch.int8), engine._conflict.view(torch.int8),
                *bufs, *dims, SOFT=(1 if soft else 0), num_warps=1,
            )
        self.launches += 1
        engine.total_bcp_rounds += 1
        self._ops_acc += self._nops.sum()
        self._props_acc += self._nprops.sum()
        self._decs_acc += self._ndecs.sum()      # every row is active at launch (post-reset) => exact
        return int(self._nrounds.max().item())

    def ops_total(self) -> int:
        """Occ entries processed by the kernel so far (syncs; call off the hot path)."""
        return int(self._ops_acc.item()) if self._ops_acc is not None else 0

    def props_total(self) -> int:
        """Literals assigned by the kernel so far (syncs; call off the hot path)."""
        return int(self._props_acc.item()) if self._props_acc is not None else 0

    def decs_total(self) -> int:
        """Decision literals assigned by the kernel so far (syncs; call off the hot path).
        Implications = props_total() - decs_total()."""
        return int(self._decs_acc.item()) if self._decs_acc is not None else 0


if _HAVE_TRITON:

    @triton.jit
    def _probe_kernel(x_ptr):
        pid = tl.program_id(0)
        tl.store(x_ptr + pid, 1)
