"""The Triton BCP kernels — the ONLY execution path of the GPAP engine (packed XOR-trick state).

Design — one program (~warp) per batch row, sequential trail, NO atomics / NO barriers:
  * Each Triton program owns row k exclusively: its row of the packed clause state and of the
    assignment are program-private, so there are no cross-program races by construction.
  * The program keeps a "trail" queue of newly-TRUE literals. It pops one literal at a time and
    walks its occurrence list (clauses where ~lit appears), updating the packed counters — one
    literal's occ list holds DISTINCT clauses, so even those updates need no atomics.
  * GPU-wide parallelism comes from the K programs (thousands to hundreds of thousands), which
    saturates the SMs; per-row work is uniform on the target formulas, so warps stay balanced.

PACKED clause state — ONE int32 word per (row, clause), `SHIFT` selects the field split
(a compile-time constexpr, so each value specialises to its own binary):
    low SHIFT bits     = sat_count (non-falsified-literal count)
    high 32-SHIFT bits = XOR of the non-falsified literal indices
  SHIFT=16 -> 16|16, covers 2N <= 65535       (sat half holds any realistic clause size)
  SHIFT=8  -> 8|24,  covers 2N <= 0xFFFFFF and max clause len <= 255 (the "wide" instances)
The engine REJECTS formulas outside both envelopes (see BCPEngine).

XOR is the remaining-literal trick (xor of a one-element set IS the element) and never
overflows. The update per falsified literal is  st = (st - 1) ^ (neg << SHIFT):
  * the -1 hits only the sat field: sat >= 1 at every decrement (sat==0 conflicts and halts
    the row before any further update), so the subtraction never borrows into the XOR field;
  * (neg << SHIFT) hits only the XOR field. For 2N near the 24-bit ceiling the shift may set
    bit 31 (int32 sign) — harmless: the arithmetic is bitwise and extraction re-masks.
sat==0 -> conflict; sat==1 -> forced literal = (st >> SHIFT) & XOR_MASK. The inner BCP loop is
memory-bound on SCATTERED clause-state sectors (uncoalesced by construction: each occ entry is
a random clause), so the single packed word is half the traffic of separate sat/xor tensors.

Two JIT kernels share the `_propagate` helper (triton inlines it — one source, two binaries per
SHIFT):
  * `_bcp_fixpoint_kernel`   — batched assumption checking: seed a (K,B) decision tensor and
    BCP to fixpoint (the engine's `apply_and_propagate`; miner/checker consume this).
  * `_bcp_trajectory_kernel` — the WHOLE randomized decide+propagate sampling loop in one
    launch, with an optional in-kernel soft risk-polarity (`SOFT`).

Semantics contract:
  * conflict/active flags are exact (BCP fixpoints are confluent).
  * NON-conflicted rows: assignment / packed counters are the exact BCP fixpoint.
  * Conflicted rows: partial counter state is propagation-order-dependent and intentionally
    unspecified (the sampler only reads rows that are active & fully-assigned).
  * Fixpoint decisions: duplicate literals are skipped; an l/~l pair or a decision contradicting
    the existing assignment conflicts the row.

Requires CUDA + triton (ships with torch on Linux/Colab — no nvcc). `kernel_available()`
probes once (compile+run a trivial kernel); the engine refuses to construct without it.
"""
import torch

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:  # noqa: BLE001 — any import/JIT env failure just means "unavailable"
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _propagate(st_row, asg_row, trail_row, occ_off_ptr, occ_cls_ptr,
                   head, tail, conflict, nops, TRAIL_CAP, SHIFT: tl.constexpr):
        """Pop literals off the trail and BCP to fixpoint (or conflict) on the packed row.
        Inlined by both kernels; returns the updated (head, tail, conflict, nops)."""
        SAT_MASK: tl.constexpr = (1 << SHIFT) - 1
        XOR_MASK: tl.constexpr = (1 << (32 - SHIFT)) - 1
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
                if sat == 0:                            # all literals falsified
                    conflict = 1
                elif sat == 1:                          # unit: remaining literal IS the XOR
                    forced = (st >> SHIFT) & XOR_MASK
                    if tl.load(asg_row + forced) == 0:
                        if tl.load(asg_row + (forced ^ 1)) != 0:
                            conflict = 1                # forced both polarities
                        elif tail < TRAIL_CAP:
                            tl.store(asg_row + forced, 1)
                            tl.store(trail_row + tail, forced)
                            tail += 1
                        else:                           # structurally unreachable (cap >= N+B)
                            conflict = 1
                o += 1
        return head, tail, conflict, nops

    @triton.jit
    def _bcp_fixpoint_kernel(
        st_ptr,        # (K, M)  int32 packed clause state (row-private per program)
        asg_ptr,       # (K, 2N) int8 assignment (1 = TRUE)
        active_ptr,    # (K,)    int8 view of bool
        conflict_ptr,  # (K,)    int8 view of bool
        dec_ptr,       # (K, B)  int64 literal indices, -1 = padding
        occ_off_ptr,   # (2N+1,) int32 CSR offsets  (occ[l] = clauses where ~l appears)
        occ_cls_ptr,   # (total,) int32 CSR clause indices
        trail_ptr,     # (K, TRAIL_CAP) int32 workspace
        nprops_ptr,    # (K,) int32 out: literals set TRUE this call (seeds + forced)
        nops_ptr,      # (K,) int32 out: occ entries processed
        ndecs_ptr,     # (K,) int32 out: DECISION literals (seeds; implications = nprops - ndecs)
        M, N2, B, TRAIL_CAP,
        SHIFT: tl.constexpr,
    ):
        k = tl.program_id(0).to(tl.int64)
        if tl.load(active_ptr + k) != 0:
            st_row = st_ptr + k * M
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
            ndecs = tail                # seeds assigned so far ARE the decisions

            head, tail, conflict, nops = _propagate(
                st_row, asg_row, trail_row, occ_off_ptr, occ_cls_ptr,
                head, tail, conflict, nops, TRAIL_CAP, SHIFT)

            if conflict != 0:
                tl.store(conflict_ptr + k, 1)
                tl.store(active_ptr + k, 0)
            tl.store(nprops_ptr + k, tail)
            tl.store(nops_ptr + k, nops)
            tl.store(ndecs_ptr + k, ndecs)

    @triton.jit
    def _bcp_trajectory_kernel(
        st_ptr, asg_ptr, active_ptr, conflict_ptr,
        order_ptr,                            # (P,) int32 var indices: the decision order (support
                                              #   first, then completion vars; a per-chunk permutation)
        occ_off_ptr, occ_cls_ptr,
        trail_ptr, nrounds_ptr, nops_ptr, nprops_ptr, ndecs_ptr,   # (K,·) workspace / outputs
        M, N2, P, NSUP, B, TRAIL_CAP, MAX_ROUNDS, seed,
        SOFT: tl.constexpr,                   # 1 = "soft": in-kernel soft risk-polarity (see below)
        SHIFT: tl.constexpr,
    ):
        """The WHOLE trajectory in one launch: repeatedly pick the next B still-free variables in
        `order`, assign each a hash-random polarity, and propagate to fixpoint.

        NSUP = number of leading `order` entries that are PHASE-1 (support) vars. A decision group
        must never straddle the phase boundary: phase 1 must reach fixpoint over the support before
        any completion var is decided (else the group decides variables that propagation would have
        determined, and conflicts explode — measured 100%->0% on b07 when straddled)."""
        SAT_MASK: tl.constexpr = (1 << SHIFT) - 1
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
            decs = B * 0                        # DECISION literals; implications = tail - decs
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
                            # soft conflict-avoidance polarity AT FUSED SPEED: risk of setting
                            # lit TRUE = sum of w(sat_count) over occ[lit] (clauses lit falsifies):
                            # w(1)=1e6 (would empty), w(2)=4 (forces a unit), w(3)=1. Pick the
                            # lower-risk polarity; hash bit breaks ties.
                            r0 = nops * 0                       # risk of lit = 2v (v TRUE)
                            r1 = nops * 0                       # risk of lit = 2v+1 (v FALSE)
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
                # --- propagate to fixpoint ---
                head, tail, conflict, nops = _propagate(
                    st_row, asg_row, trail_row, occ_off_ptr, occ_cls_ptr,
                    head, tail, conflict, nops, TRAIL_CAP, SHIFT)

            if conflict != 0:
                tl.store(conflict_ptr + k, 1)
                tl.store(active_ptr + k, 0)
            tl.store(nrounds_ptr + k, rounds)
            tl.store(nops_ptr + k, nops)
            tl.store(nprops_ptr + k, tail)    # literals assigned (decisions + implications)
            tl.store(ndecs_ptr + k, decs)     # decisions only (implications = nprops - ndecs)

    @triton.jit
    def _probe_kernel(x_ptr):
        pid = tl.program_id(0)
        tl.store(x_ptr + pid, 1)


_avail_cache: "bool | None" = None


def kernel_available() -> bool:
    """True iff triton is importable AND a trivial kernel compiles+runs on this GPU (cached)."""
    global _avail_cache
    if _avail_cache is not None:
        return _avail_cache
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
    _avail_cache = ok
    return ok
