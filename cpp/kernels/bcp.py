"""The two Triton BCP kernels — the ONLY execution path of the GPAP engine (packed SUM-trick).

BUILD INPUT, NOT RUNTIME CODE. Nothing imports this module to run it: `cpp/tools/build_kernels.py`
loads it (with `cpp/kernels/helpers.py`, the support kernels) and AOT-compiles every specialization
into `libgpap.so`. The shipped library needs `libcuda.so.1` alone — no triton, no torch, no nvcc.
Editing this file therefore changes nothing until the kernels are rebuilt; `scripts/build_core.sh`
does both steps.

Design — one program per batch row, sequential trail, atomics only WITHIN a row's warp(s):
  * Each Triton program owns row k exclusively: its row of the packed clause state and of the
    assignment bitmap are program-private, so there are no cross-PROGRAM races by construction;
    the WAVE variant's in-row lane races are resolved by the atomics' return values (below).
  * The program keeps a "trail" queue of newly-TRUE literals. It pops literals (one at a time
    scalar, up to W at once in WAVE) and walks their occurrence lists (clauses where ~lit
    appears), updating the packed counters.
  * GPU-wide parallelism comes from the K programs (thousands to hundreds of thousands), which
    saturates the SMs; per-row work is uniform on the target formulas, so warps stay balanced.

BOTH row states are single words updated by ONE op whose old value is the detection logic:

  * PACKED clause state — ONE int32 word per (row, clause), `SHIFT` selects the field split
    (compile-time constexpr): low SHIFT bits = sat_count, high 32-SHIFT bits = SUM of the
    non-falsified literal indices mod 2^(32-SHIFT).
      SHIFT=16 -> 16|16, covers 2N <= 65535     (sat half holds any realistic clause size)
      SHIFT=8  -> 8|24,  covers 2N <= 0xFFFFFF and max clause len <= 255 (the "wide" instances)
    The engine REJECTS formulas outside both envelopes (see BCPEngine). Every removal is exact
    mod 2^F, so at sat_count==1 the field IS the last literal. The update per falsified literal
    is ONE subtraction, st -= 1 + (neg << SHIFT) (count never borrows — count >= 1; sum wraps
    off the top of the word). In WAVE that subtraction is a tl.atomic_add whose RETURN gives
    race-free detection: old count==1 -> this lane emptied the clause (conflict), old count==2
    -> this lane made it unit and (old_sum - neg) & SUM_MASK IS the forced literal.
  * Assignment BITMAP — bit l of the (K, ceil(2N/32)) int32 words = literal l TRUE. l and l^1
    always share a word (2v / 2v+1), so ONE load answers free/duplicate/contradiction, and in
    WAVE the push is a tl.atomic_or whose RETURN deduplicates (old bit l set) and detects
    contradictions (old bit l^1 set) under any lane interleaving. 8x smaller than the former
    int8-per-literal array — on the big unrolls the whole bitmap approaches L2 residency.

The inner BCP loop is memory-bound on SCATTERED clause-state sectors (uncoalesced by
construction), so the single packed word is half the traffic of separate tensors.

Two entry-point kernels share the `_propagate` helper (triton inlines it — one source, two
binaries per SHIFT):
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

NO torch import, at module scope or anywhere: CI builds the library with triton ALONE (triton's
NVIDIA backend reports itself inactive without torch + a GPU, which is why build_kernels.py goes
around the JIT binder). Keep it that way or the driverless build job stops working.
"""
import triton
import triton.language as tl

@triton.jit
def _propagate(st_row, ab_row, trail_row, occ_off_ptr, occ_cls_ptr,
               head, tail, conflict, nops, TRAIL_CAP, SHIFT: tl.constexpr):
    """Pop literals off the trail and BCP to fixpoint (or conflict) on the packed row.
    Inlined by both kernels; returns the updated (head, tail, conflict, nops).

    The assignment is a BITMAP (int32 words, bit l = literal l TRUE): l and l^1 always
    share a word (2v / 2v+1), so ONE load answers duplicate + contradiction at once."""
    SAT_MASK: tl.constexpr = (1 << SHIFT) - 1
    SUM_MASK: tl.constexpr = (1 << (32 - SHIFT)) - 1
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
            st = tl.load(st_row + c) - 1 - (neg << SHIFT)
            tl.store(st_row + c, st)
            sat = st & SAT_MASK
            if sat == 0:                            # all literals falsified
                conflict = 1
            elif sat == 1:                          # unit: remaining literal IS the sum field
                forced = (st >> SHIFT) & SUM_MASK
                w = tl.load(ab_row + (forced >> 5))
                bit = 1 << (forced & 31)
                if (w & bit) == 0:
                    if (w & (1 << ((forced ^ 1) & 31))) != 0:
                        conflict = 1                # forced both polarities
                    elif tail < TRAIL_CAP:
                        tl.store(ab_row + (forced >> 5), w | bit)
                        tl.store(trail_row + tail, forced)
                        tail += 1
                    else:                           # structurally unreachable (cap >= N+B)
                        conflict = 1
            o += 1
    return head, tail, conflict, nops

@triton.jit
def _wave_propagate(st_row, ab_row, trail_row, occ_off_ptr, occ_cls_ptr,
                    head, tail, conflict, nops, TRAIL_CAP,
                    SHIFT: tl.constexpr, W: tl.constexpr):
    """Warp-wide wavefront BCP (selected by WAVE=1): pop up to W trail literals at once,
    flatten the union of their occ segments, and process W clause updates per step with
    masked vector ops — one warp-wide gather instead of a serial dependent chain (up to W
    scattered requests in flight per row instead of 1).

    BOTH state updates are one atomic whose RETURN VALUE is the detection logic:

    * clause word: two frontier literals CAN share a clause, and the modular-SUM packing
      makes the whole update a single tl.atomic_add of -(1 + (neg << SHIFT)) (count field
      never borrows — count >= 1; sum field wraps off the top). Old count==1 -> this lane
      emptied the clause (conflict); old count==2 -> this lane made it unit and the forced
      literal is (old_sum - neg) & SUM_MASK (the serialized old sum holds exactly the
      not-yet-removed literals, of which neg is one).
    * assignment BITMAP: pushing forced literal l is tl.atomic_or of bit l, and l / l^1
      always share a word (2v / 2v+1) — old bit(l) set -> duplicate (drop: replaces the
      old WxW in-warp dedup), old bit(l^1) set -> contradiction (conflict). Two lanes
      forcing l and ~l in the SAME step serialize in the L2 atomic unit, so exactly one
      of them sees the other's bit: race-free with zero extra loads.

    Propagation ORDER differs from the scalar kernel, but BCP fixpoints are confluent:
    non-conflicted rows reach the IDENTICAL fixpoint (bit-identical models — the
    no-change gate validates this variant directly); conflicted rows' partial state is
    unspecified, as ever. `nops` counts the same occ entries."""
    SAT_MASK: tl.constexpr = (1 << SHIFT) - 1
    SUM_MASK: tl.constexpr = (1 << (32 - SHIFT)) - 1
    lanes = tl.arange(0, W)
    while (head < tail) & (conflict == 0):
        # ---- frontier chunk: up to W pending literals; flatten their occ segments ----
        f = tl.minimum(tail - head, W)
        lm = lanes < f
        lit_v = tl.load(trail_row + head + lanes, mask=lm, other=0)
        neg_v = lit_v ^ 1
        s_v = tl.load(occ_off_ptr + lit_v, mask=lm, other=0)
        e_v = tl.load(occ_off_ptr + lit_v + 1, mask=lm, other=0)
        len_v = tl.where(lm, e_v - s_v, 0)
        cum_v = tl.cumsum(len_v, 0)                       # inclusive prefix sum
        T = tl.max(cum_v, 0)                              # total occ entries this chunk
        head += f
        nops += T
        t0 = f * 0
        while (t0 < T) & (conflict == 0):
            t_v = t0 + lanes
            act = t_v < T
            # lane -> (segment, offset): seg = #{j: cum_j <= t}; one-hot gathers seg attrs
            seg_v = tl.sum((cum_v[None, :] <= t_v[:, None]).to(tl.int32), 1)
            oneh = lanes[None, :] == seg_v[:, None]
            excl_v = tl.sum(tl.where(oneh, (cum_v - len_v)[None, :], 0), 1)
            base_v = tl.sum(tl.where(oneh, s_v[None, :], 0), 1)
            mneg_v = tl.sum(tl.where(oneh, neg_v[None, :], 0), 1)
            o_v = base_v + (t_v - excl_v)
            c_v = tl.load(occ_cls_ptr + o_v, mask=act, other=0).to(tl.int64)
            # ---- ONE atomic per entry: st -= 1 + (mneg << SHIFT); old value = detection
            old_v = tl.atomic_add(st_row + c_v, -(1 + (mneg_v << SHIFT)), mask=act)
            oldsat_v = old_v & SAT_MASK
            stride_conf = tl.sum((act & (oldsat_v == 1)).to(tl.int32), 0)
            # ---- unit pushes: ONE atomic_or per forced literal; old word = dedup+contra
            fm = act & (oldsat_v == 2)
            forced_v = tl.where(fm, ((old_v >> SHIFT) - mneg_v) & SUM_MASK, 0)
            bit_v = 1 << (forced_v & 31)
            oldw_v = tl.atomic_or(ab_row + (forced_v >> 5), bit_v, mask=fm)
            contra_v = fm & ((oldw_v & (1 << ((forced_v ^ 1) & 31))) != 0)
            stride_conf += tl.sum(contra_v.to(tl.int32), 0)
            keep = fm & ((oldw_v & bit_v) == 0) & ~contra_v
            k32 = keep.to(tl.int32)
            pos_v = tail + tl.cumsum(k32, 0) - k32
            tl.store(trail_row + pos_v, forced_v, mask=keep)
            tail += tl.sum(k32, 0)
            conflict = conflict | (stride_conf > 0)
            t0 += W
    return head, tail, conflict, nops

@triton.jit
def _bcp_fixpoint_kernel(
    st_ptr,        # (K, M)  int32 packed clause state (row-private per program)
    ab_ptr,        # (K, NW) int32 assignment BITMAP (bit l = literal l TRUE)
    active_ptr,    # (K,)    int8 view of bool
    conflict_ptr,  # (K,)    int8 view of bool
    dec_ptr,       # (K, B)  int64 literal indices, -1 = padding
    occ_off_ptr,   # (2N+1,) int32 CSR offsets  (occ[l] = clauses where ~l appears)
    occ_cls_ptr,   # (total,) int32 CSR clause indices
    trail_ptr,     # (K, TRAIL_CAP) int32 workspace
    nprops_ptr,    # (K,) int32 out: literals set TRUE this call (seeds + forced)
    nops_ptr,      # (K,) int32 out: occ entries processed
    ndecs_ptr,     # (K,) int32 out: DECISION literals (seeds; implications = nprops - ndecs)
    M, NW, B, TRAIL_CAP,
    SHIFT: tl.constexpr,
    WAVE: tl.constexpr,                   # 1 = warp-wide wavefront propagate (experimental)
    W: tl.constexpr,                      # wavefront lane width (power of 2)
):
    k = tl.program_id(0).to(tl.int64)
    if tl.load(active_ptr + k) != 0:
        st_row = st_ptr + k * M
        ab_row = ab_ptr + k * NW
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
                w = tl.load(ab_row + (lit >> 5))        # lit / lit^1 share this word
                if (w & (1 << ((lit ^ 1) & 31))) != 0:  # contradicts existing / earlier seed
                    conflict = 1
                elif (w & (1 << (lit & 31))) == 0:      # skip duplicates
                    tl.store(ab_row + (lit >> 5), w | (1 << (lit & 31)))
                    tl.store(trail_row + tail, lit)
                    tail += 1
        ndecs = tail                # seeds assigned so far ARE the decisions

        if WAVE:
            head, tail, conflict, nops = _wave_propagate(
                st_row, ab_row, trail_row, occ_off_ptr, occ_cls_ptr,
                head, tail, conflict, nops, TRAIL_CAP, SHIFT, W)
        else:
            head, tail, conflict, nops = _propagate(
                st_row, ab_row, trail_row, occ_off_ptr, occ_cls_ptr,
                head, tail, conflict, nops, TRAIL_CAP, SHIFT)

        if conflict != 0:
            tl.store(conflict_ptr + k, 1)
            tl.store(active_ptr + k, 0)
        tl.store(nprops_ptr + k, tail)
        tl.store(nops_ptr + k, nops)
        tl.store(ndecs_ptr + k, ndecs)

@triton.jit
def _bcp_trajectory_kernel(
    st_ptr, ab_ptr, active_ptr, conflict_ptr,
    order_ptr,                            # (P,) int32 var indices: the decision order (support
                                          #   first, then completion vars; a per-chunk permutation)
    occ_off_ptr, occ_cls_ptr,
    trail_ptr, nrounds_ptr, nops_ptr, nprops_ptr, ndecs_ptr,   # (K,·) workspace / outputs
    M, NW, P, NSUP, B, TRAIL_CAP, MAX_ROUNDS, seed,
    SOFT: tl.constexpr,                   # 1 = "soft": in-kernel soft risk-polarity (see below)
    SHIFT: tl.constexpr,
    WAVE: tl.constexpr,                   # 1 = warp-wide wavefront propagate (experimental)
    W: tl.constexpr,                      # wavefront lane width (power of 2)
    VSCAN: tl.constexpr,                  # 1 = warp-wide decision scan (both modes; GPAP_VSCAN=0 off)
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
        ab_row = ab_ptr + k * NW
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
            if WAVE and VSCAN and not SOFT:
                # Warp-wide decision SCAN (bit-identical picks: the first B still-free vars
                # in `order`, same per-(chunk,k,v) polarity hash): each step screens W
                # entries with ONE bitmap-word gather (var v's literal bits 2v/2v+1 sit in
                # word v>>4). WAVE-only: the bitmap shrank the scan's working set 8-16x
                # (small-2N rows are L1-resident), which made the serial scan cheap enough
                # that the warp-wide screen + take-atomics LOSE in scalar mode (b06 0.82x,
                # b03 0.91x) while still winning at wave's low-K regime (b04 1.13x,
                # b12/b14 1.19x). The take is an atomic_or because two taken vars can
                # share a bitmap word.
                lanes_d = tl.arange(0, W)
                while (picked < B) & (ptr < phase_end):
                    am = ptr + lanes_d < phase_end
                    v_v = tl.load(order_ptr + ptr + lanes_d, mask=am, other=0)
                    w_v = tl.load(ab_row + (v_v >> 4), mask=am, other=-1)
                    free_v = am & (((w_v >> ((v_v & 15) * 2)) & 3) == 0)
                    rank_v = tl.cumsum(free_v.to(tl.int32), 0)     # inclusive rank among free
                    need = B - picked
                    take_v = free_v & (rank_v <= need)
                    # entries consumed: through the need-th free pick (else the whole chunk)
                    hitpos = tl.min(tl.where(free_v & (rank_v == need), lanes_d, W), 0)
                    chunkw = tl.minimum(phase_end - ptr, W)
                    consumed = tl.minimum(hitpos + 1, chunkw)
                    h_v = (seed + k * -7046029254386353131
                           + v_v.to(tl.int64) * -4658895280553007687)
                    h_v = (h_v ^ (h_v >> 30)) * -4658895280553007687
                    h_v = (h_v ^ (h_v >> 27)) * -7723592293110705685
                    pol_v = ((h_v >> 13) & 1).to(tl.int32)
                    lit_v = 2 * v_v + pol_v
                    tl.atomic_or(ab_row + (lit_v >> 5), 1 << (lit_v & 31), mask=take_v)
                    tl.store(trail_row + tail + rank_v - 1, lit_v, mask=take_v)
                    n_take = tl.sum(take_v.to(tl.int32), 0)
                    tail += n_take
                    picked += n_take
                    ptr += consumed
            while (picked < B) & (ptr < phase_end):
                v = tl.load(order_ptr + ptr)
                ptr += 1
                wp = tl.load(ab_row + (v >> 4))         # var v's pair bits: word v>>4
                if ((wp >> ((v & 15) * 2)) & 3) == 0:
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
                    tl.store(ab_row + (lit >> 5), wp | (1 << (lit & 31)))
                    tl.store(trail_row + tail, lit)
                    tail += 1
                    picked += 1
            if picked > 0:
                rounds += 1
                decs += picked
            # --- propagate to fixpoint ---
            if WAVE:
                head, tail, conflict, nops = _wave_propagate(
                    st_row, ab_row, trail_row, occ_off_ptr, occ_cls_ptr,
                    head, tail, conflict, nops, TRAIL_CAP, SHIFT, W)
            else:
                head, tail, conflict, nops = _propagate(
                    st_row, ab_row, trail_row, occ_off_ptr, occ_cls_ptr,
                    head, tail, conflict, nops, TRAIL_CAP, SHIFT)

        if conflict != 0:
            tl.store(conflict_ptr + k, 1)
            tl.store(active_ptr + k, 0)
        tl.store(nrounds_ptr + k, rounds)
        tl.store(nops_ptr + k, nops)
        tl.store(nprops_ptr + k, tail)    # literals assigned (decisions + implications)
        tl.store(ndecs_ptr + k, decs)     # decisions only (implications = nprops - ndecs)

@triton.jit
def _clean_packed(st_row, init_ptr, trail_row, occ_off_ptr, occ_cls_ptr, tail, M,
                  W: tl.constexpr):
    """Rewrite every packed word this row can have touched back to the baseline row
    (`init_ptr` = the post-units-flush packed state) — the CLEAN-mode tail.

    Soundness (docs/plans/RESET_RESTORE_PLAN.md §3.1): packed writes only ever happen in the
    occ walk of literals POPPED FROM THE TRAIL, so the damage is a subset of
    ∪ occ[l] over trail[0..tail) — for surviving AND conflicted rows (a mid-list or mid-wave
    stop only shrinks the damage). Every store writes the baseline value, so over-covering
    (trail tails never propagated, occ words never reached, two trail literals sharing a
    clause) rewrites baseline over baseline: idempotent, no atomics, benign lane races.

    DEEP-ROW GUARD: scattered 4-byte stores pull 32-byte sectors (~8x amplification), so a row
    whose total occ work exceeds M/8 streams its WHOLE row from baseline instead — that row
    degrades to (its share of) the old broadcast cost, never past it.

    Vectorized W-wide in both kernel modes (the scalar/wave split is about the PROPAGATE
    algorithm; nothing here mutates shared state non-idempotently, so redundant lanes and the
    flattened-segment races are all benign)."""
    lanes = tl.arange(0, W)
    total = tail * 0
    t = tail * 0
    while t < tail:
        m = t + lanes < tail
        lit_v = tl.load(trail_row + t + lanes, mask=m, other=0)
        s_v = tl.load(occ_off_ptr + lit_v, mask=m, other=0)
        e_v = tl.load(occ_off_ptr + lit_v + 1, mask=m, other=0)
        total += tl.sum(tl.where(m, e_v - s_v, 0), 0)
        t += W
    if total > M // 8:
        j = tail * 0
        while j < M:
            jm = j + lanes < M
            v = tl.load(init_ptr + j + lanes, mask=jm, other=0)
            tl.store(st_row + j + lanes, v, mask=jm)
            j += W
    else:
        t = tail * 0
        while t < tail:
            # flatten up to W trail literals' occ segments — the _wave_propagate idiom
            f = tl.minimum(tail - t, W)
            lm = lanes < f
            lit_v = tl.load(trail_row + t + lanes, mask=lm, other=0)
            s_v = tl.load(occ_off_ptr + lit_v, mask=lm, other=0)
            e_v = tl.load(occ_off_ptr + lit_v + 1, mask=lm, other=0)
            len_v = tl.where(lm, e_v - s_v, 0)
            cum_v = tl.cumsum(len_v, 0)
            T = tl.max(cum_v, 0)
            t += f
            t0 = f * 0
            while t0 < T:
                t_v = t0 + lanes
                act = t_v < T
                seg_v = tl.sum((cum_v[None, :] <= t_v[:, None]).to(tl.int32), 1)
                oneh = lanes[None, :] == seg_v[:, None]
                excl_v = tl.sum(tl.where(oneh, (cum_v - len_v)[None, :], 0), 1)
                base_v = tl.sum(tl.where(oneh, s_v[None, :], 0), 1)
                o_v = base_v + (t_v - excl_v)
                c_v = tl.load(occ_cls_ptr + o_v, mask=act, other=0).to(tl.int64)
                v = tl.load(init_ptr + c_v, mask=act, other=0)
                tl.store(st_row + c_v, v, mask=act)
                t0 += W

@triton.jit
def _bcp_fixpoint_clean_kernel(
    st_ptr, ab_ptr, active_ptr, conflict_ptr, dec_ptr, occ_off_ptr, occ_cls_ptr,
    init_ptr,      # (M,) int32 packed_flushed — the baseline the clean tail restores
    trail_ptr, nprops_ptr, nops_ptr, ndecs_ptr,
    M, NW, B, TRAIL_CAP,
    SHIFT: tl.constexpr,
    WAVE: tl.constexpr,
    W: tl.constexpr,
):
    """CLEAN-mode batched assumption check: seed + propagate EXACTLY as _bcp_fixpoint_kernel,
    emit the same flags/counters, then restore this row's packed words to the baseline before
    exiting — so the NEXT chunk's reset needs no K·M broadcast (docs/plans/RESET_RESTORE_PLAN.md
    §9; v1 cleans PACKED only, abits/flags are restored by the cheap reset-lite).

    A caller that reads post-apply STATE (assignment_rows, sat_rows, recache_flush, the
    selection api) must use the plain kernel — after this one, the state IS the baseline.
    Flags and the (K,) counters survive; they are the result.

    The seed loop and the propagate dispatch are duplicated VERBATIM from
    _bcp_fixpoint_kernel — keep them in lockstep (the original is deliberately untouched so
    its 29 shipped cubins stay byte-identical)."""
    k = tl.program_id(0).to(tl.int64)
    if tl.load(active_ptr + k) != 0:
        st_row = st_ptr + k * M
        ab_row = ab_ptr + k * NW
        trail_row = trail_ptr + k * TRAIL_CAP

        conflict = B * 0
        tail = B * 0
        head = B * 0
        nops = B * 0

        for j in range(0, B):
            lit64 = tl.load(dec_ptr + k * B + j)
            if (lit64 >= 0) & (conflict == 0):
                lit = lit64.to(tl.int32)
                w = tl.load(ab_row + (lit >> 5))
                if (w & (1 << ((lit ^ 1) & 31))) != 0:
                    conflict = 1
                elif (w & (1 << (lit & 31))) == 0:
                    tl.store(ab_row + (lit >> 5), w | (1 << (lit & 31)))
                    tl.store(trail_row + tail, lit)
                    tail += 1
        ndecs = tail

        if WAVE:
            head, tail, conflict, nops = _wave_propagate(
                st_row, ab_row, trail_row, occ_off_ptr, occ_cls_ptr,
                head, tail, conflict, nops, TRAIL_CAP, SHIFT, W)
        else:
            head, tail, conflict, nops = _propagate(
                st_row, ab_row, trail_row, occ_off_ptr, occ_cls_ptr,
                head, tail, conflict, nops, TRAIL_CAP, SHIFT)

        if conflict != 0:
            tl.store(conflict_ptr + k, 1)
            tl.store(active_ptr + k, 0)
        tl.store(nprops_ptr + k, tail)
        tl.store(nops_ptr + k, nops)
        tl.store(ndecs_ptr + k, ndecs)

        _clean_packed(st_row, init_ptr, trail_row, occ_off_ptr, occ_cls_ptr, tail, M, W)
