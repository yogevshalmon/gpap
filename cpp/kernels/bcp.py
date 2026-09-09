"""The two Triton BCP kernels — the only execution path of the engine (packed sum-trick).

BUILD INPUT, NOT RUNTIME CODE: cpp/tools/build_kernels.py AOT-compiles every specialization into
libgpap.so, which then needs libcuda alone. Never import torch here; CI builds with triton alone.

One program per batch row, whose packed row and bitmap row are program-private (no cross-program
races; WAVE's in-row lane races are resolved by atomic return values). Each pops literals off a
trail of newly-TRUE ones — one at a time in scalar, up to W at once in WAVE — and walks their
occurrence lists. Only length>=3 clauses have a packed word, so M here is num_long; length-2
clauses are IMPLICATION EDGES in the binary prefix of each occ segment (payload = the other
literal) propagated by one bitmap push, and units live in the flush cache.

Both row states are single words updated by ONE op whose old value IS the detection logic:
  * PACKED clause word, one int32 per (row, clause): low SHIFT bits = sat_count, high 32-SHIFT
    bits = SUM of the non-falsified literal indices mod 2^(32-SHIFT) (SHIFT=16 for 2N <= 65535,
    SHIFT=8 for 2N <= 0xFFFFFF with clause length <= 255; anything else is rejected). Every
    removal is exact mod 2^F, so at sat_count==1 the sum field IS the last literal, and one
    falsified literal costs one subtraction st -= 1 + (neg << SHIFT) — the count never borrows,
    the sum wraps off the top. In WAVE that subtraction is a tl.atomic_add whose RETURN gives
    race-free detection: old count==1 means this lane emptied the clause, old count==2 means it
    made it unit and (old_sum - neg) & SUM_MASK is the forced literal.
  * Assignment BITMAP, bit l of (K, ceil(2N/32)) int32 words, where l and l^1 share a word, so
    ONE load answers free/duplicate/contradiction and in WAVE the push is a tl.atomic_or whose
    RETURN deduplicates (old bit l) and detects contradictions (old bit l^1).

`_bcp_fixpoint_kernel` seeds a (K,B) decision tensor and BCPs to fixpoint;
`_bcp_trajectory_kernel` runs the whole randomized decide+propagate loop. Contract:
conflict/active flags are exact (fixpoints are confluent) and a non-conflicted row's assignment
and counters are the exact fixpoint, while a CONFLICTED row's partial state is unspecified.
"""
import triton
import triton.language as tl

@triton.jit
def _propagate(st_row, ab_row, trail_row, occ_off_ptr, occ_mid_ptr, occ_cls_ptr,
               head, tail, conflict, nops, TRAIL_CAP, SHIFT: tl.constexpr):
    """Pop literals off the trail and BCP to fixpoint (or conflict) on the packed row. Inlined by
    both kernels; returns the updated (head, tail, conflict, nops). Each occ[lit] segment is
    [BINARY | LONG], split at occ_mid[lit], and a binary entry's payload IS the forced literal —
    the clause (~lit | payload) just lost ~lit — so its whole update is the bitmap push, with no
    clause word. Binaries run first for cheap fail-fast; confluence makes the order
    unobservable on surviving rows."""
    SAT_MASK: tl.constexpr = (1 << SHIFT) - 1
    SUM_MASK: tl.constexpr = (1 << (32 - SHIFT)) - 1
    while (head < tail) & (conflict == 0):
        lit = tl.load(trail_row + head)
        head += 1
        neg = lit ^ 1
        start = tl.load(occ_off_ptr + lit)
        mid = tl.load(occ_mid_ptr + lit)
        end = tl.load(occ_off_ptr + lit + 1)
        nops += end - start
        o = start
        while (o < mid) & (conflict == 0):          # binary prefix: payload IS the forced lit
            forced = tl.load(occ_cls_ptr + o) & 0x7FFFFFFF     # strip the binary tag bit
            w = tl.load(ab_row + (forced >> 5))
            bit = 1 << (forced & 31)
            if (w & bit) == 0:
                if (w & (1 << ((forced ^ 1) & 31))) != 0:
                    conflict = 1
                elif tail < TRAIL_CAP:
                    tl.store(ab_row + (forced >> 5), w | bit)
                    tl.store(trail_row + tail, forced)
                    tail += 1
                else:
                    conflict = 1                    # structurally unreachable (cap >= N+B)
            o += 1
        o = mid
        while (o < end) & (conflict == 0):          # long suffix: packed clause words
            c = tl.load(occ_cls_ptr + o).to(tl.int64)
            st = tl.load(st_row + c) - 1 - (neg << SHIFT)
            tl.store(st_row + c, st)
            sat = st & SAT_MASK
            if sat == 0:
                conflict = 1
            elif sat == 1:                          # unit: remaining literal IS the sum field
                forced = (st >> SHIFT) & SUM_MASK
                w = tl.load(ab_row + (forced >> 5))
                bit = 1 << (forced & 31)
                if (w & bit) == 0:
                    if (w & (1 << ((forced ^ 1) & 31))) != 0:
                        conflict = 1
                    elif tail < TRAIL_CAP:
                        tl.store(ab_row + (forced >> 5), w | bit)
                        tl.store(trail_row + tail, forced)
                        tail += 1
                    else:
                        conflict = 1
            o += 1
    return head, tail, conflict, nops

@triton.jit
def _seg_lookup(cum_v, adj_v, t_v, lanes, W: tl.constexpr):
    """Map each lane's flattened index `t` to (segment, occ offset) for a frontier chunk. `cum_v`
    is the inclusive prefix sum of the segment lengths and `adj_v = s_v - (cum_v - len_v)` the
    per-segment offset correction, so entry `t` of segment `j` is at occ index `adj_j + t`. The
    segment is `#{j: cum_j <= t}`, a branchless lower_bound of log2(W) `tl.gather` probes, each
    one `shfl.sync` — this replaced a WxW compare-reduce that Triton lowered through SHARED
    memory with barriers (docs/findings/WAVE_SHUFFLE_FINDINGS.md). In range by construction: an
    active lane has `t < T <= cum_{f-1}`, so every probe index is `<= W-2`; inactive lanes
    saturate at `W-1`, masked off by every caller. W must be a power of two."""
    seg_v = lanes * 0
    step = W // 2
    while step >= 1:
        # `step` is a constexpr but triton still emits this loop; MLIR/LLVM unroll the constant
        # trip count, which is what makes the schedule log2(W) shuffles with no shared memory.
        # An optimizer outcome: if W or this body changes, re-check the PTX.
        probe_v = tl.gather(cum_v, seg_v + (step - 1), 0)
        seg_v += tl.where(probe_v <= t_v, step, 0)
        step //= 2
    return seg_v, t_v + tl.gather(adj_v, seg_v, 0)

@triton.jit
def _wave_propagate(st_row, ab_row, trail_row, occ_off_ptr, occ_mid_ptr, occ_cls_ptr,
                    head, tail, conflict, nops, TRAIL_CAP,
                    SHIFT: tl.constexpr, W: tl.constexpr):
    """Warp-wide wavefront BCP (WAVE=1): pop up to W trail literals at once, flatten the union of
    their occ segments, and process W clause updates per step with masked vector ops — one
    warp-wide gather instead of a serial dependent chain. Both state updates are one atomic whose
    RETURN VALUE is the detection logic:
    * clause word: two frontier literals CAN share a clause, and the modular-sum packing makes
      the whole update a single tl.atomic_add of -(1 + (neg << SHIFT)). Old count==1 means this
      lane emptied the clause; old count==2 means it made it unit, and the forced literal is
      (old_sum - neg) & SUM_MASK, since the serialized old sum holds exactly the not-yet-removed
      literals, of which neg is one.
    * assignment BITMAP: pushing forced literal l is tl.atomic_or of bit l, and l / l^1 share a
      word — old bit(l) means duplicate, old bit(l^1) contradiction. Two lanes forcing l and ~l
      in one step serialize in the L2 atomic unit, so exactly one sees the other's bit.
    `occ_mid_ptr` is unused (the split is read from the payload tag) but stays in the signature so
    both variants take the same arguments. Order differs from the scalar kernel, but fixpoints
    are confluent, so non-conflicted rows reach the identical one."""
    SAT_MASK: tl.constexpr = (1 << SHIFT) - 1
    SUM_MASK: tl.constexpr = (1 << (32 - SHIFT)) - 1
    lanes = tl.arange(0, W)
    while (head < tail) & (conflict == 0):
        f = tl.minimum(tail - head, W)
        lm = lanes < f
        lit_v = tl.load(trail_row + head + lanes, mask=lm, other=0)
        neg_v = lit_v ^ 1
        s_v = tl.load(occ_off_ptr + lit_v, mask=lm, other=0)
        e_v = tl.load(occ_off_ptr + lit_v + 1, mask=lm, other=0)
        len_v = tl.where(lm, e_v - s_v, 0)
        cum_v = tl.cumsum(len_v, 0)                       # inclusive prefix sum; T = its max
        T = tl.max(cum_v, 0)
        adj_v = s_v - (cum_v - len_v)                     # entry t of seg j is at adj_j + t
        head += f
        nops += T
        t0 = f * 0
        while (t0 < T) & (conflict == 0):
            t_v = t0 + lanes
            act = t_v < T
            seg_v, o_v = _seg_lookup(cum_v, adj_v, t_v, lanes, W)
            mneg_v = tl.gather(neg_v, seg_v, 0)
            pay_v = tl.load(occ_cls_ptr + o_v, mask=act, other=0)
            # A TAGGED (negative) payload is a BINARY entry: it IS the forced literal and has no
            # clause word, so it goes straight to the push block below (reading the tag off the
            # loaded value is free; gathering occ_mid would be a third shuffle gather).
            binm = act & (pay_v < 0)
            lngm = act & (pay_v >= 0)
            c_v = pay_v.to(tl.int64)
            old_v = tl.atomic_add(st_row + c_v, -(1 + (mneg_v << SHIFT)), mask=lngm)
            oldsat_v = old_v & SAT_MASK
            # ---- forced pushes: ONE atomic_or per literal; old word = dedup + contradiction,
            #      binary lanes joining the long-unit lanes, race-free by L2 serialization.
            fm = binm | (lngm & (oldsat_v == 2))
            forced_v = tl.where(
                fm, tl.where(binm, pay_v & 0x7FFFFFFF,
                             ((old_v >> SHIFT) - mneg_v) & SUM_MASK), 0)
            bit_v = 1 << (forced_v & 31)
            oldw_v = tl.atomic_or(ab_row + (forced_v >> 5), bit_v, mask=fm)
            contra_v = fm & ((oldw_v & (1 << ((forced_v ^ 1) & 31))) != 0)
            conf_v = contra_v | (lngm & (oldsat_v == 1))   # pushed both polarities | emptied a clause
            keep = fm & ((oldw_v & bit_v) == 0) & ~contra_v
            k32 = keep.to(tl.int32)
            pos_v = tail + tl.cumsum(k32, 0) - k32
            tl.store(trail_row + pos_v, forced_v, mask=keep)
            tail += tl.sum(k32, 0)
            conflict = conflict | (tl.sum(conf_v.to(tl.int32), 0) > 0)
            t0 += W
    return head, tail, conflict, nops

@triton.jit
def _seed_and_propagate(st_row, ab_row, trail_row, dec_ptr, k,
                        occ_off_ptr, occ_mid_ptr, occ_cls_ptr,
                        active_ptr, conflict_ptr, nprops_ptr, nops_ptr, ndecs_ptr,
                        B, TRAIL_CAP,
                        SHIFT: tl.constexpr, WAVE: tl.constexpr, W: tl.constexpr):
    """The body of entry point 1: seed row k's decisions, propagate to fixpoint, write the row's
    flags and its three counters. Returns `tail`, the trail depth the CLEAN tail needs. Shared by
    _bcp_fixpoint_kernel and _bcp_fixpoint_clean_kernel, which must agree EXACTLY: CLEAN mode's
    gate (tests/test_clean_mode.py) is that a clean launch reaches state identical to a plain
    one, so a divergence here is a wrong answer, not a slowdown."""
    conflict = B * 0            # runtime-typed int32 scalar (loop-carried)
    tail = B * 0
    head = B * 0
    nops = B * 0

    # --- seed: assign each decision literal and queue it ---
    for j in range(0, B):
        lit64 = tl.load(dec_ptr + k * B + j)
        if (lit64 >= 0) & (conflict == 0):
            lit = lit64.to(tl.int32)
            w = tl.load(ab_row + (lit >> 5))        # lit / lit^1 share this word
            if (w & (1 << ((lit ^ 1) & 31))) != 0:  # contradicts existing / earlier seed
                conflict = 1
            elif (w & (1 << (lit & 31))) == 0:
                tl.store(ab_row + (lit >> 5), w | (1 << (lit & 31)))
                tl.store(trail_row + tail, lit)
                tail += 1
    ndecs = tail                # seeds assigned so far ARE the decisions

    if WAVE:
        head, tail, conflict, nops = _wave_propagate(
            st_row, ab_row, trail_row, occ_off_ptr, occ_mid_ptr, occ_cls_ptr,
            head, tail, conflict, nops, TRAIL_CAP, SHIFT, W)
    else:
        head, tail, conflict, nops = _propagate(
            st_row, ab_row, trail_row, occ_off_ptr, occ_mid_ptr, occ_cls_ptr,
            head, tail, conflict, nops, TRAIL_CAP, SHIFT)

    if conflict != 0:
        tl.store(conflict_ptr + k, 1)
        tl.store(active_ptr + k, 0)
    tl.store(nprops_ptr + k, tail)
    tl.store(nops_ptr + k, nops)
    tl.store(ndecs_ptr + k, ndecs)
    return tail


@triton.jit
def _bcp_fixpoint_kernel(
    st_ptr,        # (K, M)  int32 packed clause state (row-private per program)
    ab_ptr,        # (K, NW) int32 assignment BITMAP (bit l = literal l TRUE)
    active_ptr,    # (K,) int8 flags
    conflict_ptr,
    dec_ptr,       # (K, B)  int64 literal indices, -1 = padding
    occ_off_ptr,   # (2N+1,) int32 CSR offsets (occ[l] = entries where ~l appears); occ_mid ends
    occ_mid_ptr,   #   the binary prefix of occ[l]; occ_cls payloads the other literal (binary)
    occ_cls_ptr,   #   or the long clause id
    trail_ptr,     # (K, TRAIL_CAP) int32 workspace
    nprops_ptr,    # (K,) int32 out: literals set TRUE (seeds + forced), occ entries processed,
    nops_ptr,      #   and DECISION literals; implications = nprops - ndecs
    ndecs_ptr,
    M, NW, B, TRAIL_CAP,
    SHIFT: tl.constexpr,
    WAVE: tl.constexpr,                   # 1 = warp-wide wavefront propagate, W lanes wide
    W: tl.constexpr,
):
    k = tl.program_id(0).to(tl.int64)
    if tl.load(active_ptr + k) != 0:
        _seed_and_propagate(st_ptr + k * M, ab_ptr + k * NW, trail_ptr + k * TRAIL_CAP,
                            dec_ptr, k, occ_off_ptr, occ_mid_ptr, occ_cls_ptr,
                            active_ptr, conflict_ptr, nprops_ptr, nops_ptr, ndecs_ptr,
                            B, TRAIL_CAP, SHIFT, WAVE, W)

@triton.jit
def _risk(st_row, ab_row, occ_off_ptr, occ_mid_ptr, occ_cls_ptr, lit, SHIFT: tl.constexpr):
    """SOFT conflict-avoidance risk of setting `lit` TRUE: the weighted cost of the clauses it
    falsifies, over occ[lit]. One definition of the weights, called once per polarity. Binary
    entries are scored exactly from the bitmap (other literal true => satisfied, 0; false =>
    would empty the clause, 1e6; free => would force it, 4), sharper than a sat_count read since
    a count cannot see satisfaction. Long entries use the packed-word weights."""
    SAT_MASK: tl.constexpr = (1 << SHIFT) - 1
    r = lit * 0
    o = tl.load(occ_off_ptr + lit)
    mid = tl.load(occ_mid_ptr + lit)
    end = tl.load(occ_off_ptr + lit + 1)
    while o < mid:
        oth = tl.load(occ_cls_ptr + o) & 0x7FFFFFFF
        w = tl.load(ab_row + (oth >> 5))
        sat = (w >> (oth & 31)) & 1
        fal = (w >> ((oth ^ 1) & 31)) & 1
        r += tl.where(sat != 0, 0, tl.where(fal != 0, 1000000, 4))
        o += 1
    o = mid
    while o < end:
        c = tl.load(occ_cls_ptr + o).to(tl.int64)
        sc = tl.load(st_row + c) & SAT_MASK
        r += tl.where(sc == 1, 1000000, tl.where(sc == 2, 4, tl.where(sc == 3, 1, 0)))
        o += 1
    return r

@triton.jit
def _bcp_trajectory_kernel(
    st_ptr, ab_ptr, active_ptr, conflict_ptr,
    order_ptr,                            # (P,) int32 decision order: support vars first
    occ_off_ptr, occ_mid_ptr, occ_cls_ptr,
    trail_ptr, nrounds_ptr, nops_ptr, nprops_ptr, ndecs_ptr,   # (K,) workspace and outputs
    M, NW, P, NSUP, B, TRAIL_CAP, MAX_ROUNDS, seed,
    SOFT: tl.constexpr,                   # 1 = in-kernel soft risk-polarity (see below)
    SHIFT: tl.constexpr,
    WAVE: tl.constexpr,                   # 1 = warp-wide wavefront propagate; W = its lane
    W: tl.constexpr,                      #     width (a power of two)
    VSCAN: tl.constexpr,                  # 1 = warp-wide decision scan; read only under WAVE
                                          # and not SOFT (build_kernels.py::vscan_is_read)
):
    """The whole trajectory in one launch: repeatedly pick the next B still-free variables in
    `order`, assign each a hash-random polarity, and propagate to fixpoint. NSUP is the number of
    leading `order` entries that are phase-1 (support) vars, and a decision group must never
    straddle that boundary — phase 1 must reach fixpoint over the support first, or the group
    decides variables propagation would have determined and conflicts explode."""
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
        decs = B * 0                        # decisions; implications = tail - decs
        ptr = B * 0
        while (ptr < P) & (conflict == 0) & (rounds < MAX_ROUNDS):
            # --- decision group: next B free vars, hash-random polarity, ending at the
            #     support/completion boundary (see docstring). ---
            phase_end = NSUP if ptr < NSUP else P
            picked = B * 0
            if WAVE and VSCAN and not SOFT:
                # Warp-wide decision SCAN, bit-identical picks (the first B still-free vars in
                # `order`, same per-(chunk,k,v) polarity hash): each step screens W entries with
                # ONE bitmap-word gather, since var v's bits 2v/2v+1 sit in word v>>4. WAVE-only,
                # since the bitmap makes the serial scan win in scalar mode. The take is an
                # atomic_or because two taken vars can share a bitmap word.
                lanes_d = tl.arange(0, W)
                while (picked < B) & (ptr < phase_end):
                    am = ptr + lanes_d < phase_end
                    v_v = tl.load(order_ptr + ptr + lanes_d, mask=am, other=0)
                    w_v = tl.load(ab_row + (v_v >> 4), mask=am, other=-1)
                    free_v = am & (((w_v >> ((v_v & 15) * 2)) & 3) == 0)
                    rank_v = tl.cumsum(free_v.to(tl.int32), 0)
                    need = B - picked
                    take_v = free_v & (rank_v <= need)
                    # entries consumed: through the need-th free pick, else the whole chunk
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
                wp = tl.load(ab_row + (v >> 4))
                if ((wp >> ((v & 15) * 2)) & 3) == 0:
                    # stateless per-(chunk,k,v) polarity bit (splitmix64 mix, signed literals)
                    h = seed + k * -7046029254386353131 + v.to(tl.int64) * -4658895280553007687
                    h = (h ^ (h >> 30)) * -4658895280553007687
                    h = (h ^ (h >> 27)) * -7723592293110705685
                    pol = ((h >> 13) & 1).to(tl.int32)
                    if SOFT:
                        # Lower-risk polarity, hash bit breaking ties; weights live in _risk.
                        r0 = _risk(st_row, ab_row, occ_off_ptr, occ_mid_ptr, occ_cls_ptr,
                                   2 * v, SHIFT)
                        r1 = _risk(st_row, ab_row, occ_off_ptr, occ_mid_ptr, occ_cls_ptr,
                                   2 * v + 1, SHIFT)
                        # ε-greedy (25% hash-forced random): pure risk-argmin is
                        # row-deterministic when risks never tie, so all K trajectories would
                        # replicate ONE path. ε keeps the conflict-avoidance and diverges.
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
            if WAVE:
                head, tail, conflict, nops = _wave_propagate(
                    st_row, ab_row, trail_row, occ_off_ptr, occ_mid_ptr, occ_cls_ptr,
                    head, tail, conflict, nops, TRAIL_CAP, SHIFT, W)
            else:
                head, tail, conflict, nops = _propagate(
                    st_row, ab_row, trail_row, occ_off_ptr, occ_mid_ptr, occ_cls_ptr,
                    head, tail, conflict, nops, TRAIL_CAP, SHIFT)

        if conflict != 0:
            tl.store(conflict_ptr + k, 1)
            tl.store(active_ptr + k, 0)
        tl.store(nrounds_ptr + k, rounds)
        tl.store(nops_ptr + k, nops)
        tl.store(nprops_ptr + k, tail)    # literals assigned; decisions only in ndecs
        tl.store(ndecs_ptr + k, decs)

@triton.jit
def _clean_packed(st_row, init_ptr, trail_row, occ_off_ptr, occ_mid_ptr, occ_cls_ptr, tail, M,
                  W: tl.constexpr):
    """Rewrite every packed word this row can have touched back to the baseline row (`init_ptr` =
    the post-units-flush packed state) — the CLEAN-mode tail.

    Soundness (docs/findings/CLEAN_MODE_FINDINGS.md): packed writes only ever happen in the occ
    walk of literals POPPED FROM THE TRAIL, so the damage is a subset of the union of occ[l] over
    trail[0..tail) for surviving and conflicted rows alike, and every store writes the baseline
    value, so over-covering is idempotent — no atomics, benign lane races. Only the LONG
    sub-ranges are walked. Deep-row guard: a row whose occ work exceeds M/8 streams instead."""
    lanes = tl.arange(0, W)
    total = tail * 0
    t = tail * 0
    while t < tail:
        m = t + lanes < tail
        lit_v = tl.load(trail_row + t + lanes, mask=m, other=0)
        s_v = tl.load(occ_mid_ptr + lit_v, mask=m, other=0)
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
            s_v = tl.load(occ_mid_ptr + lit_v, mask=lm, other=0)
            e_v = tl.load(occ_off_ptr + lit_v + 1, mask=lm, other=0)
            len_v = tl.where(lm, e_v - s_v, 0)
            cum_v = tl.cumsum(len_v, 0)
            T = tl.max(cum_v, 0)
            adj_v = s_v - (cum_v - len_v)
            t += f
            t0 = f * 0
            while t0 < T:
                t_v = t0 + lanes
                act = t_v < T
                _, o_v = _seg_lookup(cum_v, adj_v, t_v, lanes, W)
                c_v = tl.load(occ_cls_ptr + o_v, mask=act, other=0).to(tl.int64)
                v = tl.load(init_ptr + c_v, mask=act, other=0)
                tl.store(st_row + c_v, v, mask=act)
                t0 += W

@triton.jit
def _bcp_fixpoint_clean_kernel(
    st_ptr, ab_ptr, active_ptr, conflict_ptr, dec_ptr, occ_off_ptr, occ_mid_ptr, occ_cls_ptr,
    init_ptr,      # (M,) int32 packed_flushed: the baseline the clean tail restores
    trail_ptr, nprops_ptr, nops_ptr, ndecs_ptr,
    M, NW, B, TRAIL_CAP,
    SHIFT: tl.constexpr,
    WAVE: tl.constexpr,
    W: tl.constexpr,
):
    """CLEAN-mode batched assumption check: seed and propagate exactly as _bcp_fixpoint_kernel,
    emit the same flags and counters, then restore this row's packed words to the baseline before
    exiting, so the next chunk's reset needs no K·M broadcast (this cleans PACKED only; abits and
    flags come back through the cheap reset-lite). A caller that reads post-apply STATE must use
    the plain kernel — after this one the state IS the baseline, and only the flags and (K,)
    counters survive. Seed and propagate are _seed_and_propagate, the plain kernel's own code."""
    k = tl.program_id(0).to(tl.int64)
    if tl.load(active_ptr + k) != 0:
        st_row = st_ptr + k * M
        trail_row = trail_ptr + k * TRAIL_CAP
        tail = _seed_and_propagate(st_row, ab_ptr + k * NW, trail_row,
                                   dec_ptr, k, occ_off_ptr, occ_mid_ptr, occ_cls_ptr,
                                   active_ptr, conflict_ptr, nprops_ptr, nops_ptr, ndecs_ptr,
                                   B, TRAIL_CAP, SHIFT, WAVE, W)
        _clean_packed(st_row, init_ptr, trail_row, occ_off_ptr, occ_mid_ptr, occ_cls_ptr,
                      tail, M, W)
