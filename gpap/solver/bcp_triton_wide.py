"""WIDE (8|24) packed trajectory kernel — the delta/XOR clause state for BIG instances.

The 16|16 packed kernel (bcp_triton._bcp_trajectory_packed_kernel) fuses sat_count + the XOR
remaining-literal trick into ONE int32, halving the scattered clause-state traffic — but its
16-bit XOR half caps literal indices at 2N <= 65535, so every instance with N > 32767 (e.g.
b12-c25 post-simplify: 2N = 67,920 — 3.6% over) silently falls back to the UNPACKED kernel:
5-6 bytes of state per clause, 2 loads + 2 stores per occ entry, a smaller K_fused budget, and
a per-chunk reset_state()+propagate() Python flush instead of the cached broadcast restore.

This module rebalances the SAME packed word to 8|24 (the observation behind the split: sat_count
is bounded by the max clause LENGTH — 5-6 on Tseitin circuit unrollings, <= 255 essentially
always — while the XOR half needs ceil(log2(2N)) bits, the part that actually grows with
instance size):

      low  8 bits = sat_count            (max clause size <= 255)
      high 24 bits = XOR of non-falsified lit indices   (2N <= 16,777,215 -> N <= 8.4M)

Same 4 bytes, same 1-load-1-store hot path, covering every realistic unrolling. The packed
update is the 16|16 one with shifted constants:  st = (st - 1) ^ (neg << 8)
  * the -1 only touches the sat byte: sat >= 1 at every decrement (sat==0 conflicts and halts
    the row first), so the subtraction never borrows into the XOR bits;
  * (neg << 8) only touches bits 8..31. For 2N near the 24-bit ceiling the shift may set bit 31
    (int32 sign) — harmless: the arithmetic is bitwise and extraction re-masks 24 bits.
sat==1 -> forced literal = (st >> 8) & 0xFFFFFF (mask kills the arithmetic-shift sign fill).

Provenance: the user's "track only the delta" idea (falsified-count + falsified-XOR with the
per-clause initial XOR consulted at unit time). Tracking the falsified XOR vs the non-falsified
XOR is the same state up to the constant initial XOR (nonfals = init ^ fals), so this keeps the
16|16 kernel's non-falsified convention — bit-for-bit the same trajectories, just wider keys.
The idea's uniform-clause-length / dummy-literal padding requirement dissolves entirely: the
count half already carries per-clause init (= clause length), so mixed lengths need no padding.

Opt-in: GPUSampler(use_kernel="wide") attaches TritonBCPWide, which packs 8|24 ONLY where the
16|16 kernel is ineligible (65535 < 2N <= 0xFFFFFF, max clause <= 255) and defers to the base
class everywhere else — small instances keep the proven 16|16 path bit-identically.
"""
import numpy as np
import torch

from gpap.solver.bcp_triton import _HAVE_TRITON, TritonBCP

if _HAVE_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _bcp_trajectory_mono8_kernel(
        st_ptr,                               # (K, M) int8 clause state: st = 127 - sat_count.
                                              #   Falsify a literal -> st += 1.
                                              #   st == 126 -> unit (sat==1); st == 127 -> conflict.
        asg_ptr, active_ptr, conflict_ptr,
        order_ptr, occ_off_ptr, occ_cls_ptr,
        cls_off_ptr,                          # (M+1,) int32 CSR: clause -> its literal indices
        cls_lits_ptr,                         # (total,) int32
        trail_ptr, nrounds_ptr, nops_ptr, nprops_ptr, ndecs_ptr,
        M, N2, P, NSUP, B, TRAIL_CAP, MAX_ROUNDS, seed,
        SOFT: tl.constexpr,
    ):
        """MONO (count-only) trajectory kernel — ONE BYTE of clause state per trajectory.

        The delta idea taken to its floor: drop the XOR/lit_sum half entirely and keep only the
        falsified count, re-based so the unit/conflict thresholds are CONSTANTS for every clause
        regardless of its length ("virtual padding": st starts at 127 - len, i.e. the missing
        literals of a hypothetical max-length clause are pre-falsified). The remaining literal of
        a unit clause is recovered by SCANNING the clause's <= 6 literals for the one whose
        negation is unassigned — only on unit events, against the trajectory's own hot assignment
        row. No index bits in the state word means NO instance-size ceiling (any N), 4x less
        state than the packed kernels (M bytes/trajectory) and 4x lighter per-chunk reset.

        st never wraps: st==127 halts the row before any further update, and a clause at st==126
        can only be touched again by falsifying its last non-falsified literal, which the forced
        assignment (or an existing TRUE) makes impossible. init >= 0 requires len <= 127
        (wrapper-gated). Trajectory semantics (order walk, hash polarity, propagation order)
        are IDENTICAL to the packed kernels — same models, bit for bit."""
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
                                sc0 = tl.load(st_row + c0).to(tl.int32)   # sat = 127 - st
                                r0 += tl.where(sc0 == 126, 1000000,
                                               tl.where(sc0 == 125, 4, tl.where(sc0 == 124, 1, 0)))
                                o0 += 1
                            e1 = tl.load(occ_off_ptr + 2 * v + 2)
                            o1 = e0
                            while o1 < e1:
                                c1 = tl.load(occ_cls_ptr + o1).to(tl.int64)
                                sc1 = tl.load(st_row + c1).to(tl.int32)
                                r1 += tl.where(sc1 == 126, 1000000,
                                               tl.where(sc1 == 125, 4, tl.where(sc1 == 124, 1, 0)))
                                o1 += 1
                            # ε-greedy (25% hash-forced random) — see the 16|16 kernel's comment.
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
                    start = tl.load(occ_off_ptr + lit)
                    end = tl.load(occ_off_ptr + lit + 1)
                    nops += end - start
                    o = start
                    while (o < end) & (conflict == 0):
                        c = tl.load(occ_cls_ptr + o).to(tl.int64)
                        st = tl.load(st_row + c).to(tl.int32) + 1
                        tl.store(st_row + c, st.to(tl.int8))
                        if st == 127:                       # all literals falsified
                            conflict = 1
                        elif st == 126:                     # unit: scan for the non-falsified lit
                            cs = tl.load(cls_off_ptr + c)
                            ce = tl.load(cls_off_ptr + c + 1)
                            forced = 0 - 1
                            j = cs
                            while (j < ce) & (forced < 0):
                                lt = tl.load(cls_lits_ptr + j)
                                if tl.load(asg_row + (lt ^ 1)) == 0:
                                    forced = lt
                                j += 1
                            if forced < 0:                  # unreachable given sat==1; defensive
                                conflict = 1
                            elif tl.load(asg_row + forced) == 0:
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

    @triton.jit
    def _bcp_trajectory_packed24_kernel(
        st_ptr,                               # (K, M) int32 PACKED clause state:
                                              #   low 8 bits   = sat_count (non-falsified count)
                                              #   high 24 bits = XOR of non-falsified lit indices
        asg_ptr, active_ptr, conflict_ptr,
        order_ptr, occ_off_ptr, occ_cls_ptr,
        trail_ptr, nrounds_ptr, nops_ptr, nprops_ptr, ndecs_ptr,
        M, N2, P, NSUP, B, TRAIL_CAP, MAX_ROUNDS, seed,
        SOFT: tl.constexpr,
    ):
        """8|24 twin of bcp_triton._bcp_trajectory_packed_kernel — identical trajectory
        semantics (order walk, hash polarity, fsoft risk scan, propagation); only the packed
        word's split differs. Kept as a verbatim copy rather than a constexpr-parametrised
        merge so the proven 16|16 kernel stays untouched."""
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
                                sc0 = tl.load(st_row + c0) & 0xFF
                                r0 += tl.where(sc0 == 1, 1000000,
                                               tl.where(sc0 == 2, 4, tl.where(sc0 == 3, 1, 0)))
                                o0 += 1
                            e1 = tl.load(occ_off_ptr + 2 * v + 2)
                            o1 = e0
                            while o1 < e1:
                                c1 = tl.load(occ_cls_ptr + o1).to(tl.int64)
                                sc1 = tl.load(st_row + c1) & 0xFF
                                r1 += tl.where(sc1 == 1, 1000000,
                                               tl.where(sc1 == 2, 4, tl.where(sc1 == 3, 1, 0)))
                                o1 += 1
                            # ε-greedy (25% hash-forced random) — see the 16|16 kernel's comment.
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
                        st = (tl.load(st_row + c) - 1) ^ (neg << 8)
                        tl.store(st_row + c, st)
                        sat = st & 0xFF
                        if sat == 0:
                            conflict = 1
                        elif sat == 1:
                            forced = (st >> 8) & 0xFFFFFF
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


class TritonBCPWide(TritonBCP):
    """TritonBCP + the 8|24 packed kernel for instances the 16|16 word can't index.

    Dispatch per formula (decided once, in _build_flush_cache):
      2N <= 65535                     -> base class verbatim (16|16 packed; bit-identical).
      65535 < 2N <= 0xFFFFFF, len<=255 -> WIDE mode: 8|24 packed word, same cached-flush
                                          broadcast reset, same trajectory semantics.
      otherwise                        -> unpacked fallback (as base).
    `_force_wide` is a test hook: pack 8|24 even when 16|16 would do (any 16-bit-eligible
    formula is trivially 24-bit-eligible), so A/B equality tests can run on small formulas."""

    def __init__(self):
        super().__init__()
        self._wide_mode = None      # True once the 8|24 kernel is selected for this formula
        self._force_wide = False    # test hook (see class docstring)

    def _build_flush_cache(self, engine) -> None:
        n2 = 2 * engine._num_vars
        M = engine._num_clauses
        max_len = int(engine._formula.clause_sizes.max().item()) if M else 0
        wide_ok = M > 0 and n2 <= 0xFFFFFF and max_len <= 255
        if not (wide_ok and (self._force_wide or n2 > 65535)):
            self._wide_mode = False
            super()._build_flush_cache(engine)      # 16|16 or unpacked, exactly as before
            return
        self._ensure_occ(engine)
        engine.reset_state(1)
        engine.propagate()
        if bool(engine._conflict[0].item()):        # formula units conflict -> UNSAT formula
            self._packed_mode = False
            self._wide_mode = False
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
        xor = np.bitwise_xor.reduceat(vals[order], starts)  # per-clause XOR of non-falsified
        # Pack in uint32 (the XOR may legitimately reach bit 31 after the shift) then reinterpret
        # as the kernel's int32 — torch integer casts would truncate but views are exact.
        packed = ((sat0.astype(np.uint32) & 0xFF)
                  | ((xor.astype(np.uint32) & 0xFFFFFF) << 8)).view(np.int32)
        self._packed_flushed = torch.from_numpy(packed).to(engine._device)
        self._packed_mode = True
        self._wide_mode = True

    def packed_sat_rows(self, row_idx: torch.Tensor) -> torch.Tensor:
        if self._wide_mode:
            return self._packed[row_idx] & 0xFF
        return super().packed_sat_rows(row_idx)

    @torch.inference_mode()
    def run_trajectory(self, engine, order: torch.Tensor, n_support: int, B: int, seed: int,
                       max_rounds: "int | None", soft: bool = False, packed: bool = False) -> int:
        if not (packed and self._wide_mode):
            return super().run_trajectory(engine, order, n_support, B, seed, max_rounds,
                                          soft=soft, packed=packed)
        K = engine._assignment.size(0)
        self._ensure(engine, K, 64)
        if self._nrounds is None or self._nrounds.size(0) != K:
            self._nrounds = torch.empty(K, dtype=torch.int32, device=engine._device)
        order = order.to(torch.int32).contiguous()
        _bcp_trajectory_packed24_kernel[(K,)](
            self._packed, engine._assignment,
            engine._active.view(torch.int8), engine._conflict.view(torch.int8),
            order, self._occ_off32, self._occ_cls32,
            self._trail, self._nrounds, self._nops, self._nprops, self._ndecs,
            engine._num_clauses, 2 * engine._num_vars, order.numel(), n_support, B,
            self._trail.size(1), (max_rounds if max_rounds is not None else 1 << 30), seed,
            SOFT=(1 if soft else 0), num_warps=1,
        )
        self.launches += 1
        engine.total_bcp_rounds += 1
        self._ops_acc += self._nops.sum()
        self._props_acc += self._nprops.sum()
        self._decs_acc += self._ndecs.sum()
        return int(self._nrounds.max().item())


class TritonBCPMono(TritonBCP):
    """TritonBCP + the MONO (count-only, 1-byte) kernel — the delta idea at its floor.

    Dispatch per formula (decided once, on the first reset_fused):
      2N <= 65535 (and not forced)      -> base class verbatim (16|16 packed / unpacked).
      2N > 65535, max clause len <= 127 -> MONO mode: (K, M) int8 state (st = 127 - sat),
                                           constant unit/conflict thresholds, unit literal
                                           recovered by scanning the clause (cls->lits CSR).
      otherwise                         -> unpacked fallback (as base).
    No instance-size ceiling: the state word carries no literal-index bits, so any N works.
    `_force_mono` is the test hook mirroring TritonBCPWide._force_wide."""

    def __init__(self):
        super().__init__()
        self._mono_mode = None      # True once the mono kernel is selected for this formula
        self._force_mono = False    # test hook
        self._mono_st = None        # (K, M) int8 clause state for mono chunks
        self._mono_flushed = None   # (M,) int8 post-units-flush row (127 - sat_flush)
        self._cls_off32 = None      # (M+1,) int32 clause -> literal CSR (shared)
        self._cls_lits32 = None     # (total,) int32

    def _build_flush_cache(self, engine) -> None:
        n2 = 2 * engine._num_vars
        M = engine._num_clauses
        max_len = int(engine._formula.clause_sizes.max().item()) if M else 0
        if not (M > 0 and max_len <= 127 and (self._force_mono or n2 > 65535)):
            self._mono_mode = False
            super()._build_flush_cache(engine)      # 16|16 or unpacked, exactly as before
            return
        self._ensure_occ(engine)
        engine.reset_state(1)
        engine.propagate()
        if bool(engine._conflict[0].item()):        # formula units conflict -> UNSAT formula
            self._packed_mode = False
            self._mono_mode = False
            return
        self._asg_flushed = engine._assignment[0].clone()
        sat0 = engine._sat_count[0].to(torch.int32).cpu().numpy()
        self._mono_flushed = torch.from_numpy((127 - sat0).astype(np.int8)).to(engine._device)
        # clause -> literals CSR, inverted from the occ CSR (occ[l] ∋ c  ⟺  literal l^1 ∈ c).
        off = self._occ_off32.cpu().numpy().astype(np.int64)
        cls = self._occ_cls32.cpu().numpy().astype(np.int64)
        lits = np.repeat(np.arange(n2, dtype=np.int64) ^ 1, off[1:] - off[:-1])
        order = np.argsort(cls, kind="stable")
        counts = np.bincount(cls, minlength=M)
        cls_off = np.zeros(M + 1, dtype=np.int32)
        np.cumsum(counts, out=cls_off[1:])
        self._cls_off32 = torch.from_numpy(cls_off).to(engine._device)
        self._cls_lits32 = torch.from_numpy(lits[order].astype(np.int32)).to(engine._device)
        self._packed_mode = True                    # "fast fused reset available" for the sampler
        self._mono_mode = True

    @torch.inference_mode()
    def reset_fused(self, engine, K: int) -> bool:
        if self._mono_mode is None:
            self._build_flush_cache(engine)
        if not self._mono_mode:
            return super().reset_fused(engine, K)   # 16|16 broadcast or ordinary-reset decline
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
        if self._mono_st is None or self._mono_st.size(0) != K:
            self._mono_st = torch.empty((K, engine._num_clauses), dtype=torch.int8, device=dev)
        self._mono_st.copy_(self._mono_flushed.unsqueeze(0).expand(K, -1))
        return True

    def packed_sat_rows(self, row_idx: torch.Tensor) -> torch.Tensor:
        if self._mono_mode:
            return 127 - self._mono_st[row_idx].to(torch.int16)
        return super().packed_sat_rows(row_idx)

    @torch.inference_mode()
    def run_trajectory(self, engine, order: torch.Tensor, n_support: int, B: int, seed: int,
                       max_rounds: "int | None", soft: bool = False, packed: bool = False) -> int:
        if not (packed and self._mono_mode):
            return super().run_trajectory(engine, order, n_support, B, seed, max_rounds,
                                          soft=soft, packed=packed)
        K = engine._assignment.size(0)
        self._ensure(engine, K, 64)
        if self._nrounds is None or self._nrounds.size(0) != K:
            self._nrounds = torch.empty(K, dtype=torch.int32, device=engine._device)
        order = order.to(torch.int32).contiguous()
        _bcp_trajectory_mono8_kernel[(K,)](
            self._mono_st, engine._assignment,
            engine._active.view(torch.int8), engine._conflict.view(torch.int8),
            order, self._occ_off32, self._occ_cls32,
            self._cls_off32, self._cls_lits32,
            self._trail, self._nrounds, self._nops, self._nprops, self._ndecs,
            engine._num_clauses, 2 * engine._num_vars, order.numel(), n_support, B,
            self._trail.size(1), (max_rounds if max_rounds is not None else 1 << 30), seed,
            SOFT=(1 if soft else 0), num_warps=1,
        )
        self.launches += 1
        engine.total_bcp_rounds += 1
        self._ops_acc += self._nops.sum()
        self._props_acc += self._nprops.sum()
        self._decs_acc += self._ndecs.sum()
        return int(self._nrounds.max().item())
