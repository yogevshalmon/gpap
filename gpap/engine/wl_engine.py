"""EXPERIMENTAL — a classic 2-watched-literal (2-WL) BCP engine, per-warp scalar, for a
head-to-head against the NobleCount sum-trick engine (gpap.engine.bcp_engine).

WHY THIS EXISTS (the experiment): the production engine's novelty is NobleCount — one packed
`sat_count | Σ-surviving-lits` word per clause, updated by a single op whose old value gives
conflict/unit/forced-lit with NO clause scan. The classic CDCL alternative is 2-watched-literals:
visit a clause only when a watched literal is falsified, then SCAN the clause for a replacement
watch. On CPU/CDCL 2-WL is the canonical BCP speedup. The question this module answers
empirically: at the SAME per-row-scalar parallelisation (one Triton program per row, serial
trail — exactly the NobleCount "scalar" kernel), is 2-WL faster or slower on OUR workload
(narrow Tseitin/ABC circuit clauses, forward-only trajectories, no backtracking)?

The engine mirrors BCPEngine's two entry points (apply_and_propagate / run_trajectory) and reuses
the SAME assignment bitmap + splitmix polarity hash, so with the same order/seed/B a 2-WL
trajectory produces BIT-IDENTICAL models to the scalar NobleCount trajectory (confluence) — that
is the correctness gate.

Data structures (contrast with NobleCount's ONE (K,M) word + shared read-only occ CSR):
  * SHARED read-only:
      cl_off (M+1,), cl_lit (T,) int32   — clause -> its literals (NobleCount stores NONE of this)
      wcap_off (2N+1,) int32             — per-literal watch-array capacity CSR (= positive occ)
  * PER-ROW MUTABLE (this is the cost — ~4-5x NobleCount's per-row state):
      wpos  (K, 2M) int32  — node (=2*clause+slot) -> watched position into cl_lit
      warr  (K, T)  int32  — per-literal watch arrays (nodes), CSR-packed by wcap_off
      wcnt  (K, 2N) int32  — current #watchers per literal
      abits (K, NW) int32  — assignment bitmap (bit l = literal l TRUE), same layout as BCPEngine

Requires CUDA + triton. Scalar only — there is deliberately NO wave variant: 2-WL's per-visit
replacement scan + watch-list mutation does not decompose into independent per-occurrence atomics
the way NobleCount's sum-word does, which is precisely the parallel-friendliness point.
"""
import numpy as np
import torch

from gpap.engine.formula import CnfFormula

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:  # noqa: BLE001
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _wl_propagate(cl_off_ptr, cl_lit_ptr, wcap_off_ptr,
                      wpos_row, warr_row, wcnt_row, ab_row, trail_row,
                      head, tail, conflict, nops, TRAIL_CAP):
        """Classic 2-WL propagate, one row, serial trail. Pop TRUE literals; for each, the
        negation `fl` just became FALSE, so walk fl's watch array and, per clause, either keep
        the watch (blocker satisfied), MIGRATE the watch to a new non-false literal (the SCAN),
        or (no replacement) force the other watched literal / conflict.

        `nops` accumulates 2-WL memory touches (one per visit + one per literal scanned) — the
        analogue of NobleCount's occ-entry count, for the work cross-check."""
        while (head < tail) & (conflict == 0):
            lit = tl.load(trail_row + head)
            head += 1
            fl = lit ^ 1                                  # this literal is now FALSE
            off = tl.load(wcap_off_ptr + fl)
            cnt = tl.load(wcnt_row + fl)
            i = fl * 0
            while (i < cnt) & (conflict == 0):
                node = tl.load(warr_row + off + i)
                nops += 1
                c = (node >> 1).to(tl.int64)
                slot = node & 1
                opos = tl.load(wpos_row + 2 * c + (1 - slot))
                olit = tl.load(cl_lit_ptr + opos)
                ow = tl.load(ab_row + (olit >> 5))
                sat = (ow >> (olit & 31)) & 1             # other watch TRUE -> clause satisfied
                moved = node * 0
                if sat == 0:
                    s = tl.load(cl_off_ptr + c)
                    e = tl.load(cl_off_ptr + c + 1)
                    p = s
                    found = s - s - 1                     # -1 sentinel (typed int32)
                    while (p < e) & (found < 0):
                        x = tl.load(cl_lit_ptr + p)
                        nx = x ^ 1
                        xw = tl.load(ab_row + (nx >> 5))
                        xfalse = (xw >> (nx & 31)) & 1
                        if (p != opos) & (xfalse == 0):   # x not falsified and not the other watch
                            found = p
                        nops += 1
                        p += 1
                    if found >= 0:
                        # migrate this node's watch to cl_lit[found] (a different literal)
                        tl.store(wpos_row + node, found)
                        newlit = tl.load(cl_lit_ptr + found)
                        moff = tl.load(wcap_off_ptr + newlit)
                        mcnt = tl.load(wcnt_row + newlit)
                        tl.store(warr_row + moff + mcnt, node)
                        tl.store(wcnt_row + newlit, mcnt + 1)
                        cnt -= 1                            # swap-remove node from fl's array
                        last = tl.load(warr_row + off + cnt)
                        tl.store(warr_row + off + i, last)
                        moved = 1
                    else:
                        nol = olit ^ 1
                        ofw = tl.load(ab_row + (nol >> 5))
                        ofalse = (ofw >> (nol & 31)) & 1
                        if ofalse != 0:
                            conflict = 1                    # other watch already false -> empty
                        else:
                            bit = 1 << (olit & 31)
                            ow2 = tl.load(ab_row + (olit >> 5))
                            if (ow2 & bit) == 0:
                                if (ow2 & (1 << ((olit ^ 1) & 31))) != 0:
                                    conflict = 1
                                elif tail < TRAIL_CAP:
                                    tl.store(ab_row + (olit >> 5), ow2 | bit)
                                    tl.store(trail_row + tail, olit)
                                    tail += 1
                                else:
                                    conflict = 1
                i += 1 - moved
            tl.store(wcnt_row + fl, cnt)
        return head, tail, conflict, nops

    @triton.jit
    def _wl_fixpoint_kernel(
        cl_off_ptr, cl_lit_ptr, wcap_off_ptr,
        wpos_ptr, warr_ptr, wcnt_ptr, ab_ptr,
        active_ptr, conflict_ptr, dec_ptr, trail_ptr,
        nprops_ptr, nops_ptr, ndecs_ptr,
        M2, T, NW, N2, B, TRAIL_CAP,
    ):
        k = tl.program_id(0).to(tl.int64)
        if tl.load(active_ptr + k) != 0:
            wpos_row = wpos_ptr + k * M2
            warr_row = warr_ptr + k * T
            wcnt_row = wcnt_ptr + k * N2
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
            head, tail, conflict, nops = _wl_propagate(
                cl_off_ptr, cl_lit_ptr, wcap_off_ptr,
                wpos_row, warr_row, wcnt_row, ab_row, trail_row,
                head, tail, conflict, nops, TRAIL_CAP)
            if conflict != 0:
                tl.store(conflict_ptr + k, 1)
                tl.store(active_ptr + k, 0)
            tl.store(nprops_ptr + k, tail)
            tl.store(nops_ptr + k, nops)
            tl.store(ndecs_ptr + k, ndecs)

    @triton.jit
    def _wl_trajectory_kernel(
        cl_off_ptr, cl_lit_ptr, wcap_off_ptr,
        wpos_ptr, warr_ptr, wcnt_ptr, ab_ptr,
        active_ptr, conflict_ptr, order_ptr,
        trail_ptr, nrounds_ptr, nops_ptr, nprops_ptr, ndecs_ptr,
        M2, T, NW, N2, P, NSUP, B, TRAIL_CAP, MAX_ROUNDS, seed,
    ):
        """The whole scalar trajectory (decide next B free vars in `order`, splitmix polarity,
        2-WL propagate to fixpoint) in one launch — mirror of the scalar NobleCount trajectory
        kernel with `_propagate` swapped for `_wl_propagate`. Same picks/polarity => bit-identical
        models (the gate)."""
        k = tl.program_id(0).to(tl.int64)
        if tl.load(active_ptr + k) != 0:
            wpos_row = wpos_ptr + k * M2
            warr_row = warr_ptr + k * T
            wcnt_row = wcnt_ptr + k * N2
            ab_row = ab_ptr + k * NW
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
                    wp = tl.load(ab_row + (v >> 4))
                    if ((wp >> ((v & 15) * 2)) & 3) == 0:
                        h = seed + k * -7046029254386353131 + v.to(tl.int64) * -4658895280553007687
                        h = (h ^ (h >> 30)) * -4658895280553007687
                        h = (h ^ (h >> 27)) * -7723592293110705685
                        pol = ((h >> 13) & 1).to(tl.int32)
                        lit = 2 * v + pol
                        tl.store(ab_row + (lit >> 5), wp | (1 << (lit & 31)))
                        tl.store(trail_row + tail, lit)
                        tail += 1
                        picked += 1
                if picked > 0:
                    rounds += 1
                    decs += picked
                head, tail, conflict, nops = _wl_propagate(
                    cl_off_ptr, cl_lit_ptr, wcap_off_ptr,
                    wpos_row, warr_row, wcnt_row, ab_row, trail_row,
                    head, tail, conflict, nops, TRAIL_CAP)
            if conflict != 0:
                tl.store(conflict_ptr + k, 1)
                tl.store(active_ptr + k, 0)
            tl.store(nrounds_ptr + k, rounds)
            tl.store(nops_ptr + k, nops)
            tl.store(nprops_ptr + k, tail)
            tl.store(ndecs_ptr + k, decs)


class WLBCPEngine:
    """2-watched-literal BCP over K rows. Experimental sibling of BCPEngine (same two entry
    points, same bitmap, scalar-only). A DROP-IN for BCPEngine: construct from a CnfFormula alone
    (the clause->literal CSR 2-WL needs is reconstructed from the occ lists), so `GPAP_BCP=wl`
    can swap it into GPUSampler. `clauses_ext` is accepted but unused (back-compat).
    since the formula object stores no clause->literal structure."""

    def __init__(self, formula: CnfFormula, clauses_ext: list | None = None):
        from gpap.engine.kernels import kernel_available
        if formula.device.type != "cuda" or not kernel_available() or not _HAVE_TRITON:
            raise RuntimeError("WLBCPEngine requires CUDA + triton")
        self._formula = formula
        self._device = formula.device
        self._num_vars = formula.num_vars
        self._num_clauses = formula.num_clauses
        N = self._num_vars
        M = self._num_clauses
        n2 = 2 * N
        self._n2 = n2
        self._nw = max(1, (n2 + 31) // 32)

        # clause -> literals CSR (lit indices), RECONSTRUCTED from the formula's occ lists so the
        # engine is a drop-in needing only a CnfFormula (which stores no clause->lit structure —
        # itself a NobleCount economy 2-WL cannot share). occ[l] = clauses where ~l appears, so
        # every (clause c in occ[l]) is an incidence of literal l^1 in c; iterating all l
        # recovers every (clause, literal) incidence exactly. Literal ORDER within a clause
        # differs from the source file => different initial watches => different propagation
        # order, but the same fixpoints (confluence) => identical models.
        occ = formula.occ_list
        occ_off = occ.occ_offsets.cpu().numpy().astype(np.int64)
        occ_cls = occ.occ_clauses.cpu().numpy().astype(np.int64)
        lit_per_inc = np.repeat(np.arange(n2, dtype=np.int64), np.diff(occ_off)) ^ 1
        order_inc = np.argsort(occ_cls, kind="stable")
        lens = np.bincount(occ_cls, minlength=M).astype(np.int64)
        cl_off = np.zeros(M + 1, dtype=np.int64)
        np.cumsum(lens, out=cl_off[1:])
        T = int(cl_off[-1])
        cl_lit = lit_per_inc[order_inc]
        self._T = T
        self._m2 = 2 * M

        # per-literal watch capacity = # clauses containing that literal (positive occ counts)
        cap = np.bincount(cl_lit, minlength=n2).astype(np.int64)
        wcap_off = np.zeros(n2 + 1, dtype=np.int64)
        np.cumsum(cap, out=wcap_off[1:])
        assert int(wcap_off[-1]) == T

        # initial watch state (pre-flush): clause c watches positions cl_off[c] (slot0) and
        # cl_off[c]+1 (slot1); len-1 clauses register only slot0.
        wpos0 = np.zeros(2 * M, dtype=np.int64)
        warr0 = np.zeros(T, dtype=np.int64)          # capacity-slack cells zeroed (hygiene; the
        #                                              algorithm never reads them — counts are exact)
        wcnt0 = np.zeros(n2, dtype=np.int64)
        fill = wcap_off[:-1].copy()                      # running write cursor per literal
        for c in range(M):
            p0 = int(cl_off[c])
            L = int(lens[c])
            wpos0[2 * c] = p0
            l0 = int(cl_lit[p0])
            warr0[fill[l0]] = 2 * c
            fill[l0] += 1
            wcnt0[l0] += 1
            if L > 1:
                p1 = p0 + 1
                wpos0[2 * c + 1] = p1
                l1 = int(cl_lit[p1])
                warr0[fill[l1]] = 2 * c + 1
                fill[l1] += 1
                wcnt0[l1] += 1
            else:
                wpos0[2 * c + 1] = p0                     # unused (node 2c+1 never inserted)

        dev = self._device
        self._cl_off = torch.from_numpy(cl_off.astype(np.int32)).to(dev)
        self._cl_lit = torch.from_numpy(cl_lit.astype(np.int32)).to(dev)
        self._wcap_off = torch.from_numpy(wcap_off.astype(np.int32)).to(dev)
        self._wpos0 = torch.from_numpy(wpos0.astype(np.int32)).to(dev)
        self._warr0 = torch.from_numpy(warr0.astype(np.int32)).to(dev)
        self._wcnt0 = torch.from_numpy(wcnt0.astype(np.int32)).to(dev)
        # clause id per literal incidence — only needed to RECOMPUTE sat_rows (see below)
        self._cl_ids = torch.from_numpy(
            np.repeat(np.arange(M, dtype=np.int64), lens).astype(np.int64)).to(dev)
        self._mode = "scalar"

        # per-row state (sized on first reset)
        self._wpos = self._warr = self._wcnt = self._abits = None
        self._active = self._conflict = None
        self._trail = self._nprops = self._nops = self._ndecs = self._nrounds = None
        self._ops_acc = torch.zeros((), dtype=torch.int64, device=dev)
        self._props_acc = torch.zeros((), dtype=torch.int64, device=dev)
        self._decs_acc = torch.zeros((), dtype=torch.int64, device=dev)
        self.launches = 0
        self._build_flush_cache()

    # ---- per-row bytes (rough): wpos 2M + warr T + wcnt 2N + abits NW, all int32
    def per_row_int32(self) -> int:
        return self._m2 + self._T + self._n2 + self._nw

    def _ensure(self, K, B):
        cap = self._num_vars + max(B, 64) + 8
        if self._trail is None or self._trail.size(0) != K or self._trail.size(1) < cap:
            dev = self._device
            self._trail = torch.empty((K, cap), dtype=torch.int32, device=dev)
            self._nprops = torch.zeros(K, dtype=torch.int32, device=dev)
            self._nops = torch.zeros(K, dtype=torch.int32, device=dev)
            self._ndecs = torch.zeros(K, dtype=torch.int32, device=dev)
            self._nrounds = torch.zeros(K, dtype=torch.int32, device=dev)

    @torch.inference_mode()
    def reset(self, K, _from_init=False):
        dev = self._device
        M2, T, N2 = self._m2, self._T, self._n2
        if self._abits is None or self._abits.size(0) != K:
            self._wpos = torch.empty((K, M2), dtype=torch.int32, device=dev)
            self._warr = torch.empty((K, T), dtype=torch.int32, device=dev)
            self._wcnt = torch.empty((K, N2), dtype=torch.int32, device=dev)
            self._abits = torch.empty((K, self._nw), dtype=torch.int32, device=dev)
            self._active = torch.empty(K, dtype=torch.bool, device=dev)
            self._conflict = torch.empty(K, dtype=torch.bool, device=dev)
        if _from_init:
            self._wpos.copy_(self._wpos0.unsqueeze(0).expand(K, -1))
            self._warr.copy_(self._warr0.unsqueeze(0).expand(K, -1))
            self._wcnt.copy_(self._wcnt0.unsqueeze(0).expand(K, -1))
            self._abits.zero_()
            self._active.fill_(True)
            self._conflict.fill_(False)
            return
        self._wpos.copy_(self._wpos_f.unsqueeze(0).expand(K, -1))
        self._warr.copy_(self._warr_f.unsqueeze(0).expand(K, -1))
        self._wcnt.copy_(self._wcnt_f.unsqueeze(0).expand(K, -1))
        self._abits.copy_(self._abits_f.unsqueeze(0).expand(K, -1))
        self._active.fill_(not self._flush_conflict)
        self._conflict.fill_(self._flush_conflict)

    @torch.inference_mode()
    def _build_flush_cache(self):
        f = self._formula
        self.reset(1, _from_init=True)
        if f.has_empty_clause:
            self._conflict.fill_(True)
            self._active.fill_(False)
        elif self._num_clauses:
            unit_mask = f.clause_sizes == 1
            if bool(unit_mask.any()):
                units = f.init_lit_sum[unit_mask].to(torch.int64).view(1, -1)
                self.apply_and_propagate(units)
        self._wpos_f = self._wpos[0].clone()
        self._warr_f = self._warr[0].clone()
        self._wcnt_f = self._wcnt[0].clone()
        self._abits_f = self._abits[0].clone()
        self._flush_conflict = bool(self._conflict[0].item())
        self._ops_acc.zero_(); self._props_acc.zero_(); self._decs_acc.zero_()
        self.launches = 0

    @torch.inference_mode()
    def apply_and_propagate(self, decisions: torch.Tensor):
        K = self._abits.size(0)
        decisions = decisions.to(self._device)
        if decisions.dim() == 1:
            decisions = decisions.unsqueeze(1)
        B = decisions.size(1)
        if decisions.numel() == 0:
            return
        self._ensure(K, B)
        self._nprops.zero_(); self._nops.zero_(); self._ndecs.zero_()
        dec = decisions.to(torch.int64).contiguous()
        _wl_fixpoint_kernel[(K,)](
            self._cl_off, self._cl_lit, self._wcap_off,
            self._wpos, self._warr, self._wcnt, self._abits,
            self._active.view(torch.int8), self._conflict.view(torch.int8), dec, self._trail,
            self._nprops, self._nops, self._ndecs,
            self._m2, self._T, self._nw, self._n2, B, self._trail.size(1),
            num_warps=1,     # ONE warp/program: the scalar kernel owns a row's private arrays;
            #                  the default 4 warps would run 128 threads redundantly, RACING on
            #                  the per-row watch arrays (torn reads -> garbage node -> OOB).
        )
        self.launches += 1
        self._ops_acc += self._nops.sum()
        self._props_acc += self._nprops.sum()
        self._decs_acc += self._ndecs.sum()

    @torch.inference_mode()
    def run_trajectory(self, order: torch.Tensor, n_support: int, B: int, seed: int,
                       *, soft: bool = False):
        # `soft` accepted for drop-in parity with BCPEngine but IGNORED: 2-WL has no soft-polarity
        # kernel (the whole point — see the findings doc); it always runs the hash-polarity path.
        K = self._abits.size(0)
        self._ensure(K, 64)
        self._nprops.zero_(); self._nops.zero_(); self._ndecs.zero_(); self._nrounds.zero_()
        order = order.to(torch.int32).contiguous()
        _wl_trajectory_kernel[(K,)](
            self._cl_off, self._cl_lit, self._wcap_off,
            self._wpos, self._warr, self._wcnt, self._abits,
            self._active.view(torch.int8), self._conflict.view(torch.int8), order,
            self._trail, self._nrounds, self._nops, self._nprops, self._ndecs,
            self._m2, self._T, self._nw, self._n2, order.numel(), n_support, B,
            self._trail.size(1), 1 << 30, seed,
            num_warps=1,     # ONE warp/program (see apply_and_propagate): avoid redundant-thread
            #                  races on the row-private watch arrays.
        )
        self.launches += 1
        self._ops_acc += self._nops.sum()
        self._props_acc += self._nprops.sum()
        self._decs_acc += self._ndecs.sum()
        return int(self._nrounds.max().item())

    # ---- accessors (validation) ----
    @property
    def active(self):
        return self._active

    @property
    def conflict(self):
        return self._conflict

    @torch.inference_mode()
    def any_free(self, pair_mask: torch.Tensor):
        w = self._abits
        return (((~(w | (w >> 1))) & pair_mask) != 0).any(dim=1)

    @torch.inference_mode()
    def make_pair_mask(self, var_mask):
        vs = (np.arange(self._num_vars, dtype=np.int64) if var_mask is None
              else np.nonzero(var_mask.cpu().numpy())[0])
        bits = np.zeros(self._nw, dtype=np.uint32)
        np.bitwise_or.at(bits, vs >> 4, np.uint32(1) << (2 * (vs & 15)).astype(np.uint32))
        return torch.from_numpy(bits.view(np.int32)).to(self._device)

    @torch.inference_mode()
    def var_true_bits(self, row_idx, var_cols):
        w = self._abits[row_idx][:, (var_cols >> 4).long()]
        return ((w >> ((var_cols & 15).to(torch.int32) * 2)) & 1).to(torch.uint8)

    # ---------------------------------------------------------------- drop-in parity surface
    def set_mode(self, mode: str) -> None:
        """Accepted for BCPEngine parity but 2-WL is SCALAR-ONLY: its per-visit replacement scan
        + watch-list mutation do not decompose into independent per-occurrence atomics, so there
        is no WAVE variant to switch to. A "wave" request is ignored (stays scalar) — that missing
        gear is part of the finding, not a bug."""
        self._mode = "scalar"

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def shift(self) -> int:
        return 0                     # no packed field split (no clause counter at all)

    @property
    def num_vars(self) -> int:
        return self._num_vars

    @property
    def num_clauses(self) -> int:
        return self._num_clauses

    @property
    def device(self):
        return self._device

    @property
    def flush_conflict(self) -> bool:
        return self._flush_conflict

    @property
    def num_propagated(self):
        return self._nprops - self._ndecs

    @torch.inference_mode()
    def recache_flush(self) -> None:
        self._wpos_f = self._wpos[0].clone()
        self._warr_f = self._warr[0].clone()
        self._wcnt_f = self._wcnt[0].clone()
        self._abits_f = self._abits[0].clone()
        self._flush_conflict = bool(self._conflict[0].item())

    @torch.inference_mode()
    def _bits_to_bytes(self, words: torch.Tensor) -> torch.Tensor:
        """(R, NW) bitmap rows -> (R, 2N) int8 (1 = literal TRUE). Contiguous unpack, row-sliced."""
        R = words.size(0)
        shifts = torch.arange(32, dtype=torch.int32, device=words.device)
        out = torch.empty((R, self._nw * 32), dtype=torch.int8, device=words.device)
        step = max(1, int(256e6 // max(self._nw * 32 * 4, 1)))
        for r0 in range(0, R, step):
            w = words[r0:r0 + step]
            out[r0:r0 + step] = ((w.unsqueeze(-1) >> shifts) & 1).to(torch.int8).view(
                w.size(0), -1)
        return out[:, :2 * self._num_vars]

    def assignment_rows(self, row_idx) -> torch.Tensor:
        return self._bits_to_bytes(self._abits[row_idx])

    @torch.inference_mode()
    def sat_rows(self, row_idx) -> torch.Tensor:
        """(R, M) int32 count of NON-FALSIFIED literals per clause — the sampler's GPU soundness
        check. NobleCount gets this for FREE (mask the packed word); 2-WL keeps no clause counter,
        so it must RECOMPUTE from the assignment over every literal incidence. That asymmetry is a
        real structural cost of dropping the counter, and it is on the sampler's timed path."""
        asg = self._bits_to_bytes(self._abits[row_idx])          # (R, 2N) int8, 1 = TRUE
        neg = (self._cl_lit.to(torch.int64) ^ 1)                 # (T,) negation of each incidence
        R = asg.size(0)
        out = torch.zeros((R, self._num_clauses), dtype=torch.int32, device=self._device)
        # literal x is falsified iff ~x is TRUE; non-falsified = 1 - that
        nonf = (1 - asg[:, neg].to(torch.int32))                 # (R, T)
        out.index_add_(1, self._cl_ids, nonf)
        return out

    def ops_total(self):
        return int(self._ops_acc.item())

    def props_total(self):
        """Literals assigned (decisions + implications) over all launches (syncs)."""
        return int(self._props_acc.item())

    def decs_total(self):
        """Decision literals over all launches (syncs). Implications = props_total - decs_total."""
        return int(self._decs_acc.item())
