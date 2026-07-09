"""GPU-parallel SAT solution sampler via exact BCP-completion.

`GPUSampler` draws satisfying assignments by running K independent randomized
DPLL-completions in parallel on the GPU. Each "trajectory" repeatedly:

  1. picks a free (unassigned) variable per sample,
  2. assigns it a *random per-sample polarity* (a decision),
  3. runs BCP to fixpoint (the engine's `apply_assumptions_bulk` + `propagate`).

A trajectory that reaches a conflict is a dead-end (discarded); a trajectory whose
variables all become assigned with no conflict is a **guaranteed-valid model** — every
clause keeps a non-falsified literal, so the assignment satisfies the formula exactly
(no rounding / repair, unlike gradient-relaxation GPU samplers).

This is the "uniform-like, speed-over-uniformity" regime (cf. QuickSampler / CMSGen):
samples are biased by the propagation order, but produced by the millions and every one
is valid. The sampler is a *throughput* primitive — its fit depends on how shallow the
decision depth is (heavily-propagating structured formulas → few decisions → few rounds).

Design knobs (the throughput vs. success-rate tradeoff):
  decisions_per_round : assign B free vars at once. B=1 = textbook DPLL (most rounds,
                        highest success rate); B>1 = a random restriction (fewer rounds,
                        more dead-ends).
  random_var          : pick a random free var per sample (diverse) vs. the first free
                        one (cheap, low-index-biased).

K is auto-sized to the GPU (capped by default to stay off the WSL2 host-RAM spill cliff).
"""
import os
# Use CUDA expandable-segments allocation BEFORE torch initializes its CUDA allocator. The sampler's
# hot loop churns many VARIABLE-sized temporaries per round (propagate scratch sized by the frontier;
# the (K,N) decision masks), which fragments the default caching allocator badly: measured peak
# RESERVED = 6.5 GB while live tensors were only 2.5 GB (62% wasted to fragmentation), and that
# fragmentation is what "grows mid-run" and forces a small K. expandable_segments grows one virtual
# segment instead of many fixed blocks → reserved tracks allocated (6.5→2.8 GB here, ~2.3× more K
# for the same VRAM). setdefault so an explicit user PYTORCH_CUDA_ALLOC_CONF still wins. Must be set
# before the first CUDA allocation; importing this module before any CUDA use guarantees that.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import time

import torch

from gpap.engine.formula import CnfFormula
from gpap.engine.bcp_engine import BCPEngine
from gpap.engine.calibrate import compute_k_safe, gpu_mem_available


class GPUSampler:
    """Batched GPU sampler: K parallel randomized BCP-completions per chunk."""

    def __init__(
        self,
        formula: CnfFormula,
        device: torch.device,
        *,
        k_cap: int | None = 16384,
        seed: int | None = None,
        use_kernel: "bool | str" = "auto",
    ):
        self.formula = formula
        self.device = device
        self.N = formula.num_vars
        self.M = formula.num_clauses
        self.engine = BCPEngine(formula)
        # Fused Triton apply+propagate kernel (one launch to fixpoint — kills the Python BCP round
        # loop's launch/sync/unique overhead + its temporary churn; see gpap.engine.kernels).
        # "auto" (default) = on when CUDA + triton compile are available; the PyTorch engine remains
        # the fallback/oracle. TritonBCP now picks the packed field split PER FORMULA in
        # _build_flush_cache (16|16 for 2N<=65535, 8|24 "wide" up to 2N<=0xFFFFFF, else unpacked), so
        # a single class covers every instance — the old "wide"/"mono" opt-in flags are accepted as
        # legacy aliases for "auto" (mono lost to wide, measured 0.93x — see docs/findings/INSIGHTS).
        if use_kernel in ("auto", "wide", "mono", True):
            from gpap.engine.kernels import TritonBCP
            use_kernel = device.type == "cuda" and self.M > 0 and TritonBCP.available()
            if use_kernel:
                self.engine.attach_kernel(TritonBCP())
        self.use_kernel = bool(use_kernel)
        self._max_cls = int(formula.clause_sizes.max().item()) if self.M else 1
        self._total_lits = int(formula.clause_sizes.sum().item()) if self.M else 0
        # RNG + state must exist before K calibration (the probe chunk in _calibrate_k uses _gen).
        self._gen = torch.Generator(device=device)
        if seed is not None:
            self._gen.manual_seed(seed)
        import random as _random
        self._pyrng = _random.Random(seed)
        self.support: list[int] | None = None   # 0-based independent-support vars (from `c ind`)
        self._recon: dict | None = None          # set when built with simplify=True (see _from_simplified)
        # K auto-sized to the GPU (see _calibrate_k): the old analytic floor 0.7·free/(70M+40N) was
        # ~4-6x too conservative because allocator FRAGMENTATION (not tensor size) inflated reserved
        # memory; enabling expandable_segments (at import) removes that, so K = 4× the floor is safe.
        free_bytes = gpu_mem_available(device)
        analytic = compute_k_safe(
            self.M, self.N, free_bytes, cap=k_cap,
            max_clause_size=self._max_cls, total_lits=self._total_lits or None,
            deduct_formula=False,
        )
        self.K = self._calibrate_k(free_bytes, k_cap, analytic)
        self._K_init = self.K
        # FUSED-config K boost: a fused-trajectory chunk never builds the (K,N) per-round decision
        # masks — its true footprint is just the engine state + the trail + a few (K,N) bool readout
        # temps: ≈ (sat_bytes+4)·M + 10·N per batch. That is far below the v1 heuristic, so fused
        # chunks can run a much larger K (throughput ∝ K). Applied per-chunk in sample() ONLY for
        # fused configs and only while K is at its construction value (a lowered K — e.g. the
        # coverage run — is honoured). Hard cap 262,144; validated under the nvidia-smi kill-switch.
        if self.use_kernel and self.M > 0:
            sat_b = 1 if self._max_cls <= 127 else (2 if self._max_cls <= 32767 else 4)
            # Hard ceiling on the fused-K boost. 262,144 was kill-switch-validated on the 12 GB 4080;
            # on a big-VRAM GPU (A100 80 GB) the caller's VRAM-proportional k_cap is larger, so let it
            # raise the ceiling — the memory-safe `budget // fused_pb` below still bounds the actual K.
            # No-op on the 4080 (its k_cap < 262,144); unlocks A100 fused headroom for the K-scaling story.
            fused_cap = max(262_144, k_cap or 0)
            n2 = 2 * self.N
            packed = n2 <= 65535 or (n2 <= 0xFFFFFF and self._max_cls <= 255)
            # Per-batch fused footprint: PACKED = ONE (K,M) int32 clause word; UNPACKED = the engine's
            # (sat_b+4)·M sat/lsum. Either way the (K,N) per-round decision masks a v1 chunk builds are
            # ABSENT in the fused path, so only ~10N of assignment + trail + readout temps remain.
            fused_pb = (4 * self.M + 300 if packed else (sat_b + 4) * self.M + 64) + 10 * self.N
            # eng_hold = the engine's (K,M) sat/lsum + (K,2N) assignment held at BASE K by a
            # NON-fused config (PACKED only — the unpacked fused path reuses those very tensors, so
            # its fused_pb already counts them). We deduct it UNCONDITIONALLY: even an all-fused
            # portfolio needs comparable headroom for the fixed 4096×M soundness-check temporary
            # (skipping it spilled b05-c50 under the nvidia-smi kill-switch).
            eng_hold = (((sat_b + 4) * self.M + 2 * self.N) * self.K) if packed else 0
            budget = max(0, int(0.6 * (free_bytes - eng_hold)))
            self.K_fused = max(self.K, min(budget // fused_pb, fused_cap))
        else:
            self.K_fused = self.K

    # K-calibration: with CUDA expandable_segments (set at module import) the sampler's per-round
    # variable-sized churn no longer fragments the allocator — reserved memory drops ~2.3× AND the
    # old super-linear (≈K^2.3) growth that spilled dense instances becomes LINEAR in K (per-batch
    # stable across K). That headroom lets K safely exceed the old proven-safe floor by this factor;
    # 4× is validated to stay well under budget for the densest measured instance (s1196: 4×floor →
    # 5.9 GB of the 8 GB safe budget, stable across chunks, no spill), with margin for denser unknowns.
    _CAL_MULT = 4

    def _calibrate_k(self, free_bytes: int, k_cap: int | None, analytic_k: int) -> int:
        """Size K to the spill-safe VRAM budget. The old analytic floor (0.7·free/(70M+40N)) was
        calibrated to never spill under the DEFAULT allocator; enabling `expandable_segments` (at
        import) both cuts reserved memory ~2.3× and linearises it in K, so K can safely be a fixed
        multiple (_CAL_MULT) of that floor — no fragile per-instance probe.

        A runtime probe was tried and abandoned: with expandable_segments the peak-reserved read at
        construction is systematically ~1.6× below production (allocation-history dependent) and the
        frontier peak varies with the RNG draw, so every probe/differencing variant under-measured
        per-batch and over-sized K → spill. The fixed 4×-floor multiple is measured-safe and robust.
        CPU / M==0: analytic only (no CUDA budget to exploit)."""
        cap = k_cap if k_cap else analytic_k
        safe_floor = max(1, int(0.7 * free_bytes / (70 * self.M + 40 * self.N + 64)))
        if self.device.type != "cuda" or self.M == 0:
            return max(1, min(analytic_k, cap))
        return max(1, min(self._CAL_MULT * safe_floor, analytic_k, cap))

    @classmethod
    def from_cnf_file(cls, path: str, device: torch.device, *, simplify: bool = False,
                      els: bool = False, **kw) -> "GPUSampler":
        """Build a sampler from a DIMACS file. `simplify=True` first applies unit-propagation
        simplification (shrinks M -> larger K -> more throughput) and samples the compacted
        formula, reconstructing full original models at collection time (see simplify_cnf).
        `els=True` additionally does equivalent-literal substitution (simplify_cnf_els): removes
        more variables/clauses -> faster propagate + bigger K (implies simplify)."""
        support = parse_independent_support(path)   # 0-based, or None
        if not (simplify or els):
            formula = CnfFormula(device)
            formula.parse_cnf_file(path)
            smp = cls(formula, device, **kw)
            smp.support = support
            return smp
        from pysat.formula import CNF
        cnf = CNF(from_file=path)
        return cls._from_simplified(cnf.nv, cnf.clauses, support, device, els=els, **kw)

    @classmethod
    def from_clauses(cls, num_vars: int, clauses, device: torch.device, *,
                     simplify: bool = False, els: bool = False,
                     support: list | None = None, **kw) -> "GPUSampler":
        if simplify or els:
            return cls._from_simplified(num_vars, clauses, support, device, els=els, **kw)
        formula = CnfFormula(device)
        formula.init_formula(num_vars, clauses)
        smp = cls(formula, device, **kw)
        smp.support = support
        return smp

    @classmethod
    def _from_simplified(cls, nv: int, clauses: list, support: list | None,
                         device: torch.device, *, els: bool = False, **kw) -> "GPUSampler":
        # `clauses` is treated read-only here and by simplify_cnf/simplify_cnf_els (they copy
        # before mutating) — no defensive deep copy: it costs ~1s per call on 1M-clause unrolls.
        if els:
            res = simplify_cnf_els(nv, clauses, support)
            if res is None:
                return cls._unsat_sampler(device, **kw)
            nv2, cl2, sup2, recon = res
            formula = CnfFormula(device)
            formula.init_formula(nv2, cl2)
            smp = cls(formula, device, **kw)
            smp.support = sup2
            smp._recon = {"unsat": False, "els": True, **recon}
            return smp
        res = simplify_cnf(nv, clauses, support)
        if res is None:                                  # UNSAT after unit propagation
            return cls._unsat_sampler(device, **kw)
        nv2, cl2, sup2, old_of_new, forced, free_support = res
        formula = CnfFormula(device)
        formula.init_formula(nv2, cl2)
        smp = cls(formula, device, **kw)
        smp.support = sup2
        smp._recon = {"unsat": False, "old_of_new": old_of_new, "forced": forced,
                      "free_support": free_support, "orig_nv": nv,
                      "orig_support": sorted(support) if support else []}
        return smp

    @classmethod
    def _unsat_sampler(cls, device: torch.device, **kw) -> "GPUSampler":
        formula = CnfFormula(device)
        formula.init_formula(1, [[1], [-1]])
        smp = cls(formula, device, **kw)
        smp.support = []
        smp._recon = {"unsat": True}
        return smp

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    # ----------------------------------------------------------------- one chunk

    @torch.inference_mode()
    def _free_mask(self) -> torch.Tensor:
        """(K, N) bool — variable v free iff neither polarity is assigned TRUE."""
        asg = self.engine.assignment            # (K, 2N) int8, 1 = TRUE
        return (asg[:, 0::2] == 0) & (asg[:, 1::2] == 0)

    @torch.inference_mode()
    def _immediate_conflict(self, rows: torch.Tensor, lits: torch.Tensor) -> torch.Tensor:
        """1-step failed-literal test: (P,) bool, True iff setting `lits[i]` TRUE immediately
        falsifies a clause — i.e. some clause in occ[lit] currently has sat_count==1 (its lone
        non-falsified literal is ~lit, so the decrement empties it). Uses the user's insight that
        a clause is conflicted exactly when sat_count hits 0. Catches direct clause-emptying, not
        deeper propagation cascades (a cheap, sound under-approximation of failed literals)."""
        occ = self.formula.occ_list
        dev = self.device
        P = rows.numel()
        if P == 0:
            return torch.zeros(0, dtype=torch.bool, device=dev)
        starts = occ.occ_offsets[lits]
        lengths = occ.occ_offsets[lits + 1] - starts
        total = int(lengths.sum().item())
        if total == 0:
            return torch.zeros(P, dtype=torch.bool, device=dev)
        cum = torch.zeros(P, dtype=torch.long, device=dev)
        if P > 1:
            cum[1:] = lengths[:-1].cumsum(0)
        flat = torch.arange(total, dtype=torch.long, device=dev)
        csi = starts.repeat_interleave(lengths) + (flat - cum.repeat_interleave(lengths))
        clauses = occ.occ_clauses[csi]
        exp_rows = rows.repeat_interleave(lengths)
        sc = self.engine.sat_count.view(-1)[exp_rows * self.M + clauses].to(torch.long)
        pair = torch.arange(P, dtype=torch.long, device=dev).repeat_interleave(lengths)
        seg_min = torch.full((P,), 1 << 30, dtype=torch.long, device=dev)
        seg_min.scatter_reduce_(0, pair, sc, reduce="amin", include_self=True)
        return seg_min <= 1

    @torch.inference_mode()
    def _polarity_risk(self, lits: torch.Tensor) -> torch.Tensor:
        """Cheap GPAP-native conflict-risk score for setting each lit TRUE (one occ-gather, no
        snapshot/propagation). Setting `lit` decrements every clause where ¬lit appears (occ[lit]);
        sat_count==1 → the clause empties → conflict; ==2 → it becomes a forced UNIT; ==3 → tightens.
        risk = Σ w(sat_count) over occ[lit], w(1)=1e6, w(2)=4, w(3)=1, else 0.

        ⚠️  RETAINED ONLY AS A NEGATIVE RESULT — do not use. The framing is WRONG for our regime:
        sat_count is the NON-FALSIFIED-literal count, not a satisfaction/quality signal, and the only
        truly bad transition is 1→0 (a real conflict). A clause going 2→1 (w=4 here) becomes a unit,
        i.e. it PROPAGATES — and in our niche (freeAftSup≈0) propagation is exactly how a trajectory
        completes into a model. So this score penalizes the engine's own success mechanism and
        collapses diversity (measured: 27.sk 49%→0% success). Correcting it to weight ONLY 1→0
        reduces to `_immediate_conflict` (a sound 1-step test) — which also doesn't help, because the
        conflicts that kill the low-supOK family are DEEP cascades no instantaneous counter can see.
        Net: there is no cheap 1-step proxy for "does this decision eventually conflict"; that lives
        in the propagation closure (only `lookahead="full"` answers it, at net-negative cost).
        `lits`: (P,) lit indices; the row for lits[i] is i (aligned with arange(P))."""
        occ = self.formula.occ_list
        dev = self.device
        P = lits.numel()
        if P == 0:
            return torch.zeros(0, device=dev)
        starts = occ.occ_offsets[lits]
        lengths = occ.occ_offsets[lits + 1] - starts
        total = int(lengths.sum().item())
        if total == 0:
            return torch.zeros(P, device=dev)
        cum = torch.zeros(P, dtype=torch.long, device=dev)
        if P > 1:
            cum[1:] = lengths[:-1].cumsum(0)
        flat = torch.arange(total, dtype=torch.long, device=dev)
        csi = starts.repeat_interleave(lengths) + (flat - cum.repeat_interleave(lengths))
        clauses = occ.occ_clauses[csi]
        rows = torch.arange(P, dtype=torch.long, device=dev).repeat_interleave(lengths)
        sc = self.engine.sat_count.view(-1)[rows * self.M + clauses].to(torch.long)
        w = torch.zeros(total, device=dev)
        w = torch.where(sc == 1, w.new_tensor(1e6), w)
        w = torch.where(sc == 2, w.new_tensor(4.0), w)
        w = torch.where(sc == 3, w.new_tensor(1.0), w)
        risk = torch.zeros(P, device=dev)
        risk.scatter_add_(0, rows, w)
        return risk

    @torch.inference_mode()
    def _pick_decisions(
        self, free: torch.Tensor, need: torch.Tensor, B: int, priority: torch.Tensor,
        soft_la: bool = False,
    ) -> torch.Tensor:
        """Build a (K, B) int64 decision tensor (lit indices, -1 = no decision).

        Per row needing a decision, choose up to B free variables — the ones with the
        smallest `priority` rank (priority is a per-chunk (N,) vector: a random permutation
        for diverse var order, or arange for deterministic). Polarity is random per pick, so
        samples diverge even under a shared variable order. Rows not in `need`, and picks with
        no remaining free var, are -1 (skipped by apply_assumptions_bulk).

        `soft_la` (B=1): pick the *lower-conflict-risk* polarity via `_polarity_risk` (random
        tie-break) instead of a uniform-random polarity. Costs two cheap occ-gathers per round —
        no snapshot, no extra propagation, no K reduction — and biases away from cascade-prone
        polarities to raise feasibility (success) while preserving sample diversity via the
        tie-break and the random variable order.
        """
        BIG = priority.numel() + 1
        masked = torch.where(free, priority.unsqueeze(0), priority.new_full((), BIG))  # (K,N)
        if soft_la and B == 1:
            var = masked.argmin(dim=1)                       # (K,) min-rank free var per row
            out = torch.full((self.engine.sat_count.size(0), 1), -1, dtype=torch.long, device=self.device)
            rows = need.nonzero(as_tuple=False).view(-1)
            if rows.numel():
                v = var[rows]
                risk_t = self._polarity_risk(2 * v)          # risk of setting v TRUE
                risk_f = self._polarity_risk(2 * v + 1)      # risk of setting v FALSE
                randp = torch.randint(0, 2, (rows.numel(),), generator=self._gen, device=self.device)
                pol = torch.where(risk_t < risk_f, torch.zeros_like(randp),
                                  torch.where(risk_f < risk_t, torch.ones_like(randp), randp))
                out[rows, 0] = 2 * v + pol
            return out
        if B == 1:
            score, var = masked.min(dim=1, keepdim=True)    # (K,1) smallest rank among free
        else:
            # clamp to N: a tiny (post-simplify) formula can have fewer columns than B
            score, var = masked.topk(min(B, masked.size(1)), dim=1, largest=False)  # (K,min(B,N))
        pol = torch.randint(0, 2, var.shape, generator=self._gen, device=self.device)
        lit = 2 * var + pol                                 # (K,B)
        valid = (score < BIG) & need.unsqueeze(1)           # picked a real free var, row needs it
        return torch.where(valid, lit, torch.full_like(lit, -1))

    @torch.inference_mode()
    def _decide_phase(self, B, priority, mask, max_rounds, rounds, lookahead=False):
        """Keep deciding free vars in `mask` (None = all free) until none remain for any active
        row (or max_rounds hit). Returns updated `rounds`.

        `lookahead` (B=1): FULL-propagation feasibility lookahead — pick one var per row, snapshot
        the engine state, try BOTH polarities (apply + propagate to fixpoint, record which conflict),
        restore, then commit the non-conflicting polarity when exactly one conflicts (else random).
        This emulates the conflict avoidance CMSGen gets from CDCL backtracking and dramatically
        raises feasibility (supOK) on formulas whose support self-conflicts via deep cascades — the
        1-step test misses those. Cost: ~3× propagations + a state snapshot per round (so sample()
        halves K under lookahead to fit the snapshot)."""
        eng = self.engine
        K = eng.assignment.size(0)
        BIG = priority.numel() + 1
        neg1 = torch.tensor(-1, device=self.device)
        while True:
            free = self._free_mask()
            dfree = free & mask if mask is not None else free
            need = eng.active & dfree.any(dim=1)
            if not bool(need.any()):
                break
            if max_rounds is not None and rounds >= max_rounds:
                break
            rounds += 1
            if lookahead in (True, "full"):
                pass  # fall through to the snapshot path below
            else:
                # off, cheap "soft" risk-polarity (handled inside _pick_decisions), or "orac"
                # (force the PyTorch engine for this config — the oracle's batched tensor ops beat
                # the per-warp kernel on ultra-shallow wide instances, e.g. b14 T=1; bandit-selected)
                decisions = self._pick_decisions(dfree, need, B, priority,
                                                 soft_la=(lookahead == "soft"))
                if lookahead == "orac":
                    eng.apply_assumptions_bulk(decisions)
                    eng.propagate()
                else:
                    eng.apply_and_propagate(decisions)
                continue
            # --- full-propagation lookahead (B=1) ---
            masked = torch.where(dfree, priority.unsqueeze(0), priority.new_full((), BIG))
            var = masked.argmin(dim=1)                              # (K,) min-rank free var
            dec_t = torch.where(need, 2 * var, neg1).unsqueeze(1)   # try var TRUE
            dec_f = torch.where(need, 2 * var + 1, neg1).unsqueeze(1)  # try var FALSE
            # snapshot (frontier is empty at fixpoint, so it needs no save/restore)
            sc, ls = eng.sat_count.clone(), eng.lit_sum.clone()
            asg, act, conf = eng.assignment.clone(), eng.active.clone(), eng.conflict.clone()

            def _restore():
                eng.sat_count.copy_(sc); eng.lit_sum.copy_(ls); eng.assignment.copy_(asg)
                eng.active.copy_(act); eng.conflict.copy_(conf)

            eng.apply_and_propagate(dec_t)
            bad_t = eng.conflict & ~conf                           # TRUE newly conflicts
            _restore()
            eng.apply_and_propagate(dec_f)
            bad_f = eng.conflict & ~conf                           # FALSE newly conflicts
            _restore()
            pol = torch.randint(0, 2, (K,), generator=self._gen, device=self.device)
            pol = torch.where(bad_t & ~bad_f, torch.ones_like(pol), pol)   # TRUE bad → FALSE
            pol = torch.where(bad_f & ~bad_t, torch.zeros_like(pol), pol)  # FALSE bad → TRUE
            dec = torch.where(need, 2 * var + pol, neg1).unsqueeze(1)
            eng.apply_and_propagate(dec)
        return rounds

    @torch.inference_mode()
    def _run_chunk(self, K: int, B: int, random_var: bool, max_rounds: int | None,
                   support_mask: torch.Tensor | None, complete: bool, lookahead: bool = False):
        """One chunk of K trajectories. Phase 1 decides the support (when `support_mask` given);
        if `complete`, phase 2 then decides the REMAINING free vars (the completion phase) so a
        full sound model is reached. Without `complete`, only the support is fixed.

        Returns (fully_assigned, support_complete, rounds, free_after_support):
          fully_assigned     : active AND no free var → a guaranteed full model.
          support_complete   : active AND no free *support* var → support projection fixed.
          free_after_support : mean # free vars left once the support is fixed (how much work
                               the completion phase faces; ~0 means support nearly determines all).
        """
        eng = self.engine
        fused_c = lookahead in ("fused", "fsoft") and eng.has_kernel
        # PACKED fused chunks (packed sat|xor state, see gpap.engine.kernels) skip reset_state+
        # propagate entirely: the post-units-flush state is row-identical every chunk, so
        # reset_fused broadcast-restores a cached row — no (K,M) sat/lsum writes, no per-chunk
        # Python-round units flush. Falls back (returns False) when ineligible: ordinary path.
        self._last_chunk_packed = bool(fused_c and eng.reset_fused(K))
        if not self._last_chunk_packed:
            eng.reset_state(K)
            eng.propagate()                                 # process the formula's own units
        if random_var:
            priority = torch.randperm(self.N, generator=self._gen, device=self.device)
        else:
            priority = torch.arange(self.N, device=self.device)

        if fused_c:
            # v2 fused-trajectory kernel: the WHOLE decide+propagate loop in ONE launch. Picking the
            # next-B-free-vars in priority order inside the kernel is exactly _pick_decisions'
            # semantics; polarity is a per-(chunk,k,v) hash bit. Decision order = support vars first
            # (phase 1), then the completion vars (phase 2) — same two-phase behaviour.
            rank = torch.argsort(priority)                  # vars in priority order
            if support_mask is not None:
                in_sup = support_mask[rank]
                sup_part = rank[in_sup]
                order = torch.cat([sup_part, rank[~in_sup]]) if complete else sup_part
                n_support = int(sup_part.numel())
            else:
                order = rank
                n_support = order.numel()                   # single phase: boundary = end
            seed = int(torch.randint(0, 2 ** 62, (1,), generator=self._gen,
                                     device=self.device).item())
            # No round cap for fused: the kernel structurally terminates in ≤ |order| decision groups
            # (each group either decides ≥1 var or exhausts the order). max_decision_rounds exists to
            # bound the v1 PYTHON loop; applying it here cripples B=1 on large-support instances
            # (measured: s832 T=25 support=450 → the 300 cap truncated every trajectory, 2× loss).
            rounds = eng.run_trajectory(order, n_support, B, seed, None,
                                        soft=(lookahead == "fsoft"),
                                        packed=self._last_chunk_packed)
            fr = self._free_mask()
            if support_mask is not None:
                support_complete = eng.active & ~(fr & support_mask).any(dim=1)
            else:
                support_complete = eng.active
            fully = eng.active & ~fr.any(dim=1)
            # free-after-support is unmeasurable mid-trajectory in the fused path (diagnostic only)
            return fully, support_complete, rounds, 0.0

        # Phase 1: fix the support (or, if no support, this decides everything).
        rounds = self._decide_phase(B, priority, support_mask, max_rounds, 0, lookahead)

        # diagnostic: free vars remaining once support is fixed (support-complete active rows)
        fr = self._free_mask()
        if support_mask is not None:
            support_complete = eng.active & ~(fr & support_mask).any(dim=1)
        else:
            support_complete = eng.active
        sc_rows = support_complete
        free_after_support = (fr.sum(dim=1).float()[sc_rows].mean().item()
                              if bool(sc_rows.any()) else 0.0)

        # Phase 2: completion — decide all remaining free vars to reach full models.
        if complete and support_mask is not None:
            rounds = self._decide_phase(B, priority, None, max_rounds, rounds, lookahead)

        free = self._free_mask()
        fully = eng.active & ~free.any(dim=1)
        return fully, support_complete, rounds, free_after_support

    @torch.inference_mode()
    def _autotune(self, K, B_hint, max_rounds, support_mask, complete, lookahead, probe_k=2048):
        """Probe a small (random_var, B) grid and return the (rv, B) with the best THROUGHPUT
        (models / wall) on this formula. Both are strong, formula-dependent levers (polarity stays
        random throughout, so diversity / 2-wise coverage are preserved — validated == CMSGen):
          - niche (support BCP-determines the rest): B=8, order irrelevant — fast, high success.
          - "service" family (search-heavy residual, freeAft≈685): random_var=False (structured order
            follows the gate DAG) + B=1 (avoid over-committing) is the ONLY combo that completes them
            (ActivityService modelOK 0.05%→12%); B≥4 OR random order → ≈0.
          - low-supOK family (7/80.sk): random_var=True.
        Selecting by throughput (not raw count) never trades a fast high-success combo for a slow one:
        a B=1 run that collects more models per chunk but takes 60 rounds loses to a 3-round B=8 run.
        Niche fast-path: if (random, B_hint) already succeeds well, return immediately so the niche
        pays one cheap probe; only the lose-cases explore the full grid (≤6 chunks)."""
        pk = max(1, min(K, probe_k))

        def probe(rv, B):
            self._sync(); t0 = time.perf_counter()
            fully, _, _, _ = self._run_chunk(pk, B, rv, max_rounds, support_mask, complete, lookahead)
            n = int(fully.sum().item()); self._sync()
            return n, time.perf_counter() - t0

        n0, dt0 = probe(True, B_hint)
        if n0 / pk > 0.1:                       # workable at (random, B_hint) — keep it; B_hint is fast.
            # (Threshold is deliberately low: a small-K probe under-rates large B — fewer rounds means
            #  kernel-launch overhead dominates its wall — so the grid over-favors B=1. If random+B_hint
            #  already produces at a usable rate, don't risk that misjudgment; only truly-stuck formulas
            #  (service family ≈0 here) fall through to the full (order,B) grid.)
            return True, B_hint
        best = (n0 / dt0 if dt0 > 0 else 0.0, True, B_hint)
        for rv in (True, False):
            for B in sorted({1, 4, B_hint}):
                if rv is True and B == B_hint:  # already probed above
                    continue
                n, dt = probe(rv, B)
                rate = n / dt if dt > 0 else 0.0
                if rate > best[0]:
                    best = (rate, rv, B)
        return best[1], best[2]

    def _select_config(self, plan, stats, chunk, exploit=False):
        """Adaptive (bandit) config selection: dynamically switch to the best-THROUGHPUT config as it
        is measured on real full-K chunks. More robust than the small-K `_autotune` probe (which
        under-rates large B because kernel-launch overhead dominates its few rounds, and commits
        permanently): here every measurement is a productive full-K chunk and the choice can change.
        Policy: try each config once, then decaying-epsilon-greedy (explore shrinks as 1/sqrt(chunk),
        floored at 5%) on cumulative models/wall — keeps re-checking alternatives but spends most of
        the budget on the winner (niche → (random,8); service → (structured,1)).
        Exploration is PLAUSIBILITY-FILTERED: only configs within 20x of the best measured rate get
        re-checked. Uniform re-checks ignored what a re-check COSTS — measured on b10 T=25 (64-chunk
        collect run): orac at 102 models/s vs the winner's 29.6k (290x) got 5 uniform picks = 20.4s
        of the 29s wall (70%). Run-to-run rate noise is ~1.9x (b06 measurements), so a 20x-slower
        config is dead on this formula, not noisy — it keeps exactly its one cold-start chunk.
        `exploit=True` skips exploration and returns the current best — used in the post-target
        top-up phase to accumulate a stable measurement window on the winner (min_measure_chunks)."""
        rate = lambda c: (stats[c]["models"] / stats[c]["wall"]) if stats[c]["wall"] > 0 else 0.0
        best = lambda: max(plan, key=rate)
        for c in plan:                                  # cold start: one real chunk per config
            if stats[c]["chunks"] == 0:
                return c
        if exploit:
            return best()
        eps = max(0.05, chunk ** -0.5)
        if self._pyrng.random() < eps:
            b = best()
            cands = [c for c in plan if c is not b and rate(c) >= rate(b) / 20]
            if cands:
                return self._pyrng.choice(cands)
        return best()

    def _reconstruct_els(self, comp_rows: list[list[int]]):
        """Reconstruct original models when the formula was reduced by ELS (simplify_cnf_els).
        Every original var resolves to its representative's value via `rep_of`: forced vars take their
        pinned value, decided/propagated reps take the sampled value, and free (don't-care) reps draw
        ONE random bit per equivalence class (so equivalent vars stay equal — a hard requirement, else
        the projection is invalid). Returns (original_models, support_projections)."""
        r = self._recon
        rep_of = r["rep_of"]                        # rep_of[v-1] = (rep_var, same_polarity)
        new_of_new = r["new_of_new"]                # rep_var -> compacted var (1-based)
        forced_map = r["forced_map"]                # var -> bool
        orig_nv = r["orig_nv"]
        orig_support = r["orig_support"]
        out_models, projs = [], []
        for cm in comp_rows:
            model_dict = {abs(cl): (cl > 0) for cl in cm}   # compacted var -> bool
            free_bits: dict[int, bool] = {}
            model = []
            for v in range(1, orig_nv + 1):
                if v in forced_map:
                    b = forced_map[v]
                else:
                    rv, pol = rep_of[v - 1]
                    if rv in forced_map:
                        rval = forced_map[rv]
                    elif rv in new_of_new:
                        rval = model_dict.get(new_of_new[rv], False)
                    else:                                    # don't-care rep: one bit per class
                        rval = free_bits.get(rv)
                        if rval is None:
                            rval = self._pyrng.random() < 0.5
                            free_bits[rv] = rval
                    b = rval if pol else (not rval)
                model.append(v if b else -v)
            out_models.append(model)
            sset = set(model)
            projs.append(tuple((v + 1) if (v + 1) in sset else -(v + 1) for v in orig_support))
        return out_models, projs

    def _reconstruct(self, comp_rows: list[list[int]]):
        """Map full models of the simplified (compacted) formula back to full ORIGINAL models, and
        return (original_models, support_projections). Forced vars are pinned to their unit-propagated
        value; the compacted vars take their sampled value; don't-care support vars (free_support) are
        sampled uniformly at random (so the support-projection distribution is preserved). Remaining
        unmapped original vars are true don't-cares set to False (any value gives a valid model)."""
        if self._recon.get("els"):
            return self._reconstruct_els(comp_rows)
        r = self._recon
        old_of_new = r["old_of_new"]              # old_of_new[i] = original var of compacted var i+1
        forced_map = {abs(l): (l > 0) for l in r["forced"]}
        free_support = r["free_support"]
        orig_nv = r["orig_nv"]
        orig_support = r["orig_support"]
        out_models, projs = [], []
        for cm in comp_rows:
            val = dict(forced_map)                # original var -> bool
            for cl in cm:
                val[old_of_new[abs(cl) - 1]] = cl > 0
            for v0 in free_support:
                val[v0 + 1] = self._pyrng.random() < 0.5
            model = [(v if val.get(v, False) else -v) for v in range(1, orig_nv + 1)]
            out_models.append(model)
            sset = set(model)
            projs.append(tuple((v + 1) if (v + 1) in sset else -(v + 1) for v in orig_support))
        return out_models, projs

    # --------------------------------------------------- efficient bit-projection output

    def _build_projection_plan(self) -> dict:
        """Precompute a per-output-column plan so a chunk's support projection is a GPU gather +
        constant fill + random fill — NO Python per-row reconstruct (the measured delivery
        bottleneck). One-time per formula; the reconstruction rules (simplify / ELS / plain)
        become tensors over the S output columns:
          gcol[i]   = compacted variable column to gather for output column i (-1 if not a gather)
          flip[i]   = XOR applied to the gathered/random bit (ELS negated-representative polarity)
          const[i]  = constant 0/1 for a forced (unit-propagated) column
          rgroup[i] = random-draw group id (-1 if none); ELS columns sharing a representative share
                      a group so equivalent variables stay EQUAL (a hard validity requirement).
        Output columns default to the original support (the sampled object); with no support they
        are all original variables (the full model). Returns/caches the plan dict."""
        if getattr(self, "_proj_plan", None) is not None:
            return self._proj_plan
        dev = self.device
        r = self._recon
        if r is not None and r.get("orig_support"):
            cols = list(r["orig_support"])                                 # 0-based support vars
        elif r is not None:
            cols = list(range(r["orig_nv"]))                              # full model
        elif self.support:
            cols = list(self.support)
        else:
            cols = list(range(self.N))
        S = len(cols)
        gcol, flip, const, rgroup = [-1] * S, [0] * S, [0] * S, [-1] * S
        groups: dict = {}

        def _group(key):
            return groups.setdefault(key, len(groups))

        if r is None:                                          # sampler decided ORIGINAL vars
            for i, v0 in enumerate(cols):
                gcol[i] = v0                                   # compacted == original
        elif r.get("els"):
            rep_of, nn, fm = r["rep_of"], r["new_of_new"], r["forced_map"]
            for i, v0 in enumerate(cols):
                v = v0 + 1
                if v in fm:
                    const[i] = 1 if fm[v] else 0
                    continue
                rv, pol = rep_of[v - 1]
                if rv in fm:
                    const[i] = 1 if (fm[rv] if pol else not fm[rv]) else 0
                elif rv in nn:
                    gcol[i], flip[i] = nn[rv] - 1, (0 if pol else 1)
                else:                                          # don't-care rep: one draw per class
                    rgroup[i], flip[i] = _group(("els", rv)), (0 if pol else 1)
        else:                                                  # plain unit-prop simplification
            new_of_old = {int(o): j for j, o in enumerate(r["old_of_new"])}
            fm = {abs(l): (l > 0) for l in r["forced"]}
            for i, v0 in enumerate(cols):
                v = v0 + 1
                if v in fm:                                    # unit-forced -> constant
                    const[i] = 1 if fm[v] else 0
                elif v in new_of_old:                          # survives compacted -> gather
                    gcol[i] = new_of_old[v]
                else:                                          # in no surviving clause == free_support
                    rgroup[i] = _group(("free", v0))
        self._proj_plan = {
            "cols_1based": [v0 + 1 for v0 in cols], "S": S, "n_groups": len(groups),
            "gcol": torch.tensor(gcol, dtype=torch.long, device=dev),
            "flip": torch.tensor(flip, dtype=torch.uint8, device=dev),
            "const": torch.tensor(const, dtype=torch.uint8, device=dev),
            "rgroup": torch.tensor(rgroup, dtype=torch.long, device=dev),
        }
        return self._proj_plan

    @torch.inference_mode()
    def _project(self, fully: torch.Tensor) -> torch.Tensor:
        """(n_done, S) uint8 support projection for the fully-assigned rows, via the cached plan.
        Pure GPU tensor ops — the whole point is to avoid the per-row Python reconstruct."""
        plan = self._build_projection_plan()
        pos = self.engine.assignment[fully][:, 0::2] == 1        # (n, N) bool: var TRUE
        n, dev = pos.size(0), self.device
        proj = plan["const"].unsqueeze(0).expand(n, -1).clone()  # (n, S) uint8
        g = plan["gcol"] >= 0
        if bool(g.any()):
            proj[:, g] = pos[:, plan["gcol"][g]].to(torch.uint8) ^ plan["flip"][g]
        rm = plan["rgroup"] >= 0
        if plan["n_groups"] and bool(rm.any()):
            bits = torch.randint(0, 2, (n, plan["n_groups"]), generator=self._gen,
                                 device=dev).to(torch.uint8)
            proj[:, rm] = bits[:, plan["rgroup"][rm]] ^ plan["flip"][rm]
        return proj

    @torch.inference_mode()
    def sample_bits(self, n_target: int, *, config=(True, 8, "fused"),
                    max_decision_rounds: "int | None" = 300, max_chunks: "int | None" = None):
        """Collect up to `n_target` support-projection bit-rows efficiently. Only the small
        (n, S) bit matrix crosses to host RAM per chunk (not full N-var models), so a big output
        is cheap. Returns (matrix, cols_1based): matrix is a numpy uint8 (got, S) array of 0/1,
        cols_1based lists the ORIGINAL variable id of each column. Sound: every row is a valid
        support projection (forced pinned, compacted vars from a real model, don't-cares free,
        ELS-equivalent vars kept equal via shared random groups)."""
        import numpy as np
        rv, B, la = config[0], max(1, config[1]), (config[2] if len(config) > 2 else False)
        plan = self._build_projection_plan()
        support_mask = None
        if self.support:
            support_mask = torch.zeros(self.N, dtype=torch.bool, device=self.device)
            support_mask[torch.tensor(self.support, dtype=torch.long, device=self.device)] = True
        K = self.K_fused if (la in ("fused", "fsoft") and self.K >= self._K_init) else self.K
        mats: list = []
        got = chunk = 0
        while got < n_target:
            if max_chunks is not None and chunk >= max_chunks:
                break
            chunk += 1
            fully, _, _, _ = self._run_chunk(K, B, rv, max_decision_rounds,
                                             support_mask, True, la)
            n_done = int(fully.sum().item())
            if n_done == 0:
                if chunk >= 3:                    # cold start + grace: instance is producing nothing
                    break
                continue
            take = min(n_done, n_target - got)
            mats.append(self._project(fully)[:take].cpu())
            got += take
        M = (torch.cat(mats).numpy() if mats
             else np.zeros((0, plan["S"]), dtype=np.uint8))
        return M, plan["cols_1based"]

    # ----------------------------------------------------------------- public

    @torch.inference_mode()
    def sample(
        self,
        n_target: int,
        *,
        decisions_per_round: int = 1,
        random_var: "bool | str" = True,
        max_decision_rounds: int | None = None,
        max_chunks: int | None = None,
        collect: bool = True,
        verify_clauses: list | None = None,
        support: list | None = "auto",
        complete: bool = True,
        lookahead: bool = False,
        configs: "list[tuple[bool, int]] | None" = None,
        config_policy: str = "adaptive",
        min_measure_chunks: int = 1,
        max_seconds: "float | None" = None,
    ) -> dict:
        """Draw at least `n_target` valid models.

        Stops early on any of the safety valves (all optional):
          - `max_chunks` chunks have run;
          - `max_seconds` of sampling wall have elapsed (checked between chunks — the intended
            "produce N samples, but never spin longer than T" contract);
          - ZERO models after every config has been measured plus a couple of grace chunks —
            the target is unreachable (e.g. a 0%-success instance would otherwise loop forever
            once no chunk cap is given).

        `support`: 0-based variable indices to restrict decisions to (independent-support
        sampling — the rest are determined by propagation). "auto" (default) uses the
        support parsed from the CNF (`self.support`) if any; None decides all variables.

        n_models counts FULLY-assigned models (sound — guaranteed valid). When sampling over
        an independent support, models are projected onto the support for diversity/collection.
        n_support_complete counts trajectories that fixed the whole support without conflict
        (equals n_models when the support is a true independent support). If `verify_clauses`
        is given, every fully-assigned model is independently checked and `all_valid` is set.

        `random_var`: True = random decision order (diverse), False = structured/index order (follows
        the gate DAG — completes search-heavy residuals), "auto" = cheaply probe a (order, B) grid and
        pick the best-throughput combo per formula (recommended; overrides `decisions_per_round` with
        the tuned B — cracks the service family with (structured, B=1) without hurting the niche's B=8).

        `configs`: a PORTFOLIO of (random_var, B) tuples, with `config_policy`:
          - "adaptive" (default): a throughput BANDIT — each chunk runs ONE config at full K, picked by
            its measured models/wall so far, dynamically switching to the winner and re-checking the
            rest (decaying-epsilon). Robust like a portfolio yet ~as fast as the best single config; it
            fixes the autotune's small-K mis-estimate (it measures full-K) and its permanent commit
            (it can switch). Recommended.
          - "static": run ALL configs as sub-batches every chunk (K split) — guarantees per-chunk
            coverage but a slow config's K-independent round count drags every chunk (~1.5-2.7× niche).
        `config_breakdown` reports each config's models/chunks/throughput. Overrides
        random_var/decisions_per_round when set.
        """
        B = max(1, decisions_per_round)
        sup = self.support if (isinstance(support, str) and support == "auto") else support
        support_mask = None
        sup_idx = None
        if sup:
            support_mask = torch.zeros(self.N, dtype=torch.bool, device=self.device)
            sup_t = torch.tensor(sup, dtype=torch.long, device=self.device)
            support_mask[sup_t] = True
            sup_idx = sup_t                                 # for projection
        if configs is None and isinstance(random_var, str) and random_var == "auto":
            random_var, B = self._autotune(self.K, B, max_decision_rounds, support_mask,
                                           complete, lookahead)
        # Config portfolio. Two policies (when len>1):
        #  - "adaptive" (default): a THROUGHPUT BANDIT — each chunk runs ONE config at full K, chosen by
        #    its measured models/wall so far, dynamically switching to the winner (and re-checking the
        #    rest). Robust like a portfolio but ~as fast as the best single config — fixes both the
        #    autotune's small-K mis-estimate and its permanent commit.
        #  - "static": run ALL configs as sub-batches every chunk (K split). Guarantees per-chunk
        #    coverage but a slow config's K-independent round count drags every chunk (~1.5-2.7×).
        # Normalise the portfolio to (random_var, B, lookahead) triples. A config may be a 2-tuple
        # (rv, B) — lookahead defaults off — or a 3-tuple (rv, B, lookahead), lookahead ∈
        # {False, "soft", "full"}. "soft" (cheap risk-polarity) is the big win on the tight/deep
        # CIRCUIT regime (conflict-avoidance polarity → far higher success at no extra memory);
        # "full" snapshots the (K,M) state so it needs K halved. off/"soft" cost no extra memory.
        def _norm(c):
            return (c[0], max(1, c[1]), (c[2] if len(c) > 2 else False))
        plan = [_norm(c) for c in configs] if configs else [(random_var, B, lookahead)]
        # NOTE: the packed K_fused keeps the conservative eng_hold deduction (__init__) even for an
        # ALL-fused portfolio. In theory an all-fused run never allocates the engine's base-K
        # sat/lsum, so skipping the deduction would recover ~9% K (measured on b12-c25). But that
        # headroom was ACCIDENTALLY covering the fixed 4096×M soundness-check temporary (~3.3 GB on
        # b05-c50's M=201k); skipping it SPILLED the big c50 wide instances under the kill-switch
        # (11.7 GB on the 12 GB 4080, b05-c50). Keeping it costs ≤0.97× on the small c25 instances —
        # within the 0.9× bar — and stays safe on the big ones. If the ~9% is ever wanted back, it
        # must go with an explicit 4096·M reservation, then re-validated under the kill-switch.
        # If ANY config in the plan uses "full" lookahead, halve K for the whole run (the fixed
        # per-chunk allocation must fit the most memory-hungry config's snapshot).
        need_full = any(c[2] in (True, "full") for c in plan)
        K = max(1, self.K // 2) if need_full else self.K
        adaptive = configs is not None and config_policy == "adaptive" and len(plan) > 1
        # Per-config work counters (decs/props) alongside models/wall so the reported dec/s, impl/s,
        # assigns/s can be attributed to the SAME config as the headline samples/s (best_rate) —
        # otherwise they blend over the bandit's exploration (incl. the slow orac chunk), understating
        # the winner ~2x and diverging across GPUs (measured on b06.c25, 2026-07-04).
        cfg_stats = {c: {"models": 0, "attempts": 0, "wall": 0.0, "chunks": 0,
                         "decs": 0, "props": 0} for c in plan}
        K_sub = max(1, K // len(plan)) if (configs and not adaptive and len(plan) > 1) else K
        models: list[list[int]] = []
        attempts = 0
        n_models = 0
        n_support_complete = 0
        rounds_hist: list[int] = []
        free_after_hist: list[float] = []
        chunk = 0
        all_valid = True
        gpu_unsound = 0          # rows that were 'fully+active' yet a clause had sat_count==0 (should be 0)
        distinct: set[tuple[int, ...]] = set()
        keep_rows = collect or verify_clauses is not None

        self._sync()
        bcp_r0 = self.engine.total_bcp_rounds
        bcp_o0 = self.engine.total_bcp_ops + self.engine.kernel_ops()
        bcp_p0 = self.engine.total_bcp_props + self.engine.kernel_props()
        bcp_d0 = self.engine.total_bcp_decisions + self.engine.kernel_decs()
        t0 = time.perf_counter()
        while True:
            # Stop once n_target is met — BUT for the adaptive bandit, always measure every config at
            # least once first. Otherwise on a big-K GPU a single chunk can already exceed n_target, so
            # only plan[0] ever runs and a faster config is never discovered (the A100 shallow-circuit
            # regression). Bounded by max_chunks. Cold-start chunks are productive (real models).
            cold = adaptive and any(cfg_stats[c]["chunks"] == 0 for c in plan)
            # Stability: don't stop until SOME config has been measured over >= min_measure_chunks
            # chunks (the bandit piles those onto the winner). A big-K GPU meets n_target in the
            # cold-start (1 chunk/config), so a single noisy chunk would set the headline rate —
            # measured on b06.c25: best_rate_sps swung 131k-249k (1.9x) run-to-run at 1 chunk.
            enough = (not adaptive) or max((cfg_stats[c]["chunks"] for c in plan),
                                           default=0) >= min_measure_chunks
            if not cold and enough and n_models >= n_target:
                break
            if max_chunks is not None and chunk >= max_chunks:
                break
            if max_seconds is not None and time.perf_counter() - t0 > max_seconds:
                break                                   # hard time cap (between chunks)
            if n_models == 0 and not cold and chunk >= len(plan) + 2:
                break                                   # every config measured + grace: hopeless
            chunk += 1
            # Once past n_target but short of the measurement window, EXPLOIT the winner (no eps
            # exploration) so its stable rate is measured on back-to-back chunks, not diluted.
            topup = adaptive and not cold and n_models >= n_target
            chunk_cfgs = ([self._select_config(plan, cfg_stats, chunk, exploit=topup)]
                          if adaptive else plan)
            for cfg in chunk_cfgs:
                rv_c, B_c, la_c = cfg
                # Fused-trajectory chunks (fused AND fsoft — both are single-launch kernels with no
                # (K,N) decide masks — see K_fused) run at the boosted K. Only while K is at its
                # construction value: a caller-lowered K (coverage run) is honoured.
                K_c = self.K_fused if (la_c in ("fused", "fsoft") and self.K >= self._K_init) else K_sub
                if adaptive:
                    self._sync()
                    # read BEFORE t_cfg so the (synced) counter reads aren't charged to wall
                    p_before = self.engine.total_bcp_props + self.engine.kernel_props()
                    d_before = self.engine.total_bcp_decisions + self.engine.kernel_decs()
                    t_cfg = time.perf_counter()
                fully, sup_complete, rounds, free_after = self._run_chunk(
                    K_c, B_c, rv_c, max_decision_rounds, support_mask, complete, la_c)
                rounds_hist.append(rounds)
                free_after_hist.append(free_after)
                attempts += K_c
                n_done = int(fully.sum().item())
                n_models += n_done
                n_support_complete += int(sup_complete.sum().item())
                cfg_stats[cfg]["models"] += n_done
                cfg_stats[cfg]["attempts"] += K_c
                cfg_stats[cfg]["chunks"] += 1
                if adaptive:
                    self._sync(); cfg_stats[cfg]["wall"] += time.perf_counter() - t_cfg
                    cfg_stats[cfg]["props"] += (self.engine.total_bcp_props
                                                + self.engine.kernel_props()) - p_before
                    cfg_stats[cfg]["decs"] += (self.engine.total_bcp_decisions
                                               + self.engine.kernel_decs()) - d_before
                if n_done:
                    # GPU soundness (cheap): for a FULLY-assigned row, non-falsified ⟺ TRUE, so
                    # "every clause has sat_count > 0" ⟺ "every clause has a TRUE literal" ⟺ satisfied.
                    # (Holds ONLY at a complete assignment — sat_count>0 on a PARTIAL assignment just
                    # means "not yet violated". We gate on `fully`, so it's valid.) Redundant with
                    # `active` (cleared the instant a clause hits 0); CPU `_satisfies()` is the real check.
                    # Capped to 4096 rows: it's a diagnostic, and an uncapped slice materialises an
                    # (n_done, M) temp that can reach GBs at the fused K. engine.sat_rows() is
                    # transparent to the packed fused state (counters live in the kernel there).
                    idx = fully.nonzero().view(-1)[:4096]
                    sat_rows = self.engine.sat_rows(idx)
                    if not bool((sat_rows > 0).all()):
                        gpu_unsound += int((sat_rows <= 0).any(dim=1).sum().item())
                if n_done and keep_rows:
                    asg = self.engine.assignment[fully]         # (n_done, 2N) int8
                    pos = asg[:, 0::2] == 1                      # (n_done, N) bool, var v TRUE
                    vidx = torch.arange(1, self.N + 1, device=self.device)
                    comp_rows = torch.where(pos, vidx, -vidx).to(torch.int32).cpu().tolist()
                    if self._recon is not None and not self._recon.get("unsat"):
                        # reconstruct full ORIGINAL models (forced pinned, don't-cares randomized)
                        out_rows, projs = self._reconstruct(comp_rows)
                    else:
                        out_rows = comp_rows
                        si = sup_idx.tolist() if sup_idx is not None else None
                        projs = [tuple(r[i] for i in si) for r in comp_rows] if si is not None else comp_rows
                    for p in projs:
                        distinct.add(tuple(p))
                    if verify_clauses is not None:
                        for r in out_rows:
                            if not _satisfies(r, verify_clauses):
                                all_valid = False
                    if collect:
                        models.extend(out_rows)
        self._sync()
        elapsed = time.perf_counter() - t0
        bcp_rounds = self.engine.total_bcp_rounds - bcp_r0     # BCP work over this timed run (delta)
        bcp_ops = self.engine.total_bcp_ops + self.engine.kernel_ops() - bcp_o0
        bcp_props = self.engine.total_bcp_props + self.engine.kernel_props() - bcp_p0
        bcp_decs = self.engine.total_bcp_decisions + self.engine.kernel_decs() - bcp_d0
        bcp_impl = bcp_props - bcp_decs                        # every non-decision assignment

        out = {
            "n_models": n_models, "n_support_complete": n_support_complete, "attempts": attempts,
            "success_rate": (n_models / attempts) if attempts else 0.0,
            "support_success_rate": (n_support_complete / attempts) if attempts else 0.0,
            "elapsed_s": elapsed,
            "samples_per_s": (n_models / elapsed) if elapsed > 0 else 0.0,
            "attempts_per_s": (attempts / elapsed) if elapsed > 0 else 0.0,
            # Lightweight BCP-work stats (free counters) — to tell success-bound from BCP-bound: a low
            # samples/s with high attempts/s = low success; a low attempts/s = each trajectory is slow
            # (deep propagation). ops/attempt = clause-cells touched per trajectory.
            "bcp_rounds": bcp_rounds, "bcp_ops": bcp_ops,
            "bcp_rounds_per_chunk": (bcp_rounds / chunk) if chunk else 0.0,
            "bcp_ops_per_attempt": (bcp_ops / attempts) if attempts else 0.0,
            # CDCL-comparable work rate: literals assigned (decisions + implications) per second —
            # what CDCL solvers report as props/sec (typically ~1e7/s single-core). Split into the
            # branch DECISIONS and the unit-propagation IMPLICATIONS they triggered (impl = assigns
            # - decisions); implications/decision is the propagation fan-out per branch.
            "assigns_per_s": (bcp_props / elapsed) if elapsed > 0 else 0.0,
            "decisions_per_s": (bcp_decs / elapsed) if elapsed > 0 else 0.0,
            "implications_per_s": (bcp_impl / elapsed) if elapsed > 0 else 0.0,
            "bcp_decisions": bcp_decs, "bcp_implications": bcp_impl,
            "mean_free_after_support": (sum(free_after_hist) / len(free_after_hist)) if free_after_hist else 0.0,
            "n_chunks": chunk, "K": K, "B": B, "random_var": bool(random_var),
            "support_size": (len(sup) if sup else None),
            "mean_decision_rounds": (sum(rounds_hist) / len(rounds_hist)) if rounds_hist else 0,
            "max_decision_rounds": max(rounds_hist) if rounds_hist else 0,
            "n_distinct": len(distinct) if keep_rows else None,
            "gpu_sound": gpu_unsound == 0, "gpu_unsound": gpu_unsound,
            "kernel": self.use_kernel,
        }
        if collect:
            out["models"] = models
        if verify_clauses is not None:
            out["all_valid"] = all_valid
        if configs is not None:
            def _rate(s):
                return (s["models"] / s["wall"]) if s["wall"] > 0 else 0.0
            dom = max(cfg_stats.items(), key=lambda kv: kv[1]["models"])[0]   # most models produced
            out["random_var"], out["B"], out["lookahead"] = bool(dom[0]), dom[1], dom[2]
            out["dominant_config"] = [bool(dom[0]), dom[1], dom[2]]
            # Best-THROUGHPUT config = max models/wall (the one a real run commits to after exploring).
            # The benchmark headline uses THIS rate, not the end-to-end blend, so the bandit's slower
            # exploration chunks (and any first-chunk warmup) never drag the reported steady-state
            # throughput. Only meaningful when adaptive (per-config wall is timed); else wall==0.
            # Prefer a WELL-MEASURED config (>= min_measure_chunks) over a lucky 1-chunk rate, then by
            # rate — so the headline rests on the config the top-up phase actually committed to and
            # measured, not a config whose single cold-start chunk happened to clock fast.
            brc, brs = max(cfg_stats.items(),
                           key=lambda kv: (kv[1]["chunks"] >= min_measure_chunks, _rate(kv[1])))
            out["best_rate_config"] = [bool(brc[0]), brc[1], brc[2]]
            out["best_rate_sps"] = _rate(brs) if brs["wall"] > 0 else None
            out["best_rate_success"] = (brs["models"] / brs["attempts"]) if brs["attempts"] else 0.0
            # Work rates for the SAME best-rate config as best_rate_sps (consistent + stable: measured
            # over the winner's chunks, not blended over the exploration). gpap_sample reports these.
            bw = brs["wall"]
            out["best_rate_decisions_per_s"] = (brs["decs"] / bw) if bw > 0 else None
            out["best_rate_implications_per_s"] = ((brs["props"] - brs["decs"]) / bw) if bw > 0 else None
            out["best_rate_assigns_per_s"] = (brs["props"] / bw) if bw > 0 else None
            out["best_rate_chunks"] = brs["chunks"]
            out["config_policy"] = "adaptive" if adaptive else "static"

            def _lab(c):
                return f"rv={c[0]},B={c[1]}" + (f",la={c[2]}" if c[2] else "")
            out["config_breakdown"] = {
                _lab(c): {
                    "models": s["models"], "attempts": s["attempts"], "chunks": s["chunks"],
                    "wall_s": s["wall"],
                    "success_rate": (s["models"] / s["attempts"]) if s["attempts"] else 0.0,
                    "samples_per_s": (s["models"] / s["wall"]) if s["wall"] > 0 else None,
                    "decisions_per_s": (s["decs"] / s["wall"]) if s["wall"] > 0 else None,
                    "implications_per_s": ((s["props"] - s["decs"]) / s["wall"]) if s["wall"] > 0 else None,
                } for c, s in cfg_stats.items()
            }
        return out


def _satisfies(dimacs_model: list[int], clauses: list) -> bool:
    """Independent soundness check: does this full assignment satisfy every clause?"""
    true_set = set(dimacs_model)
    for c in clauses:
        if not any(l in true_set for l in c):
            return False
    return True


def simplify_cnf(nv: int, clauses: list, support: list | None):
    """Unit-propagation simplification before sampling (what Arjun/preprocessors do). Asserts all
    unit clauses, propagates, drops satisfied clauses + falsified literals, and COMPACTS the
    remaining (unassigned) variables to 1..nv'. This shrinks M/N a lot on unit-heavy formulas
    (e.g. these sk_ benchmarks), so the sampler fits a much larger K → higher throughput.

    Returns (nv2, clauses2, support2, old_of_new, forced, free_support) or None if UNSAT:
      nv2          : compacted variable count.
      clauses2     : simplified clauses over compacted vars (1-based).
      support2     : 0-based support vars surviving in the compacted formula (decide these).
      old_of_new   : list; old_of_new[i] = original DIMACS var of compacted var i+1.
      forced       : DIMACS lits fixed by unit propagation (constant in EVERY full model).
      free_support : 0-based original support vars that are neither forced nor in any remaining
                     clause — true don't-cares; to preserve the support-projection distribution
                     they MUST be sampled uniformly at random in reconstruction (not dropped).

    Soundness: unit propagation only asserts already-forced units and removes satisfied clauses /
    falsified literals — it is strictly MODEL-PRESERVING (it does NOT eliminate variables by
    resolution like VE/BCE, which CMSGen disables for exactly that reason). So the set of valid
    support projections is unchanged, provided forced support vars are pinned to their forced value
    and free_support vars are randomized. Validated by tests/test_sampler.py.
    """
    import numpy as np
    from itertools import chain
    from gpap.sampling.preprocess import unit_closure
    out, assigned, status = unit_closure(clauses, [])   # reads only — no defensive copy needed
    if status == "unsat":
        return None
    nonunit = [c for c in out if len(c) > 1]
    # Vectorised variable compaction (the sorted-set + per-literal remap listcomps cost ~1.5s
    # of pure Python per call on 1M-clause unrolls; this runs on construct AND rebuild).
    lens = np.fromiter(map(len, nonunit), dtype=np.int64, count=len(nonunit))
    flat = (np.array(list(chain.from_iterable(nonunit)), dtype=np.int64) if lens.sum()
            else np.empty(0, dtype=np.int64))
    used_np = np.unique(np.abs(flat))                   # sorted unique surviving vars
    lut = np.zeros((int(used_np[-1]) + 1) if len(used_np) else 1, dtype=np.int64)
    lut[used_np] = np.arange(1, len(used_np) + 1)       # old var -> compacted var (0 = unused)
    nf = (np.sign(flat) * lut[np.abs(flat)]).tolist()
    bounds = np.zeros(len(nonunit) + 1, dtype=np.int64)
    np.cumsum(lens, out=bounds[1:])
    clauses2 = [nf[a:b] for a, b in zip(bounds[:-1].tolist(), bounds[1:].tolist())]

    def _new(v1):                                       # 1-based old var -> compacted var or 0
        return int(lut[v1]) if v1 < len(lut) else 0

    support2 = sorted({_new(v + 1) - 1 for v in (support or []) if _new(v + 1)})
    forced = [(v if assigned[v] else -v) for v in assigned]
    free_support = sorted(v for v in (support or [])
                          if (v + 1) not in assigned and not _new(v + 1))
    return len(used_np), clauses2, support2, used_np.tolist(), forced, free_support


def simplify_cnf_els(nv: int, clauses: list, support: list | None):
    """Stronger simplification: unit-closure + EQUIVALENT-LITERAL SUBSTITUTION (ELS) + compaction.

    On top of `simplify_cnf`'s unit propagation, this collapses mutually-implied literals (binary-
    implication SCCs) to a single representative, REMOVING variables. Since BCP propagation cost
    scales with M (clauses) and K with per-trajectory memory (~M), shrinking the formula speeds up
    the dominant propagate AND lets a larger K fit — it hits the profiled bottleneck twice. On the
    sk_ benchmarks the binary-graph SCCs alone remove 5-38% of variables / 3-30% of clauses.

    ELS is equisatisfiability-preserving and, for PROJECTED sampling, support-projection preserving
    **iff reconstruction is consistent**: every removed variable equals its representative literal
    in EVERY model, so a removed var (even a don't-care one) must be reconstructed from its rep, NOT
    randomized independently — otherwise an equivalent pair could be assigned unequal values, an
    invalid projection. We therefore reconstruct per equivalence-CLASS (one value per representative,
    applied to all members with the right polarity). Validated by tests/test_sampler.py.

    Returns (nv_r, clauses_r, support_r, recon) or None if UNSAT, where recon carries the per-original
    -variable resolver used by GPUSampler._reconstruct:
      rep_of[v-1]  : (rep_var, same_polarity) — original var v takes rep_var's value (negated if not).
      new_of_new   : {rep_var: compacted_idx(1-based)} — reps that survive as decided/propagated vars.
      forced_map   : {var: bool} — vars pinned by either unit-closure pass.
    """
    from gpap.engine.formula import ext_lit_to_lit_index
    from gpap.sampling.preprocess import equivalent_literal_substitution, unit_closure

    # Pass 1: unit closure over the original variables (reads only — no copy).
    out1, assigned1, st1 = unit_closure(clauses, [])
    if st1 == "unsat":
        return None
    nonunit1 = [c for c in out1 if len(c) > 1]

    # Binary-implication edges (2N lit-index scheme) from binary clauses: (l1∨l2) ⇒ ¬l1→l2, ¬l2→l1.
    edges = []
    for c in nonunit1:
        if len(c) == 2:
            i1, i2 = ext_lit_to_lit_index(c[0]), ext_lit_to_lit_index(c[1])
            edges.append((i1 ^ 1, i2))
            edges.append((i2 ^ 1, i1))
    if edges:
        els_clauses, var_map, st = equivalent_literal_substitution(nonunit1, edges, nv)
        if st.get("status") == "unsat":
            return None
    else:
        els_clauses, var_map = nonunit1, {}

    # Pass 2: ELS substitution can turn (a∨a)→(a) into new units — absorb them.
    out2, assigned2, st2 = unit_closure(els_clauses, [])
    if st2 == "unsat":
        return None
    nonunit2 = [c for c in out2 if len(c) > 1]

    used = sorted({abs(l) for c in nonunit2 for l in c})       # surviving representative vars
    new_of_new = {v: i + 1 for i, v in enumerate(used)}
    clauses_r = [[(new_of_new[abs(l)] if l > 0 else -new_of_new[abs(l)]) for l in c]
                 for c in nonunit2]
    forced_map = {**assigned1, **assigned2}                    # var -> bool (disjoint key sets)

    # Per-original-variable resolver: v -> (rep_var, same_polarity).
    rep_of = []
    for v in range(1, nv + 1):
        if v in var_map:                                       # removed by ELS -> rep DIMACS literal
            rl = var_map[v]
            rep_of.append((abs(rl), rl > 0))
        else:
            rep_of.append((v, True))                           # its own rep (forced / model / free)

    def _rep(v0):                                              # surviving decided var for support v0
        v = v0 + 1
        if v in forced_map:
            return None
        rv = rep_of[v - 1][0]
        return (new_of_new[rv] - 1) if rv in new_of_new else None

    support_r = sorted({_rep(v0) for v0 in (support or []) if _rep(v0) is not None})
    recon = {"orig_nv": nv, "orig_support": sorted(support) if support else [],
             "rep_of": rep_of, "new_of_new": new_of_new, "forced_map": forced_map}
    return len(used), clauses_r, support_r, recon


def parse_independent_support(path: str) -> list[int] | None:
    """Read the independent support from `c ind <v1> <v2> ... 0` header lines (UniGen/QuickSampler
    convention). Returns 0-based variable indices, or None if no `c ind` lines are present."""
    sup: set[int] = set()
    found = False
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("c ind") or s.startswith("cr "):
                found = True
                for tok in s.split()[2:] if s.startswith("c ind") else s.split()[1:]:
                    try:
                        v = int(tok)
                    except ValueError:
                        continue
                    if v != 0:
                        sup.add(abs(v) - 1)            # DIMACS 1-based → 0-based var
            elif s[0].isdigit() or s[0] == "-":
                break                                  # past the header into clauses
    return sorted(sup) if found else None
