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
import time

import torch

from gpap.clsdata.cnf_formula import CnfFormula
from gpap.solver.bcp_engine import BCPEngine
from gpap.utils.gpu_budget import compute_k_safe, gpu_mem_available


class GPUSampler:
    """Batched GPU sampler: K parallel randomized BCP-completions per chunk."""

    def __init__(
        self,
        formula: CnfFormula,
        device: torch.device,
        *,
        k_cap: int | None = 16384,
        seed: int | None = None,
    ):
        self.formula = formula
        self.device = device
        self.N = formula.num_vars
        self.M = formula.num_clauses
        self.engine = BCPEngine(formula)
        self._max_cls = int(formula.clause_sizes.max().item()) if self.M else 1
        self._total_lits = int(formula.clause_sizes.sum().item()) if self.M else 0
        # K auto-sized to the GPU; mem_get_info already excludes the loaded formula.
        free_bytes = gpu_mem_available(device)
        self.K = compute_k_safe(
            self.M, self.N, free_bytes, cap=k_cap,
            max_clause_size=self._max_cls, total_lits=self._total_lits or None,
            deduct_formula=False,
        )
        # IMPORTANT: compute_k_safe budgets only the BCPEngine's static tensors and is ~20x
        # OPTIMISTIC for these formulas — the propagate/apply temporaries (and the sampler's own
        # per-round (K,N) masked-priority + free masks) dominate. Measured full footprint on this
        # box ≈ 55-110 B per clause-batch + ~30-40 B per var-batch (measured: M=2215,N=1494 →
        # 238 B/batch... i.e. 0.24 MiB/batch at K; M=17828,N=4842 → ~55 B/clause). Without this
        # clamp, K spills into host RAM over PCIe on WSL2 (~12 GB/s thrash). Size K to ~70% of free
        # VRAM under the calibrated per-batch cost (70*M + 40*N covers both small- and large-M).
        per_batch_emp = 70 * self.M + 40 * self.N + 64
        k_safe_emp = int(0.7 * free_bytes / per_batch_emp)
        self.K = max(1, min(self.K, k_safe_emp))
        self._gen = torch.Generator(device=device)
        if seed is not None:
            self._gen.manual_seed(seed)
        import random as _random
        self._pyrng = _random.Random(seed)
        self.support: list[int] | None = None   # 0-based independent-support vars (from `c ind`)
        self._recon: dict | None = None          # set when built with simplify=True (see _from_simplified)

    @classmethod
    def from_cnf_file(cls, path: str, device: torch.device, *, simplify: bool = False,
                      **kw) -> "GPUSampler":
        """Build a sampler from a DIMACS file. `simplify=True` first applies unit-propagation
        simplification (shrinks M -> larger K -> more throughput) and samples the compacted
        formula, reconstructing full original models at collection time (see simplify_cnf)."""
        support = parse_independent_support(path)   # 0-based, or None
        if not simplify:
            formula = CnfFormula(device)
            formula.parse_cnf_file(path)
            smp = cls(formula, device, **kw)
            smp.support = support
            return smp
        from pysat.formula import CNF
        cnf = CNF(from_file=path)
        return cls._from_simplified(cnf.nv, [list(c) for c in cnf.clauses], support, device, **kw)

    @classmethod
    def from_clauses(cls, num_vars: int, clauses, device: torch.device, *,
                     simplify: bool = False, support: list | None = None, **kw) -> "GPUSampler":
        if simplify:
            return cls._from_simplified(num_vars, [list(c) for c in clauses], support, device, **kw)
        formula = CnfFormula(device)
        formula.init_formula(num_vars, clauses)
        smp = cls(formula, device, **kw)
        smp.support = support
        return smp

    @classmethod
    def _from_simplified(cls, nv: int, clauses: list, support: list | None,
                         device: torch.device, **kw) -> "GPUSampler":
        res = simplify_cnf(nv, clauses, support)
        if res is None:                                  # UNSAT after unit propagation
            formula = CnfFormula(device)
            formula.init_formula(1, [[1], [-1]])
            smp = cls(formula, device, **kw)
            smp.support = []
            smp._recon = {"unsat": True}
            return smp
        nv2, cl2, sup2, old_of_new, forced, free_support = res
        formula = CnfFormula(device)
        formula.init_formula(nv2, cl2)
        smp = cls(formula, device, **kw)
        smp.support = sup2
        smp._recon = {"unsat": False, "old_of_new": old_of_new, "forced": forced,
                      "free_support": free_support, "orig_nv": nv,
                      "orig_support": sorted(support) if support else []}
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
        a clause at sat_count==1 → conflict, ==2 → becomes a forced unit (cascade risk), ==3 →
        tightens. risk = Σ w(sat_count) over occ[lit], w(1)=1e6, w(2)=4, w(3)=1, else 0. A lower
        risk means a polarity less likely to (eventually) conflict. `lits`: (P,) lit indices; the
        row for lits[i] is i (caller passes one lit per active row, aligned with arange(P))."""
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
            score, var = masked.topk(B, dim=1, largest=False)   # (K,B) B smallest ranks
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
                # off, or cheap "soft" risk-polarity (handled inside _pick_decisions)
                decisions = self._pick_decisions(dfree, need, B, priority,
                                                 soft_la=(lookahead == "soft"))
                eng.apply_assumptions_bulk(decisions)
                eng.propagate()
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

            eng.apply_assumptions_bulk(dec_t); eng.propagate()
            bad_t = eng.conflict & ~conf                           # TRUE newly conflicts
            _restore()
            eng.apply_assumptions_bulk(dec_f); eng.propagate()
            bad_f = eng.conflict & ~conf                           # FALSE newly conflicts
            _restore()
            pol = torch.randint(0, 2, (K,), generator=self._gen, device=self.device)
            pol = torch.where(bad_t & ~bad_f, torch.ones_like(pol), pol)   # TRUE bad → FALSE
            pol = torch.where(bad_f & ~bad_t, torch.zeros_like(pol), pol)  # FALSE bad → TRUE
            dec = torch.where(need, 2 * var + pol, neg1).unsqueeze(1)
            eng.apply_assumptions_bulk(dec)
            eng.propagate()
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
        eng.reset_state(K)
        eng.propagate()                                     # process the formula's own units
        if random_var:
            priority = torch.randperm(self.N, generator=self._gen, device=self.device)
        else:
            priority = torch.arange(self.N, device=self.device)

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

    def _reconstruct(self, comp_rows: list[list[int]]):
        """Map full models of the simplified (compacted) formula back to full ORIGINAL models, and
        return (original_models, support_projections). Forced vars are pinned to their unit-propagated
        value; the compacted vars take their sampled value; don't-care support vars (free_support) are
        sampled uniformly at random (so the support-projection distribution is preserved). Remaining
        unmapped original vars are true don't-cares set to False (any value gives a valid model)."""
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

    # ----------------------------------------------------------------- public

    @torch.inference_mode()
    def sample(
        self,
        n_target: int,
        *,
        decisions_per_round: int = 1,
        random_var: bool = True,
        max_decision_rounds: int | None = None,
        max_chunks: int | None = None,
        collect: bool = True,
        verify_clauses: list | None = None,
        support: list | None = "auto",
        complete: bool = True,
        lookahead: bool = False,
    ) -> dict:
        """Draw at least `n_target` valid models (or until `max_chunks` chunks run).

        `support`: 0-based variable indices to restrict decisions to (independent-support
        sampling — the rest are determined by propagation). "auto" (default) uses the
        support parsed from the CNF (`self.support`) if any; None decides all variables.

        n_models counts FULLY-assigned models (sound — guaranteed valid). When sampling over
        an independent support, models are projected onto the support for diversity/collection.
        n_support_complete counts trajectories that fixed the whole support without conflict
        (equals n_models when the support is a true independent support). If `verify_clauses`
        is given, every fully-assigned model is independently checked and `all_valid` is set.
        """
        B = max(1, decisions_per_round)
        # "full" lookahead snapshots the (K,M) engine state during trials → halve K to fit it;
        # "soft" (risk-polarity) needs no extra memory.
        K = max(1, self.K // 2) if lookahead in (True, "full") else self.K
        sup = self.support if (isinstance(support, str) and support == "auto") else support
        support_mask = None
        sup_idx = None
        if sup:
            support_mask = torch.zeros(self.N, dtype=torch.bool, device=self.device)
            sup_t = torch.tensor(sup, dtype=torch.long, device=self.device)
            support_mask[sup_t] = True
            sup_idx = sup_t                                 # for projection
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
        t0 = time.perf_counter()
        while n_models < n_target:
            if max_chunks is not None and chunk >= max_chunks:
                break
            chunk += 1
            fully, sup_complete, rounds, free_after = self._run_chunk(
                K, B, random_var, max_decision_rounds, support_mask, complete, lookahead)
            rounds_hist.append(rounds)
            free_after_hist.append(free_after)
            attempts += K
            n_done = int(fully.sum().item())
            n_models += n_done
            n_support_complete += int(sup_complete.sum().item())
            if n_done:
                # GPU soundness (cheap): a fully-assigned row satisfies the formula iff every clause
                # has sat_count > 0 (no clause fully falsified). Verifies models without any CPU work.
                if not bool((self.engine.sat_count[fully] > 0).all()):
                    gpu_unsound += int((self.engine.sat_count[fully] <= 0).any(dim=1).sum().item())
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

        out = {
            "n_models": n_models, "n_support_complete": n_support_complete, "attempts": attempts,
            "success_rate": (n_models / attempts) if attempts else 0.0,
            "support_success_rate": (n_support_complete / attempts) if attempts else 0.0,
            "elapsed_s": elapsed,
            "samples_per_s": (n_models / elapsed) if elapsed > 0 else 0.0,
            "attempts_per_s": (attempts / elapsed) if elapsed > 0 else 0.0,
            "mean_free_after_support": (sum(free_after_hist) / len(free_after_hist)) if free_after_hist else 0.0,
            "n_chunks": chunk, "K": K, "B": B, "support_size": (len(sup) if sup else None),
            "mean_decision_rounds": (sum(rounds_hist) / len(rounds_hist)) if rounds_hist else 0,
            "max_decision_rounds": max(rounds_hist) if rounds_hist else 0,
            "n_distinct": len(distinct) if keep_rows else None,
            "gpu_sound": gpu_unsound == 0, "gpu_unsound": gpu_unsound,
        }
        if collect:
            out["models"] = models
        if verify_clauses is not None:
            out["all_valid"] = all_valid
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
    from gpap.solver.preprocess import unit_closure
    out, assigned, status = unit_closure([list(c) for c in clauses], [])
    if status == "unsat":
        return None
    nonunit = [c for c in out if len(c) > 1]
    used = sorted({abs(l) for c in nonunit for l in c})
    new_of_old = {v: i + 1 for i, v in enumerate(used)}
    clauses2 = [[(new_of_old[abs(l)] if l > 0 else -new_of_old[abs(l)]) for l in c] for c in nonunit]
    support2 = sorted({new_of_old[v + 1] - 1 for v in (support or []) if (v + 1) in new_of_old})
    forced = [(v if assigned[v] else -v) for v in assigned]
    free_support = sorted(v for v in (support or [])
                          if (v + 1) not in assigned and (v + 1) not in new_of_old)
    return len(used), clauses2, support2, used, forced, free_support


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
