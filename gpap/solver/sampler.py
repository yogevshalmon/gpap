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
        self.support: list[int] | None = None   # 0-based independent-support vars (from `c ind`)

    @classmethod
    def from_cnf_file(cls, path: str, device: torch.device, **kw) -> "GPUSampler":
        formula = CnfFormula(device)
        formula.parse_cnf_file(path)
        smp = cls(formula, device, **kw)
        smp.support = parse_independent_support(path)   # 0-based vars, or None
        return smp

    @classmethod
    def from_clauses(cls, num_vars: int, clauses, device: torch.device, **kw) -> "GPUSampler":
        formula = CnfFormula(device)
        formula.init_formula(num_vars, clauses)
        return cls(formula, device, **kw)

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
    def _pick_decisions(
        self, free: torch.Tensor, need: torch.Tensor, B: int, priority: torch.Tensor
    ) -> torch.Tensor:
        """Build a (K, B) int64 decision tensor (lit indices, -1 = no decision).

        Per row needing a decision, choose up to B free variables — the ones with the
        smallest `priority` rank (priority is a per-chunk (N,) vector: a random permutation
        for diverse var order, or arange for deterministic). Polarity is random per pick, so
        samples diverge even under a shared variable order. Rows not in `need`, and picks with
        no remaining free var, are -1 (skipped by apply_assumptions_bulk).

        One (K,N) temp per call (the masked-priority); generating `priority` once per chunk
        avoids a per-round (K,N) random tensor.
        """
        BIG = priority.numel() + 1
        masked = torch.where(free, priority.unsqueeze(0), priority.new_full((), BIG))  # (K,N)
        if B == 1:
            score, var = masked.min(dim=1, keepdim=True)    # (K,1) smallest rank among free
        else:
            score, var = masked.topk(B, dim=1, largest=False)   # (K,B) B smallest ranks
        pol = torch.randint(0, 2, var.shape, generator=self._gen, device=self.device)
        lit = 2 * var + pol                                 # (K,B)
        valid = (score < BIG) & need.unsqueeze(1)           # picked a real free var, row needs it
        return torch.where(valid, lit, torch.full_like(lit, -1))

    @torch.inference_mode()
    def _decide_phase(self, B, priority, mask, max_rounds, rounds):
        """Keep deciding free vars in `mask` (None = all free) until none remain for any active
        row (or max_rounds hit). Returns updated `rounds`."""
        eng = self.engine
        while True:
            free = self._free_mask()
            dfree = free & mask if mask is not None else free
            need = eng.active & dfree.any(dim=1)
            if not bool(need.any()):
                break
            if max_rounds is not None and rounds >= max_rounds:
                break
            rounds += 1
            decisions = self._pick_decisions(dfree, need, B, priority)
            eng.apply_assumptions_bulk(decisions)
            eng.propagate()
        return rounds

    @torch.inference_mode()
    def _run_chunk(self, K: int, B: int, random_var: bool, max_rounds: int | None,
                   support_mask: torch.Tensor | None, complete: bool):
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
        rounds = self._decide_phase(B, priority, support_mask, max_rounds, 0)

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
            rounds = self._decide_phase(B, priority, None, max_rounds, rounds)

        free = self._free_mask()
        fully = eng.active & ~free.any(dim=1)
        return fully, support_complete, rounds, free_after_support

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
        K = self.K
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
        distinct: set[tuple[int, ...]] = set()
        keep_rows = collect or verify_clauses is not None

        self._sync()
        t0 = time.perf_counter()
        while n_models < n_target:
            if max_chunks is not None and chunk >= max_chunks:
                break
            chunk += 1
            fully, sup_complete, rounds, free_after = self._run_chunk(
                K, B, random_var, max_decision_rounds, support_mask, complete)
            rounds_hist.append(rounds)
            free_after_hist.append(free_after)
            attempts += K
            n_done = int(fully.sum().item())
            n_models += n_done
            n_support_complete += int(sup_complete.sum().item())
            if n_done and keep_rows:
                asg = self.engine.assignment[fully]         # (n_done, 2N) int8
                pos = asg[:, 0::2] == 1                      # (n_done, N) bool, var v TRUE
                vidx = torch.arange(1, self.N + 1, device=self.device)
                full_rows = torch.where(pos, vidx, -vidx).to(torch.int32)
                # diversity is over the support projection (the sample), or the full model
                proj = full_rows[:, sup_idx] if sup_idx is not None else full_rows
                for r in proj.cpu().tolist():
                    distinct.add(tuple(r))
                if verify_clauses is not None:
                    for r in full_rows.cpu().tolist():
                        if not _satisfies(r, verify_clauses):
                            all_valid = False
                if collect:
                    models.extend(full_rows.cpu().tolist())
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

    Returns (nv2, clauses2, support2, old_of_new, forced) or None if the formula is UNSAT:
      nv2        : compacted variable count.
      clauses2   : simplified clauses over compacted vars (1-based).
      support2   : 0-based support vars surviving in the compacted formula.
      old_of_new : list; old_of_new[i] = original DIMACS var of compacted var i+1.
      forced     : DIMACS lits fixed by unit propagation (must hold in every full model).
    Vars that are neither forced nor present in any remaining clause are true don't-cares (a
    support var of this kind is reported in `support2` mapping as absent → caller sets it freely).
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
    return len(used), clauses2, support2, used, forced


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
