"""Entailed-clause mining: GPAP probing the formula against ITSELF before sampling it.

For low-success instances (the "service" family: support vars individually free but jointly
constrained through deep cascades), random completions die from LATE BCP conflicts — restarts
don't pay (measured: retry tripled success AND wall), but CLAUSES do: every mined clause is
checked by BCP once and then prunes doomed trajectories EARLY in every one of the millions of
subsequent kernel trajectories.

Tiers (all exhaustive GPU BCP probes on the original formula, PyTorch engine path for
`real_conflict` — trivial conflicts are subsumed by an existing clause and skipped):
  L1  all 2N literals, iterated to fixpoint: assuming l real-conflicts => unit (~l). This finds
      the BCP-visible BACKBONE — measured hundreds of units on every service instance, the
      single biggest lever (units also shrink the formula through the sampler's simplify).
  L2  all support pairs (a,b): conflict => binary (~a | ~b).
  L3  all support triplets: conflict => ternary clause.

Every learned clause is ENTAILED by the formula, so the model set is EXACTLY unchanged —
sampling the augmented formula is sound by construction (and CPU-verified in tests against the
original clauses).

Measured (RTX 4080, 2026-07-03, vs CMSGen reference): flips 7 of 8 losing instances to wins —
ActivityService2 0.02x -> 1.13x, IterationService 0.12x -> 2.10x, PhaseService 0.07x -> 1.12x,
UserServiceImpl 0.32x -> 6.68x, ProjectService3 0.38x -> 4.24x, ProcessBean 0.70x -> 2.56x,
c3540 0.72x -> 1.82x. Mining cost 1-10s one-time. (NotificationServiceImpl2 stays a loss:
0.00x -> ~0.16x; its joint constraint is beyond ternary reach — deeper tiers and full lookahead
both raise success but not throughput.)
"""
import itertools
import time

import torch

from gpap.clsdata.cnf_formula import CnfFormula
from gpap.solver.bcp_engine import BCPEngine


def _lit_idx(ext: int) -> int:
    return 2 * (abs(ext) - 1) + (0 if ext > 0 else 1)


def _idx_lit(i: int) -> int:
    return (i // 2 + 1) * (1 if i % 2 == 0 else -1)


class _Prober:
    """BCP-probe assumption sets against one formula, preferring the fused Triton kernel.

    The kernel path (one launch per chunk, fixed-size state, no scatter temporaries) is ~10x
    faster than the PyTorch engine AND avoids the WSL2 driver watchdog that giant propagate
    temporaries trip ("device not ready"). The post-units-flush state is cached as single rows
    and broadcast-restored per chunk, so the Python-round-loop flush runs ONCE per formula.

    Conflict semantics: candidates are pre-filtered to literals UNASSIGNED after the flush and
    to sets not already subsumed by an existing <=3-clause, so every reported conflict is a
    genuine BCP cascade => its negation clause is entailed and non-trivial."""

    def __init__(self, num_vars: int, clauses: list, device: torch.device):
        f = CnfFormula(device)
        f.init_formula(num_vars, clauses)
        self.eng = BCPEngine(f)
        if device.type == "cuda":
            from gpap.solver.bcp_triton import TritonBCP
            if TritonBCP.available():
                self.eng.attach_kernel(TritonBCP())
        self.eng.reset_state(1)
        self.eng.propagate()                               # units flush (once)
        self._snapshot()

    def _snapshot(self):
        self.unsat = bool(self.eng.conflict[0].item())
        self.asg_row = self.eng._assignment[0].clone()
        self.sat_row = self.eng._sat_count[0].clone()
        self.lsum_row = self.eng._lit_sum[0].clone()
        self.asg_cpu = self.asg_row.cpu()

    def assume_units(self, lit_indices: list[int]) -> None:
        """Fold newly-learned units into the cached flush state INCREMENTALLY (propagate just
        them on top of the current cache at K=1). Avoids rebuilding the formula/occ lists per
        L1 iteration — the rebuild dominated mining wall on large unrolled circuits (measured
        532s -> seconds). Sound: the units are entailed, so the narrowed state is exactly the
        fixpoint the rebuilt formula's flush would reach."""
        if not lit_indices or self.unsat:
            return
        eng = self.eng
        eng.reset_state(1)
        eng._sat_count.copy_(self.sat_row.unsqueeze(0))
        eng._lit_sum.copy_(self.lsum_row.unsqueeze(0))
        eng._assignment.copy_(self.asg_row.unsqueeze(0))
        t = torch.tensor([lit_indices], dtype=torch.long, device=eng._device)
        eng.apply_and_propagate(t)
        self._snapshot()

    def unassigned(self, lit: int) -> bool:
        """True iff the flush left this lit-index (and its negation) free."""
        return bool(self.asg_cpu[lit] == 0) and bool(self.asg_cpu[lit ^ 1] == 0)

    def run(self, sets: list[list[int]], k_cap: int) -> torch.Tensor:
        out = []
        eng = self.eng
        for i in range(0, len(sets), k_cap):
            chunk = sets[i:i + k_cap]
            K = len(chunk)
            L = max(len(s) for s in chunk)
            t = torch.full((K, L), -1, dtype=torch.long)
            for k, s in enumerate(chunk):
                t[k, :len(s)] = torch.tensor(s, dtype=torch.long)
            eng.reset_state(K)                             # right-sized buffers...
            eng._sat_count.copy_(self.sat_row.unsqueeze(0).expand(K, -1))   # ...flushed state
            eng._lit_sum.copy_(self.lsum_row.unsqueeze(0).expand(K, -1))
            eng._assignment.copy_(self.asg_row.unsqueeze(0).expand(K, -1))
            eng.apply_and_propagate(t.to(eng._device))     # ONE kernel launch (or torch fallback)
            out.append(eng.conflict.clone())
            if eng._device.type == "cuda":
                torch.cuda.synchronize()                   # keep the async queue shallow
        return torch.cat(out)


def mine_entailed(num_vars: int, clauses: list, support: list | None, device: torch.device,
                  *, k_cap: int = 8192, triplets: bool = True,
                  max_probes: int = 2_000_000, max_seconds: float = 60.0) -> dict:
    """Mine entailed clauses (see module docstring). `support` is 0-based var indices (the
    sampler's convention); L2/L3 probe support tuples only (the jointly-constrained set).
    Returns {"learned": [ext-lit clauses], "n_units", "n_binary", "n_ternary", "seconds"}.

    Failure-proof by design: mining is an OPTIMIZATION — any CUDA error mid-tier (observed:
    WSL2 watchdog "device not ready" under a heavy probe cascade) abandons the remaining tiers
    and returns whatever was already learned (each clause independently entailed)."""
    t0 = time.perf_counter()
    learned: list[list[int]] = []
    existing = {tuple(sorted(c)) for c in clauses if len(c) <= 3}
    cur = [list(c) for c in clauses]

    def add(cl: tuple) -> bool:
        if cl in existing:
            return False
        learned.append(list(cl))
        cur.append(list(cl))
        existing.add(cl)
        return True

    n_units = n_binary = n_ternary = 0
    prober = None

    def over_budget() -> bool:
        return time.perf_counter() - t0 > max_seconds

    try:
        # L1: failed literals, to fixpoint (new units enable new failures). The prober is built
        # ONCE; learned units fold into its cached flush state incrementally (assume_units) —
        # no per-iteration formula rebuild.
        prober = _Prober(num_vars, cur, device)
        while not prober.unsat and not over_budget():
            cand = [l for l in range(2 * num_vars) if prober.unassigned(l)]
            if not cand:
                break
            conf = prober.run([[l] for l in cand], k_cap)
            fresh = []
            for i in conf.nonzero().view(-1).tolist():
                u = -_idx_lit(cand[i])
                if add((u,)):
                    fresh.append(_lit_idx(u))
            if not fresh:
                break
            prober.assume_units(fresh)
        n_units = len(learned)

        sup = support or []
        if prober is None or prober.unsat or over_budget():
            sup = []

        def free2(ext_a: int, ext_b: int) -> bool:
            return prober.unassigned(_lit_idx(ext_a)) and prober.unassigned(_lit_idx(ext_b))

        # L2: support pairs
        pairs = []
        for va, vb in itertools.combinations(sup, 2):
            for sa, sb in itertools.product((0, 1), repeat=2):
                a, b = _idx_lit(2 * va + sa), _idx_lit(2 * vb + sb)
                if free2(a, b) and tuple(sorted((-a, -b))) not in existing:
                    pairs.append([_lit_idx(a), _lit_idx(b)])
        if pairs and len(pairs) <= max_probes:
            conf = prober.run(pairs, k_cap)
            for i in conf.nonzero().view(-1).tolist():
                add(tuple(sorted(-_idx_lit(x) for x in pairs[i])))
        n_binary = len(learned) - n_units

        # L3: support triplets (skip any triplet a learned/existing unit or pair already covers;
        # rebuild the prober iff L2 learned something, so its formula includes the new binaries)
        n_before = len(learned)
        if triplets and sup and not over_budget():
            if n_binary:
                prober = _Prober(num_vars, cur, device)
            trips = []
            for vs in itertools.combinations(sup, 3):
                for ss in itertools.product((0, 1), repeat=3):
                    lits = [_idx_lit(2 * v + s) for v, s in zip(vs, ss)]
                    neg = sorted(-x for x in lits)
                    if (all(prober.unassigned(_lit_idx(x)) for x in lits)
                            and tuple(sorted((neg[0], neg[1]))) not in existing
                            and tuple(sorted((neg[0], neg[2]))) not in existing
                            and tuple(sorted((neg[1], neg[2]))) not in existing):
                        trips.append([_lit_idx(x) for x in lits])
            if trips and len(trips) <= max_probes:
                conf = prober.run(trips, k_cap)
                for i in conf.nonzero().view(-1).tolist():
                    add(tuple(sorted(-_idx_lit(x) for x in trips[i])))
        n_ternary = len(learned) - n_before
    except RuntimeError as e:  # CUDA hiccup mid-tier: keep what's already (independently) proven
        print(f"   [miner] aborting remaining tiers after CUDA error: {e}", file=__import__('sys').stderr)
    finally:
        # Release the probe engines' VRAM: reserved-but-free memory would otherwise poison the
        # rebuilt sampler's K calibration (free-memory read) — measured K collapsing to 1.
        prober = None
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {"learned": learned, "n_units": n_units, "n_binary": n_binary,
            "n_ternary": n_ternary, "seconds": time.perf_counter() - t0}
