"""Dynamic in-solve inprocessing: drive CaDiCaL 1.9.5 with GPAP via PySAT's IPASIR-UP.

`solve_with_gpu_inprocessing(cnf_path, ...)` runs CaDiCaL in conflict-budgeted segments. An
attached `ImportanceObserver` (an IPASIR-UP `Propagator`) watches assignments to an observed
variable pool and ranks variables by how often the solver *decides* on them (a VSIDS proxy
via `solver.is_decision`). Between segments, GPAP (`GPUInprocessor`) discovers entailed short
clauses — failed-literal units and binary/ternary clauses over the currently-important
variables — which are injected with `solver.add_clause`. Every injected clause is RUP
(entailed by the original formula), so satisfiability is preserved; the clauses only reshape
the search.

The GPAP formula + engine are built once and reused (immutable formula); each round only
does reset+apply+propagate over millions of assumption sets in parallel.
"""
import time
from collections import defaultdict

import torch

from pysat.formula import CNF
from pysat.solvers import Cadical195
from pysat.engines import Propagator

from gpap.solver.inprocess import GPUInprocessor


class ImportanceObserver(Propagator):
    """IPASIR-UP observer: tallies per-variable activity to rank "important" variables.

    Observes a variable pool (set by the driver via `solver.observe`). For each assignment
    to an observed variable it increments an activity counter, and — using `is_decision` —
    a decision counter that approximates VSIDS branching activity. All reasoning callbacks
    are inert (we never propagate or reject models); we only collect telemetry. Clause
    injection is done by the driver between segments via `solver.add_clause`, not through the
    `add_clause` callback (which stays empty for the v1 segmented loop).
    """

    def __init__(self, solver_ref=None, use_is_decision: bool = True):
        super().__init__()
        self.solver = solver_ref
        self.use_is_decision = use_is_decision
        self.assign_count: dict[int, int] = defaultdict(int)
        self.decision_count: dict[int, int] = defaultdict(int)
        self.n_assign = 0

    # --- telemetry -------------------------------------------------------
    def on_assignment(self, lit, fixed=False):
        v = lit if lit > 0 else -lit
        self.assign_count[v] += 1
        self.n_assign += 1
        if self.use_is_decision and self.solver is not None:
            try:
                if self.solver.is_decision(lit):
                    self.decision_count[v] += 1
            except Exception:        # noqa: BLE001 — fall back to activity-only if unsupported
                self.use_is_decision = False

    def on_new_level(self):
        pass

    def on_backtrack(self, to):
        pass

    # --- inert reasoning callbacks (safe returns for the C layer) --------
    def check_model(self, model):
        return True                  # never reject: our clauses are RUP, any model is valid

    def decide(self):
        return 0                     # let CaDiCaL branch

    def propagate(self):
        return []                    # we infer nothing in-solve

    def provide_reason(self, lit):
        return []                    # never invoked (we never propagate)

    def add_clause(self):
        return []                    # injection happens between segments, not via callback


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _compute_degree(cnf: CNF) -> dict[int, int]:
    """Variable occurrence degree (count of clauses mentioning the variable, either polarity)."""
    deg: dict[int, int] = defaultdict(int)
    for clause in cnf.clauses:
        for lit in clause:
            deg[abs(lit)] += 1
    return deg


def _rank_vars(pool_vars, obs: ImportanceObserver, degree: dict[int, int],
               source: str, k: int) -> list[int]:
    """Return the top-k important variables (1-based) from the observed pool.

    decisions: VSIDS proxy (decision_count), degree as tiebreak.
    activity:  assignment frequency, degree as tiebreak.
    degree:    static occurrence degree only.
    blend:     decision_count primary, degree tiebreak (default) — decisions drive the
               per-round drift; degree seeds variables the solver hasn't branched on yet.
    """
    if source == "degree":
        key = lambda v: (degree.get(v, 0),)
    elif source == "activity":
        key = lambda v: (obs.assign_count.get(v, 0), degree.get(v, 0))
    elif source == "decisions":
        key = lambda v: (obs.decision_count.get(v, 0), degree.get(v, 0))
    else:  # blend
        key = lambda v: (obs.decision_count.get(v, 0), degree.get(v, 0))
    return sorted(pool_vars, key=key, reverse=True)[:k]


def _round_gpu_ms(dstats: dict) -> float:
    return sum(s["ms"] for s in dstats.values() if isinstance(s, dict) and "ms" in s)


def solve_with_gpu_inprocessing(
    cnf_path: str,
    *,
    device: torch.device | None = None,
    mode: str = "pairs",                 # probe | pairs | triples
    top_k: int = 64,
    triples_k: int = 24,
    segment_conflicts: int = 20000,
    max_clauses_per_round: int = 2000,
    max_rounds_total: int | None = None,
    time_budget_s: float | None = None,
    observe_pool: int | None = None,
    importance: str = "blend",           # blend | decisions | activity | degree
    patience: int = 3,
    gpap_max_rounds: int | None = None,
    gpap_soft_stop_frac: float | None = None,
    gpap_min_rounds: int = 0,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Solve `cnf_path` with CaDiCaL + GPU-discovered clause injection. Returns a stats dict."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.perf_counter()

    cnf = CNF(from_file=cnf_path)
    N = cnf.nv

    # GPAP discovery engine over the ORIGINAL formula — built once, reused every round.
    t_build0 = time.perf_counter()
    inproc = GPUInprocessor.from_cnf_file(
        cnf_path, device, max_rounds=gpap_max_rounds,
        soft_stop_frac=gpap_soft_stop_frac, min_rounds=gpap_min_rounds,
    )
    build_ms = (time.perf_counter() - t_build0) * 1e3

    # Observed variable pool: top-degree seed (bounded — full coverage would suppress
    # CaDiCaL's variable elimination and add per-assignment callback overhead).
    degree = _compute_degree(cnf)
    pool_size = observe_pool if observe_pool is not None else min(N, max(8 * top_k, 512))
    pool_vars = sorted(range(1, N + 1), key=lambda v: degree.get(v, 0), reverse=True)[:pool_size]

    solver = Cadical195(bootstrap_with=cnf, use_timer=True)
    obs = ImportanceObserver(solver_ref=solver)
    # Only attach the propagator when we actually observe variables — an idle propagator
    # still costs per-decision overhead in CaDiCaL. With no pool, importance falls back to
    # static degree and discovery (e.g. probe) runs without any callback cost.
    use_observer = len(pool_vars) > 0
    if use_observer:
        solver.connect_propagator(obs)
        for v in pool_vars:
            solver.observe(v)
        solver.enable_propagator()

    if verbose:
        print(f"  [gpu-inproc] N={N:,} pool={len(pool_vars)} K={inproc.K:,} "
              f"mode={mode} top_k={top_k} seg={segment_conflicts:,} imp={importance}")

    rounds = 0
    n_added = n_units = n_bins = n_terns = 0
    gpu_ms = 0.0
    empty_streak = 0
    inprocessing_active = True
    res = None

    while True:
        if time_budget_s is not None and (time.perf_counter() - t_start) > time_budget_s:
            break
        solver.conf_budget(segment_conflicts)
        res = solver.solve_limited()
        if res is not None:
            break

        if not inprocessing_active:
            # Discovery exhausted. With no wall budget to enforce, finish in one unbudgeted
            # solve (avoids repeated solve_limited segment overhead); otherwise keep segmenting.
            if time_budget_s is None:
                res = solver.solve()
                break
            continue

        rounds += 1
        do_probe = (rounds == 1)          # formula is fixed → probing again finds nothing new
        pairs_vars = triples_vars = None
        if mode in ("pairs", "triples"):
            pairs_vars = [v - 1 for v in _rank_vars(pool_vars, obs, degree, importance, top_k)]
        if mode == "triples":
            triples_vars = [v - 1 for v in _rank_vars(pool_vars, obs, degree, importance, triples_k)]

        clauses, dstats = inproc.run(
            probe=do_probe, pairs_vars=pairs_vars, triples_vars=triples_vars,
            max_clauses=max_clauses_per_round,
        )
        gpu_ms += _round_gpu_ms(dstats)

        # inproc.run already deduped vs its own history → every clause here is new.
        for c in clauses:
            solver.add_clause(c)
            L = len(c)
            n_units += (L == 1)
            n_bins += (L == 2)
            n_terns += (L == 3)
        n_added += len(clauses)

        if verbose and clauses:
            print(f"    round {rounds}: +{len(clauses)} clauses "
                  f"(u={sum(1 for c in clauses if len(c)==1)} "
                  f"b={sum(1 for c in clauses if len(c)==2)} "
                  f"t={sum(1 for c in clauses if len(c)==3)})  gpu={_round_gpu_ms(dstats):.0f}ms")

        empty_streak = empty_streak + 1 if not clauses else 0
        if empty_streak >= patience or (max_rounds_total is not None and rounds >= max_rounds_total):
            inprocessing_active = False    # stop discovery
            if time_budget_s is None:      # finish immediately in one unbudgeted solve
                res = solver.solve()
                break

    wall_ms = (time.perf_counter() - t_start) * 1e3
    solve_ms = solver.time_accum() * 1e3   # CaDiCaL-internal time across all solve calls
    result = "UNKNOWN" if res is None else ("SAT" if res else "UNSAT")
    model = solver.get_model() if res else None
    stats = solver.accum_stats()
    if use_observer:
        solver.disconnect_propagator()

    return {
        "result": result,
        "model": model,
        "wall_ms": wall_ms,
        # Time split so the GPU-discovery cost (A100/kernel-improvable) is separable from the
        # actual solving benefit. wall_no_gpu = what the wall would be if GPU discovery were free.
        "solve_ms": solve_ms,
        "build_ms": build_ms,
        "wall_no_gpu_ms": wall_ms - gpu_ms,
        "solver_stats": stats,
        "rounds": rounds,
        "clauses_added": n_added,
        "units_added": n_units,
        "binaries_added": n_bins,
        "ternaries_added": n_terns,
        "gpu_ms": gpu_ms,
        "K": inproc.K,
        "mode": mode,
        "top_k": top_k,
        "segment_conflicts": segment_conflicts,
        "importance": importance,
        "observe_pool": len(pool_vars),
        "obs_decisions_tracked": obs.use_is_decision,
    }


def solve_plain(cnf_path: str) -> dict:
    """Baseline: plain CaDiCaL solve to completion. Returns the same-shaped stats subset."""
    t0 = time.perf_counter()
    cnf = CNF(from_file=cnf_path)
    with Cadical195(bootstrap_with=cnf, use_timer=True) as solver:
        sat = solver.solve()
        stats = solver.accum_stats()
        solve_ms = solver.time_accum() * 1e3
        model = solver.get_model() if sat else None
    return {
        "result": "SAT" if sat else "UNSAT",
        "model": model,
        "wall_ms": (time.perf_counter() - t0) * 1e3,
        "solve_ms": solve_ms,
        "solver_stats": stats,
    }


def score_variables_by_decisions(cnf_path: str, warmup_conflicts: int = 8000,
                                 observe_cap: int | None = None):
    """Harvest a VSIDS proxy: per-variable branch-decision counts from a warm-start solve.

    Runs CaDiCaL for `warmup_conflicts` conflicts with an IPASIR-UP observer over the variable
    pool, counting how often each variable is a DECISION literal (``is_decision``) — i.e. picked
    by CaDiCaL's branching (VSIDS). This is the cleanest activity signal PySAT exposes (no direct
    per-variable score accessor). Returns (decision_count, assign_count, degree, solved), each a
    dict[var(1-based)] -> int. `observe_cap` bounds the observed pool to the top-degree vars to
    cap callback overhead on huge formulas (None = observe all).
    """
    cnf = CNF(from_file=cnf_path)
    N = cnf.nv
    degree = _compute_degree(cnf)
    pool = (sorted(range(1, N + 1), key=lambda v: degree.get(v, 0), reverse=True)[:observe_cap]
            if observe_cap else list(range(1, N + 1)))
    solver = Cadical195(bootstrap_with=cnf, use_timer=True)
    obs = ImportanceObserver(solver_ref=solver)
    solver.connect_propagator(obs)
    for v in pool:
        solver.observe(v)
    solver.enable_propagator()
    solver.conf_budget(warmup_conflicts)
    res = solver.solve_limited()
    solver.disconnect_propagator()
    solver.delete()
    return dict(obs.decision_count), dict(obs.assign_count), degree, (res is not None)
