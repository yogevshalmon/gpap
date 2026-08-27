"""Entailed-clause mining: prove small clauses the formula already implies, before solving it.

An application of `gpap.engine.probe` — nothing here knows about sampling. If assuming a set of
literals drives BCP to a conflict, the formula entails the negation of their conjunction, which
is a clause. Probing every k-subset of a literal pool therefore yields every entailed clause of
width k that BCP alone can see:

    k = 1   assuming l conflicts  =>  unit (~l)          the BCP-visible BACKBONE
    k = 2   assuming a,b conflicts =>  (~a | ~b)
    k = 3   ...and so on

Every clause returned is ENTAILED, so adding it to the formula leaves the model set EXACTLY
unchanged. That is what makes mining safe to apply before any downstream use: it can only make
propagation stronger, never change the answer. (Verified in tests against the original clauses,
not merely argued.)

    from gpap.mining import mine_entailed
    result = mine_entailed(num_vars, clauses, device)
    stronger = clauses + result.clauses

Why it pays: a mined clause is proven once and then prunes doomed search EARLY in every
subsequent propagation. The measured effect on solution sampling, and the tier-by-tier cost
breakdown, are in docs/findings/MINING_FINDINGS.md — results belong there, not in this docstring.
"""
from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from gpap.engine.probe import LiteralProbe, ProbeBudget

log = logging.getLogger(__name__)

__all__ = ["mine_entailed", "MiningResult", "DROP_LEVELS", "DEFAULT_WIDTHS",
           "DEFAULT_MAX_TUPLES"]

#: Widths mined by default: units, then binaries, then ternaries.
DEFAULT_WIDTHS = (1, 2, 3)

#: Per-width WORK BUDGET: how many candidates a width may probe. C(P,k) grows fast enough that a
#: wide pool at k=3 is hours of GPU work — and the cost is the PROBE, not the enumeration (probing
#: n tuples initializes n x M clause words). A width over its budget probes that many and reports
#: itself NOT exhaustive; the clauses it found are still entailed, so partial results are kept
#: rather than thrown away. An int applies to every width; a dict gives per-width budgets, which
#: is what you want when the widths cost wildly different amounts.
DEFAULT_MAX_TUPLES = 200_000


@dataclass
class MiningResult:
    """What was proven, and whether the search was complete.

    `clauses` are DIMACS clauses, each independently entailed by the input formula.
    `exhausted[k]` says width k was searched EXHAUSTIVELY — i.e. no entailed clause of that width
    escaped, which is a stronger statement than "we found these".
    """

    clauses: list[list[int]] = field(default_factory=list)
    by_width: dict[int, int] = field(default_factory=dict)
    exhausted: dict[int, bool] = field(default_factory=dict)
    skipped: dict[int, str] = field(default_factory=dict)
    #: candidates the full enumeration HAS, and how many were actually probed, per width. Without
    #: these a caller cannot tell "nothing is entailed here" from "we barely started" — the two
    #: look identical in `by_width` and mean opposite things.
    candidates: dict[int, int] = field(default_factory=dict)
    probed: dict[int, int] = field(default_factory=dict)
    #: sweeps run per width. 1 for every width >= 2; width 1 is iterated to a FIXPOINT, so its
    #: `candidates`/`probed` are SUMS over this many rounds and are NOT comparable to the pool
    #: size — the pool shrinks each round, and proving convergence always costs one final round
    #: that finds nothing. Without this the totals read as an impossible over-count of the pool.
    rounds: dict[int, int] = field(default_factory=dict)
    #: BCP implications the engine derived across every sweep. The hardware-work metric: unlike
    #: clauses/s it does not depend on how many clauses happened to be entailed, so it is what to
    #: compare across instances, GPUs or engine changes.
    implications: int = 0
    #: clauses removed because propagation already derives them (`drop="derivable"` only).
    derivable: int = 0
    seconds: float = 0.0

    @property
    def implications_per_s(self) -> float:
        return self.implications / self.seconds if self.seconds else 0.0

    def __len__(self) -> int:
        return len(self.clauses)

    @property
    def units(self) -> list[list[int]]:
        return [c for c in self.clauses if len(c) == 1]

    def summary(self) -> str:
        """One line: what was proven, how much was probed, how fast.

        The probed count is a plain total; it is deliberately NOT printed as a fraction of the
        candidate space. Summing `candidates` across widths adds a width-1 count that is itself a
        sum over fixpoint rounds to a width-2 count of C(P,2), which produced lines like
        a ratio of two incomparable things. Per-width coverage is exact and belongs per width
        (see `candidates` / `probed` / `rounds`)."""
        parts = [f"{n} of width {k}" for k, n in sorted(self.by_width.items()) if n]
        done = sum(self.probed.values())
        cover = (f", probed {done:,} tuple(s)" if done else "")
        rate = (f" ({self.implications_per_s / 1e9:.2f}G implications/s)"
                if self.implications else "")
        return (", ".join(parts) or "nothing") + cover + f" in {self.seconds:.2f}s" + rate


#: What `drop=` may remove, weakest first. Each level implies the one before it.
DROP_LEVELS = ("none", "subsumed", "derivable")


def mine_entailed(num_vars: int, clauses: list, device, *, pool: list | None = None,
                  widths=DEFAULT_WIDTHS, seconds: float | None = 5.0,
                  max_tuples: "int | dict[int, int]" = DEFAULT_MAX_TUPLES,
                  min_units_for_wide: int = 0, drop: str = "subsumed",
                  on_plan=None) -> MiningResult:
    """Mine entailed clauses of the given `widths` and return them.

    `pool`: DIMACS LITERALS to combine at widths >= 2, and the natural knob — the interesting
      joint constraints usually live in a known subset (an independent support, a set of primary
      inputs). None means every free literal. Pass both polarities of a variable to ask about it
      in general, or one to ask about a single side. Width 1 always sweeps ALL free literals: the
      backbone is cheap and is the dominant win.
    `seconds`: wall-clock budget across all widths (None = unbounded). Binds between chunks.
    `max_tuples`: per-width work budget — an int for every width, or {width: budget}. A width
      above it probes that many candidates and is reported as not exhaustive (see
      `MiningResult.candidates` / `.probed` for the coverage).
    `min_units_for_wide`: run widths >= 2 only if width 1 proved at least this many units. The
      wide tiers are the expensive ones and on many formulas they find nothing, so a caller that
      wants mining to be nearly free on barren inputs sets this to 1.
    `drop`: which REDUNDANT clauses to leave out. Three levels, each stronger than the last:
      "none"       keep everything proved (for analysis).
      "subsumed"   (default) leave out a clause some clause already covers — a subset of it is
                   present in the formula the sweep ran against, so it constrains nothing new.
                   Free: a set test over the handful of clauses found.
      "derivable"  ALSO leave out clauses propagation re-derives on demand (assume all but one
                   literal, and the last one is forced). This is the probing literature's notion
                   for hyper-binary resolvents, and it is far stronger: on b03-c25 it removes 96%
                   where "subsumed" removes 0%. Costs one extra batched launch (~40-70% of mining
                   wall). ⚠️ NOT the default, deliberately: a derivable clause still SHORTENS
                   propagation, and GPAP's own sampler is propagation-bound — keeping them is
                   measured at +29-43% end-to-end. Drop them for a consumer that wants a minimal
                   formula (a CDCL preprocessor), not for one that wants speed.

    `on_plan`: called as on_plan(width, candidates, ceiling) for each width BEFORE any probing,
      so a front-end can show what a sweep is about to cost while it is still cheap to abort.
      It lives here because the count needs a built engine: a caller doing this for itself had to
      construct a second `LiteralProbe` over the same formula and throw it away to obtain numbers
      this one already has for free.

    Width 1 is iterated to a FIXPOINT: each proven unit narrows the formula, which can expose
    further failed literals. Later widths run once, on the narrowed formula.
    """
    if drop not in DROP_LEVELS:
        raise ValueError(f"drop must be one of {DROP_LEVELS}, got {drop!r}")
    t0 = time.perf_counter()
    res = MiningResult()
    sub = _Subsumption(max(widths, default=0))
    budget = ProbeBudget(deadline=t0 + seconds if seconds is not None else None)
    current = list(clauses)          # original + everything proven so far

    def keep(clause: tuple, check: bool = True) -> bool:
        """Record a clause unless something already known SUBSUMES it (see `_Subsumption`).

        `check=False` skips the test for a caller that has already PROVEN it cannot fire. Width 1
        is the case: a mined unit (u) comes from probing ~u, and ~u was probed only because it was
        FREE — so neither u nor ~u was settled by the flush. Had (u) been a formula unit, or a unit
        mined in an earlier fixpoint round (those are folded in by `assume`), the flush would have
        set u and ~u would not have been free. Hence no width-1 unit can be subsumed by anything
        known, and the test is vacuous rather than merely cheap. Verified: the overlap between
        formula units and mined units is 0 on b03-c25, s386-c25 and b09-c25. (It is worth ~0.1% of
        mining wall — recorded for the invariant, not the speed; the recording still happens
        because later widths need it.)

        It runs on the conflicts the sweep REPORTED, never on candidates. Its predecessor filtered
        CANDIDATES and cost 65s to drop 0 of 122M.
        """
        if check and drop != "none" and sub.subsumes(clause):
            return False
        sub.add(clause)
        res.clauses.append(list(clause))
        current.append(list(clause))
        return True

    probe = None
    try:
        probe = LiteralProbe(num_vars, clauses, device)
        if on_plan is not None:
            # UPPER BOUNDS, and exact only for the first width run: width 1's proven units narrow
            # the formula, so a later width sees a smaller free pool than counted here. Width 1's
            # own count is ONE round of its fixpoint, not the total it will probe.
            for width in sorted(widths):
                on_plan(width, probe.count(pool if width > 1 else None, width),
                        _ceiling(max_tuples, width))
        for width in sorted(widths):
            if probe.unsat:
                res.skipped[width] = "formula is unsat"
                continue
            if budget.deadline is not None and time.perf_counter() > budget.deadline:
                res.skipped[width] = "budget"
                continue
            if width > 1 and res.by_width.get(1, 0) < min_units_for_wide:
                res.skipped[width] = "too few units at width 1"
                continue
            # The subsumption reference is the formula REDUCED by the baseline the probe is about
            # to run on, so it is rebuilt whenever that baseline moves — which `assume` (inside the
            # width-1 fixpoint) and a formula rebuild both do. Width 1 never consults it (its
            # clauses provably cannot be subsumed, see `keep`), so it does not pay for the pass.
            if width > 1:
                sub.rebuild(current, num_vars, probe.assigned_literals())
            ceil_w = _ceiling(max_tuples, width)
            w = (_mine_units(probe, budget, keep, ceil_w) if width == 1 else
                 _mine_width(probe, width, pool, budget, ceil_w, keep))
            found = w.found
            res.by_width[width] = w.found
            res.exhausted[width] = w.exhausted
            res.candidates[width], res.probed[width] = w.candidates, w.probed
            res.rounds[width] = w.rounds
            if not w.exhausted:
                # ONE message for both widths, from the probe's OWN stop reason. Asserting
                # "(max_tuples)" here named a bound that may never have been the binding one:
                # width 1 is just as often cut by the shared wall-clock deadline, and a user told
                # to raise a budget they had already set to unbounded has been sent nowhere.
                res.skipped[width] = (
                    f"probed {w.probed:,} of {w.candidates:,} candidates "
                    f"({w.stopped or 'incomplete'}) — clauses found are still entailed, but this "
                    f"width was NOT searched exhaustively")
            # Each width STRENGTHENS the formula the next one probes: a clause proven at width 2
            # can make a width-3 assumption conflict that would not have conflicted without it.
            # Units are folded in place by `assume`; wider clauses need the formula rebuilt, which
            # is why this only fires when a wide tier actually proved something. Skipping it costs
            # real clauses — measured 4 ternaries instead of 7 on an unrolled b03.
            if found and width > 1 and width < max(widths):
                res.implications += probe.implications      # bank before the rebuild resets them
                probe = LiteralProbe(num_vars, current, device)
    except RuntimeError as e:
        # Mining is an OPTIMIZATION: every clause already proven stays valid, so a mid-tier CUDA
        # failure abandons the rest rather than failing the caller's whole job.
        log.warning("mining stopped after a device error, keeping %d proven clause(s): %s",
                    len(res.clauses), e)
        res.skipped[-1] = f"device error: {e}"
    finally:
        if probe is not None:
            res.implications += probe.implications
        del probe
        _release(device)

    if drop == "derivable" and res.clauses:
        _drop_derivable(res, num_vars, clauses, device)
    res.seconds = time.perf_counter() - t0
    return res


def _drop_derivable(res: MiningResult, num_vars: int, clauses: list, device) -> None:
    """Remove the clauses propagation already derives, judged against what the caller will HAVE.

    That is the input formula plus the mined UNITS — not the input alone. A caller takes the whole
    result, so the units are part of the context a wider clause has to earn its place against;
    judging against the input alone understates the redundancy (on b03-c25, 53% vs 96%)."""
    from gpap.implications import derivable_mask

    units = [c for c in res.clauses if len(c) == 1]
    mask = derivable_mask(res.clauses, list(clauses) + units, num_vars, device)
    kept = [c for c, red in zip(res.clauses, mask) if not red]
    res.derivable = int(mask.sum())
    res.clauses = kept
    by_width: dict[int, int] = {}
    for c in kept:
        by_width[len(c)] = by_width.get(len(c), 0) + 1
    for w in res.by_width:
        res.by_width[w] = by_width.get(w, 0)


class _Subsumption:
    """The clauses a mined one must not repeat — as REDUCED by the baseline the probe runs on.

    Why not just the input clauses: mining folds every proven unit into the engine's flush cache
    (`assume`), so a sweep propagates on top of an assignment, and a conflict proves a clause
    relative to THAT. An input clause (a | b | c) whose `c` the baseline falsified is a BINARY
    constraint during the sweep, so assuming (~a, ~b) empties it and "proves" (a | b) — entailed,
    but redundant: any consumer that takes the mined units too will derive it by propagation. It
    is not subsumed by any INPUT clause, which is why testing against those let 17 of them through
    on b03-c25.

    That also makes the answer DETERMINISTIC. There used to be a second filter inside the engine
    (`skip_subsumed`) that read the implication counter, and on a conflicted row that counter is
    order-dependent — wave can derive implications in the same warp-wide step that empties the
    clause — so scalar dropped 17 where wave dropped 0. Same soundness either way, different
    output, and a library whose results move with a launch flag is a library that cannot be gated.
    It was deleted (it was worth 0.0-0.3% of the result buffer): redundancy is a property of a
    CLAUSE against a FORMULA, so it belongs on the host, where the formula is known. This is the
    only filter, and it is exact.
    """

    def __init__(self, max_width: int):
        self._max_width = int(max_width)
        self.known: set[tuple] = set()

    def rebuild(self, clauses: list, num_vars: int, assigned: np.ndarray) -> None:
        """Recompute the reference: `clauses` reduced by the baseline `assigned` (DIMACS literals).

        Vectorised, because it runs once per width on the whole formula — the obvious per-clause
        Python loop is ~0.5 s on a 219k-clause unroll, which would be a larger cost than the sweep
        it protects."""
        self.known = set()
        lens = np.fromiter((len(c) for c in clauses), dtype=np.int64, count=len(clauses))
        if lens.size == 0 or lens.sum() == 0:
            return
        flat = np.fromiter(itertools.chain.from_iterable(clauses), dtype=np.int32,
                           count=int(lens.sum()))
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
        val = np.zeros(int(num_vars) + 1, dtype=np.int8)
        if assigned.size:
            val[np.abs(assigned)] = np.sign(assigned).astype(np.int8)
        state = val[np.abs(flat)] * np.sign(flat).astype(np.int8)     # +1 satisfied, -1 falsified
        free = state == 0
        sat = np.add.reduceat((state > 0).astype(np.int32), starts) > 0
        width = np.add.reduceat(free.astype(np.int32), starts)
        # A satisfied clause constrains nothing, and a residue wider than the widest width mined
        # cannot subsume anything that could be emitted.
        take = ~sat & (width >= 1) & (width <= self._max_width) & (lens > 0)
        if not take.any():
            return
        pos = np.flatnonzero(free)                                    # free literal positions
        owner = np.searchsorted(starts, pos, side="right") - 1        # ...and their clause
        rank = np.arange(pos.size) - np.searchsorted(owner, owner, side="left")
        sel = take[owner]
        rows = np.zeros((len(clauses), self._max_width), dtype=np.int32)
        rows[owner[sel], rank[sel]] = flat[pos[sel]]
        for w in range(1, self._max_width + 1):
            idx = np.flatnonzero(take & (width == w))
            if idx.size:
                self.known.update(map(tuple, np.sort(rows[idx, :w], axis=1).tolist()))

    def add(self, clause: tuple) -> None:
        self.known.add(clause)

    def subsumes(self, clause: tuple) -> bool:
        """True if something known SUBSUMES `clause` — i.e. some subset of it is already a clause.
        Not a duplicate test: mining cannot emit the same clause twice (candidates within a width
        are distinct combinations, widths differ in length, and width 1's fixpoint removes each
        proven variable from the pool). Subsumption is the thing that matters: if (~a | ~b) is
        present then (~a | ~b | ~c) adds nothing while still slowing every later propagation."""
        for r in range(1, len(clause) + 1):
            for sub in itertools.combinations(clause, r):
                if sub in self.known:
                    return True
        return False


def _budget_with(budget: ProbeBudget, max_tuples: int) -> ProbeBudget:
    """The run-wide budget plus this width's work budget."""
    return ProbeBudget(seconds=budget.seconds, max_tuples=max_tuples, deadline=budget.deadline)


def _ceiling(max_tuples, width: int) -> int:
    """The candidate ceiling for one width — an int applies to all, a dict is looked up."""
    if isinstance(max_tuples, dict):
        return int(max_tuples.get(width, DEFAULT_MAX_TUPLES))
    return int(max_tuples)


class _WidthResult(NamedTuple):
    """What one width's sweep did — the SAME shape from both workers, so the caller records it
    once. `_mine_units` used to report through a positional `seen` list and drop `stopped`
    entirely, which is how the caller ended up asserting a stop reason it had not been told."""

    found: int
    exhausted: bool
    candidates: int
    probed: int
    rounds: int
    stopped: str


def _mine_units(probe: LiteralProbe, budget: ProbeBudget, keep,
                max_tuples: int = 0) -> _WidthResult:
    """Width 1, to a fixpoint: a proven unit can expose further failed literals.

    `max_tuples` is the width's WHOLE budget, spent ACROSS the fixpoint's rounds — the pool
    shrinks each round, so a per-round ceiling would let a many-round sweep spend an unbounded
    multiple of it. 0 = unbounded. This used to be hardcoded to 0 with no way to reach it, so the
    CLI printed an over-budget warning for a bound that could not fire.
    ⚠️ A caller passing a per-width DICT must give width 1 an explicit `0`: the fallback is
    DEFAULT_MAX_TUPLES, which would cap the backbone fixpoint (see pipeline.STAGE1_MAX_TUPLES)."""
    found, exhausted = 0, True
    cand = done = rounds = 0
    stopped = ""
    left = int(max_tuples)
    while not probe.unsat:
        if probe.free_literals().size == 0:
            break
        if max_tuples > 0 and left <= 0:
            exhausted = False                 # budget spent mid-fixpoint: NOT a proof of absence
            stopped = stopped or "max_tuples"
            break
        r = probe.probe(size=1, budget=_budget_with(budget, left))  # pool None = every free literal
        exhausted = exhausted and r.exhausted
        stopped = stopped or r.stopped
        cand += r.total
        done += r.probed
        rounds += 1
        left -= r.probed
        fresh = []
        for lits in r.conflicting:
            unit = -int(lits[0])                             # assuming l conflicts => ~l holds
            if keep((unit,), check=False):               # provably cannot be subsumed — see keep()
                fresh.append(unit)
        found += len(fresh)
        if not fresh or not r.exhausted:
            break
        probe.assume(fresh)                                  # narrow, then look again
    return _WidthResult(found, exhausted, cand, done, rounds, stopped)


def _mine_width(probe: LiteralProbe, width: int, pool, budget: ProbeBudget, max_tuples: int,
                keep) -> _WidthResult:
    """One width >= 2 over the pool's still-free literals."""
    # The pool is DIMACS LITERALS. Restricted to the ones still free, because a settled literal
    # cannot teach anything: TRUE makes the probe a no-op, FALSE makes it conflict for a reason
    # already known.
    free_vars = probe.free_variables()                       # DIMACS variables, 1-based
    if pool is None:
        lits = np.stack([free_vars, -free_vars], axis=1).reshape(-1).astype(np.int32)
    else:
        cand = np.asarray([int(x) for x in pool], dtype=np.int32)
        lits = cand[np.isin(np.abs(cand), free_vars)]
    if lits.size < width:
        return _WidthResult(0, True, 0, 0, 1, "")
    # COUNT FIRST, then decide. The ceiling has to be checked against an exact count computed
    # WITHOUT generating anything: `combinations` below materializes C(P,k) tuples, so a ceiling
    # applied to its output cannot protect against the very case it exists for. At pool=None and
    # width 3 on an unrolled circuit that is ~1e11 candidates (~2 TB), and this function used to
    # be OOM-killed there before it ever reached the comparison.
    # ONE call into the core, which enumerates and probes CHUNK BY CHUNK. Nothing proportional to
    # C(P,k) is materialized here — for this instance's 122M pairs the old host-side path built a
    # 0.9 GiB array first, and `max_tuples` is a work budget, so a huge width now yields whatever
    # it can reach instead of nothing.
    #
    # There is deliberately no candidate pre-filter. There used to be one (_drop_subsumed), meant
    # to skip candidates whose clause is already known and so save GPU work. MEASURED on
    # plain7824 at width 2: it cost 65.3s and dropped 0 of 122,289,160 candidates — 0.0000%. It
    # cannot do better in general, because width 1 runs first and `assume` folds its units into
    # the flush, which removes those variables from the pool entirely; what is left to subsume is
    # just the formula's existing binary clauses, a vanishing fraction. `keep()` already rejects a
    # re-derived clause, so correctness never depended on it.
    r = probe.probe(lits, width, budget=_budget_with(budget, max_tuples))
    found = 0
    for row in r.conflicting:
        found += keep(tuple(sorted(-int(x) for x in row)))
    return _WidthResult(found, r.exhausted, r.total, r.probed, 1, r.stopped)


def _release(device) -> None:
    """Hand the probe engine's VRAM back. Reserved-but-free memory poisons the next component's
    batch calibration, which reads FREE memory — measured K collapsing to 1 without this."""
    if getattr(device, "type", None) == "cuda":
        import torch
        torch.cuda.empty_cache()
