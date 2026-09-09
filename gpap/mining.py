"""Entailed-clause mining: prove small clauses the formula already implies, before solving it.

An application of `gpap.engine.probe`; nothing here knows about sampling. If assuming a set of
literals drives BCP to a conflict, the formula entails the negation of their conjunction, so
probing every k-subset of a literal pool yields every entailed clause of width k that BCP alone
can see (k=1 is the BCP-visible backbone). Every clause returned is entailed, so adding it leaves
the model set exactly unchanged (verified in tests): mining can only strengthen propagation.

    result = mine_entailed(num_vars, clauses, device); stronger = clauses + result.clauses

Measured effect on sampling and the tier-by-tier cost: docs/findings/MINING_FINDINGS.md.
"""
from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import torch

from gpap.engine.checker import AssumptionChecker
from gpap.engine.probe import LiteralProbe, ProbeBudget

log = logging.getLogger(__name__)

__all__ = ["mine_entailed", "MiningResult", "DROP_LEVELS", "DEFAULT_WIDTHS",
           "DEFAULT_MAX_TUPLES"]

DEFAULT_WIDTHS = (1, 2, 3)

#: Per-width WORK BUDGET: how many candidates a width may probe (an int for all, or a dict). A
#: width over budget probes that many and reports itself NOT exhaustive; partial results are kept
#: because the clauses found are entailed regardless.
DEFAULT_MAX_TUPLES = 200_000


@dataclass
class MiningResult:
    """What was proven, and whether the search was complete. `clauses` are DIMACS clauses, each
    entailed by the input formula; `exhausted[k]` says width k was searched EXHAUSTIVELY."""

    clauses: list[list[int]] = field(default_factory=list)
    by_width: dict[int, int] = field(default_factory=dict)
    exhausted: dict[int, bool] = field(default_factory=dict)
    skipped: dict[int, str] = field(default_factory=dict)
    #: candidates the enumeration HAS and how many were probed, per width: without them "nothing
    #: is entailed here" and "we barely started" look identical.
    candidates: dict[int, int] = field(default_factory=dict)
    probed: dict[int, int] = field(default_factory=dict)
    #: sweeps per width: 1 for width >= 2, more for width 1 (a FIXPOINT), whose
    #: `candidates`/`probed` are therefore SUMS over rounds and NOT comparable to the pool size.
    rounds: dict[int, int] = field(default_factory=dict)
    #: BCP implications derived across every sweep — the hardware-work metric, independent of how
    #: many clauses were entailed, so it is what compares across instances or engines.
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
        """One line: what was proven, how much was probed, how fast. The probed count is a plain
        total, deliberately not a fraction of the candidate space: summing `candidates` across
        widths adds a width-1 fixpoint sum to a width-2 C(P,2)."""
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

    `pool`: DIMACS literals to combine at widths >= 2 (None = every free literal); pass both
      polarities of a variable to ask about it in general, one for a single side. Width 1 always
      sweeps ALL free literals — the backbone is cheap and the dominant win.
    `seconds`: wall-clock budget across all widths (None = unbounded), binding between chunks.
    `max_tuples`: per-width work budget, an int for every width or {width: budget}. A width above
      it probes that many candidates and is reported not exhaustive (`.candidates` / `.probed`).
    `min_units_for_wide`: run widths >= 2 only if width 1 proved at least this many units, so
      mining stays nearly free where the expensive wide tiers find nothing.
    `drop`: which REDUNDANT clauses to leave out, each level stronger than the last — "none"
      keeps everything; "subsumed" (default) drops a clause the swept-against formula already
      covers, free as a set test; "derivable" ALSO drops the ones propagation re-derives on
      demand, at one extra batched launch. Derivable is deliberately not the default: such a
      clause still shortens propagation and GPAP's sampler is propagation-bound, so keeping them
      is measured faster end-to-end (docs/findings/MINING_FINDINGS.md).
    `on_plan`: on_plan(width, candidates, ceiling) before any probing, so a front-end can show
      what a sweep will cost while aborting is cheap; it lives here because the count needs a
      built engine the caller would otherwise construct twice.

    Width 1 is iterated to a FIXPOINT; later widths run once, on the narrowed formula.
    """
    if drop not in DROP_LEVELS:
        raise ValueError(f"drop must be one of {DROP_LEVELS}, got {drop!r}")
    t0 = time.perf_counter()
    res = MiningResult()
    sub = _Subsumption(max(widths, default=0))
    budget = ProbeBudget(deadline=t0 + seconds if seconds is not None else None)
    current = list(clauses)

    def keep(clause: tuple, check: bool = True) -> bool:
        """Record a clause unless something already known SUBSUMES it (see `_Subsumption`), on
        the conflicts the sweep REPORTED and never on candidates. `check=False` skips the test
        where it is provably vacuous, which is width 1: a mined unit (u) comes from probing ~u,
        and ~u was probed only because it was FREE, so had (u) been a formula unit or one mined in
        an earlier fixpoint round (both folded in by `assume`), ~u would not have been free."""
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
            # UPPER BOUNDS, exact only for the first width run: width 1's units narrow the
            # formula, and its own count is ONE fixpoint round, not the total it will probe.
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
            # The subsumption reference is the formula REDUCED by the baseline the probe runs on,
            # so it is rebuilt whenever that baseline moves; width 1 never consults it.
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
                # ONE message for both widths, from the probe's OWN stop reason: never assert
                # "(max_tuples)", since width 1 is as often cut by the shared wall-clock deadline.
                res.skipped[width] = (
                    f"probed {w.probed:,} of {w.candidates:,} candidates "
                    f"({w.stopped or 'incomplete'}) — clauses found are still entailed, but this "
                    f"width was NOT searched exhaustively")
            # Each width STRENGTHENS the formula the next one probes. Units are folded in place
            # by `assume`; wider clauses need the rebuild, and skipping it costs real clauses.
            if found and width > 1 and width < max(widths):
                res.implications += probe.implications      # bank before the rebuild resets them
                probe = LiteralProbe(num_vars, current, device)
    except RuntimeError as e:
        # Mining is an OPTIMIZATION: every clause already proven stays valid, so a mid-tier CUDA
        # failure abandons the rest rather than failing the caller's job.
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


def _derivable_mask(clauses, base_clauses, num_vars: int, device) -> np.ndarray:
    """Which of `clauses` can propagation over `base_clauses` fully replace? (n,) bool, in one
    batched launch of `sum(len(c) for c in clauses)` assumption sets.

    A clause `(l1|...|lk)` does k separate JOBS: when the other k-1 literals are false it forces
    the remaining one, and a clause is marked only when propagation covers EVERY one of them.
    Requiring all k is deliberately more conservative than the literature's transitive reduction
    for hyper-binary resolvents, which treats one direction as enough because a binary clause IS
    both of its implication-graph edges; our reference is the PROPAGATION relation, which also
    holds one-way implications squeezed out of long clauses going unit (`(-1|5),(-1|6),(-5|-6|2)`
    gives 1 => 2 but not -2 => -1), and a third of the mined wide clauses have a live direction
    the weaker rule discards. Only propagation strength is at stake, never soundness.
    """
    wide = [list(c) for c in clauses if len(c) > 1]
    out = np.zeros(len(clauses), dtype=bool)
    if not wide:
        return out
    ck = AssumptionChecker.from_clauses(int(num_vars), [list(c) for c in base_clauses], device)
    sets, tag = [], []
    for ci, cl in enumerate(wide):
        for j, want in enumerate(cl):
            sets.append([-int(x) for k, x in enumerate(cl) if k != j])
            tag.append((ci, int(want)))
    res = ck.check_assumptions(sets, assignments=True)
    conflict = res["conflict"].cpu().numpy()
    asg = res["assignment"]
    cols = torch.as_tensor([2 * (abs(w) - 1) + (0 if w > 0 else 1) for _, w in tag],
                           device=asg.device)
    forced = asg[torch.arange(len(tag), device=asg.device), cols].bool().cpu().numpy()
    hit = conflict | forced
    wide_pos = [i for i, c in enumerate(clauses) if len(c) > 1]
    owner = np.array([ci for ci, _ in tag], dtype=np.int64)
    jobs = np.bincount(owner, minlength=len(wide))
    done = np.bincount(owner, weights=hit, minlength=len(wide))
    out[np.asarray(wide_pos, dtype=np.int64)[jobs == done]] = True
    return out


def _drop_derivable(res: MiningResult, num_vars: int, clauses: list, device) -> None:
    """Remove the clauses propagation already derives, judged against what the caller will HAVE:
    the input formula plus the mined UNITS, not the input alone."""
    units = [c for c in res.clauses if len(c) == 1]
    mask = _derivable_mask(res.clauses, list(clauses) + units, num_vars, device)
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

    Not the input clauses: mining folds every proven unit into the engine's flush cache, so a
    sweep propagates on top of an assignment and a conflict proves a clause relative to THAT. An
    input clause (a | b | c) whose `c` the baseline falsified is a binary constraint during the
    sweep, so assuming (~a, ~b) "proves" (a | b) — entailed, but redundant for a consumer that
    also takes the mined units, and subsumed by no input clause. Keep this as the ONLY filter: in
    the engine it would read an order-dependent counter and make the output kernel-dependent
    (gated by test_mining_is_kernel_invariant)."""

    def __init__(self, max_width: int):
        self._max_width = int(max_width)
        self.known: set[tuple] = set()

    def rebuild(self, clauses: list, num_vars: int, assigned: np.ndarray) -> None:
        """Recompute the reference: `clauses` reduced by the baseline `assigned` (DIMACS
        literals). Vectorised: a per-clause Python loop costs more than the sweep it protects."""
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
        # A satisfied clause constrains nothing; a residue wider than the widest mined width
        # cannot subsume anything emittable.
        take = ~sat & (width >= 1) & (width <= self._max_width) & (lens > 0)
        if not take.any():
            return
        pos = np.flatnonzero(free)
        owner = np.searchsorted(starts, pos, side="right") - 1
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
        """True if something known SUBSUMES `clause`, i.e. some subset of it is already a clause
        — not a duplicate test: with (~a | ~b) present, (~a | ~b | ~c) only slows propagation."""
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
    """What one width's sweep did — the same shape from both workers, `stopped` included, so the
    caller records it once and never asserts a stop reason it was not told."""

    found: int
    exhausted: bool
    candidates: int
    probed: int
    rounds: int
    stopped: str


def _mine_units(probe: LiteralProbe, budget: ProbeBudget, keep,
                max_tuples: int = 0) -> _WidthResult:
    """Width 1, to a fixpoint: a proven unit can expose further failed literals. `max_tuples` is
    the width's WHOLE budget, spent ACROSS the fixpoint's rounds (0 = unbounded), since a
    per-round ceiling would let a many-round sweep spend an unbounded multiple of it. A caller
    passing a per-width DICT must give width 1 an explicit `0`, or the DEFAULT_MAX_TUPLES fallback
    caps the backbone fixpoint (see pipeline.STAGE1_MAX_TUPLES)."""
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
        probe.assume(fresh)
    return _WidthResult(found, exhausted, cand, done, rounds, stopped)


def _mine_width(probe: LiteralProbe, width: int, pool, budget: ProbeBudget, max_tuples: int,
                keep) -> _WidthResult:
    """One width >= 2 over the pool's still-free literals: a settled literal cannot teach
    anything (TRUE makes the probe a no-op, FALSE conflicts for a reason already known)."""
    free_vars = probe.free_variables()                       # DIMACS variables, 1-based
    if pool is None:
        lits = np.stack([free_vars, -free_vars], axis=1).reshape(-1).astype(np.int32)
    else:
        cand = np.asarray([int(x) for x in pool], dtype=np.int32)
        lits = cand[np.isin(np.abs(cand), free_vars)]
    if lits.size < width:
        return _WidthResult(0, True, 0, 0, 1, "")
    # COUNT FIRST, then decide: the ceiling must be checked against an exact count computed
    # WITHOUT generating anything (at pool=None and width 3 the enumeration is ~1e11 candidates,
    # and materializing it to compare would OOM). Then ONE call into the core, which enumerates
    # and probes CHUNK BY CHUNK. Do not add a candidate pre-filter: measured, it dropped 0 of
    # 122M candidates, because `assume` has already removed width 1's variables from the pool.
    r = probe.probe(lits, width, budget=_budget_with(budget, max_tuples))
    found = 0
    for row in r.conflicting:
        found += keep(tuple(sorted(-int(x) for x in row)))
    return _WidthResult(found, r.exhausted, r.total, r.probed, 1, r.stopped)


def _release(device) -> None:
    """Hand the probe engine's VRAM back: reserved-but-free memory poisons the next component's
    batch calibration, which reads FREE memory."""
    if getattr(device, "type", None) == "cuda":
        import torch
        torch.cuda.empty_cache()
