"""Probe every size-k combination of a literal set, in bulk, on the GPU.

This is the second shape of the engine's core primitive. `AssumptionChecker` answers "propagate
THESE assumption sets"; `LiteralProbe` answers **"propagate every k-subset of these literals, and
tell me which ones conflict"** — the question behind failed-literal detection, entailed-clause
mining, backbone extraction, implication-graph construction and lookahead scoring alike.

    probe = LiteralProbe(num_vars, clauses, device)
    res = probe.probe(size=1)                # every free literal; conflicts are failed literals
    for lits in res.conflicting:             # each is an entailed negation
        ...

    res = probe.probe(pool=[1, -1, 7, -7], size=2, budget=ProbeBudget(seconds=5))

Literals are **DIMACS** (+v TRUE, -v FALSE) throughout, like every other public surface in the
library; the internal 2N indexing never appears here. `pool=None` means every literal the
formula's own unit propagation left undetermined — probing a settled literal tells you nothing.

THE SWEEP ITSELF RUNS IN THE CORE (`cpp/src/probe.cpp`): enumeration order, chunk sizing and the
budget all live there, so a Python caller and a C++ caller get the same candidates in the same
order under the same memory model. This module only marshals. Two consequences worth knowing:

  * `ProbeResult.exhausted` is a PROOF of absence — every candidate was probed, so "no entailed
    clause of this width" is a result, not a lack of one.
  * `count()` gives the exact candidate count without generating anything, which is how a caller
    refuses an impossible width. Enumerating first and measuring afterwards does not work: at
    k=3 over the free literals of an unrolled circuit the enumeration is ~1e11 tuples.

There is deliberately no host-side enumerator here. There was one (`combinations`, numpy), kept
"for callers that filter candidates before probing" — the only such caller was deleted when
candidate pre-filtering was measured at 65s to drop 0 of 122M, and after that it existed solely so
tests could check the core's ordering. It lives in tests/reference_combinations.py now, where a
reference implementation belongs; a second enumerator inside the shipping face is a thing that can
silently disagree with the one that runs.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from gpap.engine import _core


@dataclass
class ProbeBudget:
    """When to stop. Every field is optional; the first to bind wins.

    A budget is not a nicety: C(P,k) grows fast enough that an unbounded sweep of triples over a
    wide pool is hours of work — and the cost is the PROBE, not the enumeration (probing n tuples
    initializes n x M clause words, which for a big pool runs to petabytes of state broadcast).

    `max_tuples` is therefore a WORK BUDGET: the sweep probes that many candidates and reports
    `exhausted = False`. Every conflict it found is still a genuine entailed clause, so a
    truncated sweep is SOUND — it just is not a proof of absence. `seconds` is the wall-clock
    backstop, checked between chunks. Set `refuse_over_max` if only an exhaustive answer is
    useful to you.
    """

    seconds: float | None = None
    max_tuples: int | None = None
    deadline: float | None = field(default=None)   # absolute perf_counter() stamp, if shared
    skip_subsumed: bool = False                     # drop conflicts reached with 0 implications
    refuse_over_max: bool = False                   # over the budget: refuse, instead of probing it

    def remaining(self, now: float | None = None) -> float:
        """Seconds left, as the core wants them (0.0 = unbounded). A shared `deadline` and an own
        `seconds` both apply; the tighter one wins."""
        now = time.perf_counter() if now is None else now
        limits = [s for s in (self.seconds,
                              (self.deadline - now) if self.deadline is not None else None)
                  if s is not None]
        if not limits:
            return 0.0
        return max(min(limits), 1e-9)               # a spent budget must not read as "unbounded"

    def expired(self, now: float | None = None) -> bool:
        now = time.perf_counter() if now is None else now
        return self.deadline is not None and now > self.deadline


@dataclass
class ProbeResult:
    """Which of the probed tuples conflicted.

    `conflicting` is (m, k) DIMACS literals: assuming that tuple drove BCP to a conflict, so the
    formula entails the negation of its conjunction.
    """

    conflicting: np.ndarray                 # (m, k) int32 DIMACS
    total: int                              # candidates the full enumeration has
    probed: int                             # candidates actually probed (a prefix)
    subsumed: int = 0                       # conflicts dropped as already-implied (skip_subsumed)
    seconds: float = 0.0
    stopped: str = ""                       # "" = ran to completion, else why it stopped

    @property
    def exhausted(self) -> bool:
        """True iff every candidate was probed — i.e. a negative result is a PROOF that no tuple
        of this size conflicts, not merely an absence of one."""
        return not self.stopped and self.probed == self.total

    def __len__(self) -> int:
        return int(self.conflicting.shape[0])

    @classmethod
    def _from_sweep(cls, sweep) -> "ProbeResult":
        st = sweep.stats
        return cls(conflicting=sweep.conflicts(), total=int(st.total), probed=int(st.probed),
                   subsumed=int(st.subsumed), seconds=float(st.seconds),
                   stopped=_core.PROBE_STOP.get(int(st.stop), "") if not st.exhausted else "")


class LiteralProbe:
    """BCP-probe combinations of literals against one formula, on the GPU.

    Requires CUDA and a built libgpap. The engine is built once and reused across sweeps; its
    post-units-flush state is the baseline every sweep starts from.
    """

    def __init__(self, num_vars: int, clauses: list, device: "torch.device | None" = None,
                 *, mode: str | None = None):
        """`mode` pins the propagate kernel; the default lets the CORE dispatch per chunk.

        This defaulted to "scalar" when the sweep was ported, which pinned every probe to the
        wrong kernel on any formula worth probing — measured 1.28x (b03-c25, M=14.5k) to 2.20x
        (b12-c50, M=219k) of pure loss, since the dispatch rule picks WAVE from M >= 2048 up. The
        engine resolves "auto" on every reset now, so a probe sweep gets it per chunk."""
        if device is not None and getattr(device, "type", "cuda") != "cuda":
            raise RuntimeError("LiteralProbe requires CUDA (the engine is GPU-only)")
        self._formula = _core.Formula.from_clauses(num_vars, clauses)
        self._engine = _core.Engine(
            self._formula, mode=mode or os.environ.get("GPAP_KERNEL") or "auto")
        self._num_vars = int(num_vars)
        self._refresh()

    def _refresh(self) -> None:
        self.unsat = bool(self._engine.flush_conflict)
        self._free = self._engine.free_literals()

    # ------------------------------------------------------------------ shape
    @property
    def num_vars(self) -> int:
        return self._num_vars

    @property
    def num_clauses(self) -> int:
        return int(self._engine.num_clauses)

    @property
    def implications(self) -> int:
        """BCP implications derived by every launch this probe has made.

        The hardware-work metric. Clauses/s depends on how many clauses the formula happens to
        entail; implications/s measures what the engine actually did, so it is the number to
        compare across instances, GPUs and engine changes."""
        c = self._engine.counters()
        return int(c.props - c.decs)

    @property
    def engine(self):
        """The underlying core engine — for callers that also want entry point 1."""
        return self._engine

    # ------------------------------------------------------------------ what is worth probing
    def free_literals(self) -> np.ndarray:
        """DIMACS literals whose variable the flush left undetermined.

        Probing a settled literal is wasted work: if it is already TRUE the probe is a no-op, and
        if it is FALSE the probe conflicts for a reason that is already known."""
        return self._free

    def assigned_literals(self) -> np.ndarray:
        """DIMACS literals the flush set TRUE — the baseline every sweep propagates on top of.

        The complement of `free_literals` in intent, and the thing a caller needs to reason about
        what a conflict PROVED: a clause is emptied by an assumption tuple relative to THIS
        assignment, not relative to the input formula."""
        a = self._engine.flushed_assignment()            # (2N,) int8, index 2(v-1)+(lit<0)
        idx = np.flatnonzero(a).astype(np.int64)
        v = (idx >> 1) + 1
        return np.where(idx & 1, -v, v).astype(np.int32)

    def free_variables(self) -> np.ndarray:
        """DIMACS variables (1-based) the flush left undetermined."""
        return self._free[self._free > 0]

    def is_free(self, lit: int) -> bool:
        """True iff this literal's variable is unassigned after the flush."""
        return bool(np.isin(abs(int(lit)), self.free_variables()))

    def assume(self, lits) -> None:
        """Fold PROVEN units into the cached flush state IN PLACE, narrowing later sweeps.

        Propagates them once on top of the current cache and re-caches the result, which is the
        fixpoint a rebuilt formula's flush would reach — sound as long as the literals really are
        entailed. Avoids rebuilding the formula and its occurrence lists per round, which
        dominated mining wall on large instances."""
        lits = np.ascontiguousarray(lits, dtype=np.int32).reshape(-1)
        if lits.size == 0 or self.unsat:
            return
        self._engine.assume_units(lits)
        self._refresh()

    # ------------------------------------------------------------------ probing
    def count(self, pool=None, size: int = 1, *, distinct_vars: bool = True) -> int:
        """Exact candidate count for this sweep, WITHOUT generating it. Saturates rather than
        wrapping, so comparing it against a ceiling is always safe."""
        return self._engine.probe_count(pool, size, distinct_vars)

    def probe(self, pool=None, size: int = 1, *, budget: ProbeBudget | None = None,
              distinct_vars: bool = True) -> ProbeResult:
        """Probe every size-`size` combination of `pool` (DIMACS literals).

        `pool=None` means every free literal. The core enumerates one chunk at a time, so nothing
        proportional to C(P,k) is ever materialized — the pool and the current chunk are the only
        host memory a sweep needs, whatever C(P,k) is. `budget.max_tuples` bounds how many are
        probed, not whether the sweep runs at all.
        """
        b = budget or ProbeBudget()
        if self.unsat:
            return ProbeResult(conflicting=np.empty((0, size), np.int32), total=0,
                               probed=0, stopped="unsat")
        sweep = self._engine.probe_combinations(
            pool, size, seconds=b.remaining(), max_tuples=b.max_tuples,
            distinct_vars=distinct_vars, skip_subsumed=b.skip_subsumed,
            refuse_over_max=b.refuse_over_max)
        with sweep:
            return ProbeResult._from_sweep(sweep)

    def probe_tuples(self, tuples, *, budget: ProbeBudget | None = None) -> ProbeResult:
        """Probe an explicit (n, k) array of DIMACS literals.

        The escape hatch for callers that build candidates themselves — filtered, weighted, or
        drawn from somewhere other than a plain enumeration."""
        b = budget or ProbeBudget()
        tuples = np.ascontiguousarray(tuples, dtype=np.int32)
        if tuples.ndim == 1:
            tuples = tuples.reshape(-1, 1)
        if self.unsat or tuples.shape[0] == 0:
            return ProbeResult(conflicting=tuples[:0], total=int(tuples.shape[0]), probed=0,
                               stopped="unsat" if self.unsat else "")
        sweep = self._engine.probe_tuples(tuples, seconds=b.remaining(),
                                          max_tuples=b.max_tuples, skip_subsumed=b.skip_subsumed,
                                          refuse_over_max=b.refuse_over_max)
        with sweep:
            return ProbeResult._from_sweep(sweep)
