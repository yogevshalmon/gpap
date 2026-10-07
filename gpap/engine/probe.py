"""Probe every size-k combination of a literal set, in bulk, on the GPU — the second shape of the
engine's core primitive. `AssumptionChecker` answers "propagate THESE assumption sets";
`LiteralProbe` answers "propagate every k-subset of these literals and tell me which conflict",
the question behind failed-literal detection, entailed-clause mining, backbone extraction and
lookahead scoring alike.

    probe = LiteralProbe(num_vars, clauses, device)
    res = probe.probe(size=1)                # every free literal; conflicts are failed literals
    res = probe.probe(pool=[1, -1, 7, -7], size=2, budget=ProbeBudget(seconds=5))

Literals are DIMACS (+v TRUE, -v FALSE) throughout, and `pool=None` means every literal the
formula's own unit propagation left undetermined. The sweep runs in the CORE (`cpp/src/probe.cpp`)
— enumeration order, chunk sizing and the budget — so both faces get the same candidates in the
same order; this module only marshals. Hence `ProbeResult.exhausted` is a PROOF of absence, and
`count()` gives the exact candidate count without generating anything, which is how a caller
refuses an impossible width. There is deliberately no host-side enumerator here: a second one can
silently disagree with the one that runs (reference: tests/reference/combinations.py).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from gpap.engine import _core


@dataclass
class ProbeBudget:
    """When to stop. Every field is optional; the first to bind wins. C(P,k) grows fast enough
    that an unbounded sweep of triples over a wide pool is hours of work, and the cost is the
    PROBE, not the enumeration. `max_tuples` is therefore a WORK BUDGET — the sweep probes that
    many and reports `exhausted = False`; every conflict found is still a genuine entailed clause,
    so a truncated sweep is SOUND, just not a proof of absence. `seconds` is the wall-clock
    backstop; `refuse_over_max` demands an exhaustive answer instead."""

    seconds: float | None = None
    max_tuples: int | None = None
    deadline: float | None = field(default=None)   # absolute perf_counter() stamp, if shared
    refuse_over_max: bool = False                   # over the budget: refuse, instead of probing it

    def remaining(self, now: float | None = None) -> float:
        """Seconds left, as the core wants them (0.0 = unbounded); a shared `deadline` and an own
        `seconds` both apply, the tighter one winning."""
        now = time.perf_counter() if now is None else now
        limits = [s for s in (self.seconds,
                              (self.deadline - now) if self.deadline is not None else None)
                  if s is not None]
        if not limits:
            return 0.0
        return max(min(limits), 1e-9)               # a spent budget must not read as "unbounded"


@dataclass
class ProbeResult:
    """Which of the probed tuples conflicted: `conflicting` is (m, k) DIMACS literals whose
    assumption drove BCP to a conflict, so the formula entails each one's negation."""

    conflicting: np.ndarray                 # (m, k) int32 DIMACS
    total: int                              # candidates the full enumeration has
    probed: int                             # candidates actually probed (a prefix)
    seconds: float = 0.0
    stopped: str = ""

    @property
    def exhausted(self) -> bool:
        """True iff every candidate was probed, making a negative result a PROOF that no tuple of
        this size conflicts."""
        return not self.stopped and self.probed == self.total

    def __len__(self) -> int:
        return int(self.conflicting.shape[0])

    @classmethod
    def _from_sweep(cls, sweep) -> "ProbeResult":
        st = sweep.stats
        return cls(conflicting=sweep.conflicts(), total=int(st.total), probed=int(st.probed),
                   seconds=float(st.seconds),
                   stopped=_core.PROBE_STOP.get(int(st.stop), "") if not st.exhausted else "")


class LiteralProbe:
    """BCP-probe combinations of literals against one formula, on the GPU (CUDA + libgpap). The
    engine is built once and reused; its post-units-flush state is every sweep's baseline."""

    def __init__(self, num_vars: int, clauses: list, device: "torch.device | None" = None,
                 *, mode: str | None = None):
        """`mode` pins the propagate kernel; the default lets the CORE dispatch per chunk, which
        is what a sweep wants (the rule picks WAVE from M >= 2048 up)."""
        self._init(_core.Formula.from_clauses(num_vars, clauses), device, mode)

    @classmethod
    def from_file(cls, cnf_file: str, device: "torch.device | None" = None,
                  *, mode: str | None = None) -> "LiteralProbe":
        self = cls.__new__(cls)
        self._init(_core.Formula.from_file(cnf_file), device, mode)
        return self

    def _init(self, formula, device, mode) -> None:
        if device is not None and getattr(device, "type", "cuda") != "cuda":
            raise RuntimeError("LiteralProbe requires CUDA (the engine is GPU-only)")
        self._formula = formula
        self._engine = _core.Engine(formula, mode=mode)
        self._num_vars = int(formula.info.num_vars)
        self._refresh()

    def _refresh(self) -> None:
        self.unsat = bool(self._engine.flush_conflict)
        self._free = self._engine.free_literals()

    @property
    def num_vars(self) -> int:
        return self._num_vars

    @property
    def num_clauses(self) -> int:
        return int(self._engine.num_clauses)

    @property
    def implications(self) -> int:
        """BCP implications derived by every launch this probe has made — the hardware-work
        metric, which unlike clauses/s compares across instances, GPUs and engine changes."""
        c = self._engine.counters()
        return int(c.props - c.decs)

    @property
    def engine(self):
        """The underlying `_core.Engine`, the low-level numpy binding to the C ABI — an escape
        hatch, not part of this class's stable surface; reach through it only for what the faces
        do not cover."""
        return self._engine

    def free_literals(self) -> np.ndarray:
        """DIMACS literals whose variable the flush left undetermined (a settled literal teaches
        nothing: TRUE is a no-op, FALSE conflicts for a known reason)."""
        return self._free

    def assigned_literals(self) -> np.ndarray:
        """DIMACS literals the flush set TRUE — the baseline every sweep propagates on top of, so
        a conflict empties a clause relative to THIS assignment, not to the input formula."""
        a = self._engine.flushed_assignment()
        idx = np.flatnonzero(a).astype(np.int64)
        v = (idx >> 1) + 1
        return np.where(idx & 1, -v, v).astype(np.int32)

    def free_variables(self) -> np.ndarray:
        return self._free[self._free > 0]

    def is_free(self, lit: int) -> bool:
        return bool(np.isin(abs(int(lit)), self.free_variables()))

    def assume(self, lits) -> None:
        """Fold PROVEN units into the cached flush state IN PLACE, narrowing later sweeps: they
        are propagated once on top of the cache and the result re-cached, which is the fixpoint a
        rebuilt formula's flush would reach — sound as long as they really are entailed, and it
        avoids rebuilding the occurrence lists per round."""
        lits = np.ascontiguousarray(lits, dtype=np.int32).reshape(-1)
        if lits.size == 0 or self.unsat:
            return
        self._engine.assume_units(lits)
        self._refresh()

    def count(self, pool=None, size: int = 1, *, distinct_vars: bool = True) -> int:
        """Exact candidate count for this sweep, WITHOUT generating it; saturates rather than
        wrapping, so comparing it against a ceiling is safe."""
        return self._engine.probe_count(pool, size, distinct_vars)

    def probe(self, pool=None, size: int = 1, *, budget: ProbeBudget | None = None,
              distinct_vars: bool = True) -> ProbeResult:
        """Probe every size-`size` combination of `pool` (DIMACS literals); `pool=None` means every
        free literal. The core enumerates one chunk at a time, so nothing proportional to C(P,k)
        is materialized, and `budget.max_tuples` bounds how many are probed, not whether the sweep
        runs."""
        if size < 1:
            raise ValueError(f"probe size must be >= 1, got {size}")
        b = budget or ProbeBudget()
        if self.unsat:
            return ProbeResult(conflicting=np.empty((0, size), np.int32), total=0,
                               probed=0, stopped="unsat")
        sweep = self._engine.probe_combinations(
            pool, size, seconds=b.remaining(), max_tuples=b.max_tuples,
            distinct_vars=distinct_vars, refuse_over_max=b.refuse_over_max)
        with sweep:
            return ProbeResult._from_sweep(sweep)

    def probe_tuples(self, tuples, *, budget: ProbeBudget | None = None) -> ProbeResult:
        """Probe an explicit (n, k) array of DIMACS literals (callers with own candidates)."""
        b = budget or ProbeBudget()
        try:
            tuples = np.ascontiguousarray(tuples, dtype=np.int32)
        except ValueError:
            raise ValueError("probe_tuples needs a rectangular (n, k) array — all tuples must "
                             "have the same width (probe each width separately)") from None
        if tuples.ndim == 1:
            tuples = tuples.reshape(-1, 1)
        if self.unsat or tuples.shape[0] == 0:
            return ProbeResult(conflicting=tuples[:0], total=int(tuples.shape[0]), probed=0,
                               stopped="unsat" if self.unsat else "")
        sweep = self._engine.probe_tuples(tuples, seconds=b.remaining(),
                                          max_tuples=b.max_tuples,
                                          refuse_over_max=b.refuse_over_max)
        with sweep:
            return ProbeResult._from_sweep(sweep)
