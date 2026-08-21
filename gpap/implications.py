"""What each literal implies — the one sweep behind failed literals, HBR and SAT sweeping.

Probe a literal and propagate. Two things come back, and applications differ only in which one
they read:

    conflict      ->  the literal is FAILED, so its negation is a unit    (the backbone)
    otherwise     ->  the CLOSURE: every literal BCP derived from it

The closure is the substrate. Given closures for a pool of literals, with no further GPU work:

    hyper-binary resolvents   every `m` in closure(l) IS the binary clause (~l | m)
    equivalent literals       l == m exactly when m is in closure(l) AND l is in closure(m)

    from gpap.implications import implications, equivalences, hyper_binary
    imp = implications(num_vars, clauses, device)      # one batched sweep over every free literal
    for cl in hyper_binary(imp): ...                   # binary clauses the formula entails
    edges = equivalence_edges(imp)                     # -> preprocess.equivalent_literal_substitution

WHY THIS IS THE GPAP-SHAPED WAY TO DO IT. On a CPU each literal is a separate propagation, which
is why solvers restrict probing to roots of the binary implication graph and accept the coverage
they lose. Here every literal in the pool is one row of one launch, so the exhaustive answer costs
what a partial one would.

⚠️ THE CLOSURE IS DENSE: one BYTE per (pool literal, literal) pair — 83 MiB for an unrolled b03
(7,588 free literals x 2N=11,402) and tens of GB for a big HWMCC unroll. `implications` REFUSES a
pool it cannot hold rather than reserving it; call `closure_bytes()` first and narrow the pool (an
independent support, the primary inputs) if it says no.

Soundness: `l` implying `m` means F & l |= m, so F |= (~l | m) — the binary clause is entailed, and
mutual implication gives F |= (l <-> m), so substituting one for the other preserves every model.
Both are checked against a brute-force reference in tests/test_implications.py.

FUTURE WORK (deliberately not built — this module is the simple sound core):
  * TRANSITIVE REDUCTION for `hyper_binary`. It returns EVERY closure edge, which is ~1000x the
    input formula on a real instance (b03-c25: 14,011,572 binary clauses against M=14,505), and the
    literature's whole point is that most are implied by the binary implication graph. Until that
    filter exists, treat the output as an analysis product, not something to concatenate onto a
    formula.
  * A ONE-DIRECTION `derivable_mask`. Ours requires all k forcing directions, which is strictly
    conservative; the aggressive rule drops ~1.6-2.7x more (see that function's docstring). Getting
    the aggressive one RIGHT needs the binary implication graph, not the propagation relation.
  * A CHUNKED reducer form, so a pool larger than VRAM can be swept in pieces instead of refused.
  * Wiring `equivalence_edges` into the sampler's ELS, which today builds edges from BINARY CLAUSES
    only. Closure edges are a strict superset (0 of 7,856 missing on b03-c25) and worth 1.11-1.32x
    more removable variables — but it is a behaviour change to the shipping sampler and needs its
    own uniformity gate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from gpap.engine.checker import AssumptionChecker
from gpap.engine.probe import LiteralProbe


def _idx(lit: int) -> int:
    """DIMACS literal -> internal 2N index (+v -> 2(v-1), -v -> 2(v-1)+1)."""
    return 2 * (abs(int(lit)) - 1) + (0 if int(lit) > 0 else 1)


def closure_bytes(num_vars: int, pool_size: int) -> int:
    """Device bytes `implications` needs for a pool of this size — check BEFORE calling.

    The same contract as `LiteralProbe.count`: the cost is knowable up front, so an impossible
    request is refused rather than attempted."""
    return int(pool_size) * 2 * int(num_vars)          # (P, 2N) int8


@dataclass
class Implications:
    """What each probed literal implies. Rows are in `literals` order.

    `matrix[i, j]` is 1 when probing `literals[i]` propagates internal literal `j`. Rows for FAILED
    literals are zeroed: such a literal implies everything vacuously, which would otherwise make it
    look equivalent to every other literal, when what it really is is a unit."""

    literals: np.ndarray            # (P,) DIMACS, the pool that was probed
    failed: np.ndarray              # (F,) DIMACS literals whose assumption conflicts
    matrix: torch.Tensor            # (P, 2N) bool
    num_vars: int

    @property
    def units(self) -> list[list[int]]:
        """The unit clauses the failed literals prove — assuming `l` conflicts, so `~l` holds."""
        return [[-int(l)] for l in self.failed]

    def implies(self, lit: int) -> np.ndarray:
        """DIMACS literals that propagating `lit` derives (empty if `lit` was not probed)."""
        where = np.flatnonzero(self.literals == int(lit))
        if where.size == 0:
            return np.empty(0, dtype=np.int32)
        row = np.flatnonzero(self.matrix[int(where[0])].cpu().numpy())
        v = (row >> 1) + 1
        return np.where(row & 1, -v, v).astype(np.int32)

    def __len__(self) -> int:
        return int(self.literals.size)


def implications(num_vars: int, clauses: list, device, *, pool=None,
                 max_bytes: int = 2_000_000_000) -> Implications:
    """Probe every literal in `pool` and record what each one implies.

    `pool`: DIMACS literals. None means every literal the formula's own unit propagation left
    undetermined — probing a settled literal tells you nothing, and including one would put a
    vacuous row in the matrix.

    Raises ValueError when the closure would exceed `max_bytes`; see `closure_bytes`.
    """
    probe = LiteralProbe(num_vars, clauses, device)
    lits = probe.free_literals() if pool is None else np.asarray(
        [int(x) for x in pool], dtype=np.int32)
    del probe
    if lits.size == 0:
        return Implications(literals=lits, failed=lits[:0],
                            matrix=torch.zeros((0, 2 * num_vars), dtype=torch.bool, device=device),
                            num_vars=int(num_vars))
    need = closure_bytes(num_vars, lits.size)
    if need > max_bytes:
        raise ValueError(
            f"the closure for {lits.size:,} literals x {2 * num_vars:,} would need "
            f"{need / 2**30:.2f} GiB (limit {max_bytes / 2**30:.2f} GiB). Narrow the pool — an "
            f"independent support or the primary inputs — or raise max_bytes.")

    ck = AssumptionChecker(device=device)
    ck.formula.init_formula(int(num_vars), clauses)
    out = ck.check_assumptions([[int(l)] for l in lits], assignments=True)
    conflict = out["conflict"]
    matrix = out.pop("assignment").bool()   # pop: do not hold the int8 source AND the bool copy
    # A failed literal implies EVERYTHING (the row is a conflicted snapshot, not a fixpoint), so it
    # is not a source of implications — it is a unit. Zeroing keeps every consumer honest at once.
    matrix[conflict] = False
    return Implications(literals=lits, failed=lits[conflict.cpu().numpy()], matrix=matrix,
                        num_vars=int(num_vars))


def hyper_binary(imp: Implications) -> list[list[int]]:
    """Every binary clause the closures prove: `m` in closure(l) gives `(~l | m)`.

    ⚠️ UNFILTERED, and the count is the thing to look at before using it. Every one of these is
    entailed but LOGICALLY REDUNDANT by construction — propagation derives `m` from `l` already,
    which is how they were found — and there is one per closure edge, so the output is typically
    far LARGER than the formula: b03-c25 gives 14,011,572 binary clauses against M=14,505 (966x),
    8,658 of which are verbatim copies of input clauses. This is an analysis product. Do NOT
    concatenate it onto a formula and expect the +29-43% that mining's few thousand clauses buy —
    that measurement is `gpap.mining`'s, not this function's.

    The literature's answer is transitive reduction against the binary implication graph, which is
    what makes HBR useful in a CDCL preprocessor. Not written here — see the module FUTURE WORK.
    """
    rows, cols = torch.nonzero(imp.matrix, as_tuple=True)
    if rows.numel() == 0:
        return []
    src = imp.literals[rows.cpu().numpy()].astype(np.int64)
    j = cols.cpu().numpy().astype(np.int64)
    v = (j >> 1) + 1
    dst = np.where(j & 1, -v, v)
    keep = np.abs(dst) != np.abs(src)              # a literal implying itself is not a clause
    return np.stack([-src[keep], dst[keep]], axis=1).astype(np.int32).tolist()


def equivalence_edges(imp: Implications) -> np.ndarray:
    """Mutually-implied literal pairs, as (E, 2) INTERNAL indices — the input ELS wants.

    `l == m` when each propagates the other. Both directions are emitted, because
    `preprocess.equivalent_literal_substitution` looks for the mutual pair itself.

    This is strictly stronger than building edges from binary clauses, which is what the sampler
    does today: two mutual binary clauses obviously propagate each other, so every pair that route
    finds is here too (verified: 0 of 7,856 missing on b03-c25), plus the ones that only exist
    THROUGH longer clauses. Measured 1.11-1.32x more removable variables on unrolled circuits.
    """
    if len(imp) == 0:
        return np.empty((0, 2), dtype=np.int64)
    cols = torch.as_tensor([_idx(int(l)) for l in imp.literals], device=imp.matrix.device)
    sq = imp.matrix[:, cols]                      # (P, P): does row i imply pool literal j?
    mutual = sq & sq.T.clone()
    ii, jj = torch.nonzero(mutual, as_tuple=True)
    keep = ii != jj
    a = cols[ii[keep]].cpu().numpy()
    b = cols[jj[keep]].cpu().numpy()
    return np.stack([a, b], axis=1).astype(np.int64)


def equivalence_classes(imp: Implications) -> list[list[int]]:
    """The same information as `equivalence_edges`, grouped: lists of mutually-equal DIMACS
    literals, each of length >= 2. For inspection and for callers that do their own substitution."""
    edges = equivalence_edges(imp)
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s, d in edges.tolist():
        a, b = find(int(s)), find(int(d))
        if a != b:
            parent[a] = b
    groups: dict[int, list[int]] = {}
    for x in list(parent):
        groups.setdefault(find(x), []).append(x)
    out = []
    for g in groups.values():
        if len(g) > 1:
            out.append(sorted((((i >> 1) + 1) * (-1 if i & 1 else 1)) for i in g))
    return sorted(out)


def derivable_mask(clauses, base_clauses, num_vars: int, device) -> np.ndarray:
    """Which of `clauses` can propagation over `base_clauses` fully replace? (n,) bool.

    A clause `(l1|...|lk)` does k separate JOBS: when the other k-1 literals are false it forces
    the remaining one, and which job fires depends on the search. This marks a clause only when
    propagation over `base_clauses` covers EVERY one of them — for each j, assuming the other
    literals false either conflicts (the consumer gets a backtrack, at least as informative) or
    already forces `lj`. Then the clause adds no propagation power in any direction and a consumer
    that wants a minimal formula can drop it.

    ⚠️ Derivable is NOT the same as useless. A derivable clause still SHORTENS the derivation, and
    GPAP's sampler is propagation-bound, so keeping them is measured at +29-43% end-to-end. Drop
    them for a consumer that wants a minimal formula (a CDCL preprocessor), not for one that wants
    speed. That is why `mine_entailed(drop=...)` defaults to keeping them.

    ⚠️ CONSERVATIVE, deliberately — see the module docstring's FUTURE WORK. Requiring all k jobs
    makes this weaker than the literature's transitive reduction for hyper-binary resolvents, which
    can treat one direction as enough because it runs on the binary implication graph, where a
    binary clause IS both of its edges. Our reference is the PROPAGATION relation, which also holds
    implications squeezed out of long clauses going unit, and those are one-way: `(-1|5),(-1|6),
    (-5|-6|2)` propagates 1 => 2 but not -2 => -1, because the backward step would have to conclude
    "not both 5 and 6" and unit propagation can only conclude a single literal. Measured with a
    one-direction-is-enough rule instead: 95.4% vs 60.7% dropped on b03-c25, 54.5% vs 20.3% on
    b09-c25 — i.e. a THIRD of the mined wide clauses have a live direction the weaker rule
    silently discards. Never a soundness question either way (every mined clause is entailed, so
    the model set is unchanged whatever is dropped) — only propagation strength.

    One batched launch: `sum(len(c) for c in clauses)` assumption sets.
    """
    wide = [list(c) for c in clauses if len(c) > 1]
    out = np.zeros(len(clauses), dtype=bool)
    if not wide:
        return out
    ck = AssumptionChecker(device=device)
    ck.formula.init_formula(int(num_vars), [list(c) for c in base_clauses])
    sets, tag = [], []
    for ci, cl in enumerate([c for c in clauses if len(c) > 1]):
        for j, want in enumerate(cl):
            sets.append([-int(x) for k, x in enumerate(cl) if k != j])
            tag.append((ci, int(want)))
    res = ck.check_assumptions(sets, assignments=True)
    conflict = res["conflict"].cpu().numpy()
    asg = res["assignment"]
    cols = torch.as_tensor([_idx(w) for _, w in tag], device=asg.device)
    forced = asg[torch.arange(len(tag), device=asg.device), cols].bool().cpu().numpy()
    hit = conflict | forced
    # EVERY job must be covered, not just one (see the docstring). Count per clause.
    wide_pos = [i for i, c in enumerate(clauses) if len(c) > 1]
    owner = np.array([ci for ci, _ in tag], dtype=np.int64)
    jobs = np.bincount(owner, minlength=len(wide))
    done = np.bincount(owner, weights=hit, minlength=len(wide))
    out[np.asarray(wide_pos, dtype=np.int64)[jobs == done]] = True
    return out
