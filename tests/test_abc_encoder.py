"""The ABC circuit encoder must not change WHAT WE SAMPLE.

Load-bearing claim: `--encoder abc` (unroll + dc2/fraig via berkeley-abc, then Tseitin its AIG)
yields a CNF whose set of SUPPORT PROJECTIONS — the primary-input assignments consistent with the
pinned outputs — is EXACTLY the native Tseitin encoder's. Only internal gate variables differ.

These are CPU tests. They skip when no `abc` binary is present (it is an optional dependency;
build it with scripts/build_abc.sh).
"""
import os

import pytest
from pysat.formula import CNF
from pysat.solvers import Solver

from gpap.circuits.abc_backend import abc_available, build_cnf_abc
from gpap.circuits.bench import build_cnf, parse_bench, simulate

pytestmark = pytest.mark.skipif(not abc_available(), reason="no `abc` binary (scripts/build_abc.sh)")

BENCH = "benchmarks/circuits/bench/seq"
# Small enough that the projected solution set can be enumerated exhaustively.
CASES = [("b01", 3), ("b01", 5), ("b02", 4), ("b06", 3), ("b09", 4), ("b03", 2)]


def _enum_projections(nv, clauses, support, limit=200_000):
    """All distinct assignments to `support` (1-based vars) that extend to a model."""
    s = Solver(name="cadical195", bootstrap_with=CNF(from_clauses=clauses))
    out = set()
    while s.solve() and len(out) < limit:
        m = set(s.get_model())
        proj = tuple(1 if v in m else 0 for v in support)
        out.add(proj)
        s.add_clause([(-v if v in m else v) for v in support])   # block this projection
    s.delete()
    return out


@pytest.mark.parametrize("opt", ["full", "fast"])
@pytest.mark.parametrize("name,cycles", CASES)
def test_abc_preserves_projected_solution_set(name, cycles, opt):
    path = os.path.join(BENCH, f"{name}.bench")
    circ = parse_bench(path)
    seed = 3
    bt, sup_t, tgt_t, lab_t = build_cnf(circ, cycles, seed, pin_outputs=None)
    ba, sup_a, tgt_a, lab_a = build_cnf_abc(circ, path, cycles, seed, pin_outputs=None, opt=opt)

    # Same support ORDER and the same pinned target => the two CNFs must describe the same set.
    assert lab_t == lab_a
    assert tgt_t == tgt_a
    assert len(sup_t) == len(sup_a)

    proj_t = _enum_projections(bt.nv, bt.clauses, sup_t)
    proj_a = _enum_projections(ba.nv, ba.clauses, sup_a)
    assert proj_t, "native encoder produced an UNSAT/empty instance — bad test case"
    assert proj_a == proj_t, (f"{name} T={cycles}: ABC changed the projected solution set "
                              f"(native {len(proj_t)} vs abc {len(proj_a)} projections)")


@pytest.mark.parametrize("name,cycles", [("b01", 4), ("b10", 3), ("b03", 3)])
def test_abc_encoder_is_smaller_and_short_clauses(name, cycles):
    """ABC's AIG encoding should not be larger, and must emit only 2- and 3-literal clauses
    (plus unit pins) — that uniformity is what the sum-trick BCP / packed kernel want."""
    path = os.path.join(BENCH, f"{name}.bench")
    circ = parse_bench(path)
    bt, _s, _t, _l = build_cnf(circ, cycles, 3, pin_outputs=None)
    ba, _s2, _t2, _l2 = build_cnf_abc(circ, path, cycles, 3, pin_outputs=None)
    assert len(ba.clauses) <= len(bt.clauses)
    assert max(len(c) for c in ba.clauses) <= 3


def test_abc_unroll_matches_our_simulator():
    """ABC's `zero; frames -F T -i` must reproduce our reset-state-0 unroll exactly: pin the
    outputs to OUR simulated target and the ABC CNF must still be satisfiable."""
    path = os.path.join(BENCH, "b10.bench")
    circ = parse_bench(path)
    inputs, outputs, _d, _g = circ
    T, seed = 6, 3
    ba, sup, target, _lab = build_cnf_abc(circ, path, T, seed, pin_outputs=None)
    s = Solver(name="cadical195", bootstrap_with=CNF(from_clauses=ba.clauses))
    assert s.solve(), "ABC CNF UNSAT under a reachable simulated target — unroll semantics differ"
    m = set(s.get_model())
    s.delete()
    # The model's primary inputs, replayed through OUR simulator, must reproduce the pinned target.
    bits = {v: (v in m) for v in sup}
    in_seq = [{inp: bits[sup[i * T + t]] for i, inp in enumerate(inputs)} for t in range(T)]
    assert simulate(circ, in_seq, T) == target
