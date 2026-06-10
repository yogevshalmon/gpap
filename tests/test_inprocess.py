"""Isolation tests for the GPU inprocessing discovery primitives (gpap/solver/inprocess.py).

Uses a tiny hand-checkable CNF whose failed literal and discoverable binary are known by
construction, plus a RUP soundness gate: every discovered clause, when its literals are
negated and assumed, must propagate to a conflict in CaDiCaL (i.e. it is entailed).

Formula (5 vars):
    c1: (¬1 ∨ 2)        ┐ assuming x1=TRUE forces both x2 and ¬x2  → conflict
    c2: (¬1 ∨ ¬2)       ┘ so ¬1 is a FAILED-LITERAL unit
    c3: (¬3 ∨ ¬4 ∨ 5)   ┐ assuming x3=x4=TRUE forces both x5 and ¬x5 → conflict
    c4: (¬3 ∨ ¬4 ∨ ¬5)  ┘ so (¬3 ∨ ¬4) is a discoverable BINARY (resolvent of c3,c4)
"""
import torch

from pysat.formula import CNF
from pysat.solvers import Cadical195

from gpap.solver.inprocess import (
    GPUInprocessor,
    generate_subset_pairs_chunks_gpu,
)

CLAUSES = [[-1, 2], [-1, -2], [-3, -4, 5], [-3, -4, -5]]
NUM_VARS = 5
CPU = torch.device("cpu")


def _inproc() -> GPUInprocessor:
    return GPUInprocessor.from_clauses(NUM_VARS, CLAUSES, CPU)


def _norm(clauses):
    """Order-insensitive normal form for comparing a set of clauses."""
    return sorted(tuple(sorted(c)) for c in clauses)


class TestSubsetGenerators:
    def test_pairs_over_two_vars(self):
        chunks = list(generate_subset_pairs_chunks_gpu([2, 3], 1000, CPU))  # 0-based vars x3,x4
        assert len(chunks) == 1
        rows = sorted(map(tuple, chunks[0].tolist()))
        # vars 2,3 → lit indices {4,5} × {6,7}, all 4 polarity combos
        assert rows == [(4, 6), (4, 7), (5, 6), (5, 7)]

    def test_pairs_empty_for_singleton(self):
        assert list(generate_subset_pairs_chunks_gpu([1], 1000, CPU)) == []


class TestDiscovery:
    def test_probe_finds_the_failed_literal(self):
        clauses, stats = _inproc().probe_failed_literals()
        assert _norm(clauses) == [(-1,)]
        assert stats["n_new"] == 1
        assert stats["n_rows"] == 2 * NUM_VARS

    def test_probe_dedup_across_calls(self):
        inp = _inproc()
        first, _ = inp.probe_failed_literals()
        second, _ = inp.probe_failed_literals()
        assert len(first) == 1
        assert second == []  # already emitted → not returned again

    def test_discover_binary_over_subset(self):
        # Restrict to vars x3,x4 (0-based 2,3) so the only conflict is the intended one.
        clauses, _ = _inproc().discover_binaries([2, 3])
        assert _norm(clauses) == [(-4, -3)]  # (¬3 ∨ ¬4)

    def test_discover_binaries_anchored(self):
        # Anchor on x3 (0-based 2), paired with every other variable. (x3,x4) conflicts → (¬3∨¬4).
        clauses, _ = _inproc().discover_binaries_anchored([2])
        norm = _norm(clauses)
        assert (-4, -3) in norm
        assert all(len(c) == 2 for c in clauses)            # all binaries
        assert all(3 in (abs(c[0]), abs(c[1])) for c in clauses)  # every pair touches the anchor x3

    def test_run_orders_shortest_first_and_caps(self):
        clauses, stats = _inproc().run(
            probe=True, pairs_vars=list(range(NUM_VARS)),
            triples_vars=list(range(NUM_VARS)), max_clauses=3,
        )
        assert len(clauses) <= 3
        lengths = [len(c) for c in clauses]
        assert lengths == sorted(lengths)        # shortest-first
        assert clauses[0] == [-1]                # the unit is kept first


class TestRupSoundness:
    """Every discovered clause must be entailed: assuming its negation conflicts in CaDiCaL."""

    def test_all_discovered_clauses_are_rup(self):
        inp = _inproc()
        units, _ = inp.probe_failed_literals()
        bins, _ = inp.discover_binaries(list(range(NUM_VARS)))
        terns, _ = inp.discover_ternaries(list(range(NUM_VARS)))
        discovered = units + bins + terns
        assert discovered, "expected to discover at least one clause"
        # The headline clauses must be among them.
        norm = _norm(discovered)
        assert (-1,) in norm
        assert (-4, -3) in norm

        cnf = CNF(from_clauses=CLAUSES)
        with Cadical195(bootstrap_with=cnf) as solver:
            for c in discovered:
                st, _ = solver.propagate(assumptions=[-lit for lit in c])
                assert not st, f"discovered clause {c} is NOT RUP (no conflict on ¬c)"


class TestDriverSoundness:
    """The CaDiCaL + GPU-inprocessing driver must return the same verdict as a plain solve."""

    def test_verdict_matches_plain(self, tmp_path):
        from gpap.solver.cadical_hook import solve_plain, solve_with_gpu_inprocessing

        path = str(tmp_path / "tiny.cnf")
        CNF(from_clauses=CLAUSES).to_file(path)

        plain = solve_plain(path)
        inproc = solve_with_gpu_inprocessing(
            path, device=CPU, mode="pairs", top_k=5, segment_conflicts=1, verbose=False,
        )
        assert plain["result"] == "SAT"
        assert inproc["result"] == plain["result"]
