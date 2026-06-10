"""Tests for the GPU preprocessing pipeline (gpap/solver/preprocess.py + vivify_clauses).

Two kinds of check:
  1. Hand-checkable reductions on tiny CNFs whose vivified / unit-closed form is known.
  2. A RANDOMIZED EQUIVALENCE GATE — the load-bearing soundness test. Every operation here is
     supposed to preserve the model set, so for many random CNFs the preprocessed formula must
     have the *same* SAT/UNSAT verdict as the original (checked with CaDiCaL), and any model of
     the preprocessed formula must satisfy the original. A single mismatch means an unsound
     reduction.
"""
import random

import torch
from pysat.formula import CNF
from pysat.solvers import Cadical195

from gpap.solver.inprocess import GPUInprocessor
from gpap.solver.preprocess import (
    apply_learned_subsumption,
    preprocess,
    subsume_all,
    unit_closure,
)

CPU = torch.device("cpu")


def _norm(clauses):
    return sorted(tuple(sorted(set(c))) for c in clauses)


def _verdict(clauses, nv):
    with Cadical195(bootstrap_with=CNF(from_clauses=[list(c) for c in clauses])) as s:
        r = s.solve()
        model = s.get_model() if r else None
    return ("SAT" if r else "UNSAT"), model


def _satisfies(clauses, model):
    """Does `model` (list of signed vars) satisfy every clause?"""
    truth = {abs(l): (l > 0) for l in model}
    for c in clauses:
        if not any(truth.get(abs(l), l > 0) == (l > 0) for l in c):
            return False
    return True


# --------------------------------------------------------------------------- vivification

class TestVivify:
    def test_drops_redundant_literal(self):
        # (¬3 ∨ ¬4 ∨ 5) ∧ (¬3 ∨ ¬4 ∨ ¬5): assuming 3,4 forces 5 and ¬5 → conflict, so both
        # clauses vivify to (¬3 ∨ ¬4) [the 5 / ¬5 literal is redundant].
        clauses = [[-3, -4, 5], [-3, -4, -5]]
        inp = GPUInprocessor.from_clauses(5, clauses, CPU)
        new, stats = inp.vivify_clauses(clauses)
        assert _norm(new) == _norm([[-3, -4], [-3, -4]])
        assert stats["n_vivified"] == 2
        assert stats["lits_removed"] == 2

    def test_shortens_to_unit(self):
        # (¬1 ∨ 2) ∧ (¬1 ∨ ¬2): assuming 1 forces 2 and ¬2 → conflict, so assuming ¬1-prefix on
        # each clause conflicts at p=1 → both shorten to the unit (¬1).
        clauses = [[-1, 2], [-1, -2]]
        inp = GPUInprocessor.from_clauses(2, clauses, CPU)
        new, _ = inp.vivify_clauses(clauses)
        assert _norm(new) == _norm([[-1], [-1]])

    def test_leaves_irreducible_clause(self):
        clauses = [[1, 2, 3]]            # nothing entailed → unchanged
        inp = GPUInprocessor.from_clauses(3, clauses, CPU)
        new, stats = inp.vivify_clauses(clauses)
        assert _norm(new) == _norm(clauses)
        assert stats["n_vivified"] == 0


# --------------------------------------------------------------------------- CPU reductions

class TestUnitClosure:
    def test_propagates_and_keeps_units(self):
        # unit (1) ⇒ in (¬1 ∨ 2) strip ¬1 → (2) ⇒ assign 2; (2 ∨ 3) satisfied.
        clauses = [[-1, 2], [2, 3]]
        out, assigned, status = unit_closure(clauses, units=[[1]])
        assert status == "ok"
        assert assigned == {1: True, 2: True}
        norm = _norm(out)
        assert (1,) in norm and (2,) in norm          # asserted units retained (equivalence)
        assert (2, 3) not in norm                       # satisfied clause dropped

    def test_detects_unsat(self):
        # units (1) and (¬1) contradict.
        out, _, status = unit_closure([[5, 6]], units=[[1], [-1]])
        assert status == "unsat"
        assert _verdict(out, 6)[0] == "UNSAT"


class TestSubsumption:
    def test_learned_binary_replaces_longer_clause(self):
        clauses = [[1, 2, 3], [4, 5]]
        out, stats = apply_learned_subsumption(clauses, learned=[[1, 2]])
        assert stats["subsumed"] == 1 and stats["added"] == 1
        assert _norm(out) == _norm([[4, 5], [1, 2]])    # (1∨2∨3) replaced by (1∨2)

    def test_unsubsuming_learned_clause_discarded(self):
        clauses = [[1, 2, 3]]
        out, stats = apply_learned_subsumption(clauses, learned=[[7, 8]])
        assert stats["added"] == 0                      # subsumes nothing → not added (no bloat)
        assert _norm(out) == _norm(clauses)

    def test_subsume_all_drops_supersets_and_dups(self):
        out = subsume_all([[1, 2], [1, 2, 3], [1, 2], [4]])
        assert _norm(out) == _norm([[1, 2], [4]])


# --------------------------------------------------------------------------- equivalence gate

def _rand_cnf(rng, nv, m, k=3):
    clauses = []
    for _ in range(m):
        vs = rng.sample(range(1, nv + 1), min(k, nv))
        clauses.append([v if rng.random() < 0.5 else -v for v in vs])
    return clauses


class TestEquivalenceGate:
    """The preprocessed formula must be logically equivalent to the original (same models)."""

    def test_random_cnfs_preserve_verdict_and_models(self, tmp_path):
        rng = random.Random(20260605)
        n_sat = n_unsat = 0
        for t in range(60):
            nv = rng.randint(4, 14)
            m = rng.randint(nv, nv * 4)            # span the SAT/UNSAT threshold
            clauses = _rand_cnf(rng, nv, m, k=rng.choice([2, 3, 3, 4]))
            path = str(tmp_path / f"r{t}.cnf")
            CNF(from_clauses=clauses).to_file(path)

            base_v, _ = _verdict(clauses, nv)
            res = preprocess(path, CPU, vivify_max_len=14)
            assert res["status"] in ("ok", "unsat")
            if res["status"] == "unsat":
                assert base_v == "UNSAT", f"preprocess claimed UNSAT but base is {base_v}"
                n_unsat += 1
                continue

            pre = res["clauses"]
            pre_v, pre_model = _verdict(pre, nv)
            assert pre_v == base_v, f"verdict changed {base_v}→{pre_v} on seed {t}"
            # reduction must never grow the formula
            assert res["lits_final"] <= res["lits_orig"]
            if base_v == "SAT":
                n_sat += 1
                assert _satisfies(clauses, pre_model), f"pre model violates original (seed {t})"
            else:
                n_unsat += 1
        # the suite must actually exercise both verdicts
        assert n_sat > 0 and n_unsat > 0
