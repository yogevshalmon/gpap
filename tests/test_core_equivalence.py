"""Equivalence gate: the C++ core (libgpap) vs the historical torch/Triton engine.

The two must agree EXACTLY — same formula structures, same conflict flags, same assignments,
same models — because they run the same AOT-vs-JIT compiled kernels over the same state layout
(the cubins were verified bit-identical at the SASS level). Anything less than exact agreement
here is a port bug, not a tolerance.

Semantic ground truth remains the pure-Python reference BCP (reference_bcp.py); these tests add
the core-vs-engine comparison on top so the migration cannot silently change behaviour.
"""
import random

import numpy as np
import pytest
import torch

from gpap.engine import _core
from gpap.engine.bcp_engine import make_assumption_tensor
from gpap.engine._torch_engine import BCPEngine
from gpap.engine.formula import CnfFormula
from gpap.engine.kernels import kernel_available

from tests.reference_bcp import assigned_set, ref_bcp

CUDA = torch.cuda.is_available()
AVAIL = CUDA and kernel_available()
HAVE_CORE = _core.available()
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not AVAIL, reason="requires CUDA + a working triton"),
    pytest.mark.skipif(not HAVE_CORE, reason="libgpap not built (see cpp/README.md)"),
]
DEV = torch.device("cuda") if CUDA else torch.device("cpu")


def _random_cnf(rng, nv, nc, kmax=4):
    out = []
    for _ in range(nc):
        vs = rng.sample(range(1, nv + 1), rng.randint(1, min(kmax, nv)))
        out.append([v if rng.random() < 0.5 else -v for v in vs])
    return out


def _both(num_vars, clauses, mode="scalar"):
    """(torch engine, core engine) built over the same clause list."""
    f = CnfFormula(DEV)
    f.init_formula(num_vars, clauses)
    eng = BCPEngine(f, mode=mode)
    core = _core.Engine(_core.Formula.from_clauses(num_vars, clauses), mode=mode)
    return eng, core


# ---------------------------------------------------------------- formula structures

def test_formula_structures_match():
    """The core is now the ONLY formula builder; check it reproduces the numpy one exactly."""
    rng = random.Random(7)
    for nv, nc in [(6, 10), (40, 120), (200, 900)]:
        clauses = _random_cnf(rng, nv, nc)
        f = CnfFormula(DEV)
        f.init_formula(nv, clauses)
        cf = _core.Formula.from_clauses(nv, clauses)

        assert cf.info.num_vars == f.num_vars
        assert cf.info.num_clauses == f.num_clauses
        off, cls_ = cf.occ()
        np.testing.assert_array_equal(off, f.occ_list.occ_offsets.cpu().numpy())
        np.testing.assert_array_equal(cls_, f.occ_list.occ_clauses.cpu().numpy())
        sizes, lsum = cf.clause_arrays()
        np.testing.assert_array_equal(sizes, f.clause_sizes.cpu().numpy())
        np.testing.assert_array_equal(lsum, f.init_lit_sum.cpu().numpy())


def test_dimacs_parse_matches(tmp_path):
    rng = random.Random(11)
    clauses = _random_cnf(rng, 30, 80)
    p = tmp_path / "f.cnf"
    with open(p, "w") as fh:
        fh.write("c a comment\np cnf 30 %d\n" % len(clauses))
        for c in clauses:                       # two clauses per line: the token stream matters
            fh.write(" ".join(map(str, c)) + " 0 ")
    f = CnfFormula(DEV)
    f.parse_cnf_file(str(p))
    cf = _core.Formula.from_file(str(p))
    assert (cf.info.num_vars, cf.info.num_clauses) == (f.num_vars, f.num_clauses)
    np.testing.assert_array_equal(cf.clause_arrays()[0], f.clause_sizes.cpu().numpy())


def test_k_calibration_matches():
    """The K model exists in both faces (C++ for library users, Python for the sampler). It is
    15 lines of arithmetic that was validated under the nvidia-smi kill-switch — this test is
    what keeps the two copies from drifting apart."""
    from gpap.engine.calibrate import compute_k_safe as py_k
    for M, N, total in [(4407, 1701, 11000), (109432, 42051, 250000), (201437, 77151, 500000)]:
        for budget in (1 << 30, 6 << 30):
            for cap in (0, 16384):
                assert _core.compute_k_safe(M, N, budget, cap or None, 6, total,
                                            deduct_formula=False) == \
                    py_k(M, N, budget, cap=cap or None, max_clause_size=6, total_lits=total,
                         deduct_formula=False)


def test_envelope_matches():
    from gpap.engine.bcp_engine import packed_envelope
    for nv, ml in [(10, 3), (32767, 5), (33000, 5), (40000, 300), (9_000_000, 3)]:
        assert _core.packed_envelope(nv, ml)[0] == packed_envelope(nv, ml)[0]


# ---------------------------------------------------------------- entry point 1

@pytest.mark.parametrize("mode", ["scalar", "wave"])
def test_apply_matches_engine_and_reference(mode):
    rng = random.Random(3)
    for trial in range(8):
        nv, nc = 25, 70
        clauses = _random_cnf(rng, nv, nc)
        K, L = 16, 3
        dec = [[rng.randrange(2 * nv) for _ in range(L)] for _ in range(K)]

        eng, core = _both(nv, clauses, mode)
        eng.reset(K)
        eng.apply_and_propagate(make_assumption_tensor(dec, DEV))
        core.reset(K)
        core.apply(np.array(dec, dtype=np.int64))

        t_conf = eng.conflict.cpu().numpy()
        c_act, c_conf = core.flags()
        np.testing.assert_array_equal(t_conf, c_conf)
        np.testing.assert_array_equal(eng.active.cpu().numpy(), c_act)

        # every row: core conflict == oracle conflict, and the fixpoint == the oracle's
        t_asg = eng.assignment.cpu().numpy()
        for k in range(K):
            ref_conflict, ref_asg = ref_bcp(clauses, dec[k])
            assert bool(c_conf[k]) == ref_conflict, f"row {k} trial {trial}"
            if not ref_conflict:
                assert assigned_set(t_asg[k]) == ref_asg
        np.testing.assert_array_equal(eng.num_propagated.cpu().numpy(), core.num_propagated())


@pytest.mark.parametrize("mode", ["scalar", "wave"])
def test_apply_assignments_match(mode):
    """Full (K, 2N) assignment agreement on non-conflicted rows, via the selection readout."""
    rng = random.Random(19)
    nv, nc, K = 20, 55, 12
    clauses = _random_cnf(rng, nv, nc)
    dec = [[rng.randrange(2 * nv)] for _ in range(K)]
    eng, core = _both(nv, clauses, mode)
    eng.reset(K)
    eng.apply_and_propagate(make_assumption_tensor(dec, DEV))
    core.reset(K)
    core.apply(np.array(dec, dtype=np.int64))

    t_asg = eng.assignment.cpu().numpy()
    mid = core.pair_mask(None)
    n = core.select_complete(mid, 0)
    rows = core.selection_rows(0, n)
    if n:
        c_rows = core.assignment_rows(0, n)
        for i, r in enumerate(rows):
            np.testing.assert_array_equal(t_asg[r], c_rows[i])


def test_flush_cache_and_units():
    """The formula's own unit clauses are asserted once at construction in both engines."""
    clauses = [[1], [-1, 2], [-2, 3], [4, 5]]
    eng, core = _both(5, clauses)
    np.testing.assert_array_equal(eng.flushed_assignment.cpu().numpy(), core.flushed_assignment())
    assert eng.flush_conflict == core.flush_conflict


def test_unsat_units_conflict_every_row():
    clauses = [[1], [-1]]
    eng, core = _both(2, clauses)
    assert eng.flush_conflict and core.flush_conflict
    eng.reset(4)
    core.reset(4)
    assert bool(eng.conflict.all()) and bool(core.flags()[1].all())


# ---------------------------------------------------------------- entry point 2

@pytest.mark.parametrize("mode", ["scalar", "wave"])
@pytest.mark.parametrize("B", [1, 8])
def test_trajectory_models_identical(mode, B):
    """THE gate: same order, same seed, same K => the same models, row for row."""
    rng = random.Random(5)
    nv, nc, K, seed = 60, 150, 512, 1234567
    clauses = _random_cnf(rng, nv, nc)
    eng, core = _both(nv, clauses, mode)

    order = torch.randperm(nv, generator=torch.Generator(device=DEV).manual_seed(9),
                           device=DEV).to(torch.int32).contiguous()
    eng.reset(K)
    r_t = eng.run_trajectory(order, nv, B, seed)
    core.reset(K)
    r_c = core.trajectory_dev(order.data_ptr(), nv, nv, B, seed)
    assert r_t == r_c

    pm = eng.make_pair_mask(None)
    t_fully = (eng.active & ~eng.any_free(pm)).cpu().numpy()
    mid = core.pair_mask(None)
    n = core.select_complete(mid, 0)
    assert n == int(t_fully.sum()), f"{n} core models vs {int(t_fully.sum())} torch models"
    np.testing.assert_array_equal(core.selection_rows(0, n), np.nonzero(t_fully)[0])
    if n:
        np.testing.assert_array_equal(
            core.assignment_rows(0, n),
            eng.assignment_rows(torch.from_numpy(t_fully).to(DEV)).cpu().numpy())
        assert core.check_sound(0) == 0
        # every emitted model must satisfy every clause (independent of both engines)
        bits = core.var_bits(0, np.arange(nv, dtype=np.int32), n)
        for row in bits[: min(n, 32)]:
            for c in clauses:
                assert any((row[abs(l) - 1] == 1) == (l > 0) for l in c)


@pytest.mark.parametrize("mode", ["scalar", "wave"])
def test_trajectory_counters_match(mode):
    rng = random.Random(23)
    nv, nc, K = 40, 110, 256
    clauses = _random_cnf(rng, nv, nc)
    eng, core = _both(nv, clauses, mode)
    order = torch.arange(nv, device=DEV, dtype=torch.int32).contiguous()
    eng.reset(K)
    eng.run_trajectory(order, nv, 4, 999)
    core.reset(K)
    core.trajectory_dev(order.data_ptr(), nv, nv, 4, 999)
    c = core.counters()
    assert (c.ops, c.props, c.decs) == (eng.ops_total(), eng.props_total(), eng.decs_total())


@pytest.mark.slow
@pytest.mark.parametrize("name", ["b03-c25", "b06-c10", "s386-c10"])
def test_circuit_models_identical(circuit_cnfs, name):
    """The real gate at production scale: an unrolled circuit CNF, a torch-built decision order
    (support first), production-shaped K — the two engines must emit the SAME model set."""
    if name not in circuit_cnfs:
        pytest.skip(f"{name} not in the generated matrix")
    inst = circuit_cnfs[name]
    nv, clauses, support = inst["nv"], inst["clauses"], inst["support"]
    eng, core = _both(nv, clauses)

    gen = torch.Generator(device=DEV).manual_seed(5)
    rank = torch.argsort(torch.randperm(nv, generator=gen, device=DEV))
    smask = torch.zeros(nv, dtype=torch.bool, device=DEV)
    smask[torch.tensor(support, dtype=torch.long, device=DEV)] = True
    in_sup = smask[rank]
    order = torch.cat([rank[in_sup], rank[~in_sup]]).to(torch.int32).contiguous()
    n_sup = int(in_sup.sum().item())

    K, B, seed = 2048, 8, 20260728
    eng.reset(K)
    r_t = eng.run_trajectory(order, n_sup, B, seed)
    core.reset(K)
    r_c = core.trajectory_dev(order.data_ptr(), order.numel(), n_sup, B, seed)
    assert r_t == r_c

    pm = eng.make_pair_mask(None)
    t_fully = (eng.active & ~eng.any_free(pm)).cpu().numpy()
    n = core.select_complete(core.pair_mask(None), 0)
    assert n == int(t_fully.sum())
    if n:
        np.testing.assert_array_equal(
            core.assignment_rows(0, n),
            eng.assignment_rows(torch.from_numpy(t_fully).to(DEV)).cpu().numpy())
    assert core.check_sound(0) == 0


# ---------------------------------------------------------------- degenerate shapes
# Each of these was a real bug found while porting: the flush cache used a null packed pointer
# as its "not built yet" sentinel (which M == 0 also produces), the core rejected num_vars == 0,
# and it rejected an empty decision order. All three are reachable through simplification.

def test_m0_formula_resets_every_row():
    """A formula with no clauses: reset must still restore ALL K rows, not just row 0."""
    eng, core = _both(4, [])
    for e in (eng, core):
        e.reset(2)
    eng.apply_and_propagate(make_assumption_tensor([[0], [3]], DEV))
    core.apply(np.array([[0], [3]], dtype=np.int64))
    act, conf = core.flags()
    assert act.all() and not conf.any()
    np.testing.assert_array_equal(core.rows_assignment(np.array([0, 1], dtype=np.int32)),
                                  eng.assignment.cpu().numpy())


def test_zero_variable_formula():
    """Unit propagation / ELS can determine EVERY variable, leaving an empty formula."""
    cf = _core.Formula.from_clauses(0, [])
    assert cf.info.num_vars == 0 and cf.info.num_clauses == 0
    core = _core.Engine(cf)
    core.reset(8)
    assert core.trajectory(np.zeros(0, dtype=np.int32), 0, 8, 42) == 0   # empty order = no-op
    # nothing is free, so every active row is a (vacuous) complete model
    assert core.select_complete(core.pair_mask(None), 0) == 8


def test_zero_column_readout_is_a_noop():
    """Gathering ZERO variables is legal: the sampler's collect path asks for all N variables,
    and after ELS/unit-propagation N can be 0."""
    core = _core.Engine(_core.Formula.from_clauses(0, []))
    core.reset(4)
    n = core.select_complete(core.pair_mask(None), 0)
    assert core.var_bits(0, np.zeros(0, dtype=np.int32), n).shape == (n, 0)
    assert core.assignment_rows(0, n).shape == (n, 0)


def test_readout_never_writes_past_the_caller_buffer():
    """The three gathers take an explicit row CAPACITY and write min(capacity, selection).

    Without it their output size was only *implied* by the selection count, so a caller whose
    count was stale by one chunk silently overflowed the heap. Asking for fewer rows than are
    selected must truncate — and truncate to the FIRST rows, not to arbitrary ones."""
    rng = random.Random(77)
    nv = 12
    clauses = _random_cnf(rng, nv, 18)
    core = _core.Engine(_core.Formula.from_clauses(nv, clauses))
    core.reset(64)
    core.trajectory(np.arange(nv, dtype=np.int32), nv, 4, 2024)
    n = core.select_complete(core.pair_mask(None), 0)
    assert n > 2, "need several complete rows to have something to truncate"

    cols = np.arange(nv, dtype=np.int32)
    full_bits = core.var_bits(0, cols, n)
    full_asg = core.assignment_rows(0, n)
    full_rows = core.selection_rows(0, n)
    half = n // 2
    np.testing.assert_array_equal(core.var_bits(0, cols, half), full_bits[:half])
    np.testing.assert_array_equal(core.assignment_rows(0, half), full_asg[:half])
    np.testing.assert_array_equal(core.selection_rows(0, half), full_rows[:half])
    # Asking for MORE than exists is also safe: the extra rows are simply not written.
    assert core.var_bits(0, cols, n + 5).shape == (n + 5, nv)


def test_empty_decision_order_is_a_noop():
    rng = random.Random(41)
    clauses = _random_cnf(rng, 8, 12)
    core = _core.Engine(_core.Formula.from_clauses(8, clauses))
    core.reset(4)
    before = core.flags()[0].copy()
    assert core.trajectory(np.zeros(0, dtype=np.int32), 0, 4, 7) == 0
    np.testing.assert_array_equal(core.flags()[0], before)


def test_out_of_envelope_formula_builds_but_engine_rejects():
    """The packed split is the ENGINE's choice: shape queries still work on a formula that no
    split fits, and suite drivers catch a ValueError from the engine to skip the instance."""
    cf = _core.Formula.from_clauses(40_000, [list(range(1, 300))])   # 2N>65535 AND len>255
    assert cf.info.shift == 0 and cf.info.num_vars == 40_000
    with pytest.raises(ValueError):
        _core.Engine(cf)


def test_wide_split_engine_agrees():
    """8|24 'wide' split: 2N > 65535 forces the second envelope in both engines."""
    rng = random.Random(31)
    nv = 40000
    clauses = _random_cnf(rng, nv, 60000, kmax=4)
    eng, core = _both(nv, clauses)
    assert eng.shift == core.shift == 8
    order = torch.randperm(nv, generator=torch.Generator(device=DEV).manual_seed(2),
                           device=DEV).to(torch.int32).contiguous()
    K = 64
    eng.reset(K)
    eng.run_trajectory(order, nv, 8, 4242)
    core.reset(K)
    core.trajectory_dev(order.data_ptr(), nv, nv, 8, 4242)
    pm = eng.make_pair_mask(None)
    t_n = int((eng.active & ~eng.any_free(pm)).sum().item())
    assert core.select_complete(core.pair_mask(None), 0) == t_n
