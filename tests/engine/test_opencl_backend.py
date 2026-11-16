"""The OpenCL backend gives the reference's answers, and the CUDA backend's.

`cpp/kernels/bcp.cl` is a second implementation of the fixpoint kernels (scalar and WAVE, plain
and CLEAN) and the support kernels (`GPAP_BACKEND=opencl`, docs/OPENCL.md), so every test here
is an equality. A fixpoint is unique: conflict flags must agree exactly, and so must the
assignment, the packed words and the counters of every row that does not conflict. A conflicted
row's partial state is unspecified, and under the OpenCL WAVE kernel it is not even repeatable:
its lanes race for the trail's order.

Two references. The pure-Python one (`tests/reference/bcp.py`) needs no GPU, so those tests run
on a CPU OpenCL device too, which is how CI executes the kernels. The CUDA backend is compared
on the same machine wherever both exist (`gpu` and `opencl` together).
"""
import os
import random
import re

import numpy as np
import pytest

from gpap.engine import _core
from gpap.mining import mine_entailed

from tests.conftest import REPO_ROOT
from tests.reference.bcp import assigned_set, lit_idx, ref_bcp
from tests.support import DEV, decision_matrix, random_assumptions, random_cnf, use_backend

opencl = pytest.mark.opencl
both = [pytest.mark.gpu, pytest.mark.opencl]


@pytest.fixture(autouse=True)
def _no_kernel_override(monkeypatch):
    """A `GPAP_KERNEL` in the caller's shell would pin a kernel the OpenCL backend may lack."""
    monkeypatch.delenv("GPAP_KERNEL", raising=False)


def _engine(backend, formula, **kw):
    with use_backend(backend):
        return _core.Engine(formula, **kw)


def _formulas(seed, n, nv_max=26):
    """Random formulas with units, binaries and long clauses, skipping the ones whose own units
    already conflict (there is nothing to propagate on those)."""
    rng = random.Random(seed)
    made = 0
    while made < n:
        nv = rng.randint(4, nv_max)
        clauses = random_cnf(rng, nv, rng.randint(nv, 3 * nv), kmin=1, kmax=4)
        f = _core.Formula.from_clauses(nv, clauses)
        if f.info.has_empty_clause or ref_bcp(clauses, [])[0]:
            continue
        made += 1
        yield rng, nv, clauses, f


# ---------------------------------------------------------------- against the reference

@opencl
@pytest.mark.parametrize("mode", ["scalar", "wave"])
@pytest.mark.parametrize("shift", [16, 8])
def test_fixpoint_matches_the_reference(shift, mode):
    """Conflict flags, assignments and per-clause counts of the OpenCL fixpoint kernels equal the
    pure-Python propagation, on both packed splits and both propagate kernels.

    The counts are recomputed from the reference assignment (a literal survives unless its
    negation is assigned), so a wrong packed-word update shows even where it forces nothing."""
    compared = 0
    for rng, nv, clauses, f in _formulas(2026, 20):
        eng = _engine("opencl", f, force_shift=shift, mode=mode)
        assert eng.backend == "opencl" and eng.shift == shift
        K = 24
        sets = random_assumptions(rng, nv, K, wmax=4)
        eng.reset(K)
        assert eng.mode == mode
        eng.apply(decision_matrix(sets))
        active, conflict = eng.flags()
        np.testing.assert_array_equal(active, ~conflict)
        rows = np.arange(K, dtype=np.int32)
        asg = eng.rows_assignment(rows)
        sat = eng.sat_rows(rows)
        built = f.clauses()                     # the engine's clause order
        for i, s in enumerate(sets):
            want_conflict, want_asg = ref_bcp(clauses, [lit_idx(x) for x in s])
            assert bool(conflict[i]) == want_conflict, (shift, mode, nv, s)
            if want_conflict:
                continue
            assert assigned_set(asg[i]) == want_asg, (shift, mode, nv, s)
            want_sat = [sum(lit_idx(-x) not in want_asg for x in c) for c in built]
            assert sat[i].tolist() == want_sat, (shift, mode, nv, s)
            compared += 1
    assert compared > 100, f"only {compared} surviving rows compared"


@opencl
@pytest.mark.parametrize("mode", ["scalar", "wave"])
@pytest.mark.parametrize("shift", [16, 8])
def test_a_clean_launch_restores_the_baseline(shift, mode):
    """After a CLEAN launch every packed row equals the baseline, conflicted rows included; the
    conflict flags equal the reference's and a plain launch's; the surviving rows' implication
    counts equal a plain launch's; and the next reset skips the broadcast.

    The readout is `packed_rows` (raw words): `sat_rows` masks the sum field, where a wrong
    restore could hide. The sets carry duplicates and l/~l pairs, to stop a launch each way."""
    lite = conflicted = 0
    for rng, nv, clauses, f in _formulas(1311, 10, nv_max=20):
        eng = _engine("opencl", f, force_shift=shift, mode=mode)
        twin = _engine("opencl", f, force_shift=shift, mode=mode)
        K = rng.randint(3, 39)
        rows = np.arange(K, dtype=np.int32)
        eng.reset(K)
        baseline = eng.packed_rows(rows)
        for _round in range(3):
            sets = random_assumptions(rng, nv, K, wmax=5)
            for s in sets:
                if s and rng.random() < 0.25:
                    s.append(s[0])
                if s and rng.random() < 0.25:
                    s.append(-s[0])
            d = decision_matrix(sets)
            eng.reset(K)
            eng.apply_clean(d)
            twin.reset(K)
            twin.apply(d)
            conflict = eng.flags()[1]
            np.testing.assert_array_equal(conflict, twin.flags()[1])
            want = [ref_bcp(clauses, [lit_idx(x) for x in s])[0] for s in sets]
            assert conflict.tolist() == want, (shift, mode, nv)
            np.testing.assert_array_equal(eng.packed_rows(rows), baseline)
            ok = ~conflict
            np.testing.assert_array_equal(eng.num_propagated()[ok], twin.num_propagated()[ok])
            conflicted += int(conflict.sum())
        lite += eng.reset_stats()[1]
        assert twin.reset_stats()[1] == 0, "a plain launch leaves the rows dirty"
    assert lite > 0, "no reset ever skipped its broadcast: CLEAN mode is off"
    assert conflicted > 20, f"only {conflicted} conflicted rows were restored"


@opencl
@pytest.mark.parametrize("mode", ["scalar", "wave"])
def test_a_frontier_wider_than_a_wave_is_all_propagated(mode):
    """A WAVE row pops 32 trail literals at a time. A frontier wider than that — 100 literals
    forced by one, or 100 assumed at once — is propagated to the last literal, in several waves.

    x forces y1..y100 through binary clauses, and each pair (yi, yi+1) forces zi through a
    ternary one, so a y left unpopped shows as a z left unassigned."""
    n = 100
    x, y, z = 1, list(range(2, n + 2)), list(range(n + 2, 2 * n + 1))
    clauses = [[-x, yi] for yi in y] + [[-y[i], -y[i + 1], z[i]] for i in range(n - 1)]
    nv = 2 * n
    eng = _engine("opencl", _core.Formula.from_clauses(nv, clauses), mode=mode)
    sets = [[x], y, y[:40] + [-z[50]], [-y[7]]]
    eng.reset(len(sets))
    eng.apply(decision_matrix(sets))
    conflict = eng.flags()[1]
    asg = eng.rows_assignment(np.arange(len(sets), dtype=np.int32))
    for i, s in enumerate(sets):
        want_conflict, want_asg = ref_bcp(clauses, [lit_idx(v) for v in s])
        assert bool(conflict[i]) == want_conflict, (mode, i)
        assert assigned_set(asg[i]) == want_asg, (mode, i)
    assert eng.num_propagated().tolist() == [2 * n - 1, n - 1, 39, 1]


@opencl
@pytest.mark.parametrize("mode", ["scalar", "wave"])
def test_a_trail_longer_than_a_wave_is_restored_by_the_walk(mode):
    """A CLEAN restore whose walk stays under M/8 rewrites the words under the trail's literals
    one by one; under WAVE each lane then owns several of them (lane i takes i, i+32, ...). Two
    chains forced in step, x_i & y_i => x_i+1 and y_i+1, give a trail of 160 literals in which
    each link (-x_i | -y_i | x_i+1) is touched by its own two literals only, so a literal the
    walk skips leaves its links dirty. Filler clauses nobody touches keep the walk short."""
    n = 80
    x, y = list(range(1, n + 1)), list(range(n + 1, 2 * n + 1))
    links = [[-x[i], -y[i], t] for i in range(n - 1) for t in (x[i + 1], y[i + 1])]
    rng = random.Random(150)
    filler = [rng.sample(range(2 * n + 1, 2 * n + 3001), 3) for _ in range(12_000)]
    f = _core.Formula.from_clauses(2 * n + 3000, links + filler)
    assert f.info.num_long == len(links) + len(filler) > 8 * 4 * n, "walk under M/8"
    eng = _engine("opencl", f, mode=mode)
    twin = _engine("opencl", f, mode=mode)
    sets = [[x[0], y[0]], [x[n // 2], y[n // 2]], [x[0]], [y[0]]]
    rows = np.arange(len(sets), dtype=np.int32)
    eng.reset(len(sets))
    baseline = eng.packed_rows(rows)
    eng.reset(len(sets))
    eng.apply_clean(decision_matrix(sets))
    twin.reset(len(sets))
    twin.apply(decision_matrix(sets))
    assert not eng.flags()[1].any()
    np.testing.assert_array_equal(eng.packed_rows(rows), baseline)
    assert not np.array_equal(twin.packed_rows(rows), baseline), "the launch touched no word"
    want = [2 * (n - 1), 2 * (n - 1 - n // 2), 0, 0]
    assert eng.num_propagated().tolist() == twin.num_propagated().tolist() == want


@opencl
def test_wave_and_scalar_reach_the_same_fixpoint():
    """The two OpenCL propagate kernels agree on the conflict flags and, on every surviving row,
    on the assignment, the raw packed words and the implication count. The formulas are wide
    enough for a wave to hold several literals and for lanes to meet on one clause."""
    compared = 0
    for rng, nv, _clauses, f in _formulas(909, 8, nv_max=120):
        K = 48
        d = decision_matrix(random_assumptions(rng, nv, K, wmin=1, wmax=12))
        rows = np.arange(K, dtype=np.int32)
        out = {}
        for mode in ("scalar", "wave"):
            eng = _engine("opencl", f, mode=mode)
            eng.reset(K)
            eng.apply(d)
            out[mode] = (eng.flags()[1], eng.rows_assignment(rows), eng.packed_rows(rows),
                         eng.num_propagated())
        np.testing.assert_array_equal(out["scalar"][0], out["wave"][0])
        ok = ~out["scalar"][0]
        for a, b in zip(out["scalar"][1:], out["wave"][1:]):
            np.testing.assert_array_equal(a[ok], b[ok])
        # The work counters are exact on surviving rows too; they are summed over a launch, so
        # compare a launch of the surviving rows alone.
        totals = {}
        for mode in ("scalar", "wave"):
            eng = _engine("opencl", f, mode=mode)
            eng.reset(int(ok.sum()))
            eng.apply(d[ok])
            c = eng.counters()
            totals[mode] = (c.ops, c.props, c.decs)
        assert totals["scalar"] == totals["wave"], totals
        compared += int(ok.sum())
    assert compared > 100, f"only {compared} surviving rows compared"


@opencl
def test_selections_and_readouts_match_a_host_recount():
    """The support kernels (completion mask, flag count, bit gathers, soundness check, counter
    accumulation) equal the same quantities recomputed on the host from the row state."""
    exercised = {"complete": 0, "unsound": 0, "implied": 0}
    for rng, nv, clauses, f in _formulas(11, 12, nv_max=14):
        eng = _engine("opencl", f)
        K = 40
        # Wide assumption sets, so some rows end fully assigned and some conflict.
        sets = random_assumptions(rng, nv, K, wmin=nv // 2, wmax=nv)
        eng.reset(K)
        eng.apply(decision_matrix(sets))
        active, conflict = eng.flags()
        rows = np.arange(K, dtype=np.int32)
        asg = eng.rows_assignment(rows).astype(bool)          # (K, 2N): unpack_rows
        assigned = asg[:, 0::2] | asg[:, 1::2]                # (K, N): either polarity

        # complete_mask + sum_i8_to: active rows with every masked variable assigned
        some_vars = sorted(rng.sample(range(nv), max(1, nv // 2)))
        for vars_ in (None, some_vars):
            cols = list(range(nv)) if vars_ is None else vars_
            want = np.flatnonzero(active & assigned[:, cols].all(axis=1))
            n = eng.select_complete(eng.pair_mask(vars_), slot=0)
            assert n == len(want), (nv, vars_)
            assert eng.selection_rows(0, n).tolist() == want.tolist()
            # var_bits: the selected rows' TRUE bits of those variables
            bits = eng.var_bits(0, cols, n)
            np.testing.assert_array_equal(bits.astype(bool), asg[want][:, [2 * c for c in cols]])
            exercised["complete"] += n

        # sound_rows: a row is unsound iff some clause is emptied. A long clause is judged by
        # its packed count (on a conflicted row it can lag the bitmap, which is ahead by the
        # literals still queued), a unit or binary clause by the bitmap. The short clauses are
        # what fires here: the scalar kernel cannot empty a long clause (it turns unit first,
        # and pushing its last literal conflicts), so that half only ever answers "sound".
        short = [[lit_idx(-x) for x in c] for c in f.clauses() if len(c) < 3]
        counts = eng.packed_rows(rows) & ((1 << eng.shift) - 1)
        bad = [bool((counts[r] == 0).any()) or any(all(asg[r, i] for i in c) for c in short)
               for r in range(K)]
        eng.selection_set(1, rows)
        assert eng.check_sound(1) == sum(bad)
        assert not any(b and not conflict[r] for r, b in enumerate(bad)), \
            "a clause can be emptied only on a conflicted row"
        exercised["unsound"] += sum(bad)

        # sum_i32_acc: the device accumulators ADD each launch's per-row counters
        first = int(eng.num_propagated().sum())
        eng.reset(K)
        eng.apply(decision_matrix(sets))
        c = eng.counters()
        assert c.launches == 2
        assert c.props - c.decs == first + int(eng.num_propagated().sum())
        exercised["implied"] += first
    assert exercised["complete"] > 0, "no row was ever complete: the selection never ran"
    assert exercised["unsound"] > 0, "no row ever emptied a clause: check_sound never fired"
    assert exercised["implied"] > 0, "nothing was ever implied: the counters summed zeros"


@opencl
@pytest.mark.parametrize("mode", ["scalar", "wave"])
def test_a_batch_past_65535_rows_is_whole(mode):
    """Every row of a batch larger than 65,535 is reset, propagated and read back like the rest,
    by the scalar kernel (a work item per row) and by the WAVE kernel (a work group per row).

    An OpenCL built over CUDA (PoCL's driver) allows 65,535 work groups in every launch
    dimension but the first, so a kernel that gave the rows a dimension of their own failed
    there at the engine's normal batch sizes. Every launch is one-dimensional for that reason."""
    rng = random.Random(3)
    nv = 20
    clauses = random_cnf(rng, nv, 50, kmin=2, kmax=4)
    eng = _engine("opencl", _core.Formula.from_clauses(nv, clauses), mode=mode)
    lits = [v if pos else -v for v in range(1, nv + 1) for pos in (True, False)]
    want = [ref_bcp(clauses, [lit_idx(x)]) for x in lits]       # one literal per row, cycled
    K = 70_000
    which = np.arange(K) % len(lits)
    eng.reset(K)                                                # bcast_rows over 70,000 rows
    eng.apply(decision_matrix([[lits[i]] for i in which]))
    conflict = eng.flags()[1]
    np.testing.assert_array_equal(conflict, np.array([w[0] for w in want])[which])
    assert not conflict.all(), "every row conflicts: there is no assignment to compare"

    expect = np.zeros((len(lits), 2 * nv), dtype=bool)
    for i, (_, asg) in enumerate(want):
        expect[i, sorted(asg)] = True
    got = eng.rows_assignment(np.arange(K, dtype=np.int32)).astype(bool)   # unpack_rows, all rows
    ok = ~conflict
    np.testing.assert_array_equal(got[ok], expect[which][ok])
    # var_bits, over a selection that reaches past row 65,535 as well
    rows = np.flatnonzero(ok).astype(np.int32)
    eng.selection_set(0, rows)
    bits = eng.var_bits(0, np.arange(nv, dtype=np.int32), len(rows)).astype(bool)
    np.testing.assert_array_equal(bits, expect[which][ok][:, 0::2])


# ---------------------------------------------------------------- against the CUDA backend

@pytest.mark.parametrize("mode", ["scalar", "wave"])
@pytest.mark.parametrize("shift", [16, 8])
@pytest.mark.gpu
@opencl
def test_both_backends_agree_row_for_row(shift, mode):
    """The same formula and assumptions on both backends, with either propagate kernel:
    identical conflict flags, and on every surviving row identical assignments, raw packed words
    (count AND sum field) and implication counts."""
    compared = 0
    for rng, nv, clauses, f in _formulas(404, 12, nv_max=40):
        K = 64
        d = decision_matrix(random_assumptions(rng, nv, K, wmax=5))
        rows = np.arange(K, dtype=np.int32)
        out = {}
        for backend in ("cuda", "opencl"):
            eng = _engine(backend, f, force_shift=shift, mode=mode)
            assert eng.backend == backend
            eng.reset(K)
            eng.apply(d)
            assert eng.mode == mode
            out[backend] = (eng.flags()[1], eng.rows_assignment(rows), eng.packed_rows(rows),
                            eng.num_propagated())
        (cc, ca, cp, cn), (oc, oa, op, on) = out["cuda"], out["opencl"]
        np.testing.assert_array_equal(cc, oc)
        ok = ~cc
        np.testing.assert_array_equal(ca[ok], oa[ok])
        np.testing.assert_array_equal(cp[ok], op[ok])
        np.testing.assert_array_equal(cn[ok], on[ok])
        compared += int(ok.sum())
    assert compared > 200, f"only {compared} surviving rows compared"


@pytest.mark.parametrize("shift", [16, 8])
@pytest.mark.parametrize("name", ["s386-c5", "b03-c5"])
@pytest.mark.gpu
@opencl
def test_wide_probes_agree_across_backends(circuit_cnfs, name, shift):
    """Widths 2 and 3, where mining spends its time, on circuits past the WAVE clause threshold:
    both backends sweep with WAVE and CLEAN, and report the same conflicting tuples."""
    c = circuit_cnfs[name]
    f = _core.Formula.from_clauses(c["nv"], c["clauses"])
    free = _engine("cuda", f).free_literals()
    out = {}
    for backend in ("cuda", "opencl"):
        eng = _engine(backend, f, force_shift=shift)
        found = []
        for k, pool in ((2, None), (3, free[:150])):
            with eng.probe_combinations(pool, k) as sweep:
                assert sweep.stats.exhausted
                found.append({tuple(t) for t in sweep.conflicts().tolist()})
        assert eng.mode == "wave" and eng.reset_stats()[1] > 0, (backend, eng.mode)
        out[backend] = found
    assert out["cuda"] == out["opencl"]
    assert all(out["cuda"]), "a width found no conflicting tuple: the sweep proves little"


@pytest.mark.gpu
@opencl
def test_a_cuda_tensor_reaches_an_opencl_engine_through_the_host():
    """BCPEngine hands a CUDA decision tensor to the CUDA backend by its device address, which
    means nothing to OpenCL: on an OpenCL engine the tensor is copied through the host, and the
    rows come out as the CUDA engine's."""
    import torch

    from gpap.engine.bcp_engine import BCPEngine
    from gpap.engine.formula import CnfFormula

    rng = random.Random(5)
    nv = 40
    clauses = random_cnf(rng, nv, 120, kmin=2, kmax=4)
    d = torch.tensor(decision_matrix(random_assumptions(rng, nv, 32, wmin=1, wmax=5)),
                     device=DEV)
    out = {}
    for backend in ("cuda", "opencl"):
        with use_backend(backend):
            formula = CnfFormula(DEV)
            formula.init_formula(nv, clauses)
            eng = BCPEngine(formula)
        eng.reset(d.size(0))
        eng.apply_and_propagate(d)
        out[backend] = (eng.conflict.cpu(), eng.num_propagated.cpu())
    conflict = out["cuda"][0]
    assert torch.equal(conflict, out["opencl"][0])
    ok = ~conflict                              # a conflicted row's count is unspecified
    assert torch.equal(out["cuda"][1][ok], out["opencl"][1][ok])
    assert conflict.any() and ok.any()


@pytest.mark.gpu
@opencl
def test_the_two_scalar_kernels_do_the_same_work():
    """The OpenCL scalar kernel is a literal translation of the Triton scalar kernel, so the two
    agree on the work counters of EVERY row, conflicted ones included: same occurrence entries
    walked, same literals assigned, same decisions.

    That is what makes implications per second comparable between the backends. It is stronger
    than the kernel contract, which leaves a conflicted row unspecified: a deliberate
    reordering of either kernel may relax this to the surviving rows."""
    conflicted = 0
    for rng, nv, _clauses, f in _formulas(77, 10, nv_max=40):
        K = 96
        d = decision_matrix(random_assumptions(rng, nv, K, wmin=2, wmax=8))
        out = {}
        for backend in ("cuda", "opencl"):
            eng = _engine(backend, f, mode="scalar")
            eng.reset(K)
            eng.apply(d)
            c = eng.counters()
            out[backend] = (eng.flags()[1], eng.num_propagated(), (c.ops, c.props, c.decs))
        np.testing.assert_array_equal(out["cuda"][0], out["opencl"][0])
        np.testing.assert_array_equal(out["cuda"][1], out["opencl"][1])
        assert out["cuda"][2] == out["opencl"][2], nv
        conflicted += int(out["cuda"][0].sum())
    assert conflicted > 50, f"only {conflicted} conflicted rows: the claim was barely tested"


def _mined(backend, nv, clauses, widths):
    with use_backend(backend):
        res = mine_entailed(nv, clauses, DEV, widths=widths, seconds=None)
    assert not res.skipped, f"{backend}: mining stopped early: {res.skipped}"
    return sorted(map(sorted, res.clauses))


@pytest.mark.gpu
@opencl
def test_mining_is_backend_invariant():
    """What mining returns does not depend on the backend: the mined clause SETS are identical,
    with no time budget, so neither run is cut short."""
    clauses = [[-4], [-1, -2, 4], [-1, -3, 4], [5, 6], [-5, 6], [5, -6], [2, 3], [1, 7]]
    assert _mined("cuda", 7, clauses, (1, 2)) == _mined("opencl", 7, clauses, (1, 2))
    mined = 0
    for _rng, nv, cls, _f in _formulas(31, 6, nv_max=16):
        a = _mined("cuda", nv, cls, (1, 2, 3))
        assert a == _mined("opencl", nv, cls, (1, 2, 3)), nv
        mined += len(a)
    assert mined > 0, "nothing was mined: the comparison was empty"


@pytest.mark.parametrize("name", ["s386-c5", "b03-c5", "b06-c10"])
@pytest.mark.gpu
@opencl
def test_mining_circuits_is_backend_invariant(circuit_cnfs, name):
    """The same on unrolled circuits, where the backbone is real: identical unit sets."""
    c = circuit_cnfs[name]
    assert _mined("cuda", c["nv"], c["clauses"], (1,)) == \
        _mined("opencl", c["nv"], c["clauses"], (1,))


# ---------------------------------------------------------------- what the backend refuses

@opencl
def test_unported_kernels_are_refused_by_name():
    """The OpenCL backend has no trajectory kernel and no device-pointer entry points. Asking
    for one is an error that names the backend, never a silent wrong answer."""
    f = _core.Formula.from_clauses(4, [[1, 2, 3], [-1, 2], [3, 4]])
    eng = _engine("opencl", f)
    eng.reset(4)
    with pytest.raises(_core.GpapError, match="not part of the opencl backend") as e:
        eng.trajectory(np.arange(4, dtype=np.int32), 4, 2, seed=1)
    assert e.value.status == 5                                  # GPAP_ERR_KERNEL
    with pytest.raises(_core.GpapError, match="takes host arrays"):
        eng.apply_dev(0x1000, 1)


@opencl
def test_auto_dispatches_by_clause_count_and_clean_resets_go_lite():
    """The engine's own choices on the OpenCL backend. WAVE runs from the clause threshold up;
    below it the rule asks for the device's resident warps, which OpenCL cannot report, so a
    small formula runs scalar at any batch. A CLEAN apply gives a plain apply's flags and lets
    the next reset skip its broadcast."""
    rng = random.Random(5)
    nv = 200
    clauses = random_cnf(rng, nv, 3000, kmin=3, kmax=4)         # past the WAVE threshold
    with use_backend("opencl"):
        assert _core.dispatch_wave(len(clauses), 1 << 16)
        assert not _core.dispatch_wave(6, 64)
    small = _engine("opencl", _core.Formula.from_clauses(4, [[1, 2, 3], [-1, 2], [3, 4]]))
    small.reset(8)
    assert small.mode == "scalar"

    eng = _engine("opencl", _core.Formula.from_clauses(nv, clauses))
    d = decision_matrix(random_assumptions(rng, nv, 32, wmin=1, wmax=3))
    flags = []
    for apply in (eng.apply_clean, eng.apply):
        eng.reset(32)
        assert eng.mode == "wave"
        apply(d)
        flags.append(eng.flags()[1])
    np.testing.assert_array_equal(flags[0], flags[1])
    assert eng.reset_stats()[1] == 1, "the reset after the CLEAN apply, and only it, goes lite"


@pytest.mark.core
def test_an_unknown_backend_is_an_error():
    """`GPAP_BACKEND` is validated, not defaulted: a typo must not run the other backend."""
    with use_backend("opnecl"):
        with pytest.raises(_core.GpapError, match="cuda .* or opencl"):
            _core.backend()
        with pytest.raises(_core.GpapError, match="cuda .* or opencl"):
            _core.Engine(_core.Formula.from_clauses(2, [[1, 2]]))
    with use_backend("cuda"):
        assert _core.backend() == "cuda"


# ---------------------------------------------------------------- the kernel list

def test_the_kernel_list_matches_bcp_cl():
    """Every kernel key the OpenCL plug lists is defined in bcp.cl under the name the plug
    derives from it, and bcp.cl defines no kernel the plug cannot reach. No device needed."""
    plug = open(os.path.join(REPO_ROOT, "cpp", "src", "gpu_opencl.cpp")).read()
    table = plug[plug.index("kSpecs[] = {"):plug.index("constexpr size_t kGroup")]
    keys = re.findall(r'\{"([a-z0-9_.]+)",\s*(?:ROWS|GRID|GROUP|LANES)', table)
    assert len(keys) == len(set(keys)) >= 17, keys

    src = open(os.path.join(REPO_ROOT, "cpp", "kernels", "bcp.cl")).read()
    defined = set(re.findall(r"^__kernel void (gpap_\w+)\(", src, re.M))
    defined |= set(re.findall(r"^GPAP_\w+_KERNEL\((gpap_\w+),", src, re.M))
    assert {"gpap_" + k.replace(".", "_") for k in keys} == defined


def _launch_sizes(plug: str, src: str) -> dict:
    """The group sizes the plug launches with and the ones bcp.cl is written for, and the
    WAVE kernel macros' headers."""
    num = lambda pattern, text: int(re.search(pattern, text).group(1))  # noqa: E731
    return {
        "plug_wave": num(r"constexpr int kWaveW = (\d+);", plug),
        "cl_wave": num(r"#define GPAP_WAVE_W (\d+)", src),
        "plug_reduce": num(r"constexpr size_t kReduceGroup = (\d+);", plug),
        "cl_reduce": num(r"#define GPAP_GROUP_MAX (\d+)", src),
        "wave_macros": re.findall(r"#define (GPAP_WAVE(?:_CLEAN)?_KERNEL)\(NAME, SHIFT\)\s*\\\n"
                                  r"\s*__kernel (\w*)", src),
    }


def test_the_plug_launches_the_group_sizes_bcp_cl_is_written_for():
    """A WAVE kernel's rows are its work groups of GPAP_WAVE_W lanes, and a reduction sums at
    most GPAP_GROUP_MAX items: a launch of any other shape would mix rows or lose a sum without
    an error. The plug's two constants equal bcp.cl's, and both WAVE kernel macros declare the
    group size (reqd_work_group_size), so a host that launches them otherwise is refused."""
    sizes = _launch_sizes(open(os.path.join(REPO_ROOT, "cpp", "src", "gpu_opencl.cpp")).read(),
                          open(os.path.join(REPO_ROOT, "cpp", "kernels", "bcp.cl")).read())
    assert sizes["plug_wave"] == sizes["cl_wave"]
    assert sizes["plug_reduce"] == sizes["cl_reduce"]
    assert sizes["wave_macros"] == [("GPAP_WAVE_KERNEL", "GPAP_WAVE_GROUP"),
                                    ("GPAP_WAVE_CLEAN_KERNEL", "GPAP_WAVE_GROUP")], sizes
