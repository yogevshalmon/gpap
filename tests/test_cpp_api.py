"""End-to-end validation of the C++ tier-1 API (`gpap::Sampler`, `gpap::AssumptionChecker`).

The C++ side is deliberately thin — a sampling loop over the engine — so the comprehensive
checking lives here: run the built example binaries and verify their output against the
pure-Python reference BCP. That covers the whole tier-1 path (auto-K, kernel dispatch,
support-first decision order, chunk loop, selection gather, DIMACS conversion) from the outside,
exactly as a new C++ consumer would exercise it.

Skipped unless `cmake --build build` has produced the examples (scripts/build_core.sh).
"""
import subprocess
from pathlib import Path

import pytest

from tests.reference_bcp import lit_idx, ref_bcp

REPO = Path(__file__).resolve().parent.parent
SAMPLE_BIN = REPO / "build" / "gpap_sample"
ASSUME_BIN = REPO / "build" / "gpap_assumptions"

pytestmark = pytest.mark.gpu


def _need(binary: Path):
    if not binary.exists():
        pytest.skip(f"{binary.name} not built (scripts/build_core.sh)")


def _parse_cnf(path: Path):
    """-> (num_vars, clauses as DIMACS-literal lists, independent support (1-based) or None)."""
    clauses, support, nv = [], [], None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("c ind"):
            support += [int(t) for t in line[5:].split() if t not in ("", "0")]
        elif line.startswith("p cnf"):
            nv = int(line.split()[2])
        elif line and not line.startswith(("c", "p", "%")):
            lits = [int(t) for t in line.split() if t != "0"]
            if lits:
                clauses.append(lits)
    return nv, clauses, (support or None)


def _run_sampler(cnf: Path, workdir: Path, n=2000, seed=7):
    """-> (solution rows as DIMACS literals, the binary's stdout)."""
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / "sols.txt"
    r = subprocess.run([str(SAMPLE_BIN), str(cnf), str(n), str(seed), str(out)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"gpap_sample failed:\n{r.stdout}\n{r.stderr}"
    rows = [[int(t) for t in ln.split()] for ln in out.read_text().splitlines() if ln.strip()]
    return rows, r.stdout


def _extends_to_model(clauses, row) -> bool:
    """Is the formula satisfiable with `row`'s literals asserted? This is the DEFINITION of a
    valid projected sample, decided rather than approximated."""
    from pysat.solvers import Solver
    with Solver(name="minisat22", bootstrap_with=clauses) as s:
        return s.solve(assumptions=list(row))


def _assert_rows_are_models(rows, clauses, num_vars, support):
    """Every emitted row must EXTEND to a full model.

    Propagating the row must not conflict, and then every clause must be satisfied. That second
    check is exact for a FULL assignment, and stronger than necessary for a projection: BCP is
    incomplete, so an independent support fixes a model semantically without necessarily
    *deriving* it, and a valid projection could in principle leave clauses merely non-falsified.
    Measured on every instance this suite uses (circuit Tseitin, a deep unroll, and a projected
    `sk_` CNF) the fixpoint does satisfy all of them — GPAP only emits rows that were already
    fully assigned — so the strong check is the one that runs and it catches far more than
    "did not conflict". Where it does not hold we decide the real question with a SAT call
    instead of failing a legitimate sample."""
    assert rows, "sampler emitted nothing"
    for row in rows:
        assert row, "empty solution row"
        assert len({abs(v) for v in row}) == len(row), "a variable appears twice in one row"
        for lit in row:
            assert 1 <= abs(lit) <= num_vars, f"literal {lit} outside 1..{num_vars}"
        if support is not None:
            assert sorted(abs(v) for v in row) == sorted(support), \
                "projection columns are not the declared independent support"
        conflict, assigned = ref_bcp(clauses, [lit_idx(x) for x in row])
        assert not conflict, f"emitted row conflicts under BCP: {row[:10]}..."
        unsatisfied = [cl for cl in clauses if not any(lit_idx(x) in assigned for x in cl)]
        if unsatisfied:
            assert support is not None, \
                f"a FULL assignment left clause {unsatisfied[0]} unsatisfied"
            assert _extends_to_model(clauses, row), \
                f"projection does not extend to any model: {row[:10]}..."


def test_extends_to_model_helper():
    """The projected-sample fallback, exercised directly — on our instances BCP always closes
    the formula, so without this the branch would never run anywhere."""
    # x1=TRUE forces x2 both ways => no extension; x1=FALSE extends fine.
    both_ways = [[-1, 2], [-1, -2]]
    assert not _extends_to_model(both_ways, [1])
    assert _extends_to_model(both_ways, [-1])
    # A clause BCP cannot resolve from the support alone, but which a model satisfies.
    assert _extends_to_model([[2, 3]], [1])


def test_cpp_sampler_full_assignments(tmp_path):
    """No `c ind` in the file -> the sampler reports every variable, so each row is a model."""
    _need(SAMPLE_BIN)
    cnf = tmp_path / "small.cnf"
    clauses = [[1, 2], [-1, 3], [-2, -3], [2, 3, 4], [-4, 1]]
    cnf.write_text("p cnf 4 %d\n" % len(clauses)
                   + "".join(" ".join(map(str, c)) + " 0\n" for c in clauses))

    rows, stdout = _run_sampler(cnf, tmp_path / "run", n=500)
    nv, cls, sup = _parse_cnf(cnf)
    assert sup is None
    _assert_rows_are_models(rows, cls, nv, None)
    assert all(len(r) == nv for r in rows), "a full assignment must cover every variable"
    # Everything reported must be an actual model (enumerated by brute force here).
    models = set()
    for bits in range(1 << nv):
        asg = [(v + 1) if (bits >> v) & 1 else -(v + 1) for v in range(nv)]
        if all(any(l in asg for l in c) for c in cls):
            models.add(tuple(asg))
    assert models, "test formula is unsatisfiable"
    assert {tuple(r) for r in rows} <= models
    assert "solutions in" in stdout


def test_cpp_sampler_projected_support(circuit_cnfs, tmp_path):
    """With `c ind`, columns are the independent support and each row must extend to a model."""
    _need(SAMPLE_BIN)
    cnf = Path(circuit_cnfs["b03-c5"]["path"])
    rows, stdout = _run_sampler(cnf, tmp_path / "run", n=2000)
    nv, cls, sup = _parse_cnf(cnf)
    assert sup, "expected a `c ind` header on a circuit CNF"
    _assert_rows_are_models(rows, cls, nv, sup)
    assert f"{len(sup)} variables decided first" in stdout, \
        "the sampler must decide the declared support first"


def test_cpp_sampler_is_deterministic(circuit_cnfs, tmp_path):
    """A run is a function of the seed alone — same seed, same solutions; different seed, not."""
    _need(SAMPLE_BIN)
    cnf = Path(circuit_cnfs["b03-c5"]["path"])
    a, _ = _run_sampler(cnf, tmp_path / "a", n=800, seed=11)
    b, _ = _run_sampler(cnf, tmp_path / "b", n=800, seed=11)
    c, _ = _run_sampler(cnf, tmp_path / "c", n=800, seed=12)
    assert a == b, "same seed produced different samples"
    assert a != c, "different seeds produced identical samples"


def test_cpp_assumption_checker():
    """The example's tutorial sets, checked against the reference BCP.

    This gates the EXAMPLE as well as the library: it is what a new user reads first, so its
    printed answers have to be right, not merely plausible. Each non-conflicted row is printed
    as `xN=T` / `xN=F` / `xN=free`, which must match the reference fixpoint exactly — `free`
    included, since "BCP did not determine this" is the claim the example makes about it."""
    _need(ASSUME_BIN)
    r = subprocess.run([str(ASSUME_BIN)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    clauses = [[1, 2], [-1, 3], [-2, -3]]
    lines = [ln for ln in r.stdout.splitlines() if "->" in ln]
    assumptions = ([1], [-1, -2], [2, 3], [])
    assert len(lines) == len(assumptions), r.stdout
    for line, assumption in zip(lines, assumptions):
        conflict, assigned = ref_bcp(clauses, [lit_idx(x) for x in assumption])
        assert ("CONFLICT" in line) == conflict, f"{assumption}: {line}"
        if conflict:
            continue
        got = set()
        for token in line.split(":")[-1].split():          # e.g. "x1=T", "x3=free"
            name, _, value = token.partition("=")
            var = int(name[1:])
            if value in ("T", "F"):
                got.add(var if value == "T" else -var)
            else:
                assert value == "free", f"{assumption}: unexpected value in {token!r}"
        want = {(lit // 2 + 1) if lit % 2 == 0 else -(lit // 2 + 1) for lit in assigned}
        assert got == want, f"{assumption}: {got} != {want}"
