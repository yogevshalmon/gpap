"""End-to-end validation of the C++ tier-1 API (`gpap::Sampler`, `gpap::AssumptionChecker`,
`gpap::Prober`).

The C++ side is deliberately thin — a sampling loop over the engine — so the comprehensive
checking lives here: run the built example binaries and verify their output against the
pure-Python reference BCP. That covers the whole tier-1 path (auto-K, kernel dispatch,
support-first decision order, chunk loop, selection gather, DIMACS conversion) from the outside,
exactly as a new C++ consumer would exercise it.

Skipped unless `cmake --build build` has produced the examples (scripts/build_core.sh).
"""
import re
import subprocess
from pathlib import Path

import pytest

from tests.reference_bcp import lit_idx, ref_bcp

REPO = Path(__file__).resolve().parent.parent


def _bin(name: str) -> Path:
    """An example binary in whichever build tree it was built into — the same set `_core.py`
    accepts for the library. Pinning `build/` alone meant a `cpp/build/` or `$GPAP_BUILD_DIR`
    checkout skipped every test in this file and said nothing."""
    import os
    trees = [Path(os.environ["GPAP_BUILD_DIR"])] if os.environ.get("GPAP_BUILD_DIR") else []
    trees += [REPO / "build", REPO / "cpp" / "build"]
    return next((t / name for t in trees if (t / name).exists()), trees[0] / name)


SAMPLE_BIN = _bin("gpap_sample")
ASSUME_BIN = _bin("gpap_assumptions")
PROBE_BIN = _bin("gpap_probe")
C_API_BIN = _bin("gpap_c_api")

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


def _run_sampler(cnf: Path, workdir: Path, n=2000, seed=7, k=0):
    """-> (solution rows as DIMACS literals, the binary's stdout).

    `k` pins the batch. The default sizes it to LIVE free VRAM, which is NOT part of the seed:
    comparing counts against another process (test_face_parity) needs both sides on one batch."""
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / "sols.txt"
    cmd = [str(SAMPLE_BIN), str(cnf), str(n), str(seed), str(out)]
    if k:
        cmd += ["--k", str(k)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
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
    """Same seed, same solutions; different seed, not.

    "A function of the seed" holds at a fixed BATCH, which is what running both arms in one
    process on one card gives us here — K is auto-sized from live free VRAM, so it is not itself
    a function of the seed (see gpap.h, and test_face_parity, which pins it across processes)."""
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


# ------------------------------------------------------------------ gpap::Prober

def _probe(cnf: Path, width: int):
    _need(PROBE_BIN)
    r = subprocess.run([str(PROBE_BIN), str(cnf), str(width)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"gpap_probe failed:\n{r.stdout}\n{r.stderr}"
    return r.stdout


def test_probe_finds_the_same_failed_literals_as_python(tmp_path):
    """The C++ face and the Python face are two skins on ONE core sweep, so they must agree on
    the clause SET — not merely on how many were found."""
    pytest.importorskip("torch")
    import torch
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from gpap.engine import core_available
    if not core_available():
        pytest.skip("needs a built libgpap")
    from gpap.mining import mine_entailed

    # x1 -> x2 and x1 -> ~x2, so assuming x1 conflicts: (~x1) is entailed. Same for x4 via x5.
    clauses = [[-1, 2], [-1, -2], [3, 4], [-4, 5], [-4, -5], [6, 7]]
    cnf = tmp_path / "f.cnf"
    cnf.write_text("p cnf 7 %d\n" % len(clauses)
                   + "".join(" ".join(map(str, c)) + " 0\n" for c in clauses))

    out = _probe(cnf, 1)
    cpp_units = {int(line.split("(")[1].split(")")[0])
                 for line in out.splitlines() if "entailed: (" in line}
    res = mine_entailed(7, clauses, torch.device("cuda"), widths=(1,), seconds=30.0)
    py_units = {c[0] for c in res.clauses if len(c) == 1}
    # (-1) and (-4) are the planted ones; (3) falls out too, because assuming -3 forces 4 through
    # (3|4), and 4 forces both 5 and -5. All three are reachable in a SINGLE width-1 sweep, so
    # this compares the two faces and not Python's extra fixpoint iteration.
    assert cpp_units == py_units == {-1, 3, -4}, (cpp_units, py_units)
    assert "exhaustive" in out, "a complete width-1 sweep must report itself as a proof"


def test_probe_bounds_an_impossible_width_without_enumerating_it(tmp_path):
    """REGRESSION: the ceiling has to be checked against an EXACT count computed without
    generating anything. Enumerating first and measuring afterwards OOM-killed the process — at
    width 3 over a few thousand free literals the enumeration is ~1e11 tuples.

    It gates the CONTRACT, not a wording: at the default budget `max_tuples` is a WORK BUDGET,
    so the sweep probes that many, KEEPS every clause it found, and reports itself as not
    exhaustive. This assertion used to be `"refused" in out` — which passed against an example
    that printed the word "refused" directly above 2.5 million clauses it had just delivered, so
    the test was pinning the misstatement rather than catching it."""
    nv = 400
    clauses = [[i, -(i + 1)] for i in range(1, nv)]
    cnf = tmp_path / "wide.cnf"
    cnf.write_text("p cnf %d %d\n" % (nv, len(clauses))
                   + "".join(" ".join(map(str, c)) + " 0\n" for c in clauses))
    out = _probe(cnf, 3)
    # the exact count, computed up front — the thing that must not be reached by enumerating
    assert "width 3: 84694400 candidate tuples" in out, out
    probed, total = re.search(r"probed (\d+) of (\d+)", out).groups()
    # The example sets BOTH a 5M ceiling and a 10s deadline, and probe.cpp overwrites the stop
    # reason with DEADLINE when the clock wins — which it can on a slower card or under ASan. So
    # assert the property (the sweep stopped SHORT of `total`, and says so) rather than which
    # budget got there first; `total` never being enumerated is what this test is really for.
    assert 0 < int(probed) <= 5_000_000 < int(total), out
    assert "exhaustive" not in out, "a budgeted sweep is not a proof"
    assert ("stopped at max_tuples" in out) or ("deadline" in out), out
    assert "entailed: (" in out, "a budgeted sweep still KEEPS the clauses it proved"


# ------------------------------------------------------------------ the raw C ABI

def _c_api(*args, timeout=300):
    _need(C_API_BIN)
    r = subprocess.run([str(C_API_BIN), *map(str, args)],
                       capture_output=True, text=True, timeout=timeout)
    return r


def test_c_api_example_checks_both_entry_points_through_the_raw_abi():
    """`cpp/examples/c_api.c` is the reference for `gpap/gpap.h` and the only consumer in the tree
    that touches the ABI with no C++ wrapper: 0-based variables, caller-owned buffers,
    `gpap_status` returns. CI checked that it LINKS and nothing ever ran it, so every one of those
    conventions was unexercised — a header change that broke them would have shipped green.

    The example self-checks (it exits non-zero unless x1=TRUE reaches a fixpoint and x1=FALSE
    conflicts), so the assertion is its exit status plus the verdict it printed."""
    pytest.importorskip("torch")
    import torch
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    r = _c_api()
    assert r.returncode == 0, f"gpap_c_api failed:\n{r.stdout}\n{r.stderr}"
    assert "PASS" in r.stdout, r.stdout
    assert "row 0 (x1=TRUE ): ok" in r.stdout, r.stdout
    assert "row 1 (x1=FALSE): CONFLICT" in r.stdout, r.stdout


def test_c_api_sampling_models_satisfy_the_formula(tmp_path):
    """The ABI's readout path (`gpap_select_complete` -> `gpap_var_bits`) against the clauses.

    The example prints its own soundness verdict off `gpap_check_sound`, which is the DEVICE's
    answer; this re-derives it on the host from the printed bits, so a readout that gathered the
    wrong columns cannot agree with itself. Under 32 variables, so no model is truncated."""
    pytest.importorskip("torch")
    import torch
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    # 8 variables, and every clause length the packed word distinguishes (unit, binary, long).
    clauses = [[1, 2, 3], [-1, -2], [4, 5], [-4, -5], [6], [-7, 8], [7, -8], [2, -3, 5]]
    nv = 8
    cnf = tmp_path / "abi.cnf"
    cnf.write_text("p cnf %d %d\n" % (nv, len(clauses))
                   + "".join(" ".join(map(str, c)) + " 0\n" for c in clauses))
    r = _c_api(cnf)
    assert r.returncode == 0, f"gpap_c_api failed:\n{r.stdout}\n{r.stderr}"
    assert "soundness: ok" in r.stdout, r.stdout
    models = re.findall(r"^  model \d+: ([01]+)$", r.stdout, re.M)
    assert models, f"a sound run printed no model to check:\n{r.stdout}"
    for bits in models:
        assert len(bits) == nv, f"{bits!r}: {len(bits)} columns, formula has {nv} variables"
        row = [(v + 1) if bits[v] == "1" else -(v + 1) for v in range(nv)]
        assignment = set(row)
        for clause in clauses:
            assert assignment & set(clause), f"{bits} falsifies {clause}"
