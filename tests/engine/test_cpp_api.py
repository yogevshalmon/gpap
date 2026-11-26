"""The C++ tier-1 API (`gpap::Sampler`, `gpap::AssumptionChecker`, `gpap::Prober`) and the raw
C ABI, tested by running the built example binaries and checking their output against the
reference BCP. Skipped unless the examples are built (scripts/build_core.sh).
"""
import re
import subprocess
from pathlib import Path

import pytest

from tests.reference.bcp import lit_idx, ref_bcp
from tests.support import (DEV, SAMPLE_BIN, example_binary, need_binary, parse_dimacs,
                           run_sample_example)

ASSUME_BIN = example_binary("gpap_assumptions")
PROBE_BIN = example_binary("gpap_probe")
C_API_BIN = example_binary("gpap_c_api")

pytestmark = pytest.mark.gpu


def _extends_to_model(clauses, row) -> bool:
    """Is the formula satisfiable with `row`'s literals asserted? The exact definition of a
    valid projected sample."""
    from pysat.solvers import Solver
    with Solver(name="minisat22", bootstrap_with=clauses) as s:
        return s.solve(assumptions=list(row))


def _assert_rows_are_models(rows, clauses, num_vars, support):
    """Every emitted row extends to a full model.

    The row must propagate without conflict and the fixpoint must satisfy every clause. For a
    projection BCP may leave clauses only non-falsified, so there a SAT call decides whether
    the row extends to a model."""
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
    """The SAT fallback, tested directly because BCP closes the formula on the instances
    above, so the fallback never runs there."""
    # x1=TRUE forces x2 both ways => no extension; x1=FALSE extends fine.
    both_ways = [[-1, 2], [-1, -2]]
    assert not _extends_to_model(both_ways, [1])
    assert _extends_to_model(both_ways, [-1])
    # A clause BCP cannot resolve from the support alone, but which a model satisfies.
    assert _extends_to_model([[2, 3]], [1])


def test_cpp_sampler_full_assignments(tmp_path):
    """No `c ind` in the file -> the sampler reports every variable, so each row is a model."""
    need_binary(SAMPLE_BIN)
    cnf = tmp_path / "small.cnf"
    clauses = [[1, 2], [-1, 3], [-2, -3], [2, 3, 4], [-4, 1]]
    cnf.write_text("p cnf 4 %d\n" % len(clauses)
                   + "".join(" ".join(map(str, c)) + " 0\n" for c in clauses))

    rows, stdout = run_sample_example(cnf, tmp_path / "run", n=500)
    nv, cls, sup = parse_dimacs(cnf)
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
    need_binary(SAMPLE_BIN)
    cnf = Path(circuit_cnfs["b03-c5"]["path"])
    rows, stdout = run_sample_example(cnf, tmp_path / "run", n=2000)
    nv, cls, sup = parse_dimacs(cnf)
    assert sup, "expected a `c ind` header on a circuit CNF"
    _assert_rows_are_models(rows, cls, nv, sup)
    assert f"{len(sup)} variables decided first" in stdout, \
        "the sampler must decide the declared support first"


def test_cpp_sampler_is_deterministic(circuit_cnfs, tmp_path):
    """Same seed, same solutions; different seed, different solutions.

    This needs a fixed batch. K is sized from free VRAM, which is not part of the seed; the runs
    here share a card and a moment; tests/sampling/test_face_parity.py pins K across processes."""
    need_binary(SAMPLE_BIN)
    cnf = Path(circuit_cnfs["b03-c5"]["path"])
    a, _ = run_sample_example(cnf, tmp_path / "a", n=800, seed=11)
    b, _ = run_sample_example(cnf, tmp_path / "b", n=800, seed=11)
    c, _ = run_sample_example(cnf, tmp_path / "c", n=800, seed=12)
    assert a == b, "same seed produced different samples"
    assert a != c, "different seeds produced identical samples"


def test_cpp_assumption_checker():
    """The example's printed answers match the reference BCP.

    Each non-conflicted row prints `xN=T` / `xN=F` / `xN=free`, and all three must match the
    reference fixpoint, `free` included."""
    need_binary(ASSUME_BIN)
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

def _probe(cnf: Path, width: int, *extra: str):
    need_binary(PROBE_BIN)
    r = subprocess.run([str(PROBE_BIN), str(cnf), str(width), *extra],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"gpap_probe failed:\n{r.stdout}\n{r.stderr}"
    return r.stdout


def test_probe_finds_the_same_failed_literals_as_python(tmp_path):
    """The C++ and Python faces run the same core sweep, so they find the same clauses, not
    just the same number."""
    from gpap.mining import mine_entailed

    # x1 -> x2 and x1 -> ~x2, so assuming x1 conflicts: (~x1) is entailed. Same for x4 via x5.
    clauses = [[-1, 2], [-1, -2], [3, 4], [-4, 5], [-4, -5], [6, 7]]
    cnf = tmp_path / "f.cnf"
    cnf.write_text("p cnf 7 %d\n" % len(clauses)
                   + "".join(" ".join(map(str, c)) + " 0\n" for c in clauses))

    out = _probe(cnf, 1)
    cpp_units = {int(line.split("(")[1].split(")")[0])
                 for line in out.splitlines() if "entailed: (" in line}
    res = mine_entailed(7, clauses, DEV, widths=(1,), seconds=30.0)
    py_units = {c[0] for c in res.clauses if len(c) == 1}
    # (-1) and (-4) are planted; (3) follows because -3 forces 4 through (3|4), and 4 forces 5
    # and -5. All three come from one width-1 sweep, so Python's extra fixpoint rounds add
    # nothing here.
    assert cpp_units == py_units == {-1, 3, -4}, (cpp_units, py_units)
    assert "exhaustive" in out, "a complete width-1 sweep must report itself as a proof"


def test_the_cpp_prober_adds_what_it_found(tmp_path):
    """Prober::add_clauses (Engine::add_clauses over gpap_engine_add_clauses) grows the engine's
    formula in place: with `add`, the example adds the units its sweep proved, and the engine
    then counts them as clauses and has fewer free literals."""
    clauses = [[-1, 2], [-1, -2], [3, 4], [-4, 5], [-4, -5], [6, 7]]
    cnf = tmp_path / "f.cnf"
    cnf.write_text("p cnf 7 %d\n" % len(clauses)
                   + "".join(" ".join(map(str, c)) + " 0\n" for c in clauses))
    out = _probe(cnf, 1, "add")
    m = re.search(r"added (\d+) clause\(s\): the engine now has (\d+) clauses, "
                  r"(\d+) free literals", out)
    assert m, out
    added, total, free_after = map(int, m.groups())
    free_before = int(re.search(r"(\d+) free literals", out).group(1))
    assert (added, total) == (3, len(clauses) + 3), out      # the units -1, 3 and -4
    assert free_before > free_after == 2 * (7 - 3), out      # 1, 3, 4 assigned, nothing unsat


def test_probe_bounds_an_impossible_width_without_enumerating_it(tmp_path):
    """The budget is checked against an exact count computed before anything is generated;
    enumerating a huge width first would exhaust memory.

    `max_tuples` is a work budget: the sweep probes that many, keeps every clause it found, and
    reports itself as not exhaustive."""
    nv = 400
    clauses = [[i, -(i + 1)] for i in range(1, nv)]
    cnf = tmp_path / "wide.cnf"
    cnf.write_text("p cnf %d %d\n" % (nv, len(clauses))
                   + "".join(" ".join(map(str, c)) + " 0\n" for c in clauses))
    out = _probe(cnf, 3)
    # the exact count, computed without enumerating
    assert "width 3: 84694400 candidate tuples" in out, out
    probed, total = re.search(r"probed (\d+) of (\d+)", out).groups()
    # The example sets a 5M ceiling and a 10 s deadline, and either can stop the sweep first
    # (slow card, ASan), so assert that it stopped short of `total` and says so.
    assert 0 < int(probed) <= 5_000_000 < int(total), out
    assert "exhaustive" not in out, "a budgeted sweep is not a proof"
    assert ("stopped at max_tuples" in out) or ("deadline" in out), out
    assert "entailed: (" in out, "a budgeted sweep still KEEPS the clauses it proved"


# ------------------------------------------------------------------ the raw C ABI

def _c_api(*args, timeout=300):
    need_binary(C_API_BIN)
    r = subprocess.run([str(C_API_BIN), *map(str, args)],
                       capture_output=True, text=True, timeout=timeout)
    return r


def test_c_api_example_checks_both_entry_points_through_the_raw_abi():
    """`cpp/examples/c_api.c` is the only in-tree consumer of `gpap/gpap.h` without the C++
    wrapper (0-based variables, caller-owned buffers, `gpap_status` returns). It checks itself
    (x1=TRUE reaches a fixpoint, x1=FALSE conflicts), so the test reads its exit status and
    verdict."""
    r = _c_api()
    assert r.returncode == 0, f"gpap_c_api failed:\n{r.stdout}\n{r.stderr}"
    assert "PASS" in r.stdout, r.stdout
    assert "row 0 (x1=TRUE ): ok" in r.stdout, r.stdout
    assert "row 1 (x1=FALSE): CONFLICT" in r.stdout, r.stdout


def test_c_api_sampling_models_satisfy_the_formula(tmp_path):
    """The ABI readout path (`gpap_select_complete` -> `gpap_var_bits`), checked on the host.

    The example's own verdict comes from the device (`gpap_check_sound`); re-checking the
    printed bits here catches a readout that gathered the wrong columns. Under 32 variables, so
    no model is truncated."""
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


# ------------------------------------------------------- entry point 1, flat, on the C++ Engine

_FLAT_CNF = "p cnf 4 5\n1 2 0\n-1 3 0\n-2 -3 0\n1 3 4 0\n-4 2 3 0\n"
_FLAT_SETS = [[1], [-1, -2], [2, 3], [], [-3], [4, -2]]

# Built by the test against the header as it stands, so it can never run a stale binary.
_FLAT_SRC = r"""
#include <gpap/gpap.hpp>
#include <cstdio>
static void dump(const char *face, const gpap::CheckResults &r, std::int32_t m) {
    for (std::size_t i = 0; i < r.size(); ++i) {
        std::printf("%s %zu c=%d p=%d lits=", face, i, (int)r.conflict(i), r.num_propagated(i));
        for (std::int32_t l : r.literals(i)) std::printf("%d,", l);
        std::printf(" sat=");
        for (std::int32_t c = 0; c < m; ++c) std::printf("%d,", r.sat_count(i, c));
        std::printf("\n");
    }
}
int main(int, char **argv) {
    gpap::AssumptionChecker checker(argv[1]);
    const std::vector<std::vector<std::int32_t>> sets = {SETS};
    std::vector<std::int32_t> lits;
    std::vector<std::int64_t> offsets{0};
    for (const auto &s : sets) {
        lits.insert(lits.end(), s.begin(), s.end());
        offsets.push_back(static_cast<std::int64_t>(lits.size()));
    }
    gpap::Engine &e = checker.engine();
    const std::int32_t m = e.num_clauses();
    dump("check", checker.check(sets, /*assignments=*/true, /*sat_count=*/true), m);
    dump("flat", e.check_sets(lits.data(), lits.size(), offsets.data(), sets.size(),
                              /*assignments=*/true, /*sat_count=*/true), m);
    std::printf("k default %d\n", e.k());       // the last launch's rows: chunk reached the core
    dump("chunk1", e.check_sets(lits, offsets, true, true, /*chunk=*/1), m);
    std::printf("k chunk1 %d\n", e.k());
    std::printf("build %s\n", gpap::build_id().c_str());
    const std::vector<std::int32_t> zero{0};     // the core refuses literal 0 and keeps its message
    try { e.check_sets(zero, {0, 1}); } catch (const gpap::Error &) { std::printf("zero refused\n"); }
    const std::int64_t one[] = {0, 1};           // a guard in the core must name itself
    try { e.check_sets(nullptr, 5, one, 1); } catch (const gpap::Error &x) {
        std::printf("null lits: %s|\n", x.what());
    }
    try {                               // the last set reaches one literal past what was given
        e.check_sets(lits.data(), lits.size() - 1, offsets.data(), sets.size());
        std::printf("short lits accepted\n");
    } catch (const gpap::Error &x) { std::printf("short lits rejected: %s|\n", x.what()); }
    const gpap::CheckResults flags = e.check_sets(lits, offsets);   // flags and counts alone
    try { flags.value(0, 1); std::printf("value served\n"); }
    catch (const gpap::Error &) { std::printf("value refused\n"); }
    try { flags.sat_count(0, 0); std::printf("sat served\n"); }
    catch (const gpap::Error &) { std::printf("sat refused\n"); }
}
"""


def _build_flat_program(tmp_path):
    """Compile `_FLAT_SRC` against cpp/include and the built libgpap, or skip."""
    import os
    import shutil

    from gpap.engine import _core
    import shlex
    cxx = shlex.split(os.environ["CXX"]) if os.environ.get("CXX") else (
        [c] if (c := shutil.which("c++") or shutil.which("g++")) else None)
    if not cxx:
        pytest.skip("no C++ compiler")
    lib = next((c for c in _core._candidates() if c.exists()), None)   # the one Python loads
    if lib is None:
        pytest.skip("libgpap not built (scripts/build_core.sh)")
    sets = ", ".join("{" + ", ".join(map(str, s)) + "}" for s in _FLAT_SETS)
    src = tmp_path / "flat.cpp"
    src.write_text(_FLAT_SRC.replace("SETS", sets))
    exe = tmp_path / "flat"
    r = subprocess.run([*cxx, "-std=c++17", "-O1", f"-I{Path(__file__).parents[2] / 'cpp/include'}",
                        str(src), "-o", str(exe), str(lib), f"-Wl,-rpath,{lib.parent}"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    return exe


def test_the_flat_check_on_the_cpp_engine_matches_the_python_face(tmp_path):
    """`Engine::check_sets` takes the sets flat, as the C ABI and the Python face do, and gives
    what `AssumptionChecker::check` and Python's `check_sets` give: flags, implication counts,
    assignments and sat counts, row for row (a conflicted row only its flag, the one field the
    contract makes exact there), at any `chunk`. It refuses offsets that reach past the
    literals it was given, with its own message and not the core's last one, and the defaults
    return neither assignments nor sat counts."""
    import numpy as np

    from gpap.engine import _core

    exe = _build_flat_program(tmp_path)
    cnf = tmp_path / "f.cnf"
    cnf.write_text(_FLAT_CNF)
    out = subprocess.run([str(exe), str(cnf)], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr

    nv, clauses, _ = parse_dimacs(cnf)
    lits = np.array([x for s in _FLAT_SETS for x in s], dtype=np.int32)
    offsets = np.cumsum([0] + [len(s) for s in _FLAT_SETS]).astype(np.int64)
    eng = _core.Engine(_core.Formula.from_clauses(nv, clauses))
    conflict, prop, asg, sat = eng.check_sets(lits, offsets, assignments=True, sat_count=True)
    want = []
    for i in range(len(_FLAT_SETS)):
        if conflict[i]:
            want.append(f"{i} c=1")
            continue
        true = [(j // 2 + 1) * (1 if j % 2 == 0 else -1) for j in np.flatnonzero(asg[i])]
        want.append(f"{i} c=0 p={int(prop[i])} "
                    f"lits={''.join(f'{x},' for x in sorted(true, key=abs))} "
                    f"sat={''.join(f'{int(x)},' for x in sat[i])}")
    assert conflict.any() and not conflict.all(), "the sets must exercise both outcomes"
    lines = out.stdout.splitlines()
    for face in ("check", "flat", "chunk1"):
        got = [ln.split(" ", 1)[1] for ln in lines if ln.startswith(face + " ")]
        got = [" ".join(g.split()[:2]) if " c=1 " in g else g for g in got]
        assert got == want, (face, got, want)
    # the wrapper's own refusal, after a core error, carries none of the core's stale message
    assert f"build {_core.build_id()}" in lines, "the program ran another libgpap than Python"
    assert ("k default 6" in lines and "k chunk1 1" in lines), "chunk must reach the core"
    assert "zero refused" in lines
    assert "short lits rejected: Engine::check_sets: the last set ends at 8, past the 7 literals " \
           "given|" in lines, lines
    null = next(ln for ln in lines if ln.startswith("null lits: "))
    assert "lits is NULL" in null and "literal 0" not in null, null
    assert "value refused" in lines and "sat refused" in lines
