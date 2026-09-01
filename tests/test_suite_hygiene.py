"""Invariants of the TEST SUITE itself — the ones whose breakage is silent.

A broken test does not usually fail; it stops testing. The three shapes that have actually
happened here get a guard, because none of them announces itself:

  * a path into the checkout written relative to the PROCESS's working directory, so the suite
    only passes from the repo root (32 tests died with FileNotFoundError from `tests/`, and one
    of them `pytest.skip`ped instead — a real check disappearing in silence);
  * a test with nothing that can raise — no assertion, no `pytest.raises`, no checking helper;
  * a `skipif` MARK on a helper function rather than on a test, which pytest only honours on
    collected items and which therefore guards nothing (see test_cnf_formula.read_support).

None of these catches the deepest form — a test whose assertions all hold for the wrong reason
(the 8|24 A/B comparing the base split against itself, the emission spy watching a call the
sampler no longer makes). Those are found by reading, and by making each gate fail on demand;
what is mechanical is checked here.

Host-only: this reads the test sources, it does not run them.
"""
import ast
import pathlib

import pytest

TESTS = pathlib.Path(__file__).resolve().parent
#: The directories a test may want to reach into. `tests/conftest.py` exports REPO_ROOT /
#: BENCH_DIR / SEQ_DIR / COMB_DIR; a bare relative string used as a WHOLE path is the bug.
CHECKOUT_DIRS = ("benchmarks/", "eval/", "scripts/", "cpp/", "gpap/", "build/", "docs/")


def _unanchored_path_strings(path):
    """String literals that name a checkout path and are NOT joined onto an anchor.

    Parsing rather than grepping, for two reasons. The prose in these files talks about
    `benchmarks/circuits/...` and `scripts/build_abc.sh` constantly, so a text search reads every
    mention as a use — docstrings are dropped here explicitly. And a FRAGMENT is legitimate:
    `Path(__file__).resolve().parents[1] / "cpp/include/gpap/gpap.hpp"` is anchored, and so is
    anything handed to `os.path.join(SEQ_DIR, ...)`. Only a literal standing on its own as a path
    depends on the working directory, so operands of `/` and non-leading `join` arguments are
    excluded — leaving exactly the shape that broke.
    """
    tree = ast.parse(path.read_text())
    skip = set()
    for node in ast.walk(tree):
        # docstrings
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) \
                and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant):
            skip.add(id(body[0].value))
        # `<anchor> / "fragment"` (pathlib) — either operand
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            skip.update({id(node.left), id(node.right)})
        # `os.path.join(<anchor>, "fragment", ...)` — everything after the first argument
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "join":
            skip.update(id(a) for a in node.args[1:])
    return [(n.lineno, n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in skip and n.value.startswith(CHECKOUT_DIRS)]


def test_no_test_reaches_into_the_checkout_by_a_relative_path():
    """Paths are anchored on `tests/conftest.py`'s REPO_ROOT, never on `os.getcwd()`.

    `pytest tests/` run from anywhere but the repo root used to fail 32 tests outright and skip a
    33rd — the occ/split check over a real benchmark, which looks for `benchmarks/sampling/*.cnf`
    and calls `pytest.skip("no benchmark CNFs in the checkout")` when it finds none. A check that
    turns itself off when it cannot find its input is the worst version of this: green, and gone.
    """
    offenders = []
    for path in sorted(TESTS.glob("*.py")):
        if path.name == pathlib.Path(__file__).name:
            continue                                   # CHECKOUT_DIRS itself
        offenders += [f"{path.name}:{n}: {v!r}" for n, v in _unanchored_path_strings(path)]
    assert not offenders, (
        "these paths depend on the working directory — anchor them on tests.conftest's "
        "REPO_ROOT / BENCH_DIR / SEQ_DIR / COMB_DIR instead:\n  " + "\n  ".join(offenders))


def test_the_relative_path_detector_actually_fires(tmp_path):
    """...and it must be able to see one, or the check above passes on its own blind spot.

    Both shapes at once: the bare literal must be reported and the anchored fragment must not."""
    src = tmp_path / "t.py"
    src.write_text('import os\n'
                   'A = "benchmarks/circuits/bench/seq/b03.bench"\n'
                   'B = os.path.join(SEQ_DIR, "benchmarks/x.bench")\n'
                   'C = ROOT / "cpp/include/gpap/gpap.h"\n')
    got = _unanchored_path_strings(src)
    assert [v for _, v in got] == ["benchmarks/circuits/bench/seq/b03.bench"], got


#: Tests where "it did not raise" IS the assertion, so there is nothing to assert. Named rather
#: than pattern-matched: the list is short, and each entry is a claim someone has to defend.
NO_ASSERT_BY_DESIGN = {
    # freeing the sampler after its engine must not touch freed memory — ASan is the oracle
    "test_freeing_the_sampler_does_not_touch_a_dead_engine",
    # the parser must ACCEPT these files; anything it returns is fine
    "test_dimacs_keeps_structural_leniency",
}


def test_every_test_can_actually_fail():
    """A test with no assertion, no `pytest.raises` and no checking helper cannot fail.

    That is the terminal form of the bug this file is about: it runs, it is green, and it is not
    a test. `assert*`-named calls (numpy's array comparisons) and `_check*` helpers count — they
    raise on mismatch — and the handful where an absent exception really is the whole claim are
    named in NO_ASSERT_BY_DESIGN above.
    """
    toothless = []
    for path in sorted(TESTS.glob("test_*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test") or node.name in NO_ASSERT_BY_DESIGN:
                continue
            if any(isinstance(x, ast.Assert) for x in ast.walk(node)):
                continue
            names = {getattr(c.func, "attr", "") or getattr(c.func, "id", "")
                     for c in ast.walk(node) if isinstance(c, ast.Call)}
            if any(n.startswith(("assert", "_check", "_assert")) or n == "raises" for n in names):
                continue
            toothless.append(f"{path.name}:{node.lineno} {node.name}")
    assert not toothless, (
        "these tests cannot fail — give them an assertion, or name them in "
        "NO_ASSERT_BY_DESIGN with a reason:\n  " + "\n  ".join(toothless))


def test_no_skip_mark_sits_on_a_helper_where_pytest_cannot_see_it():
    """pytest honours marks on COLLECTED items only. On a plain helper a `@pytest.mark.skipif`
    is inert decoration, so the guard it looks like silently is not one — which is how a helper
    that needs libgpap came to raise GpapError on every caller on a tree with no library."""
    offenders = []
    for path in sorted(TESTS.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("test") or node.name.startswith("_"):
                continue
            for dec in node.decorator_list:
                src = ast.dump(dec)
                if "'skipif'" in src or "'skip'" in src:
                    offenders.append(f"{path.name}:{node.lineno}: {node.name}")
    assert not offenders, (
        "a skip mark on a non-test function guards nothing; call pytest.skip() in the body "
        "instead:\n  " + "\n  ".join(offenders))


@pytest.mark.parametrize("name", ["REPO_ROOT", "BENCH_DIR", "SEQ_DIR", "COMB_DIR"])
def test_the_anchored_paths_exist(name):
    """...and the anchors themselves must point at something, or every path built on them is a
    FileNotFoundError one layer down."""
    from tests import conftest
    assert pathlib.Path(getattr(conftest, name)).is_dir(), name
