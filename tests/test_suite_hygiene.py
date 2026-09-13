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
import re

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


#: `NAME = <count your way up from __file__>`, in the two spellings the repo uses. Anything named
#: for the repo root has to BE the repo root — the depth is written by hand, and moving a file one
#: directory changes the answer without changing the line.
_ROOT_DECL = re.compile(
    r"^(ROOT|REPO|REPO_ROOT|_REPO_ROOT|_REPO)\s*=\s*(?:str\()?"
    r"(?:Path\(__file__\)\.resolve\(\)\.parents\[(?P<n>\d+)\]"
    r'|os\.path\.abspath\(os\.path\.join\(os\.path\.dirname\(__file__\)(?P<ups>(?:,\s*"\.\.")+)'
    r"|(?P<dirnames>(?:os\.path\.dirname\()+)os\.path\.abspath\(__file__\))",
    re.M)


def test_every_repo_root_constant_is_the_repo_root():
    """A constant named for the checkout that points one directory off is the quietest bug in the
    tree: `sys.path` still imports (the cwd usually saves it), and every default built on it turns
    into a directory that simply does not exist, so the driver falls back or skips.

    It has happened here — an `eval/` reorg moved eleven files a level deeper and left their
    hand-counted depths behind, and `eval/colab/colab_sampling_benchmark.py` still had one in
    2026-09-13, resolving to `<repo>/eval` and pointing `--sampling-dir` and `--circuit-dir` at
    paths with no files in them. Counted here rather than trusted, for every file in the repo."""
    repo = pathlib.Path(__file__).resolve().parents[1]
    skip = {".gpap", "build", "build-asan", "__pycache__", ".venv", "venv", ".git", "node_modules"}
    checked, wrong = 0, []
    for f in sorted(repo.rglob("*.py")):
        if skip & set(f.parts):
            continue
        for m in _ROOT_DECL.finditer(f.read_text(errors="ignore")):
            if m.group("n") is not None:
                up = int(m.group("n"))
            elif m.group("ups") is not None:
                up = m.group("ups").count("..")
            else:
                up = m.group("dirnames").count("dirname(") - 1
            checked += 1
            got = f.resolve().parents[up] if up < len(f.resolve().parents) else None
            if got != repo:
                wrong.append(f"{f.relative_to(repo)}:{f.read_text().count(chr(10), 0, m.start()) + 1}"
                             f": {m.group(1)} -> {got}")
    assert checked >= 15, f"the declaration pattern stopped matching (only {checked} found)"
    assert not wrong, "repo-root constant does not resolve to the repo root:\n  " + "\n  ".join(wrong)


#: Where `gpap-results` may be SPELLED. rival_common.results_root is the one derivation; the two
#: shell entry points force the same normalization; paper_tables names it in prose.
_RESULTS_SITES = {"eval/rivals/rival_common.py", "eval/artifact/make_artifact.sh",
                  "eval/artifact/reproduce.sh", "eval/artifact/paper_tables.py"}


def test_the_results_root_has_one_spelling(tmp_path):
    """`$GPAP_RESULTS` had four derivations, and two of them left the fallback's `..` in the
    string. That `..` is not inert: it resolves PHYSICALLY when the path is opened (the kernel
    follows the symlink, then the `..`) and LEXICALLY when the same string goes through
    `abspath`/`relpath`, so for a checkout reached through a symlink one spelling named two
    directories — and `display_path` decides what a SHIPPED provenance header says by comparing
    exactly that. One derivation now, and the two shell entry points force it with `cd -P`,
    because bash's `cd` is logical where Python's `getcwd` is not.

    Both halves are gated: nowhere else may spell it, and the two languages must agree — checked
    THROUGH A SYMLINK, because that is the only layout where the lexical and physical forms differ
    and a plain checkout would pass either way (the agreement assertion was vacuous until it ran
    here)."""
    repo = pathlib.Path(__file__).resolve().parents[1]
    skip = {".gpap", "build", "build-asan", "__pycache__", ".venv", "venv", ".git", "node_modules"}
    me = pathlib.Path(__file__).resolve()
    found, offenders = set(), []
    for f in sorted(repo.rglob("*.py")) + sorted(repo.rglob("*.sh")):
        if skip & set(f.parts) or f.resolve() == me:   # a gate naming the string is not a spelling
            continue
        rel = f.relative_to(repo).as_posix()
        for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
            if "gpap-results" not in line or line.lstrip().startswith("#"):
                continue
            found.add(rel)
            if rel not in _RESULTS_SITES:
                offenders.append(f"{rel}:{i}: a fifth spelling — use rival_common.results_root")
            elif f.suffix == ".sh" and "cd -P" not in line:
                offenders.append(f"{rel}:{i}: shell fallback must normalize with `cd -P`")
    assert not offenders, "the results root drifted:\n  " + "\n  ".join(offenders)
    assert found == _RESULTS_SITES, f"the scan stopped matching: {found} != {_RESULTS_SITES}"

    # ...and the two languages must land in the same directory. The shell halves are evaluated
    # AS WRITTEN — the `RESULTS=` line is read out of each file, not retyped here — through a
    # SYMLINKED checkout, which is the only layout where they can disagree: bash's `cd` and `$PWD`
    # are logical, so dropping the `-P` silently yields the symlink's parent instead of the
    # checkout's real one. (Python needs no forcing: an imported module's `__file__` already comes
    # back resolved. `results_root` says `realpath` anyway, so the intent is in the code rather
    # than inherited from an interpreter detail.)
    import os
    import subprocess
    import sys
    link = tmp_path / "checkout"
    link.symlink_to(repo, target_is_directory=True)
    env = {k: v for k, v in os.environ.items() if k != "GPAP_RESULTS"}
    py = subprocess.run([sys.executable, "-c",
                         "from eval.rivals.rival_common import results_root; print(results_root())"],
                        cwd=str(link), env={**env, "PYTHONPATH": str(link)},
                        capture_output=True, text=True)
    assert py.returncode == 0, py.stderr
    for rel in ("eval/artifact/reproduce.sh", "eval/artifact/make_artifact.sh"):
        line = next((ln.strip() for ln in (repo / rel).read_text().splitlines()
                     if ln.strip().startswith("RESULTS=")), None)
        assert line, f"{rel}: no RESULTS= line to check — did the entry point stop deriving it?"
        # NEWLINES, not `;`: one of these lines ends in a trailing comment, which would
        # otherwise swallow the printf and compare against an empty string.
        sh = subprocess.run(["bash", "-c", f'REPO_ROOT="$1"\n{line}\nprintf %s "$RESULTS"',
                             "_", str(link)], env=env, capture_output=True, text=True)
        assert sh.returncode == 0, f"{rel}: {sh.stderr}"
        assert sh.stdout == py.stdout.strip(), \
            (f"through a symlinked checkout {rel} resolves {sh.stdout!r} while "
             f"rival_common.results_root says {py.stdout.strip()!r}")
