"""Invariants of the test suite and repo layout whose breakage is silent (a broken test usually
stops testing rather than failing):

  * no test reaches into the checkout by a path relative to the working directory;
  * every test can fail (an assertion, `pytest.raises` or a checking helper);
  * no skip mark sits on a helper, where pytest ignores it;
  * the conftest path anchors exist;
  * every repo-root constant resolves to the repo root;
  * only the two paths modules name a managed directory;
  * every internal import resolves;
  * every shell ask for a path is guarded against an empty answer;
  * the no-git file walk sees what git sees;
  * the default pytest run covers every test directory except tests/experiments.

A test whose assertions hold for the wrong reason is not caught here. Host-only: this reads
sources and runs nothing.
"""
import ast
import os
import pathlib
import re
import subprocess

import pytest

from tests.conftest import REPO_ROOT

#: The checkout.
REPO = pathlib.Path(REPO_ROOT)
TESTS = REPO / "tests"
#: Directories a test may reach into. `tests/conftest.py` exports REPO_ROOT / BENCH_DIR /
#: SEQ_DIR / COMB_DIR; a bare relative string used as a whole path is the bug.
CHECKOUT_DIRS = ("benchmarks/", "eval/", "scripts/", "cpp/", "gpap/", "build/", "docs/")


def _unanchored_path_strings(path):
    """String literals that name a checkout path and are not joined onto an anchor.

    Parsed rather than grepped, so docstrings (which mention such paths) are skipped, and so are
    fragments joined onto an anchor: operands of `/` and non-leading `join` arguments.
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
        # `<anchor> / "fragment"` (pathlib): either operand
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            skip.update({id(node.left), id(node.right)})
        # `os.path.join(<anchor>, "fragment", ...)`: every argument after the first
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "join":
            skip.update(id(a) for a in node.args[1:])
    return [(n.lineno, n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in skip and n.value.startswith(CHECKOUT_DIRS)]


def test_no_test_reaches_into_the_checkout_by_a_relative_path():
    """Paths are anchored on `tests/conftest.py`'s REPO_ROOT, never on the working directory, so
    the suite passes from anywhere. Otherwise a test that skips when its input is missing would
    silently stop running.
    """
    offenders = []
    for path in sorted(TESTS.rglob("*.py")):
        if path.name == pathlib.Path(__file__).name:
            continue                                   # CHECKOUT_DIRS itself
        offenders += [f"{path.relative_to(TESTS)}:{n}: {v!r}" for n, v in _unanchored_path_strings(path)]
    assert not offenders, (
        "these paths depend on the working directory — anchor them on tests.conftest's "
        "REPO_ROOT / BENCH_DIR / SEQ_DIR / COMB_DIR instead:\n  " + "\n  ".join(offenders))


def test_the_relative_path_detector_actually_fires(tmp_path):
    """The detector reports a bare literal and ignores anchored fragments, so the check above is
    not blind."""
    src = tmp_path / "t.py"
    src.write_text('import os\n'
                   'A = "benchmarks/circuits/bench/seq/b03.bench"\n'
                   'B = os.path.join(SEQ_DIR, "benchmarks/x.bench")\n'
                   'C = ROOT / "cpp/include/gpap/gpap.h"\n')
    got = _unanchored_path_strings(src)
    assert [v for _, v in got] == ["benchmarks/circuits/bench/seq/b03.bench"], got


#: Tests where not raising is the whole claim. Named rather than pattern-matched, so each entry
#: has to be justified.
NO_ASSERT_BY_DESIGN = {
    # freeing the sampler after its engine must not touch freed memory (ASan is the oracle)
    "test_freeing_the_sampler_does_not_touch_a_dead_engine",
    # the parser must accept these files; what it returns does not matter
    "test_dimacs_keeps_structural_leniency",
}


def test_every_test_can_actually_fail():
    """Every test has an assertion, a `pytest.raises` or a checking helper (`assert*` and
    `_check*` calls count), or is listed in NO_ASSERT_BY_DESIGN.
    """
    toothless = []
    for path in sorted(TESTS.rglob("test_*.py")):
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
    """pytest honours marks only on collected tests, so a skip mark on a helper guards nothing."""
    offenders = []
    for path in sorted(TESTS.rglob("*.py")):
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
    """The conftest anchors point at real directories."""
    from tests import conftest
    assert pathlib.Path(getattr(conftest, name)).is_dir(), name


#: Every way this tree spells "the checkout, counted from this file". Depths are written by
#: hand, so moving a file breaks them silently, and a spelling missing here goes unchecked.
#: Each alternative is tested against a sample below.
_ROOT_DECL = re.compile(
    r"^\s*(?P<name>[A-Z_]*(?:ROOT|REPO)[A-Z_]*)\s*=\s*(?:str\()?"
    r"(?:(?:pathlib\.)?Path\(__file__\)\.resolve\(\)"
    r"(?:\.parents\[(?P<n>\d+)\]|(?P<chain>(?:\.parent)+))"
    r'|os\.path\.abspath\(os\.path\.join\(os\.path\.dirname\(__file__\)(?P<ups>(?:,\s*"\.\.")+)'
    r"|(?P<dirnames>(?:os\.path\.dirname\()+)os\.path\.abspath\(__file__\))",
    re.M)


#: What the no-git walk skips: ignored directories that can hold `.py` or `.sh` (dot-directories
#: and egg-info are skipped separately). Names, not paths, so nested ones are pruned too.
#: Checked against git by `test_the_no_git_fallback_sees_what_git_sees`.
_PRUNE = {"__pycache__", "build", "build-asan", "venv", "dist", "node_modules"}


def _walk(*suffixes):
    """The file set without git (an unpacked artifact or sdist): a walk from the repo root that
    prunes ignored trees."""
    found = set()
    for root, dirs, files in os.walk(REPO):
        dirs[:] = sorted(d for d in dirs
                         if not (d.startswith(".") or d in _PRUNE or d.endswith(".egg-info")))
        for f in files:
            if f.endswith(suffixes):
                found.add(pathlib.Path(root, f).relative_to(REPO).as_posix())
    return found


def _tracked(*suffixes):
    """Files git tracks, filtered by suffix, so local scratch never trips a gate. Without a
    repository it falls back to `_walk`; either way it asserts that something was found, since a
    scan over zero files checks nothing."""
    try:
        out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True)
        found = [f for f in out.stdout.split() if f.endswith(suffixes)] if out.returncode == 0 \
            else []
    except FileNotFoundError:                      # no git binary at all
        found = []
    if not found:
        found = sorted(_walk(*suffixes))
    assert found, f"no {suffixes} files found, by git or by walking {REPO} (not a checkout?)"
    return found


def test_every_repo_root_constant_is_the_repo_root():
    """Every repo-root constant in a tracked Python file resolves to the repo root. One that is a
    directory off still imports, but defaults built on it point nowhere, so drivers silently fall
    back or skip. Each regex alternative must match its sample, or a broken branch would pass."""
    for kind, sample in (
            ("n", "REPO = Path(__file__).resolve().parents[1]"),
            ("chain", "REPO = Path(__file__).resolve().parent.parent"),
            ("ups", 'ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))'),
            ("dirnames", "ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))")):
        m = _ROOT_DECL.match(sample)
        assert m and m.group(kind) is not None, f"the {kind} spelling no longer matches: {sample}"
    wrong = []
    for rel in _tracked(".py"):
        f = REPO / rel
        text = f.read_text(errors="ignore")
        for m in _ROOT_DECL.finditer(text):
            if m.group("n") is not None:
                up = int(m.group("n"))
            elif m.group("chain") is not None:
                # `.parent` is parents[0], so a chain of k is parents[k-1].
                up = m.group("chain").count(".parent") - 1
            elif m.group("ups") is not None:
                up = m.group("ups").count("..")
            else:
                up = m.group("dirnames").count("dirname(") - 1
            parents = f.resolve().parents
            got = parents[up] if up < len(parents) else None
            if got != REPO:
                wrong.append(f"{rel}:{text.count(chr(10), 0, m.start()) + 1}: "
                             f"{m.group('name')} -> {got}")
    assert not wrong, "repo-root constant does not resolve to the repo root:\n  " + "\n  ".join(wrong)


#: Managed directories and the one module allowed to name each. `gpap/paths.py` owns the
#: disposable-machinery root; `eval/paths.py` the experiments root, the results record and a
#: rented machine's layout. Their docstrings give the reasons.
_MANAGED = {".gpap": "gpap/paths.py", ".experiments": "eval/paths.py",
            "gpap-results": "eval/paths.py", "/content": "eval/paths.py",
            "/tmp": "gpap/paths.py"}
#: Shell has no owner module: scripts ask (`python3 -m gpap.paths home`), so a script that
#: re-derives one of these is drift.
_SHELL_DERIVATIONS = ("${GPAP_HOME:-", "${GPAP_RESULTS:-", "${GPAP_EXPERIMENTS:-")


def _is_managed_path(value: str) -> bool:
    """Whether this string is a path rooted at a managed directory, rather than prose about one.

    A path literal has no whitespace and prose does, so a token can match any path component
    (`f"{root}/.gpap/bin"`, `"../gpap-results/paper"`) without flagging help text. An absolute
    token counts only as a prefix, so `/tmp` does not match the component `"tmp"`. Paths built at
    runtime from innocent pieces, and `tempfile`'s defaults, are not caught."""
    v = value.strip()
    if not v or any(c.isspace() for c in v):
        return False
    parts = [p for p in v.replace("\\", "/").split("/") if p and p != "."]
    for tok in _MANAGED:
        if tok.startswith("/"):
            if v == tok or v.startswith(tok + "/"):
                return True
        elif tok in parts:
            return True
    return False


def test_only_the_paths_modules_name_a_managed_directory():
    """Only the two paths modules (and their tests) spell a managed directory; everything else
    asks them, so no copy of the rule can drift. Scans tracked `.py` and `.sh`, not workflow
    YAML: CI's `/tmp` libcuda stub lives on an ephemeral runner and is none of these roots."""
    exempt = set(_MANAGED.values()) | {"tests/repo/test_suite_hygiene.py",
                                       "tests/experiments/test_eval_paths.py"}
    bad = []
    for rel in _tracked(".py"):
        if rel in exempt:
            continue
        try:
            tree = ast.parse((REPO / rel).read_text(errors="ignore"))
        except SyntaxError:                       # not ours to parse; other gates cover syntax
            continue
        docs = set()
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                    and body and isinstance(body[0], ast.Expr) \
                    and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                docs.add(id(body[0].value))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and id(node) not in docs and _is_managed_path(node.value):
                bad.append(f"{rel}:{node.lineno}: the path {node.value[:40]!r} — ask its owner")
    for rel in _tracked(".sh"):
        for i, line in enumerate((REPO / rel).read_text(errors="ignore").splitlines(), 1):
            code = re.sub(r"(^|\s)#.*$", "", line)   # a comment may describe a path freely
            for d in _SHELL_DERIVATIONS:
                if d in code:
                    bad.append(f"{rel}:{i}: re-derives {d} — ask `python3 -m gpap.paths`")
            for tok in _MANAGED:
                # Matched as a component: `OUT=.experiments/x` counts (drivers cd to the repo
                # first), `$OUT/.gpap_cache` does not.
                lead = r"(?:^|[\s\"'=:(])" if tok.startswith("/") else r"(?:^|[\s\"'=:(/])"
                if re.search(lead + re.escape(tok.lstrip("/") if not tok.startswith("/") else tok)
                             + r"(?:/|\"|\}|\s|$)", code):
                    bad.append(f"{rel}:{i}: names {tok} literally — ask its owner")
    assert not bad, "a managed directory is spelled outside its owner:\n  " + "\n  ".join(bad)


def test_every_internal_import_resolves():
    """Every internal import (gpap, eval, tests) names a module that exists. A stale one inside
    an `except` that prints a skip fails silently. Resolution only: importing would need a GPU and
    a built library."""
    import importlib.util
    broken = []
    for rel in _tracked(".py", ".ipynb"):
        text = (REPO / rel).read_text(errors="ignore")
        for m in re.finditer(r"(?:^|\\n)\s*(?:from|import)\s+((?:gpap|eval|tests)[a-zA-Z0-9_.]*)",
                             text, re.M):
            mod = m.group(1).rstrip(".")
            try:
                if importlib.util.find_spec(mod) is None:
                    broken.append(f"{rel}: {mod}")
            except (ImportError, ValueError):
                broken.append(f"{rel}: {mod}")
    assert not broken, ("an internal import names a module that does not exist:\n  "
                        + "\n  ".join(sorted(set(broken))))


def test_every_shell_ask_for_a_path_is_guarded():
    """A shell script that asks a paths module for a directory refuses to continue without one.
    The drivers run `set -uo pipefail` without `-e` (a campaign must survive a failing stage), so
    a failed substitution leaves the variable empty and a later `rm -rf "$OUT/..."` would hit the
    filesystem root. Scripts under `set -euo` abort at the assignment and are exempt."""
    missing = []
    for rel in _tracked(".sh"):
        text = (REPO / rel).read_text(errors="ignore")
        if "set -euo" in text:                      # aborts at the assignment; no guard reachable
            continue
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if not re.search(r"-m (?:gpap|eval)\.paths \w+", line):
                continue
            var = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=", line)
            if not var:
                missing.append(f"{rel}:{i + 1}: an ask that assigns nothing")
                continue
            if f"{var.group(1)}:?" not in "\n".join(lines[i + 1:i + 3]):
                missing.append(f"{rel}:{i + 1}: {var.group(1)} is not guarded by `: \"${{...:?}}\"`")
    assert not missing, ("a shell ask may not continue with an empty path:\n  "
                         + "\n  ".join(missing))


def test_the_no_git_fallback_sees_what_git_sees():
    """The no-git walk (used in an unpacked artifact or sdist) sees the files git sees: tracked
    plus untracked-unignored, since a file not yet added is not a discrepancy. Skipped where git
    is unavailable, because the fallback is then the one in use."""
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=REPO,
                         capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        pytest.skip("no git here: the fallback is in use, not under test")
    suffixes = (".py", ".sh", ".ipynb")
    by_git = {f for f in out.stdout.split() if f.endswith(suffixes)}
    assert _walk(*suffixes) == by_git


def test_the_default_run_covers_every_test_directory_but_eval():
    """`testpaths` lists directories one by one, so every directory holding tests is listed,
    except tests/experiments; a new unlisted one would never run."""
    block = re.search(r"^testpaths\s*=\s*\[(.*?)\]", (REPO / "pyproject.toml").read_text(),
                      re.M | re.S).group(1)
    listed = set(re.findall(r'"([^"]+)"', block))
    holding = {p.parent.relative_to(REPO).as_posix() for p in TESTS.rglob("test_*.py")}
    assert holding - listed == {"tests/experiments"}, f"not in the default run: {holding - listed}"
    assert listed <= holding, f"testpaths names a directory with no tests: {listed - holding}"
