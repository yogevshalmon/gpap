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
import os
import pathlib
import re
import subprocess

import pytest

#: The checkout.
REPO = pathlib.Path(__file__).resolve().parents[1]

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


#: Every way this tree spells "the checkout, counted from this file". The depth is written by
#: hand, so moving a file changes the answer without changing the line — and a spelling missing
#: from here is a declaration nothing checks, which is how two of them sat wrong. Each
#: alternative is required to match below, so losing one reddens instead of passing quietly.
_ROOT_DECL = re.compile(
    r"^\s*(?P<name>[A-Z_]*(?:ROOT|REPO)[A-Z_]*)\s*=\s*(?:str\()?"
    r"(?:(?:pathlib\.)?Path\(__file__\)\.resolve\(\)"
    r"(?:\.parents\[(?P<n>\d+)\]|(?P<chain>(?:\.parent)+))"
    r'|os\.path\.abspath\(os\.path\.join\(os\.path\.dirname\(__file__\)(?P<ups>(?:,\s*"\.\.")+)'
    r"|(?P<dirnames>(?:os\.path\.dirname\()+)os\.path\.abspath\(__file__\))",
    re.M)


#: What the no-git walk steps over: every ignored directory that can hold a `.py` or `.sh` — the
#: venv, the build trees, the caches and the managed roots (all dot-directories), the egg-info.
#: Names, not paths, so a nested `build/` or `__pycache__` is pruned wherever it sits. Proven to
#: agree with git by `test_the_no_git_fallback_sees_what_git_sees`.
_PRUNE = {"__pycache__", "build", "build-asan", "venv", "dist", "node_modules"}


def _walk(*suffixes):
    """The file set WITHOUT git: a walk from the repo root that prunes the ignored trees — for an
    unpacked artifact or an sdist, where `git ls-files` has no repository to ask."""
    found = set()
    for root, dirs, files in os.walk(REPO):
        dirs[:] = sorted(d for d in dirs
                         if not (d.startswith(".") or d in _PRUNE or d.endswith(".egg-info")))
        for f in files:
            if f.endswith(suffixes):
                found.add(pathlib.Path(root, f).relative_to(REPO).as_posix())
    return found


def _tracked(*suffixes):
    """Files git tracks, filtered by suffix. `git ls-files` rather than `rglob`, so a gate can
    never redden because of somebody's local `.experiments/` or a third-party checkout under
    `.gpap/` — a gate that fires on scratch is a gate that gets muted.

    Without a repository — an unpacked artifact, an sdist — git answers nothing, and a scan over
    zero files asserts nothing: that silence turned four of these gates red (once, green) on
    exactly the tree reviewers are told to run `pytest` in. So the fallback is the pruned walk
    above, and the assertion that SOMETHING was found stands on both routes."""
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
    """A constant named for the checkout that points one directory off is the quietest bug in the
    tree: `sys.path` still imports (the cwd usually saves it), and every default built on it turns
    into a directory that simply does not exist, so the driver falls back or skips.

    It has happened here — an `eval/` reorg moved eleven files a level deeper and left their
    hand-counted depths behind, and a Colab driver still had one resolving to `<repo>/eval`,
    pointing its `--sampling-dir` and `--circuit-dir` at paths with no files in them. Counted
    here rather than trusted, for every tracked Python file.

    Each ALTERNATIVE of the pattern is required to match something, because the failure mode of a
    scan is silence: when this was one loose `>= 15`, corrupting either branch of the regex lost
    eight declarations and still passed."""
    seen, wrong = {"n": 0, "chain": 0, "ups": 0, "dirnames": 0}, []
    for rel in _tracked(".py"):
        f = REPO / rel
        text = f.read_text(errors="ignore")
        for m in _ROOT_DECL.finditer(text):
            if m.group("n") is not None:
                up, kind = int(m.group("n")), "n"
            elif m.group("chain") is not None:
                # `.parent` is parents[0], so a chain of k is parents[k-1].
                up, kind = m.group("chain").count(".parent") - 1, "chain"
            elif m.group("ups") is not None:
                up, kind = m.group("ups").count(".."), "ups"
            else:
                up, kind = m.group("dirnames").count("dirname(") - 1, "dirnames"
            seen[kind] += 1
            parents = f.resolve().parents
            got = parents[up] if up < len(parents) else None
            if got != REPO:
                wrong.append(f"{rel}:{text.count(chr(10), 0, m.start()) + 1}: "
                             f"{m.group('name')} -> {got}")
    assert not wrong, "repo-root constant does not resolve to the repo root:\n  " + "\n  ".join(wrong)
    empty = [k for k, v in seen.items() if not v]
    assert not empty, f"these spellings matched nothing — the pattern rotted: {empty} (counts {seen})"


#: The directories GPAP manages, and the ONE module allowed to name each. `gpap/paths.py` owns the
#: disposable-machinery root; `eval/paths.py` owns the experiments root, the results record and a
#: rented machine's layout. Both docstrings carry the reasoning; this is the enforcement.
_MANAGED = {".gpap": "gpap/paths.py", ".experiments": "eval/paths.py",
            "gpap-results": "eval/paths.py", "/content": "eval/paths.py",
            "/tmp": "gpap/paths.py"}
#: Shell has no owner module — every script ASKS (`python3 -m gpap.paths home`), so any script
#: re-deriving one of these is the drift this gate exists to stop.
_SHELL_DERIVATIONS = ("${GPAP_HOME:-", "${GPAP_RESULTS:-", "${GPAP_EXPERIMENTS:-")


def _is_managed_path(value: str) -> bool:
    """Is this string a PATH rooted at a managed directory, rather than prose describing one?

    A path literal has no whitespace; prose about a path does. That one discriminator is what lets
    the token match a whole path COMPONENT — which is what closes `f"{root}/.gpap/bin"`,
    `root + "/.gpap/bin"` and `"../gpap-results/paper"` — without flagging `gpap/cli.py`'s help
    text or the README demo's verbatim transcript line, and without a whitelist of prose sites.

    An ABSOLUTE token only counts as a prefix: `/tmp` must not match the bare component `"tmp"`,
    which is the commonest path component there is and used to make this gate fire on
    `tmp_path / "tmp"`.

    What it does NOT catch, stated rather than implied: a path assembled at runtime from pieces
    that are individually innocent, and `tempfile`'s own defaults (no literal to see). The
    structural rule — paths come from two modules — is the protection; this catches the
    reintroduction of a literal."""
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
    """ONE definition per managed directory, enforced rather than documented.

    Every path bug this suite has seen was one shape: a file computed a managed directory itself
    and its copy of the rule drifted — five scratch conventions, four derivations of the results
    root, a VM path built from the LOCAL machine's variable. Documenting the rule at each site is
    what produced three copies of one paragraph, one of them false. So the rule lives in two
    modules, and their own tests, and nothing else spells it.

    Scanned: tracked `.py` and `.sh`. A workflow YAML is not — `.github/workflows/ci.yml`
    links against a `/tmp/libcuda.so` stub it generates, which is right for an ephemeral
    runner with no checkout to outlive, and is deliberately none of these roots."""
    exempt = set(_MANAGED.values()) | {"tests/test_suite_hygiene.py", "tests/test_paths.py"}
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
                # A component, so `OUT=.experiments/x` counts (every driver cd's to the repo first,
                # so a leading slash was never required); `$OUT/.gpap_cache` does not.
                lead = r"(?:^|[\s\"'=:(])" if tok.startswith("/") else r"(?:^|[\s\"'=:(/])"
                if re.search(lead + re.escape(tok.lstrip("/") if not tok.startswith("/") else tok)
                             + r"(?:/|\"|\}|\s|$)", code):
                    bad.append(f"{rel}:{i}: names {tok} literally — ask its owner")
    assert not bad, "a managed directory is spelled outside its owner:\n  " + "\n  ".join(bad)


def test_every_internal_import_resolves():
    """`from eval.colab_sampling_plots import make_plots` sat in a live driver for three weeks
    after the module moved to `eval/colab/`, inside an `except Exception` that printed a one-line
    skip — so the benchmark ran to completion and silently produced no figures. Nothing in the
    suite looked at import paths, so a reorg could leave them behind invisibly.

    Resolution only: the module must EXIST where the import says. Importing it would need a GPU
    and a built library."""
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
    """A shell script that ASKS for a directory must refuse to continue without one.

    These drivers run `set -uo pipefail` deliberately — no `-e`, because a campaign must survive a
    failing stage — so a failed command substitution leaves the variable EMPTY and execution goes
    on. `make_artifact.sh` then reached `rm -rf "$OUT/..."` with `OUT=""`, i.e. at the filesystem
    root: denied for a user, DONE for root, which is what a rented VM is. Pure shell could not fail
    this way; asking a module introduced it.

    Written because the first attempt at the guards missed four sites — including the campaign
    driver — and nothing noticed. Under `set -e` the assignment itself aborts, so those scripts are
    exempt (their failure is loud already, if terse)."""
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
    """An unpacked artifact and an sdist have no `.git`, and `git ls-files` there produced
    nothing — so four gates in this file failed on exactly the tree reviewers are told to run
    `pytest` in. The fallback is a walk with a prune list; this is the proof that it scans the
    same files git does, run wherever git works (where it does not, the fallback is in use and
    has nothing to be compared against). Compared against tracked PLUS untracked-unignored, which
    is what a walk that honours the ignore list can see — a file you have not `git add`ed yet is
    not a discrepancy."""
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=REPO,
                         capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        pytest.skip("no git here: the fallback is in use, not under test")
    suffixes = (".py", ".sh", ".ipynb")
    by_git = {f for f in out.stdout.split() if f.endswith(suffixes)}
    assert _walk(*suffixes) == by_git
