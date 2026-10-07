"""Tests for the leak gate `eval/artifact/check_leaks.sh`, which the packager runs over the staged
tree. Each leak shape must be refused on its own (docs/findings/ARTIFACT_FINDINGS.md §8).
Literal names come from a denylist kept outside the repository, since the script ships in the
tarball it checks.
"""
import pathlib
import subprocess

import pytest

from tests.conftest import REPO_ROOT

REPO = pathlib.Path(REPO_ROOT)
GATE = REPO / "eval" / "artifact" / "check_leaks.sh"

# Built by concatenation so this file carries none of the shapes; otherwise the hygiene test
# over tracked files (tests/repo/test_docs_hygiene.py) would refuse it.
SHAPES = {
    "a linux home": "/home/" + "alice/proj/x.cnf",
    "a macos home": "/Users/" + "alice/proj/x.cnf",
    "a home flattened into one directory name": "/scratch/-home-" + "alice-proj-x/x.cnf",
    "an agent scratch root": "/scratch/claude-" + "1000/abc/x.cnf",
}


def _gate(tree, denylist=None):
    cmd = ["bash", str(GATE), str(tree)]
    if denylist is not None:
        cmd += ["--denylist", str(denylist)]
    return subprocess.run(cmd, capture_output=True, text=True)


def _tree(tmp_path, text):
    d = tmp_path / "stage"
    (d / "sub").mkdir(parents=True)
    (d / "sub" / "rows.jsonl").write_text(text)
    (d / "README.md").write_text("nothing to see\n")
    return d


def test_a_clean_tree_passes(tmp_path):
    r = _gate(_tree(tmp_path, '{"error": "x failed on x.cnf"}\n'))
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_each_leak_shape_is_refused_and_the_file_named(tmp_path, shape):
    r = _gate(_tree(tmp_path, f'{{"error": "x failed on {SHAPES[shape]}"}}\n'))
    assert r.returncode == 1, f"{shape}: exit {r.returncode}\n{r.stdout}{r.stderr}"
    assert "rows.jsonl" in r.stdout, r.stdout


def test_a_denylisted_literal_is_refused_case_insensitively(tmp_path):
    tree = _tree(tmp_path, '{"note": "measured by Alice on her box"}\n')
    assert _gate(tree).returncode == 0                       # no list: a name is not a shape
    deny = tmp_path / "deny.txt"
    deny.write_text("alice\n")
    r = _gate(tree, deny)
    assert r.returncode == 1 and "rows.jsonl" in r.stdout, r.stdout + r.stderr


def test_blank_and_comment_lines_in_the_denylist_match_nothing(tmp_path):
    """`grep -F -f` treats an empty line as a pattern that matches every line, so without this a
    denylist with a blank line would refuse every tree."""
    tree = _tree(tmp_path, '{"note": "clean"}\n')
    deny = tmp_path / "deny.txt"
    deny.write_text("# names that must not ship\n\n   \nalice\n\n")
    r = _gate(tree, deny)
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_packager_runs_the_gate_over_the_staged_tree():
    """The packager calls the script rather than carrying its own copy of the shapes."""
    text = (REPO / "eval" / "artifact" / "make_artifact.sh").read_text()
    assert "check_leaks.sh" in text and '"$STAGE"' in text


def test_the_harness_error_text_carries_no_directory(tmp_path):
    """The Barbarik harness error names the instance file but not its directory, which can carry
    a username (docs/findings/ARTIFACT_FINDINGS.md §9)."""
    from eval.uniformity.barbarik_gpap import _cmsgen_solutions
    cnf = tmp_path / "toy.cnf"
    cnf.write_text("p cnf 1 1\n1 0\n")
    with pytest.raises(RuntimeError) as err:
        _cmsgen_solutions(str(cnf), 1, [1], 0, "/bin/false")
    msg = str(err.value)
    assert "toy.cnf" in msg and str(tmp_path) not in msg, msg


def test_a_denylist_with_windows_line_endings_still_matches(tmp_path):
    """`grep -F -f` takes a trailing carriage return as part of the pattern, so a CRLF denylist
    would otherwise match nothing and the gate would pass silently."""
    tree = _tree(tmp_path, '{"note": "measured by Alice on her box"}\n')
    deny = tmp_path / "deny.txt"
    deny.write_bytes(b"# names\r\nalice\r\n")
    r = _gate(tree, deny)
    assert r.returncode == 1 and "rows.jsonl" in r.stdout, r.stdout + r.stderr
