"""The leak gate over SHIPPED bytes — `eval/artifact/check_leaks.sh` — and the generators
behind it.

`make_artifact.sh` runs the gate over the staged tree before anything is archived, so a
machine-specific string is refused on the bytes a reviewer receives, not on the checkout. These
tests are the proof that it can fail (a gate that cannot fail is not a gate,
docs/findings/ARTIFACT_FINDINGS.md §8), and that it fails for EACH shape on its own: the second
leak found had a shape the first gate did not know — the operator's username flattened into a
directory name by an agent's scratch layout, with no `/home/` anywhere in it — and three rows of
the published Barbarik campaign shipped carrying it.

Literal names cannot live in the gate (the script ships in the tarball it checks), so those come
from a denylist file kept OUTSIDE the repository, and the gate reads it only when told where it is.
"""
import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
GATE = REPO / "eval" / "artifact" / "check_leaks.sh"

# Built by concatenation so that this file carries none of the shapes it injects: the hygiene
# test over tracked files (tests/test_docs_hygiene.py) would otherwise refuse this file itself.
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
    """`grep -F -f` treats an empty line as a pattern that matches EVERY line, so a denylist with
    a trailing newline or a comment would refuse every tree — and the operator would set the
    override to get past it, which mutes the gate for good."""
    tree = _tree(tmp_path, '{"note": "clean"}\n')
    deny = tmp_path / "deny.txt"
    deny.write_text("# names that must not ship\n\n   \nalice\n\n")
    r = _gate(tree, deny)
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_packager_runs_the_gate_over_the_staged_tree():
    """One definition: the packager must call the script, not carry a second copy of the shapes
    that drifts from it."""
    text = (REPO / "eval" / "artifact" / "make_artifact.sh").read_text()
    assert "check_leaks.sh" in text and '"$STAGE"' in text


def test_the_harness_error_text_carries_no_directory(tmp_path):
    """Three error rows of the published Barbarik campaign carried the absolute path of the
    instance — a scratch directory with the operator's username flattened into it — because the
    message printed the argument it was handed. The row already names the instance; the message
    needs the file, never where it sat on one disk."""
    from eval.uniformity.barbarik_gpap import _cmsgen_solutions
    cnf = tmp_path / "toy.cnf"
    cnf.write_text("p cnf 1 1\n1 0\n")
    with pytest.raises(RuntimeError) as err:
        _cmsgen_solutions(str(cnf), 1, [1], 0, "/bin/false")
    msg = str(err.value)
    assert "toy.cnf" in msg and str(tmp_path) not in msg, msg


def test_a_denylist_with_windows_line_endings_still_matches(tmp_path):
    """`grep -F -f` takes a trailing carriage return as part of the pattern, so a denylist saved
    on Windows (or pasted through one) matched nothing and the gate returned CLEAN — silently,
    which is the only way a gate fails that nobody notices. It is the last thing between a
    username and a DOI that cannot be edited."""
    tree = _tree(tmp_path, '{"note": "measured by Alice on her box"}\n')
    deny = tmp_path / "deny.txt"
    deny.write_bytes(b"# names\r\nalice\r\n")
    r = _gate(tree, deny)
    assert r.returncode == 1 and "rows.jsonl" in r.stdout, r.stdout + r.stderr
