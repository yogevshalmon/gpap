"""Invariants of the DOCUMENTATION — the ones whose breakage is silent.

`docs/` is not prose beside the code: the engine, the eval layer, the public C headers and the
GENERATED paper text all cite it, and they cite it by file and by SECTION NUMBER. A pointer that
no longer resolves does not fail anything — it just quietly stops being evidence. Four such
pointers were found rotting on 2026-09-11: `INTEGRATION.md`, cited three times, had never existed
in this repository at all, and `docs/paper/KERNEL_AB.md` was still named by its own generator and
by the finding it belonged to nine days after being deleted.

What is mechanical is checked here:

  * a doc path named anywhere in the source trees exists;
  * a `FINDINGS.md §N` citation resolves to a real section of that file;
  * every findings file opens with its status block, so nobody has to guess whether a number in
    it is current, which card produced it, and whether the code still carries the rule;
  * no absolute home path or username reaches a tracked file (one lived in a committed
    `docs/slides/l4_results.tex` for a month, and `make_artifact.sh` would have shipped it);
  * a relative Markdown link resolves.

What it does NOT check is whether a doc is TRUE. A finding whose numbers have been superseded
still passes every check here; that is what the status block is for, and reading is what enforces
it. Host-only: this reads files, it runs nothing.
"""
import pathlib
import re

import pytest

from tests.conftest import REPO_ROOT

REPO = pathlib.Path(REPO_ROOT)
FINDINGS = REPO / "docs" / "findings"

#: Trees whose files may cite a doc. `docs/` itself is included: a findings file citing a dead
#: sibling is the same defect as code citing one.
SOURCE_TREES = ("cpp", "gpap", "eval", "scripts", "tests", "examples", "docs")
#: ...plus the front doors, which are not in any tree.
ROOT_DOCS = ("README.md", "ARTIFACT.md", "CLAUDE.md")
SOURCE_SUFFIXES = (".py", ".c", ".h", ".cpp", ".hpp", ".sh", ".md", ".tex", ".toml", ".cff")

#: A path mentioned in prose or code that we require to exist. Deliberately narrow: `docs/...`
#: and the two front doors are OURS, so a dead one is always our bug, whereas a bare
#: `gpap/sampling/` in a sentence is not worth the false positives.
DOC_PATH = re.compile(r"(?<![\w/.])(docs/[\w./*-]*[\w/*]|ARTIFACT\.md|cpp/README\.md)")
#: `*` is in the class so an interpolated segment lands INSIDE the match and the wildcard
#: test below sees it; leaving it out truncates the match and hides the wildcard.
#: `NAME_FINDINGS.md §12b` / `UNIFORMITY_FINDINGS.md §4b` — the citation form used everywhere.
CITATION = re.compile(r"([A-Z][A-Z0-9_]*_FINDINGS\.md)\s*§\s*(\d+[a-z]?)")
#: A heading that OPENS a section: `## 4.`, `### 16b.`, `## §16`, `## 4b —`.
HEADING = re.compile(r"^#{2,4}\s*§?\s*(\d+[a-z]?)\b", re.M)
#: Every relative Markdown link target, minus anchors and absolute URLs.
MD_LINK = re.compile(r"\[[^\]]*\]\(([^)#>]+?)(?:\s+\"[^\"]*\")?\)")

#: The fields the status block must carry, in the order a reader needs them.
REQUIRED_FIELDS = ("**Status:**", "**Measured:**")
#: ...and one of these, which is the distinction the directory exists to draw: is this file
#: evidence for a rule the CODE carries, or is it the only place the thing is written down?
PROVENANCE_FIELDS = ("**Rule lives in:**", "**Record only:**")


def _tracked_files():
    """Every file we are willing to police, as (path, text). Binaries and vendored trees out."""
    out = []
    for tree in SOURCE_TREES:
        for p in sorted((REPO / tree).rglob("*")):
            if p.is_file() and p.suffix in SOURCE_SUFFIXES and "__pycache__" not in p.parts:
                out.append(p)
    out += [REPO / n for n in ROOT_DOCS]
    return [(p, _readable(p)) for p in out if p.is_file()]


def _readable(path):
    """The file's text as a READER sees it, not as the syntax around it is written.

    LaTeX escapes an underscore (`UNIFORMITY\\_FINDINGS.md`) and Python interpolates a path
    (`f"docs/assets/animation/gpap_propagate_{...}.gif"`). Both name a real thing; neither
    matches a raw path regex. Unescaping and marking interpolation as the wildcard it already is
    keeps this check about dead pointers rather than about quoting.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".tex":
        text = text.replace("\\_", "_").replace("\\&", "&")
    return re.sub(r"\{[^{}\n]*\}", "*", text)


def _findings_files():
    return sorted(p for p in FINDINGS.glob("*.md") if p.name != "INSIGHTS.md")


def test_every_doc_path_named_in_the_tree_exists():
    """A citation that does not resolve is not a weaker pointer, it is no pointer at all.

    This is the check that would have caught `INTEGRATION.md` on the day it was written, and
    `docs/paper/KERNEL_AB.md` on the day the table was deleted.
    """
    dead = []
    for path, text in _tracked_files():
        if path.name == pathlib.Path(__file__).name:
            continue            # this file names the dead paths on purpose, as the history
        for m in DOC_PATH.finditer(text):
            target = m.group(1).rstrip(".,;:")
            if (REPO / target).exists():
                continue
            # `docs/paper/RESULTS.md` is generated into the checkout; a glob or a shell variable
            # inside the path is a pattern, not a claim about one file.
            if any(c in target for c in "*$<"):
                continue
            line = text[:m.start()].count("\n") + 1
            dead.append(f"{path.relative_to(REPO)}:{line}: {target}")
    assert not dead, (
        "these paths are cited but do not exist — fix the pointer or delete it:\n  "
        + "\n  ".join(dead))


def test_every_findings_section_citation_resolves():
    """Section numbers are an API.

    `eval/artifact/paper_tables.py` emits "RIVAL_FAIRNESS_FINDINGS.md §17" into the GENERATED
    `docs/paper/RESULTS.md`, and tests assert against section numbers by hand. So renumbering a
    section silently rewrites what a published sentence points at. Sections are append-only; this
    is the guard that says so out loud.
    """
    dead = []
    for path, text in _tracked_files():
        for m in CITATION.finditer(text):
            name, sec = m.group(1), m.group(2)
            target = FINDINGS / name
            if not target.exists():
                continue                                   # the path check above owns that case
            if sec in set(HEADING.findall(target.read_text())):
                continue
            line = text[:m.start()].count("\n") + 1
            dead.append(f"{path.relative_to(REPO)}:{line}: {name} §{sec}")
    assert not dead, (
        "these citations name a section that does not exist. NEVER renumber a section to make "
        "this pass — append one, and fix the citation:\n  " + "\n  ".join(dead))


@pytest.mark.parametrize("path", _findings_files(), ids=lambda p: p.name)
def test_every_findings_file_opens_with_its_status_block(path):
    """Status, date and card, and whether the code carries the rule — before the first claim.

    Without it a reader cannot tell a live measurement from a superseded one, and both look
    equally authoritative. Three files in this directory hold numbers that must NOT be quoted.
    """
    head = path.read_text().split("\n## ", 1)[0]
    missing = [f for f in REQUIRED_FIELDS if f not in head]
    assert not missing, (
        f"{path.name}: status block is missing {missing}. It goes in a blockquote directly under "
        f"the H1 — see any sibling.")
    assert any(f in head for f in PROVENANCE_FIELDS), (
        f"{path.name}: the status block must say either '**Rule lives in:**' (this is evidence "
        f"for something the code carries) or '**Record only:**' (this file is the only place it "
        f"is written down). That distinction is what makes the file safe to delete, or not.")


#: The shapes of a path that names one machine, mirrored from eval/artifact/check_leaks.sh: that
#: script gates the SHIPPED tree, this test the checkout, and bash and Python cannot share one
#: definition — so each names the other. tests/experiments/test_artifact_gate.py proves the script refuses each.
_LEAK_SHAPES = (r"/home/[a-z][\w.-]*/", r"/Users/[A-Za-z][\w.-]*/",
                r"-home-[a-z][\w.-]*-", r"/claude-[0-9]+/")


def test_no_tracked_file_carries_an_absolute_home_path():
    """A committed `/home/<user>/...` is a leak and a lie: it names a machine nobody else has.

    `docs/slides/l4_results.tex` carried one for a month, written there by a generator, and
    `eval/artifact/make_artifact.sh` archives HEAD — so it would have shipped to reviewers. The
    second leak found had no `/home/` in it: an agent's scratch directory with the home path
    flattened into its name. Hence shapes, not one pattern.
    """
    leaks = []
    for path, text in _tracked_files():
        if "test_docs_hygiene" in path.name:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if any(re.search(shape, line) for shape in _LEAK_SHAPES):
                leaks.append(f"{path.relative_to(REPO)}:{i}: {line.strip()[:90]}")
    assert not leaks, (
        "paths that name one machine, in tracked files — use $GPAP_RESULTS, $HOME or a relative "
        "path:\n  " + "\n  ".join(leaks))


def test_every_relative_markdown_link_resolves():
    """A dead link in a README is invisible until a reader clicks it."""
    dead = []
    docs = [p for p in (REPO / "docs").rglob("*.md")] + [REPO / n for n in ROOT_DOCS]
    docs += [REPO / "cpp" / "README.md", REPO / "eval" / "README.md",
             REPO / "examples" / "README.md", REPO / "eval" / "colab" / "README.md"]
    for path in sorted(set(d for d in docs if d.is_file())):
        text = path.read_text()
        for m in MD_LINK.finditer(text):
            target = m.group(1).strip()
            if target.startswith(("http://", "https://", "mailto:", "#", "data:")):
                continue
            if any(c in target for c in "*$<"):
                continue
            if (path.parent / target).exists():
                continue
            line = text[:m.start()].count("\n") + 1
            dead.append(f"{path.relative_to(REPO)}:{line}: {target}")
    assert not dead, "broken relative links:\n  " + "\n  ".join(dead)


def test_the_findings_index_lists_every_findings_file_and_nothing_else():
    """INSIGHTS.md is the way in. A file missing from it is a file nobody will read; a row
    pointing at a deleted file is a promise the directory does not keep.

    Only the TOPIC TABLE counts, not a passing mention in a paragraph. The table is the
    navigational contract — it is what says whether a file is current evidence or historical
    record — and a file cited only in prose has no such row. Matching prose links too made this
    check toothless: dropping a file's table row left it green.
    """
    index = (FINDINGS / "INSIGHTS.md").read_text()
    rows = [ln for ln in index.splitlines() if ln.startswith("|")]
    linked = {m.group(1) for m in re.finditer(r"\]\((\w[\w.]*\.md)\)", "\n".join(rows))}
    on_disk = {p.name for p in _findings_files()}
    assert not (on_disk - linked), (
        f"not listed in INSIGHTS.md: {sorted(on_disk - linked)}")
    assert not (linked - on_disk - {"INSIGHTS.md"}), (
        f"INSIGHTS.md links a file that is not there: {sorted(linked - on_disk - {'INSIGHTS.md'})}")


def test_the_python_package_version_is_the_same_number_in_all_three_places():
    """`pyproject.toml`, `gpap/__init__.py` and `CITATION.cff` each state the package version.

    Three copies of one number drift, and the one that drifts is the one nobody installs from:
    a `CITATION.cff` a release forgot is a citation to the wrong software.

    NOT the same number as libgpap's. `cpp/CMakeLists.txt` carries the C ABI version (0.3.0,
    SOVERSION 1) with its own break history, and `gpap_version()` returns that one. Conflating
    them would erase the ABI signal, so they are deliberately independent and this check does not
    look at CMake.
    """
    import ast
    init, proj_f, cite_f = (REPO / "gpap" / "__init__.py"), (REPO / "pyproject.toml"), \
        (REPO / "CITATION.cff")
    pkg = None
    for node in ast.parse(init.read_text()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "__version__":
            pkg = node.value.value
    proj = re.search(r"^version = \"([^\"]+)\"", proj_f.read_text(), re.M)
    cite = re.search(r"^version: (\S+)", cite_f.read_text(), re.M)
    assert pkg and proj and cite, "a version declaration went missing entirely"
    # Labels derived from the anchored paths, not written out: a literal like "gpap/__init__.py"
    # both drifts and trips tests/repo/test_suite_hygiene.py's working-directory detector.
    found = {str(init.relative_to(REPO)): pkg, str(proj_f.relative_to(REPO)): proj.group(1),
             str(cite_f.relative_to(REPO)): cite.group(1)}
    assert len(set(found.values())) == 1, f"the package version disagrees with itself: {found}"



def _export_ignored():
    """The paths `.gitattributes` keeps out of every archive `git archive` makes — which is what
    the artifact packager and GitHub's zip download both are."""
    out = []
    for line in (REPO / ".gitattributes").read_text().splitlines():
        body = line.split("#")[0].strip()
        if body.endswith("export-ignore"):
            out.append(body.split()[0])
    return out


def test_no_shipped_file_names_a_path_no_archive_carries():
    """An `export-ignore` line deletes a file from every archive while leaving it in the
    checkout, so a reference to it reads fine here and dangles for everyone who receives the
    tarball — the same dead-path class as the deleted `INTEGRATION.md`, reintroduced from the
    other end, and invisible to the test above because that one only checks that a path exists
    HERE.

    So: whatever `.gitattributes` withholds from archives may not be named by anything that
    ships. Adding an `export-ignore` line is therefore also a promise to reword its references.
    """
    ignored = _export_ignored()
    assert ignored, "no export-ignore entries — this gate would assert nothing"
    named = []
    for path, text in _tracked_files():
        if path.name in ("test_docs_hygiene.py", ".gitattributes"):
            continue                       # this gate and the file that declares the rule
        for target in ignored:
            pat = rf"(?<![\w/.]){re.escape(target.lstrip('/'))}\b"
            for i, line in enumerate(text.splitlines(), 1):
                if re.search(pat, line):
                    named.append(f"{path.relative_to(REPO)}:{i}: names {target}")
    assert not named, (
        "these SHIPPED files name a path that `.gitattributes` keeps out of every archive, so "
        "the reference dangles for everyone who unpacks one:\n  " + "\n  ".join(named))
