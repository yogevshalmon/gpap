"""Invariants of the documentation whose breakage is silent. Code, the eval layer and the
generated paper cite `docs/` by file and section number, and a dead pointer fails nothing.

  * a doc path named anywhere in the source trees exists;
  * a `FINDINGS.md §N` citation resolves to a real section of that file;
  * every findings file opens with its status block;
  * no absolute home path or username reaches a tracked file;
  * a relative Markdown link resolves;
  * the INSIGHTS.md table lists every findings file and nothing else;
  * the package version is the same in its three declarations;
  * no shipped file names a path that `.gitattributes` keeps out of archives.

None of this checks whether a doc is true. Host-only: it reads files and runs nothing.
"""
import pathlib
import re

import pytest

from tests.conftest import REPO_ROOT

REPO = pathlib.Path(REPO_ROOT)
FINDINGS = REPO / "docs" / "findings"

#: Trees whose files may cite a doc, `docs/` included.
SOURCE_TREES = ("cpp", "gpap", "eval", "scripts", "tests", "examples", "docs")
#: ...plus the front doors, which are not in any tree.
ROOT_DOCS = ("README.md", "CLAUDE.md", "CONTRIBUTING.md", "CHANGELOG.md", "THIRD_PARTY.md")
SOURCE_SUFFIXES = (".py", ".c", ".h", ".cpp", ".hpp", ".sh", ".md", ".tex", ".toml", ".cff")

#: A path that must exist. Narrow on purpose: a dead `docs/...` or front-door path is always our
#: bug, while matching any source path in prose would give false positives.
DOC_PATH = re.compile(r"(?<![\w/.])(docs/[\w./*-]*[\w/*]|cpp/README\.md)")
#: `*` is in the class so an interpolated segment stays inside the match, where the wildcard
#: test below skips it.
#: The citation form, e.g. `UNIFORMITY_FINDINGS.md §4b`.
CITATION = re.compile(r"([A-Z][A-Z0-9_]*_FINDINGS\.md)\s*§\s*(\d+[a-z]?)")
#: A heading that OPENS a section: `## 4.`, `### 16b.`, `## §16`, `## 4b —`.
HEADING = re.compile(r"^#{2,4}\s*§?\s*(\d+[a-z]?)\b", re.M)
#: Every relative Markdown link target, minus anchors and absolute URLs.
MD_LINK = re.compile(r"\[[^\]]*\]\(([^)#>]+?)(?:\s+\"[^\"]*\")?\)")

#: The fields the status block must carry.
REQUIRED_FIELDS = ("**Status:**", "**Measured:**")
#: ...and one of these: is the file evidence for a rule the code carries, or the only record?
PROVENANCE_FIELDS = ("**Rule lives in:**", "**Record only:**")


def _tracked_files():
    """Every policed file as (path, text); binaries and vendored trees excluded."""
    out = []
    for tree in SOURCE_TREES:
        for p in sorted((REPO / tree).rglob("*")):
            if p.is_file() and p.suffix in SOURCE_SUFFIXES and "__pycache__" not in p.parts:
                out.append(p)
    out += [REPO / n for n in ROOT_DOCS]
    return [(p, _readable(p)) for p in out if p.is_file()]


def _readable(path):
    """The file's text as a reader sees it: LaTeX escapes (`\\_`, `\\&`) undone and each
    interpolated `{...}` replaced by the wildcard `*`, so escaped and f-string paths still match.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".tex":
        text = text.replace("\\_", "_").replace("\\&", "&")
    return re.sub(r"\{[^{}\n]*\}", "*", text)


def _findings_files():
    return sorted(p for p in FINDINGS.glob("*.md") if p.name != "INSIGHTS.md")


def test_every_doc_path_named_in_the_tree_exists():
    """Every `docs/...` and front-door path named in the tree exists."""
    dead = []
    for path, text in _tracked_files():
        if path.name == pathlib.Path(__file__).name:
            continue            # this file may name dead paths as examples
        for m in DOC_PATH.finditer(text):
            target = m.group(1).rstrip(".,;:")
            if (REPO / target).exists():
                continue
            # a glob or shell variable in the path is a pattern, not one file
            if any(c in target for c in "*$<"):
                continue
            line = text[:m.start()].count("\n") + 1
            dead.append(f"{path.relative_to(REPO)}:{line}: {target}")
    assert not dead, (
        "these paths are cited but do not exist — fix the pointer or delete it:\n  "
        + "\n  ".join(dead))


def test_every_findings_section_citation_resolves():
    """Every findings section citation names a real section. Generated paper text cites
    section numbers, so sections are append-only: renumbering would silently repoint citations.
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
    """Each findings file states its status, when and where it was measured, and whether the
    code carries the rule, before its first claim, so a superseded number is not read as live.
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


#: Path shapes that name one machine, mirrored from eval/artifact/check_leaks.sh (which gates the
#: shipped tree; this test gates the checkout). tests/experiments/test_artifact_gate.py proves
#: the script refuses each.
_LEAK_SHAPES = (r"/home/[a-z][\w.-]*/", r"/Users/[A-Za-z][\w.-]*/",
                r"-home-[a-z][\w.-]*-", r"/claude-[0-9]+/")


def test_no_tracked_file_carries_an_absolute_home_path():
    """No tracked file names a path on one machine. Checked as several shapes, since a home
    path can also appear flattened into a scratch directory's name.
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
    """INSIGHTS.md's topic table links every findings file and no missing one. Only table rows
    count: a file linked only from prose has no status row, and counting prose would hide a
    dropped row.
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
    """`pyproject.toml`, `gpap/__init__.py` and `CITATION.cff` state the same package version.
    libgpap's ABI version (cpp/CMakeLists.txt, returned by `gpap_version()`) is independent on
    purpose and not checked here.
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
    # Labels come from the anchored paths: a literal "gpap/__init__.py" would trip
    # test_suite_hygiene.py's relative-path check.
    found = {str(init.relative_to(REPO)): pkg, str(proj_f.relative_to(REPO)): proj.group(1),
             str(cite_f.relative_to(REPO)): cite.group(1)}
    assert len(set(found.values())) == 1, f"the package version disagrees with itself: {found}"


def test_the_readme_images_are_pinned_to_the_release_tag():
    """README.md's absolute image links (the fallback outside GitHub) name the tag `v<version>`
    of the package version, so a version bump cannot leave them on an older tag.
    """
    version = re.search(r"^version = \"([^\"]+)\"", (REPO / "pyproject.toml").read_text(),
                        re.M).group(1)
    tags = re.findall(r"raw\.githubusercontent\.com/[^/]+/[^/]+/([^/]+)/",
                      (REPO / "README.md").read_text())
    assert tags, "README.md has no pinned image links — this check would assert nothing"
    assert set(tags) == {f"v{version}"}, f"README image links name {sorted(set(tags))}"


def _export_ignored():
    """Paths `.gitattributes` keeps out of every `git archive` (the artifact packager and
    GitHub's zip download)."""
    out = []
    for line in (REPO / ".gitattributes").read_text().splitlines():
        body = line.split("#")[0].strip()
        if body.endswith("export-ignore"):
            out.append(body.split()[0])
    return out


def test_no_shipped_file_names_a_path_no_archive_carries():
    """No shipped file names an `export-ignore` path: the reference resolves in the checkout but
    dangles in every archive, which the path-exists check above cannot see.
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


def _fenced_lines(text):
    """Every non-blank line inside a fenced code block, the commands and quoted outputs, with the
    indentation of a block nested in a list item removed."""
    lines, inside = set(), False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            inside = not inside
        elif inside and line.strip():
            lines.add(line.strip())
    return lines


@pytest.mark.parametrize("folder", sorted(p.parent for p in (REPO / "docs" / "papers").glob("*/AE_README.md")),
                         ids=lambda p: p.name)
def test_an_ae_readme_quotes_only_what_its_artifact_guide_says(folder):
    """A venue's AE README is an excerpt of the paper's ARTIFACT.md, the guide the archive carries
    and the image gate reads. Every command and quoted output in the excerpt appears in the
    guide, so the two cannot drift apart the way a copy does."""
    excerpt = _fenced_lines((folder / "AE_README.md").read_text())
    guide = _fenced_lines((folder / "ARTIFACT.md").read_text())
    assert excerpt, "the AE README quotes no command"
    assert not excerpt - guide, f"quoted in AE_README.md but not in ARTIFACT.md: {sorted(excerpt - guide)}"
