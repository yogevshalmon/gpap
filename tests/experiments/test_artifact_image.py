"""The artifact image, checked without Docker: what it pins, installs and copies agrees with the
files it has to agree with. Building and running it is eval/artifact/check_image.sh's job.

  * the tool image installs the torch and Triton the artifact lock pins;
  * the artifact layer installs the compared samplers' dependencies as rivals_setup.sh does;
  * every file an archive ships reaches the artifact image's /opt/gpap;
  * the entry point accepts exactly reproduce.sh's steps;
  * the image's scripts parse.
"""
import fnmatch
import pathlib
import re
import shlex
import subprocess

import pytest

from tests.conftest import REPO_ROOT

REPO = pathlib.Path(REPO_ROOT)
ARTIFACT = REPO / "eval" / "artifact"
LOCK = ARTIFACT / "requirements-artifact.txt"
PAPER_SECTION = "# The paper's files"


def _arg(name):
    """The default of one ARG in the tool image's Dockerfile."""
    m = re.search(rf"^ARG {name}=(\S+)$", (REPO / "Dockerfile").read_text(), re.M)
    assert m, f"the Dockerfile declares no ARG {name}"
    return m.group(1)


def _locked(name):
    """The version the artifact lock pins for one package."""
    m = re.search(rf"^{name}==([^\s;]+)", LOCK.read_text(), re.M)
    assert m, f"{LOCK.name} pins no {name}"
    return m.group(1)


def test_the_tool_image_pins_the_torch_and_triton_the_artifact_locks():
    """Every published row was measured with the lock's torch, and its Triton compiled the
    kernels. The tool image installs the same two from the same index, so the artifact layer,
    which installs the lock over it, changes neither and carries no second torch."""
    assert _arg("TORCH") == _locked("torch")
    assert _arg("TRITON") == _locked("triton")
    index = re.search(r"^--extra-index-url\s+(\S+)", LOCK.read_text(), re.M)
    assert index and _arg("TORCH_INDEX") == index.group(1)


def test_the_image_installs_the_compared_samplers_deps_as_rivals_setup_does():
    """`setup` runs rivals_setup.sh, which pip-installs the compared samplers' dependencies. The
    image installs the same requirements, so in a container that line is satisfied and needs no
    network. Drift would surface only in the offline gate, much later."""
    setup = (REPO / "eval" / "rivals" / "rivals_setup.sh").read_text()
    m = re.search(r"-m pip install -q (.+?)\s*\\$", setup, re.M)
    assert m, "rivals_setup.sh no longer has its pip line"
    want = shlex.split(m.group(1))
    docker = (ARTIFACT / "Dockerfile").read_text()
    have = [shlex.split(g) for g in re.findall(r"pip install (.+?)\s*(?:\\|&&|$)", docker, re.M)]
    assert want in have, f"the image does not install {want}; it installs {have}"


def _dockerignore():
    """The root .dockerignore as (local-state patterns, paper patterns)."""
    local, paper, section = [], [], None
    for line in (REPO / ".dockerignore").read_text().splitlines():
        if line.startswith(PAPER_SECTION):
            section = paper
        elif line.strip() and not line.startswith("#"):
            (section if section is not None else local).append(line.strip().rstrip("/"))
    return local, paper


def _ignored(path, patterns):
    """Whether Docker's context filter drops this relative path: a pattern matching the path or
    any directory above it, with `**/` matching at any depth."""
    parts = path.split("/")
    prefixes = ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]
    for pat in patterns:
        tail = pat[3:] if pat.startswith("**/") else None
        for pre in prefixes:
            if fnmatch.fnmatchcase(pre, pat) or (tail and fnmatch.fnmatchcase(pre.split("/")[-1], tail)):
                return True
    return False


def test_every_file_an_archive_ships_reaches_the_artifact_image():
    """An archive is the tracked files plus what make_artifact.sh adds. The tool image drops no
    tracked file, and the artifact layer copies back exactly the files .dockerignore keeps out
    of the tool image, each of which the packager writes. So in the artifact image /opt/gpap is
    the archive and MANIFEST.sha256 verifies; check_image.sh confirms that on a real image."""
    local, paper = _dockerignore()
    assert paper, f".dockerignore has no '{PAPER_SECTION}' section"
    try:
        tracked = subprocess.run(["git", "-C", str(REPO), "ls-files"], capture_output=True,
                                 text=True, check=True).stdout.split()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("no git here: the tracked files are unknown")
    dropped = [p for p in tracked if _ignored(p, local + paper)]
    assert not dropped, f".dockerignore drops tracked files from the tool image: {dropped[:10]}"
    copied = []
    for m in re.finditer(r"^COPY (.+) /opt/gpap/?\S*$", (ARTIFACT / "Dockerfile").read_text(), re.M):
        copied += m.group(1).split()
    assert sorted(pat for pat in paper if not any(fnmatch.fnmatchcase(c, pat) for c in copied)) == []
    assert sorted(c for c in copied if not _ignored(c, paper)) == []
    packager = (ARTIFACT / "make_artifact.sh").read_text()
    assert sorted(c for c in copied if c not in packager) == [], "the packager writes no such file"


def test_the_entrypoint_takes_exactly_reproduce_sh_steps():
    """The entry point runs a step through reproduce.sh and anything else as a command, so a step
    reproduce.sh gains would otherwise be run as a command that does not exist."""
    reproduce = (ARTIFACT / "reproduce.sh").read_text()
    dispatch = reproduce[reproduce.rindex('case "$STEP" in'):]
    steps = set(re.findall(r"^\s+(\w+)\)\s+step_\w+", dispatch, re.M))
    entry = (ARTIFACT / "docker_entrypoint.sh").read_text()
    accepted = re.search(r"^\s+((?:\w+\|)+\w+)\)", entry, re.M)
    assert steps and accepted and set(accepted.group(1).split("|")) == steps


@pytest.mark.parametrize("name", ["docker_entrypoint.sh", "make_image.sh", "check_image.sh"])
def test_the_image_scripts_parse(name):
    """`bash -n`: a syntax error in these shows only when someone builds or starts an image."""
    r = subprocess.run(["bash", "-n", str(ARTIFACT / name)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
