"""The tool's Docker image: what goes into it and how it starts, checked without Docker.

  * the build context leaves out local state and every paper's data;
  * the entry point compiles libgpap for the visible GPU alone, and hands it to the loader;
  * the paths the Dockerfile runs exist;
  * the entry point parses.

Building and running the image needs Docker and a GPU; `eval/artifact/check_image.sh` does that.
Host-only: these read files and run `bash -n`.
"""
import pathlib
import re
import subprocess

from eval import paths as eval_paths
from gpap import paths
from tests.conftest import REPO_ROOT

REPO = pathlib.Path(REPO_ROOT)
ENTRYPOINT = REPO / "scripts" / "docker_entrypoint.sh"


def _code(path):
    """A shell file's lines with its comments dropped: a comment may name anything."""
    return "\n".join(line for line in path.read_text().splitlines()
                     if not line.lstrip().startswith("#"))


def test_the_dockerignore_keeps_local_state_and_paper_data_out(monkeypatch):
    """A tool image built from a clone must not carry the developer's virtualenv, built tools,
    runs or git history; one built from an artifact archive must not carry the paper's rows,
    because the tool image knows no paper. The managed names come from their owners."""
    monkeypatch.delenv("GPAP_EXPERIMENTS", raising=False)
    lines = (REPO / ".dockerignore").read_text().splitlines()
    ignored = {line.strip().rstrip("/") for line in lines
               if line.strip() and not line.lstrip().startswith("#")}
    required = {".git", ".venv", paths.DIRNAME, pathlib.Path(eval_paths.experiments_dir()).name,
                "build", "artifact-data", "MANIFEST.*", "README-ARTIFACT.md"}
    assert not required - ignored, \
        f".dockerignore lets these into the tool image: {sorted(required - ignored)}"


def test_the_entrypoint_compiles_for_the_visible_gpu_and_hands_it_to_the_loader():
    """The kernels are compiled for one architecture and the engine refuses another
    (cpp/src/runtime.cpp), so the image ships no library. The entry point cross-compiles for the
    device it sees (`--arch`, which needs no driver to compile) and exports `GPAP_BUILD_DIR`,
    the variable the build script writes to and the loader reads."""
    code = _code(ENTRYPOINT)
    assert re.search(r"/build_core\.sh\"? --arch", code), "no --arch build"
    assert "export GPAP_BUILD_DIR" in code
    assert "GPAP_BUILD_DIR" in (REPO / "scripts" / "build_core.sh").read_text()
    assert "GPAP_BUILD_DIR" in (REPO / "gpap" / "engine" / "_core.py").read_text()


def test_the_paths_the_dockerfile_runs_exist():
    """The Dockerfile copies the context to /opt/gpap and runs scripts from there; a renamed
    script breaks only the image, which no test without Docker would notice."""
    text = (REPO / "Dockerfile").read_text()
    named = set(re.findall(r"/opt/gpap/([\w./-]+\.sh)", text))
    assert ENTRYPOINT.relative_to(REPO).as_posix() in named
    missing = sorted(p for p in named if not (REPO / p).is_file())
    assert not missing, f"the Dockerfile runs files the repository does not have: {missing}"


def test_the_entrypoint_parses():
    """`bash -n`: a syntax error in an entry point shows only when someone starts the image."""
    r = subprocess.run(["bash", "-n", str(ENTRYPOINT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
