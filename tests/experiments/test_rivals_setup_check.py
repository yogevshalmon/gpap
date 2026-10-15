"""`eval/rivals/rivals_setup.sh check`, the components table `reproduce.sh check` prints. On a
machine without torch (the artifact VM, which only regenerates the tables) torch is a MISSING
row, not a traceback.
"""
import os
import subprocess
import sys

from tests.conftest import REPO_ROOT

SETUP = os.path.join(REPO_ROOT, "eval", "rivals", "rivals_setup.sh")


def test_check_without_torch_reports_a_missing_row_and_no_traceback(tmp_path):
    """A `sitecustomize` makes `import torch` fail in every Python the script starts."""
    (tmp_path / "sitecustomize.py").write_text("import sys\nsys.modules['torch'] = None\n")
    env = {**os.environ, "PYTHON": sys.executable,
           "PYTHONPATH": os.pathsep.join([str(tmp_path), str(REPO_ROOT)])}
    r = subprocess.run(["bash", SETUP, "check"], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "Traceback" not in r.stdout + r.stderr, r.stdout + r.stderr
    row = next(ln for ln in r.stdout.splitlines() if ln.startswith("torch / CUDA"))
    assert "MISSING" in row
