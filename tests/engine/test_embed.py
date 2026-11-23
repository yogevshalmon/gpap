"""libgpap embedded in a host project the way mallob embeds it: built from source by the host's
CMake, static and without CUDA. Configures, builds and runs cpp/examples/embed in a fresh build
tree: the host's build type is left alone (configured with none, which a top-level libgpap would
turn into Release), no example is built, the library is static, OpenCL is the default backend,
and `GPAP_BACKEND=cuda` is refused with a message naming the option.
"""
import os
import shutil
import subprocess

import pytest

from tests.conftest import REPO_ROOT

EMBED = os.path.join(REPO_ROOT, "cpp", "examples", "embed")

pytestmark = pytest.mark.opencl


def _run(args, **kw):
    return subprocess.run([str(a) for a in args], capture_output=True, text=True, **kw)


def test_a_host_embeds_libgpap_without_cuda(tmp_path):
    if not shutil.which("cmake"):
        pytest.skip("needs cmake")
    build = tmp_path / "build"
    cfg = _run(["cmake", "-S", EMBED, "-B", build])
    assert cfg.returncode == 0, cfg.stdout + cfg.stderr
    made = _run(["cmake", "--build", build, "-j", os.cpu_count() or 2], timeout=900)
    assert made.returncode == 0, made.stdout[-4000:] + made.stderr[-4000:]
    env = {k: v for k, v in os.environ.items() if k != "GPAP_BACKEND"}
    run = _run([build / "embed_host"], env=env, timeout=300)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "all checks passed" in run.stdout, run.stdout
