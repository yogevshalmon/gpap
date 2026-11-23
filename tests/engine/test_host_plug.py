"""A host's own GPU plug (`gpap::engine_create_on`, cpp/include/gpap/gpu_api.hpp), checked by
running the gpap_host_plug example: a plug that forwards to GPAP's own plug for $GPAP_BACKEND
and records the calls. It checks that an engine on it mines what `gpap_engine_create`'s engine
mines, reports the plug's backend name, makes the plug current before allocating, chooses its
kernel from the plug's `resident_warps()`, and that a refused create frees the plug exactly once.
Skipped unless the examples are built (scripts/build_core.sh).
"""
import os
import subprocess

import pytest

from tests.conftest import REPO_ROOT
from tests.support import example_binary, need_binary

HOST_PLUG_BIN = example_binary("gpap_host_plug")

pytestmark = pytest.mark.engine


@pytest.mark.parametrize("cnf", [None, "unit_prop_chain.cnf", "conflict_2v.cnf"])
def test_an_engine_on_a_host_plug_keeps_the_plug_contract(cnf):
    need_binary(HOST_PLUG_BIN)
    args = [str(HOST_PLUG_BIN)]
    if cnf:
        args.append(os.path.join(REPO_ROOT, "tests", "test_data", cnf))
    run = subprocess.run(args, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "all checks passed" in run.stdout, run.stdout
