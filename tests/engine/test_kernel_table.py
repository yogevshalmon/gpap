"""The AOT kernel table: no duplicate cubins, and every key the engine can ask for exists.

The table is generated (cpp/tools/build_kernels.py -> cpp/src/generated/gpap_kernels.c) and
gitignored, so these tests skip on an unbuilt tree.
"""
import os
import re

import pytest

from tests.conftest import REPO_ROOT

REPO = REPO_ROOT
GENERATED = os.path.join(REPO, "cpp", "src", "generated", "gpap_kernels.c")


def _kernel_table():
    """{key: cubin bytes} parsed out of the generated C file."""
    if not os.path.exists(GENERATED):
        pytest.skip("kernels not generated (run cpp/tools/build_kernels.py)")
    src = open(GENERATED).read()
    blobs = {}
    for m in re.finditer(r"static const unsigned char (k\d+)_cubin\[\d+\] =\s*(.*?);\n", src, re.S):
        data = bytearray()
        for lit in re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(2)):
            i = 0
            while i < len(lit):
                if lit[i] == "\\":
                    data.append(int(lit[i + 1:i + 4], 8))
                    i += 4
                else:
                    data.append(ord(lit[i]))
                    i += 1
        blobs[m.group(1)] = bytes(data)
    return {m.group(1): blobs[m.group(2)]
            for m in re.finditer(r'\{\s*"([^"]+)",\s*"[^"]+",\s*(k\d+)_cubin', src)}


def test_no_two_keys_share_a_cubin():
    """A duplicate blob is a specialization on a constant the kernel never reads (e.g. VSCAN,
    read only by the wave, non-soft trajectory). The rule is build_kernels.vscan_is_read,
    mirrored by engine.cpp::traj_key."""
    table = _kernel_table()
    by_blob = {}
    for key, blob in table.items():
        by_blob.setdefault(blob, []).append(key)
    dups = [sorted(keys) for keys in by_blob.values() if len(keys) > 1]
    assert not dups, f"duplicate cubins shipped: {dups}"


def test_the_trajectory_matrix_matches_the_kernel_guard():
    """The table on disk is what the current `kernel_specs()` emits.

    `vscan_is_read` is imported, not restated, so the test cannot agree with itself by
    construction. The engine's side is `test_every_reachable_kernel_key_resolves`."""
    pytest.importorskip("triton")                 # kernel_specs() is defined beside the compiler
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gpap_build_kernels", os.path.join(REPO, "cpp", "tools", "build_kernels.py"))
    bk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bk)

    shipped = set(_kernel_table())
    expected = {key for key, *_ in bk.kernel_specs()}
    assert shipped == expected, (
        f"the generated table is not what build_kernels.py emits "
        f"(missing {sorted(expected - shipped)}, extra {sorted(shipped - expected)}) — rebuild "
        f"with python cpp/tools/build_kernels.py")
    for key in (k for k in shipped if k.startswith("trajectory.")):
        _, _, wave, vscan, soft = key.split(".")
        assert vscan == "v0" or bk.vscan_is_read(wave == "w1", soft == "f1"), \
            f"{key} specializes on VSCAN, which that kernel never reads"


def _sass_counts(blob, tmp_path, key):
    """How often a WAVE row's ordering instructions occur in one cubin's machine code: the
    GPU-wide fence (MEMBAR), the L1 flush that comes with an acquire (CCTL.IVALL) and the
    program's barrier (BAR.SYNC). cuobjdump ships with triton, beside ptxas."""
    import subprocess
    triton = pytest.importorskip("triton")
    tool = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin",
                        "cuobjdump")
    if not os.path.exists(tool):
        pytest.skip("this triton ships no cuobjdump")
    path = tmp_path / f"{key}.cubin"
    path.write_bytes(blob)
    sass = subprocess.run([tool, "-sass", str(path)], capture_output=True, text=True,
                          check=True).stdout
    found = re.findall(r"\b(MEMBAR|CCTL\.IVALL|BAR\.SYNC)", sass)
    return {op: found.count(op) for op in ("MEMBAR", "CCTL.IVALL", "BAR.SYNC")}


def test_wave_atomics_are_relaxed_and_ordered_by_barriers(tmp_path):
    """Every WAVE kernel has no fence its scalar twin lacks, and has the two barriers.

    A row's atomics are relaxed (bcp.py's docstring, "Memory ordering under WAVE"): Triton's
    default fences each one GPU-wide and flushes the L1. The scalar twin has no row atomics, so
    whatever fence it has is the job counter's, which keeps its fence. The barriers in
    `_wave_propagate`, one before every wave and one after the last, are what order a row's
    threads instead. A dropped barrier passes the identity tests on most runs, so it is
    counted here. Fails if an atomic loses `sem="relaxed"` or a barrier goes."""
    table = _kernel_table()
    checked = 0
    for key, blob in sorted(table.items()):
        if ".w1" not in key:
            continue
        twin = key.replace(".w1", ".w0").replace(".v1", ".v0")
        wave, scalar = (_sass_counts(table[k], tmp_path, k) for k in (key, twin))
        for op in ("MEMBAR", "CCTL.IVALL"):
            assert wave[op] == scalar[op], f"{key}: {wave[op]} {op}, its scalar twin {scalar[op]}"
        assert wave["BAR.SYNC"] >= scalar["BAR.SYNC"] + 2, \
            f"{key}: {wave['BAR.SYNC']} barriers, its scalar twin {scalar['BAR.SYNC']}"
        checked += 1
    assert checked == 12, f"checked {checked} WAVE kernels, expected 12"


@pytest.mark.gpu
def test_every_reachable_kernel_key_resolves():
    """The engine never asks for a cubin the build did not emit; a missing key would fail
    with GPAP_ERR_KERNEL at launch."""
    import numpy as np

    from gpap.engine import _core

    rng = np.random.default_rng(0)
    clauses = [[int(v) * (1 if rng.random() < 0.5 else -1)
                for v in rng.choice(200, 3, replace=False) + 1] for _ in range(4000)]
    formula = _core.Formula.from_clauses(200, clauses)
    order = np.arange(200, dtype=np.int32)
    launched = 0
    for mode in ("scalar", "wave"):
        for vscan in (0, 1, -1):
            for soft in (False, True):
                engine = _core.Engine(formula, mode=mode, vscan=vscan)
                engine.reset(1024)
                engine.trajectory(order, 200, 8, 7, soft)     # raises if the key is missing
                launched += 1
    # Not raising is the check, so count the launches to show the loop ran.
    assert launched == 12, f"only {launched} of the 12 reachable keys were launched"
