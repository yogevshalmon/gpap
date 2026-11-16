"""Shared test fixtures and the `gpu` / `core` skip rule.

Circuit CNFs are generated at test time from the bundled `.bench` sources
(benchmarks/circuits/bench/), never from a user-local benchmark tree.
"""
import os

import pytest
import torch

#: Paths into the checkout are anchored here, not on the working directory, so the suite runs
#: from any directory. Test modules import these names.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.join(REPO_ROOT, "benchmarks", "circuits", "bench")
SEQ_DIR = os.path.join(BENCH_DIR, "seq")
COMB_DIR = os.path.join(BENCH_DIR, "comb")
TEST_DATA_DIR = os.path.join(REPO_ROOT, "tests", "test_data")


#: Why the OpenCL backend is present but unusable (set at collection): its tests then FAIL with
#: this message instead of skipping, so a bcp.cl that does not compile cannot pass as "no device".
_OPENCL_BROKEN = None


def _opencl_available() -> bool:
    """Can libgpap open an OpenCL device? Asked the only reliable way: by building an engine on
    that backend (which loads the OpenCL library, finds a device and compiles bcp.cl). A missing
    library, driver or device (GPAP_ERR_CUDA) means "no OpenCL here"; anything else — above all
    a compile failure (GPAP_ERR_KERNEL) — is a broken backend, recorded in _OPENCL_BROKEN."""
    global _OPENCL_BROKEN
    from gpap.engine import _core
    saved = os.environ.get("GPAP_BACKEND")
    os.environ["GPAP_BACKEND"] = "opencl"
    try:
        _core.Engine(_core.Formula.from_clauses(2, [[1, 2]]))
        return True
    except _core.GpapError as e:
        if e.status != 2:
            _OPENCL_BROKEN = str(e)
        return False
    finally:
        if saved is None:
            del os.environ["GPAP_BACKEND"]
        else:
            os.environ["GPAP_BACKEND"] = saved


def pytest_collection_modifyitems(config, items):
    """`gpu` marks a test that needs CUDA: a CUDA device for its torch tensors and a built
    libgpap. `engine` marks one that needs an engine on the backend the suite runs on
    (`$GPAP_BACKEND`) and nothing else, so under `GPAP_BACKEND=opencl` it runs on any OpenCL
    device, a CPU included. `core` marks one that needs only the library (the parser, the formula
    build, the sizing arithmetic). `opencl` marks one that runs the OpenCL backend whatever the
    suite runs on: a built libgpap and an OpenCL device. A test whose need is missing is skipped
    here, so no test file checks for itself. `cuda_only` marks a test of something only the CUDA
    backend has (the device's resident-warp count), skipped under `GPAP_BACKEND=opencl`."""
    from gpap.engine import core_available
    have_core = core_available()
    have_gpu = have_core and torch.cuda.is_available()
    on_opencl = os.environ.get("GPAP_BACKEND") == "opencl"
    want_opencl = any(item.get_closest_marker("opencl")
                      or (on_opencl and item.get_closest_marker("engine")) for item in items)
    have_opencl = have_core and want_opencl and _opencl_available()
    have_engine = have_opencl if on_opencl else have_gpu
    skip_gpu = pytest.mark.skip(reason="needs CUDA and a built libgpap (scripts/build_core.sh)")
    skip_engine = pytest.mark.skip(reason="needs a built libgpap and a device for "
                                          f"GPAP_BACKEND={'opencl' if on_opencl else 'cuda'}")
    skip_core = pytest.mark.skip(reason="needs a built libgpap (scripts/build_core.sh)")
    skip_opencl = pytest.mark.skip(reason="needs a built libgpap and an OpenCL device "
                                          "(docs/OPENCL.md)")
    skip_cuda_only = pytest.mark.skip(reason="tests what only the CUDA backend has")
    for item in items:
        if item.get_closest_marker("cuda_only") and on_opencl:
            item.add_marker(skip_cuda_only)
        if _OPENCL_BROKEN and (item.get_closest_marker("opencl")
                               or (on_opencl and item.get_closest_marker("engine"))):
            continue                     # pytest_runtest_setup fails it with the reason
        if item.get_closest_marker("gpu") and not have_gpu:
            item.add_marker(skip_gpu)
        elif item.get_closest_marker("engine") and not have_engine:
            item.add_marker(skip_engine)
        elif item.get_closest_marker("opencl") and not have_opencl:
            item.add_marker(skip_opencl)
        elif item.get_closest_marker("core") and not have_core:
            item.add_marker(skip_core)


def pytest_runtest_setup(item):
    on_opencl = os.environ.get("GPAP_BACKEND") == "opencl"
    if _OPENCL_BROKEN and (item.get_closest_marker("opencl")
                           or (on_opencl and item.get_closest_marker("engine"))):
        pytest.fail(f"the OpenCL backend is installed but unusable: {_OPENCL_BROKEN}",
                    pytrace=False)


#: What the OpenCL backend says when asked to sample: the plug and the sampler refuse the
#: trajectory kernel by name (cpp/src/gpu_opencl.cpp, cpp/src/engine.cpp), `gpap sample`
#: before it starts (gpap/cli.py).
_OPENCL_REFUSALS = ("kernel 'trajectory.", "sampling needs the cuda backend")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Run the whole suite on the OpenCL backend with `GPAP_BACKEND=opencl pytest`.

    That backend ships every kernel but the sampling trajectory, and refuses that one by name.
    Under it, a test that fails on that refusal is reported as skipped: the backend did what it
    promises. Any other failure stays a failure, and nothing here applies to the default
    backend."""
    outcome = yield
    if os.environ.get("GPAP_BACKEND") != "opencl" or outcome.excinfo is None:
        return
    if any(msg in str(outcome.excinfo[1]) for msg in _OPENCL_REFUSALS):
        outcome.force_exception(pytest.skip.Exception(
            "the opencl backend does not ship the trajectory (sampling) kernel"))


# Small sequential circuits at a few unroll depths, one deeper unroll (b03 at 25 cycles), and two
# combinational ISCAS-85 circuits with one output pinned.
_SEQ = ["b03", "b06", "b10", "s386", "s526"]
_SEQ_CYCLES = [1, 5, 10]
_COMB = ["c432", "c880"]


def _emit(out, d, stem, comb, cycles, pin):
    from gpap.circuits.bench import parse_bench, build_cnf, write_dimacs
    circ = parse_bench(os.path.join(BENCH_DIR, "comb" if comb else "seq", f"{stem}.bench"))
    b, support, _, labels = build_cnf(circ, cycles, seed=1, pin_outputs=pin)
    name = f"{stem}-c{cycles}"
    path = str(d / f"{name}.cnf")
    write_dimacs(path, b, support, labels)              # embed the wire map (c wire ...)
    out[name] = {"path": path, "nv": b.nv, "clauses": b.clauses,
                 "support": [v - 1 for v in support], "labels": labels,
                 "cycles": cycles, "comb": comb}


@pytest.fixture(scope="session")
def circuit_cnfs(tmp_path_factory):
    """The small circuit-CNF matrix, generated once per session:
    {name: {path, nv, clauses, support (0-based), labels, cycles, comb}}."""
    d = tmp_path_factory.mktemp("circuit_cnfs")
    out = {}
    for stem in _SEQ:
        for c in _SEQ_CYCLES:
            _emit(out, d, stem, False, c, None)
    for stem in _COMB:
        _emit(out, d, stem, True, 1, 1)
    _emit(out, d, "b03", False, 25, None)      # the deep unroll
    return out
