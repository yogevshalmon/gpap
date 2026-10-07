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


def pytest_collection_modifyitems(config, items):
    """`gpu` marks a test that needs the engine: a CUDA device and a built libgpap. `core` marks
    one that needs only the library (the parser, the formula build, the sizing arithmetic), which
    loads wherever the NVIDIA driver does, with or without a visible GPU. A test whose need is
    missing is skipped here, so no test file checks for itself."""
    from gpap.engine import core_available
    have_core = core_available()
    have_gpu = have_core and torch.cuda.is_available()
    skip_gpu = pytest.mark.skip(reason="needs CUDA and a built libgpap (scripts/build_core.sh)")
    skip_core = pytest.mark.skip(reason="needs a built libgpap (scripts/build_core.sh)")
    for item in items:
        if item.get_closest_marker("gpu") and not have_gpu:
            item.add_marker(skip_gpu)
        elif item.get_closest_marker("core") and not have_core:
            item.add_marker(skip_core)


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
