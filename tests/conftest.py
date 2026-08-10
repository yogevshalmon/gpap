"""Shared test fixtures.

The circuit tests must NOT depend on any user-local benchmark tree — the repo bundles the small
`.bench` sources (benchmarks/circuits/bench/), and this fixture generates the CNFs on demand into
a session tmp dir. `circuit_cnfs` is a matrix of small instances (seconds to build) that the
soundness / determinism / A-B / reconstruction / contract tests parametrise over.
"""
import os

import pytest
import torch

CUDA = torch.cuda.is_available()
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.join(REPO_ROOT, "benchmarks", "circuits", "bench")


# Sequential circuits × unroll depths (kept small for CI) + a deep-cascade c25 (slow), plus two
# combinational ISCAS-85 circuits pinned to one output (the DEMOTIC-style samplable setup).
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
    """Generate the small circuit-CNF matrix once per session. Returns
    {name: {path, nv, clauses, support(0-based), cycles, comb}}."""
    d = tmp_path_factory.mktemp("circuit_cnfs")
    out = {}
    for stem in _SEQ:
        for c in _SEQ_CYCLES:
            _emit(out, d, stem, False, c, None)
    for stem in _COMB:
        _emit(out, d, stem, True, 1, 1)
    _emit(out, d, "b03", False, 25, None)      # the deep-cascade slow instance
    return out
