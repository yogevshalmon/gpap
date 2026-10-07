"""Helpers shared across the test suite: random formulas, exact oracles, and the C++ examples.

Literals are DIMACS (`+v` / `-v`, 1-based) unless a name says "index" (the engine's `2(v-1)`
encoding, `tests.reference.bcp.lit_idx`).
"""
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.conftest import REPO_ROOT
from tests.reference.bcp import lit_idx

#: Every engine test runs on the GPU; `gpu`-marked tests are skipped without one (conftest.py).
DEV = torch.device("cuda")


# ---------------------------------------------------------------- random formulas

def random_cnf(rng, nv, nc, kmin=1, kmax=4):
    """`nc` clauses of kmin..kmax distinct variables with random polarity; `rng` is a
    `random.Random`.

    `kmin` decides what the formula is good for. At kmin=1 a formula of any size almost always
    holds a contradictory pair of unit clauses, so it is UNSAT before a single decision. kmin=2
    gives real implication chains without that, and kmin=3 at a low clause ratio leaves
    solutions to find."""
    out = []
    for _ in range(nc):
        vs = rng.sample(range(1, nv + 1), rng.randint(min(kmin, nv), min(kmax, nv)))
        out.append([v if rng.random() < 0.5 else -v for v in vs])
    return out


def random_assumptions(rng, nv, n, wmin=0, wmax=4):
    """`n` assumption sets of wmin..wmax distinct variables, random polarity."""
    return [[v if rng.random() < 0.5 else -v
             for v in rng.sample(range(1, nv + 1), rng.randint(min(wmin, nv), min(wmax, nv)))]
            for _ in range(n)]


def decision_matrix(sets):
    """Assumption sets -> the engine's (n, L) int64 literal-index matrix, padded with -1."""
    L = max(1, max((len(s) for s in sets), default=1))
    d = np.full((len(sets), L), -1, dtype=np.int64)
    for i, s in enumerate(sets):
        for j, lit in enumerate(s):
            d[i, j] = lit_idx(lit)
    return d


# ---------------------------------------------------------------- exact oracles

def enumerate_projections(clauses, support, limit=None):
    """Every assignment to `support` (1-based variables) that extends to a model, as 0/1
    tuples ordered like `support`. Exact: CaDiCaL with blocking clauses, until UNSAT or
    `limit` projections."""
    from pysat.formula import CNF
    from pysat.solvers import Solver
    out = set()
    with Solver(name="cadical195", bootstrap_with=CNF(from_clauses=clauses)) as s:
        while (limit is None or len(out) < limit) and s.solve():
            m = set(s.get_model())
            out.add(tuple(1 if v in m else 0 for v in support))
            s.add_clause([-v if v in m else v for v in support])
    return out


def rows_satisfy(bits, clauses):
    """Does every row satisfy every clause? Row column j is variable j+1 (1 = TRUE)."""
    return all(any((row[abs(lit) - 1] == 1) == (lit > 0) for lit in cl)
               for row in bits for cl in clauses)


# ---------------------------------------------------------------- the C++ example binaries

def example_binary(name):
    """An example binary in whichever build tree holds it: `$GPAP_BUILD_DIR`, `build/` or
    `cpp/build/`, the same trees `_core.py` searches for the library."""
    trees = [Path(os.environ["GPAP_BUILD_DIR"])] if os.environ.get("GPAP_BUILD_DIR") else []
    trees += [Path(REPO_ROOT) / "build", Path(REPO_ROOT) / "cpp" / "build"]
    return next((t / name for t in trees if (t / name).exists()), trees[0] / name)


SAMPLE_BIN = example_binary("gpap_sample")


def need_binary(binary):
    if not binary.exists():
        pytest.skip(f"{binary.name} not built (scripts/build_core.sh)")


def parse_dimacs(path):
    """-> (num_vars, clauses, the `c ind` support or None). Independent of the engine's parser
    on purpose: the C++ tests check the engine against it."""
    clauses, support, nv = [], [], None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("c ind"):
            support += [int(t) for t in line[5:].split() if t not in ("", "0")]
        elif line.startswith("p cnf"):
            nv = int(line.split()[2])
        elif line and not line.startswith(("c", "p", "%")):
            lits = [int(t) for t in line.split() if t != "0"]
            if lits:
                clauses.append(lits)
    return nv, clauses, (support or None)


def run_sample_example(cnf, workdir, n=2000, seed=7, k=0):
    """Run `gpap_sample`; -> (solution rows as DIMACS literals, its stdout).

    `k` pins the batch. Unpinned it is sized from live free VRAM, which is not part of the seed,
    so a count compared across processes needs both sides pinned."""
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / "sols.txt"
    cmd = [str(SAMPLE_BIN), str(cnf), str(n), str(seed), str(out)]
    if k:
        cmd += ["--k", str(k)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"gpap_sample failed:\n{r.stdout}\n{r.stderr}"
    rows = [[int(t) for t in ln.split()] for ln in out.read_text().splitlines() if ln.strip()]
    return rows, r.stdout
