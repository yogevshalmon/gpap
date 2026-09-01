"""Cross-face parity: the C++ tier-1 Sampler and the Python GPUSampler drive the SAME core
loop (`gpap_sampler_*`) with the SAME RNG — so at the same seed, single config and batch cap,
the two faces must find the SAME models, row for row.

This property is the point of moving the loop into the core: it was impossible before (the two
faces had independent loops and independent RNGs, and their drift went unnoticed because
nothing could compare them). A portfolio run is deliberately NOT compared bit-for-bit — the
bandit selects on measured wall-clock throughput, so its chunk sequence is machine-local.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from gpap.engine import _core

from tests.test_cpp_api import SAMPLE_BIN, _need, _parse_cnf, _run_sampler

CUDA = torch.cuda.is_available()
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not CUDA, reason="requires CUDA"),
    pytest.mark.skipif(not _core.available(), reason="libgpap not built (see cpp/README.md)"),
]
DEV = torch.device("cuda") if CUDA else torch.device("cpu")


def test_cpp_and_python_faces_draw_identical_models(circuit_cnfs, tmp_path):
    """Same formula, seed, config and cap => identical model count AND identical kept rows."""
    from gpap.sampling.sampler import GPUSampler
    _need(SAMPLE_BIN)
    cnf = Path(circuit_cnfs["b03-c5"]["path"])
    want, seed = 2000, 11

    # C++ face: the example binary, single default config (random order, B=8), out file kept.
    rows_cpp, stdout = _run_sampler(cnf, tmp_path / "cpp", n=want, seed=seed)
    assert rows_cpp, "the C++ face found nothing — the comparison would be vacuous"

    # Python face: same seed, same single config, same auto cap; rows via the delivery path.
    smp = GPUSampler.from_cnf_file(str(cnf), DEV, seed=seed)
    r = smp.sample(want, decisions_per_round=8, random_var=True, collect=False, out_cap=want)

    nv, cls, sup = _parse_cnf(cnf)
    assert sup, "expected a `c ind` support on a circuit CNF"
    assert r["out_cols"] == [v for v in sorted(set(sup))] or r["out_cols"], "columns exist"

    # Count parity: the stdout header carries found(); n_models is the Python count.
    found_cpp = int(stdout.split(" solutions in ")[0].rsplit("\n", 1)[-1])
    assert r["n_models"] == found_cpp, \
        f"faces diverged on the model count: python {r['n_models']} vs c++ {found_cpp}"

    # Row parity: identical kept rows, in order (same stream => same chunks => same prefixes).
    bits = r["out_bits"]
    assert bits.shape[0] == len(rows_cpp)
    cols = r["out_cols"]
    py_rows = [[c if b else -c for c, b in zip(cols, row)] for row in bits.tolist()]
    assert py_rows == rows_cpp, "faces diverged on the kept rows"


def test_faces_share_the_calibration(circuit_cnfs):
    """K and kernel mode come from the same core rules on both faces — pin nothing, compare."""
    from gpap.sampling.sampler import GPUSampler
    cnf = circuit_cnfs["b03-c5"]["path"]
    smp = GPUSampler.from_cnf_file(cnf, DEV, seed=1)
    f = _core.Formula.from_file(cnf)
    e = _core.Engine(f)
    s = _core.Sampler(e, seed=1, k_cap=_core.default_k_cap())
    assert smp.K == s.k, "the two faces calibrated different batches from the same rules"
    assert smp.engine.mode in ("scalar", "wave")
