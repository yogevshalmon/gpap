"""Cross-face parity: the C++ tier-1 Sampler and the Python GPUSampler run the same core loop
(`gpap_sampler_*`) with the same RNG, so at one seed, config and batch they find the same
models, row for row.

Portfolio runs are not compared: the bandit picks configs on measured wall-clock throughput.
"""
from pathlib import Path

import pytest

from gpap.engine import _core

from tests.support import DEV, SAMPLE_BIN, need_binary, parse_dimacs, run_sample_example

pytestmark = pytest.mark.gpu


def test_cpp_and_python_faces_draw_identical_models(circuit_cnfs, tmp_path):
    """Same formula, seed, config and batch give the same model count and the same kept rows.

    The batch is pinned on both sides because K otherwise comes from live free VRAM, which can
    differ between the two processes. The rows would still match (a narrower run's rows are a
    prefix of a wider one's), but the counts would not."""
    from gpap.sampling.sampler import GPUSampler
    need_binary(SAMPLE_BIN)
    cnf = Path(circuit_cnfs["b03-c5"]["path"])
    want, seed, K = 2000, 11, 8192

    # C++ face: the example binary, single default config (random order, B=8), out file kept.
    rows_cpp, stdout = run_sample_example(cnf, tmp_path / "cpp", n=want, seed=seed, k=K)
    assert rows_cpp, "the C++ face found nothing — the comparison would be vacuous"

    # Python face: same seed, same single config, same pinned batch; rows via the delivery path.
    smp = GPUSampler.from_cnf_file(str(cnf), DEV, seed=seed)
    smp.pin_batch(K)
    r = smp.sample(want, decisions_per_round=8, random_var=True, collect=False, out_cap=want)

    nv, cls, sup = parse_dimacs(cnf)
    assert sup, "expected a `c ind` support on a circuit CNF"
    # The column order must match too.
    assert r["out_cols"] == sorted(set(sup)), \
        f"faces diverged on the projection columns: {r['out_cols'][:8]} vs {sorted(set(sup))[:8]}"

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
    """Unpinned, both faces calibrate the same K from the same core rules."""
    from gpap.sampling.sampler import GPUSampler
    cnf = circuit_cnfs["b03-c5"]["path"]
    smp = GPUSampler.from_cnf_file(cnf, DEV, seed=1)
    f = _core.Formula.from_file(cnf)
    e = _core.Engine(f)
    s = _core.Sampler(e, seed=1, k_cap=_core.default_k_cap())
    assert smp.K == s.k, "the two faces calibrated different batches from the same rules"
    assert smp.engine.mode in ("scalar", "wave")
