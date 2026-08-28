"""What the CLI PROMISES, pinned.

Every case here is a defect the CLI shipped with, and each one is the kind that lies quietly
rather than crashing: a flag that reports a bound it cannot apply, an output file with more rows
in it than the run produced models, a proof of UNSAT printed as "0 models, sound=True". The
engine was right in all of them — these are contracts of the print/plumbing layer, so they are
tested at that layer.
"""
import numpy as np
import pytest
import torch

from gpap.cli import main as cli_main

CUDA = torch.cuda.is_available()
try:
    from gpap.engine import core_available
    KERNEL = CUDA and core_available()
except Exception:  # noqa: BLE001
    KERNEL = False
# Both marks: `skipif` so the file self-skips off-CUDA like every other engine test, AND the
# repo-level `gpu` marker so `pytest -m "not gpu"` — the documented host-only lane — deselects
# them rather than collecting and skipping.
needs_gpu = pytest.mark.gpu(pytest.mark.skipif(not KERNEL, reason="engine is CUDA+triton-only"))


def _cnf(tmp_path, text, name="f.cnf"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


SMALL = "p cnf 4 2\n1 2 0\n-1 3 0\n"
UNSAT = "p cnf 1 2\n1 0\n-1 0\n"


# ------------------------------------------------------------------ argument validation (host)

@pytest.mark.parametrize("argv", [
    ["sample", "X", "-n", "0"],                     # ran a full chunk and reported millions
    ["sample", "X", "-n", "-5"],
    ["sample", "X", "--time-limit", "-1"],          # printed "0 models" with no cause given
    ["sample", "X", "--cycles", "0"],
    ["sample", "X", "--n-cov", "-1"],
    ["sample", "X", "--max-samples", "-1"],
    ["mine", "X", "--max-tuples", "-1"],            # meant "unbounded", warned "over budget"
    ["mine", "X", "--seconds", "-1"],
    ["sample", "X", "--time-limit", "nan"],         # nan < lo is False, so it passed every bound
    ["sample", "X", "--time-limit", "inf"],         # a cap that means "never" is not a cap
    ["mine", "X", "--seconds", "nan"],
    ["bench", "X", "--time-limit", "0"],            # 0 would ALSO be the rival's timeout
])
def test_degenerate_numbers_are_refused_at_parse_time(tmp_path, argv):
    argv = [a if a != "X" else _cnf(tmp_path, SMALL) for a in argv]
    with pytest.raises(SystemExit) as e:
        cli_main(argv)
    assert e.value.code != 0          # argparse error, not a run that quietly did something odd


def test_missing_input_and_output_dir_fail_before_any_work(tmp_path, monkeypatch):
    """...and WITHOUT a GPU. These are facts about the arguments, so the check runs ahead of the
    device precondition; stubbing CUDA away is how we pin that ordering, which is also what makes
    this test valid in the host-only lane."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as e:
        cli_main(["sample", str(tmp_path / "nope.cnf")])
    assert "does not exist" in str(e.value)
    # `mine` used to open --out only at the very end, losing a whole sweep to a typo.
    with pytest.raises(SystemExit) as e:
        cli_main(["mine", _cnf(tmp_path, SMALL), "--out", str(tmp_path / "no" / "dir" / "x.cnf")])
    assert "--out" in str(e.value) and "does not exist" in str(e.value)


# ------------------------------------------------------------------ mining budget + accounting

def test_the_default_mining_path_never_caps_width_1():
    """`mine_entailed` applies a ceiling to width 1 too — which is what `gpap mine --max-tuples`
    asks for, and a hazard everywhere else: the per-width lookup falls back to DEFAULT_MAX_TUPLES,
    so a dict without a key for width 1 silently truncates the backbone fixpoint on the DEFAULT
    sampling path, at a fraction of one round on a large unroll. No flag, no warning, and fewer
    mined units. Found by mutation: removing the `1: 0` survived the entire suite."""
    from gpap.mining import _ceiling
    from gpap.sampling.pipeline import STAGE1_MAX_TUPLES, STAGE2_MAX_TUPLES
    for name, budget in (("STAGE1", STAGE1_MAX_TUPLES), ("STAGE2", STAGE2_MAX_TUPLES)):
        assert _ceiling(budget, 1) == 0, (
            f"{name} caps width 1 at {_ceiling(budget, 1):,} — the backbone must stay unbounded; "
            f"give it an explicit `1: 0`")


@needs_gpu
def test_max_tuples_binds_at_width_1(tmp_path):
    """It was hardcoded to unbounded, while the plan line warned the ceiling was exceeded."""
    from gpap.mining import mine_entailed
    from gpap.engine import read_dimacs
    nv, cls, _ = read_dimacs(_cnf(tmp_path, SMALL))
    dev = torch.device("cuda")
    r = mine_entailed(nv, cls, dev, widths=(1,), max_tuples=2, seconds=None)
    assert r.probed[1] <= 2
    assert not r.exhausted[1]                       # a truncated sweep is NOT a proof of absence
    assert 1 in r.skipped                           # ...and says so
    full = mine_entailed(nv, cls, dev, widths=(1,), max_tuples=0, seconds=None)
    assert full.exhausted[1] and full.probed[1] > 2  # 0 still means unbounded


@needs_gpu
def test_width_1_totals_are_reconcilable_with_the_pool(tmp_path):
    """`candidates[1]` is a SUM over fixpoint rounds, so it can exceed one round's pool. That is
    only readable if the round count comes with it — it printed as a bare "of N candidates"."""
    from gpap.mining import mine_entailed
    from gpap.engine import read_dimacs
    from gpap.engine.probe import LiteralProbe
    nv, cls, _ = read_dimacs(_cnf(tmp_path, SMALL))
    dev = torch.device("cuda")
    r = mine_entailed(nv, cls, dev, widths=(1,), seconds=None)
    per_round = LiteralProbe(nv, cls, dev).count(None, 1)
    assert r.rounds[1] >= 1
    assert r.candidates[1] <= per_round * r.rounds[1]
    # and the cross-width total is gone from the summary line (it divided incomparable counts)
    assert "of" not in r.summary().split("probed")[-1].split("in")[0]


# ------------------------------------------------------------------ delivery contracts

@needs_gpu
def test_never_writes_more_samples_than_models_delivered(tmp_path, capsys):
    """The output pass used to re-sample from scratch, so a run capped at 2,990 models wrote the
    full 5,000 requested — and did it outside every reported wall."""
    out = str(tmp_path / "s.bits")
    cli_main(["sample", _cnf(tmp_path, SMALL), "-n", "300", "--out", out, "--k-cap", "64"])
    txt = capsys.readouterr().out
    rows = [ln for ln in open(out).read().splitlines() if not ln.startswith("c")]
    n_models = int(txt.split(" models in")[0].split()[-1].rstrip("<").replace(",", ""))
    assert len(rows) <= n_models
    assert len(rows) <= 300


@needs_gpu
def test_delivery_delivers_what_it_collected(tmp_path):
    """The LOWER bound, which the upper-bound test above cannot see.

    Found by mutation: silently dropping half of every collected block survived the whole suite,
    because every existing assertion was of the form `rows <= something`. A delivery path that
    quietly under-delivers is exactly as wrong as one that over-delivers, and reads as success."""
    from gpap.sampling.pipeline import sample_file
    info = sample_file(_cnf(tmp_path, SMALL), torch.device("cuda"), n_time=4000, out_cap=1500,
                       k_cap=64, seed=5, time_limit=60.0, mine=False)
    assert not info["chunk_capped"], "precondition: the run met its target, so nothing was cut"
    assert info["n_models"] >= 1500
    assert len(info["samples"]) == 1500, "asked for 1,500 deliverable rows and got fewer"


@needs_gpu
def test_delivery_bound_holds_when_a_valve_other_than_the_clock_stops_the_run(tmp_path):
    """`--time-limit` is not the only way a run ends short (max_chunks, the zero-progress stop).
    Bounding the top-up by the REMAINING TIME alone left those cases free to sample the full
    request afterwards — the original defect, reachable by another door."""
    from gpap.sampling.pipeline import sample_file
    info = sample_file(_cnf(tmp_path, SMALL), torch.device("cuda"), n_time=1_000_000,
                       out_cap=5000, k_cap=64, time_limit=30.0, max_chunks=2, mine=False)
    assert info["chunk_capped"]
    assert len(info["samples"]) <= info["n_models"]


@needs_gpu
@pytest.mark.parametrize("extra", [{}, {"uniformity": "fast"}, {"uniformity": "exact"},
                                   {"n_cov": 300}], ids=["off", "fast", "exact", "cov"])
def test_phases_account_for_the_whole_reported_wall(tmp_path, extra):
    """`phases` summed to more than `wall_total_s` by exactly the output pass, which was stamped
    into one and left out of the other — so the honest e2e rate read ~1.5x high."""
    from gpap.sampling.pipeline import sample_file
    info = sample_file(_cnf(tmp_path, SMALL), torch.device("cuda"), n_time=300, out_cap=300,
                       k_cap=64, time_limit=30.0, mine=False, **extra)
    # The opt-in READOUTS are billed to `phases_diag`, outside `wall_total_s` — parametrized
    # because the invariant first held only in the configuration the test happened to use.
    assert sum(info["phases"].values()) == pytest.approx(info["wall_total_s"], rel=0.05)


@needs_gpu
def test_emitted_rows_come_from_one_config(tmp_path):
    """Collecting during the timed run must not turn the output into a MIXTURE of the bandit's
    arms — the rows are bucketed per config and only the winner's are returned."""
    from gpap.sampling.pipeline import sample_file, CIRCUIT_PORTFOLIO
    info = sample_file(_cnf(tmp_path, SMALL), torch.device("cuda"), n_time=500, out_cap=200,
                       k_cap=64, time_limit=30.0, configs=CIRCUIT_PORTFOLIO)
    assert info["out_config"] is not None
    assert list(info["out_config"]) == [info["tuned_rv"], info["tuned_b"], info["tuned_la"]]


@needs_gpu
def test_unsat_is_reported_as_a_proof_not_as_zero_models(tmp_path, capsys):
    """An UNSAT formula printed "0 models ... sound=True", indistinguishable from a satisfiable
    instance the sampler merely failed on. Both detectors were already in hand."""
    cli_main(["sample", _cnf(tmp_path, UNSAT, "u.cnf"), "-n", "10"])
    assert "UNSAT" in capsys.readouterr().out


def test_zero_models_on_a_satisfiable_formula_says_why(tmp_path, capsys, monkeypatch):
    """0 models on a SATISFIABLE formula is a real outcome — GPAP discards conflicted trajectories
    instead of backtracking — but it printed as a bare 0 with no hint of what to change, and read
    identically to UNSAT. Driven through the print layer with a crafted result: forcing a genuine
    0%-success instance small enough for a unit test is not possible (B=1 solves the tight little
    ones), and the contract under test is what the CLI SAYS about that state, not how to reach it.
    """
    from gpap.sampling import pipeline

    def _fake(path, device, **kw):
        return {"samples_per_s": 0.0, "success_rate": 0.0, "nv": 3, "n_clauses": 4,
                "sup1": [1, 2, 3], "uniformity": None, "sps_e2e_total": 0.0, "phases": {},
                "n_models": 0, "chunk_capped": True, "wall_timed_s": 1.0, "wall_total_s": 1.0,
                "mined": None, "support_size": 3, "gpu_sound": True, "unsat": False,
                "samples": np.zeros((0, 3), np.uint8), "cols": [1, 2, 3], "wire_map": {},
                "out_requested": 0, "n_collected": 0}
    import gpap.cli as _cli
    monkeypatch.setattr(pipeline, "sample_file", _fake)      # _sample_run imports it from here
    monkeypatch.setattr(_cli, "_require_engine", lambda a: None)
    cli_main(["sample", _cnf(tmp_path, SMALL), "-n", "10"])
    out = capsys.readouterr().out
    assert "no models" in out and "--portfolio" in out
    assert "UNSAT" not in out                    # the opposite fact must not be implied


# ------------------------------------------------------------------ writers

@pytest.mark.parametrize("n,c,dtype,scale", [
    (7, 5, np.uint8, 1),
    (5000, 11, np.uint8, 1),          # crosses the block boundary
    (500, 37, np.int32, 1),           # a caller's own int array
    (300, 11, np.uint8, 255),         # NONZERO-IS-TRUE: a 0/255 mask, not a 0/1 one
])
def test_dimacs_writer_matches_the_reference_tokens(tmp_path, n, c, dtype, scale):
    """Vectorised, and it must stay token-identical to the row loop it replaced — INCLUDING that
    loop's coercion. The vectorised form indexes a length-2 table with the matrix values, so any
    value outside {0,1} became an IndexError where `row.astype(bool)` had accepted it; `!= 0`
    restores it. Found by mutation: dropping the `!= 0` survived a suite that only fed 0/1."""
    from gpap.sampling.output import write_dimacs
    rng = np.random.default_rng(n)
    m = ((rng.random((n, c)) > 0.5).astype(dtype)) * scale
    cols = sorted(rng.choice(np.arange(1, 9999), c, replace=False).tolist())
    p = str(tmp_path / "d.cnf")
    write_dimacs(p, m, cols)
    got = [ln.split() for ln in open(p)]
    want = [[str(v if b else -v) for v, b in zip(cols, row)] + ["0"] for row in m]
    assert got == want


def test_bench_forwards_every_flag_it_declares(tmp_path, monkeypatch):
    """`_cmd_bench` hand-copies flags into an argv for the driver, and that is exactly how
    `--allow-soft` went missing: declared on the parser, accepted by the driver, never passed
    between them. Asserted on the argv the driver actually receives — a source-text check would
    pass on the comment alone, and says nothing about the other flags copied the same way."""
    from gpap import benchmark
    seen = {}
    monkeypatch.setattr(benchmark, "main", lambda argv: seen.update(argv=argv))
    cnf = _cnf(tmp_path, SMALL)
    cli_main(["bench", cnf, "--allow-soft", "--twise", "--no-els", "--resume",
              "--tag", "t", "--seeds", "1", "2", "--encoders", "tseitin", "abc:fast",
              "--pin-outputs", "0", "--n-cov", "7", "--cycles", "3", "-n", "11"])
    argv = seen["argv"]
    for flag in ("--allow-soft", "--twise", "--no-els", "--resume"):
        assert flag in argv, f"{flag} declared on `gpap bench` but never forwarded"
    for flag, val in (("--tag", "t"), ("--n-cov", "7"), ("--cycles", "3"),
                      ("--n-time", "11"), ("--pin-outputs", "0")):
        assert argv[argv.index(flag) + 1] == val, flag


@needs_gpu
def test_out_cap_does_not_change_the_run(tmp_path):
    """Delivery must not perturb SAMPLING. The projection that fills `--out` draws random bits for
    don't-care columns; sharing the sampler's generator made `--out` shift every later chunk's
    permutation and seed, so the same command at the same --seed found different models."""
    from gpap.sampling.pipeline import sample_file, portfolio
    dev = torch.device("cuda")
    cnf = _cnf(tmp_path, SMALL)
    runs = [sample_file(cnf, dev, n_time=2000, out_cap=oc, k_cap=64, seed=5, time_limit=30.0,
                        configs=portfolio("b8"), mine=False) for oc in (0, 500, 2000)]
    assert len({r["n_models"] for r in runs}) == 1
    assert len({round(r["success_rate"], 12) for r in runs}) == 1


@needs_gpu
def test_emitted_rows_are_sound(tmp_path):
    """The delivery path CHANGED — rows now come off the timed run's chunks, per config, and may
    be spliced with a top-up. test_output.py gates soundness on `sample_bits`, which is no longer
    the path `gpap sample --out` takes, so the flagship claim needs a gate on this one too."""
    from pysat.formula import CNF
    from pysat.solvers import Solver
    from gpap.sampling.pipeline import sample_file
    cnf = _cnf(tmp_path, SMALL)
    info = sample_file(cnf, torch.device("cuda"), n_time=2000, out_cap=1500, k_cap=64,
                       seed=5, time_limit=30.0, mine=False)
    mat, cols = info["samples"], info["cols"]
    assert len(mat) > 0
    clauses = [[1, 2], [-1, 3]]
    s = Solver(name="cadical195", bootstrap_with=CNF(from_clauses=clauses))
    for row in mat:                      # every emitted row must EXTEND to a full model
        assert s.solve(assumptions=[v if b else -v for v, b in zip(cols, row)])
    s.delete()
