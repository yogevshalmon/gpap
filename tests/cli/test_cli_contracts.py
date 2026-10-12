"""Contracts of the CLI's print and plumbing layer: argument validation, mining budgets,
delivery and accounting, UNSAT reporting, the soundness tri-state, writers and defaults shared
across faces. Each guards a failure that would mislead rather than crash.
"""
import numpy as np
import pytest
import torch

from gpap.cli import main as cli_main
from tests.conftest import REPO_ROOT


def _cnf(tmp_path, text, name="f.cnf"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


SMALL = "p cnf 4 2\n1 2 0\n-1 3 0\n"
UNSAT = "p cnf 1 2\n1 0\n-1 0\n"


# ------------------------------------------------------------------ argument validation (host)

@pytest.mark.parametrize("argv", [
    ["sample", "X", "-n", "0"],
    ["sample", "X", "-n", "-5"],
    ["sample", "X", "--time-limit", "-1"],
    ["sample", "X", "--cycles", "0"],
    ["sample", "X", "--n-cov", "-1"],
    ["sample", "X", "--max-samples", "-1"],
    ["mine", "X", "--max-tuples", "-1"],
    ["mine", "X", "--seconds", "-1"],
    ["sample", "X", "--time-limit", "nan"],         # nan < lo is False, so it passes any bound
    ["sample", "X", "--time-limit", "inf"],         # a cap that means "never" is not a cap
    ["mine", "X", "--seconds", "nan"],
    ["bench", "X", "--time-limit", "0"],            # 0 would also be the rival's timeout
])
def test_degenerate_numbers_are_refused_at_parse_time(tmp_path, argv):
    argv = [a if a != "X" else _cnf(tmp_path, SMALL) for a in argv]
    with pytest.raises(SystemExit) as e:
        cli_main(argv)
    assert e.value.code != 0          # an argparse error, not a run


def test_missing_input_and_output_dir_fail_before_any_work(tmp_path, monkeypatch):
    """A missing input or output directory is refused before the device check; CUDA is stubbed
    away to pin that order."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as e:
        cli_main(["sample", str(tmp_path / "nope.cnf")])
    assert "does not exist" in str(e.value)
    # `mine` checks its --out directory before the sweep, not after.
    with pytest.raises(SystemExit) as e:
        cli_main(["mine", _cnf(tmp_path, SMALL), "--out", str(tmp_path / "no" / "dir" / "x.cnf")])
    assert "--out" in str(e.value) and "does not exist" in str(e.value)


# ------------------------------------------------------------------ mining budget + accounting

def test_the_default_mining_path_never_caps_width_1():
    """The pipeline's mining budgets leave width 1 unbounded. A budget dict without an explicit
    `1: 0` falls back to DEFAULT_MAX_TUPLES and silently truncates the backbone fixpoint."""
    from gpap.mining import _ceiling
    from gpap.sampling.pipeline import STAGE1_MAX_TUPLES, STAGE2_MAX_TUPLES
    for name, budget in (("STAGE1", STAGE1_MAX_TUPLES), ("STAGE2", STAGE2_MAX_TUPLES)):
        assert _ceiling(budget, 1) == 0, (
            f"{name} caps width 1 at {_ceiling(budget, 1):,} — the backbone must stay unbounded; "
            f"give it an explicit `1: 0`")


@pytest.mark.gpu
def test_max_tuples_binds_at_width_1(tmp_path):
    """`max_tuples` binds at width 1, and a truncated sweep says so."""
    from gpap.mining import mine_entailed
    from gpap.engine import read_dimacs
    nv, cls, _ = read_dimacs(_cnf(tmp_path, SMALL))
    dev = torch.device("cuda")
    r = mine_entailed(nv, cls, dev, widths=(1,), max_tuples=2, seconds=None)
    assert r.probed[1] <= 2
    assert not r.exhausted[1]                       # a truncated sweep is not a proof of absence
    assert 1 in r.skipped                           # ...and says so
    full = mine_entailed(nv, cls, dev, widths=(1,), max_tuples=0, seconds=None)
    assert full.exhausted[1] and full.probed[1] > 2  # 0 still means unbounded


@pytest.mark.gpu
def test_width_1_totals_are_reconcilable_with_the_pool(tmp_path):
    """`candidates[1]` sums over fixpoint rounds, so it can exceed one round's pool; the round
    count is reported with it."""
    from gpap.mining import mine_entailed
    from gpap.engine import read_dimacs
    from gpap.engine.probe import LiteralProbe
    nv, cls, _ = read_dimacs(_cnf(tmp_path, SMALL))
    dev = torch.device("cuda")
    r = mine_entailed(nv, cls, dev, widths=(1,), seconds=None)
    per_round = LiteralProbe(nv, cls, dev).count(None, 1)
    assert r.rounds[1] >= 1
    assert r.candidates[1] <= per_round * r.rounds[1]
    # the summary line has no cross-width total (it would mix incomparable counts)
    assert "of" not in r.summary().split("probed")[-1].split("in")[0]


# ------------------------------------------------------------------ delivery contracts

@pytest.mark.gpu
def test_never_writes_more_samples_than_models_delivered(tmp_path, capsys):
    """The output file holds no more rows than the run delivered models, or than -n."""
    out = str(tmp_path / "s.bits")
    cli_main(["sample", _cnf(tmp_path, SMALL), "-n", "300", "--out", out, "--k-cap", "64"])
    txt = capsys.readouterr().out
    rows = [ln for ln in open(out).read().splitlines() if not ln.startswith("c")]
    n_models = int(txt.split(" models in")[0].split()[-1].rstrip("<").replace(",", ""))
    assert len(rows) <= n_models
    assert len(rows) <= 300


@pytest.mark.gpu
def test_delivery_delivers_what_it_collected(tmp_path):
    """The lower bound: a run that met its target delivers every requested row. The upper-bound
    test above cannot see under-delivery."""
    from gpap.sampling.pipeline import sample_file
    info = sample_file(_cnf(tmp_path, SMALL), torch.device("cuda"), n_time=4000, out_cap=1500,
                       k_cap=64, seed=5, time_limit=60.0, mine=False)
    assert not info["chunk_capped"], "precondition: the run met its target, so nothing was cut"
    assert info["stop"] == "target", "a run that met its target must say so, not merely not-capped"
    assert info["n_models"] >= 1500
    assert len(info["samples"]) == 1500, "asked for 1,500 deliverable rows and got fewer"


@pytest.mark.gpu
def test_delivery_bound_holds_when_a_valve_other_than_the_clock_stops_the_run(tmp_path):
    """The delivery bound also holds when a valve other than --time-limit ends the run
    (max_chunks, the zero-progress stop).

    And the row says WHICH valve. `chunk_capped` is one boolean over four outcomes -- spent the
    clock, hit the chunk ceiling, gave up on zero yield -- which read as one fact and are not:
    the first is a property of the instance and the last of our stopping rule. The pair of
    asserts here and in the target test above is what makes the field discriminate rather than
    merely exist."""
    from gpap.sampling.pipeline import sample_file
    info = sample_file(_cnf(tmp_path, SMALL), torch.device("cuda"), n_time=1_000_000,
                       out_cap=5000, k_cap=64, time_limit=30.0, max_chunks=2, mine=False)
    assert info["chunk_capped"]
    assert info["stop"] == "chunks", \
        f"the chunk ceiling ended this run, not the clock; row says {info['stop']!r}"
    assert len(info["samples"]) <= info["n_models"]


@pytest.mark.gpu
@pytest.mark.parametrize("extra", [{}, {"n_cov": 300}, {"verify_sound": True}],
                         ids=["off", "cov", "verify"])
def test_phases_account_for_the_whole_reported_wall(tmp_path, extra):
    """`phases` sums to `wall_total_s` with and without the opt-in readouts (--n-cov,
    --verify-sound), which are billed to `phases_diag` instead."""
    from gpap.sampling.pipeline import sample_file
    info = sample_file(_cnf(tmp_path, SMALL), torch.device("cuda"), n_time=300, out_cap=300,
                       k_cap=64, time_limit=30.0, mine=False, **extra)
    assert sum(info["phases"].values()) == pytest.approx(info["wall_total_s"], rel=0.05)
    # The readout must show up in phases_diag: billing it nowhere would also pass the sum above.
    if extra.get("verify_sound"):
        assert info["phases_diag"]["verify_sound"] == pytest.approx(info["verify_sound_s"])


@pytest.mark.gpu
def test_emitted_rows_come_from_one_config(tmp_path):
    """Rows collected during the timed run come from one config (the winner's), never a mixture
    of the bandit's arms."""
    from gpap.sampling.pipeline import sample_file, CIRCUIT_PORTFOLIO
    info = sample_file(_cnf(tmp_path, SMALL), torch.device("cuda"), n_time=500, out_cap=200,
                       k_cap=64, time_limit=30.0, configs=CIRCUIT_PORTFOLIO)
    assert info["out_config"] is not None
    assert list(info["out_config"]) == [info["tuned_rv"], info["tuned_b"], info["tuned_la"]]


@pytest.mark.gpu
def test_unsat_is_reported_as_a_proof_not_as_zero_models(tmp_path, capsys):
    """An UNSAT formula is reported as UNSAT, not as "0 models"."""
    cli_main(["sample", _cnf(tmp_path, UNSAT, "u.cnf"), "-n", "10"])
    out = capsys.readouterr().out
    # Once: construction proves it here, and the CLI's second site (UNSAT found by mining) must
    # not also fire.
    assert out.count("UNSAT") == 1
    # Not marked '<' (cut short of -n): 0 models is the complete answer, so `chunk_capped`
    # excludes it.
    assert "0<" not in out


def test_zero_models_on_a_satisfiable_formula_says_why(tmp_path, capsys, monkeypatch):
    """0 models on a satisfiable formula (GPAP discards conflicted trajectories) prints a hint
    and does not read as UNSAT. Uses a stubbed result: no unit-test-sized instance reliably gets
    0% success."""
    from gpap.sampling import pipeline

    def _fake(path, device, **kw):
        return {"samples_per_s": 0.0, "success_rate": 0.0, "nv": 3, "n_clauses": 4,
                "sup1": [1, 2, 3], "sps_e2e_total": 0.0, "phases": {},
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


# --------------------------------------------------------------- the order the run happened in

@pytest.mark.gpu
def test_each_phase_is_printed_where_it_happened(tmp_path, capsys):
    """The formula and mining lines print before the throughput line, in the order they happen
    (the pipeline reports them through `progress=`)."""
    cli_main(["sample", _cnf(tmp_path, SMALL), "-n", "10", "--mine", "on"])
    out = capsys.readouterr().out.splitlines()
    where = {k: next(i for i, ln in enumerate(out) if k in ln)
             for k in ("N=4 M=2", "mined", "models in")}
    assert where["N=4 M=2"] < where["mined"] < where["models in"], out


@pytest.mark.gpu
def test_a_formula_only_mining_refutes_is_reported_as_unsat(tmp_path, capsys):
    """Every 3-literal clause over 3 variables: construction cannot refute it, but mining fails
    every literal, so UNSAT is known only after mining. The real-pipeline half of the stubbed
    test below."""
    cnf = "p cnf 3 8\n" + "".join(f"{a} {b} {c} 0\n"
                                  for a in (1, -1) for b in (2, -2) for c in (3, -3))
    cli_main(["sample", _cnf(tmp_path, cnf, "u3.cnf"), "-n", "10", "--mine", "on"])
    out = capsys.readouterr().out
    assert "UNSAT" in out, out
    assert out.count("UNSAT") == 1, out
    assert "0<" not in out, out             # nothing was cut short; there was nothing to find


def test_unsat_proved_after_construction_is_still_printed(tmp_path, capsys, monkeypatch):
    """UNSAT proved after construction (by mining) is still printed, once. Stubbed so that
    construction reports satisfiable and the run reports UNSAT."""
    from gpap.sampling import pipeline

    def _fake(path, device, progress=None, **kw):
        progress("formula", {"nv": 3, "n_clauses": 8, "n_support": 3, "support_size": 3,
                             "unsat": False})                      # construction proved nothing
        return {"samples_per_s": 0.0, "success_rate": 0.0, "nv": 3, "n_clauses": 8,
                "sup1": [1, 2, 3], "sps_e2e_total": 0.0, "phases": {},
                "n_models": 0, "chunk_capped": False, "wall_timed_s": 1.0, "wall_total_s": 1.0,
                "mined": None, "support_size": 3, "unsat": True,   # ...mining did
                "samples": np.zeros((0, 3), np.uint8), "cols": [1, 2, 3], "wire_map": {},
                "out_requested": 0, "n_collected": 0}
    import gpap.cli as _cli
    monkeypatch.setattr(pipeline, "sample_file", _fake)
    monkeypatch.setattr(_cli, "_require_engine", lambda a: None)
    cli_main(["sample", _cnf(tmp_path, SMALL), "-n", "10"])
    out = capsys.readouterr().out
    assert out.count("UNSAT") == 1               # said once, by the site that learned it


# ------------------------------------------------------------- the soundness tri-state
# `pipeline.sound_verdict` reads a row's verdict and `merge_sound_check` folds two. Every
# consumer (bench table, CLI, Colab driver, rival harnesses, paper_tables) imports them.

def test_the_three_states_are_read_back_verbatim():
    from gpap.sampling.pipeline import SOUND_BAD, SOUND_OFF, SOUND_OK, sound_verdict
    for state in (SOUND_OK, SOUND_BAD, SOUND_OFF):
        assert sound_verdict({"sound_check": state}) == state


def test_a_row_with_no_verdict_at_all_reads_off_not_ok():
    """A missing or unrecognised verdict reads as `off`: "not run" is not "passed"."""
    from gpap.sampling.pipeline import SOUND_OFF, sound_verdict
    assert sound_verdict({}) == SOUND_OFF
    assert sound_verdict({"sound_check": "yes"}) == SOUND_OFF     # junk is not a verdict
    assert sound_verdict({"sound_check": None}) == SOUND_OFF


def test_the_legacy_bool_is_a_verdict_in_both_directions():
    """Legacy rows carry `gpu_sound: bool` from a check that always ran, so the bool is the
    verdict, False included."""
    from gpap.sampling.pipeline import SOUND_BAD, SOUND_OK, sound_verdict
    assert sound_verdict({"gpu_sound": True}) == SOUND_OK
    assert sound_verdict({"gpu_sound": False}) == SOUND_BAD
    # a row with both keys is a new row; the new key wins
    assert sound_verdict({"sound_check": SOUND_OK, "gpu_sound": False}) == SOUND_OK


def test_unsound_dominates_the_merge():
    """Folding the timed and coverage verdicts: UNSOUND dominates, then evidence beats absence."""
    from gpap.sampling.pipeline import SOUND_BAD, SOUND_OFF, SOUND_OK, merge_sound_check
    assert merge_sound_check(SOUND_OK, SOUND_BAD) == SOUND_BAD
    assert merge_sound_check(SOUND_BAD, SOUND_OK) == SOUND_BAD    # order must not matter
    assert merge_sound_check(SOUND_OFF, SOUND_BAD) == SOUND_BAD
    # ...and evidence beats absence, in both orders
    assert merge_sound_check(SOUND_OK, SOUND_OFF) == SOUND_OK
    assert merge_sound_check(SOUND_OFF, SOUND_OK) == SOUND_OK
    # nothing ran, or nothing was passed at all
    assert merge_sound_check(SOUND_OFF, SOUND_OFF) == SOUND_OFF
    assert merge_sound_check() == SOUND_OFF
    assert merge_sound_check(None, "") == SOUND_OFF


@pytest.mark.gpu
def test_verify_requested_but_never_reached_reads_off(tmp_path):
    """`--verify-sound` on a run that completes nothing reports "off", not "ok": the check runs
    over completed rows, so it never ran, and `sound_rows_checked` is 0."""
    from gpap.sampling.sampler import GPUSampler
    smp = GPUSampler.from_clauses(1, [[1], [-1]], torch.device("cuda"), k_cap=16, seed=1)
    out = smp.sample(10, max_chunks=5, collect=False, verify_sound=True)
    assert out["n_models"] == 0, "fixture: this run must complete nothing"
    assert out["sound_rows_checked"] == 0
    assert out["sound_check"] == "off", "an un-run check must not be published as a passing one"


# ------------------------------------------------------------------ writers

@pytest.mark.parametrize("n,c,dtype,scale", [
    (7, 5, np.uint8, 1),
    (5000, 11, np.uint8, 1),          # crosses the block boundary
    (500, 37, np.int32, 1),           # a caller's own int array
    (300, 11, np.uint8, 255),         # nonzero is true: a 0/255 mask
])
def test_dimacs_writer_matches_the_reference_tokens(tmp_path, n, c, dtype, scale):
    """The vectorised writer is token-identical to a per-row loop, including reading any nonzero
    value as true (it indexes a length-2 table, so values outside {0,1} need the `!= 0`)."""
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
    """`gpap bench` passes every flag it declares through to the driver. `_cmd_bench` copies
    them into an argv by hand, so the check is on the argv the driver receives."""
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


@pytest.mark.gpu
def test_out_cap_does_not_change_the_run(tmp_path):
    """`--out` does not change which models a seed finds: the projection draws its don't-care
    bits from its own generator, not the sampler's."""
    from gpap.sampling.pipeline import sample_file, portfolio
    dev = torch.device("cuda")
    cnf = _cnf(tmp_path, SMALL)
    runs = [sample_file(cnf, dev, n_time=2000, out_cap=oc, k_cap=64, seed=5, time_limit=30.0,
                        configs=portfolio("b8"), mine=False) for oc in (0, 500, 2000)]
    assert len({r["n_models"] for r in runs}) == 1
    assert len({round(r["success_rate"], 12) for r in runs}) == 1


@pytest.mark.gpu
def test_emitted_rows_are_sound(tmp_path):
    """Every row on the `gpap sample --out` path (timed-run chunks, per config) extends to a full
    model. test_output.py covers `sample_bits`, which is a different path."""
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
    for row in mat:                      # every emitted row must extend to a full model
        assert s.solve(assumptions=[v if b else -v for v, b in zip(cols, row)])
    s.delete()


def test_the_two_faces_agree_on_the_decision_batch_default():
    """Every face has the same default decision batch B, which sets both throughput and
    uniformity. The values are independent literals, so only a test keeps them equal. The C++
    default is read from the header text: no build needed, and it is what C++ callers compile."""
    import inspect
    import re
    from pathlib import Path

    from gpap.sampling.pipeline import DEFAULT_PORTFOLIO, sample_clauses
    from gpap.sampling.sampler import GPUSampler

    def default(fn, name):
        return inspect.signature(fn).parameters[name].default

    hpp = (Path(REPO_ROOT) / "cpp/include/gpap/gpap.hpp").read_text()
    m = re.search(r"std::int32_t\s+batch\s*=\s*(\d+)\s*;", hpp)
    assert m, "SampleOptions::batch default not found — did the field move?"

    spellings = {
        "gpap::SampleOptions::batch": int(m.group(1)),
        "GPUSampler.sample(decisions_per_round=)": default(GPUSampler.sample,
                                                           "decisions_per_round"),
        "GPUSampler.sample_bits(config=)": default(GPUSampler.sample_bits, "config")[1],
        "pipeline.sample_clauses(b=)": default(sample_clauses, "b"),
        "DEFAULT_PORTFOLIO[0].decisions_per_round": DEFAULT_PORTFOLIO[0].decisions_per_round,
    }
    assert len(set(spellings.values())) == 1, \
        "the faces disagree on the default decision batch: " + repr(spellings)


def test_no_face_turns_on_the_soft_polarity_by_itself():
    """The Barbarik-rejected soft polarity is opt-in on every face. `--allow-soft` is covered by
    test_sampler.py::test_soft_is_not_in_any_default_portfolio; this checks the C++
    `SampleOptions::retry_with_lookahead` default (read from the header) and the ctypes binding's
    keyword default. On a plain config the retry runs the other polarity, which is soft."""
    import inspect
    import re
    from pathlib import Path

    from gpap.engine import _core

    hpp = (Path(REPO_ROOT) / "cpp/include/gpap/gpap.hpp").read_text()
    m = re.search(r"bool\s+retry_with_lookahead\s*=\s*(true|false)\s*;", hpp)
    assert m, "SampleOptions::retry_with_lookahead not found — did the field move or revert?"
    assert m.group(1) == "false", \
        "gpap::SampleOptions turns the soft-polarity retry on by default (the auto_retry bug)"
    assert inspect.signature(_core.Sampler.run).parameters["retry_with_lookahead"].default \
        is False, "the Python binding's own default enables it"
    # The rename is the guard: code using the old name must fail to compile. Matched as a
    # declaration, because the header mentions the old name in prose.
    assert not re.search(r"^\s*bool\s+auto_retry\b", hpp, re.M), \
        "the old field name is declared again — a silent flip for callers who only relink"


@pytest.mark.gpu
def test_full_models_honours_the_same_row_ceiling_as_the_bit_path(tmp_path):
    """`full_models=True` honours `max_samples` like the bit path. A chunk can overshoot the
    request, so both return shapes must be capped."""
    import gpap
    cls = [[1, 2], [-1, 3]]
    bits, bi = gpap.sample_clauses(4, cls, n=50, max_samples=10)
    models, mi = gpap.sample_clauses(4, cls, n=50, max_samples=10, full_models=True)
    # Both runs collected rows, or `<= 10` would hold on empty results.
    assert bi["n_models"] and mi["n_collected"], "precondition: the run produced rows to cap"
    # The ceiling is exact on both shapes: min(asked, collected).
    assert len(bits) == min(10, bi["n_models"]) == 10
    assert len(models) == min(10, mi["n_collected"]) == 10
    # An explicit collection request still wins over the default ceiling.
    more, xi = gpap.sample_clauses(4, cls, n=50, max_samples=10, full_models=True, n_cov=3000)
    assert len(more) == min(3000, xi["n_collected"]) > 10
