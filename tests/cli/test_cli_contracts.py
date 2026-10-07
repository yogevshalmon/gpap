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
from tests.conftest import REPO_ROOT



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


@pytest.mark.gpu
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


@pytest.mark.gpu
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

@pytest.mark.gpu
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


@pytest.mark.gpu
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


@pytest.mark.gpu
def test_delivery_bound_holds_when_a_valve_other_than_the_clock_stops_the_run(tmp_path):
    """`--time-limit` is not the only way a run ends short (max_chunks, the zero-progress stop).
    Bounding the top-up by the REMAINING TIME alone left those cases free to sample the full
    request afterwards — the original defect, reachable by another door."""
    from gpap.sampling.pipeline import sample_file
    info = sample_file(_cnf(tmp_path, SMALL), torch.device("cuda"), n_time=1_000_000,
                       out_cap=5000, k_cap=64, time_limit=30.0, max_chunks=2, mine=False)
    assert info["chunk_capped"]
    assert len(info["samples"]) <= info["n_models"]


@pytest.mark.gpu
@pytest.mark.parametrize("extra", [{}, {"n_cov": 300}, {"verify_sound": True}],
                         ids=["off", "cov", "verify"])
def test_phases_account_for_the_whole_reported_wall(tmp_path, extra):
    """`phases` summed to more than `wall_total_s` by exactly the output pass, which was stamped
    into one and left out of the other — so the honest e2e rate read ~1.5x high.

    The `verify` case is the same hole one field along: `verify_sound_s` was subtracted from the
    timed phase and billed to NO phase, so the sum fell SHORT by the self-test on every run of a
    campaign (VERIFY_SOUND=1 is the eval default) while `sps_e2e_total` still carried it."""
    from gpap.sampling.pipeline import sample_file
    info = sample_file(_cnf(tmp_path, SMALL), torch.device("cuda"), n_time=300, out_cap=300,
                       k_cap=64, time_limit=30.0, mine=False, **extra)
    # The opt-in READOUTS (--n-cov, --verify-sound) are billed to `phases_diag`, outside
    # `wall_total_s` — parametrized because the invariant first held only in the one
    # configuration the test happened to use.
    assert sum(info["phases"].values()) == pytest.approx(info["wall_total_s"], rel=0.05)
    # ...and an opt-in readout must be VISIBLE in phases_diag, not merely absent from `phases`:
    # billing it to neither would satisfy the sum above while losing the cost entirely.
    if extra.get("verify_sound"):
        assert info["phases_diag"]["verify_sound"] == pytest.approx(info["verify_sound_s"])


@pytest.mark.gpu
def test_emitted_rows_come_from_one_config(tmp_path):
    """Collecting during the timed run must not turn the output into a MIXTURE of the bandit's
    arms — the rows are bucketed per config and only the winner's are returned."""
    from gpap.sampling.pipeline import sample_file, CIRCUIT_PORTFOLIO
    info = sample_file(_cnf(tmp_path, SMALL), torch.device("cuda"), n_time=500, out_cap=200,
                       k_cap=64, time_limit=30.0, configs=CIRCUIT_PORTFOLIO)
    assert info["out_config"] is not None
    assert list(info["out_config"]) == [info["tuned_rv"], info["tuned_b"], info["tuned_la"]]


@pytest.mark.gpu
def test_unsat_is_reported_as_a_proof_not_as_zero_models(tmp_path, capsys):
    """An UNSAT formula printed "0 models ... sound=True", indistinguishable from a satisfiable
    instance the sampler merely failed on. Both detectors were already in hand."""
    cli_main(["sample", _cnf(tmp_path, UNSAT, "u.cnf"), "-n", "10"])
    out = capsys.readouterr().out
    # ONCE: construction proves it here, and the CLI has a second site for the formula only
    # MINING proves unsatisfiable — the two must not both fire on the same run.
    assert out.count("UNSAT") == 1
    # ...and NOT also marked '<' (a safety valve cut the run short of -n): 0 models is the
    # complete answer here, so `chunk_capped` excludes it and both printers agree.
    assert "0<" not in out


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
    """Mining runs BEFORE the timed run and the formula is built before either, but both lines
    were read off the returned dict and printed under the throughput — so the console said the
    sampler mined after it finished sampling. The pipeline reports them through `progress=`."""
    cli_main(["sample", _cnf(tmp_path, SMALL), "-n", "10", "--mine", "on"])
    out = capsys.readouterr().out.splitlines()
    where = {k: next(i for i, ln in enumerate(out) if k in ln)
             for k in ("N=4 M=2", "mined", "models in")}
    assert where["N=4 M=2"] < where["mined"] < where["models in"], out


@pytest.mark.gpu
def test_a_formula_only_mining_refutes_is_reported_as_unsat(tmp_path, capsys):
    """The REAL path behind the test below. Every 3-literal clause over 3 variables: construction
    cannot refute it (no units, no equivalences), but mining fails every literal, so the rebuilt
    formula's unit flush conflicts and the run is UNSAT — a fact that only exists after mining.
    The stub below pins the CLI's half; this pins the pipeline's, which reported the
    construction-time value and turned the proof into a silent "0 models"."""
    cnf = "p cnf 3 8\n" + "".join(f"{a} {b} {c} 0\n"
                                  for a in (1, -1) for b in (2, -2) for c in (3, -3))
    cli_main(["sample", _cnf(tmp_path, cnf, "u3.cnf"), "-n", "10", "--mine", "on"])
    out = capsys.readouterr().out
    assert "UNSAT" in out, out
    assert out.count("UNSAT") == 1, out
    assert "0<" not in out, out             # nothing was cut short; there was nothing to find


def test_unsat_proved_after_construction_is_still_printed(tmp_path, capsys, monkeypatch):
    """The UNSAT proof is printed from the `progress` callback, which fires at CONSTRUCTION —
    but mining can prove it later (every literal failing yields both polarities as units, and the
    rebuilt formula's flush conflicts), and that formula must not go out as a silent "0 models".
    Driven through the print layer: the state is "construction said satisfiable, the run says
    UNSAT", which is what the CLI has to notice."""
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

class TestSoundVerdict:
    """`pipeline.sound_verdict` / `merge_sound_check` — THE reader and THE fold.

    Every consumer branches on the string: the bench table, the CLI, the Colab driver, the rival
    harnesses and paper_tables all import these two. They replaced six open-coded comparisons, of
    which only two handled the legacy key — so a resumed campaign reported a fully verified sweep
    as "not run", and a legacy FAILURE as a non-verdict. Neither function had a test.
    """

    def test_the_three_states_are_read_back_verbatim(self):
        from gpap.sampling.pipeline import SOUND_BAD, SOUND_OFF, SOUND_OK, sound_verdict
        for state in (SOUND_OK, SOUND_BAD, SOUND_OFF):
            assert sound_verdict({"sound_check": state}) == state

    def test_a_row_with_no_verdict_at_all_reads_off_not_ok(self):
        """"not run" is not "passed". This is the conflation the tri-state exists to prevent, so
        an unrecognised value has to land on `off` rather than being taken at face value."""
        from gpap.sampling.pipeline import SOUND_OFF, sound_verdict
        assert sound_verdict({}) == SOUND_OFF
        assert sound_verdict({"sound_check": "yes"}) == SOUND_OFF     # junk is not a verdict
        assert sound_verdict({"sound_check": None}) == SOUND_OFF

    def test_the_legacy_bool_is_a_verdict_in_both_directions(self):
        """Rows recorded before 2026-08-30 carry `gpu_sound: bool` and checked unconditionally,
        so the bool IS the verdict there — including when it is False, which is the half that was
        being read as "no verdict"."""
        from gpap.sampling.pipeline import SOUND_BAD, SOUND_OK, sound_verdict
        assert sound_verdict({"gpu_sound": True}) == SOUND_OK
        assert sound_verdict({"gpu_sound": False}) == SOUND_BAD
        # a row carrying both keys is a NEW row; the new key wins
        assert sound_verdict({"sound_check": SOUND_OK, "gpu_sound": False}) == SOUND_OK

    def test_unsound_dominates_the_merge(self):
        """The timed and coverage runs are folded into one published verdict. Preferring either
        run's "ok" printed it beside the other's non-zero `unsound_rows`."""
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
    """`--verify-sound` on a run that COMPLETES NOTHING must report "off", not "ok".

    The check runs per chunk over the rows that chunk completed, so a 0%-success run never calls
    it — and publishing that as a pass is the same conflation one state along: an un-run check
    printed as a passing one. `sound_rows_checked` is the evidence, so it must be 0 with it.
    """
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


@pytest.mark.gpu
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


@pytest.mark.gpu
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


def test_the_two_faces_agree_on_the_decision_batch_default():
    """B has ONE default, and every face has to spell it the same.

    `GPUSampler.sample` defaulted to B=1 while `gpap::SampleOptions::batch`, `sample_bits`,
    `sample_clauses(b=)`, `gpap bench --b` and DEFAULT_PORTFOLIO's first entry all said 8 — so a
    direct library caller silently got a different sampler from the one the pipeline, the CLI
    and the C++ face all run, on the knob that is both a throughput AND a uniformity lever.
    Nothing catches that by construction: it is five independent literals, so it needs a test.

    Reads the C++ default out of the header text on purpose — no CUDA, no build required, and
    the header is the thing an out-of-tree C++ caller actually gets.
    """
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
    """The Barbarik-rejected soft polarity is OPT-IN on EVERY face, and there are three doors.

    Python's is `--allow-soft` (gated by test_sampler.py::test_soft_is_not_in_any_default_
    portfolio). The other two are defaults, so nothing catches them by construction: the C++
    `SampleOptions::retry_with_lookahead`, which shipped as `auto_retry = true` — the flip runs
    the OTHER polarity, so on a config the bandit left at plain that retry IS soft — and the
    ctypes binding's own keyword default, which is what every Python caller gets.

    Read out of the header text on purpose, like the decision-batch default beside it: no CUDA,
    no build, and the header is what an out-of-tree C++ caller actually compiles against.
    """
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
    # ...and the rename is the guard the ground rule asks for: a caller of the OLD name must not
    # still compile, or the reversal is invisible to everyone who only relinks. Matched as a
    # DECLARATION, not as a mention — the header explains the rename in prose right above the
    # field, and a bare substring search reads that explanation as the bug.
    assert not re.search(r"^\s*bool\s+auto_retry\b", hpp, re.M), \
        "the old field name is declared again — a silent flip for callers who only relink"


@pytest.mark.gpu
def test_full_models_honours_the_same_row_ceiling_as_the_bit_path(tmp_path):
    """`gpap.sample(..., full_models=True)` returned a whole COVERAGE CHUNK regardless of what
    was asked for — `n=50, max_samples=10` came back with 2,048 models, while the bit path
    honoured the same two numbers exactly. A chunk is the smallest unit of work, so the overshoot
    is intrinsic; what was wrong is that only one of the two return shapes was capped."""
    import gpap
    cls = [[1, 2], [-1, 3]]
    bits, bi = gpap.sample_clauses(4, cls, n=50, max_samples=10)
    models, mi = gpap.sample_clauses(4, cls, n=50, max_samples=10, full_models=True)
    # Both COLLECTED something first — otherwise `<= 10` would hold on two empty results and
    # this would be a test of nothing (the trap the C++ review already caught once).
    assert bi["n_models"] and mi["n_collected"], "precondition: the run produced rows to cap"
    # The ceiling is exact on both shapes: min(what was asked for, what was collected).
    assert len(bits) == min(10, bi["n_models"]) == 10
    assert len(models) == min(10, mi["n_collected"]) == 10
    # ...and an EXPLICIT collection request still wins over the default ceiling.
    more, xi = gpap.sample_clauses(4, cls, n=50, max_samples=10, full_models=True, n_cov=3000)
    assert len(more) == min(3000, xi["n_collected"]) > 10
