"""The rivals' target tables are TRANSCRIBED into our harnesses — so re-derive them.

`eval/rivals/rivals_recurrent.py::TARGETS` restates the branch table in the rival's own
`run.py`, and a transcription is exactly the kind of thing that silently rots: their table is
filename-keyed, their names strip dots (`s208.1` -> `s2081`), and a name we get wrong does not
raise — it falls into their catch-all branch and pins five different outputs, or is skipped with
a one-line message nobody reads. That already happened: `b14`/`b15` have a branch in their source
and were absent here, so the harness skipped `b15` — an instance their own paper evaluates.

So this parses THEIR source and checks ours against it (the verifier re-derives, per
docs/findings/ RIVAL_FAIRNESS lessons). Self-skips when the checkout is absent, since
`eval/rivals/rivals_setup.sh` is optional — but it must not skip silently on a MALFORMED
checkout, so a present-but-unparseable run.py is a failure.
"""
import ast
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RIVAL_RUN_PY = os.path.join(REPO, ".gpap", "src", "Recurrent-CircuitSAT-Sampler",
                            "src", "pytorch", "run.py")


def _their_branches(src):
    """[(names, target_idx, values)] for every `module_name in [...]` branch in their run.py.

    Their values come from a `torch.ones/zeros(batch, k)` followed by optional
    `target[:, SLICE] = V.` assignments, which is a small enough grammar to read exactly.
    """
    out = []
    # each branch: the name list, then target_idx, then the ones/zeros, then any overrides
    pat = re.compile(
        r"module_name in \[([^\]]*)\]:\s*\n"
        r"\s*self\.target_idx = \[([^\]]*)\]\s*\n"
        r"\s*self\.target = torch\.(ones|zeros)\([^)]*\)\s*\n"
        r"((?:\s*self\.target\[[^\n]*\n)*)")
    for m in pat.finditer(src):
        names = [ast.literal_eval(x.strip()) for x in m.group(1).split(",") if x.strip()]
        idx = [int(x) for x in m.group(2).split(",") if x.strip()]
        vals = [1 if m.group(3) == "ones" else 0] * len(idx)
        for line in m.group(4).strip().splitlines():          # e.g. self.target[:, 0] = 0.
            om = re.search(r"self\.target\[:,\s*([^\]]+)\]\s*=\s*([01])", line.strip())
            if not om:
                continue
            sl, v = om.group(1).strip(), int(om.group(2))
            if ":" in sl:                                     # a slice like 1:2
                lo, _, hi = sl.partition(":")
                lo = int(lo) if lo.strip() else 0
                hi = int(hi) if hi.strip() else len(idx)
                for i in range(lo, min(hi, len(idx))):
                    vals[i] = v
            else:
                vals[int(sl)] = v
        out.append((names, idx, vals))
    return out


def _their_catchall(src):
    """Their final `else:` — a REAL target (idx [0,15,31,63,-1]) that unlisted names land on."""
    m = re.search(r"\n\s*else:\s*\n"
                  r"\s*self\.target_idx = \[([^\]]*)\]\s*\n"
                  r"\s*self\.target = torch\.(ones|zeros)\([^)]*\)\s*\n"
                  r"((?:\s*self\.target\[[^\n]*\n)*)", src)
    if not m:
        return None
    idx = [int(x) for x in m.group(1).split(",") if x.strip()]
    vals = [1 if m.group(2) == "ones" else 0] * len(idx)
    for line in m.group(3).strip().splitlines():
        om = re.search(r"self\.target\[:,\s*([^\]]+)\]\s*=\s*([01])", line.strip())
        if not om:
            continue
        sl, v = om.group(1).strip(), int(om.group(2))
        if ":" in sl:
            lo, _, hi = sl.partition(":")
            lo = int(lo) if lo.strip() else 0
            hi = int(hi) if hi.strip() else len(idx)
            for i in range(lo, min(hi, len(idx))):
                vals[i] = v
        else:
            vals[int(sl)] = v
    return idx, vals


def _load_source():
    if not os.path.isfile(RIVAL_RUN_PY):
        pytest.skip("Recurrent checkout absent (eval/rivals/rivals_setup.sh); nothing to verify")
    src = open(RIVAL_RUN_PY).read()
    branches = _their_branches(src)
    # A present checkout that yields nothing means the parse broke, NOT that they removed their
    # table — reporting that as "verified" is the vacuous-test trap.
    assert len(branches) >= 8, \
        f"parsed only {len(branches)} branches from {RIVAL_RUN_PY} — the grammar changed"
    return branches


def _n_outputs(stem):
    p = os.path.join(REPO, "benchmarks", "circuits", "bench", "seq", f"{stem}.bench")
    if not os.path.isfile(p):
        return None
    return sum(1 for ln in open(p) if ln.lstrip().startswith("OUTPUT("))


def test_our_transcription_matches_their_source():
    branches = _load_source()
    src = open(RIVAL_RUN_PY).read()
    catchall = _their_catchall(src)
    assert catchall, "could not parse their else-branch — the grammar changed"
    from eval.rivals.rivals_recurrent import TARGETS

    # their key = our stem with dots removed (run.py: .replace('.bench','').replace('.',''))
    theirs = {}
    for names, idx, vals in branches:
        for n in names:
            theirs.setdefault(n, (idx, vals))       # first matching branch wins, as in their if/elif

    wrong = []
    for ours_name, (idx, vals) in sorted(TARGETS.items()):
        their_key = ours_name.replace(".", "")
        want = theirs.get(their_key)
        via_catchall = want is None
        if via_catchall:
            want = catchall
            # the catch-all indexes [0,15,31,63,-1], so it needs >= 64 outputs or their run
            # dies with IndexError — an entry relying on it must be a wide-output circuit
            n_out = _n_outputs(ours_name)
            if n_out is not None and n_out < 64:
                wrong.append(f"{ours_name}: relies on their catch-all but has only {n_out} "
                             f"outputs — their run.py would raise IndexError")
                continue
        if (list(idx), list(vals)) != (list(want[0]), list(want[1])):
            wrong.append(f"{ours_name}: ours={idx}/{vals} theirs={want}"
                         + ("  (catch-all)" if via_catchall else ""))
    assert not wrong, "our transcription disagrees with the rival's source:\n  " + \
                      "\n  ".join(wrong)


def test_every_committed_circuit_their_table_covers_is_transcribed():
    """A circuit we ship + a branch they define, but no entry here, is a SILENT SKIP.

    That is how `b15` — in their paper's Table II — went unmeasured: the harness prints
    'not in their target table, skipped' and moves on.
    """
    branches = _load_source()
    from eval.rivals.rivals_recurrent import TARGETS

    covered = {n for names, _, _ in branches for n in names}
    seq = os.path.join(REPO, "benchmarks", "circuits", "bench", "seq")
    missing = []
    for f in sorted(os.listdir(seq)):
        if not f.endswith(".bench"):
            continue
        stem = f[: -len(".bench")]
        if stem.replace(".", "") in covered and stem not in TARGETS:
            missing.append(stem)
    assert not missing, \
        ("these circuits are committed AND have a branch in the rival's own run.py, but are "
         f"absent from TARGETS, so the harness skips them silently: {missing}")


def test_wide_miter_and_is_decomposed_within_the_packed_envelope():
    """A miter over k outputs must not emit one k-input AND.

    Tseitin turns that into a (k+1)-literal clause, and the packed 8|24 envelope caps clause
    length at 255. `s35932` has 320 outputs, so `gpap_engine_create` refused the formula and the
    instance was SKIPPED with no row in either Recurrent arm — while sitting in that rival's own
    Table II. Both miter builders had the same line; `and_tree` is now the one home.
    """
    from eval.rivals.unroll_to_bench import _MITER_FANIN, and_tree

    # narrow: untouched, so every already-measured instance is bit-identical
    lines = []
    assert and_tree(lines, ["a", "b", "c"], "T") == ["a", "b", "c"]
    assert lines == [], "a narrow AND must emit no helper gates"

    # wide: every emitted gate stays within the fan-in, and the tree collapses to <= fanin roots
    lines = []
    roots = and_tree(lines, [f"o{i}" for i in range(320)], "T")
    assert len(roots) <= _MITER_FANIN
    widest = max(len(g.split("(")[1].rstrip(")").split(",")) for g in lines)
    assert widest <= _MITER_FANIN, f"emitted an AND of fan-in {widest}"
    assert widest + 1 <= 255, "still outside the packed 8|24 clause-length cap"

    # every original term is consumed exactly once (no term dropped or duplicated)
    used = [t for g in lines for t in g.split("(")[1].rstrip(")").split(", ")]
    assert sorted(t for t in used if t.startswith("o")) == sorted(f"o{i}" for i in range(320))


def test_oversize_instance_is_refused_before_it_can_kill_the_sweep(tmp_path):
    """An instance too large for the rival path must be REFUSED, not attempted.

    The rival consumes .bench and unrolls it itself, so our arm has to Tseitin-encode the same
    circuit — and b19 at T=25 is 9.6M vars that way (66,639 through abc:fast). Building it
    exhausts host memory and the process is KILLED: no exception, no row, and every instance
    alphabetically after it silently unmeasured. Four went missing from a 29-instance sweep
    exactly this way, so the size is checked from the .bench BEFORE anything is built.
    """
    from eval.rivals.rival_common import MAX_UNROLLED_GATES, oversize_reason

    # `avail_bytes` is pinned throughout: the guard's second term reads THIS machine's free RAM,
    # and a test whose verdict depends on what else is running is not a test.
    BIG_BOX = 256 << 30

    small = tmp_path / "small.bench"
    small.write_text("INPUT(a)\nOUTPUT(z)\nz = NOT(a)\n")
    assert oversize_reason(str(small), 25, avail_bytes=BIG_BOX) is None

    # just over the bound at T=25 -> refused, with a reason that names the arithmetic
    n = MAX_UNROLLED_GATES // 25 + 10
    big = tmp_path / "big.bench"
    big.write_text("INPUT(a)\nOUTPUT(z)\n" + "".join(f"g{i} = NOT(a)\n" for i in range(n)))
    why = oversize_reason(str(big), 25, avail_bytes=BIG_BOX)
    assert why and "unrolled gates" in why and f"{n:,}" in why
    # ...and the SAME circuit at a shallow depth is fine: it is the unroll that is oversized
    assert oversize_reason(str(big), 1, avail_bytes=BIG_BOX) is None


def test_an_instance_too_big_for_THIS_box_is_refused_before_it_is_built(tmp_path):
    """The absolute gate cap is a fixed number; the machines are not.

    b18 at T=25 peaks at 10.6 GiB (measured) — fine on a 47 GB workstation, an OOM kill on a
    13 GB Colab VM. An OOM kill is the failure mode with no exception to catch and no row to
    show for it, which is exactly how four instances vanished from a 29-instance sweep. So the
    guard also refuses what will not fit HERE, with the arithmetic in the reason.
    """
    from eval.rivals.rival_common import oversize_reason, peak_host_gib

    # ~2.86M unrolled gates: b18's shape, well under MAX_UNROLLED_GATES
    gates = 114_561
    b = tmp_path / "b18ish.bench"
    b.write_text("INPUT(a)\nOUTPUT(z)\n"
                 + "".join(f"g{i} = NOT(a)\n" for i in range(gates)))
    est = peak_host_gib(gates * 25)
    assert 9.0 < est < 13.0, f"the measured fit moved ({est:.1f} GiB for b18); re-derive the test"

    assert oversize_reason(str(b), 25, avail_bytes=13 << 30), "must be refused on a 13 GiB box"
    assert oversize_reason(str(b), 25, avail_bytes=53 << 30) is None, "must run on a 53 GiB box"
    why = oversize_reason(str(b), 25, avail_bytes=13 << 30)
    assert "host RAM" in why and "available on this machine" in why, \
        f"the reason must name the memory arithmetic, not the gate cap: {why}"


def test_the_memory_term_is_skipped_when_memory_cannot_be_read():
    """Refusing an instance on a guess is worse than attempting it."""
    from eval.rivals.rival_common import oversize_reason
    import eval.rivals.rival_common as rc

    saved = rc._available_bytes
    rc._available_bytes = lambda: None
    try:
        assert oversize_reason(os.path.join(REPO, "benchmarks", "circuits", "bench", "seq",
                                            "b13.bench"), 25) is None
    finally:
        rc._available_bytes = saved


# --------------------------------------------------------------------------- the start_point probe
#
# The probe decides whether a Recurrent cell is a comparison or an error row, and for one release
# it decided that on OUR stopwatch: each probe was capped at 60 s against the measured run's
# 300 s, and all three ways the search can fail were reported as the single string "no gradient
# path at any start_point<=10" — a claim about the RIVAL. Measured (docs/findings/
# RIVAL_FAIRNESS_FINDINGS.md §11): b14 at T=25 RUNS at start_point=1 in 743 s, where b17 really
# does refuse in 5 s. These pin the contract without needing the checkout.

def _probe(monkeypatch, replies, **kw):
    """Drive probe_start_point against a stubbed rival. `replies[sp]` is its error string, or
    None for a successful run."""
    from eval.rivals import rivals_recurrent as rr

    seen = []

    def fake_run(rival, bench, key, T, steps, batch, start_point, timeout):
        seen.append((start_point, timeout))
        err = replies.get(start_point, "does not require grad")
        return {} if err is None else {"error": err}

    monkeypatch.setattr(rr, "run_recurrent", fake_run)
    sp, why = rr.probe_start_point("R", "b.bench", "b", 25, cache_dir=None, **kw)
    return sp, why, seen


def test_the_whole_initialization_gets_one_budget_equal_to_the_run_timeout(monkeypatch):
    """The start_point search is the rival's INITIALIZATION and gets ONE budget: the run's
    timeout (300 s by default). Each probe runs with whatever is LEFT, so no probe can exceed
    the budget and the sum of the probes cannot either. It used to be 4x the run timeout, so a
    circuit whose model build takes ~1,100 s (b17) spent 4 x 1,200 s finding out it could not
    start; the 300 s the user set is generous beside the 60 s the rival is measured under."""
    from eval.rivals import rivals_recurrent as rr

    assert rr.probe_budget(rr.DEFAULT_TIMEOUT) == rr.DEFAULT_TIMEOUT == 300.0
    assert rr.probe_budget(900.0) == 900.0          # tracks --timeout, not a second constant
    _sp, _why, seen = _probe(monkeypatch, {2: None}, run_timeout=300.0)
    assert seen, "the probe never ran — the stub was not reached"
    assert all(0 < t <= 300.0 for _s, t in seen), seen
    assert all(t2 <= t1 for (_a, t1), (_b, t2) in zip(seen, seen[1:])), \
        f"a probe was given more than what was left of the budget: {seen}"


def test_a_settled_search_returns_the_start_point_and_no_reason(monkeypatch):
    sp, why, seen = _probe(monkeypatch, {2: None})
    assert (sp, why) == (2, None)
    assert [s for s, _t in seen] == [0, 1, 2], "the walk must climb from 0"


def test_a_gradient_path_refusal_is_reported_as_one(monkeypatch):
    """The only outcome that is a finding ABOUT THE RIVAL: it refused every depth we reached."""
    sp, why, _seen = _probe(monkeypatch, {}, max_sp=4)          # no-grad everywhere
    assert sp is None
    assert why == "no gradient path at any start_point<=4"


def test_a_timeout_is_reported_as_a_timeout_with_its_budget(monkeypatch):
    """b14's real case. It must NOT read as 'no gradient path', and it must name the budget so a
    reader can see whose number it is."""
    sp, why, seen = _probe(monkeypatch, {1: "timeout 300.0s"}, run_timeout=300.0)
    assert sp is None
    assert "gradient" not in why, f"a timeout must not be reported as a gradient-path refusal: {why}"
    assert "300s" in why and "budget" in why, why
    # and the walk STOPS there: a larger start_point pays the same model build
    assert [s for s, _t in seen] == [0, 1], f"the walk continued past a timeout: {seen}"


def test_an_exhausted_budget_reads_as_did_not_initialize(monkeypatch):
    """The third outcome: the probes we ran used the budget up before one completed. That is
    neither a gradient-path refusal nor a single timeout — it is the rival failing to initialize
    within the 300 s the campaign grants, and the row must say so with the budget's number."""
    sp, why, seen = _probe(monkeypatch, {}, budget=-1.0)       # budget already spent
    assert sp is None and seen == [], (sp, why, seen)
    assert why.startswith("did not initialize"), why
    assert "gradient" not in why and "OUR" not in why, why
    assert "budget" in why


def test_the_ladder_runs_once_and_never_probes_upward_or_retries_a_timeout():
    """`run_at_batch` = ONE measured configuration per instance. A timeout is recorded as the
    result (never retried at a smaller batch), a driver refusal steps ONE rung down, and a
    successful run is never followed by a bigger probe. Shared by every rival harness."""
    from eval.rivals.rival_common import run_at_batch

    calls = []

    def ok(b):
        calls.append(b)
        return {"unique_per_s": 1.0}
    r, sweep = run_at_batch(ok, 3000, ladder=[100, 1000, 3000, 10000], log=lambda *_: None)
    assert calls == [3000] and [s["batch"] for s in sweep] == [3000]

    calls.clear()

    def timeout(b):
        calls.append(b)
        return {"error": "timeout 300.0s"}
    r, sweep = run_at_batch(timeout, 3000, ladder=[100, 1000, 3000], log=lambda *_: None)
    assert calls == [3000] and "timeout" in r["error"]

    calls.clear()

    def refused_then_ok(b):
        calls.append(b)
        return {"error": "RuntimeError: CUDA driver error: device not ready"} if b == 3000 \
            else {"unique_per_s": 1.0}
    r, sweep = run_at_batch(refused_then_ok, 3000, ladder=[100, 1000, 3000],
                            log=lambda *_: None)
    assert calls == [3000, 1000] and "error" not in r
    assert [s["batch"] for s in sweep] == [3000, 1000]


def test_each_rival_scores_on_its_own_papers_instances_from_one_home():
    """`paper` expands to the rival's published set, and the tables score from the SAME dict:
    a set edited in one place cannot drift from the other."""
    from eval.artifact import paper_tables
    from eval.rivals.rival_common import PAPER_INSTANCES, expand_names

    assert paper_tables.PAPER_INSTANCES is PAPER_INSTANCES
    assert len(PAPER_INSTANCES["demotic"]) == 11 and len(PAPER_INSTANCES["rcs"]) == 16
    assert expand_names(["paper"], "rcs") == sorted(PAPER_INSTANCES["rcs"])
    assert expand_names(["b03", "paper"], "demotic") == ["b03"] + sorted(PAPER_INSTANCES["demotic"])
    assert expand_names(["b03"], "rcs") == ["b03"]           # anything else passes through


def test_the_gpap_arm_cache_key_covers_the_python_sampling_layer(tmp_path, monkeypatch):
    """A cached rival-arm result must MISS when the code that produced it changed.

    The key fingerprints libgpap's bytes (`gpap_build_id`), which is the documented rule — but
    the arm's answer is also decided by `gpap/sampling/`: the portfolio, the defaults, the
    preprocessing, the coverage-run budget. A change there moves the number and leaves the build
    id untouched, so a seeded campaign would replay stale arms as if they were current. That is
    the same failure the key's own comment warns about one layer down.
    """
    import shutil

    from eval.rivals import rival_common as rc

    f = tmp_path / "x.cnf"
    f.write_text("p cnf 1 1\n1 0\n")
    task = {"harness": "t", "cycles": 25}
    before = rc.gpap_cache_key(str(f), task)

    pipe = os.path.join(REPO, "gpap", "sampling", "pipeline.py")
    bak = tmp_path / "pipeline.bak"
    shutil.copy(pipe, bak)
    try:
        with open(pipe, "a") as fh:
            fh.write("\n# cache-key probe\n")
        rc._sampling_layer_id.cache_clear()
        after = rc.gpap_cache_key(str(f), task)
    finally:
        shutil.copy(bak, pipe)
        rc._sampling_layer_id.cache_clear()

    assert after != before, \
        "editing gpap/sampling/pipeline.py did not change the cache key — stale arms would be " \
        "replayed as current measurements"
    assert rc.gpap_cache_key(str(f), task) == before, \
        "the key did not return to its original value after the edit was reverted"


def test_the_row_says_WHICH_sampler_produced_it():
    """`--portfolio` selects GPAP's arm, so the row has to record which one ran.

    The default is the shipped adaptive bandit (`circuit`) — every recorded campaign used it, and
    changing it would move every published ratio. But it is an open asymmetry: the rivals and
    CMSGen all run one fixed configuration while the bandit measures configs against each other
    inside the timed wall and reports the winner (RIVAL_FAIRNESS_FINDINGS.md §10, §12). A caption
    cannot carry that; the row can, via `portfolio` + `adaptive`.
    """
    from gpap.sampling.pipeline import portfolio
    from eval.rivals.rivals_recurrent import GPAP_PORTFOLIO

    assert len(portfolio(GPAP_PORTFOLIO)) > 1, \
        "the default is documented as the adaptive bandit; if it is pinned now, say so here"
    assert len(portfolio("b8")) == 1, "b8 must remain a genuine single-config escape hatch"


def test_both_rival_harnesses_share_the_pin():
    """One definition. rivals_demotic imports the constant rather than restating it — two
    harnesses drifting on the sampler they measure is the drift this repo keeps re-fixing."""
    import inspect

    from eval.rivals import rivals_demotic, rivals_recurrent

    src = inspect.getsource(rivals_demotic)
    assert "GPAP_PORTFOLIO" in src, "the DEMOTIC harness must reference the shared default"
    assert rivals_demotic.GPAP_PORTFOLIO is rivals_recurrent.GPAP_PORTFOLIO


def test_the_portfolio_is_part_of_the_arm_cache_key():
    """Two runs that differ only in `--portfolio` are different measurements. The GPAP-arm cache
    keys on the task dict, so the portfolio has to be IN it — otherwise a `--portfolio b1` run
    silently serves the `b8` result that is already on disk."""
    import tempfile

    from eval.rivals import rival_common as rc

    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "x.cnf")
        open(f, "w").write("p cnf 1 1\n1 0\n")
        base = {"harness": "recurrent", "cycles": 25}
        k8 = rc.gpap_cache_key(f, {**base, "portfolio": "b8"})
        k1 = rc.gpap_cache_key(f, {**base, "portfolio": "b1"})
    assert k8 != k1, "the portfolio does not reach the cache key — a re-run would serve a " \
                     "result produced by a different sampler"


# ---------------------------------------------------------------- the GPAP arm: encoder + unique estimate

def _tiny_bench(tmp_path):
    b = tmp_path / "t.bench"
    b.write_text("INPUT(a)\nINPUT(b)\nOUTPUT(z)\nq = DFF(a)\nz = AND(q, b)\n")
    return str(b)


def _stub_arm(monkeypatch, samples, sps=1000.0, n_models=5000):
    """Stub the encoder and the pipeline under `run_gpap`; returns the recorded call kwargs."""
    import numpy as np

    from gpap.circuits import abc_backend, bench
    from gpap.sampling import pipeline

    seen = {}

    def fake_abc(circ, path, T, seed, pin_outputs=None, *, abc=None, opt="auto", target=None):
        seen["encoder_call"] = {"opt": opt, "pin_outputs": pin_outputs, "T": T}
        return bench.build_cnf(circ, T, seed=seed, pin_outputs=pin_outputs, target=target)

    def fake_sample_file(path, device, **kw):
        seen["sample_kw"] = kw
        return {"samples_per_s": sps, "sps_e2e_total": sps, "success_rate": 0.5,
                "n_models": n_models, "sound_check": "ok",
                "samples": np.asarray(samples, dtype=np.uint8)}

    monkeypatch.setattr(abc_backend, "build_cnf_abc", fake_abc)
    monkeypatch.setattr(pipeline, "sample_file", fake_sample_file)
    return seen


def test_the_gpap_arm_encodes_with_abc_fast_and_says_so(tmp_path, monkeypatch):
    """The GPAP arm of every rival row is the abc:fast formula — the one Table 1 runs — and the
    row records it. It used to be the native Tseitin with no encoder parameter at all, which made
    Table 2 the only table whose GPAP number came from a 2-10x larger formula (b14: 413,451 vars
    vs 41,938), unlabelled. RIVAL_FAIRNESS_FINDINGS.md §13."""
    from eval.rivals import rivals_recurrent as rr

    seen = _stub_arm(monkeypatch, samples=[[0, 1], [1, 0], [0, 1]])
    g = rr.run_gpap(_tiny_bench(tmp_path), 3, 1, None, [5], 100, "cpu", time_limit=1.0)
    assert rr.GPAP_ENCODER == "abc:fast"
    assert seen["encoder_call"]["opt"] == "fast" and seen["encoder_call"]["pin_outputs"] == 1
    assert g["encoder"] == "abc:fast" and g["encode_s"] >= 0.0
    assert g["portfolio"] == rr.GPAP_PORTFOLIO


def test_the_gpap_arm_runs_no_coverage_pass_and_measures_distinctness_on_its_own_rows(
        tmp_path, monkeypatch):
    """`unique_per_s_est` = samples/s x the distinct fraction of the timed run's OWN kept output
    rows (`out_cap=UNIQUE_ROWS`, GPU-projected support bits), with `n_cov=0`: the coverage pass
    reconstructed rows x nv x 72 B of full models on the host and OOM-killed b14 (61 GB), and the
    user's rule is that the eval does not need it at all."""
    from eval.rivals import rivals_recurrent as rr

    seen = _stub_arm(monkeypatch, samples=[[0, 1], [1, 0], [0, 1], [0, 1]], sps=1000.0)
    g = rr.run_gpap(_tiny_bench(tmp_path), 3, 1, None, [5], 100, "cpu", time_limit=1.0)
    kw = seen["sample_kw"]
    assert kw["n_cov"] == 0, "the coverage pass must not run on the rival arm"
    assert kw["out_cap"] == rr.UNIQUE_ROWS == 1000
    assert kw["verify_sound"] is True
    r = g["seeds"][0]
    assert r["unique_rows"] == 4 and r["unique_distinct"] == 2
    assert g["unique_per_s_est"] == pytest.approx(1000.0 * 2 / 4)
    assert g["unique_est_status"] == "ok"


def test_no_kept_row_is_not_measured_and_never_a_zero(tmp_path, monkeypatch):
    """A run that kept nothing has no distinct fraction: None, with a status the table reader
    (`paper_tables.unique_unmeasured`) recognizes — never 0.0, which every consumer buckets as
    'GPAP produced nothing'."""
    import numpy as np

    from eval.artifact.paper_tables import unique_unmeasured
    from eval.rivals import rivals_recurrent as rr

    _stub_arm(monkeypatch, samples=np.zeros((0, 2)), n_models=3)
    g = rr.run_gpap(_tiny_bench(tmp_path), 3, 1, None, [5], 100, "cpu", time_limit=1.0)
    assert g["unique_per_s_est"] is None
    assert g["unique_est_status"] == "no-sample-rows"
    assert unique_unmeasured(g), "the table must bucket this as unmeasured, not as GPAP=0"


def test_the_encoder_is_part_of_both_harnesses_cache_key():
    """A Tseitin row on disk must not be served for an abc:fast run: the task dict each harness
    hashes has to carry the encoder (and the kept-row count the estimate rests on)."""
    import inspect

    from eval.rivals import rivals_demotic, rivals_recurrent

    for mod in (rivals_recurrent, rivals_demotic):
        src = inspect.getsource(mod)
        m = re.search(r"task = \{(.*?)\}\n", src, re.S)
        assert m, f"{mod.__name__}: no task dict"
        assert '"encoder": GPAP_ENCODER' in m.group(1), f"{mod.__name__}: encoder not in the key"
        assert '"unique_rows": UNIQUE_ROWS' in m.group(1)


def test_the_oversize_guard_is_a_tseitin_guard(tmp_path):
    """Both terms of `oversize_reason` model the native Tseitin build; under abc:fast every seq
    instance in the suite built on the 13 GB L4 VM (b18: 19,859 vars against 2.86M unrolled
    gates), so a Tseitin-calibrated refusal there would omit instances the campaign already
    ran. The guard must pass everything for the abc:fast encoder and keep refusing for Tseitin."""
    from eval.rivals.rival_common import oversize_reason

    gates = 114_561                                  # b18's shape
    b = tmp_path / "b18ish.bench"
    b.write_text("INPUT(a)\nOUTPUT(z)\n" + "".join(f"g{i} = NOT(a)\n" for i in range(gates)))
    assert oversize_reason(str(b), 25, avail_bytes=13 << 30, encoder="tseitin")
    assert oversize_reason(str(b), 25, avail_bytes=13 << 30, encoder="abc:fast") is None
    assert oversize_reason(str(b), 25, avail_bytes=13 << 30) is not None, \
        "the default stays the Tseitin guard — a caller has to say abc:fast to lift it"


# ---------------------------------------------------------------- the CMSGen arm on the rival rows

def test_the_cmsgen_arm_is_the_bench_runner_timed_emission_free(monkeypatch, tmp_path):
    """The third tool of every rival row is CMSGen on the row's own CNF, run by the SAME runner
    `gpap bench` uses — `--nowrite` (no emission, like our collect=False run), the sampling
    budget passed through so `--maxtime` makes the count exact — and recorded under the bench
    driver's own `cms_*` keys, so every reader of a bench row reads a rival row unchanged."""
    import gpap.benchmark as gb
    from eval.artifact.paper_tables import cms_arm_ran
    from eval.rivals import rival_common as rc

    seen = {}

    def fake_run(path, cmsgen_bin, n, scratch, *, keep=None, timeout=60.0, write=True,
                 stats_every=1000, seed=None):
        seen.update(path=path, n=n, timeout=timeout, write=write, seed=seed)
        return {"samples_per_s": 123.0, "samples_per_s_e2e": 100.0, "n_samples": 7380,
                "mode": "nowrite", "patched": True, "count_exact": True, "budget_stop": True,
                "timed_out": False, "wall_s": 61.0, "sampling_wall_s": 60.0,
                "decisions_per_s": 1e5, "implications_per_s": 2e6, "models": []}

    monkeypatch.setattr(gb, "cmsgen_run", fake_run)
    row = rc.cmsgen_arm("/x/y.cnf", "/bin/cmsgen", 50_000, 60.0, str(tmp_path), seed=5)
    assert seen == {"path": "/x/y.cnf", "n": 50_000, "timeout": 60.0, "write": False,
                    "seed": 5}
    assert row["cms_mode"] == "nowrite" and row["cms_count_exact"] is True
    assert row["cms_sps"] == 123.0 and row["cms_n_samples"] == 7380
    assert row["cms_implications_per_s"] == 2e6            # Table 3 reads its own counters
    assert cms_arm_ran(row), "a bench-row reader must see the arm on a rival row"
    assert "cms_error" not in row


def test_a_crashed_cmsgen_subprocess_is_recorded_as_an_error_not_a_zero(monkeypatch, tmp_path):
    """A crashed baseline is `cms_error` beside a count of 0 — the ledger's own bucket, never
    'CMSGen sampled nothing' (which would score as our win)."""
    import gpap.benchmark as gb
    from eval.artifact.paper_tables import cms_arm_ran
    from eval.rivals import rival_common as rc

    monkeypatch.setattr(gb, "cmsgen_run",
                        lambda *a, **k: {"error": "OSError: boom", "samples_per_s": 0.0,
                                         "models": [], "n_samples": 0, "mode": "nowrite"})
    row = rc.cmsgen_arm("/x/y.cnf", "/bin/cmsgen", 100, 1.0, str(tmp_path))
    assert row["cms_error"] and row["cms_n_samples"] == 0
    assert cms_arm_ran(row), "it RAN and crashed: the cms_err bucket, not the absent one"


def test_run_gpap_samples_the_prebuilt_cnf_and_leaves_it_for_the_cmsgen_arm(tmp_path,
                                                                            monkeypatch):
    """Both of our arms run ONE file: the harness builds the CNF once (`build_arm_cnf`), hands
    it to `run_gpap(cnf=...)`, which samples exactly that path and must not delete it, and then
    to `cmsgen_arm`. Without `cnf=` the arm builds and removes a temporary of its own."""
    from eval.rivals import rivals_recurrent as rr
    from gpap.sampling import pipeline

    seen = _stub_arm(monkeypatch, samples=[[0, 1], [1, 0]])
    sampled, stub = [], pipeline.sample_file

    def spy(path, device, **kw):
        sampled.append(path)
        return stub(path, device, **kw)

    monkeypatch.setattr(pipeline, "sample_file", spy)
    cnf = str(tmp_path / "row.cnf")
    info = rr.build_arm_cnf(_tiny_bench(tmp_path), 3, 1, None, cnf)
    text = open(cnf).read()
    assert info["encoder"] == "abc:fast" and info["n_vars"] > 0 and info["support"] == 6
    assert "c ind " in text and "p cnf " in text, "a projected CNF CMSGen can read"
    g = rr.run_gpap(_tiny_bench(tmp_path), 3, 1, None, [5], 100, "cpu", time_limit=1.0,
                    cnf=(cnf, info))
    assert sampled == [cnf], "the GPAP arm must sample the very file the CMSGen arm gets"
    assert os.path.isfile(cnf), "the shared CNF is the harness's to remove, after the CMSGen arm"
    assert g["n_vars"] == info["n_vars"] and g["encode_s"] == info["encode_s"]
    assert seen["encoder_call"]["opt"] == "fast"


def test_both_harnesses_run_the_cmsgen_arm_on_the_cnf_the_gpap_arm_sampled():
    """Structural gate on both harness mains: a `--cmsgen` flag, ONE `build_arm_cnf` into a
    path, `run_gpap(... cnf=(<that path>, ...))` and `cmsgen_arm(<that path>, ...)` — so a
    harness cannot drift into encoding twice (or pinning differently) for the two comparisons,
    and the row carries the host CPU the CMSGen ratio needs."""
    import inspect

    from eval.rivals import rivals_demotic, rivals_recurrent

    for mod in (rivals_recurrent, rivals_demotic):
        src = inspect.getsource(mod.main)
        assert '"--cmsgen"' in src, mod.__name__
        m = re.search(r"build_arm_cnf\(\w+, [^,]+, [^,]+, [^,]+, (\w+)\)", src)
        assert m, f"{mod.__name__}: no single build_arm_cnf call"
        path = m.group(1)
        assert re.search(rf"cnf=\({path}, \w+\)", src), f"{mod.__name__}: run_gpap not on {path}"
        assert re.search(rf"cmsgen_arm\({path}, ", src), f"{mod.__name__}: cmsgen_arm not on {path}"
        assert "host_provenance(device)" in src, f"{mod.__name__}: no cpu/gpu provenance"
