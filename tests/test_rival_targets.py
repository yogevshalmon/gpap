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


def test_the_ladder_measures_the_largest_rung_that_completes():
    """`run_at_batch` = ONE measured configuration per instance, and it is the LARGEST rung the
    rival can actually finish on (2026-09-04). Rungs are rejected on FEASIBILITY — OOM, a driver
    refusal, or not finishing inside the wall budget — never on throughput, so this is still one
    measurement and not a best-of-N. Shared by every rival harness.

    It used to start at our VRAM model's rung and step down only on a driver refusal, recording
    a TIMEOUT as the rival's result. Both halves were wrong: the model was measured at
    num_steps=10 and we run 1, so it under-granted DEMOTIC 1.5-3.9x on 4 of 11 ISCAS-85 cells,
    and a batch the rival cannot finish with was never granted to it in the first place."""
    from eval.rivals.rival_common import run_at_batch

    calls = []

    def ok(b):
        calls.append(b)
        return {"unique_per_s": 1.0}
    r, sweep = run_at_batch(ok, 3000, ladder=[100, 1000, 3000, 10000], log=lambda *_: None)
    assert calls == [10000], "did not take the largest rung the rival can run"
    assert [s["batch"] for s in sweep] == [10000]

    calls.clear()

    def timeout(b):
        calls.append(b)
        return {"error": "timeout 300.0s"}
    r, sweep = run_at_batch(timeout, 3000, ladder=[100, 1000, 3000], log=lambda *_: None)
    assert "timeout" in r["error"], "an all-timeout ladder must still report the timeout"
    assert calls[0] == 3000, "did not start at the top of the ladder"

    calls.clear()

    def refused_then_ok(b):
        calls.append(b)
        return {"error": "RuntimeError: CUDA driver error: device not ready"} if b >= 3000 \
            else {"unique_per_s": 1.0}
    r, sweep = run_at_batch(refused_then_ok, 100, ladder=[100, 1000, 3000],
                            log=lambda *_: None)
    assert calls == [3000, 1000] and "error" not in r, "the refusal did not step down"
    assert [s["batch"] for s in sweep] == [3000, 1000], "the rungs tried belong in the row"


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
    silently serves the `b8` result that is already on disk.

    Gated on the harnesses' REAL task dicts (source-inspected, like the encoder test below).
    The first version built its own dict with a `portfolio` key and hashed that — a test that
    exercised nothing: deleting `portfolio` from the harness would have left it green."""
    import inspect
    import tempfile

    from eval.rivals import rival_common as rc
    from eval.rivals import rivals_demotic, rivals_recurrent

    for mod in (rivals_recurrent, rivals_demotic):
        m = re.search(r"task = \{(.*?)\}\n", inspect.getsource(mod), re.S)
        assert m, f"{mod.__name__}: no task dict"
        assert '"portfolio": args.portfolio' in m.group(1), \
            f"{mod.__name__}: the portfolio is not in the cache key"
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "x.cnf")
        open(f, "w").write("p cnf 1 1\n1 0\n")
        base = {"harness": "recurrent", "cycles": 25}
        k8 = rc.gpap_cache_key(f, {**base, "portfolio": "b8"})
        k1 = rc.gpap_cache_key(f, {**base, "portfolio": "b1"})
    assert k8 != k1, "two task dicts differing only in the portfolio must hash apart"


def test_the_results_csv_cleanup_strips_the_dot_like_the_readback_does():
    """Their `export_results` writes `sat_{dataset_str}_…` with the dot STRIPPED from the
    instance name. The read-back glob learned that in the 2026-09-03 post-mortem; the clean-up
    glob one function above it did not, so for `s208.1` / `s15850.1` no stale file was ever
    removed and the read-back's `hits[0]` was an arbitrary pick among whatever rungs had run.
    Both globs must strip the dot, and a multi-hit read-back must pick THIS call's rung."""
    import inspect

    from eval.rivals import rivals_recurrent

    src = inspect.getsource(rivals_recurrent.run_recurrent)
    globs = re.findall(r"glob\.glob\(os\.path\.join\(rival, \"src\", \"results\", (f\"[^\"]+\")", src)
    assert len(globs) >= 2, "expected a clean-up glob and a read-back glob"
    for g in globs:
        assert "key.replace('.', '')" in g, f"a results glob keeps the dot: {g}"
    assert 'h.endswith(f"_{batch}.csv")' in src, "a multi-hit read-back must select by batch"


# ---------------------------------------------------------------- the GPAP arm: encoder + unique estimate

def _tiny_bench(tmp_path):
    b = tmp_path / "t.bench"
    b.write_text("INPUT(a)\nINPUT(b)\nOUTPUT(z)\nq = DFF(a)\nz = AND(q, b)\n")
    return str(b)


def _stub_arm(monkeypatch, samples, sps=1000.0, n_models=5000, wall_total=10.0, proj_s=0.0):
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
        # `sps_e2e_total` is the library's own e2e rate, which CARRIES the output projection;
        # the arm recomputes its own with that phase removed (see the offline-dedup test).
        return {"samples_per_s": sps, "sps_e2e_total": n_models / wall_total,
                "success_rate": 0.5, "wall_total_s": wall_total,
                "phases": {"sample": wall_total - proj_s, "output": proj_s},
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
    """`unique_per_s_est` = samples/s x the EXACT distinct fraction of everything the timed run
    delivered (`out_cap=UNIQUE_ROWS`, now above any recorded delivery), with `n_cov=0`: the
    coverage pass reconstructed rows x nv x 72 B of full models on the host and OOM-killed b14
    (61 GB), and the user's rule is that the eval does not need it.

    The cap was 1,000 until 2026-09-04, which made this an ESTIMATE from a window applied to runs
    delivering up to a million models — while both rivals deduplicate their whole output exactly,
    off their own clock. Same metric, same way, or it is not their metric."""
    from eval.rivals import rivals_recurrent as rr

    seen = _stub_arm(monkeypatch, samples=[[0, 1], [1, 0], [0, 1], [0, 1]], sps=1000.0,
                     n_models=4)
    g = rr.run_gpap(_tiny_bench(tmp_path), 3, 1, None, [5], 100, "cpu", time_limit=1.0)
    kw = seen["sample_kw"]
    assert kw["n_cov"] == 0, "the coverage pass must not run on the rival arm"
    assert kw["out_cap"] == rr.UNIQUE_ROWS >= 4_000_000, "the arm must dedup the whole run"
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


# ---------------------------------------------------------------- alignment + circuit-level checks

def test_the_recurrent_miter_self_checks_and_catches_an_inverted_miter(tmp_path):
    """The rare-mode miter's P[MITER=1] must equal the instance's own p_accept — the check the
    DEMOTIC arm always ran and the Recurrent arm asserted by construction only. Sabotage: an
    inverted miter (P ~ 1 - p) must be caught, or the check proves nothing."""
    from eval.rivals import rivals_recurrent as rr

    src = os.path.join(REPO, "benchmarks", "circuits", "bench", "seq", "b06.bench")
    dst = str(tmp_path / "b06_rare.bench")
    pattern, est, p_miter, ok = rr.miter_bench(src, dst, 3, 5)
    assert ok and 0.0 <= p_miter <= 1.0 and pattern and est is not None
    text = open(dst).read()
    assert text.count("OUTPUT(") == 1 and "OUTPUT(GPAPMITER)" in text
    bad = str(tmp_path / "b06_inverted.bench")
    inv = re.sub(r"^GPAPMITER = (AND|NOT)\(", lambda m: f"GPAPMITER = {'NAND' if m.group(1) == 'AND' else 'BUFF'}(",
                 text, flags=re.M)
    assert inv != text
    open(bad, "w").write(inv)
    p_bad, ok_bad = rr.miter_self_check(src, bad, 3, 5)
    assert not ok_bad and p_bad > p_miter, "an inverted miter must fail the self-check"


def test_both_harnesses_verify_constraint_alignment_per_row():
    """Table 2's text says alignment is VERIFIED per row and the `misaligned` bucket exists for
    it; for one release only the DEMOTIC harness ran the check, so the headline sequential rows
    carried no evidence at all. Both mains must call it and write `pin_check_ok`."""
    import inspect

    from eval.rivals import rival_common, rivals_demotic, rivals_recurrent

    assert hasattr(rival_common, "native_pin_check"), "one home for the check"
    for mod in (rivals_recurrent, rivals_demotic):
        src = inspect.getsource(mod.main)
        assert "native_pin_check(" in src, mod.__name__
        assert '"pin_check_ok"' in src, mod.__name__
    assert "miter_self_check(" in inspect.getsource(rivals_recurrent.miter_bench)


def _stub_arm_with_cols(monkeypatch, samples_by_label, sps=1000.0):
    """Like `_stub_arm`, but the fake pipeline also returns the kept rows' column ids and the
    wire map, exactly as `sample_file` does — so the arm's simulate check can run."""
    import numpy as np

    from gpap.circuits import abc_backend, bench
    from gpap.sampling import pipeline

    seen = {}

    def fake_abc(circ, path, T, seed, pin_outputs=None, *, abc=None, opt="auto", target=None):
        b, support, tgt, labels = bench.build_cnf(circ, T, seed=seed, pin_outputs=pin_outputs,
                                                  target=target)
        seen["support"], seen["labels"] = list(support), list(labels)
        return b, support, tgt, labels

    def fake_sample_file(path, device, **kw):
        labels = seen["labels"]
        rows = [[row.get(lbl, 0) for lbl in labels] for row in samples_by_label]
        return {"samples_per_s": sps, "sps_e2e_total": sps, "success_rate": 0.5,
                "n_models": 10, "sound_check": "ok",
                "samples": np.asarray(rows, dtype=np.uint8), "cols": seen["support"],
                "wire_map": dict(zip(seen["support"], labels))}

    monkeypatch.setattr(abc_backend, "build_cnf_abc", fake_abc)
    monkeypatch.setattr(pipeline, "sample_file", fake_sample_file)
    return seen


def test_the_arm_simulates_its_kept_rows_through_the_original_circuit(tmp_path, monkeypatch):
    """Encoder-independent soundness ON THE CAMPAIGN PATH: the kept output rows are simulated
    through the original circuit and must reproduce the pinned outputs. The tiny bench is
    z = AND(DFF(a), b), so at T=3 the last-cycle output is a@1 & b@2: a row with those two bits
    set passes, the all-zero row fails, and the verdict folds BAD over ok over off."""
    from eval.rivals import rivals_recurrent as rr

    good = {"a@1": 1, "b@2": 1}
    bad = {}
    _stub_arm_with_cols(monkeypatch, [good, bad])
    g = rr.run_gpap(_tiny_bench(tmp_path), 3, 1, {"z": True}, [5], 100, "cpu", time_limit=1.0)
    assert g["sim_check"] == "BAD" and g["sim_check_bad"] == 1 and g["sim_check_rows"] == 2
    _stub_arm_with_cols(monkeypatch, [good, good])
    g = rr.run_gpap(_tiny_bench(tmp_path), 3, 1, {"z": True}, [5], 100, "cpu", time_limit=1.0)
    assert g["sim_check"] == "ok" and g["sim_check_bad"] == 0 and g["sim_check_rows"] == 2
    # no explicit target (never the harness's case) -> the check is OFF, never a silent pass
    _stub_arm(monkeypatch, samples=[[0, 1], [1, 0]])
    g = rr.run_gpap(_tiny_bench(tmp_path), 3, 1, None, [5], 100, "cpu", time_limit=1.0)
    assert g["sim_check"] == "off" and g["sim_check_rows"] == 0


def test_repeats_size_the_space_but_no_longer_cap_our_own_rate(tmp_path, monkeypatch):
    """THE CAP IS GONE, because what it corrected is gone (2026-09-04).

    It existed because `unique_per_s_est` was a 1,000-row window's distinct fraction applied to a
    whole run: where the window showed repeats, the implied |S| bounded what a run can deliver
    (each solution at most once) and the figure was capped at |S|/wall. Now the arm deduplicates
    every delivered row exactly, so the rate IS the measurement and there is nothing to correct.

    The cap was also defeated by the median that followed it: on both 2026-09-03 cells where it
    fired the capped seed was the slowest, so the median discarded it and the row published an
    uncapped number under a `unique_capped: true` flag.

    The space estimate stays — it feeds the `tiny` screen, and over the whole run it is a far
    better estimator than it was over 1,000 rows."""
    from eval.rivals import rivals_recurrent as rr

    rows = [[(i % 100) >> b & 1 for b in range(7)] for i in range(1000)]
    _stub_arm(monkeypatch, samples=rows, sps=1000.0, n_models=1000)
    g = rr.run_gpap(_tiny_bench(tmp_path), 3, 1, None, [5], 100, "cpu", time_limit=1.0)
    s = g["seeds"][0]
    assert s["unique_distinct"] == 100 and 80 < s["space_est"] < 130
    assert "unique_capped" not in g and "unique_capped" not in s, "the cap outlived its cause"
    assert g["unique_per_s_est"] == pytest.approx(1000.0 * 100 / 1000), \
        "the rate must be the measured distinct fraction, uncorrected"
    # below the row floor the repeat rate is too noisy to size a space: no estimate
    _stub_arm(monkeypatch, samples=[[0, 1], [1, 0], [0, 1], [0, 1]], sps=1000.0, n_models=4)
    g = rr.run_gpap(_tiny_bench(tmp_path), 3, 1, None, [5], 100, "cpu", time_limit=1.0)
    assert g["seeds"][0]["space_est"] is None
    assert g["unique_per_s_est"] == pytest.approx(1000.0 * 2 / 4)


def test_a_failed_demotic_miter_check_flags_the_row_instead_of_killing_the_stage():
    import inspect

    from eval.rivals import rivals_demotic

    src = inspect.getsource(rivals_demotic.main)
    i = src.index("write_unrolled_miter(")
    assert "except AssertionError" in src[i:i + 900], "the self-check must flag, not raise"
    assert '"pin_check_ok": pin_ok,' in src


def test_deduplication_is_offline_and_is_charged_to_no_rate(tmp_path, monkeypatch):
    """THE unique/s figure must cost the sampler nothing, on either clock.

    Two halves. The dedup itself runs after `sample_file` returns, so it is outside every clock
    the sampler keeps — `distinct_s` records what it cost, because "off the clock" is a claim and
    a claim needs evidence. The GATHER rides along inside the timed loop: `samples_per_s` already
    subtracts it (`phases["output"]`), but `sps_e2e_total` CARRIES it, which was a rounding error
    at 1,000 kept rows and is not at a million. Both rivals deduplicate their whole output off
    their own timer, so ours comes off both of ours or the e2e column pays for an instrument the
    rival is never charged for.
    """
    from eval.rivals import rivals_recurrent as rr

    rows = [[i & 1, (i >> 1) & 1] for i in range(400)]
    _stub_arm(monkeypatch, samples=rows, sps=1000.0, n_models=400,
              wall_total=10.0, proj_s=4.0)                     # 40% of the wall is delivery
    g = rr.run_gpap(_tiny_bench(tmp_path), 3, 1, None, [5], 100, "cpu", time_limit=1.0)
    seed = g["seeds"][0]

    assert seed["out_project_s"] == 4.0, "the projection wall must be recorded, not assumed"
    assert seed["distinct_s"] >= 0.0, "the offline dedup must be timed"
    # e2e is computed on the wall WITHOUT the gather: 400 / (10 - 4), not 400 / 10
    assert seed["sps_e2e"] == pytest.approx(400 / 6.0)
    assert seed["sps_e2e_with_delivery"] == pytest.approx(400 / 10.0), \
        "the library's delivery-carrying rate must still be recorded for audit"
    assert g["sps_e2e"] == pytest.approx(400 / 6.0)
    # and the steady-state rate is untouched by any of it
    assert g["sps"] == 1000.0


def test_how_the_batch_grant_was_reached_reaches_the_row():
    """Two ways the ladder can hand a rival less than the rule promises, both of which used to be
    invisible outside `batch_sweep`, which nothing read.

    `probe_truncated`: the walk spent its probe budget and fell back to the VRAM model's rung —
    the estimator measured under-granting DEMOTIC 1.5-3.9x — so a cell whose oversized rungs fail
    SLOWLY (timeouts) can settle for a smaller grant than an equally capable cell whose rungs fail
    fast (OOM). `budget_bound`: every rung timed out, so the cell is bounded by OUR wall rather
    than by the rival's capability, which is a different finding and a re-runnable one."""
    import eval.rivals.rival_common as rc

    rc._WARMED.add("probe-test")
    r, sweep = rc.run_at_batch(lambda b: {"error": "timeout 300s"}, 30_000,
                               log=lambda *_: None, probe_budget_s=0.0, tool="probe-test")
    assert r.get("budget_bound") and r.get("probe_truncated"), \
        "an all-timeout ladder must say so in the ROW, not only in the sweep"

    ok, _ = rc.run_at_batch(lambda b: {"unique_per_s": 1.0} if b <= 100_000
                            else {"error": "CUDA out of memory"}, 10_000, log=lambda *_: None)
    assert "budget_bound" not in ok and "probe_truncated" not in ok, \
        "a clean grant must not be flagged"


def test_every_paper_set_instance_has_a_target_even_on_their_default_branch():
    """A circuit their run.py handles through its catch-all `else` still needs an entry here, or
    the table-mode loop drops it with `if name not in TARGETS: continue`.

    That silent skip has now cost three instances: `b15` and `s208.1` in earlier sweeps, and it
    was waiting for `s38584` — the circuit their Table II's 8th ISCAS-89 row actually names, added
    to the paper set on 2026-09-04. The existing transcription test only checks circuits with a
    NAMED branch in their source, so a catch-all instance passes it while never being measured.
    This asserts the property that matters: everything we claim to score is scoreable."""
    from eval.rivals.rival_common import PAPER_INSTANCES
    from eval.rivals.rivals_recurrent import TARGETS

    missing = sorted(n for n in PAPER_INSTANCES["rcs"] if n not in TARGETS)
    assert not missing, (
        "these instances are in the rival's published set but have no pin transcription, so "
        f"table mode would skip them without failing: {missing}")


def test_the_rival_is_run_at_a_recorded_seed():
    """The rivals take `--seed` (their own `flags.py`, default 0) and neither harness passed it,
    so "the rival ran once" was true but the row could not say at WHICH initialization — and a
    second run would have silently re-measured the same one.

    One seed is the protocol (`rival_common.RIVAL_SEED`): the rival is measured in exactly one
    configuration, at its own default, and the asymmetry against GPAP's median-of-three is stated
    rather than hidden. What this asserts is that the choice is EXPLICIT — passed on the command
    line and recorded — not inherited from a default that could move under us."""
    import inspect

    from eval.rivals import rival_common, rivals_demotic, rivals_recurrent

    assert rival_common.RIVAL_SEED == 0, "the rivals' own default is 0; changing it is a protocol change"
    for mod in (rivals_recurrent, rivals_demotic):
        src = inspect.getsource(mod)
        assert '"--seed", str(RIVAL_SEED)' in src, \
            f"{mod.__name__} does not pass the rival its seed explicitly"
        assert '"seed_source"' in src, f"{mod.__name__} does not record which seed it measured"


# ---------------------------------------------------------------- the 2026-09-05 completion run

def test_a_failed_initialization_probe_is_cached_at_its_budget_and_retried_at_a_larger_one(tmp_path, monkeypatch):
    """The start_point probe is pin-independent, so the table stage re-paying a 1200 s failure
    the rare stage just measured is 20 minutes for the same answer — cache it WITH the budget.
    But a stuck instance may pass on a longer clock (that is what a budget means), so a larger
    budget must re-ask, and a settled start_point must still be honoured as before."""
    import json

    from eval.rivals import rivals_recurrent as rr

    bench = tmp_path / "x.bench"
    bench.write_text("INPUT(a)\nOUTPUT(y)\ny = DFF(a)\n")
    cache = str(tmp_path / "cache")
    calls = []

    def fake_run(rival, bench_file, key, T, steps, batch, start_point, timeout=None, **kw):
        calls.append((start_point, timeout))
        return {"error": "timeout 1s"}

    monkeypatch.setattr(rr, "run_recurrent", fake_run)
    sp, why = rr.probe_start_point("rival", str(bench), "x", 25, budget=300.0, cache_dir=cache)
    assert sp is None and "300s initialization budget" in why
    assert len(calls) == 1
    stored = [json.load(open(p)) for p in (tmp_path / "cache").glob("*.json")]
    assert stored and stored[0]["value"]["failed_at_budget"] == 300.0
    # same budget: served from the cache, no subprocess
    sp2, why2 = rr.probe_start_point("rival", str(bench), "x", 25, budget=300.0, cache_dir=cache)
    assert sp2 is None and why2.endswith("(cached)") and len(calls) == 1
    # a larger budget re-asks
    sp3, _ = rr.probe_start_point("rival", str(bench), "x", 25, budget=1200.0, cache_dir=cache)
    assert sp3 is None and len(calls) == 2
    # the same budget again is served the cached failure even if the tool would now pass
    monkeypatch.setattr(rr, "run_recurrent", lambda *a, **k: {"unique": 1})
    sp_same, why_same = rr.probe_start_point("rival", str(bench), "x", 25, budget=1200.0,
                                             cache_dir=cache)
    assert sp_same is None and why_same.endswith("(cached)") and len(calls) == 2
    # a larger budget re-asks; a settled start_point is stored as an int and honoured
    sp4, _ = rr.probe_start_point("rival", str(bench), "x", 25, budget=2400.0, cache_dir=cache)
    assert sp4 == 0
    sp5, _ = rr.probe_start_point("rival", str(bench), "x", 25, budget=10.0, cache_dir=cache)
    assert sp5 == 0, "a settled probe is the answer at any budget"


def test_the_calibration_measures_the_papers_configuration_too():
    """Three review rounds each found a knob the text cited and nobody ran: the tie-break
    compared each script's `num_steps` against the argparse default only, so Recurrent's paper
    (5 iterations, lr 50) and DEMOTIC's (10 iterations) were never in the candidate set. The
    calibration now probes every value the tool's own artifacts give the knob — script, argparse,
    PAPER — and the paper's paired setting; the campaign still RUNS the script configuration."""
    from eval.rivals.rival_common import RIVAL_PAPER_CONFIG, RIVAL_STEPS, resolve_lr, resolve_steps
    from eval.rivals.steps_calibration import candidates

    assert RIVAL_PAPER_CONFIG["recurrent"]["steps"] == 5 and RIVAL_PAPER_CONFIG["recurrent"]["lr"] == 50.0
    assert RIVAL_PAPER_CONFIG["demotic"]["steps"] == 10 and RIVAL_PAPER_CONFIG["demotic"]["lr"] == 15.0
    pinned, _src, vals = candidates("recurrent", "steps")
    assert pinned == 1 and vals == {1, 5, 10}
    pinned, _src, vals = candidates("demotic", "steps")
    assert pinned == 1 and vals == {1, 10}
    _p, _s, vals = candidates("recurrent", "lr")
    assert vals == {15.0, 50.0}
    # the protocol runs the SHIPPED configuration, unchanged
    assert resolve_steps("recurrent")[0] == RIVAL_STEPS["recurrent"][0] == 1
    assert resolve_lr("recurrent")[0] == 15.0 and resolve_steps("demotic")[0] == 1


def test_the_init_budget_is_its_own_number_and_the_driver_passes_it():
    """1200 s, separate from the run timeout, on every Recurrent stage of the Colab driver."""
    from tests.conftest import REPO_ROOT
    from eval.rivals import rivals_recurrent as rr

    assert rr.INIT_BUDGET_S == 1200.0
    drv = open(os.path.join(REPO_ROOT, "eval", "colab", "colab_rivals_l4.sh")).read()
    assert drv.count('--init-budget "$INIT_BUDGET"') == 4, "every Recurrent stage must pass it"
    assert 'INIT_BUDGET="${INIT_BUDGET:-1200}"' in drv
    assert 'VARIANTS_RCS="${VARIANTS_RCS:-abc}"' in drv and 'VARIANTS_DEM="${VARIANTS_DEM:-abc}"' in drv


def test_retry_errors_re_runs_only_rows_not_measured_under_the_current_budget():
    """`--retry-errors` used to drop EVERY error row at process start and re-probe it. With the
    probe cache packed only at stage end and a ~1 h VM, each fresh VM re-paid the previous VM's
    1,200 s failures and was reclaimed before reaching a new instance — the 2026-09-05 completion
    run stalled on b17/b18/b19. A row whose reason names the CURRENT budget is a finished
    measurement; only rows from another budget or an environmental failure are re-run."""
    from eval.rivals.rivals_recurrent import settled_error

    cur = "did not initialize: a 1-step start_point probe (sp=2) did not complete within the 1200s initialization budget"
    old = "did not initialize: a 1-step start_point probe (sp=1) did not complete within the 300s initialization budget"
    assert settled_error(cur, 1200.0, 1200.0), "measured under this budget: keep"
    assert settled_error(cur + " (cached)", 1200, 1200)
    assert not settled_error(old, 1200.0, 1200.0), "measured under the OLD budget: re-run"
    assert settled_error("no gradient path at any start_point<=10", 1200.0, 1200.0)
    assert settled_error("timeout 1200.0s", 1200.0, 1200.0), "a run timeout at the current cap"
    assert not settled_error("timeout 300.0s", 1200.0, 1200.0)
    assert not settled_error("no results csv", 1200.0, 1200.0), "environmental: re-run"
    assert not settled_error("", 1200.0, 1200.0) and not settled_error(None, 1200.0, 1200.0)


def test_a_known_oom_rung_is_not_walked_twice(tmp_path):
    """A rung that OOM'd is deterministic for the circuit, depth and card, and on a 20M-node
    circuit each doomed rung above the grant costs a full model build: the 2026-09-05 completion
    run spent whole VM lifetimes re-walking seven of them after every reclaim and never reached
    the run. The ceiling is cached; the next walk starts below it; the grant is unchanged."""
    from eval.rivals.rivals_recurrent import known_ladder, remember_oom_ceiling

    bench = tmp_path / "x.bench"; bench.write_text("INPUT(a)\nOUTPUT(y)\ny = DFF(a)\n")
    cache = str(tmp_path / "cache")
    ladder, prior = known_ladder(cache, str(bench), 25, (1_000, 3_000, 10_000, 30_000))
    assert ladder == (1_000, 3_000, 10_000, 30_000) and prior is None, "no ceiling known yet"
    sweep = [{"batch": 30_000, "error": "torch.OutOfMemoryError: CUDA out of memory"},
             {"batch": 10_000, "error": "timeout 1200.0s"},          # a timeout is NOT a ceiling
             {"batch": 3_000, "error": None, "unique_per_s": 5.0}]
    assert remember_oom_ceiling(cache, str(bench), 25, sweep) == 30_000
    ladder, prior = known_ladder(cache, str(bench), 25, (1_000, 3_000, 10_000, 30_000))
    assert ladder == (1_000, 3_000, 10_000) and prior == 30_000, "the OOM rung is skipped, the timeout rung is not"
    # a later walk can only TIGHTEN the ceiling
    assert remember_oom_ceiling(cache, str(bench), 25, [{"batch": 10_000, "error": "OutOfMemoryError"}], prior=30_000) == 10_000
    assert known_ladder(cache, str(bench), 25, (1_000, 3_000, 10_000, 30_000))[0] == (1_000, 3_000)
    # a different depth is a different circuit
    assert known_ladder(cache, str(bench), 50, (1_000, 3_000, 10_000, 30_000))[1] is None
