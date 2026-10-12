"""Tests for the rival harnesses in `eval/rivals/`: the Recurrent target table re-derived from the
rival's own `run.py`, the oversize guard, the start_point probe, the batch ladder and grants, the
GPAP and CMSGen arms, and the cache keys. The transcription tests skip when the rival checkout
is absent (`eval/rivals/rivals_setup.sh`) but fail on one they cannot parse.
"""
import ast
import os
import re

import pytest

from gpap import paths
from tests.conftest import REPO_ROOT, SEQ_DIR

# Through `paths.src_dir()`: `rivals_setup.sh` installs into `$GPAP_HOME/src`, so a hard-coded
# `<repo>/.gpap` would make the tests below skip whenever `GPAP_HOME` is set.
RIVAL_RUN_PY = str(paths.src_dir() / "Recurrent-CircuitSAT-Sampler" / "src" / "pytorch" / "run.py")


def _their_branches(src):
    """[(names, target_idx, values)] for every `module_name in [...]` branch in their run.py.
    Values come from `torch.ones/zeros(batch, k)` plus optional `target[:, SLICE] = V.` lines."""
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
    """Their final `else:` branch: the target (idx [0,15,31,63,-1]) unlisted names fall into."""
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
    # A present checkout that yields nothing means the parse broke, not that the table is gone.
    assert len(branches) >= 8, \
        f"parsed only {len(branches)} branches from {RIVAL_RUN_PY} — the grammar changed"
    return branches


def _n_outputs(stem):
    p = os.path.join(SEQ_DIR, f"{stem}.bench")
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
            # the catch-all indexes [0,15,31,63,-1], so an entry relying on it needs >= 64
            # outputs or their run raises IndexError
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
    """A committed circuit with a branch in their table but no TARGETS entry would be skipped
    by the harness with only a log line."""
    branches = _load_source()
    from eval.rivals.rivals_recurrent import TARGETS

    covered = {n for names, _, _ in branches for n in names}
    seq = SEQ_DIR
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
    """A miter over k outputs is a tree of narrow ANDs, not one k-input AND: Tseitin makes that a
    (k+1)-literal clause, and the packed 8|24 envelope caps clause length at 255, so a
    wide-output circuit would be refused by the engine."""
    from eval.rivals.unroll_to_bench import _MITER_FANIN, and_tree

    # narrow: unchanged, so existing instances encode bit-identically
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
    """An instance too large to Tseitin-encode is refused from the .bench size before anything is
    built. Running out of host memory kills the process with no row, and every later instance in
    the sweep goes unmeasured."""
    from eval.rivals.rival_common import MAX_UNROLLED_GATES, oversize_reason

    # `avail_bytes` is pinned so the verdict does not depend on this machine's free RAM.
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
    # ...and the same circuit at a shallow depth is fine: the unroll is what is oversized
    assert oversize_reason(str(big), 1, avail_bytes=BIG_BOX) is None


def test_an_instance_too_big_for_THIS_box_is_refused_before_it_is_built(tmp_path):
    """Besides the fixed gate cap, the guard refuses what will not fit in this machine's host
    RAM and gives the memory arithmetic as the reason. b18's shape at T=25 needs about 10 GiB."""
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
        assert oversize_reason(os.path.join(SEQ_DIR,
                                            "b13.bench"), 25) is None
    finally:
        rc._available_bytes = saved


# --------------------------------------------------------------------------- the start_point probe
#
# The probe decides whether a Recurrent cell is a comparison or an error row. Its three failure
# modes are reported apart, and only one is a claim about the rival
# (docs/findings/RIVAL_FAIRNESS_FINDINGS.md §11). These tests need no checkout.

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
    """The start_point search is the rival's initialization and gets one budget, the run
    timeout. Each probe runs with what is left, so neither one probe nor their sum exceeds it."""
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
    """The only outcome that is a finding about the rival: it refused every depth reached."""
    sp, why, _seen = _probe(monkeypatch, {}, max_sp=4)          # no-grad everywhere
    assert sp is None
    assert why == "no gradient path at any start_point<=4"


def test_a_timeout_is_reported_as_a_timeout_with_its_budget(monkeypatch):
    """A probe timeout does not read as 'no gradient path', and it names the budget so a reader
    can see whose number it is."""
    sp, why, seen = _probe(monkeypatch, {1: "timeout 300.0s"}, run_timeout=300.0)
    assert sp is None
    assert "gradient" not in why, f"a timeout must not be reported as a gradient-path refusal: {why}"
    assert "300s" in why and "budget" in why, why
    # the walk stops there: a larger start_point pays the same model build
    assert [s for s, _t in seen] == [0, 1], f"the walk continued past a timeout: {seen}"


def test_an_exhausted_budget_reads_as_did_not_initialize(monkeypatch):
    """The third outcome: the budget ran out before any probe completed. The row says the rival
    did not initialize and names the budget."""
    sp, why, seen = _probe(monkeypatch, {}, budget=-1.0)       # budget already spent
    assert sp is None and seen == [], (sp, why, seen)
    assert why.startswith("did not initialize"), why
    assert "gradient" not in why and "OUR" not in why, why
    assert "budget" in why


def test_the_ladder_measures_the_largest_rung_that_completes():
    """`run_at_batch` measures one configuration per instance: the largest rung the rival
    completes, walked down from the top. Rungs are rejected on feasibility only (OOM, driver
    refusal, wall budget), never on throughput, so it is not a best-of-N. Shared by every rival
    harness (docs/findings/RIVAL_FAIRNESS_FINDINGS.md §2)."""
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


def test_each_rival_expands_paper_to_its_own_published_instances():
    """`paper` expands to the rival's published set, defined once in rival_common. The tables no
    longer read it; the harnesses still expand the keyword."""
    from eval.rivals.rival_common import PAPER_INSTANCES, expand_names

    assert len(PAPER_INSTANCES["demotic"]) == 11 and len(PAPER_INSTANCES["rcs"]) == 16
    assert expand_names(["paper"], "rcs") == sorted(PAPER_INSTANCES["rcs"])
    assert expand_names(["b03", "paper"], "demotic") == ["b03"] + sorted(PAPER_INSTANCES["demotic"])
    assert expand_names(["b03"], "rcs") == ["b03"]           # anything else passes through


def test_the_gpap_arm_cache_key_covers_the_python_sampling_layer(tmp_path, monkeypatch):
    """A cached GPAP-arm result misses when `gpap/sampling/` changes, not only when libgpap's
    `gpap_build_id` does: the portfolio, defaults and preprocessing there also decide the number."""
    import shutil

    from eval.rivals import rival_common as rc

    f = tmp_path / "x.cnf"
    f.write_text("p cnf 1 1\n1 0\n")
    task = {"harness": "t", "cycles": 25}
    before = rc.gpap_cache_key(str(f), task)

    pipe = os.path.join(REPO_ROOT, "gpap", "sampling", "pipeline.py")
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
    """`--portfolio` selects GPAP's arm, so the row records which one ran. The default is the
    adaptive bandit while the rivals and CMSGen run one fixed config; the row discloses this via
    `portfolio` + `adaptive` (RIVAL_FAIRNESS_FINDINGS.md §10, §12)."""
    from gpap.sampling.pipeline import portfolio
    from eval.rivals.rivals_recurrent import GPAP_PORTFOLIO

    assert len(portfolio(GPAP_PORTFOLIO)) > 1, \
        "the default is documented as the adaptive bandit; if it is pinned now, say so here"
    assert len(portfolio("b8")) == 1, "b8 must remain a genuine single-config escape hatch"


def test_both_rival_harnesses_share_the_pin():
    """rivals_demotic imports the portfolio constant from rivals_recurrent rather than restating
    it."""
    import inspect

    from eval.rivals import rivals_demotic, rivals_recurrent

    src = inspect.getsource(rivals_demotic)
    assert "GPAP_PORTFOLIO" in src, "the DEMOTIC harness must reference the shared default"
    assert rivals_demotic.GPAP_PORTFOLIO is rivals_recurrent.GPAP_PORTFOLIO


def test_the_portfolio_is_part_of_the_arm_cache_key():
    """Runs that differ only in `--portfolio` are different measurements, so the portfolio must
    be in each harness's task dict (read from source); otherwise a `--portfolio b1` run would be
    served the cached `b8` result."""
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
    """Their `export_results` writes `sat_{dataset_str}_…` with the dot stripped from the instance
    name (`s208.1` -> `s2081`). Both the clean-up and the read-back globs must strip it, and a
    multi-hit read-back must pick this call's rung."""
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
        # `sps_e2e_total` includes the output projection; the arm recomputes its e2e rate
        # without it (see the offline-dedup test).
        return {"samples_per_s": sps, "sps_e2e_total": n_models / wall_total,
                "success_rate": 0.5, "wall_total_s": wall_total,
                "phases": {"sample": wall_total - proj_s, "output": proj_s},
                "n_models": n_models, "sound_check": "ok",
                "samples": np.asarray(samples, dtype=np.uint8)}

    monkeypatch.setattr(abc_backend, "build_cnf_abc", fake_abc)
    monkeypatch.setattr(pipeline, "sample_file", fake_sample_file)
    return seen


def test_the_gpap_arm_encodes_with_abc_fast_and_says_so(tmp_path, monkeypatch):
    """The GPAP arm of every rival row encodes with abc:fast, the formula Table 1 runs, and the
    row records it (RIVAL_FAIRNESS_FINDINGS.md §13)."""
    from eval.rivals import rivals_recurrent as rr

    seen = _stub_arm(monkeypatch, samples=[[0, 1], [1, 0], [0, 1]])
    g = rr.run_gpap(_tiny_bench(tmp_path), 3, 1, None, [5], 100, "cpu", time_limit=1.0)
    assert rr.GPAP_ENCODER == "abc:fast"
    assert seen["encoder_call"]["opt"] == "fast" and seen["encoder_call"]["pin_outputs"] == 1
    assert g["encoder"] == "abc:fast" and g["encode_s"] >= 0.0
    assert g["portfolio"] == rr.GPAP_PORTFOLIO


def test_the_gpap_arm_runs_no_coverage_pass_and_measures_distinctness_on_its_own_rows(
        tmp_path, monkeypatch):
    """`unique_per_s_est` is samples/s times the exact distinct fraction of every row the timed
    run delivered (`out_cap=UNIQUE_ROWS`), as both rivals deduplicate their whole output. The
    coverage pass is off (`n_cov=0`): it rebuilds full models on the host and can run out of
    memory."""
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
    """A run that kept no rows has no distinct fraction: None, with a status
    `paper_tables.unique_unmeasured` recognizes. Never 0.0, which reads as 'GPAP produced
    nothing'."""
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
    """Both terms of `oversize_reason` model the native Tseitin build. Under abc:fast every seq
    instance fits a 13 GB host, so the guard passes everything for abc:fast and keeps refusing
    for Tseitin, the default."""
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
    """CMSGen on a rival row's own CNF uses the `gpap bench` runner: `--nowrite` (no emission),
    the sampling budget passed through so `--maxtime` makes the count exact, and the bench
    driver's `cms_*` keys, so bench-row readers read a rival row unchanged."""
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
    """A crashed CMSGen run is `cms_error` beside a count of 0, never 'CMSGen sampled nothing',
    which would score as our win."""
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
    """Both of our arms run one file: the harness builds the CNF once (`build_arm_cnf`), and
    `run_gpap(cnf=...)` samples that path and leaves it for `cmsgen_arm`. Without `cnf=` the arm
    builds and removes its own temporary."""
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
    """Both harness mains have a `--cmsgen` flag, one `build_arm_cnf` into a path, and
    `run_gpap(cnf=...)` and `cmsgen_arm` on that path, so the two comparisons share one encoding
    and pin. The row also records the host CPU the CMSGen ratio needs."""
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
    """The rare-mode miter's P[MITER=1] must equal the instance's p_accept. An inverted miter
    (P ~ 1 - p) must fail the check, or the check proves nothing."""
    from eval.rivals import rivals_recurrent as rr

    src = os.path.join(SEQ_DIR, "b06.bench")
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
    """Both harness mains run the constraint-alignment check and write `pin_check_ok`, which
    Table 2's `misaligned` bucket reads."""
    import inspect

    from eval.rivals import rival_common, rivals_demotic, rivals_recurrent

    assert hasattr(rival_common, "native_pin_check"), "one home for the check"
    for mod in (rivals_recurrent, rivals_demotic):
        src = inspect.getsource(mod.main)
        assert "native_pin_check(" in src, mod.__name__
        assert '"pin_check_ok"' in src, mod.__name__
    assert "miter_self_check(" in inspect.getsource(rivals_recurrent.miter_bench)


def _stub_arm_with_cols(monkeypatch, samples_by_label, sps=1000.0):
    """Like `_stub_arm`, but the fake pipeline also returns the kept rows' column ids and wire
    map, as `sample_file` does, so the arm's simulate check can run."""
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
    """The arm simulates its kept output rows through the original circuit (encoder-independent
    soundness). The tiny bench is z = AND(DFF(a), b), so at T=3 the last-cycle output is
    a@1 & b@2: a row with both bits set passes, the all-zero row fails, and the verdict folds BAD
    over ok over off."""
    from eval.rivals import rivals_recurrent as rr

    good = {"a@1": 1, "b@2": 1}
    bad = {}
    _stub_arm_with_cols(monkeypatch, [good, bad])
    g = rr.run_gpap(_tiny_bench(tmp_path), 3, 1, {"z": True}, [5], 100, "cpu", time_limit=1.0)
    assert g["sim_check"] == "BAD" and g["sim_check_bad"] == 1 and g["sim_check_rows"] == 2
    _stub_arm_with_cols(monkeypatch, [good, good])
    g = rr.run_gpap(_tiny_bench(tmp_path), 3, 1, {"z": True}, [5], 100, "cpu", time_limit=1.0)
    assert g["sim_check"] == "ok" and g["sim_check_bad"] == 0 and g["sim_check_rows"] == 2
    # no explicit target (never the harness's case): the check is off, not a silent pass
    _stub_arm(monkeypatch, samples=[[0, 1], [1, 0]])
    g = rr.run_gpap(_tiny_bench(tmp_path), 3, 1, None, [5], 100, "cpu", time_limit=1.0)
    assert g["sim_check"] == "off" and g["sim_check_rows"] == 0


def test_repeats_size_the_space_but_no_longer_cap_our_own_rate(tmp_path, monkeypatch):
    """The repeat rate still estimates the solution space (it feeds the `tiny` screen) but no
    longer caps `unique_per_s_est`: the arm deduplicates every delivered row exactly, so the rate
    is the measurement."""
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
    """The unique/s figure costs the sampler nothing on either clock, as both rivals deduplicate
    off their own timer. The dedup runs after `sample_file` returns and is timed in
    `distinct_s`. The row gather runs inside the timed loop: `samples_per_s` already excludes it
    (`phases["output"]`), and the arm's e2e rate subtracts it too."""
    from eval.rivals import rivals_recurrent as rr

    rows = [[i & 1, (i >> 1) & 1] for i in range(400)]
    _stub_arm(monkeypatch, samples=rows, sps=1000.0, n_models=400,
              wall_total=10.0, proj_s=4.0)                     # 40% of the wall is delivery
    g = rr.run_gpap(_tiny_bench(tmp_path), 3, 1, None, [5], 100, "cpu", time_limit=1.0)
    seed = g["seeds"][0]

    assert seed["out_project_s"] == 4.0, "the projection wall must be recorded, not assumed"
    assert seed["distinct_s"] >= 0.0, "the offline dedup must be timed"
    # e2e is computed on the wall without the gather: 400 / (10 - 4), not 400 / 10
    assert seed["sps_e2e"] == pytest.approx(400 / 6.0)
    assert seed["sps_e2e_with_delivery"] == pytest.approx(400 / 10.0), \
        "the library's delivery-carrying rate must still be recorded for audit"
    assert g["sps_e2e"] == pytest.approx(400 / 6.0)
    # the steady-state rate is unchanged
    assert g["sps"] == 1000.0


def test_how_the_batch_grant_was_reached_reaches_the_row():
    """Two ways the ladder can grant a rival less than the rule promises are flagged on the row,
    not only in `batch_sweep`: `probe_truncated` (the walk spent its probe budget and fell back to
    the VRAM model's rung) and `budget_bound` (every rung timed out, so our wall bounds the cell
    rather than the rival's capability)."""
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
    """Every instance in the rival's published set has a TARGETS entry, including those their
    run.py handles through its catch-all `else`. Table mode skips a name not in TARGETS without
    failing, and the transcription test only covers named branches."""
    from eval.rivals.rival_common import PAPER_INSTANCES
    from eval.rivals.rivals_recurrent import TARGETS

    missing = sorted(n for n in PAPER_INSTANCES["rcs"] if n not in TARGETS)
    assert not missing, (
        "these instances are in the rival's published set but have no pin transcription, so "
        f"table mode would skip them without failing: {missing}")


def test_the_rival_is_run_at_a_recorded_seed():
    """Both harnesses pass the rival `--seed` explicitly and record its source. The protocol is
    one rival seed (`rival_common.RIVAL_SEED`, the rivals' own default 0), passed on the command
    line rather than inherited from a default that could move."""
    import inspect

    from eval.rivals import rival_common, rivals_demotic, rivals_recurrent

    assert rival_common.RIVAL_SEED == 0, "the rivals' own default is 0; changing it is a protocol change"
    for mod in (rivals_recurrent, rivals_demotic):
        src = inspect.getsource(mod)
        assert '"--seed", str(RIVAL_SEED)' in src, \
            f"{mod.__name__} does not pass the rival its seed explicitly"
        assert '"seed_source"' in src, f"{mod.__name__} does not record which seed it measured"


# ---------------------------------------------------------------- init budget, retries, OOM cache

def test_a_failed_initialization_probe_is_cached_at_its_budget_and_retried_at_a_larger_one(tmp_path, monkeypatch):
    """The start_point probe is pin-independent, so a failure is cached with its budget and the
    next stage does not re-pay it. A larger budget re-asks, since a stuck instance may pass on a
    longer clock; a settled start_point is honoured at any budget."""
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
    """The steps/lr calibration probes every value the tool's own artifacts give a knob (script,
    argparse, paper) and the paper's paired setting; the campaign still runs the script
    configuration."""
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
    # the protocol runs the shipped configuration
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
    """`--retry-errors` keeps an error row whose reason names the current budget (a finished
    measurement) and re-runs only rows from another budget or an environmental failure, so a
    fresh VM does not re-pay every recorded init failure."""
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
    """An OOM rung is deterministic for the circuit, depth and card, and on a large circuit each
    costs a full model build. The OOM ceiling is cached and the next walk starts below it; the
    grant is unchanged."""
    from eval.rivals.rivals_recurrent import known_ladder, remember_oom_ceiling

    bench = tmp_path / "x.bench"; bench.write_text("INPUT(a)\nOUTPUT(y)\ny = DFF(a)\n")
    cache = str(tmp_path / "cache")
    ladder, prior = known_ladder(cache, str(bench), 25, (1_000, 3_000, 10_000, 30_000))
    assert ladder == (1_000, 3_000, 10_000, 30_000) and prior is None, "no ceiling known yet"
    sweep = [{"batch": 30_000, "error": "torch.OutOfMemoryError: CUDA out of memory"},
             {"batch": 10_000, "error": "timeout 1200.0s"},          # a timeout is not a ceiling
             {"batch": 3_000, "error": None, "unique_per_s": 5.0}]
    assert remember_oom_ceiling(cache, str(bench), 25, sweep) == 30_000
    ladder, prior = known_ladder(cache, str(bench), 25, (1_000, 3_000, 10_000, 30_000))
    assert ladder == (1_000, 3_000, 10_000) and prior == 30_000, "the OOM rung is skipped, the timeout rung is not"
    # a later walk can only tighten the ceiling
    assert remember_oom_ceiling(cache, str(bench), 25, [{"batch": 10_000, "error": "OutOfMemoryError"}], prior=30_000) == 10_000
    assert known_ladder(cache, str(bench), 25, (1_000, 3_000, 10_000, 30_000))[0] == (1_000, 3_000)
    # a different depth is a different circuit
    assert known_ladder(cache, str(bench), 50, (1_000, 3_000, 10_000, 30_000))[1] is None


# --------------------------------------------------------------- protocol v1.0: the grant record

def test_the_ladder_top_is_above_the_cards_reach_and_the_walk_starts_there():
    """The ladder top (at least 30,000,000) sits above what the card completes, so the card and
    not the ladder bounds each grant (RIVAL_FAIRNESS_FINDINGS.md §16e). The walk starts at the
    top, and the rungs are the decade-and-triple ladder."""
    from eval.rivals.rival_common import BATCH_LADDER, run_at_batch

    assert BATCH_LADDER[-1] >= 30_000_000
    assert list(BATCH_LADDER) == sorted(BATCH_LADDER)
    for a, b in zip(BATCH_LADDER, BATCH_LADDER[1:]):
        assert b / a in (3.0, 10 / 3), "not the decade-and-triple ladder"
    calls = []
    run_at_batch(lambda b: (calls.append(b), {"unique_per_s": 1.0})[1], 1_000,
                 log=lambda *_: None)
    assert calls == [BATCH_LADDER[-1]], "the walk must start at the (new) top"


def test_a_recorded_grant_pins_the_walk_and_only_descends_when_this_card_cannot_finish():
    """With a grants file (eval/rivals/extract_grants.py), `pinned_ladder` measures the recorded
    rung first and offers nothing above it, so a reproduction pays no walk. On a card that cannot
    finish that rung the walk descends as usual, and `grant_source` says which happened."""
    import eval.rivals.rival_common as rc
    from eval.rivals.rival_common import grant_source, pinned_ladder, run_at_batch

    assert pinned_ladder((1_000, 3_000, 10_000, 30_000), 10_000) == (1_000, 3_000, 10_000)
    assert pinned_ladder((1_000, 3_000, 10_000), 5_000) == (1_000, 3_000, 5_000), \
        "a pinned value off the ladder is its own top rung"
    calls = []
    ok, sweep = run_at_batch(lambda b: (calls.append(b), {"unique_per_s": 1.0, "batch": b})[1],
                             1_000, ladder=pinned_ladder(rc.BATCH_LADDER, 3_000),
                             log=lambda *_: None)
    assert calls == [3_000] and [s["batch"] for s in sweep] == [3_000]
    assert grant_source(3_000, ok, "pinned from g.json") == "pinned from g.json"
    calls.clear()
    small_card = lambda b: (calls.append(b), {"error": "CUDA out of memory"}  # noqa: E731
                            if b > 1_000 else {"unique_per_s": 1.0, "batch": b})[1]
    r, sweep = run_at_batch(small_card, 1_000, ladder=pinned_ladder(rc.BATCH_LADDER, 3_000),
                            log=lambda *_: None)
    assert calls == [3_000, 1_000] and r["batch"] == 1_000
    assert "did not complete on this card, walked down" in grant_source(3_000, r, "pinned from g.json")
    assert grant_source(None, r, None) == "walked down from the ladder top"


def test_the_recurrent_harness_pins_both_the_rung_and_the_start_point_from_a_grant(tmp_path, monkeypatch):
    """Given a grant, `run_recurrent_once` skips the start_point probe and the ladder walk: it
    runs once at the recorded rung and start_point and stamps the row. Without one, it walks."""
    import eval.rivals.rival_common as rc
    import eval.rivals.rivals_recurrent as rr

    rc._WARMED.add("recurrent")                      # no warm-up subprocess in a unit test
    calls = []
    monkeypatch.setattr(rr, "run_recurrent",
                        lambda rival, f, key, T, steps, batch, sp, timeout=0, lr=None:
                        (calls.append((batch, sp)), {"unique_per_s": 1.0, "batch": batch,
                                                     "start_point": sp})[1])
    bench = tmp_path / "x.bench"
    bench.write_text("INPUT(a)\nOUTPUT(y)\ny = DFF(a)\n")
    grant = {"batch": 3_000, "start_point": 2, "source": "pinned from g.json"}
    r = rr.run_recurrent_once("rival", str(bench), "x", 25, 1, 1_000, grant["start_point"],
                              cache_dir=str(tmp_path / "c"), grant=grant)
    assert calls == [(3_000, 2)], "one run, at the recorded rung and start_point"
    assert r["grant_source"] == "pinned from g.json" and r["batch_sweep"][0]["batch"] == 3_000
    calls.clear()
    r = rr.run_recurrent_once("rival", str(bench), "x", 25, 1, 1_000, 2,
                              cache_dir=str(tmp_path / "c"))
    assert calls[0][0] == rc.BATCH_LADDER[-1] and r["grant_source"] == "walked down from the ladder top"


def test_extract_grants_reads_the_grant_off_the_rows_and_never_pins_a_failure(tmp_path):
    """The grants file is read off the campaign rows (`batch`, `batch_sweep`, Recurrent's
    `start_point`), keyed the way the harnesses look it up; a DEMOTIC native row is a
    combinational circuit at T=1 whatever its file header says. A rival failure is recorded
    without a batch, so `lookup_grant` returns None and the harness re-measures it."""
    import json

    from eval.rivals.extract_grants import extract
    from eval.rivals.rival_common import load_grants, lookup_grant

    dem = {"mode": "native", "circuit_type": "comb", "cycles": 25, "rows": [
        {"name": "c432", "variant": "abc", "mode": "native", "cycles": 25,
         "gpu": {"name": "NVIDIA L4"},
         "demotic": {"batch": 3_000_000, "unique_per_s": 8.1e6,
                     "batch_sweep": [{"batch": 10_000_000, "error": "torch.OutOfMemoryError"},
                                     {"batch": 3_000_000, "error": None}]}}]}
    rcs = [{"name": "b06", "variant": "abc", "mode": "table", "cycles": 25,
            "gpu": {"name": "NVIDIA L4"},
            "rcs": {"batch": 3_000_000, "start_point": 1, "unique_per_s": 1.1e6,
                    "batch_sweep": [{"batch": 3_000_000, "error": None}]}},
           {"name": "b17", "variant": "abc", "mode": "table", "cycles": 25,
            "rcs": {"error": "did not initialize: ... 1200s initialization budget"}}]
    (tmp_path / "rivals_demotic_native.json").write_text(json.dumps(dem))
    (tmp_path / "rivals_recurrent_table.json").write_text(json.dumps(rcs))
    (tmp_path / "rivals_steps_calibration.json").write_text(json.dumps({"tools": {}}))
    g = extract(str(tmp_path))
    assert g["gpu"] == "NVIDIA L4"
    assert g["demotic"]["c432|native|1|abc"]["batch"] == 3_000_000, "native comb = depth 1"
    assert g["demotic"]["c432|native|1|abc"]["oom_min"] == 10_000_000
    assert g["recurrent"]["b06|table|25|abc"] == {
        "batch": 3_000_000, "rungs_tried": [3_000_000], "oom_min": None,
        "unique_per_s": 1.1e6, "stage": "recurrent_table", "start_point": 1}
    assert g["recurrent"]["b17|table|25|abc"]["batch"] is None
    path = tmp_path / "g.json"
    path.write_text(json.dumps(g))
    grants = load_grants(str(path))
    assert lookup_grant(grants, "recurrent", "b06", "table", 25) == {
        "batch": 3_000_000, "start_point": 1, "source": "pinned from g.json"}
    assert lookup_grant(grants, "recurrent", "b17", "table", 25) is None, "a failure is not pinned"
    assert lookup_grant(grants, "demotic", "c432", "native", 1)["start_point"] is None
    assert lookup_grant({}, "demotic", "c432", "native", 1) is None
    (tmp_path / "bad.json").write_text("{}")
    with pytest.raises(ValueError):
        load_grants(str(tmp_path / "bad.json"))


def test_the_committed_grants_file_is_the_campaigns_and_every_pinned_rung_is_on_the_ladder():
    """eval/rivals/grants/<campaign>.json is what a reproduction pins to; it must load, name
    its card, and pin only rungs the ladder has."""
    import glob
    import os

    from eval.rivals.rival_common import BATCH_LADDER, load_grants
    from tests.conftest import REPO_ROOT

    files = glob.glob(os.path.join(REPO_ROOT, "eval", "rivals", "grants", "*.json"))
    assert files, "no committed grants file"
    for f in files:
        g = load_grants(f)
        assert g.get("gpu") and g.get("campaign")
        pinned = [e for t in ("demotic", "recurrent") for e in g[t].values() if e.get("batch")]
        assert len(pinned) >= 50, f"{f}: only {len(pinned)} pinned cells"
        assert all(e["batch"] in BATCH_LADDER for e in pinned)
        assert all(e.get("start_point") is not None for e in g["recurrent"].values()
                   if e.get("batch")), "a pinned Recurrent cell must carry its start_point"


def test_a_rival_that_dies_with_nothing_on_either_stream_is_a_refusal_not_a_crash(tmp_path, monkeypatch):
    """A Recurrent subprocess that exits non-zero with empty stdout and stderr (a SIGKILL leaves
    no traceback) is a refusal: the row says how it died and the ladder steps down past it,
    rather than crashing on the empty output."""
    import subprocess
    from eval.rivals import rivals_recurrent as rr
    from eval.rivals.rival_common import run_at_batch

    rival = tmp_path / "rival"
    for d in ("data", "src/results", "script"):
        (rival / d).mkdir(parents=True)
    bench = tmp_path / "s.bench"
    bench.write_text("INPUT(a)\nOUTPUT(a)\n")

    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, returncode=-9, stdout="", stderr="")

    monkeypatch.setattr(rr.subprocess, "run", fake_run)
    r = rr.run_recurrent(str(rival), str(bench), "s", 50, 1, 1_000_000, 0, timeout=5, lr=15)
    assert "error" in r and "no output" in r["error"] and "SIGKILL" in r["error"]
    res, sweep = run_at_batch(
        lambda b: rr.run_recurrent(str(rival), str(bench), "s", 50, 1, b, 0, timeout=5, lr=15),
        1_000, ladder=(1_000, 10_000), log=lambda *_: None)
    assert [e["batch"] for e in sweep] == [10_000, 1_000]      # walked, never raised
    assert "no output" in res["error"]
    # a plain non-zero exit with no signal reads as its code
    monkeypatch.setattr(rr.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, returncode=3, stdout="", stderr=""))
    assert "exit code 3 with no output" in rr.run_recurrent(
        str(rival), str(bench), "s", 50, 1, 1_000, 0, timeout=5, lr=15)["error"]


def test_an_interrupted_walk_resumes_below_the_rungs_it_already_saw_refused(tmp_path):
    """Each refused rung is cached as it is seen, so a walk interrupted by a VM reclaim resumes
    below it instead of from the top. A timeout is a ceiling only at its own clock, and the search
    time is carried so the probe budget is charged as one walk."""
    from eval.rivals.rival_common import run_at_batch
    from eval.rivals.rivals_recurrent import (known_ladder, known_spent, remember_oom_ceiling,
                                              remember_refusal)

    bench = tmp_path / "x.bench"; bench.write_text("INPUT(a)\nOUTPUT(y)\ny = DFF(a)\n")
    cache = str(tmp_path / "cache"); L = (1_000, 3_000, 10_000, 30_000, 100_000)
    remember_refusal(cache, str(bench), 50, 100_000, "torch.OutOfMemoryError: CUDA out of memory", 300.0, 1200.0)
    remember_refusal(cache, str(bench), 50, 30_000,
                     "exit SIGKILL with no output (killed by the host, or a crash that wrote nothing)", 400.0, 1200.0)
    # ...the VM is reclaimed here; the next walk starts below both, with their cost remembered
    assert known_ladder(cache, str(bench), 50, L, run_timeout=1200.0) == ((1_000, 3_000, 10_000), 30_000)
    assert known_spent(cache, str(bench), 50) == 700.0
    remember_refusal(cache, str(bench), 50, 10_000, "timeout 1200.0s", 1200.0, 1200.0)
    assert known_ladder(cache, str(bench), 50, L, run_timeout=1200.0) == ((1_000, 3_000), 10_000), "a timeout holds at its clock"
    assert known_ladder(cache, str(bench), 50, L, run_timeout=2400.0) == ((1_000, 3_000, 10_000), 30_000), "...not at a longer one"
    assert known_spent(cache, str(bench), 50) == 1900.0
    remember_refusal(cache, str(bench), 50, 3_000, "no results csv", 5.0, 1200.0)      # unclassified: no ceiling
    assert known_ladder(cache, str(bench), 50, L, run_timeout=1200.0)[1] == 10_000
    assert known_ladder(cache, str(bench), 25, L, run_timeout=1200.0)[1] is None, "a different depth is a different circuit"
    # the walk completes: the search time is dropped, the ceilings kept, a kill ceiling never becomes an OOM one
    remember_oom_ceiling(cache, str(bench), 50, [{"batch": 3_000, "error": None}], prior=10_000)
    assert known_spent(cache, str(bench), 50) == 0.0
    assert known_ladder(cache, str(bench), 50, L, run_timeout=1200.0) == ((1_000, 3_000), 10_000)
    assert known_ladder(cache, str(bench), 50, L, run_timeout=None) == ((1_000, 3_000, 10_000), 30_000)
    # the carried search time charges run_at_batch's probe budget as one walk would have been
    calls = []

    def run(b):
        calls.append(b)
        return {"error": "timeout 1200.0s"} if b > 1_000 else {"unique_per_s": 1.0}

    r, _sweep = run_at_batch(run, 1_000, ladder=(1_000, 3_000, 10_000), log=lambda *_: None, spent=1500.0)
    assert calls == [10_000, 1_000] and r.get("probe_truncated"), "budget already spent: one failure, then the modelled rung"


def test_the_recurrent_warmup_run_is_capped_far_below_the_measurement_timeout(tmp_path, monkeypatch):
    """The discarded warmup run is killed after WARMUP_TIMEOUT_S, since the rival's graph build
    is O(circuit x T) whatever the batch; the measured rungs keep the full timeout."""
    from eval.rivals import rivals_recurrent as rr

    seen = []

    def fake_run(rival, bench, key, T, steps, batch, start_point, timeout, retry=True, lr=None):
        seen.append((batch, timeout))
        return {"unique_per_s": 1.0, "unique": 1} if batch <= 256 else {"error": "torch.OutOfMemoryError"}

    import eval.rivals.rival_common as rc
    monkeypatch.setattr(rr, "run_recurrent", fake_run)
    rc._WARMED.discard("recurrent")
    bench = tmp_path / "s.bench"; bench.write_text("INPUT(a)\nOUTPUT(a)\n")
    rr.run_recurrent_once("R", str(bench), "s", 50, 1, 256, 0, timeout=1200.0,
                          cache_dir=str(tmp_path / "c"))
    warm = [t for b, t in seen if b == 256][0]
    assert warm == rr.WARMUP_TIMEOUT_S <= 60.0, f"warmup not capped: {warm}"
    assert all(t == 1200.0 for b, t in seen if b != 256), "a measured rung lost its full timeout"


def test_a_recorded_failure_is_only_honoured_when_asked_for():
    """`eval/artifact/reproduce.sh` can pin recorded rival failures instead of re-paying their
    init budget. That shortcut can only be unfair to the rival, so: a recorded failure is never a
    grant (`lookup_grant` returns None); honouring it is opt-in, since a faster card might settle;
    and its entry carries `not_remeasured`, so `compare_campaign.py` does not count it as
    agreement."""
    from eval.rivals.rival_common import (grant_key, lookup_grant,  # noqa: PLC0415
                                          lookup_recorded_failure)

    grants = {"_path": "grants/g.json", "recurrent": {
        grant_key("b17", "rare", 25, "abc"): {"batch": None, "error": "did not initialize: ..."},
        grant_key("b03", "rare", 25, "abc"): {"batch": 300_000, "start_point": 4},
    }}
    assert lookup_grant(grants, "recurrent", "b17", "rare", 25) is None, \
        "a recorded failure must never be read as a batch grant"
    fail = lookup_recorded_failure(grants, "recurrent", "b17", "rare", 25)
    assert fail["not_remeasured"] and "NOT re-measured" in fail["grant_source"]
    assert fail["error"].startswith("did not initialize")
    # a cell with a grant is not a failure, and an unrecorded cell is neither
    assert lookup_recorded_failure(grants, "recurrent", "b03", "rare", 25) is None
    assert lookup_recorded_failure(grants, "recurrent", "b99", "rare", 25) is None
    assert lookup_recorded_failure({}, "recurrent", "b17", "rare", 25) is None


def test_a_pinned_failure_is_itself_a_rival_failure_row():
    """Both harnesses write the lookup's dict straight out (`dict(pinned_fail)`), so this
    classifies that object: the tables see a rival failure, never a zero or a win."""
    from eval.artifact.paper_tables import rival_cells  # noqa: PLC0415
    from eval.rivals.rival_common import grant_key, lookup_recorded_failure  # noqa: PLC0415

    grants = {"_path": "g.json", "recurrent": {
        grant_key("b17", "rare", 25, "abc"):
            {"batch": None, "error": "did not initialize inside 1200 s"}}}
    row = lookup_recorded_failure(grants, "recurrent", "b17", "rare", 25, "abc")
    cell = {"name": "b17", "variant": "abc", "cycles": 25, "rcs": row,
            "gpap": {"unique_per_s_est": 1e6, "n_models": 10, "sps": 1e6}}
    (_, _, bucket, _, _), = rival_cells([cell], "rcs")
    assert bucket == "error", "a pinned failure must classify as a rival failure, not a zero"


def test_honouring_a_recorded_failure_stays_opt_in_and_settings_bound():
    """`--honour-recorded-failures` is off unless asked for, and a pin is honoured only under the
    settings that produced it. The flag is read from source because both harnesses build their
    parser inside `main()`."""
    from tests.conftest import REPO_ROOT  # noqa: PLC0415

    for mod in ("rivals_recurrent.py", "rivals_demotic.py"):
        path = os.path.join(REPO_ROOT, "eval", "rivals", mod)
        with open(path) as fh:
            tree = ast.parse(fh.read())
        calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument"
                 and n.args and getattr(n.args[0], "value", None) == "--honour-recorded-failures"]
        assert len(calls) == 1, f"{mod}: the flag is declared {len(calls)} times"
        kw = {k.arg: k.value for k in calls[0].keywords}
        assert getattr(kw.get("action"), "value", None) == "store_true", \
            f"{mod}: the flag must be store_true, i.e. OFF unless the operator asks"
        assert "default" not in kw, f"{mod}: a default on this flag turns the shortcut on"

    # The pin is bound to its settings: `settled_error` is the predicate `--retry-errors` uses,
    # so a raised --init-budget re-measures the recorded row rather than replaying it.
    from eval.rivals.rivals_recurrent import settled_error  # noqa: PLC0415

    recorded = ("did not initialize: the start_point search reached sp=1 and exhausted its "
                "1200s initialization budget")
    assert settled_error(recorded, 1200, 1200) is True
    assert settled_error(recorded, 3600, 1200) is False


# --------------------------------------------------------------- Recurrent's host snapshot, deducted

_FAKE_RUN_PY = '''import os, sys
a = sys.argv
key = os.path.basename(a[a.index("-d") + 1])[:-len(".bench")].replace(".", "")
batch = a[a.index("--batch_size") + 1]
res = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(res, exist_ok=True)
cols = {"num_unique_solutions": "100", "model_runtime": "10.0"}
if os.environ.get("FAKE_EGRESS"):
    cols["egress_runtime"] = os.environ["FAKE_EGRESS"]
with open(os.path.join(res, f"sat_{key}_1_mse_15.0_{batch}.csv"), "w") as f:
    f.write("\\t".join(cols) + "\\n" + "\\t".join(cols.values()) + "\\n")
print("[[3], [7]]")
'''


def _fake_recurrent(tmp_path):
    rival = tmp_path / "rcs"
    (rival / "src" / "pytorch").mkdir(parents=True)
    (rival / "script").mkdir()
    (rival / "src" / "pytorch" / "run.py").write_text(_FAKE_RUN_PY)
    bench = tmp_path / "s208.1.bench"
    bench.write_text("INPUT(a)\nOUTPUT(a)\n")
    return str(rival), str(bench)


def test_recurrents_rate_excludes_its_host_snapshot(tmp_path, monkeypatch):
    """Their timed loop copies every depth's candidates to the host; the eval patch reports that
    copy as `egress_runtime`, and the rate is unique / (model_runtime - egress_runtime). Their own
    clock stays `runtime_s`, the copy `egress_s`."""
    from eval.rivals.rivals_recurrent import run_recurrent  # noqa: PLC0415

    rival, bench = _fake_recurrent(tmp_path)
    monkeypatch.setenv("FAKE_EGRESS", "2.0")
    r = run_recurrent(rival, bench, "s208.1", 25, 1, 1000, 0)
    assert "error" not in r, r
    assert r["runtime_s"] == 10.0 and r["egress_s"] == 2.0
    assert r["unique_per_s"] == pytest.approx(100 / 8.0)
    assert r["unique_at_T"] == 7


def test_an_unpatched_recurrent_stops_the_run(tmp_path, monkeypatch):
    """Without `egress_runtime` the rate would carry the copy. It RAISES: returned as an error
    row it would read as a refused rung, and the ladder walk would record a smaller grant for a
    fault in the measurement."""
    from eval.rivals.rivals_recurrent import run_recurrent  # noqa: PLC0415

    rival, bench = _fake_recurrent(tmp_path)
    monkeypatch.delenv("FAKE_EGRESS", raising=False)
    with pytest.raises(RuntimeError, match="recurrent_gpap_eval.patch"):
        run_recurrent(rival, bench, "s208.1", 25, 1, 1000, 0)
    monkeypatch.setenv("FAKE_EGRESS", "10.0")       # the copy cannot be the whole clock
    with pytest.raises(RuntimeError, match="outside model_runtime"):
        run_recurrent(rival, bench, "s208.1", 25, 1, 1000, 0)


def test_resume_refuses_a_file_on_the_old_clock():
    """A row measured before the patch (no `egress_s`) carries the copy in its rate; resuming
    into its file would pool two metrics in one geomean. Error rows carry no rate and pass."""
    from eval.rivals.rivals_recurrent import rows_on_the_old_clock  # noqa: PLC0415

    new = {"name": "b03", "rcs": {"runtime_s": 9.3, "egress_s": 0.97}}
    old = {"name": "b01", "rcs": {"runtime_s": 8.1}}
    failed = {"name": "b19", "rcs": {"error": "did not initialize"}}
    assert rows_on_the_old_clock([new, failed]) == []
    assert rows_on_the_old_clock([new, old, failed]) == ["b01"]


def test_the_cross_card_comparison_puts_both_sides_on_one_clock():
    """A reference recorded before the deduction against a fresh run after it: both sides are
    rated on their own `runtime_s`, or the speed factor would measure the metric change."""
    from eval.artifact.compare_campaign import one_clock  # noqa: PLC0415

    ref = [{"name": "b03", "rcs": {"unique": 100, "runtime_s": 10.0, "unique_per_s": 10.0}}]
    new = [{"name": "b03", "rcs": {"unique": 100, "runtime_s": 10.0, "egress_s": 2.0,
                                   "unique_per_s": 12.5}}]
    r, n, converted = one_clock(ref, new)
    assert converted and r[0]["rcs"]["unique_per_s"] == 10.0 and n[0]["rcs"]["unique_per_s"] == 10.0
    assert new[0]["rcs"]["unique_per_s"] == 12.5, "the caller's rows must not be modified"
    r, n, converted = one_clock(new, new)
    assert not converted and n[0]["rcs"]["unique_per_s"] == 12.5


def test_the_recurrent_eval_patch_applies_to_the_pinned_commit(tmp_path):
    """The patch is a diff against the pinned commit, applied to that commit's pristine run.py in
    a repository of its own: `git apply` in a directory with no .git inside another repository
    prints "Skipped patch" and returns 0, which would pass this test on anything."""
    import subprocess  # noqa: PLC0415

    rival = os.path.dirname(os.path.dirname(os.path.dirname(RIVAL_RUN_PY)))
    if not os.path.isdir(os.path.join(rival, ".git")):
        pytest.skip("Recurrent git checkout absent (eval/rivals/rivals_setup.sh); nothing to verify")
    pristine = subprocess.run(["git", "-C", rival, "show", "HEAD:src/pytorch/run.py"],
                              capture_output=True, text=True, check=True).stdout
    (tmp_path / "src" / "pytorch").mkdir(parents=True)
    (tmp_path / "src" / "pytorch" / "run.py").write_text(pristine)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    patch = os.path.join(REPO_ROOT, "eval", "rivals", "recurrent_gpap_eval.patch")
    subprocess.run(["git", "-C", str(tmp_path), "apply", patch], check=True)
    patched = (tmp_path / "src" / "pytorch" / "run.py").read_text()
    assert "egress_runtime" in patched and "torch.cuda.synchronize()" in patched


def test_the_emission_audit_reads_the_measured_copy_and_models_only_older_rows(tmp_path, capsys):
    """Section 4: a row carrying `egress_s` is reported as measured; a row recorded before the
    patch gets the modelled share at ITS OWN depth (a T=50 row once was sized at T=25), and is
    counted as still carrying the copy."""
    import json  # noqa: PLC0415

    from eval.rivals.check_emission_symmetry import recurrent_egress  # noqa: PLC0415

    def row(cycles, egress):
        rcs = {"runtime_s": 20.0, "start_point": 1, "batch": 100000}
        if egress is not None:
            rcs["egress_s"] = egress
        return {"name": "b01", "cycles": cycles, "rcs": rcs}

    (tmp_path / "rivals_recurrent_table.json").write_text(json.dumps([row(25, 1.0)]))
    assert recurrent_egress(str(tmp_path)) == (True, 0)
    assert "5.00%" in capsys.readouterr().out

    (tmp_path / "rivals_recurrent_table_t50.json").write_text(
        json.dumps([row(50, None), row(25, None)]))
    assert recurrent_egress(str(tmp_path)) == (True, 2)
    out = capsys.readouterr().out
    shares = [float(m) for m in re.findall(r"modelled\s+([\d.]+)%", out)]
    assert len(shares) == 2 and shares[0] > 3 * shares[1], \
        "the T=50 row must be modelled at 50 depths, not 25"
    assert "FLATTERS GPAP" in out


def test_the_emission_audit_fails_a_stage_file_on_two_clocks(tmp_path, capsys):
    """One file holding measured and pre-patch rows would pool two metrics in one geomean. A
    pre-patch row whose .bench is absent (b17-b22 are fetched) still counts as carrying the copy."""
    import json  # noqa: PLC0415

    from eval.rivals.check_emission_symmetry import recurrent_egress  # noqa: PLC0415

    rows = [{"name": "b01", "cycles": 25,
             "rcs": {"runtime_s": 20.0, "start_point": 1, "batch": 1000, "egress_s": 1.0}},
            {"name": "no_such_bench", "cycles": 25,
             "rcs": {"runtime_s": 20.0, "start_point": 1, "batch": 1000}}]
    (tmp_path / "rivals_recurrent_table.json").write_text(json.dumps(rows))
    ok, carried = recurrent_egress(str(tmp_path))
    assert not ok and carried == 1
    assert "FAIL" in capsys.readouterr().out
