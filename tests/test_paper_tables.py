"""The reporting rules that decide what a paper claims — pinned by synthetic stage JSONs.

Every rule here exists because its absence produced a flattering number in something we had
already published (docs/findings/L4_EVAL_FINDINGS.md §2). The failures were mechanical, not
deliberate, which is exactly why a test is the right guard: reading the aggregation code did not
catch them, and a plausible-looking table is indistinguishable from a correct one.

No GPU, no data files — the fixtures are hand-built rows, so a rule change shows up as a test
diff rather than a silently different geomean months later.
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.artifact.paper_tables import (cms_family, encoders_in, prop_rates,  # noqa: E402
                                        rival_family, zero_search)


def row(name, enc="tseitin", *, models=1000, cms=1000, sps=1e6, cms_sps=1e4,
        impl=1e9, cms_impl=1e6, dec=1e7, cms_dec=1e5, **kw):
    """One benchmark cell. Defaults are a healthy GPAP win; override to build the edge cases."""
    r = {"name": name, "enc_key": enc, "n_models": models, "cms_n_samples": cms,
         "gpap_sps": sps, "cms_sps": cms_sps, "implications_per_s": impl,
         "cms_implications_per_s": cms_impl, "decisions_per_s": dec,
         "cms_decisions_per_s": cms_dec, "M": 100, "support": 10,
         "engine": {"split": "16|16", "mode": "wave"}}
    r["ratio"] = (sps / cms_sps) if cms_sps else (float("inf") if models else None)
    if not models:
        r["ratio"] = 0.0 if cms else None
    r.update(kw)
    return r


def stage(tmp_path, stem, rows, **payload):
    d = {"results": rows, "gpu": {"name": "T"}, "cpu": {"model": "C"}, "cmsgen_patched": True}
    d.update(payload)
    (tmp_path / f"{stem}.json").write_text(json.dumps(d))
    return str(tmp_path)


# --------------------------------------------------------------------- the symmetric zero ledger

def test_zero_ledger_is_symmetric_and_sums(tmp_path):
    """THE bug: a GPAP-zero row has ratio 0.0 and a both-zero row None — both falsy, so branching
    on `ratio` dropped them entirely while CMSGen-zero rows (inf) were counted as wins. 38 of
    their zeros were shown; 3 of ours and 11 mutual ones were invisible."""
    d = stage(tmp_path, "s", [
        row("win.bench"),                                   # both sampled, we win
        row("theirzero.bench", cms=0, cms_sps=0),           # CMSGen silent -> our win
        row("ourzero.bench", models=0, sps=0),              # WE are silent  -> our loss
        row("bothzero.bench", models=0, sps=0, cms=0, cms_sps=0),
    ])
    v = cms_family(d, "s", "tseitin")
    assert (v["inf"], v["gpap0"], v["both0"]) == (1, 1, 1)
    assert v["gpap0_names"] == ["ourzero"] and v["both0_names"] == ["bothzero"]
    # the ledger must balance: geomean instances + each zero bucket == instance count
    assert v["n_gm"] + v["inf"] + v["gpap0"] + v["both0"] == v["inst"] == 4
    # our loss is in the denominator of wins but is not a win; a mutual zero is neither
    assert v["wins"] == 2 and v["n"] == 3


def test_gpap_zero_is_not_silently_a_win(tmp_path):
    """A stage where we fail everywhere must not report a flattering geomean."""
    d = stage(tmp_path, "s", [row(f"i{i}.bench", models=0, sps=0) for i in range(4)])
    v = cms_family(d, "s", "tseitin")
    assert v["gpap0"] == 4 and v["wins"] == 0 and v["gm"] is None


# --------------------------------------------------------------------- zero-search degeneracy

def test_zero_search_detected_and_excluded(tmp_path):
    """Samples emitted with no decisions AND no implications mean the support is free after
    propagation — no search happened, so the ratio measures the instance. The best-encoder rule
    actively PREFERS these cells because they carry the family's top ratio."""
    free = row("free.bench", sps=1e9, dec=0, impl=0)        # 100,000x, and entirely degenerate
    assert zero_search(free) and not zero_search(row("ok.bench"))
    d = stage(tmp_path, "s", [free] + [row(f"i{i}.bench") for i in range(3)])
    v = cms_family(d, "s", "tseitin")
    assert v["free"] == 1 and v["free_names"] == ["free"]
    assert v["gm"] == pytest.approx(100.0)                   # the three honest cells only
    assert v["gm_withfree"] > v["gm"]                        # and we report what it would inflate to


def test_encoders_are_separate_rows_and_neither_selects_the_other(tmp_path):
    """No best-of-encoders rule. It used to pick, per instance, the encoder with the highest
    `ratio` — making the published statistic its own argmax key, so CMSGen's number chose which
    of our runs was reported (measured: 1.24-1.33x on the sequential families). It also actively
    preferred degenerate cells: on EPFL `ctrl` Tseitin samples at 2.0M/s with succ 0.07 while ABC
    collapses the output cone to succ 1.0 at 240M/s with no search at all, and the argmax took
    the second. Each encoder is now its own row, screened on its own merits."""
    d = stage(tmp_path, "s", [
        row("ctrl.aig", "tseitin", sps=2e6),
        row("ctrl.aig", "abc:fast", sps=2.4e8, dec=0, impl=0),
        row("a.aig"), row("b.aig"), row("c.aig"),
        row("a.aig", "abc:fast"), row("b.aig", "abc:fast"), row("c.aig", "abc:fast"),
    ])
    assert encoders_in(json.loads((Path(d) / "s.json").read_text())) == ["tseitin", "abc:fast"]
    ts, abc = cms_family(d, "s", "tseitin"), cms_family(d, "s", "abc:fast")
    assert ts["free"] == 0 and ts["n_gm"] == 4        # tseitin's ctrl really does search
    assert abc["free"] == 1 and abc["n_gm"] == 3      # abc's ctrl is the free hypercube
    # tseitin keeps its ctrl (ratio 200 against the others' 100); abc drops its degenerate one
    assert ts["gm"] == pytest.approx((200 * 100 ** 3) ** 0.25)
    assert abc["gm"] == pytest.approx(100.0)


def test_every_instance_is_counted_exactly_once(tmp_path):
    """`inst` = n_gm + CMSGen=0 + GPAP=0 + both=0 + no-search + vacuous, printed under Table 1.
    It was FALSE on 6 of 7 families: the degenerate cells were dropped from the geomean but left
    inside `inst` and inside `wins`, so a reader adding the columns up came out short."""
    d = stage(tmp_path, "seq-c25-rare", [
        row("a.bench"), row("b.bench"),                       # 2 in the geomean
        row("c.bench", cms=0, cms_sps=0),                     # CMSGen=0
        row("d.bench", models=0, sps=0),                      # GPAP=0
        row("e.bench", models=0, sps=0, cms=0, cms_sps=0),    # both=0
        row("f.bench", dec=0, impl=0),                        # no-search
        row("b02.bench", sps=1e8),                            # vacuous pin
    ])
    v = cms_family(d, "seq-c25-rare", "tseitin")
    assert v["inst"] == 7
    assert v["n_gm"] + v["inf"] + v["gpap0"] + v["both0"] + v["free"] + v["vac"] == v["inst"]
    # and the win count is over the same rows, not over the pre-screen set
    assert v["n"] == v["n_gm"] + v["inf"] + v["gpap0"] == 4
    assert v["wins"] == 3                                     # a, b, and the CMSGen-zero


# --------------------------------------------------------------------- the propagation identity

def test_propagation_identity_closes(tmp_path):
    """samples/s ratio == (impl/s ratio) / (impl-per-sample ratio). Algebra, so it can only fail
    if the columns are aggregated over different rows — which is what broke EPFL by 3x."""
    rows = [row(f"i{i}.bench", sps=10.0 ** (5 + i), impl=1e9, cms_impl=2e6 * (i + 1))
            for i in range(4)]
    d = stage(tmp_path, "s", rows)
    p, v = prop_rates(d, "s", "tseitin"), cms_family(d, "s", "tseitin")
    assert abs((p["ratio"] / p["work"]) / v["gm"] - 1.0) < 0.01


def test_propagation_columns_share_one_row_set(tmp_path):
    """An instance that propagates hard but emits NOTHING (EPFL `bar`) must not enter the
    implications ratio while being absent from work-per-sample."""
    d = stage(tmp_path, "s", [row(f"i{i}.bench") for i in range(3)]
              + [row("bar.aig", models=0, sps=0, impl=1e9)])
    p, v = prop_rates(d, "s", "tseitin"), cms_family(d, "s", "tseitin")
    assert p["n"] == 3, "the zero-emitting instance has no work-per-sample and must be excluded"
    assert abs((p["ratio"] / p["work"]) / v["gm"] - 1.0) < 0.01


# --------------------------------------------------------------------- rivals, same rules

def test_rival_zero_buckets_are_symmetric(tmp_path):
    """A cell where NEITHER side scored was being credited to us as a win (DEMOTIC-miter 18/18,
    HTS 14/15)."""
    rows = [
        {"name": "a", "variant": "raw", "demotic": {"unique_per_s": 10.0},
         "gpap": {"unique_per_s_est": 100.0}},                       # our win
        {"name": "b", "variant": "raw", "demotic": {"unique_per_s": 0.0},
         "gpap": {"unique_per_s_est": 100.0}},                       # rival silent -> our win
        {"name": "c", "variant": "raw", "demotic": {"unique_per_s": 10.0},
         "gpap": {"unique_per_s_est": 0.0}},                         # WE are silent -> our loss
        {"name": "d", "variant": "raw", "demotic": {"unique_per_s": 0.0},
         "gpap": {"unique_per_s_est": 0.0}},                         # neither -> nobody's win
        {"name": "e", "variant": "raw", "demotic": {"error": "timeout"}, "gpap": {}},
    ]
    (tmp_path / "rivals_x.json").write_text(json.dumps({"rows": rows}))
    v = rival_family(str(tmp_path), "x", "demotic")
    assert (v["zero"], v["gpap0"], v["both0"], v["errors"]) == (1, 1, 1, 1)
    assert v["wins"] == 2 and v["n"] == 3
    assert v["gm"] == pytest.approx(10.0)     # only the one finite ratio


def test_rival_unmeasured_unique_is_not_a_gpap_zero(tmp_path):
    """`unique_per_s_est` is None/0 when the OFF-timing coverage run collected nothing, which is
    NOT "GPAP produced nothing" — the timed run's `n_models` says that. Reading the first as the
    second published b11-c25 (42 models at a rare pin, ~15x the rival) as an outright LOSS."""
    rows = [
        {"name": "b11", "variant": "raw", "hts": {"unique_per_s": 0.05},        # not measured
         "gpap": {"unique_per_s_est": None, "unique_est_status": "no-coverage-sample",
                  "n_models": 42}},
        {"name": "b11old", "variant": "raw", "hts": {"unique_per_s": 0.05},     # pre-status row
         "gpap": {"unique_per_s_est": 0.0, "n_models": 42}},
        {"name": "z", "variant": "raw", "hts": {"unique_per_s": 10.0},          # genuine zero
         "gpap": {"unique_per_s_est": 0.0, "n_models": 0}},
        {"name": "w", "variant": "raw", "hts": {"unique_per_s": 10.0},
         "gpap": {"unique_per_s_est": 100.0, "n_models": 7}},
    ]
    (tmp_path / "rivals_h.json").write_text(json.dumps({"rows": rows}))
    v = rival_family(str(tmp_path), "h", "hts")
    assert v["nomeas"] == 2, "both the new and the legacy shape must be recognised"
    assert v["gpap0"] == 1, "a real zero is still a loss"
    assert v["n"] == 2 and v["wins"] == 1     # only w has a comparable ratio; z is the loss
    # the columns must still add up to the cells
    assert 1 + v["zero"] + v["gpap0"] + v["both0"] + v["nomeas"] + v["errors"] == v["cells"]


def test_rival_zero_search_cell_is_screened_not_scored(tmp_path):
    """A rival cell whose GPAP side did NO search (zero decisions, zero implications, models
    emitted) measures the instance, not either sampler — a seed-mismatched b13 cell once carried
    a 4.7e9x ratio this way. It leaves through its own `free` column, exactly as in Table 1."""
    rows = [
        {"name": "ok", "variant": "raw", "demotic": {"unique_per_s": 10.0},
         "gpap": {"unique_per_s_est": 100.0, "n_models": 5,
                  "decisions_per_s": 1e7, "implications_per_s": 1e9}},
        {"name": "b13", "variant": "raw", "demotic": {"unique_per_s": 1e-3},
         "gpap": {"unique_per_s_est": 4.7e6, "n_models": 1000,
                  "decisions_per_s": 0, "implications_per_s": 0}},
    ]
    (tmp_path / "rivals_x.json").write_text(json.dumps({"rows": rows}))
    v = rival_family(str(tmp_path), "x", "demotic")
    assert v["free"] == 1 and v["gm"] == pytest.approx(10.0), \
        "the free-hypercube cell entered the geomean"
    assert v["n"] == 1 and v["wins"] == 1
    # the columns still sum to the cells
    assert 1 + v["zero"] + v["gpap0"] + v["both0"] + v["nomeas"] + v["errors"] + v["free"] \
        + v["misaligned"] == v["cells"]


def test_rival_misaligned_row_is_excluded_and_counted(tmp_path):
    """Table 2's text says constraint alignment is VERIFIED per row (`pin_check_ok`). A row that
    FAILED that check compared two different tasks; whatever its ratio measures, it is not the
    samplers — so it must leave through its own bucket, not sit silently inside the geomean."""
    rows = [
        {"name": "ok", "variant": "raw", "pin_check_ok": True,
         "demotic": {"unique_per_s": 10.0}, "gpap": {"unique_per_s_est": 100.0}},
        {"name": "bad", "variant": "raw", "pin_check_ok": False,
         "demotic": {"unique_per_s": 1e-6}, "gpap": {"unique_per_s_est": 1e6}},  # a 1e12x "win"
    ]
    (tmp_path / "rivals_x.json").write_text(json.dumps({"rows": rows}))
    v = rival_family(str(tmp_path), "x", "demotic")
    assert v["misaligned"] == 1
    assert v["gm"] == pytest.approx(10.0) and v["n"] == 1, \
        "a row that failed the alignment check entered the ledger"


def test_rival_retry_row_replaces_rather_than_appends(tmp_path):
    """A resumed harness re-emits a row; appending gave 25 rows in an 18-instance file."""
    rows = [
        {"name": "a", "variant": "raw", "hts": {"error": "no pycryptosat"}, "gpap": {}},
        {"name": "a", "variant": "raw", "hts": {"unique_per_s": 10.0},
         "gpap": {"unique_per_s_est": 50.0}},
    ]
    (tmp_path / "rivals_h.json").write_text(json.dumps({"rows": rows}))
    v = rival_family(str(tmp_path), "h", "hts")
    assert v["cells"] == 1 and v["errors"] == 0 and v["gm"] == pytest.approx(5.0)


# --------------------------------------------------------------- summary.md, the same ledger

def test_summary_md_counts_zeros_the_papers_way(tmp_path):
    """`summary.md`'s win column used to test `ratio >= 1`, which counts a CMSGen-zero row
    (ratio == inf) as a win while showing no CMSGen=0 column at all — their zeros inside our
    wins, ours in a column, the exact asymmetry the paper's ledger exists to prevent."""
    from eval.eval_plots import write_summary
    stage(tmp_path, "s", [
        row("win.bench"),                                     # finite win
        row("lose.bench", sps=1e3),                           # finite loss (ratio 0.1)
        row("theirzero.bench", cms=0, cms_sps=0),             # CMSGen silent -> inf
        row("ourzero.bench", models=0, sps=0),                # we are silent -> 0.0
    ], tag="s")
    stages = {"s": json.loads((tmp_path / "s.json").read_text())}
    out = Path(write_summary(stages, str(tmp_path), str(tmp_path))).read_text()
    line = next(ln for ln in out.splitlines() if ln.startswith("| s |"))
    cells = [c.strip() for c in line.split("|")]
    hdr = next(ln for ln in out.splitlines() if ln.startswith("| stage |"))
    cols = [c.strip() for c in hdr.split("|")]
    got = dict(zip(cols, cells))
    assert got["wins"] == "2/4", got            # win + CMSGen-zero, over finite + both zeros
    assert got["CMSGen=0"] == "1" and got["GPAP=0"] == "1", got




def test_uniformity_screens_subcube_instances(tmp_path):
    """The campaign's own summary excludes full-subcube instances from its medians (the
    don't-care filler, not the config under test, draws the bits there — two configs return
    bit-identical statistics). Table 4 must apply the SAME screen, or the paper's median can
    disagree with the tool that produced the data."""
    from eval.artifact.paper_tables import uniformity
    rows = [
        {"kind": "circuit", "meta": {"subcube": False},
         "samplers": {"gpap:b8": {"tv_median": 0.10, "verdict": "ok", "unsound": 0}}},
        {"kind": "circuit", "meta": {"subcube": True},         # filler-only: not evidence
         "samplers": {"gpap:b8": {"tv_median": 0.00, "verdict": "ok", "unsound": 0}}},
    ]
    p = tmp_path / "u.json"
    p.write_text(json.dumps({"rows": rows, "args": {"alpha": 1e-3}}))
    uni, args = uniformity([str(p)])
    a = uni[("gpap:b8", "circuit")]
    assert a["n"] == 1 and a["tv"] == pytest.approx(0.10), \
        "the subcube instance diluted the aggregate"
    assert args["_subcube_excluded"] == 1


def test_prop_rates_has_no_row_when_every_cell_is_degenerate(tmp_path):
    """Table 3 must describe the runs Table 1 kept. When the screens remove EVERY cell of a
    family, the honest Table 3 row is no row — not a quiet fallback to the screened cells."""
    d = stage(tmp_path, "s", [row("b02.bench"), row("b05.bench")])       # all vacuous pins
    assert prop_rates(d, "s", "tseitin") is None


# --------------------------------------------------------------------- vacuous pins

def test_vacuous_pins_excluded_only_where_present(tmp_path):
    d = stage(tmp_path, "s", [row("b02.bench", sps=1e8)]          # tautological pin, huge ratio
              + [row(f"i{i}.bench") for i in range(3)])
    v = cms_family(d, "s", "tseitin")
    assert v["has_vac"] and v["gm"] == pytest.approx(100.0)
    d2 = stage(tmp_path, "t", [row(f"i{i}.bench") for i in range(3)])
    assert cms_family(d2, "t", "tseitin")["has_vac"] is False


# ------------------------------------------------- the e2e column: who is charged for setup

def timed_row(name, *, sps=1e6, cms_sps=1e4, pre=0.5, wall_total=2.0, **kw):
    """A cell whose GPAP side spent `pre` of its end-to-end wall PREPARING (parse/construct/ELS/
    warmup/mining) rather than sampling."""
    r = row(name, **kw)
    r["wall_total_s"] = wall_total
    r["phases"] = {"parse": 0.0, "construct": pre * wall_total, "warmup": 0.0,
                   "mine": 0.0, "rebuild": 0.0, "recheck": 0.0,
                   "timed": (1 - pre) * wall_total}
    # what benchmark.py records: the e2e rate is models/wall_total, so the e2e ratio is the
    # steady-state one scaled by the fraction of the wall that was actually sampling
    r["ratio_e2e"] = (sps * (1 - pre)) / cms_sps
    return r


def test_e2e_column_charges_gpap_for_its_own_preprocessing(tmp_path):
    """CMSGen's rate is samples/subprocess-wall — its parse and setup are already inside it. So a
    table that scores GPAP on its steady-state sampling rate alone charges one side and not the
    other; `gm_e2e` must exist beside `gm` and must be lower by exactly the preparation share."""
    d = stage(tmp_path, "seq-c25-rare",
              [timed_row("a", pre=0.5), timed_row("b", pre=0.5)])
    v = cms_family(d, "seq-c25-rare", "tseitin")
    assert v["gm"] == pytest.approx(100.0)          # 1e6 / 1e4, setup not charged
    assert v["gm_e2e"] == pytest.approx(50.0)       # half the wall was preparation
    assert v["pre_frac"] == pytest.approx(0.5)
    assert v["gm_e2e"] < v["gm"], "the honest column must never read higher than the headline"


def test_pre_frac_is_a_median_not_an_average_of_ratios(tmp_path):
    """One instance whose formula takes seconds to build must not drag the whole family's
    reported preparation share — the column is a median over cells, like every other spread
    number in these tables."""
    rows = [timed_row("a", pre=0.1), timed_row("b", pre=0.1), timed_row("c", pre=0.9)]
    v = cms_family(stage(tmp_path, "seq-c25-rare", rows), "seq-c25-rare", "tseitin")
    assert v["pre_frac"] == pytest.approx(0.1)


# ------------------------------------------------- the parallelism grant handed to the rivals

def test_batch_is_sized_to_the_same_share_of_the_card_we_take():
    """A rival's unique/s is near-proportional to the batch we hand it (measured: DEMOTIC c1355
    1k -> 1M is 564x), so the batch is the parallelism grant, not a detail of their setup. The
    rule is the one GPAP uses on itself — the largest that fits in 0.80 of free VRAM."""
    from eval.rivals.rival_common import GRAPH_BYTES, fit_batch
    budget = 8 << 30
    # 1000 gates x 25 frames of DEMOTIC graph: budget / (work x bytes-per-row)
    work = 1000 * 25
    b = fit_batch(work, "demotic", budget_bytes=budget)
    assert work * GRAPH_BYTES["demotic"] * b <= budget
    assert fit_batch(work, "demotic", budget_bytes=budget * 4) > b, "a bigger card must grant more"


def test_batch_never_returns_something_that_cannot_run():
    """A formula so wide that not even the smallest rung fits must still yield a runnable batch —
    the rival reporting an OOM is a result; the harness handing it 0 is a bug."""
    from eval.rivals.rival_common import BATCH_LADDER, fit_batch
    assert fit_batch(10 ** 12, "recurrent", budget_bytes=1 << 20) == BATCH_LADDER[0]


def test_run_at_batch_steps_down_when_the_card_refuses_and_records_the_rungs():
    """'device not ready' reproduces at a fixed size on some models yet passes one rung lower, and
    the measured VRAM factor is an average — so the driver, not our model, has the last word."""
    from eval.rivals.rival_common import run_at_batch

    def run(b):
        return {"error": "CUDA driver error: device not ready"} if b > 10_000 \
            else {"unique_per_s": float(b)}

    best, sweep = run_at_batch(run, 300_000, log=lambda *_: None)
    assert "error" not in best and best["unique_per_s"] == 10_000
    assert any(s.get("error") for s in sweep), "the refusals belong in the recorded sweep"
    # down to the rung that runs; the upward probe is skipped because 30k already refused
    assert [s["batch"] for s in sweep] == [300_000, 100_000, 30_000, 10_000]


def test_run_at_batch_probes_one_rung_up_because_one_factor_cannot_fit_all_sizes():
    """Measured, DEMOTIC at batch 300k: peak VRAM per gate-row falls 23.0 B (c432) -> 11.6 B
    (c3540), so any single factor over-charges one end — at 20 B the model rejects c2670 at 1M,
    which really fits an L4. One probe upward recovers that; it is bounded to one extra run."""
    from eval.rivals.rival_common import run_at_batch
    calls = []

    def run(b):
        calls.append(b)
        return {"unique_per_s": float(b)}                # bigger is genuinely better here

    best, sweep = run_at_batch(run, 100_000, log=lambda *_: None)
    assert best["unique_per_s"] == 300_000, "the under-grant was not recovered"
    assert calls == [100_000, 300_000], f"probe must be bounded to ONE rung: {calls}"


def test_run_at_batch_keeps_the_granted_rung_when_the_bigger_one_spills():
    """Too big does not always ERROR — Recurrent on s386 runs at 1M and is slower than at 300k.
    The upward probe must be kept only when it is actually faster."""
    from eval.rivals.rival_common import run_at_batch

    def run(b):
        return {"unique_per_s": 900.0 if b == 100_000 else 200.0}   # 300k spills

    best, sweep = run_at_batch(run, 100_000, log=lambda *_: None)
    assert best["unique_per_s"] == 900.0, "scored the rival on a spilled run"
    assert [x["batch"] for x in sweep] == [100_000, 300_000]


# ------------------------------------------------- mirroring: results only exist once local

def test_mirror_accepts_every_artifact_type_not_just_json(tmp_path):
    """A rented VM dies with its disk, so a result exists only once mirrored — and the mirror
    must not silently DROP an artifact because of how it validates it. Validating every file as
    JSON (which colab_supervisor did until 2026-08-25) discards non-JSON artifacts entirely: the
    GPAP-arm cache tarball, the one thing that makes a rival-side re-run cheap, never landed."""
    import json
    import tarfile
    from eval.colab.colab_supervisor import _intact

    good = tmp_path / "r.json"
    good.write_text(json.dumps([{"name": "a"}]))
    tgz = tmp_path / "gpap_cache.tgz"
    with tarfile.open(tgz, "w:gz") as t:
        t.add(good, arcname="r.json")
    assert _intact(str(good)) and _intact(str(tgz))


def test_mirror_rejects_a_download_caught_mid_write(tmp_path):
    """The check it replaced was there for a reason: a file copied while the VM was writing it
    reads back truncated, and a truncated result must never replace a good local copy."""
    import json
    import tarfile
    from eval.colab.colab_supervisor import _intact

    src = tmp_path / "r.json"
    src.write_text(json.dumps([{"name": "a"}]))
    tgz = tmp_path / "c.tgz"
    with tarfile.open(tgz, "w:gz") as t:
        t.add(src, arcname="r.json")
    (tmp_path / "half.json").write_text('[{"name":')
    (tmp_path / "half.tgz").write_bytes(tgz.read_bytes()[:20])
    assert not _intact(str(tmp_path / "half.json"))
    assert not _intact(str(tmp_path / "half.tgz"))


def test_a_timeout_does_not_walk_the_whole_batch_ladder():
    """A timeout is not a memory refusal. Measured on the L4: HTS's three tseitin cells stepped
    1M -> 1,000 at the full 300 s cap on every rung, ~40 min per cell to record an error it had
    after the first run. Their runtime does grow with the batch, so allow two rungs, not eight."""
    from eval.rivals.rival_common import run_at_batch
    calls = []

    def run(b):
        calls.append(b)
        return {"error": "timeout 300s"}

    best, sweep = run_at_batch(run, 3_000_000, log=lambda *_: None)
    assert "error" in best
    assert len(calls) <= 3, f"walked the ladder on a timeout: {calls}"


# --------------------------------------------------------------- one home for the screens

#: The rules that decide which cells a published number is computed over. CLAUDE.md states they
#: live in `eval/artifact/paper_tables.py` ONCE and that the deck imports them; nothing enforced
#: it, and "the second tool re-derived the first tool's rule" is the failure mode the whole file
#: above exists for — a re-derivation drifts silently, and the two tables then disagree about
#: what was excluded while both look correct.
SCREENS = ["zero_search", "select_cells", "cms_family", "prop_rates", "rival_family",
           "encoders_in", "unique_unmeasured", "sound_failed"]


def _eval_modules():
    root = Path(__file__).resolve().parents[1] / "eval"
    return sorted(p for p in root.rglob("*.py"))


def test_each_screening_rule_is_defined_exactly_once_across_eval():
    """A second definition anywhere under eval/ is the drift itself, whatever it is called."""
    import ast

    home = Path(__file__).resolve().parents[1] / "eval" / "artifact" / "paper_tables.py"
    found = {name: [] for name in SCREENS}
    for path in _eval_modules():
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name in found:
                found[node.name].append(path)
    for name, where in found.items():
        assert where == [home], (
            f"{name} is defined in {[str(w.name) for w in where] or 'nowhere'} — it must have "
            f"exactly one home, {home.name}")


@pytest.mark.parametrize("mod", ["slide_tables.py", "soft_price_tables.py"])
def test_the_other_generators_import_the_paper_rules_rather_than_restating_them(mod):
    """The deck and the soft-price sweep must be the SAME aggregation, by construction.

    They print numbers a reader compares against the paper's, so a private copy of `cms_family`
    in either one is a table that can quietly diverge from the one it is supposed to mirror.
    """
    import ast

    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "eval/artifact" / mod).read_text())
    imported = {alias.name for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("paper_tables")
                for alias in node.names}
    assert "cms_family" in imported, \
        f"{mod} does not import the paper's family aggregation — is it re-deriving it?"
