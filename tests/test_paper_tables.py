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

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.artifact.paper_tables import cms_family, prop_rates, rival_family, zero_search  # noqa: E402


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
    v = cms_family(d, "s")
    assert (v["inf"], v["gpap0"], v["both0"]) == (1, 1, 1)
    assert v["gpap0_names"] == ["ourzero"] and v["both0_names"] == ["bothzero"]
    # the ledger must balance: geomean instances + each zero bucket == instance count
    assert v["n_gm"] + v["inf"] + v["gpap0"] + v["both0"] == v["inst"] == 4
    # our loss is in the denominator of wins but is not a win; a mutual zero is neither
    assert v["wins"] == 2 and v["n"] == 3


def test_gpap_zero_is_not_silently_a_win(tmp_path):
    """A stage where we fail everywhere must not report a flattering geomean."""
    d = stage(tmp_path, "s", [row(f"i{i}.bench", models=0, sps=0) for i in range(4)])
    v = cms_family(d, "s")
    assert v["gpap0"] == 4 and v["wins"] == 0 and v["gm"] is None


# --------------------------------------------------------------------- zero-search degeneracy

def test_zero_search_detected_and_excluded(tmp_path):
    """Samples emitted with no decisions AND no implications mean the support is free after
    propagation — no search happened, so the ratio measures the instance. The best-encoder rule
    actively PREFERS these cells because they carry the family's top ratio."""
    free = row("free.bench", sps=1e9, dec=0, impl=0)        # 100,000x, and entirely degenerate
    assert zero_search(free) and not zero_search(row("ok.bench"))
    d = stage(tmp_path, "s", [free] + [row(f"i{i}.bench") for i in range(3)])
    v = cms_family(d, "s")
    assert v["free"] == 1 and v["free_names"] == ["free"]
    assert v["gm"] == pytest.approx(100.0)                   # the three honest cells only
    assert v["gm_withfree"] > v["gm"]                        # and we report what it would inflate to


def test_best_encoder_rule_picks_the_degenerate_cell(tmp_path):
    """Regression for the mechanism, not just the symptom: on EPFL `ctrl` the Tseitin encoding
    samples at 2.0M/s with succ 0.07 while ABC collapses the output cone to succ 1.0 at 240M/s,
    so selecting the best encoder per instance HANDS the headline the zero-search cell."""
    d = stage(tmp_path, "s", [
        row("ctrl.aig", "tseitin", sps=2e6),
        row("ctrl.aig", "abc:fast", sps=2.4e8, dec=0, impl=0),
        row("a.aig"), row("b.aig"), row("c.aig"),
    ])
    v = cms_family(d, "s")
    assert v["free"] == 1, "the selected cell for ctrl must be recognised as zero-search"
    assert v["gm"] == pytest.approx(100.0)


# --------------------------------------------------------------------- the propagation identity

def test_propagation_identity_closes(tmp_path):
    """samples/s ratio == (impl/s ratio) / (impl-per-sample ratio). Algebra, so it can only fail
    if the columns are aggregated over different rows — which is what broke EPFL by 3x."""
    rows = [row(f"i{i}.bench", sps=10.0 ** (5 + i), impl=1e9, cms_impl=2e6 * (i + 1))
            for i in range(4)]
    d = stage(tmp_path, "s", rows)
    p, v = prop_rates(d, "s"), cms_family(d, "s")
    assert abs((p["ratio"] / p["work"]) / v["gm"] - 1.0) < 0.01


def test_propagation_columns_share_one_row_set(tmp_path):
    """An instance that propagates hard but emits NOTHING (EPFL `bar`) must not enter the
    implications ratio while being absent from work-per-sample."""
    d = stage(tmp_path, "s", [row(f"i{i}.bench") for i in range(3)]
              + [row("bar.aig", models=0, sps=0, impl=1e9)])
    p, v = prop_rates(d, "s"), cms_family(d, "s")
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


# --------------------------------------------------------------------- vacuous pins

def test_vacuous_pins_excluded_only_where_present(tmp_path):
    d = stage(tmp_path, "s", [row("b02.bench", sps=1e8)]          # tautological pin, huge ratio
              + [row(f"i{i}.bench") for i in range(3)])
    v = cms_family(d, "s")
    assert v["has_vac"] and v["gm"] == pytest.approx(100.0)
    d2 = stage(tmp_path, "t", [row(f"i{i}.bench") for i in range(3)])
    assert cms_family(d2, "t")["has_vac"] is False
