"""The reporting rules behind the paper's tables, pinned on hand-built stage JSONs.

Each rule stops a table that looks right from flattering us (docs/findings/L4_EVAL_FINDINGS.md §2).
No GPU and no data files, so a rule change shows up as a test diff.
"""
import json
import os
import sys
from pathlib import Path

import pytest

from eval.artifact.paper_tables import (cms_family, encoders_in, prop_rates,
                                        rival_family, tiny_space, vacuous_pin,
                                        zero_search)
from tests.conftest import REPO_ROOT as _REPO_ROOT


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
    """A GPAP-zero row (ratio 0.0) and a both-zero row (None) are falsy, so branching on `ratio`
    alone drops them while CMSGen-zero rows (inf) count as wins. Each zero gets its own bucket."""
    d = stage(tmp_path, "s", [
        row("win.bench"),                                   # both sampled, we win
        row("theirzero.bench", cms=0, cms_sps=0),           # CMSGen silent -> our win
        row("ourzero.bench", models=0, sps=0),              # we are silent  -> our loss
        row("bothzero.bench", models=0, sps=0, cms=0, cms_sps=0),
    ])
    v = cms_family(d, "s", "tseitin")
    assert (v["inf"], v["gpap0"], v["both0"]) == (1, 1, 1)
    assert v["gpap0_names"] == ["ourzero"] and v["both0_names"] == ["bothzero"]
    # geomean instances + each zero bucket == instance count
    assert v["n_gm"] + v["inf"] + v["gpap0"] + v["both0"] == v["inst"] == 4
    # our loss counts in the win denominator; a mutual zero is in neither
    assert v["wins"] == 2 and v["n"] == 3


def test_gpap_zero_is_not_silently_a_win(tmp_path):
    """A stage where we fail everywhere must not report a flattering geomean."""
    d = stage(tmp_path, "s", [row(f"i{i}.bench", models=0, sps=0) for i in range(4)])
    v = cms_family(d, "s", "tseitin")
    assert v["gpap0"] == 4 and v["wins"] == 0 and v["gm"] is None


# --------------------------------------------------------------------- zero-search degeneracy

def test_zero_search_detected_and_excluded(tmp_path):
    """Samples with no decisions and no implications mean the support is free after propagation:
    no search happened, so the ratio measures the instance and the cell leaves the geomean."""
    free = row("free.bench", sps=1e9, dec=0, impl=0)        # 100,000x, all degenerate
    assert zero_search(free) and not zero_search(row("ok.bench"))
    d = stage(tmp_path, "s", [free] + [row(f"i{i}.bench") for i in range(3)])
    v = cms_family(d, "s", "tseitin")
    assert v["free"] == 1 and v["free_names"] == ["free"]
    assert v["gm"] == pytest.approx(100.0)                   # the three honest cells only
    assert v["gm_withfree"] > v["gm"]                        # what it would inflate to


def test_encoders_are_separate_rows_and_neither_selects_the_other(tmp_path):
    """No best-of-encoders selection: picking the encoder with the highest `ratio` lets CMSGen's
    number choose which of our runs is reported, and favours degenerate cells (ABC's `ctrl`
    searches nothing). Each encoder is its own row, screened on its own
    (docs/findings/RIVAL_FAIRNESS_FINDINGS.md §8)."""
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
    """Table 1's ledger: `inst` = n + CMSGen=0 + GPAP=0 + excluded. Every reason a cell has no
    ratio (a screen, a both-zero, a refusal) goes in the one `excluded` column, named in the text;
    the per-reason counts must sum to it (docs/findings/RIVAL_FAIRNESS_FINDINGS.md §9)."""
    d = stage(tmp_path, "seq-c25-rare", [
        row("a.bench"), row("b.bench"),                       # 2 in the geomean
        row("c.bench", cms=0, cms_sps=0),                     # CMSGen=0
        row("d.bench", models=0, sps=0),                      # GPAP=0
        row("e.bench", models=0, sps=0, cms=0, cms_sps=0),    # both=0      -> excluded
        row("f.bench", dec=0, impl=0),                        # no-search   -> excluded
        row("b02.bench", sps=1e8),                            # vacuous pin -> excluded
        row("g.bench", support=5),                            # tiny space  -> excluded
        row("h.bench", cms=13, cms_sps=0.2),                  # 13 CMSGen samples: scored, and named
    ])
    v = cms_family(d, "seq-c25-rare", "tseitin")
    assert v["inst"] == 9
    assert v["n_gm"] + v["inf"] + v["gpap0"] + v["excl"] == v["inst"]
    assert v["excl"] == v["both0"] + v["free"] + v["vac"] + v["tiny"] == 4
    assert v["excl_names"] == {"vac": ["b02"], "free": ["f"], "tiny": ["g"], "both0": ["e"]}
    assert v["thin_names"] == ["h"] and v["thin_cells"] == [("h", 13)]
    # wins are counted over the same rows, not the pre-screen set
    assert v["n"] == v["n_gm"] + v["inf"] + v["gpap0"] == 5
    assert v["wins"] == 4                                     # a, b, h and the CMSGen-zero


def test_a_thin_denominator_is_scored_and_disclosed_not_screened(tmp_path):
    """A tool that scored anything is scored: a count under `THIN_DENOMINATOR` stays in the
    geomean and is named with its count for the caveats. Table 3 covers exactly the rows Table 1
    scores (docs/findings/RIVAL_FAIRNESS_FINDINGS.md §17b)."""
    import json as _json

    from eval.artifact.paper_tables import (THIN_DENOMINATOR, rival_cms_family,
                                            rival_prop_rates)

    assert THIN_DENOMINATOR == 30
    rows = [rrow("a"), rrow("b"), rrow("b14", cms=13, cms_sps=0.2, sps=45.0)]
    (tmp_path / "rivals_recurrent_rare.json").write_text(_json.dumps(rows))
    v = rival_cms_family(str(tmp_path), "recurrent_rare", "rcs")
    assert v["n_gm"] == 3, "the thin cell must enter the geomean"
    assert v["thin_cells"] == [("b14", 13)] and v["thin_names"] == ["b14"]
    assert "cms_thin" not in v and "excl" in v
    assert v["n_gm"] + v["inf"] + v["gpap0"] + v["excl"] == v["inst"] == 3
    # Table 3 is over the same rows
    p = rival_prop_rates(str(tmp_path), "recurrent_rare", "rcs")
    assert p["n"] == v["n_gm"] == 3


def test_a_cell_whose_sign_is_inside_gpap_seed_band_is_a_tie_not_a_win(tmp_path):
    """A cell whose ratio at our slowest and fastest seed straddles 1.0 is a tie, not a win;
    a clear loss stays a loss."""
    import json as _json

    from eval.artifact.paper_tables import rival_family, seed_band, straddles

    def cell(name, sps, lo, hi, rate):
        r = rrow(name, sps=sps)
        r["gpap"].update({"unique_per_s_est": sps, "sps_min": lo, "sps_max": hi})
        r["demotic"] = {"unique": 1000, "runtime_s": 1.0, "wall_s": 3.0, "unique_per_s": rate,
                        "num_inputs": 40}
        return r
    tie = cell("t", sps=120.0, lo=80.0, hi=130.0, rate=100.0)     # 1.2x, band 0.8–1.3x
    win = cell("w", sps=300.0, lo=280.0, hi=320.0, rate=100.0)    # 3.0x, band 2.8–3.2x
    loss = cell("l", sps=84.0, lo=68.0, hi=90.0, rate=100.0)      # 0.84x, band 0.68–0.90x
    assert straddles(seed_band(tie["gpap"], 120.0, 100.0))
    assert not straddles(seed_band(loss["gpap"], 84.0, 100.0))
    (tmp_path / "rivals_demotic_native.json").write_text(_json.dumps([tie, win, loss]))
    v = rival_family(str(tmp_path), "demotic_native", "demotic")
    assert v["ties"] == 1 and v["wins"] == 1 and v["n"] == 3, (v["ties"], v["wins"], v["n"])
    assert v["gm_lo"] < v["gm"] < v["gm_hi"]


def test_the_circuit_variant_sentence_is_read_off_the_rows(tmp_path):
    """The sentence naming which circuit file each side got is read off the rows, never fixed:
    one variant everywhere names it, two variants say the tie-break applied. (WHY the asymmetry
    is left uncorrected is standing policy and lives in docs/papers/tacas-27/README.md; which
    variant actually ran is per-campaign data and belongs in the output.)"""
    import json as _json
    import types

    from eval.artifact import paper_tables

    def render(rows):
        (tmp_path / "rivals_demotic_native.json").write_text(_json.dumps(rows))
        out = tmp_path / "r.md"
        args = types.SimpleNamespace(cmsgen=None, cmsgen_ref=None, rivals=str(tmp_path),
                                     uniformity=[], barbarik=[], recheck=None, soft=None,
                                     out_md=str(out), out_json=None)
        paper_tables.build(args)
        return out.read_text()

    one = render([dict(rrow("c432"), variant="abc")])
    assert "whichever suits it" not in one
    assert "`variant: abc` on every row" in one
    two = render([dict(rrow("c432"), variant="abc"), dict(rrow("c432"), variant="raw")])
    assert "whichever suits it" in two


def test_a_rival_failure_is_described_as_a_budget_statement(tmp_path):
    """A rival cell that did not initialize inside our budget is described as a statement about
    the budget, not as an instance the rival cannot process."""
    import json as _json
    import types

    from eval.artifact import paper_tables

    r = rrow("b14")
    r["rcs"] = {"error": "did not initialize: a 1-step start_point probe (sp=1) did not "
                         "complete within the 300s initialization budget"}
    (tmp_path / "rivals_recurrent_table.json").write_text(_json.dumps([r]))
    out = tmp_path / "r.md"
    args = types.SimpleNamespace(cmsgen=None, cmsgen_ref=None, rivals=str(tmp_path),
                                 uniformity=[], barbarik=[], recheck=None, soft=None,
                                 out_md=str(out), out_json=None)
    paper_tables.build(args)
    md = out.read_text()
    assert "cannot process" not in md
    assert "statement about OUR budget" in md and "re-runnable at a longer one" in md


def test_tiny_solution_space_is_screened_in_BOTH_tables(tmp_path):
    """A solution space smaller than the sample request is enumerated, not sampled, so the ratio
    is a race between startup costs. Both tables screen it (DEMOTIC's own Table II excludes
    `c17`); in Table 2 that removes a large win of ours."""
    assert tiny_space({"support": 5}) and not tiny_space({"support": 36})
    assert tiny_space({"demotic": {"num_inputs": 5}}, "demotic")
    assert not tiny_space({"demotic": {"num_inputs": 36}}, "demotic")
    assert not tiny_space({}, "demotic"), "absent size info must not be guessed at"

    # Table 2: the enormous ratio must not reach the geomean
    rows = [
        {"name": "c432", "variant": "abc", "demotic": {"unique_per_s": 10.0, "num_inputs": 36},
         "gpap": {"unique_per_s_est": 100.0, "n_models": 5}},
        {"name": "c17", "variant": "abc", "demotic": {"unique_per_s": 14.0, "num_inputs": 5},
         "gpap": {"unique_per_s_est": 640622.0, "n_models": 900944}},
    ]
    (tmp_path / "rivals_d.json").write_text(json.dumps({"rows": rows}))
    v = rival_family(str(tmp_path), "d", "demotic")
    assert v["tiny"] == 1 and v["n"] == 1
    assert v["gm"] == pytest.approx(10.0), "the 45,520x enumeration cell entered the geomean"
    assert (v["n"] + v["both0"] + v["nomeas"] + v["errors"] + v["free"] + v["misaligned"]
            + v["vacuous"] + v["tiny"]) == v["cells"]
    # the printed ledger: cells = n + rival=0 + GPAP=0 + failed + excluded (n holds the zeros)
    assert v["excl"] == 1 and v["excl_names"] == {"tiny": ["c17"]}
    assert v["cells"] == v["n"] + v["errors"] + v["excl"]


def test_rival_table_scores_every_measured_instance_and_has_no_paper_subset_column(tmp_path):
    """Table 2 scores every measured cell as one population per row. There is no `paper set`
    column, and no cell is dropped for being outside the rival paper's own table."""
    rows = [
        {"name": "s386", "variant": "raw", "rcs": {"unique_per_s": 10.0},
         "gpap": {"unique_per_s_est": 100.0, "n_models": 5}},          # in their Table II
        {"name": "b03", "variant": "raw", "rcs": {"unique_per_s": 1.0},
         "gpap": {"unique_per_s_est": 1000.0, "n_models": 5}},         # not in it: a 1,000x cell
    ]
    (tmp_path / "rivals_r.json").write_text(json.dumps({"rows": rows}))
    v = rival_family(str(tmp_path), "r", "rcs")
    assert v["n"] == 2 and v["gm"] == pytest.approx(100.0), "a measured cell was dropped"
    assert not any(k in v for k in ("gm_paper", "n_paper", "offpaper")), sorted(v)
    assert v["cells"] == v["n"] + v["errors"] + v["excl"] == 2


def test_vacuous_pin_is_detected_per_row_not_by_a_name_list(tmp_path):
    """A vacuous pin is read off the row (`p_accept == 1.0`), not only a fixed name list, so a new
    instance with a tautological pin (`b19` here) cannot enter the geomean."""
    assert vacuous_pin({"sim_baseline": {"p_accept": 1.0}})
    assert not vacuous_pin({"sim_baseline": {"p_accept": 0.5}})
    assert vacuous_pin({"p_accept": 1.0})                    # rival rows carry it top-level
    assert vacuous_pin({}, "b02"), "the name list must still work for pre-sim_baseline rows"
    assert not vacuous_pin({}, "b03")

    d = stage(tmp_path, "seq-c25-rare", [
        row("a.bench"), row("b.bench"), row("c.bench"),
        row("b19.bench", sps=1e8, sim_baseline={"p_accept": 1.0}),   # tautological pin
    ])
    v = cms_family(d, "seq-c25-rare", "tseitin")
    assert v["vac"] == 1 and v["n_gm"] == 3, "the tautological pin entered the geomean"
    assert v["gm"] == pytest.approx(100.0)


def test_table_3_screens_the_same_rows_as_table_1(tmp_path):
    """Table 3 applies Table 1's screens; its identity only holds over the rows the ratio is
    computed on."""
    rows = [row(f"i{i}.bench") for i in range(3)]
    rows.append(row("b19.bench", sps=1e8, sim_baseline={"p_accept": 1.0}))
    rows.append(row("tiny.bench", support=5))
    d = stage(tmp_path, "seq-c25-rare", rows)
    v = cms_family(d, "seq-c25-rare", "tseitin")
    pr = prop_rates(d, "seq-c25-rare", "tseitin")
    assert pr["n"] == v["n_gm"] == 3, \
        f"Table 3 scored {pr['n']} rows where Table 1 scored {v['n_gm']}"


# --------------------------------------------------------------------- the propagation identity

def test_propagation_identity_closes(tmp_path):
    """samples/s ratio == (impl/s ratio) / (impl-per-sample ratio). This is algebra, so it fails
    only if the columns are aggregated over different rows."""
    rows = [row(f"i{i}.bench", sps=10.0 ** (5 + i), impl=1e9, cms_impl=2e6 * (i + 1))
            for i in range(4)]
    d = stage(tmp_path, "s", rows)
    p, v = prop_rates(d, "s", "tseitin"), cms_family(d, "s", "tseitin")
    assert abs((p["ratio"] / p["work"]) / v["gm"] - 1.0) < 0.01


def test_propagation_columns_share_one_row_set(tmp_path):
    """An instance that propagates but emits nothing (like EPFL `bar`) must not enter the
    implications ratio while missing from work-per-sample."""
    d = stage(tmp_path, "s", [row(f"i{i}.bench") for i in range(3)]
              + [row("bar.aig", models=0, sps=0, impl=1e9)])
    p, v = prop_rates(d, "s", "tseitin"), cms_family(d, "s", "tseitin")
    assert p["n"] == 3, "the zero-emitting instance has no work-per-sample and must be excluded"
    assert abs((p["ratio"] / p["work"]) / v["gm"] - 1.0) < 0.01


# --------------------------------------------------------------------- rivals, same rules

def test_rival_zero_buckets_are_symmetric(tmp_path):
    """A rival cell where neither side scored is not a win for us."""
    rows = [
        {"name": "c432", "variant": "raw", "demotic": {"unique_per_s": 10.0},
         "gpap": {"unique_per_s_est": 100.0}},                       # our win
        {"name": "c499", "variant": "raw", "demotic": {"unique_per_s": 0.0},
         "gpap": {"unique_per_s_est": 100.0}},                       # rival silent -> our win
        {"name": "c880", "variant": "raw", "demotic": {"unique_per_s": 10.0},
         "gpap": {"unique_per_s_est": 0.0}},                         # we are silent -> our loss
        {"name": "c1355", "variant": "raw", "demotic": {"unique_per_s": 0.0},
         "gpap": {"unique_per_s_est": 0.0}},                         # neither -> nobody's win
        {"name": "c1908", "variant": "raw", "demotic": {"error": "timeout"}, "gpap": {}},
    ]
    (tmp_path / "rivals_x.json").write_text(json.dumps({"rows": rows}))
    v = rival_family(str(tmp_path), "x", "demotic")
    assert (v["zero"], v["gpap0"], v["both0"], v["errors"]) == (1, 1, 1, 1)
    assert v["wins"] == 2 and v["n"] == 3
    assert v["gm"] == pytest.approx(10.0)     # only the one finite ratio


def test_rival_unmeasured_unique_is_not_a_gpap_zero(tmp_path):
    """`unique_per_s_est` is None or 0 when the off-timing coverage run collected nothing. That is
    not a GPAP zero, which the timed run's `n_models` decides; reading it as one turns a win into
    a loss."""
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
    # the columns add up to the cells
    assert 1 + v["zero"] + v["gpap0"] + v["both0"] + v["nomeas"] + v["errors"] == v["cells"]


def test_rival_zero_search_cell_is_screened_not_scored(tmp_path):
    """A rival cell whose GPAP side did no search (zero decisions and implications, models
    emitted) measures the instance, not either sampler. It leaves through `free`, as in Table 1."""
    rows = [
        {"name": "c432", "variant": "raw", "demotic": {"unique_per_s": 10.0},
         "gpap": {"unique_per_s_est": 100.0, "n_models": 5,
                  "decisions_per_s": 1e7, "implications_per_s": 1e9}},
        {"name": "c3540", "variant": "raw", "demotic": {"unique_per_s": 1e-3},
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
    """A row that failed the per-row constraint-alignment check (`pin_check_ok`) compared two
    different tasks, so it leaves through the exclusion column instead of entering the geomean."""
    rows = [
        {"name": "c432", "variant": "raw", "pin_check_ok": True,
         "demotic": {"unique_per_s": 10.0}, "gpap": {"unique_per_s_est": 100.0}},
        {"name": "c880", "variant": "raw", "pin_check_ok": False,
         "demotic": {"unique_per_s": 1e-6}, "gpap": {"unique_per_s_est": 1e6}},  # a 1e12x "win"
    ]
    (tmp_path / "rivals_x.json").write_text(json.dumps({"rows": rows}))
    v = rival_family(str(tmp_path), "x", "demotic")
    assert v["misaligned"] == 1
    assert v["gm"] == pytest.approx(10.0) and v["n"] == 1, \
        "a row that failed the alignment check entered the ledger"
    # it leaves through the one printed exclusion column, named in the text
    assert v["excl"] == 1 and v["excl_names"] == {"misaligned": ["c880"]}


def test_rival_retry_row_replaces_rather_than_appends(tmp_path):
    """A resumed harness re-emits a row; the later row replaces the earlier one."""
    rows = [
        {"name": "a", "variant": "raw", "hts": {"error": "no pycryptosat"}, "gpap": {}},
        {"name": "a", "variant": "raw", "hts": {"unique_per_s": 10.0},
         "gpap": {"unique_per_s_est": 50.0}},
    ]
    (tmp_path / "rivals_h.json").write_text(json.dumps({"rows": rows}))
    v = rival_family(str(tmp_path), "h", "hts")
    assert v["cells"] == 1 and v["errors"] == 0 and v["gm"] == pytest.approx(5.0)


def test_rival_vacuous_pin_is_screened_like_table_1(tmp_path):
    """s526/b02/b05/b07 have constant last-cycle outputs at T=25, so pinning them is a tautology
    and every sampler scores ~100%. Table 2 screens them as Table 1 does (Recurrent's own set
    includes s526)."""
    rows = [
        {"name": "s386", "variant": "raw", "rcs": {"unique_per_s": 10.0},
         "gpap": {"unique_per_s_est": 100.0, "n_models": 5}},
        {"name": "s526", "variant": "raw", "rcs": {"unique_per_s": 1.0},   # tautological pin
         "gpap": {"unique_per_s_est": 1e6, "n_models": 1000}},
    ]
    (tmp_path / "rivals_r.json").write_text(json.dumps({"rows": rows}))
    v = rival_family(str(tmp_path), "r", "rcs")
    assert v["vacuous"] == 1, "the vacuous pin entered the rival ledger"
    assert v["n"] == 1 and v["gm"] == pytest.approx(10.0), \
        "the tautological cell moved the geomean"
    # the ledger still counts every cell exactly once
    assert (v["n"] + v["both0"] + v["nomeas"] + v["errors"] + v["free"] + v["misaligned"]
            + v["vacuous"]) == v["cells"]


def test_rival_vacuous_pin_is_screened_before_the_rival_error_bucket(tmp_path):
    """Screens come before buckets in Table 2 as in Table 1: `b19` is both a vacuous pin and a
    rival init failure, and both tables must book it as vacuous."""
    from eval.artifact.paper_tables import rival_cms_family
    rows = [rrow("s386", p_acc=0.05),
            rrow("b19", p_acc=1.0, rcs={"error": "did not initialize: a 1-step start_point probe"})]
    (tmp_path / "rivals_recurrent_rare.json").write_text(json.dumps(rows))
    t2 = rival_family(str(tmp_path), "recurrent_rare", "rcs")
    t1 = rival_cms_family(str(tmp_path), "recurrent_rare", "rcs")
    assert t2["vacuous"] == 1 and t2["errors"] == 0, \
        "a vacuous instance was booked as a rival failure"
    assert t1["vac"] == 1 and t1["vac_names"] == ["b19"]
    assert t2["vacuous"] == t1["vac"], "Table 1 and Table 2 disagree about the same stage"
    assert (t2["n"] + t2["both0"] + t2["nomeas"] + t2["errors"] + t2["free"] + t2["misaligned"]
            + t2["vacuous"]) == t2["cells"]


def test_rival_gpap_that_produced_nothing_is_a_zero_not_unmeasured(tmp_path):
    """A GPAP run that delivered no models (`unique_est_status` "no-sample-rows") is a GPAP zero.
    The unmeasured bucket is for a run that delivered models but could not estimate its distinct
    fraction."""
    rows = [rrow("s386", p_acc=0.05), rrow("b01", p_acc=0.0, p_accept_lt=7e-8, models=0)]
    rows[1]["gpap"]["unique_est_status"] = "no-sample-rows"
    rows[1]["rcs"] = {"unique_per_s": 3.66, "unique": 30, "unique_at_T": 0, "num_inputs": 40}
    (tmp_path / "rivals_recurrent_table.json").write_text(json.dumps(rows))
    v = rival_family(str(tmp_path), "recurrent_table", "rcs")
    assert v["nomeas"] == 0 and v["gpap0"] == 1, "a GPAP zero was booked as 'unmeasured'"
    # `n` counts the geomean's cells plus the two zero columns, as in Table 1
    assert (v["n"] + v["both0"] + v["nomeas"] + v["errors"] + v["free"] + v["misaligned"]
            + v["vacuous"]) == v["cells"]


def test_rival_scatter_only_plots_cells_the_table_scores(tmp_path):
    """The rival scatter classifies cells through `rival_cells`, so a cell Table 2 screens out is
    not plotted. The figure needs 3 comparable cells, so with 2 scored and 2 screened rows it
    must draw nothing."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from eval.eval_plots import fig_rivals_sps_scatter

    def ok(name):
        return {"name": name, "variant": "raw", "demotic": {"unique_per_s": 10.0, "num_inputs": 36},
                "gpap": {"unique_per_s_est": 100.0, "n_models": 5}}

    screened = [
        {"name": "c1355", "variant": "raw", "demotic": {"unique_per_s": 10.0, "num_inputs": 41},
         "gpap": {"unique_per_s_est": None, "unique_est_status": "no-coverage-sample",
                  "n_models": 42}},                             # unmeasured, not a GPAP zero
        {"name": "c1908", "variant": "raw", "demotic": {"error": "timeout"}, "gpap": {}},
    ]
    p = tmp_path / "rivals_demotic_native.json"
    p.write_text(json.dumps({"rows": [ok("c432"), ok("c499"), ok("c880")] + screened}))
    out = fig_rivals_sps_scatter(plt, [str(p)], str(tmp_path))
    assert out is not None and (tmp_path / "sps_scatter_rivals.png").exists()

    p.write_text(json.dumps({"rows": [ok("c432"), ok("c499")] + screened}))
    assert fig_rivals_sps_scatter(plt, [str(p)], str(tmp_path)) is None, \
        "screened cells were counted as plottable points"


def test_the_cmsgen_figures_classify_through_table_1s_rule(tmp_path):
    """The GPAP-vs-CMSGen figures classify through `rival_cms_cells`, so a vacuous pin or a
    refused instance is not plotted and the dumbbell's geomean is Table 1's. Same 3-cell probe
    as the rival scatter."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from eval.eval_plots import fig_cmsgen_sps_scatter, fig_vs_cmsgen_rival

    screened = [rrow("b19", p_acc=1.0),                                      # vacuous pin
                {"name": "b21", "variant": "raw", "rcs": {},
                 "gpap": {"error": "oversize"}}]                              # refused
    p = tmp_path / "rivals_recurrent_rare.json"
    p.write_text(json.dumps([rrow("s386"), rrow("b12"), rrow("b04")] + screened))
    assert fig_cmsgen_sps_scatter(plt, [str(p)], str(tmp_path)) is not None
    assert (tmp_path / "sps_scatter_cmsgen.png").exists()
    assert (tmp_path / "sps_scatter_cmsgen.pdf").exists()      # the paper's vector copy
    out = fig_vs_cmsgen_rival(plt, str(p), str(tmp_path))
    assert out is not None and (tmp_path / "vs_cmsgen_recurrent_rare.png").exists()
    p.write_text(json.dumps([rrow("s386"), rrow("b12")] + screened))
    assert fig_cmsgen_sps_scatter(plt, [str(p)], str(tmp_path)) is None, \
        "screened cells were counted as plottable points"


# --------------------------------------------------------------- summary.md, the same ledger

def test_summary_md_counts_zeros_the_papers_way(tmp_path):
    """`summary.md` uses the paper's zero ledger: CMSGen=0 and GPAP=0 each get a column, so
    CMSGen's zeros are not hidden inside our wins."""
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


def test_a_gpap_only_sweep_is_not_a_string_of_cmsgen_zero_wins(tmp_path):
    """Absent is not zero: rows from a GPAP-only sweep (no --cmsgen) carry no CMSGen arm and
    must not read as CMSGen scoring 0."""
    from eval.eval_plots import write_summary
    gpap_only = []
    for i in range(3):
        r = row(f"i{i}.bench")
        for k in ("cms_n_samples", "cms_sps", "ratio", "ratio_e2e"):
            r.pop(k, None)
        gpap_only.append(r)
    stage(tmp_path, "s", gpap_only, tag="s")
    # Table 1: a stage with no CMSGen arm has no row at all
    assert cms_family(str(tmp_path), "s", "tseitin") is None
    stages = {"s": json.loads((tmp_path / "s.json").read_text())}
    out = Path(write_summary(stages, str(tmp_path), str(tmp_path))).read_text()
    line = next(ln for ln in out.splitlines() if ln.startswith("| s |"))
    hdr = next(ln for ln in out.splitlines() if ln.startswith("| stage |"))
    got = dict(zip([c.strip() for c in hdr.split("|")], [c.strip() for c in line.split("|")]))
    assert got["wins"] == "—" and got["CMSGen=0"] == "—", got


def _armless(name, **kw):
    """A row from a sweep that never ran the CMSGen arm (no cms_* keys at all)."""
    r = row(name, **kw)
    for k in ("cms_n_samples", "cms_sps", "cms_implications_per_s", "cms_decisions_per_s",
              "ratio", "ratio_e2e"):
        r.pop(k, None)
    return r


def test_a_mixed_arm_stage_does_not_mint_phantom_cmsgen_zero_wins(tmp_path):
    """The absent-vs-zero screen is per row, not per stage: a resumed campaign can append armless
    rows to an armed file, and those must not become CMSGen=0 wins."""
    d = stage(tmp_path, "s", [
        row("win.bench"),                                   # both ran, we win
        row("realzero.bench", cms=0, cms_sps=0),            # CMSGen ran and scored 0: a real win
        _armless("neverran.bench"),                         # the comparison never happened
    ])
    v = cms_family(d, "s", "tseitin")
    assert v["inf"] == 1, "the armless row was minted into a CMSGen=0 win"
    assert v["noarm"] == 1 and v["noarm_names"] == ["neverran"]
    assert v["wins"] == 2 and v["n"] == 2
    # the ledger balances with the excluded row's count added back
    assert (v["n_gm"] + v["inf"] + v["gpap0"] + v["both0"] + v["free"] + v["vac"]
            + v["noarm"]) == v["inst"] == 3


def test_a_crashed_cmsgen_is_not_a_zero_win(tmp_path):
    """A CMSGen spawn failure is recorded as cms_n_samples=0 plus cms_error. It is an error, not a
    zero win."""
    d = stage(tmp_path, "s", [
        row("crashed.bench", cms=0, cms_sps=0.0, cms_error="FileNotFoundError: cmsgen"),
        row("ok.bench"),
    ])
    v = cms_family(d, "s", "tseitin")
    assert v["inf"] == 0 and v["cms_err"] == 1 and v["cms_err_names"] == ["crashed"]
    assert v["wins"] == 1 and v["n"] == 1


# --------------------------------------------------------------------- the verifier re-derives

def _verify(*argv):
    import subprocess
    root = _REPO_ROOT
    return subprocess.run(
        [sys.executable, os.path.join(root, "eval", "artifact", "verify_artifact.py"),
         *[str(a) for a in argv]], capture_output=True, text=True)


def test_verify_artifact_fails_a_mixed_arm_stage(tmp_path):
    """paper_tables screens a mixed-arm stage; the verifier fails it, since the fix is to re-run
    the armless cells."""
    stage(tmp_path, "s", [row("win.bench"), _armless("neverran.bench")])
    r = _verify(tmp_path)
    assert r.returncode == 1 and "MIXED CMSGen provenance" in r.stdout, r.stdout


def test_verify_artifact_fails_a_crashed_cmsgen_row(tmp_path):
    stage(tmp_path, "s", [row("crashed.bench", cms=0, cms_sps=0.0, cms_error="boom"),
                          row("ok.bench")])
    r = _verify(tmp_path)
    assert r.returncode == 1 and "ERRORED" in r.stdout, r.stdout


def test_verify_artifact_rederives_the_recheck_claim(tmp_path):
    """`--recheck` is re-derived from the raw rows; a mismatch in a deterministic field fails."""
    orig, rec = tmp_path / "orig", tmp_path / "rec"
    orig.mkdir(), rec.mkdir()
    stage(orig, "s", [row("a.bench"), row("b.bench")])
    stage(rec, "s", [row("a.bench"), row("b.bench", M=999)])       # b re-measured a different M
    r = _verify(orig, "--recheck", rec)
    assert r.returncode == 1 and "DETERMINISTIC" in r.stdout, r.stdout
    stage(rec, "s", [row("a.bench"), row("b.bench")])              # now they agree
    r = _verify(orig, "--recheck", rec)
    assert r.returncode == 0 and "deterministic fields match on all 2 shared cells" in r.stdout, \
        r.stdout


def test_verify_artifact_names_a_stage_that_never_wrote_its_json(tmp_path):
    """A stage that crashed before writing its JSON leaves no other trace, so the verifier names
    the missing stages."""
    stage(tmp_path, "sk", [row("x.cnf", enc=None)])
    r = _verify(tmp_path)
    assert r.returncode == 0
    assert "standard stage(s) with no JSON at all" in r.stdout, r.stdout
    assert "seq-c25-rare" in r.stdout


def test_uniformity_screens_subcube_instances(tmp_path):
    """Table 4 excludes full-subcube instances, as the uniformity campaign's own summary does:
    there the don't-care filler, not the config under test, draws the bits."""
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
    """When the screens remove every cell of a family, Table 3 has no row for it rather than
    falling back to the screened cells."""
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
    """A cell whose GPAP side spent fraction `pre` of its end-to-end wall preparing (parse,
    construct, ELS, warmup, mining) rather than sampling."""
    r = row(name, **kw)
    r["wall_total_s"] = wall_total
    r["phases"] = {"parse": 0.0, "construct": pre * wall_total, "warmup": 0.0,
                   "mine": 0.0, "rebuild": 0.0, "recheck": 0.0,
                   "timed": (1 - pre) * wall_total}
    # as benchmark.py records it: e2e rate = models / wall_total, i.e. the steady-state ratio
    # scaled by the sampling share of the wall
    r["ratio_e2e"] = (sps * (1 - pre)) / cms_sps
    return r


def test_e2e_column_charges_gpap_for_its_own_preprocessing(tmp_path):
    """CMSGen's rate already includes its setup, so `gm_e2e` charges GPAP for its preparation too
    and is lower than `gm` by exactly the preparation share."""
    d = stage(tmp_path, "seq-c25-rare",
              [timed_row("a", pre=0.5), timed_row("b", pre=0.5)])
    v = cms_family(d, "seq-c25-rare", "tseitin")
    assert v["gm"] == pytest.approx(100.0)          # 1e6 / 1e4, setup not charged
    assert v["gm_e2e"] == pytest.approx(50.0)       # half the wall was preparation
    assert v["pre_frac"] == pytest.approx(0.5)
    assert v["gm_e2e"] < v["gm"], "the honest column must never read higher than the headline"


def test_pre_frac_is_a_median_not_an_average_of_ratios(tmp_path):
    """`pre_frac` is a median over cells, so one slow-to-build instance does not set the family's
    preparation share."""
    rows = [timed_row("a", pre=0.1), timed_row("b", pre=0.1), timed_row("c", pre=0.9)]
    v = cms_family(stage(tmp_path, "seq-c25-rare", rows), "seq-c25-rare", "tseitin")
    assert v["pre_frac"] == pytest.approx(0.1)


# ------------------------------------------------- the parallelism grant handed to the rivals

def test_batch_is_sized_to_the_same_share_of_the_card_we_take():
    """A rival's unique/s is near-proportional to its batch, so the batch is a parallelism grant
    (docs/findings/RIVAL_FAIRNESS_FINDINGS.md §2). `fit_batch` uses the rule GPAP uses on itself:
    the largest batch that fits the VRAM budget."""
    from eval.rivals.rival_common import GRAPH_BYTES, fit_batch
    budget = 8 << 30
    # 1000 gates x 25 frames of DEMOTIC graph: budget / (work x bytes-per-row)
    work = 1000 * 25
    b = fit_batch(work, "demotic", budget_bytes=budget)
    assert work * GRAPH_BYTES["demotic"] * b <= budget
    assert fit_batch(work, "demotic", budget_bytes=budget * 4) > b, "a bigger card must grant more"


def test_batch_never_returns_something_that_cannot_run():
    """When no rung fits, the batch is still the smallest rung: a rival OOM is a result, a batch
    of 0 is a harness bug."""
    from eval.rivals.rival_common import BATCH_LADDER, fit_batch
    assert fit_batch(10 ** 12, "recurrent", budget_bytes=1 << 20) == BATCH_LADDER[0]


def test_the_rival_is_granted_the_largest_rung_that_runs_not_the_modelled_one():
    """The grant is the largest ladder rung that runs, walked down from the top. Our VRAM model
    predicts low and its rung under-grants the rival, so the model decides nothing here
    (docs/findings/RIVAL_FAIRNESS_FINDINGS.md §16)."""
    from eval.rivals.rival_common import run_at_batch
    calls = []

    def run(b):
        calls.append(b)
        return {"error": "CUDA out of memory"} if b > 100_000 else {"unique_per_s": float(b)}

    best, sweep = run_at_batch(run, 10_000, log=lambda *_: None)      # model says 10k
    assert best["unique_per_s"] == 100_000, "granted the model's rung instead of the largest"
    assert calls == [30_000_000, 10_000_000, 3_000_000, 1_000_000, 300_000, 100_000], \
        "the walk starts at the ladder's TOP (30,000,000 since protocol v1.0)"
    assert [x["batch"] for x in sweep] == calls, "every rung tried belongs in the row"


def test_run_at_batch_is_one_measurement_per_instance():
    """Rungs are rejected on feasibility (OOM, driver refusal, not finishing inside the wall
    budget), never on throughput, and exactly one rung is measured. Keeping a bigger rung because
    it ran faster would be a best-of-two over the rival's batch."""
    from eval.rivals.rival_common import run_at_batch
    calls = []

    def run(b):
        calls.append(b)
        return {"unique_per_s": float(b)}                # every rung runs; the top one wins

    best, sweep = run_at_batch(run, 100_000, log=lambda *_: None)
    assert best["unique_per_s"] == 30_000_000 and calls == [30_000_000], \
        f"the rival was measured more than once: {calls}"
    assert [x["batch"] for x in sweep] == [30_000_000]


def test_the_first_run_of_a_process_is_discarded():
    """The first GPU run of a process is slow (CUDA context, autotuning, clocks), so one small
    warmup run per (tool, process) is discarded. Per process, because the supervisor resumes in
    a fresh interpreter after every VM reclaim; small, because the cold start does not depend on
    the batch."""
    import eval.rivals.rival_common as rc
    rc._WARMED.discard("toolX")
    calls = []
    rc.run_at_batch(lambda b: (calls.append(b), {"unique_per_s": 1.0})[1],
                    1_000, log=lambda *_: None, tool="toolX")
    assert calls[0] == 256, "no small warmup run fired before the measurement"
    n_first = len(calls)
    rc.run_at_batch(lambda b: (calls.append(b), {"unique_per_s": 1.0})[1],
                    1_000, log=lambda *_: None, tool="toolX")
    assert len(calls) == n_first + 1, "the warmup fired twice in one process"


def test_the_warmup_uses_the_short_timeout_closure_when_one_is_given():
    """The discarded warmup goes through `warmup_run`, not the measured `run`, so the caller can
    cap it at a short timeout: a large circuit's graph build makes even a discarded run costly."""
    import eval.rivals.rival_common as rc
    rc._WARMED.discard("toolW")
    warmed, measured = [], []
    rc.run_at_batch(lambda b: (measured.append(b), {"unique_per_s": 1.0})[1], 1_000,
                    log=lambda *_: None, tool="toolW",
                    warmup_run=lambda b: (warmed.append(b), {"unique_per_s": 0.0})[1])
    assert warmed == [256], "the warmup did not fire through warmup_run"
    assert 256 not in measured, "the warmup closure leaked into the measurement"


# ------------------------------------------------- mirroring: results only exist once local

def test_mirror_accepts_every_artifact_type_not_just_json(tmp_path):
    """A rented VM's disk dies with it, so a result exists only once mirrored. The mirror
    validates by file type, so non-JSON artifacts such as the GPAP-arm cache tarball are kept."""
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
    """A file copied while the VM was writing it reads back truncated and must not replace a good
    local copy."""
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


def test_a_timeout_steps_down_but_the_probe_is_bounded():
    """A rung the rival cannot finish was not granted to it, so a timeout steps down like an OOM.
    The walk stays bounded: past `probe_budget_s` it jumps to the modelled rung and records the
    truncation in the row."""
    from eval.rivals.rival_common import run_at_batch
    calls = []

    def run(b):
        calls.append(b)
        return {"error": "timeout 300s"}

    best, sweep = run_at_batch(run, 30_000, log=lambda *_: None, probe_budget_s=0.0)
    assert "error" in best
    assert calls[0] == 30_000_000, "did not start at the top of the ladder"
    assert 30_000 in calls, "never fell back to the modelled rung"
    assert any(x.get("probe_truncated") for x in sweep), "truncation not recorded in the row"
    assert len(calls) < 8, f"walked the whole ladder on timeouts: {calls}"


# --------------------------------------------------------------- one home for the screens

#: The rules that decide which cells a published number covers. They are defined once, in
#: `eval/artifact/paper_tables.py`, and imported everywhere else: a re-derived copy drifts
#: silently, and two tables then disagree about what was excluded.
SCREENS = ["zero_search", "select_cells", "cms_family", "prop_rates", "rival_family",
           "encoders_in", "unique_unmeasured", "sound_failed"]


def _eval_modules():
    root = Path(_REPO_ROOT) / "eval"
    return sorted(p for p in root.rglob("*.py"))


def test_each_screening_rule_is_defined_exactly_once_across_eval():
    """A second definition anywhere under eval/ is drift, whatever it is called."""
    import ast

    home = Path(_REPO_ROOT) / "eval" / "artifact" / "paper_tables.py"
    found = {name: [] for name in SCREENS}
    for path in _eval_modules():
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name in found:
                found[node.name].append(path)
    for name, where in found.items():
        assert where == [home], (
            f"{name} is defined in {[str(w.name) for w in where] or 'nowhere'} — it must have "
            f"exactly one home, {home.name}")


@pytest.mark.parametrize("mod", ["artifact/slide_tables.py", "profiling/soft_price_tables.py"])
def test_the_other_generators_import_the_paper_rules_rather_than_restating_them(mod):
    """The deck and soft-price tables import `cms_family`, so their numbers cannot diverge from
    the paper's. Each is named by its path under `eval/`: they do not live in the same directory
    (the soft-price pair is `eval/profiling/`, outside the artifact), and a bare basename made
    that a silent skip rather than a failure the day one of them moved."""
    import ast

    root = Path(_REPO_ROOT) / "eval"
    tree = ast.parse((root / mod).read_text())
    imported = {alias.name for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("paper_tables")
                for alias in node.names}
    assert "cms_family" in imported, \
        f"{mod} does not import the paper's family aggregation — is it re-deriving it?"


# ---------------------------------------------------------------- Table 1 on the rival rows

def rrow(name, *, models=1000, sps=1e6, cms=1000, cms_sps=1e4, p_acc=0.01, dec=1e6, impl=1e8,
         cms_impl=1e6, cms_dec=1e5, arm=True, **extra):
    """A rival-stage row: the rival, the GPAP arm and the CMSGen arm on one CNF. A tool that
    emitted nothing has a zero rate, as in a real row."""
    sps = sps if models else 0.0
    cms_sps = cms_sps if cms else 0.0
    r = {"name": name, "variant": "raw", "cycles": 25, "p_accept": p_acc, "support": 40,
         "rcs": {"unique_per_s": 10.0, "num_inputs": 40},
         "gpap": {"sps": sps, "n_models": models, "success": 0.5, "encoder": "abc:fast",
                  "decisions_per_s": dec, "implications_per_s": impl, "sound_check": "ok",
                  "unique_per_s_est": sps or None, "unique_est_status": "ok",
                  "seeds": [{"sps": sps, "sps_e2e": sps / 2}],
                  "engine": {"core": "libgpap 0.3.0+abc"}},
         "cpu": {"model": "x"}, "gpu": {"name": "L4"}}
    if arm:
        r.update({"cms_mode": "nowrite", "cms_patched": True, "cms_sps": cms_sps,
                  "cms_sps_e2e": cms_sps * 0.9, "cms_n_samples": cms, "cms_count_exact": True,
                  "cms_implications_per_s": cms_impl, "cms_decisions_per_s": cms_dec})
    r.update(extra)
    return r


def test_table_1_is_read_off_the_rival_rows_with_table_2s_screens_and_table_1s_ledger(tmp_path):
    """Table 1 (GPAP vs CMSGen) comes from the same files and cells as Table 2: screened first
    (vacuous, no-search, tiny), then the symmetric zero ledger, where absent and crashed are not
    zero. Every instance is counted once and the columns add up."""
    from eval.artifact.paper_tables import (RIVAL_CMS_SETS, RIVALS, rival_cms_family,
                                            rival_prop_rates)

    assert [s for s, _, _ in RIVAL_CMS_SETS] == [s for s, _, _, _ in RIVALS], \
        "Tables 1 and 2 must name the same stages"
    rows = [rrow("s386", sps=1e6, cms_sps=1e4),               # finite 100x, in their paper
            rrow("b12", sps=2e5, cms_sps=1e4),                #  finite 20x, in their paper
            rrow("b04", sps=5e4, cms_sps=1e4),                #  finite  5x, not in their paper
            rrow("b11", cms=0),                               # CMSGen=0: a win, no ratio
            rrow("b03", models=0),                            # GPAP=0: a loss, no ratio
            rrow("b13", models=0, cms=0),                     # both=0
            rrow("b19", p_acc=1.0),                           # vacuous pin: screened first
            rrow("b10", dec=0, impl=0),                       # no-search: screened
            rrow("b01", arm=False),                           # absent is not zero
            rrow("b09", cms=0, cms_error="OSError: boom"),    # a crash is not a zero
            {"name": "b21", "variant": "raw", "rcs": {}, "gpap": {"error": "oversize"}}]
    (tmp_path / "rivals_recurrent_rare.json").write_text(json.dumps(rows))
    v = rival_cms_family(str(tmp_path), "recurrent_rare", "rcs")
    # `inst` is every instance of the set, `cells` the ones GPAP measured: b21 is refused
    assert v["inst"] == 11 and v["cells"] == 10 and v["errors"] == 1
    assert v["refused_names"] == ["b21"]
    assert v["n_gm"] == 3 and v["gm"] == pytest.approx((100 * 20 * 5) ** (1 / 3))
    assert "gm_paper" not in v, "one population per row (2026-09-07)"
    assert v["inf"] == 1 and v["gpap0"] == 1 and v["both0"] == 1
    assert v["vac"] == 1 and v["free"] == 1 and v["tiny"] == 0
    assert v["noarm"] == 1 and v["cms_err"] == 1
    # the one printed exclusion column, itemised by reason, and the identity a reader adds up
    assert v["excl"] == 6 and v["excl_names"] == {
        "vac": ["b19"], "free": ["b10"], "both0": ["b13"], "noarm": ["b01"],
        "cms_err": ["b09"], "refused": ["b21"]}
    assert v["inst"] == v["n_gm"] + v["inf"] + v["gpap0"] + v["excl"]
    assert v["wins"] == 4 and v["n"] == 5
    assert v["gpap0_names"] == ["b03"] and v["both0_names"] == ["b13"]
    assert v["cms_err_names"] == ["b09"] and v["noarm_names"] == ["b01"]
    # e2e charges both sides for getting ready: GPAP's n / wall_total vs CMSGen's subprocess wall
    assert v["gm_e2e"] == pytest.approx(v["gm"] / 1.8) and v["pre_frac"] == pytest.approx(0.5)
    assert v["depth"] == "T=25" and v["encoder"] == "abc:fast" and v["patched"] is True
    assert v["cores"] == ["libgpap 0.3.0+abc"] and v["unsound"] == 0 and v["cpu"] == "x"
    # Table 3 over the same cells, so the identity closes
    p = rival_prop_rates(str(tmp_path), "recurrent_rare", "rcs")
    assert p["n"] == 3 and p["encoder"] == "abc:fast"
    assert p["ratio"] / p["work"] == pytest.approx(v["gm"])


def test_a_rivals_only_sweep_has_no_table_1_and_a_refusal_is_not_a_zero(tmp_path):
    """With no CMSGen arm anywhere there is no Table 1 row, not a column of CMSGen zeros. A
    GPAP-side refusal (oversize, envelope, encode failure) is an error, not a zero."""
    from eval.artifact.paper_tables import rival_cms_family

    (tmp_path / "rivals_demotic_native.json").write_text(json.dumps(
        {"rows": [rrow("c432", arm=False), rrow("c499", arm=False)]}))
    assert rival_cms_family(str(tmp_path), "demotic_native", "demotic") is None
    (tmp_path / "rivals_demotic_native.json").write_text(json.dumps(
        {"rows": [rrow("c432"), {"name": "c7552", "variant": "raw", "demotic": {},
                                 "gpap": {"error": "encode failed: no abc"}}]}))
    v = rival_cms_family(str(tmp_path), "demotic_native", "demotic")
    assert v["inst"] == 2 and v["cells"] == 1 and v["errors"] == 1
    assert v["gpap0"] == 0 and v["both0"] == 0
    assert v["excl"] == 1 and v["excl_names"] == {"refused": ["c7552"]}


def test_verify_artifact_gates_the_cmsgen_arm_on_rival_rows(tmp_path):
    """Table 1 can come from rival rows alone, under the same gates: a clean stage passes; a
    floored zero, a mixed-arm stage and a crashed CMSGen each fail."""
    riv = tmp_path / "riv"
    riv.mkdir()
    f = riv / "rivals_recurrent_rare.json"
    f.write_text(json.dumps([rrow("s386"), rrow("b12")]))
    r = _verify("--rivals", str(riv))
    assert r.returncode == 0 and "CMSGen arm on 2/2" in r.stdout, r.stdout + r.stderr
    f.write_text(json.dumps([rrow("s386"),
                             rrow("b12", cms=0, cms_count_exact=False, cms_timed_out=True)]))
    r = _verify("--rivals", str(riv))
    assert r.returncode != 0 and "FLOORED" in r.stdout
    f.write_text(json.dumps([rrow("s386"), rrow("b12", arm=False)]))
    r = _verify("--rivals", str(riv))
    assert r.returncode != 0 and "MIXED" in r.stdout
    f.write_text(json.dumps([rrow("s386"), rrow("b12", cms=0, cms_error="boom")]))
    r = _verify("--rivals", str(riv))
    assert r.returncode != 0 and "ERRORED" in r.stdout
    r = _verify()
    assert r.returncode != 0, "nothing to verify must not read as publishable"


def test_the_generator_survives_a_stage_whose_every_cell_is_degenerate(tmp_path):
    """A stage whose only cell is screened as no-search has no geomean. The generator must still
    write both files, with that stage's Table 1 row present and its no-search count stated."""
    import subprocess

    riv = tmp_path / "riv"
    riv.mkdir()
    (riv / "rivals_recurrent_rare.json").write_text(json.dumps([rrow("b13", dec=0, impl=0)]))
    (riv / "rivals_demotic_native.json").write_text(json.dumps({"rows": [rrow("c432")]}))
    root = _REPO_ROOT
    r = subprocess.run([sys.executable, os.path.join(root, "eval", "artifact", "paper_tables.py"),
                        "--rivals", str(riv), "--out-md", str(tmp_path / "R.md")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    md = (tmp_path / "R.md").read_text()
    table1 = md.split("## Table 1")[1].split("## Table 2")[0]
    t1 = [ln for ln in table1.splitlines()
          if ln.startswith("| ITC'99") or ln.startswith("| ISCAS-85")]
    assert len(t1) == 2, table1
    rare = next(ln for ln in t1 if "| rare |" in ln)
    assert "| — |" in rare and "CMSGen=0" not in rare, rare   # no geomean, and not blamed on CMSGen
    assert "no **search at all**" in md or "no search at all" in md


def test_table_2_has_one_clock_the_rivals_own_and_no_end_to_end_column(tmp_path):
    """Table 2 has one clock: GPAP's steady-state unique/s over the rival's own metric, both
    excluding model build (RIVAL_FAIRNESS_FINDINGS.md §17). The rows carry every wall field an
    e2e column would read, and nothing may read them; Table 1 keeps its e2e column."""
    import os
    import subprocess
    import sys

    from eval.artifact.paper_tables import rival_family
    from tests.conftest import REPO_ROOT

    rows = [rrow("s386", sps=1e6), rrow("b12", sps=2e5), rrow("b19", p_acc=1.0)]
    for r in rows:
        r["rcs"].update({"unique_per_s": 1e3, "unique": 1e3, "runtime_s": 1.0, "wall_s": 9.0})
        r["gpap"].update({"sps_e2e": r["gpap"]["sps"] / 2, "n_models": 1000})
    (tmp_path / "rivals_recurrent_rare.json").write_text(json.dumps(rows))
    v = rival_family(str(tmp_path), "recurrent_rare", "rcs")
    assert v["n"] == 2 and v["gm"] == pytest.approx((1e3 * 2e2) ** 0.5)
    assert v["vacuous"] == 1
    assert not any(k.startswith(("gm_e2e", "e2e", "startup")) for k in v), sorted(v)
    md = tmp_path / "R.md"
    subprocess.run([sys.executable, os.path.join(REPO_ROOT, "eval", "artifact", "paper_tables.py"),
                    "--rivals", str(tmp_path), "--out-md", str(md)],
                   check=True, capture_output=True, cwd=REPO_ROOT)
    t2 = md.read_text().split("## Table 2")[1].split("## Table 3")[0].split("\n## ")[0]
    header = next(ln for ln in t2.splitlines() if ln.startswith("| rival | task"))
    assert "| e2e |" not in header and "| GPAP/rival | wins |" in header
    assert "paper set" not in header, "one population per row (2026-09-07)"
    assert "no end-to-end column here by rule" in t2, \
        "the table must say the column is absent on purpose, not by omission"
    t1 = md.read_text().split("## Table 1")[1].split("## Table 2")[0]
    assert "| e2e |" in next(ln for ln in t1.splitlines() if ln.startswith("| instance set")), \
        "Table 1 keeps its end-to-end column: both of ITS walls are the same kind of thing"


def test_verify_artifact_fails_a_rival_row_whose_samples_break_the_circuit(tmp_path):
    riv = tmp_path / "riv"
    riv.mkdir()
    r = rrow("s386")
    r["gpap"]["sim_check"], r["gpap"]["sim_check_bad"], r["gpap"]["sim_check_rows"] = "BAD", 3, 1000
    (riv / "rivals_recurrent_rare.json").write_text(json.dumps([r, rrow("b12")]))
    res = _verify("--rivals", str(riv))
    assert res.returncode != 0 and "simulate check" in res.stdout
    r["gpap"]["sim_check"], r["gpap"]["sim_check_bad"] = "ok", 0
    (riv / "rivals_recurrent_rare.json").write_text(json.dumps([r, rrow("b12")]))
    res = _verify("--rivals", str(riv))
    assert res.returncode == 0 and "reproduce the pinned outputs" in res.stdout


def test_tiny_space_reads_the_solution_count_not_the_input_hypercube():
    """The tiny-space screen reads the solution count, not the input hypercube (`c1355` under the
    rare pin: 41 input bits, ~700 solutions). The estimate is the minimum of the hypercube,
    p_accept x hypercube (its upper bound when no hit was seen) and the repeat-rate count."""
    from eval.artifact.paper_tables import solution_space_estimate, tiny_space
    from eval.rivals.rival_common import space_from_distinct

    assert space_from_distinct(1000, 1000) is None, "no repeat seen: the window says nothing"
    s = space_from_distinct(1000, 560)
    assert 500 < s < 900
    big_support = {"support": 41, "p_accept": 0.0, "p_accept_lt": 4.6e-8,
                   "gpap": {"seeds": [{"unique_rows": 1000, "unique_distinct": 560}]}}
    assert tiny_space(big_support), "700 solutions on 41 bits is enumerated, not sampled"
    assert solution_space_estimate(big_support, 41) < 1000
    no_repeats = {"support": 41, "p_accept": 0.1,
                  "gpap": {"seeds": [{"unique_rows": 1000, "unique_distinct": 1000}]}}
    assert not tiny_space(no_repeats)
    assert tiny_space({"support": 41, "p_accept": 1e-10}), "p x 2^41 = 220 solutions"
    assert tiny_space({"support": 41, "p_accept": 0.0, "p_accept_lt": 1e-10}), \
        "the 95% upper bound is the honest estimate when no hit was observed"
    assert not tiny_space({"support": 41, "p_accept": 0.0, "p_accept_lt": 1e-6})
    # a bench row carries the repeat rate on the row itself (the coverage pass)
    assert tiny_space({"support": 41, "n_collected": 2000, "n_distinct": 300})
    assert not tiny_space({"support": 41}), "no evidence: the hypercube, as before"


# ---------------------------------------------- one run per row, exact counts, more screens

def test_the_whole_delivered_set_is_deduplicated_not_a_window():
    """Both rivals deduplicate their whole output exactly, off their own clock, so
    `exact_distinct` does the same over every delivered row instead of estimating from a window."""
    import numpy as np
    from eval.rivals.rival_common import exact_distinct

    rows = np.array([[1, 0, 1], [1, 0, 1], [0, 1, 1], [0, 0, 0]], dtype=np.uint8)
    assert exact_distinct(rows) == 3
    assert exact_distinct(np.zeros((0, 5), dtype=np.uint8)) == 0
    wide = np.zeros((1000, 250), dtype=np.uint8)          # packbits path, one distinct row
    assert exact_distinct(wide) == 1


def test_a_published_row_is_one_run_not_a_field_wise_median():
    """A row is one run. Medianing each field independently publishes a row no execution
    produced, and Table 3 would divide one seed's implications by another seed's models."""
    from eval.rivals.rival_common import median_run

    runs = [{"seed": 5, "sps": 10.0, "n_models": 111, "impl": 1.0},
            {"seed": 6, "sps": 30.0, "n_models": 222, "impl": 2.0},
            {"seed": 7, "sps": 20.0, "n_models": 333, "impl": 3.0}]
    pick, order = median_run(runs, "sps")
    assert pick["seed"] == 7 and pick["sps"] == 20.0
    assert (pick["n_models"], pick["impl"]) == (333, 3.0), "fields came from another run"
    assert (order[0]["sps"], order[-1]["sps"]) == (10.0, 30.0), "spread must stay reportable"


def test_a_proved_unsat_cell_is_screened_before_any_zero():
    """A cell proved UNSAT at its depth is a fact about the instance, not a GPAP=0 loss, and gets
    its own bucket. `None` (unasked, or unresolved inside the budget) is not a proof."""
    from eval.artifact.paper_tables import rival_cells

    row = rrow("b01", models=0, cms=0, p_acc=0.0)
    row["gpap"]["unsat_at_T"] = True
    row["rcs"] = {"unique_per_s": 30.0, "unique_at_T": 0, "num_inputs": 40}
    (name, _, bucket, ours, theirs), = rival_cells([row], "rcs")
    assert (name, bucket) == ("b01", "unsat")

    unknown = rrow("b02x", models=0, cms=0, p_acc=1e-9)
    unknown["gpap"]["unsat_at_T"] = None
    unknown["rcs"] = {"unique_per_s": 0.0, "num_inputs": 40}
    (_, _, bucket2, _, _), = rival_cells([unknown], "rcs")
    assert bucket2 == "both0", "an unresolved UNSAT probe was read as a proof"


def test_the_circuit_variant_scored_is_the_one_better_for_the_rival():
    """Where a stage holds both circuit variants, the table scores the one better for the rival
    (the tie-break `num_steps` uses) instead of counting the instance twice."""
    from eval.artifact.paper_tables import rival_cells

    raw = rrow("c880", sps=1e6)
    raw["variant"], raw["rcs"] = "raw", {"unique_per_s": 100.0, "num_inputs": 40}
    abc = rrow("c880", sps=1e6)
    abc["variant"], abc["rcs"] = "abc", {"unique_per_s": 250.0, "num_inputs": 40}
    cells = list(rival_cells([raw, abc], "rcs"))
    assert len(cells) == 1, "both variants were scored, counting the instance twice"
    name, variant, bucket, ours, theirs = cells[0]
    assert (variant, theirs) == ("abc", 250.0), "scored the variant worse for the rival"
    assert ours / theirs == 1e6 / 250.0


def test_a_nearly_free_cell_is_screened_like_a_free_one():
    """A cell doing about one implication per model is reading bits out of a hypercube, so the
    zero-search screen is a threshold, not an exact zero (`c432` under the rare pin)."""
    from eval.artifact.paper_tables import zero_search

    near_free = {"n_models": 933_900, "sps": 1.079e8, "decisions_per_s": 3.16e8,
                 "implications_per_s": 1.176e8}                    # 1.09 implications per model
    assert zero_search(near_free), "the c432-shaped cell is still in the geomean"
    # the next unscreened cell measured 44.1 implications per model, so any threshold in
    # (1.09, 44.1) gives the same partition
    real = {"n_models": 50_000, "sps": 1e5, "decisions_per_s": 1e6, "implications_per_s": 4.41e6}
    assert not zero_search(real), "a cell doing 44 implications per model is not free"
    exact_zero = {"n_models": 10, "sps": 1e5, "decisions_per_s": 0, "implications_per_s": 0}
    assert zero_search(exact_zero), "the original exact-zero rule must still fire"


def test_the_emitted_ledger_sums_to_the_cell_count(tmp_path):
    """Every ledger row of the emitted tables sums to its cell count. This parses the FILE rather
    than the dicts behind it: the dicts are tested elsewhere, and what this catches is a renderer
    printing the wrong column into the right place."""
    import json
    import re
    import subprocess
    import sys

    rows = [rrow("c17", models=8, cms=8, p_acc=0.5),                     # tiny space
            rrow("b02", models=1000, p_acc=1.0),                         # vacuous pin
            rrow("c880"), rrow("c2670"), rrow("c3540"),
            rrow("b11", models=0, cms=500),                              # GPAP zero
            rrow("b15", models=0, cms=0)]                                # both zero
    rows[6]["gpap"]["unsat_at_T"] = True                                 # ...and proved UNSAT
    for stem in ("demotic_native", "recurrent_table"):
        (tmp_path / f"rivals_{stem}.json").write_text(json.dumps({"rows": rows}))
    md = tmp_path / "R.md"
    import os

    from tests.conftest import REPO_ROOT
    subprocess.run([sys.executable, os.path.join(REPO_ROOT, "eval", "artifact", "paper_tables.py"),
                    "--rivals", str(tmp_path), "--out-md", str(md)],
                   check=True, capture_output=True, cwd=REPO_ROOT)

    # The two ledger identities, by COLUMN NAME: `total` must equal the sum of `parts`.
    LEDGERS = (("inst", ("n", "CMSGen=0", "GPAP=0", "excluded")),
               ("cells", ("n", "rival=0", "GPAP=0", "rival failed", "excluded")))

    def cells_of(line):
        return [c.strip() for c in line.strip().strip("|").split("|")]

    checked, hdr = 0, None
    for line in md.read_text().splitlines():
        # A column map is only valid inside its own section. Without this reset the rows of
        # every LATER table were read against Table 1's header, which on the real campaign's
        # output parsed a ratio cell as a count -- green on a fixture with no table after it.
        if line.startswith("#"):
            hdr = None
        if not line.startswith("| "):
            continue
        col = cells_of(line)
        if any(c in col for c, _ in LEDGERS):          # a header row: index it by name
            hdr = {name: i for i, name in enumerate(col)}
            continue
        if hdr is None or line.startswith("|---") or len(col) != len(hdr):
            continue
        for total_col, part_cols in LEDGERS:
            if total_col not in hdr or not all(p in hdr for p in part_cols):
                continue
            want = col[hdr[total_col]]
            if not re.fullmatch(r"\d+", want):
                continue
            parts = [int(col[hdr[p]]) for p in part_cols if re.fullmatch(r"\d+", col[hdr[p]])]
            assert len(parts) == len(part_cols), f"a ledger column is not a count: {col}"
            assert sum(parts) == int(want), \
                f"ledger does not close on {col[0]}: {parts} vs {want}"
            checked += 1
    assert checked >= 2, "no ledger rows were emitted to check"


def test_the_documented_protocol_describes_the_harness_that_exists():
    """The protocol write-up describes the current harness.

    This gate used to read the generator's own methods paragraph, which recited the policy into
    every RESULTS.md. The policy is standing, not per-run, so it now lives in the venue doc and
    the generator emits numbers — but the gate is the same one and matters as much: prose cannot
    be re-derived from the rows, so nothing except a phrase list stops it going stale while every
    number around it stays correct. The old rules' wording must be absent, the current rules'
    present."""
    import re
    # Whitespace-normalised: a phrase in a wrapped markdown paragraph straddles a line break,
    # and a gate that depended on where the wrap fell would fail on a reflow and pass on a
    # deleted rule.
    doc = re.sub(r"\s+", " ",
                 (Path(_REPO_ROOT) / "docs" / "papers" / "tacas-27" / "README.md").read_text())
    for phrase in ["a timeout recorded as the result",
                   "stepping down a rung only if the driver refuses",
                   "scored on whichever of two encoders is higher"]:
        assert phrase not in doc, f"the protocol doc still describes an old rule: {phrase!r}"
    for phrase in ["largest rung of a ladder walked down from 30,000,000",
                   "rejected only on feasibility, never on how fast",
                   "shipped script",
                   "Duplicates are removed after the timed run",
                   "first run of every process is discarded",
                   "1,200 s",
                   "no best-of-encoders selection"]:
        assert phrase.lower() in doc.lower(), \
            f"the protocol doc does not state the current rule: {phrase!r}"


def test_the_rival_calibration_evidence_is_printed_not_just_pointed_at(tmp_path):
    """The generator prints the rival step calibration (every arm, its ratio to the pinned arm,
    the verdict), so the results show whether the pinned setting was the better one for the
    rival."""
    import inspect
    import json as _json

    from eval.artifact import paper_tables
    from eval.artifact.paper_tables import rival_calibration

    assert rival_calibration(str(tmp_path)) == [], "no file -> nothing printed, no crash"
    (tmp_path / "rivals_steps_calibration.json").write_text(_json.dumps({
        "generated": "x", "tools": {
            "recurrent": {"pinned": 1, "pinned_source": "their script", "measured_best": 1,
                          "agrees": True, "cycles": 25,
                          "arms": {"1": {"unique_per_s": 1000.0, "wall_s": 10.0, "error": None},
                                   "5": {"unique_per_s": 400.0, "wall_s": 20.0, "error": None},
                                   "10": {"unique_per_s": 120.0, "wall_s": 30.0, "error": None},
                                   "paper(5,50)": {"unique_per_s": 300.0, "wall_s": 25.0,
                                                   "error": None, "steps": 5, "lr": 50.0}}},
            "demotic": {"pinned": 1, "pinned_source": "their script", "measured_best": 10,
                        "agrees": False, "cycles": 25,
                        "arms": {"1": {"unique_per_s": 100.0, "wall_s": 1.0, "error": None},
                                 "10": {"unique_per_s": 150.0, "wall_s": 2.0, "error": None}}}}}))
    md = "\n".join(rival_calibration(str(tmp_path)))
    for needle in ["| 1 | 1,000 | 10 | 1.00x | **pinned** |",     # the pinned arm, ratio 1
                   "| 5 | 400 | 20 | 0.40x |",                    # the paper's steps, singly
                   "| paper(5,50) | 300 | 25 | 0.30x | paper (paired) |",
                   "tie-break HOLDS",                             # recurrent's verdict
                   "**FLIP**: 10 measured faster than the pinned 1",  # demotic's, printed not acted on
                   "EVIDENCE, not configuration"]:
        assert needle in md, f"calibration section lacks {needle!r}:\n{md}"
    # build() emits it under --rivals (checked in the source: build returns nothing)
    assert "rival_calibration(args.rivals)" in inspect.getsource(paper_tables.build)


def test_the_e2e_column_belongs_to_the_same_run_as_the_rate_it_is_paired_with():
    """`_gpap_e2e` reads the e2e rate of the run picked for `sps`, not its own median across
    seeds, so a ratio never mixes two seed selections."""
    from eval.artifact.paper_tables import _gpap_e2e

    g = {"sps": 500.0, "sps_e2e": 400.0,
         "seeds": [{"sps": 100.0, "sps_e2e": 90.0},
                   {"sps": 500.0, "sps_e2e": 400.0},
                   {"sps": 900.0, "sps_e2e": 111.0}]}
    assert _gpap_e2e(g) == 400.0, "the e2e column came from a different seed than the row"
    legacy = {"seeds": [{"sps_e2e": 90.0}, {"sps_e2e": 400.0}, {"sps_e2e": 111.0}]}
    assert _gpap_e2e(legacy) == 111.0, "a pre-fix row must still read as it always did"


def test_a_rival_that_found_a_handful_of_solutions_is_scored_and_disclosed(tmp_path):
    """A rival that found only a few solutions is scored, not floored out: the cell enters the
    geomean and `rival_family` names it with its count and ratio for the caveats. A rival that
    found nothing stays a zero."""
    from eval.artifact.paper_tables import THIN_DENOMINATOR, rival_cells, rival_family

    def cell(name, n_unique, ups):
        r = rrow(name, sps=8.45e6)
        r["rcs"] = {"unique": n_unique, "unique_per_s": ups, "num_inputs": 40}
        r["gpap"]["unique_per_s_est"] = 8.45e6
        return r

    (_, _, bucket, ours, rate), = rival_cells([cell("c880", 1, 1.2)], "rcs")
    assert bucket == "ok" and ours / rate == pytest.approx(8.45e6 / 1.2), \
        "a rival that scored anything is scored (2026-09-07)"
    (_, _, ok_bucket, o2, r2), = rival_cells([cell("c2670", THIN_DENOMINATOR * 10, 500.0)], "rcs")
    assert ok_bucket == "ok" and o2 / r2 > 0
    # a rival that found nothing is an outright win, not a thin denominator
    (_, _, zero_bucket, _, _), = rival_cells([cell("c499", 0, 0.0)], "rcs")
    assert zero_bucket == "rival0", "a rival zero must stay an outright win"
    # the aggregate names the thin cell with its count and ratio
    rows = [cell("c880", 1, 1.2), cell("c2670", THIN_DENOMINATOR * 10, 500.0)]
    (tmp_path / "rivals_demotic_comb_rare.json").write_text(json.dumps(rows))
    v = rival_family(str(tmp_path), "demotic_comb_rare", "rcs")
    assert v["n"] == 2 and v["thin"] == 1 and v["thin_names"] == ["c880"]
    (nm, n_uniq, x), = v["thin_cells"]
    assert (nm, n_uniq) == ("c880", 1) and x == pytest.approx(8.45e6 / 1.2)
    assert "rival_thin" not in v and "thin" in v


def test_the_table_generators_import_without_torch_or_numpy():
    """The table generators import without torch, numpy, triton or matplotlib, so the tables can
    be regenerated on a CPU-only machine. They reach the soundness verdict through the
    stdlib-only `gpap.verdict`, since anything under `gpap.sampling` imports torch. Checked in a
    subprocess with those modules blocked, because this interpreter has them installed."""
    import subprocess
    import sys

    probe = (
        "import sys\n"
        "class B:\n"
        "    def find_spec(self, n, p=None, t=None):\n"
        "        if n.split('.')[0] in ('torch', 'numpy', 'triton', 'matplotlib'):\n"
        "            raise ImportError(n + ' is blocked')\n"
        "sys.meta_path.insert(0, B())\n"
        f"sys.path.insert(0, {_REPO_ROOT!r})\n"
        "import eval.artifact.paper_tables, eval.artifact.verify_artifact\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                         cwd=_REPO_ROOT, timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == "ok", \
        f"a table generator pulls in a GPU dependency:\n{out.stderr[-2000:]}"


# ------------------------------- the paper's absolute rate columns and the numbers its text quotes

def test_the_rate_columns_quotient_is_the_ratio_column(tmp_path):
    """The paper prints each side's ABSOLUTE rate where the generator prints their ratio, so the
    two must be one claim: a geomean of quotients is the quotient of the geomeans. If this ever
    parts, a table and the text beside it describe different cell sets and the identity that
    makes the speedup readable off the rates is gone."""
    import json as _json

    from eval.artifact.paper_tables import rival_cms_family, rival_family

    rows = [rrow("s386", sps=1e6, cms_sps=1e4), rrow("b12", sps=4e6, cms_sps=1e4)]
    (tmp_path / "rivals_recurrent_rare.json").write_text(_json.dumps(rows))
    v = rival_cms_family(str(tmp_path), "recurrent_rare", "rcs")
    assert v["gm_gpap_sps"] == pytest.approx(2e6) and v["gm_cms_sps"] == pytest.approx(1e4)
    assert v["gm_gpap_sps"] / v["gm_cms_sps"] == pytest.approx(v["gm"])
    # and on the other clock, over its own cells
    assert v["gm_gpap_e2e"] / v["gm_cms_e2e"] == pytest.approx(v["gm_e2e"])
    assert v["n_rate"] == v["n_gm"] == 2

    # Table 2 carries the same identity on the rivals' own metric
    riv = [{"name": "b11", "variant": "raw", "rcs": {"unique_per_s": 100.0, "num_inputs": 40},
            "gpap": {"unique_per_s_est": 1e4, "n_models": 10, "sps": 1e4, "support": 40}},
           {"name": "b12", "variant": "raw", "rcs": {"unique_per_s": 400.0, "num_inputs": 40},
            "gpap": {"unique_per_s_est": 1e4, "n_models": 10, "sps": 1e4, "support": 40}}]
    (tmp_path / "rivals_q.json").write_text(_json.dumps({"rows": riv}))
    w = rival_family(str(tmp_path), "q", "rcs")
    assert w["gm_rival"] == pytest.approx(200.0) and w["gm_gpap"] == pytest.approx(1e4)
    assert w["gm_gpap"] / w["gm_rival"] == pytest.approx(w["gm"])


def test_a_rate_column_stands_on_exactly_the_ratios_own_cells(tmp_path):
    """A rate geomean may not quietly widen the cell set the ratio beside it was computed over.
    A CMSGen-zero cell has a GPAP rate and no ratio: folding it into the GPAP column alone would
    raise our printed rate while leaving the ratio where it was, which is the one direction that
    flatters us and the hardest to see in a finished table."""
    import json as _json

    from eval.artifact.paper_tables import rival_cms_family

    rows = [rrow("a", sps=1e6, cms_sps=1e4),
            # `impl` scales with the rate or `zero_search` reads the cell as a free support
            # and screens it BEFORE the zero ledger — which is what this fixture first did.
            rrow("b", sps=1e9, cms=0, impl=1e12, dec=1e11),
            rrow("c", models=0, cms_sps=1e4)]            # we are silent: our loss, no ratio
    (tmp_path / "rivals_recurrent_rare.json").write_text(_json.dumps(rows))
    v = rival_cms_family(str(tmp_path), "recurrent_rare", "rcs")
    assert (v["inf"], v["gpap0"], v["n_gm"]) == (1, 1, 1)
    assert v["n_rate"] == 1, "the rate stands on the one cell with a ratio"
    assert v["gm_gpap_sps"] == pytest.approx(1e6), "the 1e9 CMSGen-zero cell must not enter it"
    assert v["gm_gpap_sps"] / v["gm_cms_sps"] == pytest.approx(v["gm"])


def test_the_delivery_count_reads_the_request_off_the_row(tmp_path):
    """"GPAP filled the request on 90 instances, CMSGen on 42" is not a rate and is in no
    geomean, so it needs its own count — over the cells either tool scored on, with the request
    taken from the row rather than assumed. A row that records no request is counted apart: "we
    do not know" and "it fell short" are different claims about a tool."""
    import json as _json

    from eval.artifact.paper_tables import filled_request, rival_cms_family

    rows = [rrow("a", models=50_000, cms=50_000, cms_n_requested=50_000),   # both filled
            rrow("b", models=60_000, cms=10, cms_n_requested=50_000),       # overshoot still fills
            rrow("c", models=7, cms=0, cms_n_requested=50_000),             # CMSGen-zero cell
            rrow("d", models=40_000, cms=50_000)]                           # no request recorded
    (tmp_path / "rivals_recurrent_rare.json").write_text(_json.dumps(rows))
    v = rival_cms_family(str(tmp_path), "recurrent_rare", "rcs")
    assert v["n_req"] == 50_000
    assert v["full_gpap"] == 2 and v["full_cms"] == 1
    assert v["full_unknown"] == 1, "the request-less row is named, not scored either way"
    # the counts share ONE denominator: the cells either tool scored on
    assert v["n"] == 4
    # and the primitive is explicit about not knowing
    assert filled_request({"cms_n_samples": 1, "gpap": {"n_models": 1}}) is None


def test_a_published_number_is_checked_at_its_own_precision():
    """`check_paper_numbers.py` compares what the PDF prints against what the generator derived.
    A printed rate claims only its significant figures, so it is compared at those; a printed
    COUNT has no precision to round to and must match exactly. Without the asymmetry the check
    would either reject every rate or accept a wrong `n`."""
    from eval.artifact.check_paper_numbers import agrees, sig_digits

    assert (sig_digits("2.31e6"), sig_digits("1.10e5"), sig_digits("0.94"),
            sig_digits("12899")) == (3, 3, 2, 5)
    assert agrees("2.31e6", 2_305_714.0)[0], "anything rounding to 2.31e6 at 3 s.f. agrees"
    assert not agrees("2.30e6", 2_305_714.0)[0], "the third figure is a claim and is checked"
    assert agrees("1.59", 1.58869)[0] and not agrees("1.60", 1.58869)[0]
    assert agrees(16, 16)[0] and not agrees(16, 17)[0], "a count is exact"
    assert not agrees("2.31e6", None)[0], "a number the generator does not produce is a failure"


def test_a_narrower_end_to_end_cell_set_is_disclosed_not_assumed(tmp_path):
    """A row prints ONE `n` and TWO rate columns. They cover the same cells only while every
    scored row carries both clocks: a row with no `sps_e2e` narrows the end-to-end geomean while
    `n` goes on claiming the wider set, and a reader dividing the printed e2e rate by the printed
    `n`-cell CMSGen rate would get a ratio over cells that were never compared. The published
    campaign has no such row — which is exactly why the path needs a test rather than a reading."""
    import os
    import subprocess
    import sys

    from eval.artifact.paper_tables import rival_cms_family
    from tests.conftest import REPO_ROOT

    rows = [rrow("s386", sps=1e6, cms_sps=1e4),
            # scored on the steady-state clock, and carrying NO end-to-end rate on either route
            dict(rrow("b12", sps=2e6, cms_sps=1e4),
                 gpap=dict(rrow("b12", sps=2e6)["gpap"], seeds=[{"sps": 2e6}]))]
    (tmp_path / "rivals_recurrent_rare.json").write_text(json.dumps(rows))
    v = rival_cms_family(str(tmp_path), "recurrent_rare", "rcs")
    assert v["n_gm"] == v["n_rate"] == 2, "both cells have a steady-state ratio"
    assert v["n_rate_e2e"] == 1, "only one carries an end-to-end rate"
    assert v["gm_gpap_e2e"] / v["gm_cms_e2e"] == pytest.approx(v["gm_e2e"]), \
        "the e2e rates must still quotient to the e2e ratio, over their own narrower set"

    md = tmp_path / "R.md"
    r = subprocess.run([sys.executable,
                        os.path.join(REPO_ROOT, "eval", "artifact", "paper_tables.py"),
                        "--rivals", str(tmp_path), "--out-md", str(md)],
                       capture_output=True, text=True, cwd=REPO_ROOT, timeout=120)
    assert r.returncode == 0, r.stderr
    text = md.read_text()
    assert "the end-to-end column stands on FEWER cells" in text, \
        "a narrower e2e cell set must be stated in the generated numbers, not left to the reader"
    assert "1 of 2" in text


def test_every_table_generator_still_imports():
    """Every module under eval/artifact/ and eval/profiling/ imports.

    `soft_price_tables.py` imported `tex_escape` from `paper_tables`, and removing that helper
    with the LaTeX renderer broke it silently: nothing in the suite imported the supplementary
    generators, so a repo-wide check for USERS of a symbol was the only thing standing between a
    dead-code sweep and a broken script. This is that check, as a test rather than a habit.

    Discovered by running it, not by reading: the dead-code scan had looked inside
    `paper_tables.py` and found no remaining use, which is true and not the question.

    BOTH directories, because the file that broke is now in the second one: the supplementary
    generators moved to `eval/profiling/` so that `eval/artifact/` holds only what the artifact
    drivers run, and a gate that followed the old path would have stopped covering the one script
    it was written for."""
    import importlib
    import pkgutil

    broken = []
    for where in ("eval.artifact", "eval.profiling"):
        pkg = importlib.import_module(where)
        for mod in pkgutil.iter_modules(pkg.__path__):
            if mod.name.startswith("_"):
                continue
            try:
                importlib.import_module(f"{where}.{mod.name}")
            except Exception as e:                       # noqa: BLE001 — report, never hide
                broken.append(f"{where}.{mod.name}: {type(e).__name__}: {e}")
    assert not broken, "modules that no longer import:\n  " + "\n  ".join(broken)


def test_verify_artifact_tells_a_gpap_zero_we_measured_from_one_we_abandoned(tmp_path):
    """A GPAP zero is only a loss if we looked.

    The sampler's zero-yield valve ends a run once a few chunks come back empty, which on a
    low-success instance is seconds into a 60 s budget. So `n_models == 0` carries two opposite
    readings -- the instance defeated us, or we stopped looking -- and only `stop` separates
    them. Measured on the published campaign's own zero cells, three of six yielded once the
    valve was lifted, so this is not a hypothetical: the gate must name the abandoned ones,
    leave the ones that spent their budget alone, and say so when the field is missing (every
    published row predates it).
    """
    riv = tmp_path / "riv"
    riv.mkdir()
    f = riv / "rivals_recurrent_rare.json"

    def zero(name, stop, wall):
        r = rrow(name, models=0)
        if stop is not None:
            r["gpap"]["stop"], r["gpap"]["wall_timed_s"] = stop, wall
        return r

    # abandoned early, spent the clock, and recorded before the field existed
    f.write_text(json.dumps([rrow("ok"), zero("gaveup", "zero_yield", 2.1),
                             zero("spent", "time", 60.0), zero("silent", None, None)]))
    out = _verify("--rivals", str(riv)).stdout
    # the NAME LIST of the warning, not the whole line: its prose says "actually spent", which a
    # substring check for the fixture's name matches -- a green-for-the-wrong-reason trap.
    import re as _re
    # greedy to the em dash: the list's own entries carry parentheses (`b17 (2.1s)`), so a
    # non-greedy match to the first `)` truncates mid-entry.
    flagged = _re.search(r"ZERO-YIELD valve[^(]*\((.*)\) —", out)
    assert flagged, out
    assert "gaveup (2.1s)" in flagged.group(1), flagged.group(1)
    assert "spent" not in flagged.group(1), \
        "a cell that spent its whole budget is a real zero and must NOT be flagged"
    assert "record no `stop`" in out and "silent" in out, out

    # nothing to warn about when every zero spent its budget
    f.write_text(json.dumps([rrow("ok"), zero("spent", "time", 60.0)]))
    out = _verify("--rivals", str(riv)).stdout
    assert "ZERO-YIELD valve" not in out and "record no `stop`" not in out, out
