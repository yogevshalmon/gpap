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
                                        rival_family, tiny_space, vacuous_pin,
                                        zero_search)


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
    """`inst` = n_gm + CMSGen=0 + GPAP=0 + both=0 + no-search + tiny-space + vacuous + CMSGen<30,
    under Table 1. It was FALSE on 6 of 7 families: the degenerate cells were dropped from the
    geomean but left inside `inst` and inside `wins`, so a reader adding the columns up came out
    short."""
    d = stage(tmp_path, "seq-c25-rare", [
        row("a.bench"), row("b.bench"),                       # 2 in the geomean
        row("c.bench", cms=0, cms_sps=0),                     # CMSGen=0
        row("d.bench", models=0, sps=0),                      # GPAP=0
        row("e.bench", models=0, sps=0, cms=0, cms_sps=0),    # both=0
        row("f.bench", dec=0, impl=0),                        # no-search
        row("b02.bench", sps=1e8),                            # vacuous pin
        row("g.bench", support=5),                            # tiny space: 2**5 < 1000
        row("h.bench", cms=13, cms_sps=0.2),                  # CMSGen<30: a thin denominator
    ])
    v = cms_family(d, "seq-c25-rare", "tseitin")
    assert v["inst"] == 9
    assert (v["n_gm"] + v["inf"] + v["gpap0"] + v["both0"] + v["free"] + v["vac"]
            + v["tiny"] + v["cms_thin"]) == v["inst"]
    assert v["cms_thin_names"] == ["h"]
    # and the win count is over the same rows, not over the pre-screen set
    assert v["n"] == v["n_gm"] + v["inf"] + v["gpap0"] == 4
    assert v["wins"] == 3                                     # a, b, and the CMSGen-zero


def test_the_thin_denominator_floor_is_the_same_on_both_sides_of_both_tables(tmp_path):
    """Table 2 screens a rival that found fewer than MIN_SOLUTIONS solutions (`rival<30`);
    Table 1 must hold CMSGen to the SAME floor, in its own column that sums with the rest.
    Until 2026-09-05 the CMSGen side was only narrated in a caveat: `b14` (13 CMSGen samples in
    60 s) sat inside the rare-pin geomean at 212x and full weight. The screen is conservative —
    every cell it removes is a win of ours."""
    import json as _json

    from eval.artifact.paper_tables import MIN_SOLUTIONS, RIVAL_MIN_SOLUTIONS, rival_cms_family

    assert RIVAL_MIN_SOLUTIONS == MIN_SOLUTIONS == 30
    rows = [rrow("a"), rrow("b"), rrow("b14", cms=13, cms_sps=0.2, sps=45.0)]
    (tmp_path / "rivals_recurrent_rare.json").write_text(_json.dumps(rows))
    v = rival_cms_family(str(tmp_path), "recurrent_rare", "rcs")
    assert v["cms_thin"] == 1 and v["cms_thin_names"] == ["b14"]
    assert v["n_gm"] == 2, "the thin cell must not enter the geomean"
    assert (v["n_gm"] + v["inf"] + v["gpap0"] + v["both0"] + v["free"] + v["vac"] + v["tiny"]
            + v["unsat"] + v["cms_thin"]) == v["inst"] == 3
    # and Table 3 is over the SAME rows — the one thing the identity can catch
    from eval.artifact.paper_tables import rival_prop_rates
    p = rival_prop_rates(str(tmp_path), "recurrent_rare", "rcs")
    assert p["n"] == v["n_gm"] == 2


def test_table_2_e2e_is_one_definition_charged_the_same_way_on_both_sides(tmp_path):
    """Table 2's `e2e` has been wrong three ways in three rounds: dividing GPAP's wall by the
    rival's UNCHANGED timer (0.8x); then by the rival's whole SUBPROCESS wall, which charged it
    ~4.5 s of interpreter + CUDA start-up per cell that GPAP's in-process wall never pays, while
    our side subtracted its delivery and carried no encoding (7.6x on cells whose whole wall is
    under a second). ONE definition now: each side's whole wall on the instance, process
    start-up excluded on both — GPAP charged encoding and delivery; the rival its timed loop plus
    its non-timed wall above the tool's start-up floor, the smallest non-timed wall it showed."""
    import json as _json

    from eval.artifact.paper_tables import rival_family

    def cell(name, unique, runtime, wall, encode_s, n_models, rate_with_delivery):
        r = rrow(name, sps=1000.0)
        r["gpap"].update({"sps_e2e": 500.0, "unique_per_s_est": 1000.0, "n_models": n_models,
                          "median_seed": 5,
                          "seeds": [{"seed": 5, "sps_e2e": 500.0,
                                     "sps_e2e_with_delivery": rate_with_delivery}]})
        r["cnf"] = {"encode_s": encode_s}
        r["demotic"] = {"unique": unique, "runtime_s": runtime, "wall_s": wall,
                        "unique_per_s": unique / runtime, "num_inputs": 40}
        return r

    # A: the tool's smallest non-timed wall (4 s) — pure start-up, so its e2e is its steady state.
    # B: 7 s non-timed, of which 3 s exceed the floor (a real model build) and are charged.
    a = cell("c1", unique=100, runtime=1.0, wall=5.0, encode_s=0.0, n_models=1000,
             rate_with_delivery=500.0)
    b = cell("c2", unique=200, runtime=2.0, wall=9.0, encode_s=1.0, n_models=1000,
             rate_with_delivery=400.0)
    (tmp_path / "rivals_demotic_native.json").write_text(_json.dumps([a, b]))
    v = rival_family(str(tmp_path), "demotic_native", "demotic")
    assert abs(v["startup_s"] - 4.0) < 1e-9, "the floor is the smallest non-timed wall"
    # GPAP A: 1000 models / (1000/500 + 0) = 500/s; rival A: 100 / (1.0 + 0) = 100/s -> 5x
    # GPAP B: 1000 / (1000/400 + 1.0) = 285.7/s; rival B: 200 / (2.0 + 3.0) = 40/s -> 7.14x
    ga, gb = 500.0, 1000.0 / (1000.0 / 400.0 + 1.0)
    ea, eb = ga / 100.0, gb / 40.0
    assert abs(v["gm_e2e"] - (ea * eb) ** 0.5) < 1e-9, "e2e: start-up excluded on both sides"
    # the printed upper bound charges the rival its WHOLE subprocess wall
    na, nb = ga / (100.0 / 5.0), gb / (200.0 / 9.0)
    assert abs(v["gm_e2e_nosub"] - (na * nb) ** 0.5) < 1e-9
    assert v["e2e_nodelivery"] == 0
    # a row without the delivery split is counted, not silently accepted
    b["gpap"]["seeds"][0].pop("sps_e2e_with_delivery")
    (tmp_path / "rivals_demotic_native.json").write_text(_json.dumps([a, b]))
    assert rival_family(str(tmp_path), "demotic_native", "demotic")["e2e_nodelivery"] == 1


def test_a_cell_whose_sign_is_inside_gpap_seed_band_is_a_tie_not_a_win(tmp_path):
    """`wins 20/20` beside a caveat calling two of those cells ties contradicted itself. A cell
    whose ratio at our slowest and fastest seed straddles 1.0 is counted in `ties`, and `wins`
    excludes it; a clear loss stays a loss."""
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
    """The methods paragraph said every rival was handed the ABC-simplified netlist AS WELL AS
    the raw one and scored on whichever suited it — over a campaign whose 82 rows were all
    `variant: abc`. A fairness procedure asserted and not performed. The sentence must come
    from the rows: one variant on every row names that variant and says the axis was not run;
    two variants says the tie-break was applied."""
    import json as _json
    import types

    from eval.artifact import paper_tables

    def render(rows):
        (tmp_path / "rivals_demotic_native.json").write_text(_json.dumps(rows))
        out = tmp_path / "r.md"
        args = types.SimpleNamespace(cmsgen=None, cmsgen_ref=None, rivals=str(tmp_path),
                                     uniformity=[], barbarik=[], recheck=None, soft=None,
                                     out_md=str(out), out_tex=None)
        paper_tables.build(args)
        return out.read_text()

    one = render([dict(rrow("c432"), variant="abc")])
    assert "whichever suits it" not in one
    assert "`variant: abc` on every row" in one and "raw-netlist axis was not run" in one
    two = render([dict(rrow("c432"), variant="abc"), dict(rrow("c432"), variant="raw")])
    assert "whichever suits it" in two


def test_a_rival_failure_is_described_as_a_budget_statement(tmp_path):
    """Every Recurrent 'failure' of the 2026-09-04 campaign was one thing — the initialization
    budget expiring — and the caveat called them 'instances the rival cannot process', which our
    own §11 measurement contradicts (`b14` initializes at a longer cap and delivers). A cell
    that did not finish inside OUR budget is a statement about the budget."""
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
                                 out_md=str(out), out_tex=None)
    paper_tables.build(args)
    md = out.read_text()
    assert "cannot process" not in md
    assert "statement about OUR budget" in md and "re-runnable at a longer one" in md


def test_the_markdown_and_latex_ledger_prose_state_the_same_sum():
    """The LaTeX caption gained `+ unsat` and the Markdown paragraph did not — the third time a
    rule fixed in one emitter was not carried to its twin. Every bucket the tables print must
    appear in BOTH prose formulas."""
    import inspect

    from eval.artifact import paper_tables

    src = inspect.getsource(paper_tables)
    assert "+ vacuous + unsat + CMSGen<30" in src.replace("\"\n         \"", ""), \
        "the Markdown ledger formula is short of the printed columns"
    assert r"\text{vac.} + \text{unsat} + \text{C}{<}30$" in src, \
        "the LaTeX ledger formula is short of the printed columns"
    # and the algebraic-identity caveat lives in both Table 3 texts
    assert src.count("ALGEBRAIC") >= 2, "the Table 3 caveat is in one emitter only"


def test_tiny_solution_space_is_screened_in_BOTH_tables(tmp_path):
    """A space smaller than the sample request is enumerated, not sampled — both tools saturate
    it and the ratio is a race between startup costs. ISCAS-85 `c17` has 5 inputs = 32 points:
    DEMOTIC finds 19 of them at 14 unique/s against GPAP's 640k/s, a 45,520x cell that alone
    lifts the DEMOTIC-native geomean from 2.7x to ~6.5x. DEMOTIC's own Table II carves it out
    ("a minimum of 1000 distinct solutions (except c17)").

    Screened SYMMETRICALLY, and it does not flatter us: in Table 2 it removes a huge win, while
    in Table 1 the same instance was DRAGGING the combinational geomean down (1,620x -> 1,348x),
    so the screen gives that number back. A screen that only ever helped would be suspect."""
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


def test_rival_table_scores_every_measured_instance_and_shows_the_paper_subset_beside_it(tmp_path):
    """Table 2 runs every b*/s* circuit the campaign can process (user's rule, 2026-09-03) and
    scores every cell that ran. The subset the rival's OWN paper evaluates is not neutral —
    b03/b04/b11/b13 are in Recurrent's run.py target table but NOT in its Table II, and with
    them 'their target table' reads 4.1x where the paper's own set reads 3.0x — so that geomean
    is printed BESIDE the whole one (`gm_paper`, `n_paper`), never instead of it, and never
    used to drop a measured cell."""
    rows = [
        {"name": "s386", "variant": "raw", "rcs": {"unique_per_s": 10.0},
         "gpap": {"unique_per_s_est": 100.0, "n_models": 5}},          # in their Table II
        {"name": "b03", "variant": "raw", "rcs": {"unique_per_s": 1.0},
         "gpap": {"unique_per_s_est": 1000.0, "n_models": 5}},         # NOT in it: a 1,000x cell
    ]
    (tmp_path / "rivals_r.json").write_text(json.dumps({"rows": rows}))
    v = rival_family(str(tmp_path), "r", "rcs")
    assert v["n"] == 2 and v["gm"] == pytest.approx(100.0), "a measured cell was dropped"
    assert v["n_paper"] == 1 and v["gm_paper"] == pytest.approx(10.0)
    assert v["offpaper"] == 1
    assert (v["n"] + v["both0"] + v["nomeas"] + v["errors"] + v["free"] + v["misaligned"]
            + v["vacuous"] + v["tiny"]) == v["cells"]


def test_vacuous_pin_is_detected_per_row_not_by_a_name_list(tmp_path):
    """`VACUOUS_PINS` was a hardcoded {b02,b05,b07,s526} — correct only while the instance list
    stood still. Running the FULL sequential family brought in `b19`, whose pin is a tautology
    (sim_baseline p_accept == 1.0 at both depths and both targets), and it entered the geomean as
    a 491x cell measuring nothing: seq-c25-rare read 72.1x instead of 64.7x.

    The signal is in the row already, so use it."""
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
    """'Every column must use the SAME rows' — the propagation table filtered on the old name
    list while Table 1 moved to the measured screens, so a vacuous or tiny-space instance could
    sit in the engine row and not the sampler row (or the reverse). The identity in Table 3 is
    only meaningful over the same set the ratio is computed on."""
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
        {"name": "c432", "variant": "raw", "demotic": {"unique_per_s": 10.0},
         "gpap": {"unique_per_s_est": 100.0}},                       # our win
        {"name": "c499", "variant": "raw", "demotic": {"unique_per_s": 0.0},
         "gpap": {"unique_per_s_est": 100.0}},                       # rival silent -> our win
        {"name": "c880", "variant": "raw", "demotic": {"unique_per_s": 10.0},
         "gpap": {"unique_per_s_est": 0.0}},                         # WE are silent -> our loss
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
    """Table 2's text says constraint alignment is VERIFIED per row (`pin_check_ok`). A row that
    FAILED that check compared two different tasks; whatever its ratio measures, it is not the
    samplers — so it must leave through its own bucket, not sit silently inside the geomean."""
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


def test_rival_vacuous_pin_is_screened_like_table_1(tmp_path):
    """s526/b02/b05/b07 have constant last-cycle outputs at T=25, so pinning them is a tautology
    and every sampler scores ~100%. Table 1 has always screened them; the rival table did not,
    which was invisible while no vacuous instance was in the rival set — and became live when the
    set became each rival's OWN published one (Recurrent's Table II includes s526)."""
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
    # and the ledger still accounts for every cell exactly once
    assert (v["n"] + v["both0"] + v["nomeas"] + v["errors"] + v["free"] + v["misaligned"]
            + v["vacuous"]) == v["cells"]


def test_rival_vacuous_pin_is_screened_before_the_rival_error_bucket(tmp_path):
    """`b19` at T=25 is vacuous (measured p_accept 1.0) AND Recurrent-CircuitSAT fails to
    initialize on it. Screens come before buckets in Table 2 exactly as in Table 1: with the
    error bucket first, Table 1 counted b19 as vacuous while Table 2 booked the same row as a
    rival failure — one stage, two ledgers (caught on the 2026-09-03 L4 campaign)."""
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
    """`b01` under Recurrent's own target at T=25 (p_accept < 7e-8): GPAP delivered 0 models in
    3 x 60 s, so no kept row and no distinct fraction — `unique_est_status` "no-sample-rows".
    That is a GPAP ZERO (the rival's depth-pooled count found 30), not "unmeasured": the
    unmeasured bucket is for a run that DID deliver models and could not estimate its distinct
    fraction, which is what its caveat says. Caught on the 2026-09-03 L4 campaign."""
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
    """The rival unique/s scatter classifies cells through `rival_cells`, imported — so a cell
    Table 2 screens out (an unmeasured unique/s estimate, a failed rival run) must not become a
    dot. The probe is the figure's 3-comparable-cell floor: with 2 ok rows plus 2 screened rows,
    an implementation that plotted everything it loaded would see 4 points and draw; returning
    None proves the screen ran before the count."""
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
                  "n_models": 42}},                             # unmeasured — NOT a GPAP zero
        {"name": "c1908", "variant": "raw", "demotic": {"error": "timeout"}, "gpap": {}},
    ]
    p = tmp_path / "rivals_demotic_native.json"
    p.write_text(json.dumps({"rows": [ok("c432"), ok("c499"), ok("c880")] + screened}))
    out = fig_rivals_sps_scatter(plt, [str(p)], str(tmp_path))
    assert out is not None and (tmp_path / "sps_scatter_rivals.png").exists()

    p.write_text(json.dumps({"rows": [ok("c432"), ok("c499")] + screened}))
    assert fig_rivals_sps_scatter(plt, [str(p)], str(tmp_path)) is None, \
        "screened cells were counted as plottable points"


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




def test_a_gpap_only_sweep_is_not_a_string_of_cmsgen_zero_wins(tmp_path):
    """ABSENT is not ZERO: rows from a GPAP-only sweep (no --cmsgen) carry no cms arm at all,
    and reading the missing key as "CMSGen scored 0" published `wins 10/10 | CMSGen=0: 10` over
    a tool that never ran — found live on the 2026-09-02 engine-A/B campaign's summary.md."""
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
    """The stage-level absent-vs-zero guard is not enough: a campaign resumed on a VM where the
    cmsgen binary failed to restore appends ARMLESS rows to an armed file, the stage passes
    "does anything here have an arm?", and every armless row falls into the CMSGen=0 bucket as
    a win over a tool that never ran. The screen must be per ROW."""
    d = stage(tmp_path, "s", [
        row("win.bench"),                                   # both ran, we win
        row("realzero.bench", cms=0, cms_sps=0),            # CMSGen RAN and scored 0: a real win
        _armless("neverran.bench"),                         # the comparison never happened
    ])
    v = cms_family(d, "s", "tseitin")
    assert v["inf"] == 1, "the armless row was minted into a CMSGen=0 win"
    assert v["noarm"] == 1 and v["noarm_names"] == ["neverran"]
    assert v["wins"] == 2 and v["n"] == 2
    # the ledger still balances once the excluded row's own count is added back
    assert (v["n_gm"] + v["inf"] + v["gpap0"] + v["both0"] + v["free"] + v["vac"]
            + v["noarm"]) == v["inst"] == 3


def test_a_crashed_cmsgen_is_not_a_zero_win(tmp_path):
    """cmsgen_run records a spawn failure (missing binary, bad glibc on a fresh VM) as
    cms_n_samples=0 plus cms_error. Scoring that zero as "CMSGen sampled nothing" publishes a
    broken environment as a stage of outright wins — the floored-count trap's sibling."""
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
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.run(
        [sys.executable, os.path.join(root, "eval", "artifact", "verify_artifact.py"),
         *[str(a) for a in argv]], capture_output=True, text=True)


def test_verify_artifact_fails_a_mixed_arm_stage(tmp_path):
    """paper_tables SCREENS a mixed stage; the verifier must FAIL it — the fix is re-running
    the armless cells, not publishing around them."""
    stage(tmp_path, "s", [row("win.bench"), _armless("neverran.bench")])
    r = _verify(tmp_path)
    assert r.returncode == 1 and "MIXED CMSGen provenance" in r.stdout, r.stdout


def test_verify_artifact_fails_a_crashed_cmsgen_row(tmp_path):
    stage(tmp_path, "s", [row("crashed.bench", cms=0, cms_sps=0.0, cms_error="boom"),
                          row("ok.bench")])
    r = _verify(tmp_path)
    assert r.returncode == 1 and "ERRORED" in r.stdout, r.stdout


def test_verify_artifact_rederives_the_recheck_claim(tmp_path):
    """--recheck used to be accepted and IGNORED — the 'VERIFIED per row, enforced by nobody'
    shape. The verifier must re-derive the deterministic-match claim from the raw rows and fail
    on a mismatch."""
    orig, rec = tmp_path / "orig", tmp_path / "rec"
    orig.mkdir(), rec.mkdir()
    stage(orig, "s", [row("a.bench"), row("b.bench")])
    stage(rec, "s", [row("a.bench"), row("b.bench", M=999)])       # b re-measured a DIFFERENT M
    r = _verify(orig, "--recheck", rec)
    assert r.returncode == 1 and "DETERMINISTIC" in r.stdout, r.stdout
    stage(rec, "s", [row("a.bench"), row("b.bench")])              # now they agree
    r = _verify(orig, "--recheck", rec)
    assert r.returncode == 0 and "deterministic fields match on all 2 shared cells" in r.stdout, \
        r.stdout


def test_verify_artifact_names_a_stage_that_never_wrote_its_json(tmp_path):
    """A stage that crashes before writing its JSON leaves no other trace: the drivers print
    their completion markers unconditionally and every other check only sees files that exist."""
    stage(tmp_path, "sk", [row("x.cnf", enc=None)])
    r = _verify(tmp_path)
    assert r.returncode == 0
    assert "standard stage(s) with no JSON at all" in r.stdout, r.stdout
    assert "seq-c25-rare" in r.stdout


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


def test_the_rival_is_granted_the_largest_rung_that_runs_not_the_modelled_one():
    """THE grant rule (2026-09-04). Our VRAM model was measured at num_steps=10 and we run
    num_steps=1, so it predicts LOW — and a gradient sampler's unique/s is near-proportional to
    its batch, so predicting low takes the rival's score away. Measured on the L4 the day before
    the 2026-09-03 campaign: the modelled rung under-grants c880 1.53x, c2670 2.12x, c3540 1.85x,
    c7552 3.87x. That campaign shipped exactly that under-grant. The walk now starts at the TOP
    of the ladder, and the model's opinion decides nothing."""
    from eval.rivals.rival_common import run_at_batch
    calls = []

    def run(b):
        calls.append(b)
        return {"error": "CUDA out of memory"} if b > 100_000 else {"unique_per_s": float(b)}

    best, sweep = run_at_batch(run, 10_000, log=lambda *_: None)      # model says 10k
    assert best["unique_per_s"] == 100_000, "granted the model's rung instead of the largest"
    assert calls == [3_000_000, 1_000_000, 300_000, 100_000]
    assert [x["batch"] for x in sweep] == calls, "every rung tried belongs in the row"


def test_run_at_batch_is_one_measurement_per_instance():
    """Rungs are rejected on FEASIBILITY — OOM, driver refusal, or not finishing inside the wall
    budget — never on throughput, and exactly one rung is ever measured. That is what separates
    this from the one-rung-up probe removed on 2026-09-03, which kept the bigger rung when it
    also ran FASTER: a best-of-two over the rival's batch."""
    from eval.rivals.rival_common import run_at_batch
    calls = []

    def run(b):
        calls.append(b)
        return {"unique_per_s": float(b)}                # every rung runs; the top one wins

    best, sweep = run_at_batch(run, 100_000, log=lambda *_: None)
    assert best["unique_per_s"] == 3_000_000 and calls == [3_000_000], \
        f"the rival was measured more than once: {calls}"
    assert [x["batch"] for x in sweep] == [3_000_000]


def test_the_first_run_of_a_process_is_discarded():
    """The first GPU run of a session is ~3.3x slow (CUDA context, autotuning, clocks), and
    whichever instance goes first eats all of it: `c1355` was published at 0.25x of its warm rate
    on 2026-09-03 for exactly this reason. One discarded run per (tool, process) — per PROCESS,
    because the supervisor resumes in a fresh interpreter after every VM reclaim."""
    import eval.rivals.rival_common as rc
    rc._WARMED.discard("toolX")
    calls = []
    rc.run_at_batch(lambda b: (calls.append(b), {"unique_per_s": 1.0})[1],
                    1_000, log=lambda *_: None, tool="toolX")
    assert calls[0] == 10_000, "no warmup run fired before the measurement"
    n_first = len(calls)
    rc.run_at_batch(lambda b: (calls.append(b), {"unique_per_s": 1.0})[1],
                    1_000, log=lambda *_: None, tool="toolX")
    assert len(calls) == n_first + 1, "the warmup fired twice in one process"


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


def test_a_timeout_steps_down_but_the_probe_is_bounded():
    """A rung the rival cannot FINISH with was not granted to it, so a timeout steps down like an
    OOM — recording it as the rival's result is how a granted-too-large cell reads as a rival
    failure. But the walk must stay bounded: measured on the L4, HTS's three tseitin cells
    stepped 1M -> 1,000 at the full 300 s cap on every rung, ~40 min per cell. Past
    `probe_budget_s` the walk jumps straight to the modelled rung and says so."""
    from eval.rivals.rival_common import run_at_batch
    calls = []

    def run(b):
        calls.append(b)
        return {"error": "timeout 300s"}

    best, sweep = run_at_batch(run, 30_000, log=lambda *_: None, probe_budget_s=0.0)
    assert "error" in best
    assert calls[0] == 3_000_000, "did not start at the top of the ladder"
    assert 30_000 in calls, "never fell back to the modelled rung"
    assert any(x.get("probe_truncated") for x in sweep), "truncation not recorded in the row"
    assert len(calls) < 8, f"walked the whole ladder on timeouts: {calls}"


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


# ---------------------------------------------------------------- Table 1 on the rival rows (the converged protocol)

def rrow(name, *, models=1000, sps=1e6, cms=1000, cms_sps=1e4, p_acc=0.01, dec=1e6, impl=1e8,
         cms_impl=1e6, cms_dec=1e5, arm=True, **extra):
    """A rival-stage row of the converged protocol: the rival, the GPAP arm and the CMSGen arm
    on ONE CNF. A tool that emitted nothing has a zero rate (as it does in a real row)."""
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
    """The converged protocol: Table 1 (GPAP vs CMSGen) comes from the SAME files and cells as
    Table 2, screened first (vacuous / no-search / tiny), then the symmetric zero ledger with
    absent-is-not-zero and crashed-is-not-zero, and the rival paper's own subset beside the
    whole set. Every instance is counted exactly once and a reader can add the columns up."""
    from eval.artifact.paper_tables import (RIVAL_CMS_SETS, RIVALS, rival_cms_family,
                                            rival_prop_rates)

    assert [s for s, _, _ in RIVAL_CMS_SETS] == [s for s, _, _, _ in RIVALS], \
        "Tables 1 and 2 must name the same stages"
    rows = [rrow("s386", sps=1e6, cms_sps=1e4),               # finite 100x, in their paper
            rrow("b12", sps=2e5, cms_sps=1e4),                #  finite 20x, in their paper
            rrow("b04", sps=5e4, cms_sps=1e4),                #  finite  5x, NOT in their paper
            rrow("b11", cms=0),                               # CMSGen=0: a win, no ratio
            rrow("b03", models=0),                            # GPAP=0: a loss, no ratio
            rrow("b13", models=0, cms=0),                     # both=0
            rrow("b19", p_acc=1.0),                           # vacuous pin: screened FIRST
            rrow("b10", dec=0, impl=0),                       # no-search: screened
            rrow("b01", arm=False),                           # absent is not zero
            rrow("b09", cms=0, cms_error="OSError: boom"),    # a crash is not a zero
            {"name": "b21", "variant": "raw", "rcs": {}, "gpap": {"error": "oversize"}}]
    (tmp_path / "rivals_recurrent_rare.json").write_text(json.dumps(rows))
    v = rival_cms_family(str(tmp_path), "recurrent_rare", "rcs")
    assert v["inst"] == 10 and v["errors"] == 1 and v["cells"] == 10
    assert v["n_gm"] == 3 and v["gm"] == pytest.approx((100 * 20 * 5) ** (1 / 3))
    assert v["gm_paper"] == pytest.approx((100 * 20) ** 0.5) and v["n_paper"] == 2
    assert v["inf"] == 1 and v["gpap0"] == 1 and v["both0"] == 1
    assert v["vac"] == 1 and v["free"] == 1 and v["tiny"] == 0
    assert v["noarm"] == 1 and v["cms_err"] == 1
    assert v["inst"] == (v["n_gm"] + v["inf"] + v["gpap0"] + v["both0"] + v["free"] + v["tiny"]
                         + v["vac"] + v["noarm"] + v["cms_err"])
    assert v["wins"] == 4 and v["n"] == 5
    assert v["gpap0_names"] == ["b03"] and v["both0_names"] == ["b13"]
    assert v["cms_err_names"] == ["b09"] and v["noarm_names"] == ["b01"]
    # e2e charges both sides for getting ready: GPAP's n / wall_total vs CMSGen's subprocess wall
    assert v["gm_e2e"] == pytest.approx(v["gm"] / 1.8) and v["pre_frac"] == pytest.approx(0.5)
    assert v["depth"] == "T=25" and v["encoder"] == "abc:fast" and v["patched"] is True
    assert v["cores"] == ["libgpap 0.3.0+abc"] and v["unsound"] == 0 and v["cpu"] == "x"
    # Table 3 over the SAME cells: the identity closes because every column shares one row set
    p = rival_prop_rates(str(tmp_path), "recurrent_rare", "rcs")
    assert p["n"] == 3 and p["encoder"] == "abc:fast"
    assert p["ratio"] / p["work"] == pytest.approx(v["gm"])


def test_a_rivals_only_sweep_has_no_table_1_and_a_refusal_is_not_a_zero(tmp_path):
    """No CMSGen arm anywhere -> no Table 1 row (not a column of CMSGen zeros); a GPAP-side
    refusal (oversize / envelope / encode failure) is an error, in no column."""
    from eval.artifact.paper_tables import rival_cms_family

    (tmp_path / "rivals_demotic_native.json").write_text(json.dumps(
        {"rows": [rrow("c432", arm=False), rrow("c499", arm=False)]}))
    assert rival_cms_family(str(tmp_path), "demotic_native", "demotic") is None
    (tmp_path / "rivals_demotic_native.json").write_text(json.dumps(
        {"rows": [rrow("c432"), {"name": "c7552", "variant": "raw", "demotic": {},
                                 "gpap": {"error": "encode failed: no abc"}}]}))
    v = rival_cms_family(str(tmp_path), "demotic_native", "demotic")
    assert v["inst"] == 1 and v["errors"] == 1 and v["gpap0"] == 0 and v["both0"] == 0


def test_verify_artifact_gates_the_cmsgen_arm_on_rival_rows(tmp_path):
    """A campaign with no `gpap bench` directory at all is legitimate now (Table 1 comes off the
    rival rows), and the SAME gates apply there: a clean stage passes; a floored zero, a
    mixed-arm stage and a crashed subprocess each FAIL."""
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
    """A rival stage where the only measured cell is a no-search cell (b13 at a shallow T
    collapses to a 2-clause CNF under ABC) has no geomean at all; the caveat that quotes 'what
    the geomean would read with them in' formatted None and crashed the whole generator on the
    first campaign directory of the converged protocol. The generator must still write both
    files, with the stage's Table 1 row present and its no-search count stated."""
    import subprocess

    riv = tmp_path / "riv"
    riv.mkdir()
    (riv / "rivals_recurrent_rare.json").write_text(json.dumps([rrow("b13", dec=0, impl=0)]))
    (riv / "rivals_demotic_native.json").write_text(json.dumps({"rows": [rrow("c432")]}))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, os.path.join(root, "eval", "artifact", "paper_tables.py"),
                        "--rivals", str(riv), "--out-md", str(tmp_path / "R.md"),
                        "--out-tex", str(tmp_path / "T.tex")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    md = (tmp_path / "R.md").read_text()
    table1 = md.split("## Table 1")[1].split("## Table 2")[0]
    t1 = [ln for ln in table1.splitlines()
          if ln.startswith("| ITC'99") or ln.startswith("| ISCAS-85")]
    assert len(t1) == 2, table1
    rare = next(ln for ln in t1 if "| rare |" in ln)
    assert "| — |" in rare and "CMSGen=0" not in rare, rare   # no geomean, and not blamed on CMSGen
    assert "no **search at all**" in md or "no search at all" in md
    assert "ISCAS-85" in (tmp_path / "T.tex").read_text()


def test_table_2_prints_the_end_to_end_ratio_beside_the_steady_state_one(tmp_path):
    """The rival-facing ratio is GPAP's STEADY-STATE unique/s over the rival's metric — it
    excludes construction, mining and warm-up, as the rival's model_runtime excludes its model
    build. Table 2 prints the e2e ratio beside it — GPAP's whole wall over the RIVAL'S whole
    wall (2026-09-05; it used to be over the rival's unchanged timer; since 2026-09-06 with the
    fixed process start-up excluded on both sides) — computed over the same `ok` cells with the
    same distinct fraction; a paragraph once claimed the first column 'includes construction'.
    Here every seed's e2e rate is half its steady state and the rival's non-timed wall is zero,
    so its whole-wall rate equals its timed rate and e2e must read exactly gm / 2."""
    from eval.artifact.paper_tables import rival_family

    rows = [rrow("s386", sps=1e6), rrow("b12", sps=2e5), rrow("b19", p_acc=1.0)]
    for r in rows:
        r["rcs"].update({"unique_per_s": 1e3, "unique": 1e3, "runtime_s": 1.0, "wall_s": 1.0})
    (tmp_path / "rivals_recurrent_rare.json").write_text(json.dumps(rows))
    v = rival_family(str(tmp_path), "recurrent_rare", "rcs")
    assert v["n"] == 2 and v["gm"] == pytest.approx((1e3 * 2e2) ** 0.5)
    assert v["gm_e2e"] == pytest.approx(v["gm"] / 2)
    assert v["vacuous"] == 1


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
    """`c1355` under the rare pin: 41 input bits (2^41 points) yet 0.56 distinct in 1,000 kept
    rows — ~700 solutions, a space both tools ENUMERATE. The old screen read the hypercube and
    let it through; the estimate is now the minimum of the hypercube, p_accept x hypercube (or
    its upper bound when no hit was observed) and the count the repeat rate implies."""
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


# ---------------------------------------------- the 2026-09-04 referee pass, screen by screen

def test_the_whole_delivered_set_is_deduplicated_not_a_window():
    """Both rivals dedup EXACTLY, over their whole output, off their own clock: Recurrent calls
    `torch.unique(dim=0)` per depth, DEMOTIC once its timer stops. We estimated a distinct
    fraction from 1,000 rows and applied it to runs delivering up to 1,048,576 models — and the
    "bias bounded at 7e-4" claim that rested on it bounded the collision rate INSIDE the window,
    not the run's distinct fraction. `exact_distinct` is the same quantity theirs is."""
    import numpy as np
    from eval.rivals.rival_common import exact_distinct

    rows = np.array([[1, 0, 1], [1, 0, 1], [0, 1, 1], [0, 0, 0]], dtype=np.uint8)
    assert exact_distinct(rows) == 3
    assert exact_distinct(np.zeros((0, 5), dtype=np.uint8)) == 0
    wide = np.zeros((1000, 250), dtype=np.uint8)          # packbits path, one distinct row
    assert exact_distinct(wide) == 1


def test_a_published_row_is_one_run_not_a_field_wise_median():
    """Medianing every column independently publishes a row no execution produced: on the
    2026-09-03 campaign 138 field/cell combinations came from a different seed than the row's
    own rate, and Table 3 then divided one seed's implications by another seed's models."""
    from eval.rivals.rival_common import median_run

    runs = [{"seed": 5, "sps": 10.0, "n_models": 111, "impl": 1.0},
            {"seed": 6, "sps": 30.0, "n_models": 222, "impl": 2.0},
            {"seed": 7, "sps": 20.0, "n_models": 333, "impl": 3.0}]
    pick, order = median_run(runs, "sps")
    assert pick["seed"] == 7 and pick["sps"] == 20.0
    assert (pick["n_models"], pick["impl"]) == (333, 3.0), "fields came from another run"
    assert (order[0]["sps"], order[-1]["sps"]) == (10.0, 30.0), "spread must stay reportable"


def test_a_proved_unsat_cell_is_screened_before_any_zero():
    """`b01` under Recurrent's own target is unreachable at exactly T=25 (their sampler runs to
    50 and pools every depth), so the 2026-09-03 campaign booked a GPAP=0 LOSS against a solution
    that does not exist. Proved-UNSAT is a fact about the instance and gets its own column;
    `None` (unasked, or unresolved inside the budget) is never read as satisfiable."""
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
    """We hand GPAP an ABC-optimized CNF; the rival must be able to have the ABC-optimized
    CIRCUIT. Where a stage holds both variants the table takes the rival's better one — the same
    tie-break `num_steps` uses — instead of pooling both, which would count every instance twice.
    """
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
    """`c432` under the rare pin leaves 4 effective variables and delivers a model per 1.08
    IMPLICATIONS, against a next-lowest surviving cell of 44.1 — and being a hair off an exact
    zero, it passed the knife-edge screen and became the ENTIRE DEMOTIC-rare geomean at 89,786x.
    A sampler doing at most one implication per solution is reading bits out of a hypercube."""
    from eval.artifact.paper_tables import zero_search

    near_free = {"n_models": 933_900, "sps": 1.079e8, "decisions_per_s": 3.16e8,
                 "implications_per_s": 1.176e8}                    # 1.09 impl / model, as measured
    assert zero_search(near_free), "the c432-shaped cell is still in the geomean"
    # the threshold sits in a two-order gap: the next unscreened cell on that campaign is b13 at
    # 44.1 implications per model, so any value in (1.09, 44.1) gives the same partition
    real = {"n_models": 50_000, "sps": 1e5, "decisions_per_s": 1e6, "implications_per_s": 4.41e6}
    assert not zero_search(real), "a cell doing 44 implications per model is not free"
    exact_zero = {"n_models": 10, "sps": 1e5, "decisions_per_s": 0, "implications_per_s": 0}
    assert zero_search(exact_zero), "the original exact-zero rule must still fire"


def test_the_emitted_latex_ledger_sums_to_the_cell_count(tmp_path):
    """THE check a referee performs in ten seconds. Both LaTeX captions assert that their columns
    account for every instance, and on the 2026-09-03 campaign neither did: the emitter dropped
    `tiny` and `vacuous` from Table 1 (11 vs 10, 29 vs 24) and `n`, `tiny` and `vacuous` from
    Table 2, so `89,786.0x` printed with nothing saying it was one cell. The Markdown emitter was
    correct throughout — only the artifact that goes into the paper was wrong, which is exactly
    why this asserts on the emitted file rather than on the dicts behind it."""
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
    tex = tmp_path / "t.tex"
    import os

    from tests.conftest import REPO_ROOT
    subprocess.run([sys.executable, os.path.join(REPO_ROOT, "eval", "artifact", "paper_tables.py"),
                    "--rivals", str(tmp_path), "--out-tex", str(tex)],
                   check=True, capture_output=True, cwd=REPO_ROOT)

    checked = 0
    for line in tex.read_text().splitlines():
        if not line.startswith("    ") or "&" not in line or "multicolumn" in line:
            continue
        col = [c.strip() for c in line.strip().rstrip("\\\\").split("&")]
        if col[0].startswith(("ISCAS", "ITC")) and len(col) == 20:       # Table 1 (+C<30, paper set, e2e, pre%, refused)
            total, parts = int(col[4]), [int(col[5])] + [int(c) for c in col[11:19]
                                                         if re.fullmatch(r"\d+", c)]
        elif col[0] in ("DEMOTIC", "Recurrent-CircuitSAT") and len(col) == 20:   # Table 2 (+paper set, ties)
            total, parts = int(col[2]), [int(col[3])] + [int(c) for c in col[9:]
                                                         if re.fullmatch(r"\d+", c)]
        else:
            continue
        assert sum(parts) == total, f"ledger does not close on {col[0]}: {parts} vs {total}"
        checked += 1
    assert checked >= 2, "no ledger rows were emitted to check"


def test_the_generated_methodology_text_describes_the_harness_that_exists():
    """The tables carry their own methods paragraph, and it is the paper's. It described the
    SUPERSEDED harness for one release — "stepping down a rung only if the driver refuses the
    size ... a timeout recorded as the result" and "a FIXED 1,000 ... rows for the distinct-
    fraction estimate" — both true before 2026-09-04 and false after, so regenerating the tables
    would have shipped a methods section that misdescribed the run that produced them.

    Prose cannot be gated by re-derivation, so it is gated by absence: the phrases that describe
    the old rules must not reappear, and the ones naming the current rules must be present."""
    import inspect

    from eval.artifact import paper_tables

    src = inspect.getsource(paper_tables)
    gone = ["a timeout recorded as the result",
            "stepping down a rung only if the driver refuses",
            "FIXED 1,000 GPU-projected support-bit rows"]
    for phrase in gone:
        assert phrase not in src, f"the methods paragraph still describes the old harness: {phrase!r}"
    # phrases short enough not to straddle a string-literal break in the source
    for phrase in ["LARGEST rung of that ladder it can run to completion",
                   "deduplicates", "discarded before anything is", "`lr`",
                   "SHIPPED script configuration", "1,200 s budget per instance",
                   "not a search for the rival's fastest rung"]:
        assert phrase in src, f"the methods paragraph does not state the current rule: {phrase!r}"


def test_the_rival_calibration_evidence_is_printed_not_just_pointed_at(tmp_path):
    """The methods prose says the papers' values are "measured beside" the shipped setting. For
    one release that measurement lived only in `rivals_steps_calibration.json`, a file the tables
    named and never read — so RESULTS.md could not say whether the setting the campaign used was
    the better one FOR THE RIVAL. The generator now prints every arm, its ratio to the pinned arm
    and the verdict, from the same run that prints the tables."""
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
    # ...and build() actually emits it under --rivals (source-inspected: build has no return)
    assert "rival_calibration(args.rivals)" in inspect.getsource(paper_tables.build)


def test_the_e2e_column_belongs_to_the_same_run_as_the_rate_it_is_paired_with():
    """A row is ONE execution, and every column of it must come from that execution. `_gpap_e2e`
    took its own median across seeds while `sps`/`unique_per_s_est` came from the picked run, so
    Table 1's e2e ratio and Table 2's e2e column mixed two different seed selections inside one
    quotient — the field-wise-median defect, reappearing in a derived quantity."""
    from eval.artifact.paper_tables import _gpap_e2e

    g = {"sps": 500.0, "sps_e2e": 400.0,
         "seeds": [{"sps": 100.0, "sps_e2e": 90.0},
                   {"sps": 500.0, "sps_e2e": 400.0},
                   {"sps": 900.0, "sps_e2e": 111.0}]}
    assert _gpap_e2e(g) == 400.0, "the e2e column came from a different seed than the row"
    legacy = {"seeds": [{"sps_e2e": 90.0}, {"sps_e2e": 400.0}, {"sps_e2e": 111.0}]}
    assert _gpap_e2e(legacy) == 111.0, "a pre-fix row must still read as it always did"


def test_a_rival_that_found_a_handful_of_solutions_is_not_a_denominator():
    """A ratio is only as good as its denominator, and this one bites on the RIVAL side an order
    of magnitude harder than on CMSGen's. Measured 2026-09-04, rare pin: DEMOTIC found ONE
    solution on `c880` (ratio 7,209,624x) and FOUR on `c3540` (86,202x); those two cells alone
    would have carried the row at a geomean of 788,345x.

    Handing the rival the ABC-simplified circuit is what created them — it let it scrape hits
    where it previously found none, converting honest `rival=0` cells (excluded from the geomean
    and counted as outright WINS) into finite ratios in the millions. Below the floor the cell is
    a capability statement and gets its own column, which is conservative by construction: every
    cell removed is a huge win of ours."""
    from eval.artifact.paper_tables import RIVAL_MIN_SOLUTIONS, rival_cells

    def cell(n_unique, ups):
        r = rrow("c880", sps=8.45e6)
        r["rcs"] = {"unique": n_unique, "unique_per_s": ups, "num_inputs": 40}
        r["gpap"]["unique_per_s_est"] = 8.45e6
        return r

    (_, _, bucket, ours, rate), = rival_cells([cell(1, 1.2)], "rcs")
    assert bucket == "rival_thin", "a one-solution denominator was published as a 7-million-fold win"
    assert (ours, rate) == (0.0, 0.0), "a screened cell must carry no rates into the geomean"

    (_, _, ok_bucket, o2, r2), = rival_cells([cell(RIVAL_MIN_SOLUTIONS * 10, 500.0)], "rcs")
    assert ok_bucket == "ok" and o2 / r2 > 0, "a real measurement must still be scored"
    # a rival that found NOTHING is a different thing — an outright win, not a thin denominator
    (_, _, zero_bucket, _, _), = rival_cells([cell(0, 0.0)], "rcs")
    assert zero_bucket == "rival0", "a rival zero must stay an outright win, not be screened away"
