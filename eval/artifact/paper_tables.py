#!/usr/bin/env python3
"""The paper's tables, regenerated from a campaign's per-cell rows.

    paper_tables.py --rivals DATA [--barbarik FILE ...] --out-md RESULTS.md --out-json NUMBERS.json
                    [--out-notes NOTES.md]

`RESULTS.md` is Tables 1-3, the uniformity counts and the numbers the paper's text quotes, laid
out as the paper prints them so a reader can match it row by row; under each table, every cell
not counted, by name and reason. `NUMBERS.json` is every published number, machine-readable: the
contract `check_paper_numbers.py` holds the PDF to. `--out-notes` writes the per-campaign analysis
behind the paper's wording (caveats, what each geomean stands on, the rivals' configuration
evidence), for the authors; the artifact does not produce it. What the columns mean and why a
cell is screened is standing policy and lives in docs/papers/tacas-27/README.md.

THE table rules, applied everywhere so they cannot diverge between tables:
  * Vacuous pins (p_accept 1.0: pinning is a tautology) are screened before any bucket.
  * Zero-search cells: no decisions and no implications means the ratio measures the instance.
  * A solution space smaller than the sample request is enumerated, not sampled.
  * Either side emitting nothing has no finite ratio: its own column, CMSGen=0 (a win) or
    GPAP=0 (a loss), so the columns sum to the instance count.
  * Rival rows that errored are counted, never folded in as zeros.
  * Every other reason a cell carries no ratio is named under its table.
  * A tool that scored anything is scored: no thin floor, thin cells disclosed by name.

The cell rules are defined here once and imported by every other table, figure and gate,
including the `gpap bench` family readers (`CMS_FAMILIES`, `cms_family`) that the soft-price
tables and the verifier use.
"""
import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gpap.verdict import sound_verdict           # noqa: E402 — a stdlib-only leaf
from eval.rivals.rival_common import grant_walk_rungs, space_from_distinct  # noqa: E402 — after sys.path
# THE uniformity rules, imported rather than restated: eligibility buckets on the control's own
# verdict, the refusal to pool two parameter settings, and what each setting costs. Stdlib-only,
# like this module, so it cannot pull a GPU dependency in behind it.
from eval.uniformity.barbarik_table import (SETTINGS as BARBARIK_SETTINGS,   # noqa: E402
                                            arm_summary, buckets,
                                            check_one_setting, load as barbarik_load,
                                            order_budget as barbarik_order_budget)

#: Pre-`sim_baseline` fallback only, measured at T=25 — see `vacuous_pin`, which reads the row
#: first. Every consumer goes through that function; a `name in VACUOUS_PINS` test does not.
VACUOUS_PINS = {"b02", "b05", "b07", "s526"}


# DEMOTIC's own Table II asks for 1000 distinct solutions "except c17": 2**support < 1000.
TINY_SPACE_FLOOR = 1000


def tiny_space(r, rival_key=None):
    """True when the projected solution space is smaller than the number of samples asked for.

    Both tools then simply ENUMERATE it and the ratio is a race between two startup costs.
    Symmetric, and it does not flatter us: the screen costs us in Table 2 and gains in Table 1.
    Reads `support`, falling back to the rival's own `num_inputs`; False when neither is there."""
    n = r.get("support")
    if n is None and rival_key:
        n = ((r.get(rival_key) or {}).get("num_inputs"))
    try:
        n = int(n)
    except (TypeError, ValueError):
        return False
    return solution_space_estimate(r, n) < TINY_SPACE_FLOOR


def solution_space_estimate(r, n_support):
    """The row's best estimate of |S|, the number of SOLUTIONS: never more than the hypercube
    2^|support|, and smaller wherever the row carries evidence — the acceptance probability
    (`p_accept`, or its 95% upper bound `p_accept_lt`) times the hypercube, and the repeat rate
    among the kept output rows (`rival_common.space_from_distinct`). The minimum of what is
    known; a row with no evidence gets the hypercube."""
    space = float(2 ** min(n_support, 64))          # min() so a huge support cannot blow up 2**n
    sb = r.get("sim_baseline") or {}
    p = sb.get("p_accept") if sb.get("p_accept") is not None else r.get("p_accept")
    if p:
        space = min(space, p * 2 ** min(n_support, 64))
    elif p == 0:
        plt = sb.get("p_accept_lt") if sb.get("p_accept_lt") is not None else r.get("p_accept_lt")
        if plt:
            space = min(space, plt * 2 ** min(n_support, 64))
    # the repeat rate: rival rows keep it per seed under `gpap`, bench rows on the row itself
    pairs = [(s.get("unique_rows"), s.get("unique_distinct"))
             for s in ((r.get("gpap") or {}).get("seeds") or [])]
    pairs.append((r.get("n_collected"), r.get("n_distinct")))
    for rows, distinct in pairs:
        if rows and distinct:
            est = space_from_distinct(rows, distinct)
            if est is not None:
                space = min(space, est)
    return space


def vacuous_pin(r, name=None):
    """True when the pinned target is a tautology — every input sequence satisfies it.

    Then "sampling" is a free hypercube over the input bits, every sampler scores ~100%, and the
    ratio measures the instance. Detected per row from `sim_baseline.p_accept`.

    THE MEASUREMENT DECIDES; `VACUOUS_PINS` is only the fallback for a row that carries no
    baseline. Vacuity is a property of (instance, depth, pin), not of a name: at T=50 `b02`
    measures 0.37-0.63 and `b07` 0.50, so a name list consulted first screens a cell the row
    itself calls samplable — and itemises it in the table as `p_accept 1.0`."""
    p = (r.get("sim_baseline") or {}).get("p_accept")
    if p is None:
        p = r.get("p_accept")                  # rival rows carry it at the top level
    if p is not None:
        return p >= 1.0
    return bool(name and name in VACUOUS_PINS)


def zero_search(r):
    """True when the row emitted samples having done NO search: no decisions, no implications.

    After propagation the support is a free hypercube, so every trajectory succeeds — a correct
    result that measures the degeneracy, and one the best-encoder rule actively selects."""
    NEARLY_FREE_IMPL_PER_MODEL = 2.0
    n = r.get("n_models") or 0
    if n <= 0:
        return False
    dec, impl = (r.get("decisions_per_s") or 0), (r.get("implications_per_s") or 0)
    sps = (r.get("sps") or r.get("samples_per_s") or 0)      # rival rows / bench rows
    if not dec and not impl:
        return True
    # ...and the continuous version, because an exact zero is a knife-edge: a sampler doing ~one
    # implication per model delivered is reading bits out of a hypercube. The threshold sits in a
    # gap in the measured values, not fitted to them (RIVAL_FAIRNESS_FINDINGS.md §17d).
    return bool(sps and impl / sps < NEARLY_FREE_IMPL_PER_MODEL)


def unsat_at_depth(r):
    """True when a solver PROVED this cell has no solution at the depth it was measured at.

    Neither sampler can produce one, so a zero on either side says nothing about either tool.
    `gpap.unsat_at_T` is True only on a proof; `None` is never read as satisfiable."""
    return ((r.get("gpap") or {}).get("unsat_at_T") is True) if isinstance(r, dict) else False


def sound_failed(r):
    """True when a row's on-GPU soundness re-check RAN and found an unsound model.

    Three-valued: `sound_check == "off"` means the opt-in check was not run, which is neither a
    pass nor a failure. ONE definition, imported by `verify_artifact.py` so the gate and the
    table cannot disagree about which cells were checked; the tri-state is `sound_verdict`."""
    if r.get("cpu_sound") is False:
        return True
    return sound_verdict(r) == "UNSOUND"


def sound_unchecked(r):
    """True when the row says nothing about soundness — the check was not run."""
    return sound_verdict(r) == "off" and r.get("cpu_sound") is None


def cms_arm_ran(r):
    """True when the CMSGen arm actually RAN on this row.

    Absent is not zero: a GPAP-only sweep carries no `cms_*` keys, and reading the missing count
    as "CMSGen scored 0" mints a win over a tool that never ran. Per ROW, because a resumed
    campaign can append armless rows to an armed file. Imported by `verify_artifact.py`."""
    return "cms_n_samples" in r or r.get("cms_sps") is not None


def unique_unmeasured(g):
    """True when GPAP's unique/s figure could not be MEASURED on a rival cell.

    Not "GPAP produced nothing" — bucketing it as one publishes a win as a loss. ONE definition,
    applied by `rival_cells`, which the tables and `eval/eval_plots.py` both import."""
    if g.get("unique_per_s_est"):
        return False
    n = g.get("n_models")
    if n is None:                       # a row too old to say how many models it delivered
        return g.get("unique_est_status") in ("no-coverage-sample", "no-sample-rows")
    # "no-sample-rows" with ZERO models is a run that delivered nothing, not an unmeasured one.
    return n > 0


CMS_FAMILIES = [
    ("seq-c25-rare", "ITC'99 + ISCAS-89 sequential", "T=25", "rare"),
    ("seq-c50-rare", "ITC'99 + ISCAS-89 sequential", "T=50", "rare"),
    ("seq-c25-random", "ITC'99 + ISCAS-89 sequential", "T=25", "random"),
    ("seq-c50-random", "ITC'99 + ISCAS-89 sequential", "T=50", "random"),
    ("comb-rare", "ISCAS-85 combinational", "T=1", "rare"),
    ("comb-random", "ISCAS-85 combinational", "T=1", "random"),
    ("hwmcc-c25-rare", "HWMCC'17 property", "T=25", "rare"),
    ("hwmcc-c50-rare", "HWMCC'17 property", "T=50", "rare"),
    ("hwmcc-c25-random", "HWMCC'17 property", "T=25", "random"),
    ("hwmcc-c50-random", "HWMCC'17 property", "T=50", "random"),
    ("epfl-rare", "EPFL combinational", "T=1", "rare"),
    ("epfl-random", "EPFL combinational", "T=1", "random"),
    ("sk", "Sketch / sk_ projected CNF", "--", "--"),
]

# One row per (rival, protocol). DEMOTIC is measured on its own paper's 11 ISCAS-85;
# Recurrent-CircuitSAT on every b*/s* sequential circuit the campaign can process — each under
# two tasks, the pins the rival's table names and our rare pin on the SAME circuits, so the pair
# isolates the task. The FULL set is the headline and the only population printed; the per-paper
# subsets are what the harnesses expand `paper` to (rival_common.py::PAPER_INSTANCES). HTS is
# absent by SCOPE, none of its 60 instances being a circuit; re-enable with RIVAL_HTS=1.


RIVALS = [
    ("demotic_native", "demotic", "DEMOTIC", "their ISCAS-85 pins"),
    ("demotic_comb_rare", "demotic", "DEMOTIC", "their ISCAS-85, our rare pin"),
    ("recurrent_table", "rcs", "Recurrent-CircuitSAT", "their target table, $T{=}25$"),
    ("recurrent_rare", "rcs", "Recurrent-CircuitSAT", "our rare pins, $T{=}25$"),
    ("recurrent_table_t50", "rcs", "Recurrent-CircuitSAT", "their target table, $T{=}50$"),
    ("recurrent_rare_t50", "rcs", "Recurrent-CircuitSAT", "our rare pins, $T{=}50$"),
]
# BOTH DEPTHS on the sequential set: Recurrent's run script says `--num_clock_cycles 50` and its
# paper's limit is eta=50, over which its sampler pools every depth, so measuring only at T=25
# would be a choice against it. GPAP's arm on every row is the abc:fast encoding, the same
# formula the CMSGen tables run, recorded per row as `gpap.encoder`.
# `demotic_miter` is measured and archived but NOT published: its instances are in NEITHER
# paper, it duplicates `demotic_comb_rare`, and its circuit is one we synthesized rather than
# cited. Dropping it removes a win of ours; re-add the tuple to publish those rows.

# ---- THE CONVERGED PROTOCOL: Table 1 is read off the RIVAL rows ----------------------------
# The paper's instance set is the rivals' own, and every cell is run by THREE tools on ONE
# constraint: the rival on its circuit, GPAP on the abc:fast CNF of that circuit and those pins,
# and CMSGen on that same CNF file (`rival_common.cmsgen_arm`). So Table 1 and Table 2 are the
# SAME cells from the SAME files, screened once; the `CMS_FAMILIES` stages are SUPPLEMENTARY.
_CMS_SET_LABEL = {"demotic": "ISCAS-85 (DEMOTIC's set)",
                  "rcs": "ITC'99 + ISCAS-89 (Recurrent's set)"}
RIVAL_CMS_SETS = [(stem, _CMS_SET_LABEL.get(key, key), "rare" if "rare" in stem else "their pins")
                  for stem, key, _, _ in RIVALS]
_RIVAL_KEY = {stem: key for stem, key, _, _ in RIVALS}


def gm(xs):
    xs = [x for x in xs if x and x > 0 and math.isfinite(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def med(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def load(path):
    return json.load(open(path)) if os.path.isfile(path) else None


def rate_gm_pair(pairs):
    """Each side's own absolute rate geomean over ONE cell set, and how many cells that is.

    Why a table may print two rates where the text quotes one speedup: a geomean of quotients
    IS the quotient of the geomeans, so `ours / theirs` here reproduces the ratio column
    exactly. That identity is the whole reason these rates are generated rather than typed off
    a ratio -- the paper prints absolute rates, the text quotes the speedups, and neither can
    drift from the other or from the cells both stand on. Pinned by
    tests/experiments/test_paper_tables.py::test_the_rate_columns_quotient_is_the_ratio_column.

    A pair is used only where BOTH sides are positive, which is the `ok` bucket's own condition,
    so a rate column cannot stand on cells the ratio column does not."""
    use = [(a, b) for a, b in pairs if a and b and a > 0 and b > 0]
    return {"ours": gm([a for a, _ in use]), "theirs": gm([b for _, b in use]), "n": len(use)}


def filled_request(r, gpap_models=None):
    """`(gpap_filled, cms_filled)` for one cell: did each tool deliver the samples it was asked
    for? `None` where the row records no request.

    THE REQUEST IS READ OFF THE ROW (`cms_n_requested`), never assumed. Both tools are handed the
    same one by the protocol, and a campaign that changes it must move this count with it rather
    than leave the text quoting a number from the last one. "Filled" is `>=`: `--n-time` overshoots
    to the end of a chunk, so a run that delivered more than it was asked still filled the request.
    A row with no request is counted apart and never as a shortfall -- "we do not know" and "it
    fell short" are different claims about a tool."""
    req = r.get("cms_n_requested")
    if not req:
        return None
    g = (r.get("gpap") or {}).get("n_models") if gpap_models is None else gpap_models
    return ((g or 0) >= req, (r.get("cms_n_samples") or 0) >= req)


def sig3(x):
    """Three significant figures, scientific at or above 1,000 -- the paper's own rate format,
    so a generated number and a printed one are compared as the same string."""
    if x is None:
        return "—"
    return f"{x:.3g}" if x < 1000 else f"{x:.2e}"


# ------------------------------------------------------------------ GPAP vs CMSGen

def _enc(r):
    """The encoder label of a cell. `enc_key` is the precise one (`abc:fast` and `abc:full` are
    different pipelines); `encoder` is the fallback, None for a family already in CNF."""
    return r.get("enc_key") or r.get("encoder")


def encoders_in(payload):
    """The encoders present in a stage, in a stable order. Circuit families carry two rows per
    instance (`tseitin`, `abc`); a CNF family such as `sk` has none and yields `[None]`."""
    encs = {_enc(r) for r in payload.get("results", [])
            if not r.get("skipped") and not r.get("error")}
    named = [e for e in ("tseitin",) if e in encs]        # Tseitin first: it is the default
    extra = sorted(e for e in encs if e is not None and e != "tseitin")
    return (named + extra) or [None]


def select_cells(payload, encoder):
    """One row per instance, for ONE encoder. Both Table 1 and Table 3 call this, so Table 3
    describes exactly the runs that produced Table 1.

    Do not select the best encoder per instance: it makes the published statistic its own argmax
    key and hands the rival's number a vote on which of our runs is reported
    (docs/findings/RIVAL_FAIRNESS_FINDINGS.md §8/§9)."""
    rows = [r for r in payload.get("results", [])
            if not r.get("skipped") and not r.get("error") and _enc(r) == encoder]
    return rows, {r["name"].split(".")[0]: r for r in rows}


def _pre_frac(r):
    """Share of GPAP's end-to-end wall spent BEFORE the clock the headline rate is measured on:
    parse + engine construction + ELS + warmup + mining + the post-mining rebuild/recheck."""
    ph = r.get("phases") or {}
    wt = r.get("wall_total_s")
    if not wt:
        return None
    return sum(ph.get(k, 0.0) for k in
               ("parse", "construct", "warmup", "mine", "rebuild", "recheck")) / wt


#: A DISCLOSURE threshold, not a screen: a cell whose ratio rests on fewer than this many events
#: on a side is named in the caveats (Poisson relative error at n=30 is ~18%). There is no floor.
THIN_DENOMINATOR = 30


def cms_family(d, stem, encoder):
    payload = load(os.path.join(d, f"{stem}.json")) if d else None
    if not payload:
        return None
    allr = payload.get("results", [])
    rows, cells = select_cells(payload, encoder)
    if not rows:
        return None
    # Absent is not zero: a stage with no CMSGen arm has no Table 1 row at all.
    if not any(cms_arm_ran(r) for r in rows):
        return None

    def scored(r, key):
        return (r.get(key) or 0) > 0

    # ---- the degeneracy screens, applied BEFORE anything is counted -------------------------
    # A cell that measures the INSTANCE takes no part in the geomean, the zero ledger OR the win
    # count. Screen first, then bucket: a zero-search cell can also be a CMSGen=0 cell.
    vac = {k for k, r in cells.items() if vacuous_pin(r, k)}
    free = {k for k, r in cells.items() if k not in vac and zero_search(r)}
    # A space smaller than the sample request is enumerated, not sampled (see `tiny_space`).
    tiny = {k for k, r in cells.items() if k not in vac | free and tiny_space(r)}
    drop = vac | free | tiny
    kept = {k: r for k, r in cells.items() if k not in drop}
    # A MIXED file passes the stage-level guard, so screen per ROW or its armless rows become
    # phantom CMSGen=0 wins.
    noarm = sorted(k for k, r in kept.items() if not cms_arm_ran(r))
    kept = {k: r for k, r in kept.items() if cms_arm_ran(r)}

    # Bucket every REMAINING instance exactly once: a GPAP-zero row has ratio == 0.0 (falsy) and
    # a both-zero row ratio == None, so keying off `ratio` alone drops both.
    finite, gpap_zero, cms_zero, both_zero, weird, cms_err = {}, [], [], [], [], []
    for k, r in kept.items():
        # A crashed CMSGen subprocess (`cms_error`, count 0) is an environment failure, not a
        # measurement: its own bucket, out of every column, named in the caveats.
        if r.get("cms_error"):
            cms_err.append(k)
            continue
        g, c = scored(r, "n_models"), scored(r, "cms_n_samples")
        # A handful of CMSGen samples IS a denominator (`THIN_DENOMINATOR`): scored, and named.
        if g and c:
            if r.get("ratio") and math.isfinite(r["ratio"]):
                finite[k] = r
            else:                       # both emitted yet no usable quotient: never silent
                weird.append(k)
        elif c:
            gpap_zero.append(k)
        elif g:
            cms_zero.append(k)
        else:
            both_zero.append(k)
    n_inf, n_g0, n_b0 = len(cms_zero), len(gpap_zero), len(both_zero)
    gm_vals = [r["ratio"] for r in finite.values()]
    # The cells either tool scored on -- ONE denominator for both delivery counts.
    scored_cells = list(finite.values()) + [cells[k] for k in cms_zero + gpap_zero]
    filled = [filled_request(r, r.get("n_models")) for r in scored_cells]

    def _all(pred):
        """geomean over cells that pass `pred`, used only for the size-of-the-error caveats."""
        return gm([r["ratio"] for k, r in cells.items()
                   if pred(k) and r.get("ratio") and math.isfinite(r["ratio"])
                   and scored(r, "n_models") and scored(r, "cms_n_samples")])

    # Absolute rates, the same four fields the rival-stage reader returns, so ONE renderer
    # prints both kinds of row. A bench row names its own rate `gpap_sps`, not `sps`.
    _ss = rate_gm_pair((r.get("gpap_sps"), r.get("cms_sps")) for r in finite.values())
    _ee = rate_gm_pair((r.get("gpap_sps_e2e"), r.get("cms_sps_e2e")) for r in finite.values())

    return {
        "encoder": encoder, "cells": len(rows), "inst": len(cells),
        "gm": gm(gm_vals),
        "gm_all": _all(lambda k: True),
        "gm_withfree": _all(lambda k: k not in vac),
        # The SAME cells scored end-to-end on BOTH sides: GPAP's wall_total (parse + engine
        # construction + ELS + warmup + MINING + the timed run) over CMSGen's subprocess wall.
        "gm_e2e": gm([r.get("ratio_e2e") for r in finite.values()]),
        "pre_frac": (statistics.median([_pre_frac(r) for r in finite.values()
                                        if _pre_frac(r) is not None] or [0.0])),
        # absolute rates, the same four fields the rival-stage reader returns, so ONE renderer
        # prints both kinds of row (a bench row names its own rate `gpap_sps`)
        "gm_gpap_sps": _ss["ours"], "gm_cms_sps": _ss["theirs"], "n_rate": _ss["n"],
        "gm_gpap_e2e": _ee["ours"], "gm_cms_e2e": _ee["theirs"], "n_rate_e2e": _ee["n"],
        "n_req": next((r.get("cms_n_requested") for r in scored_cells
                       if r.get("cms_n_requested")), None),
        "full_gpap": sum(1 for f in filled if f and f[0]),
        "full_cms": sum(1 for f in filled if f and f[1]),
        "full_unknown": sum(1 for f in filled if f is None),
        "free": len(free), "free_names": sorted(free),
        "vac": len(vac), "vac_names": sorted(vac),
        # How many instances actually ENTER the geomean — where CMSGen scores zero on most of a
        # family it collapses onto the few that remain, and a paper must show that.
        "n_gm": len(gm_vals),
        "weird": len(weird), "has_vac": bool(vac), "tiny": len(tiny),
        # wins/n is counted over exactly the geomean's instances plus the two zero columns.
        "wins": sum(1 for r in finite.values() if r["ratio"] >= 1) + n_inf,
        "n": len(finite) + n_inf + n_g0, "inf": n_inf,
        "gpap0": n_g0, "both0": n_b0,
        "gpap0_names": sorted(gpap_zero), "both0_names": sorted(both_zero),
        "noarm": len(noarm), "noarm_names": noarm,
        "cms_err": len(cms_err), "cms_err_names": sorted(cms_err),
        # scored AND disclosed: geomean cells whose CMSGen count is under the threshold
        "thin_cells": sorted((k, int(r.get("cms_n_samples") or 0)) for k, r in finite.items()
                             if (r.get("cms_n_samples") or 0) < THIN_DENOMINATOR),
        "thin_names": sorted(k for k, r in finite.items()
                             if (r.get("cms_n_samples") or 0) < THIN_DENOMINATOR),
        # THE ONE PRINTED EXCLUSION COLUMN: every cell that left the comparison for a reason
        # other than one sampler scoring zero. Identity: inst = n_gm + CMSGen=0 + GPAP=0 + excl.
        "excl": (len(vac) + len(free) + len(tiny) + n_b0 + len(weird) + len(noarm)
                 + len(cms_err)),
        "excl_names": {b: v for b, v in (("vac", sorted(vac)), ("free", sorted(free)),
                                         ("tiny", sorted(tiny)), ("both0", sorted(both_zero)),
                                         ("weird", sorted(weird)), ("noarm", noarm),
                                         ("cms_err", sorted(cms_err))) if v},
        "skipped": sum(1 for r in allr if r.get("skipped")),
        "errors": sum(1 for r in allr if r.get("error")),
        "unsound": sum(1 for r in rows if sound_failed(r)),
        "gpu": (payload.get("gpu") or {}).get("name"),
        "cpu": (payload.get("cpu") or {}).get("model"),
        "patched": payload.get("cmsgen_patched"),
        # The engine that MEASURED these rows (`gpap_build_id()` per row); more than one value
        # means the stage mixes engines.
        "cores": sorted({(r.get("engine") or {}).get("core") for r in rows} - {None}),
    }


# ------------------------------------------------------------------ propagation throughput

def _prop_stats(quads, encoder):
    """The Table 3 aggregation over (impl_g, impl_c, sps_g, sps_c, dec_g, dec_c) tuples —
    one computation, so the identity closes over one row set."""
    gi, ci, ri, rd, work, gper, cper = [], [], [], [], [], [], []
    for a, b, s, t, dg, dc in quads:
        if not all(x and x > 0 for x in (a, b, s, t)):
            continue
        gi.append(a)
        ci.append(b)
        ri.append(a / b)
        gper.append(a / s)
        cper.append(b / t)
        work.append((a / s) / (b / t))
        if dg and dc:
            rd.append(dg / dc)
    if not gi:
        return None
    return {"n": len(ri), "encoder": encoder, "gpap": gm(gi), "cms": gm(ci), "ratio": gm(ri),
            "dec": gm(rd), "work": gm(work), "gper": gm(gper), "cper": gm(cper)}


# ------------------------------------------------------------------ GPAP vs GPU rivals

def rival_cells(rows, key):
    """Classify every rival cell exactly once — THE per-cell rule Table 2 sums over.

    Yields `(name, variant, bucket, gpap_unique_per_s, rival_unique_per_s)`; only `ok` carries
    two positive rates. `eval/eval_plots.py` imports it rather than restating the screens.
    Buckets, in the order they are tested (the instance screens first, so Tables 1 and 2 cannot
    book one stage two ways):
      `vacuous`     the pin is a tautology: every sampler scores ~100% (an instance property)
      `unsat`       proved to have no solution at this depth: neither tool can produce one
      `tiny`        solution space under the sample request: enumerated, not sampled
      `error`       the rival produced no measurement inside the budget
      `gpap_error`  our encoder or packed envelope refused: a reach limit, never a zero
      `misaligned`  the row failed the constraint-alignment check (`pin_check_ok`)
      `free`        no search: the support is free after propagation (reads our counters)
      `nomeas`      GPAP's unique/s was not measurable: evidence in no direction
      `both0`       neither tool emitted
      `rival0`      the rival emitted nothing: an outright win, no finite ratio
      `gpap0`       we emitted nothing: an outright loss, no finite ratio
      `ok`          both rates positive — the only bucket the geomean sees
    """
    seen = {}
    for r in rows:
        seen[(r["name"], r.get("variant", "-"))] = r        # a retry supersedes its error row
    # ONE CELL PER INSTANCE. Where a stage holds more than one circuit variant (`raw` vs the
    # ABC-simplified netlist) the table takes THE ONE BETTER FOR THE RIVAL — the `num_steps`
    # tie-break, which can only move a ratio against us. Pooling both would double-count.
    best = {}
    for (name, variant), r in seen.items():
        rate = ((r.get(key) or {}).get("unique_per_s") or 0.0) if not (r.get(key) or {}).get("error") else -1.0
        if name not in best or rate > best[name][0]:
            best[name] = (rate, variant, r)
    for name, (_, variant, r) in best.items():
        rv, g = r.get(key) or {}, r.get("gpap") or {}
        if vacuous_pin(r, name.split(".")[0]):
            yield name, variant, "vacuous", 0.0, 0.0
            continue
        if unsat_at_depth(r):
            yield name, variant, "unsat", 0.0, 0.0
            continue
        if tiny_space(r, key):
            yield name, variant, "tiny", 0.0, 0.0
            continue
        if rv.get("error"):
            yield name, variant, "error", 0.0, 0.0
            continue
        if (g or {}).get("error"):
            yield name, variant, "gpap_error", 0.0, 0.0
            continue
        # Constraint alignment is VERIFIED per row (`pin_check_ok`), so a row that FAILED it
        # cannot sit inside the geomean the claim covers.
        if r.get("pin_check_ok") is False:
            yield name, variant, "misaligned", 0.0, 0.0
            continue
        if "decisions_per_s" in g and zero_search(g):
            yield name, variant, "free", 0.0, 0.0
            continue
        rate = rv.get("unique_per_s") or 0.0
        # A rival that scored ANYTHING is scored; thin cells are DISCLOSED, not screened. Not
        # measured is not a zero (see `unique_unmeasured`).
        if unique_unmeasured(g):
            yield name, variant, "nomeas", 0.0, rate
            continue
        ours = g.get("unique_per_s_est") or 0.0
        # Symmetric with the CMSGen table: neither zero has a finite ratio, both get a column.
        if not rate and not ours:
            bucket = "both0"
        elif not rate:
            bucket = "rival0"
        elif not ours:
            bucket = "gpap0"
        else:
            bucket = "ok"
        yield name, variant, bucket, ours, rate


def rival_family(d, stem, key):
    payload = load(os.path.join(d, f"rivals_{stem}.json")) if d else None
    if payload is None:
        return None
    rows = (payload.get("rows") if isinstance(payload, dict) else payload) or []
    cells = list(rival_cells(rows, key))
    n, names = {}, {}
    for name, _, bucket, _, _ in cells:
        n[bucket] = n.get(bucket, 0) + 1
        names.setdefault(bucket, []).append(name)
    ratios = [ours / rate for _, _, bucket, ours, rate in cells if bucket == "ok"]
    zero, gzero = n.get("rival0", 0), n.get("gpap0", 0)
    seen = {}
    for r in rows:
        seen[(r["name"], r.get("variant", "-"))] = r
    # NO SECOND CLOCK IN THIS TABLE (protocol v1.0, docs/findings/RIVAL_FAIRNESS_FINDINGS.md
    # §17). Table 2 is the rivals' OWN metric — unique solutions per second of their
    # `model_runtime`, which excludes their model build — and GPAP's figure excludes
    # construction the same way. On rows that record `egress_s` (from 2026-10-12, §17c) the
    # harness has also deducted Recurrent's in-loop host copy, as GPAP's rate excludes delivery.
    # The end-to-end comparison is Table 1's. Nothing reads `wall_s`.
    okc, unsat_rival, low_vram, ladder_top, small, thin = [], [], [], [], [], []
    for name, variant, bucket, ours, rate in cells:
        r = seen[(name, variant)]
        g, rv = r.get("gpap") or {}, r.get(key) or {}
        if bucket == "unsat" and rv.get("unique_per_s"):
            unsat_rival.append((name, rv["unique_per_s"]))
        gpu = r.get("gpu") or {}
        if gpu.get("total_gb") and gpu.get("free_gb") is not None and "sps" in g \
                and gpu["free_gb"] < 0.5 * gpu["total_gb"]:
            low_vram.append((name, gpu["free_gb"], gpu["total_gb"]))
        if bucket != "ok":
            continue
        x = ours / rate
        band = seed_band(g, ours, rate)
        okc.append((x, name, band, straddles(band)))
        # scored, and disclosed: a ratio that stands on a handful of rival solutions
        if rv.get("unique") is not None and 0 < rv["unique"] < THIN_DENOMINATOR:
            thin.append((name, int(rv["unique"]), x))
        # the walk that reached this batch: a pinned row's own sweep starts AT the grant, so it
        # is read through the recorded walk (`grant_walk_rungs`), or every pinned cell would
        # look like a top rung that completed first try and none like a small grant
        walk = grant_walk_rungs(rv)
        # cells the rival ran at its smallest surviving rung, the one above rejected: the card
        # bound the grant there, so the cell measures the tool at a small batch
        if rv.get("batch") and rv["batch"] <= 10_000 and len(walk) > 1:
            small.append((x, name, rv["batch"]))
        # cells where the ladder's TOP rung completed first try: the ladder bound the grant
        if rv.get("batch") and len(walk) == 1 and rv["batch"] == (walk[0] or 0):
            ladder_top.append((name, rv["batch"]))
    okc.sort(key=lambda t: -t[0])
    ties = sum(1 for _, _, _, st in okc if st)
    small_names = {nm for _, nm, _ in small}
    rest = [x for x, nm, _, _ in okc if nm not in small_names]
    # THE ONE PRINTED EXCLUSION COLUMN of Table 2: every cell without a ratio for a reason other
    # than a tool scoring zero or the rival failing. cells = n + rival=0 + GPAP=0 + failed + excl.
    excl_buckets = ("vacuous", "unsat", "tiny", "free", "misaligned", "nomeas", "both0",
                    "gpap_error")
    # absolute unique/s per side over the `ok` cells; their quotient is `gm` (`rate_gm_pair`)
    uq = rate_gm_pair((ours, rate) for _n, _v, b, ours, rate in cells if b == "ok")
    return {"cells": len(cells), "gm": gm(ratios), "n": len(ratios) + zero + gzero,
            "gm_gpap": uq["ours"], "gm_rival": uq["theirs"], "n_rate": uq["n"],
            # a straddling cell is a TIE: neither a win nor a loss (`ties` says how many)
            "wins": sum(1 for x, _, _, st in okc if x >= 1 and not st) + zero,
            "ties": ties, "errors": n.get("error", 0),
            "error_names": sorted(names.get("error", [])),
            "excl": sum(n.get(b, 0) for b in excl_buckets),
            "excl_names": {b: sorted(names[b]) for b in excl_buckets if names.get(b)},
            "thin": len(thin), "thin_cells": sorted(thin),
            "thin_names": sorted(nm for nm, _u, _x in thin),
            "gm_lo": gm([b[0] if b else x for x, _, b, _ in okc]),
            "gm_hi": gm([b[1] if b else x for x, _, b, _ in okc]),
            "median": statistics.median([x for x, _, _, _ in okc]) if okc else None,
            "top": [(x, nm) for x, nm, _, _ in okc[:6]],
            "gm_wo_top1": gm([x for x, _, _, _ in okc[1:]]) if len(okc) > 1 else None,
            "gm_wo_top4": gm([x for x, _, _, _ in okc[4:]]) if len(okc) > 4 else None,
            "small_grant": sorted(small, reverse=True),
            "gm_small": gm([x for x, _, _ in small]), "gm_rest": gm(rest), "n_rest": len(rest),
            "unsat_rival": unsat_rival, "low_vram": low_vram, "ladder_top": ladder_top,
            "zero": zero, "gpap0": gzero, "both0": n.get("both0", 0),
            "free": n.get("free", 0), "nomeas": n.get("nomeas", 0),
            "misaligned": n.get("misaligned", 0), "vacuous": n.get("vacuous", 0),
            "tiny": n.get("tiny", 0), "unsat": n.get("unsat", 0),
            "gpap_error": n.get("gpap_error", 0)}


# ------------------------------------------------------------------ GPAP vs CMSGen, on the rival rows

def _pa_range(rowsets):
    """The measured acceptance probabilities of a protocol's non-tautological cells, read off
    the rows, as one sentence naming the cells bounded below where no hit was observed."""
    meas, bounded = [], []
    for rows in rowsets:
        for r in rows.values():
            p, lt = r.get("p_accept"), r.get("p_accept_lt")
            if p and p < 1:
                meas.append(p)
            elif not p and lt:
                bounded.append(lt)
    if not meas:
        return "not recorded"
    out = f"{min(meas):.2g}–{max(meas):.2g}"
    if bounded:
        out += (f", {len(bounded)} cell(s) bounded below {min(bounded):.1e}–{max(bounded):.1e} "
                f"where no hit was observed")
    return out


def _rival_rows(d, stem):
    """The (name, variant) -> row map of a rival stage; a retry supersedes its error row."""
    payload = load(os.path.join(d, f"rivals_{stem}.json")) if d else None
    if payload is None:
        return None
    rows = (payload.get("rows") if isinstance(payload, dict) else payload) or []
    seen = {}
    for r in rows:
        seen[(r["name"], r.get("variant", "-"))] = r
    return seen


def _gpap_measured(r):
    """The GPAP arm of a rival row as a Table 1 cell, or None when it REFUSED the instance
    (oversize, out of envelope, encode failure) — counted, in no column, never a zero."""
    g = r.get("gpap") or {}
    return None if (g.get("error") or "sps" not in g) else g


def _gpap_e2e(g):
    """GPAP's end-to-end rate on a rival row — THE PICKED RUN'S, not a median over seeds.

    The row is one execution (`rival_common.median_run` publishes every field of the median-rate
    seed) and this column is divided by that same run's steady-state rate; a separate median
    would re-mix two seed selections inside one ratio. Legacy rows fall back to a median."""
    if g.get("sps_e2e"):
        return g["sps_e2e"]
    return med([s.get("sps_e2e") for s in (g.get("seeds") or []) if s.get("sps_e2e")])


def seed_band(g, ours, rate):
    """(lo, hi): the cell's ratio at GPAP's slowest and fastest of its seeds, or None."""
    lo, hi, sps = g.get("sps_min"), g.get("sps_max"), g.get("sps")
    if not (lo and hi and sps and rate and ours):
        return None
    a, b = ours * lo / sps / rate, ours * hi / sps / rate
    return (min(a, b), max(a, b))


def straddles(band):
    """The win/loss sign lies inside GPAP's own seed band: a tie, not a tally."""
    return band is not None and band[0] < 1.0 <= band[1]


def _cms_ratio(r):
    """GPAP steady-state samples/s over CMSGen's, both emitting; None otherwise."""
    g = r["gpap"]
    if (g.get("n_models") or 0) > 0 and (r.get("cms_n_samples") or 0) > 0 and r.get("cms_sps"):
        x = (g.get("sps") or 0.0) / r["cms_sps"]
        return x if x > 0 and math.isfinite(x) else None
    return None


def rival_cms_cells(rows, key):
    """Classify every rival-stage cell for Table 1 — GPAP vs CMSGen on the row's own CNF —
    exactly once. `rival_cms_family` sums over it, `rival_prop_rates` reads Table 3 off it and
    `eval/eval_plots.py` plots it, so table, identity and figures cannot disagree.

    Yields `(name, variant, bucket, gpap_sps, cms_sps)`; only `ok` carries two positive rates.
    The rival's own outcome is irrelevant — the CMSGen comparison ran whether or not it did.
    Buckets, in the order they are tested:
      `vacuous`   the pin is a tautology (Table 2's instance screens, applied first)
      `unsat`     proved to have no solution at this depth
      `tiny`      solution space under the sample request: enumerated, not sampled
      `refused`   GPAP's encoder or envelope refused: a reach limit, never a zero
      `free`      no search: the support is free after propagation (reads our counters)
      `noarm`     the CMSGen arm never ran — absent is not zero
      `cms_err`   its subprocess crashed: not a zero
      `weird`     both emitted yet no usable quotient
      `cms0`      CMSGen emitted nothing: an outright win, no finite ratio
      `gpap0`     we emitted nothing: an outright loss, no finite ratio
      `both0`     neither emitted
      `ok`        both rates positive — the only bucket the geomean sees
    """
    seen = {}
    for r in rows:
        seen[(r["name"], r.get("variant", "-"))] = r        # a retry supersedes its error row
    for (name, variant), r in seen.items():
        if vacuous_pin(r, name.split(".")[0]):
            yield name, variant, "vacuous", 0.0, 0.0
            continue
        if unsat_at_depth(r):
            yield name, variant, "unsat", 0.0, 0.0
            continue
        if tiny_space(r, key):
            yield name, variant, "tiny", 0.0, 0.0
            continue
        g = _gpap_measured(r)
        if g is None:
            yield name, variant, "refused", 0.0, 0.0
            continue
        if zero_search(g):
            yield name, variant, "free", 0.0, 0.0
            continue
        if not cms_arm_ran(r):
            yield name, variant, "noarm", 0.0, 0.0
            continue
        if r.get("cms_error"):
            yield name, variant, "cms_err", 0.0, 0.0
            continue
        gs, cs = (g.get("n_models") or 0) > 0, (r.get("cms_n_samples") or 0) > 0
        # A handful of CMSGen samples IS a denominator (`THIN_DENOMINATOR`): scored, and named.
        if gs and cs:
            if _cms_ratio(r):
                yield name, variant, "ok", g.get("sps") or 0.0, r.get("cms_sps") or 0.0
            else:                       # both emitted yet no usable quotient: never silent
                yield name, variant, "weird", 0.0, 0.0
        elif cs:
            yield name, variant, "gpap0", 0.0, r.get("cms_sps") or 0.0
        elif gs:
            yield name, variant, "cms0", g.get("sps") or 0.0, 0.0
        else:
            yield name, variant, "both0", 0.0, 0.0


def rival_cms_family(d, stem, key):
    """GPAP vs CMSGen over a RIVAL stage's rows — Table 1 of the converged protocol.

    The same file and the same `(name, variant)` cells Table 2 scores, each carrying the CMSGen
    arm run on the very CNF the GPAP arm sampled, classified once by `rival_cms_cells`. The
    returned dict has Table 1's shape so ONE renderer prints both kinds of row. `inst` is every
    instance, `cells` the ones GPAP measured; the difference is the refused ones, inside `excl`."""
    seen = _rival_rows(d, stem)
    if not seen or not any(cms_arm_ran(r) for r in seen.values()):
        return None                      # absent, or a rivals-only sweep: no CMSGen table
    cells = list(rival_cms_cells(list(seen.values()), key))
    names = {}
    for name, _v, bucket, _g, _c in cells:
        names.setdefault(bucket, []).append(name)
    n = {b: len(v) for b, v in names.items()}
    finite = {(name, v): g / c for name, v, bucket, g, c in cells if bucket == "ok"}
    measured = {k: r for k, r in seen.items() if _gpap_measured(r) is not None}
    n_inf, n_g0, n_b0 = n.get("cms0", 0), n.get("gpap0", 0), n.get("both0", 0)
    # THE PAPER'S RATE COLUMNS: each side's own geomean over the `ok` cells, on both clocks.
    # `gm_gpap_sps / gm_cms_sps` is `gm` and `gm_gpap_e2e / gm_cms_e2e` is `gm_e2e` (see
    # `rate_gm_pair`), so a table may print the rates and the text the speedups.
    ss_gm = rate_gm_pair((g, c) for _n, _v, b, g, c in cells if b == "ok")
    e2e, pre, e2e_pairs = [], [], []
    for k in finite:
        g, r = seen[k]["gpap"], seen[k]
        ge = _gpap_e2e(g)
        if ge and r.get("cms_sps_e2e"):
            e2e.append(ge / r["cms_sps_e2e"])
            e2e_pairs.append((ge, r["cms_sps_e2e"]))
        if ge and g.get("sps"):
            pre.append(max(0.0, 1.0 - ge / g["sps"]))     # (wall_total - timed) / wall_total
    e2e_gm = rate_gm_pair(e2e_pairs)
    # Did each tool FILL the request, on the cells either of them scored on? A rate says how
    # fast a tool sampled; this says whether it finished the job at all, which is the other half
    # of a delivered-samples claim and is in no geomean (a tool that delivered 1 sample and a
    # tool that delivered all 50,000 both have a rate).
    full_g = full_c = full_unk = 0
    for name, v, bucket, _g, _c in cells:
        if bucket not in ("ok", "cms0", "gpap0"):
            continue
        filled = filled_request(seen[(name, v)])
        if filled is None:
            full_unk += 1
        else:
            full_g += filled[0]
            full_c += filled[1]

    def _all(pred):
        """geomean over measured cells that pass `pred` — the size-of-the-error caveats only."""
        return gm([_cms_ratio(r) for k, r in measured.items()
                   if pred(k[0]) and cms_arm_ran(r) and not r.get("cms_error")])

    encs = sorted({r["gpap"].get("encoder") or "?" for r in measured.values()})
    depths = sorted({r.get("cycles") for r in measured.values() if r.get("cycles") is not None})
    armed = [r for r in measured.values() if cms_arm_ran(r)]
    vac = names.get("vacuous", [])
    excl_buckets = ("vacuous", "unsat", "free", "tiny", "both0", "weird", "noarm", "cms_err",
                    "refused")
    return {
        "encoder": "+".join(encs) if encs else None, "cells": len(measured), "inst": len(seen),
        "depth": ("T=" + "/".join(str(t) for t in depths)) if depths else None,
        "gm": gm(list(finite.values())), "gm_all": _all(lambda k: True),
        "gm_withfree": _all(lambda k: k not in vac),
        "gm_e2e": gm(e2e), "pre_frac": statistics.median(pre) if pre else 0.0,
        # absolute rates, each over the cells of the ratio beside it
        "gm_gpap_sps": ss_gm["ours"], "gm_cms_sps": ss_gm["theirs"], "n_rate": ss_gm["n"],
        "gm_gpap_e2e": e2e_gm["ours"], "gm_cms_e2e": e2e_gm["theirs"],
        "n_rate_e2e": e2e_gm["n"],
        "n_req": next((r.get("cms_n_requested") for r in measured.values()
                       if r.get("cms_n_requested")), None),
        "full_gpap": full_g, "full_cms": full_c, "full_unknown": full_unk,
        "free": n.get("free", 0), "free_names": sorted(names.get("free", [])),
        "vac": len(vac), "vac_names": sorted(vac),
        "unsat": n.get("unsat", 0),
        "n_gm": len(finite), "weird": n.get("weird", 0), "has_vac": bool(vac),
        "tiny": n.get("tiny", 0),
        "wins": sum(1 for x in finite.values() if x >= 1) + n_inf,
        "n": len(finite) + n_inf + n_g0, "inf": n_inf, "gpap0": n_g0, "both0": n_b0,
        "gpap0_names": sorted(names.get("gpap0", [])),
        "both0_names": sorted(names.get("both0", [])),
        "noarm": n.get("noarm", 0), "noarm_names": sorted(names.get("noarm", [])),
        "cms_err": n.get("cms_err", 0), "cms_err_names": sorted(names.get("cms_err", [])),
        # scored AND disclosed: geomean cells whose CMSGen count is under the threshold
        "thin_cells": sorted((k[0], int(seen[k].get("cms_n_samples") or 0)) for k in finite
                             if (seen[k].get("cms_n_samples") or 0) < THIN_DENOMINATOR),
        "thin_names": sorted(k[0] for k in finite
                             if (seen[k].get("cms_n_samples") or 0) < THIN_DENOMINATOR),
        # THE ONE PRINTED EXCLUSION COLUMN: inst = n_gm + CMSGen=0 + GPAP=0 + excl.
        "excl": sum(n.get(b, 0) for b in excl_buckets),
        "excl_names": {("vac" if b == "vacuous" else b): sorted(names[b])
                       for b in excl_buckets if names.get(b)},
        "skipped": 0, "errors": n.get("refused", 0),
        "refused_names": sorted(names.get("refused", [])),
        "unsound": sum(1 for r in measured.values()
                       if sound_failed(r["gpap"]) or r["gpap"].get("sim_check") == "BAD"),
        "gpu": next(((r.get("gpu") or {}).get("name") for r in seen.values()
                     if (r.get("gpu") or {}).get("name")), None),
        "cpu": next(((r.get("cpu") or {}).get("model") for r in seen.values()
                     if (r.get("cpu") or {}).get("model")), None),
        "patched": (all(r.get("cms_patched") for r in armed) if armed else None),
        "cores": sorted({(r["gpap"].get("engine") or {}).get("core")
                         for r in measured.values()} - {None}),
    }


def rival_prop_rates(d, stem, key):
    """Table 3 over a rival stage: EXACTLY the cells `rival_cms_family` scores — the `ok` bucket
    of `rival_cms_cells`, so Table 3 cannot stand on a different row set than Table 1."""
    seen = _rival_rows(d, stem)
    if not seen:
        return None
    quads, encs = [], set()
    for name, variant, bucket, _g, _c in rival_cms_cells(list(seen.values()), key):
        r = seen[(name, variant)]
        g = r.get("gpap") or {}
        if _gpap_measured(r) is not None:
            encs.add(g.get("encoder") or "?")
        if bucket != "ok":
            continue
        quads.append((g.get("implications_per_s"), r.get("cms_implications_per_s"),
                      g.get("sps"), r.get("cms_sps"),
                      g.get("decisions_per_s"), r.get("cms_decisions_per_s")))
    return _prop_stats(quads, "+".join(sorted(encs)) if encs else None)


# ------------------------------------------------------------------ barbarik

def barbarik(paths):
    """The standard certificate, per sampler under test.

    The eligibility rule and the per-arm counts are IMPORTED from barbarik_table.py: an instance
    counts only where the negative control (the trusted uniform reference, under test through
    the same hook) was itself Accepted, since where it is Rejected CM_test measures the
    instance."""
    rows = []
    for p in paths or []:
        if os.path.exists(p):
            rows += barbarik_load([p])
    if not rows:
        return None
    eps, eta, delta = check_one_setting(rows)      # raises on a pooled loose+strict file list
    eligible, rejected, errored = buckets(rows)
    return {"summary": arm_summary(rows, eligible), "eligible": sorted(eligible),
            "control_rejected": sorted(rejected), "control_errored": sorted(errored),
            # The budget belongs with the parameters: it is the other axis along which two of
            # these tables are not comparable, and the claim the re-run supports ("both samplers
            # under test had the same per-order budget") is only quotable if it is recorded.
            # None on rows from before the flag.
            "params": {"epsilon": eps, "eta": eta, "delta": delta,
                       "order_seconds": barbarik_order_budget(rows)},
            # provenance: this table is its own campaign, and the header above it is not its host
            "source": sorted(os.path.abspath(p) for p in (paths or []) if os.path.exists(p))}


# ------------------------------------------------------------------ the re-check reader


def recheck_cells(path):
    """One re-check stage's scored cells, keyed `(instance, encoder)`.

    THE single reader of a re-check JSON: `verify_artifact.py` gates the deterministic fields over
    it and `recheck()` below builds the reproducibility row from it. There were three copies of
    this key, one of them in an `eval/recheck_compare.py` whose whole CLI restated what these two
    callers already assert -- and three readers of a row shape is how two of them come to disagree
    about which cells were compared."""
    d = load(path) or {}
    return {(r["name"].split(".")[0], r.get("enc_key") or r.get("encoder")): r
            for r in d.get("results", []) if not r.get("skipped") and not r.get("error")}


# ------------------------------------------------------------------ rendering

# The reasons a cell sits in the one `excluded` column, in print order; the names are itemised
# under the table from the aggregate's own per-bucket lists.
EXCL_REASONS_T1 = (("vac", "vacuous pin (constant outputs, p_accept 1.0)"),
                   ("unsat", "proved UNSAT at this depth"),
                   ("free", "free support after propagation (no search)"),
                   ("tiny", "solution space under 1,000 (enumerated, not sampled)"),
                   ("both0", "neither sampler emitted"),
                   ("refused", "GPAP's encoder or envelope refused"),
                   ("noarm", "CMSGen arm never ran (absent is not zero; re-run)"),
                   ("cms_err", "CMSGen subprocess crashed (re-run)"),
                   ("weird", "both emitted, no usable quotient (fix the row)"))
EXCL_REASONS_T2 = (("vacuous", "vacuous pin (constant outputs, p_accept 1.0)"),
                   ("unsat", "proved UNSAT at this depth"),
                   ("tiny", "solution space under 1,000 (enumerated, not sampled)"),
                   ("free", "free support after propagation (no search)"),
                   ("misaligned", "failed the constraint-alignment check"),
                   ("nomeas", "GPAP's unique/s not measurable"),
                   ("both0", "neither tool emitted"),
                   ("gpap_error", "GPAP's encoder or envelope refused"))


def excl_items(names, reasons):
    """The itemised content of one row's `excluded` column, or '' when it holds nothing."""
    return "; ".join(f"{label} " + ", ".join(f"`{n}`" for n in names[b])
                     for b, label in reasons if names.get(b))


# The rival-configuration EVIDENCE files (eval/rivals/steps_calibration.py): each rival's shipped
# script value, its argparse default and its PAPER value for the knob, plus the paper's PAIRED
# setting. Schema {generated, tools: {tool: {pinned, pinned_source, measured_best, agrees,
# cycles, arms}}} — no `rows` key, which is how verify_artifact tells it from an arm.
CALIBRATION_FILES = (("rivals_steps_calibration.json", "num_steps"),
                     ("rivals_steps_calibration_t50.json", "num_steps"),
                     ("rivals_lr_calibration.json", "lr"))


def rival_calibration(rivals_dir):
    """Markdown lines stating what the rivals' OTHER author-shipped values measured, or [].

    The campaign runs each rival at its shipped script setting; this prints every arm, its ratio
    to the PINNED arm and the tie-break verdict. Evidence, never configuration: a FLIP is
    printed, not acted on."""
    import json as _json
    out = []
    for fn, knob in CALIBRATION_FILES:
        path = os.path.join(rivals_dir or "", fn)
        if not os.path.isfile(path):
            continue
        try:
            d = _json.load(open(path)) or {}
        except Exception as e:                       # noqa: BLE001 — a corrupt file is stated, not hidden
            out.append(f"* `{fn}`: unreadable ({e})")
            continue
        for tool, v in sorted((d.get("tools") or {}).items()):
            arms = v.get("arms") or {}
            pinned = v.get("pinned")
            pin = arms.get(str(pinned)) or {}
            pin_rate = pin.get("unique_per_s") or None
            # `depth` is the depth the probe RAN at, not the CLI `cycles`. No depth, no claim.
            T = v.get("depth")
            out += ["", f"**{tool}** — `{knob}`" + (f" at T={T}" if T else "") +
                    f", pinned {pinned} ({v.get('pinned_source', '?')}); file `{fn}`"
                    + (f", measured {d.get('generated')}" if d.get("generated") else ""), "",
                    f"| {knob} | unique/s | wall s | vs pinned | |", "|---|---:|---:|---:|---|"]
            for k, a in arms.items():
                rate = a.get("unique_per_s")
                ratio = (f"{rate / pin_rate:.2f}x" if (rate is not None and pin_rate) else "—")
                tag = ("**pinned**" if k == str(pinned) else
                       "paper (paired)" if k.startswith("paper(") else "")
                if a.get("error"):
                    tag = (tag + " " if tag else "") + f"error: {str(a['error'])[:60]}"
                rate_s = f"{rate:,.0f}" if rate is not None else "—"
                wall_s = f"{a['wall_s']:.0f}" if a.get("wall_s") is not None else "—"
                out.append(f"| {k} | {rate_s} | {wall_s} | {ratio} | {tag} |")
            ag = v.get("agrees")
            verdict = ("tie-break HOLDS: the pinned value is the best of the tool's own values"
                       if ag is True else
                       f"**FLIP**: {v.get('measured_best')} measured faster than the pinned "
                       f"{pinned} — recorded, not acted on (change the pin deliberately, in one place)"
                       if ag is False else
                       "INCONCLUSIVE: no arm scored")
            out.append("")
            out.append(f"* {verdict}.")
    if out:
        out = ["## Rival configuration — what their other values measured on this card", "",
               "EVIDENCE, not configuration: every campaign row runs the rival at its shipped "
               "script setting (the **pinned** arm). The other arms are the values the tool's own "
               "artifacts also give the knob — its argparse default and its PAPER — each measured "
               "once, on one instance, on the campaign's card, so the text can state what the "
               "paper's setting would have given the tool. A paper arm below 1.00x means the "
               "script setting the campaign used is the BETTER one for the rival."] + out
    return out


# The paper's names for a row. The JSON keeps the generator's labels, which the published-numbers
# file matches on; the tables print the paper's, so a reader can put them side by side.
_PAPER_SET = {"demotic": "ISCAS-85", "rcs": "Sequential"}
# The two Barbarik arms that are GPAP, said in words: the arm names are the harness's.
_ARM_NOTE = {"pipe:circuit": " (GPAP as shipped)", "gpap:b8": " (GPAP, one configuration)"}


def _paper_target(stem):
    return "rare" if "rare" in stem else "original"


def _paper_depth(stem, key, recorded=None):
    """The unrolling depth: the one the rows record, else the one the stage is named for."""
    if recorded and str(recorded).startswith("T="):
        return str(recorded)[2:]
    return "1" if key == "demotic" else ("50" if stem.endswith("_t50") else "25")


def pfmt(x):
    """A rate as the paper prints it: three significant figures, `1.78e3` from a thousand up."""
    if x is None:
        return "—"
    if abs(x) < 999.5:
        return f"{x:.3g}"
    mant, exp = f"{x:.2e}".split("e")
    return f"{mant}e{int(exp)}"


def build(args):
    # THE ROWS OF TABLES 1 AND 3, in print order. One list, so both tables and the notes describe
    # one set of rows: Table 1 is read off the rival rows, Table 2's own cells.
    riv = {s: rival_family(args.rivals, s, k) for s, k, _, _ in RIVALS}
    cms, prop, t1 = {}, {}, []
    for stem, label, tgt in RIVAL_CMS_SETS:
        v = rival_cms_family(args.rivals, stem, _RIVAL_KEY[stem])
        if not v:
            continue
        k = (f"rivals:{stem}", v["encoder"])
        cms[k] = v
        prop[k] = rival_prop_rates(args.rivals, stem, _RIVAL_KEY[stem])
        t1.append((k, label, v.get("depth") or "—", tgt))
    barb = barbarik(args.barbarik)

    prov = next((v for v in cms.values() if v), {}) or {}
    # The commit below is when the TABLES were generated, not what measured the data — that is
    # the per-row build id, unioned over every stage.
    cores = sorted({c for v in cms.values() if v for c in v.get("cores", [])})
    # A FAILED git call does not raise here — `subprocess.run` without check=True just returns a
    # non-zero code and empty stdout — so the except below never fired in the one place it was
    # meant to: the packaged artifact, which carries no .git. Say "not a git checkout" instead; a
    # blank reads as a value.
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                           text=True, timeout=10)
        commit = r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "not a git checkout"
    except Exception:                                   # noqa: BLE001
        commit = "not a git checkout"

    # Totals: `cells`/`unsound` sum over every stage, `inst`/`skipped`/`errors` are taken once per
    # stage.
    one_per_stem = {}
    for (stem, _enc), v in cms.items():
        if v and stem not in one_per_stem:
            one_per_stem[stem] = v
    tot = {kk: sum(v[kk] for v in cms.values() if v) for kk in ("cells", "unsound")}
    tot["refused"] = sum(v.get("errors", 0) for v in cms.values() if v)
    tot.update({kk: sum(v[kk] for v in one_per_stem.values())
                for kk in ("inst", "skipped", "errors")})
    refused = sorted({n for v in cms.values() if v
                      for n in (v.get("excl_names") or {}).get("refused", [])})
    other = tot["errors"] - tot["refused"]
    cell_line = (f"{tot['cells'] + tot['refused']} cells: {tot['cells']} measured, "
                 f"{tot['refused']} GPAP could not encode"
                 + (f" ({', '.join(refused)})" if refused else "")
                 + (f", {other} other error(s)" if other > 0 else "")
                 + (f", {tot['skipped']} skipped" if tot["skipped"] else "")
                 + f", {tot['unsound']} unsound")

    # THE CIRCUIT HANDED TO THE RIVAL, stated from the rows: a fairness procedure must name what
    # actually ran.
    variants = sorted({v for stem, _k, _r, _p in RIVALS
                       for (_n, v) in (_rival_rows(args.rivals, stem) or {})})
    if len(variants) > 1:
        variant_sentence = ("Each rival is handed both the ABC-simplified and the raw circuit and "
                            "is scored on whichever suits it.")
    elif variants == ["abc"]:
        variant_sentence = ("Every rival ran on the circuit file GPAP encodes from, the "
                            "ABC-simplified netlist (`variant: abc` on every row).")
    elif variants == ["raw"]:
        variant_sentence = ("Every rival is handed the raw netlist (`variant: raw` on every row) "
                            "while GPAP encodes an ABC-optimized CNF of it.")
    else:
        variant_sentence = "Circuit variant per row: " + (", ".join(variants) or "none") + "."

    # ============================================================ RESULTS.md: the paper's tables
    M = ["# The paper's tables, regenerated", "",
         f"Regenerated {time.strftime('%Y-%m-%d')} by `eval/artifact/paper_tables.py` from the "
         "campaign's per-cell rows. What each column means, and why a cell is not counted: "
         "`docs/papers/tacas-27/README.md`.", "",
         f"* GPU **{prov.get('gpu')}**, CPU **{prov.get('cpu') or 'not recorded'}**, engine "
         + (", ".join(f"`{c}`" for c in cores) or "`not recorded`")
         + f", CMSGen patched (`--nowrite`): **{prov.get('patched')}**",
         f"* **{cell_line}**",
         f"* {variant_sentence}"]

    depth_of = {}
    M += ["", "## Table 1 — GPAP vs CMSGen", "",
          "Samples per second. The last two columns are the ratios the paper's text quotes.", "",
          "| Set | Target | T | n | CMSGen | GPAP ss | GPAP e2e | Wins | C=0 "
          "| GPAP/CMSGen ss | GPAP/CMSGen e2e |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    went = []
    for k, label, depth, tgt in t1:
        v = cms[k]
        stem = k[0].split(":", 1)[1]
        key = _RIVAL_KEY[stem]
        T = depth_of[stem] = _paper_depth(stem, key, depth)
        row = f"{_PAPER_SET.get(key, label)} | {_paper_target(stem)} | {T}"
        gms = f"{v['gm']:,.1f}x" if v["gm"] else ("— (all C=0)" if v["inf"] else "—")
        if v["gm"] and v["n_gm"] < 3:
            gms += f" ⚠n={v['n_gm']}"
        e2e = f"{v['gm_e2e']:,.1f}x" if v.get("gm_e2e") else "—"
        M.append(f"| {row} | {v['n_gm']} | {pfmt(v.get('gm_cms_sps'))} "
                 f"| {pfmt(v.get('gm_gpap_sps'))} | {pfmt(v.get('gm_gpap_e2e'))} "
                 f"| {v['wins']}/{v['n']} | {v['inf']} | {gms} | {e2e} |")
        items = excl_items(v.get("excl_names") or {}, EXCL_REASONS_T1)
        went.append(f"* {_PAPER_SET.get(key, label)}, {_paper_target(stem)}, T={T}: "
                    f"{v['inst']} = {v['n_gm']} + {v['inf']} + "
                    f"{v['gpap0']} + {v.get('excl', 0)}"
                    + (f"; not counted: {items}" if items else "") + ".")
    M += ["", "Where every instance went, `instances = n + C=0 + G=0 + not counted` (G=0: GPAP "
              "emitted nothing):", ""] + went

    M += ["", "## Table 2 — GPAP vs the GPU samplers", "",
          "Unique samples per second, the rate the GPU samplers' papers report. The last column "
          "is the ratio the paper's text quotes.", "",
          "| Baseline | Target | T | n | Baseline | GPAP | Wins | Ties | C=0 | Fail | GPAP/rival |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    went = []
    for stem, key, rival, _proto in RIVALS:
        v = riv.get(stem)
        T = depth_of.get(stem) or _paper_depth(stem, key)
        row = f"{rival} | {_paper_target(stem)} | {T}"
        if not v:
            M.append(f"| {row} | — | — | — | not run | — | — | — | — |")
            continue
        # `n` is the geomean's cell count, as in Table 1; the win denominator also counts the two
        # zero columns.
        n_gm_ok = v["n"] - v["zero"] - v["gpap0"]
        gs = f"{v['gm']:,.1f}x" if v["gm"] else "—"
        if v["gm"] and n_gm_ok < 3:
            gs += f" ⚠n={n_gm_ok}"
        M.append(f"| {row} | {n_gm_ok} | {pfmt(v.get('gm_rival'))} | {pfmt(v.get('gm_gpap'))} "
                 f"| {v['wins']}/{v['n']} | {v.get('ties', 0)} | {v['zero']} | {v['errors']} "
                 f"| {gs} |")
        parts = []
        if v.get("error_names"):
            parts.append("Fail (no measurement inside the budget; the reason is per row) "
                         + ", ".join(f"`{n}`" for n in v["error_names"]))
        items = excl_items(v.get("excl_names") or {}, EXCL_REASONS_T2)
        if items:
            parts.append("not counted: " + items)
        went.append(f"* {rival}, {_paper_target(stem)}, T={T}: "
                    f"{v['cells']} = {n_gm_ok} + {v['zero']} + "
                    f"{v['gpap0']} + {v['errors']} + {v.get('excl', 0)}"
                    + ("; " + "; ".join(parts) if parts else "") + ".")
    M += ["", "Where every cell went, `cells = n + C=0 + G=0 + Fail + not counted` (C=0: the GPU "
              "sampler found nothing; G=0: GPAP found nothing):", ""] + went

    if any(prop.values()):
        M += ["", "## Table 3 — propagation throughput", "",
              "Implications per second, and per sample, over Table 1's cells. The work ratio is "
              "GPAP's implications per sample over CMSGen's; the last column is the rate ratio "
              "the paper's text quotes.", "",
              "| Set | Target | T | n | CMSGen impl/s | GPAP impl/s | CMSGen impl/sample "
              "| GPAP impl/sample | Work ratio | GPAP/CMSGen impl/s |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for k, label, _depth, _tgt in t1:
            p = prop.get(k)
            if not p:
                continue
            stem = k[0].split(":", 1)[1]
            key = _RIVAL_KEY[stem]
            M.append(f"| {_PAPER_SET.get(key, label)} | {_paper_target(stem)} | {depth_of[stem]} "
                     f"| {p['n']} | {pfmt(p['cms'])} | {pfmt(p['gpap'])} | {p['cper']:,.0f} "
                     f"| {p['gper']:,.0f} | {p['work']:.2f}x | {p['ratio']:,.0f}x |")

    if barb:
        bp = barb["params"]
        c = BARBARIK_SETTINGS.get((bp["epsilon"], bp["eta"], bp["delta"])) or {}
        n_el, rej, err = (len(barb["eligible"]), barb["control_rejected"],
                          barb["control_errored"])
        src = sorted({os.path.relpath(os.path.dirname(s)) for s in barb.get("source") or []})
        M += ["", "## Uniformity — Barbarik", "",
              f"`eps={bp['epsilon']}, eta={bp['eta']}, delta={bp['delta']}`"
              + (f", {c['under_test']:,} samples per instance from the sampler under test"
                 if c.get("under_test") else "")
              + f". {n_el} of {n_el + len(rej) + len(err)} instances are eligible: on "
              f"{len(rej)} the uniform control is itself rejected"
              + (f", and on {len(err)} it errored" if err else "")
              + ", so the test cannot speak about them. A separate run, on its own host and "
                "instances" + (f" (`{'`, `'.join(src)}`)" if src else "") + ".", "",
              "| sampler under test | Accept | Reject | no verdict |", "|---|---:|---:|---:|"]
        for arm, a in barb["summary"].items():
            M.append(f"| `{arm}`{_ARM_NOTE.get(arm, '')} | {a['accept']}/{a['n']} "
                     f"| {a['reject']} | {a['error']} |")


    # ---------------------------------------------------------------- caveats
    # These travel WITH the numbers, and each is derived from the data rather than remembered.
    C, budget_rule = [], False
    for (stem, enc), v in cms.items():
        if not v:
            continue
        stem = f"{stem}/{enc}" if enc else stem
        if v["gm"] and v["n_gm"] < 3:
            why = (f"because CMSGen scored zero on {v['inf']} of {v['n']}" if v["inf"]
                   else f"— a set of {v['inst']} instance(s) after the screens")
            C.append(f"`{stem}`: the geomean stands on **{v['n_gm']} instance(s)** {why}. "
                     f"Report the win count ({v['wins']}/{v['n']}) and the zero count, not the "
                     f"ratio.")
        if v["free"]:
            # Both geomeans can be None (every measured cell degenerate): never format blindly.
            size = (f" Leaving them in would read **{v['gm_withfree']:,.1f}x instead of "
                    f"{v['gm']:,.1f}x**." if (v.get("gm_withfree") and v.get("gm")) else
                    " Nothing else in the stage carries a ratio, so there is no geomean to move.")
            C.append(f"`{stem}`: {v['free']} instance(s) "
                     f"({', '.join('`' + n + '`' for n in v['free_names'])}) sample with **no "
                     f"search at all** — zero decisions, zero implications, success 1.0 — because "
                     f"after propagation the support is a free hypercube. Excluded for the same "
                     f"reason as vacuous pins." + size)
        if v["gpap0"]:
            C.append(f"`{stem}`: **GPAP emitted nothing on {v['gpap0']} instance(s) that CMSGen "
                     f"sampled** ({', '.join('`' + n + '`' for n in v['gpap0_names'])}). These are "
                     f"losses. They carry no finite ratio, so they cannot enter the geomean — "
                     f"which is exactly why they must be stated in the text.")
        if v["both0"]:
            C.append(f"`{stem}`: {v['both0']} instance(s) defeated **both** samplers "
                     f"({', '.join('`' + n + '`' for n in v['both0_names'])}) — a limit of the "
                     f"comparison, not evidence about either tool.")
        if v.get("weird"):
            C.append(f"`{stem}`: {v['weird']} cell(s) where both samplers emitted yet the row "
                     f"carries no finite `ratio` (a malformed or pre-schema row). They are in "
                     f"none of the printed columns, so the `inst` identity under Table 1 is "
                     f"short by exactly this count — fix the rows before publishing.")
        if v.get("noarm"):
            C.append(f"`{stem}`: **{v['noarm']} row(s) never ran the CMSGen arm at all** "
                     f"({', '.join('`' + n + '`' for n in v['noarm_names'])}) — a resume "
                     f"without the binary. Absent is not zero, so they are in none of the "
                     f"printed columns and the `inst` identity under Table 1 is short by "
                     f"exactly this count; re-run those cells before publishing.")
        if v.get("cms_err"):
            C.append(f"`{stem}`: **the CMSGen subprocess ERRORED on {v['cms_err']} row(s)** "
                     f"({', '.join('`' + n + '`' for n in v['cms_err_names'])}). A crashed "
                     f"baseline is not a zero; excluded from every column — re-run before "
                     f"publishing.")
    for stem, key, rival, proto in RIVALS:
        v = riv.get(stem)
        if v and v["errors"]:
            # Name the budget the rows were measured under: "the budget" without a number is
            # the sentence a referee cannot check.
            import re as _re
            budgets = sorted({int(m.group(1)) for r in (_rival_rows(args.rivals, stem) or {}).values()
                              for m in [_re.search(r"within the (\d+)s initialization budget",
                                                   str((r.get(key) or {}).get("error") or ""))]
                              if m})
            budget = (f"the {budgets[0]:,} s initialization budget" if len(budgets) == 1 else
                      "the initialization budget" if not budgets else
                      "the initialization budget (" + "/".join(f"{b:,} s" for b in budgets)
                      + " — MIXED, re-run to one budget before publishing)")
            # The RULE is stated once, below the list; each row carries only its own numbers.
            # Repeating it per row put the same 300 characters into the file four times.
            C.append(f"{rival} ({proto}): **{v['errors']} of {v['cells']}** runs produced no "
                     f"measurement inside the budget (did not initialize within {budget}, "
                     f"timed out, or no gradient path — the reason is recorded per row).")
            budget_rule = True
        if v and v["gpap0"]:
            C.append(f"{rival} ({proto}): **GPAP emitted nothing on {v['gpap0']} instance(s)** "
                     f"that the rival sampled. Losses, excluded from the geomean for lack of a "
                     f"finite ratio — state them.")
        if v and v.get("misaligned"):
            C.append(f"{rival} ({proto}): **{v['misaligned']} row(s) FAILED the constraint-"
                     f"alignment check** (`pin_check_ok: false` — the two sides were not given "
                     f"the same task). Excluded from every count; re-run those cells before "
                     f"quoting the family.")
        if v and v.get("nomeas"):
            C.append(f"{rival} ({proto}): {v['nomeas']} cell(s) have no unique/s estimate — no "
                     f"kept output row to measure the distinct fraction on (on rows before "
                     f"2026-09-03: an off-timing coverage run that collected nothing) while the "
                     f"timed run did produce models. Not measured, so not comparable in either "
                     f"direction; counted apart from GPAP=0, which means we produced nothing "
                     f"at all.")
    if any(v and not v["cpu"] for v in cms.values()):
        C.append("Some stages lack CPU provenance; a GPAP/CMSGen ratio is not interpretable "
                 "without it.")
    C.append("GPAP/CMSGen ratios are **host-dependent**: CMSGen is single-core, and this host is "
             "0.41x the reference laptop (verified independently by CMSGen's own propagation and "
             "decision counters). Always quote the CPU alongside the ratio.")
    # PER STAGE AND IN BOTH DIRECTIONS, never `max`: a screen that raises a headline is a result
    # and must be reported as loudly as one that lowers it. Gated on the MOVE, not on `vac`.
    effects = sorted(((k if isinstance(k, str) else "/".join(k), v["gm_all"] / v["gm"])
                      for k, v in cms.items()
                      if v and v.get("gm") and v.get("gm_all") and v["gm_all"] != v["gm"]),
                     key=lambda kv: -kv[1])
    # Named from the rows (`vac_names`) and per stage: the screen is per PIN, so an instance can
    # be a tautology under one target and not another.
    per_stage = [("/".join(k) if isinstance(k, tuple) else str(k), v["vac_names"])
                 for k, v in cms.items() if v and v.get("vac_names")]
    named = ("; ".join(", ".join(f"`{n}`" for n in names) + f" ({k})" for k, names in per_stage)
             if per_stage else "the vacuous-pin instances")
    C.append(f"Sequential geomeans **exclude** {named} — cells whose pinned outputs are constant "
             "for every input sequence (measured p_accept 1.0), so pinning them is a tautology.")
    if effects:
        moved = "; ".join(f"{k} {f:,.2f}x ({'the screens LOWER this headline' if f > 1 else 'the screens RAISE this headline'})"
                          for k, f in effects)
        C.append("What the degeneracy screens (vacuous pins, no-search and tiny-space cells) do "
                 "to each geomean, in **both** directions — unscreened / screened: " + moved
                 + ". A screen that raises a headline is reported here exactly as loudly as one "
                   "that lowers it.")
    # THE FAILURE REGIME, AS A BOUNDARY. Amplification (success / p_accept) is what propagation
    # buys over blind rejection; where it buys nothing and success is tiny, GPAP does rejection
    # sampling badly while a CDCL sampler still finds solutions.
    losses, amps = [], []
    for stem, _key, _rival, _proto in RIVALS:
        rows = _rival_rows(args.rivals, stem) or {}
        for (nm, _v), r in rows.items():
            g, pa = (r.get("gpap") or {}), r.get("p_accept")
            sps, csps = g.get("sps"), r.get("cms_sps")       # not `cms`: that is the dict
            if not (sps and csps and pa):
                continue
            amp = (g.get("success") or 0.0) / pa if pa else None
            if amp:
                amps.append((sps / csps >= 1, amp, g.get("success") or 0.0, nm))
                if sps < csps:
                    losses.append((nm, amp, g.get("success") or 0.0))
    if losses:
        worst_amp = max(a for _n, a, _s in losses)
        worst_suc = max(sc for _n, _a, sc in losses)
        safe = [x for x in amps if x[0] and (x[1] <= worst_amp and x[2] <= worst_suc)]
        C.append("**Where GPAP loses, and why**: every loss to CMSGen is a cell where propagation "
                 "buys almost nothing over blind rejection AND absolute success is tiny — "
                 "amplification (success / p_accept) at most "
                 f"{worst_amp:,.2f}x together with success at most {worst_suc:.1e} "
                 f"({', '.join(f'`{n}`' for n, _a, _s in sorted(losses))}). "
                 + (f"No cell meets both conditions and still wins ({len(safe)} counter-examples), "
                    "so the boundary is descriptive rather than fitted."
                    if not safe else
                    f"{len(safe)} winning cells also meet both conditions, so this is a tendency, "
                    "not a rule.")
                 + " That is the rejection sampler's collapse regime, and it is where a CDCL "
                   "sampler should be preferred.")
    # A ratio is only as precise as its denominator. A cell resting on a handful of samples IS
    # scored, and is named here on both sides so a reader can weigh what each geomean stands on.
    thin1 = []
    for (stem, enc), label, depth, tgt in t1:
        v = cms[(stem, enc)]
        if v.get("thin_cells"):
            thin1.append((f"{label}, {depth}, {tgt}", v["thin_cells"]))
    if thin1:
        C.append(f"**Thin CMSGen denominators, scored and stated**: cells in Table 1's geomean "
                 f"where CMSGen delivered fewer than {THIN_DENOMINATOR} samples in its whole "
                 "budget (Poisson relative error of 18% or more on its rate) — "
                 + "; ".join(f"{row}: " + ", ".join(f"`{n}` ({c})" for n, c in cells)
                             for row, cells in thin1)
                 + ". They enter the geomean (a sampler that delivered anything is scored); "
                   "read each as 'CMSGen barely samples this' — a capability contrast more "
                   "than a measured factor.")
    thin2 = []
    for stem, _key, rival, proto in RIVALS:
        v = riv.get(stem)
        if v and v.get("thin_cells"):
            thin2.append((rival, proto, v["thin_cells"]))
    if thin2:
        C.append(f"**Thin rival denominators, scored and stated**: cells in Table 2's geomean "
                 f"where the rival found fewer than {THIN_DENOMINATOR} solutions in its whole "
                 "run — "
                 + "; ".join(f"{rv} ({p}): "
                             + ", ".join(f"`{n}` ({u} solution{'s' if u != 1 else ''}, {x:,.0f}x)"
                                         for n, u, x in cells)
                             for rv, p, cells in thin2)
                 + ". Scored, as the rival papers' own tables score rates of 0.1 unique/s; the "
                   "ratio on such a cell is an order of magnitude, not a constant, and a row "
                   "whose geomean stands mostly on them says so above.")
    # THE SIGN OF A NEAR-PARITY CELL: GPAP's row is the median-rate seed of three against a
    # rival that ran once, so a ratio band straddling 1.0 is a tie, not a tally.
    straddle = []
    for stem, key, rival, proto in RIVALS:
        rows = _rival_rows(args.rivals, stem) or {}
        for name, variant, bucket, ours, rate in rival_cells(list(rows.values()), key):
            if bucket != "ok" or not rate or not ours:
                continue
            g = rows[(name, variant)].get("gpap") or {}
            lo, hi, sps = g.get("sps_min"), g.get("sps_max"), g.get("sps")
            if not (lo and hi and sps):
                continue
            r_lo, r_hi = ours * lo / sps / rate, ours * hi / sps / rate
            if min(r_lo, r_hi) < 1.0 <= max(r_lo, r_hi):
                straddle.append((rival, proto, name, ours / rate, r_lo, r_hi))
    if straddle:
        C.append("**Cells whose win/loss sign lies inside GPAP's own seed band** — the ratio at "
                 "our slowest and fastest of three seeds straddles 1.0, against a rival that ran "
                 "once: "
                 + "; ".join(f"`{n}` ({rv}, {p}: {x:.2f}x, band {a:.2f}–{b:.2f}x)"
                             for rv, p, n, x, a, b in straddle)
                 + ". Count them as ties, not as wins or losses.")
    # WHAT THE RIVAL GEOMEANS ARE MADE OF, per row: a headline resting on a few cells says so.
    for stem, key, rival, proto in RIVALS:
        v = riv.get(stem)
        if not v:
            continue
        n_ok = v["n"] - v["zero"] - v["gpap0"]
        if v.get("gm") and n_ok >= 3:
            parts = [f"**{rival} ({proto}): what the {v['gm']:,.1f}x is made of.** "
                     f"Median cell {v['median']:,.1f}x; GPAP's seed band moves the geomean to "
                     f"{v['gm_lo']:,.1f}–{v['gm_hi']:,.1f}x; {v.get('ties', 0)} tie(s) "
                     f"(sign inside that band) are excluded from `wins`"]
            if n_ok >= 8:
                parts.append("top cells " + ", ".join(f"`{nm}` {x:,.1f}x" for x, nm in v["top"][:4])
                             + f"; without the top cell {v['gm_wo_top1']:,.1f}x, without the top "
                             f"four {v['gm_wo_top4']:,.1f}x over {n_ok - 4}")
            if v.get("small_grant"):
                parts.append(f"{len(v['small_grant'])} cell(s) measured at the rival's smallest "
                             f"surviving batch (≤10,000, the rung above rejected by the card: "
                             + ", ".join(f"`{nm}` {b:,}" for _, nm, b in v["small_grant"])
                             + f") geomean {v['gm_small']:,.1f}x, against {v['gm_rest']:,.1f}x "
                             f"over the other {v['n_rest']} — a statement about this card's "
                             f"memory as much as about the tool; quote it as 'on an L4'")
            C.append("; ".join(parts) + ".")
        elif v.get("gm") and 0 < n_ok < 3:
            # the same rule Table 1 applies: a geomean over one or two cells is flagged, and its
            # cells are named
            C.append(f"**{rival} ({proto}): the {v['gm']:,.1f}x stands on {n_ok} cell(s) only** ("
                     + ", ".join(f"`{nm}` {x:,.1f}x" for x, nm in v["top"][:n_ok])
                     + f"; the rest of the row is {v['zero']} rival=0, {v.get('gpap0', 0)} "
                     f"GPAP=0, {v['errors']} rival failed, {v.get('excl', 0)} excluded of "
                     f"{v['cells']} cells). Report the win count ({v['wins']}/{v['n']}) and the "
                     f"zero count, not the ratio — it is an illustration, not an estimate.")
        elif v["cells"] and n_ok == 0:
            C.append(f"**{rival} ({proto}): NO measured ratio exists on this row** — "
                     f"{v['zero']} rival=0, {v.get('gpap0', 0)} GPAP=0, {v['errors']} rival "
                     f"failed, {v.get('excl', 0)} excluded ("
                     f"{excl_items(v.get('excl_names') or {}, EXCL_REASONS_T2) or 'nothing'}) "
                     f"of {v['cells']} cells. Its `wins` counts the rival-zero cells; read the "
                     f"row as a capability contrast, not a rate.")
        for nm, rate in v.get("unsat_rival") or []:
            C.append(f"**`{nm}` ({rival}, {proto}) is proved UNSAT at this depth and screened, "
                     f"yet the rival scores {rate:,.0f} unique/s there** — its depth-POOLED count "
                     f"admits solutions at shallower unroll depths, which the fixed-depth task has "
                     f"none of. GPAP delivers 0 on it. It is the one cell in the campaign where "
                     f"the pooled superset produces a rival-only result, and the `unsat` column "
                     f"is where it lives; say so.")
        for nm, free, total in v.get("low_vram") or []:
            C.append(f"`{nm}` ({rival}, {proto}): GPAP's arm ran with {free:.1f} of {total:.1f} GB "
                     f"free (a rival process still releasing the card), so its batch K was sized "
                     f"on a fraction of the device. Recorded per row (`gpu.free_gb`); the cell's "
                     f"verdict may be the residency, not the instance.")
        if v.get("ladder_top"):
            C.append(f"**{rival} ({proto}): on {len(v['ladder_top'])} cell(s) the batch ladder's "
                     f"TOP rung completed first try** (" + ", ".join(f"`{nm}` {b:,}" for nm, b in v["ladder_top"])
                     + "), so the card never bound the grant — the ladder did. The rival's rate is "
                     "near-proportional to its batch, so those cells may under-grant it: re-walk "
                     "them from a taller ladder (the top is 30,000,000 since protocol v1.0; "
                     "RIVAL_FAIRNESS_FINDINGS.md §16e/§17) — this caveat is the mechanical check "
                     "that the ladder's top is above the card's reach on every scored cell.")
    C.append("Rare-target rival ratios rest on near-zero denominators and a single rival run "
             "against GPAP's 3-seed median: read them as orders of magnitude, not constants.")
    if budget_rule:
        C.append("Every budget failure above is omitted from its geomean and never counted as a "
                 "zero. It is a statement about OUR budget, not about the rival's capability: "
                 "such a cell is re-runnable at a longer one (docs/findings/"
                 "RIVAL_FAIRNESS_FINDINGS.md §11, §16d).")
    # ============================================================ what the paper's TEXT quotes
    # Every number the paper states outside a table cell, derived here rather than by hand off a
    # printed table. A hand-derived number cannot be re-checked after a campaign moves a row --
    # which is exactly how a paper ends up quoting a speedup no surviving data supports.
    t1_rows = [(cms[k], label, depth, tgt) for k, label, depth, tgt in t1]
    # THE PROTOCOL'S OWN ROWS, by where they came from — not by which fields they happen to
    # carry. Selecting on `n_req` instead made every bullet below vanish on a stage that records
    # no request, and formatted the empty set as a number.
    prot = [cms[k] for k, *_ in t1 if str(k[0]).startswith("rivals:")]
    reqs = sorted({v["n_req"] for v in prot if v.get("n_req")})
    pre = [v["pre_frac"] for v in prot if v.get("pre_frac")]
    impl = [prop[k] for k, *_ in t1 if prop.get(k)]
    irate = [p["gpap"] for p in impl if p.get("gpap")]
    N = {
        "sample_request": reqs[0] if len(reqs) == 1 else (reqs or None),
        "scored_instances": sum(v["n"] for v in prot),
        "filled_request_gpap": sum(v.get("full_gpap", 0) for v in prot),
        "filled_request_cms": sum(v.get("full_cms", 0) for v in prot),
        "filled_request_unknown": sum(v.get("full_unknown", 0) for v in prot),
        "ratio_ss_min": min((v["gm"] for v in prot if v.get("gm")), default=None),
        "ratio_ss_max": max((v["gm"] for v in prot if v.get("gm")), default=None),
        "ratio_e2e_min": min((v["gm_e2e"] for v in prot if v.get("gm_e2e")), default=None),
        "ratio_e2e_max": max((v["gm_e2e"] for v in prot if v.get("gm_e2e")), default=None),
        "pre_frac_min": min(pre, default=None), "pre_frac_max": max(pre, default=None),
        # Table 3 across its rows: the rate ratio barely moves, which is the section's argument
        "impl_ratio_gm": gm([p["ratio"] for p in impl]),
        "impl_rate_min": min(irate, default=None), "impl_rate_max": max(irate, default=None),
        "impl_rate_spread": (max(irate) / min(irate)) if irate else None,
    }
    bullets = []
    if isinstance(N["sample_request"], int):
        bullets.append(f"sample request **{N['sample_request']:,}** per instance, on both tools")
    elif N["sample_request"]:
        bullets.append("the rows DISAGREE on the sample request: "
                       + ", ".join(f"{q:,}" for q in N["sample_request"])
                       + " — the delivery counts below pool two protocols")
    if N["scored_instances"]:
        bullets.append(
            f"of the **{N['scored_instances']}** instances where either sampler emits, GPAP "
            f"filled the whole request on **{N['filled_request_gpap']}**, CMSGen on "
            f"**{N['filled_request_cms']}**"
            + (f" ({N['filled_request_unknown']} of them record no request, so they are in "
               f"neither count)" if N["filled_request_unknown"] else ""))
    if N["ratio_ss_min"] and N["ratio_e2e_min"]:
        bullets.append(
            f"speedup over CMSGen: **{N['ratio_ss_min']:,.1f}x–{N['ratio_ss_max']:,.0f}x** steady "
            f"state, **{N['ratio_e2e_min']:,.1f}x–{N['ratio_e2e_max']:,.0f}x** end to end")
    # A row prints ONE `n` and TWO rate columns. They are the same cell set only while every
    # scored row carries both clocks; a row missing `sps_e2e` silently narrows the end-to-end
    # column while `n` goes on claiming the wider set. Say so rather than let the reader assume.
    split = [(label, tgt, v["n_gm"], v["n_rate_e2e"]) for v, label, _d, tgt in t1_rows
             if v.get("n_rate_e2e") is not None and v["n_rate_e2e"] != v["n_gm"]]
    if split:
        bullets.append("⚠ the end-to-end column stands on FEWER cells than the row's `n`: "
                       + "; ".join(f"{lb} ({tg}) {e2e} of {n}" for lb, tg, n, e2e in split)
                       + " — those rows carry no `sps_e2e` on every scored cell, so the two "
                         "rate columns are not over one set and `n` describes the wider one")
    if N["pre_frac_min"]:
        bullets.append(f"preparation is a median **{100 * N['pre_frac_min']:.0f}–"
                       f"{100 * N['pre_frac_max']:.0f}%** of GPAP's end-to-end wall across "
                       f"the rows")
    if N["impl_rate_min"] and N["impl_ratio_gm"]:
        bullets.append(
            f"GPAP's propagation rate spans **{N['impl_rate_min']:.3g}–{N['impl_rate_max']:.3g}** "
            f"impl/s, so the slowest row is within **{N['impl_rate_spread']:.2f}x** of the "
            f"fastest, and the rate ratio's geomean over the rows is "
            f"**{N['impl_ratio_gm']:,.0f}x** CMSGen's")
    if bullets:
        M += ["", "## Numbers the paper's text quotes", "",
              "Checked against the PDF by `eval/artifact/check_paper_numbers.py`.", ""] \
            + [f"* {b}" for b in bullets]


    if getattr(args, "out_json", None):
        _bp = (barb or {}).get("params") or {}
        _bcost = BARBARIK_SETTINGS.get((_bp.get("epsilon"), _bp.get("eta"), _bp.get("delta")), {})
        # THE CONTRACT with the paper: one machine-readable file holding every published number,
        # so checking the PDF is a diff and not a re-reading. Shapes may differ between the
        # generated table and the typeset one (the paper fits its own columns); the NUMBERS may
        # not.
        doc = {
            "generated": time.strftime("%Y-%m-%d %H:%M"), "commit": commit,
            "engine": cores, "gpu": prov.get("gpu"), "cpu": prov.get("cpu"),
            "cms_patched": prov.get("patched"),
            "table1": [{"set": label, "target": tgt, "depth": depth, "n": v["n_gm"],
                        "cms_sps": v.get("gm_cms_sps"), "gpap_sps": v.get("gm_gpap_sps"),
                        "gpap_sps_e2e": v.get("gm_gpap_e2e"),
                        "cms_sps_e2e": v.get("gm_cms_e2e"),
                        "ratio_ss": v.get("gm"), "ratio_e2e": v.get("gm_e2e"),
                        # how many cells each printed rate stands on: equal to `n` while every
                        # scored row carries both clocks, and the thing to check when it is not
                        "n_rate": v.get("n_rate"), "n_rate_e2e": v.get("n_rate_e2e"),
                        "pre_frac": v.get("pre_frac"),
                        "wins": v["wins"], "win_denom": v["n"], "cms_zero": v["inf"],
                        "gpap_zero": v["gpap0"], "excluded": v.get("excl", 0),
                        "filled_gpap": v.get("full_gpap"), "filled_cms": v.get("full_cms")}
                       for v, label, depth, tgt in t1_rows],
            "table2": [{"rival": rival, "task": proto, "n": riv[stem]["n"] - riv[stem]["zero"]
                        - riv[stem]["gpap0"],
                        "rival_unique_per_s": riv[stem].get("gm_rival"),
                        "gpap_unique_per_s": riv[stem].get("gm_gpap"),
                        "ratio": riv[stem].get("gm"), "n_rate": riv[stem].get("n_rate"),
                        "wins": riv[stem]["wins"],
                        "win_denom": riv[stem]["n"], "ties": riv[stem].get("ties", 0),
                        "rival_zero": riv[stem]["zero"], "failed": riv[stem]["errors"],
                        "excluded": riv[stem].get("excl", 0),
                        "small_grant_gm": riv[stem].get("gm_small"),
                        "small_grant_n": len(riv[stem].get("small_grant") or []),
                        "rest_gm": riv[stem].get("gm_rest"),
                        "rest_n": riv[stem].get("n_rest")}
                       for stem, _k, rival, proto in RIVALS if riv.get(stem)],
            "table3": [{"set": label, "target": tgt, "depth": depth, "n": prop[k]["n"],
                        "cms_impl_per_s": prop[k]["cms"], "gpap_impl_per_s": prop[k]["gpap"],
                        "cms_impl_per_sample": prop[k]["cper"],
                        "gpap_impl_per_sample": prop[k]["gper"],
                        "work_ratio": prop[k]["work"], "impl_ratio": prop[k]["ratio"]}
                       for k, label, depth, tgt in t1 if prop.get(k)],
            "text": N,
            # The paper's uniformity claim is PROSE, not a table, so these are the numbers it
            # quotes there: the eligible set after the control's own screen, and each arm's
            # verdicts. `pipe:circuit` is the shipped default; `gpap:b8` is the pinned config.
            "uniformity": ({
                "params": barb["params"],
                "rounds": _bcost.get("rounds"), "samples_total": _bcost.get("samples"),
                "samples_under_test": _bcost.get("under_test"),
                "eligible": len(barb["eligible"]),
                "control_rejected": len(barb["control_rejected"]),
                "arms": {k: {"accept": v["accept"], "reject": v["reject"],
                             "no_verdict": v["error"], "n": v["n"]}
                         for k, v in barb["summary"].items()},
            } if barb else None),
        }
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as fh:   # noqa: SIM115
            json.dump(doc, fh, indent=2, sort_keys=False)
            fh.write("\n")
        print(f"wrote {args.out_json}")

    if getattr(args, "out_notes", None):
        # THE AUTHORS' NOTES: what was rendered beside the tables while the paper was written.
        # Derived from the same rows, so a campaign that moves a row moves these with it.
        pa_their = _pa_range([_rival_rows(args.rivals, st) or {}
                              for st, _k, _r, _p in RIVALS if "rare" not in st])
        pa_rare = _pa_range([_rival_rows(args.rivals, st) or {}
                             for st, _k, _r, _p in RIVALS if "rare" in st])
        NM = ["# Notes on the campaign's numbers — for the paper's authors", "",
              "The per-campaign analysis behind the paper's wording, from the same rows as "
              "`RESULTS.md`. The artifact does not produce it; the standing rules are in "
              "`docs/papers/tacas-27/README.md`.", "",
              "## Caveats to state with these numbers", ""] + [f"* {c}" for c in C]
        NM += ["", "## The targets' acceptance probabilities", "",
               f"* original targets (the 1–5 outputs the rival's own table names): measured "
               f"p_accept {pa_their}",
               f"* rare targets (every output pinned to a rare reachable pattern): {pa_rare}"]
        soli = [p for p in prop.values() if p and p["n"] >= 3]
        if soli:
            lo, hi = min(p["gpap"] for p in soli), max(p["gpap"] for p in soli)
            NM += ["", "## The propagation rate across the rows", "",
                   f"Over the {len(soli)} rows of Table 3 with at least three cells, GPAP holds "
                   f"{gm([p['gpap'] for p in soli]):,.3g} implications/s, the fastest within "
                   f"{hi / lo:.2f}x of the slowest, against CMSGen's "
                   f"{gm([p['cms'] for p in soli]):,.3g}: a {gm([p['ratio'] for p in soli]):,.0f}x "
                   "engine advantage that barely moves, while Table 1's sample-rate ratios span "
                   "two orders of magnitude. The spread is how much work each sample costs, not "
                   "the propagator."]
        cal = rival_calibration(args.rivals) if args.rivals else []
        if cal:
            NM += [""] + cal
        if barb:
            NM += ["", "## Barbarik headroom", "",
                   "CM_test's own per-round bucket share (it rejects outside [0.05, 0.95]): the "
                   "verdict alone hides how much headroom an Accept had.", "",
                   "| sampler under test | mean bias | max bias |", "|---|---:|---:|"]
            for arm, a in barb["summary"].items():
                bm = "-" if a["bias_mean"] is None else f"{a['bias_mean']:.3f}"
                bx = "-" if a["bias_max"] is None else f"{a['bias_max']:.3f}"
                NM.append(f"| `{arm}` | {bm} | {bx} |")
        os.makedirs(os.path.dirname(args.out_notes) or ".", exist_ok=True)
        with open(args.out_notes, "w") as fh:   # noqa: SIM115
            fh.write("\n".join(NM) + "\n")
        print(f"wrote {args.out_notes}")

    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
        with open(args.out_md, "w") as fh:      # noqa: SIM115
            fh.write("\n".join(M) + "\n")
        print(f"wrote {args.out_md}")
    print(f"\n{cell_line}")


def main():
    ap = argparse.ArgumentParser(description="The paper's tables, from a campaign's rows.")
    ap.add_argument("--rivals", required=True,
                    help="the campaign directory (rivals_*.json): Tables 1, 2 and 3 are read off "
                         "these rows")
    ap.add_argument("--barbarik", nargs="*", default=[],
                    help="barbarik_gpap.py JSONL result files (the uniformity table)")
    ap.add_argument("--out-md", default=None, help="the tables, as the paper prints them")
    ap.add_argument("--out-json", default=None,
                    help="every published number, machine-readable: the contract "
                         "check_paper_numbers.py holds the paper to")
    ap.add_argument("--out-notes", default=None,
                    help="the per-campaign analysis behind the paper's wording, for the authors")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
