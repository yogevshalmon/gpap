#!/usr/bin/env python3
"""Build the paper's experiment section — every table, straight from the experiment JSONs.

Nothing is typed by hand: every published number is derived from the data with the exclusions
applied in ONE place. Raw data lives outside the repo in $GPAP_RESULTS (default
../gpap-results); docs/paper/README.md has the regeneration recipe and the command line.

THE table rules, applied everywhere so they cannot diverge between tables:
  * Vacuous pins (p_accept 1.0: pinning is a tautology) are screened before any bucket.
  * Zero-search cells: no decisions and no implications means the ratio measures the instance.
  * A solution space smaller than the sample request is enumerated, not sampled.
  * Either side emitting nothing has no finite ratio: its own column, CMSGen=0 (a win) or
    GPAP=0 (a loss), so the columns sum to the instance count.
  * Rival rows that errored are counted, never folded in as zeros.
  * Every other reason a cell carries no ratio is the ONE printed `excluded` column, itemised
    by name in the text under the table.
  * A tool that scored anything is scored: no thin floor, thin cells disclosed by name.
  * A stage with fewer instances than its family's maximum is marked incomplete.
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
from eval.rivals.rival_common import space_from_distinct  # noqa: E402 — after sys.path

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
    ratio measures the instance. Detected per row from `sim_baseline.p_accept`."""
    if name and name in VACUOUS_PINS:
        return True
    p = (r.get("sim_baseline") or {}).get("p_accept")
    if p is None:
        p = r.get("p_accept")                  # rival rows carry it at the top level
    return p is not None and p >= 1.0


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
    # implication per model delivered is reading bits out of a hypercube. The threshold sits
    # inside a two-order-of-magnitude gap in the measured values, not fitted to them.
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
    imported by every consumer (`rival_family` here, `eval/eval_plots.py`, `rivals_summary.py`)."""
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

    def _all(pred):
        """geomean over cells that pass `pred`, used only for the size-of-the-error caveats."""
        return gm([r["ratio"] for k, r in cells.items()
                   if pred(k) and r.get("ratio") and math.isfinite(r["ratio"])
                   and scored(r, "n_models") and scored(r, "cms_n_samples")])

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

def prop_rates(d, stem, encoder):
    """Engine-level throughput: implications and decisions per second, both samplers.

    Isolates the ENGINE from the sampler wrapped around it, and factors the headline exactly:

        samples/s ratio  =  (implications/s ratio)  /  (implications-per-sample ratio)

    The first factor is what the GPU propagator buys, the second what our algorithm SPENDS —
    GPAP discards a conflicted trajectory outright where CMSGen backtracks and keeps the partial
    work. CMSGen's counters come from `--statsevery` in scripts/cmsgen_gpap_eval.patch."""
    payload = load(os.path.join(d, f"{stem}.json")) if d else None
    if not payload:
        return None
    _, cells = select_cells(payload, encoder)
        # Same exclusions as the headline: Tables 1 and 3 must describe the same runs.
    rows = [r for k, r in cells.items()
            if not vacuous_pin(r, k) and not zero_search(r) and not tiny_space(r)]
    # EVERY column is computed over the SAME rows, the ones where all four quantities are
    # positive: filtering per column breaks the identity.
    return _prop_stats([(r.get("implications_per_s"), r.get("cms_implications_per_s"),
                         r.get("gpap_sps"), r.get("cms_sps"),
                         r.get("decisions_per_s"), r.get("cms_decisions_per_s"))
                        for r in rows], encoder)


def _prop_stats(quads, encoder):
    """The Table 3 aggregation over (impl_g, impl_c, sps_g, sps_c, dec_g, dec_c) tuples —
    shared by the bench-stage and the rival-stage readers so the identity is one computation."""
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
    # construction the same way. The end-to-end comparison is Table 1's. Nothing reads `wall_s`.
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
        sweep = rv.get("batch_sweep") or []
        # cells the rival ran at its smallest surviving rung, the one above rejected: the card
        # bound the grant there, so the cell measures the tool at a small batch
        if rv.get("batch") and rv["batch"] <= 10_000 and len(sweep) > 1:
            small.append((x, name, rv["batch"]))
        # cells where the ladder's TOP rung completed first try: the ladder bound the grant
        if rv.get("batch") and len(sweep) == 1 and rv["batch"] == (sweep[0].get("batch") or 0):
            ladder_top.append((name, rv["batch"]))
    okc.sort(key=lambda t: -t[0])
    ties = sum(1 for _, _, _, st in okc if st)
    small_names = {nm for _, nm, _ in small}
    rest = [x for x, nm, _, _ in okc if nm not in small_names]
    # THE ONE PRINTED EXCLUSION COLUMN of Table 2: every cell without a ratio for a reason other
    # than a tool scoring zero or the rival failing. cells = n + rival=0 + GPAP=0 + failed + excl.
    excl_buckets = ("vacuous", "unsat", "tiny", "free", "misaligned", "nomeas", "both0",
                    "gpap_error")
    return {"cells": len(cells), "gm": gm(ratios), "n": len(ratios) + zero + gzero,
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
    e2e, pre = [], []
    for k in finite:
        g, r = seen[k]["gpap"], seen[k]
        ge = _gpap_e2e(g)
        if ge and r.get("cms_sps_e2e"):
            e2e.append(ge / r["cms_sps_e2e"])
        if ge and g.get("sps"):
            pre.append(max(0.0, 1.0 - ge / g["sps"]))     # (wall_total - timed) / wall_total

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


# ------------------------------------------------------------------ uniformity

# Reading order for the uniformity table: controls first (without a negative control that is NOT
# flagged and a positive control that IS, "0/36 flagged" means nothing), then the B ladder.
UNI_ORDER = ["ctl:uniform", "ctl:tv0.05", "ctl:tv0.1",
             "gpap:b1", "gpap:b8", "gpap:b16", "gpap:b32", "gpap:b64",
             "pipe:default", "pipe:circuit", "pipe:circuit+soft", "gpap:soft", "cmsgen"]


def uni_sort_key(item):
    (name, kind) = item
    idx = UNI_ORDER.index(name) if name in UNI_ORDER else len(UNI_ORDER)
    return (idx, name, kind)


def uniformity(paths):
    """Aggregate the exact-oracle fuzzing campaign per sampler, split by instance kind.

    Screen first, then aggregate: an instance whose solution set is a full SUBCUBE is sampled by
    the don't-care FILLER, not by the config under test. fuzz_uniformity.py stamps
    `meta.subcube`; the excluded count is reported in `args["_subcube_excluded"]`."""
    rows = []
    args = {}
    for p in paths or []:
        d = load(p)
        if d:
            rows += d.get("rows", [])
            args = d.get("args", args)
    if not rows:
        return None, {}
    n_sub = sum(1 for r in rows if (r.get("meta") or {}).get("subcube"))
    rows = [r for r in rows if not (r.get("meta") or {}).get("subcube")]
    args = dict(args)
    args["_subcube_excluded"] = n_sub
    if not rows:
        return None, args
    agg = {}
    for r in rows:
        kind = r.get("kind", "?")
        for name, s in (r.get("samplers") or {}).items():
            k = (name, kind)
            a = agg.setdefault(k, {"tv": [], "reject": 0, "n": 0, "unsound": 0})
            a["tv"].append(s.get("tv_median"))
            a["n"] += 1
            a["unsound"] += s.get("unsound") or 0
            if s.get("verdict") == "SKEW":
                a["reject"] += 1
    out = {}
    for (name, kind), a in agg.items():
        out[(name, kind)] = {"tv": med(a["tv"]), "reject": a["reject"], "n": a["n"],
                             "unsound": a["unsound"]}
    return out, args


# ------------------------------------------------------------------ barbarik

def barbarik(paths):
    """The standard certificate, per sampler under test.

    The eligibility rule and the per-arm counts are IMPORTED from barbarik_table.py: an instance
    counts only where the negative control (the trusted uniform reference, under test through
    the same hook) was itself Accepted, since where it is Rejected CM_test measures the
    instance."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uniformity"))
    from barbarik_table import arm_summary, buckets, check_one_setting, load as bload

    rows = []
    for p in paths or []:
        if os.path.exists(p):
            rows += bload([p])
    if not rows:
        return None
    eps, eta, delta = check_one_setting(rows)      # raises on a pooled loose+strict file list
    eligible, rejected, errored = buckets(rows)
    return {"summary": arm_summary(rows, eligible), "eligible": sorted(eligible),
            "control_rejected": sorted(rejected), "control_errored": sorted(errored),
            "params": {"epsilon": eps, "eta": eta, "delta": delta},
            # provenance: this table is its own campaign, and the header above it is not its host
            "source": sorted(os.path.abspath(p) for p in (paths or []) if os.path.exists(p))}


# ------------------------------------------------------------------ reproducibility

def recheck(orig, rec, stages=("seq-c25-rare", "comb-rare", "epfl-rare", "sk")):
    if not (orig and rec):
        return None
    exact = mismatch = succ_ok = succ_bad = 0
    rates = []
    for stage in stages:
        a = load(os.path.join(orig, f"{stage}.json"))
        b = load(os.path.join(rec, f"{stage}.json"))
        if not (a and b):
            continue

        def cells(p):
            return {(r["name"].split(".")[0], r.get("enc_key") or r.get("encoder")): r
                    for r in p.get("results", []) if not r.get("skipped") and not r.get("error")}
        ca, cb = cells(a), cells(b)
        for k in set(ca) & set(cb):
            ra, rb = ca[k], cb[k]
            ea, eb = ra.get("engine") or {}, rb.get("engine") or {}
            same = (ra.get("M") == rb.get("M") and ra.get("support") == rb.get("support")
                    and ea.get("split") == eb.get("split") and ea.get("mode") == eb.get("mode"))
            exact += same
            mismatch += (not same)
            if abs((ra.get("succ") or 0) - (rb.get("succ") or 0)) <= 0.05:
                succ_ok += 1
            else:
                succ_bad += 1
            if ra.get("gpap_sps") and rb.get("gpap_sps"):
                rates.append(rb["gpap_sps"] / ra["gpap_sps"])
    return {"exact": exact, "mismatch": mismatch, "succ_ok": succ_ok, "succ_bad": succ_bad,
            "gm": gm(rates), "lo": min(rates) if rates else None,
            "hi": max(rates) if rates else None, "n": len(rates)}


# ------------------------------------------------------------------ rendering

def tex_escape(s):
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


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


def stage_encoders(d, stem):
    """Which encoders a stage's JSON contains. One table row is emitted per (family, encoder)."""
    payload = load(os.path.join(d, f"{stem}.json")) if d else None
    return encoders_in(payload) if payload else []


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
        out = ["### Rival configuration — what their other values measured on this card", "",
               "EVIDENCE, not configuration: every campaign row runs the rival at its shipped "
               "script setting (the **pinned** arm). The other arms are the values the tool's own "
               "artifacts also give the knob — its argparse default and its PAPER — each measured "
               "once, on one instance, on the campaign's card, so the text can state what the "
               "paper's setting would have given the tool. A paper arm below 1.00x means the "
               "script setting the campaign used is the BETTER one for the rival."] + out
    return out


def build(args):
    # KEY = (family stem, encoder): both encoders are reported, nothing selects between them.
    fam_encs = [(stem, e) for stem, _, _, _ in CMS_FAMILIES
                for e in stage_encoders(args.cmsgen, stem)]
    cms = {(s, e): cms_family(args.cmsgen, s, e) for s, e in fam_encs}
    ref = {(s, e): cms_family(args.cmsgen_ref, s, e)
           for s, e in [(st, en) for st, _, _, _ in CMS_FAMILIES
                        for en in stage_encoders(args.cmsgen_ref, st)]} if args.cmsgen_ref else {}
    riv = {s: rival_family(args.rivals, s, k) for s, k, _, _ in RIVALS}
    prop = {(s, e): prop_rates(args.cmsgen, s, e) for s, e in fam_encs}
    # THE ROWS OF TABLES 1 AND 3, in print order: the protocol's cells, then any supplementary
    # family stage. One list, so both tables and the caveats describe one set of rows.
    t1 = []
    for stem, label, tgt in RIVAL_CMS_SETS:
        v = rival_cms_family(args.rivals, stem, _RIVAL_KEY[stem])
        if not v:
            continue
        k = (f"rivals:{stem}", v["encoder"])
        cms[k] = v
        prop[k] = rival_prop_rates(args.rivals, stem, _RIVAL_KEY[stem])
        t1.append((k, label, v.get("depth") or "—", tgt))
    for stem, label, depth, tgt in CMS_FAMILIES:
        for enc in stage_encoders(args.cmsgen, stem):
            if cms.get((stem, enc)):
                t1.append(((stem, enc), label, depth, tgt))
    uni, uni_args = uniformity(args.uniformity)
    barb = barbarik(args.barbarik)
    rc = recheck(args.cmsgen, args.recheck) if args.cmsgen else None
    soft = load(args.soft) if args.soft else None

    expected = {}
    for (stem, _enc), v in cms.items():
        if v:
            fam = stem.split("-")[0]
            expected[fam] = max(expected.get(fam, 0), v["inst"])

    prov = next((v for v in cms.values() if v), {}) or {}
    # The commit below is when the TABLES were generated, not what measured the data — that is
    # the per-row build id, unioned over every stage.
    cores = sorted({c for v in cms.values() if v for c in v.get("cores", [])})
    # A FAILED git call does not raise here — `subprocess.run` without check=True just returns a
    # non-zero code and empty stdout — so the except below never fired in the one place it was
    # meant to: the packaged artifact, which carries no .git and rendered this field as an empty
    # pair of backticks. Say "not a git checkout" instead; a blank reads as a value.
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                           text=True, timeout=10)
        commit = r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "not a git checkout"
    except Exception:                                   # noqa: BLE001
        commit = "not a git checkout"

    # Totals: `cells`/`unsound` are per-encoder and sum over every key, while `inst`/`skipped`/
    # `errors` describe the whole STAGE and are taken once per family.
    one_per_stem = {}
    for (stem, _enc), v in cms.items():
        if v and stem not in one_per_stem:
            one_per_stem[stem] = v
    tot = {kk: sum(v[kk] for v in cms.values() if v) for kk in ("cells", "unsound")}
    tot["refused"] = sum(v.get("errors", 0) for v in cms.values() if v)
    tot.update({kk: sum(v[kk] for v in one_per_stem.values())
                for kk in ("inst", "skipped", "errors")})

    # The encoder column is printed only when the rows disagree on it.
    encs_t1 = sorted({(enc or "—") for (_s, enc), *_ in t1})
    show_enc = len(encs_t1) > 1
    enc_note = ("" if show_enc or not encs_t1 else
                f" GPAP samples the `{encs_t1[0]}` CNF on every row.")
    # ============================================================ markdown
    M = [f"# GPAP — experiment section data  ({time.strftime('%Y-%m-%d')})", "",
         f"* GPU **{prov.get('gpu')}**, CPU **{prov.get('cpu') or 'not recorded'}**",
         f"* engine {', '.join(f'`{c}`' for c in cores) or '`not recorded`'} (per-row build id)"
         f"  |  tables generated at `{commit}`"
         f"  |  CMSGen patched (`--nowrite`): **{prov.get('patched')}**",
         f"* **{tot['cells'] + tot['refused']} rows: {tot['cells']} measured cells + "
         f"{tot['refused']} refused (GPAP's encoder or envelope), {tot['inst']} instance-slots, "
         f"{tot['skipped']} skipped, {tot['errors']} errors, {tot['unsound']} unsound**", "",
         "Generated by `eval/artifact/paper_tables.py`. A cell that measures the instance rather "
         "than the samplers is excluded from every geomean; an instance where EITHER sampler "
         "emitted nothing has no finite ratio and is reported in its own column instead.", "",
         "## Table 1 — GPAP vs CMSGen", "",
         "**The same cells as Table 2.** The protocol: each GPU rival's own published benchmark "
         "set — the whole set, one population per row — under the pins its table names and "
         "under our rare pin; on every cell three tools on one constraint — the rival on its "
         "circuit, GPAP on the `abc:fast` CNF of that circuit and those pins, and CMSGen on "
         "that very CNF file. Any `seq-*`/`comb-*`/… rows below them are the older `gpap bench` "
         "family matrix (instances pinned our way), supplementary and printed only when "
         "present. One encoder; nothing selects between runs (a per-instance best-of keyed on "
         "the ratio would let CMSGen's number pick which of ours is published)." + enc_note, "",
         "Columns add up: `inst` = `n` + CMSGen=0 + GPAP=0 + excluded. `n` is the number of "
         "instances the geomean stands on; `CMSGen=0` / `GPAP=0` are the one-sided zeros (an "
         "outright win / an outright loss, no finite ratio); `excluded` is every other cell "
         "without a ratio — the instance screens (a vacuous pin, a free support, a tiny solution "
         "space, a cell proved UNSAT at this depth), a cell neither sampler reached, a cell "
         "GPAP's encoder refused — itemised by name under the table. Every instance is counted "
         "exactly once, `wins/n` is counted over the same set the geomean is (plus the two zero "
         "columns), and the degenerate cells are screened out BEFORE the ledger rather than only "
         "out of the geomean. A tool that delivered anything is scored: cells whose ratio rests "
         f"on fewer than {THIN_DENOMINATOR} samples on a side are named in the caveats, not "
         "screened.", "",
         "**Two rate columns, and each charges BOTH sides the same way.** `GPAP/CMSGen` is "
         "steady state over steady state: GPAP's winning-config sampling rate over CMSGen's "
         "samples per SAMPLING wall (its own `--statsevery` clock, which starts after it has "
         "read the CNF). `e2e` is end-to-end over end-to-end: GPAP's `wall_total` (parse + "
         "engine construction + ELS + warmup + entailed-clause MINING + the timed run) over "
         "CMSGen's whole subprocess wall. `pre%` is the median share of GPAP's end-to-end wall "
         "that preparation takes. Earlier revisions scored GPAP's steady state against "
         "CMSGen's subprocess wall, which charged one side for getting ready and not the other. "
         "Preparation is a real cost at these `n_time` values and does not amortise away until "
         "the run is much longer — quote `e2e` for a delivered-samples claim and the "
         "steady-state column only for the large-n regime, saying which. (Circuit ENCODING is "
         "in neither: it is reported per instance as `t_enc` and belongs to the encoder, not "
         "the sampler.)", "",
         "| instance set | depth | pins |" + (" enc |" if show_enc else "")
         + " inst | n | GPAP/CMSGen | e2e | pre% | wins | CMSGen=0 | GPAP=0 | excluded |"
         + (" ref GPU |" if ref else ""),
         "|---|---|---|" + ("---|" if show_enc else "")
         + "---:|---:|---:|---:|---:|---:|---:|---:|---:|" + ("---:|" if ref else "")]
    for (stem, enc), label, depth, tgt in t1:
        v = cms[(stem, enc)]
        exp = expected.get(stem.split("-")[0], v["inst"])
        mark = "" if v["inst"] >= exp else f" (partial {v['inst']}/{exp})"
        # The reference-GPU column exists only with a second sweep (--cmsgen-ref).
        rv = ref.get((stem, enc))
        refs = (f" {rv['gm']:,.1f}x |" if rv and rv["gm"] else " — |") if ref else ""
        # No finite ratio: say WHY when the reason is CMSGen zeros, "—" when a screen took them.
        gms = (f"{v['gm']:,.1f}x" if v["gm"] else
               ("— (all CMSGen=0)" if v["inf"] else "—"))
        if v["gm"] and v["n_gm"] < 3:
            gms += f" ⚠n={v['n_gm']}"
        e2e = f"{v['gm_e2e']:,.1f}x" if v.get("gm_e2e") else "—"
        M.append(f"| {label}{mark} | {depth} | {tgt} |" + (f" {enc or '—'} |" if show_enc else "")
                 + f" {v['inst']} | {v['n_gm']} | {gms} | {e2e} "
                 f"| {100 * v.get('pre_frac', 0):.0f}% | {v['wins']}/{v['n']} | {v['inf']} "
                 f"| {v['gpap0']} | {v.get('excl', 0)} |{refs}")
    # THE EXCLUDED COLUMN, ITEMISED: prose under the table, from the lists the count sums over.
    excl_lines = []
    for (stem, enc), label, depth, tgt in t1:
        v = cms[(stem, enc)]
        items = excl_items(v.get("excl_names") or {}, EXCL_REASONS_T1)
        if items:
            excl_lines.append(f"* {label}, {depth}, {tgt}: {items}.")
    M += ["",
          "**What `excluded` holds, per row** — cells that carry no ratio for a reason other "
          "than one sampler scoring zero. A cell that measures the instance (a vacuous pin, a "
          "free support, a tiny space, a proved-UNSAT cell) is evidence about neither sampler; "
          "a cell neither sampler reached bounds the comparison, not a tool; a cell GPAP's "
          "encoder refused is a reach limit, not a zero.", ""] + (excl_lines or ["* none"])
    g0 = sorted({n for v in cms.values() if v for n in v["gpap0_names"]})
    b0 = sorted({n for v in cms.values() if v for n in v["both0_names"]})
    M += ["",
          "**Instances GPAP could not sample** (CMSGen could): "
          + (", ".join(f"`{n}`" for n in g0) if g0 else "none") + ".",
          "",
          "**Instances neither sampler could touch**: "
          + (", ".join(f"`{n}`" for n in b0) if b0 else "none")
          + " — these bound the reach of the whole comparison, not of either tool."]
    # THE CIRCUIT HANDED TO THE RIVAL, stated from the rows: a fairness procedure must name what
    # actually ran.
    variants = sorted({v for stem, _k, _r, _p in RIVALS
                       for (_n, v) in (_rival_rows(args.rivals, stem) or {})})
    if len(variants) > 1:
        variant_sentence = ("Each rival is handed an ABC-simplified copy of the circuit as well "
                            "as the raw one and is scored on whichever suits it.")
    elif variants == ["abc"]:
        variant_sentence = ("Every rival is handed the SAME circuit file GPAP encodes from — the "
                            "sequential-ABC-simplified netlist (`variant: abc` on every row). "
                            "The raw-netlist axis was not run in this campaign, and GPAP's "
                            "post-unroll ABC pass has no sequential equivalent the rival could "
                            "be given; that residual is a property of GPAP's encoder and is "
                            "stated, not corrected.")
    elif variants == ["raw"]:
        variant_sentence = ("Every rival is handed the raw netlist (`variant: raw` on every row) "
                            "while GPAP encodes an ABC-optimized CNF of it — an asymmetry in "
                            "the rival's disfavour that is stated, not corrected.")
    else:
        variant_sentence = "Circuit variant per row: " + (", ".join(variants) or "none") + "."
    # THE ACCEPTANCE PROBABILITIES, read off the rows: a fixed string belongs to one campaign.
    pa_their = _pa_range([_rival_rows(args.rivals, st) or {}
                          for st, _k, _r, pr in RIVALS if "rare" not in st])
    pa_rare = _pa_range([_rival_rows(args.rivals, st) or {}
                         for st, _k, _r, pr in RIVALS if "rare" in st])
    M += ["", "## Table 2 — GPAP vs the GPU gradient samplers", "",
          "| rival | task | cells | n | GPAP/rival | wins | ties | rival=0 | GPAP=0 "
          "| rival failed | excluded |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for stem, _, rival, proto in RIVALS:
        v = riv.get(stem)
        if not v:
            M.append(f"| {rival} | {proto} | — | — | not run | — | — | — | — | — | — |")
            continue
        gs = f"{v['gm']:,.1f}x" if v["gm"] else "—"
        # `n` in Table 1 is the geomean's instance count and EXCLUDES the zero buckets; here
        # `v["n"]` is the WIN denominator and includes them. Print both.
        n_gm_ok = v["n"] - v["zero"] - v["gpap0"]
        if v["gm"] and n_gm_ok < 3:
            gs += f" ⚠n={n_gm_ok}"
        M.append(f"| {rival} | {proto} | {v['cells']} | {n_gm_ok} | {gs} | "
                 f"{v['wins']}/{v['n']} | {v.get('ties', 0)} | {v['zero']} | {v['gpap0']} "
                 f"| {v['errors']} | {v.get('excl', 0)} |")
    excl_lines = []
    for stem, _, rival, proto in RIVALS:
        v = riv.get(stem)
        if not v:
            continue
        parts = []
        if v.get("error_names"):
            parts.append("rival failed (no measurement inside the budget; the reason is per "
                         "row) " + ", ".join(f"`{n}`" for n in v["error_names"]))
        items = excl_items(v.get("excl_names") or {}, EXCL_REASONS_T2)
        if items:
            parts.append(items)
        if parts:
            excl_lines.append(f"* {rival}, {proto}: " + "; ".join(parts) + ".")
    M += ["",
          "Columns add up: `cells` = `n` + rival=0 + GPAP=0 + rival failed + excluded. `n` is "
          "what the geomean stands on; `rival=0` / `GPAP=0` are the one-sided zeros; `rival "
          "failed` counts runs that produced no measurement inside the budget (did not "
          "initialize, timed out, no gradient path — counted, never folded in as zeros); "
          "`excluded` is every other cell without a ratio (the instance screens, a misaligned "
          "or unmeasured row, a cell neither tool reached, a cell GPAP's encoder refused), "
          "itemised here. A rival that scored anything is scored — its own tables print rates "
          f"of 0.1 unique/s — and a ratio resting on fewer than {THIN_DENOMINATOR} rival "
          "solutions is named in the caveats, not screened.", "",
          "**What `rival failed` and `excluded` hold, per row:**", ""] + (excl_lines or ["* none"])
    M += ["",
          "The two protocols are DIFFERENT TASKS and must never be pooled: 'their pins' rows "
          "pin the 1–5 outputs the rival's own table names (rejection-easy by construction — "
          f"measured p_accept {pa_their}, printed per row), while miter/rare rows pin ALL outputs "
          f"to a rare reachable pattern (p_accept {pa_rare}). Recurrent-CircuitSAT is scored on "
          "its own depth-POOLED unique count — solutions satisfying the target at ANY unroll "
          "depth start_point..T, a superset of the fixed-depth task every other row measures "
          "(the fixed-depth-only count is recorded per row as `unique_at_T`). **The batch a rival "
          "is handed is its score**: measured, both are near-proportional to it "
          "over the whole range the card will run (DEMOTIC "
          "564x from 1k to 1M), so it is the same knob `K` is for us. Each rival is therefore "
          "granted the LARGEST rung of that ladder it can run to completion: the ladder is walked "
          "DOWN from the top and a rung is rejected only on feasibility — out of memory, refused "
          "by the driver, or not finished inside the wall budget — never on how fast it was, so "
          "exactly ONE configuration is measured per instance. It is a share-of-the-machine rule, "
          "not a search for the rival's fastest rung: unique/s stops tracking the batch once the "
          "model saturates the card, so a smaller rung CAN score higher, and because the walk "
          "stops at the first rung that completes the rival's per-rung optimum is neither sought "
          "nor reported — a cell where a shorter earlier budget had measured a smaller rung "
          "faster is disclosed in the campaign notes (RIVAL_FAIRNESS_FINDINGS §16c). "
          "(Until 2026-09-04 the walk started "
          "at our VRAM model's rung, which is calibrated at num_steps=10 while we run 1 and so "
          "under-granted DEMOTIC 1.5-3.9x on 4 of 11 ISCAS-85 cells, and a timeout was recorded "
          "AS the rival's result.) The first run of every process is discarded before anything is "
          "measured, because a cold CUDA context costs the session's first instance ~3.3x. "
          + variant_sentence + " The rungs tried are recorded per row as `batch_sweep`, "
          "the resolved budget as `vram_budget_bytes`. Their own defaults cannot serve as the rule — all three "
          "argparse defaults are `--batch_size 10` and their demo scripts are per-instance "
          "one-liners spanning 1,000 to 1,000,000 on comparably tiny circuits, which on our "
          "instances would hand them a small fraction of their achievable throughput and "
          "inflate the ratios below by the same factor. "
          "**Their `--num_steps` and `--lr` are theirs, not ours**: the batch is a share of the machine and "
          "must be granted, but the gradient-step count and the learning rate are tool settings "
          "with author-chosen values, so each rival runs at the ones its OWN artifacts specify — "
          "`num_steps` 1 and `lr` 15 for both (their `script/run_sampler_*.sh`). Their PAPERS "
          "state other values — DEMOTIC 10 iterations at `lr` 15, Recurrent 5 iterations at "
          "`lr` 50 — and the campaign runs the SHIPPED script configuration, one per tool, "
          "which is also the better of the two for each tool on this card at the depth measured "
          "(both saturate their batch on the first step; the paper values, singly and as the "
          "paired setting, are measured beside it in `rivals_steps_calibration*.json` so the "
          "text can say what they would have given the tool). Recorded per row as "
          "`steps`/`steps_source` and `lr`/`lr_source`. Nothing is swept, since a "
          "swept-and-best-of setting invites \"why stop there?\". Recurrent's `start_point` "
          "initialization search is given its own 1,200 s budget per instance, which BOUNDS the "
          "search and is in no published rate (this table carries no end-to-end column, see "
          "below); a cell that does not initialize inside the budget is reported with that "
          "budget named, never as a zero. "
          "Rows recorded on or before 2026-08-26 pinned 10 for both, which taxes DEMOTIC "
          "2.6x and Recurrent 8.5x (both saturate their batch on the first step) and is why the "
          "DEMOTIC and Recurrent rows here are far below their previously recorded values. "
          "Rival rows recorded before 2026-08-25 used a fixed 5 GiB budget and are not comparable "
          "to rows from the current harness. "
          "Constraint alignment is VERIFIED per row: miter mode asserts p(MITER=1) equals "
          "the instance's p_accept; native mode asserts bit-identical acceptance words between "
          "their index table on the original file and our reordered pins (`pin_check_ok`). "
          "**No tool is timed while delivering its solutions**: CMSGen runs under the patch's "
          "`--nowrite`, DEMOTIC counts and dedups its solutions after its own timer "
          "stops, and our timed run gathers no model row to the host (`collect=False`; the "
          "rival arm keeps EVERY delivered row as GPU-projected support bits and deduplicates "
          "them exactly, off the clock, the same quantity both rivals report — the dedup runs "
          "after the sampler returns and is timed as `distinct_s`, and the projection wall that "
          "gathered the rows is subtracted from the steady-state AND the end-to-end rate, so "
          "measuring distinctness cannot pay for itself out of the number it measures) "
          "— re-derived from the pinned sources by "
          "`eval/rivals/check_emission_symmetry.py`. The "
          "one exception is Recurrent-CircuitSAT, which copies its per-depth candidate "
          "assignments to the host inside its timed loop: the share of its `model_runtime` "
          "that costs is bounded per row by the audit script (section 4 of its report — quote "
          "the campaign's figure, it depends on the card's transfer rate) and left "
          "uncorrected so its rate is exactly the one it publishes. Deducting it would RAISE "
          "the ratios below, so those rows under-report GPAP rather than flattering it."]
    M += ["",
          "**One clock, and `ties`.** This table has NO end-to-end column, by rule (protocol "
          "v1.0, RIVAL_FAIRNESS_FINDINGS.md §17): the rivals' metric is unique solutions per "
          "second of their own `model_runtime`, which excludes their model build, and GPAP's "
          "figure excludes its construction the same way — that is the one comparison the "
          "rivals themselves define. An end-to-end ratio across an in-process sampler and a "
          "subprocess gradient tool had four definitions in three review rounds (0.8x, 7.6x, "
          "20.9x/132.0x and 11.8x/68.8x on the same rows), and on DEMOTIC's cells both tools "
          "finish the real work in under a second, so it measured process start-up; the "
          "end-to-end comparison is Table 1's, where both walls are the same kind of thing. "
          "`ties` counts cells whose win/loss sign lies inside GPAP's own seed band (the ratio "
          "at its slowest and fastest seed straddles 1.0, against a rival that ran once); "
          "`wins` excludes them, so wins + ties + losses = the win denominator."]
    cal = rival_calibration(args.rivals) if args.rivals else []
    if cal:
        M += [""] + cal
    soli = []
    if any(prop.values()):
        M += ["", "## Table 3 — propagation throughput (the engine, not the sampler)", "",
              "`impl/s` is Boolean implications per second: GPAP's from the packed-word kernel, "
              "CMSGen's from its own `--statsevery` counters. The last three columns factor the "
              "sample-rate result exactly — **engine advantage / work per sample = samples/s "
              "ratio** — because GPAP discards a conflicted trajectory (rejection sampling) "
              "where CMSGen backtracks and keeps the partial work. That identity is ALGEBRAIC, "
              "not a validation: defining work-per-sample as the quotient of the two rates "
              "makes it true by construction, so it can only fail if an aggregation mixes row "
              "sets — which is exactly the per-column-filtering error it is here to catch, and "
              "all it should be claimed to catch. The rows are Table 1's, under every one of "
              "its screens.", "",
              "| family | depth | targets | enc | n | GPAP impl/s | CMSGen impl/s "
              "| impl/s ratio | dec/s ratio | GPAP impl/sample | CMSGen impl/sample "
              "| work penalty |",
              "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for (stem, enc), label, depth, tgt in t1:
            v = prop.get((stem, enc))
            if not v:
                continue
            dec = f"{v['dec']:,.0f}x" if v["dec"] else "—"
            M.append(f"| {label} | {depth} | {tgt} | {enc or '—'} | {v['n']} | "
                     f"{v['gpap']:,.3g} | {v['cms']:,.3g} | {v['ratio']:,.0f}x | {dec} | "
                     # 3 significant figures, not 1 decimal: this table's caption is the identity
                     # `impl ratio / work penalty = samples/s ratio`, and it has to hand-check.
                     f"{v['gper']:,.0f} | {v['cper']:,.0f} | {v['work']:,.3g}x |")
        # The flatness claim covers the families with at least three instances — the same set
        # `propagation_summary.png` plots, so figure and table print one number.
        soli = [v for v in prop.values() if v and v["n"] >= 3]
        allr = gm([v["ratio"] for v in soli]) if soli else None
        lo, hi = ((min(v["gpap"] for v in soli), max(v["gpap"] for v in soli))
                  if soli else (None, None))
    if soli:
        M += ["", f"Across every family-encoder row with at least three instances (n={len(soli)}, the set "
              f"plotted in `propagation_summary.png`) GPAP holds **{gm([v['gpap'] for v in soli]):,.3g} "
              f"implications/s** with a spread of only {hi / lo:.2f}x between the fastest and "
              f"slowest, against CMSGen's {gm([v['cms'] for v in soli]):,.3g} — a "
              f"**{allr:,.0f}x** engine advantage that is essentially independent of the "
              "benchmark. The sample-rate ratios in Table 1 vary over nearly two orders of "
              "magnitude *on top of* this flat engine number, so that variation is a property of "
              "how much propagation each instance wastes on conflicted trajectories, not of the "
              "propagator."]
    if uni:
        sub = uni_args.get("_subcube_excluded") or 0
        M += ["", "## Table 4 — uniformity vs an exact oracle", "",
              f"`alpha={uni_args.get('alpha')}`, {uni_args.get('seeds')} seeds/instance, "
              f"n = {uni_args.get('n_mult')}x|S| samples."
              + (f" {sub} full-subcube instance(s) excluded — every sampler is uniform there "
                 f"by construction (the don't-care filler, not the config, draws the bits)."
                 if sub else ""), "",
              "| sampler | kind | instances | median TV | flagged SKEW | unsound |",
              "|---|---|---:|---:|---:|---:|"]
        for (name, kind), a in sorted(uni.items(), key=lambda kv: uni_sort_key(kv[0])):
            M.append(f"| `{name}` | {kind} | {a['n']} | "
                     f"{a['tv']:.4f} | {a['reject']}/{a['n']} | {a['unsound']} |")
    if barb:
        p = barb["params"]
        n_el = len(barb["eligible"])
        setting = {(0.3, 1.8, 0.1): "the parameters CMSGen's own authors certify at (FMCAD'21 "
                                    "Table I): 36 rounds x (1 + 1 + 50) = **1,872 samples** for "
                                    "an Accept, per-round accept band **[0.05, 0.95]**",
                   (0.1, 0.9, 0.05): "barbarik's own default: 477 rounds, **70,596 samples** for "
                                     "an Accept, per-round bands about +-0.21"
                   }.get((p["epsilon"], p["eta"], p["delta"]), "a non-standard setting")
        M += ["", "## Table 4b — Barbarik CM_test (the standard certificate)", "",
              f"`eps={p['epsilon']} eta={p['eta']} delta={p['delta']}` — {setting}. "
              f"An Accept means *not caught being eta-far* (L1 >= {p['eta']}, i.e. TV >= "
              f"{p['eta'] / 2:.2f}) and holds only under Barbarik's non-adversality assumption; "
              f"a Reject is unconditional. It is not a proof of uniformity — the exact-oracle "
              f"fuzzing campaign (docs/findings/UNIFORMITY_FINDINGS.md) is the sensitive "
              f"instrument.", "",
              f"{n_el} eligible instances "
              f"({len(barb['control_rejected'])} dropped because the trusted reference is itself "
              f"Rejected on them — a non-independent sampling set, or a delta-level false "
              f"rejection of an exactly uniform sampler; either way the test cannot speak about "
              f"the instance; {len(barb['control_errored'])} dropped on a reference error).", "",
              "**A separate campaign.** Source: "
              + ", ".join(f"`{os.path.relpath(p)}`" for p in barb.get("source") or [])
              + " — its own run, on its own host and its own instances (our pins on the "
                "circuit families), not the cells of Tables 1–3 above; see that campaign's "
                "README for the host. A CM_test verdict is a property of the sampler's "
                "distribution, not of the card, so the header's GPU line does not apply here.",
              "",
              "| sampler under test | Accept | Reject | no verdict | mean bias | max bias |",
              "|---|---:|---:|---:|---:|---:|"]
        for arm, a in barb["summary"].items():
            bm = "-" if a["bias_mean"] is None else f"{a['bias_mean']:.3f}"
            bx = "-" if a["bias_max"] is None else f"{a['bias_max']:.3f}"
            M.append(f"| `{arm}` | {a['accept']}/{a['n']} | {a['reject']} | {a['error']} | "
                     f"{bm} | {bx} |")
        M += ["", "`bias` is CM_test's own per-round bucket share (rejects outside "
                  "[0.05, 0.95]); it is reported because the verdict alone hides how much "
                  "headroom an Accept had. `no verdict` = the sampler could not fill an order "
                  "barbarik placed, which is neither an Accept nor a Reject."]
    soft_stats = None
    if soft:
        rows = [r for r in (soft.get("rows") or []) if not r.get("vacuous")]
        ratios = [r.get("ratio_new_over_base") for r in rows]
        won = [r for r in rows if r.get("soft_won_baseline")]
        # A "capability loss" is the safe portfolio finding NOTHING where soft found something:
        # a different failure from being slower, and the one a user would notice.
        lost = [r for r in rows
                if (r.get("baseline(+soft)") or {}).get("models")
                and not (r.get("new(no soft)") or {}).get("models")]
        soft_stats = {"n": len(rows), "gm": gm(ratios), "won": len(won), "lost": lost,
                      "vac": len(soft.get("rows") or []) - len(rows)}
        M += ["", "## Table 5 — the price of dropping `soft`", "",
              f"Throughput of the uniformity-safe portfolio relative to one that may pick "
              f"`soft`, over {soft_stats['n']} non-vacuous instances "
              f"({soft_stats['vac']} vacuous excluded): "
              f"**{soft_stats['gm']:.3f}x**" if soft_stats["gm"] else "(no ratios)",
              "",
              f"* `soft` won the bandit on **{soft_stats['won']}/{soft_stats['n']}** instances "
              f"— on the rest the two arms are the same configuration, so the geomean understates "
              f"the effect where it actually applies.",
              f"* capability losses (safe portfolio finds nothing, soft finds something): "
              f"**{len(lost)}**"
              + (": " + ", ".join(f"`{r['instance']}`" for r in lost) if lost else "")]
        if won:
            M += ["", "| instance | with soft (samples/s) | safe (samples/s) | ratio |",
                  "|---|---:|---:|---:|"]
            for r in sorted(won, key=lambda r: r.get("ratio_new_over_base") or 0)[:12]:
                b, n_ = r.get("baseline(+soft)") or {}, r.get("new(no soft)") or {}
                M.append(f"| `{r['instance']}` | {b.get('sps', 0):,.0f} | {n_.get('sps', 0):,.0f} "
                         f"| {r.get('ratio_new_over_base') or 0:.3f}x |")
    if rc:
        M += ["", "## Table 6 — reproducibility (fresh VM, same protocol)", "",
              f"* deterministic fields (M, support, packed split, propagate variant): "
              f"**{rc['exact']}/{rc['exact'] + rc['mismatch']} exact**",
              f"* success rate within 0.05: **{rc['succ_ok']}/{rc['succ_ok'] + rc['succ_bad']}**",
              f"* throughput recheck/original: geomean **{rc['gm']:.3f}x** "
              f"({rc['lo']:.2f}–{rc['hi']:.2f}, n={rc['n']})"]

    # ============================================================ latex
    T = ["% GENERATED by eval/artifact/paper_tables.py — do not edit by hand.",
         f"% {time.strftime('%Y-%m-%d %H:%M')}  generated at commit {commit}",
         f"% measured engine: {', '.join(cores) or 'not recorded'}",
         f"% GPU {prov.get('gpu')}  CPU {prov.get('cpu')}", "",
         r"\begin{table}[t]", r"  \centering", r"  \small",
         r"  \caption{\textbf{GPAP vs CMSGen.} " +
         f"GPU {tex_escape(prov.get('gpu'))}, CPU {tex_escape(prov.get('cpu') or 'n/a')}; "
         f"{tot['cells']} cells, {tot['skipped']} skipped, {tot['errors']} errors, "
         f"{tot['unsound']} unsound. "
         r"An instance on which \emph{either} sampler emitted nothing has no finite ratio, so it "
         r"is excluded from the geomean and counted in its own column: \emph{C$=$0} (CMSGen "
         r"silent --- an outright win for us, which makes the geomeans conservative), "
         r"\emph{G$=$0} (GPAP silent --- an outright loss). \emph{excl.} counts every other cell "
         r"that carries no ratio, on one principle --- a cell that measures the \emph{instance} "
         r"does not belong in a geomean about the \emph{sampler}: vacuous pins (constant "
         r"outputs, so pinning is a tautology), free supports (samples emitted with no search), "
         r"solution spaces smaller than the sample request, cells proved UNSAT at this depth, "
         r"cells neither sampler reached, and cells GPAP's encoder refused --- itemised by name "
         r"in the text. Every instance is counted exactly once: $\emph{inst} = n + "
         r"\text{C}{=}0 + \text{G}{=}0 + \text{excl.}$, and the win count is taken over the "
         r"same rows as the geomean plus the two zero columns. A sampler that delivered "
         r"anything is scored; a ratio resting on fewer than " + str(THIN_DENOMINATOR) +
         r" samples on a side is named in the text, not screened. \emph{e2e} charges both sides "
         r"their whole wall (GPAP parse-to-finish, CMSGen its whole subprocess; neither side's "
         r"wall holds delivery, since CMSGen writes nothing under \texttt{--nowrite}); "
         r"\emph{pre\%} is preparation's median share of GPAP's wall. "
         + (r"Each encoder is reported as its own row; nothing selects between them, since a "
            r"per-instance best-of keyed on the ratio would let CMSGen's number choose which "
            r"of our runs is published. " if show_enc else
            (f"GPAP samples the \\texttt{{{tex_escape(encs_t1[0])}}} CNF of every row. "
             if encs_t1 else ""))
         + r"$^{\dagger}$incomplete stage; $^{\ddagger}$geomean over fewer than three instances "
         r"(CMSGen scored zero on the rest), so it is an illustration, not an estimate.}",
         r"  \label{tab:cmsgen}",
         r"  \begin{tabular}{@{}lll" + ("l" if show_enc else "") + r"rrrrrrrrr@{}}",
         r"    \toprule",
         r"    family & depth & targets & " + ("enc & " if show_enc else "")
         + r"inst & $n$ & GPAP/CMSGen & e2e & pre\% & wins & C$=$0 & G$=$0 & excl. \\",
         r"    \midrule"]
    for (stem, enc), label, depth, tgt in t1:
        v = cms[(stem, enc)]
        exp = expected.get(stem.split("-")[0], v["inst"])
        mark = "" if v["inst"] >= exp else r"$^{\dagger}$"
        gtex = f"${v['gm']:,.1f}\\times$" if v["gm"] else "---"
        if v["gm"] and v["n_gm"] < 3:
            gtex += "$^{\\ddagger}$"
        e2tex = f"${v['gm_e2e']:,.1f}\\times$" if v.get("gm_e2e") else "---"
        T.append(f"    {tex_escape(label)} & ${depth}$ & {tgt} & "
                 + (f"{tex_escape(enc or '---')} & " if show_enc else "")
                 + f"{v['inst']}{mark} & {v['n_gm']} & {gtex} & {e2tex} & "
                 f"{100 * v.get('pre_frac', 0):.0f} & {v['wins']}/{v['n']} & {v['inf']} & "
                 f"{v['gpap0']} & {v.get('excl', 0)} \\\\")
    T += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    # the itemised exclusions travel with the .tex as comments, beside the table they explain
    for (stem, enc), label, depth, tgt in t1:
        items = excl_items(cms[(stem, enc)].get("excl_names") or {}, EXCL_REASONS_T1)
        if items:
            T.append(f"% excl. — {label}, {depth}, {tgt}: {items.replace('`', '')}")
    T.append("")

    T += [r"\begin{table}[t]", r"  \centering", r"  \small",
          r"  \caption{\textbf{GPAP vs the GPU gradient samplers}, their metric "
          r"(unique solutions / their \texttt{model\_runtime}, which excludes their model build; "
          r"\emph{GPAP/rival} uses our steady-state rate, which likewise excludes construction "
          r"--- the one clock this table has, since it is the one the rivals define; the "
          r"end-to-end comparison is Table~\ref{tab:cmsgen}'s). Each row is the rival's whole "
          r"published set under one task. \emph{ties} counts cells whose sign lies inside "
          r"GPAP's own seed band and \emph{wins} excludes them. As in Table~\ref{tab:cmsgen}, "
          r"a cell where either side emitted nothing carries no finite ratio and is counted "
          r"separately: \emph{R$=$0} (rival silent), \emph{G$=$0} (GPAP silent). \emph{failed} "
          r"counts rival runs that produced no measurement inside the budget (did not "
          r"initialize, timed out, no gradient path) --- counted, never folded in as zeros. "
          r"\emph{excl.} counts every other cell without a ratio --- vacuous pins, free "
          r"supports, tiny solution spaces, cells proved UNSAT at this depth, cells neither "
          r"tool reached, cells GPAP's encoder refused --- itemised by name in the text. A "
          r"rival that scored anything is scored (its own tables print rates of 0.1 unique/s); "
          r"a ratio resting on fewer than " + str(THIN_DENOMINATOR) + r" rival solutions is "
          r"named in the text, not screened. Every cell is counted exactly once: "
          r"$\emph{cells} = n + \text{R}{=}0 + \text{G}{=}0 + \text{failed} + \text{excl.}$ "
          r"$^{\ddagger}$geomean over fewer than three instances.}",
          r"  \label{tab:rivals}",
          r"  \begin{tabular}{@{}llrrrrrrrrr@{}}", r"    \toprule",
          r"    rival & task & cells & $n$ & GPAP/rival & wins & ties & R$=$0 & G$=$0 & "
          r"failed & excl. \\",
          r"    \midrule"]
    for stem, _, rival, proto in RIVALS:
        v = riv.get(stem)
        if not v:
            T.append(f"    {rival} & {tex_escape(proto)} & \\multicolumn{{9}}{{c}}{{not run}} \\\\")
            continue
        g = f"${v['gm']:,.1f}\\times$" if v["gm"] else "---"
        n_gm_ok = v["n"] - v["zero"] - v["gpap0"]        # cells carrying a finite ratio
        if v["gm"] and n_gm_ok < 3:
            g += "$^{\\ddagger}$"
        T.append(f"    {rival} & {tex_escape(proto)} & {v['cells']} & {n_gm_ok} & {g} & "
                 f"{v['wins']}/{v['n']} & {v.get('ties', 0)} & {v['zero']} & {v['gpap0']} & "
                 f"{v['errors']} & {v.get('excl', 0)} \\\\")
    T += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    for stem, _, rival, proto in RIVALS:
        v = riv.get(stem)
        if not v:
            continue
        parts = ([("failed " + ", ".join(v["error_names"]))] if v.get("error_names") else [])
        items = excl_items(v.get("excl_names") or {}, EXCL_REASONS_T2)
        if items:
            parts.append(items.replace("`", ""))
        if parts:
            T.append(f"% failed / excl. — {rival}, {proto}: " + "; ".join(parts))
    T.append("")

    if any(prop.values()):
        T += [r"\begin{table}[t]", r"  \centering", r"  \small",
              r"  \caption{\textbf{Propagation throughput --- the engine, isolated from the "
              r"sampler.} Boolean implications per second: GPAP's from the packed-word kernel, "
              r"CMSGen's from its own counters (\texttt{--statsevery}, "
              r"\texttt{scripts/cmsgen\_gpap\_eval.patch}). The identity "
              r"$\text{samples/s ratio} = \text{impl/s ratio} \,/\, \text{work-per-sample ratio}$ "
              r"holds on every family. It is an ALGEBRAIC identity, not a validation --- "
              r"defining work-per-sample as the quotient of the two rates makes it true by "
              r"construction, so it can only fail if an aggregation mixes row sets, which is "
              r"exactly the per-column-filtering error it is here to catch. What it does is "
              r"factor the sample-rate result into what the "
              r"GPU propagator buys and what rejection sampling spends: GPAP discards a "
              r"conflicted trajectory outright, whereas CMSGen backtracks and retains the partial "
              r"work. GPAP's implication rate is nearly constant across families; the "
              r"three-orders-of-magnitude spread in Table~\ref{tab:cmsgen} therefore comes from "
              r"the work-per-sample column, not from the propagator. $n$ counts instances where "
              r"\emph{both} samplers emitted, since a work-per-sample figure needs both; "
              r"$^{\ddagger}$marks rows where that leaves fewer than three.}",
              r"  \label{tab:prop}",
              r"  \begin{tabular}{@{}llllrrrrrrrr@{}}", r"    \toprule",
              r"    & & & & & \multicolumn{2}{c}{implications/s} & "
              r"\multicolumn{2}{c}{implications/sample} & & & \\",
              r"    \cmidrule(lr){6-7}\cmidrule(lr){8-9}",
              r"    family & depth & targets & enc & $n$ & GPAP & CMSGen & GPAP & CMSGen & "
              r"impl ratio & dec/s ratio & work penalty \\", r"    \midrule"]
        for (stem, enc), label, depth, tgt in t1:
            v = prop.get((stem, enc))
            if not v:
                continue
            # same low-n flag as Table 1: `n` of 1-2 illustrates, it does not estimate
            nmark = "$^{\\ddagger}$" if v["n"] < 3 else ""
            dectex = f"${v['dec']:,.0f}\\times$" if v.get("dec") else "---"
            T.append(f"    {tex_escape(label)} & ${depth}$ & {tgt} & "
                     f"{tex_escape(enc or '---')} & {v['n']}{nmark} & "
                     f"${v['gpap'] / 1e9:.2f}\\!\\times\\!10^{{9}}$ & "
                     f"${v['cms'] / 1e6:.2f}\\!\\times\\!10^{{6}}$ & "
                     f"{v['gper']:,.0f} & {v['cper']:,.0f} & "
                     f"${v['ratio']:,.0f}\\times$ & {dectex} & "
                     f"${v['work']:,.3g}\\times$ \\\\")
        T += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    if uni:
        T += [r"\begin{table}[t]", r"  \centering", r"  \small",
              r"  \caption{\textbf{Uniformity against an exact oracle.} Small instances whose "
              r"entire solution set is enumerated by brute force, so the target distribution is "
              r"known; $\chi^2$ GOF over independent seeds combined by Fisher's method. "
              r"\texttt{ctl:uniform} is an exact-uniform negative control and \texttt{ctl:tv*} a "
              r"positive control at known TV --- they calibrate the instrument.}",
              r"  \label{tab:uniformity}",
              r"  \begin{tabular}{@{}llrrrr@{}}", r"    \toprule",
              r"    sampler & kind & inst & median TV & flagged skewed & unsound \\",
              r"    \midrule"]
        for (name, kind), a in sorted(uni.items(), key=lambda kv: uni_sort_key(kv[0])):
            T.append(f"    \\texttt{{{tex_escape(name)}}} & {kind} & {a['n']} & "
                     f"${a['tv']:.4f}$ & {a['reject']}/{a['n']} & {a['unsound']} \\\\")
        T += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    if rc:
        T += [r"\begin{table}[t]", r"  \centering", r"  \small",
              r"  \caption{\textbf{Reproducibility.} The same cells re-measured on a fresh VM of "
              r"the same type, protocol pinned. Deterministic fields must match exactly; "
              r"throughput is reported, not asserted.}",
              r"  \label{tab:repro}",
              r"  \begin{tabular}{@{}lr@{}}", r"    \toprule",
              r"    quantity & result \\", r"    \midrule",
              f"    deterministic fields exact & {rc['exact']}/{rc['exact'] + rc['mismatch']}"
              r" \\",
              f"    success rate within $0.05$ & {rc['succ_ok']}/"
              f"{rc['succ_ok'] + rc['succ_bad']}" + r" \\",
              f"    throughput (geomean) & ${rc['gm']:.3f}\\times$ "
              f"(${rc['lo']:.2f}$--${rc['hi']:.2f}$, $n={rc['n']}$)" + r" \\",
              r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    # ---------------------------------------------------------------- caveats
    # These travel WITH the numbers, and each is derived from the data rather than remembered.
    C = []
    for (stem, enc), v in cms.items():
        if not v:
            continue
        stem = f"{stem}/{enc}" if enc else stem
        exp = expected.get(stem.split("-")[0], v["inst"])
        if v["inst"] < exp:
            C.append(f"`{stem}` is an **incomplete stage** ({v['inst']}/{exp} instances) — the "
                     f"rented-GPU budget ran out. Do not read it as a family result.")
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
            C.append(f"{rival} ({proto}): **{v['errors']} of {v['cells']}** runs produced no "
                     f"measurement inside the budget (did not initialize within {budget}, "
                     f"timed out, or no gradient path — the reason is recorded per row). They "
                     f"are omitted from the geomean and never counted as zeros. This is a "
                     f"statement about OUR budget, not about the rival's capability: a cell "
                     f"that did not initialize in time is re-runnable at a longer one; which "
                     f"cells did when the budget was raised, and which still did not, is "
                     f"recorded per campaign (docs/findings/RIVAL_FAIRNESS_FINDINGS.md §11, "
                     f"§16d).")
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
    M += ["", "## Caveats that MUST appear in the text", ""] + [f"* {c}" for c in C]

    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
        open(args.out_md, "w").write("\n".join(M) + "\n")
        print(f"wrote {args.out_md}")
    if args.out_tex:
        os.makedirs(os.path.dirname(args.out_tex) or ".", exist_ok=True)
        open(args.out_tex, "w").write("\n".join(T) + "\n")
        print(f"wrote {args.out_tex}")
    print(f"\n{tot['cells']} cells | {tot['skipped']} skipped | {tot['errors']} errors | "
          f"{tot['unsound']} unsound")
    if uni:
        print(f"uniformity: {len(uni)} (sampler, kind) groups")
    if rc:
        print(f"repro: {rc['exact']}/{rc['exact'] + rc['mismatch']} exact, geomean {rc['gm']:.3f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmsgen", default=None,
                    help="`gpap bench` stage directory — the older family matrix, supplementary")
    ap.add_argument("--cmsgen-ref", default=None)
    ap.add_argument("--rivals", default=None,
                    help="the campaign directory (rivals_*.json): Tables 1, 2 and 3 of the "
                         "converged protocol are read off these rows")
    ap.add_argument("--recheck", default=None)
    ap.add_argument("--uniformity", nargs="*", default=[])
    ap.add_argument("--barbarik", nargs="*", default=[],
                    help="barbarik_gpap.py JSONL result files (Table 4b)")
    ap.add_argument("--soft", default=None)
    ap.add_argument("--out-tex", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()
    if not (args.cmsgen or args.rivals):
        ap.error("nothing to read: pass --rivals (the campaign) and/or --cmsgen (supplementary)")
    build(args)


if __name__ == "__main__":
    main()
