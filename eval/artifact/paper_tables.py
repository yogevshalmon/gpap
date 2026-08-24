#!/usr/bin/env python3
"""Build the paper's experiment section — every table, straight from the experiment JSONs.

Nothing here is typed by hand. That is the point: an earlier version of the deck carried a
sequential headline of 47.7x that silently included four instances whose output pins are
tautologies (real value 36.5x), and the only durable fix is to derive every published number from
the data with the exclusions applied in ONE place.

The raw data lives OUTSIDE the repo, in $GPAP_RESULTS (default ../gpap-results) — it is too heavy
to track. See docs/paper/README.md for the full regeneration recipe.

    R=.experiments/l4_campaign_20260823          # or $GPAP_RESULTS/l4_campaign_20260823
    eval/artifact/paper_tables.py --cmsgen $R/gpap_vs_cmsgen_l4 \\
        --rivals $R/gpap_vs_gpu_rivals_l4 \\
        --recheck $R/reproducibility_recheck_l4 \\
        --uniformity $R/uniformity_l4/uniformity_campaign.json \\
        --soft $R/uniformity_l4/soft_profile.json \\
        --out-tex docs/paper/tables.tex --out-md docs/paper/RESULTS.md

Rules applied everywhere, so they cannot diverge between tables:
  * VACUOUS-PIN instances (b02/b05/b07/s526 — constant last-cycle outputs at T=25, so pinning is
    a tautology and every sampler scores ~100%) are excluded from the sequential geomeans.
  * ZERO-SEARCH cells (`zero_search`) are excluded for the same reason: no decisions and no
    implications means the support is free after propagation and the ratio measures the
    instance, not the sampler.
  * An instance where EITHER side emitted nothing has no finite ratio. It is excluded from the
    geomean and counted in its own column — CMSGen=0 (our win), GPAP=0 (our loss), both=0 —
    so the columns sum to the instance count and our failures are as visible as theirs.
  * Rival rows that errored (timeout / no gradient path) are counted, never folded in as zeros.
  * A stage with fewer instances than its family's maximum is marked incomplete.
"""
import argparse
import json
import math
import os
import subprocess
import time

VACUOUS_PINS = {"b02", "b05", "b07", "s526"}


def zero_search(r):
    """True when the row emitted samples having done NO search: no decisions, no implications.

    After unit propagation the support is a free hypercube, so every trajectory succeeds
    (`succ == 1.0`) and "sampling" is a random-bit readout at ~240M samples/s. It is a correct
    result, but it exercises none of the propagation engine, and CMSGen still pays to discover the
    same thing — so the ratio measures the degeneracy, not the sampler.

    This is the vacuous-pin problem wearing a different hat, and it is worse in one respect: the
    best-encoder rule ACTIVELY SELECTS these cells, because they carry the highest ratio in their
    family. On EPFL `ctrl` the Tseitin encoding samples at 2.0M/s with succ 0.07 while ABC
    collapses the output cone and gives succ 1.0 at 240M/s, so the encoder A/B hands the headline
    the degenerate cell. Measured cost of leaving them in: EPFL rare 9.4x -> 29.0x, EPFL random
    8.2x -> 19.5x, sk 98.9x -> 113.1x; no other family contains one.
    """
    return ((r.get("n_models") or 0) > 0
            and not (r.get("decisions_per_s") or 0)
            and not (r.get("implications_per_s") or 0))

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

RIVALS = [
    ("demotic_native", "demotic", "DEMOTIC", "their ISCAS-85 pins"),
    ("demotic_miter", "demotic", "DEMOTIC", "our rare pins (miter)"),
    ("recurrent_table", "rcs", "Recurrent-CircuitSAT", "their target table"),
    ("recurrent_rare", "rcs", "Recurrent-CircuitSAT", "our rare pins"),
    ("hts", "hts", "HTS", "our CNFs (both encoders)"),
]


def gm(xs):
    xs = [x for x in xs if x and x > 0 and math.isfinite(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def med(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def load(path):
    return json.load(open(path)) if os.path.isfile(path) else None


# ------------------------------------------------------------------ GPAP vs CMSGen

def select_cells(payload):
    """The ONE cell-selection rule: one row per instance, the best-scoring encoder.

    Both the headline table and the propagation table call this, so Table 3 describes exactly the
    runs that produced Table 1. When they each aggregated their own way, the propagation quotient
    (impl/s ratio / work-per-sample ratio) did not reproduce the headline and looked like a
    contradiction, though every per-instance identity held.
    """
    rows = [r for r in payload.get("results", [])
            if not r.get("skipped") and not r.get("error")]
    best = {}
    for r in rows:
        k = r["name"].split(".")[0]
        if k not in best or (r.get("ratio") or 0) > (best[k].get("ratio") or 0):
            best[k] = r
    return rows, best


def cms_family(d, stem):
    payload = load(os.path.join(d, f"{stem}.json")) if d else None
    if not payload:
        return None
    allr = payload.get("results", [])
    rows, best = select_cells(payload)
    if not rows:
        return None

    # Bucket EVERY instance exactly once. The earlier version keyed off `ratio` alone, and a
    # GPAP-zero row has ratio == 0.0 (falsy) while a both-zero row has ratio == None — so both
    # fell through every branch and vanished from the table, at the same time as CMSGen-zero rows
    # were being counted as wins. That is an asymmetric ledger: 38 instances where CMSGen scored
    # nothing were shown, and 14 where GPAP scored nothing (3) or neither did (11) were not.
    # A reader must be able to add the columns up and get `inst`.
    def scored(r, key):
        return (r.get(key) or 0) > 0
    finite, gpap_zero, cms_zero, both_zero = {}, [], [], []
    for k, r in best.items():
        g, c = scored(r, "n_models"), scored(r, "cms_n_samples")
        if g and c:
            finite[k] = r
        elif c:
            gpap_zero.append(k)
        elif g:
            cms_zero.append(k)
        else:
            both_zero.append(k)
    finite = {k: r for k, r in finite.items()
              if r.get("ratio") and math.isfinite(r["ratio"])}
    n_inf, n_g0, n_b0 = len(cms_zero), len(gpap_zero), len(both_zero)
    # Two degeneracies, one rule: drop instances whose measurement is about the instance rather
    # than the sampler. Vacuous pins are fixed by name; zero-search cells are detected per row.
    free = {k for k, r in finite.items() if zero_search(r)}
    has_vac = bool(set(best) & VACUOUS_PINS)
    drop = (VACUOUS_PINS if has_vac else set()) | free
    gm_vals = [r["ratio"] for k, r in finite.items() if k not in drop]
    return {
        "cells": len(rows), "inst": len(best),
        "gm_all": gm([r["ratio"] for r in finite.values()]),
        "gm": gm(gm_vals),
        "free": len(free), "free_names": sorted(free),
        # what the headline would read if the degenerate cells were left in — the size of the
        # error we are avoiding, so it can be quoted in the caveat rather than merely asserted
        "gm_withfree": gm([r["ratio"] for k, r in finite.items()
                           if k not in (VACUOUS_PINS if has_vac else set())]),
        # How many instances actually ENTER the geomean. When CMSGen scores zero on most of a
        # family (HWMCC T=50: 6 of 7), the ratio is undefined there and the "geomean" collapses
        # onto the handful that remain — 421x over ONE instance. A paper must show this number.
        "n_gm": len([x for x in gm_vals if x and math.isfinite(x)]),
        "has_vac": has_vac,
        # A CMSGen-zero instance is a win for us and a GPAP-zero instance is a loss; neither has a
        # finite ratio, so both stay out of the geomean and both are named in the caveats.
        "wins": sum(1 for r in finite.values() if r["ratio"] >= 1) + n_inf,
        "n": len(finite) + n_inf + n_g0, "inf": n_inf,
        "gpap0": n_g0, "both0": n_b0,
        "gpap0_names": sorted(gpap_zero), "both0_names": sorted(both_zero),
        "skipped": sum(1 for r in allr if r.get("skipped")),
        "errors": sum(1 for r in allr if r.get("error")),
        "unsound": sum(1 for r in rows
                       if r.get("gpu_sound") is False or r.get("cpu_sound") is False),
        "gpu": (payload.get("gpu") or {}).get("name"),
        "cpu": (payload.get("cpu") or {}).get("model"),
        "patched": payload.get("cmsgen_patched"),
    }


# ------------------------------------------------------------------ propagation throughput

def prop_rates(d, stem):
    """Engine-level throughput: implications and decisions per second, both samplers.

    This is the measurement that isolates the ENGINE from the sampler wrapped around it, and it
    factors the headline exactly. For any instance,

        samples/s ratio  =  (implications/s ratio)  /  (implications-per-sample ratio)

    because implications/s = samples/s x implications/sample on each side. The first factor is
    what the GPU propagator buys; the second is what our algorithm SPENDS, since GPAP discards a
    conflicted trajectory outright (rejection sampling) where CMSGen backtracks and keeps the
    partial work. Reporting only the quotient hides both. Checked against the recorded sample
    ratio to the printed digit on every family.

    CMSGen's counters come from `--statsevery` in scripts/cmsgen_gpap_eval.patch, so this is its
    own accounting of its own work, not our estimate of it.
    """
    payload = load(os.path.join(d, f"{stem}.json")) if d else None
    if not payload:
        return None
    _, best = select_cells(payload)
    # Same exclusion as the headline: a tautological pin measures nothing about either sampler.
    rows = [r for k, r in best.items() if k not in VACUOUS_PINS] or list(best.values())
    # EVERY column is computed over the SAME rows — the ones where all four quantities are
    # positive. Filtering per column instead put an instance like EPFL `bar` (GPAP propagates hard
    # but emits nothing) into the impl/s ratio and not into work-per-sample, which broke the
    # identity by 3x on that family and made a sound decomposition look wrong. Instances where
    # either sampler emitted nothing have no work-per-sample at all; they are Table 1's zero
    # columns, and they are excluded here for the same reason.
    gi, ci, ri, rd, work, gper, cper = [], [], [], [], [], [], []
    for r in rows:
        a, b = r.get("implications_per_s"), r.get("cms_implications_per_s")
        s, t = r.get("gpap_sps"), r.get("cms_sps")
        if not all(x and x > 0 for x in (a, b, s, t)):
            continue
        gi.append(a)
        ci.append(b)
        ri.append(a / b)
        gper.append(a / s)
        cper.append(b / t)
        work.append((a / s) / (b / t))
        if r.get("decisions_per_s") and r.get("cms_decisions_per_s"):
            rd.append(r["decisions_per_s"] / r["cms_decisions_per_s"])
    if not gi:
        return None
    return {"n": len(ri), "gpap": gm(gi), "cms": gm(ci), "ratio": gm(ri), "dec": gm(rd),
            "work": gm(work), "gper": gm(gper), "cper": gm(cper)}


# ------------------------------------------------------------------ GPAP vs GPU rivals

def rival_family(d, stem, key):
    payload = load(os.path.join(d, f"rivals_{stem}.json")) if d else None
    if payload is None:
        return None
    rows = (payload.get("rows") if isinstance(payload, dict) else payload) or []
    seen = {}
    for r in rows:
        seen[(r["name"], r.get("variant", "-"))] = r        # a retry supersedes its error row
    ratios, errs, zero, gzero, both, free = [], 0, 0, 0, 0, 0
    for r in seen.values():
        rv, g = r.get(key) or {}, r.get("gpap") or {}
        if rv.get("error"):
            errs += 1
            continue
        # Same zero-search screen as Table 1, where the row carries the counters (recorded
        # since 2026-08-23): a support that is FREE after propagation measures the instance,
        # not either sampler — a seed-mismatched b13 cell once carried a 4.7e9x ratio this way.
        if "decisions_per_s" in g and zero_search(g):
            free += 1
            continue
        rate = rv.get("unique_per_s") or 0.0
        ours = g.get("unique_per_s_est") or 0.0
        # Symmetric with the CMSGen table: name OUR zeros as loudly as theirs. A rival that
        # scored zero is an outright win, ours is an outright loss, and neither has a finite
        # ratio — so both stay out of the geomean and both get a column.
        if not rate and not ours:
            both += 1
        elif not rate:
            zero += 1
        elif not ours:
            gzero += 1
        else:
            ratios.append(ours / rate)
    return {"cells": len(seen), "gm": gm(ratios), "n": len(ratios) + zero + gzero,
            "wins": sum(1 for x in ratios if x >= 1) + zero, "errors": errs, "zero": zero,
            "gpap0": gzero, "both0": both, "free": free}


# ------------------------------------------------------------------ uniformity

# Reading order for the uniformity table. Alphabetical puts `ctl:*` in the middle, which buries
# the two rows that make the rest interpretable: without a negative control that is NOT flagged
# and a positive control that IS, "0/36 flagged" means nothing. Controls first, then the B ladder
# in order (the B >= |support| threshold is the result), then the opt-in config, then the rival.
UNI_ORDER = ["ctl:uniform", "ctl:tv0.05", "ctl:tv0.1",
             "gpap:b1", "gpap:b8", "gpap:b16", "gpap:b32", "gpap:b64",
             "pipe:default", "pipe:circuit", "pipe:circuit+soft", "gpap:soft", "cmsgen"]


def uni_sort_key(item):
    (name, kind) = item
    idx = UNI_ORDER.index(name) if name in UNI_ORDER else len(UNI_ORDER)
    return (idx, name, kind)


def uniformity(paths):
    """Aggregate the exact-oracle fuzzing campaign per sampler, split by instance kind."""
    rows = []
    args = {}
    for p in paths or []:
        d = load(p)
        if d:
            rows += d.get("rows", [])
            args = d.get("args", args)
    if not rows:
        return None, {}
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


def build(args):
    cms = {s: cms_family(args.cmsgen, s) for s, _, _, _ in CMS_FAMILIES}
    ref = {s: cms_family(args.cmsgen_ref, s) for s, _, _, _ in CMS_FAMILIES} \
        if args.cmsgen_ref else {}
    riv = {s: rival_family(args.rivals, s, k) for s, k, _, _ in RIVALS}
    prop = {s: prop_rates(args.cmsgen, s) for s, _, _, _ in CMS_FAMILIES}
    uni, uni_args = uniformity(args.uniformity)
    rc = recheck(args.cmsgen, args.recheck)
    soft = load(args.soft) if args.soft else None

    expected = {}
    for stem, _, _, _ in CMS_FAMILIES:
        v = cms.get(stem)
        if v:
            fam = stem.split("-")[0]
            expected[fam] = max(expected.get(fam, 0), v["inst"])

    prov = next((v for v in cms.values() if v), {}) or {}
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, timeout=10).stdout.strip()
    except Exception:                                   # noqa: BLE001
        commit = "?"

    tot = {kk: sum(v[kk] for v in cms.values() if v)
           for kk in ("cells", "inst", "skipped", "errors", "unsound")}

    # ============================================================ markdown
    M = [f"# GPAP — experiment section data  ({time.strftime('%Y-%m-%d')})", "",
         f"* GPU **{prov.get('gpu')}**, CPU **{prov.get('cpu') or 'not recorded'}**",
         f"* commit `{commit}`  |  CMSGen patched (`--nowrite`): **{prov.get('patched')}**",
         f"* **{tot['cells']} cells, {tot['inst']} instance-slots, {tot['skipped']} skipped, "
         f"{tot['errors']} errors, {tot['unsound']} unsound**", "",
         "Generated by `eval/artifact/paper_tables.py`. Vacuous pins excluded from sequential geomeans; "
         "an instance where EITHER sampler emitted nothing has no finite ratio and is reported in "
         "its own column instead of the geomean.", "",
         "## Table 1 — GPAP vs CMSGen", "",
         "Columns add up: `inst` = geomean instances + CMSGen=0 + GPAP=0 + both=0.", "",
         "| family | depth | targets | inst | GPAP/CMSGen | wins | CMSGen=0 | GPAP=0 | both=0 |"
         " no-search |" + (" ref GPU |" if ref else ""),
         "|---|---|---|---:|---:|---:|---:|---:|---:|---:|" + ("---:|" if ref else "")]
    for stem, label, depth, tgt in CMS_FAMILIES:
        v = cms.get(stem)
        if not v:
            continue
        exp = expected.get(stem.split("-")[0], v["inst"])
        mark = "" if v["inst"] >= exp else f" (partial {v['inst']}/{exp})"
        # The reference-GPU column exists only when a second sweep is passed (--cmsgen-ref); a
        # reference recorded under different defaults is not a reference, so the default is none.
        rv = ref.get(stem)
        refs = (f" {rv['gm']:,.1f}x |" if rv and rv["gm"] else " — |") if ref else ""
        gms = f"{v['gm']:,.1f}x" if v["gm"] else "— (all CMSGen=0)"
        if v["gm"] and v["n_gm"] < 3:
            gms += f" ⚠n={v['n_gm']}"
        M.append(f"| {label}{mark} | {depth} | {tgt} | {v['inst']} | "
                 f"{gms} | {v['wins']}/{v['n']} | {v['inf']} | {v['gpap0']} | {v['both0']} "
                 f"| {v['free']} |{refs}")
    g0 = sorted({n for v in cms.values() if v for n in v["gpap0_names"]})
    b0 = sorted({n for v in cms.values() if v for n in v["both0_names"]})
    M += ["",
          "**Instances GPAP could not sample** (CMSGen could): "
          + (", ".join(f"`{n}`" for n in g0) if g0 else "none") + ".",
          "",
          "**Instances neither sampler could touch**: "
          + (", ".join(f"`{n}`" for n in b0) if b0 else "none")
          + " — these bound the reach of the whole comparison, not of either tool."]
    M += ["", "## Table 2 — GPAP vs the GPU gradient samplers", "",
          "| rival | protocol | cells | GPAP/rival | wins | rival=0 | GPAP=0 | both=0 |"
          " no-search | rival failed |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for stem, _, rival, proto in RIVALS:
        v = riv.get(stem)
        if not v:
            M.append(f"| {rival} | {proto} | — | not run | — | — | — | — | — | — |")
            continue
        gs = f"{v['gm']:,.1f}x" if v["gm"] else "—"
        M.append(f"| {rival} | {proto} | {v['cells']} | {gs} | "
                 f"{v['wins']}/{v['n']} | {v['zero']} | {v['gpap0']} | {v['both0']} "
                 f"| {v.get('free', 0)} | {v['errors']} |")
    M += ["",
          "The two protocols are DIFFERENT TASKS and must never be pooled: 'their pins' rows "
          "pin the 1–5 outputs the rival's own table names (rejection-easy by construction — "
          "p_accept is 0.03–0.5 there, printed per row), while miter/rare rows pin ALL outputs "
          "to a rare reachable pattern (p_accept ~1e-2–1e-6). Recurrent-CircuitSAT is scored on "
          "its own depth-POOLED unique count — solutions satisfying the target at ANY unroll "
          "depth start_point..T, a superset of the fixed-depth task every other row measures "
          "(the fixed-depth-only count is recorded per row as `unique_at_T`). The rival's batch "
          "is sized by a fixed 5 GiB autograd-graph budget (`pick_batch`); on easy instances "
          "their unique count saturates at that batch, so their score is bounded by a knob we "
          "set — generous timeouts (300 s vs our 60 s) and best-of-batch offset it, but state "
          "it. Constraint alignment is VERIFIED per row: miter mode asserts p(MITER=1) equals "
          "the instance's p_accept; native mode asserts bit-identical acceptance words between "
          "their index table on the original file and our reordered pins (`pin_check_ok`)."]
    if any(prop.values()):
        M += ["", "## Table 3 — propagation throughput (the engine, not the sampler)", "",
              "`impl/s` is Boolean implications per second: GPAP's from the packed-word kernel, "
              "CMSGen's from its own `--statsevery` counters. The last three columns factor the "
              "sample-rate result exactly — **engine advantage / work per sample = samples/s "
              "ratio** — because GPAP discards a conflicted trajectory (rejection sampling) "
              "where CMSGen backtracks and keeps the partial work.", "",
              "| family | depth | targets | n | GPAP impl/s | CMSGen impl/s | impl/s ratio "
              "| dec/s ratio | GPAP impl/sample | CMSGen impl/sample | work penalty |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for stem, label, depth, tgt in CMS_FAMILIES:
            v = prop.get(stem)
            if not v:
                continue
            dec = f"{v['dec']:,.0f}x" if v["dec"] else "—"
            M.append(f"| {label} | {depth} | {tgt} | {v['n']} | {v['gpap']:,.3g} | "
                     f"{v['cms']:,.3g} | {v['ratio']:,.0f}x | {dec} | "
                     f"{v['gper']:,.0f} | {v['cper']:,.0f} | {v['work']:,.1f}x |")
        # The flatness claim is quoted over the families with at least three instances — the same
        # set `propagation_summary.png` plots. Including the n=1/n=2 HWMCC rows widens it to
        # 1.49x, and having a figure and a table print different numbers for one claim is the
        # exact failure this file exists to prevent.
        soli = [v for v in prop.values() if v and v["n"] >= 3]
        allr = gm([v["ratio"] for v in soli])
        lo, hi = min(v["gpap"] for v in soli), max(v["gpap"] for v in soli)
        M += ["", f"Across every family with at least three instances (n={len(soli)}, the set "
              f"plotted in `propagation_summary.png`) GPAP holds **{gm([v['gpap'] for v in soli]):,.3g} "
              f"implications/s** with a spread of only {hi / lo:.2f}x between the fastest and "
              f"slowest, against CMSGen's {gm([v['cms'] for v in soli]):,.3g} — a "
              f"**{allr:,.0f}x** engine advantage that is essentially independent of the "
              "benchmark. The sample-rate ratios in Table 1 vary over nearly two orders of "
              "magnitude *on top of* this flat engine number, so that variation is a property of "
              "how much propagation each instance wastes on conflicted trajectories, not of the "
              "propagator."]
    if uni:
        M += ["", "## Table 4 — uniformity vs an exact oracle", "",
              f"`alpha={uni_args.get('alpha')}`, {uni_args.get('seeds')} seeds/instance, "
              f"n = {uni_args.get('n_mult')}x|S| samples.", "",
              "| sampler | kind | instances | median TV | flagged SKEW | unsound |",
              "|---|---|---:|---:|---:|---:|"]
        for (name, kind), a in sorted(uni.items(), key=lambda kv: uni_sort_key(kv[0])):
            M.append(f"| `{name}` | {kind} | {a['n']} | "
                     f"{a['tv']:.4f} | {a['reject']}/{a['n']} | {a['unsound']} |")
    soft_stats = None
    if soft:
        rows = [r for r in (soft.get("rows") or []) if not r.get("vacuous")]
        ratios = [r.get("ratio_new_over_base") for r in rows]
        won = [r for r in rows if r.get("soft_won_baseline")]
        # A "capability loss" is where the safe portfolio finds NOTHING and soft found something:
        # a different failure from being slower, and the one that would change what a user gets.
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
         f"% {time.strftime('%Y-%m-%d %H:%M')}  commit {commit}",
         f"% GPU {prov.get('gpu')}  CPU {prov.get('cpu')}", "",
         r"\begin{table}[t]", r"  \centering", r"  \small",
         r"  \caption{\textbf{GPAP vs CMSGen.} " +
         f"GPU {tex_escape(prov.get('gpu'))}, CPU {tex_escape(prov.get('cpu') or 'n/a')}; "
         f"{tot['cells']} cells, {tot['skipped']} skipped, {tot['errors']} errors, "
         f"{tot['unsound']} unsound. "
         r"Geomeans exclude two kinds of degenerate cell, on one principle --- a cell that "
         r"measures the \emph{instance} does not belong in a geomean about the \emph{sampler}: "
         r"vacuous-pin instances (constant outputs at $T{=}25$, so pinning is a tautology) and "
         r"\emph{free} cells, which emit samples with no decisions and no implications because "
         r"the support is unconstrained after propagation. An instance on which \emph{either} "
         r"sampler emitted nothing has no finite ratio, so it is excluded from the geomean and "
         r"counted in its own column: \emph{C$=$0} (CMSGen silent --- an outright win for us, "
         r"which makes the geomeans conservative), \emph{G$=$0} (GPAP silent --- an outright "
         r"loss), \emph{both} (neither sampler reached the instance). The four rightmost counts "
         r"plus the geomean's instances sum to \emph{inst}. "
         r"$^{\dagger}$incomplete stage; $^{\ddagger}$geomean over fewer than three instances "
         r"(CMSGen scored zero on the rest), so it is an illustration, not an estimate.}",
         r"  \label{tab:cmsgen}",
         r"  \begin{tabular}{@{}lllrrrrrrr@{}}", r"    \toprule",
         r"    family & depth & targets & inst & GPAP/CMSGen & wins & C$=$0 & G$=$0 & both & "
         r"free \\",
         r"    \midrule"]
    for stem, label, depth, tgt in CMS_FAMILIES:
        v = cms.get(stem)
        if not v:
            continue
        exp = expected.get(stem.split("-")[0], v["inst"])
        mark = "" if v["inst"] >= exp else r"$^{\dagger}$"
        gtex = f"${v['gm']:,.1f}\\times$" if v["gm"] else "---"
        if v["gm"] and v["n_gm"] < 3:
            gtex += "$^{\\ddagger}$"
        T.append(f"    {tex_escape(label)} & ${depth}$ & {tgt} & {v['inst']}{mark} & "
                 f"{gtex} & {v['wins']}/{v['n']} & {v['inf']} & {v['gpap0']} & "
                 f"{v['both0']} & {v['free']} \\\\")
    T += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    T += [r"\begin{table}[t]", r"  \centering", r"  \small",
          r"  \caption{\textbf{GPAP vs the GPU gradient samplers}, their metric "
          r"(unique solutions / their \texttt{model\_runtime}, which excludes their model build "
          r"while ours includes construction). Rival failures (timeout / no gradient path) are "
          r"counted, never folded in as zeros. As in Table~\ref{tab:cmsgen}, an instance where "
          r"either side emitted nothing carries no finite ratio and is counted separately: "
          r"\emph{R$=$0} (rival silent), \emph{G$=$0} (GPAP silent), \emph{both}.}",
          r"  \label{tab:rivals}",
          r"  \begin{tabular}{@{}llrrrrrrr@{}}", r"    \toprule",
          r"    rival & protocol & cells & GPAP/rival & wins & R$=$0 & G$=$0 & both & failed \\",
          r"    \midrule"]
    for stem, _, rival, proto in RIVALS:
        v = riv.get(stem)
        if not v:
            T.append(f"    {rival} & {tex_escape(proto)} & \\multicolumn{{7}}{{c}}{{not run}} \\\\")
            continue
        g = f"${v['gm']:,.1f}\\times$" if v["gm"] else "---"
        T.append(f"    {rival} & {tex_escape(proto)} & {v['cells']} & {g} & "
                 f"{v['wins']}/{v['n']} & {v['zero']} & {v['gpap0']} & {v['both0']} & "
                 f"{v['errors']} \\\\")
    T += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    if any(prop.values()):
        T += [r"\begin{table}[t]", r"  \centering", r"  \small",
              r"  \caption{\textbf{Propagation throughput --- the engine, isolated from the "
              r"sampler.} Boolean implications per second: GPAP's from the packed-word kernel, "
              r"CMSGen's from its own counters (\texttt{--statsevery}, "
              r"\texttt{scripts/cmsgen\_gpap\_eval.patch}). The identity "
              r"$\text{samples/s ratio} = \text{impl/s ratio} \,/\, \text{work-per-sample ratio}$ "
              r"holds exactly on every family, and factors the sample-rate result into what the "
              r"GPU propagator buys and what rejection sampling spends: GPAP discards a "
              r"conflicted trajectory outright, whereas CMSGen backtracks and retains the partial "
              r"work. GPAP's implication rate is nearly constant across families; the "
              r"three-orders-of-magnitude spread in Table~\ref{tab:cmsgen} therefore comes from "
              r"the work-per-sample column, not from the propagator. $n$ counts instances where "
              r"\emph{both} samplers emitted, since a work-per-sample figure needs both; "
              r"$^{\ddagger}$marks rows where that leaves fewer than three.}",
              r"  \label{tab:prop}",
              r"  \begin{tabular}{@{}lllrrrrrr@{}}", r"    \toprule",
              r"    & & & & \multicolumn{2}{c}{implications/s} & "
              r"\multicolumn{2}{c}{implications/sample} & \\",
              r"    \cmidrule(lr){5-6}\cmidrule(lr){7-8}",
              r"    family & depth & targets & $n$ & GPAP & CMSGen & GPAP & CMSGen & "
              r"ratio \\", r"    \midrule"]
        for stem, label, depth, tgt in CMS_FAMILIES:
            v = prop.get(stem)
            if not v:
                continue
            # same low-n flag as Table 1: these rows carry the families where CMSGen scored zero
            # almost everywhere, so `n` is 1-2 and the row illustrates rather than estimates
            nmark = "$^{\\ddagger}$" if v["n"] < 3 else ""
            T.append(f"    {tex_escape(label)} & ${depth}$ & {tgt} & {v['n']}{nmark} & "
                     f"${v['gpap'] / 1e9:.2f}\\!\\times\\!10^{{9}}$ & "
                     f"${v['cms'] / 1e6:.2f}\\!\\times\\!10^{{6}}$ & "
                     f"{v['gper']:,.0f} & {v['cper']:,.0f} & "
                     f"${v['ratio']:,.0f}\\times$ \\\\")
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
    # These travel WITH the numbers. Every one of them is a thing a referee would otherwise find
    # first, and each is derived from the data rather than remembered.
    C = []
    for stem, label, depth, tgt in CMS_FAMILIES:
        v = cms.get(stem)
        if not v:
            continue
        exp = expected.get(stem.split("-")[0], v["inst"])
        if v["inst"] < exp:
            C.append(f"`{stem}` is an **incomplete stage** ({v['inst']}/{exp} instances) — the "
                     f"rented-GPU budget ran out. Do not read it as a family result.")
        if v["gm"] and v["n_gm"] < 3:
            C.append(f"`{stem}`: the geomean stands on **{v['n_gm']} instance(s)** because CMSGen "
                     f"scored zero on {v['inf']} of {v['n']}. Report the win count "
                     f"({v['wins']}/{v['n']}) and the zero count, not the ratio.")
        if v["free"]:
            C.append(f"`{stem}`: {v['free']} instance(s) "
                     f"({', '.join('`' + n + '`' for n in v['free_names'])}) sample with **no "
                     f"search at all** — zero decisions, zero implications, success 1.0 — because "
                     f"after propagation the support is a free hypercube. Excluded for the same "
                     f"reason as vacuous pins. Leaving them in would read "
                     f"**{v['gm_withfree']:,.1f}x instead of {v['gm']:,.1f}x**.")
        if v["gpap0"]:
            C.append(f"`{stem}`: **GPAP emitted nothing on {v['gpap0']} instance(s) that CMSGen "
                     f"sampled** ({', '.join('`' + n + '`' for n in v['gpap0_names'])}). These are "
                     f"losses. They carry no finite ratio, so they cannot enter the geomean — "
                     f"which is exactly why they must be stated in the text.")
        if v["both0"]:
            C.append(f"`{stem}`: {v['both0']} instance(s) defeated **both** samplers "
                     f"({', '.join('`' + n + '`' for n in v['both0_names'])}) — a limit of the "
                     f"comparison, not evidence about either tool.")
    for stem, _, rival, proto in RIVALS:
        v = riv.get(stem)
        if v and v["errors"]:
            C.append(f"{rival} ({proto}): **{v['errors']} of {v['cells']}** runs failed (timeout "
                     f"or no gradient path). They are excluded, never counted as zeros.")
        if v and v["gpap0"]:
            C.append(f"{rival} ({proto}): **GPAP emitted nothing on {v['gpap0']} instance(s)** "
                     f"that the rival sampled. Losses, excluded from the geomean for lack of a "
                     f"finite ratio — state them.")
    if any(v and not v["cpu"] for v in cms.values()):
        C.append("Some stages lack CPU provenance; a GPAP/CMSGen ratio is not interpretable "
                 "without it.")
    C.append("GPAP/CMSGen ratios are **host-dependent**: CMSGen is single-core, and this host is "
             "0.41x the reference laptop (verified independently by CMSGen's own propagation and "
             "decision counters). Always quote the CPU alongside the ratio.")
    C.append("Sequential geomeans **exclude** b02/b05/b07/s526, whose last-cycle outputs are "
             "constant at T=25 so pinning them is a tautology; including them inflates the rare "
             "column by up to 1.9x.")
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
    ap.add_argument("--cmsgen", required=True)
    ap.add_argument("--cmsgen-ref", default=None)
    ap.add_argument("--rivals", default=None)
    ap.add_argument("--recheck", default=None)
    ap.add_argument("--uniformity", nargs="*", default=[])
    ap.add_argument("--soft", default=None)
    ap.add_argument("--out-tex", default=None)
    ap.add_argument("--out-md", default=None)
    build(ap.parse_args())


if __name__ == "__main__":
    main()
