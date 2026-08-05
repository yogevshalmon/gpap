#!/usr/bin/env python3
"""Generate the deck's full-results appendix straight from the experiment JSONs.

`docs/slides/gpap_overview.tex` \\input's the file this writes. Numbers on slides are therefore
never transcribed by hand — the failure mode that put an inflated 47.7x on the headline slide
(it silently included the vacuous-pin instances) and would have put it on the appendix too.

    eval/slide_tables.py --cmsgen .experiments/cmsgen_l4_2026-08-03 \\
        --rivals .experiments/l4_rivals --out docs/slides/l4_results.tex

Everything it prints is derived, and the two rules that make the numbers honest are applied
here, in one place:
  * VACUOUS-PIN instances (constant outputs at T=25) are excluded from the sequential geomeans.
  * ratio = inf (CMSGen produced ZERO samples in budget) never enters a geomean; it is counted
    and reported separately as an outright win.
"""
import argparse
import json
import math
import os

VACUOUS_PINS = {"b02", "b05", "b07", "s526"}

# (stem, human label) in the order they should appear
CMS_STAGES = [
    ("seq-c25-rare", r"ITC'99 $+$ ISCAS-89 seq, $T{=}25$", "rare"),
    ("seq-c50-rare", r"ITC'99 $+$ ISCAS-89 seq, $T{=}50$", "rare"),
    ("seq-c25-random", r"ITC'99 $+$ ISCAS-89 seq, $T{=}25$", "random"),
    ("seq-c50-random", r"ITC'99 $+$ ISCAS-89 seq, $T{=}50$", "random"),
    ("comb-rare", r"ISCAS-85 combinational", "rare"),
    ("comb-random", r"ISCAS-85 combinational", "random"),
    ("hwmcc-c25-rare", r"HWMCC'17, $T{=}25$", "rare"),
    ("hwmcc-c50-rare", r"HWMCC'17, $T{=}50$", "rare"),
    ("hwmcc-c25-random", r"HWMCC'17, $T{=}25$", "random"),
    ("hwmcc-c50-random", r"HWMCC'17, $T{=}50$", "random"),
    ("epfl-rare", r"EPFL combinational", "rare"),
    ("epfl-random", r"EPFL combinational", "random"),
    ("sk", r"Sketch / \texttt{sk\_} projected CNFs", "---"),
]

RIVAL_STAGES = [
    ("demotic_native", "demotic", r"DEMOTIC", r"their ISCAS-85 pins"),
    ("demotic_miter", "demotic", r"DEMOTIC", r"our \keyw{rare} pins"),
    ("recurrent_table", "rcs", r"Recurrent", r"their target table"),
    ("recurrent_rare", "rcs", r"Recurrent", r"our \keyw{rare} pins"),
    ("hts", "hts", r"HTS", r"our CNFs (both encoders)"),
]


def gm(xs):
    xs = [x for x in xs if x and x > 0 and math.isfinite(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def fmt(x, suffix=r"$\times$"):
    if x is None:
        return "---"
    if x >= 1e5:
        return f"${x:,.0f}$" + suffix
    return f"${x:,.1f}$" + suffix


def cms_stage(d, stem):
    p = os.path.join(d, f"{stem}.json")
    if not os.path.isfile(p):
        return None
    payload = json.load(open(p))
    rows = [r for r in payload.get("results", []) if not r.get("skipped") and not r.get("error")]
    if not rows:
        return None
    best = {}
    for r in rows:
        k = r["name"].split(".")[0]
        if k not in best or (r.get("ratio") or 0) > (best[k].get("ratio") or 0):
            best[k] = r
    finite = {k: r for k, r in best.items()
              if r.get("ratio") and math.isfinite(r["ratio"])}
    novac = [r["ratio"] for k, r in finite.items() if k not in VACUOUS_PINS]
    n_inf = sum(1 for r in best.values() if r.get("ratio") and not math.isfinite(r["ratio"]))
    return {
        "cells": len(rows), "inst": len(best),
        "gm": gm([r["ratio"] for r in finite.values()]),
        "gm_novac": gm(novac), "n_novac": len(novac),
        "wins": sum(1 for r in finite.values() if r["ratio"] >= 1) + n_inf,
        "n": len(finite) + n_inf, "inf": n_inf,
        "vac": bool(set(best) & VACUOUS_PINS),
        "skipped": sum(1 for r in payload.get("results", []) if r.get("skipped")),
        "errors": sum(1 for r in payload.get("results", []) if r.get("error")),
    }


def rival_stage(d, stem, key):
    p = os.path.join(d, f"rivals_{stem}.json")
    if not os.path.isfile(p):
        return None
    payload = json.load(open(p))
    rows = (payload.get("rows") if isinstance(payload, dict) else payload) or []
    seen = {}
    for r in rows:
        seen[(r["name"], r.get("variant", "-"))] = r      # last wins (a retry supersedes)
    ratios, errs, zero = [], 0, 0
    for r in seen.values():
        rv, g = r.get(key) or {}, r.get("gpap") or {}
        if rv.get("error"):
            errs += 1
            continue
        rate = rv.get("unique_per_s") or 0.0
        if not rate:
            zero += 1
            continue
        ratios.append((g.get("unique_per_s_est") or 0.0) / rate)
    return {"cells": len(seen), "gm": gm(ratios), "n": len(ratios),
            "wins": sum(1 for x in ratios if x >= 1), "errors": errs, "zero": zero}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmsgen", required=True)
    ap.add_argument("--rivals", required=True)
    ap.add_argument("--uniformity", default=None,
                    help="fuzz_uniformity.py campaign JSON -> the uniformity appendix frame")
    ap.add_argument("--soft", default=None,
                    help="profile_soft_portfolio.py JSON -> the `soft` throughput-price frame")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cms = {s: cms_stage(args.cmsgen, s) for s, _, _ in CMS_STAGES}
    riv = {s: rival_stage(args.rivals, s, k) for s, k, _, _ in RIVAL_STAGES}
    payload = next((json.load(open(os.path.join(args.cmsgen, f"{s}.json")))
                    for s, _, _ in CMS_STAGES
                    if os.path.isfile(os.path.join(args.cmsgen, f"{s}.json"))), {})
    gpu = (payload.get("gpu") or {}).get("name", "?")
    cpu = payload.get("cpu") or {}
    cpu_s = f"{cpu.get('model', 'not recorded')}"
    total_cells = sum(v["cells"] for v in cms.values() if v)
    total_inst = sum(v["inst"] for v in cms.values() if v)
    n_skip = sum(v["skipped"] for v in cms.values() if v)
    n_err = sum(v["errors"] for v in cms.values() if v)

    L = []
    a = L.append
    a("% GENERATED by eval/slide_tables.py — do not edit by hand.")
    a(f"% source: {args.cmsgen} + {args.rivals}")
    a("")
    # ---------------------------------------------------------------- frame 1: vs CMSGen
    a(r"\begin{frame}{Appendix: every CMSGen family measured on the L4}")
    a(r"  \centering \scriptsize")
    a(r"  \begin{tabular}{@{}llrrrrr@{}}")
    a(r"    \toprule")
    a(r"    family & targets & inst & GPAP$/$CMSGen & \textbf{no-vac} & wins & CMSGen 0 \\")
    a(r"    \midrule")
    # A stage that ran out of budget mid-family must not look like a complete one. The family's
    # own maximum instance count is the expected size; anything short of it is marked.
    expected = {}
    for stem, _, _ in CMS_STAGES:
        v = cms.get(stem)
        if v:
            fam = stem.split("-")[0]
            expected[fam] = max(expected.get(fam, 0), v["inst"])
    partial = []
    for stem, label, tgt in CMS_STAGES:
        v = cms.get(stem)
        if not v:
            continue
        nv = fmt(v["gm_novac"]) if v["vac"] else "---"
        tgtx = r"\keyw{rare}" if tgt == "rare" else tgt
        exp = expected.get(stem.split("-")[0], v["inst"])
        mark = ""
        if v["inst"] < exp:
            mark = r"$^{\dagger}$"
            partial.append(f"{stem} ({v['inst']}/{exp})")
        a(f"    {label} & {tgtx} & {v['inst']}{mark} & {fmt(v['gm'])} & {nv} & "
          f"{v['wins']}/{v['n']} & {v['inf']} \\\\")
    a(r"    \bottomrule")
    a(r"  \end{tabular}")
    a(r"  \\[0.7em]")
    a(r"  \rem{\textbf{" + f"{total_cells} cells, {total_inst} instance-slots, {n_skip} skipped, "
      f"{n_err} errors" + r"}. GPU \textbf{" + gpu + r"}, CPU \textbf{" + cpu_s + r"}.")
    a(r"  \keyw{no-vac} drops \texttt{b02/b05/b07/s526}, whose outputs are constant at $T{=}25$ so"
      r"  any sampler scores ${\sim}100\%$ --- quote it, not the raw column.")
    a(r"  \keyw{CMSGen 0} counts instances where CMSGen emitted \emph{zero} samples in budget:"
      r"  outright wins, but ratio $=\infty$, so they are EXCLUDED from every geomean above."
      r"  The numbers are therefore conservative.")
    if partial:
        a(r"  $^{\dagger}$\bad{incomplete stage} (rented-GPU budget ran out): "
          + ", ".join(r"\texttt{" + p.replace("_", r"\_") + "}" for p in partial) + ".")
    a(r"}")
    a(r"\end{frame}")
    a("")
    # ---------------------------------------------------------------- frame 2: vs rivals
    a(r"\begin{frame}{Appendix: every GPU rival measured on the L4}")
    a(r"  \centering \scriptsize")
    a(r"  \begin{tabular}{@{}llrrrr@{}}")
    a(r"    \toprule")
    a(r"    rival & protocol & cells & GPAP$/$rival & wins & rival failed \\")
    a(r"    \midrule")
    for stem, key, rival, proto in RIVAL_STAGES:
        v = riv.get(stem)
        if not v:
            a(f"    {rival} & {proto} & \\multicolumn{{4}}{{c}}{{\\bad{{not run}}}} \\\\")
            continue
        a(f"    {rival} & {proto} & {v['cells']} & {fmt(v['gm'])} & "
          f"{v['wins']}/{v['n']} & {v['errors']} \\\\")
    a(r"    \bottomrule")
    a(r"  \end{tabular}")
    a(r"  \\[0.7em]")
    a(r"  \rem{Their metric: unique solutions $/$ their \texttt{model\_runtime}, which excludes"
      r"  their model build while ours includes construction. \keyw{rival failed} = timeout or"
      r"  no gradient path; those rows are reported, never folded in as a zero. Rows where the"
      r"  rival scored $0$ are excluded from the geomean for the same reason as above.}")
    a(r"\end{frame}")
    a("")
    # ---------------------------------------------------------------- frame 3: uniformity
    uni = json.load(open(args.uniformity)) if args.uniformity and os.path.isfile(args.uniformity) \
        else None
    if uni:
        agg = {}
        for r in uni.get("rows", []):
            for nm, sm in (r.get("samplers") or {}).items():
                k = (nm, r.get("kind"))
                e = agg.setdefault(k, {"tv": [], "skew": 0, "n": 0, "uns": 0})
                e["tv"].append(sm.get("tv_median"))
                e["n"] += 1
                e["uns"] += sm.get("unsound") or 0
                e["skew"] += (sm.get("verdict") == "SKEW")
        def med(xs):
            xs = sorted(x for x in xs if x is not None)
            return xs[len(xs) // 2] if xs else float("nan")
        # the rows worth a slide: the shipped configs, the opt-in `soft`, the B ladder, the
        # rival, and BOTH controls (without which "no skew" means nothing)
        order = ["ctl:uniform", "ctl:tv0.05", "gpap:b1", "gpap:b8", "gpap:soft",
                 "gpap:b32", "gpap:b64", "cmsgen"]
        a("\\begin{frame}{Appendix: uniformity against an exact oracle}")
        a(r"  \centering \scriptsize")
        a(r"  \begin{tabular}{@{}lrrrr@{}}")
        a(r"    \toprule")
        a(r"    sampler & \multicolumn{2}{c}{circuit} & \multicolumn{2}{c}{random $k$-CNF} \\")
        a(r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}")
        a(r"     & median TV & skewed & median TV & skewed \\")
        a(r"    \midrule")
        for nm in order:
            c, n = agg.get((nm, "circuit")), agg.get((nm, "cnf"))
            if not (c or n):
                continue
            def cell(e):
                return (f"${med(e['tv']):.4f}$", f"{e['skew']}/{e['n']}") if e else ("---", "---")
            (ct, cs), (nt, ns) = cell(c), cell(n)
            label = nm.replace("_", r"\_")
            hi = r"\bad" if nm == "gpap:soft" else (r"\rem" if nm.startswith("ctl") else "")
            a(f"    \\texttt{{{label}}} & {ct} & {cs} & {nt} & {ns} \\\\"
              if not hi else
              f"    {hi}{{\\texttt{{{label}}}}} & {ct} & {cs} & {nt} & {ns} \\\\")
        a(r"    \bottomrule")
        a(r"  \end{tabular}")
        a(r"  \\[0.7em]")
        n_inst = len(uni.get("rows", []))
        uns = sum(e["uns"] for e in agg.values())
        seeds = uni.get("args", {}).get("seeds")
        alpha = uni.get("args", {}).get("alpha")
        cap = [
            r"  \rem{", str(n_inst),
            r" generated instances whose ENTIRE solution set is enumerated by brute force, so "
            r"the target is known exactly; $\chi^2$ over ", str(seeds),
            r" seeds, Fisher-combined, $\alpha=", str(alpha), r"$. \good{", str(uns),
            r" unsound emissions}. \keyw{The controls calibrate the instrument}: "
            r"\texttt{ctl:uniform} is exactly uniform and must NOT be flagged; "
            r"\texttt{ctl:tv0.05} is skewed by a known amount and must be. "
            r"\bad{\texttt{soft} is why it is opt-in} --- and $B\ge|\text{support}|$ "
            r"(\texttt{b32}/\texttt{b64}) is exactly uniform, because discarding conflicts IS "
            r"rejection sampling.}",
        ]
        a("".join(cap))
        a(r"\end{frame}")
        a("")

    with open(args.out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"wrote {args.out}  ({len(L)} lines)")
    for stem, v in cms.items():
        if v:
            print(f"  {stem:<20} {v['inst']:>3} inst  gm={fmt(v['gm'])}  novac={fmt(v['gm_novac'])}"
                  f"  wins={v['wins']}/{v['n']}  cmsgen-zero={v['inf']}")


if __name__ == "__main__":
    main()
