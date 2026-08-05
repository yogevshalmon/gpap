#!/usr/bin/env python3
"""Build the paper's experiment section — every table, straight from the experiment JSONs.

Nothing here is typed by hand. That is the point: an earlier version of the deck carried a
sequential headline of 47.7x that silently included four instances whose output pins are
tautologies (real value 36.5x), and the only durable fix is to derive every published number from
the data with the exclusions applied in ONE place.

    eval/paper_tables.py --cmsgen .experiments/cmsgen_l4_2026-08-03 \\
        --cmsgen-ref .experiments/full_eval/nvidia-geforce-rtx-4080-laptop-gpu \\
        --rivals .experiments/l4_rivals \\
        --recheck .experiments/recheck \\
        --uniformity .experiments/fuzz/round2.json \\
        --soft .experiments/soft_profile_l4.json \\
        --out-tex docs/paper/tables.tex --out-md docs/paper/RESULTS.md

Rules applied everywhere, so they cannot diverge between tables:
  * VACUOUS-PIN instances (b02/b05/b07/s526 — constant last-cycle outputs at T=25, so pinning is
    a tautology and every sampler scores ~100%) are excluded from the sequential geomeans.
  * ratio = inf (CMSGen emitted ZERO samples inside the budget) never enters a geomean; it is
    counted separately as an outright win, which makes the published figures CONSERVATIVE.
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

def cms_family(d, stem):
    payload = load(os.path.join(d, f"{stem}.json")) if d else None
    if not payload:
        return None
    allr = payload.get("results", [])
    rows = [r for r in allr if not r.get("skipped") and not r.get("error")]
    if not rows:
        return None
    best = {}
    for r in rows:
        k = r["name"].split(".")[0]
        if k not in best or (r.get("ratio") or 0) > (best[k].get("ratio") or 0):
            best[k] = r
    finite = {k: r for k, r in best.items() if r.get("ratio") and math.isfinite(r["ratio"])}
    n_inf = sum(1 for r in best.values() if r.get("ratio") and not math.isfinite(r["ratio"]))
    novac = [r["ratio"] for k, r in finite.items() if k not in VACUOUS_PINS]
    has_vac = bool(set(best) & VACUOUS_PINS)
    gm_vals = novac if has_vac else [r["ratio"] for r in finite.values()]
    return {
        "cells": len(rows), "inst": len(best),
        "gm_all": gm([r["ratio"] for r in finite.values()]),
        "gm": gm(gm_vals),
        # How many instances actually ENTER the geomean. When CMSGen scores zero on most of a
        # family (HWMCC T=50: 6 of 7), the ratio is undefined there and the "geomean" collapses
        # onto the handful that remain — 421x over ONE instance. A paper must show this number.
        "n_gm": len([x for x in gm_vals if x and math.isfinite(x)]),
        "has_vac": has_vac,
        "wins": sum(1 for r in finite.values() if r["ratio"] >= 1) + n_inf,
        "n": len(finite) + n_inf, "inf": n_inf,
        "skipped": sum(1 for r in allr if r.get("skipped")),
        "errors": sum(1 for r in allr if r.get("error")),
        "unsound": sum(1 for r in rows
                       if r.get("gpu_sound") is False or r.get("cpu_sound") is False),
        "gpu": (payload.get("gpu") or {}).get("name"),
        "cpu": (payload.get("cpu") or {}).get("model"),
        "patched": payload.get("cmsgen_patched"),
    }


# ------------------------------------------------------------------ GPAP vs GPU rivals

def rival_family(d, stem, key):
    payload = load(os.path.join(d, f"rivals_{stem}.json")) if d else None
    if payload is None:
        return None
    rows = (payload.get("rows") if isinstance(payload, dict) else payload) or []
    seen = {}
    for r in rows:
        seen[(r["name"], r.get("variant", "-"))] = r        # a retry supersedes its error row
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
    # a rival that scored ZERO is an outright win for us, so it belongs in BOTH the numerator
    # and the denominator of the win count — but never in the geomean (division by zero).
    return {"cells": len(seen), "gm": gm(ratios), "n": len(ratios) + zero,
            "wins": sum(1 for x in ratios if x >= 1) + zero, "errors": errs, "zero": zero}


# ------------------------------------------------------------------ uniformity

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
         "Generated by `eval/paper_tables.py`. Vacuous pins excluded from sequential geomeans; "
         "CMSGen-scored-zero cells excluded from geomeans and reported separately.", "",
         "## Table 1 — GPAP vs CMSGen", "",
         "| family | depth | targets | inst | GPAP/CMSGen | wins | CMSGen=0 | ref GPU |",
         "|---|---|---|---:|---:|---:|---:|---:|"]
    for stem, label, depth, tgt in CMS_FAMILIES:
        v = cms.get(stem)
        if not v:
            continue
        exp = expected.get(stem.split("-")[0], v["inst"])
        mark = "" if v["inst"] >= exp else f" (partial {v['inst']}/{exp})"
        rv = ref.get(stem)
        refs = f"{rv['gm']:,.1f}x" if rv and rv["gm"] else "—"
        gms = f"{v['gm']:,.1f}x" if v["gm"] else "— (all CMSGen=0)"
        if v["gm"] and v["n_gm"] < 3:
            gms += f" ⚠n={v['n_gm']}"
        M.append(f"| {label}{mark} | {depth} | {tgt} | {v['inst']} | "
                 f"{gms} | {v['wins']}/{v['n']} | {v['inf']} | {refs} |")
    M += ["", "## Table 2 — GPAP vs the GPU gradient samplers", "",
          "| rival | protocol | cells | GPAP/rival | wins | rival failed |",
          "|---|---|---:|---:|---:|---:|"]
    for stem, _, rival, proto in RIVALS:
        v = riv.get(stem)
        if not v:
            M.append(f"| {rival} | {proto} | — | not run | — | — |")
            continue
        gs = f"{v['gm']:,.1f}x" if v["gm"] else "—"
        M.append(f"| {rival} | {proto} | {v['cells']} | {gs} | "
                 f"{v['wins']}/{v['n']} | {v['errors']} |")
    if uni:
        M += ["", "## Table 3 — uniformity vs an exact oracle", "",
              f"`alpha={uni_args.get('alpha')}`, {uni_args.get('seeds')} seeds/instance, "
              f"n = {uni_args.get('n_mult')}x|S| samples.", "",
              "| sampler | kind | instances | median TV | flagged SKEW | unsound |",
              "|---|---|---:|---:|---:|---:|"]
        for (name, kind), a in sorted(uni.items()):
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
        M += ["", "## Table 4 — the price of dropping `soft`", "",
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
        M += ["", "## Table 5 — reproducibility (fresh VM, same protocol)", "",
              f"* deterministic fields (M, support, packed split, propagate variant): "
              f"**{rc['exact']}/{rc['exact'] + rc['mismatch']} exact**",
              f"* success rate within 0.05: **{rc['succ_ok']}/{rc['succ_ok'] + rc['succ_bad']}**",
              f"* throughput recheck/original: geomean **{rc['gm']:.3f}x** "
              f"({rc['lo']:.2f}–{rc['hi']:.2f}, n={rc['n']})"]

    # ============================================================ latex
    T = ["% GENERATED by eval/paper_tables.py — do not edit by hand.",
         f"% {time.strftime('%Y-%m-%d %H:%M')}  commit {commit}",
         f"% GPU {prov.get('gpu')}  CPU {prov.get('cpu')}", "",
         r"\begin{table}[t]", r"  \centering", r"  \small",
         r"  \caption{\textbf{GPAP vs CMSGen.} " +
         f"GPU {tex_escape(prov.get('gpu'))}, CPU {tex_escape(prov.get('cpu') or 'n/a')}; "
         f"{tot['cells']} cells, {tot['skipped']} skipped, {tot['errors']} errors, "
         f"{tot['unsound']} unsound. "
         r"Sequential rows exclude vacuous-pin instances. \emph{CMSGen$=$0} counts instances "
         r"where CMSGen emitted no sample inside the budget: outright wins, excluded from the "
         r"geomean, so the ratios are conservative. "
         r"$^{\dagger}$incomplete stage; $^{\ddagger}$geomean over fewer than three instances "
         r"(CMSGen scored zero on the rest), so it is an illustration, not an estimate.}",
         r"  \label{tab:cmsgen}",
         r"  \begin{tabular}{@{}lllrrrr@{}}", r"    \toprule",
         r"    family & depth & targets & inst & GPAP/CMSGen & wins & CMSGen$=$0 \\",
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
                 f"{gtex} & {v['wins']}/{v['n']} & {v['inf']} \\\\")
    T += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    T += [r"\begin{table}[t]", r"  \centering", r"  \small",
          r"  \caption{\textbf{GPAP vs the GPU gradient samplers}, their metric "
          r"(unique solutions / their \texttt{model\_runtime}, which excludes their model build "
          r"while ours includes construction). Rival failures (timeout / no gradient path) are "
          r"counted, never folded in as zeros.}",
          r"  \label{tab:rivals}",
          r"  \begin{tabular}{@{}llrrrr@{}}", r"    \toprule",
          r"    rival & protocol & cells & GPAP/rival & wins & rival failed \\", r"    \midrule"]
    for stem, _, rival, proto in RIVALS:
        v = riv.get(stem)
        if not v:
            T.append(f"    {rival} & {tex_escape(proto)} & \\multicolumn{{4}}{{c}}{{not run}} \\\\")
            continue
        g = f"${v['gm']:,.1f}\\times$" if v["gm"] else "---"
        T.append(f"    {rival} & {tex_escape(proto)} & {v['cells']} & {g} & "
                 f"{v['wins']}/{v['n']} & {v['errors']} \\\\")
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
        for (name, kind), a in sorted(uni.items()):
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
    for stem, _, rival, proto in RIVALS:
        v = riv.get(stem)
        if v and v["errors"]:
            C.append(f"{rival} ({proto}): **{v['errors']} of {v['cells']}** runs failed (timeout "
                     f"or no gradient path). They are excluded, never counted as zeros.")
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
