#!/usr/bin/env python3
"""What the uniformity guarantee costs, measured on the comparison we actually publish.

`soft` (soft global lookahead) is Barbarik-rejected 5/5 against an exact oracle, so it was removed
from the default portfolio and is opt-in behind `--allow-soft`. Every number in the
shipped comparison therefore comes from a sampler that cannot choose it. This tool prices that
decision by pairing two complete sweeps that differ in exactly one flag:

    eval/artifact/soft_price_tables.py --safe <no-soft dir> --soft <--allow-soft dir> \\
        --out-md docs/paper/SOFT_PRICE.md --out-tex docs/paper/soft_price.tex

`eval/profile_soft_portfolio.py` already answers a narrower question — GPAP against itself on 19
sequential circuits, geomean 0.42x. It cannot answer this one. "We gave up 2.4x of our own
throughput" and "we gave up this much of our lead over CMSGen, and these instances outright" are
different claims, and only the second is visible to a reader of the comparison table.

Three quantities, because they say different things and the first alone is misleading:

  * THROUGHPUT price, over cells where both arms sampled. Diluted by construction: on most cells
    the bandit never picks `soft`, so the two arms run identical configurations and contribute
    1.0x. Reported both over all cells and over the cells where `soft` actually won.
  * CAPABILITY change: instances one arm samples and the other does not. A 2x slowdown and "you
    now get nothing" are not the same loss, and only this column shows the second.
  * HEADLINE effect: the published GPAP/CMSGen geomean recomputed on each arm, under the same
    exclusion rules (`eval.artifact.paper_tables`), which is the number a reader would actually see move.

Both arms must come from the same GPU and CPU — the check is enforced, not assumed, because a
CMSGen ratio is a property of the host as much as of the sampler.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.artifact.paper_tables import (CMS_FAMILIES, cms_family, encoders_in, gm, load,  # noqa: E402
                               tex_escape)


def cells(d, stem):
    """Every (instance, encoder) cell of a stage, keyed for pairing across arms."""
    payload = load(os.path.join(d, f"{stem}.json")) if d else None
    if not payload:
        return {}, {}
    out = {}
    for r in payload.get("results", []):
        if r.get("skipped") or r.get("error"):
            continue
        out[(r["name"].split(".")[0], r.get("enc_key") or r.get("encoder") or "-")] = r
    return out, payload


def soft_won(r):
    """Did the bandit actually select `soft` on this cell? `tuned` is (var_order, B, lookahead)."""
    t = r.get("tuned") or []
    return len(t) > 2 and str(t[2]).lower() == "soft"


def headline(d, stem):
    """The published geomean for one stage — literally the paper's numbers, not a copy of them.

    Reimplementing the exclusions here is the mistake this whole toolchain exists to prevent, and
    a first draft of this file made it anyway.

    Table 1 reports each encoder as its own row (nothing selects between them). This table is one
    row per STAGE, so what it needs is a single figure: the geomean OVER those per-encoder
    geomeans. It is context for the soft-vs-safe column, never a headline in its own right.
    """
    payload = load(os.path.join(d, f"{stem}.json")) if d else None
    if not payload:
        return None
    vs = [cms_family(d, stem, e) for e in encoders_in(payload)]
    return gm([v["gm"] for v in vs if v and v["gm"]])


def build(args):
    prov = {}
    for tag, d in (("safe", args.safe), ("soft", args.soft)):
        for stem, _, _, _ in CMS_FAMILIES:
            _, p = cells(d, stem)
            if p:
                prov[tag] = ((p.get("gpu") or {}).get("name"),
                             (p.get("cpu") or {}).get("model"))
                break
    if len(prov) == 2 and prov["safe"] != prov["soft"]:
        print(f"[FATAL] the two arms ran on different hardware:\n  safe {prov['safe']}\n  "
              f"soft {prov['soft']}\nA CMSGen ratio is a property of the host; pairing these "
              f"would attribute a machine difference to `soft`.", file=sys.stderr)
        return 2

    rows, per_stage = [], []
    ctrl_cms, ctrl_mismatch = [], []          # the control: what should NOT have changed
    for stem, label, depth, tgt in CMS_FAMILIES:
        a, _ = cells(args.safe, stem)
        b, _ = cells(args.soft, stem)
        shared = sorted(set(a) & set(b))
        if not shared:
            continue
        # CMSGen saw the same binary, instances and budgets in both arms, so its throughput is a
        # NEGATIVE CONTROL on the pairing: if it moved, the two arms are not comparable and the
        # `soft` effect would be confounded with a machine difference. Likewise M and |support|
        # are deterministic — any mismatch means the encodings differ and nothing else is valid.
        for k in shared:
            ca, cb = a[k].get("cms_sps"), b[k].get("cms_sps")
            if ca and cb:
                ctrl_cms.append(cb / ca)
            if (a[k].get("M") != b[k].get("M")
                    or a[k].get("support") != b[k].get("support")):
                ctrl_mismatch.append(f"{k[0]}/{k[1]}")
        both, won, cap_loss, cap_gain = [], [], [], []
        for k in shared:
            ra, rb = a[k], b[k]
            sa = ra.get("gpap_sps") or 0.0
            sb = rb.get("gpap_sps") or 0.0
            ma = (ra.get("n_models") or 0) > 0
            mb = (rb.get("n_models") or 0) > 0
            w = soft_won(rb)
            if ma and mb and sa > 0 and sb > 0:
                both.append(sa / sb)
                if w:
                    won.append(sa / sb)
            elif mb and not ma:
                cap_loss.append((stem,) + k + (rb.get("succ") or 0.0,))
            elif ma and not mb:
                cap_gain.append((stem,) + k + (ra.get("succ") or 0.0,))
            rows.append({"stage": stem, "instance": k[0], "encoder": k[1],
                         "safe_sps": sa, "soft_sps": sb, "safe_models": ra.get("n_models"),
                         "soft_models": rb.get("n_models"), "soft_won_bandit": w,
                         "safe_succ": ra.get("succ"),
                         "ratio_safe_over_soft": (sa / sb) if (sa > 0 and sb > 0) else None})
        per_stage.append({
            "stem": stem, "label": label, "depth": depth, "target": tgt,
            "cells": len(shared), "gm_all": gm(both), "n_all": len(both),
            "gm_won": gm(won), "n_won": len(won),
            "cap_loss": cap_loss, "cap_gain": cap_gain,
            # A headline comparison is only meaningful when BOTH arms ran the whole family: a
            # partial soft stage aggregates a different instance set, and reporting it next to
            # the complete safe stage compares 19 instances against 3. Seen live at 6/38 cells,
            # where it read 183.3x -> 141.8x and looked like `soft` HURTING at random pins.
            "complete": len(a) == len(b),
            "head_safe": headline(args.safe, stem),
            "head_soft": headline(args.soft, stem) if len(a) == len(b) else None})

    allr = [r["ratio_safe_over_soft"] for r in rows if r["ratio_safe_over_soft"]]
    wonr = [r["ratio_safe_over_soft"] for r in rows
            if r["ratio_safe_over_soft"] and r["soft_won_bandit"]]
    n_won = sum(1 for r in rows if r["soft_won_bandit"])
    loss = [k for s in per_stage for k in s["cap_loss"]]
    gain = [k for s in per_stage for k in s["cap_gain"]]

    def pct(x):
        return f"{(x - 1) * 100:+.0f}%" if x else "—"

    M = ["# The price of the uniformity guarantee", "",
         f"GPU **{prov.get('safe', ('?',))[0]}**, CPU **{prov.get('safe', ('?', '?'))[1]}** — "
         "both arms, same machine type, same protocol, same budgets, same encoders. The only "
         "difference is `--allow-soft`, injected through `run_full_eval.sh`'s single `bench` "
         "call so both arms traverse identical code.", "",
         f"Sources: safe `{args.safe}`, soft `{args.soft}`. **Both arms must come from the SAME "
          f"campaign** — this table is a GPAP-vs-GPAP pairing, so crossing campaigns would put "
          f"cross-campaign draw noise into a comparison that has none of the confounds the "
          f"CMSGen table has.",
          "",
          "`soft` is Barbarik-rejected 5/5 against an exact oracle (median TV 0.63 vs a 0.040 "
         "detection floor), so the shipped sampler cannot select it. Ratios below are "
         "**safe / soft**: less than 1.0 means the uniformity-safe default is slower.", "",
         "## Summary", "",
         f"* **Throughput, all paired cells: {gm(allr):.3f}x** (n={len(allr)}) — "
         f"{pct(gm(allr))} against a sampler free to choose `soft`.",
         f"* **Throughput where the bandit actually chose `soft`: "
         f"{(f'{gm(wonr):.3f}x' if wonr else 'n/a')}** (n={len(wonr)} of {n_won} such cells). "
         f"The all-cells figure is diluted: on the other cells both arms run the same "
         f"configuration and contribute exactly 1.0x.",
         f"* **Capability lost** (only the `soft` arm sampled at all): **{len(loss)}** cells"
         + (": " + ", ".join(f"`{i}`/{e} ({s}, succ {c:.1e})" for s, i, e, c in loss)
            if loss else ""),
         f"* **Capability gained** (only the safe arm sampled): **{len(gain)}** cells"
         + (": " + ", ".join(f"`{i}`/{e} ({s}, succ {c:.1e})" for s, i, e, c in gain)
            if gain else "")]
    # A capability difference is only evidence when the surviving arm was comfortably above the
    # sampler's zero-progress valve. Below ~1e-3 the valve makes the outcome BIMODAL: every
    # config is measured, a grace chunk follows, and if those chunks happen to miss, the run
    # stops and reports zero. Measured live on EPFL `div` (succ 2.1e-4): the safe arm ran 173
    # chunks over 60 s and found 441 models; the soft arm stopped at 1.72 s after 5 chunks with
    # 0 — and its bandit had chosen a NON-soft config, so the difference was a draw, not `soft`.
    FRAGILE = 1e-3
    frag = [x for x in loss + gain if x[3] < FRAGILE]
    if frag:
        M += ["",
              f"  ⚠️ **{len(frag)} of {len(loss) + len(gain)} capability differences sit below "
              f"success {FRAGILE:.0e}**, where the zero-progress stop makes the outcome bimodal "
              f"— the run halts once every config plus a grace chunk has missed. Treat those as "
              f"draws, not as evidence about `soft`: "
              + ", ".join(f"`{i}`/{e}" for _, i, e, _ in frag) + "."]
    # The number that answers the question as a reader of the comparison table would ask it:
    # not "how much throughput", but "how much of the PUBLISHED ratio".
    pub = [(s["label"], s["depth"], s["target"], s["head_safe"], s["head_soft"])
           for s in per_stage if s["head_safe"] and s["head_soft"] and s["complete"]]
    if pub:
        shift = gm([b / a for _, _, _, a, b in pub])
        M += ["",
              f"* **Effect on the published GPAP/CMSGen headline: {shift:.2f}x.** Allowing `soft` "
              f"would move the reported ratios to "
              + ", ".join(f"{a:,.1f}x → **{b:,.1f}x** ({lb.split()[0]} {d} {t})"
                          for lb, d, t, a, b in pub)
              + ". That is the size of the claim we are choosing not to make."]
    M += ["",
         "## Per family", "",
         "| family | depth | targets | cells | safe/soft (all) | safe/soft (soft-won) "
         "| soft-won cells | cap. lost | headline safe | headline soft |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in per_stage:
        nw = sum(1 for r in rows if r["stage"] == s["stem"] and r["soft_won_bandit"])
        ga = f"{s['gm_all']:.3f}x" if s["gm_all"] else "—"
        gw = f"{s['gm_won']:.3f}x" if s["gm_won"] else "—"
        hs = f"{s['head_safe']:,.1f}x" if s["head_safe"] else "—"
        ht = f"{s['head_soft']:,.1f}x" if s["head_soft"] else "— (partial)"
        mark = "" if s["complete"] else " †"
        M.append(f"| {s['label']}{mark} | {s['depth']} | {s['target']} | {s['cells']} | {ga} | "
                 f"{gw} | {nw} | {len(s['cap_loss'])} | {hs} | {ht} |")
    if any(not s["complete"] for s in per_stage):
        M += ["", "† **incomplete stage** — the `soft` arm has fewer cells than the safe arm here. "
              "The paired columns are still valid (they only use cells both arms ran), but the "
              "headline column is withheld: a family geomean over a different instance set is not "
              "comparable to the published one."]

    # WHERE the price is paid — the single most useful cut of this experiment. A geomean over the
    # whole matrix says "2.4x" and hides that the cost is ~nil on instances the sampler finds easy
    # and catastrophic on the ones it does not. Bucketing by the SAFE arm's success rate (the
    # property known before choosing a portfolio) makes it monotone and actionable.
    BUCKETS = [(0.0, 1e-4, "< 1e-4"), (1e-4, 1e-2, "1e-4 .. 1e-2"),
               (1e-2, 0.5, "1e-2 .. 0.5"), (0.5, 1.01, "> 0.5")]
    M += ["", "## Where the price is paid — by instance difficulty", "",
          "Bucketed by the **safe** arm's success rate, which is the thing you could know before "
          "choosing a portfolio. The cost is not spread evenly; it is a function of how hard the "
          "instance is.", "",
          "| safe-arm success rate | cells | safe/soft | `soft` won the bandit |",
          "|---|---:|---:|---:|"]
    for lo, hi, lab in BUCKETS:
        sel = [r for r in rows
               if r["ratio_safe_over_soft"] and r["safe_succ"] is not None
               and lo <= r["safe_succ"] < hi]
        if not sel:
            continue
        g = gm([r["ratio_safe_over_soft"] for r in sel])
        nw = sum(1 for r in sel if r["soft_won_bandit"])
        M.append(f"| {lab} | {len(sel)} | **{g:.3f}x** | {nw}/{len(sel)} |")
    M += ["",
          "This is the shape that matters for the parked design question: a bandit that fell back "
          "to `soft` **only** where every uniformity-safe config yields almost nothing would "
          "recover most of the loss while leaving the easy majority — where the safe portfolio is "
          "already within a few percent — untouched and uniform."]

    worst = sorted((r for r in rows if r["ratio_safe_over_soft"] and r["soft_won_bandit"]),
                   key=lambda r: r["ratio_safe_over_soft"])[:15]
    if worst:
        M += ["", "## Where it actually costs (bandit chose `soft`, worst 15 cells)", "",
              "| stage | instance | encoder | safe samples/s | soft samples/s | safe/soft |",
              "|---|---|---|---:|---:|---:|"]
        def rate(x):
            # A rate of 0.33 samples/s must not render as "0" next to a genuine zero in the
            # capability column — those mean different things and the table is about the difference
            return f"{x:,.0f}" if x >= 10 else f"{x:,.2f}"
        for r in worst:
            M.append(f"| {r['stage']} | `{r['instance']}` | {r['encoder']} | "
                     f"{rate(r['safe_sps'])} | {rate(r['soft_sps'])} | "
                     f"{r['ratio_safe_over_soft']:.3f}x |")

    cg = gm(ctrl_cms)
    M += ["", "## Control — what should NOT have moved", "",
          "Only GPAP's portfolio changed, so CMSGen's throughput and the formula sizes are a "
          "negative control on the pairing. If either moved, the `soft` effect would be "
          "confounded with a machine difference and nothing above would be valid.", "",
          f"* CMSGen throughput, soft arm / safe arm: **{cg:.3f}x** (n={len(ctrl_cms)}) — "
          + ("consistent with VM-to-VM noise on the same instance type."
             if cg and 0.9 <= cg <= 1.1 else
             "**OUT OF RANGE — the two arms are not comparable, do not quote the numbers above.**"),
          f"* Deterministic fields (clauses `M`, `|support|`) differing between arms: "
          f"**{len(ctrl_mismatch)}**"
          + (" — " + ", ".join(f"`{x}`" for x in ctrl_mismatch[:8]) if ctrl_mismatch
             else " (identical encodings on every paired cell, as required)."),
          "", "## Reading this", "",
          "The honest statement is not a single number. Enforcing uniformity costs "
          f"**{gm(allr):.2f}x** of throughput averaged over the whole published matrix, but the "
          "cost is concentrated: it is ~1.0x wherever the bandit would not have chosen `soft` "
          "anyway, and "
          + (f"**{gm(wonr):.2f}x** where it would have" if wonr else "larger where it would have")
          + ". Against that, `soft` fails an exact-oracle uniformity test 5 times out of 5, and "
          "a sampler is a tool whose output distribution is the product. We keep the guarantee "
          "and report the price."]

    T = ["% GENERATED by eval/artifact/soft_price_tables.py — do not edit by hand.",
         r"\begin{table}[t]", r"  \centering", r"  \small",
         r"  \caption{\textbf{The price of the uniformity guarantee.} The entire "
         r"GPAP-vs-CMSGen matrix run twice on the same machine, differing only in whether the "
         r"portfolio may select \texttt{soft} (soft global lookahead), which an exact-oracle "
         r"test rejects 5/5. Ratios are safe$/$soft, so $<1$ favours the rejected configuration. "
         r"The all-cells column is diluted by construction --- where the bandit would not choose "
         r"\texttt{soft}, both arms run identical configurations and contribute $1.0\times$ --- "
         r"so the column restricted to cells it did choose is the real cost.}",
         r"  \label{tab:softprice}",
         r"  \begin{tabular}{@{}lllrrrrr@{}}", r"    \toprule",
         r"    & & & & \multicolumn{2}{c}{safe$/$soft} & & \\",
         r"    \cmidrule(lr){5-6}",
         r"    family & depth & targets & cells & all & \texttt{soft} chosen & "
         r"$n$ & cap.\ lost \\", r"    \midrule"]
    for s in per_stage:
        nw = sum(1 for r in rows if r["stage"] == s["stem"] and r["soft_won_bandit"])
        ga = f"${s['gm_all']:.3f}\\times$" if s["gm_all"] else "---"
        gw = f"${s['gm_won']:.3f}\\times$" if s["gm_won"] else "---"
        T.append(f"    {tex_escape(s['label'])} & ${s['depth']}$ & {s['target']} & "
                 f"{s['cells']} & {ga} & {gw} & {nw} & {len(s['cap_loss'])} \\\\")
    T += [r"    \midrule",
          f"    \\textbf{{overall}} & & & {len(rows)} & "
          f"$\\mathbf{{{gm(allr):.3f}\\times}}$ & "
          + (f"$\\mathbf{{{gm(wonr):.3f}\\times}}$" if wonr else "---")
          + f" & {n_won} & {len(loss)} \\\\",
          r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    # The bucket table is the one worth putting in a paper: the per-family table says the cost
    # varies, this says WHAT it varies with, monotonically, in a quantity you know in advance.
    T += [r"\begin{table}[t]", r"  \centering", r"  \small",
          r"  \caption{\textbf{The cost of the uniformity guarantee is a function of instance "
          r"difficulty.} Paired cells bucketed by the \emph{uniformity-safe} arm's success rate "
          r"--- the quantity available before a portfolio is chosen. The guarantee is nearly free "
          r"on the instances the sampler already handles and expensive only where it does not, "
          r"which is also where \texttt{soft} wins the bandit most often. A portfolio that fell "
          r"back to \texttt{soft} only in the bottom bucket would recover most of the loss while "
          r"leaving the large majority of cells uniform.}",
          r"  \label{tab:softbucket}",
          r"  \begin{tabular}{@{}lrrr@{}}", r"    \toprule",
          r"    safe-arm success rate & cells & safe$/$soft & \texttt{soft} won bandit \\",
          r"    \midrule"]
    for lo, hi, lab in BUCKETS:
        sel = [r for r in rows
               if r["ratio_safe_over_soft"] and r["safe_succ"] is not None
               and lo <= r["safe_succ"] < hi]
        if not sel:
            continue
        g = gm([r["ratio_safe_over_soft"] for r in sel])
        nw = sum(1 for r in sel if r["soft_won_bandit"])
        tex_lab = lab.replace("1e-4", "$10^{-4}$").replace("1e-2", "$10^{-2}$") \
                     .replace("..", "--").replace("<", "$<$").replace(">", "$>$")
        T.append(f"    {tex_lab} & {len(sel)} & $\\mathbf{{{g:.3f}\\times}}$ & "
                 f"{nw}/{len(sel)} \\\\")
    T += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
        open(args.out_md, "w").write("\n".join(M) + "\n")
        print(f"wrote {args.out_md}")
    if args.out_tex:
        os.makedirs(os.path.dirname(args.out_tex) or ".", exist_ok=True)
        open(args.out_tex, "w").write("\n".join(T) + "\n")
        print(f"wrote {args.out_tex}")
    if args.out_json:
        json.dump({"provenance": prov, "per_stage": per_stage, "cells": rows,
                   "gm_all": gm(allr), "gm_won": gm(wonr), "n_cells": len(rows),
                   "n_soft_won": n_won, "capability_lost": loss, "capability_gained": gain},
                  open(args.out_json, "w"), indent=1, default=str)
        print(f"wrote {args.out_json}")
    print(f"\npaired {len(rows)} cells across {len(per_stage)} stages | "
          f"safe/soft all {gm(allr):.3f}x | soft-won {(f'{gm(wonr):.3f}x' if wonr else 'n/a')} "
          f"(n={len(wonr)}) | capability lost {len(loss)} gained {len(gain)}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--safe", required=True, help="sweep dir WITHOUT --allow-soft (the default)")
    ap.add_argument("--soft", required=True, help="sweep dir WITH --allow-soft")
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-tex", default=None)
    ap.add_argument("--out-json", default=None)
    sys.exit(build(ap.parse_args()))


if __name__ == "__main__":
    main()
