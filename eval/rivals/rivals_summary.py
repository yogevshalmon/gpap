#!/usr/bin/env python3
"""Summarise the GPU-rival head-to-heads, for one machine or as a cross-GPU A/B.

`eval/eval_plots.py` draws these rows; this prints them, and — the reason it exists — pairs two
machines' runs so the question "does our margin depend on the card?" has a number instead of an
impression.

    eval/rivals/rivals_summary.py .experiments/full_eval/nvidia-geforce-rtx-4080-laptop-gpu
    eval/rivals/rivals_summary.py <4080-dir> <l4-dir> --variant raw --markdown

Only rows present on BOTH sides at the same (instance, variant) enter a paired geomean, so a
sub-stage that one machine did not finish cannot tilt the comparison. `--variant raw` is the
usual flag for that: the expensive stages are often run raw-only on a rented GPU.

The metric is the RIVALS' own: unique solutions / their `model_runtime`, which excludes their
PyTorch model build while GPAP's includes construction. Rival rows that errored (timeout, no
gradient path) are counted and printed, never silently folded in as a zero.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.artifact.paper_tables import gm                                    # noqa: E402


# stem -> (key in the row holding the rival's numbers, human label)
STAGES = {
    "demotic_native": ("demotic", "DEMOTIC — their ISCAS-85 pins"),
    "demotic_miter": ("demotic", "DEMOTIC — our rare pins (miter)"),
    "recurrent_table": ("rcs", "Recurrent-CircuitSAT — their target table"),
    "recurrent_rare": ("rcs", "Recurrent-CircuitSAT — our rare pins"),
    "hts": ("hts", "HTS — our CNFs"),
}


def load_rows(d, stem):
    p = os.path.join(d, f"rivals_{stem}.json")
    if not os.path.isfile(p):
        return []
    payload = json.load(open(p))
    return (payload.get("rows") if isinstance(payload, dict) else payload) or []


def collect(d, stem, key, variant=None):
    """{(instance, variant): {rival, gpap, ratio, p_accept, error}}"""
    out = {}
    for r in load_rows(d, stem):
        var = r.get("variant", "-")
        if variant and var != variant:
            continue
        rv, g = r.get(key) or {}, r.get("gpap") or {}
        gpap = g.get("unique_per_s_est") or 0.0
        rec = {"gpap": gpap, "p_accept": r.get("p_accept"), "rival": None, "ratio": None,
               "error": rv.get("error")}
        if not rec["error"]:
            rec["rival"] = rv.get("unique_per_s") or 0.0
            rec["ratio"] = (gpap / rec["rival"]) if rec["rival"] else None   # None = rival got 0
        out[(r["name"], var)] = rec
    return out


def fmt(x, unit=""):
    if x is None:
        return "—"
    return f"{x:,.1f}{unit}" if x < 1e5 else f"{x:,.0f}{unit}"


def report_one(d, variant, markdown):
    print(f"\n{'#' * 8} {d}   (variant={variant or 'all'})")
    for stem, (key, label) in STAGES.items():
        rows = collect(d, stem, key, variant)
        if not rows:
            continue
        ratios = [r["ratio"] for r in rows.values() if r["ratio"]]
        only = sum(1 for r in rows.values() if r["ratio"] is None and not r["error"] and r["gpap"])
        gz = sum(1 for r in rows.values()
                 if not r["error"] and r["rival"] and not r["gpap"])
        errs = sum(1 for r in rows.values() if r["error"])
        # Count wins the way eval/artifact/paper_tables.py does, or the deck and the paper disagree in
        # print. A cell the rival scored ZERO on is an outright win and belongs in BOTH sides of
        # the fraction; a cell WE scored zero on is an outright loss and belongs in the
        # denominator only. Neither has a finite ratio, so neither enters the geomean. Printing
        # the stricter "finite ratios only" count here is a different, narrower question, and it
        # differed from the published table by exactly the zero-scoring cells.
        wins = sum(1 for x in ratios if x >= 1) + only
        n = len(ratios) + only + gz
        print(f"\n=== {label}   n={len(rows)}  geomean={fmt(gm(ratios), 'x')}  "
              f"GPAP wins {wins}/{n}  rival-scored-0: {only}  GPAP-scored-0: {gz}  "
              f"rival-errored: {errs}")
        if markdown:
            print("\n| instance | p_acc | rival u/s | GPAP u/s | ratio |\n|---|---|---|---|---|")
        for (name, var), r in rows.items():
            tag = f"{name} ({var})" if var not in ("-", None) else name
            pa = f"{r['p_accept']:.1e}" if r["p_accept"] else "—"
            if r["error"]:
                rival, ratio = f"FAILED: {r['error'][:30]}", "—"
            elif r["ratio"] is None:
                rival, ratio = "0 (found nothing)", "GPAP only"
            else:
                rival, ratio = fmt(r["rival"]), fmt(r["ratio"], "x")
            if markdown:
                print(f"| {tag} | {pa} | {rival} | {fmt(r['gpap'])} | {ratio} |")
            else:
                print(f"  {tag:<16}{pa:>10}{rival:>22}{fmt(r['gpap']):>16}{ratio:>14}")


def report_pair(a, b, variant, markdown):
    """Paired cross-GPU table: only (instance, variant) cells both machines produced."""
    na, nb = os.path.basename(a.rstrip("/")), os.path.basename(b.rstrip("/"))
    print(f"\n{'#' * 8} PAIRED   A={na}   B={nb}   (variant={variant or 'all'})")
    if markdown:
        print("\n| stage | paired n | A geomean | B geomean | B/A | A wins | B wins |")
        print("|---|---|---|---|---|---|---|")
    for stem, (key, label) in STAGES.items():
        ra, rb = collect(a, stem, key, variant), collect(b, stem, key, variant)
        keys = [k for k in ra if k in rb]
        pa = [ra[k]["ratio"] for k in keys if ra[k]["ratio"] and rb[k]["ratio"]]
        pb = [rb[k]["ratio"] for k in keys if ra[k]["ratio"] and rb[k]["ratio"]]
        if not pa:
            continue
        ga, gb = gm(pa), gm(pb)
        shift = gb / ga if ga and gb else None
        wa = sum(1 for x in pa if x >= 1)
        wb = sum(1 for x in pb if x >= 1)
        if markdown:
            print(f"| {label} | {len(pa)} | {fmt(ga, 'x')} | {fmt(gb, 'x')} | "
                  f"{fmt(shift, 'x')} | {wa}/{len(pa)} | {wb}/{len(pb)} |")
        else:
            print(f"\n=== {label}   paired n={len(pa)}")
            print(f"    {na}: geomean {fmt(ga, 'x')}  wins {wa}/{len(pa)}")
            print(f"    {nb}: geomean {fmt(gb, 'x')}  wins {wb}/{len(pb)}")
            print(f"    margin shift B/A = {fmt(shift, 'x')}")
        # where the shift comes from: each side's own throughput, not just the ratio
        for k in keys:
            if not (ra[k]["ratio"] and rb[k]["ratio"]):
                continue
            gshift = rb[k]["gpap"] / ra[k]["gpap"] if ra[k]["gpap"] else None
            rshift = rb[k]["rival"] / ra[k]["rival"] if ra[k]["rival"] else None
            if not markdown:
                print(f"      {k[0]:<10} GPAP {fmt(gshift, 'x')}   rival {fmt(rshift, 'x')}"
                      f"   ratio {fmt(ra[k]['ratio'], 'x')} -> {fmt(rb[k]['ratio'], 'x')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="full_eval directories (2 => paired A/B)")
    ap.add_argument("--variant", default=None, help="circuit variant filter, e.g. raw")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()
    for d in args.dirs:
        report_one(d, args.variant, args.markdown)
    if len(args.dirs) == 2:
        report_pair(args.dirs[0], args.dirs[1], args.variant, args.markdown)


if __name__ == "__main__":
    main()
