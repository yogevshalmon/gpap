#!/usr/bin/env python3
"""Summarise the GPU-rival head-to-heads, for one machine or as a cross-GPU A/B — the rows
`eval/eval_plots.py` draws, plus a pairing of two machines so "does our margin depend on the
card?" has a number. Only cells present on BOTH sides at the same (instance, variant) enter a
paired geomean; the metric is the RIVALS' own, unique solutions / their `model_runtime`
(Recurrent's less its host copy on rows that record `egress_s`)."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.artifact.paper_tables import gm, unique_unmeasured, zero_search    # noqa: E402


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
        # NOT MEASURED is not a zero — one definition, in paper_tables.
        nomeas = unique_unmeasured(g)
        gpap = None if nomeas else (g.get("unique_per_s_est") or 0.0)
        rec = {"gpap": gpap, "p_accept": r.get("p_accept"), "rival": None, "ratio": None,
               "error": rv.get("error"), "kind": "error" if rv.get("error") else None}
        if not rec["error"]:
            rec["rival"] = rv.get("unique_per_s") or 0.0
            # Every cell lands in exactly ONE bucket, so the printed columns SUM to the cell
            # count; the buckets and their order are paper_tables.rival_family's.
            if r.get("pin_check_ok") is False:
                rec["kind"] = "misaligned"              # the two sides ran DIFFERENT constraints
            elif "decisions_per_s" in g and zero_search(g):
                rec["kind"] = "free"                    # no search happened: the instance, not us
            elif nomeas:
                rec["kind"] = "nomeas"
            elif rec["rival"] and gpap:
                rec["ratio"], rec["kind"] = gpap / rec["rival"], "ratio"
            elif gpap:
                rec["kind"] = "rival0"                  # rival silent -> outright win
            elif rec["rival"]:
                rec["kind"] = "gpap0"                   # WE are silent -> outright loss
            else:
                rec["kind"] = "both0"                   # neither produced anything
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
        kinds = [r["kind"] for r in rows.values()]
        ratios = [r["ratio"] for r in rows.values() if r["kind"] == "ratio"]
        only = kinds.count("rival0")
        gz = kinds.count("gpap0")
        both = kinds.count("both0")
        nm = kinds.count("nomeas")
        errs = kinds.count("error")
        fr = kinds.count("free")
        mis = kinds.count("misaligned")
        # Count wins the way paper_tables does: a rival zero is an outright win and belongs in
        # BOTH sides of the fraction, one of ours is a loss, in the denominator only.
        wins = sum(1 for x in ratios if x >= 1) + only
        n = len(ratios) + only + gz
        # The columns must ADD UP to the cell count, or a reader cannot check what was dropped.
        print(f"\n=== {label}   n={len(rows)}  geomean={fmt(gm(ratios), 'x')}  "
              f"GPAP wins {wins}/{n}  rival-scored-0: {only}  GPAP-scored-0: {gz}  "
              f"both-scored-0: {both}  u/s-unmeasured: {nm}  no-search: {fr}  "
              f"pin-misaligned: {mis}  rival-errored: {errs}"
              f"   [{len(ratios)}+{only}+{gz}+{both}+{nm}+{fr}+{mis}+{errs}={len(rows)}]")
        if markdown:
            print("\n| instance | p_acc | rival u/s | GPAP u/s | ratio |\n|---|---|---|---|---|")
        for (name, var), r in rows.items():
            tag = f"{name} ({var})" if var not in ("-", None) else name
            pa = f"{r['p_accept']:.1e}" if r["p_accept"] else "—"
            if r["kind"] == "error":
                rival, ratio = f"FAILED: {r['error'][:30]}", "—"
            elif r["kind"] == "misaligned":
                rival, ratio = "—", "MISALIGNED"
            elif r["kind"] == "free":
                rival, ratio = fmt(r["rival"]), "no search"
            elif r["kind"] == "nomeas":
                rival, ratio = fmt(r["rival"]), "not measured"
            elif r["kind"] == "rival0":
                rival, ratio = "0 (found nothing)", "GPAP only"
            elif r["kind"] == "gpap0":
                rival, ratio = fmt(r["rival"]), "GPAP found nothing"
            elif r["kind"] == "both0":
                rival, ratio = "0 (found nothing)", "neither"
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
