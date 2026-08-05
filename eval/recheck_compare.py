#!/usr/bin/env python3
"""Compare a consistency re-check against the original sweep, cell by cell.

    eval/recheck_compare.py .experiments/cmsgen_l4_2026-08-03 .experiments/recheck

What counts as agreement is the whole question, so it is stated rather than assumed:

* **Deterministic — must match EXACTLY.** `M` (clauses after simplify+mine), `support`, and the
  engine provenance (packed split, propagate variant). These are functions of the instance and
  the code, not of the machine or the clock. Any difference here means the two runs were not
  measuring the same thing, and no rate comparison below it is meaningful.
* **Stochastic-but-stable — compared with a tolerance.** `succ` (success rate). Same seed and
  same config give the same trajectories, so this should be near-identical; drift means the
  sampler saw a different formula or a different config.
* **Noisy by nature — reported, never asserted.** Throughput. The SM clock swings 615-1710 MHz
  *within* a run on this class of hardware and draw noise on low-success instances reaches
  +-40% (docs/BENCHMARKING.md §3), so a rate that moved 30% is not evidence of anything. The
  geomean across many cells is the meaningful figure.

Exit code is non-zero only if a DETERMINISTIC field disagrees — that is the real alarm.
"""
import argparse
import json
import math
import os
import sys

SUCC_TOL = 0.05           # absolute; same seed + same config should be far tighter than this


def gm(xs):
    xs = [x for x in xs if x and x > 0 and math.isfinite(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def cells(path):
    if not os.path.isfile(path):
        return {}
    out = {}
    for r in json.load(open(path)).get("results", []):
        if r.get("skipped") or r.get("error"):
            continue
        out[(r["name"].split(".")[0], r.get("enc_key") or r.get("encoder"))] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("original")
    ap.add_argument("recheck")
    ap.add_argument("--stages", nargs="*",
                    default=["seq-c25-rare", "comb-rare", "epfl-rare", "sk"])
    args = ap.parse_args()

    hard_fail = []
    soft = []
    rates = []
    n_cells = 0
    for stage in args.stages:
        a = cells(os.path.join(args.original, f"{stage}.json"))
        b = cells(os.path.join(args.recheck, f"{stage}.json"))
        shared = sorted(set(a) & set(b))
        if not shared:
            continue
        print(f"\n=== {stage}   {len(shared)} shared cells")
        print(f"  {'instance':<18}{'enc':>10}{'M a/b':>16}{'succ a/b':>16}"
              f"{'gpap/s a':>12}{'gpap/s b':>12}{'b/a':>8}")
        for k in shared:
            ra, rb = a[k], b[k]
            n_cells += 1
            # --- deterministic
            for field in ("M", "support"):
                if ra.get(field) != rb.get(field):
                    hard_fail.append(f"{stage} {k} {field}: {ra.get(field)} != {rb.get(field)}")
            ea, eb = (ra.get("engine") or {}), (rb.get("engine") or {})
            for field in ("split", "mode"):
                if ea.get(field) != eb.get(field):
                    hard_fail.append(f"{stage} {k} engine.{field}: "
                                     f"{ea.get(field)} != {eb.get(field)}")
            # --- stable-ish
            sa, sb = ra.get("succ") or 0.0, rb.get("succ") or 0.0
            if abs(sa - sb) > SUCC_TOL:
                soft.append(f"{stage} {k} succ {sa:.3f} -> {sb:.3f}")
            # --- noisy
            ga, gb = ra.get("gpap_sps") or 0.0, rb.get("gpap_sps") or 0.0
            if ga and gb:
                rates.append(gb / ga)
            mflag = "" if ra.get("M") == rb.get("M") else "  <-- M DIFFERS"
            print(f"  {k[0][:17]:<18}{str(k[1])[:9]:>10}"
                  f"{str(ra.get('M'))+'/'+str(rb.get('M')):>16}"
                  f"{f'{sa:.2f}/{sb:.2f}':>16}{ga:>12,.0f}{gb:>12,.0f}"
                  f"{(gb/ga if ga else float('nan')):>8.2f}{mflag}")

    print(f"\n{'=' * 72}\nCONSISTENCY VERDICT   ({n_cells} shared cells)")
    if hard_fail:
        print(f"  DETERMINISTIC MISMATCH x{len(hard_fail)} — the two runs are NOT the same "
              f"measurement:")
        for m in hard_fail[:15]:
            print("   ", m)
    else:
        print("  deterministic fields (M, support, packed split, propagate variant): ALL MATCH")
    if soft:
        print(f"  success rate moved >{SUCC_TOL} on {len(soft)} cell(s):")
        for m in soft[:10]:
            print("   ", m)
    else:
        print(f"  success rate: within {SUCC_TOL} everywhere")
    if rates:
        print(f"  throughput recheck/original: geomean {gm(rates):.3f}x  "
              f"(min {min(rates):.2f}, max {max(rates):.2f}, n={len(rates)})")
        print("    ^ noisy BY DESIGN — SM clock swings 2.8x within a run; read the geomean only")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
