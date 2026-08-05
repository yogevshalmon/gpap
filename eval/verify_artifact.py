#!/usr/bin/env python3
"""Verify the experiment artifact is publishable: sound, complete, and self-describing.

Run this before quoting a number anywhere. It is deliberately a CHECKER, not a summariser — it
answers "is there anything here I would be embarrassed to defend?" and exits non-zero if so.

    eval/verify_artifact.py .experiments/cmsgen_l4_2026-08-03 \\
        --rivals .experiments/l4_rivals --uniformity .experiments/uniformity/*.json

Checks, and why each one exists:

  SOUNDNESS   every scored cell must have gpu_sound / cpu_sound true. An unsound row means the
              sampler emitted something that is not a model, which invalidates the whole run --
              this is the one check that can never be waived.
  FAIRNESS    CMSGen must be the PATCHED binary in every stage. Stock CMSGen is timed WITH sample
              emission (~88% of its wall on 27.sk_3_32), which silently inflates every ratio.
  PROVENANCE  each stage must record GPU *and* CPU. CMSGen is single-core, so a ratio without its
              CPU is not interpretable (measured: the same suite reads 48.5x and 113x on two
              machines purely because one host is 0.41x the other).
  AGGREGATION the vacuous-pin instances must not sit inside a sequential geomean, and a geomean
              standing on fewer than 3 instances must be flagged rather than quoted.
  COVERAGE    stages short of their family's instance count are listed, so a partial sweep can
              never be mistaken for a complete one.
  ORACLE      the uniformity campaign must include BOTH controls; without an exact-uniform
              negative and a known-TV positive, "no skew detected" is uncalibrated and unusable.
"""
import argparse
import glob as globmod
import json
import math
import os
import sys

VACUOUS_PINS = {"b02", "b05", "b07", "s526"}


def load(p):
    try:
        return json.load(open(p))
    except Exception:                                   # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmsgen_dir")
    ap.add_argument("--rivals", default=None)
    ap.add_argument("--uniformity", nargs="*", default=[])
    ap.add_argument("--recheck", default=None)
    args = ap.parse_args()

    fail, warn, ok = [], [], []

    # ---------------------------------------------------------------- CMSGen sweep
    stages = sorted(globmod.glob(os.path.join(args.cmsgen_dir, "*.json")))
    tot_cells = tot_unsound = 0
    fam_inst = {}
    for p in stages:
        d = load(p)
        if not d:
            fail.append(f"unreadable JSON: {p}")
            continue
        stem = os.path.basename(p)[:-5]
        rows = [r for r in d.get("results", []) if not r.get("skipped") and not r.get("error")]
        tot_cells += len(rows)
        uns = [r for r in rows if r.get("gpu_sound") is False or r.get("cpu_sound") is False]
        tot_unsound += len(uns)
        for r in uns:
            fail.append(f"UNSOUND cell {stem}/{r.get('name')}/{r.get('enc_key')}")
        if d.get("cmsgen_patched") is not True and any(r.get("cms_sps") for r in rows):
            fail.append(f"{stem}: CMSGen present but NOT patched — ratios inflated by emission")
        modes = {r.get("cms_mode") for r in rows if r.get("cms_mode")}
        if modes - {"nowrite"}:
            fail.append(f"{stem}: CMSGen timed in mode(s) {modes}, expected only 'nowrite'")
        if not (d.get("gpu") or {}).get("name"):
            fail.append(f"{stem}: no GPU recorded")
        if not (d.get("cpu") or {}).get("model"):
            warn.append(f"{stem}: no CPU recorded — a GPAP/CMSGen ratio is not interpretable "
                        f"without it (this stage predates the cpu provenance fix)")
        inst = {r["name"].split(".")[0] for r in rows}
        fam_inst.setdefault(stem.split("-")[0], {})[stem] = len(inst)
        # aggregation traps
        best = {}
        for r in rows:
            k = r["name"].split(".")[0]
            if k not in best or (r.get("ratio") or 0) > (best[k].get("ratio") or 0):
                best[k] = r
        finite = [r["ratio"] for k, r in best.items()
                  if r.get("ratio") and math.isfinite(r["ratio"])
                  and k not in VACUOUS_PINS]
        if 0 < len(finite) < 3:
            warn.append(f"{stem}: geomean stands on only {len(finite)} instance(s) "
                        f"(CMSGen scored zero on the rest) — flag it, do not quote it")
    if tot_unsound == 0 and tot_cells:
        ok.append(f"soundness: {tot_cells} scored cells, 0 unsound")

    # coverage: a stage below its family's max instance count is partial
    for fam, d in sorted(fam_inst.items()):
        mx = max(d.values())
        for stem, n in sorted(d.items()):
            if n < mx:
                warn.append(f"coverage: {stem} has {n}/{mx} instances (incomplete stage)")

    # ---------------------------------------------------------------- rivals
    if args.rivals:
        for p in sorted(globmod.glob(os.path.join(args.rivals, "rivals_*.json"))):
            d = load(p)
            rows = (d.get("rows") if isinstance(d, dict) else d) or []
            stem = os.path.basename(p)[:-5]
            seen = {}
            for r in rows:
                seen[(r["name"], r.get("variant", "-"))] = r
            if len(seen) != len(rows):
                warn.append(f"{stem}: {len(rows)} rows collapse to {len(seen)} cells "
                            f"(duplicate rows from a resume — de-duplicate before publishing)")
            key = next((k for k in ("demotic", "rcs", "hts")
                        if any(k in r for r in rows)), None)
            errs = sum(1 for r in seen.values() if (r.get(key) or {}).get("error"))
            if errs:
                warn.append(f"{stem}: {errs}/{len(seen)} rival runs failed "
                            f"(reported, not folded in as zeros)")
        ok.append("rivals: error rows counted, never treated as zero")

    # ---------------------------------------------------------------- uniformity oracle
    upaths = []
    for g in args.uniformity:
        upaths += sorted(globmod.glob(g))
    if upaths:
        names = set()
        unsound = 0
        for p in upaths:
            d = load(p)
            for r in (d or {}).get("rows", []):
                for nm, s in (r.get("samplers") or {}).items():
                    names.add(nm)
                    unsound += s.get("unsound") or 0
        neg = [n for n in names if n.startswith("ctl:uniform")]
        pos = [n for n in names if n.startswith("ctl:tv")]
        if not neg:
            fail.append("uniformity: no exact-uniform NEGATIVE control — "
                        "a 'no skew' result is uncalibrated")
        if not pos:
            fail.append("uniformity: no known-TV POSITIVE control — power is unmeasured")
        if unsound:
            fail.append(f"uniformity: {unsound} unsound emissions")
        else:
            ok.append(f"uniformity: {len(names)} samplers, both controls present, 0 unsound")
        if not any("soft" in n for n in names):
            warn.append("uniformity: no `soft` arm — the reason it was removed is unverified here")

    # ---------------------------------------------------------------- report
    print("=" * 78)
    for m in ok:
        print(f"  OK    {m}")
    for m in warn:
        print(f"  WARN  {m}")
    for m in fail:
        print(f"  FAIL  {m}")
    print("=" * 78)
    print(f"{len(ok)} ok | {len(warn)} warnings | {len(fail)} failures")
    if fail:
        print("\nNOT publishable until the failures above are resolved.")
    elif warn:
        print("\nPublishable, PROVIDED every warning above is stated in the text.")
    else:
        print("\nPublishable.")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
