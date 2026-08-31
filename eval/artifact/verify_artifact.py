#!/usr/bin/env python3
"""Verify the experiment artifact is publishable: sound, complete, and self-describing.

Run this before quoting a number anywhere. It is deliberately a CHECKER, not a summariser — it
answers "is there anything here I would be embarrassed to defend?" and exits non-zero if so.

    eval/artifact/verify_artifact.py <results-dir> \\
        --rivals .experiments/l4_rivals --uniformity .experiments/uniformity/*.json

Checks, and why each one exists:

  SOUNDNESS   no scored cell may REPORT itself unsound. The on-GPU re-check is opt-in
              (`gpap bench --verify-sound`), so most cells simply carry no verdict — that is
              counted, not required. An unsound row means the
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.artifact.paper_tables import sound_failed, sound_unchecked        # noqa: E402

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
    tot_cells = tot_unsound = tot_unchecked = 0
    fam_inst = {}
    for p in stages:
        d = load(p)
        if not d:
            fail.append(f"unreadable JSON: {p}")
            continue
        stem = os.path.basename(p)[:-5]
        if not isinstance(d, dict) or "results" not in d:
            warn.append(f"{p}: not a `gpap bench` stage (no `results`) — skipped, not checked")
            continue
        rows = [r for r in d.get("results", []) if not r.get("skipped") and not r.get("error")]
        tot_cells += len(rows)
        # A row that says UNSOUND is a hard failure. A row that ran NO check is not: the
        # re-check is opt-in (`gpap bench --verify-sound`), it is a self-test rather than part
        # of producing the numbers this file gates, and demanding it would make the artifact
        # depend on a flag nobody needs for a throughput claim. Counted, never required.
        uns = [r for r in rows if sound_failed(r)]
        tot_unsound += len(uns)
        for r in uns:
            fail.append(f"UNSOUND cell {stem}/{r.get('name')}/{r.get('enc_key')}")
        tot_unchecked += sum(1 for r in rows if sound_unchecked(r))
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
        # ---- CMSGen's sample count must be EXACT where it is published -----------------------
        # A killed CMSGen reports its last --statsevery multiple, so `cms_n_samples: 0` can mean
        # "fewer than 1000 in the budget" rather than "none". On the 2026-08-23 campaign that
        # was 174 of 357 rows, 81 of them recorded as zeros and counted as our wins; one of them
        # (b14/abc:fast) really delivers 941 samples, i.e. an ~10-25x LOSS published as an
        # infinite win. The patch's --maxtime makes CMSGen stop itself and print an exact final
        # count (`cms_count_exact`), so a floored count is now a defect, not a fact of life.
        floored = [r for r in rows
                   if r.get("cms_count_exact") is False or
                   (r.get("cms_count_exact") is None and r.get("cms_timed_out"))]
        if floored:
            zeros = [r for r in floored if not (r.get("cms_n_samples") or 0)]
            msg = (f"{stem}: {len(floored)} row(s) carry a FLOORED CMSGen count (killed between "
                   f"--statsevery lines), {len(zeros)} of them recorded as an exact 0")
            (fail if zeros else warn).append(
                msg + " — re-run with a --maxtime-capable cmsgen (scripts/build_cmsgen.sh)")
        # ---- aggregation traps. One cell per (instance, encoder): there is no best-of ---------
        best = {(r["name"].split(".")[0], r.get("enc_key") or r.get("encoder")): r
                for r in rows}
        finite = [r["ratio"] for (k, _e), r in best.items()
                  if r.get("ratio") and math.isfinite(r["ratio"])
                  and k not in VACUOUS_PINS]
        if 0 < len(finite) < 3:
            warn.append(f"{stem}: geomean stands on only {len(finite)} instance(s) "
                        f"(CMSGen scored zero on the rest) — flag it, do not quote it")

        # SYMMETRY of the zero ledger. Reporting the instances CMSGen could not sample while
        # dropping the ones WE could not is the single easiest way to publish a flattering table
        # by accident; it is exactly what the first version of paper_tables.py did, because a
        # GPAP-zero row carries ratio == 0.0 and fell through every branch.
        g0 = [f"{k}/{e or '-'}" for (k, e), r in best.items()
              if (r.get("cms_n_samples") or 0) > 0 and not (r.get("n_models") or 0)]
        b0 = [f"{k}/{e or '-'}" for (k, e), r in best.items()
              if not (r.get("cms_n_samples") or 0) and not (r.get("n_models") or 0)]
        if g0:
            warn.append(f"{stem}: GPAP emitted NOTHING on {len(g0)} instance(s) that CMSGen "
                        f"sampled ({', '.join(sorted(g0))}) — these are losses and must appear "
                        f"in the table, not only CMSGen's zeros")
        if b0:
            warn.append(f"{stem}: {len(b0)} instance(s) defeated both samplers "
                        f"({', '.join(sorted(b0))}) — report as a limit of the comparison")

        # ZERO-SEARCH cells: samples emitted with no decisions and no implications means the
        # support is free after propagation, so the ratio measures the instance. Worse, the
        # cell is about the instance, not the sampler.
        free = [f"{k}/{e or '-'}" for (k, e), r in best.items()
                if (r.get("n_models") or 0) > 0 and not (r.get("decisions_per_s") or 0)
                and not (r.get("implications_per_s") or 0)]
        if free:
            warn.append(f"{stem}: {len(free)} instance(s) sample with ZERO search "
                        f"({', '.join(sorted(free))}) — no decisions, no implications, the "
                        f"support is a free hypercube. Exclude as for vacuous pins")
    if tot_unsound == 0 and tot_cells:
        checked = tot_cells - tot_unchecked
        ok.append(f"soundness: 0 unsound of {tot_cells} scored cells"
                  + (f" ({checked} carry an on-GPU re-check; the rest did not run one, which is "
                     f"the default)" if checked else " (no cell ran the opt-in re-check)"))

    # ---------------------------------------------------------------- the propagation identity
    # Table 3 claims  samples/s ratio == (impl/s ratio) / (impl-per-sample ratio)  on every
    # family. It is algebra, so it cannot fail unless the aggregation is wrong — which is exactly
    # what happened once (columns filtered independently let EPFL `bar`, which propagates hard
    # and emits nothing, into the numerator only, breaking the family by 3x). Re-derive it here
    # from the raw rows rather than trusting the generator that produced the claim.
    ident_bad, ident_ok = [], 0
    for p in stages:
        d = load(p)
        if not d:
            continue
        if not isinstance(d, dict) or "results" not in d:
            continue
        stem = os.path.basename(p)[:-5]
        best = {(r["name"].split(".")[0], r.get("enc_key") or r.get("encoder")): r
                for r in d.get("results", []) if not r.get("skipped") and not r.get("error")}
        quad = []
        for (k, _e), r in best.items():
            if k in VACUOUS_PINS:
                continue
            a, b = r.get("implications_per_s"), r.get("cms_implications_per_s")
            s, t = r.get("gpap_sps"), r.get("cms_sps")
            if all(x and x > 0 for x in (a, b, s, t)):
                quad.append((a, b, s, t))
        if len(quad) < 3:
            continue

        def _gm(xs):
            return math.exp(sum(math.log(x) for x in xs) / len(xs))
        impl = _gm([a / b for a, b, _, _ in quad])
        work = _gm([(a / s) / (b / t) for a, b, s, t in quad])
        samp = _gm([s / t for _, _, s, t in quad])
        if not (0.99 <= (impl / work) / samp <= 1.01):
            ident_bad.append(f"{stem} ({impl / work:,.1f}x vs {samp:,.1f}x)")
        else:
            ident_ok += 1
    if ident_bad:
        fail.append("propagation identity BROKEN on: " + ", ".join(ident_bad)
                    + " — the impl/s and work-per-sample columns are not over the same rows")
    elif ident_ok:
        ok.append(f"propagation identity holds on {ident_ok}/{ident_ok} families "
                  f"(impl/s ratio / work-per-sample ratio == samples/s ratio, within 1%)")

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
