#!/usr/bin/env python3
"""Compare a re-measured campaign against the frozen reference, cell by cell.

    eval/artifact/compare_campaign.py $GPAP_RESULTS/paper/data .experiments/repro

THE PROBLEM THIS SOLVES. A reviewer's GPU is not ours, so "did you get our numbers?" has no
useful yes. Rates move with the card — the same engine measured 1.822x on an A100 and 0.838x on
an L4 against one 4080 (docs/findings/CROSS_GPU_FINDINGS.md) — and quoting a geomean back at a
different machine tests the machine, not the artifact. What must survive the hardware is stated
here instead, in three tiers:

* **Deterministic — must match exactly.** The formula each cell was measured on (`n_vars`,
  `n_clauses`, `support`, the encoder) and the constraint alignment (`pin_check_ok`). These are
  functions of the input and the code. A difference means the two runs were not sampling the
  same thing, and every comparison below it is void. Usually it means a different `abc` build:
  the encoder is pinned by scripts/build_abc.sh for exactly this reason.
* **Safety — must hold in the FRESH run, on its own.** No emitted model may fail the on-device
  soundness re-check, and no kept row may fail the replay through the original netlist. This one
  needs no reference at all: it is the claim "every sample is a solution", re-derived on the
  reviewer's machine.
* **Structure — reported, and the point of the exercise.** Which tool won each cell, and how
  each cell was classified. A reproduction agrees when it reaches the same verdicts, whatever
  the clock said.

Rates are printed as fresh/reference per stage — the card's speed factor, evidence about the
hardware — and never asserted.

TWO WAYS A CELL CARRIES NO EVIDENCE, both counted rather than folded into agreement:
`not_remeasured` (the reviewer pinned a recorded rival failure with --honour-recorded-failures)
and a walked-down grant (this card could not run the rival at its recorded batch, so the rival
was handicapped here and our ratio on that cell is inflated).

Exit code is non-zero only for a deterministic mismatch or a soundness failure.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from eval.artifact.paper_tables import (RIVALS, gm, rival_cells,  # noqa: E402
                                        rival_cms_cells)

#: Deterministic per-cell fields: (label, reader). A mismatch here is the real alarm.
DETERMINISTIC = [
    ("n_vars", lambda r: (r.get("cnf") or r.get("gpap") or {}).get("n_vars")),
    ("n_clauses", lambda r: (r.get("cnf") or r.get("gpap") or {}).get("n_clauses")),
    ("support", lambda r: r.get("support")),
    ("encoder", lambda r: (r.get("gpap") or {}).get("encoder")),
    ("pin_check_ok", lambda r: r.get("pin_check_ok")),
]


def rows_of(path):
    if not os.path.isfile(path):
        return None
    payload = json.load(open(path))
    return (payload.get("rows") if isinstance(payload, dict) else payload) or []


def by_cell(rows):
    return {(r["name"], r.get("variant", "-")): r for r in rows if r.get("name")}


def winners(cells):
    """{cell: "us"|"them"|bucket} from a classifier's yield — who the cell says was faster."""
    out = {}
    for name, variant, bucket, ours, theirs in cells:
        if bucket != "ok":
            out[(name, variant)] = bucket
        else:
            out[(name, variant)] = "us" if (ours or 0) > (theirs or 0) else "them"
    return out


def rate_pairs(ref_cells, new_cells, which):
    """[(cell, ref_rate, new_rate)] over cells scored `ok` in BOTH — the speed factor's input."""
    idx = 0 if which == "ours" else 1
    a = {(n, v): (o, t) for n, v, b, o, t in ref_cells if b == "ok"}
    b = {(n, v): (o, t) for n, v, bb, o, t in new_cells if bb == "ok"}
    return [(k, a[k][idx], b[k][idx]) for k in sorted(set(a) & set(b))
            if a[k][idx] and b[k][idx]]


def compare_stage(stem, key, ref_rows, new_rows, report):
    ref, new = by_cell(ref_rows), by_cell(new_rows)
    shared = sorted(set(ref) & set(new))
    report["cells"][stem] = {"reference": len(ref), "fresh": len(new), "shared": len(shared),
                             "missing": sorted(f"{n}|{v}" for n, v in set(ref) - set(new))}
    if not shared:
        report["notes"].append(f"{stem}: no shared cells — nothing to compare")
        return

    # 1. deterministic
    for cell in shared:
        for label, read in DETERMINISTIC:
            a, b = read(ref[cell]), read(new[cell])
            if a is None or b is None:           # one side never recorded it: not a mismatch
                continue
            if a != b:
                report["hard"].append(
                    f"{stem} {cell[0]}: {label} differs — reference {a!r}, fresh {b!r}. The two "
                    f"runs did not measure the same formula (check the abc pin: "
                    f"scripts/build_abc.sh)")

    # 2. safety, in the fresh run alone
    for cell in sorted(new):
        g = new[cell].get("gpap") or {}
        if g.get("sound_check") == "UNSOUND":
            report["hard"].append(f"{stem} {cell[0]}: UNSOUND — an emitted model falsifies a "
                                 f"clause ({g.get('unsound_rows')} rows)")
        if g.get("sim_check") == "BAD":
            report["hard"].append(f"{stem} {cell[0]}: sim_check BAD — a kept sample does not "
                                 f"reproduce the pinned outputs through the original circuit")
    checked = sum(1 for c in new if (new[c].get("gpap") or {}).get("sound_check") == "ok")
    simmed = sum(1 for c in new if (new[c].get("gpap") or {}).get("sim_check") == "ok")
    report["safety"][stem] = {"sound_ok": checked, "sim_ok": simmed, "rows": len(new)}

    # 3. structure: the two tables' own classifiers, on both campaigns
    for label, classify in (("vs_cmsgen", rival_cms_cells), ("vs_rival", rival_cells)):
        a = winners(classify([ref[c] for c in shared], key))
        b = winners(classify([new[c] for c in shared], key))
        both = sorted(set(a) & set(b))
        agree = [c for c in both if a[c] == b[c]]
        flips = [(c, a[c], b[c]) for c in both if a[c] != b[c]]
        report["structure"].setdefault(stem, {})[label] = {
            "cells": len(both), "agree": len(agree),
            "differ": [f"{c[0]}: {x} -> {y}" for c, x, y in flips]}

    # 4. rates, reported only
    for label, classify, which in (("gpap", rival_cms_cells, "ours"),
                                   ("cmsgen", rival_cms_cells, "theirs"),
                                   ("rival", rival_cells, "theirs")):
        pairs = rate_pairs(list(classify([ref[c] for c in shared], key)),
                           list(classify([new[c] for c in shared], key)), which)
        if pairs:
            report["rates"].setdefault(stem, {})[label] = {
                "n": len(pairs), "factor": gm([b / a for _c, a, b in pairs])}

    # 5. was the rival granted what the reference granted it?
    walked, pinned_fail, same = [], [], 0
    for cell in shared:
        rv_ref, rv_new = ref[cell].get(key) or {}, new[cell].get(key) or {}
        if rv_new.get("not_remeasured"):
            pinned_fail.append(cell[0])
            continue
        if rv_ref.get("batch") and rv_new.get("batch"):
            if rv_new["batch"] == rv_ref["batch"]:
                same += 1
            else:
                walked.append(f"{cell[0]}: {rv_ref['batch']:,} -> {rv_new['batch']:,}")
    report["grants"][stem] = {"honoured": same, "walked_down": walked,
                              "not_remeasured": sorted(pinned_fail)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", help="the frozen campaign's rows (e.g. $GPAP_RESULTS/paper/data)")
    ap.add_argument("fresh", help="the re-measured campaign's directory")
    ap.add_argument("--json", metavar="FILE", default=None, help="also write the report as JSON")
    args = ap.parse_args()

    report = {"reference": os.path.abspath(args.reference), "fresh": os.path.abspath(args.fresh),
              "hard": [], "notes": [], "cells": {}, "safety": {}, "structure": {}, "rates": {},
              "grants": {}}
    ran = 0
    for stem, key, _tool, _task in RIVALS:
        ref = rows_of(os.path.join(args.reference, f"rivals_{stem}.json"))
        new = rows_of(os.path.join(args.fresh, f"rivals_{stem}.json"))
        if ref is None:
            continue
        if new is None:
            report["notes"].append(f"rivals_{stem}: not in the fresh run — stage not attempted")
            continue
        compare_stage(f"rivals_{stem}", key, ref, new, report)
        ran += 1

    print("=" * 78)
    print(f"reference : {report['reference']}")
    print(f"fresh     : {report['fresh']}")
    if not ran:
        print("no stage present in both — nothing compared")
        return 1
    for stem in sorted(report["cells"]):
        c = report["cells"][stem]
        print(f"\n-- {stem}: {c['shared']} shared cells "
              f"(reference {c['reference']}, fresh {c['fresh']})")
        if c["missing"]:
            print(f"   not re-run: {', '.join(c['missing'][:8])}"
                  + (" ..." if len(c["missing"]) > 8 else ""))
        s = report["safety"][stem]
        print(f"   safety   : {s['sound_ok']}/{s['rows']} rows soundness-ok, "
              f"{s['sim_ok']}/{s['rows']} replayed through the original circuit")
        for label, v in sorted(report["structure"].get(stem, {}).items()):
            verdict = "AGREES" if v["agree"] == v["cells"] else "differs"
            print(f"   {label:<9}: {v['agree']}/{v['cells']} cells reach the same verdict "
                  f"({verdict})")
            for d in v["differ"][:6]:
                print(f"                {d}")
        r = report["rates"].get(stem, {})
        if r:
            print("   speed    : " + "  ".join(
                f"{k} x{v['factor']:.2f} (n={v['n']})" for k, v in sorted(r.items())
                if v["factor"]) + "   [this card vs the reference card, NOT a claim]")
        g = report["grants"][stem]
        print(f"   grants   : {g['honoured']} honoured at the recorded batch"
              + (f", {len(g['walked_down'])} WALKED DOWN" if g["walked_down"] else "")
              + (f", {len(g['not_remeasured'])} pinned failures not re-measured"
                 if g["not_remeasured"] else ""))
        for w in g["walked_down"][:6]:
            print(f"                walked down: {w}")

    walked = sum(len(g["walked_down"]) for g in report["grants"].values())
    pinned = sum(len(g["not_remeasured"]) for g in report["grants"].values())
    print("\n" + "=" * 78)
    if walked:
        print(f"WARNING: {walked} cells could not run the rival at its recorded batch on this "
              f"card.\n         The rival was handicapped there and our ratio on those cells is "
              f"inflated.\n         The reference card is an NVIDIA L4 (24 GB); every pinned "
              f"Recurrent rung sits\n         at its memory ceiling, so a smaller card walks "
              f"down on most of them.")
    if pinned:
        print(f"NOTE: {pinned} cells carry a recorded rival failure that was pinned, not "
              f"re-measured\n      (--honour-recorded-failures). They are evidence in no "
              f"direction; drop the flag\n      to re-derive them at the full initialization "
              f"budget.")
    for note in report["notes"]:
        print(f"NOTE: {note}")
    for h in report["hard"]:
        print(f"FAIL: {h}")
    ok = not report["hard"]
    print(("\nPASS — same formulas, sound samples, same verdicts." if ok else
           f"\nFAILED — {len(report['hard'])} deterministic or soundness failures above."))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
        print(f"report: {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
