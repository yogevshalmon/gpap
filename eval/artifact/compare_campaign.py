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
`not_remeasured` (the reviewer pinned a recorded rival failure with --honour-recorded-failures,
so the fresh row is a verbatim copy of the reference's and agreeing with it is a tautology) and
a walked-down grant (this card could not run the rival at its recorded batch, so the rival was
handicapped here and our ratio on that cell is inflated).

WHAT THIS TOOL MUST NEVER DO IS PASS A RUN IT DID NOT VERIFY. Three rules earn their keep:
a field recorded on one side and absent on the other is a MISMATCH, not a skip; a row that
emitted models and carries no soundness verdict FAILS, because "off" also covers
requested-but-never-reached and silence must not read as evidence; and `--expect-cells` fails a
tier that compared fewer cells than it set out to run, which is what a truncated campaign looks
like from here.

Exit code is non-zero for a deterministic mismatch, a soundness failure, or a short run.
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


def one_clock(ref_rows, new_rows):
    """`(ref_rows, new_rows, converted)` with Recurrent's rate on one clock on both sides.

    Its rate has its host copy deducted from 2026-10-12 (RIVAL_FAIRNESS_FINDINGS.md §17c); a
    reference recorded before carries it. Across that change both sides are put on their own
    `runtime_s`, or the speed factor and the near-tie verdicts would measure the metric change
    rather than the card."""
    def deducted(rows):
        return any((r.get("rcs") or {}).get("egress_s") is not None for r in rows)

    def theirs(rows):
        out = []
        for r in rows:
            rcs = r.get("rcs") or {}
            if rcs.get("egress_s") is not None and rcs.get("runtime_s"):
                r = dict(r, rcs=dict(rcs, unique_per_s=rcs["unique"] / rcs["runtime_s"]))
            out.append(r)
        return out

    if deducted(ref_rows) == deducted(new_rows):
        return ref_rows, new_rows, False
    return theirs(ref_rows), theirs(new_rows), True


def compare_stage(stem, key, ref_rows, new_rows, report):
    if key == "rcs":
        ref_rows, new_rows, converted = one_clock(ref_rows, new_rows)
        if converted:
            report["notes"].append(f"{stem}: one side predates the host-copy deduction; "
                                   f"Recurrent compared on its own model_runtime")
    ref, new = by_cell(ref_rows), by_cell(new_rows)
    shared = sorted(set(ref) & set(new))
    report["cells"][stem] = {"reference": len(ref), "fresh": len(new), "shared": len(shared),
                             "missing": sorted(f"{n}|{v}" for n, v in set(ref) - set(new))}
    # Every per-stage slot is filled up front: `main` indexes them unconditionally, and a stage
    # that returns early here used to crash the report it was about to print.
    report["safety"][stem] = {"sound_ok": 0, "sim_ok": 0, "rows": len(new), "delivered": 0}
    report["grants"][stem] = {"honoured": 0, "walked_down": [], "not_remeasured": []}
    if not new:
        report["hard"].append(f"{stem}: the fresh run wrote this stage with ZERO rows — it was "
                              f"attempted and produced nothing")
        return
    if not shared:
        report["hard"].append(f"{stem}: not one of the fresh run's {len(new)} cells appears in "
                              f"the reference — nothing could be compared")
        return

    # 1. deterministic
    for cell in shared:
        for label, read in DETERMINISTIC:
            a, b = read(ref[cell]), read(new[cell])
            if a is None and b is None:          # neither side records it: nothing to compare
                continue
            if a is None or b is None:
                # NOT a skip. A field one campaign recorded and the other did not means one of
                # the two rows was never built — a GPAP error row against a measured one, most
                # often an envelope refusal or an OOM. Silence here used to read as agreement.
                report["hard"].append(
                    f"{stem} {cell[0]}: {label} is recorded "
                    f"{'only in the reference' if b is None else 'only in the fresh run'} "
                    f"({a!r} vs {b!r}) — one of the two rows carries no measurement")
                continue
            if a != b:
                report["hard"].append(
                    f"{stem} {cell[0]}: {label} differs — reference {a!r}, fresh {b!r}. The two "
                    f"runs did not measure the same formula (check the abc pin: "
                    f"scripts/build_abc.sh)")

    # 2. safety, in the fresh run alone
    delivered = 0
    for cell in sorted(new):
        g = new[cell].get("gpap") or {}
        if g.get("sound_check") == "UNSOUND":
            report["hard"].append(f"{stem} {cell[0]}: UNSOUND — an emitted model falsifies a "
                                 f"clause ({g.get('unsound_rows')} rows)")
        if g.get("sim_check") == "BAD":
            report["hard"].append(f"{stem} {cell[0]}: sim_check BAD — a kept sample does not "
                                 f"reproduce the pinned outputs through the original circuit")
        if not (g.get("n_models") or 0):
            continue                             # nothing was emitted, so there is nothing to check
        delivered += 1
        # A cell that DELIVERED models and reports no verdict is not a pass. `off` covers
        # "requested but never reached" as well as "never requested" (`gpap/verdict.py`), and the
        # artifact runs every tier under VERIFY_SOUND=1, so `off` here means the evidence the
        # gate is about to summarise was never produced.
        for field, want in (("sound_check", "the on-device soundness re-check"),
                            ("sim_check", "the replay through the original circuit")):
            if g.get(field) != "ok":
                report["hard"].append(
                    f"{stem} {cell[0]}: {g.get('n_models'):,} models emitted but {field} is "
                    f"{g.get(field)!r} — {want} did not run. Re-run with VERIFY_SOUND=1")
    checked = sum(1 for c in new if (new[c].get("gpap") or {}).get("sound_check") == "ok")
    simmed = sum(1 for c in new if (new[c].get("gpap") or {}).get("sim_check") == "ok")
    report["safety"][stem] = {"sound_ok": checked, "sim_ok": simmed, "rows": len(new),
                              "delivered": delivered}

    # 3. structure: the two tables' own classifiers, on both campaigns.
    # A cell whose rival side was PINNED rather than measured is dropped from the rival
    # comparison: its fresh row is a verbatim copy of the reference's, so it would agree by
    # construction and pad the count with cells that carry no evidence. Its CMSGen comparison
    # survives — that arm really ran.
    tautological = {c for c in shared if ((new[c].get(key) or {}).get("not_remeasured"))}
    for label, classify in (("vs_cmsgen", rival_cms_cells), ("vs_rival", rival_cells)):
        scored = [c for c in shared if not (label == "vs_rival" and c in tautological)]
        a = winners(classify([ref[c] for c in scored], key))
        b = winners(classify([new[c] for c in scored], key))
        both = sorted(set(a) & set(b))
        agree = [c for c in both if a[c] == b[c]]
        flips = [(c, a[c], b[c]) for c in both if a[c] != b[c]]
        # OUR arm scoring in the reference and delivering nothing here is the one flip worth
        # naming on its own. It is not asserted: the tiers give every cell the same 60 s, so a
        # much slower card can genuinely reach zero on a marginal instance. It is also what a
        # broken build looks like, so it is never folded into the flip count silently.
        report["regressed"].extend(
            f"{stem} {c[0]} ({label})" for c, x, y in flips
            if x in ("us", "them") and y in ("gpap0", "gpap_error"))
        report["structure"].setdefault(stem, {})[label] = {
            "cells": len(both), "agree": len(agree),
            "pinned_out": len(tautological) if label == "vs_rival" else 0,
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
    # What the fresh run granted ITSELF, which the reference (measured before the cap existed)
    # does not record. An uncapped reproduction on a bigger card sizes a larger K than the
    # published one while the rival stays pinned, so the absence of a cap is worth saying.
    report["caps"][stem] = sorted({r.get("vram_cap_mib") for r in new.values()
                                   if r.get("vram_cap_mib")})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", help="the frozen campaign's rows (e.g. $GPAP_RESULTS/paper/data)")
    ap.add_argument("fresh", help="the re-measured campaign's directory")
    ap.add_argument("--json", metavar="FILE", default=None, help="also write the report as JSON")
    ap.add_argument("--expect-cells", type=int, default=0, metavar="N",
                    help="the number of cells the tier set out to run. Fewer compared than this "
                         "FAILS: a campaign truncated half way — a reclaimed VM, an OOM, a "
                         "missing benchmark set — otherwise reaches this gate looking like a "
                         "clean pass over whatever it managed to finish")
    args = ap.parse_args()

    report = {"reference": os.path.abspath(args.reference), "fresh": os.path.abspath(args.fresh),
              "hard": [], "notes": [], "cells": {}, "safety": {}, "structure": {}, "rates": {},
              "grants": {}, "caps": {}, "regressed": []}
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
                  f"({verdict})"
                  + (f", {v['pinned_out']} pinned not scored" if v.get("pinned_out") else ""))
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
    compared = sum(c["shared"] for c in report["cells"].values())
    if args.expect_cells and compared < args.expect_cells:
        report["hard"].append(
            f"the tier set out to run {args.expect_cells} cells and only {compared} could be "
            f"compared — the campaign did not finish, so nothing above is a verdict about it")
    print("\n" + "=" * 78)
    caps = sorted({m for v in report["caps"].values() for m in v})
    if caps:
        print(f"VRAM budget capped at {', '.join(f'{m:,} MiB' for m in caps)} — the reference "
              f"card's allowance.")
    elif report["cells"]:
        print("WARNING: the fresh rows record no VRAM cap. K is memory-bound below the batch cap\n"
              "         on large formulas, so on a card bigger than the reference L4 our batch\n"
              "         grew while the rival stayed pinned to its recorded one. Re-run with\n"
              "         reproduce.sh --vram-cap l4 to compare like with like.")
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
    flips = sum(v["cells"] - v["agree"] for st in report["structure"].values()
                for v in st.values())
    scored = sum(v["cells"] for st in report["structure"].values() for v in st.values())
    if report["regressed"]:
        print(f"WARNING: {len(report['regressed'])} cells scored in the reference and delivered "
              f"NOTHING here:\n         " + ", ".join(report["regressed"][:8])
              + ("  ..." if len(report["regressed"]) > 8 else "")
              + "\n         Every tier gives a cell the same 60 s, so a much slower card can "
                "reach zero\n         on a marginal instance — but so does a broken build. "
                "Check `gpap info` first.")
    # The cells whose verdict changed, here too: a tier's SUMMARY.txt is this closing block alone,
    # and the headline below counts them and points at them.
    for stem in sorted(report["structure"]):
        for label, v in sorted(report["structure"][stem].items()):
            for d in v["differ"][:6]:
                print(f"DIFFERS: {stem} {label} {d}")
            if len(v["differ"]) > 6:
                print(f"DIFFERS: {stem} {label} and {len(v['differ']) - 6} more (--json)")
    for note in report["notes"]:
        print(f"NOTE: {note}")
    for h in report["hard"]:
        print(f"FAIL: {h}")
    ok = not report["hard"]
    # The headline says what was actually established. It used to claim "same verdicts"
    # unconditionally, which was true of the run it was written for and false of any run whose
    # structure section was full of flips.
    verdicts = ("same verdicts" if not flips else
                f"{scored - flips} of {scored} verdicts unchanged, {flips} DIFFER (above)")
    print((f"\nPASS — {compared} cells compared: same formulas, sound samples, {verdicts}."
           if ok else
           f"\nFAILED — {len(report['hard'])} deterministic, soundness or coverage failures "
           f"above."))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
        print(f"report: {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
