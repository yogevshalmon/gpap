#!/usr/bin/env python3
"""Re-derive a rival's config tie-break on the card the campaign runs on (`--axis`).

Each rival ships TWO values for its gradient-step count — an argparse default in
`src/pytorch/flags.py` and a different one in its own demo script — and
`eval/rivals/rival_common.py::RIVAL_STEPS` resolves that by taking whichever is better FOR THE
RIVAL. That resolution is a MEASUREMENT, and it was first made on an RTX 4080. This script runs
both candidates on one instance per tool, through the ordinary drivers and at the ordinary
granted batch, so the campaign can show the tie-break holding on its own hardware instead of
importing it from another box.

It is cheap by construction: our own arm is pinned to a token budget here (`--gpap-n 2000`,
`--gpap-time-limit 5`) because only the RIVAL side is under test — the GPAP numbers in the output
JSON are meaningless and are not written into any table.

    eval/rivals/steps_calibration.py --axis steps
    eval/rivals/steps_calibration.py --axis lr --tools recurrent

The `lr` axis exists for the same reason (2026-09-04): both rivals' own run scripts say
`--lr 15e0` and Recurrent's PAPER text says 50, so that tool ships two values for the knob a
gradient method depends on most, and the rule is to run it at whichever is better FOR THE RIVAL.
DEMOTIC ships one, so its `lr` has no tie-break to re-derive.

Exit status is 0 whether or not the tie-break holds; a FLIP is reported loudly and recorded as
`agrees: false`, because the response to one is to change RIVAL_STEPS deliberately, not to have a
sweep silently reconfigure a rival mid-campaign.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval.rivals.rival_common import RIVAL_LR, RIVAL_STEPS         # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# One instance per tool: the same ones the 4080 tie-break was measured on, so the two
# measurements are directly comparable. They are also cheap — this is a direction check.
PROBES = {
    "demotic": dict(inst="c432", args=["--mode", "native", "c432"]),
    "recurrent": dict(inst="s386", args=["s386", "-T", "25"]),
    "hts": dict(inst="(first CNF)", args=[]),
}
# Our arm, pinned to a token budget: only the RIVAL side is under test here. Per tool, because
# the CNF harness has no seed axis (it samples one file) and rejects --seeds.
CHEAP = {
    "demotic": ["--gpap-n", "2000", "--gpap-time-limit", "5", "--seeds", "5"],
    "recurrent": ["--gpap-n", "2000", "--gpap-time-limit", "5", "--seeds", "5"],
    "hts": ["--gpap-n", "2000", "--gpap-time-limit", "5", "--n-cov", "0"],
}


def _rate(rows, key):
    """unique/s of the rival arm in a driver's output JSON (a list, or {rows: [...]})."""
    rows = (rows.get("rows") if isinstance(rows, dict) else rows) or []
    out = []
    for r in rows:
        rv = r.get(key) or {}
        if not rv.get("error") and rv.get("unique_per_s"):
            out.append(rv["unique_per_s"])
    return max(out) if out else 0.0


def probe(tool, value, tmp, hts_cnfs, timeout, axis="steps"):
    """Run one driver once at a fixed value of the axis under test.

    Returns (unique_per_s, wall, error-or-None). `axis` is `steps` (their gradient-step count)
    or `lr` (their learning rate) — the two knobs each rival ships more than one value for.
    """
    flag = f"--{axis}"
    out = os.path.join(tmp, f"{tool}_{axis}_{value}.json")
    py = sys.executable
    if tool == "demotic":
        cmd = [py, "eval/rivals/rivals_demotic.py", *PROBES[tool]["args"],
               flag, str(value), "--circuit-variants", "raw", "--output", out]
        key = "demotic"
    elif tool == "recurrent":
        cmd = [py, "eval/rivals/rivals_recurrent.py", *PROBES[tool]["args"],
               flag, str(value), "--circuit-variants", "raw", "--output", out]
        key = "rcs"
    else:
        cmd = [py, "eval/rivals/rivals_hts.py", hts_cnfs, flag, str(value), "--output", out]
        key = "hts"
    cmd += CHEAP[tool] + ["--timeout", str(timeout)]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if not os.path.isfile(out):
        tail = (p.stderr or p.stdout or "").strip().splitlines()[-1:] or ["no output"]
        return 0.0, wall, tail[0][:160]
    return _rate(json.load(open(out)), key), wall, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--axis", default="steps", choices=["steps", "lr"],
                    help="which knob to re-derive: `steps` (their num_steps, RIVAL_STEPS) or "
                         "`lr` (their learning rate, RIVAL_LR — their run scripts say 15 and "
                         "Recurrent's paper text says 50, so that tool ships two)")
    ap.add_argument("--tools", nargs="+", default=["demotic", "recurrent", "hts"],
                    choices=["demotic", "recurrent", "hts"])
    ap.add_argument("--hts-cnfs", default=None,
                    help="glob of CNFs for the HTS probe. The rivals stage generates them into "
                         "$OUT/hts_cnf; without this the HTS tool is skipped rather than "
                         "silently probed on some other formula")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    hts_cnfs = args.hts_cnfs
    tools = list(args.tools)
    if "hts" in tools and not hts_cnfs:
        print("hts: skipped — pass --hts-cnfs '<dir>/*.cnf' (the rivals stage's hts_cnf dir)")
        tools.remove("hts")
    tmp = tempfile.mkdtemp(prefix="steps_calib_")
    res = {}
    print(f"{'tool':<11}{'steps':>7}{'unique/s':>15}{'wall':>8}   note")
    print("-" * 62)
    table = RIVAL_STEPS if args.axis == "steps" else RIVAL_LR
    for tool in tools:
        if tool not in table:
            print(f"{tool:<11}  no {args.axis} candidates recorded — skipped")
            continue
        pinned, src, other = table[tool]
        arms = {}
        # `other` is None where the tool ships only ONE value for this knob: there is no
        # tie-break to re-derive, and probing a value the authors never use would be us tuning
        # their tool. DEMOTIC is that case for `lr`.
        for val in sorted({pinned} | ({other} if other is not None else set())):
            rate, wall, err = probe(tool, val, tmp, hts_cnfs, args.timeout, axis=args.axis)
            arms[val] = {"unique_per_s": rate, "wall_s": wall, "error": err}
            mark = "  <- pinned" if val == pinned else ""
            print(f"{tool:<11}{val:>7}{rate:>15,.0f}{wall:>7.0f}s{mark}"
                  f"{'   ' + err if err else ''}")
        if len(arms) < 2:
            print(f"{tool:<11}  only one value in their artifacts ({pinned}, {src}) — nothing "
                  f"to tie-break")
        best = max(arms, key=lambda k: arms[k]["unique_per_s"])
        # INCONCLUSIVE is not agreement. If every arm errored or scored nothing, max() just
        # returns the first key and would have printed a clean "tie-break holds" over two failed
        # runs — which is how a broken probe reports success.
        if not any(a["unique_per_s"] for a in arms.values()):
            agrees = None
            print(f"  ** INCONCLUSIVE: {tool} scored nothing at either value "
                  f"({'; '.join(a['error'] for a in arms.values() if a['error']) or 'no error'})")
        else:
            agrees = best == pinned or arms[best]["unique_per_s"] == arms[pinned]["unique_per_s"]
        res[tool] = {"pinned": pinned, "pinned_source": src, "measured_best": best,
                     "agrees": agrees, "arms": arms}
        if agrees is False:
            print(f"  ** FLIP: {tool} is faster at steps={best} "
                  f"({arms[best]['unique_per_s']:,.0f} vs {arms[pinned]['unique_per_s']:,.0f} "
                  f"u/s). RIVAL_STEPS pins {pinned}; change it deliberately, in one place.")
    out_path = args.out or f".experiments/rivals_{args.axis}_calibration.json"
    args.out = out_path
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"generated": time.strftime("%Y-%m-%d %H:%M"), "tools": res},
              open(args.out, "w"), indent=1)
    held = sum(1 for v in res.values() if v["agrees"] is True)
    inconc = sum(1 for v in res.values() if v["agrees"] is None)
    print(f"\nwrote {args.out}  |  tie-break holds: {held}/{len(res)}"
          + (f"  ({inconc} INCONCLUSIVE — neither arm scored)" if inconc else ""))


if __name__ == "__main__":
    main()
