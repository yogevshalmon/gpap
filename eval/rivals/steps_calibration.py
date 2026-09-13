#!/usr/bin/env python3
"""Re-derive a rival's config tie-break on the card the campaign runs on (`--axis steps|lr`).

Each rival ships TWO values for its gradient-step count — an argparse default and a different
one in its own demo script — and `rival_common.RIVAL_STEPS` takes whichever is better FOR THE
RIVAL. That resolution is a MEASUREMENT, so this runs both candidates on one instance per tool,
through the ordinary drivers at the ordinary granted batch, with our own arm pinned to a token
budget (only the RIVAL side is under test). The `lr` axis exists for the same reason: both run
scripts say `--lr 15e0` while Recurrent's PAPER says 50. Exit status is 0 either way — a FLIP is
reported loudly and recorded as `agrees: false`, the response being to change RIVAL_STEPS
deliberately rather than let a sweep reconfigure a rival mid-campaign.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval.rivals.rival_common import RIVAL_LR, RIVAL_PAPER_CONFIG, RIVAL_STEPS  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# One instance per tool, the same ones the original tie-break used, so the two are comparable.
PROBES = {
    "demotic": dict(inst="c432", args=["--mode", "native", "c432"]),
    "recurrent": dict(inst="s386", args=["s386", "-T", "25"]),
    "hts": dict(inst="(first CNF)", args=[]),
}
# Our arm, pinned to a token budget: only the RIVAL side is under test.
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


def candidates(tool, axis):
    """Every value the tool's own artifacts give this knob: run script (pinned), argparse
    default, PAPER — comparing only the first two leaves the paper's value never run."""
    table = RIVAL_STEPS if axis == "steps" else RIVAL_LR
    if tool not in table:
        return None, None, set()
    pinned, src, other = table[tool]
    vals = {pinned} | ({other} if other is not None else set())
    paper = (RIVAL_PAPER_CONFIG.get(tool) or {}).get(axis)
    if paper is not None:
        vals.add(paper)
    return pinned, src, vals


def probe(tool, value, tmp, hts_cnfs, timeout, axis="steps", extra=(), cycles=None):
    """Run one driver once at a fixed value of the axis under test; returns (unique_per_s, wall,
    error-or-None). `axis` is `steps` or `lr`, the knobs each rival ships two values for."""
    flag = f"--{axis}"
    tag = f"{tool}_{axis}_{value}" + ("_" + "_".join(str(x) for x in extra) if extra else "")
    out = os.path.join(tmp, tag.replace("-", "") + ".json")
    py = sys.executable
    if tool == "demotic":
        cmd = [py, "eval/rivals/rivals_demotic.py", *PROBES[tool]["args"],
               flag, str(value), *extra, "--circuit-variants", "abc", "--output", out]
        key = "demotic"
    elif tool == "recurrent":
        rargs = list(PROBES[tool]["args"])
        if cycles is not None:
            rargs[rargs.index("-T") + 1] = str(cycles)
        cmd = [py, "eval/rivals/rivals_recurrent.py", *rargs,
               flag, str(value), *extra, "--circuit-variants", "abc", "--output", out]
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
    ap.add_argument("--cycles", type=int, default=None,
                    help="unroll depth for the Recurrent probe (default: the PROBES entry, 25); "
                         "the T=50 stage passes 50 so the tie-break is re-derived at ITS depth")
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
    for tool in tools:
        pinned, src, vals = candidates(tool, args.axis)
        if pinned is None:
            print(f"{tool:<11}  no {args.axis} candidates recorded — skipped")
            continue
        arms = {}
        # Every value the tool's own artifacts give the knob; probing one they never use would be
        # us tuning their tool.
        for val in sorted(vals):
            rate, wall, err = probe(tool, val, tmp, hts_cnfs, args.timeout, axis=args.axis,
                                    cycles=args.cycles)
            arms[val] = {"unique_per_s": rate, "wall_s": wall, "error": err}
            mark = "  <- pinned" if val == pinned else ""
            paper = (RIVAL_PAPER_CONFIG.get(tool) or {}).get(args.axis)
            mark += "  (paper)" if val == paper and val != pinned else ""
            print(f"{tool:<11}{val:>7}{rate:>15,.0f}{wall:>7.0f}s{mark}"
                  f"{'   ' + err if err else ''}")
        # ...and the paper's PAIRED configuration, which no single-axis probe measures.
        pc = RIVAL_PAPER_CONFIG.get(tool)
        if args.axis == "steps" and pc:
            other_axis = "lr"
            rate, wall, err = probe(tool, pc["steps"], tmp, hts_cnfs, args.timeout, axis="steps",
                                    extra=(f"--{other_axis}", str(pc[other_axis])),
                                    cycles=args.cycles)
            arms[f"paper({pc['steps']},{pc['lr']:g})"] = {"unique_per_s": rate, "wall_s": wall,
                                                          "error": err, "steps": pc["steps"],
                                                          "lr": pc["lr"], "source": pc["source"]}
            print(f"{tool:<11}{'paper':>7}{rate:>15,.0f}{wall:>7.0f}s   steps={pc['steps']} "
                  f"lr={pc['lr']:g} ({pc['source']}){'   ' + err if err else ''}")
        if len(arms) < 2:
            print(f"{tool:<11}  only one value in their artifacts ({pinned}, {src}) — nothing "
                  f"to tie-break")
        best = max(arms, key=lambda k: arms[k]["unique_per_s"])
        best = best if not isinstance(best, str) or not best.startswith("paper(") else best
        # INCONCLUSIVE is not agreement: with every arm errored, max() returns the first key.
        if not any(a["unique_per_s"] for a in arms.values()):
            agrees = None
            print(f"  ** INCONCLUSIVE: {tool} scored nothing at either value "
                  f"({'; '.join(a['error'] for a in arms.values() if a['error']) or 'no error'})")
        else:
            agrees = best == pinned or arms[best]["unique_per_s"] == arms[pinned]["unique_per_s"]
        res[tool] = {"pinned": pinned, "pinned_source": src, "measured_best": best,
                     "agrees": agrees, "cycles": args.cycles,
                     "arms": {str(k): v for k, v in arms.items()}}
        if agrees is False:
            print(f"  ** FLIP: {tool} is faster at steps={best} "
                  f"({arms[best]['unique_per_s']:,.0f} vs {arms[pinned]['unique_per_s']:,.0f} "
                  f"u/s). RIVAL_STEPS pins {pinned}; change it deliberately, in one place.")
    # Anchored on the repo, like every other .experiments default in the tree: a cwd-relative
    # one drops the calibration wherever the shell happened to be standing.
    out_path = args.out or os.path.join(REPO, ".experiments",
                                        f"rivals_{args.axis}_calibration.json")
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
