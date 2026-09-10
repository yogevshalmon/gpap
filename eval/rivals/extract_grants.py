#!/usr/bin/env python3
"""Write a campaign's RECORDED GRANTS to one JSON, so a re-run pins them instead of searching.

    eval/rivals/extract_grants.py $GPAP_RESULTS/<campaign> --out eval/rivals/grants/<name>.json

Per rival row (calibration files skipped) the grant is the batch rung it was granted — the
largest ladder rung it completed — plus Recurrent-CircuitSAT's settled `start_point`, read off
the row. A FAILED row is recorded without a batch. The campaign driver re-measures such a cell
rather than copying it (RIVAL_FAIRNESS_FINDINGS.md §17); only the artifact's reproduction may
pin it, opt-in via `--honour-recorded-failures`, and then says so in the row."""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.rivals.rival_common import BATCH_LADDER, grant_key  # noqa: E402

TOOLS = {"demotic": "demotic", "recurrent": "rcs"}      # stage stem prefix -> row key


def _rows(payload):
    return (payload.get("rows") if isinstance(payload, dict) else payload) or []


def _demotic_depth(payload, row):
    """The depth the RIVAL ran at: native mode is T=1 whatever the header's `cycles` says."""
    mode = row.get("mode") or (payload.get("mode") if isinstance(payload, dict) else None)
    ctype = payload.get("circuit_type", "comb") if isinstance(payload, dict) else "comb"
    cyc = row.get("cycles") or (payload.get("cycles") if isinstance(payload, dict) else None)
    return cyc if (mode == "miter" or ctype == "seq") else 1


def extract(campaign_dir):
    """{tool: {grant_key: entry}} plus a header, read off every stage file in the directory."""
    out = {"generated": time.strftime("%Y-%m-%d %H:%M"),
           "campaign": os.path.basename(os.path.abspath(campaign_dir)),
           "rule": "batch = the largest rung of the ladder the rival ran to completion, walked "
                   "down from the top and rejected only on feasibility; start_point = where "
                   "Recurrent's initialization settled. Read off the campaign rows, never "
                   "recomputed. An entry without a batch is a recorded rival failure: the "
                   "campaign driver re-measures it, and only the artifact's reproduction may "
                   "pin it (--honour-recorded-failures), marking every row it does that to.",
           "ladder": list(BATCH_LADDER), "gpu": None, "demotic": {}, "recurrent": {}}
    gpus = set()
    for path in sorted(glob.glob(os.path.join(campaign_dir, "rivals_*.json"))):
        stem = os.path.basename(path)[len("rivals_"):-len(".json")]
        if "calibration" in stem:
            continue
        tool = next((t for t in TOOLS if stem.startswith(t)), None)
        if tool is None:
            continue
        payload = json.load(open(path))
        for r in _rows(payload):
            rv = r.get(TOOLS[tool]) or {}
            name, variant = r.get("name"), r.get("variant", "abc")
            mode = r.get("mode") or (payload.get("mode") if isinstance(payload, dict) else None)
            depth = _demotic_depth(payload, r) if tool == "demotic" else r.get("cycles")
            if not name or mode is None or depth is None:
                continue
            gpu = (r.get("gpu") or {}).get("name")
            if gpu:
                gpus.add(gpu)
            k = grant_key(name, mode, depth, variant)
            if rv.get("error") or not rv.get("batch"):
                e = {"batch": None, "error": (rv.get("error") or "no rival measurement")[:200],
                     "stage": stem}
            else:
                sweep = rv.get("batch_sweep") or []
                ooms = [x["batch"] for x in sweep
                        if "outofmemory" in str(x.get("error") or "").lower()]
                e = {"batch": int(rv["batch"]),
                     "rungs_tried": [x.get("batch") for x in sweep],
                     "oom_min": min(ooms) if ooms else None,
                     "unique_per_s": rv.get("unique_per_s"), "stage": stem}
                if tool == "recurrent":
                    e["start_point"] = rv.get("start_point")
            out[tool][k] = e
    out["gpu"] = sorted(gpus)[0] if len(gpus) == 1 else (sorted(gpus) or None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("campaign_dir")
    ap.add_argument("--out", default=None,
                    help="where to write (default: eval/rivals/grants/<campaign>.json)")
    args = ap.parse_args()
    g = extract(args.campaign_dir)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "grants",
                                   g["campaign"] + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(g, open(out, "w"), indent=1, sort_keys=False)
    n_ok = {t: sum(1 for e in g[t].values() if e.get("batch")) for t in TOOLS}
    n_all = {t: len(g[t]) for t in TOOLS}
    print(f"wrote {out}: gpu {g['gpu']}; "
          + "; ".join(f"{t} {n_ok[t]} pinned / {n_all[t] - n_ok[t]} recorded failures"
                      for t in TOOLS))


if __name__ == "__main__":
    main()
