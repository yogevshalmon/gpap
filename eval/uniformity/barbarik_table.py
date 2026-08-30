#!/usr/bin/env python3
"""Turn barbarik_gpap.py's JSONL rows into the table a paper can quote.

THE RULE THIS FILE ENFORCES, once, so nothing downstream can drift from it: a row is only counted
if the NEGATIVE CONTROL accepted the same instance. `spur(control)` is the trusted uniform
reference put under test through the same hook; where it is Rejected, CM_test is not measuring
sampler uniformity on that instance (see eval/uniformity/barbarik_screen.py for the mechanism — a
sampling set that is not an independent support makes the model-uniform reference
projection-non-uniform, and the test rejects the reference itself). Scoring a sampler on an
instance the reference fails would publish an instance property as a sampler property.

So every instance lands in exactly one bucket and the buckets SUM to the instance count:
    eligible (control Accepted)  +  control-failed  +  control-errored
and the per-arm Accept/Reject/error counts are reported over the eligible set alone.

`eval/artifact/paper_tables.py` IMPORTS `buckets` and `arm_summary` rather than re-deriving them,
so the paper and this table cannot disagree about which instances count.

Usage:
  PYTHONPATH=. .venv/bin/python eval/uniformity/barbarik_table.py <results.jsonl> [...]
"""
import argparse
import collections
import json
import sys

CONTROL = "spur(control)"


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def by_arm(rows):
    out = collections.defaultdict(dict)
    for r in rows:
        out[r["sampler"]][r["instance"]] = r
    return out


def check_one_setting(rows):
    """Refuse to summarise rows from DIFFERENT (eps, eta, delta) settings in one table.

    The same instance under the CMSGen paper's eps=0.3/eta=1.8/delta=0.1 and under barbarik's own
    eps=0.1/eta=0.9/delta=0.05 are two different tests — 1,872 samples with a [0.05, 0.95] band
    versus 70,596 with a [0.29, 0.71] one. Pooling them would produce an Accept count that means
    nothing, and the row format makes it a single `cat` away."""
    settings = {(r.get("epsilon"), r.get("eta"), r.get("delta")) for r in rows}
    if len(settings) > 1:
        found = ", ".join(f"(eps={e}, eta={t}, delta={d})" for e, t, d in sorted(settings))
        raise ValueError("rows mix Barbarik parameter settings — these are DIFFERENT tests, "
                         f"summarise each separately. Found: {found}")
    return settings.pop() if settings else (None, None, None)


def buckets(rows, control=CONTROL):
    """(eligible, control_rejected, control_errored) instance-name sets. They partition the
    instances the control ran on — assert it, because a screen that silently drops instances is
    how a table starts flattering the tool it measures."""
    ctl = by_arm(rows).get(control)
    if not ctl:
        raise ValueError(f"no {control} rows — the negative control gates everything else")
    eligible = {i for i, r in ctl.items() if r["verdict"] == "Accept"}
    rejected = {i for i, r in ctl.items() if r["verdict"] == "Reject"}
    errored = set(ctl) - eligible - rejected
    assert len(eligible) + len(rejected) + len(errored) == len(ctl)
    return eligible, rejected, errored


def arm_summary(rows, eligible, control=CONTROL):
    """Per-arm counts over the eligible set: {arm: {accept, reject, error, n, bias_mean,
    bias_max, bad}}. `bias` is CM_test's own per-round bucket share; it says how much headroom an
    Accept had, which the verdict alone hides."""
    out = {}
    for arm, seen in sorted(by_arm(rows).items()):
        sub = [seen[i] for i in sorted(eligible) if i in seen]
        if not sub:
            continue
        acc = [r for r in sub if r["verdict"] == "Accept"]
        rej = [r for r in sub if r["verdict"] == "Reject"]
        err = [r for r in sub if r["verdict"] not in ("Accept", "Reject")]
        b = [r["bias"] for r in acc if r.get("bias")]
        out[arm] = {
            "accept": len(acc), "reject": len(rej), "error": len(err), "n": len(sub),
            "bias_mean": (sum(x["bias_max_share_mean"] for x in b) / len(b)) if b else None,
            "bias_max": max((x["bias_max_share_max"] for x in b), default=None),
            "bad": sorted(r["instance"] for r in rej + err),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--control", default=CONTROL)
    args = ap.parse_args()

    rows = load(args.results)
    if not rows:
        sys.exit("no rows")
    try:
        eps, eta, delta = check_one_setting(rows)
        eligible, rejected, errored = buckets(rows, args.control)
    except ValueError as e:
        sys.exit(str(e))
    known = {
        (0.3, 1.8, 0.1): ("the CMSGen paper's setting (FMCAD'21 Table I): 1,872 samples, "
                          "per-round band [0.05, 0.95], rejects TV >= 0.90", "[0.05, 0.95]"),
        (0.1, 0.9, 0.05): ("barbarik's OWN default: 70,596 samples, per-round bands "
                           "[0.29, 0.71] and [0.31, 0.69], rejects TV >= 0.45", "its per-round band"),
    }
    note, band = known.get((eps, eta, delta), ("", "its per-round band"))
    print(f"CM_test at eps={eps} eta={eta} delta={delta}" + (f"\n  {note}" if note else ""))
    total = len(eligible) + len(rejected) + len(errored)

    print(f"instances the negative control ran on: {total}")
    print(f"  eligible (control Accept) : {len(eligible)}")
    print(f"  control REJECTED          : {len(rejected)}  {sorted(rejected)}")
    print(f"  control errored           : {len(errored)}  {sorted(errored)}\n")

    summ = arm_summary(rows, eligible, args.control)
    w = max(len(a) for a in summ) + 2
    print(f"{'arm':<{w}}{'Accept':>8}{'Reject':>8}{'error':>7}{'n':>5}   "
          f"{'bias mean':>10}{'bias max':>10}   rejected / errored")
    for arm, a in summ.items():
        bm = "-" if a["bias_mean"] is None else f"{a['bias_mean']:.3f}"
        bx = "-" if a["bias_max"] is None else f"{a['bias_max']:.3f}"
        print(f"{arm:<{w}}{a['accept']:>8}{a['reject']:>8}{a['error']:>7}{a['n']:>5}   "
              f"{bm:>10}{bx:>10}   {', '.join(a['bad'][:3])}"
              f"{' ...' if len(a['bad']) > 3 else ''}")
    print(f"\n  bias = max bucket share per round; CM_test rejects outside {band}.")
    print("  'error' = the sampler could not fill an order barbarik placed (no verdict either "
          "way).")
    short = {a: len(s) for a, s in by_arm(rows).items() if len(s) != total}
    if short:
        print("\n  NOTE: arms did not all cover every instance:")
        for a, n in sorted(short.items()):
            print(f"    {a}: {n} of {total}")


if __name__ == "__main__":
    main()
