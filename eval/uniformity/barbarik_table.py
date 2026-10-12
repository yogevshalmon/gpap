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

#: WHAT EACH CERTIFIED SETTING COSTS, and what a round of it accepts. One home, because these
#: numbers travel: the paper quotes `under_test`, Table 4b prints both totals, this file's own
#: header prints the band, and `barbarik_screen.py` sizes the reference budget from the round
#: count. They were spelled out in four places and the first of them to move would have been
#: right while the other three stayed plausible.
#:
#: `samples` is every draw a full Accept makes; `under_test` is the subset drawn from the sampler
#: being certified. The gap is the one reference draw per round, and it has been read as a
#: discrepancy, so both are carried rather than one derived on the spot.
SETTINGS = {
    (0.3, 1.8, 0.1): {
        "rounds": 36, "samples": 1_872, "under_test": 1_836, "per_round": "1 + 1 + 50",
        "band": "[0.05, 0.95]", "rejects_tv": 0.90,
        "source": "the parameters CMSGen's own authors certify at (FMCAD'21 Table I)"},
    (0.1, 0.9, 0.05): {
        "rounds": 477, "samples": 70_596, "under_test": None, "per_round": None,
        "band": "[0.29, 0.71] and [0.31, 0.69]", "rejects_tv": 0.45,
        "source": "barbarik's OWN default"},
}


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


def order_budget(rows):
    """The one per-order sampling budget these rows were produced under, or None for rows from
    before the budget existed. Raises on a mix, for the same reason `check_one_setting` does.

    The budget is not a performance detail here: it decides how often an order goes unfilled, and
    unfilled orders are what this campaign reports ("accepted on 60 of 65, 5 no-verdict"). Two
    budgets in one table is therefore a budget comparison wearing a uniformity label — the same
    defect as running two ARMS on two budgets, which is what `--order-seconds` was introduced to
    end. SPUR rows carry None (the budget never reaches the reference) and are ignored here."""
    seen = {r.get("order_seconds") for r in rows} - {None}
    if len(seen) > 1:
        raise ValueError(
            "rows mix PER-ORDER BUDGETS (--order-seconds "
            f"{', '.join(f'{s:g}' for s in sorted(seen))}) — the budget decides the no-verdict "
            "count this table reports, so these are DIFFERENT tests; summarise each separately. "
            "Rows with no budget recorded predate the flag and are not counted as a third.")
    return seen.pop() if seen else None


def check_one_setting(rows):
    """Refuse to summarise rows from DIFFERENT settings in one table: the (eps, eta, delta)
    parameters, and the per-order budget (`order_budget` above).

    The same instance under the CMSGen paper's eps=0.3/eta=1.8/delta=0.1 and under barbarik's own
    eps=0.1/eta=0.9/delta=0.05 are two different tests, by two orders of magnitude in samples and
    by their accept band (`SETTINGS` above has both). Pooling them would produce an Accept count
    that means nothing, and the row format makes it a single `cat` away."""
    settings = {(r.get("epsilon"), r.get("eta"), r.get("delta")) for r in rows}
    if len(settings) > 1:
        found = ", ".join(f"(eps={e}, eta={t}, delta={d})" for e, t, d in sorted(settings))
        raise ValueError("rows mix Barbarik parameter settings — these are DIFFERENT tests, "
                         f"summarise each separately. Found: {found}")
    order_budget(rows)                        # same gate, the other axis of "one test"
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
    cost = SETTINGS.get((eps, eta, delta))
    note = ("" if not cost else
            f"{cost['source']}: {cost['samples']:,} samples"
            + (f" ({cost['under_test']:,} from the sampler under test)"
               if cost["under_test"] else "")
            + f", per-round band {cost['band']}, rejects TV >= {cost['rejects_tv']:.2f}")
    band = cost["band"] if cost else "its per-round band"
    print(f"CM_test at eps={eps} eta={eta} delta={delta}" + (f"\n  {note}" if note else ""))
    # The budget belongs in the header for the same reason (eps, eta, delta) does: it is part of
    # which test this is. Rows from before the flag say so rather than printing nothing, because
    # "no budget recorded" and "30 s" are both facts and only one of them is reproducible.
    ob = order_budget(rows)
    print("  per-order sampling budget: "
          + (f"{ob:g} s for every sampler under test (SPUR exempt: not a tool under test)"
             if ob else "NOT RECORDED — these rows predate --order-seconds"))
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
