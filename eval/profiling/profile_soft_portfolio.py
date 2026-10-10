#!/usr/bin/env python3
"""Measure what one portfolio costs another in throughput, on circuit benchmarks.

Named for its ORIGINAL and still-default comparison (safe portfolio vs one that may pick `soft`),
but `--arm-a` / `--arm-b` take any two portfolios — the same interleaved method is what any
portfolio A/B on this box needs. The second use it exists for: `--arm-a circuit --arm-b b8` prices
the uniformity decision, since pinning B=8 puts the shipped pipeline at the exact-uniform TV floor
(0.0402 vs a 0.0399 floor, against the default bandit's 0.1471) and the question is what that
costs. See docs/findings/UNIFORMITY_FINDINGS.md §4c.

`soft` (the in-kernel epsilon-greedy conflict-risk polarity) was removed from CIRCUIT_PORTFOLIO on
because Barbarik rejects it on 64 of the 65 circuit instances it can sample — see docs/findings/UNIFORMITY_FINDINGS.md. That is a
uniformity win with a real throughput price, and the price is an OPEN DECISION, so this script
exists to re-measure it whenever the parameters or the engine change.

METHOD (do not "simplify" this — the numbers are meaningless without it):
  * each circuit is encoded ONCE and both portfolios run back-to-back in the SAME process, so the
    arms see the same CNF, the same GPU state and the same clocks;
  * the arm ORDER ALTERNATES between repetitions, because this box's SM clock swings 615-1710 MHz
    mid-run and a fixed-order before/after comparison cannot resolve anything smaller than that;
  * the reported rate per arm is the BEST over repetitions (drift makes any single run a lower
    bound), and `samples_per_s` is the winning config's steady-state rate, not the e2e blend;
  * VACUOUS-PIN instances (b02/b05/b07/s526 — constant last-cycle outputs at T=25, so pinning is a
    tautology) are measured but excluded from the headline geomean, by the screens that live once
    in `eval/artifact/paper_tables.py`.

  PYTHONPATH=. .venv/bin/python eval/profiling/profile_soft_portfolio.py \
      --glob 'benchmarks/circuits/bench/seq/*.bench' --encoder abc --abc-opt fast --cycles 25
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from gpap import paths                                        # noqa: E402

# Constant-output-cone instances: any sampler gets ~100% success, so their ratio is ~1.00 by
# construction and including them would dilute the headline toward "no effect".
VACUOUS = {"b02", "b05", "b07", "s526"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--glob", default="benchmarks/circuits/bench/seq/*.bench")
    ap.add_argument("--cycles", type=int, default=25)
    ap.add_argument("--encoder", default="abc", choices=["tseitin", "abc"])
    ap.add_argument("--abc-opt", default="fast", choices=["full", "fast", "auto"])
    ap.add_argument("--abc", default=str(paths.tool("abc")))
    ap.add_argument("--target", default="rare", choices=["rare", "random"])
    ap.add_argument("--portfolio", default="circuit", choices=["circuit", "default"],
                    help="base portfolio for the DEFAULT soft-vs-safe arms; ignored when "
                         "--arm-a/--arm-b are given explicitly")
    ap.add_argument("--arm-a", default=None,
                    help="baseline portfolio: circuit|default|b8|b1|s1, optionally +soft "
                         "(default: <--portfolio>+soft)")
    ap.add_argument("--arm-b", default=None,
                    help="comparison portfolio, same syntax (default: <--portfolio>)")
    ap.add_argument("--n-time", type=int, default=50_000)
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--reps", type=int, default=2, help="repetitions with the arm order alternated")
    ap.add_argument("--out", default=None, help="results JSON")
    args = ap.parse_args(argv)

    import torch

    from gpap.circuits.bench import convert_one
    from gpap.sampling.pipeline import portfolio, sample_file

    def _pf(spec):
        base, soft = (spec[:-5], True) if spec.endswith("+soft") else (spec, False)
        return portfolio(base, allow_soft=soft)

    a_name = args.arm_a or f"{args.portfolio}+soft"
    b_name = args.arm_b or args.portfolio
    legacy = (a_name, b_name) == (f"{args.portfolio}+soft", args.portfolio)
    # Legacy key names are kept for the soft pair so recorded JSONs and eval/artifact readers
    # (soft_price_tables.py, paper_tables.py) keep working; any other pair uses the arm names.
    A = "baseline(+soft)" if legacy else a_name
    B = "new(no soft)" if legacy else b_name
    arms = {A: _pf(a_name), B: _pf(b_name)}
    files = sorted(glob.glob(os.path.join(REPO_ROOT, args.glob))
                   if not os.path.isabs(args.glob) else glob.glob(args.glob))
    if not files:
        raise SystemExit(f"no benches matched {args.glob!r}")
    print(f"{len(files)} circuits | {args.encoder}"
          f"{':' + args.abc_opt if args.encoder == 'abc' else ''} | T={args.cycles} "
          f"| target={args.target} | n_time={args.n_time:,} | budget={args.time_limit}s "
          f"| {args.reps} reps (order alternated)\n")

    rows = []
    for i, src in enumerate(files):
        name = os.path.basename(src)[:-6]
        row = {"instance": name, "vacuous": name in VACUOUS}
        tmp = paths.scratch_file(name, ".cnf")
        try:
            t0 = time.perf_counter()
            convert_one(src, tmp, args.cycles, args.seed, encoder=args.encoder,
                        abc_path=args.abc, abc_opt=args.abc_opt, target_mode=args.target)
            row["encode_s"] = round(time.perf_counter() - t0, 1)
        except Exception as e:  # noqa: BLE001 — a failing instance must not kill the sweep
            row["error"] = f"encode: {type(e).__name__}: {e}"
            rows.append(row)
            print(f"[{i+1}/{len(files)}] {name}: ENCODE FAILED — {e}", flush=True)
            os.path.exists(tmp) and os.unlink(tmp)
            continue

        per = {a: [] for a in arms}
        for rep in range(args.reps):
            order = list(arms) if rep % 2 == 0 else list(arms)[::-1]
            for arm in order:
                try:
                    info = sample_file(tmp, torch.device("cuda"), n_time=args.n_time, n_cov=0,
                                       configs=arms[arm], time_limit=args.time_limit,
                                       seed=args.seed)
                    per[arm].append({"sps": info["samples_per_s"], "e2e": info["sps_e2e_total"],
                                     "succ": info["success_rate"], "models": info["n_models"],
                                     "B": info["tuned_b"], "la": str(info["tuned_la"])})
                except Exception as e:  # noqa: BLE001
                    per[arm].append({"error": f"{type(e).__name__}: {e}"})
                torch.cuda.empty_cache()
        for arm, runs in per.items():
            ok = [r for r in runs if "error" not in r]
            row[arm] = (max(ok, key=lambda r: r["sps"]) if ok
                        else {"error": runs[0].get("error", "?")})
        a, b = row[A], row[B]
        if "sps" in a and "sps" in b:
            row["ratio_new_over_base"] = (b["sps"] / a["sps"]) if a["sps"] else None
            row["soft_won_baseline"] = a["la"] == "soft"
        row["arm_a"], row["arm_b"] = a_name, b_name
        r = row.get("ratio_new_over_base")
        print(f"[{i+1}/{len(files)}] {name:<8} {'VACUOUS ' if row['vacuous'] else ''}"
              f"{a_name} {a.get('sps', 0):>11,.0f}/s (B={a.get('B')},la={a.get('la')}) | "
              f"{b_name} {b.get('sps', 0):>11,.0f}/s (B={b.get('B')}) | "
              f"ratio {'n/a' if r is None else round(r, 3)}", flush=True)
        rows.append(row)
        os.path.exists(tmp) and os.unlink(tmp)

    # ------------------------------------------------------------------ summary
    nv = [r for r in rows if not r["vacuous"] and "ratio_new_over_base" in r]
    ok = [r for r in nv if r["ratio_new_over_base"]]
    zero = [r for r in nv if r["ratio_new_over_base"] == 0]
    print(f"\nNON-VACUOUS: {len(nv)} instances, {len(ok)} comparable, "
          f"{len(zero)} where the no-soft arm produced nothing")
    if ok:
        g = math.exp(np.mean([math.log(r["ratio_new_over_base"]) for r in ok]))
        med = float(np.median([r["ratio_new_over_base"] for r in ok]))
        print(f"  geomean ratio {b_name}/{a_name} {g:.3f}  => {b_name} is "
              f"{1/g:.2f}x {'SLOWER' if g < 1 else 'FASTER'}")
        print(f"  median  ratio {med:.3f}")
    if legacy:
        won = [r["instance"] for r in nv if r.get("soft_won_baseline")]
        print(f"  soft won the bandit on: {won or 'nothing'}")
    for r in nv:
        a, b = r[A], r[B]
        if r["ratio_new_over_base"] is not None and r["ratio_new_over_base"] < 0.01:
            print(f"  CAPABILITY LOSS {r['instance']}: {a['models']:,} models @ {a['sps']:,.0f}/s"
                  f" -> {b['models']:,} @ {b['sps']:,.0f}/s")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "rows": rows}, f, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
