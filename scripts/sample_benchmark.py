#!/usr/bin/env python3
"""Benchmark the GPU BCP-completion sampler (gpap.solver.sampler.GPUSampler).

The load-bearing questions for whether sampling is a GPAP killer app:

  THROUGHPUT  — valid models/sec (samples_per_s). GPAP's engine peaks ~1.5M shallow
                sets/s; a sampler that stays near that on structured SAT formulas is the win.
  DEPTH       — mean/max decision rounds. The latency tell: heavily-propagating structured
                formulas need few decisions (shallow → throughput-bound = good); random-like
                formulas need ~N decisions (deep → latency-bound = bad fit).
  SUCCESS     — fraction of trajectories that reach a model vs. die in a conflict. Drops as
                decisions_per_round (B) rises (random restriction over-constrains).
  DIVERSITY   — distinct/total models (are we drawing varied solutions, not one repeatedly).
  SOUNDNESS   — every collected model is independently checked against the DIMACS clauses.

Per instance we first do a quick CaDiCaL SAT check (sampling only applies to SAT); UNSAT/UNK
instances are reported and skipped. Sweeps B ∈ --b-values. Optional --cpu-baseline gives a
reference "distinct models/sec" from CaDiCaL with blocking clauses (throughput reference, not
the same distribution).

Usage:
  PYTHONPATH=. .venv/bin/python scripts/sample_benchmark.py \
      --cnfs benchmarks/cnf/aa9b67...x9-10002.sat...cnf,... \
      --device cuda --b-values 1,2,4,8 --n-target 200000 --verify-n 500 --output .experiments/sample.json
"""
import argparse, glob, json, os, statistics as st, sys, time
import multiprocessing as _mp

import torch
sys.path.insert(0, ".")
from pysat.formula import CNF
from gpap.solver.sampler import GPUSampler, _satisfies

_SPAWN = _mp.get_context("spawn")


def _sat_worker(clauses, q):
    try:
        from pysat.formula import CNF as _CNF
        from pysat.solvers import Solver as _S
        s = _S(name="cadical195", bootstrap_with=_CNF(from_clauses=clauses))
        r = s.solve()
        s.delete()
        q.put("SAT" if r else "UNSAT" if r is False else "UNK")
    except Exception:  # noqa: BLE001
        q.put("UNK")


def sat_check(clauses, budget_s):
    q = _SPAWN.Queue()
    p = _SPAWN.Process(target=_sat_worker, args=([list(c) for c in clauses], q))
    p.start(); p.join(budget_s)
    if p.is_alive():
        p.terminate(); p.join(2)
        if p.is_alive():
            p.kill(); p.join()
        return "UNK"
    try:
        return q.get_nowait()
    except Exception:  # noqa: BLE001
        return "UNK"


def _cpu_enum_worker(clauses, nv, budget_s, q):
    """CPU reference: emit distinct models via CaDiCaL + blocking clauses for budget_s."""
    try:
        import time as _t
        from pysat.formula import CNF as _CNF
        from pysat.solvers import Solver as _S
        s = _S(name="cadical195", bootstrap_with=_CNF(from_clauses=clauses))
        n = 0
        t0 = _t.perf_counter()
        while _t.perf_counter() - t0 < budget_s:
            if not s.solve():
                break
            m = s.get_model()
            if m is None:
                break
            n += 1
            s.add_clause([-l for l in m])  # block this model
        dt = _t.perf_counter() - t0
        s.delete()
        q.put((n, dt))
    except Exception:  # noqa: BLE001
        q.put((0, budget_s))


def cpu_enum_rate(clauses, nv, budget_s):
    q = _SPAWN.Queue()
    p = _SPAWN.Process(target=_cpu_enum_worker, args=([list(c) for c in clauses], nv, budget_s, q))
    p.start(); p.join(budget_s + 10)
    if p.is_alive():
        p.terminate(); p.join(2)
        if p.is_alive():
            p.kill(); p.join()
        return 0.0
    try:
        n, dt = q.get_nowait()
        return n / dt if dt > 0 else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def resolve_paths(args):
    paths = []
    if args.cnfs:
        paths += [p for p in args.cnfs.split(",") if p]
    if args.suite:
        for rec in json.load(open(args.suite)):
            paths.append(rec["path"])
    if args.glob:
        paths += sorted(glob.glob(args.glob))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnfs", default=None, help="comma-separated CNF paths")
    ap.add_argument("--suite", default=None, help="suite JSON with [{path,...}]")
    ap.add_argument("--glob", default=None, help="glob of CNF paths")
    ap.add_argument("--device", default=None)
    ap.add_argument("--b-values", default="1,2,4,8")
    ap.add_argument("--random-var", action="store_true", default=True)
    ap.add_argument("--first-var", dest="random_var", action="store_false")
    ap.add_argument("--n-target", type=int, default=200_000)
    ap.add_argument("--max-chunks", type=int, default=64)
    ap.add_argument("--max-decision-rounds", type=int, default=500,
                    help="bound decision depth per trajectory (deep formulas otherwise run "
                         "thousands of 1-decision rounds). Trajectories hitting the cap are "
                         "counted as attempts, not models — so success_rate reflects completion.")
    ap.add_argument("--k-cap", type=int, default=16384)
    ap.add_argument("--max-n", type=int, default=None, help="skip instances with more than this many vars")
    ap.add_argument("--verify-n", type=int, default=500, help="independently verify this many models")
    ap.add_argument("--sat-check", action="store_true",
                    help="gate on a CaDiCaL SAT check first (skips UNSAT/UNK). Off by default — "
                         "the sampler's own success_rate already reveals SAT-richness, and hard "
                         "near-unique-solution instances (the wrong sampling target) hang the check.")
    ap.add_argument("--sat-budget", type=float, default=20.0)
    ap.add_argument("--cpu-baseline", action="store_true")
    ap.add_argument("--cpu-budget", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    bvals = [int(b) for b in args.b_values.split(",")]
    paths = resolve_paths(args)
    print(f"Device {device} | B={bvals} | n_target={args.n_target:,} | random_var={args.random_var} "
          f"| k_cap={args.k_cap}")
    print(f"{len(paths)} instance(s)\n")

    results = []
    for path in paths:
        name = os.path.basename(path)
        try:
            cnf = CNF(from_file=path)
        except Exception as e:  # noqa: BLE001
            print(f"{name[:40]:<40} parse error: {e}")
            continue
        nv, M = cnf.nv, len(cnf.clauses)
        if args.max_n is not None and nv > args.max_n:
            print(f"{name[:40]:<40} N={nv:<6} M={M:<7} -> skipped (N>{args.max_n})")
            continue
        verdict = "?"
        if args.sat_check:
            verdict = sat_check(cnf.clauses, args.sat_budget)
            if verdict != "SAT":
                print(f"{name[:40]:<40} N={nv:<6} M={M:<7} -> {verdict} (skipped; need SAT)")
                results.append({"name": name, "nv": nv, "M": M, "verdict": verdict, "runs": []})
                continue

        cpu_rate = cpu_enum_rate(cnf.clauses, nv, args.cpu_budget) if args.cpu_baseline else None
        rec = {"name": name, "nv": nv, "M": M, "verdict": verdict,
               "cpu_distinct_per_s": cpu_rate, "runs": []}
        hdr = (f"{name[:40]:<40} N={nv} M={M} SAT"
               + (f" | CPU(cadical+block) {cpu_rate:,.0f} distinct/s" if cpu_rate is not None else ""))
        print(hdr)
        print(f"    {'B':>2} {'samples/s':>12} {'attempts/s':>12} {'succ%':>6} "
              f"{'rounds(μ/max)':>14} {'distinct/N':>12} {'K':>7} {'sound':>6}")
        for B in bvals:
            smp = GPUSampler.from_cnf_file(path, device, k_cap=args.k_cap, seed=args.seed)
            out = smp.sample(args.n_target, decisions_per_round=B, random_var=args.random_var,
                             max_decision_rounds=args.max_decision_rounds, max_chunks=args.max_chunks,
                             collect=False, verify_clauses=None)
            # independent soundness check on a fresh small draw
            sound = "n/a"
            if args.verify_n > 0:
                vout = smp.sample(args.verify_n, decisions_per_round=B, random_var=args.random_var,
                                  max_decision_rounds=args.max_decision_rounds, max_chunks=4,
                                  collect=False, verify_clauses=cnf.clauses)
                sound = "OK" if vout.get("all_valid", True) else "BAD"
            del smp
            if device.type == "cuda":
                torch.cuda.empty_cache()
            run = {"B": B, **{k: out[k] for k in (
                "samples_per_s", "attempts_per_s", "success_rate", "n_models", "attempts",
                "mean_decision_rounds", "max_decision_rounds", "n_distinct", "n_chunks", "K")},
                "sound": sound}
            rec["runs"].append(run)
            dist_ratio = (out["n_distinct"] / out["n_models"]) if out["n_models"] else 0.0
            print(f"    {B:>2} {out['samples_per_s']:>12,.0f} {out['attempts_per_s']:>12,.0f} "
                  f"{100*out['success_rate']:>5.1f}% {out['mean_decision_rounds']:>6.1f}/"
                  f"{out['max_decision_rounds']:<7} {dist_ratio:>11.2f} {out['K']:>7,} {sound:>6}")
        results.append(rec)
        print()

    # summary (instances where the sampler actually produced models)
    sat = [r for r in results if r["runs"] and any(x["n_models"] > 0 for x in r["runs"])]
    if sat:
        print("=" * 78)
        print(f"SUMMARY over {len(sat)} instance(s) with models produced:")
        for B in bvals:
            sps = [next((x["samples_per_s"] for x in r["runs"] if x["B"] == B), None) for r in sat]
            sps = [x for x in sps if x]
            rounds = [next((x["mean_decision_rounds"] for x in r["runs"] if x["B"] == B), None) for r in sat]
            rounds = [x for x in rounds if x is not None]
            succ = [next((x["success_rate"] for x in r["runs"] if x["B"] == B), None) for r in sat]
            succ = [x for x in succ if x is not None]
            if sps:
                gm = st.geometric_mean(sps)
                print(f"  B={B:<2} geomean {gm:>12,.0f} samples/s | mean rounds "
                      f"{sum(rounds)/len(rounds):>6.1f} | mean succ {100*sum(succ)/len(succ):>5.1f}%")
        print("=" * 78)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        json.dump({"device": str(device), "args": vars(args), "results": results},
                  open(args.output, "w"), indent=1)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
