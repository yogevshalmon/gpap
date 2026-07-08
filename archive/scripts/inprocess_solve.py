#!/usr/bin/env python3
"""Compare plain CaDiCaL vs CaDiCaL + GPAP GPU inprocessing on a CNF.

GPAP discovers entailed short clauses (failed-literal units + binary/ternary clauses over
the variables CaDiCaL is currently branching on) and injects them between conflict-budgeted
segments through PySAT's IPASIR-UP interface. Every injected clause is RUP, so the verdict
must match the plain solve — the harness asserts this.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/inprocess_solve.py \\
        --cnf benchmarks/cnf/<file>.cnf --device cuda --mode pairs --top-k 64

    PYTHONPATH=. .venv/bin/python scripts/inprocess_solve.py \\
        --cnf <file>.cnf --mode triples --segment-conflicts 10000 --time-budget 120
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, ".")
from gpap.solver.cadical_hook import solve_plain, solve_with_gpu_inprocessing


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cnf", required=True, help="Path to DIMACS CNF file")
    ap.add_argument("--device", default=None, help="torch device (cpu/cuda)")
    ap.add_argument("--mode", default="pairs", choices=["probe", "pairs", "triples"],
                    help="Discovery: probe (units only), pairs (+binaries), triples (+ternaries)")
    ap.add_argument("--top-k", type=int, default=64, help="Important variables for pair discovery")
    ap.add_argument("--triples-k", type=int, default=24, help="Important variables for triple discovery")
    ap.add_argument("--segment-conflicts", type=int, default=20000,
                    help="Conflict budget per solve segment (default: 20000)")
    ap.add_argument("--max-clauses-per-round", type=int, default=2000,
                    help="Cap on injected clauses per round (shortest-first)")
    ap.add_argument("--max-rounds-total", type=int, default=None,
                    help="Stop discovery after this many rounds (then segment-solve to finish)")
    ap.add_argument("--time-budget", type=float, default=None,
                    help="Wall-clock budget (s) for the inprocessing solve; UNKNOWN if exceeded")
    ap.add_argument("--observe-pool", type=int, default=None,
                    help="How many variables to observe (default: min(N, max(8*top_k, 512)))")
    ap.add_argument("--importance", default="blend",
                    choices=["blend", "decisions", "activity", "degree"],
                    help="Variable ranking signal (default: blend = decisions, degree tiebreak)")
    ap.add_argument("--patience", type=int, default=3,
                    help="Stop discovery after this many consecutive empty rounds")
    ap.add_argument("--gpap-max-rounds", type=int, default=None, help="GPAP BCP hard round cap")
    ap.add_argument("--gpap-soft-stop-frac", type=float, default=None, help="GPAP soft-stop fraction")
    ap.add_argument("--gpap-min-rounds", type=int, default=0, help="GPAP min rounds before soft-stop")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-baseline", action="store_true", help="Skip the plain CaDiCaL baseline")
    args = ap.parse_args()

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"\nFormula : {os.path.basename(args.cnf)}")
    print(f"Device  : {device}\n")

    plain = None
    if not args.no_baseline:
        print("[1/2] Plain CaDiCaL ...", end=" ", flush=True)
        plain = solve_plain(args.cnf)
        print(f"{plain['result']}  {plain['wall_ms']:.0f} ms  "
              f"(conflicts={plain['solver_stats']['conflicts']:,})")

    print("[2/2] CaDiCaL + GPU inprocessing ...")
    inp = solve_with_gpu_inprocessing(
        args.cnf, device=device, mode=args.mode, top_k=args.top_k, triples_k=args.triples_k,
        segment_conflicts=args.segment_conflicts, max_clauses_per_round=args.max_clauses_per_round,
        max_rounds_total=args.max_rounds_total, time_budget_s=args.time_budget,
        observe_pool=args.observe_pool, importance=args.importance, patience=args.patience,
        gpap_max_rounds=args.gpap_max_rounds, gpap_soft_stop_frac=args.gpap_soft_stop_frac,
        gpap_min_rounds=args.gpap_min_rounds, seed=args.seed, verbose=True,
    )

    # ---- report ----
    def _row(label, a, b):
        print(f"  {label:<22}{a:>16}{b:>20}")

    print("\n" + "=" * 60)
    p = plain or {"result": "-", "wall_ms": float("nan"), "solver_stats": {}}
    ps = p.get("solver_stats", {})
    isr = inp["solver_stats"]
    _row("", "plain", "gpu-inprocess")
    _row("result", p["result"], inp["result"])
    _row("wall (ms)", f"{p['wall_ms']:.0f}" if plain else "-", f"{inp['wall_ms']:.0f}")
    _row("conflicts", f"{ps.get('conflicts','-'):,}" if plain else "-", f"{isr['conflicts']:,}")
    _row("decisions", f"{ps.get('decisions','-'):,}" if plain else "-", f"{isr['decisions']:,}")
    _row("propagations", f"{ps.get('propagations','-'):,}" if plain else "-", f"{isr['propagations']:,}")
    _row("clauses added", "-", f"{inp['clauses_added']:,}")
    _row("  units / bin / tern", "-",
         f"{inp['units_added']}/{inp['binaries_added']}/{inp['ternaries_added']}")
    _row("GPU rounds", "-", f"{inp['rounds']}")
    _row("GPU discover (ms)", "-", f"{inp['gpu_ms']:.0f}")
    if plain and inp["result"] != "UNKNOWN" and inp["wall_ms"] > 0:
        _row("speedup (plain/inproc)", "", f"{p['wall_ms'] / inp['wall_ms']:.2f}x")
    print("=" * 60)

    if plain and plain["result"] != "UNKNOWN" and inp["result"] != "UNKNOWN":
        if plain["result"] != inp["result"]:
            print(f"\n!! VERDICT MISMATCH: plain={plain['result']} inproc={inp['result']} "
                  f"— an injected clause was UNSOUND")
            sys.exit(2)
        print(f"\nVerdict agreement: {inp['result']} ✓")


if __name__ == "__main__":
    main()
