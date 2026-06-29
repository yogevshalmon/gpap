#!/usr/bin/env python3
"""Rigorous end-to-end validation of GPU ELS+vivify reduction on the velev verification niche.

Answers the load-bearing question — "do we actually boost compute?" — by measuring every piece:

  DISCOVERY (the GPU-accelerated work): probe all 2N literals, BCP to fixpoint, collect implications.
    * GPU: GPUInprocessor.discover_implications  -> T_gpu_disc, probes/s
    * CPU: the SAME 2N single-literal BCP probes via a CPU SAT solver's propagate() -> T_cpu_disc
    The compute boost is T_cpu_disc / T_gpu_disc (>1 = GPU faster at the reduction work). On a weak
    RTX 4080 this is the conservative floor; an A100's larger VRAM -> larger K -> more parallelism.

  SOLVE (hardware-independent benefit): plain vs reduced formula, on the RIGHT solvers — CaDiCaL
    1.9.5 AND kissat 4.0.4 (SOTA) — so the speedup isn't a single-solver artifact. Every reduced
    formula is verdict-checked against plain (soundness gate).

  HONEST TOTALS: x_no_gpu = plain/reduced (discovery free); x_w_gpu = plain/(reduced+T_gpu_disc);
    x_w_cpu = plain/(reduced+T_cpu_disc). Reports all three per solver.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/els_compute_boost.py --insts benchmarks/suite_champions.json \
      --device cuda --solvers cadical195,kissat404 --config els+viv+U --output .experiments/boost.json
"""
import argparse, json, os, statistics as st, sys, time

import multiprocessing as _mp

import torch
sys.path.insert(0, ".")
from pysat.formula import CNF
from gpap.solver.inprocess import GPUInprocessor
from gpap.solver.preprocess import equivalent_literal_substitution, subsume_all, unit_closure

_SPAWN = _mp.get_context("spawn")   # spawn (not fork) → safe with a CUDA-initialised parent


def _solve_worker_named(clauses, solver_name, q):
    """Solve in a child process; measure WALL time (fair across solvers, incl. non-incremental
    kissat). Returns (verdict, wall_ms, conflicts|0)."""
    try:
        import time as _t
        from pysat.formula import CNF as _CNF
        from pysat.solvers import Solver as _S
        s = _S(name=solver_name, bootstrap_with=_CNF(from_clauses=clauses), use_timer=True)
        t0 = _t.perf_counter()
        r = s.solve()
        wall = (_t.perf_counter() - t0) * 1e3
        try:
            cf = s.accum_stats().get("conflicts", 0)
        except Exception:  # noqa: BLE001 — kissat has no accum_stats
            cf = 0
        s.delete()
        q.put(("SAT" if r else "UNSAT" if r is False else "UNK", wall, cf))
    except Exception:  # noqa: BLE001
        q.put(("UNK", 0.0, 0))


def solve_time(clauses, nv, solver_name, budget):
    """Solve `clauses` with `solver_name` under a HARD wall-clock kill. Returns (verdict, wall_ms, conf)."""
    q = _SPAWN.Queue()
    p = _SPAWN.Process(target=_solve_worker_named, args=([list(c) for c in clauses], solver_name, q))
    p.start(); p.join(budget)
    if p.is_alive():
        p.terminate(); p.join(3)
        if p.is_alive():
            p.kill(); p.join()
        return "UNK", budget * 1e3, 0
    try:
        return q.get_nowait()
    except Exception:  # noqa: BLE001
        return "UNK", budget * 1e3, 0


def cpu_discovery_time(cnf_path, solver_name, n2):
    """CPU baseline for the discovery primitive: BCP each of the 2N literals to fixpoint with a
    real CPU SAT solver's propagate(), recording implied literals (same work the GPU does).
    Returns (seconds, n_failed, n_edges)."""
    from pysat.solvers import Solver
    cnf = CNF(from_file=cnf_path)
    nv = cnf.nv
    n_failed = n_edges = 0
    t0 = time.perf_counter()
    with Solver(name=solver_name, bootstrap_with=cnf, use_timer=True) as s:
        for v in range(1, nv + 1):
            for lit in (v, -v):
                ok, prop = s.propagate(assumptions=[lit])
                if not ok:
                    n_failed += 1
                else:
                    n_edges += max(0, len(prop) - 1)
    return time.perf_counter() - t0, n_failed, n_edges


def build_reduced(inp, orig, nv, config):
    """Build the reduced formula on CPU from one shared GPU discovery. Returns
    (clauses, stats, gpu_impl_s, gpu_viv_s) — implications (the BCP primitive, compared to CPU)
    and vivification timed SEPARATELY so the GPU-vs-CPU BCP comparison is apples-to-apples."""
    cuda = inp.device.type == "cuda"
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    edges, units, estats = inp.discover_implications()
    if cuda:
        torch.cuda.synchronize()
    gpu_impl_s = time.perf_counter() - t0
    gpu_viv_s = 0.0
    vivified = orig
    if "viv" in config:
        t1 = time.perf_counter()
        vivified, _ = inp.vivify_clauses(orig, max_len=12, asymmetric=True)
        if cuda:
            torch.cuda.synchronize()
        gpu_viv_s = time.perf_counter() - t1
    if config == "els":
        red, _vm, s = equivalent_literal_substitution(orig, edges, nv)
        red = subsume_all(red) if s["status"] == "ok" else [[1], [-1]]
    elif config == "els+viv+U":
        uc, _a, status = unit_closure(vivified, units)
        if status == "unsat":
            red = [[1], [-1]]
        else:
            r2, _vm, s = equivalent_literal_substitution(uc, edges, nv)
            red = subsume_all(r2) if s["status"] == "ok" else [[1], [-1]]
    else:
        raise ValueError(config)
    return red, estats, gpu_impl_s, gpu_viv_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--insts", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--solvers", default="cadical195,kissat404")
    ap.add_argument("--cpu-bcp-solver", default="cadical195", help="CPU solver for the discovery baseline")
    ap.add_argument("--config", default="els+viv+U", choices=["els", "els+viv+U"])
    ap.add_argument("--max-solve-s", type=float, default=60.0)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    solvers = args.solvers.split(",")
    insts = json.load(open(args.insts))[:args.limit]

    print(f"Device {device} | config={args.config} | solvers={solvers} | CPU-BCP={args.cpu_bcp_solver}")
    print("gpuImpl/cpuImpl = the 2N-probe BCP primitive (apples-to-apples); viv timed separately\n")
    hdr = (f"{'instance':<24}{'2N':>7}{'gpuImpl':>8}{'cpuImpl':>8}{'I↑':>5}{'gpuViv':>8}"
           + "".join(f"{s[:6]+'_pl':>10}{s[:6]+'_rd':>10}{'x':>5}" for s in solvers))
    print(hdr); print("-" * len(hdr))
    results = []
    for rec in insts:
        path = rec["path"]
        cnf = CNF(from_file=path); nv = cnf.nv; n2 = 2 * nv
        inp = GPUInprocessor.from_cnf_file(path, device)
        red, estats, gpu_impl_s, gpu_viv_s = build_reduced(inp, [list(c) for c in cnf.clauses], nv, args.config)
        del inp
        if device.type == "cuda":
            torch.cuda.empty_cache()
        cpu_disc_s, nfail, nedg = cpu_discovery_time(path, args.cpu_bcp_solver, n2)
        gpu_disc_s = gpu_impl_s + gpu_viv_s
        row = {"name": rec["name"], "family": rec.get("family"), "nv": nv, "M": len(cnf.clauses),
               "n_probes": n2, "gpu_impl_s": gpu_impl_s, "gpu_viv_s": gpu_viv_s, "cpu_disc_s": cpu_disc_s,
               "impl_speedup": (cpu_disc_s / gpu_impl_s) if gpu_impl_s else None,
               "gpu_probes_per_s": n2 / gpu_impl_s if gpu_impl_s else 0,
               "cpu_probes_per_s": n2 / cpu_disc_s if cpu_disc_s else 0,
               "n_edges_gpu": estats.get("n_edges"), "solvers": {}}
        cells = [f"{n2:>7}{gpu_impl_s*1e3:>7.0f}m{cpu_disc_s*1e3:>7.0f}m{row['impl_speedup'] or 0:>5.1f}{gpu_viv_s*1e3:>7.0f}m"]
        for sv in solvers:
            pres, pms, pconf = solve_time(cnf.clauses, nv, sv, args.max_solve_s)
            rres, rms, rconf = solve_time(red, nv, sv, args.max_solve_s)
            sound = (pres == rres) or "UNK" in (pres, rres)
            xs = (pms / rms) if (rms > 0 and pres == rres) else None
            row["solvers"][sv] = {"plain_ms": pms, "red_ms": rms, "plain_res": pres, "red_res": rres,
                                  "sound": sound, "x_no_gpu": xs,
                                  "x_w_gpu": (pms / (rms + gpu_disc_s * 1e3)) if (rms and pres == rres) else None,
                                  "x_w_cpu": (pms / (rms + cpu_disc_s * 1e3)) if (rms and pres == rres) else None,
                                  "plain_conf": pconf, "red_conf": rconf}
            tag = "" if sound else "!UNSND"
            cells.append(f"{pms:>9.0f}{rms:>10.0f}{(xs or 0):>5.2f}{tag}")
        print(f"{rec['name'][:24]:<24}" + "".join(cells))
        results.append(row)

    def gm(xs):
        xs = [x for x in xs if x and x > 0]
        return st.geometric_mean(xs) if xs else float("nan")
    print("\n" + "=" * 72)
    print(f"DISCOVERY BCP primitive — GPU implications vs CPU {args.cpu_bcp_solver} (same 2N probes):")
    print(f"  geomean speedup {gm([r['impl_speedup'] for r in results]):.2f}x   "
          f"GPU {gm([r['gpu_probes_per_s'] for r in results]):,.0f} probes/s  vs  "
          f"CPU {gm([r['cpu_probes_per_s'] for r in results]):,.0f} probes/s")
    print(f"  (vivify adds geomean {gm([r['gpu_viv_s'] for r in results])*1e3:.0f} ms GPU; "
          f"impl alone {gm([r['gpu_impl_s'] for r in results])*1e3:.0f} ms)")
    print("\nSOLVE speedup (plain/reduced), validated on each solver:")
    for sv in solvers:
        ok = [r for r in results if r["solvers"][sv]["sound"]]
        uns = sum(1 for r in results if not r["solvers"][sv]["sound"])
        print(f"  {sv:<11} x_no_gpu {gm([r['solvers'][sv]['x_no_gpu'] for r in ok]):.3f}  "
              f"x_w_gpu {gm([r['solvers'][sv]['x_w_gpu'] for r in ok]):.3f}  "
              f"x_w_cpu {gm([r['solvers'][sv]['x_w_cpu'] for r in ok]):.3f}  "
              f"improved {sum(1 for r in ok if (r['solvers'][sv]['x_no_gpu'] or 0) > 1)}/{len(ok)}  unsound {uns}")
    print("=" * 72)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        json.dump({"device": str(device), "args": vars(args), "results": results}, open(args.output, "w"), indent=1)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
