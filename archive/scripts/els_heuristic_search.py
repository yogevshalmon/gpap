#!/usr/bin/env python3
"""Search for an ELS heuristic that beats plain CaDiCaL with a SINGLE config (no portfolio).

The prior ELS finding: speedup is chaotic and uncorrelated with reduction magnitude, so the only
robust win was a per-instance plain-fallback portfolio. This script asks the next question — is
there a *deterministic* reduction heuristic (which equivalences to substitute / which variable to
keep / how vivify stacks) whose single config robustly reduces CaDiCaL's work, on the hardware-
verification niche where ELS did best?

Key metric change: **conflicts, not wall-ms.** CaDiCaL is deterministic, so the conflict count is
noise-free and reproducible even on sub-second instances (wall-ms is pure jitter there). The
target is conflict-ratio = plain_conflicts / config_conflicts > 1 on most instances, with NO
portfolio fallback. ms is still reported for sanity.

Configs (GPU discovery shared once per instance):
  plain        original formula
  els          ELS, representative = smallest variable (the baseline ELS)
  els_deg      ELS, representative = highest-DEGREE variable (keep the central var, drop peripheral)
  els_budget   ELS, only the largest ~half of equivalence classes substituted
  els_deg+vivU asymmetric-vivify → unit-closure → ELS(deg)  (the stacked reduction)

Usage:
  PYTHONPATH=. .venv/bin/python scripts/els_heuristic_search.py \
      --insts benchmarks/suite_structured_big.json --device cuda --max-solve-s 30 \
      --output .experiments/els_heur.json
"""
import argparse, json, math, os, statistics as st, sys, time

import torch
sys.path.insert(0, ".")
from gpap.solver.inprocess import GPUInprocessor
from gpap.solver.preprocess import (
    equivalent_literal_substitution, subsume_all, unit_closure,
)
from scripts.inprocess_suite import timed_solve

CONFIGS = ["plain", "els", "els_deg", "els_budget", "els_deg+vivU"]


def _read(path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0] in "cp%":
                continue
            lits = [x for x in (int(t) for t in line.split()) if x != 0]
            if lits:
                out.append(lits)
    return out


def _degree(clauses):
    deg = {}
    for c in clauses:
        for l in c:
            deg[abs(l) - 1] = deg.get(abs(l) - 1, 0) + 1     # 0-based var → count
    return deg


def discover(path, device, orig, nv, vmax_len):
    inp = GPUInprocessor.from_cnf_file(path, device)
    t0 = time.perf_counter()
    edges, units, estats = inp.discover_implications()
    vivified, _ = inp.vivify_clauses(orig, max_len=vmax_len, asymmetric=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    gpu_ms = (time.perf_counter() - t0) * 1e3
    del inp
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return edges, units, vivified, gpu_ms, estats


def build(orig, edges, units, vivified, nv, deg):
    cfg = {"plain": orig}
    els, _, s0 = equivalent_literal_substitution(orig, edges, nv)
    cfg["els"] = list(_unsat()) if s0["status"] == "unsat" else subsume_all(els)
    elsd, _, sd = equivalent_literal_substitution(orig, edges, nv, rep_score=deg)
    cfg["els_deg"] = list(_unsat()) if sd["status"] == "unsat" else subsume_all(elsd)
    # budget: only the larger half of classes (>= 1)
    nbud = max(1, s0.get("n_classes", 0) // 2)
    elsb, _, sb = equivalent_literal_substitution(orig, edges, nv, max_classes=nbud)
    cfg["els_budget"] = list(_unsat()) if sb["status"] == "unsat" else subsume_all(elsb)
    uc, _assigned, ucs = unit_closure(vivified, units)
    if ucs == "unsat":
        cfg["els_deg+vivU"] = list(_unsat())
    else:
        elsuc, _, s2 = equivalent_literal_substitution(uc, edges, nv, rep_score=deg)
        cfg["els_deg+vivU"] = list(_unsat()) if s2["status"] == "unsat" else subsume_all(elsuc)
    return cfg, {"n_classes": s0.get("n_classes", 0), "vars_removed": s0.get("vars_removed", 0)}


def _unsat():
    return [[1], [-1]]


def _gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--insts", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-solve-s", type=float, default=30.0)
    ap.add_argument("--vivify-max-len", type=int, default=12)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    insts = json.load(open(args.insts))

    print(f"Device {device} | {len(insts)} instances | metric = conflict-ratio (plain/config), "
          f"deterministic; ms shown for sanity\n")
    hdr = f"{'instance':<32}{'fam':<9}{'pConf':>9}{'elsV':>6}" + "".join(f"{c:>13}" for c in CONFIGS[1:])
    print(hdr); print("-" * len(hdr))

    results = []
    for rec in insts:
        path = rec["path"]; orig = _read(path)
        if not orig:
            continue
        nv = max(abs(l) for c in orig for l in c)
        pres, pms, pconf = timed_solve(orig, nv, args.max_solve_s)
        if pres == "UNK":
            print(f"{rec['name'][:32]:<32}{rec['family']:<9}  plain UNK (skip)")
            continue
        edges, units, vivified, gpu_ms, estats = discover(path, device, orig, nv, args.vivify_max_len)
        cfg, info = build(orig, edges, units, vivified, nv, _degree(orig))

        row = {"name": rec["name"], "family": rec["family"], "nv": nv, "M": len(orig),
               "plain_conf": pconf, "plain_ms": pms, "gpu_ms": gpu_ms,
               "els_vars_removed": info["vars_removed"], "cfg": {}}
        for name in CONFIGS[1:]:
            r, ms, conf = timed_solve(cfg[name], nv, args.max_solve_s)
            agree = r == pres
            cr = (pconf / conf) if (agree and conf > 0) else (0.001 if r == "UNK" else None)
            row["cfg"][name] = {"result": r, "ms": ms, "conf": conf, "agree": agree,
                                "conf_ratio": cr, "ms_ratio": (pms / ms) if (agree and ms > 0) else None}
        results.append(row)

        def cell(n):
            c = row["cfg"][n]
            if not c["agree"]:
                return "UNK" if c["result"] == "UNK" else "UNSND"
            return f"{c['conf_ratio']:.2f}/{c['ms_ratio']:.2f}" if c["conf_ratio"] else "-"
        print(f"{rec['name'][:32]:<32}{rec['family']:<9}{pconf:>9}{info['vars_removed']:>6}"
              + "".join(f"{cell(n):>13}" for n in CONFIGS[1:]))

    if results:
        print("\n" + "=" * 78)
        print("geomean conflict-ratio (>1 = fewer conflicts than plain) | ms-ratio | improved | TO-regress")
        for n in CONFIGS[1:]:
            crs = [r["cfg"][n]["conf_ratio"] for r in results]
            mrs = [r["cfg"][n]["ms_ratio"] for r in results]
            imp = sum(1 for r in results if (r["cfg"][n]["conf_ratio"] or 0) > 1.0)
            to = sum(1 for r in results if r["cfg"][n]["result"] == "UNK")
            unsnd = sum(1 for r in results if not r["cfg"][n]["agree"] and r["cfg"][n]["result"] != "UNK")
            print(f"  {n:<14} conf {_gm(crs):.3f}   ms {_gm(mrs):.3f}   improved {imp}/{len(results)}"
                  f"   timeouts {to}   unsound {unsnd}")
        # per-family conflict-ratio
        fams = sorted({r["family"] for r in results})
        print("\n  per-family geomean conflict-ratio:")
        print("    " + "family".ljust(10) + "".join(f"{n:>14}" for n in CONFIGS[1:]))
        for fam in fams:
            fr = [r for r in results if r["family"] == fam]
            print(f"    {fam:<10}" + "".join(f"{_gm([r['cfg'][n]['conf_ratio'] for r in fr]):>14.2f}"
                                              for n in CONFIGS[1:]) + f"   (n={len(fr)})")
        print("=" * 78)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        json.dump({"device": str(device), "args": vars(args), "results": results},
                  open(args.output, "w"), indent=1)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
