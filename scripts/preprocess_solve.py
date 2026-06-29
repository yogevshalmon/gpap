#!/usr/bin/env python3
"""Does GPU *preprocessing* (vivification + unit-closure + subsumption) beat plain CaDiCaL?

This is the reduction counterpart to the (failed) clause-*addition* experiment. Adding entailed
clauses perturbs CDCL chaotically; here we instead REDUCE the formula — shorten clauses by BCP
(vivify), assert failed-literal units and propagate, and let discovered short clauses subsume
longer ones — all equivalence-preserving. The question: does a more-exhaustive-than-the-solver
GPU reduction pass make the resulting (smaller) CNF solve faster, on top of CaDiCaL's own
inprocessing which is already on in every config?

GPU discovery (units, binaries) + vivification run ONCE per instance; the configs below are then
assembled on the CPU so they share that GPU work:
    plain   original formula (baseline; CaDiCaL still vivifies/subsumes internally)
    viv     vivify → subsume/dedup
    viv+U   vivify → unit-closure → subsume/dedup
    full    vivify → unit-closure → learned-binary subsumption → subsume/dedup

Every reduced formula is logically equivalent to the original, so the SAT/UNSAT verdict MUST
match plain — a mismatch flags an unsound reduction and is reported loudly.

Two speedups (GPU discovery is hardware-improvable; the solve benefit is not):
    ×no-GPU  = plain_solve / config_solve
    ×wGPU    = plain_solve / (config_solve + gpu_ms)
Plus a SAFE portfolio = min(plain, full) per instance (run both, keep the winner) — never regresses.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/preprocess_solve.py --insts /tmp/bigpool2.json \\
        --device cuda --max-solve-s 30 --vivify-max-len 12 --output .experiments/preproc.json
"""
import argparse
import json
import os
import random
import statistics
import sys
import time

import torch

sys.path.insert(0, ".")

from gpap.solver.inprocess import GPUInprocessor
from gpap.solver.preprocess import (
    _simplify_by_assignment,
    apply_learned_subsumption,
    equivalent_literal_substitution,
    subsume_all,
    unit_closure,
)
from scripts.inprocess_suite import timed_solve
from scripts.profile_engine import count_all_pairs

# plain racer + the existing reduction ladder + ELS (els alone, and els on top of viv+units).
CONFIGS = ["plain", "els", "viv", "viv+U", "els+viv+U", "full"]
_UNSAT = [[1], [-1]]


def _read_clauses(path):
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


def _bin_vars(N, mode, pool, deg, seed):
    if mode == "none":
        return []
    if mode == "all" or count_all_pairs(N) <= 4_000_000:
        return list(range(N))
    # degree-weighted: the highest-occurrence variables are likeliest to yield subsuming binaries
    if deg is not None:
        return [v - 1 for v in sorted(range(1, N + 1), key=lambda v: -deg.get(v, 0))[:pool]]
    return random.Random(seed).sample(range(N), min(N, pool))


def discover(path, device, orig, N, *, bin_mode, bin_pool, vmax_len, vmax_rounds, deg, seed):
    """Run the GPU phase once; return (units, bins, vivified, edges, gpu_ms, vstats, els_stats)."""
    inp = GPUInprocessor.from_cnf_file(path, device)
    t0 = time.perf_counter()
    edges, els_units, estats = inp.discover_implications()    # implications (ELS) + failed-lit units
    units = els_units                                          # implication probe ⊇ probe_failed_literals
    bins = []
    bvars = _bin_vars(N, bin_mode, bin_pool, deg, seed)
    if len(bvars) >= 2:
        inp.reset_emitted()
        bins, _ = inp.discover_binaries(bvars)
    # Complete vivification: conflict-prefix shortening + asymmetric implied-literal removal.
    vivified, vstats = inp.vivify_clauses(orig, max_len=vmax_len, max_rounds=vmax_rounds, asymmetric=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    gpu_ms = (time.perf_counter() - t0) * 1e3
    del inp
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return units, bins, vivified, edges, gpu_ms, vstats, estats


def build_configs(orig, units, bins, vivified, edges, nv):
    """Assemble the CPU configs from the shared GPU output. Returns ({name: clauses}, els_stats).

    ELS (equivalent-literal substitution) is equisat-preserving and removes variables; it is
    measured both alone (`els`) and stacked on the vivify+unit ladder (`els+viv+U`)."""
    els, _vm, els_stats = equivalent_literal_substitution(orig, edges, nv)
    cfg = {"plain": orig, "viv": subsume_all(vivified),
           "els": list(_UNSAT) if els_stats["status"] == "unsat" else subsume_all(els)}
    uc, assigned, status = unit_closure(vivified, units)
    if status == "unsat":
        for k in ("viv+U", "els+viv+U", "full"):
            cfg[k] = list(_UNSAT)
        return cfg, els_stats
    cfg["viv+U"] = subsume_all(uc)
    els_uc, _vm2, s2 = equivalent_literal_substitution(uc, edges, nv)
    cfg["els+viv+U"] = list(_UNSAT) if s2["status"] == "unsat" else subsume_all(els_uc)
    full, _ = apply_learned_subsumption(uc, _simplify_by_assignment(bins, assigned))
    cfg["full"] = subsume_all(full)
    return cfg, els_stats


def _cadical_vars_removed(orig, nv):
    """GPU-value-add diagnostic: how many variables CaDiCaL's own ELS (`decompose`) removes.

    Isolates CaDiCaL's equivalence reasoning by running `process()` with ONLY `decompose` on
    (its binary-implication-graph SCC = its ELS); everything else off. GPU-ELS works over the full
    BCP-implication closure (incl. hyper-binary implications through 3+ clauses), so the question
    is whether GPU removes MORE equivalent variables than CaDiCaL's binary-only decompose.
    Best-effort: returns None if `process()` doesn't hand back a clause list."""
    from pysat.formula import CNF
    from pysat.solvers import Cadical195
    try:
        with Cadical195(bootstrap_with=CNF(from_clauses=[list(c) for c in orig])) as s:
            status, simplified = s.process(
                rounds=2, decompose=True, elim=False, probe=False, probehbr=False,
                vivify=False, subsume=False, block=False, cover=False, condition=False)
        if not simplified:
            return None
        live = {abs(l) for c in simplified for l in c}
        return nv - len(live)
    except Exception:  # noqa: BLE001 — diagnostic only, never fail the run
        return None


def eval_instance(rec, device, args, deg_fn):
    path = rec["path"]
    orig = _read_clauses(path)
    if not orig:
        return {"name": os.path.basename(path)[:38], "skip": "empty"}
    nv = max(abs(l) for c in orig for l in c)

    res0, ms0, cf0 = timed_solve(orig, nv, args.max_solve_s)
    # plain-UNK is NOT discarded: a reduced config that solves where plain timed out is a "rescue"
    # — the strongest possible reduction result. Probe only the ELS configs there to bound cost.
    plain_unk = res0 == "UNK"

    deg = deg_fn(path) if deg_fn else None
    units, bins, vivified, edges, gpu_ms, vstats, estats = discover(
        path, device, orig, nv, bin_mode=args.bin_mode, bin_pool=args.bin_pool,
        vmax_len=args.vivify_max_len, vmax_rounds=args.vivify_max_rounds, deg=deg, seed=args.seed)
    cfgs, els_stats = build_configs(orig, units, bins, vivified, edges, nv)

    out = {"name": os.path.basename(path)[:38], "family": rec.get("family"), "nv": nv,
           "M": len(orig), "lits": sum(len(c) for c in orig), "result": res0, "plain_unk": plain_unk,
           "plain_ms": ms0, "gpu_ms": gpu_ms,
           "n_units": len(units), "n_bins": len(bins), "n_edges": estats.get("n_edges", 0),
           "n_vivified": vstats.get("n_vivified", 0), "viv_lits_removed": vstats.get("lits_removed", 0),
           "els_vars_removed": els_stats.get("vars_removed", 0),
           "els_classes": els_stats.get("n_classes", 0),
           "cadical_vars_removed": _cadical_vars_removed(orig, nv) if args.cadical_diag else None,
           "cfg": {}}
    run_names = (["els", "els+viv+U"] if plain_unk else CONFIGS[1:])
    for name in CONFIGS:
        if name == "plain":
            out["cfg"]["plain"] = {"solve_ms": ms0, "conf": cf0, "result": res0, "M": len(orig),
                                   "lits": sum(len(c) for c in orig),
                                   "speedup": None if plain_unk else 1.0,
                                   "speedup_wgpu": None if plain_unk else 1.0, "sound": True}
            continue
        if name not in run_names:
            out["cfg"][name] = {"solve_ms": None, "conf": None, "result": "SKIP", "M": len(cfgs[name]),
                                "lits": sum(len(c) for c in cfgs[name]), "speedup": None,
                                "speedup_wgpu": None, "sound": True, "rescued": False}
            continue
        cls = cfgs[name]
        r, ms, cf = timed_solve(cls, nv, args.max_solve_s)
        if plain_unk:
            sound = True                      # nothing to contradict; cross-config agreement checked later
            rescued = r != "UNK"
            speedup = speedup_wgpu = None
        else:
            sound = r in (res0, "UNK")
            rescued = False
            speedup = (ms0 / ms) if (ms > 0 and r == res0) else None
            speedup_wgpu = (ms0 / (ms + gpu_ms)) if (ms + gpu_ms) > 0 and r == res0 else None
        out["cfg"][name] = {
            "solve_ms": ms, "conf": cf, "result": r, "M": len(cls), "lits": sum(len(c) for c in cls),
            "speedup": speedup, "speedup_wgpu": speedup_wgpu, "sound": sound, "rescued": rescued,
        }
    return out


def _gm(xs):
    xs = [x for x in xs if x and x > 0]
    return statistics.geometric_mean(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--insts", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-solve-s", type=float, default=30.0)
    ap.add_argument("--vivify-max-len", type=int, default=12)
    ap.add_argument("--vivify-max-rounds", type=int, default=None)
    ap.add_argument("--bin-mode", default="degree", choices=["none", "degree", "all"])
    ap.add_argument("--bin-pool", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cadical-diag", action="store_true",
                    help="Also report how many vars CaDiCaL's own process() removes (GPU value-add)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    insts = json.load(open(args.insts))[:args.limit]

    deg_fn = None
    if args.bin_mode == "degree":
        from scripts.inprocess_heuristics import static_scores
        from pysat.formula import CNF

        def deg_fn(path):
            return static_scores(CNF(from_file=path))[0]

    print(f"Device {device} | {len(insts)} instances | vivify_max_len={args.vivify_max_len} "
          f"bin_mode={args.bin_mode} | speedup = plain / config (CaDiCaL inprocessing ON in all)\n")
    hdr = (f"{'instance':<34}{'M':>7}{'plainms':>8}{'gpums':>7}{'viv':>6}{'units':>6}{'elsV':>6}"
           f"{'Δlit%':>6}" + "".join(f"{n:>9}" for n in CONFIGS[1:]))
    print(hdr); print("-" * len(hdr))

    results = []
    for rec in insts:
        try:
            r = eval_instance(rec, device, args, deg_fn)
        except Exception as e:  # noqa: BLE001
            print(f"{os.path.basename(rec['path'])[:34]:<34}  ERROR: {type(e).__name__}: {e}")
            continue
        results.append(r)
        if "skip" in r:
            print(f"{r['name']:<34}  skip ({r['skip']})")
            continue
        dlit = 100.0 * (1 - r["cfg"]["els"]["lits"] / r["lits"]) if r["lits"] else 0.0

        def cell(n):
            c = r["cfg"][n]
            if not c["sound"]:
                return "UNSND"
            if c["result"] == "SKIP":
                return "."
            if r.get("plain_unk"):
                return "RESCUE" if c.get("rescued") else "unk"
            return f"{c['speedup']:.2f}" if c["speedup"] else "-"
        tag = "  [plain UNK]" if r.get("plain_unk") else ""
        print(f"{r['name']:<34}{r['M']:>7}{r['plain_ms']:>8.0f}{r['gpu_ms']:>7.0f}"
              f"{r['n_vivified']:>6}{r['n_units']:>6}{r['els_vars_removed']:>6}{dlit:>6.1f}"
              + "".join(f"{cell(n):>9}" for n in CONFIGS[1:]) + tag)

    allev = [r for r in results if "cfg" in r]
    ev = [r for r in allev if not r.get("plain_unk")]      # solvable: have a plain baseline
    hard = [r for r in allev if r.get("plain_unk")]        # plain timed out → rescue candidates
    if ev:
        print("\n" + "=" * 70)
        print(f"solvable instances: {len(ev)}   geomean speedup vs plain (CaDiCaL inprocessing on in all):")
        for n in CONFIGS:
            ng = [r["cfg"][n]["speedup"] for r in ev]
            wg = [r["cfg"][n]["speedup_wgpu"] for r in ev]
            imp = sum(1 for x in ng if x and x > 1.0)
            print(f"  {n:<9} ×no-GPU {_gm(ng):.3f}  ×wGPU {_gm(wg):.3f}  "
                  f"median {statistics.median([x for x in ng if x] or [0]):.2f}  improved {imp}/{len(ev)}")
        # SAFE portfolio: per instance keep the best of plain + every reduced config that agreed
        # (run them in parallel, take the winner) → never worse than plain.
        def best_solve(r):
            cands = [r["plain_ms"]] + [c["solve_ms"] for c in r["cfg"].values()
                                       if c["result"] == r["result"] and c["solve_ms"] is not None]
            return min(cands)
        safe = [r["plain_ms"] / best_solve(r) for r in ev]
        print(f"  SAFE[plain|all configs] geomean {_gm(safe):.3f} (per-instance best wall-ms, never < plain)")

        # Conflict-based portfolio: noise-free (CaDiCaL deterministic) — the honest portfolio number
        # for a niche full of sub-second instances where wall-ms is jitter.
        def best_conf(r):
            cands = [r["cfg"]["plain"]["conf"]] + [c["conf"] for c in r["cfg"].values()
                                                   if c["result"] == r["result"] and c.get("conf")]
            cands = [c for c in cands if c]
            return min(cands) if cands else None
        cport = [r["cfg"]["plain"]["conf"] / best_conf(r) for r in ev
                 if r["cfg"]["plain"].get("conf") and best_conf(r)]
        if cport:
            print(f"  SAFE[plain|all configs] geomean {_gm(cport):.3f} (per-instance fewest CONFLICTS, "
                  f"noise-free) — won by reduction on {sum(1 for x in cport if x > 1.001)}/{len(cport)}")

        # Per-family breakdown (characterize the niche): geomean per config, per family.
        fams = sorted({r.get("family") or "?" for r in ev})
        if len(fams) > 1:
            print("\n  per-family ×no-GPU geomean (n):")
            print("    " + "family".ljust(14) + "".join(f"{n:>9}" for n in CONFIGS[1:]) + "    safe")
            for fam in fams:
                fr = [r for r in ev if (r.get("family") or "?") == fam]
                cols = "".join(f"{_gm([r['cfg'][n]['speedup'] for r in fr]):>9.2f}" for n in CONFIGS[1:])
                fsafe = _gm([r["plain_ms"] / best_solve(r) for r in fr])
                print(f"    {fam[:14]:<14}{cols}{fsafe:>8.2f}  (n={len(fr)})")

        # ELS reach vs CaDiCaL's own decompose (the GPU value-add question), over ALL instances.
        gpu_vr = sum(r["els_vars_removed"] for r in allev)
        cad = [r["cadical_vars_removed"] for r in allev if r.get("cadical_vars_removed") is not None]
        wins = sum(1 for r in allev if r.get("cadical_vars_removed") is not None
                   and r["els_vars_removed"] > r["cadical_vars_removed"])
        print(f"\n  ELS vars removed (GPU total): {gpu_vr}"
              + (f" | CaDiCaL decompose total: {sum(cad)} | GPU>CaDiCaL on {wins}/{len(cad)}"
                 if cad else " | (run with --cadical-diag for the CaDiCaL comparison)"))

        # mean literal reduction of `els`
        dl = [100.0 * (1 - r["cfg"]["els"]["lits"] / r["lits"]) for r in ev if r["lits"]]
        if dl:
            print(f"  els formula reduction: mean Δliterals {statistics.mean(dl):.1f}%  "
                  f"median {statistics.median(dl):.1f}%")
        unsound = sum(1 for r in ev for c in r["cfg"].values() if not c["sound"])
        print(f"  unsound configs: {unsound}  (MUST be 0)")
        print("=" * 70)

    # Rescues: plain timed out but a reduced config solved (the strongest reduction result).
    if hard:
        rescued = [r for r in hard if any(c.get("rescued") for c in r["cfg"].values())]
        # soundness on rescues: every config that solved a given instance must agree.
        bad = 0
        for r in hard:
            verds = {c["result"] for c in r["cfg"].values() if c["result"] in ("SAT", "UNSAT")}
            bad += len(verds) > 1
        print(f"\nplain-UNK instances: {len(hard)}  | RESCUED by a reduced config: {len(rescued)}"
              f"  | rescue verdict conflicts: {bad} (MUST be 0)")
        for r in rescued:
            who = [n for n in CONFIGS[1:] if r["cfg"][n].get("rescued")]
            print(f"    {r['name']:<40} {r['family']:<10} solved by {','.join(who)} "
                  f"(elsV={r['els_vars_removed']})")
        print("=" * 70)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        json.dump({"device": str(device), "args": vars(args), "results": results},
                  open(args.output, "w"), indent=1)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
