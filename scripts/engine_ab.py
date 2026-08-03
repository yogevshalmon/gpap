"""A/B the C++ core against the historical PyTorch/Triton-JIT engine at the SAMPLER level.

The unit tests prove kernel-level equality (tests/test_core_equivalence.py). This proves the
same thing through the whole sampling path, and measures throughput, which is the "did the port
cost us anything?" question.

Protocol (the repo's no-change gate, docs/BENCHMARKING.md §3): pinned K, K_fused pinned to K,
ONE config, one seed => the model COUNT must be identical, not merely close. Rates are reported
alongside but are draw-noisy on low-success instances; the counts are the correctness signal.

    python scripts/engine_ab.py bench/*.cnf            # both engines, one process each
    python scripts/engine_ab.py --engine core f.cnf    # one side only (internal; see --help)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def run_one(path: str, K: int, B: int, seed: int, chunks: int, mode: str | None) -> dict:
    """One engine, one instance: fixed K/config/seed, `chunks` chunks, report count + rate."""
    import torch

    from gpap.engine.formula import CnfFormula
    from gpap.sampling.sampler import GPUSampler, parse_independent_support

    dev = torch.device("cuda")
    formula = CnfFormula(dev)
    formula.parse_cnf_file(path)
    smp = GPUSampler(formula, dev, k_cap=K, seed=seed)
    smp.pin_batch(K)                              # pin: no calibration, no fused boost
    smp.support = parse_independent_support(path)
    if mode:
        smp.engine.set_mode(mode)

    support_mask = None
    if smp.support:
        support_mask = torch.zeros(smp.N, dtype=torch.bool, device=dev)
        support_mask[torch.tensor(smp.support, dtype=torch.long, device=dev)] = True

    smp._run_chunk(K, B, True, None, support_mask, True, False)      # warm up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    models = 0
    for _ in range(chunks):
        n, _, _, _ = smp._run_chunk(K, B, True, None, support_mask, True, False)
        models += n
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return {
        "instance": Path(path).name, "engine": os.environ.get("GPAP_ENGINE", "core"),
        "mode": smp.engine.mode, "shift": smp.engine.shift, "K": K, "B": B,
        "models": models, "chunks": chunks, "wall_s": round(wall, 4),
        "models_per_s": round(models / wall, 1) if wall else 0.0,
        "attempts_per_s": round(K * chunks / wall, 1) if wall else 0.0,
        "props": smp.engine.props_total(), "ops": smp.engine.ops_total(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cnf", nargs="+")
    ap.add_argument("-K", type=int, default=8192, help="pinned batch size (default 8192)")
    ap.add_argument("-B", type=int, default=8, help="decisions per round")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--mode", choices=["scalar", "wave"], default=None,
                    help="force the propagate variant on both sides (default: engine default)")
    ap.add_argument("--engine", choices=["core", "torch"], default=None,
                    help=argparse.SUPPRESS)    # internal: run ONE side in this process
    args = ap.parse_args()

    if args.engine:                            # child process: emit one JSON line per instance
        for p in args.cnf:
            print(json.dumps(run_one(p, args.K, args.B, args.seed, args.chunks, args.mode)),
                  flush=True)
        return 0

    # parent: run each engine in its own process (the torch engine must not share device state)
    rows: dict[str, dict] = {}
    for engine in ("torch", "core"):
        env = dict(os.environ, GPAP_ENGINE=engine)
        cmd = [sys.executable, __file__, "--engine", engine, "-K", str(args.K), "-B", str(args.B),
               "--seed", str(args.seed), "--chunks", str(args.chunks)]
        if args.mode:
            cmd += ["--mode", args.mode]
        out = subprocess.run(cmd + args.cnf, env=env, capture_output=True, text=True)
        if out.returncode != 0:
            print(out.stdout + out.stderr, file=sys.stderr)
            return out.returncode
        for line in out.stdout.strip().splitlines():
            r = json.loads(line)
            rows.setdefault(r["instance"], {})[engine] = r

    print(f"{'instance':<16}{'mode':<7}{'K':>7}{'models':>10}{'':>3}"
          f"{'torch m/s':>12}{'core m/s':>12}{'ratio':>8}")
    ok = True
    ratios = []
    for name, pair in rows.items():
        t, c = pair.get("torch"), pair.get("core")
        if not t or not c:
            continue
        same = t["models"] == c["models"]
        ok &= same
        ratio = (c["models_per_s"] / t["models_per_s"]) if t["models_per_s"] else float("nan")
        ratios.append(ratio)
        print(f"{name:<16}{c['mode']:<7}{c['K']:>7}{c['models']:>10}"
              f"{'  =' if same else ' !!':>3}{t['models_per_s']:>12.0f}"
              f"{c['models_per_s']:>12.0f}{ratio:>7.2f}x")
        if not same:
            print(f"    MODEL COUNT MISMATCH: torch {t['models']} vs core {c['models']}")
    if ratios:
        import math
        geo = math.exp(sum(math.log(max(r, 1e-9)) for r in ratios) / len(ratios))
        print(f"\ngeomean core/torch throughput: {geo:.3f}x")
    print("model counts identical on every instance" if ok else "MISMATCHES — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
