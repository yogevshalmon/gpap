#!/usr/bin/env python3
"""Head-to-head: HTS ("High-Throughput SAT Sampling", arXiv 2502.08673 — the strongest public
member of the Berkeley gradient-descent GPU-sampler family, same framework as DiffSampler
DAC'24) vs GPAP, on IDENTICAL CNF files and the SAME GPU.

Protocol (their metric): UNIQUE solutions per second. HTS reports num_unique_solutions /
model_runtime where model_runtime is the gradient-descent loop ONLY (their `timer` wraps
run_back_prop — model build/transform excluded), i.e. their steady-state number. GPAP's
comparable figure is unique_sps_est = timed samples/s x distinct fraction (dedup measured off
the timing path); GPAP's rate additionally includes construction+simplify (sps_e2e_total is the
stricter e2e). Solution-uniqueness caveat: HTS counts unique FULL assignments, GPAP distinct
SUPPORT projections — identical on deterministic-circuit Tseitin CNFs (internal vars are
functions of the inputs), which is why the shared instances here are our circuit unrolls plus
their own s15850a/prod circuit CNFs (`c ind` present).

Setup (once):  git clone https://github.com/arashardakani/High-Throughput-SAT-Sampler \
                   ~/.cache/gpap/src/High-Throughput-SAT-Sampler
Usage:
  PYTHONPATH=. .venv/bin/python eval/rivals_hts.py '.experiments/rival_cnfs/*.cnf' \
      [--steps 10] [--batch 100 1000] [--gpap-n 50000] [--output out.json]

HTS quirks handled here: dataset paths are os.path.join'ed against ITS src/ dir (absolute paths
pass through); ../models + ../results are cwd-relative so it must run from a repo-root child
dir; results land in src/results/<params>.csv (tab-separated); zero console output by design.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

HTS_DEFAULT = os.path.expanduser("~/.cache/gpap/src/High-Throughput-SAT-Sampler")
# Per-SUBPROCESS wall cap for the rival, default 300 s = 5x the 60 s sampling budget GPAP gets
# per instance. Deliberately GENEROUS and not symmetric: the rival's wall includes building its
# PyTorch model, which its OWN metric (unique / model_runtime) excludes, so a tight cap would
# truncate runs it would have completed. MEASURED on this suite: DEMOTIC median 3.7-15.3 s
# (1/38 runs over 60 s), Recurrent-CircuitSAT median ~165 s with a 665 s tail (18/18 over 60 s,
# 5/18 over 300 s) — a 60 s cap would have produced ZERO valid Recurrent measurements.
# A timed-out run is reported as an error row with the reason, never as a zero that would enter
# a geomean as if the rival had genuinely produced nothing.
DEFAULT_TIMEOUT = 300.0


def run_hts(hts_root, cnf, steps, batch, lr="10e0", timeout=DEFAULT_TIMEOUT):
    """One HTS run; returns dict(unique, runtime_s, unique_per_s) or an 'error' dict."""
    name = os.path.basename(cnf)
    # their results filename embeds the params — clear stale ones, then glob the fresh file
    for f in glob.glob(os.path.join(hts_root, "src", "results", f"sat_{name}_*")):
        os.unlink(f)
    cmd = [sys.executable, "../src/pytorch/run.py", "-d", os.path.abspath(cnf), "-l",
           "--optimizer", "mse", "--lr", lr, "--problem_type", "cnf",
           "--num_steps", str(steps), "--batch_size", str(batch)]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=os.path.join(hts_root, "script"), capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout {timeout}s", "wall_s": timeout}
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        return {"error": (p.stderr or p.stdout).strip()[-300:], "wall_s": wall}
    hits = glob.glob(os.path.join(hts_root, "src", "results", f"sat_{name}_*"))
    if not hits:
        return {"error": "no results csv produced", "wall_s": wall}
    import csv
    with open(hits[0]) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        return {"error": "empty results csv (dataset not found?)", "wall_s": wall}
    r = rows[0]
    uniq = int(float(r["num_unique_solutions"]))
    rt = float(r["model_runtime"])
    return {"unique": uniq, "runtime_s": rt, "unique_per_s": uniq / rt if rt else 0.0,
            "wall_s": wall, "steps": steps, "batch": batch}


def run_gpap(cnf, n_time, device, time_limit=60.0, n_cov=1000):
    """`n_cov` is the OFF-timing quality pass: it COLLECTS models to host RAM and rebuilds them
    in Python, which on a deep unroll costs more wall than the timed run itself (measured 26 s of
    a 34 s instance on s1196-c25). It only feeds distinct_frac, which is ~1.0 on every non-tiny
    space — so keep it small here; this harness is about the rival comparison, not coverage."""
    from gpap.sampling.pipeline import CIRCUIT_PORTFOLIO, sample_file
    g = sample_file(cnf, device, n_time=n_time, n_cov=n_cov, configs=CIRCUIT_PORTFOLIO,
                    time_limit=time_limit, seed=5)
    return {"sps": g["samples_per_s"], "sps_e2e": g.get("sps_e2e_total"),
            "unique_per_s_est": g.get("unique_sps_est"), "success": g["success_rate"],
            "n_models": g["n_models"], "distinct_frac": (g["n_distinct"] / len(g["models"])
                                                         if g.get("models") else None),
            "wall_total_s": g["wall_total_s"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cnfs", help="glob of CNF instances (same files go to both samplers)")
    ap.add_argument("--hts", default=HTS_DEFAULT)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--batch", type=int, nargs="+", default=[100, 1000],
                    help="HTS batch sizes to try (best result reported)")
    ap.add_argument("--gpap-n", type=int, default=50_000)
    ap.add_argument("--n-cov", type=int, default=1000,
                    help="models collected for the distinct-fraction readout "
                         "(off the timing path, but its Python reconstruction is "
                         "real wall — keep it small)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                    help="per-run wall cap for the HTS subprocess (s), per batch size")
    ap.add_argument("--gpap-time-limit", type=float, default=60.0,
                    help="sampling wall cap for the GPAP side (s)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    if not os.path.isdir(os.path.join(args.hts, "src")):
        sys.exit(f"no HTS checkout at {args.hts} (see module docstring)")
    import torch
    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    files = sorted(glob.glob(args.cnfs))
    print(f"{len(files)} instances | HTS steps={args.steps} batch={args.batch} | "
          f"GPAP n_time={args.gpap_n:,} | device={device}")
    hdr = (f"{'instance':<24}{'HTS uniq':>10}{'HTS t':>8}{'HTS u/s':>12}"
           f"{'GPAP u/s~':>13}{'GPAP/s':>13}{'succ%':>7}{'ratio':>14}")
    print(hdr); print("-" * len(hdr))
    out = []
    for cnf in files:
        name = os.path.basename(cnf)
        best = None
        for b in args.batch:
            r = run_hts(args.hts, cnf, args.steps, b, timeout=args.timeout)
            if "error" not in r and (best is None or r["unique_per_s"] > best["unique_per_s"]):
                best = r
            elif "error" in r and best is None:
                best = r
        g = run_gpap(cnf, args.gpap_n, device, time_limit=args.gpap_time_limit,
                     n_cov=args.n_cov)
        row = {"name": name, "hts": best, "gpap": g}
        out.append(row)
        if args.output:                       # checkpoint after every instance, not at the end
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            json.dump(out, open(args.output, "w"), indent=1)
        if best and "error" not in best:
            ratio = ((g["unique_per_s_est"] or 0) / best["unique_per_s"] if best["unique_per_s"]
                     else float("inf"))
            rt = f"{ratio:,.1f}x" if ratio != float("inf") else "GPAP"
            print(f"{name[:22]:<24}{best['unique']:>10,}{best['runtime_s']:>7.1f}s"
                  f"{best['unique_per_s']:>12,.0f}{g['unique_per_s_est'] or 0:>13,.0f}"
                  f"{g['sps']:>13,.0f}{100*g['success']:>6.0f}%{rt:>14}", flush=True)
        else:
            err = (best or {}).get("error", "?")
            print(f"{name[:22]:<24}  HTS FAILED: {err[:60]} | GPAP u/s~="
                  f"{g['unique_per_s_est'] or 0:,.0f}", flush=True)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        json.dump(out, open(args.output, "w"), indent=1)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
