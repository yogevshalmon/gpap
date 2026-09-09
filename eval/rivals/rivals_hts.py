#!/usr/bin/env python3
"""Head-to-head: HTS ("High-Throughput SAT Sampling", arXiv 2502.08673 — the strongest public
member of the Berkeley gradient-descent GPU-sampler family) vs GPAP, on IDENTICAL CNF files and
the SAME GPU.

Protocol, their metric: UNIQUE solutions per second, num_unique_solutions / model_runtime, whose
clock is the gradient loop ONLY. GPAP's comparable figure is unique_sps_est = timed samples/s x
the distinct fraction (dedup measured off the timing path), and its rate additionally includes
construction + simplify. HTS counts unique FULL assignments, GPAP distinct SUPPORT projections,
identical on deterministic-circuit Tseitin CNFs. Their timer is emission-free, matching CMSGen's
`--nowrite` and our `collect=False` run (eval/rivals/check_emission_symmetry.py)."""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

from gpap import paths

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # eval/<sub>/ -> repo
sys.path.insert(0, REPO_ROOT)

from eval.rivals.rival_common import (cached_gpap, fit_batch,  # noqa: E402
                                      resolve_steps, rival_commit, rival_vram_budget,
                                      run_at_batch)

HTS_DEFAULT = str(paths.src_dir() / "High-Throughput-SAT-Sampler")
# Per-SUBPROCESS wall cap for the rival, 5x the sampling budget GPAP gets: not symmetric, since
# the rival's wall includes the model build its OWN metric excludes.
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
            "wall_s": wall, "steps": steps, "batch": batch}    # steps_source added by main


def n_clauses(cnf):
    """Clause count from the DIMACS header — the work unit HTS's memory scales with."""
    for line in open(cnf):
        if line.startswith("p cnf"):
            return int(line.split()[3])
    return 1


def run_gpap(cnf, n_time, device, time_limit=60.0, n_cov=1000):
    """`n_cov` is the OFF-timing quality pass: it COLLECTS models to host RAM and rebuilds them
    in Python, costing more than the timed run on a deep unroll and feeding only distinct_frac."""
    from gpap.sampling.pipeline import CIRCUIT_PORTFOLIO, sample_file
    # verify_sound: the row records `sound_check`, so the check runs; its wall leaves every rate.
    g = sample_file(cnf, device, n_time=n_time, n_cov=n_cov, configs=CIRCUIT_PORTFOLIO,
                    time_limit=time_limit, seed=5, verify_sound=True)
    return {"sps": g["samples_per_s"], "sps_e2e": g.get("sps_e2e_total"),
            "unique_per_s_est": g.get("unique_sps_est"),
            # None means NOT MEASURED, never "produced nothing" — `n_models` is that question.
            "unique_est_status": g.get("unique_est_status"), "success": g["success_rate"],
            "n_models": g["n_models"], "distinct_frac": (g["n_distinct"] / len(g["models"])
                                                         if g.get("models") else None),
            "wall_total_s": g["wall_total_s"],
            # engine provenance, as on the CMSGen sweeps: dec/s + impl/s let the zero-search screen run
            "sound_check": g.get("sound_check"),
            "decisions_per_s": g.get("decisions_per_s"),
            "implications_per_s": g.get("implications_per_s")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cnfs", help="glob of CNF instances (same files go to both samplers)")
    ap.add_argument("--hts", default=HTS_DEFAULT)
    ap.add_argument("--steps", default="auto",
                    help="their num_steps: 'auto' (default) = the value their own artifacts "
                         "specify, see rival_common.RIVAL_STEPS; or an explicit number")
    ap.add_argument("--batch", type=int, nargs="+", default=None,
                    help="explicit HTS batch sizes to try (best reported). Default: the largest "
                         "that fits in 0.80 of free VRAM — the same share of the card GPAP "
                         "sizes its own K with. Their scripts' batch 100 is ~1/295 of the rate "
                         "their tool reaches on our instances, so it cannot be the rule.")
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
    ap.add_argument("--gpap-cache", default=None,
                    help="directory of cached GPAP arms (default: <output dir>/.gpap_cache). "
                         "GPAP_RIVAL_CACHE=0 disables reuse entirely.")
    ap.add_argument("--resume", action="store_true",
                    help="keep the rows already in --output and re-run only what is missing "
                         "(a rented VM can be reclaimed mid-stage; see eval/colab/colab_supervisor.py)")
    ap.add_argument("--retry-errors", action="store_true",
                    help="with --resume, also re-run rows that previously ERRORED. Off by "
                         "default: a rival timeout is a recorded result and costs the full "
                         "timeout to rediscover on every restart. Turn it on when the failure "
                         "was ENVIRONMENTAL (e.g. a missing dependency), not the rival's.")
    args = ap.parse_args()
    if not os.path.isdir(os.path.join(args.hts, "src")):
        sys.exit(f"no HTS checkout at {args.hts} (see module docstring)")
    import torch
    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    steps, steps_src = resolve_steps("hts", args.steps)
    files = sorted(glob.glob(args.cnfs))
    bdesc = (", ".join(f"{b:,}" for b in args.batch) if args.batch
             else "fit to 0.80 of free VRAM")
    print(f"{len(files)} instances | HTS steps={steps} ({steps_src}) batch={bdesc} | "
          f"GPAP n_time={args.gpap_n:,} | device={device}")
    hdr = (f"{'instance':<24}{'HTS uniq':>10}{'HTS t':>8}{'HTS u/s':>12}"
           f"{'GPAP u/s~':>13}{'GPAP/s':>13}{'succ%':>7}{'ratio':>14}")
    print(hdr); print("-" * len(hdr))
    out = []
    done = set()
    # Our arm is the same task whatever the rival is doing; see rival_common.cached_gpap.
    cache_dir = args.gpap_cache or (os.path.join(os.path.dirname(args.output) or ".",
                                                 ".gpap_cache") if args.output else None)
    if args.resume and args.output and os.path.isfile(args.output):
        try:
            out = json.load(open(args.output)) or []
            if args.retry_errors:      # drop them so the retry REPLACES rather than appends
                out = [r for r in out if "error" not in (r.get("hts") or {})]
            done = {r["name"] for r in out}
            print(f"[resume] {len(out)} rows already in {args.output}")
        except Exception as e:      # noqa: BLE001 — a truncated checkpoint must not kill the run
            print(f"[resume] ignoring unreadable {args.output}: {e}")
            out, done = [], set()
    for cnf in files:
        name = os.path.basename(cnf)
        if name in done:
            continue
        if args.batch:                          # explicit list: measure exactly those
            best = None
            for b in args.batch:
                r = run_hts(args.hts, cnf, steps, b, timeout=args.timeout)
                if "error" not in r and (best is None or r["unique_per_s"] > best["unique_per_s"]):
                    best = r
                elif "error" in r and best is None:
                    best = r
            sweep = None
        else:
            best, sweep = run_at_batch(
                lambda b: run_hts(args.hts, cnf, steps, b, timeout=args.timeout),
                fit_batch(n_clauses(cnf), "hts"))
        if best is not None:
            best["steps_source"] = steps_src
        task = {"harness": "hts", "n_time": args.gpap_n, "time_limit": args.gpap_time_limit,
                "n_cov": args.n_cov, "portfolio": "circuit"}
        g, hit = cached_gpap(cache_dir, cnf, task,
                             lambda: run_gpap(cnf, args.gpap_n, device,
                                              time_limit=args.gpap_time_limit,
                                              n_cov=args.n_cov))
        if hit:
            print(f"{name[:22]:<24}  [gpap-cache hit — arm not re-run]", flush=True)
        row = {"name": name, "rival_commit": rival_commit(args.hts), "hts": best, "gpap": g,
               "vram_budget_bytes": rival_vram_budget()}
        if sweep:                       # what the rival was actually GRANTED, next to its score
            row["batch_sweep"] = sweep
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
