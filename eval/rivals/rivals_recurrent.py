#!/usr/bin/env python3
"""Head-to-head: Recurrent-CircuitSAT-Sampler (MLCAD'25, arXiv 2502.21226 — the Berkeley
gradient family's SEQUENTIAL-circuit member, the closest rival to GPAP's headline domain) vs
GPAP, same .bench sources, same GPU.

Their protocol (replicated exactly): per-filename target table pinning 1-2 output BITS (copied
into TARGETS below), reset-0 registers, solutions counted UNIQUE and pooled over ALL unroll
depths start_point..T (each depth's constraint sits at that depth's last cycle), model_runtime
= the gradient loop only (their timer excludes model build AND the solution re-check pass).
Both are GENEROUS to them and kept as-is; we additionally parse their per-depth stdout list for
the depth-T-only count (`uniq@T`) — the number directly comparable to GPAP's fixed-depth CNF.

EMISSION SYMMETRY (the one asymmetry that runs the OTHER way, measured and deliberately NOT
corrected): every other tool here is timed on generation alone — CMSGen via --nowrite, DEMOTIC
and HTS because their own timer wraps the descent loop only, GPAP because its timed run is
collect=False. This rival instead .cpu()s its per-depth candidate assignments INSIDE the loop its
timer wraps (run.py:186 — `|PI| x sum(d+1 for d in start_point..T-1)` strided copies, 0.5-4.0 GiB
per run), so its published rate is charged for delivery. Measured on an RTX 4080 by timing that
block directly: 0.70% (b03), 1.32-1.42% (s386), 0.88% (s1196) of model_runtime, and removing the
copy entirely (snapshots kept on device — their _check_solution already does
`par.to(self.device)`) leaves num_unique_solutions BIT-IDENTICAL while saving 1.41% on s386 over
4 interleaved repetitions. Bound for the recorded cells: <=~2.2% of their model_runtime.
NOT deducted, on purpose: it is a single rival, the correction is ~1% and it moves the ratio in
OUR favour, so leaving it in keeps their number exactly as they publish it and keeps this
harness simple. Re-derive with eval/rivals/check_emission_symmetry.py.

GPAP side: identical constraint via output-reordered .bench (their target_idx outputs moved
first, then build_cnf pin_outputs=len(idx) with the same values, pinned at cycle T), sampled at
MEDIAN OF 3 SEEDS. Rejection baseline (p_acc) comes from sim_baseline on the same reordered
bench + explicit target.

Rare mode (--rare): our adversarial fixed-T target. The bench gets a comparator cone
(MITER = AND of per-output [X]NOR terms vs the rare pattern) as its ONLY output, and the copy
is named b13.bench so their table pins [-1]=1 = MITER=1 (their table is FILENAME-keyed; every
s*-named branch would index two outputs and break on a single-output module). Their solutions
then satisfy outputs@t == pattern-of-T for any depth t — still a superset of GPAP's t=T space.

Setup (once): eval/rivals/rivals_setup.sh — clones
    https://github.com/arashardakani/Recurrent-CircuitSAT-Sampler into
    <repo>/.gpap/src/Recurrent-CircuitSAT-Sampler at the PINNED evaluated commit.
Usage:
  PYTHONPATH=. .venv/bin/python eval/rivals/rivals_recurrent.py b03 b13 s1196 [--rare] [-T 25] \
      [--steps 1] [--output out.json]

Quirks handled: Moore pipelines have no input->output gradient path until the unroll reaches
the register distance, so start_point is probed upward until backward() stops crashing (their
flag for the same purpose); dataset/results paths are joined against THEIR src/ so absolute
paths pass through; results csv is tab-separated in src/results/.
"""
import argparse
import ast
import glob
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # eval/<sub>/ -> repo
sys.path.insert(0, REPO_ROOT)

from eval.rivals.rival_common import (CACHE_MISS, VARIANTS, cache_peek,  # noqa: E402
                                      cache_store, cached_gpap, fit_batch, prepare_variant,
                                      resolve_steps, rival_commit, rival_vram_budget,
                                      run_at_batch, variant_note)

from gpap import paths
from gpap.sampling.metrics import merge_sound_check, sound_verdict

RIVAL_DEFAULT = str(paths.src_dir() / "Recurrent-CircuitSAT-Sampler")
# Per-SUBPROCESS wall cap for the rival, default 300 s = 5x the 60 s sampling budget GPAP gets
# per instance. Deliberately GENEROUS and not symmetric: the rival's wall includes building its
# PyTorch model, which its OWN metric (unique / model_runtime) excludes, so a tight cap would
# truncate runs it would have completed. MEASURED on this suite: DEMOTIC median 3.7-15.3 s
# (1/38 runs over 60 s), Recurrent-CircuitSAT median ~165 s with a 665 s tail (18/18 over 60 s,
# 5/18 over 300 s) — a 60 s cap would have produced ZERO valid Recurrent measurements.
# A timed-out run is reported as an error row with the reason, never as a zero that would enter
# a geomean as if the rival had genuinely produced nothing.
DEFAULT_TIMEOUT = 300.0
PROBE_TIMEOUT = 60.0          # per start_point probe (a 1-step run; same budget)
PROBE_BUDGET = 120.0          # total probing per instance before we give up on it
BENCH_DIR = os.path.join(REPO_ROOT, "benchmarks", "circuits", "bench", "seq")

# their run.py target table (the branches for instances we share; idx into the OUTPUT list)
TARGETS = {
    "s386": ([-1], [1]), "s400": ([-1], [1]), "s526": ([-1], [1]), "s1423": ([-1], [1]),
    "b01": ([-1], [1]), "b03": ([-1], [1]), "b04": ([-1], [1]), "b06": ([-1], [1]),
    "b08": ([-1], [1]), "b11": ([-1], [1]), "b13": ([-1], [1]),
    "b02": ([-1], [0]), "b05": ([-1], [0]), "b07": ([-1], [0]), "b09": ([-1], [0]),
    "b10": ([-2], [0]), "b12": ([-2], [0]),
    "s820": ([0, -1], [0, 1]), "s832": ([0, -1], [0, 1]), "s1238": ([0, -1], [0, 1]),
    "s1488": ([0, -1], [0, 1]),
    "s641": ([0, 1], [1, 0]), "s713": ([0, 1], [1, 0]), "s1196": ([0, 1], [1, 0]),
    "s5378": ([0, 1], [1, 0]),
}
RARE_ALIAS = "b13"      # any of their [-1]=1 names; the rare-mode MITER copy is keyed to it


# ---------------------------------------------------------------- bench rewriting helpers

def _read_bench_lines(path):
    return open(path).read().splitlines()


def _output_wires(lines):
    return [re.match(r"\s*OUTPUT\((.+)\)\s*$", l).group(1).strip()
            for l in lines if re.match(r"\s*OUTPUT\(", l)]


def _replace_outputs(lines, new_outputs):
    """Drop every OUTPUT(...) line, emit the new list where the first one stood (file order =
    output order for both our parser and theirs)."""
    out, done = [], False
    for l in lines:
        if re.match(r"\s*OUTPUT\(", l):
            if not done:
                out.extend(f"OUTPUT({w})" for w in new_outputs)
                done = True
            continue
        out.append(l)
    return out


def reorder_bench(src, dst, idx):
    """Copy `src` with the outputs at positions `idx` moved to the front."""
    lines = _read_bench_lines(src)
    outs = _output_wires(lines)
    picked = [outs[i] for i in idx]
    rest = [o for o in outs if o not in picked]
    with open(dst, "w") as f:
        f.write("\n".join(_replace_outputs(lines, picked + rest)) + "\n")
    return picked


def miter_bench(src, dst, T, seed):
    """Copy `src` with a single MITER output = [outputs at the last cycle == rare pattern],
    the rare pattern taken over ALL outputs at depth T (gpap.circuits.simulate.rare_target).
    Returns (pattern, est_freq)."""
    from gpap.circuits.bench import parse_bench
    from gpap.circuits.simulate import rare_target
    circ = parse_bench(src)
    pattern, est = rare_target("bench", circ, T, seed)
    lines = _read_bench_lines(src)
    outs = _output_wires(lines)
    terms = []
    extra = []
    for i, o in enumerate(outs):
        if pattern[o]:
            terms.append(o)
        else:
            n = f"GPAPN{i}"
            extra.append(f"{n} = NOT({o})")
            terms.append(n)
    if len(terms) == 1:
        extra.append(f"GPAPM0 = NOT({terms[0]})")
        extra.append("GPAPMITER = NOT(GPAPM0)")
    else:
        extra.append(f"GPAPMITER = AND({', '.join(terms)})")
    lines = _replace_outputs(lines, ["GPAPMITER"])
    with open(dst, "w") as f:
        f.write("\n".join(lines) + "\n" + "\n".join(extra) + "\n")
    return pattern, est


# ---------------------------------------------------------------- the two samplers

def run_recurrent(rival, bench_file, key, T, steps, batch, start_point,
                  timeout=DEFAULT_TIMEOUT, retry=True):
    """One run of their sampler; returns dict or {'error': ...}. `key` = the filename stem the
    copy is given (their target table and results csv are keyed on it)."""
    data_dir = os.path.join(rival, "data", "rival_tmp")
    os.makedirs(data_dir, exist_ok=True)
    shutil.copy(bench_file, os.path.join(data_dir, f"{key}.bench"))
    for f in glob.glob(os.path.join(rival, "src", "results", f"sat_{key}_*")):
        os.unlink(f)
    cmd = [sys.executable, os.path.join(rival, "src", "pytorch", "run.py"),
           "-d", f"../data/rival_tmp/{key}.bench", "-l", "--optimizer", "mse", "--lr", "15e0",
           "--problem_type", "BLIF", "--circuit_type", "seq", "--num_steps", str(steps),
           "--batch_size", str(batch), "--num_clock_cycles", str(T),
           "--start_point", str(start_point)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=os.path.join(rival, "script"), capture_output=True,
                           text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout {timeout}s"}
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        err = (p.stderr or p.stdout).strip().splitlines()[-1][:200]
        # transient WSL2 driver hiccup between back-to-back CUDA processes — settle and retry
        if "device not ready" in err and retry:
            time.sleep(10)
            return run_recurrent(rival, bench_file, key, T, steps, batch, start_point,
                                 timeout=timeout, retry=False)
        return {"error": err, "wall_s": wall}
    hits = glob.glob(os.path.join(rival, "src", "results", f"sat_{key}_*"))
    if not hits:
        return {"error": "no results csv", "wall_s": wall}
    import csv
    with open(hits[0]) as f:
        r = list(csv.DictReader(f, delimiter="\t"))[0]
    uniq = int(float(r["num_unique_solutions"]))
    rt = float(r["model_runtime"])
    # their stdout prints the per-depth unique-count list; the last entry is depth T
    uniq_at_t = None
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("[["):
            try:
                uniq_at_t = ast.literal_eval(line)[-1][0]
            except (ValueError, SyntaxError, IndexError):
                pass
            break
    return {"unique": uniq, "unique_at_T": uniq_at_t, "runtime_s": rt,
            "unique_per_s": uniq / rt if rt else 0.0, "wall_s": wall,
            "batch": batch, "steps": steps, "start_point": start_point}


def _sp_task(T, max_sp):
    return {"probe": "start_point", "cycles": T, "max_sp": max_sp}


def probe_start_point(rival, bench_file, key, T, max_sp=10, budget=PROBE_BUDGET,
                      cache_dir=None):
    """Their backward() crashes while the pinned output has no gradient path to any input
    (Moore register latency); walk start_point up until a tiny run survives. Bounded twice: each
    probe gets PROBE_TIMEOUT, and the whole search gets `budget` — an instance whose probe never
    settles must cost minutes, not the afternoon.

    CACHED, because it is pure: the answer depends only on the circuit and the unroll depth, yet
    every probe pays a full process launch and model build (~30 s of a Recurrent run's wall is
    build, even at --num_steps 1). Uncached, re-running this stage costs up to PROBE_BUDGET per
    instance before any measurement happens."""
    hit = cache_peek(cache_dir, bench_file, _sp_task(T, max_sp))
    if hit is not CACHE_MISS:
        return hit
    t0 = time.perf_counter()
    for sp in range(max_sp + 1):
        if time.perf_counter() - t0 > budget:
            return None
        r = run_recurrent(rival, bench_file, key, T, steps=1, batch=256, start_point=sp,
                          timeout=PROBE_TIMEOUT)
        if "error" not in r:
            return _remember_sp(cache_dir, bench_file, T, max_sp, sp)
        if "does not require grad" not in r["error"]:
            if "timeout" in r["error"]:
                return None                     # environmental: do NOT cache, it may pass later
            return _remember_sp(cache_dir, bench_file, T, max_sp, sp)
    return None


def _remember_sp(cache_dir, bench_file, T, max_sp, sp):
    """Store a SETTLED probe result. Failures to settle are deliberately not stored: a timeout
    or a reclaimed VM is about the machine, not the circuit."""
    if sp is not None:
        cache_store(cache_dir, bench_file, _sp_task(T, max_sp), sp)
    return sp



def run_recurrent_ladder(rival, bench_file, key, T, steps, batch, start_point,
                         timeout=DEFAULT_TIMEOUT):
    """run_recurrent at its granted batch, stepping DOWN if the driver refuses the size. The
    rungs tried are attached as `batch_sweep`: their unique/s is near-proportional to the batch
    (s386 T=25: 10k -> 300k is 18.6x, with model_runtime almost flat), so what the rival was
    granted belongs in the row next to its score."""
    r, sweep = run_at_batch(
        lambda b: run_recurrent(rival, bench_file, key, T, steps, b, start_point,
                                timeout=timeout),
        batch)
    return dict(r, batch_sweep=sweep) if isinstance(r, dict) else r


def pick_batch(bench_file, T, budget_bytes=None):
    """The batch Recurrent-CircuitSAT is granted: the largest that fits in 0.80 of free VRAM,
    the same share GPAP sizes its own K with (rival_common.fit_batch)."""
    n_gates = sum(1 for line in open(bench_file)
                  if "=" in line and "DFF" not in line and not line.lstrip().startswith("#"))
    return fit_batch(max(n_gates, 1) * T, "recurrent", budget_bytes)


def run_gpap(bench_file, T, pin_outputs, target, seeds, n_time, device, time_limit=60.0,
             n_cov=1000):
    """Convert once (explicit pins), then sample at each seed; report per-seed + medians."""
    from gpap.circuits.bench import parse_bench, build_cnf, write_dimacs
    from gpap.sampling.pipeline import CIRCUIT_PORTFOLIO, sample_file
    circ = parse_bench(bench_file)
    b, support, tgt, labels = build_cnf(circ, T, seed=5, pin_outputs=pin_outputs, target=target)
    with tempfile.NamedTemporaryFile("w", suffix=".cnf", delete=False) as f:
        cnf_path = f.name
    write_dimacs(cnf_path, b, support, labels)
    runs = []
    try:
        for s in seeds:
            # n_cov feeds distinct_frac only, but it COLLECTS models and rebuilds them in
            # Python — real wall on a deep unroll. Keep it small; it is not the measurement.
            # verify_sound: this row RECORDS `sound_check`, so it has to run the check —
            # otherwise the column is a permanent "off" beside a comment claiming it is the
            # on-device model check. Free to the comparison: the check's wall is excluded from
            # every rate (`verify_sound_s`), so the rival is not credited a millisecond for it.
            g = sample_file(cnf_path, device, n_time=n_time, n_cov=n_cov,
                            configs=CIRCUIT_PORTFOLIO, time_limit=time_limit, seed=s,
                            verify_sound=True)
            runs.append({"seed": s, "sps": g["samples_per_s"], "sps_e2e": g.get("sps_e2e_total"),
                         # KEEP None. `or 0.0` turned "the coverage run collected nothing" into
                         # "GPAP produced nothing", and every consumer buckets a zero as an
                         # outright loss — b11-c25 at a rare pin delivered models at ~15x the
                         # rival's rate and was published as a loss to it.
                         "unique_per_s_est": g.get("unique_sps_est"),
                         "unique_est_status": g.get("unique_est_status"),
                         "success": g["success_rate"], "n_models": g["n_models"],
                         # engine provenance per row, same as the CMSGen sweeps: sound_check is
                         # the on-device model check (opt-in), dec/s + impl/s let the zero-search screen
                         # (paper_tables.zero_search) run on rival rows too
                         "sound_check": g.get("sound_check"),
                         "decisions_per_s": g.get("decisions_per_s"),
                         "implications_per_s": g.get("implications_per_s")})
    finally:
        os.unlink(cnf_path)
    med = lambda k: statistics.median(r[k] for r in runs)
    med0 = lambda k: statistics.median((r.get(k) or 0.0) for r in runs)
    # The unique estimate is median over the seeds that HAVE one; None (not 0.0) when no seed
    # produced a coverage sample, so a consumer can tell "not measured" from "produced nothing".
    uq = [r["unique_per_s_est"] for r in runs if r["unique_per_s_est"] is not None]
    return {"seeds": runs,
            "unique_per_s_est": (statistics.median(uq) if uq else None),
            "unique_est_status": ("ok" if uq else
                                  (runs[0].get("unique_est_status") or "no-coverage-sample")),
            "sps": med("sps"),
            "success": med("success"), "n_vars": b.nv, "n_clauses": len(b.clauses),
            # merge_sound_check, not seed 0's verdict: UNSOUND dominates and "ok" beats "off",
            # so a mixed set of seeds cannot publish the first seed's silence as the answer.
            "sound_check": merge_sound_check(*(sound_verdict(r) for r in runs)),
            "decisions_per_s": med0("decisions_per_s"),
            "implications_per_s": med0("implications_per_s"),
            "n_models": int(med("n_models"))}


# ---------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", help="instance stems, e.g. b03 s1196 (see TARGETS)")
    ap.add_argument("--rival", default=RIVAL_DEFAULT)
    ap.add_argument("--bench-dir", default=BENCH_DIR)
    ap.add_argument("--rare", action="store_true",
                    help="our adversarial fixed-T rare pattern (MITER cone) instead of their table pins")
    ap.add_argument("--circuit-variants", nargs="+", default=["raw"], choices=list(VARIANTS),
                    metavar="V",
                    help="which circuit BOTH sides are given: 'raw' (as committed) and/or 'abc' "
                         "(ABC-optimized sequential AIG written back as .bench, equivalence-"
                         "checked) — so an ABC-encoded GPAP arm is never compared against a "
                         "rival that only ever saw the unoptimized circuit")
    ap.add_argument("-T", "--cycles", type=int, default=25)
    ap.add_argument("--steps", default="auto",
                    help="their num_steps: 'auto' (default) = the value their own artifacts "
                         "specify, see rival_common.RIVAL_STEPS; or an explicit number")
    ap.add_argument("--gpap-cache", default=None,
                    help="directory of cached GPAP arms (default: <output dir>/.gpap_cache). "
                         "GPAP_RIVAL_CACHE=0 disables reuse entirely.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[5, 6, 7])
    ap.add_argument("--gpap-n", type=int, default=50_000)
    ap.add_argument("--target-seed", type=int, default=5, help="rare-pattern selection seed")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                    help="per-run wall cap for the RIVAL subprocess (s); a run that exceeds it "
                         "is an error row, not a zero")
    ap.add_argument("--gpap-time-limit", type=float, default=60.0,
                    help="per-seed sampling wall cap for the GPAP side (s)")
    ap.add_argument("--output", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="keep the rows already in --output and re-run only what is missing "
                         "(a rented VM can be reclaimed mid-stage; see eval/colab/colab_supervisor.py)")
    ap.add_argument("--retry-errors", action="store_true",
                    help="with --resume, also re-run rows that previously ERRORED. Off by "
                         "default: a rival timeout is a recorded result and costs the full "
                         "timeout to rediscover on every restart. Turn it on when the failure "
                         "was ENVIRONMENTAL (e.g. a missing dependency), not the rival's.")
    args = ap.parse_args()
    if not os.path.isdir(os.path.join(args.rival, "src")):
        sys.exit(f"no rival checkout at {args.rival} (see module docstring)")
    import torch
    from gpap.circuits.simulate import sim_baseline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps, steps_src = resolve_steps("recurrent", args.steps)
    tmp = tempfile.mkdtemp(prefix="rivals_recurrent_")
    mode = "rare(fixed-T pattern, MITER)" if args.rare else "their target table"
    print(f"{len(args.names)} instances | T={args.cycles} steps={steps} ({steps_src}) | mode: {mode} | "
          f"GPAP median of seeds {args.seeds}")
    hdr = (f"{'instance':<10}{'circ':>6}{'pin':>12}{'p_acc':>10}{'RCS uniq':>11}{'@T':>9}"
           f"{'t':>7}{'RCS u/s':>10}{'GPAP u/s~':>11}{'succ%':>7}{'ratio':>10}")
    print(hdr)
    print("-" * len(hdr))
    out = []
    done = set()
    # Re-running this stage for a RIVAL-side reason (a wider batch ladder, a changed timeout) must
    # not re-pay OUR arm: 3 seeds x 60 s per cell here, and of the 2026-08-23 campaign's ~3.3 h
    # rival stage only 64 min was the rivals' own subprocesses. Keyed on the sampled file's bytes
    # + the task + the engine build + the GPU, so anything that could move the number is a miss.
    cache_dir = args.gpap_cache or (os.path.join(os.path.dirname(args.output) or ".",
                                                 ".gpap_cache") if args.output else None)
    if args.resume and args.output and os.path.isfile(args.output):
        try:
            out = json.load(open(args.output)) or []
            if args.retry_errors:      # drop them so the retry REPLACES rather than appends
                out = [r for r in out if "error" not in (r.get("rcs") or {})]
            done = {(r["name"], r.get("variant", "-")) for r in out}
            print(f"[resume] {len(out)} rows already in {args.output}")
        except Exception as e:      # noqa: BLE001 — a truncated checkpoint must not kill the run
            print(f"[resume] ignoring unreadable {args.output}: {e}")
            out, done = [], set()
    for name, variant in ((n, v) for n in args.names for v in args.circuit_variants):
        if (name, variant) in done:
            continue
        src0 = os.path.join(args.bench_dir, f"{name}.bench")
        if not os.path.isfile(src0):
            print(f"{name:<10}  no such bench file, skipped")
            continue
        # BOTH sides get the same circuit (eval/rivals/rival_common.py): with 'abc' the rival is handed
        # the ABC-optimized .bench too, gated by a bit-exact simulation equivalence check.
        src, vinfo = prepare_variant(src0, variant, tmp, cycles=args.cycles,
                                     seed=args.target_seed,
                                     target_mode="rare" if args.rare else "random")
        if vinfo.get("error"):
            print(f"{name:<10}{variant:>6}  circuit variant unusable: {vinfo['error'][:60]}",
                  flush=True)
            continue
        if args.rare:
            rival_file = os.path.join(tmp, f"{name}_{variant}_rare.bench")
            pattern, est = miter_bench(src, rival_file, args.cycles, args.target_seed)
            key = RARE_ALIAS                       # their table: [-1]=1 == MITER=1
            gpap_file, pin_n = rival_file, None    # single MITER output, pin all(=1)
            gpap_target = {"GPAPMITER": True}
            pin_desc = f"rare|{len(pattern)}o"
        else:
            if name not in TARGETS:
                print(f"{name:<10}  not in their target table, skipped")
                continue
            idx, vals = TARGETS[name]
            key = name                             # untouched copy hits their table branch
            rival_file = src
            gpap_file = os.path.join(tmp, f"{name}_{variant}_reord.bench")
            picked = reorder_bench(src, gpap_file, idx)
            gpap_target = {w: bool(v) for w, v in zip(picked, vals)}
            pin_n = len(idx)
            pin_desc = ",".join(f"[{i}]={v}" for i, v in zip(idx, vals))
        sim = sim_baseline(gpap_file, args.cycles, args.target_seed, pin_outputs=pin_n,
                           target=gpap_target)
        sp = probe_start_point(args.rival, rival_file, key, args.cycles,
                               cache_dir=cache_dir)                        # bounded, see above
        if sp is None:
            rcs = {"error": "no gradient path at any start_point<=10"}
        else:
            batch = pick_batch(rival_file, args.cycles)
            rcs = run_recurrent_ladder(args.rival, rival_file, key, args.cycles, steps,
                                       batch, sp, timeout=args.timeout)
            rcs["steps_source"] = steps_src
        task = {"harness": "recurrent", "mode": "rare" if args.rare else "table",
                "cycles": args.cycles, "pin_outputs": pin_n, "target": gpap_target,
                "seeds": args.seeds, "n_time": args.gpap_n,
                "time_limit": args.gpap_time_limit, "portfolio": "circuit"}
        g, hit = cached_gpap(cache_dir, gpap_file, task,
                             lambda: run_gpap(gpap_file, args.cycles, pin_n, gpap_target,
                                              args.seeds, args.gpap_n, device,
                                              time_limit=args.gpap_time_limit))
        if hit:
            print(f"{name:<10}{variant:>6}  [gpap-cache hit — arm not re-run]", flush=True)
        row = {"name": name, "mode": "rare" if args.rare else "table", "pin": pin_desc,
               "variant": variant, "circuit": vinfo,
               "p_accept": sim["p_accept"], "p_accept_lt": sim.get("p_accept_lt"),
               "rival_commit": rival_commit(args.rival), "rcs": rcs, "gpap": g,
               "vram_budget_bytes": rival_vram_budget()}
        out.append(row)
        if args.output:      # checkpoint after every instance: this stage runs for HOURS, and a
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)   # death at 90% used
            json.dump(out, open(args.output, "w"), indent=1)                  # to leave nothing
        pa = sim["p_accept"] if sim["p_accept"] else sim.get("p_accept_lt", 0.0)
        pa_s = f"{pa:.2e}" if sim["p_accept"] else f"<{pa:.0e}"
        # None = the off-timing coverage run collected nothing, i.e. NOT MEASURED. It is not a
        # zero: `n_models` says whether GPAP produced anything, and on a rare pin it routinely
        # did while this estimate did not exist.
        uq = g.get("unique_per_s_est")
        uq_s = f"{uq:,.0f}" if uq is not None else "n/a"
        if "error" in rcs:
            print(f"{name:<10}{variant:>6}{pin_desc:>12}{pa_s:>10}  RCS FAILED: "
                  f"{rcs['error'][:44]} | GPAP u/s~={uq_s} "
                  f"succ={100 * g['success']:.0f}%", flush=True)
            continue
        ratio = (None if uq is None else
                 (uq / rcs["unique_per_s"] if rcs["unique_per_s"] else float("inf")))
        rt = ("      n/a" if ratio is None else            # rival silent -> no ratio,
              "     GPAP" if ratio == float("inf") else    # an outright win, not "inf"
              f"{ratio:>9.1f}x")
        at_t = f"{rcs['unique_at_T']:,}" if rcs.get("unique_at_T") is not None else "—"
        print(f"{name:<10}{variant:>6}{pin_desc:>12}{pa_s:>10}{rcs['unique']:>11,}{at_t:>9}"
              f"{rcs['runtime_s']:>6.1f}s{rcs['unique_per_s']:>10,.0f}"
              f"{uq_s:>11}{100 * g['success']:>6.0f}%{rt}",
              flush=True)
    if len(args.circuit_variants) > 1:
        print("\ncircuit variants (BOTH sides get the same circuit; abc = sequential-AIG "
              "optimization, equivalence-checked):")
        for r in out:
            print(f"  {r['name']:<8} {variant_note(r['circuit'])}")
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        json.dump(out, open(args.output, "w"), indent=1)
        print(f"wrote {args.output}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
