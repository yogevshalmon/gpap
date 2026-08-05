#!/usr/bin/env python3
"""Head-to-head: DEMOTIC (ASP-DAC'25, arXiv 2502.08086 — the Berkeley gradient family's
CIRCUIT-level sampler: it consumes .bench directly, no CNF) vs GPAP, same circuits, same GPU.

Two modes, because "who wins" depends entirely on whose task is being sampled:

  --mode native  (their home turf): DEMOTIC's own benchmarks with DEMOTIC's own hard-coded
      per-module output pins — the table below is transcribed from their run.py
      (`_initialize_model` / `run_back_prop_*` / `_check_solution` all key off module_name, which
      comes from the FILENAME). GPAP is given the IDENTICAL constraint: the .bench is copied with
      those output positions moved to the front, then build_cnf pins exactly them to exactly
      their target bits. Their pins are 1-5 outputs of a wide circuit, i.e. rejection-easy by
      construction — p_accept is measured and printed so that is visible, not assumed.

  --mode miter   (our task): the GPAP CircuitSAT-sampling instance — T-cycle unroll, ALL outputs
      pinned to a `--target rare` pattern — handed to DEMOTIC as a single-output combinational
      miter (eval/unroll_to_bench.py, which self-checks p(MITER=1) == the instance's p_accept)
      named `c17.bench`, since their c17 branch pins exactly `outputs[-1] = 1`. GPAP samples the
      original instance directly. Same solution set, both sides.

Their metric (used verbatim): num_unique_solutions / model_runtime, where model_runtime times the
gradient loop ONLY — their timer excludes model build AND the solution re-check pass. GPAP's
comparable figure is unique_sps_est (raw samples/s x distinct fraction, dedup measured off the
timing path) and its rate INCLUDES construction. Both generous to them, kept that way.

Setup (once): git clone https://github.com/arashardakani/Demotic \
                  ~/.cache/gpap/src/Demotic
Usage:
  PYTHONPATH=. .venv/bin/python eval/rivals_demotic.py --mode native c432 c880 c1355
  PYTHONPATH=. .venv/bin/python eval/rivals_demotic.py --mode miter b03 b13 s1196 -T 25 \
      --target rare --output .experiments/rivals_demotic_rare.json
"""
import argparse
import glob
import json
import keyword
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from eval.rival_common import VARIANTS, prepare_variant, variant_note   # noqa: E402
from eval.rivals_recurrent import reorder_bench, run_gpap          # noqa: E402
from eval.unroll_to_bench import write_unrolled_miter              # noqa: E402

DEMOTIC_DEFAULT = os.path.expanduser("~/.cache/gpap/src/Demotic")
BENCH_COMB = os.path.join(REPO_ROOT, "benchmarks", "circuits", "bench", "comb")
BENCH_SEQ = os.path.join(REPO_ROOT, "benchmarks", "circuits", "bench", "seq")
MITER_ALIAS = "c17"          # their single-output branch: pins outputs[-1] = 1
# Per-SUBPROCESS wall cap for the rival, default 300 s = 5x the 60 s sampling budget GPAP gets
# per instance. Deliberately GENEROUS and not symmetric: the rival's wall includes building its
# PyTorch model, which its OWN metric (unique / model_runtime) excludes, so a tight cap would
# truncate runs it would have completed. MEASURED on this suite: DEMOTIC median 3.7-15.3 s
# (1/38 runs over 60 s), Recurrent-CircuitSAT median ~165 s with a 665 s tail (18/18 over 60 s,
# 5/18 over 300 s) — a 60 s cap would have produced ZERO valid Recurrent measurements.
# A timed-out run is reported as an error row with the reason, never as a zero that would enter
# a geomean as if the rival had genuinely produced nothing.
DEFAULT_TIMEOUT = 300.0


# ------------------------------------------------------------------ their target table
# Transcribed from Demotic/src/pytorch/run.py. Each entry is (output indices, target bits); the
# indices are positions in the .bench OUTPUT() order (their parser and ours both keep file order).
def demotic_target(module_name: str, circuit_type: str):
    if circuit_type == "comb":
        if module_name in ("c17",):
            return [-1], [1]
        if module_name in ("c432",):
            return [0, -1], [1, 0]
        if module_name in ("c880", "c1908", "c3540", "c499", "c1355", "c6288"):
            return [0, 15, -1], [1, 0, 1]
        return [0, 15, 31, 63, -1], [1, 0, 1, 0, 1]
    if module_name in ("s27", "s2081", "s298", "s382", "s386", "s400", "s4201", "s444", "s510",
                       "s526", "s526n", "s635", "s8381", "s938", "s1423"):
        return [-1], [1]
    if module_name in ("s344", "s349", "s1196", "s1238", "s1269", "s3271", "s4863"):
        return [0, -1], [0, 1]
    if module_name in ("s499", "s641", "s713", "s820", "s832", "s953", "s967", "s991", "s1488",
                       "s1494", "s1512", "s3384", "s5378", "s6669", "s92341", "s9234"):
        return [0, 15, -1], [0, 1, 0]
    return [0, 15, 31, 63, -1], [1, 0, 1, 0, 1]


# ------------------------------------------------------------------ bench sanitising
# DEMOTIC GENERATES PYTHON SOURCE from the .bench (signal name -> local variable) and exec's it,
# so every signal name must be a valid Python identifier. The classic ISCAS-85 distribution we
# commit names its wires with bare integers (`INPUT(1)`, `223 = NAND(...)`) — their own copies use
# `G1gat` — and their run.py dies with "SyntaxError: cannot assign to literal". Renaming a wire is
# a pure alpha-conversion: same circuit, same OUTPUT() order, so their index-based target table
# still selects the same outputs and the comparison is unaffected.
_IDENT = re.compile(r"[A-Za-z_]\w*\Z")
_IO_LINE = re.compile(r"^\s*(INPUT|OUTPUT)\s*\(([^)]*)\)\s*$", re.I)
_GATE_LINE = re.compile(r"^\s*([^=\s]+)\s*=\s*(\w+)\s*\(([^)]*)\)\s*$")


def _safe_name(tok: str) -> str:
    tok = tok.strip()
    if _IDENT.match(tok) and not keyword.iskeyword(tok):
        return tok
    return "g" + re.sub(r"\W", "_", tok)


# Their pgates module defines AND/NAND/OR/NOR/XOR/XNOR/NOT/BUF. The ISCAS distribution spells the
# buffer BUFF, which their generated source then calls as an undefined name — same gate, one
# alias apart.
_OP_ALIAS = {"BUFF": "BUF", "BUFIF": "BUF"}


def sanitize_bench(src: str, dst: str) -> bool:
    """Copy `src` to `dst` with every signal name made a valid Python identifier and every gate
    op spelled the way their pgates module defines it. Returns True if anything was rewritten.
    Comments and line order are preserved."""
    changed = False
    out = []
    for line in open(src):
        body = line.rstrip("\n")
        m = _IO_LINE.match(body)
        if m:
            nm = _safe_name(m.group(2))
            changed |= nm != m.group(2).strip()
            out.append(f"{m.group(1).upper()}({nm})")
            continue
        m = _GATE_LINE.match(body)
        if m and not body.lstrip().startswith("#"):
            lhs = _safe_name(m.group(1))
            op = _OP_ALIAS.get(m.group(2).upper(), m.group(2))
            ins = [_safe_name(x) for x in m.group(3).split(",") if x.strip()]
            changed |= (lhs != m.group(1).strip() or op != m.group(2)
                        or ins != [x.strip() for x in m.group(3).split(",") if x.strip()])
            out.append(f"{lhs} = {op}({', '.join(ins)})")
            continue
        out.append(body)
    with open(dst, "w") as f:
        f.write("\n".join(out) + "\n")
    return changed


# ------------------------------------------------------------------ running DEMOTIC

def run_demotic(root, bench_file, key, *, steps, batch, circuit_type, T, lr="15e0",
                timeout=DEFAULT_TIMEOUT, retry=True):
    """One DEMOTIC run; returns dict(unique, runtime_s, unique_per_s, ...) or {'error': ...}.

    `key` = the filename stem the copy is given: their target table AND their results-csv name
    are both keyed on it. Their `-d` path is resolved against THEIR src/, hence the ../data form.
    """
    data_dir = os.path.join(root, "data", "rival_tmp")
    os.makedirs(data_dir, exist_ok=True)
    shutil.copy(bench_file, os.path.join(data_dir, f"{key}.bench"))
    res_dir = os.path.join(root, "src", "results")
    for f in glob.glob(os.path.join(res_dir, f"sat_{key}.bench_*")):
        os.unlink(f)
    cmd = [sys.executable, os.path.join(root, "src", "pytorch", "run.py"),
           "-d", f"../data/rival_tmp/{key}.bench", "-l", "--optimizer", "mse", "--lr", lr,
           "--problem_type", "BLIF", "--circuit_type", circuit_type,
           "--num_steps", str(steps), "--batch_size", str(batch)]
    if circuit_type == "seq":
        cmd += ["--num_clock_cycles", str(T)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=os.path.join(root, "script"), capture_output=True,
                           text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout {timeout}s", "wall_s": timeout}
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        err = ((p.stderr or p.stdout).strip().splitlines() or ["?"])[-1][:200]
        # transient WSL2 driver hiccup between back-to-back CUDA processes — settle and retry
        if "device not ready" in err and retry:
            time.sleep(10)
            return run_demotic(root, bench_file, key, steps=steps, batch=batch,
                               circuit_type=circuit_type, T=T, lr=lr, timeout=timeout,
                               retry=False)
        return {"error": err, "wall_s": wall}
    hits = glob.glob(os.path.join(res_dir, f"sat_{key}.bench_*"))
    if not hits:
        return {"error": "no results csv (dataset not found?)", "wall_s": wall}
    import csv
    with open(max(hits, key=os.path.getmtime)) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        return {"error": "empty results csv", "wall_s": wall}
    r = rows[0]
    uniq = int(float(str(r["num_unique_solutions"]).strip("[] ")))
    rt = float(r["model_runtime"])
    return {"unique": uniq, "runtime_s": rt, "unique_per_s": uniq / rt if rt else 0.0,
            "wall_s": wall, "batch": batch, "steps": steps,
            "num_inputs": r.get("num_inputs"), "num_outputs": r.get("num_outputs")}


BATCH_LADDER = (1_000_000, 300_000, 100_000, 30_000, 10_000, 3_000)


def pick_batch(bench_file, T, budget_bytes=5 << 30):
    """Largest batch whose unrolled autograd graph (#gates x T x batch x 4B) fits the budget —
    this box SPILLS to host RAM instead of OOMing, so we must not rely on allocator failure."""
    n_gates = sum(1 for line in open(bench_file)
                  if "=" in line and "DFF" not in line and not line.lstrip().startswith("#"))
    for b in BATCH_LADDER:
        if max(n_gates, 1) * T * b * 4 <= budget_bytes:
            return b
    return 3_000


def run_demotic_ladder(root, bench_file, key, *, steps, batch, circuit_type, T,
                       timeout=DEFAULT_TIMEOUT):
    """run_demotic, stepping down the batch ladder when the driver refuses the size ('device not
    ready' reproduces at a fixed batch on some models yet passes one rung lower)."""
    ladder = [b for b in BATCH_LADDER if b <= batch] or [BATCH_LADDER[-1]]
    r = {"error": "empty ladder"}
    for b in ladder[:3]:
        r = run_demotic(root, bench_file, key, steps=steps, batch=b,
                        circuit_type=circuit_type, T=T, timeout=timeout)
        if "error" not in r or "device not ready" not in r["error"]:
            return r
    return r


# ------------------------------------------------------------------ driver

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="+", help="instance stems, e.g. c880 (native) or b13 (miter)")
    ap.add_argument("--mode", choices=["native", "miter"], default="native",
                    help="'native' = their benchmarks + their pins | 'miter' = OUR unrolled "
                         "rare-target instance handed to them as a single-output miter")
    ap.add_argument("--demotic", default=DEMOTIC_DEFAULT)
    ap.add_argument("--bench-dir", default=None,
                    help="default: bench/comb for --circuit-type comb, bench/seq for seq")
    ap.add_argument("--circuit-type", choices=["comb", "seq"], default="comb",
                    help="native mode: which of their two model paths (comb = ISCAS-85 at T=1, "
                         "seq = ISCAS-89/ITC'99 unrolled). miter mode always feeds them comb")
    ap.add_argument("-T", "--cycles", type=int, default=25, help="unroll depth (seq / miter)")
    ap.add_argument("--target", choices=["random", "rare"], default="rare",
                    help="miter mode: which pinned pattern OUR instance uses")
    ap.add_argument("--circuit-variants", nargs="+", default=["raw"], choices=list(VARIANTS),
                    metavar="V",
                    help="which circuit each side is given: 'raw' (as committed) and/or 'abc' "
                         "(ABC-optimized sequential AIG, written back as .bench and "
                         "equivalence-checked). Pass both to show the encoder is not what "
                         "decides the comparison — the rival gets the optimized circuit too")
    ap.add_argument("--steps", type=int, default=10, help="their num_steps (their script: 1-100)")
    ap.add_argument("--batch", type=int, default=None, help="their batch (default: fit to VRAM)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[5, 6, 7],
                    help="GPAP sampling seeds; the row reports the median")
    ap.add_argument("--gpap-n", type=int, default=50_000)
    ap.add_argument("--target-seed", type=int, default=5, help="pattern-selection seed")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                    help="per-run wall cap for the RIVAL subprocess (s)")
    ap.add_argument("--gpap-time-limit", type=float, default=60.0,
                    help="per-seed sampling wall cap for the GPAP side (s)")
    ap.add_argument("--output", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="keep the rows already in --output and re-run only what is missing "
                         "(a rented VM can be reclaimed mid-stage; see eval/colab_supervisor.py)")
    ap.add_argument("--retry-errors", action="store_true",
                    help="with --resume, also re-run rows that previously ERRORED. Off by "
                         "default: a rival timeout is a recorded result and costs the full "
                         "timeout to rediscover on every restart. Turn it on when the failure "
                         "was ENVIRONMENTAL (e.g. a missing dependency), not the rival's.")
    args = ap.parse_args()
    if not os.path.isdir(os.path.join(args.demotic, "src")):
        sys.exit(f"no DEMOTIC checkout at {args.demotic} (see module docstring)")
    import torch
    from gpap.circuits.simulate import sim_baseline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bench_dir = args.bench_dir or (BENCH_COMB if (args.mode == "native" and
                                                 args.circuit_type == "comb") else BENCH_SEQ)
    tmp = tempfile.mkdtemp(prefix="rivals_demotic_")
    mode_txt = ("their target table" if args.mode == "native"
                else f"OUR {args.target} fixed-T instance as a miter")
    print(f"{len(args.names)} instances | mode: {args.mode} ({mode_txt}) | "
          f"T={args.cycles if args.mode == 'miter' or args.circuit_type == 'seq' else 1} "
          f"steps={args.steps} | GPAP median of seeds {args.seeds}")
    hdr = (f"{'instance':<10}{'circ':>6}{'pin':>14}{'p_acc':>10}{'DEM uniq':>10}{'t':>8}"
           f"{'DEM u/s':>10}{'GPAP u/s~':>11}{'GPAP/s':>11}{'succ%':>7}{'ratio':>10}")
    print(hdr)
    print("-" * len(hdr))
    out = []
    done = set()
    if args.resume and args.output and os.path.isfile(args.output):
        try:
            prev = json.load(open(args.output))
            out = (prev.get("rows") if isinstance(prev, dict) else prev) or []
            if args.retry_errors:      # drop them so the retry REPLACES rather than appends
                out = [r for r in out if "error" not in (r.get("demotic") or {})]
            done = {(r["name"], r.get("variant", "-")) for r in out}
            print(f"[resume] {len(out)} rows already in {args.output}")
        except Exception as e:      # noqa: BLE001 — a truncated checkpoint must not kill the run
            print(f"[resume] ignoring unreadable {args.output}: {e}")
            out, done = [], set()

    def checkpoint():
        """Write the JSON after every row, not only at the end — this stage runs for hours on a
        remote box, and a run that dies at 90% used to leave nothing at all behind."""
        if not args.output:
            return
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        json.dump({"mode": args.mode, "circuit_type": args.circuit_type, "cycles": args.cycles,
                   "target": args.target, "steps": args.steps,
                   "circuit_variants": list(args.circuit_variants), "rows": out},
                  open(args.output, "w"), indent=1)

    for name, variant in ((n, v) for n in args.names for v in args.circuit_variants):
        if (name, variant) in done:
            continue
        src0 = os.path.join(bench_dir, f"{name}.bench")
        if not os.path.isfile(src0):
            print(f"{name:<10}  no such bench file ({src0}), skipped")
            continue
        # BOTH sides get the same circuit: 'abc' hands the rival the ABC-optimized .bench too,
        # gated by a bit-exact simulation equivalence check (eval/rival_common.py).
        src, vinfo = prepare_variant(src0, variant, tmp, cycles=args.cycles,
                                     seed=args.target_seed, target_mode=args.target)
        if vinfo.get("error"):
            print(f"{name:<10}{variant:>6}  circuit variant unusable: {vinfo['error'][:60]}",
                  flush=True)
            continue
        if args.mode == "miter":
            # Their side: our unrolled instance as a single-output miter, named c17 so their
            # table pins MITER=1. Our side: the ORIGINAL circuit at the same T/target — the
            # miter writer self-checks that the two describe the same solution set.
            rival_file = os.path.join(tmp, f"{name}_{variant}_miter.bench")
            n_in, p_miter = write_unrolled_miter(src, rival_file, args.cycles,
                                                 args.target_seed, target_mode=args.target)
            key, circuit_type, T_rival = MITER_ALIAS, "comb", 1
            # OUR side must pin the SAME pattern the miter encodes — resolve it through the very
            # helper write_unrolled_miter used (same seed => same pins), never let build_cnf fall
            # back to its own random target.
            from gpap.circuits.bench import parse_bench
            from gpap.circuits.simulate import _pin_and_target
            _, gpap_target, _ = _pin_and_target("bench", parse_bench(src), args.cycles,
                                                args.target_seed, None, args.target)
            gpap_file, pin_n = src, None
            pin_desc = f"{args.target}|{len(gpap_target)}o"
            sim = sim_baseline(src, args.cycles, args.target_seed, target_mode=args.target)
            gpap_T = args.cycles
        else:
            idx, vals = demotic_target(name, args.circuit_type)
            key, circuit_type = name, args.circuit_type
            T_rival = args.cycles if args.circuit_type == "seq" else 1
            gpap_T = T_rival
            rival_file = src
            gpap_file = os.path.join(tmp, f"{name}_{variant}_reord.bench")
            picked = reorder_bench(src, gpap_file, idx)      # their pins moved to the front
            gpap_target = {w: bool(v) for w, v in zip(picked, vals)}
            pin_n = len(idx)
            pin_desc = ",".join(f"[{i}]={v}" for i, v in zip(idx, vals))
            sim = sim_baseline(gpap_file, gpap_T, args.target_seed, pin_outputs=pin_n,
                               target=gpap_target)
        # Their side only: make every wire name a valid Python identifier (they exec generated
        # source). Pure alpha-rename — our side keeps the original file.
        rival_safe = os.path.join(tmp, f"{name}_{variant}_{args.mode}_safe.bench")
        renamed = sanitize_bench(rival_file, rival_safe)
        rival_file = rival_safe
        batch = args.batch or pick_batch(rival_file, T_rival)
        dem = run_demotic_ladder(args.demotic, rival_file, key, steps=args.steps, batch=batch,
                                 circuit_type=circuit_type, T=T_rival, timeout=args.timeout)
        dem["renamed_wires"] = renamed
        g = run_gpap(gpap_file, gpap_T, pin_n, gpap_target, args.seeds, args.gpap_n, device,
                     time_limit=args.gpap_time_limit)
        row = {"name": name, "mode": args.mode, "variant": variant, "circuit": vinfo,
               "pin": pin_desc, "cycles": gpap_T,
               "p_accept": sim["p_accept"], "p_accept_lt": sim.get("p_accept_lt"),
               "demotic": dem, "gpap": g, "batch": batch}
        if args.mode == "miter":
            # the miter's own acceptance probability (write_unrolled_miter asserts it equals
            # p_accept) — the evidence that both samplers were given the same task
            row["p_miter"] = p_miter
            row["miter_inputs"] = n_in
        out.append(row)
        checkpoint()
        pa = sim["p_accept"] or sim.get("p_accept_lt", 0.0)
        pa_s = f"{pa:.2e}" if sim["p_accept"] else f"<{pa:.0e}"
        pin_desc = pin_desc if len(pin_desc) <= 14 else pin_desc[:11] + "..."   # keep the columns
        if "error" in dem:
            print(f"{name:<10}{variant:>6}{pin_desc:>14}{pa_s:>10}  DEMOTIC FAILED: "
                  f"{dem['error'][:38]} | GPAP u/s~={g['unique_per_s_est']:,.0f} "
                  f"succ={100 * g['success']:.0f}%", flush=True)
            continue
        ratio = (g["unique_per_s_est"] / dem["unique_per_s"] if dem["unique_per_s"]
                 else float("inf"))
        rt = f"{ratio:,.1f}x" if ratio != float("inf") else "GPAP"
        print(f"{name:<10}{variant:>6}{pin_desc:>14}{pa_s:>10}{dem['unique']:>10,}"
              f"{dem['runtime_s']:>7.1f}s{dem['unique_per_s']:>10,.0f}"
              f"{g['unique_per_s_est']:>11,.0f}{g['sps']:>11,.0f}"
              f"{100 * g['success']:>6.0f}%{rt:>10}", flush=True)
    if len(args.circuit_variants) > 1:
        print("\ncircuit variants (both sides get the SAME circuit; abc = sequential-AIG "
              "optimization, equivalence-checked):")
        for r in out:
            print(f"  {r['name']:<8} {variant_note(r['circuit'])}")
    if args.output:
        checkpoint()
        print(f"wrote {args.output}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
