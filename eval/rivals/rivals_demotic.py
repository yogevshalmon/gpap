#!/usr/bin/env python3
"""Head-to-head: DEMOTIC (ASP-DAC'25, arXiv 2502.08086 — the Berkeley gradient family's
CIRCUIT-level sampler: it consumes .bench directly, no CNF) vs GPAP, same circuits, same GPU.

Two modes, because "who wins" depends on whose task is sampled:
  --mode native  DEMOTIC's own benchmarks under its own per-module output pins, transcribed
      below from their run.py. GPAP gets the IDENTICAL constraint: those output positions moved
      to the front of a copy, then build_cnf pins exactly them to exactly their bits. Their pins
      are rejection-easy, so p_accept is measured and printed, not assumed.
  --mode miter   our task — the T-cycle unroll with ALL outputs pinned to a rare pattern, as a
      single-output combinational miter (unroll_to_bench.py, self-checking p(MITER=1) ==
      p_accept) named `c17.bench`, their branch that pins `outputs[-1] = 1`.

Their metric verbatim: num_unique_solutions / model_runtime, timing the gradient loop ONLY,
against GPAP's `unique_per_s_est` whose rate INCLUDES construction. Their timer is emission-free,
so the comparison is like-for-like (eval/rivals/check_emission_symmetry.py).
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
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # eval/<sub>/ -> repo
sys.path.insert(0, REPO_ROOT)

from eval.rivals.rival_common import (BATCH_LADDER, VARIANTS, _available_bytes,  # noqa: E402
                                      cached_gpap, cmsgen_arm, expand_names, fit_batch,
                                      host_provenance, native_pin_check, prepare_variant,
                                      rival_commit,
                                      RIVAL_SEED, rival_vram_budget, resolve_lr, resolve_steps,
                                      run_at_batch, grant_source, load_grants, lookup_grant,
                                      lookup_recorded_failure,
                                      pinned_ladder, variant_note)
from eval.rivals.rivals_recurrent import (GPAP_ENCODER, GPAP_PORTFOLIO,   # noqa: E402
                                          UNIQUE_ROWS, _cms_cols, build_arm_cnf,
                                          reorder_bench, run_gpap)
from eval.rivals.unroll_to_bench import write_unrolled_miter              # noqa: E402

from gpap import paths

DEMOTIC_DEFAULT = str(paths.src_dir() / "Demotic")
BENCH_COMB = os.path.join(REPO_ROOT, "benchmarks", "circuits", "bench", "comb")
BENCH_SEQ = os.path.join(REPO_ROOT, "benchmarks", "circuits", "bench", "seq")
MITER_ALIAS = "c17"          # their single-output branch: pins outputs[-1] = 1
# Per-SUBPROCESS wall cap for the rival, 5x the sampling budget GPAP gets: GENEROUS and not
# symmetric, since the rival's wall includes the model build its OWN metric excludes.
DEFAULT_TIMEOUT = 300.0


# ------------------------------------------------------------------ their target table
# From Demotic/src/pytorch/run.py: (output indices, target bits), indices being positions in the
# .bench OUTPUT() order, which their parser and ours both keep.
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
# DEMOTIC GENERATES PYTHON SOURCE from the .bench and exec's it, so signal names must be valid
# identifiers; renaming is a pure alpha-conversion, same OUTPUT() order and target table.
_IDENT = re.compile(r"[A-Za-z_]\w*\Z")
_IO_LINE = re.compile(r"^\s*(INPUT|OUTPUT)\s*\(([^)]*)\)\s*$", re.I)
_GATE_LINE = re.compile(r"^\s*([^=\s]+)\s*=\s*(\w+)\s*\(([^)]*)\)\s*$")


def _safe_name(tok: str) -> str:
    tok = tok.strip()
    if _IDENT.match(tok) and not keyword.iskeyword(tok):
        return tok
    return "g" + re.sub(r"\W", "_", tok)


# Their pgates module defines BUF where the ISCAS distribution spells the buffer BUFF.
_OP_ALIAS = {"BUFF": "BUF", "BUFIF": "BUF"}


def sanitize_bench(src: str, dst: str) -> bool:
    """Copy `src` to `dst` with valid identifiers and their pgates spelling; True if rewritten."""
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

def run_demotic(root, bench_file, key, *, steps, batch, circuit_type, T, lr=None,
                timeout=DEFAULT_TIMEOUT, retry=True):
    """One DEMOTIC run; dict(unique, runtime_s, ...) or {'error': ...}. `key` = the stem their
    target table and results-csv name are keyed on."""
    data_dir = os.path.join(root, "data", "rival_tmp")
    os.makedirs(data_dir, exist_ok=True)
    shutil.copy(bench_file, os.path.join(data_dir, f"{key}.bench"))
    res_dir = os.path.join(root, "src", "results")
    for f in glob.glob(os.path.join(res_dir, f"sat_{key}.bench_*")):
        os.unlink(f)
    cmd = [sys.executable, os.path.join(root, "src", "pytorch", "run.py"),
           "-d", f"../data/rival_tmp/{key}.bench", "-l", "--optimizer", "mse",
           # their own run script's value; RIVAL_LR carries the provenance and the tie-break
           "--lr", f"{(lr if lr is not None else resolve_lr('demotic')[0]):g}e0"
           if not isinstance(lr, str) else lr,
           "--problem_type", "BLIF", "--circuit_type", circuit_type,
           # THEIR seed, passed explicitly so the row can say which initialization was measured
           "--seed", str(RIVAL_SEED),
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
    # keep the traceback TAIL, not just the last line, which rarely says which frame raised
        err = (p.stderr or p.stdout or "?").strip()[-300:]
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


def pick_batch(bench_file, T, budget_bytes=None):
    """The batch DEMOTIC is granted: the largest fitting 0.80 of free VRAM (fit_batch)."""
    n_gates = sum(1 for line in open(bench_file)
                  if "=" in line and "DFF" not in line and not line.lstrip().startswith("#"))
    return fit_batch(max(n_gates, 1) * T, "demotic", budget_bytes)


def run_demotic_ladder(root, bench_file, key, *, steps, batch, circuit_type, T,
                       timeout=DEFAULT_TIMEOUT, lr=None, grant=None):
    """run_demotic at the LARGEST ladder rung that completes (rival_common.run_at_batch, which
    also fires this process's one discarded warmup). The rungs tried go in `batch_sweep`."""
    pinned = grant["batch"] if grant else None
    ladder = pinned_ladder(BATCH_LADDER, pinned) if pinned else BATCH_LADDER
    r, sweep = run_at_batch(
        lambda b: run_demotic(root, bench_file, key, steps=steps, batch=b,
                              circuit_type=circuit_type, T=T, timeout=timeout, lr=lr),
        batch, ladder=ladder, tool="demotic")
    if isinstance(r, dict):
        r = dict(r, batch_sweep=sweep)
        r["grant_source"] = grant_source(pinned, r, grant["source"] if grant else None)
    return r


# ------------------------------------------------------------------ driver

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="+",
                    help="instance stems, e.g. c880 (native) or b13 (miter); `paper` = the 11 "
                         "ISCAS-85 their own Table II evaluates (rival_common.PAPER_INSTANCES)")
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
    ap.add_argument("--steps", default="auto",
                    help="their num_steps: 'auto' (default) = the value their own artifacts "
                         "specify, see rival_common.RIVAL_STEPS; or an explicit number")
    ap.add_argument("--lr", default="auto",
                    help="their learning rate: 'auto' (default) = their own run script's value "
                         "(rival_common.RIVAL_LR); or an explicit number")
    ap.add_argument("--batch", type=int, default=None,
                    help="pin ONE batch rung for every instance (default: the largest rung of "
                         "the ladder that completes, walked down from the top)")
    ap.add_argument("--honour-recorded-failures", action="store_true",
                    help="with --grants: accept a cell the grants file records as a rival "
                         "FAILURE instead of re-deriving it. OFF by default — a recorded "
                         "failure is 'did not settle on THAT card' and honouring it can only "
                         "be unfair to the rival; the row records `not_remeasured`.")
    ap.add_argument("--grants", default=None, metavar="FILE",
                    help="a grants file (eval/rivals/extract_grants.py): pin each cell's batch "
                         "rung to the value a previous campaign measured — no ladder walk; a "
                         "rung this card cannot finish on is walked down from")
    ap.add_argument("--gpap-cache", default=None,
                    help="directory of cached GPAP arms (default: <output dir>/.gpap_cache). "
                         "GPAP_RIVAL_CACHE=0 disables reuse entirely.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[5, 6, 7],
                    help="GPAP sampling seeds; the row reports the median")
    ap.add_argument("--gpap-n", type=int, default=50_000)
    ap.add_argument("--portfolio", default=GPAP_PORTFOLIO,
                    help=f"GPAP sampling portfolio for our arm (default {GPAP_PORTFOLIO!r}); a "
                         f"single-config name pins the algorithm, as every rival's run is. See "
                         f"rivals_recurrent.GPAP_PORTFOLIO for the measurement behind the pin.")
    ap.add_argument("--target-seed", type=int, default=5, help="pattern-selection seed")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                    help="per-run wall cap for the RIVAL subprocess (s)")
    ap.add_argument("--gpap-time-limit", type=float, default=60.0,
                    help="per-seed sampling wall cap for the GPAP side (s)")
    ap.add_argument("--cmsgen", default=None,
                    help="patched cmsgen binary: run the CMSGen arm on the VERY CNF the GPAP arm "
                         "samples, same n / budget (the converged protocol's third tool; "
                         "rival_common.cmsgen_arm). Absent = no CMSGen columns on these rows")
    ap.add_argument("--cmsgen-seed", type=int, default=None, help="cmsgen -s seed")
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
    args.names = expand_names(args.names, "demotic")
    grants = load_grants(args.grants)          # {} = walk the ladder, as always
    if grants:
        print(f"[grants] {args.grants}: {len(grants.get('demotic') or {})} recorded DEMOTIC "
              f"cells (measured on {grants.get('gpu')}); pinned where recorded")
    if not os.path.isdir(os.path.join(args.demotic, "src")):
        sys.exit(f"no DEMOTIC checkout at {args.demotic} (see module docstring)")
    import torch
    from gpap.circuits.simulate import sim_baseline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bench_dir = args.bench_dir or (BENCH_COMB if (args.mode == "native" and
                                                 args.circuit_type == "comb") else BENCH_SEQ)
    steps, steps_src = resolve_steps("demotic", args.steps)
    lr, lr_src = resolve_lr("demotic", args.lr)
    tmp = paths.scratch_dir("rivals_demotic")
    mode_txt = ("their target table" if args.mode == "native"
                else f"OUR {args.target} fixed-T instance as a miter")
    if args.cmsgen and not os.access(args.cmsgen, os.X_OK):
        sys.exit(f"--cmsgen {args.cmsgen}: not an executable (scripts/build_cmsgen.sh)")
    print(f"{len(args.names)} instances | mode: {args.mode} ({mode_txt}) | "
          f"T={args.cycles if args.mode == 'miter' or args.circuit_type == 'seq' else 1} "
          f"steps={steps} ({steps_src}) | GPAP median of seeds {args.seeds} | CMSGen arm: "
          f"{'on (same CNF, --nowrite --maxtime)' if args.cmsgen else 'OFF'}")
    hdr = (f"{'instance':<10}{'circ':>6}{'pin':>14}{'p_acc':>10}{'DEM uniq':>10}{'t':>8}"
           f"{'DEM u/s':>10}{'GPAP u/s~':>11}{'GPAP/s':>11}{'succ%':>7}{'ratio':>10}"
           f"{'CMS/s':>10}{'G/C':>9}")
    print(hdr)
    print("-" * len(hdr))
    # Our arm is the same task whatever the rival is doing; see rival_common.cached_gpap.
    cache_dir = args.gpap_cache or (os.path.join(os.path.dirname(args.output) or ".",
                                                 ".gpap_cache") if args.output else None)
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
        """Write the JSON after every row: this stage runs for hours on a remote box."""
        if not args.output:
            return
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        json.dump({"mode": args.mode, "circuit_type": args.circuit_type, "cycles": args.cycles,
                   "target": args.target, "steps": steps, "steps_source": steps_src,
                   "circuit_variants": list(args.circuit_variants),
                   # HOST ram: `oversize_reason`'s second term reads it, so a refusal row needs it
                   "host_mem_avail_bytes": _available_bytes(),
                   "rival_commit": rival_commit(args.demotic), "rows": out},
                  open(args.output, "w"), indent=1)

    missing = []
    for name, variant in ((n, v) for n in args.names for v in args.circuit_variants):
        if (name, variant) in done:
            continue
        src0 = os.path.join(bench_dir, f"{name}.bench")
        if not os.path.isfile(src0):
            print(f"{name:<10}  no such bench file ({src0}), skipped")
            missing.append((name, f"no bench file in {bench_dir}"))
            continue
        # BOTH sides get the same circuit, gated by a bit-exact equivalence check, at the run's depth.
        T_chk = args.cycles if (args.mode == "miter" or args.circuit_type == "seq") else 1
        src, vinfo = prepare_variant(src0, variant, tmp, cycles=T_chk,
                                     seed=args.target_seed, target_mode=args.target)
        if vinfo.get("error"):
        # A ROW, not a silent skip: `prepare_variant` fails closed and a `continue` would drop it.
            print(f"{name:<10}{variant:>6}  circuit variant unusable: {vinfo['error'][:60]}",
                  flush=True)
            out.append({"name": name, "variant": variant, "mode": args.mode,
                        "cycles": args.cycles, "circuit": vinfo,
                        "demotic": {"error": f"circuit variant unusable: "
                                             f"{vinfo['error'][:160]}"},
                        "gpap": {"error": "not attempted: no usable circuit for this variant"}})
            missing.append((name, f"variant unusable: {vinfo['error'][:60]}"))
            continue
        if args.mode == "miter":
            # Their side: our unrolled instance as a single-output miter named c17, so their
            # table pins MITER=1; ours is the ORIGINAL circuit at the same T/target.
            rival_file = os.path.join(tmp, f"{name}_{variant}_miter.bench")
        # The self-check (p(MITER=1) == p_accept) must not RAISE: a misaligned miter is a flagged row.
            try:
                n_in, p_miter = write_unrolled_miter(src, rival_file, args.cycles,
                                                     args.target_seed, target_mode=args.target)
                pin_ok = True
            except AssertionError as e:
                n_in, p_miter, pin_ok = None, None, False
                print(f"{name:<10}{variant:>6}  MITER CHECK FAILED — {str(e)[:60]}; row flagged",
                      flush=True)
            key, circuit_type, T_rival = MITER_ALIAS, "comb", 1
            # OUR side must pin the SAME pattern the miter encodes, via write_unrolled_miter.
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
            pin_ok = native_pin_check(src, gpap_file, idx, vals, pin_n, gpap_T)
            if not pin_ok:
                print(f"{name:<10}{variant:>6}  PIN CHECK FAILED — reordered pins do not match "
                      f"their index table; row flagged", flush=True)
        # Their side only: wire names made valid Python identifiers (they exec generated source).
        rival_safe = os.path.join(tmp, f"{name}_{variant}_{args.mode}_safe.bench")
        renamed = sanitize_bench(rival_file, rival_safe)
        rival_file = rival_safe
        batch = args.batch or pick_batch(rival_file, T_rival)
        # a recorded grant pins the walk (rival_common.load_grants); `--batch` pins every cell
        grant = ({"batch": args.batch, "start_point": None, "source": "pinned by --batch"}
                 if args.batch else
                 lookup_grant(grants, "demotic", name, args.mode, T_rival, variant))
        pinned_fail = (lookup_recorded_failure(grants, "demotic", name, args.mode, T_rival,
                                               variant)
                       if args.honour_recorded_failures and not args.batch else None)
        if pinned_fail:
            # The reviewer's shortcut: the recorded outcome stands, unmeasured, and says so.
            # The lookup returns the ROW, so its shape has one definition, over in rival_common.
            dem = dict(pinned_fail)
        else:
            dem = run_demotic_ladder(args.demotic, rival_file, key, steps=steps, batch=batch,
                                     circuit_type=circuit_type, T=T_rival, timeout=args.timeout,
                                     lr=lr, grant=grant)
        dem["renamed_wires"] = renamed
        dem["steps_source"] = steps_src
        dem["lr"], dem["lr_source"] = lr, lr_src
        dem["seed"], dem["seed_source"] = RIVAL_SEED, "their flags.py default; one run"
        task = {"harness": "demotic", "mode": args.mode, "cycles": gpap_T, "pin_outputs": pin_n,
                "target": gpap_target, "seeds": args.seeds, "n_time": args.gpap_n,
                "time_limit": args.gpap_time_limit, "portfolio": args.portfolio,
                "encoder": GPAP_ENCODER, "unique_rows": UNIQUE_ROWS}
        # ONE CNF PER ROW, both of our arms on it; an encode failure is a GPAP-side error row.
        cnf_path = os.path.join(tmp, f"{name}_{variant}_{args.mode}.cnf")
        cnf_info, hit = None, False
        try:
            cnf_info = build_arm_cnf(gpap_file, gpap_T, pin_n, gpap_target, cnf_path)
            g, hit = cached_gpap(cache_dir, gpap_file, task,
                                 lambda: run_gpap(gpap_file, gpap_T, pin_n, gpap_target,
                                                  args.seeds, args.gpap_n, device,
                                                  time_limit=args.gpap_time_limit,
                                                  portfolio_name=args.portfolio,
                                                  cnf=(cnf_path, cnf_info)))
        except Exception as e:                              # noqa: BLE001 — one cell, not the run
            what = "encode failed" if cnf_info is None else "GPAP arm failed"
            print(f"{name:<10}{variant:>6}  {what}: {type(e).__name__}: {str(e)[:80]}",
                  flush=True)
            g = {"error": f"{what}: {type(e).__name__}: {str(e)[:200]}"}
        if hit:
            print(f"{name:<10}{variant:>6}  [gpap-cache hit — arm not re-run]", flush=True)
        cms = (cmsgen_arm(cnf_path, args.cmsgen, args.gpap_n, args.gpap_time_limit, tmp,
                          seed=args.cmsgen_seed) if (args.cmsgen and cnf_info) else {})
        if os.path.exists(cnf_path):
            os.remove(cnf_path)
        row = {"name": name, "mode": args.mode, "variant": variant, "circuit": vinfo,
               "pin": pin_desc, "cycles": gpap_T,
               "pin_check_ok": pin_ok,
               "p_accept": sim["p_accept"], "p_accept_lt": sim.get("p_accept_lt"),
               "demotic": dem, "gpap": g, "batch": batch,
               "cnf": cnf_info, "support": (cnf_info or {}).get("support"),
               **cms, **host_provenance(device),
               # what parallelism the rival was GRANTED, beside its score: the VRAM budget and rung
               "vram_budget_bytes": rival_vram_budget()}
        if args.mode == "miter":
               # the miter's acceptance probability — evidence both samplers got the same task
            row["p_miter"] = p_miter
            row["miter_inputs"] = n_in
        out.append(row)
        checkpoint()
        pa = sim["p_accept"] or sim.get("p_accept_lt", 0.0)
        pa_s = f"{pa:.2e}" if sim["p_accept"] else f"<{pa:.0e}"
        pin_desc = pin_desc if len(pin_desc) <= 14 else pin_desc[:11] + "..."   # keep the columns
        # `unique_per_s_est` is None when the timed run kept no output row to measure on, not
        # when GPAP produced nothing (`n_models` says that), so it prints "n/a" and takes no part
        # in a ratio rather than being coerced to a zero that reads as a loss.
        uq = g.get("unique_per_s_est")
        uq_s = f"{uq:,.0f}" if uq is not None else "n/a"
        if "error" in dem:
            succ_s = (f"{100 * g['success']:.0f}%" if g.get("success") is not None
                      else "arm not run")
            print(f"{name:<10}{variant:>6}{pin_desc:>14}{pa_s:>10}  DEMOTIC FAILED: "
                  f"{dem['error'][:38]} | GPAP u/s~={uq_s} succ={succ_s} |{_cms_cols(g, cms)}",
                  flush=True)
            continue
        if "error" in g:
            print(f"{name:<10}{variant:>6}{pin_desc:>14}{pa_s:>10}{dem['unique']:>10,}"
                  f"  GPAP FAILED: {g['error'][:50]} |{_cms_cols(g, cms)}", flush=True)
            continue
        ratio = (None if uq is None else
                 (uq / dem["unique_per_s"] if dem["unique_per_s"] else float("inf")))
        rt = ("n/a" if ratio is None else
              (f"{ratio:,.1f}x" if ratio != float("inf") else "GPAP"))
        print(f"{name:<10}{variant:>6}{pin_desc:>14}{pa_s:>10}{dem['unique']:>10,}"
              f"{dem['runtime_s']:>7.1f}s{dem['unique_per_s']:>10,.0f}"
              f"{uq_s:>11}{g['sps']:>11,.0f}"
              f"{100 * g['success']:>6.0f}%{rt:>10}{_cms_cols(g, cms)}", flush=True)
    if len(args.circuit_variants) > 1:
        print("\ncircuit variants (both sides get the SAME circuit; abc = sequential-AIG "
              "optimization, equivalence-checked):")
        for r in out:
            print(f"  {r['name']:<8} {variant_note(r['circuit'])}")
    if args.output:
        checkpoint()
        print(f"wrote {args.output}")
    shutil.rmtree(tmp, ignore_errors=True)
    # AN ARM THAT MEASURED NOTHING MUST SAY SO LOUDLY: a JSON holding `"rows": []` looks exactly
    # like a stage that never started, so a non-zero exit is what a driver can see.
    if missing and not out:
        sys.exit(f"[FATAL] every instance was skipped ({len(missing)}) — the arm measured "
                 f"nothing:\n    "
                 + "\n    ".join(f"{n:<12} {why}" for n, why in missing[:8])
                 + ("\n    ..." if len(missing) > 8 else "")
                 + "\n  (a missing bench file usually means --bench-dir points elsewhere)")
    if missing:
        print(f"[warn] {len(missing)} instance(s) did not produce a measured row:")
        for n, why in missing:
            print(f"    {n:<12} {why}")


if __name__ == "__main__":
    main()
