#!/usr/bin/env python3
"""WAVE vs SCALAR propagate-kernel A/B across the benchmark families — GPAP only.

The engine picks the propagate kernel per formula (`cpp/src/calibrate.cpp::gpap_dispatch_wave`:
WAVE, one warp per row, when M >= 2048; SCALAR, the serial trail, below). This measures what
that choice is worth per instance and binned by clause count, with the kernel FORCED on each
arm via `$GPAP_KERNEL` (read by the core binding at engine construction, so it can be toggled
between `sample_file` calls in one process).

METHOD — every rule here was a wrong number once; do not "simplify":
  * each CNF is encoded ONCE and both arms run in the SAME process, order ALTERNATED per rep
    (SM clocks drift within a run; an un-alternated before/after cannot resolve a few percent);
  * the config is PINNED (`--portfolio b8`, no bandit): a free bandit picks differently per arm
    and the A/B would be measuring the bandit, not the kernel;
  * MINING IS OFF: mining is wall-budgeted, so a faster kernel mines MORE clauses and the two
    arms would run DIFFERENT formulas (M, K and everything downstream would move);
  * VALIDATION before timing: both arms at a pinned K (`--val-k-cap`) for a fixed number of
    chunks (`--val-chunks`) must pass the soundness re-check and produce IDENTICAL model counts — BCP is
    confluent, so at equal K/seed/chunks the kernels agree on every model (the repo's no-change
    gate). Every arm also records the sampler's own `engine.mode` provenance, the proof that the
    forced kernel is what ran;
  * TIMING: a fixed wall per arm (`--time-limit`, the cross-GPU protocol's 20 s so several chunks
    run), `--reps` repetitions with the order alternated, best-of per arm (drift makes any
    single run a lower bound); ratio = wave/scalar of the bests; K is recorded per arm and
    flagged if the arms differ.
  * the dispatch rule's own pick (`dispatch_mode(M, K)`) is recorded per row, so the result
    doubles as a check of the rule on this GPU.

Resumable (`--resume`): rows are keyed by (family, instance, encoder); the JSON is written after
every CNF. Vacuous-pin and zero-search instances are measured but flagged and excluded from the
headline geomean, exactly as the paper tables do.

    PYTHONPATH=. python eval/kernel_ab.py --out .experiments/kernel_ab/kernel_ab.json --resume
"""
import argparse
import glob
import json
import math
import os
import sys
import tempfile
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from gpap import paths                                        # noqa: E402

VACUOUS = {"b02", "b05", "b07", "s526"}          # constant last-cycle outputs at T=25
KERNELS = ("scalar", "wave")
M_BINS = ((0, 2048, "M<2k"), (2048, 20_000, "2k-20k"), (20_000, 100_000, "20k-100k"),
          (100_000, 10**12, ">100k"))

# family -> (glob, kind, cycles-key, pin_outputs)   cycles-key: "T" = --cycles, 1 = fixed
FAMILIES = {
    "seq":   ("benchmarks/circuits/bench/seq/*.bench",  "bench", "T", None),
    "comb":  ("benchmarks/circuits/bench/comb/*.bench", "bench", 1,   1),
    "epfl":  ("benchmarks/circuits/epfl/*.aig",         "aig",   1,   None),
    "hwmcc": ("benchmarks/circuits/hwmcc17/*.aig",      "aig",   "T", None),
    "sk":    ("benchmarks/sampling/*.cnf",              "cnf",   None, None),
}


def log(msg):
    print(msg, flush=True)


def parse_encoder(spec):
    enc, _, opt = spec.partition(":")
    return enc, (opt or "fast")


def convert(kind, src, out, cycles, seed, pin_outputs, encoder, abc_opt, abc_path, target):
    if kind == "bench":
        from gpap.circuits.bench import convert_one
        convert_one(src, out, cycles, seed, pin_outputs=pin_outputs, encoder=encoder,
                    abc_path=abc_path, quiet=True, abc_opt=abc_opt, target_mode=target)
    else:
        from gpap.circuits.aig2cnf import convert_aig_one
        convert_aig_one(src, out, cycles, seed, pin_outputs=pin_outputs, encoder=encoder,
                        abc_path=abc_path, quiet=True, abc_opt=abc_opt, target_mode=target)


def run_arm(kernel, cnf, device, configs, *, n_time, time_limit, k_cap, max_chunks, seed,
            verify_sound=False):
    """One sampling run with the kernel FORCED. Returns the fields the A/B needs, or an error."""
    import torch
    from gpap.sampling.pipeline import sample_file
    os.environ["GPAP_KERNEL"] = kernel
    try:
        g = sample_file(cnf, device, n_time=n_time, n_cov=0, configs=configs,
                        time_limit=time_limit, seed=seed, mine="off",
                        k_cap=k_cap, max_chunks=max_chunks, verify_sound=verify_sound)
    except Exception as e:                      # noqa: BLE001 — envelope rejection, OOM, ...
        return {"error": f"{type(e).__name__}: {e}"[:300]}
    finally:
        os.environ.pop("GPAP_KERNEL", None)
        torch.cuda.empty_cache()
    eng = g.get("engine") or {}
    return {"sps": g["samples_per_s"], "e2e": g.get("sps_e2e_total"),
            "succ": g["success_rate"], "models": g["n_models"],
            "wall_timed_s": g.get("wall_timed_s"), "chunk_capped": g.get("chunk_capped"),
            "dec_s": g.get("decisions_per_s"), "impl_s": g.get("implications_per_s"),
            "sound": g.get("sound_check"), "mode": eng.get("mode"), "K": eng.get("K"),
            "batch": eng.get("batch"), "split": eng.get("split"),
            "M": g.get("n_clauses"), "nv": g.get("nv"), "support": len(g.get("sup1") or [])}


def measure(cnf, device, configs, args):
    """Validation (pinned K, fixed chunks, identical models) then timing (alternated reps)."""
    row = {}
    # ---- validation: same K, same seed, same chunk count -> the kernels must agree exactly
    val = {}
    for k in KERNELS:
        # verify_sound ON for the VALIDATION arms only: this is the pass whose job is to
        # prove both kernels emit real models, and it is untimed. The timing arms below leave it
        # off so nothing is charged to a measured wall.
        val[k] = run_arm(k, cnf, device, configs, n_time=10**9, time_limit=args.val_time_limit,
                         k_cap=args.val_k_cap, max_chunks=args.val_chunks, seed=args.seed,
                         verify_sound=True)
    ok = all("error" not in val[k] for k in KERNELS)
    row["validation"] = {
        "K": {k: val[k].get("K") for k in KERNELS},
        "models": {k: val[k].get("models") for k in KERNELS},
        "mode_recorded": {k: val[k].get("mode") for k in KERNELS},
        "sound": {k: val[k].get("sound") for k in KERNELS},
        "errors": {k: val[k]["error"] for k in KERNELS if "error" in val[k]},
    }
    if ok:
        same_k = val["scalar"]["K"] == val["wave"]["K"]
        row["validation"]["same_K"] = same_k
        row["validation"]["identical_models"] = (same_k and
                                                val["scalar"]["models"] == val["wave"]["models"])
        row["validation"]["mode_ok"] = all(val[k]["mode"] == k for k in KERNELS)
        # `sound` is the TRI-STATE string, so it must be compared to "ok" — truthiness passes
        # for "off" AND for "UNSOUND", which silently disarmed this gate when the key changed
        # from the old `gpu_sound` bool.
        row["validation"]["sound_ok"] = all(val[k]["sound"] == "ok" for k in KERNELS)
        row["validation"]["passed"] = bool(row["validation"]["identical_models"]
                                           and row["validation"]["mode_ok"]
                                           and row["validation"]["sound_ok"])
    else:
        row["validation"]["passed"] = False
    # ---- timing: fixed wall, reps alternated, best-of per arm
    reps = {k: [] for k in KERNELS}
    for rep in range(args.reps):
        order = KERNELS if rep % 2 == 0 else KERNELS[::-1]
        for k in order:
            reps[k].append(run_arm(k, cnf, device, configs, n_time=args.n_time,
                                   time_limit=args.time_limit, k_cap=args.k_cap,
                                   max_chunks=None, seed=args.seed))
    for k in KERNELS:
        good = [r for r in reps[k] if "error" not in r]
        best = max(good, key=lambda r: r["sps"]) if good else {"error": reps[k][0].get("error")}
        best = dict(best)
        best["reps_sps"] = [r.get("sps") for r in reps[k]]
        row[k] = best
    s, w = row["scalar"], row["wave"]
    if "sps" in s and "sps" in w:
        row["M"], row["nv"], row["support"] = s["M"], s["nv"], s["support"]
        row["split"] = s["split"]
        row["same_K_timing"] = s["K"] == w["K"]
        row["mode_ok_timing"] = s["mode"] == "scalar" and w["mode"] == "wave"
        row["zero_search"] = not s["dec_s"] and not s["impl_s"] and not w["dec_s"] and not w["impl_s"]
        if s["sps"] and w["sps"]:
            row["ratio"] = w["sps"] / s["sps"]
        elif w["sps"]:
            row["ratio"] = float("inf")             # scalar produced nothing in the budget
        elif s["sps"]:
            row["ratio"] = 0.0
        else:
            row["ratio"] = None                     # neither did
        from gpap.engine.calibrate import dispatch_mode
        row["auto_pick"] = dispatch_mode(int(s["M"]), int(s["K"] or 1), device)
    else:
        row["ratio"] = None
        row["error"] = (s.get("error") or w.get("error"))
    return row


def m_bin(m):
    for lo, hi, name in M_BINS:
        if lo <= m < hi:
            return name
    return "?"


def summarize(rows):
    head = [r for r in rows if r.get("ratio") not in (None, 0.0, float("inf"))
            and not r.get("vacuous") and not r.get("zero_search")]
    val_pass = sum(1 for r in rows if (r.get("validation") or {}).get("passed"))
    val_tot = sum(1 for r in rows if r.get("validation"))
    log(f"\n==== WAVE / SCALAR  ({len(rows)} rows; validation passed {val_pass}/{val_tot})")
    if not head:
        log("  no comparable rows")
        return
    gm = math.exp(sum(math.log(r["ratio"]) for r in head) / len(head))
    srt = sorted(r["ratio"] for r in head)
    med = srt[len(srt) // 2]
    log(f"  headline (non-vacuous, non-zero-search, both>0): n={len(head)} "
        f"geomean {gm:.3f}x  median {med:.3f}x  min {srt[0]:.3f}  max {srt[-1]:.3f}  "
        f"wave wins {sum(1 for r in head if r['ratio'] > 1)}/{len(head)}")
    for lo, hi, name in M_BINS:
        b = [r for r in head if lo <= r["M"] < hi]
        if b:
            g = math.exp(sum(math.log(r["ratio"]) for r in b) / len(b))
            log(f"  {name:>9}: n={len(b):3d} geomean {g:.3f}x  wave wins {sum(1 for r in b if r['ratio'] > 1)}/{len(b)}")
    agree = sum(1 for r in head if (r["ratio"] > 1) == (r["auto_pick"] == "wave"))
    log(f"  dispatch rule picked the faster kernel on {agree}/{len(head)}")
    inf = [r for r in rows if r.get("ratio") == float("inf")]
    zero = [r for r in rows if r.get("ratio") == 0.0]
    neither = [r for r in rows if r.get("ratio") is None and "error" not in r]
    if inf or zero or neither:
        log(f"  scalar-only-zero {len(inf)} | wave-only-zero {len(zero)} | neither {len(neither)}")
    bad = [r for r in rows if r.get("validation") and not r["validation"].get("passed")]
    for r in bad:
        log(f"  VALIDATION FAILED {r['family']}/{r['instance']}/{r['encoder']}: {r['validation']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    for fam, (g, *_rest) in FAMILIES.items():
        ap.add_argument(f"--{fam}-glob", default=g)
    ap.add_argument("--cycles", type=int, default=25)
    ap.add_argument("--encoders", nargs="+", default=["tseitin", "abc:fast"])
    ap.add_argument("--target", default="rare", choices=["rare", "random"])
    ap.add_argument("--abc", default=str(paths.tool("abc")))
    ap.add_argument("--portfolio", default="b8", help="PINNED single config (b8 | b1 | s1)")
    ap.add_argument("--n-time", type=int, default=2_000_000)
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--k-cap", default="auto")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--val-k-cap", type=int, default=4096)
    ap.add_argument("--val-chunks", type=int, default=3)
    ap.add_argument("--val-time-limit", type=float, default=120.0)
    ap.add_argument("--max-clauses", type=int, default=4_000_000)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, metavar="FAMILY/INSTANCE/ENCODER",
                    help="measure only these cells (a re-measurement pass; merge by key)")
    args = ap.parse_args(argv)
    if args.k_cap != "auto":
        args.k_cap = int(args.k_cap)

    import torch
    from gpap.sampling.pipeline import portfolio
    device = torch.device("cuda")
    configs = portfolio(args.portfolio)
    gpu = {"name": torch.cuda.get_device_properties(0).name, "torch": torch.__version__,
           "cuda": torch.version.cuda}

    rows, done = [], set()
    if args.resume and os.path.isfile(args.out):
        try:
            rows = json.load(open(args.out)).get("rows", [])
            done = {(r["family"], r["instance"], r["encoder"]) for r in rows}
            log(f"[resume] {len(rows)} rows already in {args.out}")
        except Exception as e:                  # noqa: BLE001
            log(f"[resume] ignoring unreadable {args.out}: {e}")
            rows, done = [], set()

    def save():
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"schema": 1, "args": vars(args), "gpu": gpu, "rows": rows}, f, indent=1)

    # ---- the work list
    work = []
    for fam in args.families:
        g, kind, cyc, pin = FAMILIES[fam]
        g = getattr(args, f"{fam}_glob")
        files = sorted(glob.glob(g if os.path.isabs(g) else os.path.join(REPO_ROOT, g)))
        cycles = args.cycles if cyc == "T" else cyc
        for src in files:
            inst = os.path.splitext(os.path.basename(src))[0]
            for enc in (["cnf"] if kind == "cnf" else args.encoders):
                work.append((fam, inst, enc, src, kind, cycles, pin))
    log(f"{len(work)} (instance, encoder) cells | portfolio {args.portfolio} (pinned) | "
        f"mining OFF | validation K<={args.val_k_cap} x {args.val_chunks} chunks | "
        f"timing {args.time_limit}s x {args.reps} reps alternated | {gpu['name']}")

    if args.only:
        keep = {tuple(k.split("/")) for k in args.only}
        work = [w for w in work if (w[0], w[1], w[2]) in keep]
        # MERGE BY KEY, as the help says: a re-measurement replaces the old row. Without this,
        # --resume skipped the named cells (already "done") and a run without --resume threw
        # away every OTHER row — either way the pass could not do what it exists for.
        rows = [r for r in rows if (r["family"], r["instance"], r["encoder"]) not in keep]
        done -= keep
    for i, (fam, inst, enc, src, kind, cycles, pin) in enumerate(work):
        if (fam, inst, enc) in done:
            continue
        row = {"family": fam, "instance": inst, "encoder": enc, "cycles": cycles,
               "vacuous": inst in VACUOUS, "target": args.target}
        tmp = None
        try:
            if kind == "cnf":
                cnf = src
            else:
                e, opt = parse_encoder(enc)
                tmp = tempfile.NamedTemporaryFile(suffix=".cnf", delete=False).name
                t0 = time.perf_counter()
                convert(kind, src, tmp, cycles, args.seed, pin, e, opt, args.abc, args.target)
                row["encode_s"] = round(time.perf_counter() - t0, 1)
                cnf = tmp
            # size guard, like the sweep's --max-clauses (a 50-cycle giant parses for minutes)
            with open(cnf) as fh:
                for line in fh:
                    if line.startswith("p cnf"):
                        m_hdr = int(line.split()[3])
                        break
                else:
                    m_hdr = 0
            if m_hdr > args.max_clauses:
                row["skipped"] = f"{m_hdr:,} clauses > --max-clauses"
            else:
                row.update(measure(cnf, device, configs, args))
        except Exception as e:                  # noqa: BLE001 — one cell must not kill the sweep
            row["error"] = f"{type(e).__name__}: {e}"[:300]
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)
        rows.append(row)
        save()
        r = row.get("ratio")
        rs = "n/a" if r is None else ("inf" if r == float("inf") else f"{r:.3f}x")
        v = (row.get("validation") or {}).get("passed")
        log(f"[{i+1}/{len(work)}] {fam:<5} {inst:<22} {enc:<8} M={row.get('M') or '-':>9} "
            f"K={(row.get('scalar') or {}).get('K') or '-':>7} "
            f"scalar {((row.get('scalar') or {}).get('sps') or 0):>12,.0f}/s  "
            f"wave {((row.get('wave') or {}).get('sps') or 0):>12,.0f}/s  "
            f"wave/scalar {rs:>8}  auto={row.get('auto_pick', '-'):<6} "
            f"val={'PASS' if v else ('FAIL' if v is False else '-')}"
            f"{'  ' + row['error'] if row.get('error') else ''}"
            f"{'  ' + row['skipped'] if row.get('skipped') else ''}")
    summarize(rows)
    save()
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
