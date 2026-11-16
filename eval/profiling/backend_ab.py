#!/usr/bin/env python3
"""CUDA vs OpenCL on probing work, across the benchmark families: the same tuples on every arm.

The OpenCL backend is a second implementation of the fixpoint kernels (docs/OPENCL.md). This
measures it against the CUDA backend on the work entailed-clause mining does — probing tuples of
`--width` literals (1 to 3) to their BCP fixpoint — kernel by kernel, on ONE GPU:

    arm                  backend  kernel  CLEAN
    cuda-wave-clean      cuda     wave    on      (what CUDA runs on a large formula)
    cuda-wave            cuda     wave    off
    cuda-scalar-clean    cuda     scalar  on
    cuda-scalar          cuda     scalar  off
    opencl-wave-clean    opencl   wave    on      (what OpenCL runs on a large formula)
    opencl-wave          opencl   wave    off
    opencl-scalar-clean  opencl   scalar  on
    opencl-scalar        opencl   scalar  off

Two batch rules (`--batch`): a number pins ONE batch for every arm (bounded by every arm's core
model), so the arms do the same launches; 0 gives each arm the batch its core sizes for a fresh
engine, which is what a user gets (the batch is pinned per arm from that engine's model, so a
second sweep on one engine cannot shrink it: GitHub issue 56).

Method, following docs/BENCHMARKING.md:
  * count-terminated, never wall-terminated: every arm probes the SAME matrix of random literal
    tuples, so the work is identical and only the time differs;
  * one process, arm order alternated per repetition, because SM clocks drift within a run; the
    best repetition is kept per arm (drift makes any single run a lower bound);
  * an engine per (arm, repetition), created and warmed outside the timed sweep; one engine alive
    at a time, so the batch is not bounded by another arm's memory;
  * the check before any rate: every arm reports the same conflicting tuples, the scalar arms
    report the same work counters, and each arm ran the kernel and the CLEAN mode it names. A
    row that fails it is flagged and left out of the summary;
  * the rate is implications per second over the sweep's own wall (`gpap_probe_tuples` times
    itself); the phase split (reset, apply) is timed separately, with a readback as the sync.
    The WAVE arms count a conflicted row's implications in their own order, so the summary also
    gives every ratio in tuples per second, which is the same work for every arm.

Resumable (`--resume`, with the same settings only): rows are keyed by family/instance[@cycles]; the
JSON is rewritten after each.

    PYTHONPATH=. python eval/profiling/backend_ab.py --out .experiments/backend_ab/backend_ab.json --resume
"""
import argparse
import contextlib
import glob
import hashlib
import json
import math
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from gpap import paths                                        # noqa: E402

#: arm -> (backend, kernel, GPAP_CLEAN)
ARMS = {
    "cuda-wave-clean": ("cuda", "wave", "1"),
    "cuda-wave": ("cuda", "wave", "0"),
    "cuda-scalar-clean": ("cuda", "scalar", "1"),
    "cuda-scalar": ("cuda", "scalar", "0"),
    "opencl-wave-clean": ("opencl", "wave", "1"),
    "opencl-wave": ("opencl", "wave", "0"),
    "opencl-scalar-clean": ("opencl", "scalar", "1"),
    "opencl-scalar": ("opencl", "scalar", "0"),
}
M_BINS = ((0, 2048, "M<2k"), (2048, 20_000, "2k-20k"), (20_000, 100_000, "20k-100k"),
          (100_000, 10**12, ">100k"))
#: family -> (glob, kind, cycles: "T" = --cycles or a fixed count, pin_outputs)
FAMILIES = {
    "seq":   ("benchmarks/circuits/bench/seq/*.bench",  "bench", "T", None),
    "comb":  ("benchmarks/circuits/bench/comb/*.bench", "bench", 1,   1),
    "epfl":  ("benchmarks/circuits/epfl/*.aig",         "aig",   1,   None),
    "hwmcc": ("benchmarks/circuits/hwmcc17/*.aig",      "aig",   "T", None),
    "sk":    ("benchmarks/sampling/*.cnf",              "cnf",   None, None),
}
#: The comparisons the summary reports: name -> (numerator arm, denominator arm).
RATIOS = {
    "opencl-scalar / cuda-scalar": ("opencl-scalar", "cuda-scalar"),
    "opencl-scalar-clean / cuda-scalar-clean": ("opencl-scalar-clean", "cuda-scalar-clean"),
    "opencl-wave / cuda-wave": ("opencl-wave", "cuda-wave"),
    "opencl-wave-clean / cuda-wave-clean": ("opencl-wave-clean", "cuda-wave-clean"),
    "cuda-wave / cuda-scalar": ("cuda-wave", "cuda-scalar"),
    "cuda-scalar-clean / cuda-scalar": ("cuda-scalar-clean", "cuda-scalar"),
    "cuda-wave-clean / cuda-wave": ("cuda-wave-clean", "cuda-wave"),
    "opencl-wave / opencl-scalar": ("opencl-wave", "opencl-scalar"),
    "opencl-scalar-clean / opencl-scalar": ("opencl-scalar-clean", "opencl-scalar"),
    "opencl-wave-clean / opencl-wave": ("opencl-wave-clean", "opencl-wave"),
    "opencl-wave-clean / opencl-scalar-clean": ("opencl-wave-clean", "opencl-scalar-clean"),
    "opencl-scalar / cuda-wave-clean": ("opencl-scalar", "cuda-wave-clean"),
    "opencl-scalar-clean / cuda-wave-clean": ("opencl-scalar-clean", "cuda-wave-clean"),
}


def log(msg):
    print(msg, flush=True)


def convert(kind, src, out, cycles, seed, pin_outputs, target):
    if kind == "bench":
        from gpap.circuits.bench import convert_one
        convert_one(src, out, cycles, seed, pin_outputs=pin_outputs, encoder="tseitin",
                    quiet=True, target_mode=target)
    else:
        from gpap.circuits.aig2cnf import convert_aig_one
        convert_aig_one(src, out, cycles, seed, pin_outputs=pin_outputs, encoder="tseitin",
                        quiet=True, target_mode=target)


@contextlib.contextmanager
def environment(**values):
    """Set these variables for the block. The core reads them when an engine is CREATED."""
    saved = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def arm_spec(arm):
    """(backend, kernel, clean) of an arm name."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}: one of {', '.join(ARMS)}")
    return ARMS[arm]


def make_engine(arm, formula):
    from gpap.engine import _core
    backend, kernel, clean = arm_spec(arm)
    with environment(GPAP_BACKEND=backend, GPAP_CLEAN=clean, GPAP_KERNEL=None):
        return _core.Engine(formula, mode=kernel)


def own_batch(arm, formula, n_tuples, width):
    """The batch this arm's core sizes for a sweep of `n_tuples`, on a fresh engine: the probe
    sweep's model (check_chunk_size is the same arithmetic), with the device's single-buffer
    limit. One engine at a time, so no arm's budget sees another's buffers."""
    eng = make_engine(arm, formula)
    try:
        return int(eng.check_chunk_size(n_tuples, width))
    finally:
        del eng


def random_tuples(free, n, width, seed):
    """`n` tuples of `width` free literals on distinct variables, DIMACS, from a fixed seed."""
    import numpy as np
    rng = np.random.default_rng(seed)
    free = np.asarray(free, dtype=np.int32)
    cols = [rng.integers(0, len(free), size=n) for _ in range(width)]
    for j in range(1, width):                           # redraw a literal that clashes
        while True:
            clash = np.zeros(n, dtype=bool)
            for i in range(j):
                clash |= np.abs(free[cols[j]]) == np.abs(free[cols[i]])
            if not clash.any():
                break
            cols[j][clash] = rng.integers(0, len(free), size=int(clash.sum()))
    return np.stack([free[c] for c in cols], axis=1).astype(np.int32)


def sweep(eng, tuples, K):
    """One timed sweep: (seconds, implications, occurrence entries, conflict count, digest)."""
    c0 = eng.counters()
    before = (c0.props - c0.decs, c0.ops)
    with eng.probe_tuples(tuples, batch=K) as s:
        seconds, probed = float(s.stats.seconds), int(s.stats.probed)
        conf = s.conflicts()
    c1 = eng.counters()
    assert probed == len(tuples), (probed, len(tuples))
    digest = hashlib.sha256(conf[np_lexsort(conf)].tobytes()).hexdigest()[:16]
    return (seconds, int(c1.props - c1.decs - before[0]), int(c1.ops - before[1]), len(conf),
            digest)


def np_lexsort(conf):
    import numpy as np
    if not len(conf):
        return np.zeros(0, dtype=np.int64)
    return np.lexsort(tuple(conf[:, j] for j in reversed(range(conf.shape[1]))))


def phases(eng, arm, decisions, K, n=3):
    """Seconds per chunk of the two phases, each closed by a flag readback: reset(K), and the
    fixpoint launch. The first chunk is dropped (a CLEAN arm's first reset is a full one)."""
    apply = eng.apply_clean if arm_spec(arm)[2] == "1" else eng.apply
    t_reset, t_apply = [], []
    for _ in range(n + 1):
        t0 = time.perf_counter()
        eng.reset(K)
        eng.flags()
        t1 = time.perf_counter()
        apply(decisions)
        eng.flags()
        t2 = time.perf_counter()
        t_reset.append(t1 - t0)
        t_apply.append(t2 - t1)
    return sum(t_reset[1:]) / n, sum(t_apply[1:]) / n


def measure(cnf, arms, args):
    import numpy as np
    from gpap.engine import _core

    f = _core.Formula.from_file(cnf)
    info = f.info
    row = {"M": int(info.num_clauses), "num_long": int(info.num_long), "nv": int(info.num_vars)}
    if info.has_empty_clause:
        return dict(row, skip="the formula has an empty clause")
    probe = make_engine("cuda-scalar", f)        # the free literals; not a measured arm
    if probe.flush_conflict:
        return dict(row, skip="the formula's own units conflict")
    free = probe.free_literals()
    del probe
    if len(free) < 8:
        return dict(row, skip=f"only {len(free)} free literals")
    if args.batch > 0:                                  # one batch, the arms' common bound
        K = min(own_batch(a, f, args.batch, args.width) for a in arms)
        Ks = dict.fromkeys(arms, K)
    else:
        Ks = {a: own_batch(a, f, 1 << 30, args.width) for a in arms}
        K = Ks[arms[0]]
    row.update(K=K, free_literals=int(len(free)),
               auto_pick="wave" if _core.dispatch_wave(row["M"], K) else "scalar")
    if args.batch <= 0:
        row["K_by_arm"] = Ks

    # Calibrate the sweep length once, on the first arm: enough chunks for --target-s seconds.
    tuples = random_tuples(free, args.max_chunks * K, args.width, args.seed)
    eng = make_engine(arms[0], f)
    sweep(eng, tuples[:K], K)                                           # warm-up
    t = sweep(eng, tuples[:2 * K], K)[0] / 2
    del eng
    chunks = int(min(args.max_chunks, max(args.min_chunks, round(args.target_s / max(t, 1e-6)))))
    tuples = tuples[:chunks * K]
    first = tuples[:K].astype(np.int64)                 # DIMACS -> the engine's literal index
    decisions = 2 * (np.abs(first) - 1) + (first < 0)
    row.update(chunks=chunks, tuples=int(len(tuples)))

    runs = {a: [] for a in arms}
    for rep in range(args.reps):
        for arm in (arms if rep % 2 == 0 else arms[::-1]):
            try:
                eng = make_engine(arm, f)
                k = Ks[arm]
                sweep(eng, tuples[:k], k)                               # warm-up, untimed
                lite0 = eng.reset_stats()[1]
                sec, impl, ops, n_conf, digest = sweep(eng, tuples, k)
                lite = eng.reset_stats()[1] - lite0
                r = {"seconds": sec, "implications": impl, "ops": ops, "conflicts": n_conf,
                     "digest": digest, "mode": eng.mode, "resets_lite": int(lite)}
                if rep == 0 and args.batch > 0:                 # the phases at the one batch
                    r["reset_s"], r["apply_s"] = phases(eng, arm, decisions, K)
                    r["device"] = eng.device_name
                del eng
            except Exception as e:                      # noqa: BLE001 — OOM, a refused kernel
                r = {"error": f"{type(e).__name__}: {e}"[:300]}
            runs[arm].append(r)

    row["arms"] = {}
    for arm in arms:
        good = [r for r in runs[arm] if "error" not in r]
        if not good:
            row["arms"][arm] = {"error": runs[arm][0]["error"]}
            continue
        best = min(good, key=lambda r: r["seconds"])
        row["arms"][arm] = {
            "seconds": best["seconds"], "reps_seconds": [r.get("seconds") for r in runs[arm]],
            "implications": best["implications"], "ops": best["ops"],
            "conflicts": best["conflicts"], "digest": best["digest"], "mode": best["mode"],
            "resets_lite": best["resets_lite"],
            "impl_per_s": best["implications"] / best["seconds"] if best["seconds"] else None,
            "tuples_per_s": len(tuples) / best["seconds"] if best["seconds"] else None,
            "reset_s": good[0].get("reset_s"), "apply_s": good[0].get("apply_s"),
            "device": good[0].get("device"),
            # Every repetition of an arm finds the same conflicts and does the same work, or
            # the arm is not deterministic. The OpenCL WAVE kernel is exempt from the second:
            # its lanes race for the trail's order, so a conflicted row's count varies.
            "stable": len({r["digest"] for r in good}) == 1 and (
                arm_spec(arm)[:2] == ("opencl", "wave")
                or len({r["implications"] for r in good}) == 1)}
    ok = [a for a in arms if "error" not in row["arms"][a]]
    scalar = [a for a in ok if arm_spec(a)[1] == "scalar"]
    row["checks"] = {
        "same_conflicts": len({row["arms"][a]["digest"] for a in ok}) == 1,
        "same_work_scalar": len({(row["arms"][a]["implications"], row["arms"][a]["ops"])
                                 for a in scalar}) <= 1,
        "modes_as_asked": all(row["arms"][a]["mode"] == arm_spec(a)[1] for a in ok),
        # A CLEAN arm's resets skip the broadcast; a plain arm's never do.
        "clean_as_asked": all((row["arms"][a]["resets_lite"] > 0) == (arm_spec(a)[2] == "1")
                              for a in ok),
        "stable": all(row["arms"][a]["stable"] for a in ok),
        "all_arms_ran": len(ok) == len(arms)}
    row["valid"] = all(row["checks"].values())
    return row


def m_bin(m):
    for lo, hi, name in M_BINS:
        if lo <= m < hi:
            return name
    return "?"


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def summarize(rows, arms):
    """Geometric-mean rate ratios over the valid rows, overall and binned by clause count."""
    valid = [r for r in rows.values() if r.get("valid")]
    out = {"rows": len(rows), "valid": len(valid),
           "skipped": sum(1 for r in rows.values() if "skip" in r),
           "invalid": sorted(k for k, r in rows.items() if "arms" in r and not r.get("valid"))}
    ratios = dict(RATIOS)

    def ratio(r, num, den, rate):
        a, b = r["arms"].get(num, {}), r["arms"].get(den, {})
        return a[rate] / b[rate] if a.get(rate) and b.get(rate) else None

    # "ratios" is in implications per second; "ratios_tuples" in tuples per second, which is
    # the same work for every arm (a WAVE arm counts a conflicted row's implications its way).
    for field, rate in (("ratios", "impl_per_s"), ("ratios_tuples", "tuples_per_s")):
        out[field] = {}
        for name, (num, den) in ratios.items():
            if num not in arms or den not in arms:
                continue
            cell = {"all": geomean([ratio(r, num, den, rate) for r in valid])}
            for _, _, b in M_BINS:
                xs = [ratio(r, num, den, rate) for r in valid if m_bin(r["M"]) == b]
                cell[b] = {"geomean": geomean(xs), "n": len([x for x in xs if x])}
            out[field][name] = cell
    return out


def write(path, meta, rows, arms):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"meta": meta, "summary": summarize(rows, arms), "rows": rows}, fh, indent=1)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--target", default="rare", choices=["rare", "random"])
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS), metavar="ARM",
                    help="the arms to run (default: all eight)")
    ap.add_argument("--width", type=int, default=2, choices=[1, 2, 3],
                    help="literals per probed tuple")
    ap.add_argument("--batch", type=int, default=65536,
                    help="rows per launch, pinned for all arms; 0 = each arm's own batch")
    ap.add_argument("--target-s", type=float, default=1.5, help="seconds one sweep should take")
    ap.add_argument("--min-chunks", type=int, default=4)
    ap.add_argument("--max-chunks", type=int, default=32)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-clauses", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--only", nargs="*", default=None, metavar="FAMILY/INSTANCE[@CYCLES]",
                    help="these instances only; @CYCLES unrolls one to its own depth")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    import torch
    from gpap.engine import _core
    arms = list(args.arms)
    settings = {k: v for k, v in vars(args).items() if k not in ("out", "resume", "only")}
    rows = {}
    if args.resume and os.path.isfile(args.out):
        old = json.load(open(args.out))
        if old.get("meta", {}).get("args") != settings:
            raise SystemExit(f"--resume: {args.out} was written with other settings "
                             f"({old.get('meta', {}).get('args')}); rows would mix")
        rows = old.get("rows", {})
        log(f"[resume] {len(rows)} rows")
    meta = {"date": time.strftime("%F %T"), "gpu": torch.cuda.get_device_name(0),
            "libgpap": _core.build_id(), "torch": torch.__version__, "arms": arms,
            "args": settings}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = paths.tmp_dir()
    os.makedirs(tmp, exist_ok=True)

    wanted = {}                                     # FAMILY/INSTANCE -> [depth or None]
    for spec in args.only or []:
        k, _, c = spec.partition("@")
        wanted.setdefault(k, []).append(int(c) if c else None)
    todo = []
    for fam in args.families:
        for src in sorted(glob.glob(os.path.join(REPO_ROOT, FAMILIES[fam][0]))):
            name = os.path.splitext(os.path.basename(src))[0]
            if args.only and f"{fam}/{name}" not in wanted:
                continue
            todo += [(fam, src, name, d) for d in wanted.get(f"{fam}/{name}", [None])]
    for fam, src, name, depth in todo:
        _pattern, kind, cyc, pin = FAMILIES[fam]
        key = f"{fam}/{name}" + (f"@{depth}" if depth else "")
        if key in rows and "error" not in rows[key]:
            continue
        t0 = time.time()
        try:
            if kind == "cnf":
                cnf = src
            else:
                cycles = depth or (args.cycles if cyc == "T" else cyc)
                cnf = os.path.join(tmp, f"backend_ab-{fam}-{name}-c{cycles}-s{args.seed}-"
                                        f"{args.target}.cnf")
                if not os.path.isfile(cnf):         # written whole, or not at all
                    part = f"{cnf}.{os.getpid()}.part"
                    convert(kind, src, part, cycles, args.seed, pin, args.target)
                    os.replace(part, cnf)
            with open(cnf) as fh:
                header = next((ln for ln in fh if ln.startswith("p cnf")), "p cnf 0 0")
            m = int(header.split()[3])
            if m > args.max_clauses:
                row = {"M": m, "skip": f"{m:,} clauses is past --max-clauses"}
            else:
                row = measure(cnf, arms, args)
        except Exception as e:                      # noqa: BLE001 — recorded, never fatal
            row = {"error": f"{type(e).__name__}: {e}"[:300]}
        row["family"], row["wall_s"] = fam, round(time.time() - t0, 1)
        rows[key] = row
        write(args.out, meta, rows, arms)
        if "arms" in row:
            rate = {a: (row["arms"][a].get("impl_per_s") or 0) / 1e9 for a in arms}
            log(f"{key:28s} M={row['M']:>9,} K={row['K']:>6} chunks={row['chunks']:>2} "
                + " ".join(f"{a.replace('cuda-', 'c-').replace('opencl-', 'o-')}="
                           f"{rate[a]:.2f}G" for a in arms)
                + ("" if row["valid"] else f"  INVALID {row['checks']}"))
        else:
            log(f"{key:28s} {row.get('skip') or row.get('error')}")
    s = summarize(rows, arms)
    log(f"\n{s['valid']} valid rows of {s['rows']} ({s['skipped']} skipped, "
        f"{len(s['invalid'])} invalid)")
    for name, cell in s["ratios"].items():
        bins = "  ".join(f"{b}: {c['geomean']:.2f}x (n={c['n']})" if c["geomean"] else f"{b}: -"
                         for b, c in cell.items() if b != "all")
        tup = s["ratios_tuples"][name]["all"]
        log(f"  {name:42s} {cell['all']:.2f}x (tuples/s {tup:.2f}x)   {bins}" if cell["all"]
            else f"  {name}: -")
    log("BACKEND_AB_COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
