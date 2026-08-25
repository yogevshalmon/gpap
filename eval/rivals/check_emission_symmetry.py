#!/usr/bin/env python3
"""EMISSION SYMMETRY: prove that no tool in the comparison is charged for delivering its
solutions while another is not.

GPAP's timed run emits NOTHING — `sample(collect=False)` never gathers a model row to the host
(`keep_rows` is False, so `var_bits` is not called), and the sample writer runs in a separate
pass after the clock stops. CMSGen is timed with the patch's `--nowrite` for exactly this
reason (a stock binary writing every sample cost 8.2x of its wall on 27.sk_3_32). The GPU
gradient rivals must therefore also be timed on generation alone — otherwise our ratio is
partly a measurement of THEIR disk/PCIe traffic, which is not what the paper claims.

This checker RE-DERIVES that from the pinned rival sources instead of asserting it in prose:

  1. rival timed regions  — AST-scan each rival's run.py for the function its own `timer(...)`
     wraps, and report every host-transfer / IO call inside it (.cpu/.item/.numpy/.tolist,
     open/print, to_csv/save). The modules that function reaches (model/, utils/) are scanned
     too. Anything not in ALLOWED below is a FAILURE.
  2. GPAP timed region    — AST-check that every `.sample(...)` call inside pipeline.py's timed
     window passes `collect=False`.
  3. CMSGen               — audit `cms_mode` over a recorded results tree; anything but
     `nowrite` means that row timed CMSGen WITH sample emission.
  4. Recurrent egress bound — the one known, documented exception (below) is quantified per
     recorded cell from its own parameters, so the paper can state a bound rather than a hope.

Usage:
  PYTHONPATH=. python eval/rivals/check_emission_symmetry.py \
      [--results .experiments/l4_campaign_20260823] [--rivals-root .gpap/src]
"""
import argparse
import ast
import glob
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # eval/<sub>/ -> repo
sys.path.insert(0, REPO_ROOT)

# host-transfer / IO markers. `.detach()`/`.clone()` stay on device and are NOT egress.
EGRESS_ATTRS = {"cpu", "item", "numpy", "tolist", "to_csv", "save", "write", "writelines", "dump"}
EGRESS_NAMES = {"open", "print"}

# The ONLY tolerated host transfer inside any timed region, kept because removing it would mean
# editing the rival's measured algorithm. Recurrent-CircuitSAT snapshots its per-depth candidate
# assignments to the host at the end of every unroll depth; the cost is bounded in section 4 and
# it is charged to THEM, i.e. it can only make our ratio look better than it is.
ALLOWED = {
    ("Recurrent-CircuitSAT-Sampler", "run_back_prop_seq", "cpu"):
        "per-depth candidate-solution snapshot — measured at 0.7-1.4% of their model_runtime "
        "(section 4), left uncorrected so their rate stays the one they publish",
}

RIVALS = [("Demotic", "src/pytorch/run.py"),
          ("Recurrent-CircuitSAT-Sampler", "src/pytorch/run.py"),
          ("High-Throughput-SAT-Sampler", "src/pytorch/run.py")]

# Per-copy cost of THEIR access pattern (strided (batch,T) column -> pageable host), measured on
# an RTX 4080 Laptop at ramped clocks: 0.098 ms @30k, 0.185 ms @100k, 0.367 ms @300k rows =>
# t(ms) ~ 0.069 + 0.261 * MB (3.7 GB/s effective). PESSIMISTIC doubles it, covering a slower
# host PCIe path than the box the fit came from.
COPY_FIXED_MS, COPY_MS_PER_MB, PESSIMISM = 0.069, 0.261, 2.0


def _marks(node, helpers=None):
    """Every host-transfer / IO call inside an AST subtree, as (lineno, marker).

    `helpers` maps module-level function names to their defs, and calls into them are resolved
    ONE level deep. Without that, wrapping the transfer in a helper hides it — which is exactly
    what the setup script's egress meter does to Recurrent's `.cpu()` (`_gpap_snap`), so the
    checker would have gone quiet on the one case it exists to catch.
    """
    hits = []
    for n in ast.walk(node):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Attribute) and f.attr in EGRESS_ATTRS:
            hits.append((n.lineno, "." + f.attr + "()"))
        elif isinstance(f, ast.Name) and f.id in EGRESS_NAMES:
            hits.append((n.lineno, f.id + "()"))
        elif isinstance(f, ast.Name) and helpers and f.id in helpers:
            hits += [(n.lineno, m + f" [via {f.id}()]") for _, m in _marks(helpers[f.id])]
    return hits


def timed_functions(tree):
    """Names of the functions the rival's own `timer(self.NAME)(...)` wraps."""
    names = []
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Call)
                and isinstance(n.func.func, ast.Name) and n.func.func.id == "timer"):
            for a in n.func.args:
                if isinstance(a, ast.Attribute):
                    names.append(a.attr)
    return sorted(set(names))


def scan_rivals(root):
    ok = True
    for name, rel in RIVALS:
        path = os.path.join(root, name, rel)
        print(f"\n[1] {name}")
        if not os.path.isfile(path):
            print(f"    SKIP — no checkout at {path}")
            continue
        src = open(path).read()
        tree = ast.parse(src)
        lines = src.splitlines()
        fns = timed_functions(tree)
        defs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        helpers = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        if not fns:
            print("    FAIL — no timer(self.X) call found; the timed region cannot be located")
            ok = False
            continue
        for fn in fns:
            body = defs.get(fn)
            if body is None:
                print(f"    FAIL — timed function {fn} not found")
                ok = False
                continue
            hits = _marks(body, helpers)
            if not hits:
                print(f"    timed `{fn}`: CLEAN — no host transfer, no IO")
                continue
            for ln, marker in hits:
                key = (name, fn, marker.split(" [")[0].strip(".()"))
                note = ALLOWED.get(key)
                tag = "allowed" if note else "UNACCOUNTED"
                print(f"    timed `{fn}`: {tag} {marker} at run.py:{ln}  "
                      f"| {lines[ln - 1].strip()[:70]}")
                if note:
                    print(f"        ^ {note}")
                else:
                    ok = False
        # the modules the timed function reaches
        reach = []
        for sub in ("model", "utils"):
            for f in sorted(glob.glob(os.path.join(os.path.dirname(path), sub, "**", "*.py"),
                                      recursive=True)):
                hits = _marks(ast.parse(open(f).read()))
                if hits:
                    reach.append((os.path.relpath(f, os.path.dirname(path)), hits))
        if reach:
            for f, hits in reach:
                print(f"    reachable module {f}: {len(hits)} marker(s) — REVIEW")
            ok = False
        else:
            print("    reachable modules (model/, utils/): CLEAN")
    return ok


def scan_gpap():
    """Every `.sample(...)` inside pipeline.py's timed window must pass collect=False."""
    path = os.path.join(REPO_ROOT, "gpap", "sampling", "pipeline.py")
    tree = ast.parse(open(path).read())
    ends = [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "wall" for t in n.targets)
            and isinstance(n.value, ast.BinOp)]
    starts = [n.lineno for n in ast.walk(tree)
              if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "t0" for t in n.targets)]
    print("\n[2] GPAP timed region (gpap/sampling/pipeline.py)")
    if not ends or not starts:
        print("    FAIL — could not locate the timed window (t0 ... wall = ... - t0)")
        return False
    end = min(ends)
    start = max(s for s in starts if s < end)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "sample"
             and start < n.lineno < end]
    if not calls:
        print(f"    FAIL — no .sample() call inside lines {start}-{end}")
        return False
    ok = True
    for c in calls:
        kw = {k.arg: k.value for k in c.keywords}
        v = kw.get("collect")
        good = isinstance(v, ast.Constant) and v.value is False
        print(f"    line {c.lineno}: collect={'False' if good else ast.dump(v) if v else 'MISSING'}"
              f"  {'OK' if good else 'FAIL'}")
        ok &= good
    print("    => the timed run never gathers a model row to the host (keep_rows is False)"
          if ok else "    => TIMED RUN MAY EMIT")
    return ok


def scan_cmsgen(results):
    print("\n[3] CMSGen emission mode")
    files = sorted(glob.glob(os.path.join(results, "**", "*.json"), recursive=True))
    modes, n = {}, 0
    for f in files:
        try:
            d = json.load(open(f))
        except (ValueError, OSError):
            continue
        for r in (d.get("results") or []) if isinstance(d, dict) else []:
            if not isinstance(r, dict) or "cms_mode" not in r:
                continue
            n += 1
            modes[r["cms_mode"]] = modes.get(r["cms_mode"], 0) + 1
    if not n:
        print("    no rows with cms_mode found — nothing to audit")
        return True
    print(f"    {n} rows: " + ", ".join(f"{k}={v}" for k, v in sorted(modes.items())))
    ok = set(modes) == {"nowrite"}
    print("    => CMSGen timed emission-free on every row" if ok
          else "    => FAIL: some rows timed CMSGen WITH sample emission")
    return ok


def recurrent_egress(results, cycles=25):
    """Bound the one allowed exception, per recorded cell, from its own parameters.

    Copy count x the measured per-copy cost of that access pattern. Corroborated end-to-end by
    timing the block directly inside the rival on an RTX 4080: 0.70% (b03@100k), 1.32-1.42%
    (s386@100k), 0.88% (s1196@30k) of model_runtime, and removing the copy entirely leaves
    num_unique_solutions bit-identical while saving 1.41% on s386 over 4 interleaved reps. The
    direct measurements land near the PESSIMISTIC column here, so quote that one.
    """
    bench = os.path.join(REPO_ROOT, "benchmarks", "circuits", "bench", "seq")
    print("\n[4] Recurrent-CircuitSAT per-depth host snapshot — measured bound")
    rows = []
    for f in sorted(glob.glob(os.path.join(results, "**", "rivals_recurrent_*.json"),
                              recursive=True)):
        mode = os.path.basename(f).replace("rivals_recurrent_", "").replace(".json", "")
        for r in json.load(open(f)):
            rcs = r.get("rcs") or {}
            if not rcs.get("runtime_s"):
                continue
            bf = os.path.join(bench, r["name"] + ".bench")
            if not os.path.isfile(bf):
                continue
            pi = sum(1 for l in open(bf) if l.strip().upper().startswith("INPUT("))
            sp, batch = rcs["start_point"], rcs["batch"]
            copies = pi * sum(d + 1 for d in range(sp, cycles))
            mb = batch * 4 / 2 ** 20
            sec = copies * (COPY_FIXED_MS + COPY_MS_PER_MB * mb) / 1e3
            rows.append((mode, r["name"], copies, copies * mb / 1024, rcs["runtime_s"], sec))
    if not rows:
        print("    no timed Recurrent cells found")
        return True
    print(f"    {'mode':<7}{'inst':<8}{'copies':>8}{'GiB':>7}{'their rt':>10}"
          f"{'egress s':>10}{'%':>7}{'% pessim':>10}")
    worst = 0.0
    for mode, name, copies, gib, rt, sec in rows:
        pct, pes = 100 * sec / rt, 100 * sec * PESSIMISM / rt
        worst = max(worst, pes)
        print(f"    {mode:<7}{name:<8}{copies:>8,}{gib:>7.2f}{rt:>10.1f}{sec:>10.2f}"
              f"{pct:>6.2f}%{pes:>9.2f}%")
    print(f"    => worst case {worst:.2f}% of their reported model_runtime. NOT corrected: it is "
          f"one rival, worth ~1%, and\n       deducting it would RAISE our ratio — so an "
          f"uncorrected cell under-reports us, it never flatters us.")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rivals-root", default=None,
                    help="checkout root (default: the repo's .gpap/src)")
    ap.add_argument("--results", default=None,
                    help="campaign dir to audit (cms_mode + the Recurrent egress bound)")
    ap.add_argument("--cycles", type=int, default=25, help="unroll depth the rival rows used")
    args = ap.parse_args()
    root = args.rivals_root
    if root is None:
        from gpap import paths
        root = str(paths.src_dir())
    print(f"emission symmetry check | rivals={root}"
          + (f" | results={args.results}" if args.results else ""))
    ok = scan_rivals(root)
    ok &= scan_gpap()
    if args.results:
        ok &= scan_cmsgen(args.results)
        recurrent_egress(args.results, args.cycles)
    print("\n" + ("PASS — every tool is timed on generation alone (allowed exception quantified)"
                  if ok else "FAIL — an unaccounted emission cost is inside a timed region"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
