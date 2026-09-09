#!/usr/bin/env python3
"""Which instances is Barbarik's CM_test able to say anything about?

Two things must hold before a verdict means anything, and neither is about the sampler under
test. Both are checked here so the exclusion list is re-derived from the instances rather than
chosen after seeing which rows were inconvenient.

1. The trusted reference must be able to sample it. CM_test draws one sample per round from
   SPUR, so an instance SPUR cannot compile is outside the test's reach.

2. The declared sampling set must be an independent support. CM_test's kernel pins every declared
   indVar to sigma1's or sigma2's values, so the two buckets hold the extensions of sigma1 and of
   sigma2 times the chain formula's designed ratio — while SPUR is uniform over MODELS. When the
   extension count varies across projections, SPUR's share of a bucket is off by exactly that
   factor and CM_test rejects the trusted reference itself. Measured, the correspondence is
   exact: every constant-spread instance Accepts and every varying-spread one Rejects. A Reject
   there is a statement about the instance, so such instances are dropped from a uniformity
   table exactly as a vacuous pin is dropped from a throughput table
   (docs/findings/UNIFORMITY_FINDINGS.md §4b).

The spread is estimated from a sample of projections, so it is a lower bound: constant spread
over k projections is evidence of independence, not proof, and the SPUR control is the gate.

Usage:
  PYTHONPATH=. .venv/bin/python eval/uniformity/barbarik_screen.py <cnf> [<cnf> ...] \
      [--spur-seconds 90] [--projections 8] [--out eligible.txt]
"""
import argparse
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)


def _ind(path):
    ind = []
    for line in open(path):
        if line.startswith("c ind"):
            ind += [int(x) for x in line.split()[2:] if x != "0"]
    return ind


def _spur_ok(spur, path, seconds):
    """Can the trusted reference produce samples here at all, inside the budget?

    Screened at `-s 1`, which is what CM_test asks for once per round — and NOT the same cost as
    `-s 50`: ProcessBean.sk_8_64 returns 50 samples in 0.0s and then runs >500s on a single one
    (spur takes a different path for s=1). Screening at 50 would have declared it eligible and the
    sweep would have hung on it."""
    t0 = time.perf_counter()
    out = "/tmp/barbarik_screen_spur.out"
    rc = -1
    for n in ("1", "50"):
        try:
            p = subprocess.run([spur, "-seed", "1", "-q", "-s", n, "-out", out, "-cnf", path],
                               capture_output=True, timeout=seconds)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = -1
        if os.path.exists(out):
            os.unlink(out)
        if rc != 0:
            break
    return rc == 0, time.perf_counter() - t0


def _count(spur, nv, clauses, extra):
    tmp = "/tmp/barbarik_screen_count.cnf"
    with open(tmp, "w") as f:
        f.write(f"p cnf {nv} {len(clauses) + len(extra)}\n")
        for c in clauses:
            f.write(" ".join(map(str, c)) + " 0\n")
        for lit in extra:
            f.write(f"{lit} 0\n")
    try:
        p = subprocess.run([spur, "-count-only", "-cnf", tmp], capture_output=True, text=True,
                           timeout=300)
    except subprocess.TimeoutExpired:
        return -1
    finally:
        os.unlink(tmp)
    lines = p.stdout.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("# solutions"):
            try:
                return int(lines[i + 1].strip())
            except (ValueError, IndexError):
                return -1
    return -1


def _spread(spur, path, k):
    """max/min full-model count over k distinct projections of the declared support."""
    from pysat.formula import CNF
    from pysat.solvers import Minisat22

    cnf = CNF(from_file=path)
    ind = set(_ind(path))
    counts = []
    with Minisat22(bootstrap_with=cnf.clauses) as S:
        for _ in range(k):
            if not S.solve():
                break
            proj = [lit for lit in S.get_model() if abs(lit) in ind]
            n = _count(spur, cnf.nv, cnf.clauses, proj)
            if n > 0:
                counts.append(n)
            S.add_clause([-lit for lit in proj])
    if not counts:
        return None, 0
    return max(counts) / min(counts), len(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cnfs", nargs="+")
    ap.add_argument("--spur", default=os.path.join(REPO_ROOT, ".gpap/src/barbarik/samplers/spur"))
    ap.add_argument("--spur-seconds", type=float, default=20.0,
                    help="reference budget for ONE sample. CM_test needs 36 of them from the "
                         "reference plus 1,836 from the sampler under test, so a 65 s/sample "
                         "instance is an hours-long certificate per arm — that is a cost bound "
                         "on the TEST, stated up front, not a result-dependent exclusion")
    ap.add_argument("--projections", type=int, default=8,
                    help="how many distinct projections to count extensions for")
    ap.add_argument("--out", default=None, help="write the ELIGIBLE instance paths here")
    args = ap.parse_args()

    print(f"{'instance':<42}{'|ind|':>6}{'spur':>10}{'spread':>10}  verdict")
    eligible = []
    for path in args.cnfs:
        path = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
        name = os.path.basename(path)
        ok, dt = _spur_ok(args.spur, path, args.spur_seconds)
        if not ok:
            print(f"{name:<42}{len(_ind(path)):>6}{'TIMEOUT':>10}{'-':>10}  "
                  f"EXCLUDE (reference cannot sample it, {dt:.0f}s)")
            continue
        spread, n = _spread(args.spur, path, args.projections)
        sp = "?" if spread is None else f"{spread:.1f}x"
        if spread is not None and spread > 1.0:
            print(f"{name:<42}{len(_ind(path)):>6}{dt:>9.1f}s{sp:>10}  "
                  f"EXCLUDE (support not independent over {n} projections -> the "
                  f"model-uniform reference is not projection-uniform)")
            continue
        print(f"{name:<42}{len(_ind(path)):>6}{dt:>9.1f}s{sp:>10}  ELIGIBLE")
        eligible.append(path)
    print(f"\n{len(eligible)} of {len(args.cnfs)} eligible")
    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(eligible) + "\n")


if __name__ == "__main__":
    main()
