#!/usr/bin/env python3
"""Run Barbarik (Chakraborty & Meel, AAAI'19 — meelgroup/barbarik) with GPAP or CMSGen as the
sampler under test: it draws one sample from the sampler and one from a trusted uniform reference
(SPUR), builds a kernel CNF whose solutions mix the two in a known ratio, has the sampler under
test sample that kernel, and checks the observed bias. The tester is `CM_test` (uniform sampling);
Barbarik3's `PM_test` targets weighted distributions and needs a dual oracle, so it does not apply.

Two parameter settings, never pooled. The default here is the CMSGen paper's (eps 0.3, eta 1.8,
delta 0.1; FMCAD'21 §V-A), which unrolls to one loop of 36 rounds x (1 + 1 + 50) = 1,872 samples,
1,836 of them from the sampler under test, rejecting a round whose observed bias leaves
[0.05, 0.95]. Barbarik's own default (eps 0.1 / eta 0.9 / delta 0.05) is 477 rounds and 70,596
samples and rejects TV >= 0.45 rather than TV >= 0.90. Say which one produced a table.
`--explain` prints the derivation at any (eps, eta, delta).

How to read a verdict:
  * Reject is unconditional; Accept holds only under the paper's non-adversality assumption with
    respect to Kernel, and only against eta. Skew inside the band is invisible by design — for a
    sensitive test on enumerable spaces use eval/uniformity/fuzz_uniformity.py.
  * CMSGen's own Accept was fitted to this test (their Remark 1 tunes its restarts against
    Barbarik's rejections), so passing here is parity with a sampler tuned to pass here.
  * The control: `--sampler spur` puts the trusted uniform reference under test through this
    file's own hook and must Accept, or "GPAP Accepts" cannot be told from "our hook hides bias".
    The positive control `--config soft` is deliberately biased and must Reject; it shows the
    test can Reject at all, and UNIFORMITY_FINDINGS.md §4b records it rather than each campaign.

GPAP is hooked in-process (their SolutionRetriever monkeypatched) for one CUDA init per test; SPUR
keeps their stock ./samplers/spur path; CMSGen runs as a subprocess, as it normally is.

Setup (once), pinned at the commit the recorded runs used:
  git clone https://github.com/meelgroup/barbarik <repo>/.gpap/src/barbarik
  cd <repo>/.gpap/src/barbarik \
      && git checkout 2c0a6691b97c796a829408dc4db70d9f59f49382 \
      && git submodule update --init

Usage:
  PYTHONPATH=. .venv/bin/python eval/uniformity/barbarik_gpap.py <cnf> [<cnf> ...] \
      [--sampler gpap|pipe|cmsgen|spur|spur-raw] [--config b8|b1|s1|soft|bsup] [--draw batch|iid] \
      [--eta 1.8] [--epsilon 0.3] [--delta 0.1] [--seed 1] [--max N] [--out results.json]
Verdict per instance: "Accept:1 ..." or "RejectIteration:..."; no verdict when `--max` truncates.
An instance that errors is recorded as such and the sweep continues.
"""
import argparse
import io
import json
import os
import subprocess
import sys
import time
import traceback

from gpap import paths

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # eval/<sub>/ -> repo
if REPO_ROOT not in sys.path:        # guarded: unconditional, it prepended a duplicate entry on
    sys.path.insert(0, REPO_ROOT)    # every import, which is a side effect on the whole session
# Error text lands in a row that ships; a message that prints the argument it was handed put an
# operator's scratch directory into three rows of the published campaign (tests/experiments/test_artifact_gate.py).
from eval.paths import display_path  # noqa: E402 — after sys.path

# (random_var, B, lookahead), as gpap.sampling.pipeline defines a config. Every config a default
# portfolio can select must appear here or the certificate has a hole: the bandit emits only the
# winning config's rows, so "the pipeline is Accepted" means "every portfolio member is Accepted".
# "bsup" is special-cased below: B becomes the sampling set's size, making a chunk exact rejection
# sampling over the projection (eval/uniformity/fuzz_uniformity.py).
CONFIGS = {"b8": (True, 8, False), "b1": (True, 1, False), "s1": (False, 1, False),
           "soft": (True, 8, "soft"), "bsup": (True, None, False)}

#: OUR per-order sampling budget, for the GPAP arms alone. CM_test places ~72 orders per
#: instance at the CMSGen paper's setting, so 72x this bounds a GPAP certificate (worst case ~6
#: of 65 instances x 72 x 30 s = 3.6 h an arm). Neither upstream tool offers a number to inherit
#: -- barbarik times no sampler, stock CMSGen has no wall budget -- so it rests on our own rows:
#: 3x the ~10 s an order the slowest instance that still ACCEPTED averaged (c2670-c1-random,
#: 695 s over 72), and 1.5x the 20 s `barbarik_screen.py --spur-seconds` uses. A SAMPLING budget:
#: the `pipe` arm's parse, ELS and up to MINE_SECONDS of mining sit outside it (~5-10 s).
ORDER_SECONDS = 30.0

#: The rival's budget, left where the recorded campaign had it: effectively unbounded.
#: THE ASYMMETRY RUNS AGAINST US, the only direction it may run. ORDER_SECONDS came from OUR
#: costs, and imposing it on CMSGen was measured taking an Accept it had earned off
#: `c6288-c1-random`; unbounded it cannot, so every no-verdict here is ours and no CMSGen Accept
#: owes anything to a number we chose. The table's claim is therefore not "the same budget" but
#: "GPAP was held to 30 s an order; CMSGen was not held to anything".
#: `--maxtime` is still passed, at THIS value, so CMSGen ends its own loop and exits normally and
#: the samples it wrote can be read; the kill stays above it (as `gpap.benchmark.cmsgen_run`).
CMSGEN_ORDER_SECONDS = 1800.0

# How hard to try to fill one of barbarik's sample orders before recording "no verdict".
_FILL_ATTEMPTS = 24        # collector attempts per order
_FILL_GIVE_UP = 3          # consecutive ATTEMPTS returning NOTHING -> the instance really is dry


def _gpap_solutions(input_file, num_solutions, ind_var_list, seed, device, config, k_cap,
                    draw="batch", order_seconds=ORDER_SECONDS):
    """Sample `num_solutions` from GPAP on `input_file`, projected on ind_var_list (1-based).
    Returns solutions in barbarik's format: one list of signed ints per solution, consistently
    ordered — biasFind keys solutions by their literal string, and compares to the SPUR
    reference by SET, so any fixed order is valid.

    `draw` decides WHERE the samples come from, and the distinction is not cosmetic:

      * "batch" — one chunk, `num_solutions` of its surviving rows. This is what the shipped tool
        delivers. Rows in a chunk share ONE variable permutation (the core loop shuffles the
        order per chunk, not per row) and differ only in their polarity hash, so they are i.i.d. only
        CONDITIONAL on that order. Their marginal is the sampler's distribution; their joint is
        not a product.
      * "iid" — one chunk per sample, so every sample gets a fresh permutation and the batch is
        i.i.d., which is the premise Barbarik's guarantee is stated under.

    Certifying only "batch" would leave open whether an Accept survives the correlation; certifying
    only "iid" would certify something the user never receives. Run both.
    """
    import torch
    from pysat.formula import CNF

    from gpap.sampling.sampler import GPUSampler

    cnf = CNF(from_file=input_file)
    sup0 = [v - 1 for v in ind_var_list]
    rv, B, la = config
    if B is None:                              # "bsup": one decision round over the whole support
        B = max(1, len(ind_var_list))
    smp = GPUSampler.from_clauses(cnf.nv, cnf.clauses, torch.device(device), support=sup0,
                                  seed=seed, k_cap=k_cap)
    # ONE DEADLINE FOR THE WHOLE ORDER, and the ONLY bound: `max_chunks=None`, so nothing ends an
    # order before the budget we mean to give. Per CALL it would not be a per-order budget at all
    # (the batch arm makes up to _FILL_ATTEMPTS calls, the iid arm num_solutions per attempt);
    # with the 400-chunk cap it was the opposite, ending an order after 1.4 s of 30 (400 chunks
    # is 0.17 s on a 20-var formula, measured).
    #
    # The loop stays as a safety net, not as the budget: `sample_bits` no longer gives up on its
    # own, so one call fills the order or spends the deadline. Every call is a fresh set of
    # independent chunks, so concatenating them changes no distribution. An `iid` attempt ends at
    # its first dry call, so budgeting that arm in CALLS would cap it far lower for structural
    # reasons -- hence the scaled `attempt_cap`, which the error below quotes so no row can read
    # "173 of 24 attempts".
    rows, cols, empty, tries, gave_up, spent = [], None, 0, 0, False, False
    attempt_cap = _FILL_ATTEMPTS * (num_solutions if draw == "iid" else 1)
    deadline = time.monotonic() + order_seconds
    for _ in range(attempt_cap):
        left = deadline - time.monotonic()
        if left <= 0.0:
            spent = True
            break
        tries += 1
        before = len(rows)
        if draw == "iid":                      # a fresh chunk == a fresh variable permutation
            for _ in range(num_solutions - len(rows)):
                left = deadline - time.monotonic()
                if left <= 0.0:
                    spent = True
                    break
                b, cols = smp.sample_bits(1, config=(rv, B, la), max_chunks=None,
                                          max_seconds=left)
                if b is None or len(b) < 1:
                    break
                rows.append(b[0])
        else:
            b, cols = smp.sample_bits(num_solutions - len(rows), config=(rv, B, la),
                                      max_chunks=None, max_seconds=left)
            if b is not None and len(b):
                rows.extend(b)
        if len(rows) >= num_solutions or spent:
            break
        # "Dry" is an ATTEMPT that produced nothing, never one call inside it: both draws obey
        # the rule `_FILL_GIVE_UP` states, so the collector's stopping rule is never charged to
        # the sampler on a low-yield kernel.
        empty = 0 if len(rows) > before else empty + 1
        if empty >= _FILL_GIVE_UP:              # genuinely producing nothing, not merely rarely
            gave_up = True
            break
    bits = rows
    if bits is None or len(bits) < num_solutions:
        # WHICH STOPPING RULE ENDED THE ATTEMPT, in the row: `sample_bits` returns no stats, so
        # the core's own `stop` is out of reach here and the collector's own accounting stands in.
        # Without it a no-verdict reads as a sampler that cannot do the instance at all.
        why = (f"gave up on {_FILL_GIVE_UP} consecutive dry attempts" if gave_up
               else f"the {order_seconds:g}s order budget was spent" if spent
               else "attempt budget exhausted")
        raise RuntimeError(f"GPAP produced {0 if bits is None else len(bits)}/{num_solutions} "
                           f"samples on {display_path(input_file)} "
                           f"[{tries} of {attempt_cap} attempts; {why}]")
    if sorted(cols) != sorted(ind_var_list):   # biasFind needs one literal per declared indVar
        raise RuntimeError(f"projection columns {len(cols)} != declared support "
                           f"{len(ind_var_list)} on {display_path(input_file)}")
    return [[v if row[i] else -v for i, v in enumerate(cols)] for row in bits[:num_solutions]]


def _pipeline_solutions(input_file, num_solutions, ind_var_list, seed, device, portfolio_name,
                        k_cap, order_seconds=ORDER_SECONDS):
    """The SHIPPED pipeline under test, not one of its components.

    `sample_file` is exactly what `gpap sample` runs: parse, unit closure, ELS, entailed-clause
    mining, the adaptive bandit over the portfolio, and the GPU bit-projection delivery path.
    Certifying it directly closes by MEASUREMENT what the per-config arms close by argument (the
    bandit emits only the winning config's rows, so "every portfolio member Accepts" implies the
    pipeline does). Costs ~72 full pipeline invocations per instance, mining budget included —
    run it on a subset, not the whole suite.
    """
    import torch

    from gpap.sampling.pipeline import portfolio, sample_file

    info = sample_file(input_file, torch.device(device), n_time=num_solutions,
                       out_cap=num_solutions, seed=seed, k_cap=k_cap,
                       time_limit=order_seconds, configs=portfolio(portfolio_name))
    bits, cols = info.get("samples"), info.get("cols") or []
    if bits is None or len(bits) < num_solutions:
        # NAME THE VALVE AND THE WALL IN THE ROW. This becomes the `error` of a published
        # no-verdict, and "produced 0/50" alone reads as "the sampler cannot do this instance"
        # where it may mean our own stopping rule ended the attempt seconds into the budget.
        raise RuntimeError(f"pipeline produced {0 if bits is None else len(bits)}/"
                           f"{num_solutions} samples on {display_path(input_file)} "
                           f"[stop={info.get('stop')} after {info.get('wall_timed_s') or 0.0:.1f}s "
                           f"of a {order_seconds:.1f}s budget]")
    if sorted(cols) != sorted(ind_var_list):
        raise RuntimeError(f"projection columns {len(cols)} != declared support "
                           f"{len(ind_var_list)} on {display_path(input_file)}")
    return [[v if row[i] else -v for i, v in enumerate(cols)] for row in bits[:num_solutions]]


def _cmsgen_solutions(input_file, num_solutions, ind_var_list, seed, cmsgen_bin,
                      order_seconds=CMSGEN_ORDER_SECONDS):
    """Same contract, CMSGen as the sampler under test. CMSGen emits FULL assignments (it ignores
    `c ind`), so the projection onto ind_var_list happens here.

    `order_seconds` defaults to CMSGEN_ORDER_SECONDS, not `ORDER_SECONDS` -- see there for why
    the asymmetry is the conservative choice. SPUR is likewise not bounded by us: barbarik
    invokes the reference on its own stock path and it is not a tool under test."""
    from gpap.benchmark import cmsgen_supports_maxtime
    out = paths.scratch_file("cmsgen_barbarik", ".out")
    cmd = [cmsgen_bin, "--samples", str(num_solutions), "--seed", str(int(seed)),
           "--verb", "0", "--samplefile", out, input_file]
    deadline = order_seconds if cmsgen_supports_maxtime(cmsgen_bin) else 0.0
    if deadline:
        cmd[1:1] = ["--maxtime", f"{deadline:g}"]
    # Above the deadline, never instead of it. On a stock binary (no --maxtime) the kill IS the
    # budget, and the row says so.
    hard = (deadline + max(30.0, 0.5 * deadline)) if deadline else order_seconds
    # The budget, in the words both failure paths use: a no-verdict row has to distinguish
    # "could not fill the order" from "we cut it off".
    budget = (f"[{order_seconds:g}s budget" +
              ("" if deadline else ", enforced as a subprocess kill: this binary has no "
                                  "--maxtime") + "]")
    try:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=hard)
        except subprocess.TimeoutExpired:
            # A stock binary has no deadline to exit on, so the kill is its budget: the same
            # fact as a short count, and `TimeoutExpired` would leave the row unreadable.
            raise RuntimeError(f"cmsgen produced 0/{num_solutions} on "
                               f"{display_path(input_file)} {budget} — killed at {hard:g}s of "
                               f"wall, so no samples could be read back") from None
        # 10 = SATISFIABLE. A `--maxtime` exit is NOT a failure: the patch returns the DIMACS
        # "unknown" code (15, measured) through its normal path having written what it managed,
        # which is why `gpap.benchmark.cmsgen_run` ignores the code entirely. Keyed on the
        # patch's own marker, not on 15, so a genuine crash exiting 15 is still a failure.
        # Measured: without this a budget stop on c6288-c1-random read as "cmsgen failed".
        stopped_on_budget = "c gpap-maxtime reached" in (p.stdout or "")
        if p.returncode not in (0, 10) and not stopped_on_budget:
            raise RuntimeError(f"cmsgen failed on {display_path(input_file)}: "
                               f"{p.stdout[-300:]}{p.stderr[-300:]}")
        sols = []
        with open(out) as f:
            for line in f:
                lits = {int(x) for x in line.split() if x.lstrip("-").isdigit()}
                if lits:
                    sols.append([v if v in lits else -v for v in ind_var_list])
    finally:
        # Every raise above used to leave the file behind, and a campaign errors per cell, so the
        # scratch root filled with sample files nobody would ever read.
        try:
            os.unlink(out)
        except OSError:
            pass
    if len(sols) < num_solutions:
        raise RuntimeError(f"cmsgen produced {len(sols)}/{num_solutions} on "
                           f"{display_path(input_file)} {budget}")
    return sols[:num_solutions]


def _spur_raw_solutions(stock, C, input_file, num_solutions, ind_var_list, seed):
    """The same control WITHOUT this file's reformatting — barbarik's own retriever output handed
    straight back. Pairs with `_spur_solutions` to separate "our projection step is wrong" from
    "this instance is not a valid subject for the test"."""
    return stock(input_file, num_solutions, C.SAMPLER_SPUR, ind_var_list, 0, seed)[:num_solutions]


def _spur_solutions(stock, C, input_file, num_solutions, ind_var_list, seed):
    """NEGATIVE CONTROL: the trusted uniform reference, put UNDER TEST.

    Same dispatch hook, same "list of signed ints over ind_var_list" reformatting that GPAP and
    CMSGen go through — only the source of the assignments is exact-uniform. Barbarik must Accept
    this. If it does not, the defect is in this harness and no other row on the page is evidence.
    SPUR emits full assignments over 1..n, so the projection happens here (as it does for CMSGen).
    """
    sols = stock(input_file, num_solutions, C.SAMPLER_SPUR, ind_var_list, 0, seed)
    return [[v if v in set(s) else -v for v in ind_var_list] for s in sols[:num_solutions]]


class _QuietOS:
    """barbarik's `os`, with its housekeeping `rm` done in Python instead of a shell.

    CM_test `os.unlink`s the kernel CNF as soon as the sampler has consumed it (cnf.py:1008) and
    then, at the end of the round and again on Accept, runs `os.system("rm *.nnf <that file>")`
    (cnf.py:974, 1032). The file is already gone, and `*.nnf` is d-DNNF output from the WAPS
    reference path, which CM_test with SPUR never produces — so every round prints two
    `rm: cannot remove` warnings for nothing (74 lines per Accept). The return value is ignored
    upstream, so nothing is behind them. The checkout is pinned, so the shim lives here: it does
    the same deletion, quietly, and passes every other `os.system` call through untouched.
    Installed as `interfaces.cnf.os` only — the global `os` module is not patched."""

    def __init__(self, base):
        self._base = base

    def __getattr__(self, name):
        return getattr(self._base, name)

    def system(self, cmd):
        if not cmd.startswith("rm *.nnf "):
            return self._base.system(cmd)
        import glob
        for f in glob.glob("*.nnf") + cmd.split()[2:]:
            try:
                self._base.unlink(f)
            except FileNotFoundError:
                pass
        return 0


class _Tee(io.StringIO):
    """Capture CM_test's stdout (it PRINTS its verdict rather than returning it) while still
    streaming it, so a long sweep stays watchable and still ends machine-readable."""

    def __init__(self, live):
        super().__init__()
        self._live = live

    def write(self, s):
        self._live.write(s)
        self._live.flush()
        return super().write(s)


def _verdict(text):
    """(verdict, samples) from CM_test's output. It prints exactly one of these lines."""
    for line in reversed(text.splitlines()):
        if line.startswith("Accept:1"):
            return "Accept", int(line.split("TotalSamplesGenerated:")[1].strip())
        if line.startswith("RejectIteration:"):
            return "Reject", int(line.split("TotalSolutionsGenerated:")[1].strip())
    return "no-verdict", None


def _bias_range(text):
    """The per-round bias values CM_test actually observed, from the bucket counts it prints.

    The verdict alone hides HOW MUCH headroom an Accept had. The band is [0.05, 0.95] at the
    CMSGen paper's parameters, which is wide; a sampler sitting at 0.5 every round and one
    grazing 0.94 both print "Accept:1". Recording max(count)/sum(count) per round makes the
    difference visible, and makes a Reject's margin visible too.
    """
    vals = []
    for line in text.splitlines():
        if "Printing counts of each sample:" in line:
            try:
                counts = json.loads(line.split(":", 1)[1].strip())
            except (ValueError, IndexError):
                continue
            tot = sum(counts)
            if tot:
                vals.append(max(counts) / tot)
    if not vals:
        return None
    return {"rounds": len(vals), "bias_max_share_min": round(min(vals), 4),
            "bias_max_share_max": round(max(vals), 4),
            "bias_max_share_mean": round(sum(vals) / len(vals), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cnfs", nargs="*", help="CNF instances (must carry `c ind` sampling sets)")
    ap.add_argument("--sampler", default="gpap", choices=["gpap", "pipe", "cmsgen", "spur", "spur-raw"],
                    help="which sampler is under test (the SPUR reference is unchanged). "
                         "'gpap' = one pinned config; 'pipe' = the whole shipped pipeline "
                         "(ELS + mining + bandit + delivery); "
                         "'spur' puts the trusted uniform sampler under test = NEGATIVE CONTROL")
    ap.add_argument("--cmsgen", default=str(paths.tool("cmsgen")),
                    help="cmsgen binary, for --sampler cmsgen")
    ap.add_argument("--barbarik", default=str(paths.src_dir() / "barbarik"),
                    help="path to a meelgroup/barbarik checkout (with submodules)")
    ap.add_argument("--eta", type=float, default=1.8,
                    help="farness to reject, in L1 (CMSGen paper default 1.8 == TV 0.9)")
    ap.add_argument("--epsilon", type=float, default=0.3, help="closeness to accept (default 0.3)")
    ap.add_argument("--delta", type=float, default=0.1, help="confidence (default 0.1)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max", type=int, default=sys.maxsize,
                    help="cap on total samples (capping ends the test WITHOUT a verdict)")
    ap.add_argument("--device", default="cuda",
                    help="gpap device. The engine is CUDA-only since the packed-kernel refactor; "
                         "there is no CPU path any more")
    ap.add_argument("--config", default="b8", choices=list(CONFIGS),
                    help="gpap sampling config under test (bsup = B set to |sampling set|)")
    ap.add_argument("--portfolio", default="circuit",
                    choices=["circuit", "default", "b8", "b1", "s1"],
                    help="for --sampler pipe: which portfolio the bandit selects from")
    ap.add_argument("--order-seconds", type=float, default=ORDER_SECONDS,
                    help=f"sampling budget for ONE order barbarik places, for the GPAP arms only "
                         f"(default {ORDER_SECONDS:g} s). CM_test places about 72 orders per "
                         f"instance, so 72 x this bounds a GPAP certificate. The rival is NOT "
                         f"held to it -- see --cmsgen-order-seconds -- so every no-verdict this "
                         f"campaign reports is ours. SPUR is likewise unbounded on barbarik's "
                         f"own stock path: it is not a tool under test")
    ap.add_argument("--cmsgen-order-seconds", type=float, default=CMSGEN_ORDER_SECONDS,
                    help=f"the rival's per-order budget, deliberately NOT --order-seconds "
                         f"(default {CMSGEN_ORDER_SECONDS:g} s, effectively unbounded and what "
                         f"the recorded campaign gave it). Ours was derived from our own costs, "
                         f"so imposing it on CMSGen let a number we chose take a verdict off the "
                         f"rival -- measured: it turned an earned Accept on c6288-c1-random into "
                         f"a no-verdict. The asymmetry runs against us, which is the only "
                         f"direction it may run")
    ap.add_argument("--draw", default="batch", choices=["batch", "iid"],
                    help="how the requested samples are drawn from GPAP. 'batch' = one chunk, as "
                         "the tool delivers (rows share the chunk's variable permutation). "
                         "'iid' = one chunk per sample, which is the independence premise "
                         "barbarik's guarantee is stated under. Certify BOTH")
    ap.add_argument("--k-cap", default="auto",
                    help="per-chunk batch ceiling, as `gpap sample` takes it. K is a batch size, "
                         "not a distribution knob (every row is an independent trajectory), but a "
                         "SMALL K makes short chunks and the collector's zero-yield stop fires "
                         "sooner — which shows up as 'produced 9/50', an inability to fill "
                         "barbarik's order rather than a verdict. Default matches the shipped tool")
    ap.add_argument("--out", default=None,
                    help="append one JSON row per instance here (verdict, samples, wall)")
    ap.add_argument("--explain", action="store_true",
                    help="print what the chosen (eps, eta, delta) unroll to, and exit")
    args = ap.parse_args()

    if args.explain:
        _explain(args.epsilon, args.eta, args.delta)
        return
    if not args.cnfs:
        sys.exit("no instances given (use --explain to see what the parameters mean)")

    # ABSOLUTE BEFORE THE CHDIR. Everything else a caller can pass is already resolved against
    # REPO_ROOT (the instances below, `--out`) because this process moves its own cwd; `--barbarik`
    # was not, so a relative `--barbarik .gpap/src/barbarik` inserted a relative sys.path entry
    # and then chdir'd out from under it -- `ModuleNotFoundError: No module named 'interfaces'`,
    # pointing at the checkout rather than at the path handling. Same for the binary: the chdir
    # would strip a relative `--cmsgen` of its meaning inside the sampler arm.
    args.barbarik = os.path.abspath(args.barbarik)
    if args.cmsgen:
        args.cmsgen = os.path.abspath(args.cmsgen)
    if not os.path.isdir(os.path.join(args.barbarik, "interfaces")):
        sys.exit(f"no barbarik checkout at {args.barbarik} (see module docstring)")
    sys.path.insert(0, args.barbarik)
    os.chdir(args.barbarik)               # their code invokes ./samplers/spur and `rm *.nnf`
    import interfaces.cnf as C

    # Hook: dispatch SAMPLER_CUSTOM to the sampler under test, everything else (the SPUR uniform
    # reference) to the stock retriever.
    config = CONFIGS[args.config]
    stock = C.SolutionRetriever.getSolutionFromSampler
    _draw = "" if args.draw == "batch" else f"/{args.draw}"
    label = {"gpap": f"gpap:{args.config}{_draw}", "pipe": f"pipe:{args.portfolio}",
             "cmsgen": "cmsgen", "spur": "spur(control)",
             "spur-raw": "spur-raw(control)"}[args.sampler]

    def dispatch(inputFile, numSolutions, samplerType, indVarList, verbosity, newSeed):
        if samplerType != C.SAMPLER_CUSTOM:
            return stock(inputFile, numSolutions, samplerType, indVarList, verbosity, newSeed)
        seed = int(newSeed) + args.seed
        if args.sampler == "gpap":
            sols = _gpap_solutions(inputFile, numSolutions, indVarList, seed, args.device,
                                   config, args.k_cap, args.draw, args.order_seconds)
        elif args.sampler == "pipe":
            sols = _pipeline_solutions(inputFile, numSolutions, indVarList, seed, args.device,
                                       args.portfolio, args.k_cap, args.order_seconds)
        elif args.sampler == "cmsgen":
            # NOT args.order_seconds: the rival keeps its own, far larger budget on purpose.
            sols = _cmsgen_solutions(inputFile, numSolutions, indVarList, seed, args.cmsgen,
                                     args.cmsgen_order_seconds)
        elif args.sampler == "spur":
            sols = _spur_solutions(stock, C, inputFile, numSolutions, indVarList, seed)
        else:
            sols = _spur_raw_solutions(stock, C, inputFile, numSolutions, indVarList, seed)
        if verbosity:
            print(f"{label} returned {len(sols)} solutions")
        return sols

    C.SolutionRetriever.getSolutionFromSampler = staticmethod(dispatch)
    _stock_name = C.get_sampler_string
    C.get_sampler_string = lambda t: label if t == C.SAMPLER_CUSTOM else _stock_name(t)
    C.os = _QuietOS(C.os)                 # silence its double-delete `rm` warnings (see class)

    rows = []
    for cnf in args.cnfs:
        cnf = cnf if os.path.isabs(cnf) else os.path.abspath(os.path.join(REPO_ROOT, cnf))
        print(f"\n=== barbarik CM_test: {label} on {cnf} "
              f"(eta={args.eta} eps={args.epsilon} delta={args.delta}) ===", flush=True)
        # THE KNOBS THAT MOVE A VERDICT, IN THE ROW THAT CARRIES IT: `order_seconds` decides how
        # often an order goes unfilled and the unfilled orders ARE the headline, `k_cap` sizes
        # the chunk and so how much an order yields. THIS ARM's budget, not the invocation's --
        # the arms are deliberately not on one (CMSGEN_ORDER_SECONDS), so one number per file
        # would be a lie about one of them, and so would stamping one on the unbounded SPUR arms.
        budget = {"gpap": args.order_seconds, "pipe": args.order_seconds,
                  "cmsgen": args.cmsgen_order_seconds}.get(args.sampler)
        row = {"sampler": label, "instance": os.path.basename(cnf), "seed": args.seed,
               "eta": args.eta, "epsilon": args.epsilon, "delta": args.delta,
               "order_seconds": budget, "k_cap": args.k_cap, **_instance_shape(cnf)}
        t0 = time.perf_counter()
        tee = _Tee(sys.stdout)
        try:
            import contextlib
            with contextlib.redirect_stdout(tee):
                t = C.cnf_test(C.SAMPLER_CUSTOM, cnf, args.eta, args.epsilon, args.delta,
                               args.max, verbosity=0)
                t.CM_test(args.seed)
            row["verdict"], row["samples"] = _verdict(tee.getvalue())
            row["bias"] = _bias_range(tee.getvalue())
        except BaseException as exc:          # SystemExit too: barbarik exits on some failures
            row["verdict"], row["error"] = "error", f"{type(exc).__name__}: {exc}"
            traceback.print_exc(file=sys.stdout)
        row["wall_s"] = round(time.perf_counter() - t0, 2)
        print(f"--> {label} {row['instance']}: {row['verdict']} "
              f"samples={row.get('samples')} wall={row['wall_s']}s", flush=True)
        rows.append(row)
        if args.out:                          # write after EVERY instance: a killed sweep keeps
            out = args.out if os.path.isabs(args.out) else os.path.join(REPO_ROOT, args.out)
            with open(out, "a") as f:         # what it already proved
                f.write(json.dumps(row) + "\n")

    n_acc = sum(r["verdict"] == "Accept" for r in rows)
    n_rej = sum(r["verdict"] == "Reject" for r in rows)
    n_err = sum(r["verdict"] == "error" for r in rows)
    print(f"\n===== {label}: Accept {n_acc} / Reject {n_rej} / error {n_err} "
          f"of {len(rows)} instances =====")


def _instance_shape(path):
    """(n_vars, n_clauses, n_ind) for the results table — a verdict without the instance's shape
    cannot be checked against anyone else's run of the same benchmark."""
    nv = nc = 0
    ind = 0
    with open(path) as f:
        for line in f:
            if line.startswith("p cnf"):
                _, _, nv, nc = line.split()[:4]
            elif line.startswith("c ind"):
                ind += len([x for x in line.split()[2:] if x != "0"])
    return {"n_vars": int(nv), "n_clauses": int(nc), "n_ind": ind}


def _explain(epsilon, eta, delta):
    """Print the tester's own unrolled constants. The verdict line reports a sample COUNT, and
    nobody should have to reverse-engineer what it means from barbarik's source."""
    from math import ceil, e, log
    total_loops = int(ceil(log(2.0 / (eta + 2 * epsilon), 2)) + 1)
    print(f"CM_test with epsilon={epsilon} eta={eta} delta={delta}\n")
    print(f"  loops: {total_loops}")
    budget = 0
    for j in range(total_loops, 0, -1):
        tj = int(ceil(pow(2, j) * (2 * epsilon + eta) / ((eta - 2 * epsilon) ** 2)
                      * log(4.0 / (eta + 2 * epsilon), 2) * (4 * e / (e - 1) * log(1.0 / delta))))
        beta = (pow(2, j - 1) + 1) * (eta + 2 * epsilon) / (4 + (2 * epsilon + eta) * (pow(2, j - 1) - 1))
        gamma = (beta - 2 * epsilon) / 4
        n_sol = int(ceil(ceil(1 / (8.79 * gamma * gamma))
                         * log(16 * (e / (e - 1)) * (1 / (delta * (eta - 2 * epsilon) ** 2))
                               * log(4 / (eta + 2 * epsilon), 2) * log(1 / delta), 2)))
        lo, hi = (1 - (beta + 2 * epsilon) / 2) / 2, (1 + (beta + 2 * epsilon) / 2) / 2
        print(f"  loop j={j}: {tj} rounds x (1 under-test + 1 SPUR + {n_sol} kernel samples)"
              f" = {tj * (n_sol + 2)} samples")
        print(f"             a round REJECTS when the kernel bias leaves [{lo:.4f}, {hi:.4f}]"
              f"  (i.e. <= {int(lo * n_sol)} or >= {int(ceil(hi * n_sol))} of {n_sol})")
        budget += tj * (n_sol + 2)
    print(f"\n  Accept costs {budget} samples; a Reject stops at the first bad round.")
    print(f"  Guarantee: Accept w.p. >= {1 - delta:.0%} if the sampler is {epsilon}-close to "
          f"uniform,\n             Reject w.p. >= {1 - delta:.0%} if it is {eta}-far in L1 "
          f"(= TV >= {eta / 2:.2f}).\n  Between those it promises NOTHING — an Accept is "
          f"'not caught being {eta}-far', not 'uniform'.")


if __name__ == "__main__":
    main()
