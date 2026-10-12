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

# How hard to try to fill one of barbarik's sample orders before recording "no verdict".
#: THE PER-ORDER SAMPLING BUDGET, and the same one for every sampler under test. CM_test places
#: about 72 orders per instance at the CMSGen paper's setting, so this times 72 bounds a
#: certificate. Neither upstream tool offers a number to inherit: barbarik runs every sampler
#: through a bare `os.system` with no timeout at all, and stock CMSGen has no wall budget either
#: (`--maxtime` is our own patch, defaulting to unlimited). So the number is ours, and it rests
#: on this campaign's own rows: the slowest instance that still ACCEPTED averaged 9.7 s per order
#: (c2670-c1-random, 695 s over 72), so a 10 s budget would have turned a pass into a no-verdict.
#: 30 s is about 3x that, and 1.5x the 20 s `barbarik_screen.py --spur-seconds` already uses for
#: one reference order. The cost is bounded by the instances that would otherwise run forever:
#: ~6 of 65, so 6 x 72 x 30 s = 3.6 h per arm in the worst case. It is a SAMPLING budget: the
#: `pipe` arm re-runs the whole shipped pipeline per order, so parse, ELS and up to MINE_SECONDS
#: of mining sit outside it (~5-10 s an order). That setup is part of what the arm certifies and
#: is not stripped to make the arithmetic tidier -- it is stated instead.
ORDER_SECONDS = 30.0

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
    # WHY RETRY AT ALL, now that `sample_bits` no longer gives up on its own. It used to stop
    # after the cold start plus two dry chunks -- right for "produces nothing", wrong for
    # "produces rarely", and Barbarik's kernel is the second case by construction (it enlarges
    # the sampling set while pinning the support, so a rejection sampler's yield drops). That
    # rule is opt-in now and this arm does not ask for it, so a single call already runs to its
    # budget. The loop stays because the two stopping rules answer different questions: a call
    # ends when ITS budget or chunk cap is out, the loop ends when the ORDER is filled, dry, or
    # out of time. Every call is a fresh set of independent chunks, so concatenating them
    # changes no distribution. An
    # `iid` attempt ends at its first dry call, so budget that arm in CALLS or the same attempt
    # count would cap it far lower for purely structural reasons. A genuinely dry instance is
    # stopped by the give-up rule below, in 3 attempts either way.
    # ONE DEADLINE FOR THE WHOLE ORDER, and it is the ONLY bound: `max_chunks=None`. The wall
    # is the budget we mean to give, so nothing else may end an order first.
    #
    # Both halves of that were wrong before and in opposite directions. `order_seconds` per CALL
    # is not a per-order budget at all -- the `batch` arm makes up to _FILL_ATTEMPTS calls
    # (24 x 30 s = 12 min for one of ~72 orders) and the `iid` arm up to num_solutions per
    # attempt -- while a `max_chunks` cap ended an order far SHORT of the budget: 400 chunks is
    # 0.17 s on a 20-var formula (measured), so three dry attempts abandoned the order in 1.4 s
    # of its 30 s and the arms were not on the same budget after all, which is the whole claim
    # this number exists to support. The sampler's own give-up rule used to hide both; it is
    # off by default now, so the bound has to be exactly the one we state.
    #
    # The retry loop survives as a safety net, not as the budget: one call now either fills the
    # order or spends the deadline, so `_FILL_GIVE_UP` is reachable only on a sampler that
    # returns short for its own reasons.
    rows, cols, empty, tries, gave_up, spent = [], None, 0, 0, False, False
    # Named, because the error below quotes it: the iid arm's cap is NOT `_FILL_ATTEMPTS` (see
    # the note above on why it is scaled), and a row reading "173 of 24 attempts" is a defect.
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
        # WHICH STOPPING RULE ENDED THE ATTEMPT, in the row. `sample_bits` returns no stats, so
        # the core's own `stop` is not reachable here (unlike the pipeline arm) -- but the
        # COLLECTOR's accounting is, and it answers the same question: `gave_up` means three
        # consecutive attempts came back empty, `attempts` says how many were spent of the
        # budget. Without it a no-verdict cannot be told from a sampler that cannot do the
        # instance at all.
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
        # NAME THE VALVE IN THE ROW. This message becomes the `error` of a published no-verdict,
        # and "produced 0/50" alone reads as "the sampler cannot do this instance" when what it
        # often means is that OUR stopping rule ended the attempt after a few empty chunks — on
        # a Barbarik kernel, which is adversarial for a rejection sampler by construction and is
        # therefore exactly where the two readings diverge. Carry the wall too: a reader can then
        # see at a glance that 120 s were available and 2 were used.
        raise RuntimeError(f"pipeline produced {0 if bits is None else len(bits)}/"
                           f"{num_solutions} samples on {display_path(input_file)} "
                           f"[stop={info.get('stop')} after {info.get('wall_timed_s') or 0.0:.1f}s "
                           f"of a {order_seconds:.1f}s budget]")
    if sorted(cols) != sorted(ind_var_list):
        raise RuntimeError(f"projection columns {len(cols)} != declared support "
                           f"{len(ind_var_list)} on {display_path(input_file)}")
    return [[v if row[i] else -v for i, v in enumerate(cols)] for row in bits[:num_solutions]]


def _cmsgen_solutions(input_file, num_solutions, ind_var_list, seed, cmsgen_bin,
                      order_seconds=ORDER_SECONDS):
    """Same contract, CMSGen as the sampler under test. CMSGen emits FULL assignments (it ignores
    `c ind`), so the projection onto ind_var_list happens here.

    `order_seconds` IS THE SAME BUDGET THE GPAP ARMS GET, and that is the point. This call used
    to be bounded only by a 1,800 s subprocess kill with no `--maxtime`, i.e. effectively
    unbounded per order, while the GPAP arms were ended by the sampler's own give-up rule in
    about two seconds. The headline of this campaign is a NO-VERDICT count -- "GPAP accepted on
    60 of 65, CMSGen on 65" -- and a no-verdict is a coverage outcome, which is exactly what a
    per-order budget decides. Two arms on two budgets make that number a budget comparison
    wearing a uniformity label. `--maxtime` bounds CMSGen's own sampling loop so it exits
    normally; the subprocess kill stays as a backstop above it, since the solver's deadline is
    CPU time and does not cover reading a large CNF (the same split as
    `gpap.benchmark.cmsgen_run`).

    SPUR, the trusted reference, is deliberately left unbounded: barbarik invokes it through its
    own stock path and it is not a tool under test. Stated here rather than discovered."""
    from gpap.benchmark import cmsgen_supports_maxtime
    out = paths.scratch_file("cmsgen_barbarik", ".out")
    cmd = [cmsgen_bin, "--samples", str(num_solutions), "--seed", str(int(seed)),
           "--verb", "0", "--samplefile", out, input_file]
    deadline = order_seconds if cmsgen_supports_maxtime(cmsgen_bin) else 0.0
    if deadline:
        cmd[1:1] = ["--maxtime", f"{deadline:g}"]
    # Above the deadline, never instead of it: a stock binary has no deadline, so there the kill
    # IS the budget and the arms are only as equal as that makes them (reported in the row).
    hard = (deadline + max(30.0, 0.5 * deadline)) if deadline else order_seconds
    # The budget, in the words both failure paths use. A reader of a no-verdict row has to be
    # able to tell "could not fill the order" from "we cut it off", and on a stock binary the
    # kill IS the budget, so the kill is an ordinary outcome of this arm.
    budget = (f"[{order_seconds:g}s budget" +
              ("" if deadline else ", enforced as a subprocess kill: this binary has no "
                                  "--maxtime") + "]")
    try:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=hard)
        except subprocess.TimeoutExpired:
            # Reached on a stock binary (no deadline to exit on) or when a patched one's CPU
            # deadline did not cover the wall. Either way it is the same fact as a short count,
            # and reporting it as `TimeoutExpired` would leave the row unreadable.
            raise RuntimeError(f"cmsgen produced 0/{num_solutions} on "
                               f"{display_path(input_file)} {budget} — killed at {hard:g}s of "
                               f"wall, so no samples could be read back") from None
        if p.returncode not in (0, 10):        # 10 = SATISFIABLE
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
                    help=f"sampling budget for ONE order barbarik places, THE SAME for every "
                         f"sampler under test -- GPAP and CMSGen alike (default "
                         f"{ORDER_SECONDS:g} s). CM_test places about 72 orders per instance, so "
                         f"72 x this bounds a certificate. Two arms on two budgets would make "
                         f"the no-verdict count a budget comparison rather than a uniformity "
                         f"one. SPUR, the trusted reference, stays unbounded on barbarik's own "
                         f"stock path: it is not a tool under test")
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
            sols = _cmsgen_solutions(inputFile, numSolutions, indVarList, seed, args.cmsgen,
                                     args.order_seconds)
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
        # THE KNOBS THAT MOVE A VERDICT, IN THE ROW THAT CARRIES IT. `order_seconds` decides
        # how often an order goes unfilled, and the unfilled orders ARE this campaign's headline
        # ("accepted on 60 of 65, 5 no-verdict"), so a row without it cannot be compared with
        # another run of the same instance -- ours included. `k_cap` sizes the chunk and so
        # decides how much one order yields (see --k-cap's help). None on the SPUR arms: the
        # budget does not reach them (barbarik's own stock path invokes the reference), and
        # stamping a number they never saw would make `order_budget` below refuse a file that is
        # perfectly consistent.
        under_test = args.sampler in ("gpap", "pipe", "cmsgen")
        row = {"sampler": label, "instance": os.path.basename(cnf), "seed": args.seed,
               "eta": args.eta, "epsilon": args.epsilon, "delta": args.delta,
               "order_seconds": args.order_seconds if under_test else None,
               "k_cap": args.k_cap, **_instance_shape(cnf)}
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
