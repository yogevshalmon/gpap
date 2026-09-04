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
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # eval/<sub>/ -> repo
sys.path.insert(0, REPO_ROOT)

from eval.rivals.rival_common import (CACHE_MISS, VARIANTS, cache_peek,  # noqa: E402
                                      cache_store, cached_gpap, cmsgen_arm, fit_batch,
                                      _available_bytes, expand_names, host_provenance,
                                      native_pin_check, oversize_reason, prepare_variant,
                                      RIVAL_SEED, exact_distinct, median_run, resolve_lr,
                                      space_from_distinct, unsat_at_depth,
                                      resolve_steps, rival_commit, rival_vram_budget,
                                      run_at_batch, variant_note)
from eval.rivals.unroll_to_bench import and_tree                      # noqa: E402

from gpap import paths
from gpap.sampling.pipeline import merge_sound_check, sound_verdict

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
# INITIALIZATION budget: the whole start_point search — every probe's model build included —
# must settle inside ONE run timeout (300 s by default), else the instance is an error row for
# the rival ("did not initialize"). The probe is a 1-step run whose wall is the model build, the
# same work the measured run does, so a per-probe cap smaller than the run's is not a faster
# probe, only a failing one: it read 60 s against the run's 300 s for one release, and on b14/b17
# at T=25 every start_point >= 1 hit it — reported as "no gradient path", a claim about the
# rival made from our stopwatch. Now each probe gets what is LEFT of the budget, and a probe
# that runs out of it ends the search with the budget named in the reason.
PROBE_BUDGET_FRACTION = 1.0


def probe_budget(run_timeout=DEFAULT_TIMEOUT):
    """Total initialization wall per instance, over all start_point probes."""
    return float(run_timeout) * PROBE_BUDGET_FRACTION


BENCH_DIR = os.path.join(REPO_ROOT, "benchmarks", "circuits", "bench", "seq")

# Their run.py target table (the branches for instances we share; idx into the OUTPUT list).
# ⚠️ THE KEY IS OUR FILENAME STEM, THEIRS IS THAT WITH DOTS REMOVED: their run.py does
# `basename.replace('.bench','').replace('.','')`, so `s208.1.bench` selects their `s2081`
# branch and `s9234.1` their `s92341`. Transcribe against the STRIPPED name or a dotted
# instance silently lands in the else-branch below.
# ⚠️ The final `else` is a REAL target, not an error path — idx [0,15,31,63,-1] with
# values [1,0,1,1,1] — and it is what their own `s15850.1` row runs at, since only the
# undotted `s15850` has a curated branch. It needs >= 64 outputs, so it raises IndexError on
# a small circuit; that is why an unlisted instance is skipped rather than guessed at.
# Gated against their source by tests/test_rival_targets.py, which re-derives this dict.
TARGETS = {
    "s386": ([-1], [1]), "s400": ([-1], [1]), "s526": ([-1], [1]), "s1423": ([-1], [1]),
    "s208.1": ([-1], [1]),                                     # their `s2081`
    "b01": ([-1], [1]), "b03": ([-1], [1]), "b04": ([-1], [1]), "b06": ([-1], [1]),
    "b08": ([-1], [1]), "b11": ([-1], [1]), "b13": ([-1], [1]),
    "b02": ([-1], [0]), "b05": ([-1], [0]), "b07": ([-1], [0]), "b09": ([-1], [0]),
    "b10": ([-2], [0]), "b12": ([-2], [0]),
    # b14/b15/b17 share one branch (target ZEROS). Their absence here silently skipped b15 —
    # an instance their OWN paper's Table II evaluates.
    "b14": ([0, -1], [0, 0]), "b15": ([0, -1], [0, 0]), "b17": ([0, -1], [0, 0]),
    "b18": ([0, -1], [1, 0]), "b19": ([0, -1], [1, 0]),
    "b20": ([0, 7, -1], [1, 1, 1]), "b21": ([0, 7, -1], [1, 1, 1]),
    "b22": ([0, 7, -1], [1, 1, 1]),
    "s820": ([0, -1], [0, 1]), "s832": ([0, -1], [0, 1]), "s1238": ([0, -1], [0, 1]),
    "s1488": ([0, -1], [0, 1]), "s4863": ([0, -1], [0, 1]),
    "s641": ([0, 1], [1, 0]), "s713": ([0, 1], [1, 0]), "s1196": ([0, 1], [1, 0]),
    "s5378": ([0, 1], [1, 0]),
    "s35932": ([0, 15, 31, -1], [1, 1, 1, 0]),
    "s15850.1": ([0, 15, 31, 63, -1], [1, 0, 1, 1, 1]),        # their default branch
    # ...and so is s38584, the circuit their Table II's 8th ISCAS-89 row actually names (their
    # run.py has no branch for it, so it takes the same `else`). Missing this entry would have
    # SILENTLY SKIPPED the instance in table mode — `if name not in TARGETS: continue` — which
    # is exactly how `b15` and `s208.1` disappeared from earlier sweeps, and it would have gone
    # unnoticed because the transcription test only checked circuits with a NAMED branch.
    "s38584": ([0, 15, 31, 63, -1], [1, 0, 1, 1, 1]),          # their default branch
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


def miter_self_check(src, dst, T, seed, n_words=2048):
    """P[MITER=1] on the miter copy must equal the ORIGINAL instance's rare-pin acceptance
    probability: the same check `unroll_to_bench.write_unrolled_miter` runs for the DEMOTIC
    arm, so both harnesses' rare rows carry the same alignment evidence (RIVAL_FAIRNESS_FINDINGS
    §14 — for one release the Recurrent miter was asserted by construction only). Returns
    (p_miter, ok); a failure is a row flag (`pin_check_ok: false`, the `misaligned` bucket,
    verify_artifact FAILS), never a dead harness. Tolerance is 4 sigma of the simulation
    estimate, floored at 1e-5 — at a rare pin both probabilities sit near the floor, so the
    check has power against a WRONG pattern (p_miter 0 vs a p_accept of order 1e-4 and up) and
    against an inverted miter, not against a one-in-a-million miss."""
    import numpy as np
    from gpap.circuits.bench import parse_bench
    from gpap.circuits.simulate import _popcount, _run_bench, _topo_gates, sim_baseline
    mcirc = parse_bench(dst)
    outs = _run_bench(mcirc, _topo_gates(mcirc), T, n_words, rng=np.random.default_rng(3))
    p_miter = _popcount(outs["GPAPMITER"]) / (n_words * 64)
    p_ref = sim_baseline(src, T, seed, target_mode="rare", n_min=n_words * 64,
                         budget_s=0.5, hard_cap_s=5.0)["p_accept"]
    tol = 4 * max((p_ref * (1 - p_ref) / (n_words * 64)) ** 0.5, 1e-5)
    ok = abs(p_miter - p_ref) <= tol or (p_ref == 0 and p_miter == 0)
    return p_miter, bool(ok)


def miter_bench(src, dst, T, seed):
    """Copy `src` with a single MITER output = [outputs at the last cycle == rare pattern],
    the rare pattern taken over ALL outputs at depth T (gpap.circuits.simulate.rare_target).
    Returns (pattern, est_freq, p_miter, pin_ok) — the last two from `miter_self_check`."""
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
    # Wide AND -> bounded fan-in tree, or a 320-output circuit like s35932 emits a 321-literal
    # clause that the packed envelope refuses (see unroll_to_bench.and_tree).
    terms = and_tree(extra, terms, "GPAPAND")
    if len(terms) == 1:
        extra.append(f"GPAPM0 = NOT({terms[0]})")
        extra.append("GPAPMITER = NOT(GPAPM0)")
    else:
        extra.append(f"GPAPMITER = AND({', '.join(terms)})")
    lines = _replace_outputs(lines, ["GPAPMITER"])
    with open(dst, "w") as f:
        f.write("\n".join(lines) + "\n" + "\n".join(extra) + "\n")
    p_miter, ok = miter_self_check(src, dst, T, seed)
    return pattern, est, p_miter, ok


# ---------------------------------------------------------------- the two samplers

def run_recurrent(rival, bench_file, key, T, steps, batch, start_point,
                  timeout=DEFAULT_TIMEOUT, retry=True, lr=None):
    """One run of their sampler; returns dict or {'error': ...}. `key` = the filename stem the
    copy is given (their target table and results csv are keyed on it)."""
    data_dir = os.path.join(rival, "data", "rival_tmp")
    os.makedirs(data_dir, exist_ok=True)
    shutil.copy(bench_file, os.path.join(data_dir, f"{key}.bench"))
    for f in glob.glob(os.path.join(rival, "src", "results", f"sat_{key}_*")):
        os.unlink(f)
    cmd = [sys.executable, os.path.join(rival, "src", "pytorch", "run.py"),
           "-d", f"../data/rival_tmp/{key}.bench", "-l", "--optimizer", "mse",
           # their own run script's value; RIVAL_LR carries the provenance and the tie-break
           "--lr", f"{(lr if lr is not None else resolve_lr('recurrent')[0]):g}e0",
           "--problem_type", "BLIF", "--circuit_type", "seq", "--num_steps", str(steps),
           "--batch_size", str(batch), "--num_clock_cycles", str(T),
           # THEIR seed, passed explicitly so the row can say which initialization was measured
           # (their flags.py defaults to it; we run one seed, on purpose — rival_common.RIVAL_SEED)
           "--seed", str(RIVAL_SEED),
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
    # THEIR filename, not ours: run.py builds `f"{problem_type}_{dataset_str}_{num_steps}"` with
    # `dataset_str = basename.replace('.bench','').replace('.','')` — the DOT IS STRIPPED. Globbing
    # the key verbatim finds nothing for a dotted instance, and "the rival produced no results"
    # is then recorded for a run that in fact completed. It cost `s208.1` its whole table-mode
    # row in the 2026-09-03 campaign — six rungs ran, all six were discarded, and the cell was
    # published as one of that rival's 11 failures. `s15850.1` and `s38584.1` are the same shape.
    hits = glob.glob(os.path.join(rival, "src", "results", f"sat_{key.replace('.', '')}_*"))
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


def probe_start_point(rival, bench_file, key, T, max_sp=10, budget=None,
                      cache_dir=None, run_timeout=DEFAULT_TIMEOUT):
    """Their backward() crashes while the pinned output has no gradient path to any input
    (Moore register latency); walk start_point up until a tiny run survives, inside ONE
    initialization budget (`probe_budget`, = the run timeout).

    Returns `(start_point, reason)`. `start_point` is None when the search did not settle, and
    then `reason` says which of two things happened — both are outcomes of the RIVAL under a
    stated rule, and they are kept apart because they mean different things:

      "no gradient path at any start_point<=N"   their backward() refused every depth the search
                                                 reached. A finding about the rival.
      "did not initialize ... within the Ns      a 1-step probe ran out of the initialization
       initialization budget"                    budget. The search stops there: a larger
                                                 start_point pays the same model build, so it
                                                 cannot be faster.

    CACHED, because it is pure: the answer depends only on the circuit and the unroll depth, yet
    every probe pays a full process launch and model build (~30 s of a Recurrent run's wall is
    build, even at --num_steps 1). A timeout is NOT cached — it may pass on a faster box."""
    budget = probe_budget(run_timeout) if budget is None else float(budget)
    hit = cache_peek(cache_dir, bench_file, _sp_task(T, max_sp))
    if hit is not CACHE_MISS:
        return hit, None
    t0 = time.perf_counter()
    for sp in range(max_sp + 1):
        left = budget - (time.perf_counter() - t0)
        if left <= 0:
            return None, (f"did not initialize: start_point probes 0..{sp - 1} used up the "
                          f"{budget:.0f}s initialization budget")
        r = run_recurrent(rival, bench_file, key, T, steps=1, batch=256, start_point=sp,
                          timeout=left)
        if "error" not in r:
            return _remember_sp(cache_dir, bench_file, T, max_sp, sp), None
        if "does not require grad" not in r["error"]:
            if "timeout" in r["error"]:
                return None, (f"did not initialize: a 1-step start_point probe (sp={sp}) did not "
                              f"complete within the {budget:.0f}s initialization budget")
            return _remember_sp(cache_dir, bench_file, T, max_sp, sp), None
    return None, f"no gradient path at any start_point<={max_sp}"


def _remember_sp(cache_dir, bench_file, T, max_sp, sp):
    """Store a SETTLED probe result. Failures to settle are deliberately not stored: a timeout
    or a reclaimed VM is about the machine, not the circuit."""
    if sp is not None:
        cache_store(cache_dir, bench_file, _sp_task(T, max_sp), sp)
    return sp



def run_recurrent_once(rival, bench_file, key, T, steps, batch, start_point,
                       timeout=DEFAULT_TIMEOUT, lr=None):
    """ONE measured run at the rival's granted batch — the LARGEST ladder rung that completes
    (rival_common.run_at_batch, which also fires this process's one discarded warmup run). The
    rungs tried are attached as `batch_sweep`: their unique/s is near-proportional to the batch
    (s386 T=25: 10k -> 300k is 18.6x, with model_runtime almost flat), so what the rival was
    granted belongs in the row next to its score."""
    r, sweep = run_at_batch(
        lambda b: run_recurrent(rival, bench_file, key, T, steps, b, start_point,
                                timeout=timeout, lr=lr),
        batch, tool="recurrent")
    return dict(r, batch_sweep=sweep) if isinstance(r, dict) else r


def pick_batch(bench_file, T, budget_bytes=None):
    """The batch Recurrent-CircuitSAT is granted: the largest that fits in 0.80 of free VRAM,
    the same share GPAP sizes its own K with (rival_common.fit_batch)."""
    n_gates = sum(1 for line in open(bench_file)
                  if "=" in line and "DFF" not in line and not line.lstrip().startswith("#"))
    return fit_batch(max(n_gates, 1) * T, "recurrent", budget_bytes)


# GPAP's sampling portfolio for our arm. `circuit` is the ADAPTIVE bandit — the shipped default
# for circuit inputs and what every recorded campaign has used — so it stays the default here;
# changing it would move every published rival ratio.
#
# ⚠️ It is an open ASYMMETRY, not a settled choice: every rival runs one fixed configuration and
# so does CMSGen, while the bandit measures configs against each other inside the timed wall and
# reports the winner's rate (RIVAL_FAIRNESS_FINDINGS.md §10, §12). `--portfolio b8` pins a single
# config if a campaign wants to close that. MEASURED on the rival path (T=25, their pins,
# Tseitin, seed 5, 20 s): the bandit picks B=8 on b13/b04/b14 and B=1 on b12, and pinning `b8`
# costs 4.5-8.7% of samples/s (b13 -4.6%, b12 -4.5%, b04 -8.7%, b14 -5.3%) where pinning `b1`
# costs 13.6-31.2% on three of the four.
GPAP_PORTFOLIO = "circuit"


# The GPAP arm's encoder. The rival unrolls the RAW circuit itself, so the encoder is ours to
# choose, and it is the one every other table in the paper runs: abc:fast (`--encoders abc:fast`
# in eval/run_full_eval.sh). It used to be the native Tseitin — the rival path was written
# before the ABC route existed and never moved — which made this table the only one whose
# GPAP number came from a formula 2-10x larger than the paper's, and made b14's GPAP arm a
# 413,451-variable CNF where abc:fast gives 41,938 (docs/findings/RIVAL_FAIRNESS_FINDINGS.md
# §13). There is NO Tseitin fallback: without an `abc` binary the arm is an error row, never a
# silently different encoding under the same label. Recorded in every row as `encoder`.
GPAP_ENCODER = "abc:fast"

# Rows of the timed run's own output kept, to DEDUPLICATE — not to estimate from. This is what
# `--out`/`--max-samples` keep: GPU-projected support bits of the WINNING config, `out_project_s`
# excluded from every rate — so it costs rows x |support| bits of host memory, not the coverage
# run's rows x nv x 72 B (which is what OOM-killed b14 and took 7 instances out of a sweep with
# it). The coverage pass (`n_cov`) is not run on this path at all.
#
# IT USED TO BE 1,000, AND THAT WAS THE WRONG SHAPE OF ANSWER (2026-09-04). Both rivals dedup
# their WHOLE output exactly, off their own clock; we alone estimated a distinct fraction from a
# window and applied it to runs delivering up to 1,048,576 models. The cap is now above any
# delivered count we have ever recorded, so `unique/s` is measured the same way theirs is. Cost
# at the worst cell on record (b13, 1,048,576 models x 250 support bits): 262 MB gathered,
# 32 MB packed, both off the timed clock. If a future run ever delivers more than this, the row
# says so — `unique_rows` is recorded and `unique_est_status` reads `capped-gather`.
UNIQUE_ROWS = 4_000_000
# Below this many kept rows a repeat rate is too noisy to size the solution space from. The
# space estimate now feeds only the `tiny` screen, never a cap on our own rate.
UNIQUE_CAP_MIN_ROWS = 100


def build_arm_cnf(bench_file, T, pin_outputs, target, cnf_path, encoder=GPAP_ENCODER):
    """Encode ONE instance of a rival row to `cnf_path` — the formula BOTH of our arms run.

    The GPAP arm samples it and the CMSGen arm (`rival_common.cmsgen_arm`) reads the same
    file, so the two comparisons a row carries are over one CNF by construction rather than by
    deriving it twice and hoping. `abc:fast` is the campaign's encoder; `tseitin` stays
    reachable for an A/B and nothing else, and there is NO fallback between them (no abc
    binary = an error row, never a silently different encoding under the same label).
    Returns the row's `cnf` provenance: encoder, encode wall, sizes, support width."""
    from gpap.circuits.bench import parse_bench, write_dimacs
    circ = parse_bench(bench_file)
    t_enc = time.perf_counter()
    if encoder == "abc:fast":
        from gpap.circuits.abc_backend import build_cnf_abc     # raises without an abc binary
        b, support, _tgt, labels = build_cnf_abc(circ, bench_file, T, seed=5,
                                                 pin_outputs=pin_outputs, opt="fast",
                                                 target=target)
    elif encoder == "tseitin":
        from gpap.circuits.bench import build_cnf
        b, support, _tgt, labels = build_cnf(circ, T, seed=5, pin_outputs=pin_outputs,
                                             target=target)
    else:
        raise ValueError(f"encoder must be abc:fast or tseitin, got {encoder!r}")
    encode_s = time.perf_counter() - t_enc
    write_dimacs(cnf_path, b, support, labels)
    return {"encoder": encoder, "encode_s": encode_s, "n_vars": b.nv,
            "n_clauses": len(b.clauses), "support": len(support)}


def run_gpap(bench_file, T, pin_outputs, target, seeds, n_time, device, time_limit=60.0,
             portfolio_name=GPAP_PORTFOLIO, encoder=GPAP_ENCODER, unique_rows=UNIQUE_ROWS,
             cnf=None):
    """Convert once (explicit pins, `encoder`), then sample at each seed; report per-seed +
    medians. `unique_per_s_est` = samples/s x the distinct fraction of the timed run's own
    kept output rows (`unique_rows` of them) — no separate coverage run.

    `cnf=(path, info)` is a CNF `build_arm_cnf` already wrote for this row — the harnesses pass
    it so the CMSGen arm can read the very same file afterwards; it is left in place. Without
    it the arm encodes into a temporary file of its own and removes it."""
    import numpy as np
    from gpap.sampling.pipeline import portfolio, sample_file
    if cnf is None:
        with tempfile.NamedTemporaryFile("w", suffix=".cnf", delete=False) as f:
            cnf_path = f.name
        info, own = build_arm_cnf(bench_file, T, pin_outputs, target, cnf_path, encoder), True
    else:
        (cnf_path, info), own = cnf, False
    runs, unsat, unsat_prov = [], None, {}
    try:
        for s in seeds:
            # verify_sound: this row RECORDS `sound_check`, so it has to run the check —
            # otherwise the column is a permanent "off" beside a comment claiming it is the
            # on-device model check. Free to the comparison: the check's wall is excluded from
            # every rate (`verify_sound_s`), so the rival is not credited a millisecond for it.
            g = sample_file(cnf_path, device, n_time=n_time, n_cov=0, out_cap=unique_rows,
                            configs=portfolio(portfolio_name), time_limit=time_limit, seed=s,
                            verify_sound=True)
            # Distinct fraction over the rows the timed run DELIVERED (the (n, S) support-bit
            # matrix `--out` would have written). Exact on large spaces (every row distinct),
            # honest on tiny ones where re-sampling dominates.
            bits = g.get("samples")
            n_coll = 0 if bits is None else len(bits)
            # OFFLINE, AND MEASURABLY SO. The dedup itself runs here, after the sampler has
            # returned, so it is outside every clock the sampler keeps — and it is timed anyway,
            # because "it is off the clock" is a claim and `distinct_s` is its evidence.
            _t0 = time.perf_counter()
            n_distinct = exact_distinct(bits) if n_coll else 0
            distinct_s = time.perf_counter() - _t0
            # The GATHER is the other half, and it is not free at this row count: the projection
            # rides along inside the timed loop and is billed to phases["output"], which
            # `samples_per_s` already subtracts — but `sps_e2e_total` CARRIES it, which was
            # harmless at 1,000 rows and is not at a million. Both rivals deduplicate their whole
            # output OFF their own timer, so ours must come off both of our clocks or the e2e
            # column pays for an instrument the rival is not charged for. Recomputed here rather
            # than in the library, because a real `--out` user genuinely does pay delivery; it is
            # only as a MEASUREMENT that it belongs outside the wall.
            proj_s = float((g.get("phases") or {}).get("output") or 0.0)
            wall_e2e = float(g.get("wall_total_s") or 0.0) - proj_s
            sps_e2e = (g["n_models"] / wall_e2e) if wall_e2e > 0 else None
            # ENCODER-INDEPENDENT soundness on the kept rows: simulate each through the
            # ORIGINAL circuit and require the pinned outputs (the `gpap sample --sim-check`
            # check, on the campaign path). The on-device re-check above verifies the CNF; this
            # verifies the unroll + ABC + Tseitin chain that produced it. Runs only with an
            # explicit target (the harnesses always pass one) and a wire map for every kept
            # column; otherwise "off", never a silent pass.
            sim_bad, sim_rows = None, 0
            cols, wm = g.get("cols"), g.get("wire_map") or {}
            if n_coll and target is not None and cols is not None and all(v in wm for v in cols):
                from gpap.circuits.simulate import simulate_check
                sim_bad = simulate_check(bench_file, T, 5, np.asarray(bits), [wm[v] for v in cols],
                                         pin_outputs=pin_outputs, target=target)
                sim_rows = n_coll
            # KEEP None. `or 0.0` turned "nothing to measure on" into "GPAP produced nothing",
            # and every consumer buckets a zero as an outright loss — b11-c25 at a rare pin
            # delivered models at ~15x the rival's rate and was published as a loss to it.
            # EXACT, over everything the timed run delivered — the same quantity the rivals
            # report. `n_coll == n_models` unless the gather cap bound (recorded), in which case
            # this is a distinct fraction over a prefix and the status says so.
            uniq = (g["samples_per_s"] * n_distinct / n_coll) if n_coll else None
            # TRUNCATION means the GATHER CAP BOUND, and nothing else. Comparing the kept rows
            # against `n_models` looks right and is wrong: on the adaptive path `n_models` counts
            # every config's models while the kept rows are the WINNER's only (one sampler is
            # delivered, never a mixture), so a healthy bandit run trips it by construction. It
            # did on 3 of 82 cells in the 2026-09-04 campaign (b14 1,666 rows of 2,672 models,
            # b18 8,125 of 10,263, s4863 34,272 of 38,073) and every one was a false alarm: the
            # distinct fraction is the winner's, multiplied by the winner's rate, which is
            # coherent. What is NOT coherent is a fraction taken over a capped prefix, so that is
            # what this flags.
            partial = bool(n_coll >= unique_rows)
            # The space estimate survives ONLY to feed the `tiny` screen (paper_tables): a
            # measured repeat rate over the whole run is a far better estimator than the same
            # rate over 1,000 rows. It no longer caps anything — with an exact dedup there is
            # nothing left to correct, and the cap it used to apply was in any case discarded by
            # the median that followed it (the capped seed was the slowest on both cells where
            # it fired, so the row published an uncapped number under a `unique_capped` flag).
            space = (space_from_distinct(n_coll, n_distinct)
                     if n_coll >= UNIQUE_CAP_MIN_ROWS else None)
            runs.append({"seed": s, "sps": g["samples_per_s"], "sps_e2e": sps_e2e,
                         # what the offline dedup cost, and what it would have cost the e2e rate
                         # had it been left inside: the audit trail for the sentence above
                         "distinct_s": distinct_s, "out_project_s": proj_s,
                         "sps_e2e_with_delivery": g.get("sps_e2e_total"),
                         "unique_per_s_est": uniq, "space_est": space,
                         "unique_est_status": ("partial-gather" if partial else
                                               "ok" if uniq is not None else "no-sample-rows"),
                         "unique_rows": n_coll, "unique_distinct": n_distinct,
                         "sim_check_bad": sim_bad, "sim_check_rows": sim_rows,
                         "success": g["success_rate"], "n_models": g["n_models"],
                         # engine provenance per row, same as the CMSGen sweeps: sound_check is
                         # the on-device model check (opt-in), dec/s + impl/s let the zero-search screen
                         # (paper_tables.zero_search) run on rival rows too
                         "sound_check": g.get("sound_check"),
                         "decisions_per_s": g.get("decisions_per_s"),
                         "implications_per_s": g.get("implications_per_s"),
                         # the engine that measured it (packed split / kernel / K / build id),
                         # exactly what a `gpap bench` row records — verify_artifact's
                         # one-engine-per-stage check reads it here too
                         "engine": g.get("engine")})
        # NOTHING AT ALL? Then ask whether there is anything to find. A cell where both
        # samplers come up empty is ambiguous — hard, or empty — and `b01` under Recurrent's own
        # target is the second: its published pattern is unreachable at exactly T=25 (their
        # sampler runs to 50 and pools every depth), so booking a GPAP zero there scores us
        # against a solution that does not exist. Proved-UNSAT cells get their own column.
        # Budgeted, and `None` (unknown) is never read as satisfiable.
        if not any(r["n_models"] for r in runs):
            unsat = unsat_at_depth(cnf_path, provenance=unsat_prov)
    finally:
        if own:
            os.unlink(cnf_path)
    # THE MEDIAN RUN, not a median of each field (2026-09-04). Taking `statistics.median` of
    # every column independently publishes a row NO EXECUTION EVER PRODUCED: on the 2026-09-03
    # campaign 138 field/cell combinations came from a different seed than the row's own `sps`,
    # and Table 3 then divided one seed's implication count by another seed's model count. The
    # row is now ONE run — the one whose sampling rate is the median — and every rate, count and
    # counter in it belongs to that run.
    #
    # The SAFETY verdicts are still folded over ALL seeds, and must be: an unsound row or a
    # failed circuit-level check anywhere in the set cannot be hidden by which run got picked.
    pick, order = median_run(runs, "sps")
    return {"seeds": runs,
            # WHICH sampler produced this row. A fixed config is only a fixed config if the row
            # says which one; `config_breakdown` is per-seed and this is the claim itself.
            "portfolio": portfolio_name,
            # True only when a solver PROVED there is none at this depth; None = not asked, or
            # asked and not resolved inside the budget. paper_tables screens True, never None.
            "unsat_at_T": unsat, "unsat_probe": (unsat_prov or None),
            "adaptive": len(portfolio(portfolio_name)) > 1,
            # WHICH formula: the encoder is a claim about the row as much as the portfolio is.
            "encoder": encoder, "encode_s": info["encode_s"],
            "engine": pick.get("engine"),
            "median_seed": pick["seed"],
            "unique_per_s_est": pick["unique_per_s_est"],
            "space_est": pick.get("space_est"),
            "unique_rows": pick.get("unique_rows"),
            "unique_distinct": pick.get("unique_distinct"),
            "unique_est_status": (pick.get("unique_est_status") or "no-sample-rows"),
            "sps": pick["sps"], "sps_e2e": pick.get("sps_e2e"),
            "success": pick["success"], "n_vars": info["n_vars"], "n_clauses": info["n_clauses"],
            # merge_sound_check, not seed 0's verdict: UNSOUND dominates and "ok" beats "off",
            # so a mixed set of seeds cannot publish the first seed's silence as the answer.
            "sound_check": merge_sound_check(*(sound_verdict(r) for r in runs)),
            # the circuit-level check, folded the same way: BAD dominates, "ok" beats "off"
            "sim_check": ("BAD" if any((r.get("sim_check_bad") or 0) > 0 for r in runs) else
                          "ok" if any(r.get("sim_check_rows") for r in runs) else "off"),
            "sim_check_rows": sum(r.get("sim_check_rows") or 0 for r in runs),
            "sim_check_bad": sum(r.get("sim_check_bad") or 0 for r in runs),
            "decisions_per_s": pick.get("decisions_per_s") or 0.0,
            "implications_per_s": pick.get("implications_per_s") or 0.0,
            # spread over the seeds, so a reader can see how much the median is worth: a cell
            # whose seeds span 3x is not evidence for a 1.06x win (2026-09-03: p90 of max/min
            # was 1.51 and six published wins sat inside their own seed spread).
            "sps_min": order[0]["sps"], "sps_max": order[-1]["sps"],
            "n_models": int(pick["n_models"])}


def _cms_cols(g, cms):
    """The live table's CMSGen columns: its samples/s on the row's CNF, and GPAP/CMSGen."""
    if not cms or "cms_sps" not in cms:
        return f"{'—':>10}{'—':>9}"
    if cms.get("cms_error"):
        return f"{'ERR':>10}{'—':>9}"
    c = cms["cms_sps"] or 0.0
    gs = (g or {}).get("sps")
    gc = "GPAP" if (gs and not c) else (f"{gs / c:,.1f}x" if (gs and c) else "—")
    return f"{c:>10,.0f}{gc:>9}"


# ---------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+",
                    help="instance stems, e.g. b03 s1196 (see TARGETS); `paper` = the 16 their "
                         "own Table II evaluates (rival_common.PAPER_INSTANCES)")
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
    ap.add_argument("--lr", default="auto",
                    help="their learning rate: 'auto' (default) = the value their own run "
                         "script specifies (rival_common.RIVAL_LR); or an explicit number. "
                         "Their paper text names 50 where their script says 15, so the "
                         "tie-break is re-derived by steps_calibration.py --axis lr")
    ap.add_argument("--gpap-cache", default=None,
                    help="directory of cached GPAP arms (default: <output dir>/.gpap_cache). "
                         "GPAP_RIVAL_CACHE=0 disables reuse entirely.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[5, 6, 7])
    ap.add_argument("--gpap-n", type=int, default=50_000)
    ap.add_argument("--portfolio", default=GPAP_PORTFOLIO,
                    help=f"GPAP sampling portfolio for our arm (default {GPAP_PORTFOLIO!r}). "
                         f"A SINGLE-config name (b8/b1/s1) pins the algorithm, which is what "
                         f"every rival and CMSGen run; 'circuit'/'default' turn the adaptive "
                         f"bandit back on and the row then reports an adaptive sampler.")
    ap.add_argument("--target-seed", type=int, default=5, help="rare-pattern selection seed")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                    help="per-run wall cap for the RIVAL subprocess (s); a run that exceeds it "
                         "is an error row, not a zero")
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
    args.names = expand_names(args.names, "rcs")
    if not os.path.isdir(os.path.join(args.rival, "src")):
        sys.exit(f"no rival checkout at {args.rival} (see module docstring)")
    import torch
    from gpap.circuits.simulate import sim_baseline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps, steps_src = resolve_steps("recurrent", args.steps)
    lr, lr_src = resolve_lr("recurrent", args.lr)
    tmp = tempfile.mkdtemp(prefix="rivals_recurrent_")
    mode = "rare(fixed-T pattern, MITER)" if args.rare else "their target table"
    if args.cmsgen and not os.access(args.cmsgen, os.X_OK):
        sys.exit(f"--cmsgen {args.cmsgen}: not an executable (scripts/build_cmsgen.sh)")
    print(f"{len(args.names)} instances | T={args.cycles} steps={steps} ({steps_src}) | mode: {mode} | "
          f"GPAP median of seeds {args.seeds} | CMSGen arm: "
          f"{'on (same CNF, --nowrite --maxtime)' if args.cmsgen else 'OFF'}")
    hdr = (f"{'instance':<10}{'circ':>6}{'pin':>12}{'p_acc':>10}{'RCS uniq':>11}{'@T':>9}"
           f"{'t':>7}{'RCS u/s':>10}{'GPAP u/s~':>11}{'succ%':>7}{'ratio':>10}"
           f"{'CMS/s':>10}{'G/C':>9}")
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
    # EVERY instance that produces no row is tracked, so a stage cannot quietly measure less
    # than it was asked to. Without this, `b19`-`b22` and `s35932` vanished from a 29-instance
    # sweep and the only evidence was one line each in a log nobody reads (the same failure the
    # DEMOTIC harness had). Reasons are reported at the end; a stage that measured NOTHING exits
    # non-zero so the driver sees it.
    skipped = []
    for name, variant in ((n, v) for n in args.names for v in args.circuit_variants):
        if (name, variant) in done:
            continue
        src0 = os.path.join(args.bench_dir, f"{name}.bench")
        if not os.path.isfile(src0):
            print(f"{name:<10}  no such bench file ({src0}), skipped")
            skipped.append((name, "no bench file"))
            continue
        # Checked BEFORE anything is built: an oversized Tseitin unroll is OOM-killed mid-build,
        # which writes no row and silently truncates every instance after it (see
        # oversize_reason). It gates OUR arm only, and only under the Tseitin encoder — with
        # abc:fast (the default) the guard passes everything: every seq instance built on the
        # 13 GB L4 VM. The rival unrolls the .bench its own way and always gets its run.
        too_big = oversize_reason(src0, args.cycles, encoder=GPAP_ENCODER)
        if too_big:
            print(f"{name:<10}  GPAP arm not attempted (oversize): {too_big[:70]}", flush=True)
        # BOTH sides get the same circuit (eval/rivals/rival_common.py): with 'abc' the rival is handed
        # the ABC-optimized .bench too, gated by a bit-exact simulation equivalence check.
        src, vinfo = prepare_variant(src0, variant, tmp, cycles=args.cycles,
                                     seed=args.target_seed,
                                     target_mode="rare" if args.rare else "random")
        if vinfo.get("error"):
            # A ROW, not a silent skip. `prepare_variant` fails closed (ABC missing, ABC crashed,
            # or the equivalence check refused the copy) and this used to `continue`, so the
            # instance vanished from the sweep with one line in a log. Every other failure mode
            # here records what happened; so does this one.
            print(f"{name:<10}{variant:>6}  circuit variant unusable: {vinfo['error'][:60]}",
                  flush=True)
            out.append({"name": name, "variant": variant,
                        "mode": "rare" if args.rare else "table", "circuit": vinfo,
                        "rcs": {"error": f"circuit variant unusable: {vinfo['error'][:160]}"},
                        "gpap": {"error": "not attempted: no usable circuit for this variant"}})
            skipped.append((name, f"variant unusable: {vinfo['error'][:40]}"))
            continue
        if args.rare:
            rival_file = os.path.join(tmp, f"{name}_{variant}_rare.bench")
            pattern, est, p_miter, pin_ok = miter_bench(src, rival_file, args.cycles,
                                                        args.target_seed)
            key = RARE_ALIAS                       # their table: [-1]=1 == MITER=1
            gpap_file, pin_n = rival_file, None    # single MITER output, pin all(=1)
            gpap_target = {"GPAPMITER": True}
            pin_desc = f"rare|{len(pattern)}o"
            if not pin_ok:
                print(f"{name:<10}{variant:>6}  MITER CHECK FAILED — p(MITER=1)={p_miter:.3e} "
                      f"differs from the instance's p_accept; row flagged", flush=True)
        else:
            if name not in TARGETS:
                print(f"{name:<10}  not in their target table, skipped")
                skipped.append((name, "not in their target table"))
                continue
            idx, vals = TARGETS[name]
            key = name                             # untouched copy hits their table branch
            rival_file = src
            gpap_file = os.path.join(tmp, f"{name}_{variant}_reord.bench")
            picked = reorder_bench(src, gpap_file, idx)
            gpap_target = {w: bool(v) for w, v in zip(picked, vals)}
            pin_n = len(idx)
            pin_desc = ",".join(f"[{i}]={v}" for i, v in zip(idx, vals))
            # Their index table on the ORIGINAL file and our reordered prefix must accept
            # bit-identical input sequences — the same per-row check the DEMOTIC harness runs.
            p_miter = None
            pin_ok = native_pin_check(src, gpap_file, idx, vals, pin_n, args.cycles)
            if not pin_ok:
                print(f"{name:<10}{variant:>6}  PIN CHECK FAILED — reordered pins do not match "
                      f"their index table; row flagged", flush=True)
        sim = sim_baseline(gpap_file, args.cycles, args.target_seed, pin_outputs=pin_n,
                           target=gpap_target)
        sp, why = probe_start_point(args.rival, rival_file, key, args.cycles,
                                    cache_dir=cache_dir,                   # bounded, see above
                                    run_timeout=args.timeout)
        if sp is None:
            # `why` is one of two rival outcomes under a stated rule (no gradient path, or did
            # not initialize within the budget); it is recorded verbatim, never as a zero.
            rcs = {"error": why}
        else:
            batch = pick_batch(rival_file, args.cycles)
            rcs = run_recurrent_once(args.rival, rival_file, key, args.cycles, steps,
                                     batch, sp, timeout=args.timeout, lr=lr)
            rcs["steps_source"] = steps_src
            rcs["lr"], rcs["lr_source"] = lr, lr_src
            rcs["seed"], rcs["seed_source"] = RIVAL_SEED, "their flags.py default; one run"
        task = {"harness": "recurrent", "mode": "rare" if args.rare else "table",
                "cycles": args.cycles, "pin_outputs": pin_n, "target": gpap_target,
                "seeds": args.seeds, "n_time": args.gpap_n,
                "time_limit": args.gpap_time_limit, "portfolio": args.portfolio,
                "encoder": GPAP_ENCODER, "unique_rows": UNIQUE_ROWS}
        # ONE BAD INSTANCE MUST NOT TRUNCATE THE SWEEP. `b19` Tseitin-unrolls to 2N=19,182,702,
        # past the 8|24 envelope's 16,777,215, and the raised EnvelopeError killed the harness —
        # so b20/b21/b22/s35932, all alphabetically after it, were never attempted and the stage
        # simply ended 4 instances short with nothing marking the gap. Recorded as a GPAP-side
        # error row (its own bucket in paper_tables: an envelope refusal is NOT "GPAP produced
        # nothing"), exactly as a rival crash is recorded on their side.
        # OUR ARM RUNS EVEN WHEN THE RIVAL FAILED. There is no ratio in such a row — `rival_cells`
        # buckets it as `error` before it reads our side — but "the rival has no gradient path
        # here and GPAP delivers N models/s on the same circuit" is a capability contrast worth
        # recording, and it is what the sweep is asked to cover: every b*/s* on all three tools.
        #
        # It was skipped for one release because it was the half that BLEW UP: b14's table-mode
        # arm peaked at 43.5 GB and was OOM-killed on a 13 GB Colab VM *and* on a 47 GB box,
        # taking the 8 instances after it out of the sweep. That was never about the rival — it
        # was the COVERAGE RUN (`n_cov`), which reconstructs rows x nv x 72 B on the host (b14
        # through the Tseitin encoding this path used then is 413,451 vars: 2048 rows = 61 GB).
        # The arm no longer runs a coverage pass at all (the distinct fraction comes from the
        # timed run's own kept output rows, see `run_gpap`) and encodes with abc:fast, so there
        # is nothing left to dodge. Set GPAP_RIVAL_SKIP_ON_ERROR=1 to restore the skip.
        skip_on_err = os.environ.get("GPAP_RIVAL_SKIP_ON_ERROR", "0") == "1"
        # ONE CNF PER ROW, both of our arms on it. Built here rather than inside run_gpap so the
        # CMSGen arm reads the very file the GPAP arm sampled — the converged protocol's third
        # tool (RIVAL_FAIRNESS_FINDINGS.md §14) — and built even on a GPAP-cache hit, since it is
        # what CMSGen runs on. An encode failure (no abc binary) is a GPAP-side error row.
        cnf_path, cnf_info, cnf_err = None, None, None
        if not too_big:
            cnf_path = os.path.join(tmp, f"{name}_{variant}_{'rare' if args.rare else 'table'}.cnf")
            try:
                cnf_info = build_arm_cnf(gpap_file, args.cycles, pin_n, gpap_target, cnf_path)
            except Exception as e:                          # noqa: BLE001 — one cell, not the run
                cnf_err = f"{type(e).__name__}: {str(e)[:200]}"
                print(f"{name:<10}{variant:>6}  encode failed: {cnf_err[:80]}", flush=True)
        if too_big:
            g, hit = {"error": f"oversize: {too_big}"}, False
        elif cnf_err:
            g, hit = {"error": f"encode failed: {cnf_err}"}, False
        elif rcs.get("error") and skip_on_err:
            g, hit = {"skipped": "rival failed; GPAP arm not run (unused in an error row) — "
                                 "see the CMSGen table for GPAP on this instance"}, False
        else:
            try:
                g, hit = cached_gpap(cache_dir, gpap_file, task,
                                     lambda: run_gpap(gpap_file, args.cycles, pin_n, gpap_target,
                                                      args.seeds, args.gpap_n, device,
                                                      time_limit=args.gpap_time_limit,
                                                      portfolio_name=args.portfolio,
                                                      cnf=(cnf_path, cnf_info)))
            except Exception as e:                          # noqa: BLE001 — one cell, not the run
                print(f"{name:<10}{variant:>6}  GPAP arm failed: {type(e).__name__}: "
                      f"{str(e)[:80]}", flush=True)
                g, hit = {"error": f"{type(e).__name__}: {str(e)[:200]}"}, False
        if hit:
            print(f"{name:<10}{variant:>6}  [gpap-cache hit — arm not re-run]", flush=True)
        # The CMSGen arm: the same CNF file, the same n and sampling budget as the GPAP arm.
        # Not cached (a `--resume` skips finished rows; only `--retry-errors` re-pays it).
        cms = (cmsgen_arm(cnf_path, args.cmsgen, args.gpap_n, args.gpap_time_limit, tmp,
                          seed=args.cmsgen_seed) if (args.cmsgen and cnf_info) else {})
        if cnf_path and os.path.exists(cnf_path):
            os.remove(cnf_path)                 # a b19-class CNF is hundreds of MB; not kept
        row = {"name": name, "mode": "rare" if args.rare else "table", "pin": pin_desc,
               "variant": variant, "circuit": vinfo, "cycles": args.cycles,
               # constraint alignment, verified per row: table mode = bit-identical acceptance
               # words (native_pin_check); rare mode = the miter's P[MITER=1] equals p_accept
               "pin_check_ok": pin_ok, "p_miter": p_miter,
               # WAS THIS ARM MEASURED IN THIS RUN, OR SERVED FROM THE CACHE? Printed to the log
               # since the cache existed, but never written to the row — so an artifact evaluator
               # reading the JSON alone could not tell a freshly measured cell from one restored
               # from a cache keyed on an older build.
               "gpap_cache_hit": bool(hit),
               "p_accept": sim["p_accept"], "p_accept_lt": sim.get("p_accept_lt"),
               "rival_commit": rival_commit(args.rival), "rcs": rcs, "gpap": g,
               # the ONE formula both of our arms ran (encoder, sizes, support width); `support`
               # at the top level is what `paper_tables.tiny_space` reads, as on a bench row
               "cnf": cnf_info, "support": (cnf_info or {}).get("support"),
               **cms, **host_provenance(device),
               "vram_budget_bytes": rival_vram_budget(),
               # HOST ram, recorded because it decides what this machine could attempt at all:
               # `oversize_reason`'s second term refuses an instance whose measured-fit peak
               # exceeds a share of it, so a refusal row is only interpretable next to this.
               "host_mem_avail_bytes": _available_bytes()}
        out.append(row)
        if args.output:      # checkpoint after every instance: this stage runs for HOURS, and a
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)   # death at 90% used
            json.dump(out, open(args.output, "w"), indent=1)                  # to leave nothing
        pa = sim["p_accept"] if sim["p_accept"] else sim.get("p_accept_lt", 0.0)
        pa_s = f"{pa:.2e}" if sim["p_accept"] else f"<{pa:.0e}"
        # None = the timed run kept no output row to measure on, i.e. NOT MEASURED. It is not
        # a zero: `n_models` says whether GPAP produced anything.
        uq = g.get("unique_per_s_est")
        uq_s = f"{uq:,.0f}" if uq is not None else "n/a"
        if "error" in rcs:
            # `g` may hold no measurement at all now (the arm is skipped when the rival has
            # already failed), so every field read here has to tolerate that.
            succ_s = (f"{100 * g['success']:.0f}%" if g.get("success") is not None
                      else "arm not run")
            print(f"{name:<10}{variant:>6}{pin_desc:>12}{pa_s:>10}  RCS FAILED: "
                  f"{rcs['error'][:44]} | GPAP u/s~={uq_s} succ={succ_s} |{_cms_cols(g, cms)}",
                  flush=True)
            continue
        if "error" in g:
            print(f"{name:<10}{variant:>6}{pin_desc:>12}{pa_s:>10}{rcs['unique']:>11,}"
                  f"  GPAP FAILED: {g['error'][:50]} |{_cms_cols(g, cms)}", flush=True)
            continue
        ratio = (None if uq is None else
                 (uq / rcs["unique_per_s"] if rcs["unique_per_s"] else float("inf")))
        rt = ("      n/a" if ratio is None else            # rival silent -> no ratio,
              "     GPAP" if ratio == float("inf") else    # an outright win, not "inf"
              f"{ratio:>9.1f}x")
        at_t = f"{rcs['unique_at_T']:,}" if rcs.get("unique_at_T") is not None else "—"
        print(f"{name:<10}{variant:>6}{pin_desc:>12}{pa_s:>10}{rcs['unique']:>11,}{at_t:>9}"
              f"{rcs['runtime_s']:>6.1f}s{rcs['unique_per_s']:>10,.0f}"
              f"{uq_s:>11}{100 * g['success']:>6.0f}%{rt}{_cms_cols(g, cms)}",
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
    if skipped:
        print(f"\n[warn] {len(skipped)} instance(s) produced NO row:")
        for n, why in skipped:
            print(f"    {n:<12} {why}")
    if skipped and not out:
        sys.exit(f"[FATAL] every instance was skipped ({len(skipped)}) — the arm measured "
                 f"nothing. See the reasons above.")


if __name__ == "__main__":
    main()
