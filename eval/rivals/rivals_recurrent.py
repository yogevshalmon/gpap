#!/usr/bin/env python3
"""Head-to-head: Recurrent-CircuitSAT-Sampler (MLCAD'25, arXiv 2502.21226 — the Berkeley
gradient family's SEQUENTIAL-circuit member) vs GPAP, same .bench sources, same GPU.

Their protocol, replicated exactly: a per-filename target table pinning 1-2 output BITS (copied
into TARGETS below), reset-0 registers, solutions counted UNIQUE and pooled over ALL unroll
depths start_point..T, `model_runtime` = the gradient loop only (their timer excludes model build
and the solution re-check). Both are generous to them and kept as-is; we additionally parse their
per-depth stdout for the depth-T-only count (`uniq@T`), comparable to GPAP's fixed-depth CNF.

EMISSION SYMMETRY, the one asymmetry running the OTHER way and deliberately NOT corrected: this
rival copies its candidate assignments to host INSIDE the loop its timer wraps, so its published
rate is charged for delivery (bounded at ~2.2% of model_runtime, measured), while every other
tool here is timed on generation alone. Leaving it in keeps their number as they publish it;
re-derive with eval/rivals/check_emission_symmetry.py.

GPAP side: the identical constraint via an output-reordered .bench (their target_idx outputs
first, then build_cnf pin_outputs=len(idx) with the same values, pinned at cycle T), sampled at
MEDIAN OF 3 SEEDS. `--rare` is our adversarial fixed-T target: the bench gets a comparator cone
(MITER = AND of per-output [X]NOR terms vs the rare pattern) as its ONLY output, named b13.bench
so their FILENAME-keyed table pins [-1]=1 = MITER=1.

Setup (once): eval/rivals/rivals_setup.sh clones the pinned upstream commit. Moore pipelines have
no input->output gradient path until the unroll reaches the register distance, so start_point is
probed upward until backward() stops crashing.
"""
import argparse
import ast
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # eval/<sub>/ -> repo
sys.path.insert(0, REPO_ROOT)

from eval.rivals.rival_common import (BATCH_LADDER, CACHE_MISS, VARIANTS, cache_peek,  # noqa: E402
                                      cache_store, cached_gpap, cmsgen_arm, fit_batch,
                                      _available_bytes, expand_names, host_provenance,
                                      native_pin_check, oversize_reason, prepare_variant,
                                      RIVAL_SEED, exact_distinct, median_run, resolve_lr,
                                      space_from_distinct, unsat_at_depth,
                                      resolve_steps, rival_commit, rival_vram_budget,
                                      run_at_batch, variant_note, grant_source, load_grants,
                                      lookup_grant, pinned_ladder)
from eval.rivals.unroll_to_bench import and_tree                      # noqa: E402

from gpap import paths
from gpap.sampling.pipeline import merge_sound_check, sound_verdict

RIVAL_DEFAULT = str(paths.src_dir() / "Recurrent-CircuitSAT-Sampler")
# Per-SUBPROCESS wall cap for the rival, 5x the sampling budget GPAP gets: GENEROUS and not
# symmetric, since the rival's wall includes the PyTorch model build its OWN metric excludes.
DEFAULT_TIMEOUT = 300.0
# INITIALIZATION budget: the whole start_point search — every probe's model build included — must
# settle inside it, else the instance is an error row ("did not initialize"); each probe gets what
# is LEFT. A per-probe cap smaller than the run's turns a slow instance into a false "no gradient
# path". Its own number, not tied to the run timeout (RIVAL_FAIRNESS_FINDINGS.md §11), counted in
# the rival's e2e figure the way our preparation is; its steady-state metric is untouched.
INIT_BUDGET_S = 1200.0
# The cold-start warmup (rival_common.warmup_rival) is discarded, so it is killed after this long.
WARMUP_TIMEOUT_S = 60.0
PROBE_BUDGET_FRACTION = 1.0


def probe_budget(run_timeout=DEFAULT_TIMEOUT):
    """Legacy: the budget as a fraction of the run timeout; the campaign passes --init-budget."""
    return float(run_timeout) * PROBE_BUDGET_FRACTION


BENCH_DIR = os.path.join(REPO_ROOT, "benchmarks", "circuits", "bench", "seq")

# Their run.py target table (the branches for instances we share; idx into the OUTPUT list).
# THE KEY IS OUR FILENAME STEM, THEIRS IS THAT WITH DOTS REMOVED (`s208.1` selects their `s2081`),
# so a mis-transcribed name lands in the else-branch — which is a REAL target needing >= 64
# outputs, and why an unlisted instance is skipped rather than guessed at. Gated against their
# source by tests/test_rival_targets.py, which re-derives this dict.
TARGETS = {
    "s386": ([-1], [1]), "s400": ([-1], [1]), "s526": ([-1], [1]), "s1423": ([-1], [1]),
    "s208.1": ([-1], [1]),                                     # their `s2081`
    "b01": ([-1], [1]), "b03": ([-1], [1]), "b04": ([-1], [1]), "b06": ([-1], [1]),
    "b08": ([-1], [1]), "b11": ([-1], [1]), "b13": ([-1], [1]),
    "b02": ([-1], [0]), "b05": ([-1], [0]), "b07": ([-1], [0]), "b09": ([-1], [0]),
    "b10": ([-2], [0]), "b12": ([-2], [0]),
    # b14/b15/b17 share one branch (target ZEROS); omitting any silently skips an instance.
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
    # ...and so is s38584, their Table II's 8th ISCAS-89 row: a missing entry SILENTLY SKIPS the
    # instance in table mode.
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
    """Drop every OUTPUT(...) line, emitting the new list where the first stood (file order =
    output order for both parsers)."""
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
    probability — the same check the DEMOTIC arm runs, so both harnesses' rare rows carry the
    same alignment evidence (docs/findings/RIVAL_FAIRNESS_FINDINGS.md §14). Returns (p_miter, ok);
    a failure is a row flag (the `misaligned` bucket), never a dead harness. Tolerance is 4 sigma
    floored at 1e-5: power against a WRONG or inverted pattern, not against a rare miss."""
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
    """Copy `src` with a single MITER output = [outputs at the last cycle == rare pattern], the
    pattern over ALL outputs at depth T. Returns (pattern, est_freq, p_miter, pin_ok)."""
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
    # Wide AND -> bounded fan-in tree, or a many-output circuit emits a clause the envelope refuses.
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
    """One run of their sampler; dict or {'error': ...}. `key` = the stem the copy is given."""
    data_dir = os.path.join(rival, "data", "rival_tmp")
    os.makedirs(data_dir, exist_ok=True)
    shutil.copy(bench_file, os.path.join(data_dir, f"{key}.bench"))
    # THE DOT IS STRIPPED in their filename, so the clean-up must strip it too.
    for f in glob.glob(os.path.join(rival, "src", "results", f"sat_{key.replace('.', '')}_*")):
        os.unlink(f)
    cmd = [sys.executable, os.path.join(rival, "src", "pytorch", "run.py"),
           "-d", f"../data/rival_tmp/{key}.bench", "-l", "--optimizer", "mse",
           # their own run script's value; RIVAL_LR carries the provenance and the tie-break
           "--lr", f"{(lr if lr is not None else resolve_lr('recurrent')[0]):g}e0",
           "--problem_type", "BLIF", "--circuit_type", "seq", "--num_steps", str(steps),
           "--batch_size", str(batch), "--num_clock_cycles", str(T),
           # THEIR seed, passed explicitly so the row can say which initialization was measured
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
        # A rival that dies with NOTHING on either stream must not raise here: a rung that does
        # not COMPLETE is a refusal, so the row records the exit status and the walk steps down.
        lines = (p.stderr or p.stdout or "").strip().splitlines()
        if lines:
            err = lines[-1][:200]
        else:
            try:
                how = signal.Signals(-p.returncode).name if p.returncode < 0 else f"code {p.returncode}"
            except ValueError:
                how = f"code {p.returncode}"
            err = f"exit {how} with no output (killed by the host, or a crash that wrote nothing)"
        # transient WSL2 driver hiccup between back-to-back CUDA processes — settle and retry
        if "device not ready" in err and retry:
            time.sleep(10)
            return run_recurrent(rival, bench_file, key, T, steps, batch, start_point,
                                 timeout=timeout, retry=False)
        return {"error": err, "wall_s": wall}
    # THEIR filename, not ours: the DOT IS STRIPPED, so globbing the key verbatim would record
    # "no results" for a run that completed.
    hits = glob.glob(os.path.join(rival, "src", "results", f"sat_{key.replace('.', '')}_*"))
    if not hits:
        return {"error": "no results csv", "wall_s": wall}
    # If more than one results file survives, the only acceptable pick is the rung this call ran.
    if len(hits) > 1:
        hits = [h for h in hits if h.endswith(f"_{batch}.csv")]
        if len(hits) != 1:
            return {"error": f"ambiguous results csv ({len(hits)} candidates for batch {batch})",
                    "wall_s": wall}
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


def settled_error(err, init_budget, run_timeout):
    """Is this rival error row a FINISHED measurement under the current protocol? True for a
    failure to initialize inside the CURRENT budget, a timeout at the CURRENT timeout, and "no
    gradient path"; False for another budget or an environmental error (`--retry-errors`)."""
    err = str(err or "")
    if err.startswith("no gradient path"):
        return True
    if f"{float(init_budget):.0f}s initialization budget" in err:
        return True
    if err.startswith("timeout ") and err.rstrip("s").split()[-1] in (f"{float(run_timeout):g}", f"{float(run_timeout)}"):
        return True
    return False


def _sp_task(T, max_sp):
    return {"probe": "start_point", "cycles": T, "max_sp": max_sp}


def probe_start_point(rival, bench_file, key, T, max_sp=10, budget=None,
                      cache_dir=None, run_timeout=DEFAULT_TIMEOUT):
    """Their backward() crashes while the pinned output has no gradient path to any input (Moore
    register latency); walk start_point up until a tiny run survives, inside ONE budget.

    Returns `(start_point, reason)`; `start_point` is None when the search did not settle, and
    `reason` separates two outcomes of the RIVAL under a stated rule — "no gradient path at any
    start_point<=N" and "did not initialize ... within the Ns initialization budget" (a larger
    start_point pays the same model build, so it cannot be faster).

    CACHED, because it is pure: the answer depends only on the circuit and the depth, yet every
    probe pays a process launch and model build. A FAILURE is stored WITH the budget it failed at
    and honoured only at that budget or smaller — a stuck instance may pass on a longer clock."""
    budget = probe_budget(run_timeout) if budget is None else float(budget)
    hit = cache_peek(cache_dir, bench_file, _sp_task(T, max_sp))
    if isinstance(hit, dict) and "failed_at_budget" in hit:
        if float(hit["failed_at_budget"]) >= budget:
            return None, hit.get("reason", "did not initialize") + " (cached)"
    elif hit is not CACHE_MISS:
        return hit, None
    t0 = time.perf_counter()
    for sp in range(max_sp + 1):
        left = budget - (time.perf_counter() - t0)
        if left <= 0:
            return None, _remember_failure(
                cache_dir, bench_file, T, max_sp, budget,
                f"did not initialize: start_point probes 0..{sp - 1} used up the "
                f"{budget:.0f}s initialization budget")
        r = run_recurrent(rival, bench_file, key, T, steps=1, batch=256, start_point=sp,
                          timeout=left)
        if "error" not in r:
            return _remember_sp(cache_dir, bench_file, T, max_sp, sp), None
        if "does not require grad" not in r["error"]:
            if "timeout" in r["error"]:
                # PHRASE IT AT THE SEARCH LEVEL, not the probe: the search had the whole budget
                # while probe sp got what its predecessors left. The substring
                # `{budget}s initialization budget` is kept so `settled_error` still matches.
                return None, _remember_failure(
                    cache_dir, bench_file, T, max_sp, budget,
                    f"did not initialize: the start_point search reached sp={sp} and exhausted "
                    f"its {budget:.0f}s initialization budget ({left:.0f}s remained for that probe)")
            return _remember_sp(cache_dir, bench_file, T, max_sp, sp), None
    return None, f"no gradient path at any start_point<={max_sp}"


def _remember_sp(cache_dir, bench_file, T, max_sp, sp):
    """Store a SETTLED probe result (an int)."""
    if sp is not None:
        cache_store(cache_dir, bench_file, _sp_task(T, max_sp), sp)
    return sp


def _remember_failure(cache_dir, bench_file, T, max_sp, budget, reason):
    """Store a failure to settle, tagged with the budget it exhausted (see probe_start_point)."""
    cache_store(cache_dir, bench_file, _sp_task(T, max_sp),
                {"failed_at_budget": float(budget), "reason": reason})
    return reason



def _oom_task(T):
    return {"probe": "oom_ceiling", "cycles": T}


def _refusal_ceiling(hit, run_timeout=None):
    """The rung below which cached refusals say this walk must start: an OOM or a silent kill
    holds on this card at any clock, a timeout only at the clock it was seen at."""
    if not isinstance(hit, dict):
        return None
    cands = [hit.get("oom_min"), hit.get("killed_min")]
    if run_timeout is not None and hit.get("timeout_s") == float(run_timeout):
        cands.append(hit.get("timeout_min"))
    cands = [c for c in cands if c]
    return min(cands) if cands else None


def known_ladder(cache_dir, bench_file, T, ladder=BATCH_LADDER, run_timeout=None):
    """The batch ladder minus the rungs a previous walk on THIS card already saw refused: an
    out-of-memory rung is deterministic for a circuit, depth and card, and skipping a KNOWN
    failure changes nothing about the grant. A timeout is honoured only at its own clock."""
    ceiling = _refusal_ceiling(cache_peek(cache_dir, bench_file, _oom_task(T)), run_timeout)
    if ceiling:
        trimmed = tuple(b for b in ladder if b < ceiling)
        return trimmed or ladder[:1], ceiling
    return tuple(ladder), None


def known_spent(cache_dir, bench_file, T):
    """Search time an interrupted walk already spent on refused rungs, charged across VM
    lifetimes as one walk."""
    hit = cache_peek(cache_dir, bench_file, _oom_task(T))
    return float(hit.get("spent_s") or 0.0) if isinstance(hit, dict) else 0.0


def remember_refusal(cache_dir, bench_file, T, batch, error, wall_s, run_timeout):
    """Store one refused rung THE MOMENT IT IS SEEN: `remember_oom_ceiling` writes the ceiling
    only when the walk ENDS, and a walk whose rungs each cost minutes of model build can outlast
    a VM and restart from the ladder top with nothing remembered. The RULE is unchanged; only the
    bookkeeping moved. Tightens, never loosens; an unclassifiable error is not a ceiling."""
    if not cache_dir:
        return None
    hit = cache_peek(cache_dir, bench_file, _oom_task(T))
    rec = dict(hit) if isinstance(hit, dict) else {}
    e = str(error or "").lower()
    if "outofmemory" in e:
        rec["oom_min"] = min(batch, rec.get("oom_min") or batch)
    elif "with no output" in e and "sigkill" in e:
        rec["killed_min"] = min(batch, rec.get("killed_min") or batch)
    elif e.startswith("timeout"):
        if rec.get("timeout_s") == float(run_timeout) and rec.get("timeout_min"):
            rec["timeout_min"] = min(batch, rec["timeout_min"])
        else:
            rec["timeout_min"], rec["timeout_s"] = batch, float(run_timeout)
    else:
        return None
    rec["spent_s"] = float(rec.get("spent_s") or 0.0) + float(wall_s or 0.0)
    cache_store(cache_dir, bench_file, _oom_task(T), rec)
    return rec


def remember_oom_ceiling(cache_dir, bench_file, T, sweep, prior=None):
    """Store the smallest rung the walk saw OOM (tightening, never loosening) and drop the
    search time `remember_refusal` accumulated for it."""
    ooms = [x["batch"] for x in (sweep or []) if "outofmemory" in str(x.get("error") or "").lower()]
    hit = cache_peek(cache_dir, bench_file, _oom_task(T))
    rec = dict(hit) if isinstance(hit, dict) else {}
    had_spent = rec.pop("spent_s", None) is not None
    if not ooms:
        if had_spent:
            cache_store(cache_dir, bench_file, _oom_task(T), rec)
        return prior
    # `prior` may be a timeout or kill ceiling (known_ladder); only an OOM tightens oom_min
    known = rec.get("oom_min") or (prior if not rec else None)
    rec["oom_min"] = min(ooms + ([known] if known else []))
    cache_store(cache_dir, bench_file, _oom_task(T), rec)
    return rec["oom_min"]


def run_recurrent_once(rival, bench_file, key, T, steps, batch, start_point,
                       timeout=DEFAULT_TIMEOUT, lr=None, cache_dir=None, grant=None):
    """ONE measured run at the rival's granted batch — the LARGEST ladder rung that completes
    (rival_common.run_at_batch, which also fires this process's one discarded warmup). The rungs
    tried are attached as `batch_sweep`, since unique/s is near-proportional to the batch. A
    RECORDED grant pins the walk to its rung; known-OOM rungs are skipped (`known_ladder`)."""
    ladder, prior = known_ladder(cache_dir, bench_file, T, run_timeout=timeout)
    spent0 = known_spent(cache_dir, bench_file, T)
    pinned = grant["batch"] if grant else None
    if pinned:
        ladder = pinned_ladder(ladder, pinned)

    def rung(b):
        t0 = time.perf_counter()
        r = run_recurrent(rival, bench_file, key, T, steps, b, start_point, timeout=timeout, lr=lr)
        if "error" in r:            # remembered NOW, not when the walk ends (remember_refusal)
            remember_refusal(cache_dir, bench_file, T, b, r["error"], time.perf_counter() - t0, timeout)
        return r

    def warm(b):
        # a DISCARDED cold-start run, killed after WARMUP_TIMEOUT_S; the cap moves no number.
        return run_recurrent(rival, bench_file, key, T, steps, b, start_point,
                             timeout=min(WARMUP_TIMEOUT_S, timeout), lr=lr)

    r, sweep = run_at_batch(rung, batch, ladder=ladder, tool="recurrent", spent=spent0,
                            warmup_run=warm)
    remember_oom_ceiling(cache_dir, bench_file, T, sweep, prior)
    if isinstance(r, dict):
        r = dict(r, batch_sweep=sweep)
        if prior:
            r["oom_ceiling_cached"] = prior
        if spent0:
            r["ladder_spent_cached_s"] = round(spent0, 1)
        r["grant_source"] = grant_source(pinned, r, grant["source"] if grant else None)
    return r


def pick_batch(bench_file, T, budget_bytes=None):
    """The batch this rival is granted: the largest fitting 0.80 of free VRAM (fit_batch)."""
    n_gates = sum(1 for line in open(bench_file)
                  if "=" in line and "DFF" not in line and not line.lstrip().startswith("#"))
    return fit_batch(max(n_gates, 1) * T, "recurrent", budget_bytes)


# GPAP's sampling portfolio for our arm. `circuit` is the ADAPTIVE bandit, the shipped default for
# circuit inputs and what every recorded campaign used, so it stays the default. It is an open
# ASYMMETRY: every rival and CMSGen run one fixed configuration while the bandit reports its
# winner's rate (RIVAL_FAIRNESS_FINDINGS.md §10, §12); `--portfolio b8` pins a single config.
GPAP_PORTFOLIO = "circuit"


# The GPAP arm's encoder: the rival unrolls the RAW circuit itself, so the encoder is ours to
# choose, and it is the one every other table runs (RIVAL_FAIRNESS_FINDINGS.md §13). NO Tseitin
# fallback — without an `abc` binary the arm is an error row, never a different encoding under
# the same label.
GPAP_ENCODER = "abc:fast"

# Rows of the timed run's own output kept, to DEDUPLICATE — not to estimate from: GPU-projected
# support bits of the WINNING config, `out_project_s` excluded from every rate (`n_cov` is not run
# here at all). The cap is above any delivered count on record, so `unique/s` is measured the way
# both rivals measure theirs; a run that exceeds it says `unique_est_status: capped-gather`.
UNIQUE_ROWS = 4_000_000
# Below this many kept rows a repeat rate is too noisy to size the solution space from.
UNIQUE_CAP_MIN_ROWS = 100


def build_arm_cnf(bench_file, T, pin_outputs, target, cnf_path, encoder=GPAP_ENCODER):
    """Encode ONE instance of a rival row to `cnf_path` — the formula BOTH of our arms run, so a
    row's two comparisons are over one CNF by construction. `abc:fast` is the campaign's encoder,
    `tseitin` stays reachable for an A/B, and there is NO fallback between them."""
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
    medians. `unique_per_s_est` = samples/s x the distinct fraction of the timed run's own kept
    output rows. `cnf=(path, info)` is passed so the CMSGen arm reads the very same file."""
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
            # verify_sound: this row RECORDS `sound_check`, so the check runs; its wall is
            # excluded from every rate.
            g = sample_file(cnf_path, device, n_time=n_time, n_cov=0, out_cap=unique_rows,
                            configs=portfolio(portfolio_name), time_limit=time_limit, seed=s,
                            verify_sound=True)
            # Distinct fraction over the rows the timed run DELIVERED: exact on large spaces.
            bits = g.get("samples")
            n_coll = 0 if bits is None else len(bits)
            # OFFLINE, and timed anyway: the dedup runs after the sampler returned, so it is
            # outside every clock the sampler keeps, and `distinct_s` is the evidence.
            _t0 = time.perf_counter()
            n_distinct = exact_distinct(bits) if n_coll else 0
            distinct_s = time.perf_counter() - _t0
            # The GATHER is the other half and is not free at this row count: it rides inside the
            # timed loop, billed to phases["output"], which `samples_per_s` subtracts but
            # `sps_e2e_total` CARRIES. Both rivals dedup off their own timer, so ours comes off
            # both of our clocks.
            proj_s = float((g.get("phases") or {}).get("output") or 0.0)
            wall_e2e = float(g.get("wall_total_s") or 0.0) - proj_s
            sps_e2e = (g["n_models"] / wall_e2e) if wall_e2e > 0 else None
            # ENCODER-INDEPENDENT soundness on the kept rows: simulate each through the ORIGINAL
            # circuit. The on-device re-check verifies the CNF; this verifies the encode chain.
            sim_bad, sim_rows = None, 0
            cols, wm = g.get("cols"), g.get("wire_map") or {}
            if n_coll and target is not None and cols is not None and all(v in wm for v in cols):
                from gpap.circuits.simulate import simulate_check
                sim_bad = simulate_check(bench_file, T, 5, np.asarray(bits), [wm[v] for v in cols],
                                         pin_outputs=pin_outputs, target=target)
                sim_rows = n_coll
            # KEEP None: `or 0.0` turns "nothing to measure on" into "GPAP produced nothing", and
            # every consumer buckets a zero as an outright loss.
            uniq = (g["samples_per_s"] * n_distinct / n_coll) if n_coll else None
            # TRUNCATION means the GATHER CAP BOUND, and nothing else: on the adaptive path
            # `n_models` counts every config's models while the kept rows are the WINNER's only.
            partial = bool(n_coll >= unique_rows)
            # The space estimate survives ONLY to feed the `tiny` screen; it caps nothing now.
            space = (space_from_distinct(n_coll, n_distinct)
                     if n_coll >= UNIQUE_CAP_MIN_ROWS else None)
            runs.append({"seed": s, "sps": g["samples_per_s"], "sps_e2e": sps_e2e,
                         # what the offline dedup cost, and what it would have cost the e2e rate
                         "distinct_s": distinct_s, "out_project_s": proj_s,
                         "sps_e2e_with_delivery": g.get("sps_e2e_total"),
                         "unique_per_s_est": uniq, "space_est": space,
                         "unique_est_status": ("partial-gather" if partial else
                                               "ok" if uniq is not None else "no-sample-rows"),
                         "unique_rows": n_coll, "unique_distinct": n_distinct,
                         "sim_check_bad": sim_bad, "sim_check_rows": sim_rows,
                         "success": g["success_rate"], "n_models": g["n_models"],
                         # engine provenance per row: dec/s + impl/s let paper_tables.zero_search
                         # read rival rows too
                         "sound_check": g.get("sound_check"),
                         "decisions_per_s": g.get("decisions_per_s"),
                         "implications_per_s": g.get("implications_per_s"),
                         # the engine that measured it, read by verify_artifact's one-engine check
                         "engine": g.get("engine")})
        # NOTHING AT ALL? Then ask whether there is anything to find: booking a GPAP zero on an
        # instance with no solution at this depth scores us against one that does not exist.
        if not any(r["n_models"] for r in runs):
            unsat = unsat_at_depth(cnf_path, provenance=unsat_prov)
    finally:
        if own:
            os.unlink(cnf_path)
    # THE MEDIAN RUN, not a median of each field, which publishes a row NO EXECUTION EVER
    # PRODUCED. The SAFETY verdicts are still folded over ALL seeds: an unsound row anywhere in
    # the set cannot be hidden by which run got picked.
    pick, order = median_run(runs, "sps")
    return {"seeds": runs,
            # WHICH sampler produced this row: a fixed config is only fixed if the row says which.
            "portfolio": portfolio_name,
            # True only when a solver PROVED there is none at this depth; None = not asked or not
            # resolved inside the budget, and paper_tables screens True, never None.
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
            # merge_sound_check, not seed 0's verdict: UNSOUND dominates and "ok" beats "off", so
            # a mixed set of seeds cannot publish the first seed's silence as the answer.
            "sound_check": merge_sound_check(*(sound_verdict(r) for r in runs)),
            # the circuit-level check, folded the same way: BAD dominates, "ok" beats "off"
            "sim_check": ("BAD" if any((r.get("sim_check_bad") or 0) > 0 for r in runs) else
                          "ok" if any(r.get("sim_check_rows") for r in runs) else "off"),
            "sim_check_rows": sum(r.get("sim_check_rows") or 0 for r in runs),
            "sim_check_bad": sum(r.get("sim_check_bad") or 0 for r in runs),
            "decisions_per_s": pick.get("decisions_per_s") or 0.0,
            "implications_per_s": pick.get("implications_per_s") or 0.0,
            # spread over the seeds, so a reader can see what the median is worth
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
    ap.add_argument("--init-budget", type=float, default=INIT_BUDGET_S,
                    help="total wall for the start_point initialization search per instance "
                         "(s); it BOUNDS the search and is in no published rate. A cell that "
                         "does not initialize inside it is an error row naming the budget")
    ap.add_argument("--grants", default=None, metavar="FILE",
                    help="a grants file (eval/rivals/extract_grants.py): pin each cell's batch "
                         "rung and start_point to the values a previous campaign measured — "
                         "no ladder walk, no probe; a rung this card cannot finish on is "
                         "walked down from. Cells recorded as rival failures are re-measured.")
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
    grants = load_grants(args.grants)          # {} = walk the ladder and probe, as always
    if grants:
        print(f"[grants] {args.grants}: {len(grants.get('recurrent') or {})} recorded "
              f"Recurrent cells (measured on {grants.get('gpu')}); pinned where recorded")
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
    # Re-running this stage for a RIVAL-side reason must not re-pay OUR arm; keyed on the sampled
    # file's bytes + task + engine build + GPU, so anything that moves the number is a miss.
    cache_dir = args.gpap_cache or (os.path.join(os.path.dirname(args.output) or ".",
                                                 ".gpap_cache") if args.output else None)
    if args.resume and args.output and os.path.isfile(args.output):
        try:
            out = json.load(open(args.output)) or []
            if args.retry_errors:      # drop them so the retry REPLACES rather than appends
                # ...but ONLY the rows this protocol has not measured yet: an error row naming the
                # CURRENT budget is a finished measurement, and re-paying it stalls a resume.
                keep, retry = [], []
                for r in out:
                    err = (r.get("rcs") or {}).get("error")
                    if not err:
                        keep.append(r)
                    elif settled_error(err, args.init_budget, args.timeout):
                        keep.append(r)
                    else:
                        retry.append(r["name"])
                out = keep
                if retry:
                    print(f"[resume] retrying {len(retry)} error row(s) not measured under this "
                          f"budget: {', '.join(sorted(retry))}")
            done = {(r["name"], r.get("variant", "-")) for r in out}
            print(f"[resume] {len(out)} rows already in {args.output}")
        except Exception as e:      # noqa: BLE001 — a truncated checkpoint must not kill the run
            print(f"[resume] ignoring unreadable {args.output}: {e}")
            out, done = [], set()
    # EVERY instance that produces no row is tracked; a stage that measured NOTHING exits non-zero.
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
        # writing no row and silently truncating every instance after it. Gates OUR arm only.
        too_big = oversize_reason(src0, args.cycles, encoder=GPAP_ENCODER)
        if too_big:
            print(f"{name:<10}  GPAP arm not attempted (oversize): {too_big[:70]}", flush=True)
        # BOTH sides get the same circuit (rival_common.prepare_variant): with 'abc' the rival is
        # handed the ABC-optimized .bench too, gated by a bit-exact simulation equivalence check.
        src, vinfo = prepare_variant(src0, variant, tmp, cycles=args.cycles,
                                     seed=args.target_seed,
                                     target_mode="rare" if args.rare else "random")
        if vinfo.get("error"):
            # A ROW, not a silent skip: `prepare_variant` fails closed, and a `continue` here
            # would drop the instance from the sweep with one line in a log.
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
            # bit-identical input sequences — the per-row check the DEMOTIC harness also runs.
            p_miter = None
            pin_ok = native_pin_check(src, gpap_file, idx, vals, pin_n, args.cycles)
            if not pin_ok:
                print(f"{name:<10}{variant:>6}  PIN CHECK FAILED — reordered pins do not match "
                      f"their index table; row flagged", flush=True)
        sim = sim_baseline(gpap_file, args.cycles, args.target_seed, pin_outputs=pin_n,
                           target=gpap_target)
        # A RECORDED grant pins both halves of the rival's search (the settled start_point and the
        # granted rung), so a reproduction pays neither; a recorded FAILURE has no grant.
        grant = lookup_grant(grants, "recurrent", name, "rare" if args.rare else "table",
                             args.cycles, variant)
        if grant and grant.get("start_point") is not None:
            sp, why, sp_src = int(grant["start_point"]), None, grant["source"]
        else:
            sp, why = probe_start_point(args.rival, rival_file, key, args.cycles,
                                        budget=args.init_budget,           # bounded, see above
                                        cache_dir=cache_dir, run_timeout=args.timeout)
            sp_src = "probed"
        if sp is None:
            # `why` is one of two rival outcomes under a stated rule, recorded verbatim, never
            # as a zero.
            rcs = {"error": why}
        else:
            batch = pick_batch(rival_file, args.cycles)
            rcs = run_recurrent_once(args.rival, rival_file, key, args.cycles, steps,
                                     batch, sp, timeout=args.timeout, lr=lr,
                                     cache_dir=cache_dir, grant=grant)
            rcs["start_point_source"] = sp_src
            rcs["steps_source"] = steps_src
            rcs["lr"], rcs["lr_source"] = lr, lr_src
            rcs["seed"], rcs["seed_source"] = RIVAL_SEED, "their flags.py default; one run"
        task = {"harness": "recurrent", "mode": "rare" if args.rare else "table",
                "cycles": args.cycles, "pin_outputs": pin_n, "target": gpap_target,
                "seeds": args.seeds, "n_time": args.gpap_n,
                "time_limit": args.gpap_time_limit, "portfolio": args.portfolio,
                "encoder": GPAP_ENCODER, "unique_rows": UNIQUE_ROWS}
        # ONE BAD INSTANCE MUST NOT TRUNCATE THE SWEEP: an envelope refusal used to raise and kill
        # the harness. It is recorded as a GPAP-side error row (its own bucket in paper_tables —
        # an envelope refusal is NOT "GPAP produced nothing"), as a rival crash is on their side.
        # OUR ARM RUNS EVEN WHEN THE RIVAL FAILED: there is no ratio in such a row, but the
        # capability contrast is what the sweep is asked to cover. GPAP_RIVAL_SKIP_ON_ERROR=1
        # restores the old skip.
        skip_on_err = os.environ.get("GPAP_RIVAL_SKIP_ON_ERROR", "0") == "1"
        # ONE CNF PER ROW, both of our arms on it: built here rather than inside run_gpap so the
        # CMSGen arm reads the very file the GPAP arm sampled, and built even on a cache hit.
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
        # The CMSGen arm: the same CNF file, n and sampling budget as the GPAP arm. Not cached
        # (a `--resume` skips finished rows; only `--retry-errors` re-pays it).
        cms = (cmsgen_arm(cnf_path, args.cmsgen, args.gpap_n, args.gpap_time_limit, tmp,
                          seed=args.cmsgen_seed) if (args.cmsgen and cnf_info) else {})
        if cnf_path and os.path.exists(cnf_path):
            os.remove(cnf_path)                 # a b19-class CNF is hundreds of MB; not kept
        row = {"name": name, "mode": "rare" if args.rare else "table", "pin": pin_desc,
               "variant": variant, "circuit": vinfo, "cycles": args.cycles,
               # constraint alignment, verified per row: table mode = bit-identical acceptance
               # words; rare mode = the miter's P[MITER=1] equals p_accept
               "pin_check_ok": pin_ok, "p_miter": p_miter,
               # WAS THIS ARM MEASURED IN THIS RUN, OR SERVED FROM THE CACHE? An artifact
               # evaluator reading the JSON alone must be able to tell.
               "gpap_cache_hit": bool(hit),
               "p_accept": sim["p_accept"], "p_accept_lt": sim.get("p_accept_lt"),
               "rival_commit": rival_commit(args.rival), "rcs": rcs, "gpap": g,
               # the ONE formula both of our arms ran; `support` at the top level is what
               # `paper_tables.tiny_space` reads, as on a bench row
               "cnf": cnf_info, "support": (cnf_info or {}).get("support"),
               **cms, **host_provenance(device),
               "vram_budget_bytes": rival_vram_budget(),
               # HOST ram, recorded because it decides what this machine could attempt at all:
               # `oversize_reason`'s second term reads it, so a refusal row needs it
               "host_mem_avail_bytes": _available_bytes()}
        out.append(row)
        if args.output:      # checkpoint after every instance: this stage runs for HOURS, and a
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)   # death at 90% used
            json.dump(out, open(args.output, "w"), indent=1)                  # to leave nothing
        pa = sim["p_accept"] if sim["p_accept"] else sim.get("p_accept_lt", 0.0)
        pa_s = f"{pa:.2e}" if sim["p_accept"] else f"<{pa:.0e}"
        # None = the timed run kept no output row to measure on, i.e. NOT MEASURED, not a zero:
        # `n_models` says whether GPAP produced anything.
        uq = g.get("unique_per_s_est")
        uq_s = f"{uq:,.0f}" if uq is not None else "n/a"
        if "error" in rcs:
            # `g` may hold no measurement (the arm is skipped when the rival already failed).
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
