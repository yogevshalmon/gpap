"""Shared pieces of the GPU-rival head-to-heads (eval/rivals_{demotic,recurrent,hts}.py).

The one policy that lives here: **circuit variants**. GPAP's headline circuit numbers are measured
on two encodings of the same instance — our native Tseitin unroll, and an ABC-optimized one. The
circuit-level rivals (DEMOTIC, Recurrent-CircuitSAT) consume `.bench` and unroll it themselves, so
if only GPAP sees the ABC-optimized circuit the encoder's win is silently credited to the sampler.
`prepare_variant` gives BOTH sides the same circuit:

    raw  — the benchmark file as committed
    abc  — `strash; zero; dc2; dc2; fraig` on the SEQUENTIAL AIG, written back as a
           primitive-gate .bench (gpap.circuits.abc_backend.simplify_bench_abc)

and gates the swap on a bit-exact simulation check: both circuits are fed identical stimuli and
must produce identical words on EVERY primary output (and so, in particular, accept exactly the
same input sequences for the pinned target). Comparing only the acceptance word is vacuous at a
rare pin — both sides are all-zero there — which is why the gate is output equality.

Honest limit, worth stating in any write-up: the rivals can only be given the SEQUENTIAL
optimization. GPAP's `abc` encoder also runs a pass AFTER unrolling, where the cross-frame
equivalence collapse lives (b12-class 12-14x) — that pass has no sequential equivalent, because
once frames are expanded the circuit can no longer be handed to a tool that unrolls it itself.
Measured on the ITC'99/ISCAS-89 benches, the sequential-only pass is nearly free (b03 129 -> 127
ANDs, s386 166 -> 113), so in practice the rivals see essentially the same circuit either way —
which is itself the point: their gap is not an encoding artifact.
"""
import functools
import os
import subprocess

VARIANTS = ("raw", "abc")


def rival_commit(path: str):
    """HEAD of the rival checkout at `path`, or None. Recorded into every head-to-head JSON:
    rivals_setup.sh pins the clones, but a pin can be overridden or a checkout moved by hand —
    the result must carry what actually ran, not what was intended."""
    try:
        out = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:                                   # noqa: BLE001 — provenance, never fatal
        return None


def prepare_variant(src: str, variant: str, tmpdir: str, *, cycles: int, seed: int,
                    target_mode: str = "random", pin_outputs=None, target=None) -> tuple:
    """(path, info) for one circuit variant of `src`. `info` carries the ABC stats and the
    equivalence-check result; `info["error"]` is set (and the RAW path returned) when ABC is
    unavailable or the check fails — a rival comparison must never silently run on a circuit
    that is not the instance."""
    if variant == "raw":
        return src, {"variant": "raw"}
    if variant != "abc":
        raise ValueError(f"unknown circuit variant {variant!r} (raw | abc)")
    from gpap.circuits.abc_backend import abc_available, bench_equiv_check, simplify_bench_abc
    info = {"variant": "abc"}
    if not abc_available():
        return src, {**info, "error": "no abc binary (scripts/build_abc.sh)"}
    dst = os.path.join(tmpdir, f"{os.path.splitext(os.path.basename(src))[0]}_abc.bench")
    try:
        info.update(simplify_bench_abc(src, dst))
        chk = bench_equiv_check(src, dst, cycles, seed, pin_outputs=pin_outputs,
                                target_mode=target_mode, target=target)
    except Exception as e:  # noqa: BLE001 — fall back to raw rather than compare the wrong circuit
        return src, {**info, "error": f"{type(e).__name__}: {e}"}
    info["equiv_ok"] = chk["ok"]
    info["equiv_hits"] = [chk["hits_src"], chk["hits_dst"], chk["n"]]
    # WHICH half of the check had content, recorded per row. The acceptance-word half is
    # vacuous at a rare target (both words all-zero, 0 hits on both sides — the state of every
    # ABC row in the recorded ISCAS-85 campaign), so the OUTPUT-equality half is what actually
    # gates the swap; a row that shows `equiv_outputs` proves the gate was not comparing nothing.
    info["equiv_outputs"] = chk.get("outputs")
    info["equiv_outputs_ok"] = chk.get("outputs_ok")
    info["equiv_accept_ok"] = chk.get("accept_ok")
    if not chk["ok"]:
        why = ("outputs differ on " + ", ".join(map(str, chk.get("mismatched") or []))
               if not chk.get("outputs_ok") else "acceptance differs on the pinned target")
        return src, {**info, "error": f"ABC copy is NOT equivalent: {why}"}
    return dst, info


def variant_note(info: dict) -> str:
    """One-line human summary for the table row."""
    if info.get("variant") == "raw":
        return "raw"
    if info.get("error"):
        return f"abc FAILED ({info['error'][:40]}) -> raw"
    return f"abc {info.get('ands_before')}->{info.get('ands_after')} ANDs"


# --------------------------------------------------------------------------- GPAP arm cache
#
# The rival stage spends most of its wall NOT on the rival: of the 2026-08-23 L4 campaign's
# ~3.3 h, the three rivals' own subprocesses accounted for 64 min. The rest is our side —
# the GPAP arm (Recurrent alone runs 3 seeds x 18 cells), the start_point probes, sim_baseline.
# Almost all of it is re-paid verbatim whenever a stage is re-run for a RIVAL-side reason
# (a wider batch ladder, a fixed seed, a re-measured timeout), even though the GPAP side is
# bit-for-bit the same task.
#
# So: cache the GPAP arm on disk, keyed by everything that can change its answer — the exact
# bytes of the file sampled, the task (cycles/pins/target), the sampling knobs, the ENGINE BUILD
# and the GPU. Any of those moving is a cache miss, which is the point: a stale GPAP number is
# far worse than a re-run. `GPAP_RIVAL_CACHE=0` disables it outright.

def _canonical(obj):
    """Stable JSON for hashing: sorted keys, no whitespace, tuples as lists."""
    import json
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


@functools.lru_cache(maxsize=1)
def _core_build_id() -> str:
    """Fingerprint of the libgpap BUILD — the bytes, not the version it advertises.

    `_core_provenance()` reads `gpap_version()`, which is the hardcoded literal "0.1.0"
    (cpp/src/runtime.cpp) and has never changed, so keying the cache on it made a REBUILT engine
    a cache HIT — exactly the case the key exists to miss. A stale arm is not a stale number, it
    is a wrong one: the 2026-08-27 wave rewrite is 1.80x at the same K, so a rival re-run against
    a pre-rewrite cache would publish today's rival subprocess against last month's GPAP.
    """
    import hashlib
    try:
        from gpap.engine import _core
        path = getattr(_core.load(), "_name", None)
        if not path or not os.path.isfile(path):
            return "unknown"
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return f"{os.path.basename(path)}:{h.hexdigest()[:16]}"
    except Exception:                       # noqa: BLE001 — no engine, no GPAP arm to cache
        return "unknown"


def gpap_cache_key(sampled_file: str, task: dict) -> str:
    """Identity of one GPAP measurement. Includes the file's CONTENT, not its path — the rival
    harnesses write their inputs into per-run temp dirs, so paths differ every invocation while
    the instance is identical."""
    import hashlib
    h = hashlib.sha1()
    with open(sampled_file, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    env = {"file_sha1": h.hexdigest(), "task": task}
    try:                                    # a result from another engine build or GPU is NOT ours
        from gpap.sampling.pipeline import _core_provenance
        env["core"] = _core_provenance()
    except Exception:                       # noqa: BLE001
        env["core"] = "unknown"
    env["build"] = _core_build_id()         # the engine BINARY, since "core" cannot change
    try:
        import torch
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:                       # noqa: BLE001
        env["gpu"] = "unknown"
    return hashlib.sha1(_canonical(env).encode()).hexdigest()


CACHE_MISS = object()          # distinct from a stored None


def cache_peek(cache_dir, sampled_file, task):
    """Read-only lookup: the stored value, or `CACHE_MISS`. Separate from cached_gpap because a
    caller that only wants to KNOW must not be able to write a placeholder — passing a
    `lambda: None` as the compute would store None and turn every later lookup into a false hit
    that silently skips the work."""
    import json
    if not cache_dir or os.environ.get("GPAP_RIVAL_CACHE") == "0":
        return CACHE_MISS
    path = os.path.join(cache_dir, gpap_cache_key(sampled_file, task) + ".json")
    if not os.path.isfile(path):
        return CACHE_MISS
    try:
        return json.load(open(path))["value"]
    except Exception:                               # noqa: BLE001 — a corrupt entry is a miss
        return CACHE_MISS


def cache_store(cache_dir, sampled_file, task, value):
    """Write one entry, atomically. No-op when caching is off."""
    import json
    if not cache_dir or os.environ.get("GPAP_RIVAL_CACHE") == "0":
        return
    path = os.path.join(cache_dir, gpap_cache_key(sampled_file, task) + ".json")
    try:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = path + ".tmp"
        json.dump({"task": task, "file": os.path.basename(sampled_file), "value": value},
                  open(tmp, "w"), indent=1)
        os.replace(tmp, path)
    except Exception as e:                          # noqa: BLE001
        print(f"[gpap-cache] could not store {path}: {e}")


def cached_gpap(cache_dir, sampled_file, task, compute):
    """`compute()` unless this exact GPAP measurement is already on disk under `cache_dir`.

    Returns (result, hit). The stored record keeps the task next to the value so a cache
    directory can be read by a human — a bare hash tells you nothing when a number looks wrong.
    """
    import json
    if not cache_dir or os.environ.get("GPAP_RIVAL_CACHE") == "0":
        return compute(), False
    key = gpap_cache_key(sampled_file, task)
    path = os.path.join(cache_dir, key + ".json")
    if os.path.isfile(path):
        try:
            rec = json.load(open(path))
            return rec["value"], True
        except Exception as e:              # noqa: BLE001 — a corrupt entry must only cost a re-run
            print(f"[gpap-cache] ignoring unreadable {path}: {e}")
    value = compute()
    try:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = path + ".tmp"
        json.dump({"task": task, "file": os.path.basename(sampled_file), "value": value},
                  open(tmp, "w"), indent=1)
        os.replace(tmp, path)               # atomic: a VM reclaimed mid-write leaves no half entry
    except Exception as e:                  # noqa: BLE001
        print(f"[gpap-cache] could not store {path}: {e}")
    return value, False


# ------------------------------------------------------------------ the parallelism grant
#
# MEASURED on an RTX 4080 (2026-08-25): every one of the three rivals scores in near-PROPORTION
# to the batch it is handed, across the whole range we can run —
#
#   HTS       b03-c25-rare   batch    100 ->    100,000 :   1 ->    294 unique/s  (295x)
#   Recurrent s386 T=25      batch 10,000 ->    300,000 : 3.8k ->  69.9k unique/s ( 18.6x)
#   DEMOTIC   c432           batch  1,000 ->    300,000 : 2.5k -> 654k  unique/s (258x)
#   DEMOTIC   c1355          batch  1,000 ->  1,000,000 : 1.2k -> 662k  unique/s (564x)
#
# model_runtime barely moves across those (c432: 0.39 s -> 0.46 s for a 300x batch), i.e. the GPU
# is nowhere near saturated at the small end. The batch is therefore not a detail of their setup:
# it IS the parallelism grant, the same knob K is for us — and GPAP sizes K at 0.80 of FREE VRAM
# (gpap.engine.calibrate.gpu_mem_available). Handing a rival a constant instead means the two
# sides get different fractions of the machine, and worse, a DIFFERENT fraction on every card:
# the recorded L4 campaign gave the rivals a fixed 5 GiB of a 24 GB board while GPAP took ~19 GB.
#
# So the budget is VRAM-proportional, by the same rule and the same fraction we apply to
# ourselves. Override with GPAP_RIVAL_VRAM_FRACTION (or an explicit budget) when reproducing an
# older sweep.

RIVAL_VRAM_FRACTION = 0.80


def rival_vram_budget(fraction: float = None) -> int:
    """Bytes of GPU memory a rival's autograd graph may occupy: the same share of the same card
    GPAP grants itself. Falls back to the historical 5 GiB when no CUDA device is visible, so a
    CPU-only smoke run still picks a sane batch."""
    if fraction is None:
        fraction = float(os.environ.get("GPAP_RIVAL_VRAM_FRACTION", RIVAL_VRAM_FRACTION))
    try:
        import torch
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return int(free * fraction)
    except Exception:                                   # noqa: BLE001 — sizing must never be fatal
        pass
    return 5 << 30


BATCH_LADDER = (1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000)

# Bytes of real peak VRAM per unit of "work", MEASURED on an RTX 4080 (2026-08-25) rather than
# modelled — the naive `n_gates * T * batch * 4` undercounts by 2-6x and the factor is
# tool-dependent, so a budget built on it is not the budget you think you set:
#
#   Recurrent s386  batch 100k: est 1.48 GiB, peak 3.03 GiB (2.0x)   300k: 4.44 -> 8.36 (1.9x)
#   DEMOTIC   c432  batch 300k: est 0.18 GiB, peak 1.02 GiB (5.7x)     1M: 0.60 -> 2.97 (5.0x)
#   HTS       b03-c25-rare (14,505 clauses) batch 100k: peak 11.30 GiB = 8.4 B per row-clause
#
# These size the batch; the driver still gets the last word (a refused rung steps down).
# Use the factors AS MEASURED, not rounded up: the budget already keeps 20% of the card in
# reserve, and rounding up costs whole ladder rungs — at 2.0x instead of the measured 1.88x,
# Recurrent on s386 drops from 300k to 100k, i.e. 3x of its own throughput, for nothing. If a
# rung turns out not to fit after all, the driver refuses it and run_at_batch steps down.
GRAPH_BYTES = {"demotic": 4 * 5.0, "recurrent": 4 * 1.9}     # per gate x frame x row
CNF_BYTES_PER_ROW_CLAUSE = 8.4                                # HTS, per clause x row


def fit_batch(work_units, tool, budget_bytes=None, ladder=BATCH_LADDER):
    """The largest ladder rung that fits the rival inside its share of the card.

    `work_units` is gates x frames for the circuit rivals, clauses for HTS. THE RULE, chosen
    2026-08-25: each rival gets the batch that fits in **0.80 of free VRAM** — the same fraction,
    by the same rule, that GPAP sizes its own K with (gpap.engine.calibrate.gpu_mem_available).

    Their own "defaults" cannot serve as that rule: all three argparse defaults are `--batch_size
    10`, a placeholder, and their published demo scripts are per-instance one-liners with a
    1000x spread on comparably tiny circuits (DEMOTIC 1,000 on c17's 6 gates; Recurrent 1,000,000
    on s27's 10 gates; HTS 100 on a 75-variable CNF). Transplanted onto our instances those would
    hand DEMOTIC ~1/500 and HTS ~1/295 of the throughput their tool actually reaches on the card
    — inflating OUR ratios by the same factor. A share of the machine is the only rule that means
    the same thing for both sides.
    """
    per_row = max(work_units, 1) * (CNF_BYTES_PER_ROW_CLAUSE if tool == "hts"
                                    else GRAPH_BYTES.get(tool, 4 * 5.5))
    budget = budget_bytes or rival_vram_budget()
    for b in reversed(ladder):
        if per_row * b <= budget:
            return b
    return ladder[0]


# --------------------------------------------------------------------------- gradient steps
# `--num_steps` is the rivals' OTHER first-order knob (gradient-descent steps per batch), and
# unlike the batch it is not a share of the machine — it is a tool setting with an author-chosen
# value, so the fair thing is to use theirs rather than to tune it.
#
# THE RULE: run each rival at the value its OWN artifacts specify. Where a tool ships two (an
# argparse default and a demo script that overrides it), take the one that is better FOR THE
# RIVAL — a tie-break that can only ever move a ratio against us, and that never reaches for a
# value the authors do not themselves use. Resolved ONCE by measurement (RTX 4080, at each
# rival's granted batch; re-verified per campaign by eval/rivals/steps_calibration.py):
#
#   DEMOTIC    c432 @300k       steps=1  1,698,654 u/s   vs  steps=10    650,006  -> 1  (x2.61)
#   Recurrent  s386 @100k       steps=1    153,726 u/s   vs  steps=10     18,081  -> 1  (x8.50)
#   HTS        b03-c25-rare@30k steps=5          5 u/s   vs  steps=10        183  -> 10 (x36)
#
# The mechanism is visible in the counts, not just the rates: DEMOTIC and Recurrent SATURATE
# their batch on the first step (Recurrent finds 2,198,776 unique at steps=1 and 2,134,454 at
# steps=10 — every later step is pure cost), while HTS genuinely needs the descent on our CNFs
# and finds nothing at 5. One global value cannot be right for both, which is why this is
# per tool. The campaign to 2026-08-26 pinned 10 everywhere, taxing DEMOTIC 2.6x and Recurrent
# 8.5x; docs/findings/RIVAL_FAIRNESS_FINDINGS.md carries the full table.
RIVAL_STEPS = {
    # tool:      (steps, where the value comes from,                  the other value we rejected)
    "demotic":   (1,  "their script/run_sampler_comb.sh", 10),
    "recurrent": (1,  "their script/run_sampler_seq.sh", 10),
    "hts":       (10, "their src/pytorch/flags.py default", 5),
}


def resolve_steps(tool, requested="auto"):
    """(steps, provenance) for a rival. `requested` is a number, or "auto" for the tool's own.

    Recorded in every row as `steps` / `steps_source`, so a table can state which setting each
    rival ran at instead of implying they all ran at one.
    """
    steps, src, other = RIVAL_STEPS[tool]
    if requested in (None, "auto", "-1"):
        return steps, f"author default ({src}; their other value {other} is worse for them)"
    return int(requested), "explicit --steps"


def _one_line(err):
    """A rival's error field can be a traceback tail; the sweep only needs the reason."""
    if not err:
        return err
    line = [x for x in str(err).splitlines() if x.strip()]
    return (line[-1] if line else str(err))[:160]


def run_at_batch(run, batch, ladder=BATCH_LADDER, log=print):
    """Run the rival at its granted batch — then check the two ways the grant can be wrong.

    DOWN, if the driver refuses the size: 'device not ready' reproduces at a fixed size on some
    models yet passes one rung lower, so the machine gets the last word over our estimate.

    UP, once, if the granted rung ran: a single bytes-per-gate-row factor cannot be accurate
    across the circuit sizes we run — MEASURED for DEMOTIC at batch 300k, peak VRAM per gate-row
    falls from 23.0 B (c432, 160 gates) to 11.6 B (c3540, 1669 gates), because part of the cost
    is per-ROW rather than per-gate. Any single factor therefore over-charges one end: at 20 B
    the model rejects c2670 at batch 1M, which really needs ~16 GiB and fits an L4 fine. One
    probe upward removes that under-grant, and it is kept ONLY if the bigger rung also runs
    FASTER — which is the spill guard, since too-big does not always error (Recurrent on s386:
    1M runs, and is slower than 300k).

    Returns (result, sweep) — every rung tried, so the row records what the rival was granted.
    """
    rungs = list(ladder)
    start = max((k for k, b in enumerate(rungs) if b <= batch), default=0)
    sweep, best, i, timeouts = [], None, start, 0
    while i >= 0:
        r = run(rungs[i])
        sweep.append({"batch": rungs[i], "unique_per_s": r.get("unique_per_s"),
                      "error": _one_line(r.get("error"))})
        if "error" not in r:
            best = r
            break
        # A TIMEOUT is not a memory refusal, and walking the whole ladder on one is pure waste:
        # measured, HTS's three tseitin timeouts stepped 1M -> 1,000 at the full 300 s cap each,
        # ~40 min per cell to record the error it already had. Their runtime does grow with the
        # batch, so give it two rungs to come good and then stop.
        if "timeout" in str(r.get("error", "")).lower():
            timeouts += 1
            if timeouts > 2:
                log(f"    [batch] {rungs[i]:,} timed out again — recording the timeout")
                break
        log(f"    [batch] {rungs[i]:,} refused ({_one_line(r.get('error'))[:60]}) — stepping down")
        i -= 1
    if best is None:
        return (sweep[-1] if sweep else {"error": "empty ladder"}), sweep
    tried = {x["batch"] for x in sweep}
    if i + 1 < len(rungs) and rungs[i + 1] not in tried:   # never re-run a rung that refused
        r = run(rungs[i + 1])
        sweep.append({"batch": rungs[i + 1], "unique_per_s": r.get("unique_per_s"),
                      "error": _one_line(r.get("error"))})
        if "error" in r:
            log(f"    [batch] {rungs[i + 1]:,} refused — {rungs[i]:,} stands")
        elif (r.get("unique_per_s") or 0) > (best.get("unique_per_s") or 0):
            log(f"    [batch] {rungs[i + 1]:,} beats the granted {rungs[i]:,} — taking it")
            best = r
        else:
            log(f"    [batch] {rungs[i + 1]:,} ran but is SLOWER (spilling) — {rungs[i]:,} stands")
    return best, sweep
