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

    THE fingerprint is the core's own `gpap_build_id()`; this is a thin call, not a second
    implementation. It exists because keying the cache on `gpap_version()` — the literal "0.1.0"
    for the project's whole life — made a REBUILT engine a cache HIT, exactly the case the key
    exists to miss. A stale arm is not a stale number, it is a wrong one: a kernel rewrite can
    move GPAP's rate by more than the effect being published, so a rival re-run against a
    pre-rewrite cache pairs a current rival subprocess with a months-old GPAP.
    (docs/findings/WAVE_SHUFFLE_FINDINGS.md has the figure.)
    """
    try:
        from gpap.engine import _core
        build = _core.build_id()
    except Exception:                       # noqa: BLE001 — no engine, no GPAP arm to cache
        return "unknown"
    # The core reports "<version>+unknown" when it cannot locate its own file (static link,
    # stripped dl info). That is a STABLE string across every rebuild, i.e. exactly the constant
    # key this helper exists to avoid — and it looks like a real fingerprint, so it would fail
    # silently. Fall back to hashing the library file ourselves; if that is not possible either,
    # say "unknown" and let the caller's own disable-on-unknown path handle it.
    if not build.endswith("+unknown"):
        return build
    import hashlib
    path = getattr(_core.load(), "_name", None)
    if not path or not os.path.isfile(path):
        return "unknown"
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"{os.path.basename(path)}:{h.hexdigest()[:16]}"


@functools.lru_cache(maxsize=1)
def _sampling_layer_id() -> str:
    """SHA1 over the python modules whose contents decide a GPAP arm's result. Missing files are
    hashed as their name alone, so a moved module is a cache MISS rather than a silent hit."""
    import hashlib
    from gpap import sampling
    root = os.path.dirname(os.path.abspath(sampling.__file__))
    h = hashlib.sha1()
    for name in ("pipeline.py", "sampler.py", "preprocess.py"):
        h.update(name.encode())
        try:
            with open(os.path.join(root, name), "rb") as f:
                h.update(f.read())
        except OSError:
            pass
    return h.hexdigest()[:16]


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
    # ONE engine-identity field. It used to be two — `core` (gpap_version(), a literal that never
    # changed) plus `build` (a local hash of the .so, added because of that) — and now that both
    # resolve to gpap_build_id() they would hash the same string twice under two names, with a
    # comment claiming one of them cannot change. A reader who trusts that comment and drops the
    # "redundant" field brings the stale-arm bug straight back.
    env["build"] = _core_build_id()
    # ...AND the PYTHON sampling layer, because the arm's answer is not decided by the C core
    # alone. `gpap_build_id()` fingerprints libgpap's bytes; the portfolio, the defaults, the
    # preprocessing and the coverage-run budget all live above it in `gpap/sampling/`, and a
    # change there moves the number while leaving the build id untouched. That happened on
    # 2026-09-03 (`pipeline.cov_batch`), and the comment right above this one warns about exactly
    # this failure one layer down. Hash the source of the modules that decide the answer.
    env["sampling_py"] = _sampling_layer_id()
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

# ---------------------------------------------------------------- oversize pre-check
# THE RIVAL PATH CANNOT USE abc:fast. A circuit-level rival consumes .bench and unrolls it
# itself, so our arm must encode THE SAME circuit — which means Tseitin, and Tseitin is far
# larger: b19 at T=25 is 66,639 vars through abc:fast (what Table 1 uses) and 9,591,351 vars
# through Tseitin, whose 2N=19,182,702 exceeds the packed 8|24 cap of 16,777,215.
#
# Worse than a clean refusal: building that CNF in Python exhausts host memory and the process
# is KILLED, so no exception is raised and no row is written. Measured 2026-09-03 — b19
# (231,266 gates x 25 = 5.8M unrolled) is OOM-killed; b18 (114,561 x 25 = 2.9M) completes. The
# harness therefore ended four instances short (b20/b21/b22/s35932, all alphabetically after
# b19) with nothing marking the gap.
#
# So the size is checked BEFORE anything is built, from the .bench alone, and an oversized
# instance becomes a recorded row like any other refusal. The bound is deliberately below the
# measured failure and above the measured success; it bounds the BUILD, not the envelope, which
# is why it is a gate count and not a 2N figure.
MAX_UNROLLED_GATES = 4_000_000

# The SECOND term is the box's own RAM, because the first one is a fixed number and the machines
# are not: the same instance that runs on a 47 GB workstation is an OOM kill on a 13 GB VM, and
# an OOM kill is the failure with no exception to catch and no row to show for it.
#
# MEASURED on this suite (2026-09-03, one process per instance, peak RSS of the whole arm —
# unroll + Tseitin + DIMACS round-trip + engine construction + the capped coverage collection):
#
#   b14  250,300 unrolled gates ->  2.12 GiB      b22    747,425 ->  4.30 GiB
#   b20  504,300               ->  3.20 GiB      b18  2,864,025 -> 10.56 GiB
#   b17  804,800               ->  3.77 GiB
#
# which is linear to ~15%: about 1.4 GiB fixed plus 3.3 GiB per million unrolled gates (the
# Tseitin build alone is a very clean 0.73 GiB per million; the rest is the DIMACS parse and the
# clause lists it produces). Refuse at 60% of AVAILABLE memory, not total: the campaign's own
# torch process and the rival's are resident at the same time.
PEAK_GIB_FIXED = 1.4
PEAK_GIB_PER_MGATE = 3.3
HOST_MEM_FRACTION = 0.60


def _available_bytes():
    """Available (not free) host RAM, or None when it cannot be read — in which case the memory
    term is simply not applied, since refusing an instance on a guess is worse than attempting
    it."""
    try:
        with open("/proc/meminfo") as fh:
            for ln in fh:
                if ln.startswith("MemAvailable:"):
                    return int(ln.split()[1]) * 1024
    except OSError:
        pass
    return None


def peak_host_gib(unrolled_gates):
    """Estimated peak host RSS of one rival-path GPAP arm, in GiB. See the fit above."""
    return PEAK_GIB_FIXED + PEAK_GIB_PER_MGATE * (unrolled_gates / 1e6)


def oversize_reason(bench_file, T, limit=MAX_UNROLLED_GATES, avail_bytes=None,
                    encoder="tseitin"):
    """A stated reason string when this (circuit, depth) must not be attempted, else None.

    Both terms model the NATIVE TSEITIN build (the fit above is its peak RSS), so they apply
    only when that is the encoder. Under `abc:fast` — the GPAP arm's encoder since 2026-09-03 —
    the unroll happens inside `abc` and the CNF is 2-20x smaller (b14 at T=25: 41,938 vars vs
    413,451; b18 19,859), and EVERY seq instance in the suite, b19 and s35932 at T=50 included,
    built and sampled on the 13 GB L4 VM in the CMSGen stage of the 2026-09-02 campaign. A
    Tseitin-calibrated refusal there would omit instances the campaign has already run, so the
    guard passes them; an abc build that fails still becomes an error row, never a kill."""
    if encoder != "tseitin":
        return None
    try:
        n_gates = sum(1 for ln in open(bench_file)
                      if "=" in ln and not ln.lstrip().startswith("#"))
    except OSError:
        return None
    unrolled = n_gates * max(T, 1)
    if unrolled > limit:
        return (f"{n_gates:,} gates x T={T} = {unrolled:,} unrolled gates > {limit:,}: the "
                f"Tseitin CNF the rival path requires exceeds the packed envelope and its build "
                f"exhausts host memory (abc:fast, used in the CMSGen table, is far smaller)")
    avail = _available_bytes() if avail_bytes is None else avail_bytes
    if avail:
        need, budget = peak_host_gib(unrolled), HOST_MEM_FRACTION * avail / (1 << 30)
        if need > budget:
            return (f"{n_gates:,} gates x T={T} = {unrolled:,} unrolled gates needs ~{need:.1f} "
                    f"GiB of host RAM (measured fit), over {HOST_MEM_FRACTION:.0%} of the "
                    f"{avail / (1 << 30):.1f} GiB available on this machine — refused here "
                    f"rather than OOM-killed mid-build, which writes no row at all")
    return None


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


# ---------------------------------------------------------------- the instance sets
#
# THE SET EACH RIVAL IS RUN AND SCORED ON IS ITS OWN PAPER'S — not a list we picked. ONE home:
# the harnesses expand the instance name `paper` to it, and `eval/artifact/paper_tables.py`
# imports it to decide what Table 2 scores, so the sweep and the table cannot drift (a curated
# list is the one thing in this protocol a referee cannot check, and the drivers had already
# drifted from each other twice).
#   DEMOTIC   (arXiv 2502.08086 Table II): all 11 ISCAS-85, c17 included.
#   Recurrent (arXiv 2502.21226 Table II): 8 ISCAS-89 + 8 ITC'99. Their "s35844" is s35932 — no
#             such circuit exists in ISCAS-89 and s35932 is the only s359xx branch in their
#             run.py. b17/b19/b22 are too large to commit; scripts/fetch_benchmarks.sh itc99
#             installs them (a harness reports a missing one as skipped, never silently).
PAPER_INSTANCES = {
    "demotic": frozenset({"c17", "c432", "c499", "c880", "c1355", "c1908", "c2670", "c3540",
                          "c5315", "c6288", "c7552"}),
    "rcs": frozenset({"s208.1", "s386", "s526", "s832", "s1196", "s4863", "s15850.1", "s35932",
                      "b01", "b06", "b10", "b12", "b15", "b17", "b19", "b22"}),
}


def expand_names(names, key):
    """`paper` -> that rival's published set (sorted, so a sweep's order is reproducible);
    anything else is passed through."""
    out = []
    for n in names:
        if n == "paper":
            out.extend(sorted(PAPER_INSTANCES[key]))
        else:
            out.append(n)
    return out


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
    """ONE measured run per instance, at the rival's granted batch.

    The only fallback is a DRIVER REFUSAL: 'device not ready' reproduces at a fixed size on some
    models yet passes one rung lower, so when the granted rung produces no run at all the machine
    gets the last word over our estimate and the next rung down is tried. A refused rung is not a
    measurement, so the row still holds exactly one — the largest rung the driver accepted.

    A TIMEOUT is a result, not a refusal: the rival ran and did not finish inside its budget, and
    it is recorded as such. It is NOT retried lower (their runtime grows with the batch, so a
    smaller rung might come in under the cap — but that is a second configuration, and the rule
    is one). The one-rung probe UPWARD that used to follow a successful run — kept only when the
    bigger rung also ran faster — was likewise a best-of-two and is gone (2026-09-03).

    Returns (result, sweep) — every rung tried, so the row records what the rival was granted.
    """
    rungs = list(ladder)
    i = max((k for k, b in enumerate(rungs) if b <= batch), default=0)
    sweep = []
    while i >= 0:
        r = run(rungs[i])
        sweep.append({"batch": rungs[i], "unique_per_s": r.get("unique_per_s"),
                      "error": _one_line(r.get("error"))})
        if "error" not in r:
            return r, sweep
        if "timeout" in str(r.get("error", "")).lower():
            log(f"    [batch] {rungs[i]:,} timed out — recording the timeout")
            return r, sweep
        log(f"    [batch] {rungs[i]:,} refused ({_one_line(r.get('error'))[:60]}) — stepping down")
        i -= 1
    return (sweep[-1] if sweep else {"error": "empty ladder"}), sweep


# ---------------------------------------------------------------- the CMSGen arm of a rival row

def cmsgen_arm(cnf_path, cmsgen_bin, n, timeout, scratch, seed=None):
    """CMSGen on the VERY CNF the GPAP arm sampled — the third tool of every rival row.

    The converged protocol (docs/findings/RIVAL_FAIRNESS_FINDINGS.md §14) compares three tools
    on one instance: the rival on its own circuit, GPAP on the `abc:fast` CNF of that circuit
    and those pins, and CMSGen on that same CNF FILE. It used to be a separate `gpap bench`
    sweep that pinned the circuits its own way (one output on ISCAS-85, the outputs directly
    rather than a miter on the unrolls) and never saw a rival's own pins at all, so the CMSGen
    table and the rival table described two different instance sets.

    Same runner as `gpap bench` (gpap.benchmark.cmsgen_run): the patched binary timed under
    `--nowrite` (emission-free, like our `collect=False` run) with `--maxtime` (it stops itself
    and prints an EXACT count — a killed run reports a floored one), one run at the same
    sampling budget the GPAP arm gets. The keys are the bench driver's own `cms_*` names, so
    every reader of a bench row — `paper_tables.cms_arm_ran`, `verify_artifact`'s floored-count
    and emission-mode checks, `check_emission_symmetry` — reads a rival row unchanged. A
    subprocess failure is recorded as `cms_error` beside a count of 0: a crashed baseline is not
    a zero (the ledger screens it; `verify_artifact` fails on it).
    """
    from gpap.benchmark import cmsgen_run
    t = cmsgen_run(cnf_path, cmsgen_bin, n, scratch, timeout=timeout, write=False, seed=seed)
    row = {"cms_mode": t.get("mode"), "cms_patched": t.get("patched"),
           "cms_sps": t.get("samples_per_s"), "cms_sps_e2e": t.get("samples_per_s_e2e"),
           "cms_n_samples": t.get("n_samples", 0), "cms_timed_out": t.get("timed_out"),
           "cms_count_exact": t.get("count_exact"), "cms_budget_stop": t.get("budget_stop"),
           "cms_wall_s": t.get("wall_s"), "cms_sampling_wall_s": t.get("sampling_wall_s"),
           "cms_decisions_per_s": t.get("decisions_per_s"),
           "cms_implications_per_s": t.get("implications_per_s"),
           "cms_props_per_s": t.get("props_per_s"), "cms_conflicts": t.get("conflicts"),
           "cms_n_requested": n, "cms_timeout_s": timeout}
    if t.get("error"):
        row["cms_error"] = t["error"]
    return row


def host_provenance(device):
    """`cpu` + `gpu` on every rival row, the bench driver's own dicts.

    A GPAP/CMSGen ratio is not interpretable without the CPU — CMSGen is single-core, and the
    same instances read 0.41x differently between two hosts on the CPU alone — so the rival
    rows, which now carry that ratio, record what `gpap bench` records (docs/BENCHMARKING.md)."""
    from gpap.benchmark import _cpu_info, _gpu_info
    try:
        gpu = _gpu_info(device)
    except Exception:                       # noqa: BLE001 — provenance must never break a sweep
        gpu = {"name": str(device)}
    return {"cpu": _cpu_info(), "gpu": gpu}
