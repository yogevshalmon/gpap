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
import time

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
    pkg = os.path.dirname(root)                       # gpap/
    here = os.path.dirname(os.path.abspath(__file__))  # eval/rivals/
    h = hashlib.sha1()
    # EVERY file that decides what a GPAP arm returns, not just the ones in the sampling package.
    # This was stopped one directory short twice: first at the C core (the python layer moved the
    # number while `gpap_build_id` did not), then at `gpap/sampling/` (the ARM itself — the
    # encoder call, the portfolio, the dedup, the rate accounting — lives in the harness files,
    # and a logic fix there changed no key at all). The rule is the same each time: hash the
    # source of everything that computes the answer.
    files = [os.path.join(root, n) for n in ("pipeline.py", "sampler.py", "preprocess.py")]
    files += [os.path.join(here, n) for n in ("rival_common.py", "rivals_recurrent.py",
                                              "rivals_demotic.py", "rivals_hts.py")]
    files += [os.path.join(pkg, "mining.py"),
              os.path.join(pkg, "engine", "calibrate.py")]
    circuits = os.path.join(pkg, "circuits")
    files += [os.path.join(circuits, n) for n in
              ("bench.py", "abc_backend.py", "aig2cnf.py", "simulate.py")]
    for path in files:
        h.update(os.path.basename(path).encode())
        try:
            with open(path, "rb") as f:
                h.update(f.read())
        except OSError:
            pass
    return h.hexdigest()[:16]


@functools.lru_cache(maxsize=1)
def _abc_fingerprint() -> str:
    """Identity of the `abc` binary that encodes the arm's CNF, as its content hash.

    `--version` would be weaker: ABC's version string moves far less often than its behaviour.
    A missing binary hashes to "absent", which is itself a distinguishing value — a run without
    abc cannot produce an abc:fast arm at all.
    """
    import hashlib
    try:
        from gpap import paths
        exe = os.path.join(paths.gpap_home(), "bin", "abc")
    except Exception:                                          # noqa: BLE001
        return "unknown"
    try:
        h = hashlib.sha1()
        with open(exe, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError:
        return "absent"


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
    # ...AND THE ENCODER BINARY. The arm samples the abc:fast CNF of the circuit, so the CNF is
    # a function of the `abc` executable as much as of the .bench bytes — and nothing else in
    # this key can see an ABC rebuild that changes what it emits. Hash the binary's own bytes;
    # a missing one is recorded as such rather than skipped silently.
    env["abc"] = _abc_fingerprint()
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
#   Recurrent (arXiv 2502.21226 Table II): 8 ISCAS-89 + 8 ITC'99. Their 8th ISCAS-89 row is
#             `s38584` — 12 PI / 278 PO / 19,253 gates / 1,452 DFF, which is exactly the
#             s38584.bench in their own data/ISCAS/. THIS WAS WRONG UNTIL 2026-09-04: the row
#             was read as "s35844" and `s35932` substituted for it, because s35932 is the only
#             s359xx branch in their run.py. Their code covers more circuits than their paper
#             reports; the paper is what PAPER_INSTANCES is a list of. s35932 stays in the swept
#             set as a circuit WE added, and is scored outside their 16.
#             b17/b19/b22 are too large to commit; scripts/fetch_benchmarks.sh itc99
#             installs them (a harness reports a missing one as skipped, never silently).
PAPER_INSTANCES = {
    "demotic": frozenset({"c17", "c432", "c499", "c880", "c1355", "c1908", "c2670", "c3540",
                          "c5315", "c6288", "c7552"}),
    "rcs": frozenset({"s208.1", "s386", "s526", "s832", "s1196", "s4863", "s15850.1", "s38584",
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


# The ladder's TOP must sit above what the card can run on any scored cell: the walk starts
# there, and a top rung that completes first try means the LADDER bound the grant, not the
# card — a fourth rejection reason the "feasibility only" rule does not state. That happened
# on six cells of the 2026-09-04 campaign at a 3,000,000 top (c1355 c1908 c432 c499 c880 under
# DEMOTIC's pins, b06 under Recurrent's; RIVAL_FAIRNESS_FINDINGS.md §16e). The top is
# 30,000,000 since 2026-09-06 (protocol v1.0, §17): on the campaign's 24 GB L4 no scored cell
# completes it (c17 would, and is screened as a tiny space). `paper_tables` still names any
# cell whose top rung completed first try, so a larger card cannot silently re-open the hole.
# The two extra rungs cost a big circuit two more doomed model builds on a FRESH card only: a
# known OOM ceiling is cached (rivals_recurrent.known_ladder) and a RECORDED grant skips the
# walk entirely (`pinned_ladder` / `load_grants`, below).
BATCH_LADDER = (1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000,
                10_000_000, 30_000_000)


# ------------------------------------------------------------------ recorded grants
# THE GRANT IS PART OF THE RECORD (protocol v1.0, 2026-09-06). A campaign's rows say which rung
# each rival was granted and, for Recurrent, which `start_point` its initialization settled on;
# `eval/rivals/extract_grants.py` writes them to one JSON per campaign (committed under
# eval/rivals/grants/), and a harness run with `--grants FILE` PINS them: no ladder walk, no
# start_point probe, one run at the recorded rung. That is what makes a re-run reproducible and
# cheap (the walk and the probe are hours of a campaign, and neither is a measurement). The
# RULE does not change — the pinned rung is the largest that completed, and a card that cannot
# finish on it descends exactly as the walk would — and a cell the campaign recorded as a
# rival FAILURE is never pinned: a failure is re-measured, not copied.

def grant_key(name, mode, cycles, variant="abc"):
    """One cell: circuit | task | unroll depth | circuit variant."""
    return f"{name}|{mode}|{int(cycles)}|{variant}"


def pinned_ladder(ladder, pinned):
    """The ladder a RECORDED grant turns the walk into: the pinned rung on top, the rungs below
    it kept. `run_at_batch` then measures the pinned rung first and descends only if THIS card
    cannot finish on it (a smaller card than the one that recorded the grant) — the rule is
    unchanged, only the search is skipped. A pinned value off the ladder is its own top rung."""
    pinned = int(pinned)
    return tuple(sorted({b for b in ladder if b < pinned} | {pinned}))


def load_grants(path):
    """The grants file, or {} when no path is given. A malformed file is an error, never a
    silent empty pin — a campaign that believes it is reproducing one and is walking instead
    would look right in every row and cost hours."""
    import json
    if not path:
        return {}
    with open(path) as f:
        g = json.load(f)
    if not isinstance(g, dict) or not any(k in g for k in ("demotic", "recurrent")):
        raise ValueError(f"{path}: not a grants file (no 'demotic'/'recurrent' table)")
    g["_path"] = path
    return g


def lookup_grant(grants, tool, name, mode, cycles, variant="abc"):
    """The recorded grant for one cell, or None (unrecorded, or recorded as a rival failure).
    Returns {"batch", "start_point", "source"}; `start_point` is None for DEMOTIC."""
    if not grants:
        return None
    e = (grants.get(tool) or {}).get(grant_key(name, mode, cycles, variant))
    if not e or not e.get("batch"):
        return None
    return {"batch": int(e["batch"]), "start_point": e.get("start_point"),
            "source": f"pinned from {os.path.basename(grants.get('_path') or 'grants')}"}


def grant_source(pinned, result, source):
    """What the row says about HOW the rung was reached — a pin honoured, a pin this card could
    not finish on (walked down from it), or the plain walk from the ladder top."""
    if not pinned:
        return "walked down from the ladder top"
    got = (result or {}).get("batch")
    if got == pinned:
        return source
    return f"{source}; rung {pinned:,} did not complete on this card, walked down"

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

# THE PAPERS' OWN CONFIGURATIONS — a third source, and the one a referee cites. DEMOTIC
# (arXiv 2502.08086 §IV): "we set the learning rate to 15, the batch size to 500,000, and the
# number of iterations to 10". Recurrent-CircuitSAT (arXiv 2502.21226 §IV): "GD was used as the
# optimizer during training, with a learning rate of 50 and a total of 5 iterations". Neither
# is what their run scripts ship (steps 1, lr 15 for both), and the campaign runs the SCRIPT
# configuration (2026-09-05, the user's call: one configuration, the tool's own shipped one,
# which is also the better of the two for each tool on this card at T=25 — §6). The paper
# values are MEASURED beside it by steps_calibration.py, singly and as the paired setting, so
# the text can state what they would have given the tool instead of asserting it.
RIVAL_PAPER_CONFIG = {
    "demotic":   {"steps": 10, "lr": 15.0, "source": "arXiv 2502.08086 §IV"},
    "recurrent": {"steps": 5,  "lr": 50.0, "source": "arXiv 2502.21226 §IV"},
}

# THE LEARNING RATE, by the same rule — and it is the knob that matters most to a method whose
# whole search IS gradient descent, so leaving it undocumented was the `num_steps` hole one knob
# along. Both rivals' own run scripts say `--lr 15e0` (DEMOTIC script/run_sampler_comb.sh:6,
# Recurrent script/run_sampler_seq.sh), which is the value we run and the same artifact
# RIVAL_STEPS is taken from. Recurrent's PAPER text names 50 instead, so that tool ships two
# values and the tie-break rule applies: whichever is better FOR THE RIVAL, re-derived on the
# campaign's own card by eval/rivals/steps_calibration.py --axis lr.
# THE RIVAL'S SEED, and the fact that there is exactly one of it. Both tools take `--seed`
# (their `src/pytorch/flags.py`, default 0) and neither harness used to pass it — so "the rival
# ran once" was true, but the row could not say at WHICH initialization, and a second run would
# have re-measured the same one without anyone noticing. It is now passed explicitly and recorded.
#
# ONE seed is a deliberate protocol choice (2026-09-04), not an oversight: the rival is run in
# exactly one configuration, at the value its own artifacts default to, and the asymmetry against
# GPAP's median-of-three is STATED rather than hidden. Three rival seeds would cost about an hour
# and a half per campaign and would narrow an interval we do not use — what the tables need from
# the rival is one honest measurement at its own settings.
RIVAL_SEED = 0

RIVAL_LR = {
    # tool:      (lr,  where the value comes from,                    the other value it ships)
    "demotic":   (15.0, "their script/run_sampler_comb.sh", None),
    "recurrent": (15.0, "their script/run_sampler_seq.sh", 50.0),
}


def resolve_lr(tool, requested="auto"):
    """The learning rate this run should use, and where it came from. `auto` = RIVAL_LR."""
    lr, src, _other = RIVAL_LR.get(tool, (15.0, "harness default", None))
    if requested in (None, "auto"):
        return lr, src
    return float(requested), "explicit --lr"


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


# The first GPU run of a process is SLOW — CUDA context, kernel autotuning and clocks ramping —
# and whichever instance goes first eats all of it. Measured on the L4 the day before the
# 2026-09-03 campaign: the first DEMOTIC run of a session is ~3.3x slower than the same run
# later (c432, same unique count, 1.60s vs 0.48s). That campaign fired no warmup, so `c1355`
# — alphabetically first in the DEMOTIC stage — was published at 0.25x of its warm rate.
# One discarded run per (tool, process) fixes it; the probe scripts in the calibration directory
# have always done this and the campaign harness did not.
_WARMED = set()


def warmup_rival(tool, run, batch=256, log=print):
    """Discard one SMALL run before this process measures anything for `tool`. Returns True if fired.

    Called by run_at_batch, so both circuit harnesses get it without remembering to. It must be
    per PROCESS, not per campaign: the Colab supervisor rebuilds the VM on reclaim and the driver
    resumes in a fresh interpreter, so the first cell measured after every resume would otherwise
    pay the cold start again.

    THE WARMUP MUST BE BOUNDED, and the caller does it with a short-timeout `warmup_run`
    (run_at_batch), not with the batch. The cold start this absorbs — CUDA context, clocks
    ramping, the torch import paged in — is a fixed few seconds and batch-INDEPENDENT; it is
    worth a lot on a sub-second cell (c432: 1.60 s vs 0.48 s, 3.3x) and nothing on a cell whose
    one run is minutes. But the rival's per-run cost is O(circuit x T) in GRAPH BUILD, not in
    batch: on s15850.1 at T=50 even a discarded batch-256 run took ~1,030 s (measured, T=50
    appendix run 2026-09-07) — and, re-paid on every ~1 h VM reclaim, it kept the heavy
    ITC'99/ISCAS-89 cells from ever landing. So the warmup runs at a small batch AND is killed
    after WARMUP_TIMEOUT_S: the cold start is long absorbed by then, and the run is discarded, so
    neither the batch nor the cap moves any number. (batch 256, was 10,000 before 2026-09-07.)
    """
    if tool in _WARMED:
        return False
    _WARMED.add(tool)
    t0 = time.perf_counter()
    try:
        run(batch)
    except Exception as e:                                    # noqa: BLE001
        log(f"    [warmup] {tool} warmup raised ({_one_line(e)[:60]}) — continuing")
    log(f"    [warmup] {tool}: discarded run at batch {batch:,} ({time.perf_counter() - t0:.1f}s)")
    return True


def run_at_batch(run, batch, ladder=BATCH_LADDER, log=print, tool=None, probe_budget_s=1200.0,
                 spent=0.0, warmup_run=None):
    """ONE measured run per instance: the LARGEST ladder rung that COMPLETES.

    THE RULE (2026-09-04). Walk DOWN from the top of the ladder; the first rung that runs to
    completion is the measurement. `batch` — the rung our VRAM model predicts — no longer
    decides anything and is recorded beside the sweep as the model's opinion.

    WHY NOT THE MODEL: `GRAPH_BYTES` was measured on a 4080 at `num_steps=10`, and we run
    `num_steps=1`, where the autograd graph is far smaller — so the model predicts LOW. A
    gradient sampler's unique/s is near-proportional to its batch, so predicting low is taking
    the rival's score away. Measured on the L4 (gpap-results/demotic_config_calibration_20260902,
    finding 4): the modelled rung under-grants c880 1.53x, c2670 2.12x, c3540 1.85x, c7552 3.87x.
    The 2026-09-03 campaign shipped exactly that under-grant, because the one-rung-up probe that
    had been masking it was removed the same day for being a best-of-two.

    THIS IS NOT A BEST-OF-N, and that distinction is the whole design: a rung is rejected on
    FEASIBILITY — it OOM'd, the driver refused it, or it did not finish inside the wall budget —
    never on throughput, and exactly one rung is ever measured. Rejecting a rung the rival cannot
    finish with is not charity either: the previous rule recorded that timeout AS the result,
    which is how a granted-too-large cell reads as a rival failure.

    `probe_budget_s` bounds the search: a ladder of timeouts could otherwise cost 8 x the wall
    budget on one instance. When probing has spent that much, the walk jumps straight to the
    modelled rung and the sweep records `probe_truncated`. `spent` is what an INTERRUPTED walk
    on this cell already paid (rivals_recurrent.known_spent), so a walk resumed on a fresh VM
    is charged exactly as one uninterrupted walk would be.

    `warmup_run`, if given, is the closure the discarded warmup fires instead of `run` — used
    to warm at a SHORT timeout so the cold-start absorption never pays a full O(circuit) run
    (see warmup_rival).

    Returns (result, sweep) — every rung tried, so the row records what the rival was granted.
    """
    if tool:
        warmup_rival(tool, warmup_run or run, log=log)
    rungs = list(ladder)
    sweep, spent, truncated = [], float(spent or 0.0), False
    grant = {}                       # what the row must say about HOW the grant was reached
    i = len(rungs) - 1
    while i >= 0:
        t0 = time.perf_counter()
        r = run(rungs[i])
        spent += time.perf_counter() - t0
        entry = {"batch": rungs[i], "unique_per_s": r.get("unique_per_s"),
                 "error": _one_line(r.get("error"))}
        if truncated:
            entry["probe_truncated"] = True
        sweep.append(entry)
        if "error" not in r:
            if rungs[i] > batch:
                log(f"    [batch] granted {rungs[i]:,} — the VRAM model said {batch:,}")
            return dict(r, **grant), sweep
        why = "timed out" if "timeout" in str(r.get("error", "")).lower() else "refused"
        log(f"    [batch] {rungs[i]:,} {why} ({_one_line(r.get('error'))[:60]}) — stepping down")
        if spent > probe_budget_s and not truncated and batch < rungs[i]:
            log(f"    [batch] probe budget {probe_budget_s:.0f}s spent — jumping to the "
                f"modelled rung {batch:,}")
            # THE ROW HAS TO SAY THIS. Truncating falls back to the VRAM model's rung — the very
            # estimator measured under-granting DEMOTIC 1.5-3.9x — so a cell whose oversized rungs
            # fail SLOWLY (timeouts) can settle for a smaller grant than one whose rungs fail fast
            # (OOM), on two circuits of equal capability. The evidence was buried in `batch_sweep`
            # with no consumer; it is now a row-level flag the tables and the gate can count.
            truncated = True
            grant["probe_truncated"] = True
            i = max((k for k, b in enumerate(rungs) if b <= batch), default=0)
            continue
        i -= 1
    # EVERY rung failed. If they all failed for the same reason and that reason is the wall,
    # the cell is bounded by OUR budget rather than by the rival's capability, and those are
    # different findings — the first is re-runnable at a longer timeout, the second is a result.
    if sweep and all("timeout" in str(x.get("error") or "").lower() for x in sweep):
        grant["budget_bound"] = True
    return (dict(sweep[-1], **grant) if sweep else {"error": "empty ladder"}), sweep


# ---------------------------------------------------------------- constraint alignment
# Shared by BOTH circuit harnesses: for one release only the DEMOTIC harness ran it, so the
# headline sequential rows carried no alignment evidence and Table 2's `misaligned` column
# could never fire for them (RIVAL_FAIRNESS_FINDINGS.md §14).

def native_pin_check(orig, reordered, idx, vals, pin_n, T, n_words=256):
    """Verify the native-mode constraint alignment END TO END: pinning the outputs at THEIR
    table indices `idx` on the ORIGINAL file must accept exactly the same input sequences as
    pinning the FIRST pin_n outputs of our REORDERED copy (what GPAP's build_cnf gets).

    Both files share the same primary inputs in the same order, so the same rng feeds them
    identical stimuli and the acceptance words must match BIT FOR BIT — the same style of check
    the miter mode already runs (p_miter == p_accept). This closes the one link that was only
    asserted by construction: negative-index resolution and the reorder itself. The remaining
    cross-parser link — that DEMOTIC's own parser reads OUTPUT() in file order — is verified by
    reading their comb_parser/seq_parser at the pinned commit (outputs.append per OUTPUT line,
    emitted in that order by the generated forward)."""
    import numpy as np
    from gpap.circuits.bench import parse_bench
    from gpap.circuits.simulate import _accept_words, _run_bench, _topo_gates
    co, cr = parse_bench(orig), parse_bench(reordered)
    tgt_o = {co[1][i]: bool(v) for i, v in zip(idx, vals)}
    tgt_r = {w: bool(v) for w, v in zip(cr[1][:pin_n], vals)}
    lo = _run_bench(co, _topo_gates(co), T, n_words, rng=np.random.default_rng(7))
    lr = _run_bench(cr, _topo_gates(cr), T, n_words, rng=np.random.default_rng(7))
    return bool(np.array_equal(_accept_words(lo, tgt_o), _accept_words(lr, tgt_r)))


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


# ---------------------------------------------------------------- how big is the solution space?

def median_run(runs, key="sps"):
    """THE run whose `key` is the median — not a median of each field across runs.

    A row assembled field by field is a composite no execution ever produced: on the 2026-09-03
    campaign 138 field/cell combinations came from a different seed than the row's own rate, so
    Table 3 divided one seed's implication count by another seed's model count. Repeat-runs exist
    to damp noise in ONE number; the honest way to spend them is to pick the run that number
    puts in the middle and report that run.

    Odd n gives the median; even n gives the lower middle, so the result is always a real run.
    """
    order = sorted(runs, key=lambda r: (r.get(key) or 0.0))
    return order[(len(order) - 1) // 2], order


def exact_distinct(bits):
    """The number of DISTINCT support projections among `bits`, exactly.

    THE RULE (2026-09-04): the GPAP arm deduplicates every row the timed run delivered, because
    that is what both circuit rivals do. Recurrent calls `torch.unique(dim=0)` over its whole
    candidate set per depth and counts the survivors that hit the target; DEMOTIC calls
    `torch.unique` over its whole solution list once its timer has stopped. Both dedup EXACTLY,
    over the WHOLE run, off the clock.

    Until now our arm estimated the distinct fraction from a 1,000-row window and applied it to
    runs delivering up to a million models — a window that shows no repeat certifies far less
    than it looks like it does, and the "bias bounded at 7e-4" claim that rested on it bounded
    the wrong quantity (the collision rate INSIDE the window, not the run's distinct fraction).
    With the whole set deduplicated there is no estimator left to bound, and no need for the
    unique/s cap that used to correct it.

    Packed to bits first: a (10^6, 250) uint8 matrix is 250 MB and np.unique over it is slow,
    while the packed (10^6, 32) view is 32 MB and dedups as fixed-width records.
    """
    import numpy as np
    a = np.ascontiguousarray(np.asarray(bits, dtype=np.uint8))
    if a.size == 0:
        return 0
    packed = np.ascontiguousarray(np.packbits(a, axis=1))
    return int(len(np.unique(packed.view([("", packed.dtype)] * packed.shape[1]))))


def unsat_at_depth(cnf_path, conf_budget=2_000_000, provenance=None):
    """Is this pinned instance UNSATISFIABLE at the depth it was built for? True / False / None.

    A cell where BOTH samplers produce nothing is ambiguous: either the task is hard, or it has
    no solution at all. `b01` under Recurrent's own target is the second — its published target
    pattern is unreachable at exactly T=25 (their own sampler runs to 50 and pools every depth)
    — and booking that as a GPAP zero scores us against a solution that does not exist. A cell
    proved UNSAT belongs in its own column, like a vacuous pin: it measures the instance, not
    the sampler.

    Bounded by a conflict budget so a hard-but-satisfiable instance returns None (unknown)
    rather than hanging the sweep. None is NOT "satisfiable" and must not be bucketed as one.
    """
    # WHICH solver said so, and under what budget, belongs in the row: `python-sat` is installed
    # unpinned, so a re-run can resolve a different version — or lose Cadical153 and fall back —
    # and silently move a cell between the `unsat` bucket and `both0` with no trace.
    name = None
    try:
        from pysat.formula import CNF
        from pysat.solvers import Cadical153 as _Solver
        name = "Cadical153"
    except ImportError:
        try:
            from pysat.formula import CNF
            from pysat.solvers import Minisat22 as _Solver
            name = "Minisat22"
        except ImportError:
            if provenance is not None:
                provenance.update(solver=None, conf_budget=conf_budget)
            return None
    if provenance is not None:
        try:
            import pysat
            ver = getattr(pysat, "__version__", "?")
        except Exception:                                      # noqa: BLE001
            ver = "?"
        provenance.update(solver=name, pysat=ver, conf_budget=int(conf_budget))
    try:
        with _Solver(bootstrap_with=CNF(from_file=cnf_path).clauses) as sv:
            sv.conf_budget(int(conf_budget))
            res = sv.solve_limited(expect_interrupt=False)
    except Exception:                                          # noqa: BLE001
        return None
    return None if res is None else (not res)


def space_from_distinct(n_rows, n_distinct):
    """|S| such that `n_rows` uniform draws from |S| points yield `n_distinct` distinct ones in
    expectation, E[d] = S(1 - (1 - 1/S)^n), solved by bisection. None when no repeat was seen
    (d == n): the window then says nothing beyond S >~ n^2/2.

    THE HOLE THIS CLOSES (second referee pass, 2026-09-03): a 1,000-row distinct fraction is a
    narrower deduplication window than a rival's whole-run dedup. On `c1355` under the rare pin
    the window showed 0.56 (|S| ~ 700) while the run delivered ~800k models, so the whole-run
    unique count cannot exceed ~700 and 'samples/s x 0.56' overstated unique/s by ~10^5. The
    `tiny` screen could not catch it either: it read 2^|S| with |S| = 41 input bits, and the
    SOLUTION space, not the input hypercube, is what is enumerated. So the repeat rate the row
    already carries is turned into a size estimate, and both the screen (paper_tables.tiny_space)
    and the estimate (rivals_recurrent.run_gpap's cap of unique/s at |S| per run) read it."""
    n, d = int(n_rows), int(n_distinct)
    if n <= 0 or d <= 0 or d >= n:
        return None
    lo, hi = float(d), 1e15
    for _ in range(200):
        mid = (lo * hi) ** 0.5
        if mid * (1.0 - (1.0 - 1.0 / mid) ** n) < d:
            lo = mid
        else:
            hi = mid
    return hi
