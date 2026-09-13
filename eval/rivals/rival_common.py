"""Shared pieces of the GPU-rival head-to-heads (eval/rivals_{demotic,recurrent,hts}.py).

THE FAIRNESS POLICY LIVES HERE, three rules:
  * Circuit variants — `prepare_variant` hands BOTH sides the same circuit (`raw` as committed,
    or `abc`: the SEQUENTIAL AIG through `strash; zero; dc2; dc2; fraig`, written back as a
    primitive-gate .bench), gated on a bit-exact simulation check over EVERY primary output,
    since comparing only the acceptance word is vacuous at a rare pin.
  * The parallelism grant — each rival gets the largest batch fitting the same share of the card
    GPAP sizes its own K with (see "the parallelism grant" below).
  * The GPAP-arm cache — a rival-side re-run costs only the rival (see "GPAP arm cache").

Stated limit: the rivals get the SEQUENTIAL optimization only, since GPAP's post-unroll `abc`
pass cannot be handed to a tool that unrolls for itself. Measured nearly free on these benches.
"""
import functools
import os
import subprocess
import time

VARIANTS = ("raw", "abc")


def rival_commit(path: str):
    """HEAD of the rival checkout at `path`, or None: the row must carry what actually ran, since
    a rivals_setup.sh pin can be overridden or a checkout moved by hand."""
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
    unavailable or the check fails — a comparison must never run on a different circuit."""
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
    # WHICH half of the check had content, recorded per row: the acceptance-word half is vacuous
    # at a rare target (both words all-zero), so OUTPUT equality is what gates the swap.
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
# Most of a rival stage's wall is OUR side (the GPAP arm, the start_point probes, sim_baseline),
# and all of it is re-paid verbatim when a stage is re-run for a RIVAL-side reason. So the arm is
# cached on disk, keyed by everything that can change its answer: the bytes of the file sampled,
# the task, the sampling knobs, the ENGINE BUILD and the GPU. Any of those moving is a MISS,
# which is the point — a stale GPAP number is worse than a re-run. `GPAP_RIVAL_CACHE=0` disables.

def _canonical(obj):
    """Stable JSON for hashing: sorted keys, no whitespace, tuples as lists."""
    import json
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


@functools.lru_cache(maxsize=1)
def _core_build_id() -> str:
    """Fingerprint of the libgpap BUILD — the bytes, not the version it advertises: the core's
    own `gpap_build_id()`, not a second implementation. Do not key the cache on `gpap_version()`,
    a constant that makes a REBUILT engine a HIT — exactly the case the key exists to miss."""
    try:
        from gpap.engine import _core
        build = _core.build_id()
    except Exception:                       # noqa: BLE001 — no engine, no GPAP arm to cache
        return "unknown"
    # The core reports "<version>+unknown" when it cannot locate its own file (static link,
    # stripped dl info) — a STABLE string across every rebuild, i.e. the constant key this
    # helper exists to avoid. Hash the library file ourselves instead.
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
    # EVERY file that decides what a GPAP arm returns: the ARM itself lives in the harness files.
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
    """Identity of the `abc` binary that encodes the arm's CNF, as its content hash. `--version`
    would be weaker (ABC's version string moves far less often than its behaviour); a missing
    binary hashes to "absent", itself a distinguishing value."""
    import hashlib
    try:
        from gpap import paths
        # The GPAP-BUILT abc. NOT necessarily the one that encoded: `abc_backend.find_abc`
        # prefers an explicit path, then $GPAP_ABC, then PATH, so where abc comes from PATH this
        # hashes "absent" and the key stops identifying the encoder. Open: the fix invalidates
        # every recorded cache entry (docs/findings/RIVAL_FAIRNESS_FINDINGS.md §4).
        exe = str(paths.tool("abc"))
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
    # ONE engine-identity field, `gpap_build_id()`: two names for one string invite a cleanup.
    env["build"] = _core_build_id()
    # ...AND the PYTHON sampling layer, which moves the number with the build id untouched.
    env["sampling_py"] = _sampling_layer_id()
    # ...AND THE ENCODER BINARY: the CNF is a function of `abc` as much as of the .bench bytes.
    env["abc"] = _abc_fingerprint()
    # ...AND THE VRAM CEILING, which decides K and therefore the rate this key stands for. Without
    # it an uncapped run and a capped one shared a cache entry, so `reproduce.sh --vram-cap l4`
    # could serve numbers measured with the cap OFF while printing the cap in its provenance —
    # the grown-K inflation the cap exists to prevent, wearing the cap's own label.
    env["vram_cap_mib"] = int(_vram_cap_mib())
    try:
        import torch
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:                       # noqa: BLE001
        env["gpu"] = "unknown"
    return hashlib.sha1(_canonical(env).encode()).hexdigest()


CACHE_MISS = object()          # distinct from a stored None


def cache_peek(cache_dir, sampled_file, task):
    """Read-only lookup: the stored value, or `CACHE_MISS`. Separate from cached_gpap because a
    caller that only wants to KNOW must not write a placeholder — a `lambda: None` compute would
    store None and turn every later lookup into a false hit."""
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
    Returns (result, hit); the record keeps the task beside the value so a cache directory can
    be read by a human."""
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


def _vram_cap_mib() -> int:
    """The GPAP_VRAM_BUDGET_MIB ceiling in MiB, 0 for none — for RECORDING it in a row and in the
    cache key. The clamp itself is the core's, inside gpap_mem_available. Imported lazily because
    the table generators import this module for `space_from_distinct` and must stay free of torch
    and the engine (docs/findings/ARTIFACT_FINDINGS.md §7); every caller of this one is a rival
    harness, which has libgpap loaded anyway."""
    from gpap.engine.calibrate import vram_budget_cap_mib
    return vram_budget_cap_mib()


# ------------------------------------------------------------------ the parallelism grant
# Measured: each rival scores in near-PROPORTION to the batch it is handed, over the whole range
# we can run, while `model_runtime` barely moves (docs/findings/RIVAL_FAIRNESS_FINDINGS.md). The
# batch is not a detail of their setup: it IS the parallelism grant, the knob K is for us, and
# GPAP sizes K against `gpap_mem_available`. The rival is sized against THE SAME CALL, so the two
# sides get the same bytes of the same card and one GPAP_VRAM_BUDGET_MIB reproduces both. A
# constant instead gives the two sides different fractions of the machine, and a different
# fraction on every card. Override with GPAP_RIVAL_VRAM_FRACTION (or an explicit budget) when
# reproducing an older sweep.

# ---------------------------------------------------------------- oversize pre-check
# THE RIVAL PATH CANNOT USE abc:fast: a circuit-level rival consumes .bench and unrolls it
# itself, so our arm must encode THE SAME circuit — Tseitin, far larger, which can exceed the
# packed 8|24 cap and, worse, exhaust host memory so the process is KILLED with no exception and
# no row. So the size is checked BEFORE anything is built, from the .bench alone. The bound is a
# GATE COUNT: it bounds the BUILD, not the envelope.
MAX_UNROLLED_GATES = 4_000_000

# The SECOND term is the box's own RAM, because the first is a fixed number and the machines are
# not: an OOM kill has no exception to catch and no row to show for it. Peak RSS of one arm is
# linear in unrolled gates to ~15% (docs/findings/RIVAL_FAIRNESS_FINDINGS.md). Refuse at 60% of
# AVAILABLE memory, not total: our torch process and the rival's are resident together.
PEAK_GIB_FIXED = 1.4
PEAK_GIB_PER_MGATE = 3.3
HOST_MEM_FRACTION = 0.60


def _available_bytes():
    """Available (not free) host RAM, or None when it cannot be read — the memory term is then
    not applied, since refusing an instance on a guess is worse than attempting it."""
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

    Both terms model the NATIVE TSEITIN build, so they apply only when that is the encoder: under
    `abc:fast` the unroll happens inside `abc` and the CNF is 2-20x smaller, so a
    Tseitin-calibrated refusal would omit instances the campaign has already run."""
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
RIVAL_VRAM_FALLBACK = 5 << 30                           # no CUDA visible: a sane batch, not zero


def rival_vram_budget(fraction: float = None) -> int:
    """Bytes of GPU memory a rival's autograd graph may occupy — THE SAME NUMBER, from the same
    function, that the engine grants itself: `gpap_mem_available`, i.e. (free - the driver
    reserve) x fraction, then capped by GPAP_VRAM_BUDGET_MIB.

    IT IS ONE BUDGET ON PURPOSE. This used to be a second implementation — `free x 0.80` with no
    reserve — which read 3.4% higher than ours on the same card, so "the same share of the same
    card" was true of the fraction and false of the bytes, and a single cap could not reproduce
    both sides at once. There is now nothing to keep in step: change the model in
    cpp/src/calibrate.cpp and both sides move.

    Falls back to 5 GiB, still under the ceiling, when no CUDA device is visible, so a CPU-only
    smoke run still picks a sane batch. `GPAP_RIVAL_VRAM_FRACTION` still overrides the fraction
    for reproducing an older sweep; it feeds the same function."""
    if fraction is None:
        fraction = float(os.environ.get("GPAP_RIVAL_VRAM_FRACTION", RIVAL_VRAM_FRACTION))
    try:
        import torch

        from gpap.engine.calibrate import gpu_mem_available
        if torch.cuda.is_available():
            return gpu_mem_available(torch.device("cuda", 0), fraction)
        # No card to measure means no core call to carry the ceiling, so this branch applies it
        # itself — the same reader, not a second one. It was missed once because the branch a
        # developer box takes is not the branch CI takes: with the clamp only inside
        # `gpap_mem_available`, a cap on a CPU-only machine bounded our side and left the rival
        # at the full fallback, which is exactly the asymmetry the cap exists to remove.
        cap = _vram_cap_mib() << 20
        return min(RIVAL_VRAM_FALLBACK, cap) if cap else RIVAL_VRAM_FALLBACK
    except Exception:                                   # noqa: BLE001 — sizing must never be fatal
        pass
    return RIVAL_VRAM_FALLBACK


# ---------------------------------------------------------------- the instance sets
# THE SET EACH RIVAL IS RUN AND SCORED ON IS ITS OWN PAPER'S, not a list we picked. ONE home: the
# harnesses expand the instance name `paper` to it and `eval/artifact/paper_tables.py` imports
# it, so the sweep and the table cannot drift.
#   DEMOTIC   (arXiv 2502.08086 Table II): all 11 ISCAS-85, c17 included.
#   Recurrent (arXiv 2502.21226 Table II): 8 ISCAS-89 + 8 ITC'99, its 8th ISCAS-89 row `s38584`.
#             Their code covers more circuits than their paper reports and the paper is what this
#             lists; `s35932` is a circuit WE add, scored outside their 16. b17/b19/b22 are too
#             large to commit — scripts/fetch_benchmarks.sh itc99 installs them.
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
# there, and a top rung that completes first try means the LADDER bound the grant, not the card,
# a fourth rejection reason the "feasibility only" rule does not state
# (docs/findings/RIVAL_FAIRNESS_FINDINGS.md §16e, §17). `paper_tables` names such cells.
BATCH_LADDER = (1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000,
                10_000_000, 30_000_000)


# ------------------------------------------------------------------ recorded grants
# THE GRANT IS PART OF THE RECORD (protocol v1.0). A campaign's rows say which rung each rival
# was granted and which `start_point` Recurrent settled on; `eval/rivals/extract_grants.py`
# writes them per campaign and `--grants FILE` PINS them — no walk, no probe, one run at the
# recorded rung. A cell recorded as a rival FAILURE is re-measured by default; the artifact's
# reproduction may pin it instead, opt-in per harness (`--honour-recorded-failures`,
# `lookup_recorded_failure`), and every row that took that shortcut says so.

def grant_key(name, mode, cycles, variant="abc"):
    """One cell: circuit | task | unroll depth | circuit variant."""
    return f"{name}|{mode}|{int(cycles)}|{variant}"


def pinned_ladder(ladder, pinned):
    """The ladder a RECORDED grant turns the walk into: the pinned rung on top, the rungs below
    kept, so `run_at_batch` descends only if THIS card cannot finish on the pin."""
    pinned = int(pinned)
    return tuple(sorted({b for b in ladder if b < pinned} | {pinned}))


def load_grants(path):
    """The grants file, or {} when no path is given. A malformed file is an error, never a
    silent empty pin: a campaign walking when it believes it is pinning looks right in every row."""
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


def lookup_recorded_failure(grants, tool, name, mode, cycles, variant="abc"):
    """The recorded FAILURE for one cell, or None. Opt-in (`--honour-recorded-failures`), never
    the harness default, because it asserts a rival outcome instead of measuring it.

    Why it exists: re-deriving the campaign's 28 recorded failures costs each one the full
    initialization budget, 9.3 h of a 19 h reproduction, to produce rows that say "no
    measurement". An artifact reviewer may reasonably decline to pay that.

    Why it is NOT the default: a failure is "did not settle inside 1,200 s ON THAT CARD", and a
    faster card might settle. Honouring it can therefore only be unfair to the RIVAL, which is
    exactly why the row records `not re-measured` and the comparator refuses to count such a
    cell as agreement.

    The CALLER must also check that the recorded failure was produced under the settings in
    force now — `rivals_recurrent.settled_error` — because a run that raised the initialization
    budget is asking to give the rival more time, and replaying the old failure would answer a
    different question under the new flag."""
    if not grants:
        return None
    e = (grants.get(tool) or {}).get(grant_key(name, mode, cycles, variant))
    if not e or e.get("batch") or not e.get("error"):
        return None
    return {"error": e["error"],
            "grant_source": f"recorded failure pinned from "
                            f"{os.path.basename(grants.get('_path') or 'grants')}; "
                            f"NOT re-measured on this card",
            "not_remeasured": True}


def grant_source(pinned, result, source):
    """What the row says about HOW the rung was reached — a pin honoured, a pin this card could
    not finish on (walked down from it), or the plain walk from the ladder top."""
    if not pinned:
        return "walked down from the ladder top"
    got = (result or {}).get("batch")
    if got == pinned:
        return source
    return f"{source}; rung {pinned:,} did not complete on this card, walked down"

# Bytes of real peak VRAM per unit of "work", MEASURED rather than modelled — the naive
# `n_gates * T * batch * 4` undercounts by 2-6x, tool-dependently. Use the factors AS MEASURED,
# not rounded up: the budget already keeps 20% of the card in reserve, and rounding up costs
# whole ladder rungs. The driver still gets the last word, since a refused rung steps down.
GRAPH_BYTES = {"demotic": 4 * 5.0, "recurrent": 4 * 1.9}     # per gate x frame x row
CNF_BYTES_PER_ROW_CLAUSE = 8.4                                # HTS, per clause x row


def fit_batch(work_units, tool, budget_bytes=None, ladder=BATCH_LADDER):
    """The largest ladder rung that fits the rival inside its share of the card.

    `work_units` is gates x frames for the circuit rivals, clauses for HTS. THE RULE: each rival
    gets the batch that fits the budget `gpap_mem_available` reports — not the same rule as the
    one GPAP sizes K with, the SAME CALL, so neither side can be granted a different share of
    the card and one GPAP_VRAM_BUDGET_MIB reproduces both. Their own "defaults"
    cannot serve as that rule: all three argparse defaults are `--batch_size 10`, a placeholder,
    and their demo scripts are per-instance one-liners with a 1000x spread on comparably tiny
    circuits, which transplanted here would inflate OUR ratios by the same factor."""
    per_row = max(work_units, 1) * (CNF_BYTES_PER_ROW_CLAUSE if tool == "hts"
                                    else GRAPH_BYTES.get(tool, 4 * 5.5))
    budget = budget_bytes or rival_vram_budget()
    for b in reversed(ladder):
        if per_row * b <= budget:
            return b
    return ladder[0]


# --------------------------------------------------------------------------- gradient steps
# `--num_steps` is the rivals' OTHER first-order knob (gradient steps per batch) and, unlike the
# batch, is not a share of the machine but a tool setting with an author-chosen value.
# THE RULE: run each rival at the value its OWN artifacts specify; where a tool ships two, take
# the one better FOR THE RIVAL — a tie-break that can only move a ratio against us. Resolved by
# measurement and re-verified per campaign by steps_calibration.py; it is per tool because
# DEMOTIC and Recurrent saturate their batch on the first step while HTS needs the descent.
RIVAL_STEPS = {
    # tool:      (steps, where the value comes from,                  the other value we rejected)
    "demotic":   (1,  "their script/run_sampler_comb.sh", 10),
    "recurrent": (1,  "their script/run_sampler_seq.sh", 10),
    "hts":       (10, "their src/pytorch/flags.py default", 5),
}

# THE PAPERS' OWN CONFIGURATIONS — a third source, and the one a referee cites. Neither is what
# the run scripts ship, and the campaign runs the SCRIPT configuration (one configuration, the
# tool's own, also the better of the two for each tool on this card at T=25 —
# docs/findings/RIVAL_FAIRNESS_FINDINGS.md §6). The paper values are MEASURED beside it by
# steps_calibration.py, singly and as the paired setting, so the text can state them.
RIVAL_PAPER_CONFIG = {
    "demotic":   {"steps": 10, "lr": 15.0, "source": "arXiv 2502.08086 §IV"},
    "recurrent": {"steps": 5,  "lr": 50.0, "source": "arXiv 2502.21226 §IV"},
}

# THE LEARNING RATE, by the same rule — the knob that matters most to a method whose whole search
# IS gradient descent. Both rivals' run scripts say `--lr 15e0`, the artifact RIVAL_STEPS comes
# from; Recurrent's PAPER names 50, so that tool ships two and the tie-break applies.
# THE RIVAL'S SEED, and that there is exactly ONE of it: both tools take `--seed` (default 0),
# now passed and recorded. One seed is a deliberate protocol choice, and the asymmetry against
# GPAP's median-of-three is STATED, not hidden.
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
    Recorded per row as `steps` / `steps_source`, so a table can state which setting each rival
    ran at instead of implying they all ran at one."""
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


# The first GPU run of a process is SLOW (CUDA context, autotuning, clocks ramping) and whichever
# instance goes first eats all of it; one discarded run per (tool, process) fixes it.
_WARMED = set()


def warmup_rival(tool, run, batch=256, log=print):
    """Discard one SMALL run before this process measures anything for `tool`. Returns True if fired.

    Called by run_at_batch, so both circuit harnesses get it without remembering to. Per PROCESS,
    not per campaign: a supervisor that rebuilds the VM resumes in a fresh interpreter.
    THE WARMUP MUST BE BOUNDED, and the caller does that with a short-timeout `warmup_run`, not
    with the batch: the cold start is a fixed few seconds and batch-INDEPENDENT, while a rival's
    per-run cost is O(circuit x T) in GRAPH BUILD. The run is discarded either way."""
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

    THE RULE: walk DOWN from the top of the ladder; the first rung that runs to completion is the
    measurement. `batch`, the rung our VRAM model predicts, decides nothing and is recorded as
    the model's opinion — `GRAPH_BYTES` was measured at `num_steps=10` while we run 1, so it
    predicts LOW, and a gradient sampler's unique/s is near-proportional to its batch.

    NOT A BEST-OF-N: a rung is rejected on FEASIBILITY — it OOM'd, the driver refused it, or it
    did not finish inside the wall budget — never on throughput, and one rung is ever measured.

    `probe_budget_s` bounds the search: past it the walk jumps to the modelled rung and records
    `probe_truncated`. `spent` is what an INTERRUPTED walk already paid. Returns (result, sweep),
    every rung tried, so the row records what the rival was granted."""
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
            # THE ROW HAS TO SAY THIS: truncating falls back to the VRAM model's rung, so a cell
            # whose oversized rungs fail SLOWLY can settle for a smaller grant than one whose
            # rungs fail fast, on two circuits of equal capability.
            truncated = True
            grant["probe_truncated"] = True
            i = max((k for k, b in enumerate(rungs) if b <= batch), default=0)
            continue
        i -= 1
    # EVERY rung failed. All failing for the same reason, and that reason being the wall, means
    # the cell is bounded by OUR budget rather than the rival's capability: a re-runnable finding.
    if sweep and all("timeout" in str(x.get("error") or "").lower() for x in sweep):
        grant["budget_bound"] = True
    return (dict(sweep[-1], **grant) if sweep else {"error": "empty ladder"}), sweep


# ---------------------------------------------------------------- constraint alignment
# Shared by BOTH circuit harnesses (docs/findings/RIVAL_FAIRNESS_FINDINGS.md §14).

def native_pin_check(orig, reordered, idx, vals, pin_n, T, n_words=256):
    """Verify the native-mode constraint alignment END TO END: pinning the outputs at THEIR table
    indices `idx` on the ORIGINAL file must accept exactly the same input sequences as pinning
    the FIRST pin_n outputs of our REORDERED copy (what GPAP's build_cnf gets). Both files share
    the same primary inputs in the same order, so identical stimuli must give BIT-IDENTICAL
    acceptance words — closing the link otherwise asserted by construction."""
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

    The converged protocol (docs/findings/RIVAL_FAIRNESS_FINDINGS.md §14) compares three tools on
    one instance: the rival on its own circuit, GPAP on the `abc:fast` CNF of that circuit and
    those pins, and CMSGen on that same CNF FILE. Same runner as `gpap bench`
    (gpap.benchmark.cmsgen_run): the patched binary timed under `--nowrite` (emission-free, like
    our `collect=False` run) with `--maxtime` (it stops itself and prints an EXACT count; a
    killed run reports a floored one), at the same budget the GPAP arm gets. The keys are the
    bench driver's own `cms_*` names, so every reader of a bench row reads a rival row unchanged.
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
    """`cpu` + `gpu` on every rival row, the bench driver's own dicts. A GPAP/CMSGen ratio is not
    interpretable without the CPU — CMSGen is single-core and hosts differ by more than a factor
    of two on it alone (docs/BENCHMARKING.md). `vram_cap_mib` joins them because it changes what
    BOTH sides were granted, so a row measured under a cap is not comparable to one without."""
    from gpap.benchmark import _cpu_info, _gpu_info
    try:
        gpu = _gpu_info(device)
    except Exception:                       # noqa: BLE001 — provenance must never break a sweep
        gpu = {"name": str(device)}
    out = {"cpu": _cpu_info(), "gpu": gpu}
    cap = _vram_cap_mib()
    if cap > 0:
        out["vram_cap_mib"] = cap
    return out


# ---------------------------------------------------------------- how big is the solution space?

def median_run(runs, key="sps"):
    """THE run whose `key` is the median — not a median of each field across runs, which is a
    composite no execution ever produced (Table 3 would divide one seed's implication count by
    another seed's model count). Odd n gives the median, even n the lower middle, so the result
    is always a real run."""
    order = sorted(runs, key=lambda r: (r.get(key) or 0.0))
    return order[(len(order) - 1) // 2], order


def exact_distinct(bits):
    """The number of DISTINCT support projections among `bits`, exactly.

    THE RULE: the GPAP arm deduplicates every row the timed run delivered, because that is what
    both circuit rivals do — each calls `torch.unique` over its whole candidate set, off the
    clock. Do not estimate the fraction from a window: one showing no repeat certifies far less
    than it looks like it does. Packed to bits first, so np.unique dedups fixed-width records."""
    import numpy as np
    a = np.ascontiguousarray(np.asarray(bits, dtype=np.uint8))
    if a.size == 0:
        return 0
    packed = np.ascontiguousarray(np.packbits(a, axis=1))
    return int(len(np.unique(packed.view([("", packed.dtype)] * packed.shape[1]))))


def unsat_at_depth(cnf_path, conf_budget=2_000_000, provenance=None):
    """Is this pinned instance UNSATISFIABLE at the depth it was built for? True / False / None.

    A cell where BOTH samplers produce nothing is ambiguous: either the task is hard, or it has
    no solution at all. A cell proved UNSAT belongs in its own column, like a vacuous pin — it
    measures the instance, not the sampler. Bounded by a conflict budget, so a hard-but-
    satisfiable instance returns None (unknown) rather than hanging the sweep, and None is NOT
    "satisfiable"."""
    # WHICH solver said so, and under what budget, belongs in the row: `python-sat` is installed
    # unpinned, so a re-run can resolve a different version — or fall back — and silently move a
    # cell between the `unsat` bucket and `both0` with no trace.
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

    A narrow distinct-fraction window can overstate whole-run unique/s by orders of magnitude,
    and the `tiny` screen cannot catch it because it reads the input hypercube, not the SOLUTION
    space; both that screen and the arm's unique/s cap read this instead."""
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
