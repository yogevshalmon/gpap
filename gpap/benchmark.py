"""Benchmark the GPU BCP-completion sampler (gpap.sampling) against CMSGen — a thin CLI/table
layer over gpap.sampling.pipeline.sample_file (the shared e2e flow). Lives INSIDE the package so
the installed `gpap bench` entry point works from any directory.

GPAP runs CMSGen's UniformLikeWitness algorithm (random var + random polarity + unit propagation)
massively in parallel on the GPU. This harness compares them with the SAT-sampling literature's
own metrics (Golia et al., FMCAD'21):

  THROUGHPUT  — samples/sec (wall time to produce --n-time samples, including setup).
  WORK RATES  — decisions/s + unit-prop implications/s for BOTH samplers (CMSGen's come from its
                own CryptoMiniSat counters; needs the patched binary, see --cmsgen below).
  QUALITY     — 2-wise coverage %: for each pair of sampling-set vars, the fraction of pairs whose
                all four (±,±) combinations appear across the samples (the combinatorial-testing
                quality metric CMSGen reports). Higher = better solution-space spread.
  SOUNDNESS   — GPAP: GPU check (every model's clauses all have sat_count>0) + optional CPU verify.

### Emission fairness (the CMSGen protocol; read before quoting a ratio)

GPAP's timed run EMITS NOTHING — models stay on the GPU and delivery is a separate, off-timing
pass (docs/BENCHMARKING.md §4). Stock CMSGen instead formats and writes every sample as a full
N-variable DIMACS assignment, which on wide formulas costs MORE than the sampling itself
(measured on 27.sk_3_32, 50k samples: 0.42 s generating vs 3.6 s with --samplefile — 373 MB of
output). Comparing GPAP-without-emission to CMSGen-with-emission measures file I/O, not sampling.

So the driver prefers a CMSGen built with scripts/build_cmsgen.sh (upstream + the small in-repo
patch scripts/cmsgen_gpap_eval.patch) which adds `--nowrite` (generate, emit nothing) and
`--statsevery` (parseable propagation/decision counters). It auto-detects those flags:
  * patched  -> the timed CMSGen run uses --nowrite (like-for-like), and the write-mode rate is
                reported too whenever a samplefile run happens anyway (--n-cov) or --cmsgen-both;
  * stock    -> falls back to the samplefile run, and the row is marked cms_mode="write" so the
                handicap is visible in the JSON rather than silently folded into the ratio.

The glob may mix DIMACS CNFs with circuits (.bench / .aig / .aag): circuits are converted first
(same --encoder / --abc-opt / --cycles / --pin-outputs / --target flags as `gpap sample`; the
encode wall is reported per instance and in the JSON, NOT folded into the sampler's e2e rate —
both samplers consume the identical CNF, so the encode cost is common to the comparison).
`--encoders tseitin abc:fast` runs EVERY circuit through each encoder back-to-back in one pass,
emitting one row per (instance, encoder) — the paired A/B the encoder plots consume.

Usage:
  gpap bench 'benchmarks/sampling/*.cnf' --cmsgen .gpap/bin/cmsgen --n-cov 5000
  gpap bench 'benchmarks/circuits/bench/seq/*.bench' --encoders tseitin abc:fast --target rare
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

# Shared with the `gpap` CLI — INCLUDING the numeric validators. This module owns its own parser
# and eval/run_full_eval.sh invokes it as `python -m gpap.benchmark`, so a bound that lived only in
# gpap/cli.py's wrapper would be absent on exactly the campaign path it was written to protect
# (`--time-limit 0` here is also the rival subprocess's timeout).
from gpap.cli import CIRCUIT_EXTS, _NONNEG_INT, _POS_INT, _POS_SECS, _k_cap_arg

SCHEMA = 3          # results-JSON schema version (docs/BENCHMARKING.md §7)
#   1: original flat rows
#   2: + encode_s / encoder / sim_baseline / per_seed
#   3: + one row per (instance, ENCODER) with `enc_key`; engine provenance (`engine`); CMSGen
#      work-rate stats + emission mode (`cms_*`); error rows; resumable incremental output


# --------------------------------------------------------------------------- CMSGen runner

_GPAP_STATS_RE = re.compile(
    r"c gpap-stats samples=(\d+) props=(\d+) decisions=(\d+) conflicts=(\d+) "
    r"cpu_s=([0-9.]+) wall_s=([0-9.]+)(?: final=([01]))?")
_nowrite_cache: dict = {}
_maxtime_cache: dict = {}


def cmsgen_supports_nowrite(cmsgen_bin: str) -> bool:
    """True iff this binary carries the GPAP eval patch (--nowrite / --statsevery). Cached."""
    if cmsgen_bin not in _nowrite_cache:
        try:
            h = subprocess.run([cmsgen_bin, "--help"], capture_output=True, text=True, timeout=20)
            txt = (h.stdout or "") + (h.stderr or "")
        except Exception:  # noqa: BLE001 — a missing/unrunnable binary is handled by the caller
            txt = ""
        _nowrite_cache[cmsgen_bin] = "--nowrite" in txt and "--statsevery" in txt
        _maxtime_cache[cmsgen_bin] = "--maxtime" in txt
    return _nowrite_cache[cmsgen_bin]


def cmsgen_supports_maxtime(cmsgen_bin: str) -> bool:
    """True iff this binary can enforce its own sampling deadline (--maxtime). Cached.

    Added to the patch after the count-flooring finding; an older patched binary has --nowrite
    and --statsevery but not this, so the two capabilities are probed separately rather than
    assumed together."""
    if cmsgen_bin not in _maxtime_cache:
        cmsgen_supports_nowrite(cmsgen_bin)                     # fills both caches
    return _maxtime_cache.get(cmsgen_bin, False)


def _parse_cms_stats(text: str) -> dict | None:
    """Last `c gpap-stats` line of a CMSGen run -> counters dict.

    `final` is True when that line is the one printed AFTER the sampling loop, which is the only
    one carrying an exact count. The cadence lines are printed every --statsevery samples, so a
    run that is KILLED reports the last multiple of the cadence and silently drops everything
    since — measured on the 2026-08-23 L4 campaign, all 174 killed runs reported an exact
    multiple of 1000 and 81 of them reported 0, i.e. "CMSGen sampled nothing" meant "CMSGen
    sampled fewer than 1000". `--maxtime` (below) exists so that never happens; `final` is how a
    caller proves it did not."""
    last = None
    for m in _GPAP_STATS_RE.finditer(text or ""):
        last = m
    if last is None:
        return None
    n, props, decs, confl, cpu_s, wall_s, final = last.groups()
    return {"samples": int(n), "props": int(props), "decisions": int(decs),
            "conflicts": int(confl), "cpu_s": float(cpu_s), "wall_s": float(wall_s),
            "final": final == "1"}


def cmsgen_run(path, cmsgen_bin, n, scratch, *, keep=None, timeout=60.0, write=True,
               stats_every=1000, seed=None) -> dict:
    """Run CMSGen once and return a dict describing the run.

    write=True  -> stock protocol: samples are written to a samplefile (which is then parsed for
                   the model list the quality pass needs). This INCLUDES the emission cost.
    write=False -> `--nowrite` (patched binaries only): generate and count, emit nothing. The
                   apples-to-apples mode against GPAP's non-emitting timed run; the sample count
                   then comes from the `c gpap-stats` counters instead of the file.

    `keep` caps how many parsed models are retained in memory (the rest are only COUNTED). This is
    essential on low-RAM hosts (a 12 GB Colab): retaining all `n` full models is `n × N` Python
    ints (~8-9 GB at n=50k, N=5k) and OOM-kills the session. None = keep all.

    `timeout` is the SAMPLING budget, the same quantity GPAP's --time-limit bounds (parse and
    construction sit outside both). A patched binary enforces it itself via `--maxtime` and exits
    through its normal path, so the final counters are exact; the subprocess timeout is only a
    backstop for the phases the deadline does not cover (parsing) and therefore allows a grace
    margin on top. On a stock binary there is no deadline flag, so the subprocess kill IS the
    budget and the counters are floored to the last `--statsevery` line.

    Two rates are returned, and they differ only in who is charged for getting ready:
    `samples_per_s` over the SAMPLING wall (CMSGen's steady state, the like-for-like partner of
    GPAP's `samples_per_s`) and `samples_per_s_e2e` over the whole subprocess wall, parse
    included (the partner of GPAP's `sps_e2e_total`). Scoring CMSGen's steady state on its
    subprocess wall while scoring GPAP's on its sampling wall charges one side twice."""
    out = os.path.join(scratch, "cms_" + os.path.basename(path) + ".txt")
    cmd = [cmsgen_bin, "--samples", str(n)]
    patched = cmsgen_supports_nowrite(cmsgen_bin)
    deadline = patched and cmsgen_supports_maxtime(cmsgen_bin) and timeout
    if patched:
        cmd += ["--statsevery", str(stats_every)]
    if deadline:
        cmd += ["--maxtime", f"{timeout:g}"]
    if seed is not None:
        cmd += ["-s", str(seed)]
    if write or not patched:
        cmd += ["--samplefile", out]
    else:
        cmd += ["--nowrite"]
    cmd.append(path)
    t0 = time.perf_counter()
    timed_out = False
    stdout = ""
    # With --maxtime, CMSGen stops itself after `timeout` seconds of SAMPLING; the subprocess
    # cap must therefore leave room for the phases the deadline does not cover (reading a
    # million-clause CNF), or the backstop would fire first and put us back on a floored count.
    hard = timeout + max(30.0, 0.5 * timeout) if deadline else timeout
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=hard)
        stdout = (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        timed_out = True                        # partial output still carries the counters
        stdout = ((e.stdout or "") if isinstance(e.stdout, str)
                  else (e.stdout or b"").decode("utf-8", "replace"))
    except Exception as ex:  # noqa: BLE001
        return {"error": f"{type(ex).__name__}: {ex}", "samples_per_s": 0.0, "models": [],
                "n_samples": 0, "mode": "write" if (write or not patched) else "nowrite"}
    dt = time.perf_counter() - t0
    stats = _parse_cms_stats(stdout)
    models, count = [], 0
    if (write or not patched) and os.path.exists(out):
        for line in open(out):
            lits = [int(x) for x in line.split() if x.lstrip("-").isdigit()]
            if lits and lits[-1] == 0:
                lits = lits[:-1]
            if not lits:
                continue
            count += 1                           # the rate uses the full count, not just kept
            if keep is None or len(models) < keep:
                models.append(lits)
        try:
            os.remove(out)                       # the samplefile can be >1 GB; don't accumulate
        except OSError:
            pass
    elif stats:
        count = stats["samples"]
    # The SAMPLING wall: CMSGen's own measurement of its generation loop, which starts after it
    # has read the CNF. That is the wall GPAP's steady-state rate is measured on, so it is the
    # one the headline ratio must use; `dt` (the whole subprocess) is the e2e partner.
    samp_wall = stats["wall_s"] if (stats and stats.get("wall_s")) else dt
    r = {"samples_per_s": (count / samp_wall if samp_wall else 0.0),
         "samples_per_s_e2e": (count / dt if dt else 0.0),
         "models": models, "n_samples": count,
         "wall_s": dt, "sampling_wall_s": samp_wall, "timed_out": timed_out,
         # The count is exact only when CMSGen printed the post-loop line: either it finished, or
         # its own --maxtime stopped it. A kill leaves the last cadence multiple instead.
         "count_exact": bool(stats and stats.get("final")) and not timed_out,
         "budget_stop": "c gpap-maxtime" in stdout,
         "mode": "write" if (write or not patched) else "nowrite", "patched": patched}
    if stats:
        # CMSGen's own counters over its own sampling wall -> the CPU numbers our dec/s and
        # impl/s face. implications = propagations - decisions (same definition as the GPAP
        # columns). Same wall as `samples_per_s`, so Table 3's identity
        # (samples/s ratio = impl/s ratio / impl-per-sample ratio) stays exact.
        r["stats"] = stats
        w = samp_wall
        r["decisions_per_s"] = stats["decisions"] / w if w else 0.0
        r["props_per_s"] = stats["props"] / w if w else 0.0
        r["implications_per_s"] = max(0, stats["props"] - stats["decisions"]) / w if w else 0.0
        r["conflicts"] = stats["conflicts"]
    return r


def cmsgen_sample(path, cmsgen_bin, n, scratch, keep=None, timeout=60.0):
    """Back-compat wrapper (eval/colab/colab_sampling_benchmark.py imports this): stock write-mode run,
    returning (samples_per_s, models)."""
    r = cmsgen_run(path, cmsgen_bin, n, scratch, keep=keep, timeout=timeout, write=True)
    return r["samples_per_s"], r["models"]


# --------------------------------------------------------------------------- encoder specs

def parse_encoder_specs(tokens, default_encoder, default_abc_opt):
    """['tseitin', 'abc:fast'] -> [('tseitin', None), ('abc', 'fast')].

    A bare 'abc' inherits --abc-opt. The returned order is the run order; every circuit is
    converted+sampled once per spec (back-to-back, same process, same GPU state — the paired A/B
    the encoder plots want)."""
    if not tokens:
        return [(default_encoder, default_abc_opt if default_encoder != "tseitin" else None)]
    specs = []
    for tok in tokens:
        enc, _, opt = tok.partition(":")
        if enc not in ("tseitin", "abc", "auto"):
            raise SystemExit(f"unknown encoder {enc!r} in --encoders (tseitin|abc|auto[:opt])")
        if enc == "tseitin":
            specs.append(("tseitin", None))
        else:
            specs.append((enc, opt or default_abc_opt))
    return specs


def enc_key(encoder, abc_opt):
    return encoder if encoder == "tseitin" else f"{encoder}:{abc_opt}"


# --------------------------------------------------------------------------- driver

def _build_argparser():
    ap = argparse.ArgumentParser(prog="gpap bench")
    ap.add_argument("--glob", default="benchmarks/sampling/*.cnf",
                    help="instances: DIMACS .cnf and/or circuits (.bench/.aig/.aag, converted "
                         "first with the encoder flags below)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--cmsgen", default=None,
                    help="path to a cmsgen binary (enables the comparison). Prefer one built by "
                         "scripts/build_cmsgen.sh: its --nowrite makes the emission-free "
                         "comparison possible and --statsevery yields CMSGen's own work rates")
    ap.add_argument("--cmsgen-both", action="store_true",
                    help="with a patched cmsgen, ALSO time the write-mode (samplefile) run even "
                         "when no quality pass needs the models — the emission-cost appendix")
    ap.add_argument("--cmsgen-seed", type=int, default=None, help="cmsgen -s seed")
    # circuit-conversion flags (identical to `gpap sample` / `gpap circuit2cnf`)
    from gpap.cli import _add_encoder_args
    _add_encoder_args(ap)
    ap.add_argument("--encoders", nargs="+", default=None, metavar="ENC",
                    help="run every circuit through EACH of these encoders, back to back, one "
                         "row per (instance, encoder): e.g. --encoders tseitin abc:fast "
                         "(tokens: tseitin | abc[:full|fast|auto] | auto). Default: the single "
                         "--encoder/--abc-opt pair")
    ap.add_argument("--cycles", type=_POS_INT, default=25,
                    help="unroll depth for circuit inputs (combinational AIGs clamp to 1)")
    ap.add_argument("--pin-outputs", type=_NONNEG_INT, default=None,
                    help="outputs to pin for circuit inputs (default: all)")
    ap.add_argument("--seed", type=int, default=5, help="conversion + sampling seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="run the timed sampling at EACH of these seeds and report the "
                         "median-throughput run per instance (docs/BENCHMARKING.md: single-seed "
                         "rates are draw-noisy up to ±40%% on low-success instances). The "
                         "conversion/target seed stays --seed so every run samples the SAME CNF; "
                         "CMSGen still runs once (CDCL rates are not draw-noisy the way "
                         "trajectory sampling is). Adds a ±%% spread column; per-seed raw "
                         "numbers land in the JSON")
    ap.add_argument("--mine", choices=["auto", "on", "off"], default="auto",
                    help="entailed-clause mining: 'auto' (on, except ABC-encoded CNFs — their "
                         "backbone is already gone) | 'on' (force) | 'off'")
    ap.add_argument("--uniformity", choices=["off", "fast", "exact"], default="off",
                    help="per-instance uniformity readout: 'off' (default — costs an extra "
                         "sampling pass per instance) | 'fast' (repeat-rate check, gross skew "
                         "only, no solver work) | 'exact' (chi-square on enumerable spaces)")
    ap.add_argument("--twise", action="store_true",
                    help="also compute 2-wise coverage (the CMSGen paper's spread metric, "
                         "O(S^2) — opt-in; auto-enables n_cov=2000 if --n-cov unset)")
    ap.add_argument("--sim-baseline", action="store_true",
                    help="(circuit instances) measure the rejection-sampling rival per instance "
                         "(random inputs + bit-parallel simulation, no CNF): p_acc column = "
                         "acceptance probability of the pinned target, rej/s = accepted "
                         "samples/s it achieves on this CPU")
    ap.add_argument("--n-time", type=_POS_INT, default=50_000,
                    help="samples to PRODUCE in the timed run (a contract, not a hint). Only the "
                         "safety valves end the run short of this ('<' after sols): --time-limit, "
                         "or zero models after every config was measured (hopeless instance)")
    ap.add_argument("--time-limit", type=_POS_SECS, default=60.0,
                    help="per-instance sampling wall cap in seconds (the safety valve that bounds "
                         "very-low-success instances; e.g. at 1k samples/s, 200k samples would "
                         "otherwise take 200s)")
    ap.add_argument("--max-chunks", type=_POS_INT, default=None,
                    help="expert knob: hard chunk cap for the timed run (default none — --n-time "
                         "and --time-limit govern)")
    ap.add_argument("--n-cov", type=_NONNEG_INT, default=0,
                    help="samples for the QUALITY pass (2-wise coverage + distinct count). 0 (default) "
                         "SKIPS it: the collect run's Python model-reconstruction dominated instance "
                         "wall (measured 26s of 34s on s1196 T=25) while distinct≈100%% on all non-tiny "
                         "circuits. Pass e.g. 5000 for the paper-quality metrics")
    ap.add_argument("--portfolio", choices=["circuit", "default", "b8", "b1", "s1"],
                    default="circuit",
                    help="bandit config portfolio: 'circuit' (the CircuitSAT focus, default) or "
                         "'default' (the sk_ projected-sampling portfolio). 'b8'/'b1'/'s1' pin "
                         "ONE config and disable the bandit — use those for any cross-GPU or "
                         "before/after MEASUREMENT, since the bandit optimises models/wall with "
                         "no stability term and can pick differently per machine (measured: the same "
                         "instance chose a different config on two GPUs), which silently "
                         "confounds the comparison")
    ap.add_argument("--allow-soft", action="store_true",
                    help="add the soft risk-polarity config to the portfolio. Higher success on "
                         "tight/deep circuits but NOT uniform (Barbarik rejects it on 64 of 65 circuits) — never "
                         "use for a distribution claim")
    ap.add_argument("--b", type=_POS_INT, default=8,
                    help="decisions per round (8 is the throughput sweet spot: fewer rounds while "
                         "success barely drops; B>=16 over-constrains and collapses success)")
    ap.add_argument("--k-cap", type=_k_cap_arg, default="auto",
                    help="ceiling on the per-chunk batch K. 'auto' (default) is the library's "
                         "cap, identical on every GPU "
                         "(throughput is ~linear in K); no-op at or below the reference "
                         "budget. Pass an int to FIX the cap — what a cross-GPU A/B needs, "
                         "since 'auto' deliberately differs per card")
    ap.add_argument("--no-simplify", dest="simplify", action="store_false", default=True)
    ap.add_argument("--no-els", action="store_true",
                    help="disable equivalent-literal substitution (ON by default: sound and "
                         "model-preserving, and a solid geomean win on circuit instances). Pass "
                         "this to reproduce a run made before it became the default)")
    ap.add_argument("--also-plain", action="store_true", help="also run GPAP without simplification")
    ap.add_argument("--verify-cpu", action="store_true", help="also CPU-verify a few models vs clauses")
    # OFF, like every other entry point: a soundness re-check is a self-test, not part of
    # measuring throughput, and a row is only counted when it is fully assigned with no clause
    # emptied. `eval/run_full_eval.sh` turns it on for the campaign, which is where the extra
    # coverage is worth its wall (excluded from every rate, but it grows with the formula).
    ap.add_argument("--verify-sound", action="store_true",
                    help="re-check on the GPU that every emitted model satisfies every clause")
    ap.add_argument("--max-clauses", type=_POS_INT, default=None,
                    help="skip any instance whose CNF exceeds this many clauses (recorded as a "
                         "skipped row with the size). The engine itself only rejects formulas "
                         "outside the packed envelope — this is the WALL-TIME guard for a "
                         "sweep: a 50-cycle b15 unroll parses for minutes before it samples")
    ap.add_argument("--max-vars", type=_POS_INT, default=None,
                    help="same, on variables (the packed-word envelope caps 2N at 16.7M anyway)")
    ap.add_argument("--scratch", default="/tmp")
    ap.add_argument("--keep-cnf", action="store_true",
                    help="keep each converted circuit CNF in --scratch after its row finishes. "
                         "Default: delete it — a full matrix (encoders x targets x depths) of "
                         "deep unrolls is tens of GB, which fills a Colab disk mid-sweep")
    ap.add_argument("--output", default=None,
                    help="results JSON. Written INCREMENTALLY (after every row) so a dropped "
                         "Colab session keeps everything finished so far")
    ap.add_argument("--resume", action="store_true",
                    help="skip (instance, encoder) rows already present in --output and append "
                         "to it — re-run the same command after an interruption")
    ap.add_argument("--tag", default=None, help="free-form label stored in the results JSON")
    return ap


def _row_key(row):
    return (row.get("name"), row.get("enc_key"), row.get("cycles"), row.get("target"))


def _dimacs_header(path):
    """(num_vars, num_clauses) from the `p cnf N M` line — a few bytes, no clause parse."""
    try:
        with open(path) as f:
            for line in f:
                s = line.strip()
                if s.startswith("p "):
                    parts = s.split()
                    if len(parts) >= 4:
                        return int(parts[2]), int(parts[3])
                    return None, None
                if s and s[0] not in "c%":
                    break                          # no header (rare) — let the parse decide
    except OSError:
        pass
    return None, None


def main(argv=None):
    import numpy as np
    import torch
    from gpap.sampling.metrics import distinct_projections, format_uniformity, two_wise_coverage
    from gpap.sampling.pipeline import soft_warning, describe_portfolio, portfolio as _portfolio, sample_file

    ap = _build_argparser()
    args = ap.parse_args(argv)
    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    files = sorted(glob.glob(args.glob))
    if not files:
        print(f"[warn] no instances matched {args.glob!r} — nothing to run "
              f"(is the glob quoted so the shell did not expand it?)", file=sys.stderr)
    portfolio = _portfolio(args.portfolio, allow_soft=args.allow_soft)
    if args.allow_soft:
        print(soft_warning())
    if args.twise and args.n_cov == 0:
        args.n_cov = 2000                          # 2-wise needs the collected quality sample
    quality = args.n_cov > 0
    mine = {"auto": "auto", "on": True, "off": False}[args.mine]
    specs = parse_encoder_specs(args.encoders, args.encoder, args.abc_opt)
    os.makedirs(args.scratch, exist_ok=True)       # converted CNFs + CMSGen samplefiles land here
    show_unif = args.uniformity != "off"           # gates the unif column
    has_circuits = any(f.endswith(CIRCUIT_EXTS) for f in files)
    multi_enc = len(specs) > 1
    cms_patched = bool(args.cmsgen) and cmsgen_supports_nowrite(args.cmsgen)
    if args.cmsgen and not cms_patched:
        print("[warn] this cmsgen has no --nowrite (stock build): its rate INCLUDES writing every "
              "sample to a file, which GPAP's timed run does not do. Build the patched binary "
              "with scripts/build_cmsgen.sh for the like-for-like number.", file=sys.stderr)

    # ---- resume / incremental output -------------------------------------------------------
    results, done = [], set()
    if args.output and args.resume and os.path.exists(args.output):
        try:
            prev = json.load(open(args.output))
            results = prev.get("results", [])
            done = {_row_key(r) for r in results}
            print(f"[resume] {len(results)} row(s) already in {args.output}")
            # Resuming a file written by a DIFFERENT engine silently mixes provenance in one
            # result set. Rows carry engine.core since the libgpap port (absent = the retired
            # torch/Triton-JIT engine), so say so rather than let it into a summary unnoticed.
            from gpap.sampling.pipeline import _core_provenance
            here = _core_provenance()
            stale = sorted({(r.get("engine") or {}).get("core") or "pre-libgpap (torch engine)"
                            for r in results
                            if (r.get("engine") or {}).get("core") not in (None, here)
                            or ((r.get("engine") or {}).get("core") is None and r.get("engine"))})
            if stale:
                print(f"[resume] WARNING: {args.output} contains rows from another engine "
                      f"({', '.join(stale)}); this run appends {here}. Delete the file to "
                      f"re-measure everything on one engine.", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"[resume] could not read {args.output} ({e}); starting fresh", file=sys.stderr)

    def _save():
        if not args.output:
            return
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        payload = {"schema": SCHEMA, "device": str(device), "tag": args.tag,
                   "gpu": _gpu_info(device), "cpu": _cpu_info(), "args": vars(args),
                   "encoders": [enc_key(e, o) for e, o in specs],
                   "cmsgen_patched": cms_patched, "results": results}
        tmp = args.output + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=1)
        os.replace(tmp, args.output)               # atomic: a kill never truncates the results

    enc_note = (f" | encoders={','.join(enc_key(e, o) for e, o in specs)} cycles={args.cycles} "
                f"target={args.target}" if has_circuits else "")
    cms_note = ("no" if not args.cmsgen else
                ("yes [patched: nowrite]" if cms_patched else "yes [STOCK: writes samples]"))
    print(f"  portfolio [{args.portfolio}]: {describe_portfolio(portfolio)}")
    print(f"device={device} | simplify={args.simplify} | portfolio={args.portfolio} "
          f"| n_time={args.n_time:,} (limit {args.time_limit:g}s/inst) n_cov={args.n_cov:,} "
          f"| cmsgen={cms_note}{enc_note}\n{len(files)} benchmarks x {len(specs)} encoder(s)\n")

    hdr = f"{'instance':<24}"
    if multi_enc:
        hdr += f"{'enc':>10}"
    # Widths sized for the real range: rates reach 6e7 samples/s and sols 1e7 on shallow
    # circuits, and a column that overflows silently glues itself to its neighbour.
    hdr += (f"{'N':>7}{'M':>8}{'|sup|':>6}{'GPAP/s':>13}{'e2e/s':>13}"
            f"{'dec/s':>9}{'impl/s':>9}{'succ%':>6}{'sols':>12}{'t_slv':>7}{'t_tot':>7}")
    if args.seeds:
        hdr += f"{'±%':>5}"
    if has_circuits:
        hdr += f"{'t_enc':>7}"
    if quality:
        hdr += f"{'uniq/s~':>10}"
    if args.twise:
        hdr += f"{'2wiseG':>8}"
    if show_unif:
        hdr += f"{'unif':>9}"
    if args.sim_baseline:
        hdr += f"{'p_acc':>9}{'rej/s':>11}{'amp':>9}"
    hdr += f"{'krn':>5}{'snd':>4}"
    if args.cmsgen:
        hdr += f"{'CMS/s':>10}{'Cdec/s':>9}{'Cimpl/s':>9}{'GPAP/CMS':>10}{'e2e/CMS':>9}"
        if args.twise:
            hdr += f"{'2wiseC':>7}"
    print(hdr)
    print("-" * len(hdr))

    t_suite = time.perf_counter()
    total_sols = 0
    for path0 in files:
        name = os.path.basename(path0)
        is_circuit = path0.endswith(CIRCUIT_EXTS)
        for encoder, abc_opt in (specs if is_circuit else [(None, None)]):
            ek = enc_key(encoder, abc_opt) if is_circuit else None
            key = (name, ek, args.cycles if is_circuit else None,
                   args.target if is_circuit else None)
            if key in done:
                continue
            row = {"name": name, "enc_key": ek, "encoder": encoder, "abc_opt": abc_opt,
                   "cycles": args.cycles if is_circuit else None,
                   "target": args.target if is_circuit else None,
                   "seed": args.seed, "n_time": args.n_time, "time_limit": args.time_limit}
            try:
                r = _run_one(path0, row, args, device, portfolio, mine, quality,
                             sample_file, two_wise_coverage, distinct_projections)
            except KeyboardInterrupt:
                raise
            except Exception as e:  # noqa: BLE001 — one bad instance must never kill the suite
                row["error"] = f"{type(e).__name__}: {e}"
                _drop_cnf(row, args)
                results.append(row)
                _save()
                lbl = f"{name[:22]}" + (f" [{ek}]" if multi_enc else "")
                print(f"{lbl:<24} ERROR: {row['error'][:90]}", file=sys.stderr, flush=True)
                continue
            _drop_cnf(r, args)
            results.append(r)
            _save()
            if r.get("skipped"):
                lbl = f"{name[:22]}" + (f" [{ek}]" if multi_enc else "")
                print(f"{lbl:<24} SKIPPED: {r['skipped']}", flush=True)
                continue
            total_sols += r["n_models"]
            print(_format_row(r, args, multi_enc, has_circuits, quality, show_unif,
                              format_uniformity), flush=True)

    suite_wall = time.perf_counter() - t_suite
    good = [r for r in results if not r.get("error") and not r.get("skipped")]
    n_err = sum(1 for r in results if r.get("error"))
    n_skip = sum(1 for r in results if r.get("skipped"))
    print(f"\nTOTAL: {total_sols:,} solutions over {len(good)} rows in {suite_wall:,.1f}s "
          f"suite wall (solver e2e {sum(r['wall_total_s'] for r in good):,.1f}s, of which "
          f"timed sampling {sum(r['wall_timed_s'] for r in good):,.1f}s)"
          + (f"  [{n_err} error, {n_skip} skipped]" if (n_err or n_skip) else ""))
    _print_summaries(good, args, [enc_key(e, o) for e, o in specs] if multi_enc else None, np)
    _save()
    if args.output:
        print(f"wrote {args.output}")
    return results


def _drop_cnf(row, args):
    """Delete the CNF this cell generated (unless --keep-cnf). A full circuit matrix is tens of
    GB of intermediate DIMACS; on a Colab disk that is the difference between finishing and
    dying at 80%. The path stays in the row for provenance."""
    p = row.get("cnf_path")
    if args.keep_cnf or not p or p == row.get("name") or not row.get("encode_s"):
        return                                     # not ours to delete (plain-CNF input)
    try:
        os.remove(p)
    except OSError:
        pass


def _cpu_info():
    """Host CPU provenance — REQUIRED to quote any GPAP/CMSGen ratio.

    CMSGen is single-core CPU, so the ratio moves with the HOST as much as with the GPU. The
    same instance set has been measured at very different ratios on two machines purely because
    their CPUs differ — CMSGen's own propagation and decision counters confirm the slower host is
    the cause. A sweep that records the GPU but not the CPU cannot tell "our GPU got faster" from
    "their CPU got slower", and a reviewer reading a large ratio without the CPU alongside would
    be right to call it a rigged baseline (docs/BENCHMARKING.md)."""
    info = {"count": os.cpu_count()}
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    info["model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        import platform
        info["machine"] = platform.machine()
    except Exception:      # noqa: BLE001 — provenance must never break a sweep
        pass
    return info


def _gpu_info(device):
    import torch
    if device.type != "cuda":
        return {"name": "cpu"}
    p = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    return {"name": p.name, "total_gb": round(p.total_memory / 1e9, 2),
            "free_gb": round(free / 1e9, 2), "sms": p.multi_processor_count,
            "capability": f"{p.major}.{p.minor}", "torch": torch.__version__,
            "cuda": torch.version.cuda}


def _run_one(path0, row, args, device, portfolio, mine, quality,
             sample_file, two_wise_coverage, distinct_projections):
    """One (instance, encoder) cell: convert -> optional rejection baseline -> GPAP (x seeds)
    -> optional CMSGen. Returns the result row. Raises only on unexpected failures (the caller
    turns those into an error row); expected 'cannot run this instance' conditions (packed
    envelope, conversion failure) come back as a `skipped` row."""
    from gpap.cli import _convert_circuit
    name = row["name"]
    path = path0
    sim = None
    if path0.endswith(CIRCUIT_EXTS):
        cnf = os.path.join(args.scratch, f"{os.path.splitext(name)[0]}-c{args.cycles}"
                           f"-{row['enc_key'].replace(':', '_')}-{args.target}.cnf")
        conv = argparse.Namespace(**vars(args))    # per-encoder view of the conversion flags
        conv.encoder, conv.abc_opt = row["encoder"], row["abc_opt"]
        _t0 = time.perf_counter()
        try:
            _convert_circuit(path0, cnf, conv, quiet=True)
        except Exception as e:  # noqa: BLE001 — a conversion failure skips the CELL, not the run
            row["skipped"] = f"convert failed: {type(e).__name__}: {e}"
            return row
        row["encode_s"] = time.perf_counter() - _t0
        if args.sim_baseline:
            # The rejection-sampling rival at the SAME cycles/seed/pin/target (simulate.py).
            from gpap.circuits.simulate import sim_baseline
            sim = sim_baseline(path0, args.cycles, args.seed, pin_outputs=args.pin_outputs,
                               target_mode=args.target)
        path = cnf
    row["cnf_path"] = path

    # Packed-envelope pre-check (cheap, header only): the sum-trick word cannot represent every
    # formula — 2N bounds its literal-SUM field and the max clause length bounds its COUNT field
    # (gpap.engine.packed_envelope). The var count is readable from the DIMACS header without a
    # parse, and it is the condition a real instance actually hits; the clause-length half is
    # caught below by the engine's own raise. Either way the CELL is skipped with a reason
    # recorded — a suite sweep must not die on one oversized unroll.
    from gpap.engine import packed_envelope
    nv_hdr, m_hdr = _dimacs_header(path)
    if nv_hdr:
        row["nv"], row["M"] = nv_hdr, m_hdr
        if packed_envelope(nv_hdr, 1)[0] is None:
            row["skipped"] = (f"formula outside the packed envelopes: 2N={2 * nv_hdr} "
                              f"(max {0xFFFFFF}) — needs a wider packed word")
            return row
        if args.max_clauses and m_hdr and m_hdr > args.max_clauses:
            row["skipped"] = f"{m_hdr:,} clauses > --max-clauses {args.max_clauses:,}"
            return row
        if args.max_vars and nv_hdr > args.max_vars:
            row["skipped"] = f"{nv_hdr:,} vars > --max-vars {args.max_vars:,}"
            return row

    # --seeds: one full timed run per seed; the ROW is the median-throughput run (a real,
    # self-consistent run — not a per-metric mix), the others land in the JSON as per_seed.
    runs = []
    for s in (args.seeds or [args.seed]):
        try:
            g = sample_file(path, device, simplify=args.simplify, els=not args.no_els,
                            n_time=args.n_time, n_cov=args.n_cov, b=args.b,
                            k_cap=args.k_cap, verify_cpu=args.verify_cpu,
                            verify_sound=args.verify_sound,
                            configs=portfolio, max_chunks=args.max_chunks, seed=s,
                            mine=mine, uniformity=args.uniformity, time_limit=args.time_limit)
        except ValueError as e:                    # the engine's packed-envelope rejection
            if "packed envelope" not in str(e):
                raise
            row["nv"], row["M"] = nv_hdr, m_hdr
            row["skipped"] = str(e)
            return row
        g["seed"] = s
        runs.append(g)
    runs.sort(key=lambda r: r["samples_per_s"])
    g = runs[len(runs) // 2]
    spread = None
    if len(runs) > 1 and g["samples_per_s"]:
        spread = (runs[-1]["samples_per_s"] - runs[0]["samples_per_s"]) / (2 * g["samples_per_s"])
    sup1 = g["sup1"]                       # metadata from the pipeline's single parse
    row.update({
        "nv": g["nv"], "M": g["n_clauses"], "support": len(sup1),
        "gpap_sps": g["samples_per_s"], "gpap_sps_e2e": g.get("sps_e2e_total"),
        "n_models": g["n_models"], "wall_timed_s": g["wall_timed_s"],
        "wall_total_s": g["wall_total_s"], "chunk_capped": g.get("chunk_capped"),
        "uniformity": g.get("uniformity"), "sim_baseline": sim,
        # "ok" | "UNSOUND" | "off" — see gpap.sampling.sampler. NOT a bool: "off" (the check
        # was not run) must not read as either verdict.
        "succ": g["success_rate"], "sound_check": g.get("sound_check", "off"),
        "unsound_rows": g.get("unsound_rows"), "cpu_sound": g.get("cpu_sound"),
        "n_distinct": g["n_distinct"], "n_collected": g.get("n_collected"),
        "unique_sps_est": g.get("unique_sps_est"), "assigns_per_s": g.get("assigns_per_s"),
        "decisions_per_s": g.get("decisions_per_s"),
        "implications_per_s": g.get("implications_per_s"), "mined": g.get("mined"),
        "phases": g.get("phases"), "config_breakdown": g.get("config_breakdown"),
        # WHICH kernel/packed split actually ran (sum-trick provenance) + the calibrated K.
        "engine": g.get("engine"),
        "tuned": [g.get("tuned_rv"), g.get("tuned_b"), g.get("tuned_la")],
    })
    row["twise_gpap"] = two_wise_coverage(g["models"], sup1) if args.twise else None
    if args.seeds:
        row["median_seed"] = g["seed"]
        row["sps_spread_frac"] = spread
        row["per_seed"] = [{"seed": r["seed"], "sps": r["samples_per_s"],
                            "sps_e2e": r.get("sps_e2e_total"), "succ": r["success_rate"],
                            "n_models": r["n_models"], "wall_timed_s": r["wall_timed_s"]}
                           for r in runs]
    if sim is not None and sim.get("hits"):
        row["amplification"] = ((g["success_rate"] / sim["p_accept"]) if sim["p_accept"] else 0.0)
    elif sim is not None:
        row["amplification"] = None
    if args.also_plain:
        plain = sample_file(path, device, simplify=False, n_time=args.n_time, n_cov=0,
                            b=args.b, k_cap=args.k_cap, configs=portfolio,
                            max_chunks=args.max_chunks, time_limit=args.time_limit)
        row["gpap_plain_sps"] = plain["samples_per_s"]

    # ---- CMSGen (same per-instance wall budget as GPAP's --time-limit) ---------------------
    if args.cmsgen:
        patched = cmsgen_supports_nowrite(args.cmsgen)
        # The TIMED run: emission-free when the binary supports it (like GPAP's timed run). On a
        # stock binary it is the samplefile run, so keep the models the quality pass needs from
        # it (a patched build gets them from the separate write run below instead).
        timed = cmsgen_run(path, args.cmsgen, args.n_time, args.scratch,
                           keep=(0 if patched else (args.n_cov or 1)),
                           timeout=args.time_limit, write=not patched,
                           seed=args.cmsgen_seed)
        cmodels = timed.get("models") or []
        row["cms_mode"] = timed["mode"]
        row["cms_sps"] = timed["samples_per_s"]                 # over CMSGen's SAMPLING wall
        row["cms_sps_e2e"] = timed.get("samples_per_s_e2e")     # over its whole subprocess wall
        row["cms_n_samples"] = timed["n_samples"]
        row["cms_timed_out"] = timed.get("timed_out")
        # True = the count is CMSGen's own final number, not the last --statsevery multiple
        # before a kill. A False here means `cms_n_samples` is a LOWER BOUND, so it must never
        # be published as an exact zero (verify_artifact fails the stage on it).
        row["cms_count_exact"] = timed.get("count_exact")
        row["cms_budget_stop"] = timed.get("budget_stop")
        row["cms_wall_s"] = timed.get("wall_s")
        row["cms_sampling_wall_s"] = timed.get("sampling_wall_s")
        row["cms_decisions_per_s"] = timed.get("decisions_per_s")
        row["cms_implications_per_s"] = timed.get("implications_per_s")
        row["cms_props_per_s"] = timed.get("props_per_s")
        row["cms_conflicts"] = timed.get("conflicts")
        if timed.get("error"):
            row["cms_error"] = timed["error"]
        # A samplefile run when the quality pass needs models (or --cmsgen-both): its rate is the
        # WITH-EMISSION number — reported separately, never used for the headline ratio.
        if patched and (quality or args.cmsgen_both):
            wr = cmsgen_run(path, args.cmsgen, args.n_time, args.scratch,
                            keep=(args.n_cov or 1), timeout=args.time_limit, write=True,
                            seed=args.cmsgen_seed)
            cmodels = wr.get("models") or []
            row["cms_sps_write"] = wr["samples_per_s"]
            # >1 = how much throughput CMSGen loses by emitting its samples (the handicap every
            # published GPAP-vs-stock-CMSGen ratio silently contained).
            row["cms_write_penalty"] = (timed["samples_per_s"] / wr["samples_per_s"]
                                        if wr["samples_per_s"] else None)
        elif not patched:
            row["cms_sps_write"] = timed["samples_per_s"]   # stock: the timed run IS write-mode
        gsps, crate = g["samples_per_s"], row["cms_sps"]
        ce2e = row.get("cms_sps_e2e") or 0.0
        # The two ratios pair LIKE WITH LIKE: steady state against steady state, and end-to-end
        # against end-to-end. Both sides' preparation is either in or out; charging GPAP for its
        # parse while CMSGen's rate was scored on a wall containing its own parse (the earlier
        # `ratio_e2e`) moved only one side of the fraction.
        # ratio: inf = GPAP-only, 0 = CMSGen-only, None = neither produced in the budget
        row["ratio"] = (gsps / crate) if crate else (float("inf") if gsps else None)
        row["ratio_e2e"] = ((((g.get("sps_e2e_total") or 0) / ce2e) if ce2e
                             else (float("inf") if gsps else None)))
        if quality and cmodels:
            cd, cn = distinct_projections(cmodels[:args.n_cov], sup1)
            row["cms_distinct"], row["cms_distinct_of"] = cd, cn
        if args.twise:
            row["twise_cms"] = two_wise_coverage(cmodels[:args.n_cov], sup1) if cmodels else None
    return row


def _format_row(r, args, multi_enc, has_circuits, quality, show_unif, format_uniformity):
    eng = r.get("engine") or {}
    line = f"{r['name'][:22]:<24}"
    if multi_enc:
        line += f"{(r.get('enc_key') or '—'):>10}"
    dps = f"{r['decisions_per_s']:.1e}" if r.get("decisions_per_s") else "—"
    ips = f"{r['implications_per_s']:.1e}" if r.get("implications_per_s") else "—"
    # '+' after succ% = entailed-clause mining ran; '<' after sols = a safety valve
    # (--time-limit / zero-progress) stopped the run before it delivered --n-time samples.
    mn = "+" if r.get("mined") else " "
    sols_s = f"{r['n_models']:,}" + ("<" if r.get("chunk_capped") else "")
    line += (f"{r['nv']:>7}{r['M']:>8}{r['support']:>6}"
             f"{r['gpap_sps']:>13,.0f}{r.get('gpap_sps_e2e') or 0:>13,.0f}{dps:>9}{ips:>9}"
             f"{100*r['succ']:>5.0f}%{mn}{sols_s:>11}"
             f"{r['wall_timed_s']:>6.1f}s{r['wall_total_s']:>6.1f}s")
    if args.seeds:
        sp = r.get("sps_spread_frac")
        line += f"{100 * sp:>4.0f}%" if sp is not None else f"{'—':>5}"
    if has_circuits:
        line += f"{r['encode_s']:>6.1f}s" if r.get("encode_s") is not None else f"{'—':>7}"
    if quality:
        uq = f"{r['unique_sps_est']:,.0f}" if r.get("unique_sps_est") else "—"
        line += f"{uq:>10}"
    if args.twise:
        line += f"{(r.get('twise_gpap') or 0):>7.1f}%"
    if show_unif:
        line += f"{format_uniformity(r.get('uniformity')):>9}"
    if args.sim_baseline:
        sim = r.get("sim_baseline")
        if sim is None:
            line += f"{'—':>9}{'—':>11}{'—':>9}"
        elif sim["hits"]:
            amp = r.get("amplification") or 0.0
            amps = f"{amp:,.0f}x" if amp < 1e6 else f"{amp:.0e}x"
            line += f"{sim['p_accept']:>9.1e}{sim['accepted_per_s']:>11,.0f}{amps:>9}"
        else:                                      # no hit seen: rejection is unusable here
            line += f"{'<' + format(sim['p_accept_lt'], '.0e'):>9}{'~0':>11}{'inf':>9}"
    # kernel provenance: w/s = wave/scalar propagate, 16 or 8 = the packed split's SHIFT
    krn = f"{eng.get('mode', '?')[0]}{eng.get('shift', '')}" if eng else "—"
    # Three outcomes, three glyphs: "-" is "not checked", which is neither OK nor BAD.
    _sc = r.get("sound_check")
    if _sc is None:                            # a row recorded before the key existed
        _sc = ("ok" if r["gpu_sound"] else "UNSOUND") if "gpu_sound" in r else "off"
    snd = ("BAD" if _sc == "UNSOUND" or r.get("cpu_sound") is False else
           "OK" if _sc == "ok" else "-")
    line += f"{krn:>5}{snd:>4}"
    if args.cmsgen:
        cd = f"{r['cms_decisions_per_s']:.1e}" if r.get("cms_decisions_per_s") else "—"
        ci = f"{r['cms_implications_per_s']:.1e}" if r.get("cms_implications_per_s") else "—"
        ratio, ratio_e = r.get("ratio"), r.get("ratio_e2e")
        rtxt = "—" if ratio is None else (f"{ratio:.2f}x" if ratio != float("inf") else "GPAP")
        rtxt_e = "—" if ratio_e is None else (f"{ratio_e:.2f}x" if ratio_e != float("inf") else "GPAP")
        line += f"{r.get('cms_sps') or 0:>10,.0f}{cd:>9}{ci:>9}{rtxt:>10}{rtxt_e:>9}"
        if args.twise:
            line += f"{(r.get('twise_cms') or 0):>6.1f}%"
    return line


def _print_summaries(rows, args, enc_order, np):
    import statistics as st

    def _gm(xs):
        xs = [x for x in xs if x and 0 < x != float("inf")]
        return st.geometric_mean(xs) if xs else None

    if args.cmsgen:
        ratios = [r.get("ratio") for r in rows]
        fin = [x for x in ratios if x and 0 < x != float("inf")]
        gm = _gm(ratios) or 0.0
        gm_e2e = _gm([r.get("ratio_e2e") for r in rows]) or 0.0
        n_gpap_only = sum(1 for x in ratios if x == float("inf"))
        n_cms_only = sum(1 for x in ratios if x == 0)
        n_neither = sum(1 for x in ratios if x is None)
        mode = {r.get("cms_mode") for r in rows if r.get("cms_mode")}
        mtxt = ("emission-free (--nowrite)" if mode == {"nowrite"}
                else "STOCK, includes writing every sample" if mode == {"write"} else str(mode))
        print(f"geomean GPAP/CMSGen throughput = {gm:.2f}x steady-state, {gm_e2e:.2f}x "
              f"end-to-end incl. setup+mining, over {len(fin)} both-produced rows "
              f"(+{n_gpap_only} GPAP-only, {n_cms_only} CMSGen-only, {n_neither} neither)  "
              f"(GPAP=GPU, CMSGen=single-core CPU, {mtxt})")
        pen = [r["cms_write_penalty"] for r in rows if r.get("cms_write_penalty")]
        if pen:
            print(f"  CMSGen emission cost: writing the samples costs it {_gm(pen):.2f}x "
                  f"throughput (geomean over {len(pen)} rows) — the like-for-like ratio above "
                  f"uses the emission-free rate")
        dq = [r["twise_gpap"] - r["twise_cms"] for r in rows
              if r.get("twise_cms") is not None and r.get("twise_gpap") is not None]
        if dq:
            print(f"mean 2-wise coverage: GPAP "
                  f"{np.mean([r['twise_gpap'] for r in rows if r.get('twise_gpap') is not None]):.1f}% "
                  f"vs CMSGen {np.mean([r['twise_cms'] for r in rows if r.get('twise_cms') is not None]):.1f}% "
                  f"(Δ {np.mean(dq):+.1f}pp)")
    if enc_order:
        # Paired encoder A/B: only instances that produced a row under EVERY encoder count, and
        # the BASE is the first --encoders token (the run order), not an alphabetical accident.
        by_inst = {}
        for r in rows:
            by_inst.setdefault(r["name"], {})[r.get("enc_key")] = r
        present = {r.get("enc_key") for r in rows if r.get("enc_key")}
        encs = [e for e in enc_order if e in present]
        if len(encs) < 2:
            return
        base = encs[0]
        print(f"\nencoder A/B (paired, base = {base}):")
        for e in encs[1:]:
            pairs = [(v[base], v[e]) for v in by_inst.values() if base in v and e in v]
            if not pairs:
                continue
            sps = _gm([b["gpap_sps"] / a["gpap_sps"] for a, b in pairs if a["gpap_sps"]])
            e2e = _gm([(b.get("gpap_sps_e2e") or 0) / (a.get("gpap_sps_e2e") or 1)
                       for a, b in pairs if a.get("gpap_sps_e2e")])
            shrink = _gm([a["M"] / b["M"] for a, b in pairs if b["M"]])
            enc_a = sum(a.get("encode_s") or 0 for a, _ in pairs)
            enc_b = sum(b.get("encode_s") or 0 for _, b in pairs)
            wins = sum(1 for a, b in pairs if b["gpap_sps"] > a["gpap_sps"])
            print(f"  {e:<12} vs {base}: {sps or 0:.2f}x steady, {e2e or 0:.2f}x e2e, "
                  f"{shrink or 0:.2f}x fewer clauses, wins {wins}/{len(pairs)}, "
                  f"encode {enc_a:.1f}s -> {enc_b:.1f}s (total over {len(pairs)} instances)")
            if args.cmsgen:
                cms = _gm([(b.get("cms_sps") or 0) / (a.get("cms_sps") or 1)
                           for a, b in pairs if a.get("cms_sps")])
                if cms:
                    print(f"  {'':<12} (CMSGen gains {cms:.2f}x on the same CNFs — the encoder "
                          f"helps both sides; ratio shift = {(sps or 0) / cms:.2f}x)")


if __name__ == "__main__":
    main()
