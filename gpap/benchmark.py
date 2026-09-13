"""Benchmark the GPU sampler against CMSGen — a CLI/table layer over
gpap.sampling.pipeline.sample_file, inside the package so the installed `gpap bench` works from
any directory. Columns follow the SAT-sampling literature (Golia et al., FMCAD'21): samples/s,
decisions/s and implications/s for both samplers, 2-wise coverage, GPAP soundness.

Emission fairness (docs/BENCHMARKING.md §4, read before quoting a ratio): GPAP's timed run emits
nothing, while stock CMSGen writes every sample as a full N-variable assignment, which on wide
formulas costs more than the sampling. The driver therefore prefers a CMSGen built with
scripts/build_cmsgen.sh (upstream + scripts/cmsgen_gpap_eval.patch), adding `--nowrite` and
`--statsevery`; patched times the emission-free run, stock falls back to the samplefile run and
marks the row cms_mode="write" so the handicap stays visible. The glob may mix DIMACS CNFs with
circuits (.bench / .aig / .aag), which are converted first with the same flags as `gpap sample`;
the encode wall is reported per instance but folded into neither sampler's rate, and
`--encoders tseitin abc:fast` gives one row per (instance, encoder).
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

# Shared with the `gpap` CLI, validators included: this module owns its own parser and is also
# invoked as `python -m gpap.benchmark`, so a bound living only in gpap/cli.py would be absent on
# the campaign path (`--time-limit 0` here is also the rival subprocess's timeout).
from gpap.cli import CIRCUIT_EXTS, _NONNEG_INT, _POS_INT, _POS_SECS, _k_cap_arg
from gpap.sampling.pipeline import merge_sound_check, sound_verdict
from gpap.verdict import PCT_WIDTH, fmt_pct

#: The succ% column: one wider than the widest rate, so a full-width "<0.01%" still has a space
#: between it and impl/s. Header and row read the same constant, or they drift apart.
_SUCC_W = PCT_WIDTH + 1

SCHEMA = 4          # results-JSON schema version (docs/BENCHMARKING.md §9)
#   1: flat rows;  2: + encode_s / encoder / sim_baseline / per_seed
#   3: + one row per (instance, ENCODER) with `enc_key`; engine provenance; CMSGen work-rate
#      stats + emission mode (`cms_*`); error rows; resumable incremental output
#   4: - `uniformity` (the readout moved to eval/uniformity/; schema-3 files still read)



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
    """True iff this binary can enforce its own sampling deadline (--maxtime). Cached. Probed
    separately from --nowrite/--statsevery: an older patched binary has those but not this."""
    if cmsgen_bin not in _maxtime_cache:
        cmsgen_supports_nowrite(cmsgen_bin)
    return _maxtime_cache.get(cmsgen_bin, False)


def _parse_cms_stats(text: str) -> dict | None:
    """Last `c gpap-stats` line of a CMSGen run -> counters dict. `final` is True when that line
    is the one printed AFTER the sampling loop, the only one with an exact count: cadence lines
    print every --statsevery samples, so a KILLED run reports the last cadence multiple and a
    nonzero count can floor to 0. `--maxtime` avoids that, and `final` proves it did."""
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
    """Run CMSGen once and return a dict describing the run. `write=True` is the stock protocol
    (samples to a samplefile, parsed for the quality pass; the rate includes the emission cost);
    `write=False` uses `--nowrite` on a patched binary — generate and count, emit nothing, the
    like-for-like partner of GPAP's non-emitting timed run — taking the count from
    `c gpap-stats`. `keep` caps how many parsed models stay in memory; None keeps all and
    OOM-kills a low-RAM host. `timeout` is the SAMPLING budget, the quantity GPAP's --time-limit
    bounds: a patched binary enforces it via `--maxtime` and exits normally so the counters are
    exact, while on a stock binary the subprocess kill IS the budget and the counters floor to
    the last `--statsevery` line. Returns `samples_per_s` over the SAMPLING wall (the partner of
    GPAP's `samples_per_s`) and `samples_per_s_e2e` over the whole subprocess wall, parse
    included (the partner of `sps_e2e_total`)."""
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
    # With --maxtime, CMSGen stops itself after `timeout` seconds of SAMPLING, so the subprocess
    # cap must leave room for what the deadline does not cover (reading a huge CNF).
    hard = timeout + max(30.0, 0.5 * timeout) if deadline else timeout
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=hard)
        stdout = (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        timed_out = True
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
            count += 1
            if keep is None or len(models) < keep:
                models.append(lits)
        try:
            os.remove(out)                       # the samplefile can be >1 GB; don't accumulate
        except OSError:
            pass
    elif stats:
        count = stats["samples"]
    # The SAMPLING wall: CMSGen's own measure of its generation loop, the partner of GPAP's
    # steady-state rate; `dt` (whole subprocess) is the e2e partner.
    samp_wall = stats["wall_s"] if (stats and stats.get("wall_s")) else dt
    r = {"samples_per_s": (count / samp_wall if samp_wall else 0.0),
         "samples_per_s_e2e": (count / dt if dt else 0.0),
         "models": models, "n_samples": count,
         "wall_s": dt, "sampling_wall_s": samp_wall, "timed_out": timed_out,
         # Exact only when CMSGen printed the post-loop line (it finished, or --maxtime stopped
         # it); a kill leaves the last cadence multiple instead.
         "count_exact": bool(stats and stats.get("final")) and not timed_out,
         "budget_stop": "c gpap-maxtime" in stdout,
         "mode": "write" if (write or not patched) else "nowrite", "patched": patched}
    if stats:
        # CMSGen's counters over its own sampling wall, so implications = propagations -
        # decisions as in the GPAP columns and Table 3's identity stays exact.
        r["stats"] = stats
        w = samp_wall
        r["decisions_per_s"] = stats["decisions"] / w if w else 0.0
        r["props_per_s"] = stats["props"] / w if w else 0.0
        r["implications_per_s"] = max(0, stats["props"] - stats["decisions"]) / w if w else 0.0
        r["conflicts"] = stats["conflicts"]
    return r


def cmsgen_sample(path, cmsgen_bin, n, scratch, keep=None, timeout=60.0):
    """Back-compat wrapper (eval/colab/colab_sampling_benchmark.py imports it): stock write-mode
    run, returning (samples_per_s, models)."""
    r = cmsgen_run(path, cmsgen_bin, n, scratch, keep=keep, timeout=timeout, write=True)
    return r["samples_per_s"], r["models"]



def parse_encoder_specs(tokens, default_encoder, default_abc_opt):
    """['tseitin', 'abc:fast'] -> [('tseitin', None), ('abc', 'fast')]; a bare 'abc' inherits
    --abc-opt. The returned order is the run order — each circuit is converted+sampled once per
    spec, back-to-back in one process, which is what makes the encoder A/B paired."""
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
                         "cap, identical on every GPU (throughput is ~linear in K, but the "
                         "cap only binds for small M — above that the analytic VRAM bound is "
                         "lower). Pass an int to FIX the cap, e.g. to reproduce a sweep "
                         "recorded under an older, per-card cap")
    ap.add_argument("--no-simplify", dest="simplify", action="store_false", default=True)
    ap.add_argument("--no-els", action="store_true",
                    help="disable equivalent-literal substitution (ON by default: sound and "
                         "model-preserving, and a solid geomean win on circuit instances). Pass "
                         "this to reproduce a run made before it became the default)")
    ap.add_argument("--also-plain", action="store_true", help="also run GPAP without simplification")
    ap.add_argument("--verify-cpu", action="store_true", help="also CPU-verify a few models vs clauses")
    # Off, like every other entry point: the re-check is a self-test, not part of measuring
    # throughput. `eval/run_full_eval.sh` turns it on for the campaign.
    ap.add_argument("--verify-sound", action="store_true",
                    help="re-check on the GPU that every emitted model satisfies every clause")
    ap.add_argument("--max-clauses", type=_POS_INT, default=None,
                    help="skip any instance whose CNF exceeds this many clauses (recorded as a "
                         "skipped row with the size). The engine itself only rejects formulas "
                         "outside the packed envelope — this is the WALL-TIME guard for a "
                         "sweep: a 50-cycle b15 unroll parses for minutes before it samples")
    ap.add_argument("--max-vars", type=_POS_INT, default=None,
                    help="same, on variables (the packed-word envelope caps 2N at 16.7M anyway)")
    # The repo-local scratch (`<repo>/.gpap/tmp`), one place for everything a run writes and
    # deletes; the campaign driver passes its own `--scratch "$OUT/cnf"` and is unaffected.
    ap.add_argument("--scratch", default=None,
                    help="where converted circuit CNFs and the rival's sample file are written "
                         "(default: <repo>/.gpap/tmp)")
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


def _cms_arm_mismatch(results, cmsgen):
    """Resume guard: refuse to append rows with a different CMSGen arm than the file holds;
    returns the refusal message, or None. Absent is not zero, so armless rows in an armed file
    would read as CMSGen zeros; the reverse mix is refused too, since resume skips finished rows
    and old armless ones would never gain the comparison."""
    live = [r for r in results if not r.get("skipped") and not r.get("error")]
    if not live:
        return None
    prev_armed = any("cms_n_samples" in r or r.get("cms_sps") is not None for r in live)
    if prev_armed == bool(cmsgen):
        return None
    if prev_armed:
        return ("[resume] REFUSED: the existing rows carry a CMSGen arm but this run has no "
                "--cmsgen. Appending would mix armed and armless rows in one stage (every "
                "missing count then reads as a CMSGen zero — a phantom win). Point --cmsgen at "
                "the binary (scripts/build_cmsgen.sh), or delete the file to re-measure "
                "GPAP-only.")
    return ("[resume] REFUSED: the existing rows are GPAP-only but this run adds a CMSGen arm. "
            "--resume skips finished rows, so the old rows would never gain the comparison and "
            "the stage would stay half-measured. Delete the file to re-measure with the arm, "
            "or drop --cmsgen to keep extending the GPAP-only sweep.")


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
                    break
    except OSError:
        pass
    return None, None


def main(argv=None):
    import numpy as np
    import torch
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
        args.n_cov = 2000
    quality = args.n_cov > 0
    mine = {"auto": "auto", "on": True, "off": False}[args.mine]
    specs = parse_encoder_specs(args.encoders, args.encoder, args.abc_opt)
    if args.scratch is None:
        from gpap import paths
        args.scratch = str(paths.tmp_dir())
    os.makedirs(args.scratch, exist_ok=True)
    has_circuits = any(f.endswith(CIRCUIT_EXTS) for f in files)
    multi_enc = len(specs) > 1
    cms_patched = bool(args.cmsgen) and cmsgen_supports_nowrite(args.cmsgen)
    if args.cmsgen and not cms_patched:
        print("[warn] this cmsgen has no --nowrite (stock build): its rate INCLUDES writing every "
              "sample to a file, which GPAP's timed run does not do. Build the patched binary "
              "with scripts/build_cmsgen.sh for the like-for-like number.", file=sys.stderr)

    results, done = [], set()
    if args.output and args.resume and os.path.exists(args.output):
        try:
            prev = json.load(open(args.output))
            results = prev.get("results", [])
            done = {_row_key(r) for r in results}
            print(f"[resume] {len(results)} row(s) already in {args.output}")
            mismatch = _cms_arm_mismatch(results, args.cmsgen)
            if mismatch:
                raise SystemExit(mismatch)
            # Resuming a file written by a DIFFERENT engine mixes provenance in one result set.
            # Rows carry engine.core (absent = the retired torch/Triton-JIT engine), so say so.
            from gpap.sampling.pipeline import _core_provenance
            here = _core_provenance()
            others = {(r.get("engine") or {}).get("core") or "pre-libgpap (torch engine)"
                      for r in results
                      if (r.get("engine") or {}).get("core") not in (None, here)
                      or ((r.get("engine") or {}).get("core") is None and r.get("engine"))}
            # A REBUILD of the same version is not a different engine: the stamp is the build id
            # and a staged campaign rebuilds per VM, so only a VERSION change is loud.
            ver = lambda c: c.split("+", 1)[0]          # noqa: E731 -- "libgpap 0.3.0"
            rebuilt = sorted(c for c in others if ver(c) == ver(here))
            changed = sorted(c for c in others if ver(c) != ver(here))
            if rebuilt:
                print(f"[resume] note: {args.output} has rows from another BUILD of "
                      f"{ver(here)} ({', '.join(rebuilt)}); same version, appending.",
                      file=sys.stderr)
            if changed:
                print(f"[resume] WARNING: {args.output} contains rows from another engine "
                      f"({', '.join(changed)}); this run appends {here}. Delete the file to "
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

    hdr = _format_header(args, multi_enc, has_circuits, quality)
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
                r = _run_one(path0, row, args, device, portfolio, mine, quality, sample_file)
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
            print(_format_row(r, args, multi_enc, has_circuits, quality), flush=True)

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
    """Delete the CNF this cell generated (unless --keep-cnf) — a full circuit matrix is tens of
    GB of intermediate DIMACS. The path stays in the row for provenance."""
    p = row.get("cnf_path")
    if args.keep_cnf or not p or p == row.get("name") or not row.get("encode_s"):
        return
    try:
        os.remove(p)
    except OSError:
        pass


def _cpu_info():
    """Host CPU provenance, required to quote any GPAP/CMSGen ratio: CMSGen is single-core CPU,
    so the ratio moves with the host as much as with the GPU and a sweep recording only the GPU
    cannot tell "our GPU got faster" from "their CPU got slower" (docs/BENCHMARKING.md)."""
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


# The quality-pass metrics, computed OFF the throughput timing path (the timed run is
# collect=False): folding reconstruction/dedup into it understates GPU throughput by up to an
# order of magnitude. They live in the suite driver because the SUITE reports them.


def distinct_projections(models, support):
    """Count DISTINCT support-projections among `models` (support = 1-based vars); returns
    (n_distinct, n_models), on the same collected set for GPAP and CMSGen so they compare."""
    if not models:
        return 0, 0
    seen = set()
    for m in models:
        s = set(m)
        seen.add(tuple(v if v in s else -v for v in support))
    return len(seen), len(models)


def two_wise_coverage(models, support):
    """% of sampling-var pairs (1-based) whose all 4 (±,±) joint polarities appear in the sample
    — the combinatorial-testing COVERAGE metric the CMSGen paper reports, NOT a uniformity test;
    opt-in via `--twise`. One matmul gives the 11-counts, the rest follow from column sums."""
    import numpy as np

    S = len(support)
    if S < 2 or not models:
        return 0.0
    idx = {v: i for i, v in enumerate(support)}
    M = np.zeros((len(models), S), dtype=np.float32)
    for r, m in enumerate(models):
        s = set(m)
        for v in support:
            if v in s:
                M[r, idx[v]] = 1.0
    n = M.shape[0]
    c11 = (M.T @ M).round().astype(np.int64)       # sgemm; exact for counts < 2^24
    s1 = M.sum(axis=0).round().astype(np.int64)
    c10 = s1[:, None] - c11
    c01 = s1[None, :] - c11
    c00 = n - s1[:, None] - s1[None, :] + c11
    ok = (c11 > 0) & (c10 > 0) & (c01 > 0) & (c00 > 0)
    iu = np.triu_indices(S, k=1)
    total = iu[0].size
    return 100.0 * int(ok[iu].sum()) / total if total else 0.0


def _run_one(path0, row, args, device, portfolio, mine, quality, sample_file):
    """One (instance, encoder) cell: convert -> optional rejection baseline -> GPAP (x seeds) ->
    optional CMSGen, returning the row. Raises only on unexpected failures (the caller turns
    those into an error row); "cannot run this instance" comes back as a `skipped` row."""
    from gpap.cli import _convert_circuit
    name = row["name"]
    path = path0
    sim = None
    if path0.endswith(CIRCUIT_EXTS):
        cnf = os.path.join(args.scratch, f"{os.path.splitext(name)[0]}-c{args.cycles}"
                           f"-{row['enc_key'].replace(':', '_')}-{args.target}.cnf")
        conv = argparse.Namespace(**vars(args))
        conv.encoder, conv.abc_opt = row["encoder"], row["abc_opt"]
        _t0 = time.perf_counter()
        try:
            _convert_circuit(path0, cnf, conv, quiet=True)
        except Exception as e:  # noqa: BLE001 — a conversion failure skips the CELL, not the run
            row["skipped"] = f"convert failed: {type(e).__name__}: {e}"
            return row
        row["encode_s"] = time.perf_counter() - _t0
        if args.sim_baseline:
            from gpap.circuits.simulate import sim_baseline
            sim = sim_baseline(path0, args.cycles, args.seed, pin_outputs=args.pin_outputs,
                               target_mode=args.target)
        path = cnf
    row["cnf_path"] = path

    # Packed-envelope pre-check from the DIMACS header alone: 2N bounds the packed word's
    # literal-SUM field (gpap.engine.packed_envelope); the clause-length half is caught below by
    # the engine's own raise. Either way the CELL is skipped with a reason, never raised.
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
    # self-consistent one, not a per-metric mix), the others land in the JSON as per_seed.
    runs = []
    for s in (args.seeds or [args.seed]):
        try:
            g = sample_file(path, device, simplify=args.simplify, els=not args.no_els,
                            n_time=args.n_time, n_cov=args.n_cov, b=args.b,
                            k_cap=args.k_cap, verify_cpu=args.verify_cpu,
                            verify_sound=args.verify_sound,
                            configs=portfolio, max_chunks=args.max_chunks, seed=s,
                            mine=mine, time_limit=args.time_limit)
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
    sup1 = g["sup1"]
    # The soundness verdict folds over EVERY seed, since `per_seed` carries none. UNSOUND
    # dominates (merge_sound_check); "off" stays None so an unchecked run is not a clean zero.
    sound = merge_sound_check(*(sound_verdict(r) for r in runs))
    checked = sum(int(r.get("sound_rows_checked") or 0) for r in runs)
    unsound = sum(int(r.get("unsound_rows") or 0) for r in runs)
    row.update({
        "nv": g["nv"], "M": g["n_clauses"], "support": len(sup1),
        "gpap_sps": g["samples_per_s"], "gpap_sps_e2e": g.get("sps_e2e_total"),
        "n_models": g["n_models"], "wall_timed_s": g["wall_timed_s"],
        "wall_total_s": g["wall_total_s"], "chunk_capped": g.get("chunk_capped"),
        "sim_baseline": sim,
        # "ok" | "UNSOUND" | "off" (gpap.sampling.pipeline). Not a bool: "off" is neither
        # verdict; `sound_rows_checked` is the evidence a verdict is a statement about.
        "succ": g["success_rate"], "sound_check": sound,
        "unsound_rows": unsound if sound != "off" else None,
        "sound_rows_checked": checked if sound != "off" else None,
        "cpu_sound": g.get("cpu_sound"),
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
                            "n_models": r["n_models"], "wall_timed_s": r["wall_timed_s"],
                            "sound_check": sound_verdict(r),
                            "unsound_rows": r.get("unsound_rows")}
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

    if args.cmsgen:
        patched = cmsgen_supports_nowrite(args.cmsgen)
        # The TIMED run: emission-free when the binary supports it. On a stock binary it is the
        # samplefile run, so keep the quality pass's models from it.
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
        # True = CMSGen's own final count, not the last --statsevery multiple before a kill;
        # False makes `cms_n_samples` a LOWER BOUND, never publishable as an exact zero.
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
        # A samplefile run when the quality pass needs models (or --cmsgen-both): a
        # WITH-EMISSION rate, reported separately, never the headline ratio.
        if patched and (quality or args.cmsgen_both):
            wr = cmsgen_run(path, args.cmsgen, args.n_time, args.scratch,
                            keep=(args.n_cov or 1), timeout=args.time_limit, write=True,
                            seed=args.cmsgen_seed)
            cmodels = wr.get("models") or []
            row["cms_sps_write"] = wr["samples_per_s"]
            # >1 = how much throughput CMSGen loses by emitting its samples.
            row["cms_write_penalty"] = (timed["samples_per_s"] / wr["samples_per_s"]
                                        if wr["samples_per_s"] else None)
        elif not patched:
            row["cms_sps_write"] = timed["samples_per_s"]
        gsps, crate = g["samples_per_s"], row["cms_sps"]
        ce2e = row.get("cms_sps_e2e") or 0.0
        # The two ratios pair LIKE WITH LIKE (steady state to steady state, e2e to e2e), so both
        # sides' preparation is either in or out. inf = GPAP-only, 0 = CMSGen-only, None = both.
        row["ratio"] = (gsps / crate) if crate else (float("inf") if gsps else None)
        row["ratio_e2e"] = ((((g.get("sps_e2e_total") or 0) / ce2e) if ce2e
                             else (float("inf") if gsps else None)))
        if quality and cmodels:
            cd, cn = distinct_projections(cmodels[:args.n_cov], sup1)
            row["cms_distinct"], row["cms_distinct_of"] = cd, cn
        if args.twise:
            row["twise_cms"] = two_wise_coverage(cmodels[:args.n_cov], sup1) if cmodels else None
    return row


def _format_header(args, multi_enc, has_circuits, quality):
    """The table header, beside `_format_row` and in the same optional-column order: the only
    thing that makes either right is that the two agree. An overflowing value glues itself to its
    left neighbour rather than erroring, so widening a value means widening its label here in the
    same edit (tests/test_benchmark_harness.py::TestTableAlignment)."""
    hdr = f"{'instance':<24}"
    if multi_enc:
        hdr += f"{'enc':>10}"
    hdr += (f"{'N':>7}{'M':>8}{'|sup|':>6}{'GPAP/s':>13}{'e2e/s':>13}"
            f"{'dec/s':>9}{'impl/s':>9}{'succ%':>{_SUCC_W}}{'sols':>13}{'t_slv':>7}{'t_tot':>7}")
    if args.seeds:
        hdr += f"{'±%':>5}"
    if has_circuits:
        hdr += f"{'t_enc':>7}"
    if quality:
        hdr += f"{'uniq/s~':>13}"
    if args.twise:
        hdr += f"{'2wiseG':>8}"
    if args.sim_baseline:
        hdr += f"{'p_acc':>9}{'rej/s':>15}{'amp':>9}"
    hdr += f"{'krn':>5}{'snd':>4}"
    if args.cmsgen:
        hdr += f"{'CMS/s':>10}{'Cdec/s':>9}{'Cimpl/s':>9}{'GPAP/CMS':>10}{'e2e/CMS':>9}"
        if args.twise:
            hdr += f"{'2wiseC':>7}"
    return hdr


def _format_row(r, args, multi_enc, has_circuits, quality):
    eng = r.get("engine") or {}
    line = f"{r['name'][:22]:<24}"
    if multi_enc:
        line += f"{(r.get('enc_key') or '—'):>10}"
    dps = f"{r['decisions_per_s']:.1e}" if r.get("decisions_per_s") else "—"
    ips = f"{r['implications_per_s']:.1e}" if r.get("implications_per_s") else "—"
    # '+' after succ% = mining ran; '<' after sols = a safety valve (--time-limit /
    # zero-progress) stopped the run short of --n-time samples.
    mn = "+" if r.get("mined") else " "
    sols_s = f"{r['n_models']:,}" + ("<" if r.get("chunk_capped") else "")
    line += (f"{r['nv']:>7}{r['M']:>8}{r['support']:>6}"
             f"{r['gpap_sps']:>13,.0f}{r.get('gpap_sps_e2e') or 0:>13,.0f}{dps:>9}{ips:>9}"
             f"{fmt_pct(r['succ']):>{_SUCC_W}}{mn}{sols_s:>12}"
             f"{r['wall_timed_s']:>6.1f}s{r['wall_total_s']:>6.1f}s")
    if args.seeds:
        sp = r.get("sps_spread_frac")
        line += f"{100 * sp:>4.0f}%" if sp is not None else f"{'—':>5}"
    if has_circuits:
        line += f"{r['encode_s']:>6.1f}s" if r.get("encode_s") is not None else f"{'—':>7}"
    if quality:
        uq = f"{r['unique_sps_est']:,.0f}" if r.get("unique_sps_est") else "—"
        line += f"{uq:>13}"
    if args.twise:
        line += f"{(r.get('twise_gpap') or 0):>7.1f}%"
    if args.sim_baseline:
        sim = r.get("sim_baseline")
        if sim is None:
            line += f"{'—':>9}{'—':>15}{'—':>9}"
        elif sim["hits"]:
            amp = r.get("amplification") or 0.0
            amps = f"{amp:,.0f}x" if amp < 1e6 else f"{amp:.0e}x"
            line += f"{sim['p_accept']:>9.1e}{sim['accepted_per_s']:>15,.0f}{amps:>9}"
        else:
            line += f"{'<' + format(sim['p_accept_lt'], '.0e'):>9}{'~0':>15}{'inf':>9}"
    # kernel provenance: w/s = wave/scalar propagate, 16 or 8 = the packed split's SHIFT
    krn = f"{eng.get('mode', '?')[0]}{eng.get('shift', '')}" if eng else "—"
    # Three outcomes, three glyphs: "-" is "not checked", neither OK nor BAD; the verdict comes
    # from the one reader in gpap.sampling.pipeline (legacy `gpu_sound` rows included).
    _sc = sound_verdict(r)
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
        # Paired encoder A/B: only instances with a row under EVERY encoder count, and the BASE
        # is the first --encoders token (run order), not an alphabetical accident.
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
