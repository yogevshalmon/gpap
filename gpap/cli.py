"""`gpap` command-line entry point — the print layer (library code uses `logging`).

Subcommands: `sample` one instance, `bench` a suite, `mine` entailed clauses, `circuit2cnf`
(.bench/.aig/.aag -> projected CNF) and `info`. The circuit-encoder flags (--encoder, --abc,
--abc-opt) are shared by sample / mine / circuit2cnf; --mine auto|on|off by sample / bench.
"""
import argparse
import contextlib
import math
import os
import sys
from gpap.verdict import fmt_pct

CIRCUIT_EXTS = (".bench", ".aig", ".aag")


def _k_cap_arg(v):
    """--k-cap accepts 'auto' or a POSITIVE int; 0 would mean "uncapped", a host-RAM-spill
    footgun on the CLI, so ask for it through the Python API (k_cap=None)."""
    if v == "auto":
        return v
    return _bounded(int, 1, "'auto' or a positive integer")(v)


def _bounded(kind, lo, what, zero_means=None, exclusive=False):
    """An argparse `type` that REJECTS a value the command cannot act on, at parse time rather
    than as a nonsense run. `zero_means` documents a 0 that is meaningful, not degenerate."""
    def parse(v):
        try:
            n = kind(v)
        except (ValueError, OverflowError):
            raise argparse.ArgumentTypeError(f"expected {what}, got {v!r}") from None
        # `nan` compares False against every bound, so `n < lo` waves it through and a NaN
        # deadline disables its own cap. `inf` is refused likewise: a bound cannot be "never".
        bad = (not math.isfinite(n)) if kind is float else False
        if bad or n < lo or (exclusive and n == lo):
            extra = f" (0 = {zero_means})" if zero_means and lo == 0 else ""
            raise argparse.ArgumentTypeError(f"expected {what}{extra}, got {v!r}")
        return n
    return parse


_POS_INT = _bounded(int, 1, "a positive integer")
#: 0 is MEANINGFUL for these (unbounded / none / pin nothing); each flag's help says which.
_NONNEG_INT = _bounded(int, 0, "a non-negative integer")
_NONNEG_SECS = _bounded(float, 0.0, "a non-negative number of seconds", zero_means="unbounded")
#: `bench` gets the strict one: its --time-limit also bounds the rival subprocess, where 0 would
#: mean "kill it now" rather than "no cap", so the ambiguous value is refused.
_POS_SECS = _bounded(float, 0.0, "a positive number of seconds", exclusive=True)

# How `sound_check` prints. Three values, not two: "off" is not "unsound".
_SOUND_LABEL = {"ok": "OK", "UNSOUND": "*** UNSOUND ***", "off": "not run (--verify-sound)"}

#: A PROOF, not a failure to find anything — the two print identically otherwise. One string,
#: because it has two sites: construction proves it for almost every UNSAT formula, but mining can
#: prove it later (a formula whose every literal fails yields both polarities as units).
_UNSAT_LINE = ("  UNSAT: the formula has no models (proved by propagation, before sampling). "
               "Nothing to sample.")


def _sound_label(info):
    """The printed verdict; `sound_verdict` is the ONE rule, shared with the bench and paper
    tables, so a console line and a published cell cannot disagree."""
    from gpap.sampling.pipeline import sound_verdict
    return _SOUND_LABEL[sound_verdict(info)]



def _ensure_cnf(args):
    """(cnf_path, temp_path_or_None, encode_seconds_or_None) for a CNF or a circuit input, and
    shared by every subcommand that takes an instance. A circuit is unrolled to a scratch CNF
    whose `c wire ...` / `c encoder ...` comments let output name each column by its wire.

    The scratch CNF goes to the one scratch directory, and its name says what it holds because
    this path is printed (gpap/paths.py)."""
    path = args.input
    if not path.endswith(CIRCUIT_EXTS):
        return path, None, None
    import time as _time

    from gpap import paths
    stem = f"{os.path.splitext(os.path.basename(path))[0]}-c{args.cycles}"
    tmp = paths.scratch_file(stem, ".cnf")
    # Repo-relative only when the reader stands in the checkout, where `.gpap/tmp/x.cnf` is the
    # short name they know; absolute otherwise, since from an ancestor the relative form is a
    # bare `gpu-sat/gpap/.gpap/...` that reads like someone else's tree. Nothing chdirs, so
    # both forms open the same file.
    root = paths.repo_root()
    cwd = os.getcwd()
    if root is not None and (cwd == str(root) or cwd.startswith(str(root) + os.sep)):
        rel = os.path.relpath(tmp)
        tmp = tmp if rel.startswith(os.pardir) else rel
    t0 = _time.perf_counter()
    try:
        if path.endswith(".bench"):
            from gpap.circuits.bench import convert_one
            convert_one(path, tmp, args.cycles, args.seed, pin_outputs=args.pin_outputs,
                        encoder=args.encoder, abc_path=args.abc, abc_opt=args.abc_opt,
                        target_mode=args.target)
        else:
            from gpap.circuits.aig2cnf import convert_aig_one
            convert_aig_one(path, tmp, args.cycles, args.seed, pin_outputs=args.pin_outputs,
                            encoder=args.encoder, abc_path=args.abc, abc_opt=args.abc_opt,
                            target_mode=args.target)
    except BaseException:
        with contextlib.suppress(OSError):   # a failed conversion must not leak its temp file
            os.unlink(tmp)
        raise
    encode_s = _time.perf_counter() - t0
    print_now(f"  encoded in {encode_s:.1f}s [{args.encoder}"
              f"{'/' + args.abc_opt if args.encoder != 'tseitin' else ''}]")
    return tmp, tmp, encode_s


def _check_out_dirs(args, *flags):
    """Fail on an undeliverable destination BEFORE the run, not after minutes of GPU work."""
    for flag in flags:
        dest = getattr(args, flag.lstrip("-").replace("-", "_"), None)
        if dest:
            d = os.path.dirname(dest) or "."
            if not os.path.isdir(d):
                raise SystemExit(f"{flag}: directory {d!r} does not exist")


def _check_input(args):
    """The instance must exist before anything is printed about it, or the parse error arrives
    after the banner and reads as though the run had started."""
    if not os.path.exists(args.input):
        raise SystemExit(f"input: {args.input!r} does not exist")


def _require_engine(args):
    """The two preconditions every GPU subcommand shares -> (device,)."""
    import torch

    from gpap.engine import core_available
    if not torch.cuda.is_available():
        raise SystemExit("this command requires a CUDA GPU — there is no CPU execution path. "
                         "See `gpap info`.")
    if not core_available():
        raise SystemExit("the engine core (libgpap) is not built. Run scripts/build_core.sh, "
                         "then `gpap info` to confirm.")
    return torch.device(args.device) if args.device else torch.device("cuda")


def print_now(*a):
    """A line the user should see WHEN IT IS PRINTED, not when the process exits: stdout is
    block-buffered whenever it is not a terminal."""
    print(*a, flush=True)


def _sample_progress(args):
    """The `sample_file` progress callback: print each phase AS IT HAPPENS.

    Everything here used to be read off the returned dict and printed after the run, which put
    the mining line under the throughput lines — as though the sampler had mined after it
    finished sampling. Mining runs BEFORE the timed run (and can spend `MINE_SECONDS`), and the
    formula's shape is known before either, so both belong where they happen. The library does
    not print; this is where its phase order becomes the console's order.

    Every line goes through `print_now`, which says why it flushes."""
    def report(event, d):
        if event == "formula":
            # |sup| = the DECLARED support (the width of a printed row); `support_size` is what
            # stays ACTIVE after simplification/ELS. Print both: either alone misleads.
            act = d.get("support_size")
            report.said_support = d["n_support"]
            report.said_active = act if act is not None else d["n_support"]
            act_s = ("" if act is None or act == d["n_support"]
                     else f" ({act:,} active after simplify)")
            print_now(f"{os.path.basename(args.input)}: N={d['nv']} M={d['n_clauses']} "
                  f"|sup|={d['n_support']}{act_s}")
            if d["n_clauses"] == 0:
                print_now("  [warn] the formula has 0 clauses (empty or comment-only input?) — every "
                      "assignment is a model, so these numbers measure nothing")
            if d["unsat"]:
                report.said_unsat = True     # printed here, because this is where it is known
                print_now(_UNSAT_LINE)
        elif event == "mined":
            n = d["n_units"] + d["n_binary"] + d["n_ternary"]
            if not n:
                print_now(f"  mined nothing in {d['seconds']:.1f}s")
                return
            # Whether the clauses were KEPT is the point of the line: width >= 3 finds shrink K,
            # so a mine that found plenty can still be reverted. `gain` is the projected
            # models-per-chunk ratio (K x success), a proxy and not a measurement.
            g = d.get("gain")
            gain_s = f", {g:.2f}x projected models/chunk" if g else ""
            verdict = f"kept{gain_s}" if d["kept"] else f"reverted — did not pay{gain_s}"
            # Mining rebuilds the formula, so simplification and ELS run again over
            # `clauses + learned` and the ACTIVE support is recomputed: mined units fix support
            # variables (fewer), and the added clauses also change the equivalence classes, so a
            # variable ELS had merged away can come back as its own (more). It moves both ways —
            # hence "after mining", not "still". Without this the console and the `--stats` JSON
            # report different support sizes for the same run.
            act = d.get("support_size")
            if act is not None and report.said_active is not None and act != report.said_active:
                verdict += f"; {act:,} of {report.said_support:,} support vars active after mining"
            print_now(f"  mined {d['n_units']} units / {d['n_binary']} binary / {d['n_ternary']} "
                  f"ternary in {d['seconds']:.1f}s ({verdict})")
    report.said_unsat = False
    report.said_support = report.said_active = None
    return report


def _cmd_sample(args):
    # Host checks FIRST: a missing file or an undeliverable --out is a fact about the arguments,
    # not the machine, so it must not need a GPU to report.
    _check_input(args)
    _check_out_dirs(args, "--out", "--stats")
    device = _require_engine(args)
    path, tmp, encode_s = _ensure_cnf(args)
    try:
        _sample_run(args, device, path, tmp, encode_s)
    finally:
        if tmp:
            with contextlib.suppress(OSError):
                os.unlink(tmp)


def _sample_run(args, device, path, tmp, encode_s):
    """The body of `gpap sample`, split out so the temp CNF is unlinked on every exit path."""
    import json
    from gpap.sampling.pipeline import (describe_config, describe_portfolio,
                                        portfolio as _portfolio, sample_file, soft_warning)

    portfolio = _portfolio(args.portfolio, allow_soft=args.allow_soft)
    if args.allow_soft:
        print(soft_warning())
    print_now(f"  portfolio [{args.portfolio}]: {describe_portfolio(portfolio)}")
    # Collect sample bits when an output file is requested (up to `-n`, never past the
    # --max-samples ceiling), or when --sim-check needs rows to simulate.
    out_cap = min(args.n, args.max_samples) if (args.out or args.sim_check) else 0
    mine = {"auto": "auto", "on": True, "off": False}[args.mine]
    progress = _sample_progress(args)
    info = sample_file(path, device, n_time=args.n, n_cov=args.n_cov, out_cap=out_cap,
                       seed=args.seed, simplify=not args.no_simplify, els=not args.no_els, mine=mine,
                       configs=portfolio, k_cap=args.k_cap,
                       verify_sound=args.verify_sound, progress=progress,
                       # 0 = no cap (the pipeline reads None that way); a literal 0 expires the
                       # sampling loop before its first chunk.
                       time_limit=args.time_limit or None)
    if encode_s is not None:
        info["encode_s"] = encode_s
        info["encoder"] = args.encoder
    # '<' = a safety valve stopped the run short of -n. `chunk_capped` already excludes a
    # proved-UNSAT run, so this printer and the bench table's mark the same rows.
    capped = "<" if info.get("chunk_capped") else ""
    # The formula line, the UNSAT proof and the mining stats are printed by `_sample_progress`,
    # at the point in the run where each becomes true. The exception is a formula only MINING
    # proves unsatisfiable, which construction could not have reported.
    if info.get("unsat") and not progress.said_unsat:
        print(_UNSAT_LINE)
    print(f"  {info['n_models']:,}{capped} models in {info['wall_total_s']:.1f}s e2e "
          f"({info['wall_timed_s']:.1f}s sampling)")
    print(f"  {info['samples_per_s']:,.0f} samples/s (steady-state), "
          f"{info.get('sps_e2e_total') or 0:,.0f}/s end-to-end | success {fmt_pct(info['success_rate'])} "
          f"| sound-check {_sound_label(info)}")
    if info["n_models"] == 0 and not info.get("unsat") and info["n_clauses"]:
        # Every trajectory conflicted. The three causes want different responses, so name them.
        print("  [warn] no models: every trajectory hit a conflict. GPAP discards conflicted\n"
              "         trajectories rather than backtracking, so a tightly-constrained instance\n"
              "         can yield nothing. Try --portfolio b1 (one decision per round), a longer\n"
              "         --time-limit, or --allow-soft (faster on tight circuits, NOT uniform).")
    if info.get("n_collected"):
        nd, nc = info.get("n_distinct"), info["n_collected"]
        cpu = info.get("cpu_sound")
        print(f"  quality [--n-cov]: {nd:,} distinct of {nc:,} collected "
              f"({100.0 * nd / nc:.0f}% distinct)"
              + (f" | cpu-verified={cpu}" if cpu is not None else ""))
    if args.sim_baseline:
        # The rejection-sampling rival (simulate.py): random inputs + bit-parallel simulation,
        # no CNF. Perfectly uniform by construction; its only weakness is p_accept.
        if tmp is None:
            print("  sim-baseline: skipped (CNF input — no circuit to simulate)")
        else:
            from gpap.circuits.simulate import sim_baseline
            r = sim_baseline(args.input, args.cycles, args.seed, pin_outputs=args.pin_outputs,
                             target_mode=args.target)
            info["sim_baseline"] = r
            if r["hits"]:
                amp = info["success_rate"] / r["p_accept"] if r["p_accept"] else float("inf")
                print(f"  rejection baseline (CPU bit-parallel sim): p_accept={r['p_accept']:.2e} "
                      f"({r['hits']:,}/{r['n_sim']:,}), {r['patterns_per_s']:.2e} patterns/s "
                      f"=> {r['accepted_per_s']:,.0f} accepted/s "
                      f"(GPAP amplification {amp:,.1f}x)")
            else:
                print(f"  rejection baseline (CPU bit-parallel sim): p_accept < "
                      f"{r['p_accept_lt']:.1e} (0 hits in {r['n_sim']:,}; 95%), "
                      f"{r['patterns_per_s']:.2e} patterns/s => rejection is unusable here")
    if args.sim_check:
        if tmp is None:
            print("  sim-check: skipped (CNF input — no circuit to simulate)")
        else:
            from gpap.circuits.simulate import simulate_check
            mat, cols = info.get("samples"), info.get("cols")
            wm = info.get("wire_map") or {}
            if mat is None or len(mat) == 0:
                print("  sim-check: no samples collected")
            elif any(v not in wm for v in cols):
                print("  sim-check: skipped (no wire map for some sample columns)")
            else:
                nbad = simulate_check(args.input, args.cycles, args.seed, mat,
                                      [wm[v] for v in cols], pin_outputs=args.pin_outputs,
                                      target_mode=args.target)
                info["sim_check_bad"] = nbad
                if nbad == 0:
                    print(f"  sim-check: {len(mat):,}/{len(mat):,} emitted samples reproduce "
                          f"the pinned outputs through the circuit (encoder-independent)")
                else:
                    print(f"  SIM-CHECK FAILED: {nbad:,}/{len(mat):,} emitted samples do NOT "
                          f"drive the circuit to the pinned outputs")
    if args.out:
        from gpap.sampling.output import write_samples
        mat, cols = info.get("samples"), info.get("cols")
        want = info.get("out_requested") or 0
        if mat is None or len(mat) == 0:
            why = ("--max-samples 0 asked for none" if want == 0 else
                   "the run produced no models")
            print(f"  no samples to write ({why})")
        else:
            meta = {"instance": os.path.basename(args.input), "seed": args.seed,
                    "success_rate": round(info["success_rate"], 4)}
            write_samples(args.out, mat, cols, fmt=args.out_format,
                          wire_map=info.get("wire_map"), meta=meta)
            short = f" of {want:,} requested" if (want and len(mat) < want) else ""
            print(f"  wrote {len(mat):,} samples{short} ({mat.shape[1]} cols) -> {args.out} "
                  f"[{args.out_format}]")
            # A short file is explained where it happens: rows come off the timed run, so a
            # capped run writes what it delivered and says why.
            if short:
                why = ("the run stopped at {:,} models (--time-limit / no progress)".format(
                    info["n_models"]) if info.get("chunk_capped") else
                    "the winning config produced {:,} rows".format(len(mat)))
                print(f"         short because {why}; emitted rows all come from ONE config "
                      f"({describe_config(info.get('out_config'))}), never a mix of the portfolio's")
    if args.stats:
        stats = {k: v for k, v in info.items() if k not in ("samples", "models", "cols")}
        json.dump(stats, open(args.stats, "w"), indent=1, default=str)
        print(f"  wrote stats -> {args.stats}")


def _write_cnf(path, num_vars, clauses):
    with open(path, "w") as f:
        f.write(f"p cnf {num_vars} {len(clauses)}\n")
        for c in clauses:
            f.write(" ".join(map(str, c)) + " 0\n")


def _parse_pool(spec, cnf_path, num_vars):
    """Resolve --pool into DIMACS literals, or None for "every free literal": 'all' -> None (the
    default), 'support' -> the declared independent support (`c ind`), or an explicit "1,-2,7" /
    "@file". A variable expands to BOTH polarities; a signed literal is taken as given."""
    if spec in (None, "all"):
        return None
    if spec == "support":
        from gpap.sampling import parse_independent_support
        sup = parse_independent_support(cnf_path)
        if not sup:
            raise SystemExit(f"--pool support: {cnf_path} declares no `c ind` support. "
                             f"Use --pool all, or name the literals explicitly.")
        return [s for v in sup for s in (v + 1, -(v + 1))]
    text = spec
    if spec.startswith("@"):
        try:
            with open(spec[1:]) as fh:
                text = fh.read()
        except OSError as e:
            raise SystemExit(f"--pool: cannot read {spec[1:]!r} ({e.strerror or e})") from None
    lits = []
    for tok in text.replace(",", " ").split():
        try:
            v = int(tok)
        except ValueError:
            raise SystemExit(f"--pool: {tok!r} is not an integer literal") from None
        if v == 0 or abs(v) > num_vars:
            raise SystemExit(f"--pool: literal {v} out of range for a {num_vars}-variable formula")
        lits.append(v)
    if not lits:
        raise SystemExit("--pool: no literals given")
    return lits


def total_probed_hint(res) -> int:
    return sum(res.probed.values())


def _cmd_mine(args):
    """Prove clauses the formula already entails by propagating every k-subset of a literal set:
    a set that conflicts proves the formula entails its negation. Adding the result leaves the
    model set exactly unchanged while making later propagation stronger."""
    import json

    from gpap.mining import mine_entailed
    _check_input(args)
    _check_out_dirs(args, "--out", "--out-mined", "--stats")
    device = _require_engine(args)
    path, tmp, encode_s = _ensure_cnf(args)
    try:
        from gpap.engine import read_dimacs
        nv, clauses, _sup = read_dimacs(path)
        pool = _parse_pool(args.pool, path, nv)
        widths = sorted(set(args.width))
        if widths and widths[0] < 1:
            raise SystemExit(f"--width: widths must be >= 1, got {widths}")
        print(f"{os.path.basename(args.input)}: N={nv} M={len(clauses)} "
              f"pool={args.pool}{'' if pool is None else f' ({len(pool)} literals)'} "
              f"widths={widths}")

        # What each width WOULD cost, before spending the budget: the candidate count is exact
        # and needs no GPU work. These are UPPER BOUNDS, since width 1's units narrow the formula
        # first. Printed from INSIDE the sweep (on_plan) so it reuses that engine.
        def _plan(w, n, ceiling):
            # Width 1's count is ONE round of a fixpoint over a shrinking pool, never a bound.
            per = " per fixpoint round" if w == 1 else ""
            # For wider widths the count is an UPPER BOUND, so a flat "> --max-tuples N" would
            # assert a truncation the same run can contradict. A plan line may warn, not conclude.
            over = (f"  > --max-tuples {ceiling:,} unless an earlier width shrinks the pool"
                    if (ceiling > 0 and n > ceiling) else "")
            print(f"  width {w}: {n:,} candidates{per}{over}")

        res = mine_entailed(nv, clauses, device,
                            pool=pool,
                            widths=tuple(widths),
                            seconds=None if args.seconds <= 0 else args.seconds,
                            max_tuples=args.max_tuples, drop=args.drop, on_plan=_plan)
        print(f"  mined {res.summary()}")
        if res.seconds:
            print(f"  engine: {res.implications:,} implications "
                  f"({res.implications_per_s / 1e9:.2f}G/s), "
                  f"{total_probed_hint(res):,} tuples probed "
                  f"({total_probed_hint(res) / res.seconds:,.0f}/s)")
        for w in widths:
            cand, done = res.candidates.get(w, 0), res.probed.get(w, 0)
            cover = f"{100.0 * done / cand:.3g}%" if cand else "-"
            mark = ("exhaustive — nothing of this width escaped" if res.exhausted.get(w)
                    else res.skipped.get(w, "incomplete"))
            # Width 1's numbers are SUMS over its fixpoint rounds; name the rounds so the total
            # reconciles with the pool it started from.
            rnd = res.rounds.get(w, 1)
            over_rounds = f" over {rnd} fixpoint rounds" if rnd > 1 else ""
            print(f"    width {w}: {res.by_width.get(w, 0):,} clause(s)   "
                  f"probed {done:,}/{cand:,} ({cover}){over_rounds}   [{mark}]")
        if not res.clauses:
            # Per-width, never summed: width 1 is a fixpoint SUM and width 2 a C(P,2).
            done = sum(res.probed.values())
            partial = [w for w in widths if res.probed.get(w, 0) < res.candidates.get(w, 0)]
            if partial:
                cov = ", ".join(
                    f"width {w} {100.0 * res.probed.get(w, 0) / res.candidates[w]:.3g}%"
                    for w in partial if res.candidates.get(w))
                # An empty result from a PARTIAL sweep says nothing about the formula.
                print(f"  (nothing found in the {done:,} tuple(s) probed — that is only "
                      f"{cov} of each width's space, so this is NOT evidence that nothing is "
                      f"entailed. Raise --max-tuples/--seconds, or narrow --pool.)")
            elif all(res.exhausted.get(w, False) for w in widths):
                print("  (nothing entailed was found, and the search WAS exhaustive — the "
                      "formula's propagation is already tight at these widths)")
            else:
                # No candidates generated (a skipped or errored width): never claim exhaustive.
                print("  (nothing was probed — see the per-width status above)")

        if args.out:
            # The STRENGTHENED formula: original + mined, sound since every mined clause is
            # entailed.
            _write_cnf(args.out, nv, clauses + list(res.clauses))
            print(f"  wrote {args.out} ({len(clauses):,} original + {len(res.clauses):,} "
                  f"mined clauses)")
        if args.out_mined:
            _write_cnf(args.out_mined, nv, list(res.clauses))
            print(f"  wrote {args.out_mined} ({len(res.clauses):,} mined clauses only)")
        if args.stats:
            # Report WHAT WAS DONE, not only what was found: an empty `clauses` without coverage
            # cannot distinguish "entails nothing" from "barely started".
            total_probed = sum(res.probed.values())
            info = {"input": args.input, "nv": nv, "n_clauses": len(clauses),
                    "pool": args.pool,
                    "pool_size": (2 * nv if pool is None else len(pool)),
                    "pool_note": ("all free literals (resolved per width)" if pool is None
                                  else "explicit"),
                    "widths": widths, "seconds": res.seconds, "encode_s": encode_s,
                    "max_tuples": args.max_tuples, "seconds_budget": args.seconds,
                    "drop": args.drop, "n_dropped_derivable": res.derivable,
                    "n_mined": len(res.clauses),
                    "probed_total": total_probed,
                    "coverage": {str(w): (res.probed.get(w, 0) / res.candidates[w])
                                 for w in sorted(res.candidates) if res.candidates.get(w)},
                    "probes_per_s": (total_probed / res.seconds) if res.seconds else None,
                    # The HARDWARE-work metric: clauses/s and probes/s depend on the instance,
                    # implications/s on the engine, so only it compares across instances.
                    "implications": res.implications,
                    "implications_per_s": res.implications_per_s,
                    "complete": all(res.exhausted.get(w, False) for w in widths),
                    "by_width": {str(k): v for k, v in sorted(res.by_width.items())},
                    "candidates": {str(k): v for k, v in sorted(res.candidates.items())},
                    "probed": {str(k): v for k, v in sorted(res.probed.items())},
                    "rounds": {str(k): v for k, v in sorted(res.rounds.items())},
                    "exhausted": {str(k): bool(v) for k, v in sorted(res.exhausted.items())},
                    "skipped": {str(k): str(v) for k, v in sorted(res.skipped.items())},
                    "clauses": res.clauses}
            with open(args.stats, "w") as fh:
                json.dump(info, fh, indent=1)
            print(f"  wrote {args.stats}")
    finally:
        if tmp:
            with contextlib.suppress(OSError):
                os.unlink(tmp)


def _cmd_bench(args):
    from gpap import benchmark
    argv = ["--glob", args.glob, "--n-time", str(args.n),
            "--time-limit", str(args.time_limit), "--n-cov", str(args.n_cov),
            "--portfolio", args.portfolio,
            "--encoder", args.encoder, "--abc-opt", args.abc_opt, "--mine", args.mine,
            "--target", args.target,
            "--cycles", str(args.cycles), "--seed", str(args.seed),
            "--k-cap", str(args.k_cap)]
    if args.seeds:
        argv += ["--seeds"] + [str(s) for s in args.seeds]
    if args.twise:
        argv += ["--twise"]
    if args.allow_soft:
        # Must be forwarded, or a bench run that asked for the soft config silently gets the
        # uniformity-safe portfolio.
        argv += ["--allow-soft"]
    if args.no_els:
        argv += ["--no-els"]
    if args.verify_sound:
        argv += ["--verify-sound"]
    if args.sim_baseline:
        argv += ["--sim-baseline"]
    if args.abc:
        argv += ["--abc", args.abc]
    if args.pin_outputs is not None:
        argv += ["--pin-outputs", str(args.pin_outputs)]
    if args.cmsgen:
        argv += ["--cmsgen", args.cmsgen]
    if args.cmsgen_both:
        argv += ["--cmsgen-both"]
    if args.encoders:
        argv += ["--encoders"] + list(args.encoders)
    if args.resume:
        argv += ["--resume"]
    if args.tag:
        argv += ["--tag", args.tag]
    if args.device:
        argv += ["--device", args.device]
    if args.output:
        argv += ["--output", args.output]
    benchmark.main(argv)


def _convert_circuit(path, outp, args, quiet=False):
    """Dispatch one circuit file (.bench or .aig/.aag) to its converter. `args` needs the shared
    conversion attributes (cycles, seed, pin_outputs, encoder, abc, abc_opt, target)."""
    tm = getattr(args, "target", "random")
    if path.endswith(".bench"):
        from gpap.circuits.bench import convert_one
        convert_one(path, outp, args.cycles, args.seed, pin_outputs=args.pin_outputs,
                    encoder=args.encoder, abc_path=args.abc, abc_opt=args.abc_opt, quiet=quiet,
                    target_mode=tm)
    else:
        from gpap.circuits.aig2cnf import convert_aig_one
        convert_aig_one(path, outp, args.cycles, args.seed, pin_outputs=args.pin_outputs,
                        encoder=args.encoder, abc_path=args.abc, abc_opt=args.abc_opt,
                        quiet=quiet, target_mode=tm)


def _cmd_circuit2cnf(args):
    import glob as _glob
    _check_input(args)
    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        files = sorted(f for ext in CIRCUIT_EXTS
                       for f in _glob.glob(os.path.join(args.input, "**", "*" + ext),
                                           recursive=True))
        for fp in files:
            stem = os.path.splitext(os.path.basename(fp))[0]
            outp = os.path.join(args.output, f"{stem}-c{args.cycles}.cnf")
            try:
                _convert_circuit(fp, outp, args)
            except Exception as e:  # noqa: BLE001
                print(f"{fp}: SKIPPED ({type(e).__name__}: {e})", file=sys.stderr)
    else:
        _check_out_dirs(args, "output")
        _convert_circuit(args.input, args.output, args)


def _print_abc_status():
    from gpap.circuits.abc_backend import find_abc
    exe = find_abc()
    if exe:
        print(f"abc circuit encoder: {exe}  (use --encoder abc)")
    else:
        print("abc circuit encoder: not found — native Tseitin only "
              "(build with scripts/build_abc.sh to enable --encoder abc)")


def _cmd_info(args):
    import torch
    print(f"torch {torch.__version__}  (CUDA {torch.version.cuda})")
    if not torch.cuda.is_available():
        print("device: CPU (no CUDA) — the engine CANNOT run (packed-kernel-only, no CPU path);"
              " only circuit2cnf / preprocessing work here")
        _print_abc_status()
        return
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    free, total = torch.cuda.mem_get_info(dev)
    print(f"device: {props.name}  {total/1e9:.1f} GB ({free/1e9:.1f} GB free)  "
          f"SM {props.major}.{props.minor} x{props.multi_processor_count}")
    # A capped budget changes K on memory-bound formulas, so it belongs beside the device, not
    # in a log line somebody has to go looking for.
    try:
        from gpap.engine.calibrate import gpu_mem_available, vram_budget_cap
        cap = vram_budget_cap()
        if cap:
            got = gpu_mem_available(dev)
            print(f"VRAM budget: capped at {cap/(1<<20):,.0f} MiB by GPAP_VRAM_BUDGET_MIB "
                  f"(engine sizes K against {got/1e9:.1f} GB; "
                  f"{'the cap binds' if got >= cap else 'the CARD binds — the cap is inert here'})")
    except Exception as e:  # noqa: BLE001 — provenance, never fatal
        print(f"VRAM budget: unavailable ({e})")
    try:
        from gpap.engine import _core
        if _core.available():
            path = next(p for p in _core._candidates() if p.exists())
            print(f"engine core: libgpap {_core.version()}  ({path})")
        else:
            print("engine core: NOT BUILT — the engine cannot run.\n"
                  "             build it with:  scripts/build_core.sh")
    except Exception as e:  # noqa: BLE001
        print(f"engine core: unavailable ({e})")
    try:
        from gpap.engine._triton import kernel_available
        ok = kernel_available()
        print(f"triton (BUILD only — needed to REBUILD kernels, not to run them): "
              f"{'available' if ok else 'unavailable'}")
    except Exception as e:  # noqa: BLE001
        print(f"triton (BUILD only): import failed ({e})")
    _print_abc_status()
    # Power-limit / clock warning: before a headline benchmark, confirm the card is not
    # power-capped. Ratios within one state are valid, absolute numbers across states are not.
    try:
        import subprocess

        def _f(x):
            try:
                return float(x)
            except ValueError:
                return None
        q = subprocess.run(["nvidia-smi", "--query-gpu=power.limit,power.default_limit,clocks.sm,"
                            "clocks.max.sm", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        cur, default, sm, sm_max = (x.strip() for x in q.stdout.strip().split(","))
        note = ""
        if _f(cur) and _f(default) and _f(cur) < 0.95 * _f(default):
            note = f"  ⚠️  power capped {cur} W < default {default} W — benchmark numbers throttled"
        elif cur == "[N/A]":
            note = "  (power.limit N/A — common on WSL2; check clocks under load instead)"
        print(f"power limit: {cur} W (default {default} W), SM clock {sm}/{sm_max} MHz{note}")
    except Exception:  # noqa: BLE001
        pass


def _add_encoder_args(sp):
    """Circuit-encoder flags, shared by every subcommand that accepts a circuit input."""
    sp.add_argument("--encoder", choices=["tseitin", "abc", "auto"], default="tseitin",
                    help="circuit->CNF encoder: 'tseitin' (native, default) | 'abc' (unroll + "
                         "dc2/fraig via berkeley-abc, 2-8x fewer clauses, same solution space; "
                         "needs an abc binary — see scripts/build_abc.sh) | 'auto' (abc if found)")
    sp.add_argument("--abc", default=None,
                    help="path to an `abc` binary (else $GPAP_ABC, PATH, <repo>/.gpap/bin/abc)")
    sp.add_argument("--abc-opt", choices=["full", "fast", "auto"], default="auto",
                    help="abc optimization effort: 'full' (optimize the unrolled AIG — best "
                         "shrink, expensive on big unrolls) | 'fast' (optimize pre-unroll + one "
                         "cheap post sweep) | 'auto' (full when small, fast when big; default)")
    sp.add_argument("--target", choices=["random", "rare"], default="random",
                    help="pinned output pattern: 'random' (default — outputs of one seed-random "
                         "simulated sequence; size-biased toward high-probability patterns) | "
                         "'rare' (rarest pattern seen in a 262k-sequence bit-parallel simulation "
                         "— still reachable/SAT, but adversarial for rejection sampling)")


def _user_error_types():
    """Exceptions that mean "bad input or bad environment", not "gpap has a bug"; only these
    collapse to a one-line message, anything else keeps its traceback. `GpapError` is imported
    lazily, since the core may not be built."""
    types = [OSError, ValueError]
    try:
        from gpap.engine._core import GpapError
        types.append(GpapError)
    except Exception:                                     # noqa: BLE001
        pass
    return tuple(types)


def main(argv=None):
    # Its OWN sentence, not `__doc__`: this module's docstring opens with a note to whoever edits
    # it (that the print layer lives here), and `gpap --help` was greeting users with it.
    p = argparse.ArgumentParser(
        prog="gpap",
        description="GPU-parallel propagation over many partial assignments at once, and the "
                    "solution sampler built on it.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="sample models from one CNF, .bench or .aig/.aag file")
    s.add_argument("input", help="a DIMACS .cnf, or a circuit (.bench / .aig / .aag — unrolled first)")
    s.add_argument("-n", type=_POS_INT, default=50_000,
                   help="models to deliver (a target; default 50k). The batch K is the unit of "
                        "work, so a run overshoots to the end of a chunk and can fall short if a "
                        "safety valve fires — the delivered count is always printed")
    s.add_argument("--time-limit", type=_NONNEG_SECS, default=60.0,
                   help="per-instance sampling wall cap in seconds (0 = no cap). Bounds the "
                        "sampling loop AND any top-up needed to fill --out")
    s.add_argument("--seed", type=int, default=5)
    s.add_argument("--n-cov", type=_NONNEG_INT, default=0,
                   help="collect this many models for the quality/soundness readout (distinct "
                        "count + optional CPU verification); 0 = skip that pass")
    s.add_argument("--verify-sound", action="store_true",
                   help="re-check on the GPU that EVERY emitted model satisfies EVERY clause. "
                        "Off by default: it is a self-test, not part of sampling (a row is only "
                        "counted when it is fully assigned with no clause emptied). Its wall is "
                        "excluded from every rate, and it grows with the formula")
    s.add_argument("--device", default=None)
    s.add_argument("--k-cap", type=_k_cap_arg, default="auto",
                   help="ceiling on the per-chunk batch K. 'auto' (default) is the library's "
                        "cap, the same on every GPU; pass an int to fix it explicitly")
    s.add_argument("--portfolio", choices=["circuit", "default", "b8", "b1", "s1"],
                   default="circuit",
                   help="single-config choices (b8/b1/s1) pin ONE config and disable the bandit — use them for any cross-GPU or before/after MEASUREMENT, since the bandit can otherwise pick differently per machine and confound the comparison")
    s.add_argument("--allow-soft", action="store_true",
                   help="add the soft risk-polarity config to the bandit portfolio. Higher "
                        "success on tight/deep circuits, but it BIASES the distribution "
                        "(Barbarik rejects it on 64 of 65 circuits) — opt in only when throughput matters and "
                        "uniformity does not")
    s.add_argument("--no-simplify", action="store_true",
                   help="disable unit-propagation simplification (ON by default; it is "
                        "model-preserving, unlike the variable elimination a sampler must switch "
                        "off, and it is most of the win over an unpreprocessed formula)")
    s.add_argument("--no-els", action="store_true",
                   help="disable equivalent-literal substitution (ON by default: it merges "
                        "mutually-implied literals, removing variables and clauses — a measured "
                        "sampling win, and uniformity-fuzzed at 0 verdict flips / 0 unsound rows "
                        "against the exact oracle; docs/findings/INSIGHTS.md)")
    s.add_argument("--cycles", type=_POS_INT, default=25,
                   help="unroll depth (circuit inputs; combinational AIGs clamp to 1)")
    s.add_argument("--pin-outputs", type=_NONNEG_INT, default=None,
                   help="how many circuit outputs to pin (default: all). 0 pins NOTHING — the "
                        "unconstrained free-hypercube baseline, not a degenerate value")
    s.add_argument("--mine", choices=["auto", "on", "off"], default="auto",
                   help="entailed-clause mining: 'auto' (on, bounded by a 5s total budget) | "
                        "'on' (force, ignores the size gate) | 'off'")
    _add_encoder_args(s)
    s.add_argument("--sim-baseline", action="store_true",
                   help="(circuit inputs) also measure the REJECTION-SAMPLING rival: random "
                        "inputs + bit-parallel circuit simulation, no CNF — reports p_accept, "
                        "sim patterns/s, accepted/s and GPAP's constraint amplification")
    s.add_argument("--sim-check", action="store_true",
                   help="(circuit inputs) simulate every emitted sample through the ORIGINAL "
                        "circuit and verify the pinned outputs — an encoder-independent "
                        "end-to-end soundness check (validates unroll+Tseitin/ABC, not just CNF)")
    s.add_argument("--out", default=None,
                   help="write the sampled solutions here (compact support projections, one per "
                        "sample). For a .bench input each column is a primary-input wire@cycle")
    s.add_argument("--out-format", choices=["bits", "npy", "dimacs"], default="bits",
                   help="--out format: 'bits' (0/1 text, default) | 'npy' (numpy uint8 array) | "
                        "'dimacs' (signed lits, CMSGen-style)")
    s.add_argument("--max-samples", type=_NONNEG_INT, default=100_000,
                   help="CEILING on solutions written to --out (default 100k). It can only lower "
                        "the count: the file gets min(-n, --max-samples, models delivered) rows, "
                        "so raising it above -n changes nothing — raise -n for a bigger dump")
    s.add_argument("--stats", default=None, help="write the run's info/stats JSON here")
    s.set_defaults(func=_cmd_sample)

    m = sub.add_parser("mine", help="prove clauses the formula already entails (BCP-visible)",
                       description="Probe every k-subset of a literal set: assuming a set and "
                                   "propagating to a CONFLICT proves the formula entails the "
                                   "negation of that set — a clause. Every clause found is "
                                   "entailed, so adding them leaves the solution set exactly "
                                   "unchanged and only makes later propagation stronger.")
    m.add_argument("input", help="a DIMACS .cnf, or a .bench/.aig/.aag circuit (converted first)")
    m.add_argument("--width", type=int, nargs="+", default=[1, 2],
                   help="tuple sizes to sweep (default: 1 2). Width 1 = failed literals, i.e. the "
                        "BCP-visible backbone, and it is iterated to a fixpoint; width 2 = binary "
                        "clauses; and so on. Cost grows as C(pool, width) — check the candidate "
                        "counts this prints before asking for 3")
    m.add_argument("--pool", default="all",
                   help="which literals to combine: 'all' (default — every literal propagation "
                        "left undetermined) | 'support' (the declared `c ind` independent "
                        "support; for a circuit that is the primary inputs) | an explicit list "
                        "like '1,-2,7' or '@lits.txt'. A bare variable expands to both "
                        "polarities. Width 1 always sweeps every free literal — the backbone is "
                        "cheap and is the dominant win")
    m.add_argument("--seconds", type=_NONNEG_SECS, default=10.0,
                   help="wall-clock budget across all widths (default 10; 0 = unbounded)")
    m.add_argument("--max-tuples", type=_NONNEG_INT, default=200_000,
                   help="WORK BUDGET per width: how many candidate tuples to probe (default "
                        "200k). A width with more candidates than this probes the first "
                        "--max-tuples of them and is reported as not exhaustive — the clauses it "
                        "finds are still entailed. 0 = unbounded. Width 1 is iterated to a "
                        "fixpoint and spends this budget ACROSS its rounds. Cost is the PROBE, "
                        "not the enumeration: probing n tuples initializes n x M clause words")
    m.add_argument("--drop", choices=["none", "subsumed", "derivable"], default="subsumed",
                   help="which REDUNDANT clauses to leave out. 'subsumed' (default) drops a clause "
                        "some clause already covers — free. 'derivable' ALSO drops the ones unit "
                        "propagation re-derives on demand in EVERY direction (61%% of them on "
                        "b03-c25), which is what a CDCL preprocessor wants; it costs one extra "
                        "pass and is NOT the default because a derivable clause still shortens "
                        "propagation, worth +29-43%% to GPAP's own sampler. 'none' keeps "
                        "everything proved")
    m.add_argument("--out", default=None,
                   help="write the STRENGTHENED CNF here (original + mined clauses)")
    m.add_argument("--out-mined", default=None, help="write ONLY the mined clauses here")
    m.add_argument("--stats", default=None,
                   help="write the mining report JSON here (includes every mined clause)")
    m.add_argument("--device", default=None)
    m.add_argument("--seed", type=int, default=5, help="(circuit input) unroll/target seed")
    m.add_argument("--cycles", type=_POS_INT, default=25,
                   help="(circuit input) unroll depth")
    m.add_argument("--pin-outputs", type=_NONNEG_INT, default=None,
                   help="(circuit input) outputs to pin (default: all; 0 pins nothing)")
    _add_encoder_args(m)
    m.set_defaults(func=_cmd_mine)

    b = sub.add_parser("bench", help="benchmark a glob of instances (vs CMSGen if given)")
    b.add_argument("glob", help="DIMACS .cnf and/or circuits (.bench/.aig/.aag, converted first)")
    b.add_argument("-n", "--n-time", dest="n", type=_POS_INT, default=50_000)
    b.add_argument("--time-limit", type=_POS_SECS, default=60.0,
                   help="per-instance sampling wall cap in seconds (also the rival's timeout)")
    b.add_argument("--n-cov", type=_NONNEG_INT, default=0)
    b.add_argument("--verify-sound", action="store_true",
                   help="re-check on the GPU that EVERY emitted model satisfies EVERY clause. "
                        "Off by default, as everywhere else: it is a self-test, not part of "
                        "measuring throughput. Its wall is excluded from every rate, but it "
                        "grows with the formula (eval/run_full_eval.sh turns it on)")
    b.add_argument("--k-cap", type=_k_cap_arg, default="auto",
                   help="ceiling on the per-chunk batch K. 'auto' (default) is the library's "
                        "cap, the same on every GPU; pass an int to fix it explicitly")
    b.add_argument("--cmsgen", default=None,
                   help="path to a cmsgen binary (enables the comparison). Prefer the one built "
                        "by scripts/build_cmsgen.sh — its --nowrite gives the emission-free, "
                        "like-for-like rate (stock CMSGen's wall is dominated by writing every "
                        "sample) and --statsevery its own decisions/implications per second")
    b.add_argument("--cmsgen-both", action="store_true",
                   help="also time CMSGen's write-mode run (the emission-cost appendix)")
    b.add_argument("--encoders", nargs="+", default=None, metavar="ENC",
                   help="run every circuit through EACH encoder, one row per (instance, "
                        "encoder): e.g. --encoders tseitin abc:fast")
    b.add_argument("--resume", action="store_true",
                   help="skip rows already in --output and append (re-run after an interruption)")
    b.add_argument("--tag", default=None, help="free-form label stored in the results JSON")
    b.add_argument("--portfolio", choices=["circuit", "default", "b8", "b1", "s1"],
                   default="circuit",
                   help="single-config choices (b8/b1/s1) pin ONE config and disable the bandit — use them for any cross-GPU or before/after MEASUREMENT, since the bandit can otherwise pick differently per machine and confound the comparison")
    b.add_argument("--allow-soft", action="store_true",
                   help="add the soft risk-polarity config to the bandit portfolio. Higher "
                        "success on tight/deep circuits, but it BIASES the distribution "
                        "(Barbarik rejects it on 64 of 65 circuits) — opt in only when throughput matters and "
                        "uniformity does not")
    b.add_argument("--device", default=None)
    b.add_argument("--output", default=None)
    b.add_argument("--cycles", type=_POS_INT, default=25,
                   help="unroll depth (circuit inputs; combinational AIGs clamp to 1)")
    b.add_argument("--seed", type=int, default=5, help="conversion + sampling seed")
    b.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="run the timed sampling at each seed, report the median-throughput run "
                        "per instance (+ ±%% spread column; per-seed raws in the JSON). "
                        "Conversion/target stays --seed")
    b.add_argument("--pin-outputs", type=_NONNEG_INT, default=None,
                   help="outputs to pin (circuit input; default all, 0 pins nothing)")
    b.add_argument("--mine", choices=["auto", "on", "off"], default="auto",
                   help="entailed-clause mining: 'auto' (on, bounded by a 5s total budget) | "
                        "'on' (force, ignores the size gate) | 'off'")
    b.add_argument("--no-els", action="store_true",
                   help="disable equivalent-literal substitution (ON by default; pass this to "
                        "reproduce a run made before it became the default)")
    b.add_argument("--twise", action="store_true",
                   help="also compute 2-wise coverage (the CMSGen paper's spread metric; needs "
                        "the --n-cov quality pass — auto-enables n_cov=2000 if unset)")
    b.add_argument("--sim-baseline", action="store_true",
                   help="(circuit instances) also measure the rejection-sampling rival per "
                        "instance: p_accept + bit-parallel sim rate -> p_acc / rej/s columns")
    _add_encoder_args(b)
    b.set_defaults(func=_cmd_bench)

    c = sub.add_parser("circuit2cnf",
                       help="unroll a circuit (.bench/.aig/.aag, or a dir of them) to projected CNF")
    c.add_argument("input", help="a .bench/.aig/.aag file or a directory of them")
    c.add_argument("output", help="output .cnf (single) or output directory")
    c.add_argument("--cycles", type=_POS_INT, default=25)
    # 5 = the same conversion/target seed `gpap sample` and `gpap bench` default to; another
    # value would pin a different rare pattern than an in-process conversion of the same circuit.
    c.add_argument("--seed", type=int, default=5)
    c.add_argument("--pin-outputs", type=_NONNEG_INT, default=None)
    _add_encoder_args(c)
    c.set_defaults(func=_cmd_circuit2cnf)

    i = sub.add_parser("info", help="device, engine core, kernel/encoder availability, VRAM cap, "
                                    "power-limit warning")
    i.set_defaults(func=_cmd_info)

    args = p.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        raise SystemExit("interrupted") from None
    except _user_error_types() as e:
        # The core's own diagnostics are good: print the message, keep the traceback behind
        # $GPAP_TRACEBACK=1.
        if os.environ.get("GPAP_TRACEBACK"):
            raise
        raise SystemExit(f"gpap {args.cmd}: {e}") from None


if __name__ == "__main__":
    main()
