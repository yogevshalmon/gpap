"""`gpap` command-line entry point.

Subcommands:
  gpap sample <cnf|bench|aig|aag> [-n] [--out FILE] [--stats FILE] [--encoder abc]  — sample one instance
  gpap bench  <glob> [--cmsgen PATH] [--encoder abc] [--n-cov] ...  — benchmark a suite (CNFs and/or circuits)
  gpap circuit2cnf <circuit|dir> --cycles T [--encoder abc]         — .bench/.aig/.aag -> projected CNF
  gpap info                                                         — device / triton / abc / power

Circuit-encoder flags (--encoder tseitin|abc|auto, --abc PATH, --abc-opt full|fast|auto) are shared
by sample / bench / circuit2cnf; --mine auto|on|off by sample / bench.
Library code uses `logging`; user-facing output lives here (this is the print layer).
"""
import argparse
import contextlib
import math
import os
import sys

CIRCUIT_EXTS = (".bench", ".aig", ".aag")   # the single source for "is this a circuit input?"


def _k_cap_arg(v):
    """--k-cap accepts the literal 'auto' or a POSITIVE int; the pipeline resolves 'auto' per
    device. 0 would mean "uncapped" to the memory model — on the CLI that is a footgun (an
    uncapped K on a big-VRAM box is the host-RAM-spill hazard), so ask for it explicitly via the
    Python API (k_cap=None) if that is really what you want.
    Shared with gpap/benchmark.py (which already imports this module for CIRCUIT_EXTS)."""
    if v == "auto":
        return v
    return _bounded(int, 1, "'auto' or a positive integer")(v)


def _bounded(kind, lo, what, zero_means=None, exclusive=False):
    """An argparse `type` that REJECTS a value the command cannot act on.

    Every one of these was reachable and silently produced a nonsense run: `-n 0` and `-n -5` both
    sampled a full chunk anyway; `--time-limit 0` and `--time-limit -1` printed "0 models" with no
    hint that the cap was the cause; `--max-tuples -1` meant unbounded while the plan line warned
    that the ceiling had been exceeded. Refusing at parse time costs nothing and cannot mislead.
    `zero_means` documents a 0 that is meaningful (usually "unbounded") rather than degenerate."""
    def parse(v):
        try:
            n = kind(v)
        except (ValueError, OverflowError):
            raise argparse.ArgumentTypeError(f"expected {what}, got {v!r}") from None
        # `nan` compares False against EVERY bound, so a plain `n < lo` waves it through — and a
        # NaN deadline is the one value that silently disables the cap it was passed to set
        # (`elapsed > nan` is never true). `inf` is refused for the same reason: a flag whose whole
        # job is to bound something must not accept "never".
        bad = (not math.isfinite(n)) if kind is float else False
        if bad or n < lo or (exclusive and n == lo):
            extra = f" (0 = {zero_means})" if zero_means and lo == 0 else ""
            raise argparse.ArgumentTypeError(f"expected {what}{extra}, got {v!r}")
        return n
    return parse


_POS_INT = _bounded(int, 1, "a positive integer")
#: 0 is always MEANINGFUL for these (unbounded / none / pin nothing) — what it means is
#: stated at each flag's own help, which is where a reader looks.
_NONNEG_INT = _bounded(int, 0, "a non-negative integer")
_NONNEG_SECS = _bounded(float, 0.0, "a non-negative number of seconds", zero_means="unbounded")
#: `bench` gets the strict one: its --time-limit ALSO bounds the rival subprocess, where 0 would
#: mean "kill it immediately" rather than "no cap" — the two readings are not reconcilable, so the
#: measurement path simply does not accept the ambiguous value.
_POS_SECS = _bounded(float, 0.0, "a positive number of seconds", exclusive=True)


def _ensure_cnf(args):
    """(cnf_path, temp_path_or_None, encode_seconds_or_None) for a CNF **or** circuit input.

    A circuit is unrolled to a temporary CNF first; the conversion embeds the wire map as
    `c wire ...` and the encoder provenance as `c encoder ...`, so downstream output can name each
    column by its circuit wire and the pipeline can skip work the encoder already did. Shared by
    every subcommand that takes an instance, so the circuit flags mean one thing.
    """
    path = args.input
    if not path.endswith(CIRCUIT_EXTS):
        return path, None, None
    import tempfile
    import time as _time
    tmp = tempfile.NamedTemporaryFile(suffix=".cnf", delete=False).name
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
    print(f"  encoded in {encode_s:.1f}s [{args.encoder}"
          f"{'/' + args.abc_opt if args.encoder != 'tseitin' else ''}]")
    return tmp, tmp, encode_s


def _check_out_dirs(args, *flags):
    """Fail on an undeliverable destination BEFORE the run, not after minutes of GPU work.

    Every subcommand that writes a file uses this. `mine` and `circuit2cnf` used to skip it and
    open the path only at the very end, so a typo in --out cost the whole sweep and then raised a
    FileNotFoundError traceback over results that were already computed and are now gone."""
    for flag in flags:
        dest = getattr(args, flag.lstrip("-").replace("-", "_"), None)
        if dest:
            d = os.path.dirname(dest) or "."
            if not os.path.isdir(d):
                raise SystemExit(f"{flag}: directory {d!r} does not exist")


def _check_input(args):
    """The instance has to exist before anything is printed about it. Without this the parse error
    arrived AFTER the portfolio banner, which reads as though the run had started."""
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


def _cmd_sample(args):
    # Host checks FIRST: a missing file or an undeliverable --out is a fact about the arguments,
    # not about the machine, so it must not be reportable only on a box with a GPU.
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
    """The body of `gpap sample`, after the environment checks and the circuit conversion —
    split out so the converted temp CNF is unlinked on every exit path."""
    import json
    from gpap.sampling.metrics import format_uniformity
    from gpap.sampling.pipeline import (describe_config, describe_portfolio,
                                        portfolio as _portfolio, sample_file, soft_warning)

    portfolio = _portfolio(args.portfolio, allow_soft=args.allow_soft)
    if args.allow_soft:
        print(soft_warning())
    print(f"  portfolio [{args.portfolio}]: {describe_portfolio(portfolio)}")
    # Collect sample bits when an output file is requested (up to `-n` solutions, never more
    # than the --max-samples safety ceiling) — or when --sim-check needs rows to simulate.
    out_cap = min(args.n, args.max_samples) if (args.out or args.sim_check) else 0
    mine = {"auto": "auto", "on": True, "off": False}[args.mine]
    info = sample_file(path, device, n_time=args.n, n_cov=args.n_cov, out_cap=out_cap,
                       seed=args.seed, simplify=not args.no_simplify, els=not args.no_els, mine=mine,
                       uniformity=args.uniformity, configs=portfolio, k_cap=args.k_cap,
                       # 0 = no cap (the pipeline reads None that way); as a literal 0 it would
                       # expire the sampling loop before its first chunk and report "0 models".
                       time_limit=args.time_limit or None)
    if encode_s is not None:
        info["encode_s"] = encode_s          # circuit->CNF encode wall (stats sidecar honesty)
        info["encoder"] = args.encoder
    capped = "<" if info.get("chunk_capped") else ""
    # |sup| = the DECLARED support (the width of every printed solution row). `support_size` is
    # what is left ACTIVE after simplification/ELS; printing only the first made a run look wider
    # than it searched, and only the second made the output file's column count unexplainable.
    act = info.get("support_size")
    act_s = "" if act is None or act == len(info["sup1"]) else f" ({act:,} active after simplify)"
    print(f"{os.path.basename(args.input)}: N={info['nv']} M={info['n_clauses']} "
          f"|sup|={len(info['sup1'])}{act_s}")
    if info["n_clauses"] == 0:
        print("  [warn] the formula has 0 clauses (empty or comment-only input?) — every "
              "assignment is a model, so these numbers measure nothing")
    if info.get("unsat"):
        # A PROOF, not a failure to find anything — say which, because the two print identically.
        print("  UNSAT: the formula has no models (proved during construction). Nothing to sample.")
    print(f"  {info['n_models']:,}{capped} models in {info['wall_total_s']:.1f}s e2e "
          f"({info['wall_timed_s']:.1f}s sampling)")
    print(f"  {info['samples_per_s']:,.0f} samples/s (steady-state), "
          f"{info.get('sps_e2e_total') or 0:,.0f}/s end-to-end | success {info['success_rate']*100:.0f}% "
          f"| sound={info['gpu_sound']}")
    if info["n_models"] == 0 and not info.get("unsat") and info["n_clauses"]:
        # Every trajectory conflicted. Nothing above says why, and the three causes want three
        # different responses from the user, so name them rather than leaving a bare 0.
        print("  [warn] no models: every trajectory hit a conflict. GPAP discards conflicted\n"
              "         trajectories rather than backtracking, so a tightly-constrained instance\n"
              "         can yield nothing. Try --portfolio b1 (one decision per round), a longer\n"
              "         --time-limit, or --allow-soft (faster on tight circuits, NOT uniform).")
    if info.get("n_collected"):
        # --n-cov's whole result used to be invisible without --stats: same printout with and
        # without the flag, while it silently paid for an extra collection pass.
        nd, nc = info.get("n_distinct"), info["n_collected"]
        cpu = info.get("cpu_sound")
        print(f"  quality [--n-cov]: {nd:,} distinct of {nc:,} collected "
              f"({100.0 * nd / nc:.0f}% distinct)"
              + (f" | cpu-verified={cpu}" if cpu is not None else ""))
    if info.get("mined"):
        m = info["mined"]
        print(f"  mined {m['n_units']} units / {m['n_binary']} binary / {m['n_ternary']} ternary "
              f"in {m['seconds']:.1f}s")
    if info.get("uniformity") is not None:
        print(f"  uniformity: {format_uniformity(info['uniformity'], verbose=True)}")
    if args.sim_baseline:
        # The rejection-sampling rival: random inputs + bit-parallel circuit simulation, no CNF.
        # Perfectly uniform by construction — its only weakness is p_accept. See simulate.py.
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
            # Two different reasons, and the old message asserted the wrong one unconditionally:
            # `--max-samples 0` used to print "0 models produced" after a run that produced
            # plenty of them.
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
            # A short file is explained where it happens. It used to be the opposite problem:
            # the writer went and got the full request with a second, unbounded sampling run, so a
            # capped run reported one model count and then wrote a larger number of samples.
            if short:
                why = ("the run stopped at {:,} models (--time-limit / no progress)".format(
                    info["n_models"]) if info.get("chunk_capped") else
                    "the winning config produced {:,} rows".format(len(mat)))
                print(f"         short because {why}; emitted rows all come from ONE config "
                      f"({describe_config(info.get('out_config'))}), never a mix of the portfolio's")
    if args.stats:
        # Drop the bulky arrays from the JSON — it is the stats/info sidecar, not the samples.
        stats = {k: v for k, v in info.items() if k not in ("samples", "models", "cols")}
        json.dump(stats, open(args.stats, "w"), indent=1, default=str)
        print(f"  wrote stats -> {args.stats}")


def _write_cnf(path, num_vars, clauses):
    """Plain DIMACS writer (header + 0-terminated clauses)."""
    with open(path, "w") as f:
        f.write(f"p cnf {num_vars} {len(clauses)}\n")
        for c in clauses:
            f.write(" ".join(map(str, c)) + " 0\n")


def _parse_pool(spec, cnf_path, num_vars):
    """Resolve --pool into DIMACS literals, or None for "every free literal".

    Three shapes, because the useful pool depends on what you know about the instance:
      'all'      -> None: sweep everything propagation left undetermined (the default)
      'support'  -> the formula's declared independent support (`c ind` lines) — for a circuit
                    CNF that is the primary inputs, where the interesting joint constraints are
      explicit  -> "1,-2,7" or "@file" with whitespace/comma-separated literals
    A variable is expanded to BOTH polarities; a signed literal is taken as given, which is how
    you ask about one side only.
    """
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
    """Tuples probed across every width — the sweep's own work unit."""
    return sum(res.probed.values())


def _cmd_mine(args):
    """Prove clauses the formula already entails, by propagating every k-subset of a literal set.

    Assuming a set of literals and reaching a conflict PROVES the formula entails the negation of
    that set — a clause. So this finds every entailed clause of each requested width that BCP
    alone can see, and adding them leaves the model set EXACTLY unchanged while making later
    propagation stronger.
    """
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

        # What each width WOULD cost, before spending the budget on it: the candidate count is
        # exact and needs no GPU work, and at width 3 over a wide pool it is the difference
        # between a 2-second sweep and one that cannot be run at all.
        # These are UPPER BOUNDS. Width 1 runs first and its proven units narrow the formula, so a
        # later width sees fewer free literals and its real count is smaller than printed here.
        # Printed from INSIDE the sweep (on_plan) so it reuses that engine — this used to build a
        # whole second LiteralProbe and drop it, for counts that cost nothing once one exists.
        def _plan(w, n, ceiling):
            # Width 1's count is ONE round of a fixpoint that reprobes the shrinking pool until
            # nothing new appears, so it is not the total it will probe; saying "up to N" there
            # invited exactly the comparison that made the totals look impossible.
            per = " per fixpoint round" if w == 1 else ""
            over = f"  > --max-tuples {ceiling:,}" if (ceiling > 0 and n > ceiling) else ""
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
            # can be reconciled with the pool it started from instead of dwarfing it unexplained.
            rnd = res.rounds.get(w, 1)
            over_rounds = f" over {rnd} fixpoint rounds" if rnd > 1 else ""
            print(f"    width {w}: {res.by_width.get(w, 0):,} clause(s)   "
                  f"probed {done:,}/{cand:,} ({cover}){over_rounds}   [{mark}]")
        if not res.clauses:
            # Per-width, never summed: width 1's counts are a fixpoint SUM and width 2's are a
            # C(P,2), so a pooled ratio divides two incomparable things (which is why it was
            # removed from MiningResult.summary()).
            done = sum(res.probed.values())
            partial = [w for w in widths if res.probed.get(w, 0) < res.candidates.get(w, 0)]
            if partial:
                cov = ", ".join(
                    f"width {w} {100.0 * res.probed.get(w, 0) / res.candidates[w]:.3g}%"
                    for w in partial if res.candidates.get(w))
                # The distinction that matters and that the old output erased: an empty result
                # from a partial sweep says nothing about the formula.
                print(f"  (nothing found in the {done:,} tuple(s) probed — that is only "
                      f"{cov} of each width's space, so this is NOT evidence that nothing is "
                      f"entailed. Raise --max-tuples/--seconds, or narrow --pool.)")
            elif all(res.exhausted.get(w, False) for w in widths):
                print("  (nothing entailed was found, and the search WAS exhaustive — the "
                      "formula's propagation is already tight at these widths)")
            else:
                # No candidates were even generated (a skipped or errored width) — say so
                # instead of claiming exhaustiveness the sweep never earned.
                print("  (nothing was probed — see the per-width status above)")

        if args.out:
            # The STRENGTHENED formula: original + mined. Sound by construction — every mined
            # clause is entailed, so the model set is unchanged.
            _write_cnf(args.out, nv, clauses + list(res.clauses))
            print(f"  wrote {args.out} ({len(clauses):,} original + {len(res.clauses):,} "
                  f"mined clauses)")
        if args.out_mined:
            _write_cnf(args.out_mined, nv, list(res.clauses))
            print(f"  wrote {args.out_mined} ({len(res.clauses):,} mined clauses only)")
        if args.stats:
            # Report WHAT WAS DONE, not only what was found. An empty `clauses` with no coverage
            # numbers is unreadable: it cannot distinguish "this formula entails nothing of this
            # width" from "we probed 0.0008% of the space and ran out of budget".
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
                    # PER WIDTH. A pooled `coverage` used to be written here — the same
                    # incomparable ratio the terminal stopped printing, but durable on disk.
                    "coverage": {str(w): (res.probed.get(w, 0) / res.candidates[w])
                                 for w in sorted(res.candidates) if res.candidates.get(w)},
                    "probes_per_s": (total_probed / res.seconds) if res.seconds else None,
                    # The HARDWARE-work metric. clauses/s and probes/s both depend on the
                    # instance (how many clauses it entails, how fast a tuple conflicts);
                    # implications/s is what the engine actually did, so it is the number to
                    # compare across instances, GPUs and engine changes.
                    "implications": res.implications,
                    "implications_per_s": res.implications_per_s,
                    "complete": all(res.exhausted.get(w, False) for w in widths),
                    "by_width": {str(k): v for k, v in sorted(res.by_width.items())},
                    "candidates": {str(k): v for k, v in sorted(res.candidates.items())},
                    "probed": {str(k): v for k, v in sorted(res.probed.items())},
                    # width 1's candidates/probed are SUMS over this many fixpoint rounds — without
                    # it a reader cannot reconcile them with the pool the sweep started from.
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
    # Delegate to the benchmark table driver (gpap/benchmark.py — inside the package, so this
    # works from any directory).
    from gpap import benchmark
    argv = ["--glob", args.glob, "--n-time", str(args.n),
            "--time-limit", str(args.time_limit), "--n-cov", str(args.n_cov),
            "--portfolio", args.portfolio,
            "--encoder", args.encoder, "--abc-opt", args.abc_opt, "--mine", args.mine,
            "--uniformity", args.uniformity, "--target", args.target,
            "--cycles", str(args.cycles), "--seed", str(args.seed),
            "--k-cap", str(args.k_cap)]
    if args.seeds:
        argv += ["--seeds"] + [str(s) for s in args.seeds]
    if args.twise:
        argv += ["--twise"]
    if args.allow_soft:
        # Was DEFINED on this parser and accepted by the driver, but never forwarded: the flag
        # silently did nothing, and unlike `gpap sample` it printed no warning either, so a bench
        # run asked for the soft config and quietly got the uniformity-safe portfolio instead.
        argv += ["--allow-soft"]
    if args.no_els:
        argv += ["--no-els"]
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
    conversion attributes: cycles, seed, pin_outputs, encoder, abc, abc_opt, target (see
    _add_encoder_args) — the bench front-end reuses this for circuit globs."""
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
    """Report the OPTIONAL berkeley-abc circuit encoder (--encoder abc)."""
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
    # The engine core (libgpap): the C++ library Python and C++ consumers share.
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
    # Power-limit / clock warning (the measured 40 W-vs-80 W + battery-clock gotchas). Before any
    # headline benchmark, confirm the card isn't power-capped or downclocked — ratios in one state
    # are valid but absolute numbers across states are not.
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
    """Circuit-encoder flags, shared by every subcommand that accepts a circuit input
    (`sample`, `mine`, `circuit2cnf`) so the conversion means one thing everywhere."""
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
    """Exceptions that mean "bad input or bad environment", not "gpap has a bug".

    Only these are collapsed to a one-line message; anything else keeps its traceback, because a
    traceback for an internal fault is information and a traceback for a mistyped filename is not.
    `GpapError` is imported lazily — the core may not be built, and that case has its own message."""
    types = [OSError, ValueError]
    try:
        from gpap.engine._core import GpapError
        types.append(GpapError)
    except Exception:                                     # noqa: BLE001
        pass
    return tuple(types)


def main(argv=None):
    p = argparse.ArgumentParser(prog="gpap", description=__doc__.splitlines()[0])
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
    s.add_argument("--device", default=None)
    s.add_argument("--k-cap", type=_k_cap_arg, default="auto",
                   help="ceiling on the per-chunk batch K. 'auto' (default) sizes it to this "
                        "GPU's VRAM; pass an int to fix it (needed to A/B two different GPUs)")
    s.add_argument("--portfolio", choices=["circuit", "default", "b8", "b1", "s1"],
                   default="circuit",
                   help="single-config choices (b8/b1/s1) pin ONE config and disable the bandit — use them for any cross-GPU or before/after MEASUREMENT, since the bandit can otherwise pick differently per machine and confound the comparison")
    s.add_argument("--allow-soft", action="store_true",
                   help="add the soft risk-polarity config to the bandit portfolio. Higher "
                        "success on tight/deep circuits, but it BIASES the distribution "
                        "(Barbarik rejects it 5/5) — opt in only when throughput matters and "
                        "uniformity does not")
    s.add_argument("--no-simplify", action="store_true")
    s.add_argument("--no-els", action="store_true",
                   help="disable equivalent-literal substitution (ON by default: it merges "
                        "mutually-implied literals, removing variables and clauses. Measured 1.42x "
                        "geomean over 24 circuit instances, and uniformity-fuzzed at 0 verdict "
                        "flips / 0 unsound rows against the exact oracle)")
    s.add_argument("--cycles", type=_POS_INT, default=25,
                   help="unroll depth (circuit inputs; combinational AIGs clamp to 1)")
    s.add_argument("--pin-outputs", type=_NONNEG_INT, default=None,
                   help="how many circuit outputs to pin (default: all). 0 pins NOTHING — the "
                        "unconstrained free-hypercube baseline, not a degenerate value")
    s.add_argument("--mine", choices=["auto", "on", "off"], default="auto",
                   help="entailed-clause mining: 'auto' (on, bounded by a 5s total budget) | "
                        "'on' (force, ignores the size gate) | 'off'")
    s.add_argument("--uniformity", choices=["off", "fast", "exact"], default="off",
                   help="uniformity readout: 'off' (default — the readout costs an extra "
                        "sampling pass) | 'fast' (repeat-rate check: l2_ratio, ~1.0 = uniform, "
                        ">1.5 = certified skew; no solver work) | 'exact' (also enumerate small "
                        "solution spaces and run a chi-square test vs uniform — the sensitive "
                        "check). Real certificates on big spaces: eval/uniformity/barbarik_gpap.py")
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
    b.add_argument("--k-cap", type=_k_cap_arg, default="auto",
                   help="ceiling on the per-chunk batch K. 'auto' (default) sizes it to this "
                        "GPU's VRAM; pass an int to fix it (needed to A/B two different GPUs)")
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
                        "(Barbarik rejects it 5/5) — opt in only when throughput matters and "
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
    b.add_argument("--uniformity", choices=["off", "fast", "exact"], default="off",
                   help="per-instance uniformity readout: 'off' (default — costs an extra "
                        "sampling pass) | 'fast' (repeat-rate check) | 'exact' (chi-square on "
                        "enumerable spaces)")
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
    # 5 = the same conversion/target seed `gpap sample` and `gpap bench` default to. It was 1
    # here, so CNFs built via this subcommand pinned a DIFFERENT rare pattern than the sweep's
    # in-process conversions of the same circuit — the HTS rival arm inherited a degenerate
    # (zero-search) b13 instance from exactly that mismatch.
    c.add_argument("--seed", type=int, default=5)
    c.add_argument("--pin-outputs", type=_NONNEG_INT, default=None)
    _add_encoder_args(c)
    c.set_defaults(func=_cmd_circuit2cnf)

    i = sub.add_parser("info", help="device, triton availability, power-limit warning")
    i.set_defaults(func=_cmd_info)

    args = p.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        raise SystemExit("interrupted") from None
    except _user_error_types() as e:
        # The core already produces excellent diagnostics ("<file> line 3: unexpected character
        # 'x' — not DIMACS"); what reached the user was a ctypes stack trace with that message
        # buried at the bottom. Print the message, keep the traceback for $GPAP_TRACEBACK=1.
        if os.environ.get("GPAP_TRACEBACK"):
            raise
        raise SystemExit(f"gpap {args.cmd}: {e}") from None


if __name__ == "__main__":
    main()
