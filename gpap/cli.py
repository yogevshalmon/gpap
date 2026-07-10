"""`gpap` command-line entry point.

Subcommands:
  gpap sample <cnf|bench> [-n] [--time-limit] [--seed] [--json out.json]  — sample one instance
  gpap bench  <glob> [--cmsgen PATH] [--n-cov] ...                         — benchmark a suite
  gpap circuit2cnf <bench|dir> --cycles T [--pin-outputs k]                — unroll .bench -> CNF
  gpap info                                                                — device / triton / power

Library code uses `logging`; user-facing output lives here (this is the print layer).
"""
import argparse
import os
import sys


def _cmd_sample(args):
    import json
    import torch
    from gpap.sampling.pipeline import sample_file, CIRCUIT_PORTFOLIO, DEFAULT_PORTFOLIO

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    path = args.input
    tmp = None
    encode_s = None
    if path.endswith((".bench", ".aig", ".aag")):
        # unroll to a temporary CNF first (the conversion embeds the wire map as `c wire ...` and
        # the encoder provenance as `c encoder ...`, so sample output can name each column by its
        # circuit wire and the pipeline can skip work the encoder already did).
        import tempfile
        import time as _time
        tmp = tempfile.NamedTemporaryFile(suffix=".cnf", delete=False).name
        _t0 = _time.perf_counter()
        if path.endswith(".bench"):
            from gpap.circuits.bench import convert_one
            convert_one(path, tmp, args.cycles, args.seed, pin_outputs=args.pin_outputs,
                        encoder=args.encoder, abc_path=args.abc, abc_opt=args.abc_opt)
        else:
            from gpap.circuits.aig2cnf import convert_aig_one
            convert_aig_one(path, tmp, args.cycles, args.seed, pin_outputs=args.pin_outputs,
                            encoder=args.encoder, abc_path=args.abc, abc_opt=args.abc_opt)
        encode_s = _time.perf_counter() - _t0
        print(f"  encoded in {encode_s:.1f}s [{args.encoder}"
              f"{'/' + args.abc_opt if args.encoder != 'tseitin' else ''}]")
        path = tmp
    portfolio = CIRCUIT_PORTFOLIO if args.portfolio == "circuit" else DEFAULT_PORTFOLIO
    # Only collect sample bits when an output file is requested: up to `-n` solutions, never more
    # than the --max-samples safety ceiling (default 100k; raise it for a full dump).
    out_cap = min(args.n, args.max_samples) if args.out else 0
    mine = {"auto": "auto", "on": True, "off": False}[args.mine]
    info = sample_file(path, device, n_time=args.n, n_cov=args.n_cov, out_cap=out_cap,
                       seed=args.seed, simplify=not args.no_simplify, els=args.els, mine=mine,
                       configs=portfolio, time_limit=args.time_limit, use_kernel=args.use_kernel)
    if encode_s is not None:
        info["encode_s"] = encode_s          # circuit->CNF encode wall (stats sidecar honesty)
        info["encoder"] = args.encoder
    capped = "<" if info.get("chunk_capped") else ""
    print(f"{os.path.basename(args.input)}: N={info['nv']} M={info['n_clauses']} "
          f"|sup|={info['support_size']}")
    print(f"  {info['n_models']:,}{capped} models in {info['wall_total_s']:.1f}s e2e "
          f"({info['wall_timed_s']:.1f}s sampling)")
    print(f"  {info['samples_per_s']:,.0f} samples/s (steady-state), "
          f"{info.get('sps_e2e_total') or 0:,.0f}/s end-to-end | success {info['success_rate']*100:.0f}% "
          f"| sound={info['gpu_sound']}")
    if info.get("mined"):
        m = info["mined"]
        print(f"  mined {m['n_units']} units / {m['n_binary']} binary / {m['n_ternary']} ternary "
              f"in {m['seconds']:.1f}s")
    if args.out:
        from gpap.sampling.output import write_samples
        mat, cols = info.get("samples"), info.get("cols")
        if mat is None or len(mat) == 0:
            print("  no samples to write (0 models produced)")
        else:
            meta = {"instance": os.path.basename(args.input), "seed": args.seed,
                    "success_rate": round(info["success_rate"], 4)}
            write_samples(args.out, mat, cols, fmt=args.out_format,
                          wire_map=info.get("wire_map"), meta=meta)
            print(f"  wrote {len(mat):,} samples ({mat.shape[1]} cols) -> {args.out} "
                  f"[{args.out_format}]")
    stats_path = args.stats or args.json
    if stats_path:
        # Drop the bulky arrays from the JSON — it is the stats/info sidecar, not the samples.
        stats = {k: v for k, v in info.items() if k not in ("samples", "models", "cols")}
        json.dump(stats, open(stats_path, "w"), indent=1, default=str)
        print(f"  wrote stats -> {stats_path}")
    if tmp:
        os.unlink(tmp)


def _cmd_bench(args):
    # Delegate to the benchmark table script (kept thin around the pipeline).
    from scripts import sample_benchmark
    sys.argv = ["gpap bench", "--glob", args.glob, "--n-time", str(args.n),
                "--time-limit", str(args.time_limit), "--n-cov", str(args.n_cov),
                "--portfolio", args.portfolio, "--use-kernel", args.use_kernel]
    if args.cmsgen:
        sys.argv += ["--cmsgen", args.cmsgen]
    if args.device:
        sys.argv += ["--device", args.device]
    if args.output:
        sys.argv += ["--output", args.output]
    sample_benchmark.main()


def _convert_circuit(path, outp, args):
    """Dispatch one circuit file (.bench or .aig/.aag) to its converter."""
    if path.endswith(".bench"):
        from gpap.circuits.bench import convert_one
        convert_one(path, outp, args.cycles, args.seed, pin_outputs=args.pin_outputs,
                    encoder=args.encoder, abc_path=args.abc, abc_opt=args.abc_opt)
    else:
        from gpap.circuits.aig2cnf import convert_aig_one
        convert_aig_one(path, outp, args.cycles, args.seed, pin_outputs=args.pin_outputs,
                        encoder=args.encoder, abc_path=args.abc, abc_opt=args.abc_opt)


def _cmd_circuit2cnf(args):
    import glob as _glob
    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        files = sorted(f for pat in ("*.bench", "*.aig", "*.aag")
                       for f in _glob.glob(os.path.join(args.input, "**", pat), recursive=True))
        for fp in files:
            stem = os.path.splitext(os.path.basename(fp))[0]
            outp = os.path.join(args.output, f"{stem}-c{args.cycles}.cnf")
            try:
                _convert_circuit(fp, outp, args)
            except Exception as e:  # noqa: BLE001
                print(f"{fp}: SKIPPED ({type(e).__name__}: {e})", file=sys.stderr)
    else:
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
        print("device: CPU (no CUDA) — the PyTorch BCP oracle runs; fused kernels are off")
        _print_abc_status()
        return
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    free, total = torch.cuda.mem_get_info(dev)
    print(f"device: {props.name}  {total/1e9:.1f} GB ({free/1e9:.1f} GB free)  "
          f"SM {props.major}.{props.minor} x{props.multi_processor_count}")
    try:
        from gpap.engine.kernels import TritonBCP
        print(f"triton fused kernels: {'available' if TritonBCP.available() else 'UNAVAILABLE'}")
    except Exception as e:  # noqa: BLE001
        print(f"triton fused kernels: import failed ({e})")
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
    """Circuit-encoder flags, shared by `sample` (circuit input) and `circuit2cnf`."""
    sp.add_argument("--encoder", choices=["tseitin", "abc", "auto"], default="tseitin",
                    help="circuit->CNF encoder: 'tseitin' (native, default) | 'abc' (unroll + "
                         "dc2/fraig via berkeley-abc, 2-8x fewer clauses, same solution space; "
                         "needs an abc binary — see scripts/build_abc.sh) | 'auto' (abc if found)")
    sp.add_argument("--abc", default=None,
                    help="path to an `abc` binary (else $GPAP_ABC, PATH, ~/.cache/gpap/bin/abc)")
    sp.add_argument("--abc-opt", choices=["full", "fast", "auto"], default="auto",
                    help="abc optimization effort: 'full' (optimize the unrolled AIG — best "
                         "shrink, expensive on big unrolls) | 'fast' (optimize pre-unroll + one "
                         "cheap post sweep) | 'auto' (full when small, fast when big; default)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="gpap", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="sample models from one CNF, .bench or .aig/.aag file")
    s.add_argument("input", help="a DIMACS .cnf, or a circuit (.bench / .aig / .aag — unrolled first)")
    s.add_argument("-n", type=int, default=50_000, help="models to deliver (a contract; default 50k)")
    s.add_argument("--time-limit", type=float, default=60.0, help="per-instance sampling wall cap (s)")
    s.add_argument("--seed", type=int, default=5)
    s.add_argument("--n-cov", type=int, default=0, help="collect this many for quality/soundness")
    s.add_argument("--device", default=None)
    s.add_argument("--portfolio", choices=["circuit", "default"], default="circuit")
    s.add_argument("--no-simplify", action="store_true")
    s.add_argument("--els", action="store_true", help="equivalent-literal substitution (removes vars)")
    s.add_argument("--use-kernel", default="auto")
    s.add_argument("--cycles", type=int, default=25,
                   help="unroll depth (circuit inputs; combinational AIGs clamp to 1)")
    s.add_argument("--pin-outputs", type=int, default=None, help="outputs to pin (circuit input)")
    s.add_argument("--mine", choices=["auto", "on", "off"], default="auto",
                   help="entailed-clause mining: 'auto' (on, except for ABC-encoded CNFs, whose "
                        "backbone dc2/fraig already removed — measured 0-45 units for a full 5s "
                        "budget) | 'on' (force) | 'off'")
    _add_encoder_args(s)
    s.add_argument("--out", default=None,
                   help="write the sampled solutions here (compact support projections, one per "
                        "sample). For a .bench input each column is a primary-input wire@cycle")
    s.add_argument("--out-format", choices=["bits", "npy", "dimacs"], default="bits",
                   help="--out format: 'bits' (0/1 text, default) | 'npy' (numpy uint8 array) | "
                        "'dimacs' (signed lits, CMSGen-style)")
    s.add_argument("--max-samples", type=int, default=100_000,
                   help="cap on solutions written to --out (default 100k; raise for a full dump)")
    s.add_argument("--stats", default=None, help="write the run's info/stats JSON here")
    s.add_argument("--json", default=None, help=argparse.SUPPRESS)   # deprecated alias for --stats
    s.set_defaults(func=_cmd_sample)

    b = sub.add_parser("bench", help="benchmark a glob of instances (vs CMSGen if given)")
    b.add_argument("glob")
    b.add_argument("-n", "--n-time", dest="n", type=int, default=50_000)
    b.add_argument("--time-limit", type=float, default=60.0)
    b.add_argument("--n-cov", type=int, default=0)
    b.add_argument("--cmsgen", default=None, help="path to a cmsgen binary (enables comparison)")
    b.add_argument("--portfolio", choices=["circuit", "default"], default="circuit")
    b.add_argument("--use-kernel", default="auto")
    b.add_argument("--device", default=None)
    b.add_argument("--output", default=None)
    b.set_defaults(func=_cmd_bench)

    c = sub.add_parser("circuit2cnf",
                       help="unroll a circuit (.bench/.aig/.aag, or a dir of them) to projected CNF")
    c.add_argument("input", help="a .bench/.aig/.aag file or a directory of them")
    c.add_argument("output", help="output .cnf (single) or output directory")
    c.add_argument("--cycles", type=int, default=25)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--pin-outputs", type=int, default=None)
    _add_encoder_args(c)
    c.set_defaults(func=_cmd_circuit2cnf)

    i = sub.add_parser("info", help="device, triton availability, power-limit warning")
    i.set_defaults(func=_cmd_info)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
