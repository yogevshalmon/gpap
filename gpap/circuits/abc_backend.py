"""OPTIONAL ABC-based circuit encoder: unroll + logic-optimize with berkeley-abc, then Tseitin
the resulting AIG. A drop-in alternative to the native encoders in `bench.py` (.bench) and
`aig2cnf.py` (.aig/.aag).

Why: propagate is ~80% of sampling wall and scales with M. ABC's structural hashing, `dc2`
rewriting and `fraig` SAT-sweeping shrink the unrolled circuit far below a gate-by-gate Tseitin
(measured on this repo's benches: 2.2x on b10-c25 up to 7.8x on b14-c25 vs our Tseitin after its
own unit_closure).

SOUNDNESS. `strash`/`dc2`/`fraig`/`&fraig`/`scorr` are function-preserving, and `zero` +
`frames -F T -i` reproduces exactly our reset-state-0 unroll — verified against `bench.simulate`
(0/24 mismatches) and gated by tests/test_abc_encoder.py, which enumerates the projected solution
set under both encoders and asserts SET EQUALITY. Only internal gate variables are restructured.
`zero` is MANDATORY: without it ABC turns the T=0 register state into FREE primary inputs.

COST — the `opt` presets (measured 2026-07-10, see docs/findings/INSIGHTS.md):
  "full" : optimize AFTER unrolling (`dc2; dc2; fraig` on the T-times-larger AIG). Best shrink —
           the post-unroll fraig catches cross-frame equivalences (b12-c25 collapses 14x) — but
           that fraig is also the whole cost: ~99 s on b15-c25 (160k ANDs).
  "fast" : optimize the SEQUENTIAL AIG first (it is T times smaller; seconds), unroll, then one
           post `dc2`. Bounded and predictable (<=4 s everywhere measured); equals full's shrink
           on b14-c25 (1.47x vs 1.46x) but forgoes the cross-frame sweep (b15: 1.18x vs 2.17x —
           worth it below ~10^5-sample contracts; pass opt="full" beyond).
  "auto" : "full" when the estimated unrolled size is small — trivially cheap there AND it is
           where the cross-frame collapses live (b12 est 26k ANDs) — else "fast".

ABC is never vendored and never a hard dependency — `scripts/build_abc.sh` fetches a pinned
upstream commit and installs the binary into the GPAP cache; callers fall back to the native
Tseitin encoders when no binary is found.
"""
import os
import shutil
import subprocess
import tempfile

from gpap.circuits.aiger import Aig, read_aiger, split_frame_name

# (pre_frames, post_frames) optimization script halves; either may be empty. "fast"'s post pass
# is dc2 ONLY, deliberately: a post-unroll equivalence sweep (fraig or &fraig) is where both the
# b12-class cross-frame collapse (12-14x) AND the unbounded cost live (&fraig >300s on b15-c25's
# 160k-AND unroll) — a default must be predictable, and the cross-frame cases are small enough
# that `auto` routes them to "full" anyway (2026-07-10 profiling).
ABC_OPT_PRESETS = {
    "full": ("", "dc2; dc2; fraig"),
    "fast": ("dc2; dc2; fraig", "dc2"),
}
# `auto` switches full->fast above this estimated raw unrolled AND count (gate-count x T for
# .bench, A x T for AIGER — a deliberate overestimate; strash typically hashes 2-4x away).
AUTO_FAST_EST_ANDS = 50_000


def find_abc(explicit: str | None = None) -> str | None:
    """Locate an `abc` binary: explicit path -> $GPAP_ABC -> PATH -> $GPAP_ABC_HOME/bin/abc
    (default ~/.cache/gpap/bin/abc, where scripts/build_abc.sh installs it). None if absent."""
    for cand in (explicit, os.environ.get("GPAP_ABC")):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    found = shutil.which("abc")
    if found:
        return found
    home = os.environ.get("GPAP_ABC_HOME") or os.path.join(os.path.expanduser("~"), ".cache/gpap")
    cand = os.path.join(home, "bin", "abc")
    return cand if os.path.isfile(cand) and os.access(cand, os.X_OK) else None


def abc_available(explicit: str | None = None) -> bool:
    return find_abc(explicit) is not None


class _AbcError(RuntimeError):
    pass


def resolve_opt(opt: str, est_unrolled_ands: int) -> str:
    if opt == "auto":
        return "fast" if est_unrolled_ands > AUTO_FAST_EST_ANDS else "full"
    if opt not in ABC_OPT_PRESETS:
        raise ValueError(f"unknown abc opt preset {opt!r} (full | fast | auto)")
    return opt


def _run_abc(abc: str, script: str, timeout: float = 1800.0) -> str:
    # Run in a scratch cwd: abc drops an `abc.history` file next to wherever it is invoked, and we
    # must not litter the caller's project directory. All paths in `script` are absolute.
    with tempfile.TemporaryDirectory(prefix="gpap-abc-") as cwd:
        p = subprocess.run([abc, "-q", script], capture_output=True, text=True,
                           timeout=timeout, cwd=cwd)
    if p.returncode != 0:
        raise _AbcError(f"abc failed (rc={p.returncode}): {(p.stderr or p.stdout).strip()[:400]}")
    return p.stdout


def unroll_optimize_to_aig(input_path: str, cycles: int, aig_path: str, *,
                           abc: str | None = None, opt: str = "full") -> None:
    """Run ABC: read circuit (.bench or .aig/.aag) -> AIG -> pin registers to 0 -> [pre-opt] ->
    unroll `cycles` frames -> [post-opt] -> write binary AIGER with the I/O symbol table."""
    exe = find_abc(abc)
    if exe is None:
        raise _AbcError("no `abc` binary found (build it with scripts/build_abc.sh)")
    reader = "read_aiger" if input_path.endswith((".aig", ".aag")) else "read_bench"
    pre, post = ABC_OPT_PRESETS[opt]
    pre = f"{pre}; " if pre else ""
    post = f"{post}; " if post else ""
    input_path, aig_path = os.path.abspath(input_path), os.path.abspath(aig_path)
    out = _run_abc(exe, f"{reader} {input_path}; strash; zero; {pre}frames -F {cycles} -i; {post}"
                        f"write_aiger -s {aig_path}")
    if not os.path.exists(aig_path) or os.path.getsize(aig_path) == 0:
        # abc reports command failures on stdout with exit code 0 — surface its message.
        raise _AbcError(f"abc produced no framed AIG for {input_path}: {out.strip()[-300:]}")


def _read_framed(input_path: str, cycles: int, abc, opt) -> Aig:
    tmp = tempfile.NamedTemporaryFile(suffix=".aig", delete=False).name
    try:
        unroll_optimize_to_aig(input_path, cycles, tmp, abc=abc, opt=opt)
        return read_aiger(tmp)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _framed_to_cnf(faig: Aig, T: int, in_names: list[str], out_names: list[str],
                   target: list[bool], pin_count: int | None):
    """Tseitin the framed (combinational) AIG and pin the last-cycle outputs.

    Signal identity is POSITIONAL: ABC's `frames` emits CIs/COs frame-major in original order
    (framed input t*I+i = input i at cycle t; likewise outputs) — so this works for nameless
    AIGER inputs too. When the symbol table has parseable `<name>_<frame>` entries they are
    VERIFIED against the positional layout (mismatch raises; never silently mis-mapped)."""
    from gpap.circuits.bench import CnfBuilder

    n_in, n_out = len(in_names), len(out_names)
    if faig.n_latches:
        raise _AbcError("framed AIG still has latches — `frames` did not unroll")
    if faig.n_inputs != n_in * T or faig.n_outputs != n_out * T:
        raise _AbcError(f"framed AIG I/O count mismatch: I={faig.n_inputs} (want {n_in}x{T}), "
                        f"O={faig.n_outputs} (want {n_out}x{T})")
    for k, nm in faig.input_names.items():
        sp = split_frame_name(nm)
        if sp and sp[0] in in_names and (sp[0] != in_names[k % n_in] or sp[1] != k // n_in):
            raise _AbcError(f"framed input {k} named {nm!r} contradicts positional layout")
    for k, nm in faig.output_names.items():
        sp = split_frame_name(nm)
        if sp and sp[0] in out_names and (sp[0] != out_names[k % n_out] or sp[1] != k // n_out):
            raise _AbcError(f"framed output {k} named {nm!r} contradicts positional layout")

    # CNF var `v` == AIG var `v` (1..M); one extra var pinned FALSE encodes the AIGER constants.
    b = CnfBuilder()
    b.nv = faig.max_var + 1
    const = b.nv
    b.clauses = [[-const]]

    def lit(aig_lit: int) -> int:
        if aig_lit == 0:
            return const                           # constant FALSE
        if aig_lit == 1:
            return -const                          # constant TRUE
        v = aig_lit >> 1
        return -v if (aig_lit & 1) else v

    for lhs, r0, r1 in faig.ands:                  # y <-> a AND b : 2- and 3-literal clauses only
        y, a, c = lit(lhs), lit(r0), lit(r1)
        b.clauses.append([-y, a])
        b.clauses.append([-y, c])
        b.clauses.append([y, -a, -c])

    pin = range(n_out) if pin_count is None else range(min(pin_count, n_out))
    for j in pin:                                  # last-cycle output j = framed CO (T-1)*n_out+j
        ol = lit(faig.outputs[(T - 1) * n_out + j])
        b.clauses.append([ol if target[j] else -ol])

    # Support var for input i at cycle t = framed CI t*n_in+i -> AIG var (index+1). Ordered
    # input-major to match the native encoders' label convention.
    support = [t * n_in + i + 1 for i in range(n_in) for t in range(T)]
    labels = [f"{in_names[i]}@{t}" for i in range(n_in) for t in range(T)]
    return b, support, labels


def build_cnf_abc(circ, bench_path: str, cycles: int, seed: int, pin_outputs: int | None = None,
                  *, abc: str | None = None, opt: str = "auto", target=None):
    """ABC-encoded counterpart of `bench.build_cnf` (.bench input).

    Returns (builder, support, target, support_labels) exactly like the native encoder. The
    pinned target comes from the SAME simulation of the ORIGINAL circuit with the SAME seed
    (or an explicit `target` override), so the two encoders describe the same projected
    solution space."""
    from gpap.circuits.bench import random_target

    inputs, outputs, _dffs, gates = circ
    if target is None:
        target = random_target(circ, cycles, seed)

    opt = resolve_opt(opt, len(gates) * cycles)
    faig = _read_framed(bench_path, cycles, abc, opt)
    b, support, labels = _framed_to_cnf(faig, cycles, list(inputs), list(outputs),
                                        [target[o] for o in outputs], pin_outputs)
    return b, support, target, labels


def build_cnf_abc_aig(aig: Aig, aig_path: str, cycles: int, seed: int,
                      pin_outputs: int | None = None, *, abc: str | None = None,
                      opt: str = "auto", target=None):
    """ABC-encoded counterpart of `aig2cnf.build_cnf_aig` (.aig/.aag input)."""
    from gpap.circuits.aig2cnf import input_labels, random_target_aig

    if any(r for _cur, _nxt, r in aig.latches):
        # ABC's `zero` (needed for .bench, harmless for all-zero resets) would silently override
        # a reset-1 latch to 0 and change the sampled space — refuse rather than mis-encode.
        raise _AbcError(f"{aig_path}: latches with reset=1 — use the native encoder "
                        "(encoder='tseitin'), which honours AIGER reset values")
    T = 1 if aig.n_latches == 0 else cycles
    if target is None:
        target = random_target_aig(aig, T, seed)

    opt = resolve_opt(opt, len(aig.ands) * T)
    # Hand ABC a canonical BINARY rewrite of the parsed circuit rather than the original file:
    # ABC's read_aiger does not accept ASCII `aag`, and canonicalization also normalizes exotic
    # (valid) variable numberings. Symbols are preserved so the frames name-verify still applies.
    from gpap.circuits.aiger import write_aiger_binary
    tmp = tempfile.NamedTemporaryFile(suffix=".aig", delete=False).name
    try:
        write_aiger_binary(aig, tmp)
        faig = _read_framed(tmp, T, abc, opt)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    names = input_labels(aig)
    out_names = [aig.output_names.get(j, f"o{j}") for j in range(aig.n_outputs)]
    b, support, labels = _framed_to_cnf(faig, T, names, out_names, target, pin_outputs)
    return b, support, target, labels
