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

COST — the `opt` presets (measured; see docs/findings/INSIGHTS.md):
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

from gpap import paths

from gpap.circuits.aiger import Aig, read_aiger, split_frame_name

# (pre_frames, post_frames) optimization script halves; either may be empty. "fast"'s post pass
# is dc2 ONLY, deliberately: a post-unroll equivalence sweep (fraig or &fraig) is where both the
# b12-class cross-frame collapse (12-14x) AND the unbounded cost live (&fraig >300s on b15-c25's
# 160k-AND unroll) — a default must be predictable, and the cross-frame cases are small enough
# that `auto` routes them to "full" anyway.
ABC_OPT_PRESETS = {
    "full": ("", "dc2; dc2; fraig"),
    "fast": ("dc2; dc2; fraig", "dc2"),
}
# `auto` switches full->fast above this estimated raw unrolled AND count (gate-count x T for
# .bench, A x T for AIGER — a deliberate overestimate; strash typically hashes 2-4x away).
AUTO_FAST_EST_ANDS = 50_000


def find_abc(explicit: str | None = None) -> str | None:
    """Locate an `abc` binary: explicit path -> $GPAP_ABC -> PATH -> GPAP's own tool dir
    (see `gpap.paths`, where scripts/build_abc.sh installs it). None if absent."""
    for cand in (explicit, os.environ.get("GPAP_ABC")):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    found = shutil.which("abc")
    if found:
        return found
    cand = str(paths.tool("abc"))
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


# --------------------------------------------------------------------------------------------
# Circuit-level simplification: .bench -> ABC -> .bench
#
# For the GPU-RIVAL comparison. DEMOTIC and Recurrent-CircuitSAT consume .bench directly and do
# their own unrolling, so if GPAP's ABC arm is measured on an ABC-optimized circuit, the rivals
# must be offered the SAME optimized circuit — otherwise the encoder win is credited to the
# sampler. The route is AIGER, NOT abc's `write_bench`: that command emits `LUT 0x8 (a, b)` and
# `DFFRSE(d, gnd, gnd, gnd, gnd)`, which neither our parser nor either rival's understands.
# An AIG is 2-input ANDs + inversions + reset-0 latches, which is exactly the .bench primitive
# set everyone here accepts.
#
# What the rivals get is the SEQUENTIAL-AIG optimization (`dc2; dc2; fraig` — the pre-unroll half
# of our "fast" preset). The post-unroll pass in GPAP's abc encoder has no sequential equivalent:
# once frames are expanded the circuit is combinational and can no longer be handed to a tool
# that unrolls it itself. Documented, not hidden.

def write_bench_from_aig(aig: Aig, path: str, *, name: str = "abc") -> None:
    """Emit an AIG as a primitive-gate .bench (INPUT/OUTPUT/AND/NOT/BUF/DFF).

    Names: primary inputs and latches keep their AIGER symbol names when present (so the
    rivals' filename/positional target tables still line up); AND nodes become `n<var>`.
    Primary outputs get a fresh `PO<j>_<name>` buffer signal so an inverted or shared driver
    can still be named, and OUTPUT() order is the AIG's output order — the pinning convention
    everywhere in this repo is positional."""
    import re

    def _clean(s, fallback):
        s = re.sub(r"\W", "_", s or "")
        return s if s and not s[0].isdigit() else (fallback if not s else f"n_{s}")

    var_name = {}
    lines_io, lines_gate = [], []
    for k in range(aig.n_inputs):
        nm = _clean(aig.input_names.get(k), f"pi{k}")
        var_name[aig.inputs[k] >> 1] = nm
        lines_io.append(f"INPUT({nm})")
    for k, (cur, _nxt, _r) in enumerate(aig.latches):
        var_name[cur >> 1] = f"lat{k}"          # AIGER latch symbols are not parsed (aiger.py)
    # constants: an AIG literal may be 0/1; build them from the first PI (x AND ~x), the same
    # trick eval/rivals/unroll_to_bench.py uses, so no gate library extension is needed.
    anchor = var_name.get(aig.inputs[0] >> 1) if aig.n_inputs else None
    const0 = "GPAP_ZERO"
    if anchor:
        lines_gate.append(f"GPAP_NOT_ANCHOR = NOT({anchor})")
        lines_gate.append(f"{const0} = AND({anchor}, GPAP_NOT_ANCHOR)")
    neg_of = {}

    def sig(lit: int) -> str:
        """Signal name for an AIG literal, materialising an inverter (once) when negated."""
        if lit <= 1:
            if anchor is None:
                raise _AbcError("constant-driven circuit with no primary input")
            if lit == 0:
                return const0
            if "GPAP_ONE" not in neg_of:
                lines_gate.append(f"GPAP_ONE = NOT({const0})")
                neg_of["GPAP_ONE"] = "GPAP_ONE"
            return "GPAP_ONE"
        base = var_name[lit >> 1]
        if not (lit & 1):
            return base
        if base not in neg_of:
            neg_of[base] = f"{base}_NEG"
            lines_gate.append(f"{base}_NEG = NOT({base})")
        return neg_of[base]

    # sorted by lhs == topological: AIGER requires lhs > rhs0 >= rhs1 for every AND
    for lhs, r0, r1 in sorted(aig.ands):
        var_name[lhs >> 1] = f"n{lhs >> 1}"
        lines_gate.append(f"n{lhs >> 1} = AND({sig(r0)}, {sig(r1)})")
    for k, (cur, nxt, _r) in enumerate(aig.latches):   # reset 0 (ABC `zero` guarantees it)
        lines_gate.append(f"{var_name[cur >> 1]} = DFF({sig(nxt)})")
    for j, ol in enumerate(aig.outputs):
        po = f"PO{j}_{_clean(aig.output_names.get(j), str(j))}"
        # BUF, not BUFF: our parser takes either, but DEMOTIC's and
        # Recurrent-CircuitSAT's pgates modules only define BUF.
        lines_gate.append(f"{po} = BUF({sig(ol)})")
        lines_io.append(f"OUTPUT({po})")
    with open(path, "w") as f:
        # The header must not contain "INPUT"/"OUTPUT"/"DFF" or an '=': DEMOTIC's bench parser
        # classifies lines by SUBSTRING without skipping comments, so a chatty header makes it
        # treat this line as a flip-flop and crash in its own split('=').
        f.write(f"# gpap abc_backend from {name}: {aig.n_inputs} in / {aig.n_outputs} out / "
                f"{aig.n_latches} flop / {len(aig.ands)} and\n")
        f.write("\n".join(lines_io) + "\n" + "\n".join(lines_gate) + "\n")


def simplify_bench_abc(src: str, dst: str, *, abc: str | None = None,
                       script: str = "dc2; dc2; fraig") -> dict:
    """ABC-optimize a .bench circuit and write an equivalent primitive-gate .bench.

    Sequential-preserving (`strash; zero; <script>` — latches survive, reset 0), so the result is
    still something a tool that does its own unrolling can consume. Returns a stats dict with the
    AND counts before/after. Correctness is the caller's to gate — `bench_equiv_check` below
    simulates both circuits on identical stimuli and compares the pinned-output acceptance."""
    exe = find_abc(abc)
    if exe is None:
        raise _AbcError("no `abc` binary found (build it with scripts/build_abc.sh)")
    src, dst = os.path.abspath(src), os.path.abspath(dst)
    tmp = tempfile.NamedTemporaryFile(suffix=".aig", delete=False).name
    try:
        pre = _run_abc(exe, f"read_bench {src}; strash; print_stats")
        out = _run_abc(exe, f"read_bench {src}; strash; zero; {script}; "
                            f"print_stats; write_aiger -s {tmp}")
        if not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
            raise _AbcError(f"abc produced no AIG for {src}: {out.strip()[-300:]}")
        aig = read_aiger(tmp)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    write_bench_from_aig(aig, dst, name=os.path.basename(src))

    def _ands(txt):
        import re
        m = re.search(r"and =\s*(\d+)", txt)
        return int(m.group(1)) if m else None
    return {"ands_before": _ands(pre), "ands_after": len(aig.ands),
            "n_inputs": aig.n_inputs, "n_outputs": aig.n_outputs, "n_latches": aig.n_latches,
            "script": script}


def bench_equiv_check(src: str, dst: str, cycles: int, seed: int, *, pin_outputs=None,
                      target_mode: str = "random", target=None, n_words: int = 512) -> dict:
    """Verify that `dst` (an ABC-simplified copy of `src`) computes the SAME OUTPUTS as `src`
    under identical stimuli — and, in particular, accepts exactly the same input sequences for
    the pinned target.

    Both circuits have the same primary inputs in the same order, so the bit-parallel simulator
    driven from the same seed feeds them IDENTICAL stimuli; every primary output must then match
    bit for bit, which is far stronger than comparing acceptance rates.

    ⚠️ EVERY OUTPUT is compared, not only the pinned-target acceptance word, and that is the whole
    point: at a `rare` target the acceptance probability is below 1/(n_words*64) by construction,
    so both acceptance words are ALL ZERO and comparing them alone returns ok=True having tested
    nothing. Every ABC row of the recorded ISCAS-85 rival campaign passed the check that way
    (hits [0, 0, 32768]). Output equality is target-independent and cannot degenerate.

    Returns {ok, outputs_ok, accept_ok, outputs, mismatched, hits_src, hits_dst, n, target}: `ok`
    is the gate (both halves), and the components are recorded so a row can prove which half had
    content. Raises nothing: the caller decides what a mismatch means."""
    import numpy as np
    from gpap.circuits.simulate import (_accept_words, _pin_and_target, _popcount, _run_bench,
                                        _topo_gates)
    from gpap.circuits.bench import parse_bench

    c_src, c_dst = parse_bench(src), parse_bench(dst)
    pin, pin_target, T = _pin_and_target("bench", c_src, cycles, seed, pin_outputs, target_mode,
                                         target)
    # positional correspondence: ABC preserves primary-output ORDER, and write_bench_from_aig
    # emits OUTPUT() in that order — so output j of dst is output j of src.
    if len(c_dst[1]) != len(c_src[1]):          # positional correspondence is the premise
        return {"ok": False, "outputs_ok": False, "accept_ok": False,
                "outputs": len(c_src[1]), "mismatched": ["<output count differs: "
                                                         f"{len(c_src[1])} -> {len(c_dst[1])}>"],
                "hits_src": 0, "hits_dst": 0, "n": n_words * 64, "target": {}}
    idx = {o: j for j, o in enumerate(c_src[1])}
    dst_target = {c_dst[1][idx[o]]: v for o, v in pin_target.items()}
    out_a = _run_bench(c_src, _topo_gates(c_src), T, n_words, rng=np.random.default_rng(seed))
    out_b = _run_bench(c_dst, _topo_gates(c_dst), T, n_words, rng=np.random.default_rng(seed))
    mism = [o for o in c_src[1] if not np.array_equal(out_a[o], out_b[c_dst[1][idx[o]]])]
    a, b = _accept_words(out_a, pin_target), _accept_words(out_b, dst_target)
    accept_ok = bool(np.array_equal(a, b))
    return {"ok": bool(not mism and accept_ok), "outputs_ok": not mism, "accept_ok": accept_ok,
            "outputs": len(c_src[1]), "mismatched": mism[:8],
            "hits_src": _popcount(a), "hits_dst": _popcount(b),
            "n": n_words * 64, "target": dst_target}


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
    # Like the native encoder, only the PINNED prefix of the outputs is read from `target`: a
    # caller pinning `pin_outputs` may hand a partial dict holding just those (the rival
    # harnesses' target tables do), and indexing every output here raised a KeyError on it.
    pinned = list(outputs) if pin_outputs is None else list(outputs)[:pin_outputs]
    values = [target[o] for o in pinned] + [False] * (len(outputs) - len(pinned))
    b, support, labels = _framed_to_cnf(faig, cycles, list(inputs), list(outputs),
                                        values, pin_outputs)
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
