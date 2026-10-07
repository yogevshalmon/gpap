"""The ABC circuit encoder must not change WHAT WE SAMPLE.

Load-bearing claim: `--encoder abc` (unroll + dc2/fraig via berkeley-abc, then Tseitin its AIG)
yields a CNF whose set of SUPPORT PROJECTIONS — the primary-input assignments consistent with the
pinned outputs — is EXACTLY the native Tseitin encoder's. Only internal gate variables differ.

These are CPU tests. They skip when no `abc` binary is present (it is an optional dependency;
build it with scripts/build_abc.sh).
"""
import os

import pytest
from pysat.formula import CNF
from pysat.solvers import Solver

from gpap.circuits.abc_backend import abc_available, build_cnf_abc
from gpap.circuits.bench import build_cnf, parse_bench, simulate
from tests.conftest import COMB_DIR, SEQ_DIR

pytestmark = pytest.mark.skipif(not abc_available(), reason="no `abc` binary (scripts/build_abc.sh)")

BENCH = SEQ_DIR
# Small enough that the projected solution set can be enumerated exhaustively.
CASES = [("b01", 3), ("b01", 5), ("b02", 4), ("b06", 3), ("b09", 4), ("b03", 2)]


def _enum_projections(nv, clauses, support, limit=200_000):
    """All distinct assignments to `support` (1-based vars) that extend to a model."""
    s = Solver(name="cadical195", bootstrap_with=CNF(from_clauses=clauses))
    out = set()
    while s.solve() and len(out) < limit:
        m = set(s.get_model())
        proj = tuple(1 if v in m else 0 for v in support)
        out.add(proj)
        s.add_clause([(-v if v in m else v) for v in support])   # block this projection
    s.delete()
    return out


@pytest.mark.parametrize("opt", ["full", "fast"])
@pytest.mark.parametrize("name,cycles", CASES)
def test_abc_preserves_projected_solution_set(name, cycles, opt):
    path = os.path.join(BENCH, f"{name}.bench")
    circ = parse_bench(path)
    seed = 3
    bt, sup_t, tgt_t, lab_t = build_cnf(circ, cycles, seed, pin_outputs=None)
    ba, sup_a, tgt_a, lab_a = build_cnf_abc(circ, path, cycles, seed, pin_outputs=None, opt=opt)

    # Same support ORDER and the same pinned target => the two CNFs must describe the same set.
    assert lab_t == lab_a
    assert tgt_t == tgt_a
    assert len(sup_t) == len(sup_a)

    proj_t = _enum_projections(bt.nv, bt.clauses, sup_t)
    proj_a = _enum_projections(ba.nv, ba.clauses, sup_a)
    assert proj_t, "native encoder produced an UNSAT/empty instance — bad test case"
    assert proj_a == proj_t, (f"{name} T={cycles}: ABC changed the projected solution set "
                              f"(native {len(proj_t)} vs abc {len(proj_a)} projections)")


@pytest.mark.parametrize("name,cycles", [("b06", 3), ("b09", 4)])
def test_abc_pins_a_prefix_from_a_partial_target_like_the_native_encoder(name, cycles):
    """`pin_outputs=k` reads `target` only for the first k outputs, on BOTH encoders: the rival
    harnesses hand a partial dict holding just the pinned outputs (their target tables), and the
    ABC route used to index every output and KeyError on it — so the abc:fast rival arm could
    not run at all in table mode. Same target, same pins => the same projected solution set."""
    path = os.path.join(BENCH, f"{name}.bench")
    circ = parse_bench(path)
    outputs = list(circ[1])
    k = max(1, len(outputs) // 2)
    full = build_cnf(circ, cycles, 3, pin_outputs=None)[2]
    partial = {o: full[o] for o in outputs[:k]}          # nothing for the unpinned outputs
    bt, sup_t, _t, _l = build_cnf(circ, cycles, 3, pin_outputs=k, target=partial)
    ba, sup_a, _t2, _l2 = build_cnf_abc(circ, path, cycles, 3, pin_outputs=k, opt="fast",
                                        target=partial)
    proj_t = _enum_projections(bt.nv, bt.clauses, sup_t)
    proj_a = _enum_projections(ba.nv, ba.clauses, sup_a)
    assert proj_t and proj_a == proj_t
    # And it genuinely pins fewer outputs than pinning all of them would.
    ball = build_cnf_abc(circ, path, cycles, 3, pin_outputs=None, opt="fast", target=full)[0]
    assert len(_enum_projections(ball.nv, ball.clauses, sup_a)) <= len(proj_a)


@pytest.mark.parametrize("name,cycles", [("b01", 4), ("b10", 3), ("b03", 3)])
def test_abc_encoder_is_smaller_and_short_clauses(name, cycles):
    """ABC's AIG encoding should not be larger, and must emit only 2- and 3-literal clauses
    (plus unit pins) — that uniformity is what the sum-trick BCP / packed kernel want."""
    path = os.path.join(BENCH, f"{name}.bench")
    circ = parse_bench(path)
    bt, _s, _t, _l = build_cnf(circ, cycles, 3, pin_outputs=None)
    ba, _s2, _t2, _l2 = build_cnf_abc(circ, path, cycles, 3, pin_outputs=None)
    assert len(ba.clauses) <= len(bt.clauses)
    assert max(len(c) for c in ba.clauses) <= 3


def test_abc_unroll_matches_our_simulator():
    """ABC's `zero; frames -F T -i` must reproduce our reset-state-0 unroll exactly: pin the
    outputs to OUR simulated target and the ABC CNF must still be satisfiable."""
    path = os.path.join(BENCH, "b10.bench")
    circ = parse_bench(path)
    inputs, outputs, _d, _g = circ
    T, seed = 6, 3
    ba, sup, target, _lab = build_cnf_abc(circ, path, T, seed, pin_outputs=None)
    s = Solver(name="cadical195", bootstrap_with=CNF(from_clauses=ba.clauses))
    assert s.solve(), "ABC CNF UNSAT under a reachable simulated target — unroll semantics differ"
    m = set(s.get_model())
    s.delete()
    # The model's primary inputs, replayed through OUR simulator, must reproduce the pinned target.
    bits = {v: (v in m) for v in sup}
    in_seq = [{inp: bits[sup[i * T + t]] for i, inp in enumerate(inputs)} for t in range(T)]
    assert simulate(circ, in_seq, T) == target


# --------------------------------------------------------------------------------------------
# Circuit-level simplification (.bench -> ABC -> .bench) — the RIVAL-fairness path.
# The GPU rivals unroll .bench themselves, so they must be offered the same ABC-optimized circuit
# GPAP's abc arm gets. That swap is only legitimate if the circuit is unchanged as a function.

@pytest.mark.parametrize("name", ["b03", "b06", "b10", "s386"])
@pytest.mark.parametrize("target_mode", ["random", "rare"])
def test_simplify_bench_abc_is_equivalent(tmp_path, name, target_mode):
    """The simplified copy must accept EXACTLY the same input sequences: identical stimuli in,
    bit-identical acceptance words out (not merely the same acceptance rate)."""
    from gpap.circuits.abc_backend import bench_equiv_check, simplify_bench_abc
    src = os.path.join(BENCH, f"{name}.bench")
    dst = str(tmp_path / f"{name}_abc.bench")
    st = simplify_bench_abc(src, dst)
    assert st["ands_after"] <= st["ands_before"]
    chk = bench_equiv_check(src, dst, 25, 5, target_mode=target_mode)
    assert chk["ok"], f"{name}: ABC copy accepts a different set ({chk})"
    assert chk["hits_src"] == chk["hits_dst"]
    assert chk["outputs_ok"] and chk["outputs"] > 0, "the output-equality half must have content"


def test_bench_equiv_check_catches_a_sabotaged_copy_at_a_rare_pin(tmp_path):
    """The gate must not be able to pass by comparing nothing.

    At a `rare` target the acceptance probability is below 1/(n_words*64), so BOTH acceptance
    words are all-zero and comparing only them returns ok=True whatever the circuit does — every
    ABC row of the recorded ISCAS-85 rival campaign passed that way (hits [0, 0, 32768]). The
    check therefore compares every primary output; this pins that a functional change is caught
    even in the regime where the acceptance half is vacuous."""
    from gpap.circuits.abc_backend import bench_equiv_check
    src = os.path.join(COMB_DIR, "c432.bench")
    lines, done, out = open(src).read().splitlines(), False, []
    for line in lines:                                  # flip exactly one gate's function
        if not done and "NAND(" in line:
            line, done = line.replace("NAND(", "AND(", 1), True
        out.append(line)
    assert done, "no NAND gate to sabotage"
    dst = str(tmp_path / "c432_sabotaged.bench")
    open(dst, "w").write("\n".join(out) + "\n")
    chk = bench_equiv_check(src, dst, 1, 5, target_mode="rare")
    assert chk["accept_ok"], "precondition: at a rare pin the acceptance halves agree (0 vs 0)"
    assert chk["hits_src"] == 0 and chk["hits_dst"] == 0
    assert not chk["ok"] and not chk["outputs_ok"], "a sabotaged circuit must NOT pass the gate"
    assert chk["mismatched"], "the failing outputs must be named in the row"


def test_simplify_bench_abc_output_is_parseable_and_primitive(tmp_path):
    """The emitted .bench must use only primitives every consumer here understands (ours and both
    circuit-level rivals' parsers): AND / NOT / BUF / DFF — never abc's LUT/DFFRSE."""
    import re
    from gpap.circuits.abc_backend import simplify_bench_abc
    src = os.path.join(BENCH, "b03.bench")
    dst = str(tmp_path / "b03_abc.bench")
    simplify_bench_abc(src, dst)
    circ = parse_bench(dst)
    ops = {op for op, _ins in circ[3].values()}
    assert ops <= {"AND", "NOT", "BUF"}, ops
    assert circ[2], "latches vanished — the sequential structure must survive"
    text = open(dst).read()
    assert "LUT" not in text and "DFFRSE" not in text
    # every signal name must be a valid Python identifier: DEMOTIC exec's generated source
    for tok in re.findall(r"^\s*([^=\s#]+)\s*=", text, re.M):
        assert re.fullmatch(r"[A-Za-z_]\w*", tok), tok
    # ...and the header must not contain the substrings their line classifier keys on
    assert not any(k in text.splitlines()[0] for k in ("INPUT", "OUTPUT", "DFF", "="))


def test_abc_paths_survive_a_space(tmp_path):
    """abc splits its script on whitespace, so an unquoted path with a space in it became two
    arguments: the command wrote nothing and the caller reported "abc produced no framed AIG".
    It binds on the USER's circuit path, and — since the intermediates live in `<repo>/.gpap/tmp`
    — on a checkout whose own directory has a space in it, which is not the user's doing at all."""
    import shutil

    from gpap.circuits.abc_backend import unroll_optimize_to_aig

    d = tmp_path / "a dir with spaces"
    d.mkdir()
    src = str(d / "b01 copy.bench")
    shutil.copy(os.path.join(BENCH, "b01.bench"), src)
    out = str(d / "framed out.aig")
    unroll_optimize_to_aig(src, 2, out, opt="fast")      # spaced INPUT and spaced OUTPUT
    # The OTHER abc-script site: the `--circuit-variants abc` route that writes a rival's
    # primitive-gate .bench. Reverting the quoting there passed every other test.
    from gpap.circuits.abc_backend import simplify_bench_abc
    simplify_bench_abc(src, str(d / "simplified out.bench"))
    assert os.path.getsize(str(d / "simplified out.bench")) > 0


@pytest.mark.parametrize("ch", ['"', "'", "\n", "\r", "\t"])
def test_abc_paths_it_cannot_carry_are_refused_by_name(tmp_path, ch):
    """Quoting survives a space, a `;`, a glob, a backtick, `&` and UTF-8 — but abc rewrites tab,
    newline and carriage return inside a quoted argument into a space, ends the quote at a `"`,
    and strips a `'`, swallowing the rest of the script into the filename. Every one of them wrote
    nothing and surfaced as "abc produced no framed AIG", the misleading message quoting was added
    to remove, so they are refused up front with a message that names the character."""
    import shutil

    from gpap.circuits.abc_backend import _AbcError, unroll_optimize_to_aig

    src = str(tmp_path / f"b01{ch}x.bench")
    shutil.copy(os.path.join(BENCH, "b01.bench"), src)
    with pytest.raises(_AbcError, match="cannot be given a path containing"):
        unroll_optimize_to_aig(src, 2, str(tmp_path / "out.aig"), opt="fast")
