"""The ABC encoder (`--encoder abc`) must not change what we sample: its CNF has exactly the
native Tseitin encoder's set of support projections; only internal gate variables differ.
CPU tests; they skip without an `abc` binary (scripts/build_abc.sh).
"""
import os

import pytest
from pysat.formula import CNF
from pysat.solvers import Solver

from gpap.circuits.abc_backend import abc_available, build_cnf_abc
from gpap.circuits.bench import build_cnf, parse_bench, simulate
from tests.conftest import COMB_DIR, SEQ_DIR
from tests.support import enumerate_projections

pytestmark = pytest.mark.skipif(not abc_available(), reason="no `abc` binary (scripts/build_abc.sh)")

BENCH = SEQ_DIR
# Small enough that the projected solution set can be enumerated exhaustively.
CASES = [("b01", 3), ("b01", 5), ("b02", 4), ("b06", 3), ("b09", 4), ("b03", 2)]


@pytest.mark.parametrize("opt", ["full", "fast"])
@pytest.mark.parametrize("name,cycles", CASES)
def test_abc_preserves_projected_solution_set(name, cycles, opt):
    path = os.path.join(BENCH, f"{name}.bench")
    circ = parse_bench(path)
    seed = 3
    bt, sup_t, tgt_t, lab_t = build_cnf(circ, cycles, seed, pin_outputs=None)
    ba, sup_a, tgt_a, lab_a = build_cnf_abc(circ, path, cycles, seed, pin_outputs=None, opt=opt)

    # Same support order and pinned target, so the two CNFs must describe the same set.
    assert lab_t == lab_a
    assert tgt_t == tgt_a
    assert len(sup_t) == len(sup_a)

    proj_t = enumerate_projections(bt.clauses, sup_t, limit=200_000)
    proj_a = enumerate_projections(ba.clauses, sup_a, limit=200_000)
    assert proj_t, "native encoder produced an UNSAT/empty instance — bad test case"
    assert proj_a == proj_t, (f"{name} T={cycles}: ABC changed the projected solution set "
                              f"(native {len(proj_t)} vs abc {len(proj_a)} projections)")


@pytest.mark.parametrize("name,cycles", [("b06", 3), ("b09", 4)])
def test_abc_pins_a_prefix_from_a_partial_target_like_the_native_encoder(name, cycles):
    """With `pin_outputs=k` both encoders read `target` only for the first k outputs, since the
    rival harnesses pass a partial dict holding just the pinned outputs."""
    path = os.path.join(BENCH, f"{name}.bench")
    circ = parse_bench(path)
    outputs = list(circ[1])
    k = max(1, len(outputs) // 2)
    full = build_cnf(circ, cycles, 3, pin_outputs=None)[2]
    partial = {o: full[o] for o in outputs[:k]}          # nothing for the unpinned outputs
    bt, sup_t, _t, _l = build_cnf(circ, cycles, 3, pin_outputs=k, target=partial)
    ba, sup_a, _t2, _l2 = build_cnf_abc(circ, path, cycles, 3, pin_outputs=k, opt="fast",
                                        target=partial)
    proj_t = enumerate_projections(bt.clauses, sup_t, limit=200_000)
    proj_a = enumerate_projections(ba.clauses, sup_a, limit=200_000)
    assert proj_t and proj_a == proj_t
    # Pinning every output gives at most as many projections.
    ball = build_cnf_abc(circ, path, cycles, 3, pin_outputs=None, opt="fast", target=full)[0]
    assert len(enumerate_projections(ball.clauses, sup_a, limit=200_000)) <= len(proj_a)


@pytest.mark.parametrize("name,cycles", [("b01", 4), ("b10", 3), ("b03", 3)])
def test_abc_encoder_is_smaller_and_short_clauses(name, cycles):
    """ABC's AIG encoding is no larger than the native one and has no clause over 3 literals."""
    path = os.path.join(BENCH, f"{name}.bench")
    circ = parse_bench(path)
    bt, _s, _t, _l = build_cnf(circ, cycles, 3, pin_outputs=None)
    ba, _s2, _t2, _l2 = build_cnf_abc(circ, path, cycles, 3, pin_outputs=None)
    assert len(ba.clauses) <= len(bt.clauses)
    assert max(len(c) for c in ba.clauses) <= 3


def test_abc_unroll_matches_our_simulator():
    """ABC's `zero; frames -F T -i` matches our reset-0 unroll: pinned to our simulated target,
    the ABC CNF is satisfiable and its model replays to that target."""
    path = os.path.join(BENCH, "b10.bench")
    circ = parse_bench(path)
    inputs, outputs, _d, _g = circ
    T, seed = 6, 3
    ba, sup, target, _lab = build_cnf_abc(circ, path, T, seed, pin_outputs=None)
    s = Solver(name="cadical195", bootstrap_with=CNF(from_clauses=ba.clauses))
    assert s.solve(), "ABC CNF UNSAT under a reachable simulated target — unroll semantics differ"
    m = set(s.get_model())
    s.delete()
    # Replay the model's primary inputs through our simulator.
    bits = {v: (v in m) for v in sup}
    in_seq = [{inp: bits[sup[i * T + t]] for i, inp in enumerate(inputs)} for t in range(T)]
    assert simulate(circ, in_seq, T) == target


# --------------------------------------------------------------------------------------------
# Circuit-level simplification (.bench -> ABC -> .bench) gives the rivals, which unroll .bench
# themselves, the same ABC-optimized circuit. Valid only if the circuit's function is unchanged.

@pytest.mark.parametrize("name", ["b03", "b06", "b10", "s386"])
@pytest.mark.parametrize("target_mode", ["random", "rare"])
def test_simplify_bench_abc_is_equivalent(tmp_path, name, target_mode):
    """The simplified copy accepts exactly the same input sequences: identical stimuli give
    bit-identical acceptance words, not just the same rate."""
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
    """A one-gate change is caught even at a rare pin. There both acceptance words are all zero,
    so comparing only them would pass any circuit; the check also compares every output."""
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
    """The emitted .bench uses only AND / NOT / BUF / DFF, which our parser and both
    circuit-level rivals' parsers accept, never abc's LUT/DFFRSE."""
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
    # every signal name is a Python identifier (DEMOTIC execs generated source)
    for tok in re.findall(r"^\s*([^=\s#]+)\s*=", text, re.M):
        assert re.fullmatch(r"[A-Za-z_]\w*", tok), tok
    # the header line holds none of the keywords their line classifier looks for
    assert not any(k in text.splitlines()[0] for k in ("INPUT", "OUTPUT", "DFF", "="))


def test_abc_paths_survive_a_space(tmp_path):
    """abc splits its script on whitespace, so every path handed to it is quoted. This covers
    the user's circuit path and the scratch intermediates of a checkout under a spaced directory."""
    import shutil

    from gpap.circuits.abc_backend import unroll_optimize_to_aig

    d = tmp_path / "a dir with spaces"
    d.mkdir()
    src = str(d / "b01 copy.bench")
    shutil.copy(os.path.join(BENCH, "b01.bench"), src)
    out = str(d / "framed out.aig")
    unroll_optimize_to_aig(src, 2, out, opt="fast")      # spaced INPUT and spaced OUTPUT
    # The other abc-script site (`--circuit-variants abc`); no other test covers its quoting.
    from gpap.circuits.abc_backend import simplify_bench_abc
    simplify_bench_abc(src, str(d / "simplified out.bench"))
    assert os.path.getsize(str(d / "simplified out.bench")) > 0


@pytest.mark.parametrize("ch", ['"', "'", "\n", "\r", "\t"])
def test_abc_paths_it_cannot_carry_are_refused_by_name(tmp_path, ch):
    """abc cannot carry a tab, newline, carriage return, `"` or `'` inside a quoted path, so such
    paths are refused up front with a message naming the character, instead of failing later
    as "abc produced no framed AIG"."""
    import shutil

    from gpap.circuits.abc_backend import _AbcError, unroll_optimize_to_aig

    src = str(tmp_path / f"b01{ch}x.bench")
    shutil.copy(os.path.join(BENCH, "b01.bench"), src)
    with pytest.raises(_AbcError, match="cannot be given a path containing"):
        unroll_optimize_to_aig(src, 2, str(tmp_path / "out.aig"), opt="fast")
