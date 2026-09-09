"""Native .aig/.aag input: unroll a (possibly sequential) AIGER circuit over T cycles and
Tseitin-encode it into a projected-sampling CNF. The AIGER twin of `bench.py`, sharing its
conventions so the two input formats are interchangeable downstream:

  * latch state at t=0 is its reset value (0 by default, explicit 0/1 honoured, `x` rejected);
    at t>0 it is the next-state literal's value at t-1, by variable sharing and no clause;
  * primary inputs get a fresh variable per cycle, and these are the sampling support;
  * outputs at the last cycle are pinned to a simulated (hence reachable, hence SAT) target;
  * support labels are `<input>@<cycle>`.

This is the no-dependency path. With berkeley-abc installed, `encoder="abc"` routes the same file
through `dc2`/`fraig` first (abc_backend) for a much smaller CNF over the same projected space.

Everything walks the AIG with explicit worklists, never recursion: ASCII files may list AND
definitions in any order, and unrolled dependency chains run far past any Python recursion limit.
"""
import random

from gpap.circuits.aiger import Aig, read_aiger


def input_labels(aig: Aig) -> list[str]:
    return [aig.input_names.get(k, f"i{k}") for k in range(aig.n_inputs)]


def simulate_aig(aig: Aig, in_seq: list[list[bool]], T: int) -> list[bool]:
    """Forward-simulate T cycles from the reset state; returns the LAST cycle's output values.
    `in_seq[t][k]` = value of input k at cycle t."""
    and_def = aig.and_defs()
    state = {cur >> 1: bool(r) for cur, _nxt, r in aig.latches}
    out = []
    for t in range(T):
        values = {0: False}                        # var 0 = constant FALSE
        for k, lit in enumerate(aig.inputs):
            values[lit >> 1] = in_seq[t][k]
        values.update(state)

        def ev(lit: int) -> bool:
            v = lit >> 1
            if v not in values:
                stack = [v]
                while stack:
                    x = stack[-1]
                    if x in values:
                        stack.pop()
                        continue
                    r0, r1 = and_def[x]
                    need = [u >> 1 for u in (r0, r1) if (u >> 1) not in values]
                    if need:
                        stack.extend(need)
                        continue
                    values[x] = ((values[r0 >> 1] ^ bool(r0 & 1))
                                 and (values[r1 >> 1] ^ bool(r1 & 1)))
                    stack.pop()
            return values[v] ^ bool(lit & 1)

        out = [ev(o) for o in aig.outputs]
        state = {cur >> 1: ev(nxt) for cur, nxt, _r in aig.latches}
    return out


def random_target_aig(aig: Aig, T: int, seed: int) -> list[bool]:
    """Default pinned target: last-cycle outputs of ONE random input sequence (size-biased toward
    high-probability patterns — see gpap/circuits/simulate.py's rare_target). T must already be
    combinational-clamped by the caller."""
    rng = random.Random(seed)
    in_seq = [[rng.random() < 0.5 for _ in range(aig.n_inputs)] for _ in range(T)]
    return simulate_aig(aig, in_seq, T)


def build_cnf_aig(aig: Aig, T: int, seed: int, pin_outputs: int | None = None, target=None):
    """Unroll T cycles -> Tseitin CNF over the AIG structure (3 clauses per AND instance).
    Returns (builder, support, target, support_labels) exactly like `bench.build_cnf`; `target`
    is the list of simulated last-cycle output values (all outputs, pinned or not) — pass one
    (e.g. simulate.rare_target) to override the default random_target_aig pattern."""
    from gpap.circuits.bench import CnfBuilder

    if T < 1:
        raise ValueError(f"cycles must be >= 1, got {T}")
    if aig.n_latches == 0:
        T = 1                                      # combinational: unrolling = independent copies
    names = input_labels(aig)
    if target is None:
        target = random_target_aig(aig, T, seed)

    b = CnfBuilder()
    false_var = b.newvar()
    b.add(-false_var)                              # constant-FALSE anchor (reset values, const lits)
    input_vars = {}                                # (input k, t) -> CNF var  (the support)
    for k in range(aig.n_inputs):
        for t in range(T):
            input_vars[(k, t)] = b.newvar()
    in_var_of = {lit >> 1: k for k, lit in enumerate(aig.inputs)}
    latch_next = {cur >> 1: (nxt, r) for cur, nxt, r in aig.latches}
    and_def = aig.and_defs()
    memo = {}                                      # (var, t) -> CNF var/literal of the POSITIVE var

    def _resolved(v: int, t: int) -> bool:
        return v == 0 or v in in_var_of or (v, t) in memo

    def _value(lit: int, t: int) -> int:
        """CNF literal for AIG literal `lit` at frame t (parts must already be resolved)."""
        v = lit >> 1
        if v == 0:
            base = false_var
        elif v in in_var_of:
            base = input_vars[(in_var_of[v], t)]
        else:
            base = memo[(v, t)]
        return -base if (lit & 1) else base

    def resolve(v0: int, t0: int) -> None:
        if _resolved(v0, t0):
            return
        stack = [(v0, t0)]
        while stack:
            v, t = stack[-1]
            if _resolved(v, t):
                stack.pop()
                continue
            if v in latch_next:
                nxt, r = latch_next[v]
                if t == 0:                         # reset value as a constant literal
                    memo[(v, 0)] = -false_var if r else false_var
                    stack.pop()
                    continue
                if not _resolved(nxt >> 1, t - 1):
                    stack.append((nxt >> 1, t - 1))
                    continue
                memo[(v, t)] = _value(nxt, t - 1)  # share the D-input's literal at t-1
                stack.pop()
                continue
            r0, r1 = and_def[v]                    # AND gate
            deps = [(u >> 1, t) for u in (r0, r1) if not _resolved(u >> 1, t)]
            if deps:
                stack.extend(deps)
                continue
            y = b.newvar()                         # y <-> a AND c : 3 clauses
            a, c = _value(r0, t), _value(r1, t)
            b.add(-y, a)
            b.add(-y, c)
            b.add(y, -a, -c)
            memo[(v, t)] = y
            stack.pop()

    # Realise the FULL unrolled circuit (every AND + next-state cone at every cycle), matching
    # bench.build_cnf's whole-unrolling Tseitin (not just the pinned-output cone).
    for t in range(T):
        for lhs, _r0, _r1 in aig.ands:
            resolve(lhs >> 1, t)
        for _cur, nxt, _r in aig.latches:
            resolve(nxt >> 1, t)
    pin = range(aig.n_outputs) if pin_outputs is None else range(min(pin_outputs, aig.n_outputs))
    for j in pin:
        resolve(aig.outputs[j] >> 1, T - 1)
        ol = _value(aig.outputs[j], T - 1)
        b.add(ol if target[j] else -ol)

    support = [input_vars[(k, t)] for k in range(aig.n_inputs) for t in range(T)]
    support_labels = [f"{names[k]}@{t}" for k in range(aig.n_inputs) for t in range(T)]
    return b, support, target, support_labels


def convert_aig_one(aig_path, out_path, cycles, seed, pin_outputs=None, encoder="tseitin",
                    abc_path=None, quiet=False, abc_opt="auto", target_mode="random"):
    """Convert a single .aig/.aag file to a projected-sampling DIMACS CNF. Mirrors
    `bench.convert_one` (same encoder choices, same target_mode random|rare, same
    `c ind`/`c wire`/provenance output) — both are faces of `bench.convert_common`."""
    from gpap.circuits.bench import convert_common

    aig = read_aiger(aig_path)

    def native_build(target):
        return build_cnf_aig(aig, cycles, seed, pin_outputs=pin_outputs, target=target)

    def abc_build(target):
        from gpap.circuits.abc_backend import build_cnf_abc_aig
        return build_cnf_abc_aig(aig, aig_path, cycles, seed, pin_outputs=pin_outputs,
                                 abc=abc_path, opt=abc_opt, target=target)

    b, support, rare_est, enc = convert_common("aig", aig, cycles, seed, pin_outputs,
                                               encoder, abc_path, target_mode, out_path,
                                               native_build, abc_build)
    if not quiet:
        T = 1 if aig.n_latches == 0 else cycles
        rare_s = f" target=rare(freq~{rare_est:.2e})" if rare_est is not None else ""
        print(f"{aig_path} -> {out_path}: PI={aig.n_inputs} PO={aig.n_outputs} "
              f"L={aig.n_latches} ands={len(aig.ands)} | cycles={T} enc={enc}{rare_s} -> "
              f"vars={b.nv} clauses={len(b.clauses)} support={len(support)}")
    return b
