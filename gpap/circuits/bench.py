#!/usr/bin/env python3
"""Unroll a sequential ISCAS-89 `.bench` circuit over T clock cycles and emit a projected-sampling
DIMACS CNF, reproducing the CircuitSAT-sampling setup of Ardakani et al., "Recurrent CircuitSAT
Sampling for Sequential Circuits" (arXiv:2502.21226).

  unroll   : T copies of the combinational logic; a flip-flop's output at cycle t is its D-input's
             value at t-1 (variable SHARING, no clause), at t=0 the reset value 0. Primary inputs
             get a FRESH variable per cycle — these are the sampling support.
  Tseitin  : each gate instance -> a fresh CNF var + the standard gate clauses.
  constrain: simulate one random input sequence and pin the last-cycle primary OUTPUTS to the
             simulated (hence reachable, hence SAT) values — the CRV setup.
  emit     : DIMACS with a `c ind` header listing the primary-input variables across all cycles.

The CLI drives this module (`gpap circuit2cnf`), and also handles AIGER and the ABC encoder.
"""
import random
import re

_GATE_RE = re.compile(r"^\s*([\w.\[\]]+)\s*=\s*([A-Za-z]+)\s*\(([^)]*)\)")
_IO_RE = re.compile(r"^\s*(INPUT|OUTPUT)\s*\(\s*([\w.\[\]]+)\s*\)")


def parse_bench(path):
    """Return (inputs[list], outputs[list], dffs{out:din}, gates{out:(op,[ins])})."""
    inputs, outputs, dffs, gates = [], [], {}, {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            m = _IO_RE.match(line)
            if m:
                (inputs if m.group(1) == "INPUT" else outputs).append(m.group(2))
                continue
            m = _GATE_RE.match(line)
            if m:
                out, op, args = m.group(1), m.group(2).upper(), m.group(3)
                ins = [a.strip() for a in args.split(",") if a.strip()]
                if op == "DFF":
                    dffs[out] = ins[0]
                else:
                    gates[out] = (op, ins)
    return inputs, outputs, dffs, gates


def _eval_gate(op, vals):
    if op == "AND":
        return all(vals)
    if op == "NAND":
        return not all(vals)
    if op == "OR":
        return any(vals)
    if op == "NOR":
        return not any(vals)
    if op == "NOT":
        return not vals[0]
    if op in ("BUFF", "BUF"):
        return vals[0]
    if op in ("XOR", "XNOR"):
        r = False
        for v in vals:
            r ^= bool(v)
        return r if op == "XOR" else not r
    raise ValueError(f"unknown gate {op}")


def simulate(circ, in_seq, T):
    """Forward-simulate (reset state 0). in_seq[t][input]=bool. Returns outputs at the LAST cycle."""
    inputs, outputs, dffs, gates = circ
    state = {d: False for d in dffs}
    last_out = {}
    for t in range(T):
        vals = dict(state)
        for inp in inputs:
            vals[inp] = in_seq[t][inp]

        memo = {}

        def ev(sig):
            if sig in vals:
                return vals[sig]
            if sig in memo:
                return memo[sig]
            op, ins = gates[sig]
            r = _eval_gate(op, [ev(i) for i in ins])
            memo[sig] = r
            return r

        last_out = {o: ev(o) for o in outputs}
        state = {d: ev(din) for d, din in dffs.items()}
    return last_out


class CnfBuilder:
    def __init__(self):
        self.nv = 0
        self.clauses = []

    def newvar(self):
        self.nv += 1
        return self.nv

    def add(self, *lits):
        self.clauses.append(list(lits))

    def _xor2(self, y, a, b):
        self.add(-y, -a, -b); self.add(-y, a, b); self.add(y, -a, b); self.add(y, a, -b)

    def tseitin(self, op, o, ins):
        """o <-> op(ins). Standard CNF (handles AND/OR/NAND/NOR/NOT/BUFF/XOR/XNOR, multi-input)."""
        if op in ("NOT", "BUFF", "BUF"):
            a = ins[0]
            same = (op != "NOT")
            # o <-> (a or ~a):  (¬o ∨ s·a)(o ∨ ¬s·a) with s = +1 buff / -1 not
            self.add(-o, a if same else -a)
            self.add(o, -a if same else a)
            return
        if op in ("XOR", "XNOR"):
            # chain 2-input XORs into acc = odd-parity(ins); o <-> acc (XOR) / o <-> ¬acc (XNOR)
            acc = ins[0]
            for b in ins[1:]:
                y = self.newvar()
                self._xor2(y, acc, b)
                acc = y
            if op == "XOR":
                self.add(-o, acc); self.add(o, -acc)
            else:
                self.add(-o, -acc); self.add(o, acc)
            return
        pos = op in ("AND", "NAND")
        neg_out = op in ("NAND", "NOR")
        # y <-> AND(ins) for AND/NAND; for OR/NOR use De Morgan via y <-> OR
        if pos:
            y = o if not neg_out else self.newvar()
            for a in ins:
                self.add(-y, a)
            self.add(y, *[-a for a in ins])
        else:
            y = o if not neg_out else self.newvar()
            for a in ins:
                self.add(y, -a)
            self.add(-y, *ins)
        if neg_out:
            self.add(-o, -y); self.add(o, y)


def random_target(circ, T, seed):
    """The default pinned target: simulate ONE random input sequence (reset state 0) and take the
    last-cycle outputs. It draws from the circuit's own output DISTRIBUTION, so it is size-biased
    toward rejection-easy patterns; `simulate.rare_target` is the adversarial alternative."""
    inputs = circ[0]
    rng = random.Random(seed)
    in_seq = [{inp: (rng.random() < 0.5) for inp in inputs} for _ in range(T)]
    return simulate(circ, in_seq, T)


def build_cnf(circ, T, seed, pin_outputs=None, target=None):
    """Unroll T cycles -> Tseitin CNF, sampling the primary inputs (the support). `pin_outputs` is
    how many final-cycle outputs to constrain to the simulated target: None = all, right for a
    sequential circuit; for a COMBINATIONAL one that over-constrains to a tiny solution set, so
    pass a small count. `target` overrides the pinned pattern ({output: bool}) and MUST be
    simulation-reachable or the CNF is UNSAT (default: random_target(seed))."""
    if T < 1:
        raise ValueError(f"cycles must be >= 1, got {T}")
    inputs, outputs, dffs, gates = circ
    if target is None:
        target = random_target(circ, T, seed)

    b = CnfBuilder()
    false_var = b.newvar()
    b.add(-false_var)
    input_vars = {}
    for inp in inputs:
        for t in range(T):
            input_vars[(inp, t)] = b.newvar()
    gate_memo = {}

    def var_of(sig, t):
        if (sig, t) in input_vars:
            return input_vars[(sig, t)]
        if (sig, t) in gate_memo:
            return gate_memo[(sig, t)]
        if sig in dffs:
            v = false_var if t == 0 else var_of(dffs[sig], t - 1)
            gate_memo[(sig, t)] = v
            return v
        op, ins = gates[sig]
        o = b.newvar()
        gate_memo[(sig, t)] = o
        b.tseitin(op, o, [var_of(i, t) for i in ins])
        return o

    # Realise the FULL unrolled circuit (every gate at every cycle), matching the paper's
    # Tseitin-of-the-whole-unrolling rather than just the final-output cone.
    for t in range(T):
        for g in gates:
            var_of(g, t)
    pin = outputs if pin_outputs is None else outputs[:pin_outputs]
    for o in pin:
        v = var_of(o, T - 1)
        b.add(v if target[o] else -v)

    support = [input_vars[(inp, t)] for inp in inputs for t in range(T)]
    support_labels = [f"{inp}@{t}" for inp in inputs for t in range(T)]
    return b, support, target, support_labels


def write_dimacs(path, builder, support, support_labels=None, encoder=None):
    """Write the projected-sampling CNF. The `c ind` header lists the support variables; with
    `support_labels` a `c wire <var> <input@cycle>` map is embedded too, so sample output can be
    named in circuit-wire terms. `encoder` records provenance as `c encoder <name>`."""
    with open(path, "w") as f:
        f.write("c unrolled CircuitSAT sampling instance (support = primary inputs x cycles)\n")
        if encoder:
            f.write(f"c encoder {encoder}\n")
        for i in range(0, len(support), 10):
            f.write("c ind " + " ".join(map(str, support[i:i + 10])) + " 0\n")
        if support_labels:
            for v, lbl in zip(support, support_labels):
                f.write(f"c wire {v} {lbl}\n")
        f.write(f"p cnf {builder.nv} {len(builder.clauses)}\n")
        for c in builder.clauses:
            f.write(" ".join(map(str, c)) + " 0\n")


def parse_support_labels(path):
    """Read the `c wire <var> <label>` map write_dimacs embeds: {var: label}, or empty."""
    out = {}
    with open(path) as f:
        for line in f:
            if line.startswith("c wire "):
                parts = line.split()
                if len(parts) >= 4:
                    out[int(parts[2])] = parts[3]
            elif line and line[0] not in "c":
                break
    return out


def parse_encoder_comment(path):
    with open(path) as f:
        for line in f:
            if line.startswith("c encoder "):
                return line.split()[2]
            if line and line[0] not in "c":
                break
    return None


def convert_common(kind, circ, cycles, seed, pin_outputs, encoder, abc_path, target_mode,
                   out_path, native_build, abc_build):
    """THE conversion driver both circuit formats share (`convert_one`, `aig2cnf.convert_aig_one`):
    resolve the target mode and the encoder, build, write, so that dispatch has one home.
    `native_build(target)` / `abc_build(target)` return (builder, support, target, labels); this
    returns (builder, support, rare_est, resolved_encoder)."""
    target = rare_est = None
    if target_mode == "rare":
        from gpap.circuits.simulate import rare_target
        target, rare_est = rare_target(kind, circ, cycles, seed, pin_outputs)
    elif target_mode != "random":
        raise ValueError(f"unknown target_mode {target_mode!r} (random | rare)")
    if encoder == "auto":
        from gpap.circuits.abc_backend import abc_available
        encoder = "abc" if abc_available(abc_path) else "tseitin"
    if encoder == "abc":
        b, support, target, labels = abc_build(target)
    elif encoder == "tseitin":
        b, support, target, labels = native_build(target)
    else:
        raise ValueError(f"unknown encoder {encoder!r} (tseitin | abc | auto)")
    write_dimacs(out_path, b, support, labels, encoder=encoder)
    return b, support, rare_est, encoder


def convert_one(bench_path, out_path, cycles, seed, pin_outputs=None, encoder="tseitin",
                abc_path=None, quiet=False, abc_opt="auto", target_mode="random"):
    """Convert a single .bench file to a DIMACS CNF at out_path; returns the CnfBuilder.

    `encoder` is "tseitin" (default, the native gate-by-gate Tseitin), "abc" (unroll +
    logic-optimize with berkeley-abc, then Tseitin its AIG — abc_backend.py; raises without an
    `abc` binary, `abc_opt` picks the preset), or "auto". Both encoders describe the SAME projected
    solution space over the primary inputs at one seed; only internal gate variables differ, and
    the default stays "tseitin" so results never depend on whether abc is installed.

    `target_mode` is "random" (default: pin the outputs of one seed-random simulated sequence) or
    "rare" (the RAREST pattern in a large simulation batch — still reachable, but adversarial for
    rejection sampling)."""
    circ = parse_bench(bench_path)
    inputs, outputs, dffs, gates = circ

    def native_build(target):
        return build_cnf(circ, cycles, seed, pin_outputs=pin_outputs, target=target)

    def abc_build(target):
        from gpap.circuits.abc_backend import build_cnf_abc
        return build_cnf_abc(circ, bench_path, cycles, seed, pin_outputs=pin_outputs,
                             abc=abc_path, opt=abc_opt, target=target)

    b, support, rare_est, enc = convert_common("bench", circ, cycles, seed, pin_outputs,
                                               encoder, abc_path, target_mode, out_path,
                                               native_build, abc_build)
    if not quiet:
        rare_s = f" target=rare(freq~{rare_est:.2e})" if rare_est is not None else ""
        print(f"{bench_path} -> {out_path}: PI={len(inputs)} PO={len(outputs)} DFF={len(dffs)} "
              f"gates={len(gates)} | cycles={cycles} enc={enc}{rare_s} -> vars={b.nv} "
              f"clauses={len(b.clauses)} support={len(support)}")
    return b


