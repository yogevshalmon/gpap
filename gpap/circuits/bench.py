#!/usr/bin/env python3
"""Unroll a sequential ISCAS-89 `.bench` circuit over T clock cycles and emit a projected-sampling
DIMACS CNF — reproducing the CircuitSAT-sampling setup of Ardakani et al., "Recurrent CircuitSAT
Sampling for Sequential Circuits" (arXiv:2502.21226), so GPAP can be compared vs CMSGen on the same
instances.

Pipeline:
  parse  : INPUT()/OUTPUT()/`g = DFF(d)`/`g = AND|OR|NAND|NOR|NOT|BUFF(...)` (sequential .bench).
  unroll : T copies of the combinational logic; a flip-flop's output at cycle t is its D-input's
           value at cycle t-1 (variable SHARING, no clause); at t=0 the state is the reset value 0.
           Primary inputs get a FRESH variable per cycle — these are the sampling support.
  Tseitin: each gate instance -> a fresh CNF var + the standard gate clauses.
  constrain: simulate one random input sequence (reset state 0) and pin the primary OUTPUTS at the
           last cycle to the simulated (hence reachable -> SAT) values. This is the CRV setup: sample
           input stimuli consistent with an output constraint.
  emit   : DIMACS with a `c ind` header listing the primary-input variables across all cycles
           (the independent support GPAP/CMSGen sample over).

Usage (the CLI drives this module; it also handles AIGER and the ABC encoder):
  # single instance
  gpap circuit2cnf s386.bench s386-c25.cnf --cycles 25 --seed 1
  # whole folder of circuits -> folder of .cnf files (named <stem>-c<cycles>.cnf)
  gpap circuit2cnf benches/ cnf_out/ --cycles 25 --seed 1
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
    if op in ("XOR", "XNOR"):                 # ISCAS-85 uses XOR/XNOR
        r = False
        for v in vals:
            r ^= bool(v)
        return r if op == "XOR" else not r
    raise ValueError(f"unknown gate {op}")


def simulate(circ, in_seq, T):
    """Forward-simulate (reset state 0). in_seq[t][input]=bool. Returns outputs at the LAST cycle."""
    inputs, outputs, dffs, gates = circ
    state = {d: False for d in dffs}                      # t=0 reset state
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
        state = {d: ev(din) for d, din in dffs.items()}   # next-state = D-input now
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
        """y <-> a ⊕ b (4 clauses)."""
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
        pos = op in ("AND", "NAND")          # AND-like (conjunction) vs OR-like
        neg_out = op in ("NAND", "NOR")      # output inverted
        # y <-> AND(ins) for AND/NAND (y is the conjunction); for OR/NOR use De Morgan via y<->OR
        if pos:                               # y = AND(ins); o = y (AND) or ¬y (NAND)
            y = o if not neg_out else self.newvar()
            for a in ins:
                self.add(-y, a)               # y -> a
            self.add(y, *[-a for a in ins])   # (a∧b∧..) -> y
        else:                                 # y = OR(ins); o = y (OR) or ¬y (NOR)
            y = o if not neg_out else self.newvar()
            for a in ins:
                self.add(y, -a)               # a -> y
            self.add(-y, *ins)                # y -> (a∨b∨..)
        if neg_out:
            self.add(-o, -y); self.add(o, y)  # o <-> ¬y


def random_target(circ, T, seed):
    """The default pinned target: simulate ONE random input sequence (reset state 0) and take the
    last-cycle outputs. NOTE this draws the target from the circuit's own output DISTRIBUTION, so
    it is size-biased toward high-probability (rejection-easy) patterns — see
    gpap/circuits/simulate.py's rare_target for the adversarial alternative."""
    inputs = circ[0]
    rng = random.Random(seed)
    in_seq = [{inp: (rng.random() < 0.5) for inp in inputs} for _ in range(T)]
    return simulate(circ, in_seq, T)


def build_cnf(circ, T, seed, pin_outputs=None, target=None):
    """Unroll T cycles → Tseitin CNF; sample the primary inputs (support). `pin_outputs` = how many
    final-cycle primary outputs to constrain to the simulated target: None = all (sequential circuits,
    where the whole output pattern is the spec). For COMBINATIONAL circuits (ISCAS-85, many outputs vs
    inputs) pinning all outputs over-constrains to a tiny solution set — pass a small count (e.g. 1) so
    the CircuitSAT problem keeps a large, samplable solution space (the DEMOTIC-style single-output
    setup). `target` overrides the pinned pattern ({output: bool}, e.g. simulate.rare_target);
    it MUST be a simulation-reachable pattern or the CNF is UNSAT — default = random_target(seed)."""
    if T < 1:
        raise ValueError(f"cycles must be >= 1, got {T}")
    inputs, outputs, dffs, gates = circ
    if target is None:
        target = random_target(circ, T, seed)

    b = CnfBuilder()
    false_var = b.newvar()
    b.add(-false_var)                                     # a constant-0 literal for t=0 DFF state
    input_vars = {}                                       # (input, t) -> var  (the support)
    for inp in inputs:
        for t in range(T):
            input_vars[(inp, t)] = b.newvar()
    gate_memo = {}                                        # (sig, t) -> var  (memoized Tseitin)

    def var_of(sig, t):
        if (sig, t) in input_vars:                        # primary input (the support)
            return input_vars[(sig, t)]
        if (sig, t) in gate_memo:
            return gate_memo[(sig, t)]
        if sig in dffs:                                   # flip-flop output
            v = false_var if t == 0 else var_of(dffs[sig], t - 1)   # share with D-input at t-1
            gate_memo[(sig, t)] = v
            return v
        op, ins = gates[sig]                              # combinational gate -> Tseitin
        o = b.newvar()
        gate_memo[(sig, t)] = o
        b.tseitin(op, o, [var_of(i, t) for i in ins])
        return o

    # Realise the FULL unrolled circuit (every gate at every cycle) so the CNF matches the paper's
    # Tseitin-of-the-whole-unrolling, not just the final-output cone.
    for t in range(T):
        for g in gates:
            var_of(g, t)
    # Pin the primary OUTPUTS at the last cycle to the simulated (reachable => SAT) target.
    pin = outputs if pin_outputs is None else outputs[:pin_outputs]
    for o in pin:
        v = var_of(o, T - 1)
        b.add(v if target[o] else -v)

    support = [input_vars[(inp, t)] for inp in inputs for t in range(T)]
    # A human label per support variable: the primary-input wire and the cycle it belongs to.
    support_labels = [f"{inp}@{t}" for inp in inputs for t in range(T)]
    return b, support, target, support_labels


def write_dimacs(path, builder, support, support_labels=None, encoder=None):
    """Write the projected-sampling CNF. The `c ind` header lists the support variables; when
    `support_labels` is given (from a .bench conversion) a `c wire <var> <input@cycle>` map is
    embedded too, so sample output can be reported in circuit-wire terms and the map travels WITH
    the CNF (readable back via parse_support_labels). `encoder` records provenance as a
    `c encoder <name>` line — the sampling pipeline reads it to skip work the encoder already did
    (ABC's dc2/fraig removes the backbone the entailed-clause miner would otherwise re-probe)."""
    with open(path, "w") as f:
        f.write("c unrolled CircuitSAT sampling instance (support = primary inputs x cycles)\n")
        if encoder:
            f.write(f"c encoder {encoder}\n")
        # `c ind ... 0` lines (UniGen/QuickSampler convention), chunked
        for i in range(0, len(support), 10):
            f.write("c ind " + " ".join(map(str, support[i:i + 10])) + " 0\n")
        if support_labels:                                 # wire map: var -> input@cycle
            for v, lbl in zip(support, support_labels):
                f.write(f"c wire {v} {lbl}\n")
        f.write(f"p cnf {builder.nv} {len(builder.clauses)}\n")
        for c in builder.clauses:
            f.write(" ".join(map(str, c)) + " 0\n")


def parse_support_labels(path):
    """Read the `c wire <var> <label>` map embedded by write_dimacs. Returns {var(int): label} or
    an empty dict for a CNF without a wire map (e.g. a non-circuit instance)."""
    out = {}
    with open(path) as f:
        for line in f:
            if line.startswith("c wire "):
                parts = line.split()
                if len(parts) >= 4:
                    out[int(parts[2])] = parts[3]
            elif line and line[0] not in "c":
                break                                      # past the header
    return out


def parse_encoder_comment(path):
    """Read the `c encoder <name>` provenance line embedded by write_dimacs (None if absent)."""
    with open(path) as f:
        for line in f:
            if line.startswith("c encoder "):
                return line.split()[2]
            if line and line[0] not in "c":
                break                                      # past the header
    return None


def convert_common(kind, circ, cycles, seed, pin_outputs, encoder, abc_path, target_mode,
                   out_path, native_build, abc_build):
    """THE conversion driver both circuit formats share (`convert_one` for .bench,
    `aig2cnf.convert_aig_one` for AIGER): resolve the target mode, resolve the encoder, build,
    write — so the target/encoder dispatch rule has one home rather than a twin per format.
    `native_build(target)` / `abc_build(target)` return (builder, support, target, labels).
    Returns (builder, support, rare_est, resolved_encoder)."""
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
    """Convert a single .bench file to a DIMACS CNF at out_path. Returns the CnfBuilder.

    `encoder`:
      "tseitin" (default) — the native gate-by-gate Tseitin of the unrolled circuit.
      "abc"               — unroll + logic-optimize with berkeley-abc, then Tseitin its AIG
                            (2-8x fewer clauses; see gpap/circuits/abc_backend.py). Raises if no
                            `abc` binary is found. `abc_opt` picks the optimization preset
                            (full | fast | auto — see abc_backend.ABC_OPT_PRESETS).
      "auto"              — "abc" when a binary is available, else "tseitin".
    The two encoders describe the SAME projected solution space over the primary inputs (same seed
    => same simulated output target); only the internal gate variables differ. The default stays
    "tseitin" so results never depend silently on whether abc happens to be installed.

    `target_mode`: "random" (default — pin the outputs of one seed-random simulated sequence;
    size-biased toward high-probability patterns) | "rare" (pin the RAREST pattern observed in a
    large bit-parallel simulation batch — still reachable => SAT, but adversarial for rejection
    sampling; see gpap/circuits/simulate.py)."""
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


