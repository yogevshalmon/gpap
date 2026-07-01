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

Usage:
  python scripts/bench_to_cnf.py --bench s386.bench --cycles 25 --out s386_c25.cnf --seed 1
"""
import argparse
import random
import re
import sys

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

    def tseitin(self, op, o, ins):
        """o <-> op(ins). Standard CNF (handles AND/OR/NAND/NOR/NOT/BUFF, multi-input)."""
        if op in ("NOT", "BUFF", "BUF"):
            a = ins[0]
            same = (op != "NOT")
            # o <-> (a or ~a):  (¬o ∨ s·a)(o ∨ ¬s·a) with s = +1 buff / -1 not
            self.add(-o, a if same else -a)
            self.add(o, -a if same else a)
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


def build_cnf(circ, T, seed):
    inputs, outputs, dffs, gates = circ
    rng = random.Random(seed)
    in_seq = [{inp: (rng.random() < 0.5) for inp in inputs} for _ in range(T)]
    target = simulate(circ, in_seq, T)

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
    for o in outputs:
        v = var_of(o, T - 1)
        b.add(v if target[o] else -v)

    support = [input_vars[(inp, t)] for inp in inputs for t in range(T)]
    return b, support, target


def write_dimacs(path, builder, support):
    with open(path, "w") as f:
        f.write(f"c unrolled CircuitSAT sampling instance (support = primary inputs x cycles)\n")
        # `c ind ... 0` lines (UniGen/QuickSampler convention), chunked
        for i in range(0, len(support), 10):
            f.write("c ind " + " ".join(map(str, support[i:i + 10])) + " 0\n")
        f.write(f"p cnf {builder.nv} {len(builder.clauses)}\n")
        for c in builder.clauses:
            f.write(" ".join(map(str, c)) + " 0\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--cycles", type=int, default=25)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    circ = parse_bench(args.bench)
    inputs, outputs, dffs, gates = circ
    b, support, target = build_cnf(circ, args.cycles, args.seed)
    write_dimacs(args.out, b, support)
    print(f"{args.bench}: PI={len(inputs)} PO={len(outputs)} DFF={len(dffs)} gates={len(gates)} "
          f"| cycles={args.cycles} -> vars={b.nv} clauses={len(b.clauses)} support={len(support)}")


if __name__ == "__main__":
    main()
