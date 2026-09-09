#!/usr/bin/env python3
"""Emit GPAP's circuit-sampling task as a SINGLE-OUTPUT combinational .bench miter, so
circuit-level rivals (DEMOTIC / Recurrent-CircuitSAT) can run on OUR instances untouched.

The circuit is the T-cycle unrolling (reset state 0, fresh inputs per cycle, named `<wire>_c<t>`
to match gpap's `wire@cycle` support order) plus MITER = AND over pinned outputs of (output at
cycle T-1 == target bit): solutions of MITER=1 are EXACTLY gpap's projected solution set for that
bench/T/seed/pin/target, verified here by bit-parallel simulation. module_name comes from the
FILENAME, so a single-output 'c17' gets DEMOTIC's target=ones."""
import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # eval/<sub>/ -> repo
sys.path.insert(0, REPO_ROOT)

from gpap.circuits.bench import parse_bench                                   # noqa: E402
from gpap.circuits.simulate import (_pin_and_target, _run_bench, _topo_gates,  # noqa: E402
                                    _popcount, sim_baseline)


# Max fan-in of any AND in a miter tree: 64 keeps the Tseitin clause inside the packed 8|24
# envelope's 255-literal cap with room for the pin/unit clauses.
_MITER_FANIN = 64


def and_tree(lines, terms, prefix, fanin=_MITER_FANIN):
    """Reduce a wide AND to a tree of bounded fan-in, appending the intermediate gates to `lines`.
    One AND over k inputs Tseitin-encodes to a (k+1)-literal clause and the envelope caps clause
    length at 255, so a many-output miter would be refused outright. Shared by both miter
    builders; both call sites re-verify p_miter == p_accept."""
    idx = 0
    while len(terms) > fanin:
        nxt = []
        for i in range(0, len(terms), fanin):
            chunk = terms[i:i + fanin]
            if len(chunk) == 1:
                nxt.append(chunk[0])
                continue
            nm = f"{prefix}{idx}"
            idx += 1
            lines.append(f"{nm} = AND({', '.join(chunk)})")
            nxt.append(nm)
        terms = nxt
    return terms


def write_unrolled_miter(bench_path, out_path, T, seed, pin_outputs=None,
                         target_mode="random"):
    """Unroll + miter-ify to a combinational single-output .bench; returns (n_inputs, p_accept)."""
    circ = parse_bench(bench_path)
    inputs, outputs, dffs, gates = circ
    pin, pin_target, T = _pin_and_target("bench", circ, T, seed, pin_outputs, target_mode)
    order = _topo_gates(circ)

    lines = []
    name_at = {}                                   # (sig, t) -> emitted signal name
    for t in range(T):
        for inp in inputs:
            name_at[(inp, t)] = f"{inp}_c{t}"
            lines.append(f"INPUT({inp}_c{t})")
    # constant 0 for the t=0 register state: x AND NOT x over the first input
    i0 = f"{inputs[0]}_c0"
    lines.append(f"CONST0N = NOT({i0})")
    lines.append(f"CONST0 = AND({i0}, CONST0N)")

    def ref(sig, t):
        if (sig, t) in name_at:
            return name_at[(sig, t)]
        if sig in dffs:                            # register: alias of its D-input at t-1
            nm = "CONST0" if t == 0 else ref(dffs[sig], t - 1)
            name_at[(sig, t)] = nm
            return nm
        raise KeyError(f"unresolved {sig}@{t}")

    for t in range(T):
        for sig in order:                          # topo order: inputs of each gate exist
            op, ins = gates[sig]
            name_at[(sig, t)] = f"{sig}_c{t}"
            lines.append(f"{sig}_c{t} = {op}({', '.join(ref(i, t) for i in ins)})")
    terms = []
    for o in pin:
        nm = ref(o, T - 1)
        if pin_target[o]:
            terms.append(nm)
        else:
            lines.append(f"MITN_{o} = NOT({nm})")
            terms.append(f"MITN_{o}")
    # WIDE AND -> a bounded fan-in tree, a clause past 255 literals being refused by the envelope.
    terms = and_tree(lines, terms, "MITAND")
    lines.append(f"MITER = {'AND(' + ', '.join(terms) + ')' if len(terms) > 1 else f'BUFF({terms[0]})'}")
    # A second (unconstrained) output so DEMOTIC's generated forward returns a TUPLE: with one
    # output their `outputs_list[-1]` slices the BATCH dim, so the shadow comes FIRST.
    lines.append("MITER_SHADOW = NOT(MITER)")
    header = [f"# unrolled miter of {os.path.basename(bench_path)} T={T} seed={seed} "
              f"target={target_mode} pins={len(pin)} (gpap eval/rivals/unroll_to_bench.py)",
              "OUTPUT(MITER_SHADOW)", "OUTPUT(MITER)"]
    with open(out_path, "w") as f:
        f.write("\n".join(header + lines) + "\n")

    # SELF-CHECK: the miter's P[MITER=1] must equal the original instance's p_accept.
    import numpy as np
    mcirc = parse_bench(out_path)
    morder = _topo_gates(mcirc)
    nw = 2048
    outs = _run_bench(mcirc, morder, 1, nw, rng=np.random.default_rng(3))
    p_miter = _popcount(outs["MITER"]) / (nw * 64)
    ref_r = sim_baseline(bench_path, T, seed, pin_outputs=pin_outputs, target_mode=target_mode,
                         n_min=nw * 64, budget_s=0.5, hard_cap_s=5.0)
    p_ref = ref_r["p_accept"]
    tol = 4 * max((p_ref * (1 - p_ref) / (nw * 64)) ** 0.5, 1e-5)
    if abs(p_miter - p_ref) > tol and not (p_ref == 0 and p_miter == 0):
        raise AssertionError(f"miter p={p_miter:.3e} != original p_accept={p_ref:.3e}")
    return len(inputs) * T, p_miter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("out")
    ap.add_argument("--cycles", type=int, default=25)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--pin-outputs", type=int, default=None)
    ap.add_argument("--target", choices=["random", "rare"], default="random")
    args = ap.parse_args()
    n_in, p = write_unrolled_miter(args.bench, args.out, args.cycles, args.seed,
                                   pin_outputs=args.pin_outputs, target_mode=args.target)
    print(f"{args.bench} -> {args.out}: {n_in} miter inputs, verified p(MITER=1)={p:.3e}")


if __name__ == "__main__":
    main()
