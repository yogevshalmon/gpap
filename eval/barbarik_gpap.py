#!/usr/bin/env python3
"""Run Barbarik (Chakraborty & Meel, AAAI'19 — meelgroup/barbarik) against GPAP as the sampler
under test. Barbarik is THE uniformity certifier for CNF samplers: it draws one sample from the
sampler and one from a trusted uniform reference (SPUR), builds a "kernel" CNF whose solutions
mix the two with a known ratio, has the SAMPLER UNDER TEST sample the kernel, and checks the
observed bias — accepting an eps-close-to-uniform sampler / rejecting an eta-far one w.p. >= 1-delta
with O(1/(eta-eps)^4) samples. This complements the cheap in-pipeline readout
(gpap.sampling.metrics.uniformity_stats), which can only CERTIFY on exactly-enumerable spaces.

GPAP is hooked IN-PROCESS (their SolutionRetriever monkeypatched): one torch init for the whole
test instead of a subprocess per call; the SPUR reference keeps their stock ./samplers/spur path.

Setup (once):
  git clone https://github.com/meelgroup/barbarik ~/.cache/gpap/src/barbarik
  cd ~/.cache/gpap/src/barbarik && git submodule update --init

Usage:
  PYTHONPATH=. .venv/bin/python eval/barbarik_gpap.py <cnf> [<cnf> ...] \
      [--eta 0.9] [--epsilon 0.1] [--delta 0.05] [--seed 1] [--max N] \
      [--device cpu] [--B 8] [--barbarik PATH]
Verdict per instance: "Accept:1 ..." (uniform-consistent at the chosen eta/eps/delta) or
"RejectIteration:..." (non-uniformity detected). No verdict when --max truncates the test.
"""
import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)


def _gpap_solutions(input_file, num_solutions, ind_var_list, seed, device, config):
    """Sample `num_solutions` from GPAP on `input_file`, projected on ind_var_list (1-based).
    Returns solutions in barbarik's format: one list of signed ints per solution, consistently
    ordered — biasFind keys solutions by their literal string, and compares to the SPUR
    reference by SET, so any fixed order is valid."""
    import torch
    from pysat.formula import CNF
    from gpap.sampling.sampler import GPUSampler

    cnf = CNF(from_file=input_file)
    sup0 = [v - 1 for v in ind_var_list]
    smp = GPUSampler.from_clauses(cnf.nv, cnf.clauses, torch.device(device), support=sup0,
                                  seed=seed, k_cap=4096, use_kernel=(device == "cuda" and "auto"))
    bits, cols = smp.sample_bits(num_solutions, config=config, max_chunks=200)
    if bits is None or len(bits) < num_solutions:
        raise RuntimeError(f"GPAP produced {0 if bits is None else len(bits)}/{num_solutions} "
                           f"samples on {input_file}")
    out = []
    for row in bits[:num_solutions]:
        out.append([v if row[i] else -v for i, v in enumerate(cols)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cnfs", nargs="+", help="CNF instances (must carry `c ind` sampling sets)")
    ap.add_argument("--barbarik", default=os.path.expanduser("~/.cache/gpap/src/barbarik"),
                    help="path to a meelgroup/barbarik checkout (with submodules)")
    ap.add_argument("--eta", type=float, default=0.9, help="farness to reject (paper default 0.9)")
    ap.add_argument("--epsilon", type=float, default=0.1, help="closeness to accept (default 0.1)")
    ap.add_argument("--delta", type=float, default=0.05, help="confidence (default 0.05)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max", type=int, default=sys.maxsize,
                    help="cap on total samples (capping ends the test WITHOUT a verdict)")
    ap.add_argument("--device", default="cpu",
                    help="gpap device for the kernel-CNF sampling (default cpu: the kernel "
                         "formulas are tiny and per-call CUDA overhead dominates on GPU)")
    ap.add_argument("--B", type=int, default=8, help="gpap decisions/round under test (default 8)")
    args = ap.parse_args()

    if not os.path.isdir(os.path.join(args.barbarik, "interfaces")):
        sys.exit(f"no barbarik checkout at {args.barbarik} (see module docstring)")
    sys.path.insert(0, args.barbarik)
    os.chdir(args.barbarik)               # their code invokes ./samplers/spur and `rm *.nnf`
    import interfaces.cnf as C

    # GPAP hook: dispatch SAMPLER_CUSTOM to the in-process sampler, everything else (the SPUR
    # uniform reference) to the stock retriever.
    config = (True, args.B, False)
    stock = C.SolutionRetriever.getSolutionFromSampler

    def dispatch(inputFile, numSolutions, samplerType, indVarList, verbosity, newSeed):
        if samplerType == C.SAMPLER_CUSTOM:
            sols = _gpap_solutions(inputFile, numSolutions, indVarList,
                                   int(newSeed) + args.seed, args.device, config)
            if verbosity:
                print(f"GPAP returned {len(sols)} solutions")
            return sols
        return stock(inputFile, numSolutions, samplerType, indVarList, verbosity, newSeed)

    C.SolutionRetriever.getSolutionFromSampler = staticmethod(dispatch)
    _stock_name = C.get_sampler_string
    C.get_sampler_string = lambda t: "GPAP" if t == C.SAMPLER_CUSTOM else _stock_name(t)

    for cnf in args.cnfs:
        cnf = os.path.abspath(os.path.join(REPO_ROOT, cnf)) if not os.path.isabs(cnf) else cnf
        print(f"\n=== barbarik CM_test: GPAP (B={args.B}, {args.device}) on {cnf} "
              f"(eta={args.eta} eps={args.epsilon} delta={args.delta}) ===", flush=True)
        t = C.cnf_test(C.SAMPLER_CUSTOM, cnf, args.eta, args.epsilon, args.delta,
                       args.max, verbosity=0)
        t.CM_test(args.seed)


if __name__ == "__main__":
    main()
