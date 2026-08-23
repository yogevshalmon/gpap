#!/usr/bin/env python3
"""Run Barbarik (Chakraborty & Meel, AAAI'19 — meelgroup/barbarik) against GPAP or CMSGen as the
sampler under test. Barbarik is THE uniformity certifier for CNF samplers: it draws one sample
from the sampler and one from a trusted uniform reference (SPUR), builds a "kernel" CNF whose
solutions mix the two with a known ratio, has the SAMPLER UNDER TEST sample the kernel, and checks
the observed bias — accepting an eps-close-to-uniform sampler / rejecting an eta-far one w.p.
>= 1-delta with O(1/(eta-eps)^4) samples.

The tester is `CM_test` (the AAAI'19 uniform test). The repo also carries Barbarik3's `PM_test`
(NeurIPS'22, Pote & Meel), but that targets arbitrary WEIGHTED distributions and needs a DUAL
oracle — sampling plus probability queries — on the target. For uniform sampling CM_test is the
right and cheaper instrument.

DEFAULT PARAMETERS ARE THE CMSGen PAPER'S (eps 0.3, eta 1.8, delta 0.1 — "Designing Samplers is
Easy", FMCAD'21, Table I), which is what makes a like-for-like comparison possible and costs
~1,836 samples per instance instead of the ~70,000 that eps 0.1 / eta 0.9 demands. Note what that
buys: at eta 1.8 the test only rejects samplers that are 1.8-far in L1, i.e. TV > 0.9. Skew well
inside that is invisible here BY DESIGN — for a sensitive test on enumerable spaces use
eval/uniformity/fuzz_uniformity.py, which measures total variation against the exact solution set.

GPAP is hooked IN-PROCESS (their SolutionRetriever monkeypatched): one CUDA init for the whole
test instead of a subprocess per call; the SPUR reference keeps their stock ./samplers/spur path.
CMSGen runs as a subprocess, which is how it is normally invoked.

Setup (once) — pinned at the commit the recorded certification runs used:
  git clone https://github.com/meelgroup/barbarik <repo>/.gpap/src/barbarik
  cd <repo>/.gpap/src/barbarik \
      && git checkout 2c0a6691b97c796a829408dc4db70d9f59f49382 \
      && git submodule update --init

Usage:
  PYTHONPATH=. .venv/bin/python eval/uniformity/barbarik_gpap.py <cnf> [<cnf> ...] \
      [--sampler gpap|cmsgen] [--config b8|b1|soft|bsup] [--cmsgen PATH] \
      [--eta 1.8] [--epsilon 0.3] [--delta 0.1] [--seed 1] [--max N] [--barbarik PATH]
Verdict per instance: "Accept:1 ..." (uniform-consistent at the chosen eta/eps/delta) or
"RejectIteration:..." (non-uniformity detected). No verdict when --max truncates the test.
"""
import argparse
import os
import subprocess
import sys
import tempfile

from gpap import paths

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # eval/<sub>/ -> repo
sys.path.insert(0, REPO_ROOT)

# (random_var, B, lookahead), as gpap.sampling.pipeline defines a config. "bsup" is special-cased
# below: B is set to the size of the sampling set, which makes a chunk exact rejection sampling
# over the projection (see eval/uniformity/fuzz_uniformity.py for the measurement that motivates it).
CONFIGS = {"b8": (True, 8, False), "b1": (True, 1, False), "soft": (True, 8, "soft"),
           "bsup": (True, None, False)}


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
    rv, B, la = config
    if B is None:                              # "bsup": one decision round over the whole support
        B = max(1, len(ind_var_list))
    smp = GPUSampler.from_clauses(cnf.nv, cnf.clauses, torch.device(device), support=sup0,
                                  seed=seed, k_cap=4096)
    bits, cols = smp.sample_bits(num_solutions, config=(rv, B, la), max_chunks=400)
    if bits is None or len(bits) < num_solutions:
        raise RuntimeError(f"GPAP produced {0 if bits is None else len(bits)}/{num_solutions} "
                           f"samples on {input_file}")
    return [[v if row[i] else -v for i, v in enumerate(cols)] for row in bits[:num_solutions]]


def _cmsgen_solutions(input_file, num_solutions, ind_var_list, seed, cmsgen_bin):
    """Same contract, CMSGen as the sampler under test. CMSGen emits FULL assignments (it ignores
    `c ind`), so the projection onto ind_var_list happens here."""
    out = os.path.join(tempfile.gettempdir(), f"cmsgen_barbarik_{os.getpid()}.out")
    p = subprocess.run([cmsgen_bin, "--samples", str(num_solutions), "--seed", str(int(seed)),
                        "--verb", "0", "--samplefile", out, input_file],
                       capture_output=True, text=True, timeout=1800)
    if p.returncode not in (0, 10):            # 10 = SATISFIABLE
        raise RuntimeError(f"cmsgen failed on {input_file}: {p.stdout[-300:]}{p.stderr[-300:]}")
    sols = []
    with open(out) as f:
        for line in f:
            lits = {int(x) for x in line.split() if x.lstrip("-").isdigit()}
            if lits:
                sols.append([v if v in lits else -v for v in ind_var_list])
    os.unlink(out)
    if len(sols) < num_solutions:
        raise RuntimeError(f"cmsgen produced {len(sols)}/{num_solutions} on {input_file}")
    return sols[:num_solutions]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cnfs", nargs="+", help="CNF instances (must carry `c ind` sampling sets)")
    ap.add_argument("--sampler", default="gpap", choices=["gpap", "cmsgen"],
                    help="which sampler is under test (the SPUR reference is unchanged)")
    ap.add_argument("--cmsgen", default=str(paths.tool("cmsgen")),
                    help="cmsgen binary, for --sampler cmsgen")
    ap.add_argument("--barbarik", default=str(paths.src_dir() / "barbarik"),
                    help="path to a meelgroup/barbarik checkout (with submodules)")
    ap.add_argument("--eta", type=float, default=1.8,
                    help="farness to reject, in L1 (CMSGen paper default 1.8 == TV 0.9)")
    ap.add_argument("--epsilon", type=float, default=0.3, help="closeness to accept (default 0.3)")
    ap.add_argument("--delta", type=float, default=0.1, help="confidence (default 0.1)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max", type=int, default=sys.maxsize,
                    help="cap on total samples (capping ends the test WITHOUT a verdict)")
    ap.add_argument("--device", default="cuda",
                    help="gpap device. The engine is CUDA-only since the packed-kernel refactor; "
                         "there is no CPU path any more")
    ap.add_argument("--config", default="b8", choices=list(CONFIGS),
                    help="gpap sampling config under test (bsup = B set to |sampling set|)")
    args = ap.parse_args()

    if not os.path.isdir(os.path.join(args.barbarik, "interfaces")):
        sys.exit(f"no barbarik checkout at {args.barbarik} (see module docstring)")
    sys.path.insert(0, args.barbarik)
    os.chdir(args.barbarik)               # their code invokes ./samplers/spur and `rm *.nnf`
    import interfaces.cnf as C

    # Hook: dispatch SAMPLER_CUSTOM to the sampler under test, everything else (the SPUR uniform
    # reference) to the stock retriever.
    config = CONFIGS[args.config]
    stock = C.SolutionRetriever.getSolutionFromSampler
    label = f"gpap:{args.config}" if args.sampler == "gpap" else "cmsgen"

    def dispatch(inputFile, numSolutions, samplerType, indVarList, verbosity, newSeed):
        if samplerType != C.SAMPLER_CUSTOM:
            return stock(inputFile, numSolutions, samplerType, indVarList, verbosity, newSeed)
        seed = int(newSeed) + args.seed
        if args.sampler == "gpap":
            sols = _gpap_solutions(inputFile, numSolutions, indVarList, seed, args.device, config)
        else:
            sols = _cmsgen_solutions(inputFile, numSolutions, indVarList, seed, args.cmsgen)
        if verbosity:
            print(f"{label} returned {len(sols)} solutions")
        return sols

    C.SolutionRetriever.getSolutionFromSampler = staticmethod(dispatch)
    _stock_name = C.get_sampler_string
    C.get_sampler_string = lambda t: label if t == C.SAMPLER_CUSTOM else _stock_name(t)

    for cnf in args.cnfs:
        cnf = cnf if os.path.isabs(cnf) else os.path.abspath(os.path.join(REPO_ROOT, cnf))
        print(f"\n=== barbarik CM_test: {label} on {cnf} "
              f"(eta={args.eta} eps={args.epsilon} delta={args.delta}) ===", flush=True)
        t = C.cnf_test(C.SAMPLER_CUSTOM, cnf, args.eta, args.epsilon, args.delta,
                       args.max, verbosity=0)
        t.CM_test(args.seed)


if __name__ == "__main__":
    main()
