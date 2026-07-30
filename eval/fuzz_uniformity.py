#!/usr/bin/env python3
"""Fuzz sampling uniformity against an EXACT oracle, GPAP vs CMSGen on identical instances.

The problem with uniformity claims on real instances is that nobody knows the true distribution:
|S| is astronomical, so every statistic is either a gross-skew heuristic (the `--uniformity fast`
l2_ratio) or a conditional-sampling certificate at parameters coarse enough to be cheap (Barbarik
— see `eval/barbarik_gpap.py`, and the note on its sensitivity below). This harness takes the
other route: GENERATE small formulas whose entire solution set can be enumerated by brute force,
so the target distribution is known exactly and a chi-square goodness-of-fit test has real power.

Two generators, both chosen so that "uniform over the support projections" is UNAMBIGUOUS:

  cnf      random k-CNF, support = every variable. Projection is the identity, so uniform-over-
           models and uniform-over-projections are the same distribution.
  circuit  a random combinational gate DAG (AND/OR/XOR with random inversions) over k inputs,
           Tseitin-encoded, with its outputs PINNED to a pattern that some input vector produces.
           The support is the inputs, which is an INDEPENDENT support by construction (every gate
           var is functionally determined), so again each projection has exactly one extension.
           This is the flagship circuit-sampling regime in miniature — an unrolled sequential
           circuit is just a combinational one over |PI|x T inputs.

Why the independence matters: CMSGen samples full models and we project, while GPAP decides the
support first. On a NON-independent support those two target different distributions (projections
would be weighted by their extension count), and the comparison would be meaningless. Both
generators sidestep that by construction.

The oracle is deliberately solver-free — exhaustive evaluation over all 2^n assignments (cnf) or
all 2^k input vectors (circuit) — so a bug in our BCP, our Tseitin encoder or pysat cannot hide a
bug in the sampler.

Two things are checked per (instance, sampler, config, seed):
  SOUNDNESS  every emitted row must be in the enumerated solution set. A miss is a hard failure,
             and a far more serious one than skew.
  UNIFORMITY chi-square GOF + total variation against uniform over all |S| solutions
             (gpap.sampling.metrics.uniformity_stats, mode "exact"). Independent seeds are
             combined with Fisher's method, because a single p=0.01 is noise, not a finding.

Sensitivity note, and the reason this exists alongside Barbarik: at the CMSGen paper's own
parameters (eps 0.3, eta 1.8, delta 0.1) Barbarik only rejects samplers that are ~1.8-far in L1.
The instance-dependent skew we have actually measured (TV~0.17, on BOTH GPAP and CMSGen) is two
orders of magnitude inside that, so Barbarik accepts it by design. Barbarik is the standard
certificate; this is the sensitive test.

  PYTHONPATH=. .venv/bin/python eval/fuzz_uniformity.py --count 40 \
      --cmsgen ~/.cache/gpap/bin/cmsgen --out .experiments/fuzz_uniformity.json
  # then certify the interesting ones with the standard tester:
  PYTHONPATH=. .venv/bin/python eval/barbarik_gpap.py --sampler gpap  <keep-dir>/*.cnf
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from gpap.sampling.metrics import chi2_sf, uniformity_stats  # noqa: E402

# Configs are (random_var, B, lookahead) exactly as gpap.sampling.pipeline defines them. All three
# CIRCUIT_PORTFOLIO members are swept by default: the bandit picks one per instance in production,
# so a uniformity claim that only holds for one of them is not a claim about the tool. "soft" is
# the one to watch — it biases polarity by conflict risk, which is a deliberate trade of
# distribution for success rate, and nothing had ever measured what it costs.
CONFIGS = {
    "b8": (True, 8, False),
    "b1": (True, 1, False),
    "soft": (True, 8, "soft"),
    # B sweep. B is the number of variables decided per propagation round, and it turns out to be
    # a uniformity knob: at B >= N a chunk assigns every variable a uniform random polarity in one
    # round and then propagates, which — because GPAP DISCARDS conflicted trajectories instead of
    # backtracking out of them — is exactly rejection sampling, hence exactly uniform. Small B
    # interleaves propagation with the decisions and weights solutions by their attraction basin.
    "b2": (True, 2, False), "b4": (True, 4, False), "b16": (True, 16, False),
    "b32": (True, 32, False), "b64": (True, 64, False), "b256": (True, 256, False),
}


# --------------------------------------------------------------------------- exact oracles

def all_assignments(n: int) -> np.ndarray:
    """(2^n, n) bool matrix of every assignment, column v = value of variable v+1."""
    idx = np.arange(1 << n, dtype=np.uint64)
    return ((idx[:, None] >> np.arange(n, dtype=np.uint64)[None, :]) & 1).astype(bool)


def brute_force_models(nv: int, clauses) -> np.ndarray:
    """Every satisfying assignment of `clauses`, as a (S, nv) uint8 matrix. Exhaustive: no SAT
    solver is involved anywhere in the oracle path."""
    bits = all_assignments(nv)
    ok = np.ones(bits.shape[0], dtype=bool)
    for cl in clauses:
        sat = np.zeros(bits.shape[0], dtype=bool)
        for lit in cl:
            col = bits[:, abs(lit) - 1]
            sat |= col if lit > 0 else ~col
        ok &= sat
        if not ok.any():                       # early out: an UNSAT draw is common near threshold
            return np.zeros((0, nv), dtype=np.uint8)
    return bits[ok].astype(np.uint8)


# --------------------------------------------------------------------------- generators

@dataclass
class Instance:
    name: str
    kind: str
    nv: int
    clauses: list
    support: list                              # 1-based vars, the `c ind` set
    sols: np.ndarray                           # (S, |support|) uint8 — the EXACT solution set
    meta: dict = field(default_factory=dict)

    @property
    def S(self) -> int:
        return int(self.sols.shape[0])

    def dimacs(self) -> str:
        head = f"c fuzz {self.kind} S={self.S} {json.dumps(self.meta, sort_keys=True)}\n"
        head += f"p cnf {self.nv} {len(self.clauses)}\n"
        head += "c ind " + " ".join(str(v) for v in self.support) + " 0\n"
        return head + "".join(" ".join(map(str, c)) + " 0\n" for c in self.clauses)


def gen_random_cnf(rng, s_window, nv_range=(12, 18), k_choices=(3, 4)) -> "Instance | None":
    """A random k-CNF over every variable. The clause/variable ratio is drawn wide and the
    instance kept only if |S| lands in the window — rejection sampling is simpler and less
    biased than trying to hit a solution count analytically."""
    nv = int(rng.integers(nv_range[0], nv_range[1] + 1))
    k = int(rng.choice(k_choices))
    ratio = float(rng.uniform(1.0, 3.2 if k == 3 else 5.5))
    m = max(1, int(round(ratio * nv)))
    clauses = []
    for _ in range(m):
        vs = rng.choice(nv, size=k, replace=False) + 1
        signs = rng.integers(0, 2, size=k) * 2 - 1
        clauses.append([int(v * s) for v, s in zip(vs, signs)])
    sols = brute_force_models(nv, clauses)
    if not (s_window[0] <= sols.shape[0] <= s_window[1]):
        return None
    return Instance("", "cnf", nv, clauses, list(range(1, nv + 1)), sols,
                    {"k": k, "ratio": round(ratio, 2), "m": m})


_TSEITIN = {
    # z = op(a, b) as clauses over the signed literals a, b and the gate output z.
    "and": lambda z, a, b: [[-z, a], [-z, b], [z, -a, -b]],
    "or": lambda z, a, b: [[z, -a], [z, -b], [-z, a, b]],
    "xor": lambda z, a, b: [[-z, a, b], [-z, -a, -b], [z, -a, b], [z, a, -b]],
}


def _simulate_block(k, gates, lo, hi):
    """Wire values for input vectors [lo, hi) -> (hi-lo, k+len(gates)) bool.

    Chunked so a wide support stays affordable: the full truth table at k=24 would be ~1 GB of
    wire values, while one block is a few tens of MB."""
    idx = np.arange(lo, hi, dtype=np.uint64)
    vals = np.zeros((hi - lo, k + len(gates)), dtype=bool)
    vals[:, :k] = ((idx[:, None] >> np.arange(k, dtype=np.uint64)[None, :]) & 1).astype(bool)
    for i, (op, a, b) in enumerate(gates):
        x = vals[:, abs(a) - 1] if a > 0 else ~vals[:, abs(a) - 1]
        y = vals[:, abs(b) - 1] if b > 0 else ~vals[:, abs(b) - 1]
        vals[:, k + i] = (x & y) if op == "and" else (x | y) if op == "or" else (x ^ y)
    return vals


def _verify_encoding(k, gates, gate_clauses, pin_clauses, accept, block=1 << 18):
    """Prove the encoding right, for THIS instance, before it is ever used as an oracle.

    Every gate var is functionally determined by the inputs, so the simulated row is the ONLY
    candidate extension of an input vector. Hence: models(CNF) projected onto the inputs == the
    accepted set, exactly when (a) the simulated rows satisfy every gate clause for ALL inputs and
    (b) they satisfy the pin clauses exactly on `accept`. Both are checked over the whole 2^k
    truth table (in blocks), so a wrong Tseitin clause cannot pass."""
    for lo in range(0, 1 << k, block):
        hi = min(1 << k, lo + block)
        vals = _simulate_block(k, gates, lo, hi)

        def sat(cl, vals=vals):
            s = np.zeros(vals.shape[0], dtype=bool)
            for lit in cl:
                col = vals[:, abs(lit) - 1]
                s |= col if lit > 0 else ~col
            return s

        if not all(sat(cl).all() for cl in gate_clauses):
            raise AssertionError("Tseitin gate clauses are not satisfied by the simulated values")
        pinned = np.ones(vals.shape[0], dtype=bool)
        for cl in pin_clauses:
            pinned &= sat(cl)
        if not np.array_equal(pinned, accept[lo:hi]):
            raise AssertionError("pinned CNF disagrees with the simulated accept set")


def gen_circuit(rng, s_window, k_range=(10, 15), gate_range=(10, 40),
                target="rare") -> "Instance | None":
    """A random combinational circuit, outputs pinned to a reachable pattern.

    Variables map 1:1 onto wires: 1..k are the primary inputs (the support), k+1..k+g the gates.
    `target` picks the pinned pattern the way the circuit pipeline does: "random" pins whatever a
    random input vector produces (size-biased toward easy, high-probability patterns) and "rare"
    pins the least frequent observed pattern (the regime our headline numbers use)."""
    k = int(rng.integers(k_range[0], k_range[1] + 1))
    g = int(rng.integers(gate_range[0], gate_range[1] + 1))
    ops = list(_TSEITIN)
    gates = []
    for i in range(g):
        z = k + i + 1                          # this gate's wire/var
        srcs = rng.choice(z - 1, size=2, replace=(z - 1 < 2)) + 1
        signs = rng.integers(0, 2, size=2) * 2 - 1
        gates.append((str(rng.choice(ops)), int(srcs[0] * signs[0]), int(srcs[1] * signs[1])))

    # Pin count must scale with the input count or |S| ~ 2^(k - n_out) leaves the enumerable
    # window as soon as the support grows: 2-6 outputs is right at k=12 and hopeless at k=20.
    lo_out = max(2, k - 14)
    n_out = int(rng.integers(lo_out, min(g, max(lo_out + 1, k - 4)) + 1))
    outs = [k + g - j for j in range(n_out)]   # the last gates are the primary outputs
    ocols = [o - 1 for o in outs]
    # Output pattern of EVERY input vector, computed in blocks (a wide support cannot hold the
    # whole wire table). Only the (2^k,) code array is kept.
    block = 1 << 18
    codes = np.empty(1 << k, dtype=np.uint32)
    w = (1 << np.arange(n_out)).astype(np.uint32)
    for lo in range(0, 1 << k, block):
        hi = min(1 << k, lo + block)
        codes[lo:hi] = _simulate_block(k, gates, lo, hi)[:, ocols].dot(w)
    uniq, cnts = np.unique(codes, return_counts=True)
    pick = uniq[cnts.argmin()] if target == "rare" else codes[int(rng.integers(0, 1 << k))]
    accept = codes == pick

    n_sol = int(accept.sum())
    if not (s_window[0] <= n_sol <= s_window[1]):
        return None                            # reject BEFORE the expensive exhaustive proof

    gate_clauses = []
    for i, (op, a, b) in enumerate(gates):
        gate_clauses += _TSEITIN[op](k + i + 1, a, b)
    bitpat = [(int(pick) >> j) & 1 for j in range(n_out)]
    pin_clauses = [[o if bit else -o] for o, bit in zip(outs, bitpat)]
    _verify_encoding(k, gates, gate_clauses, pin_clauses, accept)
    clauses = gate_clauses + pin_clauses

    # Unpack ONLY the accepted input vectors (|S| rows), never the full 2^k table.
    idx = np.flatnonzero(accept).astype(np.uint64)
    sols = ((idx[:, None] >> np.arange(k, dtype=np.uint64)[None, :]) & 1).astype(np.uint8)
    return Instance("", "circuit", k + g, clauses, list(range(1, k + 1)), sols,
                    {"inputs": k, "gates": g, "outputs": n_out, "target": target,
                     "p_accept": round(float(accept.mean()), 6),
                     # The pin constrained NOTHING: the chosen outputs are constant over every
                     # input, so "sampling" is a free hypercube. Same degeneracy as b02/b05/b07/
                     # s526 at T=25 (CLAUDE.md). Kept — a sampler that cannot be uniform on a free
                     # hypercube is badly broken — but reported separately, never as evidence
                     # about CONSTRAINED sampling.
                     "vacuous": bool(accept.all())})


def generate(rng, count, kinds, s_window, max_tries=400, sup_min=0, k_range=(10, 15)):
    """Rejection-sample `count` instances whose solution count lands in `s_window`.

    `sup_min` additionally requires |support| >= sup_min. This is what makes a campaign
    INFORMATIVE about a given B: at B >= |support| the sampler is exact rejection sampling and
    trivially uniform, so instances with a small support measure nothing about a B=8 config. Set
    it above the B under test."""
    out, tries, made = [], 0, {"cnf": 0, "circuit": 0}
    while len(out) < count and tries < max_tries * count:
        tries += 1
        kind = kinds[len(out) % len(kinds)] if len(kinds) > 1 else kinds[0]
        # Alternate the pin target across CIRCUITS (not across all instances — the kinds already
        # alternate, which would otherwise pin every circuit the same way).
        inst = (gen_random_cnf(rng, s_window) if kind == "cnf" else
                gen_circuit(rng, s_window, k_range=k_range,
                            target="rare" if made["circuit"] % 2 else "random"))
        if inst is None or len(inst.support) < sup_min:
            continue
        made[kind] += 1
        inst.name = f"{inst.kind}-{len(out):03d}-n{inst.nv}-S{inst.S}"
        out.append(inst)
    return out


# --------------------------------------------------------------------------- samplers

def _counts_from_rows(inst: Instance, rows) -> "tuple[np.ndarray, int]":
    """Histogram emitted support-projections over the enumerated solution set.
    Returns (counts over the S known solutions, number of rows that are NOT solutions)."""
    index = {tuple(int(b) for b in r): i for i, r in enumerate(inst.sols)}
    counts = np.zeros(inst.S, dtype=np.int64)
    unsound = 0
    for r in rows:
        i = index.get(tuple(int(b) for b in r))
        if i is None:
            unsound += 1
        else:
            counts[i] += 1
    return counts, unsound


def sample_gpap(inst: Instance, n: int, seed: int, config, device="cuda"):
    """-> (counts, unsound, n_emitted, wall_s). Raw formula: no unit closure / ELS, so the
    measurement is of the sampler and not of a preprocessing pass."""
    import torch

    from gpap.sampling.sampler import GPUSampler
    t0 = time.perf_counter()
    smp = GPUSampler.from_clauses(inst.nv, inst.clauses, torch.device(device),
                                  support=[v - 1 for v in inst.support], seed=seed, k_cap=8192)
    bits, cols = smp.sample_bits(n, config=config, max_chunks=4000)
    dt = time.perf_counter() - t0
    if bits is None or len(bits) == 0:
        return np.zeros(inst.S, dtype=np.int64), 0, 0, dt
    # sample_bits reports which original variable each column is; reorder to `support` order
    # rather than assuming the engine kept it (it compacts and can reorder).
    order = [cols.index(v) for v in inst.support]
    return (*_counts_from_rows(inst, bits[:, order]), len(bits), dt)


def sample_pipeline(inst: Instance, path, n: int, seed: int, portfolio_name: str,
                    allow_soft: bool, device="cuda"):
    """-> (counts, unsound, n_emitted, wall_s), through the FULL shipped pipeline.

    Unlike sample_gpap (which drives one config directly) this is `sample_file` exactly as the CLI
    calls it: parse, unit-closure simplify, warmup, entailed-clause mining, the adaptive bandit
    over a portfolio, and the GPU bit-projection output path. It answers the question that
    actually matters to a user — "is the tool, as shipped and with its defaults, uniform?" — which
    per-config numbers cannot, because the bandit chooses the config at runtime."""
    import torch

    from gpap.sampling.pipeline import portfolio as _portfolio
    from gpap.sampling.pipeline import sample_file
    t0 = time.perf_counter()
    info = sample_file(path, torch.device(device), n_time=min(n, 5000), out_cap=n, seed=seed,
                       configs=_portfolio(portfolio_name, allow_soft=allow_soft),
                       n_cov=0, uniformity="off", time_limit=120.0)
    dt = time.perf_counter() - t0
    bits, cols = info.get("samples"), info.get("cols")
    if bits is None or len(bits) == 0:
        return np.zeros(inst.S, dtype=np.int64), 0, 0, dt
    order = [list(cols).index(v) for v in inst.support]
    return (*_counts_from_rows(inst, np.asarray(bits)[:, order]), len(bits), dt)


def sample_control(inst: Instance, n: int, seed: int, tv: float):
    """A synthetic sampler with a KNOWN distance from uniform — the harness testing itself.

    tv=0 is the negative control: draws are exactly uniform over the enumerated solution set, so
    any SKEW verdict against it is a bug in the statistics, not a finding about a sampler. tv>0 is
    the positive control: mass is shifted by a fixed amount onto half the solutions, giving a
    distribution whose total variation from uniform is known in closed form, which shows the test
    actually has the power to detect a skew of that size at the chosen n. Without both, "CMSGen
    skews on 5/6" is indistinguishable from "our chi-square is wrong"."""
    S = inst.S
    p = np.full(S, 1.0 / S)
    if tv > 0 and S >= 2:
        h = S // 2                             # leave the middle cell alone when S is odd
        p[:h] += 2.0 * tv / S
        p[S - h:] -= 2.0 * tv / S
        p = np.clip(p, 0.0, None)
        p /= p.sum()
    t0 = time.perf_counter()
    idx = np.random.default_rng(seed).choice(S, size=n, p=p)
    counts = np.bincount(idx, minlength=S).astype(np.int64)
    return counts, 0, int(n), time.perf_counter() - t0


def control_tv(S: int, tv: float) -> float:
    """The TV the positive control actually realises (the closed form above is exact only for
    even S) — reported so the printed number is the truth, not the requested target."""
    p = np.full(S, 1.0 / S)
    if tv > 0 and S >= 2:
        h = S // 2
        p[:h] += 2.0 * tv / S
        p[S - h:] -= 2.0 * tv / S
        p = np.clip(p, 0.0, None)
        p /= p.sum()
    return float(0.5 * np.abs(p - 1.0 / S).sum())


def sample_cmsgen(inst: Instance, path, n: int, seed: int, cmsgen_bin, scratch):
    """-> (counts, unsound, n_emitted, wall_s). CMSGen emits FULL assignments (it ignores
    `c ind`), so we project onto the support here — sound only because both generators make the
    support independent."""
    out = os.path.join(scratch, "cms.txt")
    t0 = time.perf_counter()
    p = subprocess.run([cmsgen_bin, "--samples", str(n), "--seed", str(seed), "--verb", "0",
                        "--samplefile", out, path], capture_output=True, text=True, timeout=900)
    dt = time.perf_counter() - t0
    if p.returncode not in (0, 10):             # 10 = SATISFIABLE, the DIMACS solver convention
        raise RuntimeError(f"cmsgen failed on {path}: {p.stdout[-400:]}{p.stderr[-400:]}")
    cols = [v - 1 for v in inst.support]
    rows = []
    with open(out) as f:
        for line in f:
            lits = [int(x) for x in line.split() if x.lstrip("-").isdigit()]
            if lits and lits[-1] == 0:
                lits = lits[:-1]
            if not lits:
                continue
            # A variable absent from the line would silently read as FALSE below and MANUFACTURE
            # skew, so require the full assignment CMSGen documents itself as writing rather than
            # trusting it. Cheap, and the failure it guards against is invisible otherwise.
            if len({abs(x) for x in lits}) != inst.nv:
                raise RuntimeError(f"cmsgen line covers {len({abs(x) for x in lits})} of "
                                   f"{inst.nv} vars — the projection would be wrong")
            pos = np.zeros(inst.nv, dtype=np.uint8)
            for lit in lits:
                if lit > 0:
                    pos[lit - 1] = 1
            rows.append(pos[cols])
    os.remove(out)
    return (*_counts_from_rows(inst, rows), len(rows), dt)


# --------------------------------------------------------------------------- statistics

def fisher_combine(pvals):
    """Fisher's method over INDEPENDENT seeds: -2*sum(ln p) ~ chi2 with 2R dof. A real skew
    replicates across seeds and its combined p collapses; a fluke does not."""
    ps = [max(float(p), 1e-300) for p in pvals if p is not None]
    if not ps:
        return None
    return chi2_sf(-2.0 * sum(math.log(p) for p in ps), 2 * len(ps))


def evaluate(inst, draw, seeds, n):
    """Run one sampler over `seeds` and reduce to a verdict dict. `draw(n, seed)` returns
    (counts, unsound, emitted, wall)."""
    per, tvs, ps, unsound, emitted, wall = [], [], [], 0, 0, 0.0
    for sd in seeds:
        counts, bad, got, dt = draw(n, sd)
        unsound += bad
        emitted += got
        wall += dt
        u = uniformity_stats(counts, int(counts.sum()), S=inst.S)
        per.append({"seed": sd, "n": int(counts.sum()), "emitted": got, "unsound": bad,
                    "tv": u.get("tv"), "p": u.get("p"), "chi2": u.get("chi2")})
        if u.get("tv") is not None:
            tvs.append(u["tv"])
        ps.append(u.get("p"))
    return {"per_seed": per, "tv_median": float(np.median(tvs)) if tvs else None,
            "tv_max": float(max(tvs)) if tvs else None, "p_fisher": fisher_combine(ps),
            "unsound": unsound, "emitted": emitted, "wall_s": round(wall, 3)}


def verdict(res, alpha):
    if res["emitted"] == 0:
        return "NO-SAMPLES"
    if res["unsound"]:
        return "UNSOUND"
    p = res["p_fisher"]
    if p is None:
        return "n/a"
    return "SKEW" if p < alpha else "ok"


# --------------------------------------------------------------------------- driver

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--count", type=int, default=24, help="instances to generate")
    ap.add_argument("--kinds", nargs="+", default=["cnf", "circuit"], choices=["cnf", "circuit"])
    ap.add_argument("--seed", type=int, default=1, help="master seed for GENERATION")
    ap.add_argument("--seeds", type=int, default=3, help="sampler seeds per instance")
    ap.add_argument("--circuit-inputs", type=int, nargs=2, default=[10, 15],
                    metavar=("MIN", "MAX"),
                    help="primary-input count (= |support|) range for the circuit generator. The "
                         "ratio |support|/B is what drives skew — a real unroll has |support| in "
                         "the hundreds against B=8, so a suite at 12-16 tests only ~2 decision "
                         "rounds. The oracle is exhaustive over 2^MAX, so MAX ~24 is the ceiling")
    ap.add_argument("--sup-min", type=int, default=12,
                    help="require |support| >= this. At B >= |support| the sampler is exact "
                         "rejection sampling and trivially uniform, so a campaign meant to say "
                         "something about B=8 must only contain instances with a LARGER support "
                         "(default 12 > the default B of 8)")
    ap.add_argument("--s-min", type=int, default=8, help="smallest solution count to accept")
    ap.add_argument("--s-max", type=int, default=400, help="largest solution count to accept")
    ap.add_argument("--n-mult", type=int, default=100,
                    help="samples per instance = n-mult * |S| (chi2 needs >= 5|S|; 100 gives the "
                         "test real power against a TV around 0.05)")
    ap.add_argument("--n-max", type=int, default=200_000)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS),
                    help="raw single-config arms (drive one config directly)")
    ap.add_argument("--pipelines", nargs="+", default=[], metavar="NAME",
                    help="FULL-pipeline arms — 'default', 'circuit', or either with '+soft' "
                         "(e.g. circuit+soft). These run sample_file exactly as the CLI does "
                         "(simplify + mining + the adaptive bandit), which is the only way to "
                         "test what a user actually gets")
    ap.add_argument("--cmsgen", default=None, help="cmsgen binary -> run the head-to-head")
    ap.add_argument("--controls", type=float, nargs="*", default=[0.0, 0.05, 0.10],
                    help="synthetic control samplers at these TV distances from uniform. 0.0 is "
                         "the negative control and MUST come out ok (otherwise the statistics are "
                         "broken and no verdict here means anything); values > 0 measure the "
                         "test's POWER against a real skew of that size at this n, which is what "
                         "makes an 'ok' interpretable. Pass --controls with no values to disable")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alpha", type=float, default=1e-3,
                    help="per-instance significance for the SKEW flag. The default keeps the "
                         "family-wise error near 5%% over ~50 instances (Bonferroni)")
    ap.add_argument("--out", default=None, help="write the full result JSON here")
    ap.add_argument("--keep-dir", default=None,
                    help="write every generated instance as DIMACS here (feed them to "
                         "eval/barbarik_gpap.py for the standard certificate)")
    ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp"))
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    insts = generate(rng, args.count, args.kinds, (args.s_min, args.s_max),
                     sup_min=args.sup_min, k_range=tuple(args.circuit_inputs))
    if len(insts) < args.count:
        print(f"warning: only generated {len(insts)}/{args.count} instances in the |S| window "
              f"with |support| >= {args.sup_min}")
    seeds = list(range(1, args.seeds + 1))
    if args.keep_dir:
        os.makedirs(args.keep_dir, exist_ok=True)
    print(f"{len(insts)} instances | configs {args.configs} | {len(seeds)} seeds "
          f"| n = {args.n_mult}*|S| (cap {args.n_max:,})\n")

    hdr = f"{'instance':<28} {'|S|':>5} {'n':>7} {'sampler':<10} {'TVmed':>7} {'p_fisher':>10} verdict"
    print(hdr)
    print("-" * len(hdr))
    rows, flags = [], []
    for inst in insts:
        n = min(args.n_max, max(args.n_mult * inst.S, 2000))
        path = os.path.join(args.keep_dir or args.scratch, inst.name + ".cnf")
        with open(path, "w") as f:
            f.write(inst.dimacs())
        row = {"instance": inst.name, "kind": inst.kind, "nv": inst.nv,
               "clauses": len(inst.clauses), "support": len(inst.support), "S": inst.S,
               "n": n, "meta": inst.meta, "samplers": {}}
        for cfg in args.configs:
            res = evaluate(inst, lambda n, sd, c=cfg: sample_gpap(inst, n, sd, CONFIGS[c],
                                                                 args.device), seeds, n)
            row["samplers"][f"gpap:{cfg}"] = res
        for spec in args.pipelines:
            pname, soft = (spec[:-5], True) if spec.endswith("+soft") else (spec, False)
            row["samplers"][f"pipe:{spec}"] = evaluate(
                inst, lambda n, sd, p=pname, s=soft: sample_pipeline(inst, path, n, sd, p, s,
                                                                     args.device), seeds, n)
        if args.cmsgen:
            row["samplers"]["cmsgen"] = evaluate(
                inst, lambda n, sd: sample_cmsgen(inst, path, n, sd, args.cmsgen, args.scratch),
                seeds, n)
        for tv in args.controls:
            key = "ctl:uniform" if tv == 0 else f"ctl:tv{tv:g}"
            res = evaluate(inst, lambda n, sd, t=tv: sample_control(inst, n, sd, t), seeds, n)
            res["control_tv"] = round(control_tv(inst.S, tv), 4)
            row["samplers"][key] = res
        for name, res in row["samplers"].items():
            v = verdict(res, args.alpha)
            res["verdict"] = v
            tv = res["tv_median"]
            p = res["p_fisher"]
            print(f"{inst.name:<28} {inst.S:>5} {n:>7} {name:<10} "
                  f"{tv if tv is None else round(tv, 4):>7} "
                  f"{'n/a' if p is None else f'{p:.2e}':>10} {v}")
            if v in ("SKEW", "UNSOUND", "NO-SAMPLES") and not name.startswith("ctl:"):
                flags.append((inst.name, name, v, tv))
        rows.append(row)
        if not args.keep_dir:
            os.remove(path)

    # ------------------------------------------------------------------ summary
    names = list(rows[0]["samplers"]) if rows else []
    print("\n=== summary ===")
    for name in names:
        if name.startswith("ctl:"):
            continue
        vs = [r["samplers"][name]["verdict"] for r in rows]
        tvs = [r["samplers"][name]["tv_median"] for r in rows
               if r["samplers"][name]["tv_median"] is not None]
        print(f"  {name:<12} ok {vs.count('ok'):>3}/{len(vs)}  skew {vs.count('SKEW'):>3}  "
              f"unsound {vs.count('UNSOUND'):>3}  TV med {np.median(tvs):.4f} "
              f"max {max(tvs):.4f}" if tvs else f"  {name:<12} no data")
    # Controls. The negative control is a genuine PASS/FAIL gate on the statistics; the positive
    # controls are not — they measure the test's POWER at this n, which is the number a reader
    # needs to interpret every "ok" above ("ok" means "no skew detectable at this sensitivity",
    # never "uniform").
    ctl_ok = True
    for name in [n for n in names if n.startswith("ctl:")]:
        vs = [r["samplers"][name]["verdict"] for r in rows]
        hits = vs.count("SKEW")
        tvs = {r["samplers"][name].get("control_tv") for r in rows}
        if name == "ctl:uniform":
            ctl_ok = hits == 0
            print(f"  {name:<12} flagged {hits:>3}/{len(vs)}  negative control (expects 0)"
                  f"  {'PASS' if ctl_ok else 'FAIL — the statistics are wrong'}")
        else:
            print(f"  {name:<12} detected {hits:>3}/{len(vs)}  power against a real "
                  f"TV of {min(tvs):.3f}-{max(tvs):.3f} at n={args.n_mult}|S| x "
                  f"{len(seeds)} seeds")
    if not ctl_ok:
        print("  !! the negative control was flagged — treat every verdict above as unvalidated")
    if args.cmsgen and "cmsgen" in names:
        # The question that actually matters: does GPAP skew where CMSGen does NOT? Shared skew is
        # a property of the algorithm class (random var + random polarity + propagate, no
        # backtracking to re-balance), which both tools implement.
        for name in names:
            if name == "cmsgen" or name.startswith("ctl:"):
                continue
            both = sum(1 for r in rows if r["samplers"][name]["verdict"] == "SKEW"
                       and r["samplers"]["cmsgen"]["verdict"] == "SKEW")
            only = sum(1 for r in rows if r["samplers"][name]["verdict"] == "SKEW"
                       and r["samplers"]["cmsgen"]["verdict"] != "SKEW")
            print(f"  {name} vs cmsgen: both skew {both}, {name}-only {only}")
    if flags:
        print("\nflagged:")
        for nm, sampler, v, tv in flags:
            print(f"  {v:<10} {sampler:<12} {nm}" + (f"  TV={tv:.4f}" if tv else ""))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"schema": 1, "args": vars(args), "rows": rows}, f, indent=1)
        print(f"\nwrote {args.out}")
    return 1 if any(v in ("UNSOUND",) for _, _, v, _ in flags) else 0


if __name__ == "__main__":
    sys.exit(main())
