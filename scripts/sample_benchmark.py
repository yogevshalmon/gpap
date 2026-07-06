#!/usr/bin/env python3
"""Benchmark the GPU BCP-completion sampler (gpap.solver.sampler.GPUSampler) against CMSGen.

GPAP runs CMSGen's UniformLikeWitness algorithm (random var + random polarity + unit propagation)
massively in parallel on the GPU. This harness compares them with the SAT-sampling literature's
own metrics (Golia et al., FMCAD'21):

  THROUGHPUT  — samples/sec (wall time to produce --n-time samples, including setup).
  QUALITY     — 2-wise coverage %: for each pair of sampling-set vars, the fraction of pairs whose
                all four (±,±) combinations appear across the samples (the combinatorial-testing
                quality metric CMSGen reports). Higher = better solution-space spread.
  SOUNDNESS   — GPAP: GPU check (every model's clauses all have sat_count>0) + optional CPU verify.

GPAP uses unit-propagation simplification (--simplify, on by default): strictly model-preserving
(unlike the VE/BCE that CMSGen disables), validated in tests/test_sampler.py — it shrinks M so a
much larger K fits, which is where the GPU throughput win comes from.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/sample_benchmark.py \
      --glob 'benchmarks/sampling/*.cnf' --device cuda \
      --cmsgen /path/to/cmsgen --n-time 50000 --n-cov 5000 --output .experiments/sample.json
"""
import argparse, glob, json, os, subprocess, sys, time

import numpy as np
import torch
sys.path.insert(0, ".")
from pysat.formula import CNF
from gpap.solver.sampler import GPUSampler, parse_independent_support


# CIRCUIT portfolio (the default here — this harness is currently evaluated on CircuitSAT
# unrollings). Measured per-config on 22+ circuit instances (RTX 4080, 2026-07-02..04); each entry
# is the outright winner somewhere, the adaptive bandit picks per instance:
#   (True, 8, "fused")  — fused-trajectory kernel at boosted K_fused: broad deep-circuit winner.
#   (True, 1, "fused")  — B=1's success edge is free in-kernel: wins the conflict-heavy deep set.
#   (True, 8, "fsoft")  — in-kernel ε-greedy soft risk-polarity: wins the tight-cascade family.
# (True, 64, "orac") — high-B on the PYTHON engine — was REMOVED 2026-07-06: measured on the full
# 64-instance circuits dir (RTX 4080, per-config bandit breakdown), it wins 0/64 and is never
# within 1.05x of the winner (closest: b09 T=1); its original niche (ultra-shallow wide, b14 T=1)
# is now >3x behind the boosted-K fused configs, while its cold-start chunk cost 0.2-5.1s per
# instance (~67s across the dir). Re-add if a GPU/regime shifts the balance (e.g. A100 large-K
# PyTorch-engine batching) — the "orac" lookahead mode itself remains fully supported.
# (scripts/colab_sampling_benchmark.py imports this same list — single source of truth.)
CIRCUIT_PORTFOLIO = [(True, 8, "fused"), (True, 1, "fused"), (True, 8, "fsoft")]


def distinct_projections(models, support):
    """Count DISTINCT support-projections among `models` (support = 1-based vars). Returns
    (n_distinct, n_models). This is the "unique solutions" verification metric — computed OFF the
    throughput timing path (the timed run uses collect=False), purely to confirm the sampler emits
    genuinely distinct solutions (not re-emitting the same model). Used for both GPAP and CMSGen on
    the SAME collected set so the distinct COUNTS are directly comparable."""
    if not models:
        return 0, 0
    seen = set()
    for m in models:
        s = set(m)
        seen.add(tuple(v if v in s else -v for v in support))
    return len(seen), len(models)


def two_wise_coverage(models, support):
    """% of sampling-var pairs whose all 4 (±,±) combos appear. support: 1-based vars."""
    S = len(support)
    if S < 2 or not models:
        return 0.0
    idx = {v: i for i, v in enumerate(support)}
    M = np.zeros((len(models), S), dtype=np.int8)
    for r, m in enumerate(models):
        s = set(m)
        for v in support:
            if v in s:
                M[r, idx[v]] = 1
    covered = total = 0
    for i in range(S):
        ci = M[:, i].astype(np.int16) << 1
        for j in range(i + 1, S):
            total += 1
            if np.unique(ci | M[:, j]).size == 4:
                covered += 1
    return 100.0 * covered / total if total else 0.0


# Default config portfolio for the adaptive bandit (per-config measured 2026-07-02):
#   (True, 8, False)   — the classic niche config (BCP-determined support; fast tensor path).
#   (True, 1, "fused") — v2 fused-trajectory Triton kernel at boosted K_fused: the new sk_ winner
#                        (35.sk 1.44M samples/s vs 654k for B8 — B=1's success edge is free
#                        in-kernel and K_fused adds ~5x parallelism). Degrades to (True,1) if the
#                        kernel is unavailable.
#   (False, 1, False)  — structured/index order for the search-heavy "service" family.
# The bandit measures each at full K and spends the budget on the winner per formula.
DEFAULT_PORTFOLIO = [(True, 8, False), (True, 1, "fused"), (False, 1, False)]


def gpap_sample(path, device, *, simplify, n_time, n_cov, b, k_cap, max_chunks=None,
                time_limit=60.0, cov_chunks=4, max_rounds=300, lookahead=False,
                verify_clauses=None, verify_cpu=False, cov_k_cap=2048, els=False,
                random_var="auto", configs="default", use_kernel="auto", mine="auto",
                mine_gate=0.15, min_measure_chunks=3):
    """Throughput run timed end-to-end (construction + simplify + sample, to match CMSGen's
    'produce N samples' wall clock) + a small collect run for coverage/soundness. `n_time` is a
    CONTRACT: the timed run keeps sampling until it has delivered that many models. Safety
    valves (both reported via the '<' marker when they fire): `time_limit` seconds of sampling
    wall, and the sampler's zero-progress stop (0 models after every config is measured — a
    0%-success instance would otherwise spin forever). `max_chunks` (default None = unbounded)
    remains as an expert knob; the old default of 8 silently capped low-success instances far
    below the requested n_time.

    `configs` (default = adaptive portfolio DEFAULT_PORTFOLIO): a throughput bandit that runs the
    best config per chunk at full K, dynamically switching to the winner — cracks the search-heavy
    "service" family (structured order + B=1) AND keeps the niche fast (random order + B=8) in one
    run, more robustly than the small-K `random_var="auto"` probe. Pass configs=None to fall back to
    `random_var` (e.g. "auto"). The dominant config is REUSED for the coverage run (focused).

    The coverage run COLLECTS models to host RAM (one chunk = K models × N ints reconstructed), so
    its K is capped to `cov_k_cap` independently of the (full-K) timing run — otherwise a large
    auto-K on a low-RAM host (12 GB Colab) OOM-kills the session collecting a full chunk."""
    if configs == "default":
        configs = DEFAULT_PORTFOLIO

    def _warmup(s):
        # Warm up CUDA kernels (first-chunk JIT / allocation) at a tiny K so the timed run isn't
        # skewed by one-time warmup — this is what inflated the first shallow circuit's number.
        # Returns the probe chunk's stats (a FREE gate signal for mining) and restores the RNG
        # so the timed run stays deterministic.
        # FULL round budget + one chunk per FUSED-FAMILY config: with the old min(10, max_rounds)
        # cap NO deep-circuit trajectory could complete in the warmup, so success_rate read 0% on
        # every c25 unroll; and measuring only configs[0] (fused B=8) misreads instances whose
        # success lives in another config (b03 T=25: B=8 4.5%, B=1 99%). Either way the miner's
        # "stuck" gate fired suite-wide and burned the full L2/L3 budget everywhere (77.6s of the
        # c25 set's 99s overhead). Fused/fsoft chunks are in-kernel — cheap at any round budget —
        # while an orac chunk can cost seconds even at K=1024 (measured 2.5s on b10 T=25), so the
        # slow configs keep paying their own cold-start in the timed run exactly as before.
        # Returns "best_success" = the best warmed config's completion rate — THE stuck signal.
        cw = [c for c in (configs or []) if len(c) > 2 and c[2] in ("fused", "fsoft")]
        cw = cw or (configs[:1] if configs else None)
        _k = s.K
        s.K = min(s.K, 1024)
        w = s.sample(1, configs=cw, decisions_per_round=b, random_var=random_var,
                     max_decision_rounds=max_rounds, max_chunks=len(cw) if cw else 1,
                     collect=False, lookahead=lookahead)
        bd = (w.get("config_breakdown") or {}).values()
        w["best_success"] = max((c["success_rate"] for c in bd), default=w["success_rate"])
        s.K = _k
        s._gen.manual_seed(5)
        import random as _r
        s._pyrng = _r.Random(5)
        torch.cuda.synchronize()
        return w

    t_start = time.perf_counter()   # TRUE end-to-end clock: parse + construction + simplify +
    #                                 warmup + mining + the timed run — the wall CMSGen is charged for
    phases = {}                     # per-phase seconds (overhead attribution; lands in the JSON)
    # THE single parse. It is shared by sampler construction, mining and the caller's table
    # metadata — the flow previously parsed every CNF three times (from_cnf_file + miner + main),
    # ~2s/parse extra on a 1M-clause unroll.
    cnf0 = CNF(from_file=path)
    sup0 = parse_independent_support(path)
    phases["parse"] = time.perf_counter() - t_start
    _tc = time.perf_counter()
    smp = GPUSampler.from_clauses(cnf0.nv, cnf0.clauses, device, simplify=simplify, els=els,
                                  support=sup0, k_cap=k_cap, seed=5, use_kernel=use_kernel)
    phases["construct"] = time.perf_counter() - _tc
    mine_info = None
    if device.type == "cuda":
        _tw = time.perf_counter()
        w = _warmup(smp)
        phases["warmup"] = time.perf_counter() - _tw
        # Entailed-clause mining (learned clauses are ENTAILED — the model set is exactly
        # unchanged; measured: service family 0.02-0.7x -> 1.1-6.7x vs CMSGen, circuit backbone
        # instances 1.2-16x). Mining is now nearly self-gating (see gpap/solver/miner.py): the L1
        # backbone sweep costs 0.1-0.5s when there is nothing to learn, and the expensive L2/L3
        # support-tuple tiers only run when L1 found units — so mine on every non-tiny formula
        # (M >= 3000; below that the whole timed run is ~instant and even 0.3s of probing would
        # dominate the e2e wall). The warmup success rate (< mine_gate = truly stuck) FORCES the
        # full L2/L3 tiers even without L1 yield — the original service-family behaviour.
        if mine == "auto" and smp.M >= 3000:
            try:
                from gpap.solver.miner import mine_entailed
                # Two-stage mining. Stage 1: L1 backbone units ALWAYS (cheap fixpoint, THE
                # measured circuit win) + the L2/L3 tiers that are CHEAP for this support size
                # (pairs="auto", see the miner docstring — b09's 4 ternaries are 1.4x for 0.7s,
                # s386's 18 binaries 1.48x for ~2s; wide-support tiers never paid). Stage 2
                # (full tiers — the service-family lever) only if the instance is still stuck
                # AFTER the units are in: pre-mining success is low almost everywhere because
                # mining is exactly what lifts it (b03 T=25: best config 4.5% before its 1007
                # units, 99% after), so gating the tiers on the PRE-mining rate burned the full
                # L2/L3 budget suite-wide (77.6s of the c25 set's 99s overhead).
                _tm = time.perf_counter()
                mi = mine_entailed(cnf0.nv, cnf0.clauses, sup0, device, pairs="auto",
                                   pairs_min_units=0)
                phases["mine"] = time.perf_counter() - _tm
                learned = mi["learned"]
                stuck = w["best_success"] < mine_gate
                if learned:
                    # Rebuild on the augmented formula. No trailing re-warmup needed: kernels
                    # are JIT'd and the fresh seed=5 sampler starts the timed run from the same
                    # RNG state the warmup+reseed dance produced.
                    _tm = time.perf_counter()
                    smp = GPUSampler.from_clauses(
                        cnf0.nv, cnf0.clauses + learned, device, simplify=simplify, els=els,
                        support=sup0, k_cap=k_cap, seed=5, use_kernel=use_kernel)
                    phases["rebuild"] = time.perf_counter() - _tm
                    _tm = time.perf_counter()
                    stuck = _warmup(smp)["best_success"] < mine_gate   # post-units stuck signal
                    phases["recheck"] = time.perf_counter() - _tm
                if stuck:
                    _tm = time.perf_counter()
                    mi2 = mine_entailed(cnf0.nv, cnf0.clauses + learned, sup0, device,
                                        pairs=True, pairs_min_units=0,
                                        max_seconds=max(1.0, 5.0 - mi["seconds"]))
                    phases["mine"] += time.perf_counter() - _tm
                    if mi2["learned"]:
                        learned = learned + mi2["learned"]
                        _tm = time.perf_counter()
                        smp = GPUSampler.from_clauses(
                            cnf0.nv, cnf0.clauses + learned, device, simplify=simplify,
                            els=els, support=sup0, k_cap=k_cap, seed=5, use_kernel=use_kernel)
                        phases["rebuild"] = (phases.get("rebuild", 0.0)
                                             + time.perf_counter() - _tm)
                    for k2 in ("n_units", "n_binary", "n_ternary", "seconds"):
                        mi[k2] += mi2[k2]
                mine_info = {k2: mi[k2] for k2 in ("n_units", "n_binary", "n_ternary", "seconds")}
            except RuntimeError as e:   # mining is an optimization — never let it kill the run
                print(f"   [mine] skipped after error: {e}", file=sys.stderr)
    t0 = time.perf_counter()
    # A big-K GPU meets n_time in the bandit cold-start (1 chunk/config), so the headline rates would
    # rest on a single noisy chunk (measured: best_rate_sps ±1.9x run-to-run). min_measure_chunks makes
    # the winner run >= that many back-to-back chunks post-target for a stable, config-consistent rate.
    mmc = max(1, min(min_measure_chunks,
                     (max_chunks - (len(configs) if configs else 1)) if max_chunks
                     else min_measure_chunks))
    if configs:
        t = smp.sample(n_time, configs=configs, max_decision_rounds=max_rounds,
                       max_chunks=max_chunks, collect=False, lookahead=lookahead,
                       min_measure_chunks=mmc, max_seconds=time_limit)
    else:
        t = smp.sample(n_time, decisions_per_round=b, random_var=random_var,
                       max_decision_rounds=max_rounds, max_chunks=max_chunks, collect=False,
                       lookahead=lookahead, min_measure_chunks=mmc, max_seconds=time_limit)
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    # Headline throughput = the best-THROUGHPUT config's steady-state rate (best_rate_sps), NOT the
    # end-to-end blend. With the adaptive bandit the run measures every config once (see sampler.py);
    # on a big-K GPU a single chunk exceeds n_time, so the blend would report whatever config ran
    # first and be dragged by slower exploration chunks. best_rate_sps is the rate a real run achieves
    # after committing to the winner (construction/warmup amortised). Falls back to the blend if a
    # single non-adaptive config was used (no per-config timing).
    sps_e2e = t["n_models"] / wall if wall > 0 else 0.0
    # TRUE end-to-end rate (construction + simplify + warmup + MINING + sampling) — the honest
    # apples-to-apples number vs CMSGen's subprocess wall. The steady-state headline (best_rate_sps)
    # remains valid for large-N workloads where the one-time costs amortise away.
    wall_total = time.perf_counter() - t_start
    sps_e2e_total = t["n_models"] / wall_total if wall_total > 0 else 0.0
    sps = t.get("best_rate_sps") or sps_e2e
    brc = t.get("best_rate_config")
    # Work rates for the SAME config as `sps` (best_rate). Falls back to the whole-run blend for a
    # single non-adaptive config. This keeps dec/s, impl/s, assigns/s consistent with samples/s —
    # otherwise the blend is dragged by the bandit's slow exploration chunk (e.g. orac).
    dps = t.get("best_rate_decisions_per_s") if t.get("best_rate_decisions_per_s") is not None else t.get("decisions_per_s")
    ips = t.get("best_rate_implications_per_s") if t.get("best_rate_implications_per_s") is not None else t.get("implications_per_s")
    aps = t.get("best_rate_assigns_per_s") if t.get("best_rate_assigns_per_s") is not None else t.get("assigns_per_s")
    if brc:                                          # (rv, B, lookahead) of the fastest config
        tuned_rv, tuned_b, tuned_la = brc[0], brc[1], brc[2]
        succ = t.get("best_rate_success", t["success_rate"])
    else:
        tuned_rv, tuned_b = t["random_var"], t["B"]
        tuned_la = t.get("lookahead", lookahead)
        succ = t["success_rate"]
    if n_cov:
        smp.K = min(smp.K, cov_k_cap)                # bound host RAM from collected/reconstructed models
    # Coverage / UNIQUE-solution run: collect models with the winning config and count distinct
    # support-projections. Deliberately SEPARATE from the timed run above (which is collect=False), so
    # the model reconstruction / dedup cost is NOT charged to samples/s — this is a correctness/quality
    # readout ("do we actually emit distinct solutions?"), not a throughput number.
    vc = cnf0.clauses if (verify_cpu and verify_clauses is None) else verify_clauses
    c = (smp.sample(n_cov, decisions_per_round=tuned_b, random_var=tuned_rv, max_decision_rounds=max_rounds,
                    max_chunks=cov_chunks, collect=True, verify_clauses=vc,
                    lookahead=tuned_la) if n_cov else {})
    del smp
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # Estimated UNIQUE solutions/s: timed raw rate × the distinct fraction measured post-hoc on the
    # coverage sample. Dedup/reconstruction NEVER touches the timed run (collect=False) — this is a
    # pure post-processing estimate (exact on large spaces where distinct_frac≈1; honest on tiny
    # spaces where re-sampling dominates). None when no coverage run was requested.
    n_coll = len(c.get("models", []))
    uniq_sps = (sps * c["n_distinct"] / n_coll) if (n_coll and c.get("n_distinct")) else None
    phases["timed"] = wall
    return {"samples_per_s": sps, "success_rate": succ,
            "nv": cnf0.nv, "n_clauses": len(cnf0.clauses),  # ORIGINAL sizes (pre-simplify) — the
            #   caller's table metadata; saves main() a third parse of the same file
            "sup1": [v + 1 for v in (sup0 or range(cnf0.nv))],
            "sps_e2e_total": sps_e2e_total,                   # incl. construction+simplify+mining
            "phases": phases,                                 # per-phase seconds of the e2e wall
            "n_models": t["n_models"],                        # total solutions found in the timed run
            # True when a safety valve (time_limit / zero-progress / explicit max_chunks) stopped
            # the timed run BEFORE it delivered n_time models. Rates stay valid either way.
            "chunk_capped": t["n_models"] < n_time,
            "wall_timed_s": wall,                             # the solver's sampling wall (timed run)
            "wall_total_s": wall_total,                       # e2e wall incl. construction+simplify+mining
            "mined": mine_info,                               # entailed-clause mining stats (None = skipped)
            "unique_sps_est": uniq_sps,
            "assigns_per_s": aps,                             # literals assigned/s (CDCL props/s analogue)
            "decisions_per_s": dps,                           # branch decisions/s (best-rate config)
            "implications_per_s": ips,                        # unit-prop implications/s (=assigns-decisions)
            "kernel": t.get("kernel", False),                 # fused Triton BCP kernel active?
            "attempts_per_s": t.get("attempts_per_s"),        # raw trajectory throughput (K/chunk rate)
            "bcp_rounds_per_chunk": t.get("bcp_rounds_per_chunk"),
            "bcp_ops_per_attempt": t.get("bcp_ops_per_attempt"),
            # gpu_sound: from the coverage run when it exists, else from the timed run (the timed
            # run's capped per-chunk check) — so soundness is reported even with quality checks off
            "support_size": t["support_size"],
            "gpu_sound": c.get("gpu_sound", t.get("gpu_sound", True)),
            "cpu_sound": c.get("all_valid"), "models": c.get("models", []),
            "n_distinct": c.get("n_distinct"), "n_collected": n_coll,
            "tuned_rv": tuned_rv, "tuned_b": tuned_b, "tuned_la": tuned_la,
            "config_breakdown": t.get("config_breakdown")}


def cmsgen_sample(path, cmsgen_bin, n, scratch, keep=None):
    """Run CMSGen to produce n samples; return (samp/s, parsed models).

    `keep` caps how many parsed models are retained in memory (the rest are only COUNTED, for the
    rate). This is essential on low-RAM hosts (e.g. a 12 GB Colab): retaining all `n` full models
    is `n × N` Python ints (~8-9 GB at n=50k, N=5k) and OOM-kills the session. None = keep all."""
    out = os.path.join(scratch, "cms_" + os.path.basename(path) + ".txt")
    t0 = time.perf_counter()
    try:
        subprocess.run([cmsgen_bin, "--samples", str(n), "--samplefile", out, path],
                       capture_output=True, timeout=1200)
    except Exception:  # noqa: BLE001
        return 0.0, []
    dt = time.perf_counter() - t0
    models = []
    count = 0
    if os.path.exists(out):
        for line in open(out):
            lits = [int(x) for x in line.split() if x.lstrip("-").isdigit()]
            if lits and lits[-1] == 0:
                lits = lits[:-1]
            if not lits:
                continue
            count += 1                                   # rate uses the full count, not just kept
            if keep is None or len(models) < keep:
                models.append(lits)
        try:
            os.remove(out)                               # the samplefile can be >1 GB; don't accumulate
        except OSError:
            pass
    return (count / dt if dt else 0.0), models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="benchmarks/sampling/*.cnf")
    ap.add_argument("--device", default=None)
    ap.add_argument("--cmsgen", default=None, help="path to the cmsgen binary (enables comparison)")
    ap.add_argument("--n-time", type=int, default=50_000,
                    help="samples to PRODUCE in the timed run (a contract, not a hint). Only the "
                         "safety valves end the run short of this ('<' after sols): --time-limit, "
                         "or zero models after every config was measured (hopeless instance)")
    ap.add_argument("--time-limit", type=float, default=60.0,
                    help="per-instance sampling wall cap in seconds (the safety valve that bounds "
                         "very-low-success instances; e.g. at 1k samples/s, 200k samples would "
                         "otherwise take 200s)")
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="expert knob: hard chunk cap for the timed run (default none — --n-time "
                         "and --time-limit govern)")
    ap.add_argument("--n-cov", type=int, default=0,
                    help="samples for the QUALITY pass (2-wise coverage + distinct count). 0 (default) "
                         "SKIPS it: the collect run's Python model-reconstruction dominated instance "
                         "wall (measured 26s of 34s on s1196 T=25) while distinct≈100%% on all non-tiny "
                         "circuits. Pass e.g. 5000 for the paper-quality metrics")
    ap.add_argument("--portfolio", choices=["circuit", "default"], default="circuit",
                    help="bandit config portfolio: 'circuit' (fused/fsoft/orac — the CircuitSAT "
                         "focus, default) or 'default' (the sk_ projected-sampling portfolio)")
    ap.add_argument("--b", type=int, default=8,
                    help="decisions per round (8 is the throughput sweet spot: fewer rounds while "
                         "success barely drops; B>=16 over-constrains and collapses success)")
    ap.add_argument("--k-cap", type=int, default=49152)
    ap.add_argument("--no-simplify", dest="simplify", action="store_false", default=True)
    ap.add_argument("--els", action="store_true",
                    help="also do equivalent-literal substitution (removes vars: faster propagate + "
                         "bigger K). Sound + model-preserving; ~1.5x throughput on the niche")
    ap.add_argument("--also-plain", action="store_true", help="also run GPAP without simplification")
    ap.add_argument("--verify-cpu", action="store_true", help="also CPU-verify a few models vs clauses")
    ap.add_argument("--scratch", default="/tmp")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    files = sorted(glob.glob(args.glob))
    portfolio = CIRCUIT_PORTFOLIO if args.portfolio == "circuit" else DEFAULT_PORTFOLIO
    quality = args.n_cov > 0
    print(f"device={device} | simplify={args.simplify} | portfolio={args.portfolio} "
          f"| n_time={args.n_time:,} (limit {args.time_limit:g}s/inst) n_cov={args.n_cov:,} "
          f"| cmsgen={'yes' if args.cmsgen else 'no'}\n{len(files)} benchmarks\n")
    hdr = (f"{'instance':<24}{'N':>6}{'M':>7}{'|sup|':>6}{'GPAP/s':>10}{'e2e/s':>10}"
           f"{'dec/s':>9}{'impl/s':>9}{'succ%':>6}{'sols':>10}{'t_slv':>7}{'t_tot':>7}")
    if quality:
        hdr += f"{'uniq/s~':>10}{'2wiseG':>8}"
    hdr += f"{'snd':>4}"
    if args.cmsgen:
        hdr += f"{'CMS/s':>9}{'GPAP/CMS':>9}{'e2e/CMS':>8}"
        if quality:
            hdr += f"{'2wiseC':>7}"
    print(hdr); print("-" * len(hdr))
    results, ratios = [], []
    t_suite = time.perf_counter()
    total_sols = 0
    for path in files:
        name = os.path.basename(path)
        g = gpap_sample(path, device, simplify=args.simplify, els=args.els, n_time=args.n_time,
                        n_cov=args.n_cov, b=args.b, k_cap=args.k_cap, verify_cpu=args.verify_cpu,
                        configs=portfolio, max_chunks=args.max_chunks,
                        time_limit=args.time_limit)
        sup1 = g["sup1"]                       # metadata from gpap_sample's single parse
        twg = two_wise_coverage(g["models"], sup1) if quality else None
        snd = "OK" if g["gpu_sound"] and (g["cpu_sound"] in (None, True)) else "BAD"
        total_sols += g["n_models"]
        row = {"name": name, "nv": g["nv"], "M": g["n_clauses"], "support": len(sup1),
               "gpap_sps": g["samples_per_s"], "gpap_sps_e2e": g.get("sps_e2e_total"),
               "n_models": g["n_models"], "wall_timed_s": g["wall_timed_s"],
               "wall_total_s": g["wall_total_s"],
               "succ": g["success_rate"], "twise_gpap": twg,
               "gpu_sound": g["gpu_sound"], "n_distinct": g["n_distinct"],
               "unique_sps_est": g.get("unique_sps_est"), "assigns_per_s": g.get("assigns_per_s"),
               "decisions_per_s": g.get("decisions_per_s"),
               "implications_per_s": g.get("implications_per_s"), "mined": g.get("mined"),
               "phases": g.get("phases"), "config_breakdown": g.get("config_breakdown")}
        dps = f"{g['decisions_per_s']:.1e}" if g.get("decisions_per_s") else "—"
        ips = f"{g['implications_per_s']:.1e}" if g.get("implications_per_s") else "—"
        mi = g.get("mined")
        # '+' after succ% = entailed-clause mining ran (low-success gate); details land in the JSON.
        # '<' after sols = a safety valve (--time-limit, or zero-progress on a hopeless instance)
        # stopped the run before it delivered --n-time samples; rates are still valid.
        mn = "+" if mi else " "
        sols_s = f"{g['n_models']:,}" + ("<" if g.get("chunk_capped") else "")
        line = (f"{name[:22]:<24}{g['nv']:>6}{g['n_clauses']:>7}{len(sup1):>6}"
                f"{g['samples_per_s']:>10,.0f}{g.get('sps_e2e_total') or 0:>10,.0f}{dps:>9}{ips:>9}"
                f"{100*g['success_rate']:>5.0f}%{mn}{sols_s:>9}"
                f"{g['wall_timed_s']:>6.1f}s{g['wall_total_s']:>6.1f}s")
        if quality:
            uq = f"{g['unique_sps_est']:,.0f}" if g.get("unique_sps_est") else "—"
            line += f"{uq:>10}{twg:>7.1f}%"
        line += f"{snd:>4}"
        plain = None
        if args.also_plain:
            plain = gpap_sample(path, device, simplify=False, n_time=args.n_time, n_cov=0,
                                b=args.b, k_cap=args.k_cap, configs=portfolio,
                                max_chunks=args.max_chunks, time_limit=args.time_limit)
            row["gpap_plain_sps"] = plain["samples_per_s"]
        if args.cmsgen:
            # single CMSGen run at n_time (fair samp/s); use a subset of its samples for coverage
            crate_t, cmodels = cmsgen_sample(path, args.cmsgen, args.n_time, args.scratch,
                                             keep=args.n_cov or 1)
            ratio = (g["samples_per_s"] / crate_t) if crate_t else float("inf")
            ratio_e2e = ((g.get("sps_e2e_total") or 0) / crate_t) if crate_t else float("inf")
            row.update(cms_sps=crate_t, ratio=ratio, ratio_e2e=ratio_e2e)
            ratios.append(ratio)
            line += f"{crate_t:>9,.0f}{ratio:>8.2f}x{ratio_e2e:>7.2f}x"
            if quality:
                twc = two_wise_coverage(cmodels[:args.n_cov], sup1)
                # CMSGen distinct fraction on the same subset — the papers' metric is UNIQUE
                # solutions/s, so report the dedup factor for BOTH samplers, not just GPAP
                cd, cn = distinct_projections(cmodels[:args.n_cov], sup1)
                row.update(twise_cms=twc, cms_distinct=cd, cms_distinct_of=cn)
                line += f"{twc:>6.1f}%"
        print(line, flush=True)
        results.append(row)

    suite_wall = time.perf_counter() - t_suite
    print(f"\nTOTAL: {total_sols:,} solutions over {len(results)} instances in {suite_wall:,.1f}s "
          f"suite wall (solver e2e {sum(r['wall_total_s'] for r in results):,.1f}s, of which "
          f"timed sampling {sum(r['wall_timed_s'] for r in results):,.1f}s)")
    if ratios:
        import statistics as st
        gm = st.geometric_mean([r for r in ratios if r and r > 0 and r != float("inf")])
        e2e_rs = [r["ratio_e2e"] for r in results
                  if r.get("ratio_e2e") and 0 < r["ratio_e2e"] != float("inf")]
        gm_e2e = st.geometric_mean(e2e_rs) if e2e_rs else 0.0
        print(f"geomean GPAP/CMSGen throughput = {gm:.2f}x steady-state, {gm_e2e:.2f}x "
              f"end-to-end incl. setup+mining  (GPAP=GPU, CMSGen=single-core CPU)")
        # quality: mean 2-wise coverage delta (only when the quality pass ran)
        dq = [r["twise_gpap"] - r["twise_cms"] for r in results
              if r.get("twise_cms") is not None and r.get("twise_gpap") is not None]
        if dq:
            print(f"mean 2-wise coverage: GPAP "
                  f"{np.mean([r['twise_gpap'] for r in results if r['twise_gpap'] is not None]):.1f}% "
                  f"vs CMSGen {np.mean([r['twise_cms'] for r in results if r.get('twise_cms') is not None]):.1f}% "
                  f"(Δ {np.mean(dq):+.1f}pp)")
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        json.dump({"device": str(device), "args": vars(args), "results": results},
                  open(args.output, "w"), indent=1)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
