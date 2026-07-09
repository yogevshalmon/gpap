"""The end-to-end GPAP sampling pipeline — the flow every consumer shares (Python API, CLI,
benchmark script, Colab driver):

    parse → construct (simplify/ELS) → warmup → two-stage entailed-clause mining →
    contract-timed run → optional quality/coverage pass

`n_time` is a CONTRACT: the timed run keeps sampling until it has delivered that many models.
Safety valves (both reported via `chunk_capped` and the CLI's '<' marker): `time_limit` seconds
of sampling wall, and the sampler's zero-progress stop (0 models after every config is
measured). The headline `samples_per_s` is the best-measured config's steady-state rate
(`best_rate_sps`); `sps_e2e_total` is the honest end-to-end rate including construction,
simplification and mining (what a CMSGen comparison must charge).
"""
import sys
import time

import torch

from gpap.sampling.sampler import GPUSampler, parse_independent_support

# CIRCUIT portfolio (the default for circuit-unroll evaluation). Measured per-config on 22+
# circuit instances (RTX 4080, 2026-07-02..04); each entry is the outright winner somewhere,
# the adaptive bandit picks per instance:
#   (True, 8, "fused")  — fused-trajectory kernel at boosted K_fused: broad deep-circuit winner.
#   (True, 1, "fused")  — B=1's success edge is free in-kernel: wins the conflict-heavy deep set.
#   (True, 8, "fsoft")  — in-kernel ε-greedy soft risk-polarity: wins the tight-cascade family.
# (True, 64, "orac") — high-B on the PYTHON engine — was REMOVED 2026-07-06: measured 0/64 wins
# on the full circuits dir while its cold-start chunks cost ~67s across the suite; its old niche
# (ultra-shallow wide, b14 T=1) is now >3x behind the boosted-K fused configs. Re-audition on
# hardware that shifts the balance (e.g. A100 large-K PyTorch batching) — the "orac" lookahead
# mode itself remains fully supported.
CIRCUIT_PORTFOLIO = [(True, 8, "fused"), (True, 1, "fused"), (True, 8, "fsoft")]

# Default portfolio for projected sampling (the sk_ / CMSGen-comparison story):
#   (True, 8, False)   — the classic niche config (BCP-determined support; fast tensor path).
#   (True, 1, "fused") — v2 fused-trajectory kernel at boosted K_fused: the sk_ winner
#                        (B=1's success edge is free in-kernel). Degrades to (True,1) if the
#                        kernel is unavailable.
#   (False, 1, False)  — structured/index order for the search-heavy "service" family.
DEFAULT_PORTFOLIO = [(True, 8, False), (True, 1, "fused"), (False, 1, False)]


def sample_file(path, device, **kw):
    """Run the full sampling pipeline on a DIMACS CNF file (support read from `c ind` lines).
    Returns the same rich dict as sample_clauses; the parse is charged to phases["parse"] and
    to the end-to-end wall."""
    from pysat.formula import CNF

    t_start = time.perf_counter()   # TRUE end-to-end clock: parse + construction + simplify +
    #                                 warmup + mining + the timed run
    phases = {}
    # THE single parse. It is shared by sampler construction, mining and the caller's table
    # metadata — parsing a 1M-clause unroll costs ~2s, so it must happen exactly once.
    cnf0 = CNF(from_file=path)
    sup0 = parse_independent_support(path)
    phases["parse"] = time.perf_counter() - t_start
    # Wire map (var -> input@cycle) embedded by a .bench conversion — lets sample output name each
    # column by its circuit wire. Empty for a plain (non-circuit) CNF.
    from gpap.circuits.bench import parse_support_labels
    wire_map = parse_support_labels(path)
    return sample_clauses(cnf0.nv, cnf0.clauses, sup0, device,
                          _t_start=t_start, _phases=phases, _wire_map=wire_map, **kw)


def sample_clauses(nv, clauses, support, device, *, simplify=True, els=False, n_time=50_000,
                   n_cov=0, out_cap=0, b=8, k_cap=49152, max_chunks=None, time_limit=60.0,
                   cov_chunks=4, max_rounds=300, lookahead=False, verify_clauses=None,
                   verify_cpu=False, cov_k_cap=2048, random_var="auto", configs="default",
                   use_kernel="auto", mine="auto", mine_gate=0.15, min_measure_chunks=3, seed=5,
                   _t_start=None, _phases=None, _wire_map=None):
    """Throughput run timed end-to-end (construction + simplify + sample, to match CMSGen's
    'produce N samples' wall clock) + a small collect run for coverage/soundness. `n_time` is a
    CONTRACT: the timed run keeps sampling until it has delivered that many models. Safety
    valves (both reported via `chunk_capped`): `time_limit` seconds of sampling wall, and the
    sampler's zero-progress stop (0 models after every config is measured — a 0%-success
    instance would otherwise spin forever). `max_chunks` (default None = unbounded) remains as
    an expert knob.

    `configs` (default = the adaptive-bandit DEFAULT_PORTFOLIO; pass CIRCUIT_PORTFOLIO for
    circuit unrolls): a throughput bandit that runs the best config per chunk at full K,
    dynamically switching to the winner — cracks the search-heavy "service" family (structured
    order + B=1) AND keeps the niche fast (random order + B=8) in one run. Pass configs=None to
    fall back to `random_var` (e.g. "auto"). The dominant config is REUSED for the coverage run.

    The coverage run COLLECTS models to host RAM (one chunk = K models × N ints reconstructed),
    so its K is capped to `cov_k_cap` independently of the (full-K) timing run — otherwise a
    large auto-K on a low-RAM host (12 GB Colab) OOM-kills the session collecting a full chunk."""
    if configs == "default":
        configs = DEFAULT_PORTFOLIO
    t_start = _t_start if _t_start is not None else time.perf_counter()
    phases = _phases if _phases is not None else {}

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
        s._gen.manual_seed(seed)
        import random as _r
        s._pyrng = _r.Random(seed)
        torch.cuda.synchronize()
        return w

    _tc = time.perf_counter()
    smp = GPUSampler.from_clauses(nv, clauses, device, simplify=simplify, els=els,
                                  support=support, k_cap=k_cap, seed=seed, use_kernel=use_kernel)
    phases["construct"] = time.perf_counter() - _tc
    mine_info = None
    if device.type == "cuda":
        _tw = time.perf_counter()
        w = _warmup(smp)
        phases["warmup"] = time.perf_counter() - _tw
        # Entailed-clause mining (learned clauses are ENTAILED — the model set is exactly
        # unchanged; measured: service family 0.02-0.7x -> 1.1-6.7x vs CMSGen, circuit backbone
        # instances 1.2-16x). Mining is nearly self-gating (see gpap/sampling/miner.py): the L1
        # backbone sweep costs 0.1-0.5s when there is nothing to learn, and the expensive L2/L3
        # support-tuple tiers only run when L1 found units — so mine on every non-tiny formula
        # (M >= 3000; below that the whole timed run is ~instant and even 0.3s of probing would
        # dominate the e2e wall). The warmup success rate (< mine_gate = truly stuck) FORCES the
        # full L2/L3 tiers even without L1 yield — the original service-family behaviour.
        if mine == "auto" and smp.M >= 3000:
            try:
                from gpap.sampling.miner import mine_entailed
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
                mi = mine_entailed(nv, clauses, support, device, pairs="auto",
                                   pairs_min_units=0)
                phases["mine"] = time.perf_counter() - _tm
                learned = mi["learned"]
                stuck = w["best_success"] < mine_gate
                if learned:
                    # Rebuild on the augmented formula. No trailing re-warmup needed: kernels
                    # are JIT'd and the fresh-seeded sampler starts the timed run from the same
                    # RNG state the warmup+reseed dance produced.
                    _tm = time.perf_counter()
                    smp = GPUSampler.from_clauses(
                        nv, clauses + learned, device, simplify=simplify, els=els,
                        support=support, k_cap=k_cap, seed=seed, use_kernel=use_kernel)
                    phases["rebuild"] = time.perf_counter() - _tm
                    _tm = time.perf_counter()
                    stuck = _warmup(smp)["best_success"] < mine_gate   # post-units stuck signal
                    phases["recheck"] = time.perf_counter() - _tm
                if stuck:
                    _tm = time.perf_counter()
                    mi2 = mine_entailed(nv, clauses + learned, support, device,
                                        pairs=True, pairs_min_units=0,
                                        max_seconds=max(1.0, 5.0 - mi["seconds"]))
                    phases["mine"] += time.perf_counter() - _tm
                    if mi2["learned"]:
                        learned = learned + mi2["learned"]
                        _tm = time.perf_counter()
                        smp = GPUSampler.from_clauses(
                            nv, clauses + learned, device, simplify=simplify,
                            els=els, support=support, k_cap=k_cap, seed=seed,
                            use_kernel=use_kernel)
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
    # otherwise the blend is dragged by the bandit's slow exploration chunks.
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
    # Sample OUTPUT: collect up to `out_cap` support-projection bit-rows with the winning config —
    # efficient GPU projection (only the (n, S) bit matrix crosses to host, not full N-var models).
    # Off the timing path (samples/s already measured). Kept BEFORE the coverage run's K-lowering so
    # it runs at full throughput K.
    samples = cols = None
    phases["output"] = 0.0
    if out_cap:
        _to = time.perf_counter()
        samples, cols = smp.sample_bits(out_cap, config=(tuned_rv, tuned_b, tuned_la),
                                        max_decision_rounds=max_rounds)
        phases["output"] = time.perf_counter() - _to
    if n_cov:
        smp.K = min(smp.K, cov_k_cap)                # bound host RAM from collected/reconstructed models
    # Coverage / UNIQUE-solution run: collect models with the winning config and count distinct
    # support-projections. Deliberately SEPARATE from the timed run above (which is collect=False), so
    # the model reconstruction / dedup cost is NOT charged to samples/s — this is a correctness/quality
    # readout ("do we actually emit distinct solutions?"), not a throughput number.
    vc = clauses if (verify_cpu and verify_clauses is None) else verify_clauses
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
            "nv": nv, "n_clauses": len(clauses),              # ORIGINAL sizes (pre-simplify)
            "sup1": [v + 1 for v in (support or range(nv))],
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
            # Sample OUTPUT (present when out_cap>0): `samples` = numpy uint8 (n, S) 0/1 matrix of
            # support projections; `cols` = the ORIGINAL variable id of each column; `wire_map` =
            # {var: input@cycle} for a circuit CNF (empty otherwise). See gpap.sampling.output.
            "samples": samples, "cols": cols, "wire_map": _wire_map or {},
            "tuned_rv": tuned_rv, "tuned_b": tuned_b, "tuned_la": tuned_la,
            "config_breakdown": t.get("config_breakdown")}
