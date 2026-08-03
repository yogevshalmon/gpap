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
from enum import Enum
from typing import NamedTuple

import torch

from gpap.engine import resolve_k_cap
from gpap.sampling.sampler import GPUSampler, parse_independent_support

class VarOrder(int, Enum):
    """Which unassigned variable each decision picks.

    int-valued on purpose: a config IS the `(random_var, B, lookahead)` tuple the sampler has
    always consumed, and the kernel path tests it with a plain `if random_var:`. An `int` mixin
    keeps RANDOM truthy and STRUCTURED falsy, so nothing downstream has to learn about the enum.
    """

    RANDOM = 1        # uniform random free variable — CMSGen's Algorithm 1. Diverse; the default.
    STRUCTURED = 0    # first free variable in index order (= the gate DAG order on a Tseitin CNF).
    #                   Wins the search-heavy "service" family, where a random order keeps picking
    #                   variables propagation would have implied anyway.


class Lookahead(str, Enum):
    """Polarity heuristic inside the fused trajectory kernel.

    str-valued, and NONE is the EMPTY string so it stays falsy and compares unequal to "soft" —
    the sampler normalises `lookahead in ("soft", "fsoft")` and treats everything else as off.
    """

    NONE = ""         # uniform random polarity. THE UNIFORMITY-SAFE CHOICE: Barbarik Accepts B=8
    #                   and B=1 5/5 at the CMSGen paper's parameters.
    SOFT = "soft"     # ε-greedy CONFLICT-RISK polarity: prefer the value less likely to conflict.
    #                   Big success-rate win on the tight/deep circuit family (b03-c25: 10% -> 88%)
    #                   — but it BIASES the distribution by construction, and Barbarik REJECTS it
    #                   5/5 (in 156-936 samples, vs the 1,872 an Accept costs; exact-oracle TV
    #                   median 0.694, max 0.907). See docs/findings/UNIFORMITY_FINDINGS.md.
    #                   OPT-IN ONLY — never in a default portfolio.


class SamplingConfig(NamedTuple):
    """One portfolio entry: how a trajectory decides, and how many decisions it makes per
    propagation round. Subclasses tuple, so it is accepted anywhere the old
    `(random_var, B, lookahead)` triples were.

    `decisions_per_round` (B) is both a throughput and a UNIFORMITY knob. At B >= |support| one
    round assigns the whole support a uniform random polarity and, because GPAP DISCARDS
    conflicted trajectories rather than backtracking, keeping the survivors is exact rejection
    sampling — i.e. exactly uniform (measured; see UNIFORMITY_FINDINGS.md §3). Smaller B
    interleaves propagation with the decisions and weights solutions by their attraction basin.
    """

    var_order: VarOrder
    decisions_per_round: int
    lookahead: Lookahead = Lookahead.NONE

    def describe(self) -> str:
        la = "" if self.lookahead is Lookahead.NONE else f", {self.lookahead.name.lower()}-polarity"
        return f"{self.var_order.name.lower()} order, B={self.decisions_per_round}{la}"


# CIRCUIT portfolio (circuit-unroll evaluation). Measured per-config on 22+ circuit instances
# (RTX 4080, 2026-07-02..04); each entry is the outright winner somewhere and the adaptive bandit
# picks per instance. ("orac" — high-B on the retired PYTHON engine — was removed 2026-07-06 after
# 0/64 wins, and the engine path it needed was deleted 2026-07-13.)
CIRCUIT_PORTFOLIO = [
    SamplingConfig(VarOrder.RANDOM, 8),    # broad deep-circuit winner
    SamplingConfig(VarOrder.RANDOM, 1),    # B=1's success edge is free in-kernel: wins the
    #                                        conflict-heavy deep set
]

# Default portfolio for projected sampling (the sk_ / CMSGen-comparison story).
# NOTE 2026-07-13: the (RANDOM,8) and (STRUCTURED,1) entries used to run the retired PyTorch v1
# path; they now run in-kernel — sk_-suite numbers published before this date predate that switch.
DEFAULT_PORTFOLIO = [
    SamplingConfig(VarOrder.RANDOM, 8),        # the classic niche config (BCP-determined support)
    SamplingConfig(VarOrder.RANDOM, 1),        # the sk_ winner
    SamplingConfig(VarOrder.STRUCTURED, 1),    # the search-heavy "service" family
]

# OPT-IN ONLY (`--allow-soft`). Removed from CIRCUIT_PORTFOLIO 2026-07-30: the bandit selects on
# models/second with NO uniformity term, so leaving soft in the default portfolio meant any
# instance where it won silently emitted a Barbarik-REJECTED distribution. Nothing unsound — every
# row is a real solution — but it is not what someone asking for a *sampler* expects. It stays
# reachable because the regime-dependence rule stands: it is the right config for "many solutions
# fast, distribution does not matter".
SOFT_CONFIG = SamplingConfig(VarOrder.RANDOM, 8, Lookahead.SOFT)

PORTFOLIOS = {"circuit": CIRCUIT_PORTFOLIO, "default": DEFAULT_PORTFOLIO}


def portfolio(name: str = "default", allow_soft: bool = False) -> list:
    """The named portfolio, optionally with the non-uniform soft-polarity config appended."""
    if name not in PORTFOLIOS:
        raise ValueError(f"unknown portfolio {name!r} (have: {', '.join(PORTFOLIOS)})")
    return [*PORTFOLIOS[name], SOFT_CONFIG] if allow_soft else list(PORTFOLIOS[name])


def describe_portfolio(configs) -> str:
    """Human-readable one-line-per-config rendering, for CLI output and results metadata."""
    return "; ".join(c.describe() if isinstance(c, SamplingConfig)
                     else SamplingConfig(*c).describe() for c in configs)


def _core_provenance() -> str:
    """Which engine build produced a result: "libgpap <version>", or "torch-jit" when the
    GPAP_ENGINE=torch migration escape hatch is in use."""
    import os
    if os.environ.get("GPAP_ENGINE") == "torch":
        return "torch-jit"
    try:
        from gpap.engine import _core
        return f"libgpap {_core.version()}"
    except Exception:       # noqa: BLE001 — provenance must never break a run
        return "unknown"


def _uniformity_readout(smp, clauses, sup1, rate, config, max_rounds, mode="fast", target=2000):
    """The uniformity readout (OFF the timing path; see metrics.py for the statistics).

    mode="fast" (the default): collect a small sample of projections and check their REPEAT
    behaviour — no SAT solver, no enumeration, one cheap sample_bits call. Catches gross skew
    (l2_ratio) with zero setup; cannot certify uniformity or see subtle bias.
    mode="exact": additionally try to ENUMERATE the projected solution space (small formulas
    only: M <= 200k, <= 4096 projections, <= 2 s); when enumeration completes, run the real
    goodness-of-fit test (chi-square p + TV) over all |S| solutions + a soundness cross-check.

    The sample size is clamped to ~2 s of the measured steady rate either way (a 100/s stuck
    instance gets a 256-sample readout, a 1M/s instance its full target in milliseconds)."""
    import numpy as np
    from collections import Counter
    from gpap.sampling.metrics import enumerate_projections, uniformity_stats

    projs, status = (None, "not-attempted")
    if mode == "exact":
        if len(clauses) <= 200_000:
            projs, status = enumerate_projections(clauses, sup1, cap=4096, budget_s=2.0)
        else:                                      # exact REQUESTED but the formula defeats it —
            status = "skipped-large"               # the hint must not suggest exact again
    S = len(projs) if (projs is not None and status == "complete") else None
    if S is not None:
        target = min(max(target, 5 * S), 20_000)   # n >= 5*S makes the chi-square p valid
    if rate:
        target = max(256, min(target, int(rate * 2.0)))
    bits, cols = smp.sample_bits(target, config=config, max_decision_rounds=max_rounds,
                                 max_chunks=40)
    if bits is None or len(bits) == 0:
        return {"mode": "empty", "n": 0, "note": "no samples collected", "enum_status": status}
    ci = {v: i for i, v in enumerate(cols)}
    arr = np.ascontiguousarray(np.asarray(bits, dtype=np.uint8)[:, [ci[v] for v in sup1]])
    packed = np.packbits(arr, axis=1)
    counter = Counter(packed[i].tobytes() for i in range(packed.shape[0]))
    unsound = 0
    if S is not None:
        pmat = np.array(sorted(projs), dtype=np.uint8)
        pkeys = {row.tobytes() for row in np.packbits(pmat, axis=1)}
        unsound = sum(1 for k in counter if k not in pkeys)   # must be 0 (soundness cross-check)
    out = uniformity_stats(list(counter.values()), int(arr.shape[0]), S=S)
    out["enum_status"] = status
    if unsound:
        out["unsound_projections"] = unsound
        out["note"] = f"{unsound} observed projections NOT in the enumerated solution set (BUG)"
    return out


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
    from gpap.circuits.bench import parse_encoder_comment, parse_support_labels
    wire_map = parse_support_labels(path)
    # Encoder provenance gate (2026-07-10): on an ABC-encoded CNF the entailed-clause miner is
    # pure overhead — dc2/fraig already removed the backbone L1 harvests structurally (measured:
    # 0-45 units found vs 59-1007 on the same circuits' Tseitin CNFs, while costing its full 5 s
    # budget; b15-c25: 4.8 s of a 9.4 s e2e wall for 45 units on a 222k-clause formula). The gate
    # only downgrades the DEFAULT ("auto"); pass mine=True to force probing anyway.
    if kw.get("mine", "auto") == "auto" and parse_encoder_comment(path) == "abc":
        kw["mine"] = False
    return sample_clauses(cnf0.nv, cnf0.clauses, sup0, device,
                          _t_start=t_start, _phases=phases, _wire_map=wire_map, **kw)


def sample_clauses(nv, clauses, support, device, *, simplify=True, els=False, n_time=50_000,
                   n_cov=0, out_cap=0, uniformity="off", n_uni=2000, b=8, k_cap="auto",
                   max_chunks=None, time_limit=60.0, cov_chunks=4, max_rounds=300,
                   lookahead=False, verify_clauses=None, verify_cpu=False, cov_k_cap=2048,
                   random_var="auto", configs="default", mine="auto",
                   mine_gate=0.15, min_measure_chunks=3, seed=5,
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
    large auto-K on a low-RAM host (12 GB Colab) OOM-kills the session collecting a full chunk.

    `k_cap="auto"` (default) sizes the cap to THIS GPU's VRAM (engine.default_k_cap) instead of
    the historical hardcoded 49,152 — which pinned every GPU to the 12 GB laptop's batch size and
    made a bigger card report the laptop's throughput. It is an exact no-op at or below the
    reference budget; pass an int to force a fixed cap (what a cross-GPU A/B needs)."""
    if configs == "default":
        configs = DEFAULT_PORTFOLIO
    # Resolve ONCE: every GPUSampler built below (initial + post-mining rebuilds) must get the
    # same cap, or a rebuild could land on a different K than the warmed-up run it replaces.
    k_cap = resolve_k_cap(k_cap, device)
    t_start = _t_start if _t_start is not None else time.perf_counter()
    phases = _phases if _phases is not None else {}

    def _warmup(s):
        # Warm up CUDA kernels (first-chunk JIT / allocation) at a tiny K so the timed run isn't
        # skewed by one-time warmup — this is what inflated the first shallow circuit's number.
        # Returns the probe chunk's stats (a FREE gate signal for mining) and restores the RNG
        # so the timed run stays deterministic.
        # FULL round budget + one chunk per config: with the old min(10, max_rounds) cap NO
        # deep-circuit trajectory could complete in the warmup, so success_rate read 0% on
        # every c25 unroll; and measuring only configs[0] (B=8) misreads instances whose
        # success lives in another config (b03 T=25: B=8 4.5%, B=1 99%). Either way the miner's
        # "stuck" gate fired suite-wide and burned the full L2/L3 budget everywhere (77.6s of
        # the c25 set's 99s overhead). Every config is now an in-kernel chunk — cheap at any
        # round budget at the warmup's K<=1024.
        # Returns "best_success" = the best warmed config's completion rate — THE stuck signal.
        cw = list(configs) if configs else None
        # pinned_batch, not a raw `s.K = min(s.K, 1024)`: lowering K only disables the fused boost
        # when it lands BELOW the construction value, so on a sampler built with a small k_cap the
        # "cheap" warmup silently ran a full boosted chunk instead. Pinning makes the bound real
        # and restores both models on the way out.
        with s.pinned_batch(min(s.K, 1024)):
            w = s.sample(1, configs=cw, decisions_per_round=b, random_var=random_var,
                         max_decision_rounds=max_rounds, max_chunks=len(cw) if cw else 1,
                         collect=False, lookahead=lookahead)
        bd = (w.get("config_breakdown") or {}).values()
        w["best_success"] = max((c["success_rate"] for c in bd), default=w["success_rate"])
        s._gen.manual_seed(seed)
        import random as _r
        s._pyrng = _r.Random(seed)
        torch.cuda.synchronize()
        return w

    _tc = time.perf_counter()
    smp = GPUSampler.from_clauses(nv, clauses, device, simplify=simplify, els=els,
                                  support=support, k_cap=k_cap, seed=seed)
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
        if mine in ("auto", True) and smp.M >= 3000:   # True = forced (bypasses provenance gate)
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
                        support=support, k_cap=k_cap, seed=seed)
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
                            els=els, support=support, k_cap=k_cap, seed=seed)
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
    # Engine provenance for the results JSON: WHICH packed split and WHICH kernel variant actually
    # ran, and at what K. Captured here — after the (possible) post-mining rebuild, before the
    # coverage pass lowers K. This is the evidence that a headline number came from the sum-trick
    # packed kernel inside its non-overflowing envelope (see engine.packed_envelope).
    # `core` records WHICH build produced the number — results from before the C++ core
    # (2026-07-28) carry no such field, which is exactly how a pre/post-port comparison is told
    # apart. The kernels themselves are unchanged (AOT vs JIT of the same source, bit-identical
    # SASS), so the two are comparable; the field just makes the provenance explicit.
    engine_info = {"kernel": "triton-packed-sum", "mode": smp.engine.mode,
                   "shift": smp.engine.shift,
                   "split": f"{smp.engine.shift}|{32 - smp.engine.shift}",
                   # k_cap is the RESOLVED ceiling (auto -> VRAM-proportional). Without it a
                   # cross-GPU row is unreadable: a small K could mean "this GPU is small" or
                   # "the cap bound it", and those are different results.
                   # `batch` is what actually ran; K/K_fused are the two memory models behind it
                   # (see GPUSampler.batch_size) and are kept for provenance.
                   "batch": smp.batch_size,
                   "K": smp.K, "K_fused": smp.K_fused, "k_cap": k_cap,
                   "N_eff": smp.N, "M_eff": smp.M,
                   "core": _core_provenance(),
                   "max_clause_len": smp.formula.max_clause_len if smp.M else 0}
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
        # Bound host RAM: this run RECONSTRUCTS every completed row into python lists, so the batch
        # is a hard requirement, not a preference — pin_batch guarantees it, whereas the old
        # `smp.K = min(smp.K, cov_k_cap)` left the fused boost on whenever K was already <=
        # cov_k_cap (the same hole that made one test collect a 242k-row chunk at 28 GB).
        smp.pin_batch(min(smp.K, cov_k_cap))
    # Coverage / UNIQUE-solution run: collect models with the winning config and count distinct
    # support-projections. Deliberately SEPARATE from the timed run above (which is collect=False), so
    # the model reconstruction / dedup cost is NOT charged to samples/s — this is a correctness/quality
    # readout ("do we actually emit distinct solutions?"), not a throughput number.
    vc = clauses if (verify_cpu and verify_clauses is None) else verify_clauses
    c = (smp.sample(n_cov, decisions_per_round=tuned_b, random_var=tuned_rv, max_decision_rounds=max_rounds,
                    max_chunks=cov_chunks, collect=True, verify_clauses=vc,
                    lookahead=tuned_la) if n_cov else {})
    # Uniformity readout, OPT-IN (default "off": it costs an extra ~2s sampling pass even off the
    # timing path). "fast" = collision/repeat check only (no solver work, gross-skew detector).
    # "exact" adds enumeration + chi-square on small spaces (the sensitive test — subtle bias like
    # b06-c4's TV=0.17 is only visible here). See _uniformity_readout / metrics.py.
    sup1 = [v + 1 for v in (support or range(nv))]
    uni = None
    phases["uniformity"] = 0.0
    if uniformity and uniformity != "off":
        _tu = time.perf_counter()
        try:
            uni = _uniformity_readout(smp, clauses, sup1, sps, (tuned_rv, tuned_b, tuned_la),
                                      max_rounds, mode=uniformity, target=int(n_uni))
        except Exception as e:  # noqa: BLE001 — a diagnostic must never kill the run
            uni = {"mode": "error", "note": f"{type(e).__name__}: {e}"}
        phases["uniformity"] = time.perf_counter() - _tu
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
            "sup1": sup1,
            "uniformity": uni,                                # opt-in readout (see metrics.py)
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
            "engine": engine_info,                            # packed split / kernel variant / K
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
