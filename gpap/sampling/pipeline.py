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
import logging
import time
from enum import Enum
from typing import NamedTuple

import numpy as np
import torch

from gpap.engine.calibrate import resolve_k_cap
from gpap.mining import mine_entailed
from gpap.sampling.metrics import merge_sound_check

from gpap.sampling.sampler import GPUSampler

log = logging.getLogger(__name__)

__all__ = ["sample_file", "sample_clauses",
           "portfolio", "describe_portfolio", "PORTFOLIOS", "SamplingConfig", "VarOrder",
           "Lookahead", "DEFAULT_PORTFOLIO", "CIRCUIT_PORTFOLIO", "SINGLE_PORTFOLIOS",
           "SOFT_CONFIG", "soft_warning", "describe_config"]


def _support_literals(support):
    """Support variables (0-based) -> the DIMACS literals mining combines: both polarities of
    each. `mine_entailed`'s pool is literals, so that a caller can ask about one side only."""
    return None if not support else [s for v in support for s in (v + 1, -(v + 1))]


# Mining budget and per-width candidate ceilings, as the pipeline uses them. Width 1 (the
# backbone) always runs; the wide widths are gated on their candidate COUNT, which is known
# before any GPU work and is what actually predicts their cost.
#: TOTAL wall budget for mining. Both stages draw from ONE deadline, so this bounds the whole
#: phase rather than either half of it. Honored to CHUNK granularity: a chunk is the smallest unit
#: of GPU work and is never abandoned half-written, so a formula with a very large M can overshoot
#: by one chunk.
MINE_SECONDS = 5.0
#: Stage 1 — only the wide widths that are cheap for this support size.
#: ⚠️ The width-2 ceiling is a COST calibration, and the cost moved: at 80,000 it was set when a
#: probe chunk paid a full K·M state broadcast, and it TRUNCATED the support-pool sweep on any
#: instance with |sup| >~ 200 (C(2·275,2) = 150,975 on b04/b10, 244,650 on s1196) — so those
#: swept a lexicographic PREFIX, which MINING_FINDINGS notes is not a uniform sample of the
#: space. With the reset broadcast gone (CLEAN mode) the whole support pool is exhaustible for
#: free: measured on b04-c25 the uncapped sweep ran 0.40 s vs 0.46 s capped, reported
#: `exhausted` instead of a truncation, and found a clause the ceiling had cut. 1,000,000 covers
#: |sup| up to ~700 exhaustively; past that MINE_SECONDS is the real (and correct) bound.
#: ⚠️ `1: 0` (unbounded) is LOAD-BEARING, not decoration. `mine_entailed` applies a ceiling to
#: width 1 as well, and a dict with no key 1 falls through to DEFAULT_MAX_TUPLES — which would cap
#: the backbone fixpoint on the default sampling path, silently, at a fraction of ONE round on a
#: large unroll. Width 1 is cheap and is the dominant win; only `gpap mine --max-tuples` should be
#: able to bound it.
STAGE1_MAX_TUPLES = {1: 0, 2: 1_000_000, 3: 40_000}
#: Stage 2 — forced wide widths for an instance that is still stuck after the units are in.
#: A dict for the same reason as stage 1: an int applies to EVERY width, width 1 included.
STAGE2_MAX_TUPLES = {1: 0, 2: 2_000_000, 3: 2_000_000}
#: Formulas smaller than this skip mining under `mine="auto"` (`--mine on` forces past it).
#: Set from measured mining cost at small M, not from a guess: see docs/findings/MINING_FINDINGS.md.
#: ⚠️ Raising it excludes ABC-encoded CNFs, which are small by construction and where mining pays
#: MOST.
MINE_MIN_CLAUSES = 500


def _mine_score(smp, warm) -> float:
    """Models a chunk is expected to yield — batch rows x completion rate.

    THE proxy for "did mining pay?", and it costs no extra GPU work: both terms are already in
    hand. K is sized as ~budget/row_bytes with row_bytes ~ 4*M_long (the len>=3 STATE clauses;
    binaries are occ implication edges and cost no per-row state), so `K x M_long` is roughly the
    memory budget WHATEVER the formula is — a chunk therefore costs about the same wall before
    and after mining, and what actually moves is how many rows fit and how many of them complete.
    Mined clauses of width >= 3 push those two OPPOSITE ways (more state clauses => smaller K,
    stronger propagation => higher success); **mined width-1/2 clauses are K-FREE** (units fold
    into the flush, binaries into the implication list), so for them the proxy reduces to "did
    success improve" — the gate needs no retuning, `smp.K` on the rebuilt sampler already
    charges only its num_long.

    ⚠️ A PROXY, not a measurement. The warmup pins K to a small batch, so its success rate is read
    at a batch the timed run will not use; and holding chunk wall constant ignores that a longer
    trail costs more per row. It is deliberately cheap — the alternative is a second timed run,
    which costs more than the clauses are worth on the instances where this fires.
    """
    return float(smp.K) * float(warm["best_success"])


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

    NONE = ""         # uniform random polarity. THE UNIFORMITY-SAFE CHOICE: at the CMSGen
    #                   paper's Barbarik parameters, the config the bandit delivers is Accepted on
    #                   64/66 circuit instances with ZERO rejections (the 2 are orders GPAP could
    #                   not fill), matching CMSGen's 66/66 — and 6/6 vs 6/6 at barbarik's own
    #                   stricter default. docs/findings/UNIFORMITY_FINDINGS.md §4b.
    SOFT = "soft"     # ε-greedy CONFLICT-RISK polarity: prefer the value less likely to conflict.
    #                   Big success-rate win on the tight/deep circuit family (b03-c25: 10% -> 88%)
    #                   — but it BIASES the distribution by construction, and Barbarik REJECTS it
    #                   on 64 of the 65 circuit instances it can sample (exact-oracle TV median
    #                   0.694, max 0.907). See docs/findings/UNIFORMITY_FINDINGS.md §4b.
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


# CIRCUIT portfolio (circuit-unroll evaluation). Every entry is the outright winner on some
# instance, and the adaptive bandit picks per instance; configs that never won anywhere were
# removed. Per-config measurements: docs/findings/GPU_SAMPLING_FINDINGS.md.
CIRCUIT_PORTFOLIO = [
    SamplingConfig(VarOrder.RANDOM, 8),    # broad deep-circuit winner
    SamplingConfig(VarOrder.RANDOM, 1),    # B=1's success edge is free in-kernel: wins the
    #                                        conflict-heavy deep set
]

# Default portfolio for projected sampling (the sk_ / CMSGen-comparison story).
DEFAULT_PORTFOLIO = [
    SamplingConfig(VarOrder.RANDOM, 8),        # the classic niche config (BCP-determined support)
    SamplingConfig(VarOrder.RANDOM, 1),        # the sk_ winner
    SamplingConfig(VarOrder.STRUCTURED, 1),    # the search-heavy "service" family
]

# OPT-IN ONLY (`--allow-soft`), and deliberately NOT a default-portfolio member: the bandit
# selects on models/second with NO uniformity term, so leaving soft in the default portfolio meant
# any instance where it won silently emitted a Barbarik-REJECTED distribution. Nothing unsound — every
# row is a real solution — but it is not what someone asking for a *sampler* expects. It stays
# reachable because the regime-dependence rule stands: it is the right config for "many solutions
# fast, distribution does not matter".
SOFT_CONFIG = SamplingConfig(VarOrder.RANDOM, 8, Lookahead.SOFT)

# SINGLE-CONFIG portfolios — the MEASUREMENT escape hatch. A one-entry portfolio leaves the
# bandit nothing to select (`sample()` sets adaptive only when len(plan) > 1), so the run is a
# fixed algorithm rather than an adaptive one.
#
# Why this exists: the bandit optimises models/wall with no stability term, so on instances where
# two configs are close it can pick differently between machines or between repeats of the same
# command — and then a comparison silently A/Bs the bandit instead of the thing under test.
# Measured: one instance chose B=1 on one GPU and B=8 on another, which accounted for most of an
# apparent cross-GPU speedup difference; and another alternated ~18% in model count across repeated
# identical local runs. Pin a config for any cross-GPU or before/after comparison.
# Numbers: docs/findings/CROSS_GPU_FINDINGS.md.
SINGLE_PORTFOLIOS = {
    "b8": [SamplingConfig(VarOrder.RANDOM, 8)],        # throughput-leaning; over-constrains
    "b1": [SamplingConfig(VarOrder.RANDOM, 1)],        # success-leaning; the deep-circuit winner
    "s1": [SamplingConfig(VarOrder.STRUCTURED, 1)],    # gate-DAG order, the "service" family
}

PORTFOLIOS = {"circuit": CIRCUIT_PORTFOLIO, "default": DEFAULT_PORTFOLIO, **SINGLE_PORTFOLIOS}


def portfolio(name: str = "default", allow_soft: bool = False) -> list:
    """The named portfolio, optionally with the non-uniform soft-polarity config appended.

    `b8`/`b1`/`s1` are single-config portfolios: no bandit, no adaptive selection. Use one when
    the run is a MEASUREMENT and the config must not be a free variable (see SINGLE_PORTFOLIOS).
    """
    if name not in PORTFOLIOS:
        raise ValueError(f"unknown portfolio {name!r} (have: {', '.join(PORTFOLIOS)})")
    return [*PORTFOLIOS[name], SOFT_CONFIG] if allow_soft else list(PORTFOLIOS[name])


def describe_portfolio(configs) -> str:
    """Human-readable one-line-per-config rendering, for CLI output and results metadata."""
    return "; ".join(c.describe() if isinstance(c, SamplingConfig)
                     else SamplingConfig(*c).describe() for c in configs)


def soft_warning() -> str:
    """THE warning printed whenever the soft config is in a portfolio.

    One string, because both front-ends print it and the two hand-typed copies had already drifted
    apart in the commit that introduced the second. What it says is a uniformity VERDICT about
    `SOFT_CONFIG`, which lives here — so it belongs beside it, not in a print layer."""
    return ("  [warn] --allow-soft: the soft risk-polarity config is in the portfolio. It raises\n"
            "         success rate on tight circuits but is NOT uniform (Barbarik rejects it\n"
            "         on 64 of the 65 circuit instances it can sample)\n"
            "         — do not use rows produced by it for a distribution claim.")


def describe_config(cfg) -> str:
    """Render ONE config the way `describe_portfolio` renders each of a portfolio's.

    Takes the raw `(random_var, B, lookahead)` triple that crosses the results JSON (where the
    enums have decayed to bool/int/str), so the CLI cannot describe the same arm differently from
    the portfolio banner printed ten lines earlier."""
    if not cfg:
        return "?"
    la = cfg[2] if len(cfg) > 2 else ""
    return SamplingConfig(VarOrder(int(bool(cfg[0]))), int(cfg[1]),
                          Lookahead(la or "")).describe()


def _core_provenance() -> str:
    """Which engine BUILD produced a result, for the results JSON: "libgpap <build id>".

    The build id, not the version: the version has read "0.1.0" for the project's whole life
    while the engine changed underneath it, so a row stamped with it said nothing about what
    ran. Two rows that disagree now genuinely came from different libraries."""
    try:
        from gpap.engine import _core
        return f"libgpap {_core.build_id()}"
    except Exception:       # noqa: BLE001 — provenance must never break a run
        return "unknown"


def _uniformity_readout(smp, clauses, sup1, rate, config, mode="fast", target=2000):
    """The uniformity readout (OFF the timing path; see metrics.py for the statistics).

    mode="fast" (the default): collect a small sample of projections and check their REPEAT
    behaviour — no SAT solver, no enumeration, one cheap sample_bits call. Catches gross skew
    (l2_ratio) with zero setup; cannot certify uniformity or see subtle bias.
    mode="exact": additionally try to ENUMERATE the projected solution space (small formulas
    only: M <= 200k, <= 4096 projections, <= 2 s); when enumeration completes, run the real
    goodness-of-fit test (chi-square p + TV) over all |S| solutions + a soundness cross-check.

    The sample size is clamped to ~2 s of the measured steady rate either way (a 100/s stuck
    instance gets a 256-sample readout, a 1M/s instance its full target in milliseconds)."""
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
    bits, cols = smp.sample_bits(target, config=config, max_chunks=40)
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
    from gpap.engine import read_dimacs

    t_start = time.perf_counter()   # TRUE end-to-end clock: parse + construction + simplify +
    #                                 warmup + mining + the timed run
    phases = {}
    # THE single parse (the core's — clauses AND the `c ind` support in one pass). It is shared
    # by sampler construction, mining and the caller's table metadata — parsing a 1M-clause
    # unroll is seconds, so it must happen exactly once.
    nv0, cls0, sup0 = read_dimacs(path)
    phases["parse"] = time.perf_counter() - t_start
    # Wire map (var -> input@cycle) embedded by a .bench conversion — lets sample output name each
    # column by its circuit wire. Empty for a plain (non-circuit) CNF.
    from gpap.circuits.bench import parse_support_labels
    wire_map = parse_support_labels(path)
    # There is deliberately NO encoder-provenance gate here. One used to turn mining off on any
    # CNF carrying `c encoder abc`, reasoning that dc2/fraig has already removed the backbone —
    # but the unit COUNT it keyed on never measured the sampling rate, and mining is in fact worth
    # MORE on ABC-encoded CNFs than on Tseitin ones. MINE_SECONDS bounds the cost instead, and the
    # keep-if-it-helps guard bounds the risk. Measurements: docs/findings/MINING_FINDINGS.md.
    return sample_clauses(nv0, cls0, sup0, device,
                          _t_start=t_start, _phases=phases, _wire_map=wire_map, **kw)


def sample_clauses(nv, clauses, support, device, *, simplify=True, els=True, n_time=50_000,
                   n_cov=0, out_cap=0, uniformity="off", n_uni=2000, b=8, k_cap="auto",
                   max_chunks=None, time_limit=60.0, cov_chunks=4,
                   lookahead=False, verify_clauses=None, verify_cpu=False, cov_k_cap=2048,
                   verify_sound=False,
                   random_var=True, configs="default", mine="auto",
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
    run the single fixed (`random_var`, `b`) config with no bandit — the other way, besides a
    one-entry portfolio, to pin the algorithm. The dominant config is REUSED for the coverage run.

    The coverage run COLLECTS models to host RAM (one chunk = K models × N ints reconstructed),
    so its K is capped to `cov_k_cap` independently of the (full-K) timing run — otherwise a
    large auto-K on a low-RAM host (12 GB Colab) OOM-kills the session collecting a full chunk.

    `k_cap="auto"` (default) uses the library's batch cap (engine.default_k_cap), which is the
    SAME number on every GPU so two machines' rows are comparable without pinning an override;
    pass an int to force a fixed cap.

    `verify_sound` (default OFF) re-checks on the GPU that every emitted model satisfies every
    clause, reporting `sound_check` ("ok" / "UNSOUND" / "off") and `unsound_rows`. Opt-in
    because it is a self-test rather than part of sampling; its wall is excluded from every
    rate, and it grows with the formula. `gpap bench` turns it on by default — those rows get
    published, and a soundness column that was never computed is not evidence."""
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
                         max_chunks=len(cw) if cw else 1,
                         collect=False, lookahead=lookahead)
        bd = (w.get("config_breakdown") or {}).values()
        w["best_success"] = max((c["success_rate"] for c in bd), default=w["success_rate"])
        s.reseed(seed)          # BOTH streams — sampling and the delivery projection
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
        # Entailed-clause mining (gpap.mining, an application of the engine's tuple-probe
        # primitive). Every learned clause is ENTAILED, so the model set is exactly unchanged and
        # mining can only make propagation stronger — but NOT necessarily the sampler faster, which
        # is what KEEP_IF_IT_HELPS below is about. Skipped on tiny formulas (MINE_MIN_CLAUSES);
        # `mine=True` ("--mine on") forces past that gate. Effect: docs/findings/MINING_FINDINGS.md.
        if mine is True or (mine == "auto" and smp.M >= MINE_MIN_CLAUSES):
            try:
                # Two stages. Stage 1 mines the backbone (width 1, always — the cheap fixpoint
                # and the dominant win) plus the wider widths that are AFFORDABLE for this
                # support size. Stage 2 forces the wide widths, and only if the instance is
                # still stuck AFTER the units are in: pre-mining success is low almost
                # everywhere because mining is exactly what lifts it, so gating the wide widths
                # on the PRE-mining rate burned the budget suite-wide.
                _tm = time.perf_counter()
                score_before = _mine_score(smp, w)       # what we have to beat to keep the clauses
                mine_deadline = _tm + MINE_SECONDS      # ONE budget, shared by both stages
                mi = mine_entailed(nv, clauses, device, pool=_support_literals(support),
                                   max_tuples=STAGE1_MAX_TUPLES, seconds=MINE_SECONDS)
                phases["mine"] = time.perf_counter() - _tm
                learned = list(mi.clauses)
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
                    warm = _warmup(smp)                 # post-units: the stuck signal AND the score
                    stuck = warm["best_success"] < mine_gate
                    phases["recheck"] = time.perf_counter() - _tm
                left = mine_deadline - time.perf_counter()
                if stuck and left > 0.25:               # no floor: an exhausted budget stays spent
                    _tm = time.perf_counter()
                    mi2 = mine_entailed(nv, clauses + learned, device, pool=_support_literals(support),
                                        max_tuples=STAGE2_MAX_TUPLES, seconds=left)
                    phases["mine"] += time.perf_counter() - _tm
                    if mi2.clauses:
                        learned = learned + list(mi2.clauses)
                        _tm = time.perf_counter()
                        smp = GPUSampler.from_clauses(
                            nv, clauses + learned, device, simplify=simplify,
                            els=els, support=support, k_cap=k_cap, seed=seed)
                        phases["rebuild"] = (phases.get("rebuild", 0.0)
                                             + time.perf_counter() - _tm)
                        _tm = time.perf_counter()
                        warm = _warmup(smp)             # re-score: stage 2 changed the formula
                        phases["recheck"] = phases.get("recheck", 0.0) + time.perf_counter() - _tm
                    for width, n in mi2.by_width.items():
                        mi.by_width[width] = mi.by_width.get(width, 0) + n
                    mi.seconds += mi2.seconds
                # KEEP THE CLAUSES ONLY IF THEY HELP. Mined clauses are always
                # sound, but width>=3 ones enlarge num_long, and K is sized as budget/row_bytes
                # with row_bytes ~ 4*M_long — so a bigger STATE formula buys a SMALLER batch
                # (width-1/2 finds are K-free). Since K x M_long is then roughly constant, a
                # chunk costs about the same either way and models-per-chunk `K x success` is
                # the throughput proxy, readable with no extra GPU work. Compare
                # it against the pre-mining sampler and revert when mining did not pay. This is
                # what turns "mining helps on average" into "mining never hurts": on an
                # instance that already completes there is no success left to win, and the batch
                # loss is all that happens.
                score_after = _mine_score(smp, warm if learned else w)
                # `score_before > 0` guard: when the un-mined sampler completed NOTHING in the
                # warmup there is no baseline to be worse than, and mining is the only
                # thing that has ever converted those instances. Keep the clauses.
                if learned and score_before > 0 and score_after <= score_before:
                    _tm = time.perf_counter()
                    smp = GPUSampler.from_clauses(nv, clauses, device, simplify=simplify, els=els,
                                                  support=support, k_cap=k_cap, seed=seed)
                    phases["rebuild"] = phases.get("rebuild", 0.0) + time.perf_counter() - _tm
                    # Re-warm. EVERY other route into the timed run has been through `_warmup`
                    # (the no-mining path warms at the top, the keep path warms in the recheck),
                    # and a reverted run must not be the one exception — otherwise the instances
                    # this guard fires on are also the only ones paying first-touch cost inside
                    # the measurement, which would show up as a regression caused by the guard
                    # rather than by the clauses.
                    _tm = time.perf_counter()
                    _warmup(smp)
                    phases["recheck"] = phases.get("recheck", 0.0) + time.perf_counter() - _tm
                    log.info("mining reverted: %d clauses did not pay (%.2fx projected)",
                             len(learned), score_after / score_before if score_before else 0.0)
                    learned = []
                mine_info = {"n_units": mi.by_width.get(1, 0), "n_binary": mi.by_width.get(2, 0),
                             "n_ternary": mi.by_width.get(3, 0), "seconds": mi.seconds,
                             "kept": bool(learned),
                             "gain": (score_after / score_before) if score_before else None}
            except RuntimeError as e:   # mining is an optimization — never let it kill the run
                log.warning("mining skipped after a device error: %s", e)
    t0 = time.perf_counter()
    # A big-K GPU meets n_time in the bandit cold-start (1 chunk/config), so the headline rates would
    # rest on a single noisy chunk (measured: best_rate_sps ±1.9x run-to-run). min_measure_chunks makes
    # the winner run >= that many back-to-back chunks post-target for a stable, config-consistent rate.
    mmc = max(1, min(min_measure_chunks,
                     (max_chunks - (len(configs) if configs else 1)) if max_chunks
                     else min_measure_chunks))
    # out_cap rides ALONG with the timed run (sampler.sample) instead of re-running the sampler
    # afterwards: the rows are projected off chunks that execute either way.
    if configs:
        t = smp.sample(n_time, configs=configs,
                       max_chunks=max_chunks, collect=False, lookahead=lookahead,
                       min_measure_chunks=mmc, max_seconds=time_limit, out_cap=out_cap,
                       verify_sound=verify_sound)
    else:
        t = smp.sample(n_time, decisions_per_round=b, random_var=random_var,
                       max_chunks=max_chunks, collect=False,
                       lookahead=lookahead, min_measure_chunks=mmc, max_seconds=time_limit,
                       out_cap=out_cap, verify_sound=verify_sound)
    if device.type == "cuda":
        torch.cuda.synchronize()
    # The projection that fills `--out` runs inside the loop above but is DELIVERY, not sampling:
    # bill it to phases["output"] and keep it out of `wall`, so every rate below is the same
    # number the run would have reported without `--out`. (The sampler's own `cfg_stats` wall
    # already excludes it, but only on the adaptive path — a pinned single config has none.)
    proj_s = float(t.get("out_project_s") or 0.0)
    # The soundness re-check is a SELF-TEST, not sampling, and its contract is that turning it on
    # cannot move a published rate. Same shape as `proj_s` above and the same reason it has to be
    # subtracted HERE: the sampler's per-config wall already excludes it, but that stamp is only
    # taken on the adaptive path, so a PINNED single config -- the mode every measurement uses --
    # falls through to `n_models / wall` and would carry the check.
    verify_s = float(t.get("verify_sound_s") or 0.0)
    wall = time.perf_counter() - t0 - proj_s - verify_s
    # Engine provenance for the results JSON: WHICH packed split and WHICH kernel variant actually
    # ran, and at what K. Captured here — after the (possible) post-mining rebuild, before the
    # coverage pass lowers K. This is the evidence that a headline number came from the sum-trick
    # packed kernel inside its non-overflowing envelope (see engine.packed_envelope).
    # `core` records WHICH build produced the number, so rows from different engine builds are
    # distinguishable in a results file rather than silently comparable.
    engine_info = {"kernel": "triton-packed-sum", "mode": smp.engine.mode,
                   "shift": smp.engine.shift,
                   "split": f"{smp.engine.shift}|{32 - smp.engine.shift}",
                   # k_cap is the RESOLVED ceiling (auto -> the library cap). Without it a
                   # cross-GPU row is unreadable: a small K could mean "this GPU is small" or
                   # "the cap bound it", and those are different results.
                   # `batch` is what actually ran; `K` is the calibrated value behind it
                   # (see GPUSampler.batch_size) and are kept for provenance.
                   "batch": smp.batch_size,
                   "K": smp.K, "k_cap": k_cap,
                   # M_eff = ALL clauses; m_long = the len>=3 STATE clauses the K model charges
                   # (binaries are occ implication edges, units live in the flush bitmap).
                   "N_eff": smp.N, "M_eff": smp.M, "m_long": smp.M_state,
                   "core": _core_provenance(),
                   "max_clause_len": smp.formula.max_clause_len if smp.M else 0}
    # Headline throughput = the best-THROUGHPUT config's steady-state rate (best_rate_sps), NOT the
    # end-to-end blend. With the adaptive bandit the run measures every config once (see sampler.py);
    # on a big-K GPU a single chunk exceeds n_time, so the blend would report whatever config ran
    # first and be dragged by slower exploration chunks. best_rate_sps is the rate a real run achieves
    # after committing to the winner (construction/warmup amortised). Falls back to the blend if a
    # single non-adaptive config was used (no per-config timing).
    sps_e2e = t["n_models"] / wall if wall > 0 else 0.0
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
    # Sample OUTPUT: up to `out_cap` support-projection bit-rows from the WINNING config — only
    # the (n, S) bit matrix ever crosses to host, never full N-var models.
    # The timed run above already kept them (`out_cap=` -> `out_bits`), so the common case costs
    # NOTHING here. It used to be a second full sampling pass over the same instance, which (a)
    # roughly doubled the wall, (b) obeyed no time limit, and (c) delivered `out_cap` rows even
    # when the timed run had been CUT SHORT — so a capped run reported one model count and then
    # wrote a larger number of samples. docs/findings/CLI_DELIVERY_FINDINGS.md.
    # A top-up runs only when the collection came up short of the request, and it is bounded by
    # what is LEFT of the sampling budget: a run that already spent `time_limit` writes what it
    # has (short, and said so) rather than silently sampling for another `time_limit`.
    # Always WELL-SHAPED: with out_cap=0, or when nothing was found (UNSAT, 0%-success), this is
    # a (0, S) matrix over the same columns rather than None, so a caller can index it
    # unconditionally. `projection_cols` is the shape alone — the per-column plan is only built
    # when there are rows to fill.
    cols = [v + 1 for v in smp.projection_cols()]
    samples = np.zeros((0, len(cols)), dtype=np.uint8)
    phases["output"] = proj_s
    if out_cap:
        samples = t.get("out_bits", samples)
        cols = t.get("out_cols", cols)
        # A file must never claim more than the run produced. The ceiling is what the timed run
        # DELIVERED, not what was requested: a run cut short by a safety valve now writes what it
        # has, instead of quietly sampling the rest afterwards and reporting the two numbers side
        # by side. Bounded by the remaining sampling budget as well, so a spent `--time-limit`
        # buys no extra sampling at all.
        want = min(out_cap, t["n_models"])
        left = None if time_limit is None else max(0.0, time_limit - wall)
        if len(samples) < want and (left is None or left > 0):
            _to = time.perf_counter()
            extra, cols = smp.sample_bits(want - len(samples),
                                          config=(tuned_rv, tuned_b, tuned_la),
                                          max_seconds=left)
            if len(extra):
                samples = np.concatenate([samples, extra]) if len(samples) else extra
            phases["output"] += time.perf_counter() - _to
    # AFTER the output block: these rows are part of what the run DELIVERS, so the end-to-end wall
    # has to carry their cost. Charging it to `phases` while leaving it out of `wall_total_s` made
    # the two disagree by exactly the output pass, and inflated the reported e2e rate.
    # ...but the soundness re-check is NOT delivery: it is an opt-in SELF-TEST, so it comes out
    # of the e2e wall the same way it comes out of `wall` above. Leaving it in broke the
    # `sum(phases) == wall_total_s` invariant by exactly `verify_sound_s` (nothing billed it to a
    # phase) AND charged the check to `sps_e2e_total` -- the column paired against CMSGen's
    # subprocess wall -- so a campaign with VERIFY_SOUND=1 published a rate no un-verified run
    # would have reported. It is billed to `phases_diag` below, beside the other opt-in readouts.
    wall_total = time.perf_counter() - t_start - verify_s
    sps_e2e_total = t["n_models"] / wall_total if wall_total > 0 else 0.0
    # Everything past this point is a DIAGNOSTIC the caller opted into (--n-cov, --uniformity),
    # not part of delivering samples. Its wall is reported separately, in `phases_diag`, because
    # `phases` has to sum to `wall_total_s` and `sps_e2e_total` must not be dragged down by a
    # readout the user asked for on the side. Charging uniformity to `phases` alone left the two
    # disagreeing by exactly its cost.
    phases_diag: dict = {}
    # The timed run's soundness re-check (the coverage run's is already inside "coverage").
    phases_diag["verify_sound"] = verify_s
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
    _tcov = time.perf_counter()
    c = (smp.sample(n_cov, decisions_per_round=tuned_b, random_var=tuned_rv,
                    max_chunks=cov_chunks, collect=True, verify_clauses=vc,
                    lookahead=tuned_la, verify_sound=verify_sound) if n_cov else {})
    phases_diag["coverage"] = (time.perf_counter() - _tcov) if n_cov else 0.0
    # Uniformity readout, OPT-IN (default "off": it costs an extra ~2s sampling pass even off the
    # timing path). "fast" = collision/repeat check only (no solver work, gross-skew detector).
    # "exact" adds enumeration + chi-square on small spaces (the sensitive test — subtle bias like
    # b06-c4's TV=0.17 is only visible here). See _uniformity_readout / metrics.py.
    sup1 = [v + 1 for v in (support or range(nv))]
    uni = None
    phases_diag["uniformity"] = 0.0
    if uniformity and uniformity != "off":
        _tu = time.perf_counter()
        try:
            uni = _uniformity_readout(smp, clauses, sup1, sps, (tuned_rv, tuned_b, tuned_la),
                                      mode=uniformity, target=int(n_uni))
        except Exception as e:  # noqa: BLE001 — a diagnostic must never kill the run
            uni = {"mode": "error", "note": f"{type(e).__name__}: {e}"}
        phases_diag["uniformity"] = time.perf_counter() - _tu
    # Was the formula PROVED to have no models? Both detectors are already in hand and neither
    # reached the caller before: preprocessing returning None (`_unsat_sampler`), and the engine's
    # construction-time units flush conflicting. Without this an UNSAT instance is reported as
    # "0 models, success 0%, sound=True" — indistinguishable from a satisfiable instance the
    # sampler merely failed on, which are opposite facts.
    unsat = bool((getattr(smp, "_recon", None) or {}).get("unsat")) or bool(
        smp.engine.flush_conflict)
    del smp
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # Estimated UNIQUE solutions/s: timed raw rate × the distinct fraction measured post-hoc on the
    # coverage sample. Dedup/reconstruction NEVER touches the timed run (collect=False) — this is a
    # pure post-processing estimate (exact on large spaces where distinct_frac≈1; honest on tiny
    # spaces where re-sampling dominates). None when it could not be measured — `unique_est_status`
    # says which reason, and the two are NOT interchangeable (see below).
    n_coll = len(c.get("models", []))
    uniq_sps = (sps * c["n_distinct"] / n_coll) if (n_coll and c.get("n_distinct")) else None
    # WHY it is None is not a detail: "no coverage run was requested" and "the coverage run
    # collected nothing" are different facts, and a consumer that folds both to 0.0 publishes the
    # second as "GPAP produced nothing". The coverage run is a few thousand trajectories, so on a
    # very-low-success instance it comes back empty while the TIMED run delivers models — b11-c25
    # at a rare pin was recorded that way and scored as an outright loss to a rival it beat.
    uniq_status = ("ok" if uniq_sps is not None
                   else ("not-requested" if not n_cov else "no-coverage-sample"))
    phases["timed"] = wall
    return {"samples_per_s": sps, "success_rate": succ,
            "nv": nv, "n_clauses": len(clauses),              # ORIGINAL sizes (pre-simplify)
            "sup1": sup1,
            "uniformity": uni,                                # opt-in readout (see metrics.py)
            "sps_e2e_total": sps_e2e_total,                   # incl. construction+simplify+mining
            "phases": phases,                                 # DELIVERY phases; these sum to wall_total_s
            "phases_diag": phases_diag,                       # opt-in readouts, OUTSIDE wall_total_s
            "n_models": t["n_models"],                        # total solutions found in the timed run
            # True when a safety valve (time_limit / zero-progress / explicit max_chunks) stopped
            # the timed run BEFORE it delivered n_time models. Rates stay valid either way.
            "chunk_capped": t["n_models"] < n_time,
            "wall_timed_s": wall,                             # the solver's sampling wall (timed run)
            "wall_total_s": wall_total,                       # e2e wall incl. construction+simplify+mining
            "mined": mine_info,                               # entailed-clause mining stats (None = skipped)
            "unique_sps_est": uniq_sps,
            "unique_est_status": uniq_status,                 # ok | not-requested | no-coverage-sample
            "assigns_per_s": aps,                             # literals assigned/s (CDCL props/s analogue)
            "decisions_per_s": dps,                           # branch decisions/s (best-rate config)
            "implications_per_s": ips,                        # unit-prop implications/s (=assigns-decisions)
            "engine": engine_info,                            # packed split / kernel variant / K
            "attempts_per_s": t.get("attempts_per_s"),        # raw trajectory throughput (K/chunk rate)
            "bcp_rounds_per_chunk": t.get("bcp_rounds_per_chunk"),
            "bcp_ops_per_attempt": t.get("bcp_ops_per_attempt"),
            "support_size": t["support_size"],
            # `sound_check` is OPT-IN (`verify_sound`), so its third value is "off" — a run that
            # did not check is NOT a run that failed, and a consumer folding the two into a bool
            # publishes the first as the second.
            # UNSOUND DOMINATES, over both runs. Preferring the timed run's verdict silently
            # discarded the coverage run's, and since `unsound_rows` sums BOTH, the row could say
            # `sound_check: "ok"` beside `unsound_rows: 7` — and every consumer branches on the
            # string. The coverage run is not a lesser witness: it samples the same formula under
            # a DIFFERENT config and batch, which is exactly where a config-specific bug shows.
            "sound_check": merge_sound_check(t.get("sound_check"), c.get("sound_check")),
            "sound_rows_checked": (((t.get("sound_rows_checked") or 0)
                                    + (c.get("sound_rows_checked") or 0))
                                   if verify_sound else None),
            "unsound_rows": ((t.get("unsound_rows") or 0) + (c.get("unsound_rows") or 0)
                             if verify_sound else None),
            "verify_sound_s": (float(t.get("verify_sound_s") or 0.0)
                               + float(c.get("verify_sound_s") or 0.0)) if verify_sound else None,
            "cpu_sound": c.get("all_valid"), "models": c.get("models", []),
            "n_distinct": c.get("n_distinct"), "n_collected": n_coll,
            # Sample OUTPUT (present when out_cap>0): `samples` = numpy uint8 (n, S) 0/1 matrix of
            # support projections; `cols` = the ORIGINAL variable id of each column; `wire_map` =
            # {var: input@cycle} for a circuit CNF (empty otherwise). See gpap.sampling.output.
            "samples": samples, "cols": cols, "wire_map": _wire_map or {},
            # How many rows were ASKED for vs kept, so a short file is self-explaining, and which
            # config produced them (they all come from one — see sampler.sample's out_cap).
            "out_requested": out_cap, "out_config": t.get("out_config"),
            "unsat": unsat,                                   # PROVED no models (not merely none found)
            "tuned_rv": tuned_rv, "tuned_b": tuned_b, "tuned_la": tuned_la,
            "config_breakdown": t.get("config_breakdown")}
