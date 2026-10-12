"""The end-to-end GPAP sampling pipeline shared by the Python API, the CLI, the benchmark
driver and the Colab driver:

    parse -> construct (simplify/ELS) -> warmup -> two-stage entailed-clause mining ->
    contract-timed run -> optional quality/coverage pass

`n_time` is a contract: the timed run samples until it has delivered that many models. The
safety valves (`time_limit` seconds of sampling wall, and the sampler's zero-progress stop)
are reported via `chunk_capped` and the CLI's '<' marker (never set on a proved-UNSAT run,
which delivered everything there was). `samples_per_s` is the best-measured
config's steady-state rate (`best_rate_sps`); `sps_e2e_total` is the end-to-end rate including
construction, simplification and mining, which is what a CMSGen comparison must charge.

This module re-exports `sound_verdict` / `merge_sound_check`, the tri-state readers for the
`sound_check` key it writes; every consumer imports them from here, and they are DEFINED in
`gpap/verdict.py` so that a table generator can read a recorded row without a CUDA wheel.
"""
import logging
import os
import time
from enum import Enum
from typing import NamedTuple

import numpy as np
import torch

from gpap.engine.calibrate import resolve_k_cap
from gpap.mining import mine_entailed
from gpap.sampling.sampler import GPUSampler
# THE soundness reader and fold, re-exported: they live in a stdlib-only leaf module so that a
# table generator on a GPU-less box can read a recorded row without importing the sampler, while
# every consumer keeps importing them from here (`__all__` below is the contract).
from gpap.verdict import (SOUND_BAD, SOUND_OFF, SOUND_OK,  # noqa: F401
                          merge_sound_check, sound_verdict)

log = logging.getLogger(__name__)

__all__ = ["sample_file", "sample_clauses",
           "portfolio", "describe_portfolio", "PORTFOLIOS", "SamplingConfig", "VarOrder",
           "Lookahead", "DEFAULT_PORTFOLIO", "CIRCUIT_PORTFOLIO", "SINGLE_PORTFOLIOS",
           "SOFT_CONFIG", "soft_warning", "describe_config",
           "sound_verdict", "merge_sound_check", "SOUND_OK", "SOUND_BAD", "SOUND_OFF"]


def _support_literals(support):
    """Support vars (0-based) -> both polarities as DIMACS literals, the pool mining combines."""
    return None if not support else [s for v in support for s in (v + 1, -(v + 1))]


# Mining budget and per-width candidate ceilings. Width 1 (the backbone) always runs; the wide
# widths are gated on their candidate count, known before any GPU work.
#: Total wall budget for mining, one deadline shared by both stages. Honoured to chunk
#: granularity (a chunk is never abandoned half-written), so a large-M formula can overshoot
#: by one chunk.
MINE_SECONDS = 5.0
#: Stage 1: width 1 unbounded, plus the wide widths cheap for this support size (the width-2
#: ceiling exhausts |sup| up to ~700; past that MINE_SECONDS binds). `1: 0` is load-bearing:
#: `mine_entailed` caps width 1 too, and a dict without key 1 falls back to DEFAULT_MAX_TUPLES,
#: which would silently cap the backbone on the default sampling path.
STAGE1_MAX_TUPLES = {1: 0, 2: 1_000_000, 3: 40_000}
#: Stage 2: forced wide widths for an instance still stuck after the units are in. A dict for
#: the same reason as stage 1.
STAGE2_MAX_TUPLES = {1: 0, 2: 2_000_000, 3: 2_000_000}
#: Formulas smaller than this skip mining under `mine="auto"` (`--mine on` forces past it).
#: Do not raise it: ABC-encoded CNFs are small by construction and mining pays most there
#: (docs/findings/MINING_FINDINGS.md).
MINE_MIN_CLAUSES = 500


def _mine_score(smp, warm) -> float:
    """Models a chunk is expected to yield: batch rows x completion rate — the proxy for "did
    mining pay?", free because both terms are already in hand.

    K ~ budget / (4 * M_long), so chunk wall is roughly constant whatever the formula and what
    moves is how many rows fit and how many complete. Mined width >= 3 clauses push those two
    opposite ways; mined width-1/2 clauses are K-free (flush bitmap / implication list). A
    proxy, not a measurement: the warmup reads success at a pinned small batch."""
    return float(smp.K) * float(warm["best_success"])


class VarOrder(int, Enum):
    """Which unassigned variable each decision picks. int-valued on purpose: the kernel path
    tests `if random_var:`, so RANDOM stays truthy and STRUCTURED falsy."""

    RANDOM = 1        # uniform random free variable (CMSGen's Algorithm 1); the default
    STRUCTURED = 0    # first free variable in index order (the gate DAG order on a Tseitin CNF)


class Lookahead(str, Enum):
    """Polarity heuristic inside the fused trajectory kernel. str-valued, and NONE is the empty
    string so it stays falsy; the sampler treats anything but "soft"/"fsoft" as off."""

    NONE = ""         # uniform random polarity: the uniformity-safe choice (Barbarik-accepted
    #                   where the bandit delivers it; docs/findings/UNIFORMITY_FINDINGS.md §4b)
    SOFT = "soft"     # epsilon-greedy conflict-risk polarity: a big success win on tight circuits
    #                   but biased by construction and Barbarik-rejected almost everywhere
    #                   (UNIFORMITY_FINDINGS.md §4b). Opt-in only, never in a default portfolio.


class SamplingConfig(NamedTuple):
    """One portfolio entry: how a trajectory decides and how many decisions per propagation
    round. A tuple subclass, accepted wherever the old `(random_var, B, lookahead)` triples were.

    `decisions_per_round` (B) is a throughput and a uniformity knob: at B >= |support| keeping
    the survivors is exact rejection sampling, i.e. exactly uniform (UNIFORMITY_FINDINGS.md §3);
    smaller B weights solutions by their attraction basin."""

    var_order: VarOrder
    decisions_per_round: int
    lookahead: Lookahead = Lookahead.NONE

    def describe(self) -> str:
        la = "" if self.lookahead is Lookahead.NONE else f", {self.lookahead.name.lower()}-polarity"
        return f"{self.var_order.name.lower()} order, B={self.decisions_per_round}{la}"


# Circuit portfolio: every entry is the outright winner on some instance and the bandit picks
# per instance (docs/findings/GPU_SAMPLING_FINDINGS.md).
CIRCUIT_PORTFOLIO = [
    SamplingConfig(VarOrder.RANDOM, 8),    # broad deep-circuit winner
    SamplingConfig(VarOrder.RANDOM, 1),    # wins the conflict-heavy deep set
]

# Default portfolio for projected sampling (the sk_ / CMSGen-comparison story).
DEFAULT_PORTFOLIO = [
    SamplingConfig(VarOrder.RANDOM, 8),        # the classic niche config (BCP-determined support)
    SamplingConfig(VarOrder.RANDOM, 1),        # the sk_ winner
    SamplingConfig(VarOrder.STRUCTURED, 1),    # the search-heavy "service" family
]

# Opt-in only (`--allow-soft`), never a default-portfolio member: the bandit selects on
# models/s with no uniformity term, so a winning soft config silently emits a Barbarik-rejected
# distribution. Still the right config for "many solutions fast, distribution irrelevant".
SOFT_CONFIG = SamplingConfig(VarOrder.RANDOM, 8, Lookahead.SOFT)

# Single-config portfolios: the measurement escape hatch. One entry leaves the bandit nothing
# to select, so the run is a fixed algorithm. Pin one for any cross-GPU or before/after
# comparison: the bandit has no stability term and can pick differently per machine or per
# repeat (docs/findings/CROSS_GPU_FINDINGS.md).
SINGLE_PORTFOLIOS = {
    "b8": [SamplingConfig(VarOrder.RANDOM, 8)],        # throughput-leaning; over-constrains
    "b1": [SamplingConfig(VarOrder.RANDOM, 1)],        # success-leaning; the deep-circuit winner
    "s1": [SamplingConfig(VarOrder.STRUCTURED, 1)],    # gate-DAG order, the "service" family
}

PORTFOLIOS = {"circuit": CIRCUIT_PORTFOLIO, "default": DEFAULT_PORTFOLIO, **SINGLE_PORTFOLIOS}


def portfolio(name: str = "default", allow_soft: bool = False) -> list:
    """The named portfolio, optionally with the non-uniform soft config appended. `b8`/`b1`/`s1`
    are single-config (no bandit): use one when the config must not be a free variable."""
    if name not in PORTFOLIOS:
        raise ValueError(f"unknown portfolio {name!r} (have: {', '.join(PORTFOLIOS)})")
    return [*PORTFOLIOS[name], SOFT_CONFIG] if allow_soft else list(PORTFOLIOS[name])


def describe_portfolio(configs) -> str:
    """Human-readable one-line-per-config rendering, for CLI output and results metadata."""
    return "; ".join(c.describe() if isinstance(c, SamplingConfig)
                     else SamplingConfig(*c).describe() for c in configs)


def soft_warning() -> str:
    """The warning printed whenever the soft config is in a portfolio: one string for both
    front-ends, kept beside the verdict it states."""
    return ("  [warn] --allow-soft: the soft risk-polarity config is in the portfolio. It raises\n"
            "         success rate on tight circuits but is NOT uniform (Barbarik rejects it\n"
            "         on 64 of the 65 circuit instances it can sample)\n"
            "         — do not use rows produced by it for a distribution claim.")


def describe_config(cfg) -> str:
    """Render one config the way `describe_portfolio` does, from the raw `(random_var, B,
    lookahead)` triple that crosses the results JSON (enums decayed to bool/int/str)."""
    if not cfg:
        return "?"
    la = cfg[2] if len(cfg) > 2 else ""
    return SamplingConfig(VarOrder(int(bool(cfg[0]))), int(cfg[1]),
                          Lookahead(la or "")).describe()


def _core_provenance() -> str:
    """Which engine build produced a result: "libgpap <build id>". The build id, not the
    version: two rows that disagree on it genuinely came from different libraries."""
    try:
        from gpap.engine import _core
        return f"libgpap {_core.build_id()}"
    except Exception:       # noqa: BLE001 — provenance must never break a run
        return "unknown"


def sample_file(path, device, **kw):
    """Run the full pipeline on a DIMACS CNF file (support from `c ind` lines). Returns the
    dict of sample_clauses; the parse is charged to phases["parse"] and the e2e wall."""
    from gpap.engine import read_dimacs

    t_start = time.perf_counter()   # the end-to-end clock: parse through the timed run
    phases = {}
    # One parse (clauses and the `c ind` support), shared by construction, mining and the caller.
    nv0, cls0, sup0 = read_dimacs(path)
    phases["parse"] = time.perf_counter() - t_start
    # Wire map (var -> input@cycle) from a .bench conversion; empty for a plain CNF.
    from gpap.circuits.bench import parse_support_labels
    wire_map = parse_support_labels(path)
    # No encoder-provenance gate here: mining is worth more on ABC-encoded CNFs than on Tseitin
    # ones (docs/findings/MINING_FINDINGS.md); MINE_SECONDS bounds the cost.
    return sample_clauses(nv0, cls0, sup0, device,
                          _t_start=t_start, _phases=phases, _wire_map=wire_map, **kw)


# ---------------------------------------------------------------- coverage-run host budget
# The coverage run reconstructs every collected row into Python objects at 72 bytes per
# (row x variable) (measured), so the bound is on the product, not on rows alone: a row-only
# ceiling calibrated on small formulas OOM-killed the sweep on a large Tseitin unroll.
COV_BYTES_PER_CELL = 72

# Peak host bytes the collection may take; small because this is a diagnostic readout that has
# to fit beside the timed run on the smallest campaign box.
COV_HOST_BUDGET = int(os.environ.get("GPAP_COV_HOST_BUDGET", 1 << 31))       # 2 GiB


def cov_batch(nv: int, cov_k_cap: int = 2048, cov_chunks: int = 4,
              budget: int | None = None) -> int:
    """Rows the coverage run may collect per chunk, from the host-RAM budget and `nv`. Divided
    by `cov_chunks` because kept rows accumulate over the whole run. Never 0: one row per chunk
    is always attempted, so an oversized formula reports a few rows rather than none
    (`unique_est_status` and `n_collected` say what the estimate rests on)."""
    budget = COV_HOST_BUDGET if budget is None else budget
    per_row = max(1, int(nv) * COV_BYTES_PER_CELL)
    return max(1, min(int(cov_k_cap), int(budget) // (per_row * max(1, int(cov_chunks)))))


def sample_clauses(nv, clauses, support, device, *, simplify=True, els=True, n_time=50_000,
                   n_cov=0, out_cap=0, b=8, k_cap="auto",
                   max_chunks=None, time_limit=60.0, cov_chunks=4,
                   lookahead=False, verify_clauses=None, verify_cpu=False, cov_k_cap=2048,
                   verify_sound=False,
                   random_var=True, configs="default", mine="auto",
                   mine_gate=0.15, min_measure_chunks=3, give_up_chunks=0, seed=5, progress=None,
                   _t_start=None, _phases=None, _wire_map=None):
    """The throughput run, timed end-to-end, plus an optional collect run for coverage/soundness.

    `n_time` is a contract: the timed run samples until it has delivered that many models; the
    safety valves (`time_limit` seconds of sampling wall, `max_chunks`, and the OPT-IN
    `give_up_chunks`) are reported via `chunk_capped`, and `stop` says which one fired.

    `give_up_chunks=0` (the default) means the run spends the budget it was given. Set it only
    where abandoning a dry formula early is worth more than the measurement -- it ends a run
    that has found nothing after every config ran plus that many chunks, which on a
    low-success instance is seconds into a minute-long budget. It was on by default in the core
    and that is how a 60 s campaign cell came to report zero after 2 s on instances the budget
    could sample.
    `configs`: a portfolio run as a throughput bandit (one config per chunk at full K, switching
    to the winner); None runs the single fixed (`random_var`, `b`) config. The winner is reused
    for the coverage run, whose batch is `min(cov_k_cap, cov_batch(nv))` so reconstructing
    models to host RAM stays bounded.
    `k_cap="auto"` is the library's cap, the same on every GPU; an int forces it.
    `verify_sound` (default off) re-checks every emitted model on the GPU and reports
    `sound_check` ("ok" / "UNSOUND" / "off") plus `unsound_rows`; its wall is in no rate.
    `progress(event, data)` is called as the run reaches each of them — "formula" once the
    formula is built, "mined" once mining has been kept or reverted — so a front-end can report a
    phase WHEN IT HAPPENS instead of reading it off the returned dict minutes later. This module
    never prints; the callback is how the print layer learns the order."""
    if configs == "default":
        configs = DEFAULT_PORTFOLIO
    # Resolve once so every rebuild below lands on the same cap.
    k_cap = resolve_k_cap(k_cap, device)
    t_start = _t_start if _t_start is not None else time.perf_counter()
    phases = _phases if _phases is not None else {}

    def _warmup(s):
        # Warm up kernels/allocations at a tiny K with the full round budget and one chunk per
        # config, so the timed run pays no first-touch cost and every config's completion rate is
        # read (the best is "best_success", the miner's stuck signal). Restores the RNG after.
        cw = list(configs) if configs else None
        # pinned_batch, not a raw K assignment: the pin bounds the chunk regardless and restores K.
        with s.pinned_batch(min(s.K, 1024)):
            w = s.sample(1, configs=cw, decisions_per_round=b, random_var=random_var,
                         max_chunks=len(cw) if cw else 1,
                         collect=False, lookahead=lookahead)
        bd = (w.get("config_breakdown") or {}).values()
        w["best_success"] = max((c["success_rate"] for c in bd), default=w["success_rate"])
        s.reseed(seed)          # both streams: sampling and the delivery projection
        torch.cuda.synchronize()
        return w

    def _active_support(s):
        """Support variables still active after preprocessing, or None when the formula declared
        no support. 0 and None are DIFFERENT facts — a support propagation determines entirely is
        0 active, not "no support declared" — and reporting both as None hid the largest change
        there is."""
        return None if s.support is None else len(s.support)

    def _unsat(s) -> bool:
        """PROVED no models (preprocessing returned None, or the construction flush conflicted),
        which is a different fact from "found nothing". One definition: it is reported at
        construction time and again in the returned dict."""
        return bool((getattr(s, "_recon", None) or {}).get("unsat")) or bool(s.engine.flush_conflict)

    def _report(event, data):
        if progress is not None:
            progress(event, data)

    _tc = time.perf_counter()
    smp = GPUSampler.from_clauses(nv, clauses, device, simplify=simplify, els=els,
                                  support=support, k_cap=k_cap, seed=seed)
    phases["construct"] = time.perf_counter() - _tc
    # The shape of the instance is known now, and everything after it (warmup, up to MINE_SECONDS
    # of mining, the timed run) takes real time: report it before spending any of that.
    _report("formula", {"nv": nv, "n_clauses": len(clauses),
                        "n_support": len(support) if support else nv,
                        "support_size": _active_support(smp),
                        "unsat": _unsat(smp)})
    mine_info = None
    if device.type == "cuda":
        _tw = time.perf_counter()
        w = _warmup(smp)
        phases["warmup"] = time.perf_counter() - _tw
        # Entailed-clause mining (gpap.mining): every learned clause is entailed, so the model
        # set is unchanged; whether the sampler gets faster is what the keep-if-it-helps gate
        # below decides. Skipped on tiny formulas unless `mine=True` (MINING_FINDINGS.md).
        if mine is True or (mine == "auto" and smp.M >= MINE_MIN_CLAUSES):
            try:
                # Stage 1 mines the backbone plus the affordable wide widths; stage 2 forces
                # the wide widths only if the instance is still stuck after the units are in.
                _tm = time.perf_counter()
                score_before = _mine_score(smp, w)       # what we have to beat to keep the clauses
                mine_deadline = _tm + MINE_SECONDS      # one budget, shared by both stages
                mi = mine_entailed(nv, clauses, device, pool=_support_literals(support),
                                   max_tuples=STAGE1_MAX_TUPLES, seconds=MINE_SECONDS)
                phases["mine"] = time.perf_counter() - _tm
                learned = list(mi.clauses)
                stuck = w["best_success"] < mine_gate
                if learned:
                    # Rebuild on the augmented formula and re-warm: the stuck signal and the score.
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
                # Keep the clauses only if they help: width >= 3 finds enlarge num_long and so
                # shrink K (width-1/2 finds are K-free), and `K x success` is the models-per-chunk
                # proxy. Revert when it did not beat the pre-mining sampler.
                score_after = _mine_score(smp, warm if learned else w)
                # No baseline when the un-mined warmup completed nothing: keep the clauses.
                if learned and score_before > 0 and score_after <= score_before:
                    _tm = time.perf_counter()
                    smp = GPUSampler.from_clauses(nv, clauses, device, simplify=simplify, els=els,
                                                  support=support, k_cap=k_cap, seed=seed)
                    phases["rebuild"] = phases.get("rebuild", 0.0) + time.perf_counter() - _tm
                    # Re-warm: every other route into the timed run went through `_warmup`, and
                    # a reverted run must not pay first-touch cost inside the measurement.
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
                # `smp` is now the sampler the timed run will use (kept or reverted), so the
                # active support here is the FINAL one. It is not merely smaller: the rebuild
                # re-runs simplification and ELS over the augmented formula, which fixes support
                # variables (fewer) but also splits equivalence classes (more). The console
                # reported the pre-mining count with no way to see either.
                _report("mined", {**mine_info,
                                  "support_size": _active_support(smp)})
            except RuntimeError as e:   # mining is an optimization — never let it kill the run
                log.warning("mining skipped after a device error: %s", e)
    t0 = time.perf_counter()
    # A big-K GPU meets n_time in the bandit cold-start, so `min_measure_chunks` makes the winner
    # run that many back-to-back chunks for a stable, config-consistent rate.
    mmc = max(1, min(min_measure_chunks,
                     (max_chunks - (len(configs) if configs else 1)) if max_chunks
                     else min_measure_chunks))
    # out_cap rides along with the timed run: rows are projected off chunks that run anyway.
    if configs:
        t = smp.sample(n_time, configs=configs,
                       max_chunks=max_chunks, collect=False, lookahead=lookahead,
                       min_measure_chunks=mmc, max_seconds=time_limit, out_cap=out_cap,
                       give_up_chunks=give_up_chunks, verify_sound=verify_sound)
    else:
        t = smp.sample(n_time, decisions_per_round=b, random_var=random_var,
                       max_chunks=max_chunks, collect=False,
                       lookahead=lookahead, min_measure_chunks=mmc, max_seconds=time_limit,
                       out_cap=out_cap, give_up_chunks=give_up_chunks,
                       verify_sound=verify_sound)
    if device.type == "cuda":
        torch.cuda.synchronize()
    # The `--out` projection is delivery, not sampling: bill it to phases["output"] and keep it
    # out of `wall`, so every rate is what the run reports without `--out`.
    proj_s = float(t.get("out_project_s") or 0.0)
    # The soundness re-check is a self-test with the same rule; subtracted here because a pinned
    # single config has no per-config wall and would otherwise carry it.
    verify_s = float(t.get("verify_sound_s") or 0.0)
    wall = time.perf_counter() - t0 - proj_s - verify_s
    # Engine provenance: which packed split, kernel variant, K and build produced the number,
    # captured after any post-mining rebuild and before the coverage pass lowers K.
    engine_info = {"kernel": "triton-packed-sum", "mode": smp.engine.mode,
                   "shift": smp.engine.shift,
                   "split": f"{smp.engine.shift}|{32 - smp.engine.shift}",
                   # k_cap is the resolved ceiling; `batch` is what ran, `K` the calibrated value.
                   "batch": smp.batch_size,
                   "K": smp.K, "k_cap": k_cap,
                   # M_eff = all clauses; m_long = the len>=3 state clauses the K model charges.
                   "N_eff": smp.N, "M_eff": smp.M, "m_long": smp.M_state,
                   "core": _core_provenance(),
                   "max_clause_len": smp.formula.max_clause_len if smp.M else 0}
    # Headline throughput = the best-throughput config's steady-state rate (best_rate_sps), not
    # the end-to-end blend, which the bandit's exploration chunks drag; falls back to the blend
    # for a single non-adaptive config.
    sps_e2e = t["n_models"] / wall if wall > 0 else 0.0
    sps = t.get("best_rate_sps") or sps_e2e
    brc = t.get("best_rate_config")
    # Work rates for the same config as `sps`, so dec/s, impl/s, assigns/s stay consistent.
    dps =t.get("best_rate_decisions_per_s") if t.get("best_rate_decisions_per_s") is not None else t.get("decisions_per_s")
    ips = t.get("best_rate_implications_per_s") if t.get("best_rate_implications_per_s") is not None else t.get("implications_per_s")
    aps = t.get("best_rate_assigns_per_s") if t.get("best_rate_assigns_per_s") is not None else t.get("assigns_per_s")
    if brc:                                          # (rv, B, lookahead) of the fastest config
        tuned_rv, tuned_b, tuned_la = brc[0], brc[1], brc[2]
        succ = t.get("best_rate_success", t["success_rate"])
    else:
        tuned_rv, tuned_b = t["random_var"], t["B"]
        tuned_la = t.get("lookahead", lookahead)
        succ = t["success_rate"]
    # Sample output: up to `out_cap` support-projection bit rows from the winning config, kept by
    # the timed run itself (`out_bits`). A top-up runs only when the collection came up short,
    # bounded by what is left of the sampling budget. Always well-shaped: a (0, S) matrix when
    # nothing was found (docs/findings/CLI_DELIVERY_FINDINGS.md).
    cols = [v + 1 for v in smp.projection_cols()]
    samples = np.zeros((0, len(cols)), dtype=np.uint8)
    phases["output"] = proj_s
    if out_cap:
        samples = t.get("out_bits", samples)
        cols = t.get("out_cols", cols)
        # A file never claims more than the run delivered; a spent `--time-limit` buys no top-up.
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
    # The delivered rows are inside the e2e wall; the soundness self-test is not (it is billed to
    # `phases_diag`), so `phases` sums to `wall_total_s` and `sps_e2e_total` is the same number
    # whether or not the check ran.
    wall_total = time.perf_counter() - t_start - verify_s
    sps_e2e_total = t["n_models"] / wall_total if wall_total > 0 else 0.0
    # Opt-in diagnostics (--n-cov, --verify-sound) are billed to `phases_diag`, outside
    # `wall_total_s`.
    phases_diag: dict = {}
    # The timed run's soundness re-check (the coverage run's is already inside "coverage").
    phases_diag["verify_sound"] = verify_s
    if n_cov:
        # Bound host RAM: this run reconstructs every row into Python lists, so pin the batch to
        # `cov_batch` (rows x nv, not rows alone).
        smp.pin_batch(min(smp.K, cov_batch(nv, cov_k_cap, cov_chunks)))
    # Coverage run: collect with the winning config and count distinct support projections.
    # Separate from the timed run so reconstruction/dedup is never charged to samples/s.
    vc = clauses if (verify_cpu and verify_clauses is None) else verify_clauses
    _tcov = time.perf_counter()
    c = (smp.sample(n_cov, decisions_per_round=tuned_b, random_var=tuned_rv,
                    max_chunks=cov_chunks, collect=True, verify_clauses=vc,
                    lookahead=tuned_la, verify_sound=verify_sound) if n_cov else {})
    phases_diag["coverage"] = (time.perf_counter() - _tcov) if n_cov else 0.0
    # The declared support (every emitted row's width), beside `support_size` (what stays active).
    sup1 = [v + 1 for v in (support or range(nv))]
    unsat = _unsat(smp)
    del smp
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # Estimated unique solutions/s: timed rate x the distinct fraction of the coverage sample.
    # None when unmeasured; `unique_est_status` says why.
    n_coll = len(c.get("models", []))
    uniq_sps = (sps * c["n_distinct"] / n_coll) if (n_coll and c.get("n_distinct")) else None
    # "not requested" and "collected nothing" are different facts: a consumer folding both to 0
    # publishes a low-success instance the timed run delivered on as "produced nothing".
    uniq_status =("ok" if uniq_sps is not None
                   else ("not-requested" if not n_cov else "no-coverage-sample"))
    phases["timed"] = wall
    return {"samples_per_s": sps, "success_rate": succ,
            "nv": nv, "n_clauses": len(clauses),              # ORIGINAL sizes (pre-simplify)
            "sup1": sup1,
            "sps_e2e_total": sps_e2e_total,                   # incl. construction+simplify+mining
            "phases": phases,                                 # DELIVERY phases; these sum to wall_total_s
            "phases_diag": phases_diag,                       # opt-in readouts, OUTSIDE wall_total_s
            "n_models": t["n_models"],                        # total solutions found in the timed run
            # A safety valve stopped the run short of `n_time` (rates stay valid). NOT set on a proved-UNSAT
            # formula, where 0 models is the complete answer and nothing was cut: the flag is
            # what both printers mark with '<', and marking that row would invent a valve.
            "chunk_capped": (t["n_models"] < n_time) and not unsat,
            # WHICH valve, not merely that one fired. `target` | `time` | `chunks` | `zero_yield`
            # | `unsat` (gpap.engine._core.SAMPLE_STOP). The core has always distinguished them
            # and this function used to discard the answer, leaving `chunk_capped` to conflate
            # "spent its whole budget" with "abandoned after the cold start plus two grace
            # chunks" -- which are opposite facts about a tool and about an instance. A row that
            # delivered nothing cannot be read without it: on the paper campaign's 15 GPAP-zero
            # cells nothing in the record says whether 60 s were spent or four chunks were.
            "stop": t.get("stop"),
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
            # tri-state, opt-in: "off" is not a verdict; UNSOUND dominates over both runs
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
            # samples = uint8 (n, S) support projections; cols = each column's original var;
            # wire_map = {var: input@cycle} for a circuit CNF (see gpap.sampling.output)
            "samples": samples, "cols": cols, "wire_map": _wire_map or {},
            # rows asked for vs kept, and the one config that produced them
            "out_requested": out_cap, "out_config": t.get("out_config"),
            "unsat": unsat,                                   # PROVED no models (not merely none found)
            "tuned_rv": tuned_rv, "tuned_b": tuned_b, "tuned_la": tuned_la,
            "config_breakdown": t.get("config_breakdown")}
