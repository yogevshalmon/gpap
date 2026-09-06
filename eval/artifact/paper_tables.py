#!/usr/bin/env python3
"""Build the paper's experiment section — every table, straight from the experiment JSONs.

Nothing here is typed by hand. That is the point: an earlier version of the deck carried a
sequential headline of 47.7x that silently included four instances whose output pins are
tautologies (real value 36.5x), and the only durable fix is to derive every published number from
the data with the exclusions applied in ONE place.

The raw data lives OUTSIDE the repo, in $GPAP_RESULTS (default ../gpap-results) — it is too heavy
to track. See docs/paper/README.md for the full regeneration recipe.

    R=${GPAP_RESULTS:-../gpap-results}/l4_abc_20260902   # the current campaign
    eval/artifact/paper_tables.py --cmsgen $R/gpap_vs_cmsgen_l4_abc \\
        --rivals $R/gpap_vs_gpu_rivals_l4_abc \\
        --barbarik $R/barbarik_abc/circ_abc_*.jsonl \\
        --out-tex docs/paper/tables.tex --out-md docs/paper/RESULTS.md

(--recheck / --uniformity / --soft add their tables when a campaign carries those arms; the
current one deliberately does not — docs/paper/README.md states what is in scope and why.)

Rules applied everywhere, so they cannot diverge between tables:
  * VACUOUS-PIN instances (constant last-cycle outputs at T=25, so pinning is a tautology and
    every sampler scores ~100%: b02/b05/b07/s526/b19, detected per row from the measured
    p_accept) are excluded from every geomean, and screened BEFORE any error or zero bucket.
  * ZERO-SEARCH cells (`zero_search`) are excluded for the same reason: no decisions and no
    implications means the support is free after propagation and the ratio measures the
    instance, not the sampler.
  * An instance where EITHER side emitted nothing has no finite ratio. It is excluded from the
    geomean and counted in its own column — CMSGen=0 (our win), GPAP=0 (our loss), both=0 —
    so the columns sum to the instance count and our failures are as visible as theirs.
  * Rival rows that errored (timeout / no gradient path) are counted, never folded in as zeros.
  * A stage with fewer instances than its family's maximum is marked incomplete.
"""
import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gpap.sampling.pipeline import sound_verdict          # noqa: E402 — after sys.path
from eval.rivals.rival_common import (PAPER_INSTANCES,   # noqa: E402 — ONE home for the sets
                                      space_from_distinct)

VACUOUS_PINS = {"b02", "b05", "b07", "s526"}


# DEMOTIC's own Table II asks each sampler for "a minimum of 1000 distinct solutions
# (except c17)" — the exception is the authors telling us the metric cannot be applied to an
# instance whose whole space is smaller than the request. 2**support < 1000 is that rule.
TINY_SPACE_FLOOR = 1000


def tiny_space(r, rival_key=None):
    """True when the projected solution space is smaller than the number of samples asked for.

    Both tools then simply ENUMERATE it, and the ratio is a race between two startup costs
    rather than a sampling measurement. ISCAS-85 `c17` has 5 inputs = 32 points: DEMOTIC finds
    19 of them at 14 unique/s while GPAP reads out 640k/s, a 45,520x cell that would lift the
    DEMOTIC-native geomean from 2.7x to ~6.5x on its own.

    Deliberately SYMMETRIC, and it does not flatter us: removing c17 costs us 2.4x in Table 2
    and gains 1.2x in Table 1 (comb 1,348x -> 1,620x), because there the tiny space makes us look
    WORSE. A screen that only ever helped would be the thing to distrust.

    Reads `support` on a bench row; on a rival row that field does not exist, so it falls back to
    the rival's own recorded `num_inputs` (miter mode included: |PI| x T). Returns False when
    neither is present rather than guessing.
    """
    n = r.get("support")
    if n is None and rival_key:
        n = ((r.get(rival_key) or {}).get("num_inputs"))
    try:
        n = int(n)
    except (TypeError, ValueError):
        return False
    return solution_space_estimate(r, n) < TINY_SPACE_FLOOR


def solution_space_estimate(r, n_support):
    """The row's best estimate of |S|, the number of SOLUTIONS — never more than the input
    hypercube 2^|support|, and smaller wherever the row carries evidence:

      * the measured acceptance probability (`p_accept`, or its 95% upper bound `p_accept_lt`
        when no hit was observed) times the hypercube — every solution is an accepted input;
      * the repeat rate among the kept output rows (`rival_common.space_from_distinct`) — the
        measurement that exposed `c1355` under the rare pin: 41 input bits, so 2^41 points,
        yet 0.56 distinct in 1,000 kept rows, i.e. ~700 solutions, a space both tools ENUMERATE.

    The minimum of what is known. A row with no evidence gets the hypercube, as before."""
    space = float(2 ** min(n_support, 64))          # min() so a huge support cannot blow up 2**n
    sb = r.get("sim_baseline") or {}
    p = sb.get("p_accept") if sb.get("p_accept") is not None else r.get("p_accept")
    if p:
        space = min(space, p * 2 ** min(n_support, 64))
    elif p == 0:
        plt = sb.get("p_accept_lt") if sb.get("p_accept_lt") is not None else r.get("p_accept_lt")
        if plt:
            space = min(space, plt * 2 ** min(n_support, 64))
    # the repeat rate: rival rows keep it per seed under `gpap`, bench rows on the row itself
    pairs = [(s.get("unique_rows"), s.get("unique_distinct"))
             for s in ((r.get("gpap") or {}).get("seeds") or [])]
    pairs.append((r.get("n_collected"), r.get("n_distinct")))
    for rows, distinct in pairs:
        if rows and distinct:
            est = space_from_distinct(rows, distinct)
            if est is not None:
                space = min(space, est)
    return space


def vacuous_pin(r, name=None):
    """True when the pinned target is a TAUTOLOGY — every input sequence satisfies it.

    Then "sampling" is a free hypercube over the input bits, every sampler scores ~100%, and the
    ratio measures the instance. `VACUOUS_PINS` used to be a hardcoded four (b02/b05/b07/s526),
    which worked only for as long as the instance list did not grow: running the FULL sequential
    family (2026-09-03) brought in `b19`, whose pin has p_accept == 1.0 at both depths and both
    targets, and it sailed into the geomean as a 491x cell measuring nothing.

    So the test is now the MEASUREMENT the row already carries — `sim_baseline.p_accept` — with
    the name list kept only as a fallback for rows written before `--sim-baseline` was standard.
    Detected per row, not by a name list: the same rule `zero_search` follows, and for the same
    reason.
    """
    if name and name in VACUOUS_PINS:
        return True
    p = (r.get("sim_baseline") or {}).get("p_accept")
    if p is None:
        p = r.get("p_accept")                  # rival rows carry it at the top level
    return p is not None and p >= 1.0


def zero_search(r):
    """True when the row emitted samples having done NO search: no decisions, no implications.

    After unit propagation the support is a free hypercube, so every trajectory succeeds
    (`succ == 1.0`) and "sampling" is a random-bit readout at ~240M samples/s. It is a correct
    result, but it exercises none of the propagation engine, and CMSGen still pays to discover the
    same thing — so the ratio measures the degeneracy, not the sampler.

    This is the vacuous-pin problem wearing a different hat, and it is worse in one respect: the
    best-encoder rule ACTIVELY SELECTS these cells, because they carry the highest ratio in their
    family. On EPFL `ctrl` the Tseitin encoding samples at 2.0M/s with succ 0.07 while ABC
    collapses the output cone and gives succ 1.0 at 240M/s, so the encoder A/B hands the headline
    the degenerate cell. Measured cost of leaving them in: EPFL rare 9.4x -> 29.0x, EPFL random
    8.2x -> 19.5x, sk 98.9x -> 113.1x; no other family contains one.
    """
    NEARLY_FREE_IMPL_PER_MODEL = 2.0
    n = r.get("n_models") or 0
    if n <= 0:
        return False
    dec, impl = (r.get("decisions_per_s") or 0), (r.get("implications_per_s") or 0)
    sps = (r.get("sps") or r.get("samples_per_s") or 0)      # rival rows / bench rows
    if not dec and not impl:
        return True
    # ...AND the continuous version of the same predicate. An EXACT zero is a knife-edge, and a
    # cell can sit a hair off it and still have done no search worth the name: `c432` under our
    # rare pin (2026-09-03) leaves 4 effective variables and 3 clauses after propagation, and
    # delivers a model per 2.93 decisions and 1.08 IMPLICATIONS — against a next-lowest surviving
    # cell of 44.1 on the same campaign, a 40x gap. It was the only DEMOTIC-rare cell with a
    # finite ratio, so that stage's entire geomean was this one degenerate cell at 89,786x.
    # A sampler performing ~one implication per solution it delivers is reading bits out of a
    # hypercube, whatever its counters say. The threshold is placed inside a two-order-of-
    # magnitude GAP rather than fitted: implications per delivered model over the 2026-09-03
    # campaign run 0.83 (c17, already tiny), 1.09 (c432-rare), 6.32 (c17, tiny), 10.0 (b02,
    # already vacuous), then 44.1 (b13) and up. Every value in (1.09, 44.1) yields the same
    # partition, so the rule does not depend on where in that gap it sits.
    return bool(sps and impl / sps < NEARLY_FREE_IMPL_PER_MODEL)


def unsat_at_depth(r):
    """True when a solver PROVED this cell has no solution at the depth it was measured at.

    Then neither sampler can produce one, and a zero on either side says nothing about either
    tool. `b01` under Recurrent's own target is the case that forced this: its published output
    pattern is unreachable at exactly T=25 for parity reasons (their own sampler runs to 50 and
    pools every depth), so the 2026-09-03 campaign booked a GPAP=0 loss against a solution that
    does not exist. Recorded by the harness as `gpap.unsat_at_T`, which is True only on a proof;
    `None` means unasked or unresolved inside the budget and is NEVER read as satisfiable.
    """
    return ((r.get("gpap") or {}).get("unsat_at_T") is True) if isinstance(r, dict) else False


def sound_failed(r):
    """True when a row's on-GPU soundness re-check RAN and found an unsound model.

    Three-valued, and the third value is the point: `sound_check == "off"` means the check was
    not run — it is opt-in everywhere (`gpap bench --verify-sound`), so most rows carry no
    verdict at all, which is neither a pass nor a failure. Reading a missing check as a failure
    fails an honest sweep; reading it as a pass publishes an unverified one. Rows recorded before
    the key existed carry the old boolean `gpu_sound` and are read the same way.

    ONE definition, imported by `verify_artifact.py` — the gate and the table must not disagree
    about which cells were actually checked. The tri-state itself (including how a legacy
    `gpu_sound` bool reads) comes from `gpap.sampling.pipeline.sound_verdict`, which is also what
    the bench table, the CLI and the Colab driver use, so a reader and a writer of the same JSON
    cannot disagree either."""
    if r.get("cpu_sound") is False:
        return True
    return sound_verdict(r) == "UNSOUND"


def sound_unchecked(r):
    """True when the row says nothing about soundness — the check was not run."""
    return sound_verdict(r) == "off" and r.get("cpu_sound") is None


def cms_arm_ran(r):
    """True when the CMSGen arm actually RAN on this row.

    ABSENT is not ZERO (the `sound_check` lesson): a row written by a GPAP-only sweep carries no
    `cms_*` keys at all, and reading the missing count as "CMSGen scored 0" mints a win over a
    tool that never ran. The dangerous shape is the MIXED stage — a campaign resumed on a VM
    where the cmsgen binary failed to restore appends armless rows to an armed file, and a
    stage-level "does anything here have an arm?" test then waves every one of them into the
    zero ledger. So the test is per ROW, defined once, and imported by `verify_artifact.py`
    (which FAILS a mixed stage) and by `eval/eval_plots.py`'s summary ledger."""
    return "cms_n_samples" in r or r.get("cms_sps") is not None


def unique_unmeasured(g):
    """True when GPAP's unique/s figure could not be MEASURED on a rival cell.

    That is NOT "GPAP produced nothing", and bucketing it as one publishes a win as a loss. The
    estimate is the timed rate times a distinct fraction. Since 2026-09-03 that fraction is
    measured on the timed run's OWN kept output rows (`unique_est_status` "ok" |
    "no-sample-rows"), so it exists whenever the run delivered anything; rows before that
    measured it on a separate OFF-timing coverage run of a few thousand trajectories, which on a
    very-low-success instance collected nothing while the TIMED run delivered models
    ("no-coverage-sample"). b11-c25 at a rare pin was recorded exactly that way — 42 models, no
    estimate — against the rival's 0.0469 u/s, an ~15x win that was published as an outright
    loss. Rows written before `unique_est_status` existed are caught the same way: no estimate,
    but models in the timed run.

    ONE definition, imported by every consumer (`rival_family` here, `eval/eval_plots.py`,
    `eval/rivals/rivals_summary.py`) — a rule with three copies is a rule that will drift.
    """
    if g.get("unique_per_s_est"):
        return False
    n = g.get("n_models")
    if n is None:                       # a row too old to say how many models it delivered
        return g.get("unique_est_status") in ("no-coverage-sample", "no-sample-rows")
    # "no-sample-rows" with ZERO models is not an unmeasured estimate, it is a run that
    # delivered nothing: b01 under Recurrent's own target (p_accept < 7e-8 at T=25) was
    # booked "unmeasured" this way while the rival's depth-pooled count found 30 — a GPAP
    # zero published as no-verdict. Unmeasured means models WERE delivered.
    return n > 0


CMS_FAMILIES = [
    ("seq-c25-rare", "ITC'99 + ISCAS-89 sequential", "T=25", "rare"),
    ("seq-c50-rare", "ITC'99 + ISCAS-89 sequential", "T=50", "rare"),
    ("seq-c25-random", "ITC'99 + ISCAS-89 sequential", "T=25", "random"),
    ("seq-c50-random", "ITC'99 + ISCAS-89 sequential", "T=50", "random"),
    ("comb-rare", "ISCAS-85 combinational", "T=1", "rare"),
    ("comb-random", "ISCAS-85 combinational", "T=1", "random"),
    ("hwmcc-c25-rare", "HWMCC'17 property", "T=25", "rare"),
    ("hwmcc-c50-rare", "HWMCC'17 property", "T=50", "rare"),
    ("hwmcc-c25-random", "HWMCC'17 property", "T=25", "random"),
    ("hwmcc-c50-random", "HWMCC'17 property", "T=50", "random"),
    ("epfl-rare", "EPFL combinational", "T=1", "rare"),
    ("epfl-random", "EPFL combinational", "T=1", "random"),
    ("sk", "Sketch / sk_ projected CNF", "--", "--"),
]

# One row per (rival, protocol). DEMOTIC is measured on its own paper's 11 ISCAS-85;
# Recurrent-CircuitSAT on EVERY b*/s* sequential circuit the campaign can process (its own
# paper's 16 are a subset, scored beside the whole) — each under two tasks: the pins the
# rival's table names, and our rare pin on the SAME circuits, so the pair isolates the task.
# ⚠️ HTS is absent since 2026-09-02, and its absence is a SCOPE statement, not a loss: its paper
# (arXiv 2502.08673) evaluates 60 instances of which none is a circuit (random q-family,
# OR-configuration, prod-*, s15850a_*), so in a circuits-only evaluation every HTS cell would be
# our task on instances we chose. Re-enable with RIVAL_HTS=1 and restore the row here.
# THE INSTANCE SET: every b*/s* sequential circuit the campaign can process (user's rule,
# 2026-09-03: "run all seq circuits of b* and s* on their pins and our rare pins; what we cannot
# process is omitted and said so"). Table 2 scores every cell that ran; the geomean over the
# subset each rival's OWN paper evaluates is printed BESIDE it (`paper set`), never instead of
# it, because the two differ and a reader must see by how much.
#
# ⚠️ It is not a formality: the four ITC'99 instances b03/b04/b11/b13 are in Recurrent's run.py
# target table but NOT in its Table II, and they INFLATE us — "their target table" reads 4.1x
# with them and 3.0x on the paper's own set. Both numbers are in the table.
#   DEMOTIC   arXiv 2502.08086 Table II: all 11 ISCAS-85.
#   Recurrent arXiv 2502.21226 Table II: 8 ISCAS-89 + 8 ITC'99. Its 8th ISCAS-89 row is `s38584`
#                                        (12 PI / 278 PO / 1,452 DFF). Read as `s35844` until
#                                        2026-09-04, with `s35932` substituted for it; `s35932`
#                                        is now a circuit WE add, scored outside their set.
# Each rival's own published instance set is defined ONCE in eval/rivals/rival_common.py
# (`PAPER_INSTANCES`, imported above): the harnesses expand `paper` to it, so what a sweep runs
# and what this table scores are the same object.


RIVALS = [
    ("demotic_native", "demotic", "DEMOTIC", "their ISCAS-85 pins"),
    ("demotic_comb_rare", "demotic", "DEMOTIC", "their ISCAS-85, our rare pin"),
    ("recurrent_table", "rcs", "Recurrent-CircuitSAT", "their target table, $T{=}25$"),
    ("recurrent_rare", "rcs", "Recurrent-CircuitSAT", "our rare pins, $T{=}25$"),
    ("recurrent_table_t50", "rcs", "Recurrent-CircuitSAT", "their target table, $T{=}50$"),
    ("recurrent_rare_t50", "rcs", "Recurrent-CircuitSAT", "our rare pins, $T{=}50$"),
]
# BOTH DEPTHS on the sequential set (2026-09-04). T=25 was described as "the rival's own default
# depth" and it is not: their run script says `--num_clock_cycles 50` and their paper sets the
# limit to eta=50, over which their sampler pools every depth. A shallower unroll gives them a
# SMALLER pool to find solutions in, so measuring only at 25 was another choice against them.
# Both depths are run and both are reported; a stage file that does not exist is simply absent
# from the tables, so a campaign may still be published at one depth.
# A 2x2: each rival under two tasks — the pins its paper names, and our rare pin on the SAME
# circuits. Only the task varies across a row, so the comparison isolates it. DEMOTIC's rows are
# its paper's 11 ISCAS-85; Recurrent's are every seq b*/s* (its paper's 16 scored beside).
# GPAP's arm on every row is the abc:fast encoding — the same formula the CMSGen tables run —
# recorded per row as `gpap.encoder` (before 2026-09-03 it was the native Tseitin, a 2-10x
# larger formula; those rows are superseded).
#
# ⚠️ `demotic_miter` is measured and archived but NOT published here (2026-09-02). It runs
# DEMOTIC on our SEQUENTIAL instances by unrolling them T cycles and collapsing the rare-pattern
# check into a single-output comparator cone (eval/rivals/unroll_to_bench.py), which DEMOTIC can
# eat because it is combinational. Three reasons it is out: its instances are in NEITHER paper
# (DEMOTIC's evaluation is ISCAS-85 only; b*/s* are Recurrent's set), it is now redundant with
# `demotic_comb_rare`, which answers the same "DEMOTIC under our task" question on DEMOTIC's own
# circuits, and it is the only arm whose circuit we synthesized rather than cited. Dropping it
# removes a 557x WIN, so the trim is conservative. The rows stay in the campaign directory as a
# supplementary robustness check; re-add the tuple to publish them.

# ---- THE CONVERGED PROTOCOL (2026-09-03): Table 1 is read off the RIVAL rows ----------------
# The paper's instance set is the rivals' own — each rival's published benchmarks, under the
# pins its table names and under our rare pin — and every one of those cells is run by THREE
# tools on ONE constraint: the rival on its circuit, GPAP on the abc:fast CNF of that circuit
# and those pins, and CMSGen on that same CNF file (`rivals_*.py --cmsgen`,
# `rival_common.cmsgen_arm`). So the CMSGen table (Table 1) and the rival table (Table 2) are
# the SAME cells from the SAME files, screened once, and a reader can line the rows up. It used
# to be two instance sets: a separate `gpap bench` sweep pinned the circuits its own way (one
# output on ISCAS-85, the outputs directly rather than a miter on the unrolls, targets `random`
# and depth 50 besides) and never saw a rival's pins at all. Those `CMS_FAMILIES` stages are
# SUPPLEMENTARY now: printed after these rows when their JSONs exist, never the headline.
# Derived from RIVALS so the two tables cannot name different stages.
_CMS_SET_LABEL = {"demotic": "ISCAS-85 (DEMOTIC's set)",
                  "rcs": "ITC'99 + ISCAS-89 (Recurrent's set)"}
RIVAL_CMS_SETS = [(stem, _CMS_SET_LABEL.get(key, key), "rare" if "rare" in stem else "their pins")
                  for stem, key, _, _ in RIVALS]
_RIVAL_KEY = {stem: key for stem, key, _, _ in RIVALS}


def gm(xs):
    xs = [x for x in xs if x and x > 0 and math.isfinite(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def med(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def load(path):
    return json.load(open(path)) if os.path.isfile(path) else None


# ------------------------------------------------------------------ GPAP vs CMSGen

def _enc(r):
    """The encoder label of a cell. `enc_key` is the precise one (`abc:fast` vs `abc:full` are
    different pipelines with different encode costs); `encoder` is the fallback for rows written
    before it existed, and None for a family that is already CNF."""
    return r.get("enc_key") or r.get("encoder")


def encoders_in(payload):
    """The encoders present in a stage, in a stable order.

    Circuit families carry two rows per instance (`tseitin`, `abc`); a CNF family such as `sk`
    has no encoder at all and yields `[None]`, i.e. exactly one row per instance.
    """
    encs = {_enc(r) for r in payload.get("results", [])
            if not r.get("skipped") and not r.get("error")}
    named = [e for e in ("tseitin",) if e in encs]        # Tseitin first: it is the default
    extra = sorted(e for e in encs if e is not None and e != "tseitin")
    return (named + extra) or [None]


def select_cells(payload, encoder):
    """One row per instance, for ONE encoder. Both Table 1 and Table 3 call this, so Table 3
    describes exactly the runs that produced Table 1.

    There is NO best-of-encoders selection here, deliberately. It used to pick, per instance,
    the encoder with the highest `ratio` — which makes the published statistic its own argmax
    key, and hands the rival's number a vote on which of our runs is reported. The effect was
    measured on the 2026-08-23 campaign: seq-c25-rare 66.7x -> 84.3x, seq-c25-random 250.8x ->
    333.8x, seq-c50-rare 34.0x -> 42.1x (a wash in the pooled geomean, 1.24-1.33x on the
    flagship sequential families). The mechanism is that ABC helps CMSGen MORE than it helps us
    on some instances, so maximising the quotient awards a shared input-preprocessing gain to
    one side only.

    Reporting each encoder as its own row removes the question instead of answering it: both
    pipelines are shown, neither is selected, and the encoder A/B — which is a result in its own
    right, and not a monotone one (ABC unlocks b14 and collapses b13's rare pin) — stays visible.
    """
    rows = [r for r in payload.get("results", [])
            if not r.get("skipped") and not r.get("error") and _enc(r) == encoder]
    return rows, {r["name"].split(".")[0]: r for r in rows}


def _pre_frac(r):
    """Share of GPAP's end-to-end wall spent BEFORE the clock the headline rate is measured on:
    parse + engine construction + ELS + warmup + mining + the post-mining rebuild/recheck."""
    ph = r.get("phases") or {}
    wt = r.get("wall_total_s")
    if not wt:
        return None
    return sum(ph.get(k, 0.0) for k in
               ("parse", "construct", "warmup", "mine", "rebuild", "recheck")) / wt


#: Fewest samples a tool must deliver before its rate is a ratio's denominator — ONE floor for
#: both sides of both tables. Below it the cell says the tool could not sample the instance,
#: which is a result, not a factor (Poisson relative error at n=30 is ~18%; at n=1 the figure
#: means nothing). Until 2026-09-05 the floor was ENFORCED on the rival side of Table 2 and only
#: NARRATED on the CMSGen side of Table 1, where `b14` (13 CMSGen samples in 60 s) sat inside the
#: rare-pin geomean at full weight — the per-column-filtering trap between two tables.
MIN_SOLUTIONS = 30


def cms_family(d, stem, encoder):
    payload = load(os.path.join(d, f"{stem}.json")) if d else None
    if not payload:
        return None
    allr = payload.get("results", [])
    rows, cells = select_cells(payload, encoder)
    if not rows:
        return None
    # ABSENT is not ZERO (the sound_check lesson, again): a row with no cms arm at all — a
    # GPAP-only sweep, e.g. an engine A/B re-measurement — must not be read as "CMSGen scored 0"
    # and counted as an outright win over a tool that never ran. A stage with no CMSGen arm has
    # no Table 1 row; the zero ledger below is only for rows where the comparison actually ran.
    if not any(cms_arm_ran(r) for r in rows):
        return None

    def scored(r, key):
        return (r.get(key) or 0) > 0

    # ---- the degeneracy screens, applied BEFORE anything is counted -------------------------
    # A cell that measures the INSTANCE rather than the sampler is not evidence about either
    # one, so it leaves through its own column and takes no part in the geomean, the zero
    # ledger OR the win count. Screening only the geomean (what this did until 2026-08-26) left
    # them inside `inst` and inside `wins`, so the identity printed under the table — inst =
    # geomean instances + the zero columns — was false on 6 of 7 families, which a referee can
    # check in ten seconds. Note a zero-search cell can also be a CMSGen=0 cell (EPFL `ctrl`
    # abc: succ 1.0 at 240M samples/s against CMSGen's nothing), so the screen has to come
    # first or those degenerate cells are published as outright wins.
    vac = {k for k, r in cells.items() if vacuous_pin(r, k)}
    free = {k for k, r in cells.items() if k not in vac and zero_search(r)}
    # A space smaller than the sample request is enumerated, not sampled (see `tiny_space`).
    tiny = {k for k, r in cells.items() if k not in vac | free and tiny_space(r)}
    drop = vac | free | tiny
    kept = {k: r for k, r in cells.items() if k not in drop}
    # The stage-level guard above catches a fully GPAP-only file; a MIXED file — a campaign
    # resumed on a VM where the cmsgen binary failed to restore — passes it, and its armless
    # rows would fall straight into the CMSGen=0 bucket below as phantom wins. Screen per ROW:
    # a row whose comparison never ran is in none of the columns (the `weird` precedent), the
    # caveats name it, and verify_artifact FAILS the stage.
    noarm = sorted(k for k, r in kept.items() if not cms_arm_ran(r))
    kept = {k: r for k, r in kept.items() if cms_arm_ran(r)}

    # Bucket every REMAINING instance exactly once. An earlier version keyed off `ratio` alone,
    # and a GPAP-zero row has ratio == 0.0 (falsy) while a both-zero row has ratio == None — so
    # both fell through every branch and vanished, at the same time as CMSGen-zero rows were
    # counted as wins. That is an asymmetric ledger. A reader must be able to add the columns up.
    finite, gpap_zero, cms_zero, both_zero, weird, cms_err = {}, [], [], [], [], []
    cms_thin = []
    for k, r in kept.items():
        # A crashed CMSGen subprocess is recorded as `cms_error` with a count of 0 — an
        # environment failure, not a measurement. Scoring that 0 as "CMSGen sampled nothing"
        # publishes a broken binary as a stage of outright wins (the floored-count trap's
        # sibling). Its own bucket, out of every column, named in the caveats.
        if r.get("cms_error"):
            cms_err.append(k)
            continue
        g, c = scored(r, "n_models"), scored(r, "cms_n_samples")
        # A handful of CMSGen samples is a capability statement, not a denominator — the same
        # floor Table 2 applies to the rival (MIN_SOLUTIONS), so one rule screens both sides.
        if g and c and (r.get("cms_n_samples") or 0) < MIN_SOLUTIONS:
            cms_thin.append(k)
            continue
        if g and c:
            if r.get("ratio") and math.isfinite(r["ratio"]):
                finite[k] = r
            else:                       # both emitted yet no usable quotient: never silent
                weird.append(k)
        elif c:
            gpap_zero.append(k)
        elif g:
            cms_zero.append(k)
        else:
            both_zero.append(k)
    n_inf, n_g0, n_b0 = len(cms_zero), len(gpap_zero), len(both_zero)
    gm_vals = [r["ratio"] for r in finite.values()]

    def _all(pred):
        """geomean over cells that pass `pred`, used only for the size-of-the-error caveats."""
        return gm([r["ratio"] for k, r in cells.items()
                   if pred(k) and r.get("ratio") and math.isfinite(r["ratio"])
                   and scored(r, "n_models") and scored(r, "cms_n_samples")])

    return {
        "encoder": encoder, "cells": len(rows), "inst": len(cells),
        "gm": gm(gm_vals),
        # what the headline would read with the degenerate cells left in — the size of the error
        # being avoided, so the caveat can quote it rather than merely assert it
        "gm_all": _all(lambda k: True),
        "gm_withfree": _all(lambda k: k not in vac),
        # The SAME cells, scored end-to-end on BOTH sides: GPAP's wall_total (parse + engine
        # construction + ELS + warmup + MINING + the timed run) over CMSGen's subprocess wall,
        # against the steady-state column's sampling-wall-over-sampling-wall. Each column
        # charges both sides for getting ready, or neither.
        "gm_e2e": gm([r.get("ratio_e2e") for r in finite.values()]),
        "pre_frac": (statistics.median([_pre_frac(r) for r in finite.values()
                                        if _pre_frac(r) is not None] or [0.0])),
        "free": len(free), "free_names": sorted(free),
        "vac": len(vac), "vac_names": sorted(vac),
        # How many instances actually ENTER the geomean. When CMSGen scores zero on most of a
        # family (HWMCC T=50: 6 of 7), the ratio is undefined there and the "geomean" collapses
        # onto the handful that remain — 421x over ONE instance. A paper must show this number.
        "n_gm": len(gm_vals),
        "weird": len(weird), "has_vac": bool(vac), "tiny": len(tiny),
        "tiny_names": sorted(tiny),
        # A CMSGen-zero instance is a win for us and a GPAP-zero instance is a loss; neither has
        # a finite ratio, so both stay out of the geomean and both are named in the caveats.
        # wins/n are counted over EXACTLY the instances the geomean is computed over, plus the
        # zero columns — the same set, so the numbers in one row are mutually consistent.
        "wins": sum(1 for r in finite.values() if r["ratio"] >= 1) + n_inf,
        "n": len(finite) + n_inf + n_g0, "inf": n_inf,
        "gpap0": n_g0, "both0": n_b0,
        "gpap0_names": sorted(gpap_zero), "both0_names": sorted(both_zero),
        "noarm": len(noarm), "noarm_names": noarm,
        "cms_err": len(cms_err), "cms_err_names": sorted(cms_err),
        "cms_thin": len(cms_thin), "cms_thin_names": sorted(cms_thin),
        "skipped": sum(1 for r in allr if r.get("skipped")),
        "errors": sum(1 for r in allr if r.get("error")),
        "unsound": sum(1 for r in rows if sound_failed(r)),
        "unchecked": sum(1 for r in rows if sound_unchecked(r)),
        "gpu": (payload.get("gpu") or {}).get("name"),
        "cpu": (payload.get("cpu") or {}).get("model"),
        "patched": payload.get("cmsgen_patched"),
        # The engine that MEASURED these rows — `gpap_build_id()` stamped per row, the ground
        # rule's identity for a result (`gpap_version` read "0.1.0" while the engine got 1.8x
        # faster under it). More than one value here means the stage mixes engines.
        "cores": sorted({(r.get("engine") or {}).get("core") for r in rows} - {None}),
    }


# ------------------------------------------------------------------ propagation throughput

def prop_rates(d, stem, encoder):
    """Engine-level throughput: implications and decisions per second, both samplers.

    This is the measurement that isolates the ENGINE from the sampler wrapped around it, and it
    factors the headline exactly. For any instance,

        samples/s ratio  =  (implications/s ratio)  /  (implications-per-sample ratio)

    because implications/s = samples/s x implications/sample on each side. The first factor is
    what the GPU propagator buys; the second is what our algorithm SPENDS, since GPAP discards a
    conflicted trajectory outright (rejection sampling) where CMSGen backtracks and keeps the
    partial work. Reporting only the quotient hides both. Checked against the recorded sample
    ratio to the printed digit on every family.

    CMSGen's counters come from `--statsevery` in scripts/cmsgen_gpap_eval.patch, so this is its
    own accounting of its own work, not our estimate of it.
    """
    payload = load(os.path.join(d, f"{stem}.json")) if d else None
    if not payload:
        return None
    _, cells = select_cells(payload, encoder)
    # Same exclusions as the headline, and for the same reason: a cell that measures the
    # instance measures neither engine. Table 1 and Table 3 must describe the same runs.
    # No fallback to the unscreened cells: a family whose every cell is degenerate has no
    # Table 3 row — quietly re-admitting the screened cells there would make Table 3
    # describe runs Table 1 refused to.
    rows = [r for k, r in cells.items()
            if not vacuous_pin(r, k) and not zero_search(r) and not tiny_space(r)]
    # EVERY column is computed over the SAME rows — the ones where all four quantities are
    # positive. Filtering per column instead put an instance like EPFL `bar` (GPAP propagates hard
    # but emits nothing) into the impl/s ratio and not into work-per-sample, which broke the
    # identity by 3x on that family and made a sound decomposition look wrong. Instances where
    # either sampler emitted nothing have no work-per-sample at all; they are Table 1's zero
    # columns, and they are excluded here for the same reason.
    return _prop_stats([(r.get("implications_per_s"), r.get("cms_implications_per_s"),
                         r.get("gpap_sps"), r.get("cms_sps"),
                         r.get("decisions_per_s"), r.get("cms_decisions_per_s"))
                        for r in rows], encoder)


def _prop_stats(quads, encoder):
    """The Table 3 aggregation over (impl_g, impl_c, sps_g, sps_c, dec_g, dec_c) tuples —
    shared by the bench-stage and the rival-stage readers so the identity is one computation."""
    gi, ci, ri, rd, work, gper, cper = [], [], [], [], [], [], []
    for a, b, s, t, dg, dc in quads:
        if not all(x and x > 0 for x in (a, b, s, t)):
            continue
        gi.append(a)
        ci.append(b)
        ri.append(a / b)
        gper.append(a / s)
        cper.append(b / t)
        work.append((a / s) / (b / t))
        if dg and dc:
            rd.append(dg / dc)
    if not gi:
        return None
    return {"n": len(ri), "encoder": encoder, "gpap": gm(gi), "cms": gm(ci), "ratio": gm(ri),
            "dec": gm(rd), "work": gm(work), "gper": gm(gper), "cper": gm(cper)}


# ------------------------------------------------------------------ GPAP vs GPU rivals

#: The rival-side name of the one floor above (kept: consumers and tests import it by this name).
RIVAL_MIN_SOLUTIONS = MIN_SOLUTIONS


def rival_cells(rows, key):
    """Classify every rival cell exactly once — THE per-cell rule Table 2 sums over.

    Yields `(name, variant, bucket, gpap_unique_per_s, rival_unique_per_s)` with bucket one of
    `ok` / `error` / `gpap_error` / `misaligned` / `vacuous` / `free` / `tiny` / `nomeas` /
    `both0` / `rival0` / `gpap0`;
    only `ok` carries two positive rates. `eval/eval_plots.py`'s rival scatter IMPORTS this
    instead of restating the screens, so the figure and the table cannot disagree about which
    cells are comparable (the `cms_arm_ran` precedent).
    """
    seen = {}
    for r in rows:
        seen[(r["name"], r.get("variant", "-"))] = r        # a retry supersedes its error row
    # ONE CELL PER INSTANCE, and where a stage holds more than one circuit variant (`raw` vs the
    # ABC-simplified netlist) the one that goes into the table is THE ONE BETTER FOR THE RIVAL.
    # Same tie-break as `num_steps`: where our protocol offers a tool two configurations it could
    # legitimately be run under, the rival gets the better of them, which can only move a ratio
    # against us. It also closes the asymmetry a referee sees immediately — GPAP samples an
    # ABC-optimized CNF while the rival is handed the raw netlist — for the price of running the
    # variant axis, and pooling both variants instead (the old behaviour) would count every
    # instance twice in one geomean.
    best = {}
    for (name, variant), r in seen.items():
        rate = ((r.get(key) or {}).get("unique_per_s") or 0.0) if not (r.get(key) or {}).get("error") else -1.0
        if name not in best or rate > best[name][0]:
            best[name] = (rate, variant, r)
    for name, (_, variant, r) in best.items():
        rv, g = r.get(key) or {}, r.get("gpap") or {}
        # SCREEN FIRST, THEN BUCKET — the same order Table 1 uses. The instance-only screens
        # (a vacuous pin, a tiny solution space) depend on the circuit and the pin, not on
        # whether either tool ran, so they come before either tool's error bucket: `b19` at
        # T=25 is vacuous (p_accept 1.0) AND Recurrent fails to initialize on it, and with the
        # error bucket first Table 1 counted it as vacuous while Table 2 counted it as a rival
        # failure — the same stage, two different ledgers. The `free` screen reads OUR
        # counters, so it stays behind the gpap_error bucket.
        if vacuous_pin(r, name.split(".")[0]):
            yield name, variant, "vacuous", 0.0, 0.0
            continue
        # No solution exists at this depth (proved) — neither sampler can produce one, so a zero
        # on either side is a fact about the instance. Screened with the other instance-only
        # screens, ahead of every tool's error and zero bucket.
        if unsat_at_depth(r):
            yield name, variant, "unsat", 0.0, 0.0
            continue
        if tiny_space(r, key):
            yield name, variant, "tiny", 0.0, 0.0
            continue
        if rv.get("error"):
            yield name, variant, "error", 0.0, 0.0
            continue
        # OUR side refused the instance (e.g. outside the packed envelope: b19 Tseitin-unrolls to
        # 2N=19.2M against the 8|24 cap of 16.8M). That is a REACH limit, not "GPAP produced
        # nothing" — bucketing it as gpap0 would publish a capability boundary as a loss on the
        # sampler, and bucketing it as a win would be worse. Its own column, like the rival's.
        if (g or {}).get("error"):
            yield name, variant, "gpap_error", 0.0, 0.0
            continue
        # (The vacuous-pin screen — b02/b05/b07/s526/b19 have constant last-cycle outputs at
        # T=25, so pinning them is a tautology and every sampler scores ~100% — is applied
        # above, before the error buckets. Table 1 has always screened these; the rival table
        # did not, which did not matter while no vacuous instance was in the rival set and
        # matters now that the set is each rival's OWN published one. A screen that lives in
        # one table and not its neighbour is the per-column-filtering trap one table along.)
        # Table 2's text says constraint alignment is VERIFIED per row (`pin_check_ok`), so a
        # row that FAILED that check cannot sit inside the geomean the claim covers: the two
        # sides were not given the same constraint, and whatever the ratio measures, it is not
        # the samplers. The harness prints the failure and records it; this is where it is
        # enforced, and the caveats name the count whenever it is non-zero.
        if r.get("pin_check_ok") is False:
            yield name, variant, "misaligned", 0.0, 0.0
            continue
        # Same zero-search screen as Table 1, where the row carries the counters (recorded
        # since 2026-08-23): a support that is FREE after propagation measures the instance,
        # not either sampler — a seed-mismatched b13 cell once carried a 4.7e9x ratio this way.
        if "decisions_per_s" in g and zero_search(g):
            yield name, variant, "free", 0.0, 0.0
            continue
        # (The tiny-space screen — Table 1's, read off the RIVAL's own num_inputs where a row
        # carries no `support`; c17 is DEMOTIC's own carve-out — is applied above as well.)
        rate = rv.get("unique_per_s") or 0.0
        # A RATIO IS ONLY AS GOOD AS ITS DENOMINATOR. A rival that scraped a handful of solutions
        # in its whole run has told us it cannot sample the instance; dividing by the rate those
        # few events imply turns that into a spurious factor. Measured on the 2026-09-04 rare
        # stage: DEMOTIC found ONE solution on `c880` (ratio 7,209,624x) and FOUR on `c3540`
        # (86,202x), and those two cells alone would have carried the row at a geomean of
        # 788,345x. Handing the rival the ABC-simplified circuit is what created them — it let it
        # scrape hits where it previously found none, converting honest rival=0 cells (excluded,
        # counted as outright wins) into finite ratios in the millions.
        #
        # Below the floor the cell is a CAPABILITY statement, not a measured factor, and it gets
        # its own column. This is conservative by construction: every cell it removes is a huge
        # win of ours. The floor is the same 30 events the CMSGen-side caveat uses — Poisson
        # relative error at n=30 is ~18%, and at n=1 the figure means nothing at all.
        n_uniq = rv.get("unique")
        if rate and n_uniq is not None and 0 < n_uniq < RIVAL_MIN_SOLUTIONS:
            yield name, variant, "rival_thin", 0.0, 0.0
            continue
        # NOT MEASURED is not a zero (see `unique_unmeasured`): it gets its own column, because
        # it is not evidence in either direction.
        if unique_unmeasured(g):
            yield name, variant, "nomeas", 0.0, rate
            continue
        ours = g.get("unique_per_s_est") or 0.0
        # Symmetric with the CMSGen table: name OUR zeros as loudly as theirs. A rival that
        # scored zero is an outright win, ours is an outright loss, and neither has a finite
        # ratio — so both stay out of the geomean and both get a column.
        if not rate and not ours:
            bucket = "both0"
        elif not rate:
            bucket = "rival0"
        elif not ours:
            bucket = "gpap0"
        else:
            bucket = "ok"
        yield name, variant, bucket, ours, rate


def rival_family(d, stem, key):
    payload = load(os.path.join(d, f"rivals_{stem}.json")) if d else None
    if payload is None:
        return None
    rows = (payload.get("rows") if isinstance(payload, dict) else payload) or []
    cells = list(rival_cells(rows, key))
    n = {}
    for _, _, bucket, _, _ in cells:
        n[bucket] = n.get(bucket, 0) + 1
    ratios = [ours / rate for _, _, bucket, ours, rate in cells if bucket == "ok"]
    # The same ratio restricted to the instances the rival's OWN paper evaluates — printed
    # beside the full geomean, not instead of it (see the note above RIVALS).
    paper = PAPER_INSTANCES.get(key)
    in_paper = lambda name: paper is None or name in paper           # noqa: E731
    paper_ratios = [ours / rate for name, _, bucket, ours, rate in cells
                    if bucket == "ok" and in_paper(name)]
    zero, gzero = n.get("rival0", 0), n.get("gpap0", 0)
    seen = {}
    for r in rows:
        seen[(r["name"], r.get("variant", "-"))] = r
    # NO SECOND CLOCK IN THIS TABLE (protocol v1.0, 2026-09-06 — RIVAL_FAIRNESS_FINDINGS.md
    # §17). Table 2 is the rivals' OWN metric — unique solutions per second of their
    # `model_runtime`, which excludes their model build — and GPAP's figure excludes construction
    # the same way. An end-to-end column across an in-process sampler and a subprocess gradient
    # tool has no definition either tool would recognise: it had FOUR in three review rounds
    # (0.8x, 7.6x, 20.9x/132x, 11.8x/68.8x on the same rows), and on DEMOTIC's cells both tools
    # finish the real work in under a second, so whatever it read was process start-up. Removed
    # by rule, not redefined a fifth time; the end-to-end comparison lives in Table 1, where the
    # two walls are the same kind of thing. Nothing here reads `wall_s`.
    okc, unsat_rival, low_vram, ladder_top, small = [], [], [], [], []
    for name, variant, bucket, ours, rate in cells:
        r = seen[(name, variant)]
        g, rv = r.get("gpap") or {}, r.get(key) or {}
        if bucket == "unsat" and rv.get("unique_per_s"):
            unsat_rival.append((name, rv["unique_per_s"]))
        gpu = r.get("gpu") or {}
        if gpu.get("total_gb") and gpu.get("free_gb") is not None and "sps" in g \
                and gpu["free_gb"] < 0.5 * gpu["total_gb"]:
            low_vram.append((name, gpu["free_gb"], gpu["total_gb"]))
        if bucket != "ok":
            continue
        x = ours / rate
        band = seed_band(g, ours, rate)
        okc.append((x, name, band, straddles(band)))
        sweep = rv.get("batch_sweep") or []
        # cells the rival ran at its two smallest surviving rungs with a rung above rejected:
        # the card bound the grant there, and the cell measures the tool at a small batch
        if rv.get("batch") and rv["batch"] <= 10_000 and len(sweep) > 1:
            small.append((x, name, rv["batch"]))
        # cells where the ladder's TOP rung completed first try: nothing above it was offered,
        # so the ladder bound the grant, not the card (a fourth rejection reason the rule does
        # not state; disclosed in the caveats)
        if rv.get("batch") and len(sweep) == 1 and rv["batch"] == (sweep[0].get("batch") or 0):
            ladder_top.append((name, rv["batch"]))
    okc.sort(key=lambda t: -t[0])
    ties = sum(1 for _, _, _, st in okc if st)
    small_names = {nm for _, nm, _ in small}
    rest = [x for x, nm, _, _ in okc if nm not in small_names]
    return {"cells": len(cells), "gm": gm(ratios), "n": len(ratios) + zero + gzero,
            "gm_paper": gm(paper_ratios), "n_paper": len(paper_ratios),
            "offpaper": len(ratios) - len(paper_ratios),
            # a straddling cell is a TIE: neither a win nor a loss (`ties` says how many)
            "wins": sum(1 for x, _, _, st in okc if x >= 1 and not st) + zero,
            "ties": ties, "errors": n.get("error", 0),
            "gm_lo": gm([b[0] if b else x for x, _, b, _ in okc]),
            "gm_hi": gm([b[1] if b else x for x, _, b, _ in okc]),
            "median": statistics.median([x for x, _, _, _ in okc]) if okc else None,
            "top": [(x, nm) for x, nm, _, _ in okc[:6]],
            "gm_wo_top1": gm([x for x, _, _, _ in okc[1:]]) if len(okc) > 1 else None,
            "gm_wo_top4": gm([x for x, _, _, _ in okc[4:]]) if len(okc) > 4 else None,
            "small_grant": sorted(small, reverse=True),
            "gm_small": gm([x for x, _, _ in small]), "gm_rest": gm(rest), "n_rest": len(rest),
            "unsat_rival": unsat_rival, "low_vram": low_vram, "ladder_top": ladder_top,
            "zero": zero, "gpap0": gzero, "both0": n.get("both0", 0),
            "free": n.get("free", 0), "nomeas": n.get("nomeas", 0),
            "misaligned": n.get("misaligned", 0), "vacuous": n.get("vacuous", 0),
            "tiny": n.get("tiny", 0), "unsat": n.get("unsat", 0),
            "thin": n.get("rival_thin", 0),
            "gpap_error": n.get("gpap_error", 0)}


# ------------------------------------------------------------------ GPAP vs CMSGen, on the rival rows

def _pa_range(rowsets):
    """'0.020–0.97, 2 cell(s) bounded below 8.1e-08–2.3e-05 where no hit was observed' — the
    measured acceptance probabilities of a protocol's non-tautological cells, from the rows."""
    meas, bounded = [], []
    for rows in rowsets:
        for r in rows.values():
            p, lt = r.get("p_accept"), r.get("p_accept_lt")
            if p and p < 1:
                meas.append(p)
            elif not p and lt:
                bounded.append(lt)
    if not meas:
        return "not recorded"
    out = f"{min(meas):.2g}–{max(meas):.2g}"
    if bounded:
        out += (f", {len(bounded)} cell(s) bounded below {min(bounded):.1e}–{max(bounded):.1e} "
                f"where no hit was observed")
    return out


def _rival_rows(d, stem):
    """The (name, variant) -> row map of a rival stage; a retry supersedes its error row."""
    payload = load(os.path.join(d, f"rivals_{stem}.json")) if d else None
    if payload is None:
        return None
    rows = (payload.get("rows") if isinstance(payload, dict) else payload) or []
    seen = {}
    for r in rows:
        seen[(r["name"], r.get("variant", "-"))] = r
    return seen


def _gpap_measured(r):
    """The GPAP arm of a rival row as a Table 1 cell, or None when it REFUSED the instance
    (oversize, outside the packed envelope, encode failure). That is an error row — counted,
    in no column — never a zero: a reach limit is not 'GPAP produced nothing'."""
    g = r.get("gpap") or {}
    return None if (g.get("error") or "sps" not in g) else g


def _gpap_e2e(g):
    """GPAP's end-to-end rate on a rival row — THE PICKED RUN'S, not a median over seeds.

    The row is one execution (`rival_common.median_run` picks the median-rate seed and publishes
    all of its fields), and this column is divided by, and multiplied against, that same run's
    steady-state rate. Taking a separate median here re-mixed two seed selections inside one
    ratio — the field-wise-median defect the row-level fix removed, reappearing in a derived
    quantity the fix did not touch. Legacy rows, written before the harness published a top-level
    `sps_e2e`, still fall back to the median over seeds.
    """
    if g.get("sps_e2e"):
        return g["sps_e2e"]
    return med([s.get("sps_e2e") for s in (g.get("seeds") or []) if s.get("sps_e2e")])


def _picked_seed(g):
    """The `seeds[]` entry of the run the row publishes (`median_seed` is the seed VALUE)."""
    seeds = g.get("seeds") or []
    for s in seeds:
        if s.get("seed") == g.get("median_seed"):
            return s
    for s in seeds:
        if s.get("sps_e2e") and g.get("sps_e2e") and abs(s["sps_e2e"] - g["sps_e2e"]) < 1e-9:
            return s
    return None


def seed_band(g, ours, rate):
    """(lo, hi): the cell's ratio at GPAP's slowest and fastest of its seeds, or None."""
    lo, hi, sps = g.get("sps_min"), g.get("sps_max"), g.get("sps")
    if not (lo and hi and sps and rate and ours):
        return None
    a, b = ours * lo / sps / rate, ours * hi / sps / rate
    return (min(a, b), max(a, b))


def straddles(band):
    """The win/loss sign lies inside GPAP's own seed band: a tie, not a tally."""
    return band is not None and band[0] < 1.0 <= band[1]


def _cms_ratio(r):
    """GPAP steady-state samples/s over CMSGen's, both emitting; None otherwise."""
    g = r["gpap"]
    if (g.get("n_models") or 0) > 0 and (r.get("cms_n_samples") or 0) > 0 and r.get("cms_sps"):
        x = (g.get("sps") or 0.0) / r["cms_sps"]
        return x if x > 0 and math.isfinite(x) else None
    return None


def rival_cms_family(d, stem, key):
    """GPAP vs CMSGen over a RIVAL stage's rows — Table 1 of the converged protocol.

    The same file and the same `(name, variant)` cells Table 2 scores (`rival_family`), each
    carrying the CMSGen arm run on the very CNF the GPAP arm sampled. The screens are Table 2's,
    applied FIRST (vacuous pin / no-search / tiny space: a cell that measures the instance
    measures neither sampler); the ledger is Table 1's (finite / CMSGen=0 / GPAP=0 / both=0,
    absent is not zero, a crashed baseline is not zero); the returned dict has Table 1's shape
    so ONE renderer prints both kinds of row. The rival's own outcome is irrelevant here — the
    CMSGen comparison ran whether or not the rival did. `gm_paper` is the same ratio over the
    rival paper's own instance subset, printed beside the whole set exactly as Table 2 does.
    """
    seen = _rival_rows(d, stem)
    if not seen or not any(cms_arm_ran(r) for r in seen.values()):
        return None                      # absent, or a rivals-only sweep: no CMSGen table
    cells, errors = {}, 0
    for k, r in seen.items():
        if _gpap_measured(r) is None:
            errors += 1
        else:
            cells[k] = r
    vac = {k for k, r in cells.items() if vacuous_pin(r, k[0].split(".")[0])}
    # PROVED to have no solution at this depth: neither sampler can produce one, so a zero on
    # either side is a fact about the instance (`b01` under Recurrent's target at T=25). Screened
    # in BOTH tables — a screen that lives in one table and not its neighbour is the
    # per-column-filtering trap one table along.
    unsat = {k for k, r in cells.items() if k not in vac and unsat_at_depth(r)}
    free = {k for k, r in cells.items() if k not in vac | unsat and zero_search(r["gpap"])}
    tiny = {k for k, r in cells.items() if k not in vac | unsat | free and tiny_space(r, key)}
    kept = {k: r for k, r in cells.items() if k not in vac | unsat | free | tiny}
    noarm = sorted(k[0] for k, r in kept.items() if not cms_arm_ran(r))
    kept = {k: r for k, r in kept.items() if cms_arm_ran(r)}
    finite, gpap_zero, cms_zero, both_zero, weird, cms_err = {}, [], [], [], [], []
    cms_thin = []
    for k, r in kept.items():
        if r.get("cms_error"):
            cms_err.append(k[0])
            continue
        g = r["gpap"]
        gs, cs = (g.get("n_models") or 0) > 0, (r.get("cms_n_samples") or 0) > 0
        # the CMSGen-side floor, the same MIN_SOLUTIONS the rival side of Table 2 is held to
        if gs and cs and (r.get("cms_n_samples") or 0) < MIN_SOLUTIONS:
            cms_thin.append(k[0])
            continue
        if gs and cs:
            x = _cms_ratio(r)
            if x:
                finite[k] = x
            else:                       # both emitted yet no usable quotient: never silent
                weird.append(k[0])
        elif cs:
            gpap_zero.append(k[0])
        elif gs:
            cms_zero.append(k[0])
        else:
            both_zero.append(k[0])
    n_inf, n_g0, n_b0 = len(cms_zero), len(gpap_zero), len(both_zero)
    paper = PAPER_INSTANCES.get(key)
    paper_ratios = [x for (name, _v), x in finite.items() if paper is None or name in paper]
    e2e, pre = [], []
    for k in finite:
        g, r = kept[k]["gpap"], kept[k]
        ge = _gpap_e2e(g)
        if ge and r.get("cms_sps_e2e"):
            e2e.append(ge / r["cms_sps_e2e"])
        if ge and g.get("sps"):
            pre.append(max(0.0, 1.0 - ge / g["sps"]))     # (wall_total - timed) / wall_total

    def _all(pred):
        """geomean over cells that pass `pred`, used only for the size-of-the-error caveats."""
        return gm([_cms_ratio(r) for k, r in cells.items()
                   if pred(k) and cms_arm_ran(r) and not r.get("cms_error")])

    encs = sorted({r["gpap"].get("encoder") or "?" for r in cells.values()})
    depths = sorted({r.get("cycles") for r in cells.values() if r.get("cycles") is not None})
    armed = list(kept.values())
    return {
        "encoder": "+".join(encs) if encs else None, "cells": len(cells), "inst": len(cells),
        "depth": ("T=" + "/".join(str(t) for t in depths)) if depths else None,
        "gm": gm(list(finite.values())), "gm_all": _all(lambda k: True),
        "gm_withfree": _all(lambda k: k not in vac),
        "gm_paper": gm(paper_ratios), "n_paper": len(paper_ratios),
        "gm_e2e": gm(e2e), "pre_frac": statistics.median(pre) if pre else 0.0,
        "free": len(free), "free_names": sorted(k[0] for k in free),
        "vac": len(vac), "vac_names": sorted(k[0] for k in vac),
        "unsat": len(unsat), "unsat_names": sorted(k[0] for k in unsat),
        "n_gm": len(finite), "weird": len(weird), "has_vac": bool(vac),
        "tiny": len(tiny), "tiny_names": sorted(k[0] for k in tiny),
        "wins": sum(1 for x in finite.values() if x >= 1) + n_inf,
        "n": len(finite) + n_inf + n_g0, "inf": n_inf, "gpap0": n_g0, "both0": n_b0,
        "gpap0_names": sorted(gpap_zero), "both0_names": sorted(both_zero),
        "noarm": len(noarm), "noarm_names": noarm,
        "cms_err": len(cms_err), "cms_err_names": sorted(cms_err),
        "cms_thin": len(cms_thin), "cms_thin_names": sorted(cms_thin),
        "skipped": 0, "errors": errors,
        "unsound": sum(1 for r in cells.values()
                       if sound_failed(r["gpap"]) or r["gpap"].get("sim_check") == "BAD"),
        "unchecked": sum(1 for r in cells.values() if sound_unchecked(r["gpap"])),
        "gpu": next(((r.get("gpu") or {}).get("name") for r in seen.values()
                     if (r.get("gpu") or {}).get("name")), None),
        "cpu": next(((r.get("cpu") or {}).get("model") for r in seen.values()
                     if (r.get("cpu") or {}).get("model")), None),
        "patched": (all(r.get("cms_patched") for r in armed) if armed else None),
        "cores": sorted({(r["gpap"].get("engine") or {}).get("core")
                         for r in cells.values()} - {None}),
    }


def rival_prop_rates(d, stem, key):
    """Table 3 over a rival stage: the cells `rival_cms_family` scores (same screens, no
    fallback), GPAP's counters from its arm and CMSGen's from its own `--statsevery`."""
    seen = _rival_rows(d, stem)
    if not seen:
        return None
    quads, encs = [], set()
    for (name, _v), r in seen.items():
        g = _gpap_measured(r)
        if g is None:
            continue
        encs.add(g.get("encoder") or "?")
        if not cms_arm_ran(r) or r.get("cms_error"):
            continue
        if vacuous_pin(r, name.split(".")[0]) or zero_search(g) or tiny_space(r, key):
            continue
        # the two screens Table 1 applies that this loop did not: a proved-UNSAT cell and a
        # CMSGen count under the floor. Without them Table 3 is over a different row set than
        # Table 1, which is the one failure the algebraic identity exists to catch.
        if unsat_at_depth(r) or 0 < (r.get("cms_n_samples") or 0) < MIN_SOLUTIONS:
            continue
        quads.append((g.get("implications_per_s"), r.get("cms_implications_per_s"),
                      g.get("sps"), r.get("cms_sps"),
                      g.get("decisions_per_s"), r.get("cms_decisions_per_s")))
    return _prop_stats(quads, "+".join(sorted(encs)) if encs else None)


# ------------------------------------------------------------------ uniformity

# Reading order for the uniformity table. Alphabetical puts `ctl:*` in the middle, which buries
# the two rows that make the rest interpretable: without a negative control that is NOT flagged
# and a positive control that IS, "0/36 flagged" means nothing. Controls first, then the B ladder
# in order (the B >= |support| threshold is the result), then the opt-in config, then the rival.
UNI_ORDER = ["ctl:uniform", "ctl:tv0.05", "ctl:tv0.1",
             "gpap:b1", "gpap:b8", "gpap:b16", "gpap:b32", "gpap:b64",
             "pipe:default", "pipe:circuit", "pipe:circuit+soft", "gpap:soft", "cmsgen"]


def uni_sort_key(item):
    (name, kind) = item
    idx = UNI_ORDER.index(name) if name in UNI_ORDER else len(UNI_ORDER)
    return (idx, name, kind)


def uniformity(paths):
    """Aggregate the exact-oracle fuzzing campaign per sampler, split by instance kind.

    SCREEN FIRST, THEN AGGREGATE — the same rule as every other table here, and the campaign's
    own summary already applies it: an instance whose solution set is a full SUBCUBE (every
    support variable pinned or free) is sampled by the don't-care FILLER, not by the config
    under test — two different configs return bit-identical statistics there — so it is
    evidence about neither sampler. fuzz_uniformity.py stamps `meta.subcube` per instance;
    excluding those rows here keeps Table 4 in agreement with the tool that produced the data
    (it used not to, and the paper's median could disagree with the campaign's own printout).
    The excluded count is reported in `args["_subcube_excluded"]` so the table can state it.
    """
    rows = []
    args = {}
    for p in paths or []:
        d = load(p)
        if d:
            rows += d.get("rows", [])
            args = d.get("args", args)
    if not rows:
        return None, {}
    n_sub = sum(1 for r in rows if (r.get("meta") or {}).get("subcube"))
    rows = [r for r in rows if not (r.get("meta") or {}).get("subcube")]
    args = dict(args)
    args["_subcube_excluded"] = n_sub
    if not rows:
        return None, args
    agg = {}
    for r in rows:
        kind = r.get("kind", "?")
        for name, s in (r.get("samplers") or {}).items():
            k = (name, kind)
            a = agg.setdefault(k, {"tv": [], "reject": 0, "n": 0, "unsound": 0})
            a["tv"].append(s.get("tv_median"))
            a["n"] += 1
            a["unsound"] += s.get("unsound") or 0
            if s.get("verdict") == "SKEW":
                a["reject"] += 1
    out = {}
    for (name, kind), a in agg.items():
        out[(name, kind)] = {"tv": med(a["tv"]), "reject": a["reject"], "n": a["n"],
                             "unsound": a["unsound"]}
    return out, args


# ------------------------------------------------------------------ barbarik

def barbarik(paths):
    """The standard certificate, per sampler under test.

    The eligibility rule and the per-arm counts are IMPORTED from
    eval/uniformity/barbarik_table.py, not re-derived: an instance counts only where the negative
    control (the trusted uniform reference, put under test through the same hook) was itself
    Accepted. Where the control is Rejected, CM_test is not measuring sampler uniformity — the
    declared sampling set is not an independent support, so the model-uniform reference is not
    projection-uniform. Publishing those cells would report an instance property as a sampler one.
    """
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uniformity"))
    from barbarik_table import arm_summary, buckets, check_one_setting, load as bload

    rows = []
    for p in paths or []:
        if os.path.exists(p):
            rows += bload([p])
    if not rows:
        return None
    eps, eta, delta = check_one_setting(rows)      # raises on a pooled loose+strict file list
    eligible, rejected, errored = buckets(rows)
    return {"summary": arm_summary(rows, eligible), "eligible": sorted(eligible),
            "control_rejected": sorted(rejected), "control_errored": sorted(errored),
            "params": {"epsilon": eps, "eta": eta, "delta": delta},
            # provenance: this table is its own campaign, and the header above it is not its host
            "source": sorted(os.path.abspath(p) for p in (paths or []) if os.path.exists(p))}


# ------------------------------------------------------------------ reproducibility

def recheck(orig, rec, stages=("seq-c25-rare", "comb-rare", "epfl-rare", "sk")):
    if not (orig and rec):
        return None
    exact = mismatch = succ_ok = succ_bad = 0
    rates = []
    for stage in stages:
        a = load(os.path.join(orig, f"{stage}.json"))
        b = load(os.path.join(rec, f"{stage}.json"))
        if not (a and b):
            continue

        def cells(p):
            return {(r["name"].split(".")[0], r.get("enc_key") or r.get("encoder")): r
                    for r in p.get("results", []) if not r.get("skipped") and not r.get("error")}
        ca, cb = cells(a), cells(b)
        for k in set(ca) & set(cb):
            ra, rb = ca[k], cb[k]
            ea, eb = ra.get("engine") or {}, rb.get("engine") or {}
            same = (ra.get("M") == rb.get("M") and ra.get("support") == rb.get("support")
                    and ea.get("split") == eb.get("split") and ea.get("mode") == eb.get("mode"))
            exact += same
            mismatch += (not same)
            if abs((ra.get("succ") or 0) - (rb.get("succ") or 0)) <= 0.05:
                succ_ok += 1
            else:
                succ_bad += 1
            if ra.get("gpap_sps") and rb.get("gpap_sps"):
                rates.append(rb["gpap_sps"] / ra["gpap_sps"])
    return {"exact": exact, "mismatch": mismatch, "succ_ok": succ_ok, "succ_bad": succ_bad,
            "gm": gm(rates), "lo": min(rates) if rates else None,
            "hi": max(rates) if rates else None, "n": len(rates)}


# ------------------------------------------------------------------ rendering

def tex_escape(s):
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def stage_encoders(d, stem):
    """Which encoders a stage's JSON contains. One table row is emitted per (family, encoder)."""
    payload = load(os.path.join(d, f"{stem}.json")) if d else None
    return encoders_in(payload) if payload else []


# The rival-configuration EVIDENCE files (eval/rivals/steps_calibration.py): each rival's
# shipped script value, its argparse default and its PAPER value for the knob, plus the paper's
# PAIRED setting, each measured once on the campaign's own card. Schema {generated, tools:
# {tool: {pinned, pinned_source, measured_best, agrees, cycles, arms: {str(value): {unique_per_s,
# wall_s, error}}}}} — no `rows` key, which is how verify_artifact tells it from an arm.
CALIBRATION_FILES = (("rivals_steps_calibration.json", "num_steps"),
                     ("rivals_steps_calibration_t50.json", "num_steps"),
                     ("rivals_lr_calibration.json", "lr"))


def rival_calibration(rivals_dir):
    """Markdown lines stating what the rivals' OTHER author-shipped values measured, or [].

    The methods paragraph says the campaign runs each rival at its shipped script setting and
    that the papers' values are "measured beside it". Until this existed the measurement lived
    only in a JSON nobody opened: the prose pointed at a file and the text never said the
    number, so a reader could not tell whether the script setting flattered us. Now the
    generator prints every arm, its ratio to the PINNED arm and the tie-break verdict, so the
    disclosure is produced by the same run that produces the tables (evidence, never
    configuration — a FLIP is printed, not acted on)."""
    import json as _json
    out = []
    for fn, knob in CALIBRATION_FILES:
        path = os.path.join(rivals_dir or "", fn)
        if not os.path.isfile(path):
            continue
        try:
            d = _json.load(open(path)) or {}
        except Exception as e:                       # noqa: BLE001 — a corrupt file is stated, not hidden
            out.append(f"* `{fn}`: unreadable ({e})")
            continue
        for tool, v in sorted((d.get("tools") or {}).items()):
            arms = v.get("arms") or {}
            pinned = v.get("pinned")
            pin = arms.get(str(pinned)) or {}
            pin_rate = pin.get("unique_per_s") or None
            # `depth` is the depth the probe RAN at (recorded by steps_calibration since
            # 2026-09-06); `cycles` was the CLI argument, which DEMOTIC's combinational probe
            # ignores, so an older file mislabelled DEMOTIC "at T=25". No depth, no claim.
            T = v.get("depth")
            out += ["", f"**{tool}** — `{knob}`" + (f" at T={T}" if T else "") +
                    f", pinned {pinned} ({v.get('pinned_source', '?')}); file `{fn}`"
                    + (f", measured {d.get('generated')}" if d.get("generated") else ""), "",
                    f"| {knob} | unique/s | wall s | vs pinned | |", "|---|---:|---:|---:|---|"]
            for k, a in arms.items():
                rate = a.get("unique_per_s")
                ratio = (f"{rate / pin_rate:.2f}x" if (rate is not None and pin_rate) else "—")
                tag = ("**pinned**" if k == str(pinned) else
                       "paper (paired)" if k.startswith("paper(") else "")
                if a.get("error"):
                    tag = (tag + " " if tag else "") + f"error: {str(a['error'])[:60]}"
                rate_s = f"{rate:,.0f}" if rate is not None else "—"
                wall_s = f"{a['wall_s']:.0f}" if a.get("wall_s") is not None else "—"
                out.append(f"| {k} | {rate_s} | {wall_s} | {ratio} | {tag} |")
            ag = v.get("agrees")
            verdict = ("tie-break HOLDS: the pinned value is the best of the tool's own values"
                       if ag is True else
                       f"**FLIP**: {v.get('measured_best')} measured faster than the pinned "
                       f"{pinned} — recorded, not acted on (change the pin deliberately, in one place)"
                       if ag is False else
                       "INCONCLUSIVE: no arm scored")
            out.append("")
            out.append(f"* {verdict}.")
    if out:
        out = ["### Rival configuration — what their other values measured on this card", "",
               "EVIDENCE, not configuration: every campaign row runs the rival at its shipped "
               "script setting (the **pinned** arm). The other arms are the values the tool's own "
               "artifacts also give the knob — its argparse default and its PAPER — each measured "
               "once, on one instance, on the campaign's card, so the text can state what the "
               "paper's setting would have given the tool. A paper arm below 1.00x means the "
               "script setting the campaign used is the BETTER one for the rival."] + out
    return out


def build(args):
    # KEY = (family stem, encoder). Every circuit instance was run through both encoders and
    # both are reported; nothing selects between them (see select_cells).
    fam_encs = [(stem, e) for stem, _, _, _ in CMS_FAMILIES
                for e in stage_encoders(args.cmsgen, stem)]
    cms = {(s, e): cms_family(args.cmsgen, s, e) for s, e in fam_encs}
    ref = {(s, e): cms_family(args.cmsgen_ref, s, e)
           for s, e in [(st, en) for st, _, _, _ in CMS_FAMILIES
                        for en in stage_encoders(args.cmsgen_ref, st)]} if args.cmsgen_ref else {}
    riv = {s: rival_family(args.rivals, s, k) for s, k, _, _ in RIVALS}
    prop = {(s, e): prop_rates(args.cmsgen, s, e) for s, e in fam_encs}
    # THE ROWS OF TABLES 1 AND 3, in print order: the converged protocol's cells (the rival
    # stages, CMSGen on the row's own CNF) first, then whatever supplementary family stages
    # exist. One list, so the two tables and the caveats describe the same rows.
    t1 = []
    for stem, label, tgt in RIVAL_CMS_SETS:
        v = rival_cms_family(args.rivals, stem, _RIVAL_KEY[stem])
        if not v:
            continue
        k = (f"rivals:{stem}", v["encoder"])
        cms[k] = v
        prop[k] = rival_prop_rates(args.rivals, stem, _RIVAL_KEY[stem])
        t1.append((k, label, v.get("depth") or "—", tgt))
    for stem, label, depth, tgt in CMS_FAMILIES:
        for enc in stage_encoders(args.cmsgen, stem):
            if cms.get((stem, enc)):
                t1.append(((stem, enc), label, depth, tgt))
    uni, uni_args = uniformity(args.uniformity)
    barb = barbarik(args.barbarik)
    rc = recheck(args.cmsgen, args.recheck) if args.cmsgen else None
    soft = load(args.soft) if args.soft else None

    expected = {}
    for (stem, _enc), v in cms.items():
        if v:
            fam = stem.split("-")[0]
            expected[fam] = max(expected.get(fam, 0), v["inst"])

    prov = next((v for v in cms.values() if v), {}) or {}
    # The commit below is when the TABLES were generated, not what measured the data — a
    # regeneration at a later HEAD must not silently re-attribute the campaign. The measured
    # engine's identity is the per-row build id, unioned over every stage.
    cores = sorted({c for v in cms.values() if v for c in v.get("cores", [])})
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, timeout=10).stdout.strip()
    except Exception:                                   # noqa: BLE001
        commit = "?"

    # Totals: `cells` and `unsound` are per-encoder counts and add up over every key, while
    # `inst`, `skipped` and `errors` describe the whole STAGE and would be double-counted by a
    # sum over encoders — so they are taken once per family.
    one_per_stem = {}
    for (stem, _enc), v in cms.items():
        if v and stem not in one_per_stem:
            one_per_stem[stem] = v
    tot = {kk: sum(v[kk] for v in cms.values() if v) for kk in ("cells", "unsound")}
    tot["refused"] = sum(v.get("errors", 0) for v in cms.values() if v)
    tot.update({kk: sum(v[kk] for v in one_per_stem.values())
                for kk in ("inst", "skipped", "errors")})

    # ============================================================ markdown
    M = [f"# GPAP — experiment section data  ({time.strftime('%Y-%m-%d')})", "",
         f"* GPU **{prov.get('gpu')}**, CPU **{prov.get('cpu') or 'not recorded'}**",
         f"* engine {', '.join(f'`{c}`' for c in cores) or '`not recorded`'} (per-row build id)"
         f"  |  tables generated at `{commit}`"
         f"  |  CMSGen patched (`--nowrite`): **{prov.get('patched')}**",
         f"* **{tot['cells'] + tot['refused']} rows: {tot['cells']} measured cells + "
         f"{tot['refused']} refused (GPAP's encoder or envelope), {tot['inst']} instance-slots, "
         f"{tot['skipped']} skipped, {tot['errors']} errors, {tot['unsound']} unsound**", "",
         "Generated by `eval/artifact/paper_tables.py`. Vacuous pins excluded from sequential geomeans; "
         "an instance where EITHER sampler emitted nothing has no finite ratio and is reported in "
         "its own column instead of the geomean.", "",
         "## Table 1 — GPAP vs CMSGen", "",
         "**The same cells as Table 2.** The protocol: each GPU rival's own published benchmark "
         "set, under the pins its table names and under our rare pin; on every cell three tools "
         "on one constraint — the rival on its circuit, GPAP on the `abc:fast` CNF of that "
         "circuit and those pins, and CMSGen on that very CNF file. `paper set` is the same "
         "ratio over the instances the rival's own paper evaluates. Any `seq-*`/`comb-*`/… rows "
         "below them are the older `gpap bench` family matrix (instances pinned our way), "
         "supplementary and printed only when present. One encoder; nothing selects between "
         "runs (a per-instance best-of keyed on the ratio would let CMSGen's number pick which "
         "of ours is published).", "",
         "Columns add up: `inst` = `n_gm` + CMSGen=0 + GPAP=0 + both=0 + no-search + tiny-space "
         "+ vacuous + unsat + CMSGen<30. "
         "Every one of those instances is counted exactly once, `wins/n` is counted over the "
         "same set the geomean is (plus the two zero columns), and the degenerate cells are "
         "screened out BEFORE the ledger rather than only out of the geomean.", "",
         "**Two rate columns, and each charges BOTH sides the same way.** `GPAP/CMSGen` is "
         "steady state over steady state: GPAP's winning-config sampling rate over CMSGen's "
         "samples per SAMPLING wall (its own `--statsevery` clock, which starts after it has "
         "read the CNF). `e2e` is end-to-end over end-to-end: GPAP's `wall_total` (parse + "
         "engine construction + ELS + warmup + entailed-clause MINING + the timed run) over "
         "CMSGen's whole subprocess wall. `pre%` is the median share of GPAP's end-to-end wall "
         "that preparation takes. Earlier revisions scored GPAP's steady state against "
         "CMSGen's subprocess wall, which charged one side for getting ready and not the other. "
         "Preparation is a real cost at these `n_time` values and does not amortise away until "
         "the run is much longer — quote `e2e` for a delivered-samples claim and the "
         "steady-state column only for the large-n regime, saying which. (Circuit ENCODING is "
         "in neither: it is reported per instance as `t_enc` and belongs to the encoder, not "
         "the sampler.)", "",
         "| instance set | depth | pins | enc | inst | n_gm | GPAP/CMSGen | paper set | e2e "
         "| pre% | wins | CMSGen=0 | GPAP=0 | both=0 | no-search | tiny-space | vacuous | unsat "
         "| CMSGen<30 | refused |" + (" ref GPU |" if ref else ""),
         "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
         + ("---:|" if ref else "")]
    for (stem, enc), label, depth, tgt in t1:
        v = cms[(stem, enc)]
        exp = expected.get(stem.split("-")[0], v["inst"])
        mark = "" if v["inst"] >= exp else f" (partial {v['inst']}/{exp})"
        # The reference-GPU column exists only when a second sweep is passed (--cmsgen-ref);
        # a reference recorded under different defaults is not a reference, so it defaults
        # to none.
        rv = ref.get((stem, enc))
        refs = (f" {rv['gm']:,.1f}x |" if rv and rv["gm"] else " — |") if ref else ""
        # No finite ratio: say WHY when the reason is CMSGen zeros (an outright-win column), and
        # just "—" when the stage's cells all left through a screen (vacuous / no-search / tiny).
        gms = (f"{v['gm']:,.1f}x" if v["gm"] else
               ("— (all CMSGen=0)" if v["inf"] else "—"))
        if v["gm"] and v["n_gm"] < 3:
            gms += f" ⚠n={v['n_gm']}"
        ps = (f"{v['gm_paper']:,.1f}x ({v['n_paper']})" if v.get("gm_paper") else "—")
        e2e = f"{v['gm_e2e']:,.1f}x" if v.get("gm_e2e") else "—"
        M.append(f"| {label}{mark} | {depth} | {tgt} | {enc or '—'} | {v['inst']} "
                 f"| {v['n_gm']} | {gms} | {ps} | {e2e} | {100 * v.get('pre_frac', 0):.0f}% "
                 f"| {v['wins']}/{v['n']} | {v['inf']} | {v['gpap0']} | {v['both0']} "
                 f"| {v['free']} | {v.get('tiny', 0)} | {v['vac']} | {v.get('unsat', 0)} "
                 f"| {v.get('cms_thin', 0)} | {v.get('errors', 0)} |{refs}")
    g0 = sorted({n for v in cms.values() if v for n in v["gpap0_names"]})
    b0 = sorted({n for v in cms.values() if v for n in v["both0_names"]})
    M += ["",
          "**Instances GPAP could not sample** (CMSGen could): "
          + (", ".join(f"`{n}`" for n in g0) if g0 else "none") + ".",
          "",
          "**Instances neither sampler could touch**: "
          + (", ".join(f"`{n}`" for n in b0) if b0 else "none")
          + " — these bound the reach of the whole comparison, not of either tool."]
    # THE CIRCUIT HANDED TO THE RIVAL, stated from the rows rather than from a fixed string. This
    # paragraph claimed "as well as the raw one ... whichever suits it" over a campaign in which
    # every row was the ABC variant (2026-09-04) — a fairness procedure asserted and not
    # performed. It now names what actually ran.
    variants = sorted({v for stem, _k, _r, _p in RIVALS
                       for (_n, v) in (_rival_rows(args.rivals, stem) or {})})
    if len(variants) > 1:
        variant_sentence = ("Each rival is handed an ABC-simplified copy of the circuit as well "
                            "as the raw one and is scored on whichever suits it.")
    elif variants == ["abc"]:
        variant_sentence = ("Every rival is handed the SAME circuit file GPAP encodes from — the "
                            "sequential-ABC-simplified netlist (`variant: abc` on every row). "
                            "The raw-netlist axis was not run in this campaign, and GPAP's "
                            "post-unroll ABC pass has no sequential equivalent the rival could "
                            "be given; that residual is a property of GPAP's encoder and is "
                            "stated, not corrected.")
    elif variants == ["raw"]:
        variant_sentence = ("Every rival is handed the raw netlist (`variant: raw` on every row) "
                            "while GPAP encodes an ABC-optimized CNF of it — an asymmetry in "
                            "the rival's disfavour that is stated, not corrected.")
    else:
        variant_sentence = "Circuit variant per row: " + (", ".join(variants) or "none") + "."
    # THE ACCEPTANCE PROBABILITIES, read off the rows rather than typed: the fixed strings this
    # paragraph carried ("0.03–0.5", "~1e-2–1e-6") were the 20260903 campaign's and wrong on
    # both ends for its successor (0.020–0.97; 0.5 down to 3.9e-8).
    pa_their = _pa_range([_rival_rows(args.rivals, st) or {}
                          for st, _k, _r, pr in RIVALS if "rare" not in st])
    pa_rare = _pa_range([_rival_rows(args.rivals, st) or {}
                         for st, _k, _r, pr in RIVALS if "rare" in st])
    M += ["", "## Table 2 — GPAP vs the GPU gradient samplers", "",
          "| rival | protocol | cells | n_gm | GPAP/rival | paper set | wins | ties "
          "| rival=0 | GPAP=0 | both=0 | u/s unmeasured | rival<30 | no-search | tiny-space "
          "| vacuous | unsat | misaligned | rival failed | out-of-envelope |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for stem, _, rival, proto in RIVALS:
        v = riv.get(stem)
        if not v:
            M.append(f"| {rival} | {proto} | — | — | not run | — | — | — | — | — | — | — "
                     f"| — | — | — | — | — | — | — | — |")
            continue
        gs = f"{v['gm']:,.1f}x" if v["gm"] else "—"
        ps = (f"{v['gm_paper']:,.1f}x ({v['n_paper']})" if v.get("gm_paper") else "—")
        # `n` in Table 1 is the geomean's instance count and EXCLUDES the zero buckets; here
        # `v["n"]` is the WIN denominator and includes them. Printing that under the same header
        # made the two tables' `n` mean different things and made this row fail an addition a
        # reader is invited to do (13 + 0 + 1 + 5 + 11 = 30 against 29 cells). Print both: the
        # geomean's count, and the win fraction over its own denominator.
        n_gm_ok = v["n"] - v["zero"] - v["gpap0"]
        M.append(f"| {rival} | {proto} | {v['cells']} | {n_gm_ok} | {gs} | {ps} | "
                 f"{v['wins']}/{v['n']} | {v.get('ties', 0)} | {v['zero']} | {v['gpap0']} | {v['both0']} "
                 f"| {v.get('nomeas', 0)} | {v.get('thin', 0)} | {v.get('free', 0)} "
                 f"| {v.get('tiny', 0)} | {v.get('vacuous', 0)} | {v.get('unsat', 0)} "
                 f"| {v.get('misaligned', 0)} | {v['errors']} | {v.get('gpap_error', 0)} |")
    M += ["",
          "The two protocols are DIFFERENT TASKS and must never be pooled: 'their pins' rows "
          "pin the 1–5 outputs the rival's own table names (rejection-easy by construction — "
          f"measured p_accept {pa_their}, printed per row), while miter/rare rows pin ALL outputs "
          f"to a rare reachable pattern (p_accept {pa_rare}). Recurrent-CircuitSAT is scored on "
          "its own depth-POOLED unique count — solutions satisfying the target at ANY unroll "
          "depth start_point..T, a superset of the fixed-depth task every other row measures "
          "(the fixed-depth-only count is recorded per row as `unique_at_T`). **The batch a rival "
          "is handed is its score**: measured, both are near-proportional to it "
          "over the whole range the card will run (DEMOTIC "
          "564x from 1k to 1M), so it is the same knob `K` is for us. Each rival is therefore "
          "granted the LARGEST rung of that ladder it can run to completion: the ladder is walked "
          "DOWN from the top and a rung is rejected only on feasibility — out of memory, refused "
          "by the driver, or not finished inside the wall budget — never on how fast it was, so "
          "exactly ONE configuration is measured per instance. It is a share-of-the-machine rule, "
          "not a search for the rival's fastest rung: unique/s stops tracking the batch once the "
          "model saturates the card, so a smaller rung CAN score higher, and because the walk "
          "stops at the first rung that completes the rival's per-rung optimum is neither sought "
          "nor reported — a cell where a shorter earlier budget had measured a smaller rung "
          "faster is disclosed in the campaign notes (RIVAL_FAIRNESS_FINDINGS §16c). "
          "(Until 2026-09-04 the walk started "
          "at our VRAM model's rung, which is calibrated at num_steps=10 while we run 1 and so "
          "under-granted DEMOTIC 1.5-3.9x on 4 of 11 ISCAS-85 cells, and a timeout was recorded "
          "AS the rival's result.) The first run of every process is discarded before anything is "
          "measured, because a cold CUDA context costs the session's first instance ~3.3x. "
          + variant_sentence + " The rungs tried are recorded per row as `batch_sweep`, "
          "the resolved budget as `vram_budget_bytes`. Their own defaults cannot serve as the rule — all three "
          "argparse defaults are `--batch_size 10` and their demo scripts are per-instance "
          "one-liners spanning 1,000 to 1,000,000 on comparably tiny circuits, which on our "
          "instances would hand them a small fraction of their achievable throughput and "
          "inflate the ratios below by the same factor. "
          "**Their `--num_steps` and `--lr` are theirs, not ours**: the batch is a share of the machine and "
          "must be granted, but the gradient-step count and the learning rate are tool settings "
          "with author-chosen values, so each rival runs at the ones its OWN artifacts specify — "
          "`num_steps` 1 and `lr` 15 for both (their `script/run_sampler_*.sh`). Their PAPERS "
          "state other values — DEMOTIC 10 iterations at `lr` 15, Recurrent 5 iterations at "
          "`lr` 50 — and the campaign runs the SHIPPED script configuration, one per tool, "
          "which is also the better of the two for each tool on this card at the depth measured "
          "(both saturate their batch on the first step; the paper values, singly and as the "
          "paired setting, are measured beside it in `rivals_steps_calibration*.json` so the "
          "text can say what they would have given the tool). Recorded per row as "
          "`steps`/`steps_source` and `lr`/`lr_source`. Nothing is swept, since a "
          "swept-and-best-of setting invites \"why stop there?\". Recurrent's `start_point` "
          "initialization search is given its own 1,200 s budget per instance, which BOUNDS the "
          "search and is in no published rate (this table carries no end-to-end column, see "
          "below); a cell that does not initialize inside the budget is reported with that "
          "budget named, never as a zero. "
          "Rows recorded on or before 2026-08-26 pinned 10 for both, which taxes DEMOTIC "
          "2.6x and Recurrent 8.5x (both saturate their batch on the first step) and is why the "
          "DEMOTIC and Recurrent rows here are far below their previously recorded values. "
          "Rival rows recorded before 2026-08-25 used a fixed 5 GiB budget and are not comparable "
          "to rows from the current harness. "
          "Constraint alignment is VERIFIED per row: miter mode asserts p(MITER=1) equals "
          "the instance's p_accept; native mode asserts bit-identical acceptance words between "
          "their index table on the original file and our reordered pins (`pin_check_ok`). "
          "**No tool is timed while delivering its solutions**: CMSGen runs under the patch's "
          "`--nowrite`, DEMOTIC counts and dedups its solutions after its own timer "
          "stops, and our timed run gathers no model row to the host (`collect=False`; the "
          "rival arm keeps EVERY delivered row as GPU-projected support bits and deduplicates "
          "them exactly, off the clock, the same quantity both rivals report — the dedup runs "
          "after the sampler returns and is timed as `distinct_s`, and the projection wall that "
          "gathered the rows is subtracted from the steady-state AND the end-to-end rate, so "
          "measuring distinctness cannot pay for itself out of the number it measures) "
          "— re-derived from the pinned sources by "
          "`eval/rivals/check_emission_symmetry.py`. The "
          "one exception is Recurrent-CircuitSAT, which copies its per-depth candidate "
          "assignments to the host inside its timed loop: the share of its `model_runtime` "
          "that costs is bounded per row by the audit script (section 4 of its report — quote "
          "the campaign's figure, it depends on the card's transfer rate) and left "
          "uncorrected so its rate is exactly the one it publishes. Deducting it would RAISE "
          "the ratios below, so those rows under-report GPAP rather than flattering it."]
    M += ["",
          "**One clock, and `ties`.** This table has NO end-to-end column, by rule (protocol "
          "v1.0, RIVAL_FAIRNESS_FINDINGS.md §17): the rivals' metric is unique solutions per "
          "second of their own `model_runtime`, which excludes their model build, and GPAP's "
          "figure excludes its construction the same way — that is the one comparison the "
          "rivals themselves define. An end-to-end ratio across an in-process sampler and a "
          "subprocess gradient tool had four definitions in three review rounds (0.8x, 7.6x, "
          "20.9x/132.0x and 11.8x/68.8x on the same rows), and on DEMOTIC's cells both tools "
          "finish the real work in under a second, so it measured process start-up; the "
          "end-to-end comparison is Table 1's, where both walls are the same kind of thing. "
          "`ties` counts cells whose win/loss sign lies inside GPAP's own seed band (the ratio "
          "at its slowest and fastest seed straddles 1.0, against a rival that ran once); "
          "`wins` excludes them, so wins + ties + losses = the win denominator."]
    cal = rival_calibration(args.rivals) if args.rivals else []
    if cal:
        M += [""] + cal
    soli = []
    if any(prop.values()):
        M += ["", "## Table 3 — propagation throughput (the engine, not the sampler)", "",
              "`impl/s` is Boolean implications per second: GPAP's from the packed-word kernel, "
              "CMSGen's from its own `--statsevery` counters. The last three columns factor the "
              "sample-rate result exactly — **engine advantage / work per sample = samples/s "
              "ratio** — because GPAP discards a conflicted trajectory (rejection sampling) "
              "where CMSGen backtracks and keeps the partial work. That identity is ALGEBRAIC, "
              "not a validation: defining work-per-sample as the quotient of the two rates "
              "makes it true by construction, so it can only fail if an aggregation mixes row "
              "sets — which is exactly the per-column-filtering error it is here to catch, and "
              "all it should be claimed to catch. The rows are Table 1's, under every one of "
              "its screens.", "",
              "| family | depth | targets | enc | n | GPAP impl/s | CMSGen impl/s "
              "| impl/s ratio | dec/s ratio | GPAP impl/sample | CMSGen impl/sample "
              "| work penalty |",
              "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for (stem, enc), label, depth, tgt in t1:
            v = prop.get((stem, enc))
            if not v:
                continue
            dec = f"{v['dec']:,.0f}x" if v["dec"] else "—"
            M.append(f"| {label} | {depth} | {tgt} | {enc or '—'} | {v['n']} | "
                     f"{v['gpap']:,.3g} | {v['cms']:,.3g} | {v['ratio']:,.0f}x | {dec} | "
                     # 3 significant figures, not 1 decimal: the identity in this table's own
                     # caption is `impl ratio / work penalty = samples/s ratio`, and a work
                     # penalty printed as "0.5x" makes a reader's hand-check miss by 8% on a
                     # number that is exact to every digit it is computed with.
                     f"{v['gper']:,.0f} | {v['cper']:,.0f} | {v['work']:,.3g}x |")
        # The flatness claim is quoted over the families with at least three instances — the same
        # set `propagation_summary.png` plots. Including the n=1/n=2 HWMCC rows widens it to
        # 1.49x, and having a figure and a table print different numbers for one claim is the
        # exact failure this file exists to prevent.
        soli = [v for v in prop.values() if v and v["n"] >= 3]
        allr = gm([v["ratio"] for v in soli]) if soli else None
        lo, hi = ((min(v["gpap"] for v in soli), max(v["gpap"] for v in soli))
                  if soli else (None, None))
    if soli:
        M += ["", f"Across every family-encoder row with at least three instances (n={len(soli)}, the set "
              f"plotted in `propagation_summary.png`) GPAP holds **{gm([v['gpap'] for v in soli]):,.3g} "
              f"implications/s** with a spread of only {hi / lo:.2f}x between the fastest and "
              f"slowest, against CMSGen's {gm([v['cms'] for v in soli]):,.3g} — a "
              f"**{allr:,.0f}x** engine advantage that is essentially independent of the "
              "benchmark. The sample-rate ratios in Table 1 vary over nearly two orders of "
              "magnitude *on top of* this flat engine number, so that variation is a property of "
              "how much propagation each instance wastes on conflicted trajectories, not of the "
              "propagator."]
    if uni:
        sub = uni_args.get("_subcube_excluded") or 0
        M += ["", "## Table 4 — uniformity vs an exact oracle", "",
              f"`alpha={uni_args.get('alpha')}`, {uni_args.get('seeds')} seeds/instance, "
              f"n = {uni_args.get('n_mult')}x|S| samples."
              + (f" {sub} full-subcube instance(s) excluded — every sampler is uniform there "
                 f"by construction (the don't-care filler, not the config, draws the bits)."
                 if sub else ""), "",
              "| sampler | kind | instances | median TV | flagged SKEW | unsound |",
              "|---|---|---:|---:|---:|---:|"]
        for (name, kind), a in sorted(uni.items(), key=lambda kv: uni_sort_key(kv[0])):
            M.append(f"| `{name}` | {kind} | {a['n']} | "
                     f"{a['tv']:.4f} | {a['reject']}/{a['n']} | {a['unsound']} |")
    if barb:
        p = barb["params"]
        n_el = len(barb["eligible"])
        setting = {(0.3, 1.8, 0.1): "the parameters CMSGen's own authors certify at (FMCAD'21 "
                                    "Table I): 36 rounds x (1 + 1 + 50) = **1,872 samples** for "
                                    "an Accept, per-round accept band **[0.05, 0.95]**",
                   (0.1, 0.9, 0.05): "barbarik's own default: 477 rounds, **70,596 samples** for "
                                     "an Accept, per-round bands about +-0.21"
                   }.get((p["epsilon"], p["eta"], p["delta"]), "a non-standard setting")
        M += ["", "## Table 4b — Barbarik CM_test (the standard certificate)", "",
              f"`eps={p['epsilon']} eta={p['eta']} delta={p['delta']}` — {setting}. "
              f"An Accept means *not caught being eta-far* (L1 >= {p['eta']}, i.e. TV >= "
              f"{p['eta'] / 2:.2f}) and holds only under Barbarik's non-adversality assumption; "
              f"a Reject is unconditional. It is not a proof of uniformity — the exact-oracle "
              f"fuzzing campaign (docs/findings/UNIFORMITY_FINDINGS.md) is the sensitive "
              f"instrument.", "",
              f"{n_el} eligible instances "
              f"({len(barb['control_rejected'])} dropped because the trusted reference is itself "
              f"Rejected on them — a non-independent sampling set, or a delta-level false "
              f"rejection of an exactly uniform sampler; either way the test cannot speak about "
              f"the instance; {len(barb['control_errored'])} dropped on a reference error).", "",
              "**A separate campaign.** Source: "
              + ", ".join(f"`{os.path.relpath(p)}`" for p in barb.get("source") or [])
              + " — its own run, on its own host and its own instances (our pins on the "
                "circuit families), not the cells of Tables 1–3 above; see that campaign's "
                "README for the host. A CM_test verdict is a property of the sampler's "
                "distribution, not of the card, so the header's GPU line does not apply here.",
              "",
              "| sampler under test | Accept | Reject | no verdict | mean bias | max bias |",
              "|---|---:|---:|---:|---:|---:|"]
        for arm, a in barb["summary"].items():
            bm = "-" if a["bias_mean"] is None else f"{a['bias_mean']:.3f}"
            bx = "-" if a["bias_max"] is None else f"{a['bias_max']:.3f}"
            M.append(f"| `{arm}` | {a['accept']}/{a['n']} | {a['reject']} | {a['error']} | "
                     f"{bm} | {bx} |")
        M += ["", "`bias` is CM_test's own per-round bucket share (rejects outside "
                  "[0.05, 0.95]); it is reported because the verdict alone hides how much "
                  "headroom an Accept had. `no verdict` = the sampler could not fill an order "
                  "barbarik placed, which is neither an Accept nor a Reject."]
    soft_stats = None
    if soft:
        rows = [r for r in (soft.get("rows") or []) if not r.get("vacuous")]
        ratios = [r.get("ratio_new_over_base") for r in rows]
        won = [r for r in rows if r.get("soft_won_baseline")]
        # A "capability loss" is where the safe portfolio finds NOTHING and soft found something:
        # a different failure from being slower, and the one that would change what a user gets.
        lost = [r for r in rows
                if (r.get("baseline(+soft)") or {}).get("models")
                and not (r.get("new(no soft)") or {}).get("models")]
        soft_stats = {"n": len(rows), "gm": gm(ratios), "won": len(won), "lost": lost,
                      "vac": len(soft.get("rows") or []) - len(rows)}
        M += ["", "## Table 5 — the price of dropping `soft`", "",
              f"Throughput of the uniformity-safe portfolio relative to one that may pick "
              f"`soft`, over {soft_stats['n']} non-vacuous instances "
              f"({soft_stats['vac']} vacuous excluded): "
              f"**{soft_stats['gm']:.3f}x**" if soft_stats["gm"] else "(no ratios)",
              "",
              f"* `soft` won the bandit on **{soft_stats['won']}/{soft_stats['n']}** instances "
              f"— on the rest the two arms are the same configuration, so the geomean understates "
              f"the effect where it actually applies.",
              f"* capability losses (safe portfolio finds nothing, soft finds something): "
              f"**{len(lost)}**"
              + (": " + ", ".join(f"`{r['instance']}`" for r in lost) if lost else "")]
        if won:
            M += ["", "| instance | with soft (samples/s) | safe (samples/s) | ratio |",
                  "|---|---:|---:|---:|"]
            for r in sorted(won, key=lambda r: r.get("ratio_new_over_base") or 0)[:12]:
                b, n_ = r.get("baseline(+soft)") or {}, r.get("new(no soft)") or {}
                M.append(f"| `{r['instance']}` | {b.get('sps', 0):,.0f} | {n_.get('sps', 0):,.0f} "
                         f"| {r.get('ratio_new_over_base') or 0:.3f}x |")
    if rc:
        M += ["", "## Table 6 — reproducibility (fresh VM, same protocol)", "",
              f"* deterministic fields (M, support, packed split, propagate variant): "
              f"**{rc['exact']}/{rc['exact'] + rc['mismatch']} exact**",
              f"* success rate within 0.05: **{rc['succ_ok']}/{rc['succ_ok'] + rc['succ_bad']}**",
              f"* throughput recheck/original: geomean **{rc['gm']:.3f}x** "
              f"({rc['lo']:.2f}–{rc['hi']:.2f}, n={rc['n']})"]

    # ============================================================ latex
    T = ["% GENERATED by eval/artifact/paper_tables.py — do not edit by hand.",
         f"% {time.strftime('%Y-%m-%d %H:%M')}  generated at commit {commit}",
         f"% measured engine: {', '.join(cores) or 'not recorded'}",
         f"% GPU {prov.get('gpu')}  CPU {prov.get('cpu')}", "",
         r"\begin{table}[t]", r"  \centering", r"  \small",
         r"  \caption{\textbf{GPAP vs CMSGen.} " +
         f"GPU {tex_escape(prov.get('gpu'))}, CPU {tex_escape(prov.get('cpu') or 'n/a')}; "
         f"{tot['cells']} cells, {tot['skipped']} skipped, {tot['errors']} errors, "
         f"{tot['unsound']} unsound. "
         r"Geomeans exclude two kinds of degenerate cell, on one principle --- a cell that "
         r"measures the \emph{instance} does not belong in a geomean about the \emph{sampler}: "
         r"vacuous-pin instances (constant outputs at $T{=}25$, so pinning is a tautology) and "
         r"\emph{free} cells, which emit samples with no decisions and no implications because "
         r"the support is unconstrained after propagation. An instance on which \emph{either} "
         r"sampler emitted nothing has no finite ratio, so it is excluded from the geomean and "
         r"counted in its own column: \emph{C$=$0} (CMSGen silent --- an outright win for us, "
         r"which makes the geomeans conservative), \emph{G$=$0} (GPAP silent --- an outright "
         r"loss), \emph{both} (neither sampler reached the instance). Every instance is counted "
         r"exactly once: $\emph{inst} = n + \text{C}{=}0 + \text{G}{=}0 + \text{both} + "
         r"\text{free} + \text{tiny} + \text{vac.} + \text{unsat} + \text{C}{<}30$ --- where "
         r"\emph{tiny} holds fewer solutions than the sample request, \emph{unsat} was PROVED "
         r"to have none at this depth and \emph{C$<$30} is a CMSGen count under the same "
         r"30-sample floor the rival side of Table~\ref{tab:rivals} is held to --- and the win "
         r"count is taken over the same rows as the "
         r"geomean plus the two zero columns. \emph{paper set} is the same ratio over the "
         r"instances the rival's own paper evaluates, \emph{e2e} charges both sides their whole "
         r"wall (GPAP parse-to-finish, CMSGen its whole subprocess; neither side's wall holds "
         r"delivery, since CMSGen writes nothing under \texttt{--nowrite}), \emph{pre\%} is "
         r"preparation's median share of GPAP's wall, and \emph{refused} counts rows GPAP's "
         r"encoder or envelope refused, which lie outside \emph{inst}. "
         r"Each encoder is reported as its own row; nothing "
         r"selects between them, since a per-instance best-of keyed on the ratio would let "
         r"CMSGen's number choose which of our runs is published. "
         r"$^{\dagger}$incomplete stage; $^{\ddagger}$geomean over fewer than three instances "
         r"(CMSGen scored zero on the rest), so it is an illustration, not an estimate.}",
         r"  \label{tab:cmsgen}",
         r"  \begin{tabular}{@{}llllrrrrrrrrrrrrrrrr@{}}", r"    \toprule",
         r"    family & depth & targets & enc & inst & $n$ & GPAP/CMSGen & paper set & e2e & "
         r"pre\% & wins & C$=$0 & G$=$0 & both & free & tiny & vac. & unsat & C$<$30 & "
         r"refused \\",
         r"    \midrule"]
    for (stem, enc), label, depth, tgt in t1:
        v = cms[(stem, enc)]
        exp = expected.get(stem.split("-")[0], v["inst"])
        mark = "" if v["inst"] >= exp else r"$^{\dagger}$"
        gtex = f"${v['gm']:,.1f}\\times$" if v["gm"] else "---"
        if v["gm"] and v["n_gm"] < 3:
            gtex += "$^{\\ddagger}$"
        pstex = (f"${v['gm_paper']:,.1f}\\times$ ({v['n_paper']})" if v.get("gm_paper") else "---")
        e2tex = f"${v['gm_e2e']:,.1f}\\times$" if v.get("gm_e2e") else "---"
        T.append(f"    {tex_escape(label)} & ${depth}$ & {tgt} & "
                 f"{tex_escape(enc or '---')} & {v['inst']}{mark} & {v['n_gm']} & "
                 f"{gtex} & {pstex} & {e2tex} & {100 * v.get('pre_frac', 0):.0f} & "
                 f"{v['wins']}/{v['n']} & {v['inf']} & {v['gpap0']} & "
                 f"{v['both0']} & {v['free']} & {v.get('tiny', 0)} & "
                 f"{v.get('vac', 0)} & {v.get('unsat', 0)} & {v.get('cms_thin', 0)} & "
                 f"{v.get('errors', 0)} \\\\")
    T += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    T += [r"\begin{table}[t]", r"  \centering", r"  \small",
          r"  \caption{\textbf{GPAP vs the GPU gradient samplers}, their metric "
          r"(unique solutions / their \texttt{model\_runtime}, which excludes their model build "
          r"and \emph{GPAP/rival} uses our steady-state rate, which likewise excludes construction --- the one clock this table has, since it is the one the rivals define; the end-to-end comparison is Table~\ref{tab:cmsgen}'s; \emph{ties} counts cells whose sign lies inside GPAP's own seed band and \emph{wins} excludes them; \emph{paper set} is the same ratio over the rival's own published instances). Rival failures (timeout / no gradient path) are "
          r"counted, never folded in as zeros. As in Table~\ref{tab:cmsgen}, an instance where "
          r"either side emitted nothing carries no finite ratio and is counted separately: "
          r"\emph{R$=$0} (rival silent), \emph{G$=$0} (GPAP silent), \emph{both}. "
          r"\emph{free} cells sampled with no search, \emph{tiny} hold fewer solutions than the "
          r"sample request, \emph{vac.} pins are tautologies, and \emph{unsat} cells were "
          r"PROVED to have no solution at this depth --- four ways of measuring the instance "
          r"rather than the sampler --- while \emph{mis.} rows failed the per-row "
          r"constraint-alignment check. All are excluded from the geomean, $n$ is what it is "
          r"taken over, and every column sums to \emph{cells}. "
          r"$^{\ddagger}$geomean over fewer than three instances.}",
          r"  \label{tab:rivals}",
          r"  \begin{tabular}{@{}llrrrrrrrrrrrrrrrrr@{}}", r"    \toprule",
          r"    rival & protocol & cells & $n$ & GPAP/rival & paper set & wins & ties & "
          r"R$=$0 & G$=$0 & both & u$/$s n$/$a & R$<$30 & free & tiny & vac. & unsat & mis. & "
          r"failed \\",
          r"    \midrule"]
    for stem, _, rival, proto in RIVALS:
        v = riv.get(stem)
        if not v:
            T.append(f"    {rival} & {tex_escape(proto)} & \\multicolumn{{17}}{{c}}{{not run}} \\\\")
            continue
        g = f"${v['gm']:,.1f}\\times$" if v["gm"] else "---"
        ps = (f"${v['gm_paper']:,.1f}\\times$ ({v['n_paper']})" if v.get("gm_paper") else "---")
        n_gm_ok = v["n"] - v["zero"] - v["gpap0"]        # cells carrying a finite ratio
        if v["gm"] and n_gm_ok < 3:
            g += "$^{\\ddagger}$"
        T.append(f"    {rival} & {tex_escape(proto)} & {v['cells']} & {n_gm_ok} & {g} & {ps} & "
                 f"{v['wins']}/{v['n']} & {v.get('ties', 0)} & {v['zero']} & {v['gpap0']} & "
                 f"{v['both0']} & "
                 f"{v.get('nomeas', 0)} & {v.get('thin', 0)} & {v.get('free', 0)} & "
                 f"{v.get('tiny', 0)} & {v.get('vacuous', 0)} & {v.get('unsat', 0)} & "
                 f"{v.get('misaligned', 0)} & {v['errors']} \\\\")
    T += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    if any(prop.values()):
        T += [r"\begin{table}[t]", r"  \centering", r"  \small",
              r"  \caption{\textbf{Propagation throughput --- the engine, isolated from the "
              r"sampler.} Boolean implications per second: GPAP's from the packed-word kernel, "
              r"CMSGen's from its own counters (\texttt{--statsevery}, "
              r"\texttt{scripts/cmsgen\_gpap\_eval.patch}). The identity "
              r"$\text{samples/s ratio} = \text{impl/s ratio} \,/\, \text{work-per-sample ratio}$ "
              r"holds on every family. It is an ALGEBRAIC identity, not a validation --- "
              r"defining work-per-sample as the quotient of the two rates makes it true by "
              r"construction, so it can only fail if an aggregation mixes row sets, which is "
              r"exactly the per-column-filtering error it is here to catch. What it does is "
              r"factor the sample-rate result into what the "
              r"GPU propagator buys and what rejection sampling spends: GPAP discards a "
              r"conflicted trajectory outright, whereas CMSGen backtracks and retains the partial "
              r"work. GPAP's implication rate is nearly constant across families; the "
              r"three-orders-of-magnitude spread in Table~\ref{tab:cmsgen} therefore comes from "
              r"the work-per-sample column, not from the propagator. $n$ counts instances where "
              r"\emph{both} samplers emitted, since a work-per-sample figure needs both; "
              r"$^{\ddagger}$marks rows where that leaves fewer than three.}",
              r"  \label{tab:prop}",
              r"  \begin{tabular}{@{}llllrrrrrrrr@{}}", r"    \toprule",
              r"    & & & & & \multicolumn{2}{c}{implications/s} & "
              r"\multicolumn{2}{c}{implications/sample} & & & \\",
              r"    \cmidrule(lr){6-7}\cmidrule(lr){8-9}",
              r"    family & depth & targets & enc & $n$ & GPAP & CMSGen & GPAP & CMSGen & "
              r"impl ratio & dec/s ratio & work penalty \\", r"    \midrule"]
        for (stem, enc), label, depth, tgt in t1:
            v = prop.get((stem, enc))
            if not v:
                continue
            # same low-n flag as Table 1: these rows carry the families where CMSGen scored
            # zero almost everywhere, so `n` is 1-2 and the row illustrates, not estimates
            nmark = "$^{\\ddagger}$" if v["n"] < 3 else ""
            dectex = f"${v['dec']:,.0f}\\times$" if v.get("dec") else "---"
            T.append(f"    {tex_escape(label)} & ${depth}$ & {tgt} & "
                     f"{tex_escape(enc or '---')} & {v['n']}{nmark} & "
                     f"${v['gpap'] / 1e9:.2f}\\!\\times\\!10^{{9}}$ & "
                     f"${v['cms'] / 1e6:.2f}\\!\\times\\!10^{{6}}$ & "
                     f"{v['gper']:,.0f} & {v['cper']:,.0f} & "
                     f"${v['ratio']:,.0f}\\times$ & {dectex} & "
                     f"${v['work']:,.3g}\\times$ \\\\")
        T += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    if uni:
        T += [r"\begin{table}[t]", r"  \centering", r"  \small",
              r"  \caption{\textbf{Uniformity against an exact oracle.} Small instances whose "
              r"entire solution set is enumerated by brute force, so the target distribution is "
              r"known; $\chi^2$ GOF over independent seeds combined by Fisher's method. "
              r"\texttt{ctl:uniform} is an exact-uniform negative control and \texttt{ctl:tv*} a "
              r"positive control at known TV --- they calibrate the instrument.}",
              r"  \label{tab:uniformity}",
              r"  \begin{tabular}{@{}llrrrr@{}}", r"    \toprule",
              r"    sampler & kind & inst & median TV & flagged skewed & unsound \\",
              r"    \midrule"]
        for (name, kind), a in sorted(uni.items(), key=lambda kv: uni_sort_key(kv[0])):
            T.append(f"    \\texttt{{{tex_escape(name)}}} & {kind} & {a['n']} & "
                     f"${a['tv']:.4f}$ & {a['reject']}/{a['n']} & {a['unsound']} \\\\")
        T += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    if rc:
        T += [r"\begin{table}[t]", r"  \centering", r"  \small",
              r"  \caption{\textbf{Reproducibility.} The same cells re-measured on a fresh VM of "
              r"the same type, protocol pinned. Deterministic fields must match exactly; "
              r"throughput is reported, not asserted.}",
              r"  \label{tab:repro}",
              r"  \begin{tabular}{@{}lr@{}}", r"    \toprule",
              r"    quantity & result \\", r"    \midrule",
              f"    deterministic fields exact & {rc['exact']}/{rc['exact'] + rc['mismatch']}"
              r" \\",
              f"    success rate within $0.05$ & {rc['succ_ok']}/"
              f"{rc['succ_ok'] + rc['succ_bad']}" + r" \\",
              f"    throughput (geomean) & ${rc['gm']:.3f}\\times$ "
              f"(${rc['lo']:.2f}$--${rc['hi']:.2f}$, $n={rc['n']}$)" + r" \\",
              r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    # ---------------------------------------------------------------- caveats
    # These travel WITH the numbers. Every one of them is a thing a referee would otherwise find
    # first, and each is derived from the data rather than remembered.
    C = []
    for (stem, enc), v in cms.items():
        if not v:
            continue
        stem = f"{stem}/{enc}" if enc else stem
        exp = expected.get(stem.split("-")[0], v["inst"])
        if v["inst"] < exp:
            C.append(f"`{stem}` is an **incomplete stage** ({v['inst']}/{exp} instances) — the "
                     f"rented-GPU budget ran out. Do not read it as a family result.")
        if v["gm"] and v["n_gm"] < 3:
            why = (f"because CMSGen scored zero on {v['inf']} of {v['n']}" if v["inf"]
                   else f"— a set of {v['inst']} instance(s) after the screens")
            C.append(f"`{stem}`: the geomean stands on **{v['n_gm']} instance(s)** {why}. "
                     f"Report the win count ({v['wins']}/{v['n']}) and the zero count, not the "
                     f"ratio.")
        if v["free"]:
            # Both geomeans can be None — a stage whose every measured cell is degenerate has
            # no ratio at all (b13 at a shallow T collapses to a 2-clause CNF under ABC), and
            # formatting None here used to crash the whole generator on such a campaign.
            size = (f" Leaving them in would read **{v['gm_withfree']:,.1f}x instead of "
                    f"{v['gm']:,.1f}x**." if (v.get("gm_withfree") and v.get("gm")) else
                    " Nothing else in the stage carries a ratio, so there is no geomean to move.")
            C.append(f"`{stem}`: {v['free']} instance(s) "
                     f"({', '.join('`' + n + '`' for n in v['free_names'])}) sample with **no "
                     f"search at all** — zero decisions, zero implications, success 1.0 — because "
                     f"after propagation the support is a free hypercube. Excluded for the same "
                     f"reason as vacuous pins." + size)
        if v["gpap0"]:
            C.append(f"`{stem}`: **GPAP emitted nothing on {v['gpap0']} instance(s) that CMSGen "
                     f"sampled** ({', '.join('`' + n + '`' for n in v['gpap0_names'])}). These are "
                     f"losses. They carry no finite ratio, so they cannot enter the geomean — "
                     f"which is exactly why they must be stated in the text.")
        if v["both0"]:
            C.append(f"`{stem}`: {v['both0']} instance(s) defeated **both** samplers "
                     f"({', '.join('`' + n + '`' for n in v['both0_names'])}) — a limit of the "
                     f"comparison, not evidence about either tool.")
        if v.get("weird"):
            C.append(f"`{stem}`: {v['weird']} cell(s) where both samplers emitted yet the row "
                     f"carries no finite `ratio` (a malformed or pre-schema row). They are in "
                     f"none of the printed columns, so the `inst` identity under Table 1 is "
                     f"short by exactly this count — fix the rows before publishing.")
        if v.get("noarm"):
            C.append(f"`{stem}`: **{v['noarm']} row(s) never ran the CMSGen arm at all** "
                     f"({', '.join('`' + n + '`' for n in v['noarm_names'])}) — a resume "
                     f"without the binary. Absent is not zero, so they are in none of the "
                     f"printed columns and the `inst` identity under Table 1 is short by "
                     f"exactly this count; re-run those cells before publishing.")
        if v.get("cms_err"):
            C.append(f"`{stem}`: **the CMSGen subprocess ERRORED on {v['cms_err']} row(s)** "
                     f"({', '.join('`' + n + '`' for n in v['cms_err_names'])}). A crashed "
                     f"baseline is not a zero; excluded from every column — re-run before "
                     f"publishing.")
    for stem, key, rival, proto in RIVALS:
        v = riv.get(stem)
        if v and v["errors"]:
            # Name the budget the rows were measured under — it is in every error reason
            # ("... within the 1200s initialization budget"), and "the budget" without a number
            # is exactly the sentence a referee cannot check.
            import re as _re
            budgets = sorted({int(m.group(1)) for r in (_rival_rows(args.rivals, stem) or {}).values()
                              for m in [_re.search(r"within the (\d+)s initialization budget",
                                                   str((r.get(key) or {}).get("error") or ""))]
                              if m})
            budget = (f"the {budgets[0]:,} s initialization budget" if len(budgets) == 1 else
                      "the initialization budget" if not budgets else
                      "the initialization budget (" + "/".join(f"{b:,} s" for b in budgets)
                      + " — MIXED, re-run to one budget before publishing)")
            C.append(f"{rival} ({proto}): **{v['errors']} of {v['cells']}** runs produced no "
                     f"measurement inside the budget (did not initialize within {budget}, "
                     f"timed out, or no gradient path — the reason is recorded per row). They "
                     f"are omitted from the geomean and never counted as zeros. This is a "
                     f"statement about OUR budget, not about the rival's capability: a cell "
                     f"that did not initialize in time is re-runnable at a longer one; which "
                     f"cells did when the budget was raised, and which still did not, is "
                     f"recorded per campaign (docs/findings/RIVAL_FAIRNESS_FINDINGS.md §11, "
                     f"§16d).")
        if v and v["gpap0"]:
            C.append(f"{rival} ({proto}): **GPAP emitted nothing on {v['gpap0']} instance(s)** "
                     f"that the rival sampled. Losses, excluded from the geomean for lack of a "
                     f"finite ratio — state them.")
        if v and v.get("misaligned"):
            C.append(f"{rival} ({proto}): **{v['misaligned']} row(s) FAILED the constraint-"
                     f"alignment check** (`pin_check_ok: false` — the two sides were not given "
                     f"the same task). Excluded from every count; re-run those cells before "
                     f"quoting the family.")
        if v and v.get("nomeas"):
            C.append(f"{rival} ({proto}): {v['nomeas']} cell(s) have no unique/s estimate — no "
                     f"kept output row to measure the distinct fraction on (on rows before "
                     f"2026-09-03: an off-timing coverage run that collected nothing) while the "
                     f"timed run did produce models. Not measured, so not comparable in either "
                     f"direction; counted apart from GPAP=0, which means we produced nothing "
                     f"at all.")
    if any(v and not v["cpu"] for v in cms.values()):
        C.append("Some stages lack CPU provenance; a GPAP/CMSGen ratio is not interpretable "
                 "without it.")
    C.append("GPAP/CMSGen ratios are **host-dependent**: CMSGen is single-core, and this host is "
             "0.41x the reference laptop (verified independently by CMSGen's own propagation and "
             "decision counters). Always quote the CPU alongside the ratio.")
    # PER STAGE AND IN BOTH DIRECTIONS, never `max` (2026-09-04). Reporting the maximum picked
    # out the one stage where screening was CONSERVATIVE and hid the three where it lifts the
    # headline: on the 2026-09-03 campaign the screens raised ISCAS-85/their-pins 1.19x,
    # ISCAS-85/rare 1.24x and seq/their-table 1.24x, and lowered seq/rare 1.42x — and only the
    # 1.42x was printed. A screen that helps you three times out of four is not a caveat, it is
    # a result, and it has to be stated as one.
    # EVERY stage whose geomean the screens moved — gated on the MOVE, not on whether the stage
    # happened to contain a vacuous pin. Gating on `vac` dropped both ISCAS-85 stages, which have
    # no vacuous cell and whose headlines the screens RAISE by 1.19x and 1.06x: the same
    # "only some stages reported" failure the max() fix was written to close, one gate along.
    effects = sorted(((k if isinstance(k, str) else "/".join(k), v["gm_all"] / v["gm"])
                      for k, v in cms.items()
                      if v and v.get("gm") and v.get("gm_all") and v["gm_all"] != v["gm"]),
                     key=lambda kv: -kv[1])
    # Named from the rows (each family's `vac_names`), not from a typed list: the list said
    # b02/b05/b07/s526 for as long as the suite had four, and the full sequential set has b19.
    # Per stage, because the screen is per PIN: b13 is a tautology under Recurrent's own target
    # and not under our rare pin, b19 the other way round.
    per_stage = [("/".join(k) if isinstance(k, tuple) else str(k), v["vac_names"])
                 for k, v in cms.items() if v and v.get("vac_names")]
    named = ("; ".join(", ".join(f"`{n}`" for n in names) + f" ({k})" for k, names in per_stage)
             if per_stage else "the vacuous-pin instances")
    C.append(f"Sequential geomeans **exclude** {named} — cells whose pinned outputs are constant "
             "for every input sequence (measured p_accept 1.0), so pinning them is a tautology.")
    if effects:
        moved = "; ".join(f"{k} {f:,.2f}x ({'the screens LOWER this headline' if f > 1 else 'the screens RAISE this headline'})"
                          for k, f in effects)
        C.append("What the degeneracy screens (vacuous pins, no-search and tiny-space cells) do "
                 "to each geomean, in **both** directions — unscreened / screened: " + moved
                 + ". A screen that raises a headline is reported here exactly as loudly as one "
                   "that lowers it.")
    # THE FAILURE REGIME, AS A BOUNDARY. Amplification (success / p_accept) is what propagation
    # buys over blind rejection on that cell; where it buys nothing AND absolute success is tiny,
    # GPAP is doing rejection sampling badly while a CDCL sampler still finds solutions. Derived
    # from fields every row already carries, so it costs no measurement — and a boundary is worth
    # more to a reader than the list of instance names it replaces.
    losses, amps = [], []
    for stem, _key, _rival, _proto in RIVALS:
        rows = _rival_rows(args.rivals, stem) or {}
        for (nm, _v), r in rows.items():
            g, pa = (r.get("gpap") or {}), r.get("p_accept")
            sps, cms = g.get("sps"), r.get("cms_sps")
            if not (sps and cms and pa):
                continue
            amp = (g.get("success") or 0.0) / pa if pa else None
            if amp:
                amps.append((sps / cms >= 1, amp, g.get("success") or 0.0, nm))
                if sps < cms:
                    losses.append((nm, amp, g.get("success") or 0.0))
    if losses:
        worst_amp = max(a for _n, a, _s in losses)
        worst_suc = max(sc for _n, _a, sc in losses)
        safe = [x for x in amps if x[0] and (x[1] <= worst_amp and x[2] <= worst_suc)]
        C.append("**Where GPAP loses, and why**: every loss to CMSGen is a cell where propagation "
                 "buys almost nothing over blind rejection AND absolute success is tiny — "
                 "amplification (success / p_accept) at most "
                 f"{worst_amp:,.2f}x together with success at most {worst_suc:.1e} "
                 f"({', '.join(f'`{n}`' for n, _a, _s in sorted(losses))}). "
                 + (f"No cell meets both conditions and still wins ({len(safe)} counter-examples), "
                    "so the boundary is descriptive rather than fitted."
                    if not safe else
                    f"{len(safe)} winning cells also meet both conditions, so this is a tendency, "
                    "not a rule.")
                 + " That is the rejection sampler's collapse regime, and it is where a CDCL "
                   "sampler should be preferred.")
    # A ratio is only as precise as its denominator. A cell where the baseline emitted a handful
    # of samples in its whole budget is a capability statement, not a measured factor.
    thin = []
    for stem, _key, _rival, _proto in RIVALS:
        for (nm, _v), r in (_rival_rows(args.rivals, stem) or {}).items():
            n_s = r.get("cms_n_samples")
            if r.get("cms_sps") and n_s is not None and 0 < n_s < MIN_SOLUTIONS:
                thin.append((nm, n_s))
    if thin:
        C.append(f"Cells where CMSGen delivered fewer than {MIN_SOLUTIONS} samples in its whole "
                 "budget — " + "; ".join(f"`{n}` ({c})" for n, c in sorted(thin))
                 + " — are SCREENED into the `CMSGen<30` column, the same floor the rival side "
                   "of Table 2 is held to, and do not enter the geomean. Read them as 'CMSGen "
                   "could not sample this', not as a measured factor.")
    # THE SIGN OF A NEAR-PARITY CELL. GPAP's row is the median-rate seed of three and carries
    # `sps_min`/`sps_max`; the rival ran once. Where the ratio at our slowest and at our fastest
    # seed straddle 1.0, the win/loss verdict is inside our own seed band, and a "wins n/n"
    # that counts it is a coin flip presented as a tally.
    straddle = []
    for stem, key, rival, proto in RIVALS:
        rows = _rival_rows(args.rivals, stem) or {}
        for name, variant, bucket, ours, rate in rival_cells(list(rows.values()), key):
            if bucket != "ok" or not rate or not ours:
                continue
            g = rows[(name, variant)].get("gpap") or {}
            lo, hi, sps = g.get("sps_min"), g.get("sps_max"), g.get("sps")
            if not (lo and hi and sps):
                continue
            r_lo, r_hi = ours * lo / sps / rate, ours * hi / sps / rate
            if min(r_lo, r_hi) < 1.0 <= max(r_lo, r_hi):
                straddle.append((rival, proto, name, ours / rate, r_lo, r_hi))
    if straddle:
        C.append("**Cells whose win/loss sign lies inside GPAP's own seed band** — the ratio at "
                 "our slowest and fastest of three seeds straddles 1.0, against a rival that ran "
                 "once: "
                 + "; ".join(f"`{n}` ({rv}, {p}: {x:.2f}x, band {a:.2f}–{b:.2f}x)"
                             for rv, p, n, x, a, b in straddle)
                 + ". Count them as ties, not as wins or losses.")
    # WHAT THE RIVAL GEOMEANS ARE MADE OF, generated per row (2026-09-06, referee round 4): a
    # headline of 21.5x that is 116x over six cells and 10x over the other fourteen has to say
    # so where the paper text will see it, and so does a row with no measured ratio at all.
    for stem, key, rival, proto in RIVALS:
        v = riv.get(stem)
        if not v:
            continue
        n_ok = v["n"] - v["zero"] - v["gpap0"]
        if v.get("gm") and n_ok >= 3:
            parts = [f"**{rival} ({proto}): what the {v['gm']:,.1f}x is made of.** "
                     f"Median cell {v['median']:,.1f}x; GPAP's seed band moves the geomean to "
                     f"{v['gm_lo']:,.1f}–{v['gm_hi']:,.1f}x; {v.get('ties', 0)} tie(s) "
                     f"(sign inside that band) are excluded from `wins`"]
            if n_ok >= 8:
                parts.append("top cells " + ", ".join(f"`{nm}` {x:,.1f}x" for x, nm in v["top"][:4])
                             + f"; without the top cell {v['gm_wo_top1']:,.1f}x, without the top "
                             f"four {v['gm_wo_top4']:,.1f}x over {n_ok - 4}")
            if v.get("small_grant"):
                parts.append(f"{len(v['small_grant'])} cell(s) measured at the rival's smallest "
                             f"surviving batch (≤10,000, the rung above rejected by the card: "
                             + ", ".join(f"`{nm}` {b:,}" for _, nm, b in v["small_grant"])
                             + f") geomean {v['gm_small']:,.1f}x, against {v['gm_rest']:,.1f}x "
                             f"over the other {v['n_rest']} — a statement about this card's "
                             f"memory as much as about the tool; quote it as 'on an L4'")
            C.append("; ".join(parts) + ".")
        elif v["cells"] and n_ok == 0:
            C.append(f"**{rival} ({proto}): NO measured ratio exists on this row** — "
                     f"{v['zero']} rival=0, {v.get('thin', 0)} rival<30, {v.get('tiny', 0)} tiny, "
                     f"{v.get('free', 0)} free, {v.get('vacuous', 0)} vacuous, "
                     f"{v['errors']} rival failed, {v.get('gpap0', 0)} GPAP=0 of {v['cells']} cells. "
                     f"Its `wins` counts the rival-zero cells; read the row as a capability "
                     f"contrast, not a rate.")
        for nm, rate in v.get("unsat_rival") or []:
            C.append(f"**`{nm}` ({rival}, {proto}) is proved UNSAT at this depth and screened, "
                     f"yet the rival scores {rate:,.0f} unique/s there** — its depth-POOLED count "
                     f"admits solutions at shallower unroll depths, which the fixed-depth task has "
                     f"none of. GPAP delivers 0 on it. It is the one cell in the campaign where "
                     f"the pooled superset produces a rival-only result, and the `unsat` column "
                     f"is where it lives; say so.")
        for nm, free, total in v.get("low_vram") or []:
            C.append(f"`{nm}` ({rival}, {proto}): GPAP's arm ran with {free:.1f} of {total:.1f} GB "
                     f"free (a rival process still releasing the card), so its batch K was sized "
                     f"on a fraction of the device. Recorded per row (`gpu.free_gb`); the cell's "
                     f"verdict may be the residency, not the instance.")
        if v.get("ladder_top"):
            C.append(f"**{rival} ({proto}): on {len(v['ladder_top'])} cell(s) the batch ladder's "
                     f"TOP rung completed first try** (" + ", ".join(f"`{nm}` {b:,}" for nm, b in v["ladder_top"])
                     + "), so the card never bound the grant — the ladder did. The rival's rate is "
                     "near-proportional to its batch, so those cells may under-grant it: re-walk "
                     "them from a taller ladder (the top is 30,000,000 since protocol v1.0; "
                     "RIVAL_FAIRNESS_FINDINGS.md §16e/§17) — this caveat is the mechanical check "
                     "that the ladder's top is above the card's reach on every scored cell.")
    C.append("Rare-target rival ratios rest on near-zero denominators and a single rival run "
             "against GPAP's 3-seed median: read them as orders of magnitude, not constants.")
    M += ["", "## Caveats that MUST appear in the text", ""] + [f"* {c}" for c in C]

    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
        open(args.out_md, "w").write("\n".join(M) + "\n")
        print(f"wrote {args.out_md}")
    if args.out_tex:
        os.makedirs(os.path.dirname(args.out_tex) or ".", exist_ok=True)
        open(args.out_tex, "w").write("\n".join(T) + "\n")
        print(f"wrote {args.out_tex}")
    print(f"\n{tot['cells']} cells | {tot['skipped']} skipped | {tot['errors']} errors | "
          f"{tot['unsound']} unsound")
    if uni:
        print(f"uniformity: {len(uni)} (sampler, kind) groups")
    if rc:
        print(f"repro: {rc['exact']}/{rc['exact'] + rc['mismatch']} exact, geomean {rc['gm']:.3f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmsgen", default=None,
                    help="`gpap bench` stage directory — the older family matrix, supplementary")
    ap.add_argument("--cmsgen-ref", default=None)
    ap.add_argument("--rivals", default=None,
                    help="the campaign directory (rivals_*.json): Tables 1, 2 and 3 of the "
                         "converged protocol are read off these rows")
    ap.add_argument("--recheck", default=None)
    ap.add_argument("--uniformity", nargs="*", default=[])
    ap.add_argument("--barbarik", nargs="*", default=[],
                    help="barbarik_gpap.py JSONL result files (Table 4b)")
    ap.add_argument("--soft", default=None)
    ap.add_argument("--out-tex", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()
    if not (args.cmsgen or args.rivals):
        ap.error("nothing to read: pass --rivals (the campaign) and/or --cmsgen (supplementary)")
    build(args)


if __name__ == "__main__":
    main()
