#!/usr/bin/env python3
"""Figures + summary tables for the GPAP full evaluation (eval/run_full_eval.sh).

Reads every bench results JSON in a directory (one per stage, written by gpap/benchmark.py)
plus the rival head-to-head JSONs, and writes PNGs + a markdown summary:

  encoder_ab_<stage>.png      GPAP throughput, tseitin vs abc:fast, paired per instance (the
                              headline encoder A/B) — log scale, geomean in the subtitle
  encoder_size_<stage>.png    what the encoder actually changed: CNF clauses and encode wall
  vs_cmsgen_<stage>.png       GPAP vs CMSGen, emission-free both sides
  cmsgen_emission.png         CMSGen with vs without writing its samples (the fairness control)
  work_rates_<stage>.png      implications/s and decisions/s, GPU vs CPU, log-log
  propagation_summary.png     the engine across ALL families at once: GPAP's implication rate is
                              flat (~1e9/s, 1.41x spread) while the sample-rate ratio moves three
                              orders of magnitude, and the exact factorisation of why
  amplification_<stage>.png   p_accept vs GPAP's constraint amplification (the rejection rival)
  rivals_<name>.png           unique solutions/s vs DEMOTIC / Recurrent-CircuitSAT / HTS
  summary.md                  every stage's geomeans, win counts and skips, as a table

Usage:
  PYTHONPATH=. .venv/bin/python eval/eval_plots.py .experiments/full_eval/<gpu> [--out-dir DIR]
"""
import argparse
import glob
import json
import os
import statistics as st
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from eval.artifact.paper_tables import (VACUOUS_PINS, cms_arm_ran,  # noqa: E402 — shared rules
                                        unique_unmeasured, zero_search)

# Categorical palette: fixed slot order, never cycled (blue/orange/aqua/violet). Validated
# colorblind-safe against the light chart surface (all pairs, OKLab dE>=8) — see the dataviz
# reference palette. One role per hue, kept identical across every figure so a colour always
# means the same thing: encoder A / encoder B / CMSGen / rival-or-rejection.
C_TSEITIN, C_ABC, C_CMS, C_RIVAL = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"

# VACUOUS_PINS and zero_search are IMPORTED, never restated: the figures and summary.md quote
# geomeans a reader will compare against the paper's, so the exclusion rules must be the paper's
# by construction (a private copy is a rule that drifts — see tests/test_paper_tables.py).
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4"
ENC_COLOR = {"tseitin": C_TSEITIN, "abc:fast": C_ABC, "abc:full": C_ABC, "abc:auto": C_ABC}


def _gm(xs):
    xs = [x for x in xs if x and x > 0 and x != float("inf")]
    return st.geometric_mean(xs) if xs else None


def _fig(plt, w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, axis="y", which="major", color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    return fig, ax


def _title(ax, title, subtitle=None):
    """Title + subtitle drawn above the axes at a FIXED point offset (not an axes fraction):
    these figures vary in height with the instance count, and a fractional offset collapses the
    two lines onto each other on the short ones."""
    import textwrap
    from matplotlib.transforms import offset_copy
    off = lambda pts: offset_copy(ax.transAxes, fig=ax.figure, x=0, y=pts, units="points")  # noqa: E731
    lines = textwrap.wrap(subtitle, 108) if subtitle else []
    # the title sits ABOVE the whole wrapped subtitle block (va="bottom" anchors the block's
    # bottom, so it grows upward — a 3-line subtitle used to grow straight through the title)
    ax.text(0, 1, title, transform=off(8 + 12.5 * len(lines) + (6 if lines else 0)),
            fontsize=12, fontweight="bold", color=INK, va="bottom")
    if lines:
        ax.text(0, 1, "\n".join(lines), transform=off(8), fontsize=8.5,
                color=INK2, va="bottom")


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def _short(n):
    return (n.replace(".bench", "").replace(".cnf", "").replace(".aig", "")
             .replace(".sk", "").split("-c")[0][:22])


def _dumbbell(plt, names, a_vals, b_vals, a_label, b_label, a_color, b_color, *,
              xlabel, title, subtitle=None, path=None, floor=1e-1):
    """The paired-magnitude form: one row per instance, a thin connector between the two
    measurements, a dot at each end, log x. Deliberately NOT grouped bars — a bar encodes
    magnitude by AREA from a zero baseline, which a log axis destroys; a dot encodes position,
    which a log axis is exactly right for, and the connector length reads as the ratio."""
    import numpy as np
    n = len(names)
    fig, ax = plt.subplots(figsize=(8.2, max(2.6, 0.34 * n + 1.9)))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, axis="x", which="major", color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    y = np.arange(n)
    a = [max(v or 0, floor) for v in a_vals]
    b = [max(v or 0, floor) for v in b_vals]
    for i in range(n):
        ax.plot([a[i], b[i]], [y[i], y[i]], "-", color=GRID, lw=2.0, zorder=1,
                solid_capstyle="round")
    ax.scatter(a, y, s=52, color=a_color, edgecolor=SURFACE, linewidth=1.4, zorder=3,
               label=a_label)
    ax.scatter(b, y, s=52, color=b_color, edgecolor=SURFACE, linewidth=1.4, zorder=3,
               label=b_label)
    ax.set_yticks(y)
    ax.set_yticklabels([_short(x) for x in names], fontsize=8, color=INK2)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    _title(ax, title, subtitle)
    # Legend UNDER the plot: these labels are long ("CMSGen --samplefile (stock: writes every
    # sample)") and any in-axes corner collides with either the dots or the subtitle.
    fig.legend(frameon=False, fontsize=8.5, labelcolor=INK2, ncols=2, loc="upper center",
               bbox_to_anchor=(0.5, 0.02))
    return _save(fig, path)


# --------------------------------------------------------------------------- loading

def load_stages(d):
    """{stage: payload} for every bench JSON in `d` (rival files are loaded apart).

    Any payload with a `results` list is read, whatever its `schema`: the schema records when a
    COLUMN appeared or went away, and every consumer here already reads its columns with .get().
    """
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem.startswith("rivals"):
            continue
        try:
            payload = json.load(open(p))
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {p}: {e}")
            continue
        if "results" not in payload:
            continue
        out[payload.get("tag") or stem] = payload
    return out


def rows_of(payload):
    return [r for r in payload.get("results", []) if not r.get("error") and not r.get("skipped")]


def paired_by_encoder(rows):
    """[(instance, {enc_key: row})] for instances present under >= 2 encoders, run order kept."""
    by = {}
    for r in rows:
        if r.get("enc_key"):
            by.setdefault(r["name"], {})[r["enc_key"]] = r
    return {k: v for k, v in by.items() if len(v) >= 2}


# --------------------------------------------------------------------------- figures

def fig_encoder_ab(plt, stage, payload, out_dir):
    """Paired magnitude comparison -> grouped bars, log y (rates span decades), sorted by the
    per-instance ratio so the win/lose split is the shape of the chart."""
    rows = rows_of(payload)
    pairs = paired_by_encoder(rows)
    if not pairs:
        return None
    encs = [e for e in (payload.get("encoders") or []) if any(e in v for v in pairs.values())]
    if len(encs) < 2:
        return None
    base, other = encs[0], encs[1]
    items = [(n, v[base], v[other]) for n, v in pairs.items() if base in v and other in v]
    if not items:
        return None
    items.sort(key=lambda t: (t[2]["gpap_sps"] / t[1]["gpap_sps"]) if t[1]["gpap_sps"] else 0)
    gm = _gm([y["gpap_sps"] / x["gpap_sps"] for _, x, y in items if x["gpap_sps"]])
    wins = sum(1 for _, x, y in items if y["gpap_sps"] > x["gpap_sps"])
    return _dumbbell(
        plt, [n for n, _, _ in items], [x["gpap_sps"] for _, x, _ in items],
        [y["gpap_sps"] for _, _, y in items], base, other,
        ENC_COLOR.get(base, C_TSEITIN), ENC_COLOR.get(other, C_ABC),
        xlabel="GPAP throughput (samples/s, steady state) — right is faster",
        title=f"Encoder A/B — {stage}",
        subtitle=(f"{other} vs {base}: geomean {gm:.2f}x, wins {wins}/{len(items)} instances"
                  if gm else None),
        path=os.path.join(out_dir, f"encoder_ab_{stage}.png"))


def fig_encoder_size(plt, stage, payload, out_dir):
    """What the encoder DID: clause count and encode wall. Two facets, one measure each —
    never a second y-axis on the throughput chart."""
    pairs = paired_by_encoder(rows_of(payload))
    encs = [e for e in (payload.get("encoders") or []) if any(e in v for v in pairs.values())]
    if len(encs) < 2:
        return None
    base, other = encs[0], encs[1]
    items = [(n, v[base], v[other]) for n, v in pairs.items() if base in v and other in v]
    if not items:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.grid(True, which="both", color=GRID, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
    xs = [x["M"] for _, x, _ in items]
    ys = [y["M"] for _, _, y in items]
    lo, hi = min(xs + ys) * 0.7, max(xs + ys) * 1.4
    axes[0].plot([lo, hi], [lo, hi], "--", color=INK2, lw=1.0, label="equal size")
    axes[0].scatter(xs, ys, s=34, color=C_ABC, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlim(lo, hi); axes[0].set_ylim(lo, hi)
    axes[0].set_xlabel(f"clauses — {base}", fontsize=9, color=INK2)
    axes[0].set_ylabel(f"clauses — {other}", fontsize=9, color=INK2)
    shrink = _gm([x["M"] / y["M"] for _, x, y in items if y["M"]])
    axes[0].set_title(f"CNF size  (geomean {shrink:.2f}x fewer clauses)" if shrink else "CNF size",
                      fontsize=10, color=INK, loc="left")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
    ea = [x.get("encode_s") or 0 for _, x, _ in items]
    eb = [y.get("encode_s") or 0 for _, _, y in items]
    import numpy as np
    idx = np.arange(len(items))
    axes[1].bar(idx - 0.19, ea, 0.36, color=ENC_COLOR.get(base, C_TSEITIN), label=base)
    axes[1].bar(idx + 0.19, eb, 0.36, color=ENC_COLOR.get(other, C_ABC), label=other)
    axes[1].set_ylabel("encode wall (s)", fontsize=9, color=INK2)
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels([n.split(".")[0][:12] for n, _, _ in items], rotation=45,
                            ha="right", fontsize=7, color=INK2)
    fmt = (lambda s: f"{s:.1f}s") if max(sum(ea), sum(eb)) < 100 else (lambda s: f"{s:.0f}s")
    axes[1].set_title(f"encode cost  ({fmt(sum(ea))} -> {fmt(sum(eb))} total)", fontsize=10,
                      color=INK, loc="left")
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK2, ncols=2)
    fig.suptitle(f"What the ABC encoder changes — {stage}", fontsize=12, fontweight="bold",
                 color=INK, x=0.02, ha="left")
    return _save(fig, os.path.join(out_dir, f"encoder_size_{stage}.png"))


def fig_vs_cmsgen(plt, stage, payload, out_dir):
    rows = [r for r in rows_of(payload) if r.get("cms_sps") is not None]
    if not rows:
        return None
    # one row per instance: the best encoder (the configuration a user would actually ship)
    best = {}
    for r in rows:
        k = r["name"]
        if k not in best or r["gpap_sps"] > best[k]["gpap_sps"]:
            best[k] = r
    items = sorted(best.values(), key=lambda r: (r.get("ratio") or 0))
    # The count printed beside the geomean is the geomean's OWN population: a zero-scoring cell
    # (ratio 0.0 / inf / None) stays on the figure — the dumbbell shows which side was silent —
    # but cannot enter the geomean, so "over N instances" must not count it (fig_rivals rule).
    ratios = [r.get("ratio") for r in items
              if r.get("ratio") and r["ratio"] > 0 and r["ratio"] != float("inf")]
    gm = _gm(ratios)
    mode = {r.get("cms_mode") for r in items}
    mtxt = ("emission-free on both sides" if mode == {"nowrite"}
            else "CMSGen writes its samples (stock build — handicapped)")
    n_txt = (f"over {len(ratios)} instances" if len(ratios) == len(items)
             else f"over {len(ratios)} of {len(items)} instances (the rest scored zero on a side)")
    return _dumbbell(
        plt, [r["name"] for r in items], [r["cms_sps"] for r in items],
        [r["gpap_sps"] for r in items], "CMSGen (1 CPU core)", "GPAP (GPU)", C_CMS, C_TSEITIN,
        xlabel="samples/s (steady state)",
        title=f"GPAP vs CMSGen — {stage}",
        subtitle=(f"geomean {gm:.1f}x {n_txt}; {mtxt}" if gm else mtxt),
        path=os.path.join(out_dir, f"vs_cmsgen_{stage}.png"))


def fig_cmsgen_emission(plt, stages, out_dir):
    """The fairness control: CMSGen's own rate with and without emitting its samples."""
    items = []
    for stage, payload in stages.items():
        for r in rows_of(payload):
            if r.get("cms_sps") and r.get("cms_sps_write") and r.get("cms_mode") == "nowrite":
                items.append((f"{r['name'].split('.')[0]}", r["cms_sps"], r["cms_sps_write"]))
    if not items:
        return None
    items.sort(key=lambda t: (t[1] / t[2] if t[2] else 0))
    items = items[:28]
    gm = _gm([x[1] / x[2] for x in items if x[2]])
    return _dumbbell(
        plt, [x[0] for x in items], [x[2] for x in items], [x[1] for x in items],
        "--samplefile (stock: writes every sample)", "--nowrite (generation only)",
        C_RIVAL, C_CMS,
        xlabel="CMSGen samples/s",
        title="CMSGen: the cost of emitting samples",
        subtitle=(f"writing full N-variable assignments costs it {gm:.1f}x throughput (geomean); "
                  f"GPAP's timed run emits nothing, so the headline uses --nowrite" if gm else None),
        path=os.path.join(out_dir, "cmsgen_emission.png"))


def fig_work_rates(plt, stage, payload, out_dir):
    """CDCL-comparable work: unit-prop implications/s and decisions/s, GPU vs single CPU core."""
    rows = [r for r in rows_of(payload)
            if r.get("implications_per_s") and r.get("cms_implications_per_s")]
    if not rows:
        return None
    fig, ax = _fig(plt, 6.4, 5.6)
    ax.grid(True, which="both", color=GRID, lw=0.6, alpha=0.7)
    xs = [r["cms_implications_per_s"] for r in rows]
    ys = [r["implications_per_s"] for r in rows]
    # one filter for BOTH coordinates, or a row with one side missing shifts the pairing
    dec = [(r["cms_decisions_per_s"], r["decisions_per_s"]) for r in rows
           if r.get("cms_decisions_per_s") and r.get("decisions_per_s")]
    xd, yd = ([d[0] for d in dec], [d[1] for d in dec])
    lo = min(xs + ys + xd + yd) * 0.5
    hi = max(xs + ys + xd + yd) * 2
    ax.plot([lo, hi], [lo, hi], "--", color=INK2, lw=1.0, label="parity (y = x)")
    ax.scatter(xs, ys, s=38, color=C_TSEITIN, edgecolor=SURFACE, linewidth=1.2, zorder=3,
               label="unit-prop implications/s")
    ax.scatter(xd, yd, s=38, color=C_ABC, marker="^", edgecolor=SURFACE, linewidth=1.2, zorder=3,
               label="branch decisions/s")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("CMSGen (1 CPU core)", fontsize=9, color=INK2)
    ax.set_ylabel("GPAP (GPU)", fontsize=9, color=INK2)
    gm = _gm([y / x for x, y in zip(xs, ys) if x])
    _title(ax, f"Propagation work rate — {stage}",
           f"implications/s geomean {gm:,.0f}x the CPU sampler" if gm else None)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right")
    return _save(fig, os.path.join(out_dir, f"work_rates_{stage}.png"))


def fig_prop_summary(plt, stages, out_dir):
    """The engine claim, across every family at once, and what the sampler does with it.

    Left: implication rate per family. GPAP's is nearly FLAT (the propagator is occupancy-bound,
    so its rate is a property of the GPU, not of the instance) while CMSGen's varies with the
    formula. Right: the exact factorisation of the headline,

        samples/s ratio  =  (implications/s ratio)  /  (work-per-sample ratio)

    which separates what the engine buys from what rejection sampling spends — GPAP throws away a
    conflicted trajectory, CMSGen backtracks and keeps the partial work. The sample-rate spread
    across families is three orders of magnitude on top of a flat engine number, so it belongs to
    the right-hand factor.
    """
    fams = []
    for stage, payload in sorted(stages.items()):
        # One row per instance, chosen by OUR rate — never by `ratio`: keying the argmax on the
        # published statistic lets CMSGen's number pick which of our runs enters the figure
        # (the same rule fig_vs_cmsgen and write_summary already follow).
        best = {}
        for r in rows_of(payload):
            k = r["name"].split(".")[0]
            if k not in best or (r.get("gpap_sps") or 0) > (best[k].get("gpap_sps") or 0):
                best[k] = r
        keep = []
        for k, r in best.items():
            if k in VACUOUS_PINS:
                continue
            a, b = r.get("implications_per_s"), r.get("cms_implications_per_s")
            s, t = r.get("gpap_sps"), r.get("cms_sps")
            if all(x and x > 0 for x in (a, b, s, t)):
                keep.append((a, b, s, t))
        if len(keep) >= 3:
            fams.append((stage,
                         _gm([a for a, _, _, _ in keep]), _gm([b for _, b, _, _ in keep]),
                         _gm([a / b for a, b, _, _ in keep]),
                         _gm([(a / s) / (b / t) for a, b, s, t in keep])))
    if len(fams) < 2:
        return None
    names = [f[0] for f in fams]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.7)
        for sp in ax.spines.values():
            sp.set_color(GRID)
    fig.patch.set_facecolor(SURFACE)
    x = range(len(fams))
    w = 0.38
    axes[0].bar([i - w / 2 for i in x], [f[1] for f in fams], w, color=C_TSEITIN, label="GPAP (GPU)")
    axes[0].bar([i + w / 2 for i in x], [f[2] for f in fams], w, color=C_CMS,
                label="CMSGen (1 CPU core)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("implications / s", fontsize=9, color=INK2)
    spread = max(f[1] for f in fams) / min(f[1] for f in fams)
    axes[0].set_title(f"Propagation rate is a property of the engine, not the benchmark\n"
                      f"GPAP varies only {spread:.2f}x across families",
                      fontsize=10.5, color=INK, loc="left")
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2)

    axes[1].bar([i - w / 2 for i in x], [f[3] for f in fams], w, color=C_TSEITIN,
                label="engine advantage (impl/s ratio)")
    axes[1].bar([i + w / 2 for i in x], [f[4] for f in fams], w, color=C_RIVAL,
                label="rejection cost (work per sample)")
    axes[1].plot(list(x), [f[3] / f[4] for f in fams], "o-", color=C_ABC, lw=1.6, ms=5,
                 label="net = samples/s ratio")
    axes[1].axhline(1.0, color=INK2, ls="--", lw=1.0)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("x (log)", fontsize=9, color=INK2)
    axes[1].set_title("What the engine buys, what rejection sampling spends\n"
                      "left bar / right bar = the published ratio, exactly",
                      fontsize=10.5, color=INK, loc="left")
    axes[1].legend(frameon=False, fontsize=8.5, labelcolor=INK2)
    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8, color=INK2)
        ax.tick_params(colors=INK2, labelsize=8)
    fig.tight_layout()
    return _save(fig, os.path.join(out_dir, "propagation_summary.png"))


FAMILY_COLOR = {"seq": C_TSEITIN, "comb": C_ABC, "hwmcc": C_CMS, "epfl": C_RIVAL,
                "sk": "#b0b02a"}


def fig_sps_scatter(plt, stages, out_dir):
    """Every measured cell of the campaign on ONE picture: CMSGen samples/s (x) against GPAP
    samples/s (y), log-log, one dot per cell, colored by FAMILY.

    The per-family tables answer "what is the geomean"; this answers the question a geomean
    hides — on a family we beat 100x, how are the instances actually DISTRIBUTED around that
    ratio? The constant-ratio guides (1x/10x/100x/1000x diagonals) make the answer readable:
    a family's spread ALONG a guide is instance difficulty (both samplers slow down together),
    spread ACROSS guides is where the ratio itself varies. Marker = target (circle rare,
    triangle random; square where the stage has no target axis, e.g. sk).

    Same screens as the paper's Table 1, imported: vacuous pins and zero-search cells are out
    (a cell that measures the instance is not a point about the sampler), armless rows are out
    (absent is not zero), and a cell where either side emitted nothing has no log-log position
    — those are the tables' zero columns, counted in the subtitle so the plot states what it
    cannot show.
    """
    pts = []                                       # (family, target, cms_sps, gpap_sps)
    n_deg = n_zero = 0
    for stage, payload in sorted(stages.items()):
        fam = stage.split("-")[0]
        tgt_default = (payload.get("args") or {}).get("target")
        for r in rows_of(payload):
            if not cms_arm_ran(r):
                continue
            k = r["name"].split(".")[0]
            if k in VACUOUS_PINS or zero_search(r):
                n_deg += 1
                continue
            g, c = r.get("gpap_sps") or 0, r.get("cms_sps") or 0
            if g <= 0 or c <= 0:
                n_zero += 1
                continue
            pts.append((fam, r.get("target") or tgt_default, c, g))
    if len(pts) < 3:
        return None
    fig, ax = _fig(plt, 7.6, 6.4)
    ax.grid(True, which="both", color=GRID, lw=0.6, alpha=0.7)
    lo = min(min(p[2] for p in pts), min(p[3] for p in pts)) * 0.4
    hi = max(max(p[2] for p in pts), max(p[3] for p in pts)) * 2.5
    # constant-ratio guides: the frame the eye reads the cloud against
    for k, lab in ((1, "1x"), (10, "10x"), (100, "100x"), (1000, "1,000x"), (10000, "10,000x")):
        ax.plot([lo, hi], [lo * k, hi * k], "--", color=INK2 if k == 1 else GRID,
                lw=1.0 if k == 1 else 0.8, zorder=1)
        y_at = lo * 3 * k
        if lo < y_at < hi:
            ax.annotate(lab, (lo * 3, y_at), fontsize=7.5, color=INK2,
                        xytext=(2, 3), textcoords="offset points", rotation=29)
    MARKER = {"rare": "o", "random": "^", None: "s"}
    fams = sorted({p[0] for p in pts})
    for fam in fams:
        col = FAMILY_COLOR.get(fam, INK2)
        mine = [p for p in pts if p[0] == fam]
        first = True
        for tgt, mk in MARKER.items():
            sub = [p for p in mine if p[1] == tgt]
            if not sub:
                continue
            # no per-family geomean in the label: it would POOL rare+random across depths,
            # which no table does — the distribution on the guides is the figure's number
            ax.scatter([p[2] for p in sub], [p[3] for p in sub], s=44, color=col,
                       marker=mk, edgecolor=SURFACE, linewidth=1.1, zorder=3,
                       label=(f"{fam}  (n={len(mine)})" if first else None))
            first = False
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi * 0.9)
    ax.set_xlabel("CMSGen samples/s (1 CPU core, emission-free)", fontsize=9, color=INK2)
    ax.set_ylabel("GPAP samples/s (GPU, steady state)", fontsize=9, color=INK2)
    excl = []
    if n_zero:
        excl.append(f"{n_zero} cell(s) with a zero side (the tables' zero columns)")
    if n_deg:
        excl.append(f"{n_deg} degenerate cell(s) (vacuous pin / zero-search)")
    _title(ax, "Every cell at once — samples/s, GPAP vs CMSGen",
           "one dot per (instance, stage); circle = rare target, triangle = random; "
           "diagonals = constant ratio"
           + ("; not plottable: " + ", ".join(excl) if excl else ""))
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right")
    return _save(fig, os.path.join(out_dir, "sps_scatter_all.png"))


def fig_amplification(plt, stage, payload, out_dir):
    """The rejection-sampling rival: where does propagation actually earn its keep?"""
    rows = [r for r in rows_of(payload)
            if r.get("sim_baseline") and r.get("amplification")]
    if not rows:
        return None
    xs = [r["sim_baseline"]["p_accept"] or 1e-9 for r in rows]
    ys = [max(r["amplification"], 1e-3) for r in rows]
    fig, ax = _fig(plt, 6.6, 5.0)
    ax.grid(True, which="both", color=GRID, lw=0.6, alpha=0.7)
    ax.axhline(1.0, color=INK2, ls="--", lw=1.0)
    ax.scatter(xs, ys, s=40, color=C_RIVAL, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    for r, x, y in zip(rows, xs, ys):                 # selective labels: the extremes only
        if y > 100 or y < 1:
            ax.annotate(r["name"].split(".")[0], (x, y), textcoords="offset points",
                        xytext=(5, 3), fontsize=7, color=INK2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("p(accept) of the pinned target — rejection sampling's whole story",
                  fontsize=9, color=INK2)
    ax.set_ylabel("GPAP amplification (success / p_accept)", fontsize=9, color=INK2)
    _title(ax, f"Constraint amplification — {stage}",
           "above the dashed line, BCP beats simulate-and-filter on hit rate")
    return _save(fig, os.path.join(out_dir, f"amplification_{stage}.png"))


def fig_rivals(plt, path, out_dir):
    """GPU gradient-descent rivals, their own metric (unique solutions / their model_runtime)."""
    try:
        payload = json.load(open(path))
    except Exception:  # noqa: BLE001
        return None
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not rows:
        return None
    key = "demotic" if any("demotic" in r for r in rows) else ("rcs" if any("rcs" in r for r in rows) else "hts")
    label = {"demotic": "DEMOTIC", "rcs": "Recurrent-CircuitSAT", "hts": "HTS"}[key]
    # rows may carry a circuit VARIANT (raw / abc — both sides got the same circuit); when both
    # are present the label must say which, or the two rows collide on one axis tick.
    multi_var = len({r.get("variant") for r in rows if r.get("variant")}) > 1
    # A MISSING unique/s estimate is not a zero: it means the off-timing coverage run collected
    # nothing, on an instance whose timed run may have delivered models at many times the rival's
    # rate (b11-c25 at a rare pin). Coercing it to 0 drew GPAP at the axis floor — the picture of
    # an outright loss — for a cell nobody measured. The rule is imported, not restated, so this
    # figure and Table 2 cannot disagree about which cells are comparable.
    items, unmeasured = [], []
    for r in rows:
        rv = r.get(key) or {}
        if not rv or "error" in rv:
            continue
        tag = f"{r['name']} ({r['variant']})" if multi_var else r["name"]
        if unique_unmeasured(r.get("gpap") or {}):
            unmeasured.append(tag)
            continue
        items.append((tag, (r.get("gpap") or {}).get("unique_per_s_est") or 0,
                      rv.get("unique_per_s") or 0))
    if not items:
        return None
    items.sort(key=lambda t: (t[1] / t[2] if t[2] else float("inf")))
    # A cell where EITHER side scored zero has no finite ratio, so it stays on the figure (it is a
    # real outcome, and the dumbbell shows which side was silent) but cannot enter the geomean.
    # The count beside the geomean is therefore its own, never the plotted-cell count.
    ratios = [x[1] / x[2] for x in items if x[1] and x[2]]
    gm = _gm(ratios)
    stem = os.path.splitext(os.path.basename(path))[0]
    mode = payload.get("mode") if isinstance(payload, dict) else None
    sub = (f"geomean {gm:,.1f}x over {len(ratios)} of {len(items)} plotted cells"
           + ("" if len(ratios) == len(items) else "; the rest scored zero on one side")) if gm \
        else ""
    if mode:
        sub += ("  |  their own pins (rejection-easy by construction)" if mode == "native"
                else "  |  our fixed-depth rare-target instance, handed to them as a miter")
    if unmeasured:      # never drop cells silently — name them on the figure itself
        sub += (f"  |  omitted, no unique/s estimate (coverage run collected nothing): "
                f"{', '.join(sorted(unmeasured))}")
    return _dumbbell(
        plt, [x[0] for x in items], [x[2] for x in items], [x[1] for x in items], label, "GPAP",
        C_RIVAL, C_TSEITIN,
        xlabel="unique solutions/s (their metric: uniques / model_runtime)",
        title=f"GPAP vs {label}", subtitle=sub or None,
        path=os.path.join(out_dir, f"{stem}.png"))


# --------------------------------------------------------------------------- summary

def write_summary(stages, out_dir, src_dir):
    lines = ["# GPAP full evaluation — summary", ""]
    gpu = next((p.get("gpu", {}).get("name") for p in stages.values() if p.get("gpu")), "?")
    cpu = next((p.get("cpu", {}) for p in stages.values() if p.get("cpu")), {})
    cpu_s = f"{cpu.get('model', '?')} ({cpu.get('count', '?')} threads)" if cpu else "**not recorded**"
    lines += [f"GPU: **{gpu}**  |  CPU: **{cpu_s}**  |  source: `{src_dir}`", "",
              "The CPU matters as much as the GPU here: CMSGen is single-core, so the ratio moves",
              "with the host. The same instance set has been measured at roughly double the ratio",
              "on a machine whose CPU is ~0.4x as fast — none of that difference is GPAP.",
              "Never quote a ratio without the CPU beside it.", "",
              "`GPAP/CMSGen` is per INSTANCE at its best encoder — the configuration a user would",
              "ship. The per-ROW figure (every instance x every encoder) is given alongside it;",
              "they differ whenever one encoder collapses on an instance the other handles, which",
              "is a real per-instance effect, not an aggregate one.", "",
              "`wins` is counted the paper's way: finite ratios >= 1 plus the instances CMSGen",
              "scored ZERO on (our outright wins), over finite + both zero columns — the zeros",
              "are their own columns, never silently inside a win count.", "",
              "**clean** applies the two exclusions the paper generator applies, for the same",
              "reason — a cell that measures the INSTANCE is not a measurement of the sampler:",
              f"(1) VACUOUS PINS ({', '.join(sorted(VACUOUS_PINS))}), whose last-cycle outputs are",
              "constant at T=25 so pinning them is a tautology (up to 1.9x on the rare column); and",
              "(2) ZERO-SEARCH cells, which emit samples with no decisions AND no implications",
              "because the support is free after propagation (EPFL 29.0x -> 9.4x). **Quote the",
              "clean column.** `eval/artifact/paper_tables.py` is authoritative; this is the same rule.", "",
              "| stage | rows | inst | skipped | errors | encoders | GPAP/CMSGen (best enc) | "
              "**clean** | (per row) | wins | CMSGen=0 | GPAP=0 | encoder A/B "
              "| CMSGen emission cost |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for stage, payload in sorted(stages.items()):
        rows = rows_of(payload)
        allr = payload.get("results", [])
        n_skip = sum(1 for r in allr if r.get("skipped"))
        n_err = sum(1 for r in allr if r.get("error"))
        encs = payload.get("encoders") or []
        gm_row = _gm([r.get("ratio") for r in rows])
        # One row per instance, chosen by OUR rate — the configuration a user would ship, and
        # the same rule fig_vs_cmsgen uses. Never by `ratio`: that is the published statistic,
        # so keying the argmax on it lets CMSGen's number pick which of our runs is reported
        # (measured: 1.24-1.33x on the sequential families). The paper's tables go further and
        # report each encoder as its own row, selecting nothing — see paper_tables.select_cells;
        # this summary keeps a per-instance view because the figures are per-instance.
        best = {}
        for r in rows:
            k = r["name"]
            if k not in best or (r.get("gpap_sps") or 0) > (best[k].get("gpap_sps") or 0):
                best[k] = r
        gm = _gm([r.get("ratio") for r in best.values()])
        # `clean` = the paper's rule, IMPORTED (paper_tables.zero_search + VACUOUS_PINS): drop
        # vacuous pins and cells that sampled without searching — the ratio describes the
        # instance there, not the sampler.
        gm_nv = _gm([r.get("ratio") for k, r in best.items()
                     if k.split(".")[0] not in VACUOUS_PINS and not zero_search(r)])
        # THE SYMMETRIC LEDGER (paper_tables.cms_family's rule): a CMSGen-zero cell (ratio inf)
        # is an outright win and a GPAP-zero cell an outright loss — neither has a finite ratio,
        # so `wins` counts finite wins plus CMSGen's zeros, over finite + both zero columns, and
        # each zero kind gets its own column. Counting inf inside a plain `ratio >= 1` (what
        # this did) shows their zeros as wins while never showing the zeros as zeros.
        fin = [r["ratio"] for r in best.values()
               if r.get("ratio") and r["ratio"] > 0 and r["ratio"] != float("inf")]
        # ABSENT is not ZERO: only rows that CARRY a cms arm enter the ledger — a GPAP-only
        # sweep (no --cmsgen) must read "no comparison", never "wins n/n over CMSGen=0 x n".
        cmsr = [r for r in best.values() if "cms_n_samples" in r or r.get("cms_sps") is not None]
        n_c0 = sum(1 for r in cmsr
                   if (r.get("n_models") or 0) > 0 and not (r.get("cms_n_samples") or 0))
        n_g0 = sum(1 for r in cmsr
                   if (r.get("cms_n_samples") or 0) > 0 and not (r.get("n_models") or 0))
        n_win = sum(1 for x in fin if x >= 1) + n_c0
        n_cmp = len(fin) + n_c0 + n_g0
        pairs = paired_by_encoder(rows)
        ab = "—"
        if len(encs) >= 2 and pairs:
            base, other = encs[0], encs[1]
            items = [(v[base], v[other]) for v in pairs.values() if base in v and other in v]
            r = _gm([b["gpap_sps"] / a["gpap_sps"] for a, b in items if a["gpap_sps"]])
            if r:
                wins = sum(1 for a, b in items if b["gpap_sps"] > a["gpap_sps"])
                ab = f"{other} {r:.2f}x vs {base} (wins {wins}/{len(items)})"
        pen = _gm([r.get("cms_write_penalty") for r in rows])
        nv = f"**{gm_nv:.1f}x**" if (gm_nv and gm and abs(gm_nv - gm) > 1e-9) else "—"
        led = (f"{n_win}/{n_cmp} | {n_c0} | {n_g0}" if cmsr
               else "— | — | —")                          # no cms arm ran: no ledger to print
        lines.append(f"| {stage} | {len(rows)} | {len(best)} | {n_skip} | {n_err} | "
                     f"{'+'.join(encs) or '—'} | {f'{gm:.1f}x' if gm else '—'} | {nv} | "
                     f"{f'{gm_row:.1f}x' if gm_row else '—'} | "
                     f"{led} | {ab} | "
                     f"{f'{pen:.1f}x' if pen else '—'} |")
    # skipped/error detail — a sweep that silently dropped instances is not a result
    detail = []
    for stage, payload in sorted(stages.items()):
        for r in payload.get("results", []):
            if r.get("skipped") or r.get("error"):
                detail.append(f"| {stage} | {r.get('name')} | {r.get('enc_key') or '—'} | "
                              f"{(r.get('skipped') or r.get('error'))[:110]} |")
    if detail:
        lines += ["", "## Instances not measured", "",
                  "| stage | instance | encoder | reason |", "|---|---|---|---|"] + detail
    # engine provenance: which packed split / kernel variant produced the numbers
    seen = {}
    for payload in stages.values():
        for r in rows_of(payload):
            e = r.get("engine") or {}
            if e:
                seen[(e.get("split"), e.get("mode"))] = seen.get((e.get("split"), e.get("mode")), 0) + 1
    if seen:
        lines += ["", "## Engine provenance (sum-trick packed kernel)", "",
                  "| packed split (count\\|sum bits) | propagate variant | rows |", "|---|---|---|"]
        for (split, mode), n in sorted(seen.items(), key=lambda kv: -kv[1]):
            # the split is literally "16|16" — escape it or it becomes two table cells
            lines.append(f"| {str(split).replace('|', chr(92) + '|')} | {mode} | {n} |")
    p = os.path.join(out_dir, "summary.md")
    with open(p, "w") as f:
        f.write("\n".join(lines) + "\n")
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results_dir", help="directory of stage JSONs (eval/run_full_eval.sh's OUT)")
    ap.add_argument("--out-dir", default=None, help="default: <results_dir>/plots")
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.results_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        sys.exit(f"matplotlib unavailable ({e}) — cannot plot")
    stages = load_stages(args.results_dir)
    if not stages:
        sys.exit(f"no results JSONs in {args.results_dir}")
    made = []
    for stage, payload in sorted(stages.items()):
        for fn in (fig_encoder_ab, fig_encoder_size, fig_vs_cmsgen, fig_work_rates,
                   fig_amplification):
            try:
                p = fn(plt, stage, payload, out_dir)
            except Exception as e:  # noqa: BLE001 — one bad figure never blocks the rest
                print(f"  [warn] {fn.__name__}({stage}): {type(e).__name__}: {e}")
                p = None
            if p:
                made.append(p)
    for fn in (fig_cmsgen_emission, fig_prop_summary, fig_sps_scatter):
        try:
            p = fn(plt, stages, out_dir)
            if p:
                made.append(p)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] {fn.__name__}: {e}")
    for rp in sorted(glob.glob(os.path.join(args.results_dir, "rivals*.json"))):
        try:
            p = fig_rivals(plt, rp, out_dir)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] fig_rivals({os.path.basename(rp)}): {e}")
            p = None
        if p:
            made.append(p)
    made.append(write_summary(stages, out_dir, args.results_dir))
    print(f"{len(made)} file(s) in {out_dir}:")
    for p in made:
        print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
