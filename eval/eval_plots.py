#!/usr/bin/env python3
"""Figures + summary tables for the GPAP full evaluation (eval/run_full_eval.sh).

Reads every bench results JSON in a directory (one per stage, from gpap/benchmark.py) plus the
rival head-to-head JSONs, and writes PNGs + `summary.md`: the encoder A/B, GPAP vs CMSGen per
stage and per rival stage (Table 1's row, drawn), the CMSGen emission control, work rates,
propagation_summary, amplification, the rival dumbbells (Table 2's row, drawn) and three log-log
scatters, the two all-cells ones drawn at the paper's print size as PNG + PDF. Every screen and
cell classification is IMPORTED from eval/artifact/paper_tables.py, so a figure's geomean is the
paper's over the paper's cells.

  PYTHONPATH=. .venv/bin/python eval/eval_plots.py .experiments/full_eval/<gpu> [--out-dir DIR]
"""
import argparse
import glob
import json
import math
import os
import re
import statistics as st
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from eval.artifact.paper_tables import (RIVAL_CMS_SETS, RIVALS,  # noqa: E402 — shared rules
                                        VACUOUS_PINS, _RIVAL_KEY, cms_arm_ran, rival_cells,
                                        rival_cms_cells, zero_search)

# Categorical palette: fixed slot order, never cycled, colorblind-safe; one role per hue.
C_TSEITIN, C_ABC, C_CMS, C_RIVAL = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"

# VACUOUS_PINS and zero_search are IMPORTED, never restated: a private copy of a paper rule drifts.
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
    """Title + subtitle at a FIXED point offset, not an axes fraction, which collapses on short figures."""
    import textwrap
    from matplotlib.transforms import offset_copy
    off = lambda pts: offset_copy(ax.transAxes, fig=ax.figure, x=0, y=pts, units="points")  # noqa: E731
    lines = textwrap.wrap(subtitle, 108) if subtitle else []
    # the title sits ABOVE the wrapped subtitle block, so the offset is measured from the block
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
    """The paired-magnitude form: one row per instance, a connector between the two measurements,
    a dot at each end, log x. Not grouped bars: a bar encodes magnitude by AREA from a zero
    baseline, which a log axis destroys, while the connector length reads as the ratio."""
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
    # Legend UNDER the plot: these labels are long and any in-axes corner collides with the data.
    fig.legend(frameon=False, fontsize=8.5, labelcolor=INK2, ncols=2, loc="upper center",
               bbox_to_anchor=(0.5, 0.02))
    return _save(fig, path)


# --------------------------------------------------------------------------- loading

def load_stages(d):
    """{stage: payload} for every bench JSON in `d`, whatever its `schema` (consumers use .get())."""
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
    """Paired magnitude comparison as grouped bars, log y, sorted by the per-instance ratio."""
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
    """What the encoder DID: clause count and encode wall, one measure per facet."""
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
    best = {}
    for r in rows:
        k = r["name"]
        if k not in best or r["gpap_sps"] > best[k]["gpap_sps"]:
            best[k] = r
    items = sorted(best.values(), key=lambda r: (r.get("ratio") or 0))
    # The count beside the geomean is the geomean's OWN population: a zero-scoring cell is drawn
    # but not counted.
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

    Left: implication rate per family — GPAP's is nearly FLAT (occupancy-bound, a property of the
    GPU) while CMSGen's varies with the formula. Right: samples/s ratio = (implications/s ratio)
    / (work-per-sample ratio), separating what the engine buys from what rejection costs."""
    fams = []
    for stage, payload in sorted(stages.items()):
        # One row per instance, chosen by OUR rate, never by `ratio`: an argmax on the published
        # statistic lets CMSGen's number pick which of our runs enters the figure.
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

    It answers what a geomean hides: spread ALONG a constant-ratio guide is instance difficulty,
    spread ACROSS guides is where the ratio itself varies. Same screens as Table 1, imported; a
    cell where either side emitted nothing has no log-log position and is counted in the
    subtitle, so the plot states what it cannot show."""
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
            # no per-family geomean in the label: it would POOL rare+random across depths
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


# What each non-plottable bucket is called on a figure — the buckets of
# `paper_tables.rival_cells` (Table 2) and `rival_cms_cells` (Table 1).
RIVAL_NOTE = {"rival0": "rival=0 (our outright wins)", "gpap0": "GPAP=0 (our outright losses)",
              "both0": "both=0", "nomeas": "no unique/s estimate", "free": "zero-search",
              "tiny": "tiny space", "vacuous": "vacuous pin", "unsat": "proved UNSAT",
              "misaligned": "misaligned pins", "error": "rival failed",
              "gpap_error": "GPAP refused"}
CMS_NOTE = {"cms0": "CMSGen=0 (our outright wins)", "gpap0": "GPAP=0 (our outright losses)",
            "both0": "both=0", "free": "zero-search", "tiny": "tiny space",
            "vacuous": "vacuous pin", "unsat": "proved UNSAT", "refused": "GPAP refused",
            "noarm": "no CMSGen arm", "cms_err": "CMSGen crashed", "weird": "no usable quotient"}


def _plain(proto):
    """'their target table, $T{=}25$' -> 'their target table, T=25': RIVALS labels carry TeX."""
    return (proto or "").replace("$T{=}", "T=").replace("$", "")


def _rival_payload(path):
    """(rows, stem, spec) of a rival stage file; spec = (key, rival, proto) from `RIVALS`, or
    guessed from the rows for a stage the table does not know."""
    try:
        payload = json.load(open(path))
    except Exception:  # noqa: BLE001
        return None, None, None
    rows = (payload.get("rows") if isinstance(payload, dict) else payload) or []
    stem = os.path.splitext(os.path.basename(path))[0]
    short = stem[len("rivals_"):] if stem.startswith("rivals_") else stem
    spec = {s: (key, rival, proto) for s, key, rival, proto in RIVALS}
    if short in spec:
        return rows, stem, spec[short]
    key = next((k for k in ("demotic", "rcs", "hts") if any(k in r for r in rows)), None)
    if key is None:
        return rows, stem, None
    label = {"demotic": "DEMOTIC", "rcs": "Recurrent-CircuitSAT", "hts": "HTS"}[key]
    return rows, stem, (key, label, payload.get("mode") if isinstance(payload, dict) else None)


def _ledger_note(drop, note):
    """The cells a figure could not draw, named on the figure itself, never dropped silently."""
    return ", ".join(f"{len(v)} {note.get(b, b)} ({', '.join(sorted(v))})"
                     for b, v in sorted(drop.items()))


def fig_rivals(plt, path, out_dir):
    """GPU gradient-descent rivals, their own metric (unique solutions / their model_runtime,
    Recurrent's less its host copy on rows that record `egress_s`),
    one dumbbell per cell of ONE stage — Table 2's row, drawn. Classification is
    `paper_tables.rival_cells`, IMPORTED, so the subtitle's geomean is Table 2's over Table 2's
    cells; a cell where ONE side scored zero stays at the axis floor but not in the geomean."""
    rows, stem, spec = _rival_payload(path)
    if not rows or spec is None:
        return None
    key, label, proto = spec
    # rows may carry a circuit VARIANT (raw / abc); with both present the label must say which
    multi_var = len({r.get("variant") for r in rows if r.get("variant")}) > 1
    items, drop = [], {}
    for name, variant, bucket, ours, rate in rival_cells(rows, key):
        if bucket in ("ok", "rival0", "gpap0"):
            items.append((f"{name} ({variant})" if multi_var else name, ours, rate))
        else:
            drop.setdefault(bucket, []).append(name)
    if not items:
        return None
    items.sort(key=lambda t: (t[1] / t[2] if t[2] else float("inf")))
    ratios = [x[1] / x[2] for x in items if x[1] and x[2]]
    gm = _gm(ratios)
    n_zero = len(items) - len(ratios)
    sub = (f"geomean {gm:,.1f}x over {len(ratios)} cells — Table 2's row" if gm
           else f"{len(items)} cells, none with a finite ratio")
    if n_zero:
        sub += f"; {n_zero} more drawn at the axis floor scored zero on one side"
    if drop:
        sub += "  |  not drawn: " + _ledger_note(drop, RIVAL_NOTE)
    return _dumbbell(
        plt, [x[0] for x in items], [x[2] for x in items], [x[1] for x in items], label, "GPAP",
        C_RIVAL, C_TSEITIN,
        xlabel="unique solutions/s (their own metric)",
        title=f"GPAP vs {label} — {_plain(proto)}" if proto else f"GPAP vs {label}",
        subtitle=sub, path=os.path.join(out_dir, f"{stem}.png"))


def fig_vs_cmsgen_rival(plt, path, out_dir):
    """GPAP vs CMSGen on ONE rival stage's cells — Table 1's row of the converged protocol,
    drawn: both samplers on the abc:fast CNF of the same circuit and pins, steady-state
    samples/s. Classification is `paper_tables.rival_cms_cells`, IMPORTED."""
    rows, stem, spec = _rival_payload(path)
    if not rows or spec is None:
        return None
    short = stem[len("rivals_"):] if stem.startswith("rivals_") else stem
    key = _RIVAL_KEY.get(short, spec[0])
    label, tgt = next(((lb, tg) for s, lb, tg in RIVAL_CMS_SETS if s == short), (spec[1], ""))
    depths = sorted({r.get("cycles") for r in rows if r.get("cycles") is not None})
    depth = ("T=" + "/".join(str(t) for t in depths)) if depths else ""
    multi_var = len({r.get("variant") for r in rows if r.get("variant")}) > 1
    items, drop = [], {}
    for name, variant, bucket, ours, cms in rival_cms_cells(rows, key):
        if bucket in ("ok", "cms0", "gpap0"):
            items.append((f"{name} ({variant})" if multi_var else name, ours, cms))
        else:
            drop.setdefault(bucket, []).append(name)
    if not items:
        return None
    items.sort(key=lambda t: (t[1] / t[2] if t[2] else float("inf")))
    ratios = [x[1] / x[2] for x in items if x[1] and x[2]]
    gm = _gm(ratios)
    n_zero = len(items) - len(ratios)
    sub = (f"geomean {gm:,.1f}x over {len(ratios)} cells — Table 1's row; steady state, "
           f"emission-free on both sides" if gm else f"{len(items)} cells, none with a finite ratio")
    if n_zero:
        sub += f"; {n_zero} more drawn at the axis floor scored zero on one side"
    if drop:
        sub += "  |  not drawn: " + _ledger_note(drop, CMS_NOTE)
    where = ", ".join(x for x in (label, tgt, depth) if x)
    return _dumbbell(
        plt, [x[0] for x in items], [x[2] for x in items], [x[1] for x in items],
        "CMSGen (1 CPU core, --nowrite)", "GPAP (GPU)", C_CMS, C_TSEITIN,
        xlabel="samples/s (steady state)",
        title=f"GPAP vs CMSGen — {where}", subtitle=sub,
        path=os.path.join(out_dir, f"vs_cmsgen_{short}.png"))


# The paper's figures are drawn at their print width, the LNCS text width (12.2 cm), so LaTeX
# places them 1:1 and the point size set here is the one printed. The height is the page budget.
PRINT_W, PRINT_H = 4.8, 3.5


def _save_print(fig, path):
    """A paper figure, laid out at its print size, saved at exactly that size (a tight bbox
    would crop it and LaTeX would rescale the fonts): the PNG at 300 dpi and a vector PDF beside
    it, with TrueType fonts (not Type 3) and no timestamp."""
    import matplotlib.pyplot as plt
    with plt.rc_context({"pdf.fonttype": 42}):
        fig.savefig(path, dpi=300, facecolor=SURFACE)
        fig.savefig(os.path.splitext(path)[0] + ".pdf", facecolor=SURFACE,
                    metadata={"CreationDate": None})
    plt.close(fig)
    return path


def _dots(ax):
    """Every scatter marker on screen as (x, y, radius), stroke included, padded by 1 pt."""
    dots, pt = [], ax.figure.dpi / 72
    for c in ax.collections:
        rad = ((math.sqrt(c.get_sizes()[0]) + c.get_linewidths()[0]) / 2 + 1) * pt
        dots += [(x, y, rad) for x, y in ax.transData.transform(c.get_offsets())]
    return dots


def _ratio_guides(ax, lo, hi, dots=(), boxes=()):
    """Constant-ratio diagonals from 1× up by decades, extended as far as the data spreads. Call
    it once the limits and the layout are final: a label takes its line's angle on screen and
    starts near the left end. Given `dots` (`_dots`) and `boxes` (screen Bboxes) it slides along
    its line until its rotated footprint is clear of them; a label that cannot sit clear and
    inside the axes is left out (its neighbours name it)."""
    r = ax.figure.canvas.get_renderer()
    frame = ax.get_window_extent(r)
    (x0, y0), (x1, y1) = ax.transData.transform([(lo, lo), (hi, hi)])
    a = math.atan2(y1 - y0, x1 - x0)
    cos, sin = math.cos(a), math.sin(a)

    def clear(t):
        bb = t.get_window_extent(r)
        if not (frame.x0 <= bb.x0 and bb.x1 <= frame.x1 and frame.y0 <= bb.y0
                and bb.y1 <= frame.y1) or any(bb.overlaps(b) for b in boxes):
            return False
        t.set_rotation(0)                          # the text's own frame: anchor + w x h
        flat = t.get_window_extent(r)
        t.set_rotation(math.degrees(a))
        for x, y, rad in dots:
            u = (x - flat.x0) * cos + (y - flat.y0) * sin
            v = (y - flat.y0) * cos - (x - flat.x0) * sin
            if -rad < u < flat.width + rad and -rad < v < flat.height + rad:
                return False
        return True

    k = 1
    while lo * k < hi:
        ax.plot([lo, hi], [lo * k, hi * k], "--", color=INK2 if k == 1 else GRID,
                lw=0.8 if k == 1 else 0.6, zorder=1)
        x = lo * 3
        while x < hi:                              # 2.5 pt above the line, along it
            t = ax.annotate(f"{k:,d}×", (x, x * k), fontsize=7.5, color=INK2,
                            xytext=(-2.5 * sin, 2.5 * cos), textcoords="offset points",
                            rotation=math.degrees(a), rotation_mode="anchor", va="bottom")
            if clear(t):
                break
            t.remove()
            x *= 10 ** 0.125
        k *= 10


def _row_label(name, rare, depth):
    """A legend entry in the paper tables' words, "Sequential, rare, $T$=25"; a combinational
    set (T=1, nothing unrolled) carries no depth."""
    pin = "rare" if rare else "theirs"
    return f"{name}, {pin}" + (f", $T$={depth}" if (depth or 1) > 1 else "")


def _cells_scatter(plt, groups, drop, note, xlabel, ylabel, path):
    """The two all-cells scatters, drawn for print: no title (the caption carries the
    encoding), log-log over ONE range on both axes, so 1× is the identity and a constant ratio a
    diagonal, with the legend framed in the empty corner below 1×. `groups` is
    [(label, colour, marker, hollow, xs, ys)] in the table's row order. What it does not draw is
    left to the caption and printed here, with the dots per legend row."""
    fig, ax = _fig(plt, PRINT_W, PRINT_H)
    ax.grid(True, which="major", color=GRID, lw=0.5, alpha=0.7)
    vals = [v for *_, xs, ys in groups for v in xs + ys]
    lo, hi = min(vals) * 0.4, max(vals) * 2.5
    for label, color, marker, hollow, xs, ys in groups:
        ax.scatter(xs, ys, s=20, marker=marker, facecolors="none" if hollow else color,
                   edgecolors=color if hollow else SURFACE, linewidth=0.9 if hollow else 0.5,
                   zorder=3, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi * 0.9)
    ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    ax.tick_params(which="minor", color=GRID)
    fig.tight_layout(pad=0.3)
    leg = ax.legend(loc="lower right", fontsize=8, labelcolor=INK2, frameon=True,
                    facecolor=SURFACE, framealpha=0.9, edgecolor=GRID, borderaxespad=0.3,
                    borderpad=0.4, labelspacing=0.35, handlelength=1.0, handletextpad=0.4)
    leg.get_frame().set_linewidth(0.5)
    leg.set_zorder(5)                              # above the points
    dots = _dots(ax)
    frame = leg.get_window_extent(fig.canvas.get_renderer())
    _ratio_guides(ax, lo, hi, dots=dots, boxes=[frame])
    hidden = sum(frame.x0 - d < x < frame.x1 + d and frame.y0 - d < y < frame.y1 + d
                 for x, y, d in dots)
    excl = [f"{drop[b]} {note[b]}" for b in note if drop.get(b)]
    print(f"  {os.path.basename(path)}: {len(dots)} dots ("
          + ", ".join(f"{g[0].replace('$', '')}: {len(g[4])}" for g in groups) + ")"
          + ("; not drawn: " + ", ".join(excl) if excl else "")
          + (f"; WARNING: the legend covers {hidden} dot(s), move it" if hidden else ""))
    return _save_print(fig, path)


SET_COLOR = {"demotic": C_ABC, "rcs": C_TSEITIN}
SET_NAME = {"demotic": "ISCAS-85", "rcs": "Sequential"}      # Table 1's Set column, as printed


def fig_cmsgen_sps_scatter(plt, rival_paths, out_dir):
    """Every Table 1 cell of the campaign on ONE picture: CMSGen's samples/s (x) against GpapS's
    (y), log-log — the paper's figure beside Table 1, as `sps_scatter_rivals.png` is Table 2's.
    Colour = benchmark set, marker = task, filled = the paper's depth, hollow = the T=50
    appendix; tasks are never pooled, so there is no geomean. Classification is
    `rival_cms_cells`, IMPORTED."""
    row = {s: i for i, (s, _lb, _tg) in enumerate(RIVAL_CMS_SETS)}   # Table 1's row order
    pts, drop = [], {}                             # (row, key, tgt, depth, cms, gpap)
    for path in rival_paths:
        rows, stem, spec = _rival_payload(path)
        short = (stem or "")[len("rivals_"):] if (stem or "").startswith("rivals_") else stem
        if not rows or short not in _RIVAL_KEY:
            continue                               # e.g. rivals_steps_calibration.json
        key = _RIVAL_KEY[short]
        tgt = next(tg for s, _lb, tg in RIVAL_CMS_SETS if s == short)
        for name, _v, bucket, ours, cms in rival_cms_cells(rows, key):
            if bucket == "ok":
                depth = next((r.get("cycles") for r in rows if r["name"] == name), None)
                pts.append((row[short], key, tgt, depth, cms, ours))
            else:
                drop[bucket] = drop.get(bucket, 0) + 1
    if len(pts) < 3:
        return None
    groups = []
    for g in sorted({p[:4] for p in pts}, key=lambda g: (g[0], g[3] or 0)):
        _row, key, tgt, depth = g
        sub = [p for p in pts if p[:4] == g]
        groups.append((_row_label(SET_NAME[key], tgt == "rare", depth), SET_COLOR[key],
                       "o" if tgt == "their pins" else "^", depth == 50,
                       [p[4] for p in sub], [p[5] for p in sub]))
    return _cells_scatter(plt, groups, drop, CMS_NOTE, "CMSGen samples/s", "GpapS samples/s",
                          os.path.join(out_dir, "sps_scatter_cmsgen.png"))


# a rival wears its benchmark set's colour, so the pair of scatters agrees
RIVAL_COLOR = {"DEMOTIC": C_ABC, "Recurrent-CircuitSAT": C_TSEITIN, "HTS": C_CMS}
RIVAL_NAME = {"Recurrent-CircuitSAT": "Recurrent"}           # Table 2's Rival column, as printed


def fig_rivals_sps_scatter(plt, rival_paths, out_dir):
    """Every rival cell of the campaign on ONE picture: the rival's unique solutions/s (x)
    against GpapS's (y), log-log — the rivals' own metric, and the companion to Table 2 that
    `sps_scatter_cmsgen.png` is to Table 1. Colour = rival, marker = task; the two protocols are
    never pooled, so there is no geomean. Classification is `rival_cells`, IMPORTED."""
    spec = {stem: (key, rival, proto) for stem, key, rival, proto in RIVALS}
    pts = []                                       # (rival, proto, rival_rate, gpap_rate)
    drop = {}
    for path in rival_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        stem = stem[len("rivals_"):] if stem.startswith("rivals_") else stem
        if stem not in spec:                       # e.g. rivals_steps_calibration.json
            continue
        key, rival, proto = spec[stem]
        try:
            payload = json.load(open(path))
        except Exception:  # noqa: BLE001
            continue
        rows = (payload.get("rows") if isinstance(payload, dict) else payload) or []
        for _name, _var, bucket, ours, rate in rival_cells(rows, key):
            if bucket == "ok":
                pts.append((rival, proto, rate, ours))
            else:
                drop[bucket] = drop.get(bucket, 0) + 1
    if len(pts) < 3:
        return None
    groups = []
    for stem, _key, rival, proto in RIVALS:        # RIVALS order = Table 2 row order
        sub = [p for p in pts if p[0] == rival and p[1] == proto]
        if not sub:
            continue
        m = re.search(r"T\{=\}(\d+)", proto)
        depth = int(m.group(1)) if m else None
        # the TASK decides the marker: "their ISCAS-85, our rare pin" is our task on their
        # circuits (a `startswith("their")` test drew it as the rival's own task)
        groups.append((_row_label(RIVAL_NAME.get(rival, rival), "rare" in proto, depth),
                       RIVAL_COLOR[rival], "^" if "rare" in proto else "o", depth == 50,
                       [p[2] for p in sub], [p[3] for p in sub]))
    return _cells_scatter(plt, groups, drop, RIVAL_NOTE, "Rival unique solutions/s",
                          "GpapS unique solutions/s",
                          os.path.join(out_dir, "sps_scatter_rivals.png"))


# --------------------------------------------------------------------------- summary

def write_summary(stages, out_dir, src_dir):
    lines = ["# GPAP full evaluation — summary", ""]
    gpu = next((p.get("gpu", {}).get("name") for p in stages.values() if p.get("gpu")), "?")
    cpu = next((p.get("cpu", {}) for p in stages.values() if p.get("cpu")), {})
    cpu_s = f"{cpu.get('model', '?')} ({cpu.get('count', '?')} threads)" if cpu else "**not recorded**"
    # AN INVENTORY, NOT A DISCUSSION. This file used to explain each column in prose carrying
    # dated numbers ("up to 1.9x on the rare column", "EPFL 29.0x -> 9.4x", a CPU "~0.4x as
    # fast"). Those are results, and a result written into a sentence in a source file is one
    # nothing regenerates: the campaign moves and the sentence does not. What a column MEANS is
    # a rule and lives with the generator; what it says is the paper's to discuss.
    lines += [f"GPU: **{gpu}**  |  CPU: **{cpu_s}**  |  source: `{src_dir}`", "",
              "What ran, what did not, and the aggregate per stage — for the supplementary "
              "`gpap bench` families. `eval/artifact/paper_tables.py` is authoritative for every "
              "published number, and `docs/papers/*/README.md` states what each column means.",
              "",
              "Two column rules, because the table is unreadable without them: **quote `clean`**, "
              "which applies the paper's instance screens (vacuous pins, zero-search cells); and "
              "`GPAP/CMSGen (best enc)` is per instance at its fastest encoder **by our own "
              "rate**, never by the ratio — an argmax on the published statistic would let "
              "CMSGen's number pick which of our runs is reported.", "",
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
        # One row per instance, chosen by OUR rate, never by `ratio`: an argmax on the published
        # statistic lets CMSGen's number pick which run is reported. The paper's tables select
        # nothing at all (paper_tables.select_cells).
        best = {}
        for r in rows:
            k = r["name"]
            if k not in best or (r.get("gpap_sps") or 0) > (best[k].get("gpap_sps") or 0):
                best[k] = r
        gm = _gm([r.get("ratio") for r in best.values()])
        # `clean` = the paper's rule, IMPORTED: drop vacuous pins and no-search cells.
        gm_nv = _gm([r.get("ratio") for k, r in best.items()
                     if k.split(".")[0] not in VACUOUS_PINS and not zero_search(r)])
        # THE SYMMETRIC LEDGER (paper_tables.cms_family's rule): a CMSGen-zero cell is an outright
        # win and a GPAP-zero cell an outright loss, neither with a finite ratio, so each gets its
        # own column and `wins` counts finite wins plus CMSGen's zeros.
        fin = [r["ratio"] for r in best.values()
               if r.get("ratio") and r["ratio"] > 0 and r["ratio"] != float("inf")]
        # ABSENT is not ZERO: only rows that CARRY a cms arm enter the ledger.
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
    detail = []
    for stage, payload in sorted(stages.items()):
        for r in payload.get("results", []):
            if r.get("skipped") or r.get("error"):
                detail.append(f"| {stage} | {r.get('name')} | {r.get('enc_key') or '—'} | "
                              f"{(r.get('skipped') or r.get('error'))[:110]} |")
    if detail:
        lines += ["", "## Instances not measured", "",
                  "| stage | instance | encoder | reason |", "|---|---|---|---|"] + detail
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
    rival_paths = sorted(glob.glob(os.path.join(args.results_dir, "rivals*.json")))
    # A campaign keeps its rival JSONs in their OWN directory, so a sibling is searched.
    if not stages and not rival_paths:
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
    if stages:
        for fn in (fig_cmsgen_emission, fig_prop_summary, fig_sps_scatter):
            try:
                p = fn(plt, stages, out_dir)
                if p:
                    made.append(p)
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] {fn.__name__}: {e}")
    for rp in rival_paths:
        for fn in (fig_rivals, fig_vs_cmsgen_rival):
            try:
                p = fn(plt, rp, out_dir)
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] {fn.__name__}({os.path.basename(rp)}): {e}")
                p = None
            if p:
                made.append(p)
    if rival_paths:
        for fn in (fig_rivals_sps_scatter, fig_cmsgen_sps_scatter):
            try:
                p = fn(plt, rival_paths, out_dir)
                if p:
                    made.append(p)
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] {fn.__name__}: {e}")
    if stages:
        made.append(write_summary(stages, out_dir, args.results_dir))
    print(f"{len(made)} file(s) in {out_dir}:")
    for p in made:
        print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
