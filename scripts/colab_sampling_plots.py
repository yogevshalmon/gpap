#!/usr/bin/env python3
"""Plots for the GPAP cross-GPU SAMPLING benchmark payload (scripts/colab_sampling_benchmark.py).

make_plots(payload, out_dir) writes up to three PNGs next to the per-GPU result JSON and returns
their paths. Best-effort: if matplotlib is missing it warns and returns []. Figures:

  1. <gpu>_kscaling.png        — samples/s vs K (log-log), one line per swept instance, with this
     GPU's auto-K marked. The A100 story: throughput ∝ K, so bigger VRAM sits further up the line.
  2. <gpu>_vs_cmsgen.png       — per niche instance: GPAP vs CMSGen samples/s (grouped bars) + the
     GPAP/CMSGen ratio annotated. The headline throughput win.
  3. <gpu>_quality_parity.png  — 2-wise coverage %, GPAP vs CMSGen scatter on the y=x parity line:
     points on the diagonal = equal quality (the speed win is not from worse coverage).
"""
import os
import re


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (name or "gpu").lower()).strip("-") or "gpu"


def make_plots(payload: dict, out_dir: str) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"  [plot] matplotlib unavailable ({e}); skipping plots")
        return []

    gpu_name = payload.get("gpu", {}).get("name", "gpu")
    slug = _slug(gpu_name)
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    p1 = _fig_kscaling(plt, payload, gpu_name,
                       os.path.join(out_dir, f"{slug}_kscaling.png"))
    if p1:
        paths.append(p1)
    p2 = _fig_vs_cmsgen(plt, payload, gpu_name,
                        os.path.join(out_dir, f"{slug}_vs_cmsgen.png"))
    if p2:
        paths.append(p2)
    p3 = _fig_quality(plt, payload, gpu_name,
                      os.path.join(out_dir, f"{slug}_quality_parity.png"))
    if p3:
        paths.append(p3)
    return paths


def _fig_kscaling(plt, payload, gpu_name, path):
    sweeps = payload.get("ksweeps") or []
    sweeps = [s for s in sweeps if len([p for p in s["points"] if p["samples_per_s"] > 0]) >= 2]
    if not sweeps:
        return None
    auto_K = payload.get("aggregate", {}).get("auto_K", 0)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_title(f"{gpu_name} — sampling throughput scales with K (batch size)",
                 fontsize=12, fontweight="bold")
    for s in sweeps:
        pts = [p for p in s["points"] if p["samples_per_s"] > 0]
        xs = [p["K"] for p in pts]; ys = [p["samples_per_s"] for p in pts]
        ax.plot(xs, ys, "o-", lw=1.8, ms=5, label=s["name"])
    # Reference: ideal linear scaling (slope 1 in log-log) anchored at the first point of sweep 0.
    p0 = [p for p in sweeps[0]["points"] if p["samples_per_s"] > 0][0]
    xref = [p["K"] for s in sweeps for p in s["points"] if p["samples_per_s"] > 0]
    xlo, xhi = min(xref), max(xref)
    ax.plot([xlo, xhi], [p0["samples_per_s"] * xlo / p0["K"], p0["samples_per_s"] * xhi / p0["K"]],
            "--", color="gray", lw=1.2, label="ideal ∝ K")
    if auto_K:
        ax.axvline(auto_K, color="tab:red", ls=":", lw=1.5, label=f"this GPU auto-K={auto_K:,}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("K (parallel trajectories / batch size)")
    ax.set_ylabel("throughput (samples/s)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _fig_vs_cmsgen(plt, payload, gpu_name, path):
    rows = [r for r in payload.get("instances", [])
            if "error" not in r and r.get("in_niche") and "cms_sps" in r]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r.get("ratio") or 0, reverse=True)
    names = [r["name"].split(".sk")[0] + ".sk" for r in rows]
    gp = [r["gpap_sps"] for r in rows]; cm = [r["cms_sps"] for r in rows]
    import numpy as np
    x = np.arange(len(rows)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(rows)), 5))
    ax.set_title(f"{gpu_name} — GPAP vs CMSGen sampling throughput (niche)",
                 fontsize=12, fontweight="bold")
    ax.bar(x - w / 2, gp, w, label="GPAP (GPU)", color="tab:blue")
    ax.bar(x + w / 2, cm, w, label="CMSGen (CPU)", color="tab:orange")
    for xi, r in zip(x, rows):
        if r.get("ratio"):
            ax.annotate(f"{r['ratio']:.1f}×", (xi, max(r['gpap_sps'], r['cms_sps'])),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8,
                        fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("samples/s (end-to-end wall)")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _fig_quality(plt, payload, gpu_name, path):
    rows = [r for r in payload.get("instances", [])
            if "error" not in r and "twise_cms" in r and r.get("in_niche")]
    if not rows:
        return None
    g = [r["twise_gpap"] for r in rows]; c = [r["twise_cms"] for r in rows]
    lo = min(min(g), min(c)) * 0.95; hi = max(max(g), max(c)) * 1.02
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f"{gpu_name} — 2-wise coverage: GPAP vs CMSGen (parity)",
                 fontsize=12, fontweight="bold")
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1.2, label="y = x (equal quality)")
    ax.scatter(c, g, s=40, color="tab:blue", zorder=3)
    for r in rows:
        ax.annotate(r["name"].split(".sk")[0], (r["twise_cms"], r["twise_gpap"]),
                    textcoords="offset points", xytext=(4, 2), fontsize=7)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("CMSGen 2-wise coverage (%)")
    ax.set_ylabel("GPAP 2-wise coverage (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
