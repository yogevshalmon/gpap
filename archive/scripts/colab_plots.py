#!/usr/bin/env python3
"""Plots for the GPAP cross-GPU benchmark payload (scripts/colab_benchmark.py).

make_plots(payload, out_dir) writes up to three PNGs next to the per-GPU result JSON and returns
their paths. Best-effort: if matplotlib is missing it warns and returns []. Figures:

  1. <gpu>_throughput_by_lits.png — per-L grid: GPU throughput (sets/s) vs total_lits, one line per
     soft-stop setting (baseline=gpu_sets_s, soft-stop=successful_sets/s) + a CPU reference line.
  2. <gpu>_speedup_by_M.png       — per-L grid: speedup vs CPU, vs M, one line per soft-stop setting.
  3. <gpu>_softstop_tradeoff.png  — for the deepest instance (most timed-out under soft-stop):
     successful_sets/s and timed_out_frac vs soft_stop_frac (the cutoff tradeoff curve).
"""
import math
import os


def _frac_label(f) -> str:
    return "baseline (off)" if f is None else f"ss={f:g}"


def _good_instances(payload: dict) -> list[dict]:
    return [r for r in payload.get("instances", []) if "error" not in r and r.get("l_sweep")]


def _l_entry(inst: dict, L: int):
    return next((le for le in inst["l_sweep"] if le["L"] == L), None)


def _setting(le: dict, frac):
    return next((s for s in le["settings"] if s["soft_stop_frac"] == frac), None)


def _grid(n: int):
    """Rows, cols for n subplots (≤2 cols)."""
    cols = 1 if n <= 1 else 2
    rows = math.ceil(n / cols)
    return rows, cols


def make_plots(payload: dict, out_dir: str) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"  [plot] matplotlib unavailable ({e}); skipping plots")
        return []

    insts = _good_instances(payload)
    if not insts:
        print("  [plot] no successful instances to plot")
        return []

    gpu_name = payload.get("gpu", {}).get("name", "gpu")
    slug = __import__("re").sub(r"[^a-z0-9]+", "-", gpu_name.lower()).strip("-") or "gpu"
    l_values = payload.get("l_values") or sorted({le["L"] for r in insts for le in r["l_sweep"]})
    settings = [None] + list(payload.get("soft_stop_fracs") or [])
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    # ----------------------------------------------------------------- Fig 1
    paths.append(_fig_throughput(plt, insts, l_values, settings, gpu_name,
                                 os.path.join(out_dir, f"{slug}_throughput_by_lits.png")))
    # ----------------------------------------------------------------- Fig 2
    paths.append(_fig_speedup(plt, insts, l_values, settings, gpu_name,
                              os.path.join(out_dir, f"{slug}_speedup_by_M.png")))
    # ----------------------------------------------------------------- Fig 3
    p3 = _fig_tradeoff(plt, insts, l_values, settings, gpu_name,
                       os.path.join(out_dir, f"{slug}_softstop_tradeoff.png"))
    if p3:
        paths.append(p3)
    return [p for p in paths if p]


def _y_for_setting(le, frac):
    """Throughput to plot for a setting: baseline uses raw sets/s, soft-stop uses successful/s."""
    s = _setting(le, frac)
    if not s:
        return None
    return s["gpu_sets_s"] if frac is None else s["successful_sets_s"]


def _fig_throughput(plt, insts, l_values, settings, gpu_name, path) -> str:
    rows, cols = _grid(len(l_values))
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.2 * rows), squeeze=False)
    fig.suptitle(f"{gpu_name} — GPU set-propagation throughput vs formula size (by total literals)",
                 fontsize=13, fontweight="bold")

    for idx, L in enumerate(l_values):
        ax = axes[idx // cols][idx % cols]
        pts = sorted(((r["total_lits"], r) for r in insts if _l_entry(r, L)), key=lambda t: t[0])
        if not pts:
            ax.set_visible(False)
            continue
        xs = [t[0] for t in pts]
        for frac in settings:
            ys = [_y_for_setting(_l_entry(r, L), frac) for _, r in pts]
            xy = [(x, y) for x, y in zip(xs, ys) if y]
            if not xy:
                continue
            style = dict(marker="o", ms=4, lw=1.6)
            if frac is None:
                style.update(color="black", lw=2.2)
            ax.plot([p[0] for p in xy], [p[1] for p in xy], label=_frac_label(frac), **style)
        # CPU reference (per-L cpu_sets_s; same for all settings)
        cpu = [( _l_entry(r, L)["cpu_sets_s"]) for _, r in pts]
        cxy = [(x, y) for x, y in zip(xs, cpu) if y]
        if cxy:
            ax.plot([p[0] for p in cxy], [p[1] for p in cxy], "--", color="gray",
                    lw=1.4, label="CPU (PySAT)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"L = {L} assumptions/set")
        ax.set_xlabel("total literals in formula"); ax.set_ylabel("throughput (sets/s)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
    for j in range(len(l_values), rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _speedup_for_setting(le, frac):
    s = _setting(le, frac)
    if not s:
        return None
    return s["speedup"] if frac is None else s["speedup_successful"]


def _fig_speedup(plt, insts, l_values, settings, gpu_name, path) -> str:
    rows, cols = _grid(len(l_values))
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.2 * rows), squeeze=False)
    fig.suptitle(f"{gpu_name} — speedup over sequential CPU vs formula clause count (M)",
                 fontsize=13, fontweight="bold")

    for idx, L in enumerate(l_values):
        ax = axes[idx // cols][idx % cols]
        pts = sorted(((r["M"], r) for r in insts if _l_entry(r, L)), key=lambda t: t[0])
        if not pts:
            ax.set_visible(False)
            continue
        xs = [t[0] for t in pts]
        for frac in settings:
            ys = [_speedup_for_setting(_l_entry(r, L), frac) for _, r in pts]
            xy = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if not xy:
                continue
            style = dict(marker="o", ms=4, lw=1.6)
            if frac is None:
                style.update(color="black", lw=2.2)
            ax.plot([p[0] for p in xy], [p[1] for p in xy], label=_frac_label(frac), **style)
        ax.axhline(1.0, color="gray", ls=":", lw=1, label="parity (1×)")
        ax.set_xscale("log")
        ax.set_title(f"L = {L} assumptions/set")
        ax.set_xlabel("clauses M"); ax.set_ylabel("speedup vs CPU (×)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
    for j in range(len(l_values), rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _fig_tradeoff(plt, insts, l_values, settings, gpu_name, path):
    fracs = [f for f in settings if f is not None]
    if not fracs:
        return None
    L = max(l_values)
    # Pick the instance with the largest timed-out fraction (deepest propagation) at this L.
    best = None
    for r in insts:
        le = _l_entry(r, L)
        if not le:
            continue
        tmax = max((s["timed_out_frac"] for s in le["settings"]), default=0.0)
        if best is None or tmax > best[0]:
            best = (tmax, r, le)
    if best is None or best[0] <= 0:
        return None  # nothing timed out anywhere — soft-stop never bit; tradeoff plot is moot
    _, inst, le = best

    xs = list(fracs)
    succ = [(_setting(le, f) or {}).get("successful_sets_s") for f in xs]
    tout = [(_setting(le, f) or {}).get("timed_out_frac") for f in xs]
    base = _setting(le, None)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f"{gpu_name} — soft-stop tradeoff on {inst['name'][:40]} "
                 f"(N={inst['N']}, M={inst['M']}, L={L})", fontsize=11, fontweight="bold")
    ax.plot(xs, succ, "o-", color="tab:green", label="successful sets/s")
    if base:
        ax.axhline(base["gpu_sets_s"], color="black", ls="--", lw=1.5,
                   label=f"baseline {base['gpu_sets_s']:,.0f} sets/s (no cutoff)")
    ax.set_xlabel("soft_stop_frac (higher = more aggressive cutoff)")
    ax.set_ylabel("successful sets/s", color="tab:green")
    ax.tick_params(axis="y", labelcolor="tab:green")
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(xs, [t * 100 if t is not None else None for t in tout], "s--",
             color="tab:red", label="timed-out %")
    ax2.set_ylabel("timed-out batches (%)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
