"""Comparison figures: measured vs reconstructed |chi(R)|, and the structure."""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from exafs_gpu.scattering import ELEMENTS

C_DATA = "#1a1a1a"
C_FIT = "#0072B2"
C_IDEAL = "#D55E00"
C_3 = "#009E73"
INK, INK2 = "#1a1a1a", "#555555"
BAND = "#eeeeee"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "grid.color": "#dddddd",
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def fit_comparison(path, R, data, model, ideal, mask, title, r_factor,
                   r_factor_ideal, shells=None, spread=None):
    """Three panels (one per edge): data, reconstruction, and residual."""
    n = data.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(2.75 * n, 3.9), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.6, 1.0], hspace=0.08,
                                              wspace=0.28))
    lo, hi = R[mask].min(), R[mask].max()
    for e in range(n):
        ax = axes[0, e]
        ax.axvspan(lo, hi, color=BAND, zorder=0, lw=0)
        if spread is not None:
            ax.fill_between(R, data[e] - spread[e], data[e] + spread[e],
                            color=C_DATA, alpha=0.13, lw=0, zorder=1)
        ax.plot(R, data[e], color=C_DATA, lw=1.4, zorder=3, label="measured")
        ax.plot(R, ideal[e], color=C_IDEAL, lw=1.0, ls=(0, (4, 1.6)), zorder=2,
                label=f"ideal FCC (R={r_factor_ideal:.4f})")
        ax.plot(R, model[e], color=C_FIT, lw=1.3, ls=(0, (1.4, 1.4)), zorder=4,
                label=f"fit (R={r_factor:.4f})")
        if shells is not None:
            for s in shells[:3]:
                ax.axvline(s, color=INK2, lw=0.6, ls=":", zorder=1)
        ax.set_xlim(0.8, 6.0)
        ax.set_title(f"{ELEMENTS[e]} edge", fontsize=8.5, pad=4)
        ax.grid(zorder=0); ax.set_axisbelow(True)
        if e == 0:
            ax.set_ylabel(r"$|\chi(R)|$   (a.u.)")
            ax.legend(loc="upper right", handlelength=1.9, labelspacing=0.25,
                      borderaxespad=0.3, framealpha=0.9)
        ax2 = axes[1, e]
        ax2.axvspan(lo, hi, color=BAND, zorder=0, lw=0)
        ax2.axhline(0, color=INK2, lw=0.7)
        ax2.plot(R, model[e] - data[e], color=C_FIT, lw=1.0)
        ax2.set_xlabel(r"$R$  (Å)")
        ax2.grid(zorder=0); ax2.set_axisbelow(True)
        if e == 0:
            ax2.set_ylabel("fit − data")
    fig.suptitle(title, fontsize=9.5, y=1.005)
    fig.savefig(path)
    plt.close(fig)


def rdf_figure(path, r, g, g_tot, shells, title, wc, wc_band=None):
    """Partial RDFs, plus the Warren-Cowley values with their systematic band."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9),
                             gridspec_kw=dict(width_ratios=[1.55, 1.0], wspace=0.32))
    ax = axes[0]
    colors = {(0, 0): "#0072B2", (1, 1): "#D55E00", (2, 2): "#009E73",
              (0, 1): "#56B4E9", (0, 2): "#CC79A7", (1, 2): "#E69F00"}
    for (A, B), c in colors.items():
        ax.plot(r, g[A, B], color=c, lw=1.2,
                label=f"{ELEMENTS[A]}–{ELEMENTS[B]}")
    ax.plot(r, g_tot, color=INK, lw=1.4, ls=(0, (4, 1.6)), label="total")
    for s in shells[:3]:
        ax.axvline(s, color=INK2, lw=0.6, ls=":")
    ax.set_xlabel(r"$r$  (Å)"); ax.set_ylabel(r"$g_{AB}(r)$")
    ax.set_xlim(1.8, 5.2)
    ax.legend(ncol=2, handlelength=1.6, labelspacing=0.22, columnspacing=1.0,
              borderaxespad=0.3)
    ax.grid(zorder=0); ax.set_axisbelow(True)
    ax.set_title("Partial RDFs", fontsize=8.5, pad=4)

    ax = axes[1]
    labels = [f"$\\alpha_{{{ELEMENTS[i]}{ELEMENTS[j]}}}$"
              for i, j in zip(*np.triu_indices(3))]
    y = np.arange(len(labels))[::-1]
    if wc_band is not None:
        lo, hi = np.asarray(wc_band[0]), np.asarray(wc_band[1])
        ax.barh(y, hi - lo, left=lo, height=0.55, color="#cccccc",
                label="calibration band")
    ax.plot(np.asarray(wc), y, "o", color=C_FIT, ms=5, zorder=3, label="fitted")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("Warren–Cowley $\\alpha$ (shell 1)")
    ax.set_xlim(-0.55, 0.55)
    ax.grid(axis="x", zorder=0); ax.set_axisbelow(True)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.legend(loc="lower center", ncol=2, borderaxespad=0.3, framealpha=0.95,
              fontsize=6.8)
    ax.set_title(r"$\alpha$, shell 1", fontsize=8.5, pad=4)
    fig.suptitle(title, fontsize=9.5, y=1.02)
    fig.savefig(path)
    plt.close(fig)


def pressure_summary(path, cal, per_pressure, wc_band):
    """Lattice constant, sigma^2 and the WC band against pressure."""
    P = sorted(per_pressure)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.5), gridspec_kw=dict(wspace=0.34))
    a = [per_pressure[p]["a0"] for p in P]
    nn = [per_pressure[p]["nn_distance"] for p in P]
    s2 = [per_pressure[p]["sigma2_nn"] for p in P]
    ax = axes[0]
    ax.plot(P, nn, "o-", color=C_FIT, lw=1.4, ms=5)
    ax.set_xlabel("pressure (GPa)"); ax.set_ylabel(r"$r_{\rm NN}$  (Å)")
    ax.set_title(r"$r_{\rm NN}$", fontsize=8.5, pad=4)
    ax.grid(zorder=0); ax.set_axisbelow(True)
    ax = axes[1]
    ax.plot(P, np.array(s2) * 1e3, "o-", color=C_IDEAL, lw=1.4, ms=5)
    ax.set_xlabel("pressure (GPa)")
    ax.set_ylabel(r"$\sigma^2_{\rm NN}$  ($10^{-3}$ Å$^2$)")
    ax.set_title(r"$\sigma^2_{\rm NN}$", fontsize=8.5, pad=4)
    ax.grid(zorder=0); ax.set_axisbelow(True)
    ax = axes[2]
    labels = [f"$\\alpha_{{{ELEMENTS[i]}{ELEMENTS[j]}}}$"
              for i, j in zip(*np.triu_indices(3))]
    x = np.arange(len(labels))
    for n, p in enumerate(P):
        lo = np.array(wc_band[p]["wc_min"]); hi = np.array(wc_band[p]["wc_max"])
        ax.bar(x + (n - 1.5) * 0.2, hi - lo, bottom=lo, width=0.18,
               color=["#0072B2", "#D55E00", "#009E73", "#CC79A7"][n],
               label=f"{p} GPa", alpha=0.85)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.5, rotation=30)
    ax.set_ylabel(r"$\alpha$ range")
    ax.set_title(r"$\alpha$ range", fontsize=8.5, pad=4)
    ax.legend(fontsize=6.5, ncol=2, borderaxespad=0.2)
    ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)

    fig.savefig(path)
    plt.close(fig)


def location_map(path, P, samples, values, label, title):
    """Fitted quantity against sample index - spatial variation across the map."""
    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    v = np.asarray(values, float)
    ax.plot(samples, v, "o-", color=C_FIT, lw=1.0, ms=3.5)
    med = np.median(v)
    ax.axhline(med, color=C_IDEAL, lw=1.0, ls=(0, (4, 1.6)),
               label=f"median {med:.4f}")
    ax.set_xlabel("sample index (map position)"); ax.set_ylabel(label)
    ax.set_title(title, fontsize=9)
    ax.grid(zorder=0); ax.set_axisbelow(True)
    ax.legend(borderaxespad=0.3)
    fig.savefig(path)
    plt.close(fig)


def sro_by_shell(path, shells, alpha, elements, title=None, band1=None,
                 void=None, note=None, nshell=4):
    """Warren-Cowley parameters per shell, one bar per ordered pair.

    Matches the layout of EXAFS_RDF_RMC's sro_by_shell.png so the two
    pipelines can be read side by side: shells across the x axis, the nine
    ordered pairs as grouped bars, clustering above zero and ordering below.

    `band1` (lo, hi) draws the shell-1 calibration systematic as an error bar,
    and `void` greys out pairs with no independent data behind them.
    """
    nsp = len(elements)
    pairs = [(a, b) for a in range(nsp) for b in range(nsp)]
    names = [f"{elements[a]}-{elements[b]}" for a, b in pairs]
    order = np.argsort(names)
    ns = min(nshell, len(shells), len(alpha))
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    fig.subplots_adjust(right=0.80)

    ylim = max(0.02, 1.18 * float(abs(np.asarray(alpha[:ns])).max()))
    ax.axhspan(0, ylim, color="#fdf0f0", zorder=0, lw=0)
    ax.axhspan(-ylim, 0, color="#f0f2fb", zorder=0, lw=0)
    ax.text(0.988, 0.972, "Clustering", transform=ax.transAxes, ha="right",
            va="top", fontsize=9, color="#c25f5f")
    ax.text(0.988, 0.028, "Ordering", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=9, color="#7d86cc")

    w = 0.088
    cmap = plt.get_cmap("tab10")
    for slot, pi in enumerate(order):
        a, b = pairs[pi]
        vals = [alpha[s][a][b] for s in range(ns)]
        x = np.arange(ns) + (slot - (len(pairs) - 1) / 2) * w
        ax.bar(x, vals, width=w * 0.92, zorder=3, color=cmap(slot % 10),
               edgecolor="none", label=names[pi])

    ax.axhline(0, color=INK, lw=1.1, zorder=4)
    ax.set_xticks(np.arange(ns))
    ax.set_xticklabels([f"{s + 1}\n({shells[s]:.2f} Å)" for s in range(ns)])
    ax.set_xlabel("Shell number")
    ax.set_ylabel(r"$\alpha$ (Warren–Cowley)")
    ax.set_ylim(-ylim, ylim)
    ax.set_xlim(-0.55, ns - 0.45)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.015, 0.5),
              framealpha=1.0, handlelength=1.3, labelspacing=0.45,
              borderaxespad=0.0, title="pair", title_fontsize=8)
    ax.set_title(title or "Short-Range Order Parameters by Shell", fontsize=10.5,
                 pad=8)
    fig.savefig(path)
    plt.close(fig)
