"""Comparison figures: measured vs reconstructed chi(R), and the structure."""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .system import wc_labels
from .dataio import axis_label, point_label

# one colour per unordered pair, like pairs first, then the cross pairs
PAIR_COLOURS = ["#0072B2", "#D55E00", "#009E73", "#56B4E9", "#CC79A7",
                "#E69F00", "#000000", "#999999", "#8C564B", "#17BECF",
                "#BCBD22", "#E377C2", "#2CA02C", "#9467BD", "#FF7F0E"]


def pair_order(nsp):
    """Unordered pairs with the like pairs first, then the unlike ones."""
    like = [(a, a) for a in range(nsp)]
    unlike = [(a, b) for a in range(nsp) for b in range(a + 1, nsp)]
    return like + unlike


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


def _channel_titles(n, labels, elements=None):
    if labels is not None:
        return list(labels)
    els = tuple(elements) if elements is not None else ()
    if n == len(els):
        return [f"{e} edge" for e in els]
    return [f"channel {i}" for i in range(n)]


def fit_comparison(path, R, data, model, ideal, mask, title, r_factor,
                   r_factor_ideal, shells=None, spread=None, labels=None,
                   elements=None):
    """One column per channel: data, reconstruction, and residual.

    Real arrays are |chi(R)| (two rows). Complex arrays - a fit to chi(k)
    data - add a row with Re chi(R), where the phase information lives, and
    the residual row then shows Re (solid) and Im (dotted).
    """
    n = data.shape[0]
    cplx = np.iscomplexobj(data)
    titles = _channel_titles(n, labels, elements)
    nrow = 3 if cplx else 2
    heights = [2.2, 1.7, 1.0] if cplx else [2.6, 1.0]
    fig, axes = plt.subplots(nrow, n, figsize=(2.75 * n, 1.55 * sum(heights)),
                             sharex=True, squeeze=False,
                             gridspec_kw=dict(height_ratios=heights, hspace=0.08,
                                              wspace=0.28))
    lo, hi = R[mask].min(), R[mask].max()
    mag = (np.abs(data), np.abs(model), np.abs(ideal))
    for e in range(n):
        ax = axes[0, e]
        ax.axvspan(lo, hi, color=BAND, zorder=0, lw=0)
        if spread is not None and not cplx:
            ax.fill_between(R, data[e] - spread[e], data[e] + spread[e],
                            color=C_DATA, alpha=0.13, lw=0, zorder=1)
        ax.plot(R, mag[0][e], color=C_DATA, lw=1.4, zorder=3, label="measured")
        ax.plot(R, mag[2][e], color=C_IDEAL, lw=1.0, ls=(0, (4, 1.6)), zorder=2,
                label=f"ideal FCC (R={r_factor_ideal:.4f})")
        ax.plot(R, mag[1][e], color=C_FIT, lw=1.3, ls=(0, (1.4, 1.4)), zorder=4,
                label=f"fit (R={r_factor:.4f})")
        if shells is not None:
            for s in shells[:3]:
                ax.axvline(s, color=INK2, lw=0.6, ls=":", zorder=1)
        ax.set_xlim(0.8, 6.0)
        ax.set_title(titles[e], fontsize=8.5, pad=4)
        ax.grid(zorder=0); ax.set_axisbelow(True)
        if e == 0:
            ax.set_ylabel(r"$|\chi(R)|$   (a.u.)")
            ax.legend(loc="upper right", handlelength=1.9, labelspacing=0.25,
                      borderaxespad=0.3, framealpha=0.9)
        if cplx:
            axr = axes[1, e]
            axr.axvspan(lo, hi, color=BAND, zorder=0, lw=0)
            axr.axhline(0, color=INK2, lw=0.7)
            axr.plot(R, data[e].real, color=C_DATA, lw=1.2, zorder=3)
            axr.plot(R, ideal[e].real, color=C_IDEAL, lw=0.9, ls=(0, (4, 1.6)), zorder=2)
            axr.plot(R, model[e].real, color=C_FIT, lw=1.2, ls=(0, (1.4, 1.4)), zorder=4)
            axr.grid(zorder=0); axr.set_axisbelow(True)
            if e == 0:
                axr.set_ylabel(r"Re $\chi(R)$")
        ax2 = axes[-1, e]
        ax2.axvspan(lo, hi, color=BAND, zorder=0, lw=0)
        ax2.axhline(0, color=INK2, lw=0.7)
        resid = model[e] - data[e]
        if cplx:
            ax2.plot(R, resid.real, color=C_FIT, lw=1.0, label="Re")
            ax2.plot(R, resid.imag, color=C_FIT, lw=0.9, ls=(0, (1.2, 1.2)), label="Im")
            if e == 0:
                ax2.legend(loc="upper right", ncol=2, borderaxespad=0.2, fontsize=6.8)
        else:
            ax2.plot(R, resid, color=C_FIT, lw=1.0)
        ax2.set_xlabel(r"$R$  (Å)")
        ax2.grid(zorder=0); ax2.set_axisbelow(True)
        if e == 0:
            ax2.set_ylabel("fit − data")
    fig.suptitle(title, fontsize=9.5, y=1.005)
    fig.savefig(path)
    plt.close(fig)


def k_comparison(path, chik, S, species, scales, title, kweight=None, pos=None,
                 ideal_species=None, ideal_scales=None):
    """k-space view of a fit to chi(k) data: k^w chi(k) measured against the
    reconstruction (and the ideal model), one panel per edge, with the
    window shaded. The model is drawn on each edge's experimental k axis
    (its fitted E0 applied), which is the axis the data are on.
    """
    w = int(S.kweights[0] if kweight is None else kweight)
    n = chik.nsp
    els = chik.elements or S.elements
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 2.6), squeeze=False,
                             gridspec_kw=dict(wspace=0.28))
    chi_fit = S.chi_k(species, pos)
    chi_ideal = None if ideal_species is None else S.chi_k(ideal_species)
    # kmin/kmax may be one number or one per edge (--per-edge-kmax), so each
    # panel shades the window ITS edge is actually fitted under
    lo_all = np.atleast_1d(S.kmin).ravel()
    hi_all = np.atleast_1d(S.kmax).ravel()
    for e in range(n):
        ax = axes[0, e]
        kmin_e = float(lo_all[e % lo_all.size])
        kmax_e = float(hi_all[e % hi_all.size])
        ax.axvspan(kmin_e, kmax_e, color=BAND, zorder=0, lw=0)
        ax.axhline(0, color=INK2, lw=0.6)
        k, y = chik.weighted(e, w)
        ax.plot(k, y, color=C_DATA, lw=1.3, zorder=3, label="measured")
        ke = S.k_exp(e)
        if chi_ideal is not None:
            si = scales if ideal_scales is None else ideal_scales
            ax.plot(ke, ke ** w * chi_ideal[e] * si[e], color=C_IDEAL, lw=0.9,
                    ls=(0, (4, 1.6)), zorder=2, label="ideal FCC")
        ax.plot(ke, ke ** w * chi_fit[e] * scales[e], color=C_FIT, lw=1.2,
                ls=(0, (1.4, 1.4)), zorder=4, label="fit")
        ax.set_xlim(max(0.0, kmin_e - 1.5), kmax_e + 1.5)
        ax.set_xlabel(r"$k$  (Å$^{-1}$)")
        ax.set_title(f"{els[e]} edge", fontsize=8.5, pad=4)
        ax.grid(zorder=0); ax.set_axisbelow(True)
        if e == 0:
            ax.set_ylabel(rf"$k^{w}\,\chi(k)$")
            ax.legend(loc="upper right", handlelength=1.9, labelspacing=0.25,
                      borderaxespad=0.3, framealpha=0.9, fontsize=6.8)
    fig.suptitle(title, fontsize=9.5, y=1.02)
    fig.savefig(path)
    plt.close(fig)


def rdf_figure(path, r, g, g_tot, shells, title, wc, wc_band=None, elements=None):
    """Partial RDFs, plus the Warren-Cowley values with their systematic band."""
    els = (tuple(elements) if elements is not None
           else tuple(f"s{i}" for i in range(np.shape(g)[0])))
    nsp = len(els)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9),
                             gridspec_kw=dict(width_ratios=[1.55, 1.0], wspace=0.32))
    ax = axes[0]
    for n, (A, B) in enumerate(pair_order(nsp)):
        ax.plot(r, g[A, B], color=PAIR_COLOURS[n % len(PAIR_COLOURS)], lw=1.2,
                label=f"{els[A]}–{els[B]}")
    ax.plot(r, g_tot, color=INK, lw=1.4, ls=(0, (4, 1.6)), label="total")
    for s in shells[:3]:
        ax.axvline(s, color=INK2, lw=0.6, ls=":")
    ax.set_xlabel(r"$r$  (Å)"); ax.set_ylabel(r"$g_{AB}(r)$")
    ax.set_xlim(r[0], r[-1])
    ax.legend(ncol=2, handlelength=1.6, labelspacing=0.22, columnspacing=1.0,
              borderaxespad=0.3)
    ax.grid(zorder=0); ax.set_axisbelow(True)
    ax.set_title("Partial RDFs", fontsize=8.5, pad=4)

    ax = axes[1]
    labels = wc_labels(els, tex=True)
    y = np.arange(len(labels))[::-1]
    if wc_band is not None:
        lo, hi = np.asarray(wc_band[0]), np.asarray(wc_band[1])
        ax.barh(y, hi - lo, left=lo, height=0.55, color="#cccccc",
                label="calibration band")
    ax.plot(np.asarray(wc), y, "o", color=C_FIT, ms=5, zorder=3, label="fitted")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("Warren–Cowley $\\alpha$ (shell 1)")
    lim = max(0.55, 1.1 * float(np.max(np.abs(wc))))
    if wc_band is not None:
        lim = max(lim, 1.1 * float(np.max(np.abs(np.asarray(wc_band)))))
    ax.set_xlim(-lim, lim)
    ax.grid(axis="x", zorder=0); ax.set_axisbelow(True)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.legend(loc="lower center", ncol=2, borderaxespad=0.3, framealpha=0.95,
              fontsize=6.8)
    ax.set_title(r"$\alpha$, shell 1", fontsize=8.5, pad=4)
    fig.suptitle(title, fontsize=9.5, y=1.02)
    fig.savefig(path)
    plt.close(fig)


def series_summary(path, per_point, wc_band, elements, unit=""):
    """Nearest-neighbour distance, sigma^2 and the WC band along the series."""
    els = tuple(elements)
    P = sorted(per_point)
    xlab = axis_label(unit)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.5), gridspec_kw=dict(wspace=0.34))
    nn = [per_point[p]["nn_distance"] for p in P]
    s2 = [per_point[p]["sigma2_nn"] for p in P]
    ax = axes[0]
    ax.plot(P, nn, "o-", color=C_FIT, lw=1.4, ms=5)
    ax.set_xlabel(xlab); ax.set_ylabel(r"$r_{\rm NN}$  (Å)")
    ax.set_title(r"$r_{\rm NN}$", fontsize=8.5, pad=4)
    ax.grid(zorder=0); ax.set_axisbelow(True)
    ax = axes[1]
    ax.plot(P, np.array(s2) * 1e3, "o-", color=C_IDEAL, lw=1.4, ms=5)
    ax.set_xlabel(xlab)
    ax.set_ylabel(r"$\sigma^2_{\rm NN}$  ($10^{-3}$ Å$^2$)")
    ax.set_title(r"$\sigma^2_{\rm NN}$", fontsize=8.5, pad=4)
    ax.grid(zorder=0); ax.set_axisbelow(True)
    ax = axes[2]
    labels = wc_labels(els, tex=True)
    x = np.arange(len(labels))
    pcol = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
            "#000000", "#999999"]
    wbar = 0.8 / max(len(P), 1)
    for n, p in enumerate(P):
        lo = np.array(wc_band[p]["wc_min"]); hi = np.array(wc_band[p]["wc_max"])
        ax.bar(x + (n - (len(P) - 1) / 2) * wbar, hi - lo, bottom=lo, width=0.9 * wbar,
               color=pcol[n % len(pcol)],
               label=point_label(p, unit), alpha=0.85)
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

    Shells across the x axis, the ordered pairs as grouped bars, clustering
    above zero and ordering below.

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
