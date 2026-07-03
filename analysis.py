#!/usr/bin/env python3
"""
Spin@Threshold'26 — analysis plots.
Run: uv run python analysis.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from matplotlib.transforms import blended_transform_factory

matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.labelsize":    14,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "legend.fontsize":   12,
    "legend.frameon":    False,
})

# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR  = Path("input_data")
OUT_DIR   = Path("plots")
FSR       = "afterFSR"
M_COL     = f"spin_{FSR}_m_ttbar_NOSYS"
COS_COL   = f"spin_{FSR}_cos_phi_NOSYS"
W_NOM     = "weight_mc_NOSYS"
W_NRC_I0  = "weight_mc_GEN_ithreshold0"
W_CL_LO   = "weight_mc_GEN_ithreshold33_coulombScaleFact05"
W_CL_HI   = "weight_mc_GEN_ithreshold33_coulombScaleFact2"
THRESHOLD = 345.0   # GeV

# Physical cross sections [pb] — used to set relative normalisation when
# combining the continuum (NRCi0) with the toponium signal samples.
XS_HVQ      = 87.87    # Powheg hvq total xs
XS_TOPONIUM = 0.675    # Fuks and Toy toponium xs (identical for both)

LABEL = {
    "hvq":         r"Powheg hvq + Pythia 8",
    "amcatnlo":    r"aMC@NLO + Pythia 8",
    "madgraph":    r"MadGraph (parton-level)",
    "nrc_nominal": r"Powheg hvq + Pythia 8 (incl. NRC)",
    "nrc_i0":      r"Powheg hvq + Pythia 8 (no NRC)",
    "fuks":        r"$t\bar{t}_\mathrm{GFRW}$",
    "toy":         r"$\eta_t$",
    "nrc_i0_fuks": r"Powheg hvq + Pythia 8 (no NRC) + $t\bar{t}_\mathrm{GFRW}$",
    "nrc_i0_toy":  r"Powheg hvq + Pythia 8 (no NRC) + $\eta_t$",
    "coulomb":     r"Coulomb scale variation",
    "gg":          r"$gg$",
    "qq":          r"$q\bar{q}$",
    "qg":          r"$qg$",
    "no_corr":     r"No correction",
    "gg_corr":     r"$gg$: $D{=}{-1}$ below 345 GeV",
    "gg_qq_corr":  r"$gg$: $D{=}{-1}$,  $q\bar{q}$: $D{=}{+1/3}$ below 345 GeV",
}

COLOR = {
    "hvq":         "#1f77b4",
    "amcatnlo":    "#d62728",
    "madgraph":    "#2ca02c",
    "nrc_nominal": "black",
    "nrc_i0":      "#1f77b4",
    "fuks":        "#d62728",
    "toy":         "#2ca02c",
    "nrc_i0_fuks": "#d62728",
    "nrc_i0_toy":  "#2ca02c",
    "gg":          "#d62728",
    "qq":          "#1f77b4",
    "qg":          "#2ca02c",
    "no_corr":     "black",
    "gg_corr":     "#d62728",
    "gg_qq_corr":  "#1f77b4",
}

MTTBAR_XLABEL = r"$m(t\bar{t})$ [GeV]"
D_YLABEL      = r"$D = -3\langle\cos\theta\rangle$"
MTTBAR_YLABEL = r"Normalised events / GeV"

# ─────────────────────────────────────────────────────────────────────────────
#  Data loading (lazy, cached)
# ─────────────────────────────────────────────────────────────────────────────

_CACHE: dict[str, pd.DataFrame] = {}

_PATHS = {
    "hvq":      DATA_DIR / "cache_hvq_afterFSR.parquet",
    "amcatnlo": DATA_DIR / "cache_amcatnlo_afterFSR.parquet",
    "nrc":      DATA_DIR / "cache_nrc_afterFSR.parquet",
    "fuks":     DATA_DIR / "cache_toponium_Fuks_afterFSR.parquet",
    "toy":      DATA_DIR / "cache_toponium_Toy_afterFSR.parquet",
    "madgraph": DATA_DIR / "cache_MadGraph_LO_lhe.parquet",
}


def _load(name: str) -> pd.DataFrame:
    if name not in _CACHE:
        print(f"  loading {name} …")
        df = pd.read_parquet(_PATHS[name])
        if name != "madgraph":
            df[M_COL] = df[M_COL] / 1000.0   # MeV → GeV
        _CACHE[name] = df
    return _CACHE[name]


def _mw(name: str, w_col: str = W_NOM) -> tuple[np.ndarray, np.ndarray]:
    """Return (m_ttbar_GeV, weights)."""
    df = _load(name)
    if name == "madgraph":
        return df["m_ttbar"].to_numpy(), df["weight"].to_numpy()
    return df[M_COL].to_numpy(), df[w_col].to_numpy()


def _mcw(name: str, w_col: str = W_NOM) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (m_ttbar_GeV, cos_theta, weights)."""
    df = _load(name)
    if name == "madgraph":
        return (df["m_ttbar"].to_numpy(),
                df["cos_hel"].to_numpy(),
                df["weight"].to_numpy())
    return df[M_COL].to_numpy(), df[COS_COL].to_numpy(), df[w_col].to_numpy()


def _channel_masks(name: str) -> dict[str, np.ndarray]:
    df = _load(name)
    if name == "madgraph":
        ch = df["channel"].to_numpy()
        return {"gg": ch == "gg", "qq": ch == "qq", "qg": ch == "qg"}
    id1 = np.abs(df["PDFinfo_PDGID1"].to_numpy())
    id2 = np.abs(df["PDFinfo_PDGID2"].to_numpy())
    g1, g2 = id1 == 21, id2 == 21
    return {"gg": g1 & g2, "qq": ~g1 & ~g2, "qg": g1 ^ g2}


# ─────────────────────────────────────────────────────────────────────────────
#  Sufficient statistics
# ─────────────────────────────────────────────────────────────────────────────

def bin_stats(m: np.ndarray, cos: np.ndarray, w: np.ndarray,
              bin_edges: np.ndarray) -> dict:
    """Per-bin sufficient statistics for D computation (vectorised)."""
    n   = len(bin_edges) - 1
    idx = np.digitize(m, bin_edges) - 1
    sel = (idx >= 0) & (idx < n)
    idx, c, wi = idx[sel], cos[sel], w[sel]

    out = {k: np.zeros(n) for k in ["w", "wc", "wc2", "w2", "w2c2"]}
    out["empty"] = np.ones(n, dtype=bool)
    if len(idx) == 0:
        return out

    order = np.argsort(idx, kind="stable")
    idx, c, wi = idx[order], c[order], wi[order]
    wi2 = wi * wi
    wc  = wi * c

    first  = np.searchsorted(idx, np.arange(n), side="left")
    starts = np.where(first < len(idx), first, len(idx) - 1)
    counts = np.diff(np.append(first, len(idx)))
    empty  = (first >= len(idx)) | (counts == 0)

    ws    = np.add.reduceat(wi,         starts)
    wcs   = np.add.reduceat(wc,         starts)
    w2s   = np.add.reduceat(wi2,        starts)
    w2c2s = np.add.reduceat(wi2 * c**2, starts)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(empty, 0.0, wcs / ws)
        wc2s = np.add.reduceat(wi2 * (c - mean[idx])**2, starts)

    for k, arr in zip(["w", "wc", "wc2", "w2", "w2c2"],
                       [ws, wcs, wc2s, w2s, w2c2s]):
        arr[empty] = 0.0
        out[k] = arr
    out["empty"] = empty
    return out


def merge_stats(s1: dict, s2: dict) -> dict:
    """Element-wise sum of two bin_stats dicts (combines two event pools)."""
    out = {k: s1[k] + s2[k] for k in ["w", "wc", "wc2", "w2", "w2c2"]}
    out["empty"] = out["w"] == 0
    return out


def scale_stats(s: dict, factor: float) -> dict:
    """Multiply all weight-like arrays by factor (leaves empty mask unchanged)."""
    return {k: (s[k] * factor if k != "empty" else s[k]) for k in s}


def D_diff(s: dict) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(s["empty"], np.nan, s["wc"] / s["w"])
        var  = np.where(s["empty"], np.nan, s["wc2"] / s["w"]**2)
    return -3.0 * mean, 3.0 * np.sqrt(var)


def D_cum(s: dict) -> tuple[np.ndarray, np.ndarray]:
    W    = np.cumsum(s["w"])
    WC   = np.cumsum(s["wc"])
    W2   = np.cumsum(s["w2"])
    W2C2 = np.cumsum(s["w2c2"])
    e    = W == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        mc  = WC / W
        var = (W2C2 - mc**2 * W2) / W**2
    return (np.where(e, np.nan, -3.0 * mc),
            np.where(e, np.nan,  3.0 * np.sqrt(np.maximum(var, 0.0))))


def apply_corrections(ch_stats: dict, bin_edges: np.ndarray,
                      corrections: dict, cumulative: bool = False):
    """
    Build D(m) with per-channel threshold corrections.
    corrections: e.g. {"gg": -1.0, "qq": 1/3, "qg": None}
    Below THRESHOLD, the mean cos is forced to corrections[ch] / -3.
    """
    n     = len(bin_edges) - 1
    cen   = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    below = cen < THRESHOLD
    agg   = {k: np.zeros(n) for k in ["w", "wc", "wc2", "w2", "w2c2"]}

    for ch, s in ch_stats.items():
        enforced = (corrections or {}).get(ch)
        w, wc, wc2, w2, w2c2 = (s["w"].copy(), s["wc"].copy(), s["wc2"].copy(),
                                  s["w2"].copy(), s["w2c2"].copy())
        if enforced is not None:
            ce   = enforced / -3.0
            wc   = np.where(below, w * ce,       wc)
            wc2  = np.where(below, 0.0,           wc2)
            w2c2 = np.where(below, w2 * ce**2,   w2c2)
        for k, arr in zip(["w", "wc", "wc2", "w2", "w2c2"],
                           [w,   wc,  wc2,   w2,  w2c2]):
            agg[k] += arr

    agg["empty"] = agg["w"] == 0
    return D_cum(agg) if cumulative else D_diff(agg)


def weighted_hist(m: np.ndarray, w: np.ndarray,
                  bin_edges: np.ndarray,
                  norm: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Weighted histogram normalised to events/GeV.  norm defaults to sum(w)."""
    counts,  _ = np.histogram(m, bins=bin_edges, weights=w)
    errors2, _ = np.histogram(m, bins=bin_edges, weights=w * w)
    denom = w.sum() if norm is None else norm
    bw    = np.diff(bin_edges)
    return counts / denom / bw, np.sqrt(errors2) / denom / bw


def norm_stats(m: np.ndarray, cos: np.ndarray, w: np.ndarray,
               bin_edges: np.ndarray) -> dict:
    """bin_stats with weights normalised to unit total (shape-only D)."""
    total = w.sum()
    return bin_stats(m, cos, w / total, bin_edges) if total > 0 else bin_stats(m, cos, w, bin_edges)


# ─────────────────────────────────────────────────────────────────────────────
#  Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _spin_label(ax: plt.Axes) -> None:
    ax.text(0.03, 0.97, "Spin@Threshold'26",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=13, fontweight="bold")


def _thresh_line(ax: plt.Axes) -> None:
    ax.axvline(THRESHOLD, color="lightgray", lw=1.0, ls="--", zorder=0)
    # x in data coords, y in axes fraction — robust regardless of ylim
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(THRESHOLD + 0.5, 0.97, r"$2m_t = 345\,\mathrm{GeV}$",
            transform=trans, color="lightgray", fontsize=10,
            va="top", ha="left", rotation=90, rotation_mode="anchor",
            clip_on=True)


def _style_main(ax: plt.Axes, xlabel: str, ylabel: str,
                xlim: tuple, ylim: tuple | None = None,
                legend_loc: str = "best",
                threshold: bool = True,
                D_lines: bool = False,
                show_xlabel: bool = True) -> None:
    if threshold:
        _thresh_line(ax)
    if D_lines:
        ax.axhline(0,  color="k",    lw=0.7, ls=":",  zorder=0)
        ax.axhline(-1, color="gray", lw=0.7, ls="--", alpha=0.5, zorder=0)
    if show_xlabel:
        ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=12, loc=legend_loc)
    _spin_label(ax)


def _ratio_fig(figsize: tuple = (8, 6), hr: tuple = (3, 1)):
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs  = gridspec.GridSpec(2, 1, height_ratios=hr, hspace=0.05, figure=fig)
    axm = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=axm)
    axm.tick_params(axis="x", labelbottom=False)
    return fig, axm, axr


def _style_ratio(axr: plt.Axes, xlabel: str, xlim: tuple,
                 ylim: tuple = (0.5, 1.5),
                 ylabel: str = "Ratio to nominal",
                 threshold: bool = True) -> None:
    axr.axhline(1.0, color="black", lw=0.8, zorder=0)
    if threshold:
        _thresh_line(axr)
    axr.set_xlabel(xlabel, fontsize=14)
    axr.set_ylabel(ylabel, fontsize=12)
    axr.set_xlim(*xlim)
    axr.set_ylim(*ylim)
    axr.tick_params(labelsize=11)


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    p = OUT_DIR / f"{name}.pdf"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


# ── drawing primitives ────────────────────────────────────────────────────────

def _draw_steps(ax: plt.Axes, bin_edges: np.ndarray, curves: list[dict]) -> None:
    """
    Draw step histograms with optional MC stat error bars at bin centres.
    Each curve dict: vals, label, color, errs (optional), lw (optional).
    """
    xstep = np.append(bin_edges[:-1], bin_edges[-1])
    cen   = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    for c in curves:
        v  = c["vals"]
        lw = c.get("lw", 1.8)
        ax.step(xstep, np.append(v, v[-1]), where="post",
                color=c["color"], lw=lw, label=c["label"])
        if c.get("errs") is not None:
            ax.errorbar(cen, v, yerr=c["errs"],
                        fmt="none", color=c["color"],
                        capsize=0, lw=0.8, rasterized=True)


def _draw_band(ax: plt.Axes, bin_edges: np.ndarray,
               lo: np.ndarray, hi: np.ndarray,
               color: str, label: str, alpha: float = 0.30) -> None:
    xstep = np.append(bin_edges[:-1], bin_edges[-1])
    lo_s  = np.append(lo, lo[-1])
    hi_s  = np.append(hi, hi[-1])
    ax.fill_between(xstep, lo_s, hi_s, step="post",
                    alpha=alpha, color=color, label=label, zorder=0)


def _draw_errorbars(ax: plt.Axes, bin_edges: np.ndarray,
                    curves: list[dict]) -> None:
    """
    Draw D errorbar plots on ax.
    Each curve dict: vals, errs, label, color, marker (optional), ms (optional).
    """
    x = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    for c in curves:
        ax.errorbar(x, c["vals"], yerr=c.get("errs"),
                    color=c["color"], label=c["label"],
                    marker=c.get("marker", "o"),
                    ms=c.get("ms", 3.5),
                    ls="none", capsize=2, rasterized=True)


def _draw_ratio_steps(axr: plt.Axes, bin_edges: np.ndarray,
                      curves: list[dict], ref_idx: int = 0) -> None:
    ref  = curves[ref_idx]["vals"]
    xstep = np.append(bin_edges[:-1], bin_edges[-1])
    for i, c in enumerate(curves):
        if i == ref_idx:
            axr.step(xstep, np.append(np.ones_like(c["vals"]),
                                       np.ones(1)),
                     where="post", color=c["color"], lw=1.5)
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(ref != 0, c["vals"] / ref, np.nan)
        axr.step(xstep, np.append(ratio, ratio[-1]),
                 where="post", color=c["color"], lw=1.5)


def _draw_ratio_errorbars(axr: plt.Axes, bin_edges: np.ndarray,
                          curves: list[dict], ref_idx: int = 0) -> None:
    ref = curves[ref_idx]["vals"]
    x   = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    for i, c in enumerate(curves):
        if i == ref_idx:
            axr.axhline(1.0, color=c["color"], lw=1.5)
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(ref != 0, c["vals"] / ref, np.nan)
        axr.scatter(x, ratio, color=c["color"],
                    marker=c.get("marker", "o"), s=10, rasterized=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Generic plot drivers
# ─────────────────────────────────────────────────────────────────────────────

def plot_mttbar(bin_edges: np.ndarray, curves: list[dict],
                xlim: tuple, ylim: tuple | None = None,
                legend_loc: str = "best",
                ratio_to: int | None = None,
                ratio_ylim: tuple = (0.5, 1.5),
                bands: list[dict] | None = None,
                figsize: tuple = (8, 5),
                log_y: bool = False,
                output: str | None = None) -> None:
    """
    Plot normalised m(ttbar) step histograms.

    curves:  list of dicts with keys: vals, errs (optional), label, color
    bands:   list of dicts with keys: lo, hi, label, color  (Coulomb-style)
    ratio_to: index of the reference curve for the ratio panel (None = no ratio)
    """
    has_ratio = ratio_to is not None
    if has_ratio:
        fig, axm, axr = _ratio_fig(figsize=figsize)
    else:
        fig, axm = plt.subplots(figsize=figsize)

    if bands:
        for b in bands:
            _draw_band(axm, bin_edges, b["lo"], b["hi"],
                       b["color"], b["label"], alpha=b.get("alpha", 0.28))
    _draw_steps(axm, bin_edges, curves)

    if log_y:
        axm.set_yscale("log")
    _style_main(axm, MTTBAR_XLABEL, MTTBAR_YLABEL, xlim, ylim,
                legend_loc=legend_loc, show_xlabel=not has_ratio)

    if has_ratio:
        if bands:
            ref = curves[ratio_to]["vals"]
            for b in bands:
                with np.errstate(invalid="ignore", divide="ignore"):
                    rlo = np.where(ref != 0, b["lo"] / ref, np.nan)
                    rhi = np.where(ref != 0, b["hi"] / ref, np.nan)
                _draw_band(axr, bin_edges, rlo, rhi,
                           b["color"], "", alpha=b.get("alpha", 0.28))
        _draw_ratio_steps(axr, bin_edges, curves, ref_idx=ratio_to)
        _style_ratio(axr, MTTBAR_XLABEL, xlim, ylim=ratio_ylim)

    if not has_ratio:
        fig.tight_layout()
    if output:
        _save(fig, output)


def plot_D(bin_edges: np.ndarray, curves: list[dict],
           xlim: tuple, ylim: tuple = (-1.0, 1.0),
           legend_loc: str = "upper right",
           ratio_to: int | None = None,
           ratio_ylim: tuple = (0.5, 1.5),
           figsize: tuple = (8, 5),
           xlabel: str = MTTBAR_XLABEL,
           output: str | None = None) -> None:
    """
    Plot D = -3<cos theta> errorbar curves.
    Optionally adds a ratio panel if ratio_to is set.
    """
    has_ratio = ratio_to is not None
    if has_ratio:
        fig, axm, axr = _ratio_fig(figsize=figsize)
    else:
        fig, axm = plt.subplots(figsize=figsize)

    _draw_errorbars(axm, bin_edges, curves)
    _style_main(axm, xlabel, D_YLABEL, xlim, ylim,
                legend_loc=legend_loc, D_lines=True,
                show_xlabel=not has_ratio)

    if has_ratio:
        _draw_ratio_errorbars(axr, bin_edges, curves, ref_idx=ratio_to)
        _style_ratio(axr, xlabel, xlim, ylim=ratio_ylim,
                     ylabel=r"Ratio to nominal")

    if not has_ratio:
        fig.tight_layout()
    if output:
        _save(fig, output)


# ─────────────────────────────────────────────────────────────────────────────
#  Individual plots
# ─────────────────────────────────────────────────────────────────────────────

M_MIN, M_MAX = 300.0, 500.0
M_STEP       = 2.0
BIN_EDGES    = np.arange(M_MIN, M_MAX + M_STEP, M_STEP)
XLIM_FULL    = (M_MIN, M_MAX)

M_THR_STEP   = 0.5
M_THR_MIN    = 330.0
M_THR_MAX    = 370.0
BIN_EDGES_THR = np.arange(M_THR_MIN, M_THR_MAX + M_THR_STEP, M_THR_STEP)
XLIM_THR     = (M_THR_MIN, M_THR_MAX)


def _make_D_curve(name: str, key: str,
                  w_col: str = W_NOM,
                  m_arr: np.ndarray | None = None,
                  cos_arr: np.ndarray | None = None,
                  w_arr: np.ndarray | None = None,
                  bin_edges: np.ndarray | None = None,
                  cumulative: bool = False) -> dict:
    """Build a D curve dict from a named sample (or pre-supplied arrays)."""
    if bin_edges is None:
        bin_edges = BIN_EDGES
    if m_arr is None:
        m_arr, cos_arr, w_arr = _mcw(name, w_col)
    s = norm_stats(m_arr, cos_arr, w_arr, bin_edges)
    vals, errs = D_cum(s) if cumulative else D_diff(s)
    return {"vals": vals, "errs": errs, "label": LABEL[key], "color": COLOR[key]}


# ── Plot 1: m(ttbar) distribution, 3 generators ──────────────────────────────

def plot1_mttbar_generators():
    print("Plot 1: m(ttbar) — generators")
    curves = []
    for name, key in [("hvq", "hvq"), ("amcatnlo", "amcatnlo"), ("madgraph", "madgraph")]:
        m, w = _mw(name)
        v, e = weighted_hist(m, w, BIN_EDGES)
        curves.append({"vals": v, "errs": e, "label": LABEL[key], "color": COLOR[key]})

    plot_mttbar(BIN_EDGES, curves, xlim=XLIM_FULL,
                legend_loc="upper right",
                figsize=(8, 5), log_y=True,
                output="plot1_mttbar_generators")


# ── Plot 2: D differential, 3 generators ─────────────────────────────────────

def plot2_D_generators():
    print("Plot 2: D — generators")
    curves = [
        _make_D_curve("hvq",      "hvq"),
        _make_D_curve("amcatnlo", "amcatnlo"),
        _make_D_curve("madgraph", "madgraph"),
    ]
    plot_D(BIN_EDGES, curves, xlim=XLIM_FULL,
           legend_loc="lower right", figsize=(8, 5),
           output="plot2_D_generators")


# ── Plot 3: D per channel, 3 generators, 3 panels ────────────────────────────

def plot3_D_channels():
    print("Plot 3: D per channel")
    channels   = ["gg", "qq", "qg"]
    gen_specs  = [
        ("hvq",      "hvq",      "-"),
        ("amcatnlo", "amcatnlo", "--"),
        ("madgraph", "madgraph", ":"),
    ]
    x = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, ch in zip(axes, channels):
        for name, key, ls in gen_specs:
            m, cos, w = _mcw(name)
            masks = _channel_masks(name)
            m_ch, cos_ch, w_ch = m[masks[ch]], cos[masks[ch]], w[masks[ch]]
            s = norm_stats(m_ch, cos_ch, w_ch, BIN_EDGES)
            vals, errs = D_diff(s)
            ax.errorbar(x, vals, yerr=errs,
                        color=COLOR[key], label=LABEL[key],
                        marker="o", ms=3.5, ls="none",
                        capsize=2, rasterized=True)

        _thresh_line(ax)
        ax.axhline(0,  color="k",    lw=0.7, ls=":",  zorder=0)
        ax.axhline(-1, color="gray", lw=0.7, ls="--", alpha=0.5, zorder=0)
        ax.set_title(LABEL[ch], fontsize=13)
        ax.set_xlabel(MTTBAR_XLABEL, fontsize=14)
        ax.set_xlim(*XLIM_FULL)
        ax.set_ylim(-1.0, 1.0)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=11, loc="lower right")
        _spin_label(ax)

    axes[0].set_ylabel(D_YLABEL, fontsize=14)
    fig.tight_layout()
    _save(fig, "plot3_D_channels")


# ── Plots 4 & 5: D with threshold corrections for hvq ────────────────────────

def _hvq_channel_stats(bin_edges: np.ndarray) -> dict:
    masks = _channel_masks("hvq")
    m, cos, w = _mcw("hvq")
    total_w = w.sum()
    ch_stats = {}
    for ch, mask in masks.items():
        ch_stats[ch] = norm_stats(m[mask], cos[mask], w[mask], bin_edges)
    return ch_stats


def plot4_D_corrections_diff():
    print("Plot 4: D corrections (differential)")
    ch_stats = _hvq_channel_stats(BIN_EDGES)
    x        = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])

    corr_specs = [
        (None,                                 "no_corr"),
        ({"gg": -1.0},                         "gg_corr"),
        ({"gg": -1.0, "qq": 1/3},              "gg_qq_corr"),
    ]
    curves = []
    for corrections, key in corr_specs:
        vals, errs = apply_corrections(ch_stats, BIN_EDGES, corrections)
        curves.append({"vals": vals, "errs": errs,
                       "label": LABEL[key], "color": COLOR[key]})

    plot_D(BIN_EDGES, curves, xlim=XLIM_FULL,
           legend_loc="lower right", figsize=(8, 5),
           output="plot4_D_corrections_diff")


def plot5_D_corrections_cum():
    print("Plot 5: D corrections (cumulative)")
    ch_stats = _hvq_channel_stats(BIN_EDGES)
    x        = BIN_EDGES[1:]   # upper edges for cumulative

    corr_specs = [
        (None,                   "no_corr"),
        ({"gg": -1.0},           "gg_corr"),
        ({"gg": -1.0, "qq": 1/3}, "gg_qq_corr"),
    ]
    curves = []
    for corrections, key in corr_specs:
        vals, errs = apply_corrections(ch_stats, BIN_EDGES, corrections,
                                       cumulative=True)
        curves.append({"vals": vals, "errs": errs,
                       "label": LABEL[key], "color": COLOR[key]})

    ylabel_cum = r"$D$ (integrated up to $m(t\bar{t})$)"
    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_errorbars(ax, BIN_EDGES, curves)
    _style_main(ax, MTTBAR_XLABEL, ylabel_cum, XLIM_FULL, ylim=(-1.0, 1.0),
                legend_loc="lower right", D_lines=True)
    fig.tight_layout()
    _save(fig, "plot5_D_corrections_cum")


# ── Plot 6: m(ttbar) NRC nominal vs NRCi0, Coulomb band, ratio ───────────────

def plot6_mttbar_nrc():
    print("Plot 6: m(ttbar) NRC variations")
    m, _ = _mw("nrc")   # preload to share m array
    df_nrc = _load("nrc")
    m_nrc = df_nrc[M_COL].to_numpy()

    w_nom  = df_nrc[W_NOM].to_numpy()
    w_i0   = df_nrc[W_NRC_I0].to_numpy()
    w_cl_lo = df_nrc[W_CL_LO].to_numpy()
    w_cl_hi = df_nrc[W_CL_HI].to_numpy()

    # Normalise to nominal total weight so ratio ≈ 1 far from threshold
    ref_norm = w_nom.sum()

    v_nom, e_nom = weighted_hist(m_nrc, w_nom, BIN_EDGES_THR, norm=ref_norm)
    v_i0,  e_i0  = weighted_hist(m_nrc, w_i0,  BIN_EDGES_THR, norm=ref_norm)
    v_lo,  _     = weighted_hist(m_nrc, w_cl_lo, BIN_EDGES_THR, norm=ref_norm)
    v_hi,  _     = weighted_hist(m_nrc, w_cl_hi, BIN_EDGES_THR, norm=ref_norm)

    curves = [
        {"vals": v_nom, "errs": e_nom, "label": LABEL["nrc_nominal"], "color": COLOR["nrc_nominal"]},
        {"vals": v_i0,  "errs": e_i0,  "label": LABEL["nrc_i0"],      "color": COLOR["nrc_i0"]},
    ]
    lo_env = np.minimum(v_lo, v_hi)
    hi_env = np.maximum(v_lo, v_hi)
    bands  = [{"lo": lo_env, "hi": hi_env,
               "label": LABEL["coulomb"], "color": "gray", "alpha": 0.35}]

    plot_mttbar(BIN_EDGES_THR, curves, xlim=XLIM_THR,
                legend_loc="upper right",
                ratio_to=0, ratio_ylim=(0.5, 1.5),
                bands=bands, figsize=(8, 6),
                output="plot6_mttbar_nrc")


# ── Plot 7: m(ttbar) NRC nominal vs NRCi0+Fuks vs NRCi0+Toy, ratio ──────────

def _xs_combined_hist(m_cont, w_cont, xs_cont,
                      m_sig,  w_sig,  xs_sig,
                      bin_edges: np.ndarray):
    """
    Combine a continuum and a signal sample with xs-weighted normalisation.

    Each sample is first normalised to unit area (shape), then mixed with
    weights xs_cont and xs_sig respectively, and finally renormalised so the
    combined distribution also integrates to 1.  This sets the relative
    contribution of signal vs continuum by their physical cross sections.

    Returns (vals, errs) in [events / GeV], normalised to unit area.
    """
    h_cont, e_cont = weighted_hist(m_cont, w_cont, bin_edges)   # shape of continuum
    h_sig,  e_sig  = weighted_hist(m_sig,  w_sig,  bin_edges)   # shape of signal
    total_xs = xs_cont + xs_sig
    vals = (xs_cont * h_cont + xs_sig * h_sig) / total_xs
    errs = np.sqrt((xs_cont * e_cont)**2 + (xs_sig * e_sig)**2) / total_xs
    return vals, errs


def _xs_combined_stats(m_cont, cos_cont, w_cont, xs_cont,
                       m_sig,  cos_sig,  w_sig,  xs_sig,
                       bin_edges: np.ndarray) -> dict:
    """
    Combine sufficient statistics for D with xs-weighted normalisation.

    Each sample's weights are normalised to unit area, then scaled by its
    cross section before merging.  D is a ratio so the absolute scale of the
    combined stats cancels, but the xs ratio sets the correct bin-by-bin
    mixing of continuum and signal contributions.
    """
    s_cont = scale_stats(norm_stats(m_cont, cos_cont, w_cont, bin_edges), xs_cont)
    s_sig  = scale_stats(norm_stats(m_sig,  cos_sig,  w_sig,  bin_edges), xs_sig)
    return merge_stats(s_cont, s_sig)


def plot7_mttbar_toponium():
    print("Plot 7: m(ttbar) NRC nominal vs toponium combinations")
    df_nrc = _load("nrc")
    m_nrc  = df_nrc[M_COL].to_numpy()
    w_nom  = df_nrc[W_NOM].to_numpy()
    w_i0   = df_nrc[W_NRC_I0].to_numpy()

    m_fuks, w_fuks = _mw("fuks")
    m_toy,  w_toy  = _mw("toy")

    v_nom,  e_nom  = weighted_hist(m_nrc, w_nom, BIN_EDGES_THR)
    v_fuks, e_fuks = _xs_combined_hist(m_nrc, w_i0, XS_HVQ,
                                        m_fuks, w_fuks, XS_TOPONIUM, BIN_EDGES_THR)
    v_toy,  e_toy  = _xs_combined_hist(m_nrc, w_i0, XS_HVQ,
                                        m_toy,  w_toy,  XS_TOPONIUM, BIN_EDGES_THR)

    curves = [
        {"vals": v_nom,  "errs": e_nom,  "label": LABEL["nrc_nominal"], "color": COLOR["nrc_nominal"]},
        {"vals": v_fuks, "errs": e_fuks, "label": LABEL["nrc_i0_fuks"], "color": COLOR["nrc_i0_fuks"]},
        {"vals": v_toy,  "errs": e_toy,  "label": LABEL["nrc_i0_toy"],  "color": COLOR["nrc_i0_toy"]},
    ]
    plot_mttbar(BIN_EDGES_THR, curves, xlim=XLIM_THR,
                legend_loc="upper right",
                ratio_to=0, ratio_ylim=(0.5, 1.5),
                figsize=(8, 6), output="plot7_mttbar_toponium")


# ── Plot 8: D NRC nominal vs NRCi0+Fuks vs NRCi0+Toy, ratio ─────────────────

def plot8_D_toponium():
    print("Plot 8: D NRC nominal vs toponium combinations")
    df_nrc  = _load("nrc")
    m_nrc   = df_nrc[M_COL].to_numpy()
    cos_nrc = df_nrc[COS_COL].to_numpy()
    w_nom   = df_nrc[W_NOM].to_numpy()
    w_i0    = df_nrc[W_NRC_I0].to_numpy()

    m_fuks, cos_fuks, w_fuks = _mcw("fuks")
    m_toy,  cos_toy,  w_toy  = _mcw("toy")

    s_nom  = norm_stats(m_nrc, cos_nrc, w_nom, BIN_EDGES)
    s_fuks = _xs_combined_stats(m_nrc, cos_nrc, w_i0,   XS_HVQ,
                                 m_fuks, cos_fuks, w_fuks, XS_TOPONIUM, BIN_EDGES)
    s_toy  = _xs_combined_stats(m_nrc, cos_nrc, w_i0,   XS_HVQ,
                                 m_toy,  cos_toy,  w_toy,  XS_TOPONIUM, BIN_EDGES)

    v_nom,  e_nom  = D_diff(s_nom)
    v_fuks, e_fuks = D_diff(s_fuks)
    v_toy,  e_toy  = D_diff(s_toy)

    curves = [
        {"vals": v_nom,  "errs": e_nom,  "label": LABEL["nrc_nominal"], "color": COLOR["nrc_nominal"]},
        {"vals": v_fuks, "errs": e_fuks, "label": LABEL["nrc_i0_fuks"], "color": COLOR["nrc_i0_fuks"]},
        {"vals": v_toy,  "errs": e_toy,  "label": LABEL["nrc_i0_toy"],  "color": COLOR["nrc_i0_toy"]},
    ]
    plot_D(BIN_EDGES, curves, xlim=XLIM_FULL,
           legend_loc="lower right",
           ratio_to=0, ratio_ylim=(0.5, 1.5),
           figsize=(8, 6), output="plot8_D_toponium")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    for name in ["hvq", "amcatnlo", "madgraph", "nrc", "fuks", "toy"]:
        _load(name)

    print("\nGenerating plots …")
    plot1_mttbar_generators()
    plot2_D_generators()
    plot3_D_channels()
    plot4_D_corrections_diff()
    plot5_D_corrections_cum()
    plot6_mttbar_nrc()
    plot7_mttbar_toponium()
    plot8_D_toponium()

    print("\nDone.")


if __name__ == "__main__":
    main()
