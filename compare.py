#!/usr/bin/env python3
"""
Compare integrated D predictions against published measurements.
Run: uv run python compare.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle as MplRect
from matplotlib.transforms import blended_transform_factory
from pathlib import Path

matplotlib.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.labelsize":  12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 11,
    "legend.frameon":  False,
})

# ── Constants ──────────────────────────────────────────────────────────────────

DATA_DIR    = Path("input_data")
OUT_DIR     = Path("plots")
FSR         = "afterFSR"
M_COL       = f"spin_{FSR}_m_ttbar_NOSYS"
COS_COL     = f"spin_{FSR}_cos_phi_NOSYS"
BZ_COL      = "beta_z"
W_NOM       = "weight_mc_NOSYS"
W_NRC_I0    = "weight_mc_GEN_ithreshold0"
THRESHOLD   = 345.0   # GeV
XS_HVQ      = 87.87   # pb
XS_TOPONIUM = 0.675   # pb
XS_SINGLET  = 0.07    # pb

# ── Setup catalogue ────────────────────────────────────────────────────────────
# solid=False → hollow marker  (hvq continuum, no gg+qq correction)
# solid=True  → filled marker  (hvq continuum with gg+qq correction below threshold)
# no_err=True → marker only, no error bar drawn (NRC nominal reference)

SETUPS: dict[str, dict] = {
    "hvq": {
        "label":        r"Ph hvq + Py8",
        "legend_label": "Ph hvq + Py8 (no NRC)",
        "color": "#1f77b4", "marker": "o", "solid": False, "no_err": False,
    },
    "hvq_corr": {
        "label":        r"Ph hvq + Py8 (gg+qq corr.)",
        "legend_label": "Ph hvq + gg+qq corr.",
        "color": "#1f77b4", "marker": "o", "solid": True,  "no_err": False,
    },
    "hvq_fuks": {
        "label":        r"Ph hvq + Py8 + $t\bar{t}_\mathrm{GFRW}$",
        "legend_label": r"+ $t\bar{t}_\mathrm{GFRW}$",
        "color": "#d62728", "marker": "o", "solid": False, "no_err": False,
    },
    "hvq_fuks_ms": {
        "label":        r"Ph hvq + Py8 + $t\bar{t}_\mathrm{GFRW}$ $-$ singlet",
        "legend_label": r"+ $t\bar{t}_\mathrm{GFRW}$ $-$ singlet",
        "color": "#ff7f0e", "marker": "o", "solid": False, "no_err": False,
    },
    "hvq_toy": {
        "label":        r"Ph hvq + Py8 + $\eta_t$",
        "legend_label": r"+ $\eta_t$",
        "color": "#2ca02c", "marker": "o", "solid": False, "no_err": False,
    },
    "corr_hvq_fuks": {
        "label":        r"Ph hvq + Py8 (gg+qq corr.) + $t\bar{t}_\mathrm{GFRW}$",
        "legend_label": r"+ $t\bar{t}_\mathrm{GFRW}$ [corr.]",
        "color": "#d62728", "marker": "o", "solid": True,  "no_err": False,
    },
    "corr_hvq_fuks_ms": {
        "label":        r"Ph hvq + Py8 (gg+qq corr.) + $t\bar{t}_\mathrm{GFRW}$ $-$ singlet",
        "legend_label": r"+ $t\bar{t}_\mathrm{GFRW}$ $-$ singlet [corr.]",
        "color": "#ff7f0e", "marker": "o", "solid": True,  "no_err": False,
    },
    "corr_hvq_toy": {
        "label":        r"Ph hvq + Py8 (gg+qq corr.) + $\eta_t$",
        "legend_label": r"+ $\eta_t$ [corr.]",
        "color": "#2ca02c", "marker": "o", "solid": True,  "no_err": False,
    },
    "nrc_nominal": {
        "label":        r"Ph hvq + Py8 (incl. NRC)",
        "legend_label": "Ph hvq + Py8 (incl. NRC)",
        "color": "black",   "marker": "^", "solid": True,  "no_err": True,
    },
    # ── MiNNLOps variant ───────────────────────────────────────────────────────
    "mnn": {
        "label":        r"Ph MiNNLOps + Py8",
        "legend_label": "Ph MiNNLOps + Py8 (no NRC)",
        "color": "#1f77b4", "marker": "o", "solid": False, "no_err": False,
    },
    "mnn_corr": {
        "label":        r"Ph MiNNLOps + Py8 (gg+qq corr.)",
        "legend_label": "Ph MiNNLOps + gg+qq corr.",
        "color": "#1f77b4", "marker": "o", "solid": True,  "no_err": False,
    },
    "mnn_fuks": {
        "label":        r"Ph MiNNLOps + Py8 + $t\bar{t}_\mathrm{GFRW}$",
        "legend_label": r"+ $t\bar{t}_\mathrm{GFRW}$",
        "color": "#d62728", "marker": "o", "solid": False, "no_err": False,
    },
    "mnn_fuks_ms": {
        "label":        r"Ph MiNNLOps + Py8 + $t\bar{t}_\mathrm{GFRW}$ $-$ singlet",
        "legend_label": r"+ $t\bar{t}_\mathrm{GFRW}$ $-$ singlet",
        "color": "#ff7f0e", "marker": "o", "solid": False, "no_err": False,
    },
    "mnn_toy": {
        "label":        r"Ph MiNNLOps + Py8 + $\eta_t$",
        "legend_label": r"+ $\eta_t$",
        "color": "#2ca02c", "marker": "o", "solid": False, "no_err": False,
    },
    "corr_mnn_fuks": {
        "label":        r"Ph MiNNLOps + Py8 (gg+qq corr.) + $t\bar{t}_\mathrm{GFRW}$",
        "legend_label": r"+ $t\bar{t}_\mathrm{GFRW}$ [corr.]",
        "color": "#d62728", "marker": "o", "solid": True,  "no_err": False,
    },
    "corr_mnn_fuks_ms": {
        "label":        r"Ph MiNNLOps + Py8 (gg+qq corr.) + $t\bar{t}_\mathrm{GFRW}$ $-$ singlet",
        "legend_label": r"+ $t\bar{t}_\mathrm{GFRW}$ $-$ singlet [corr.]",
        "color": "#ff7f0e", "marker": "o", "solid": True,  "no_err": False,
    },
    "corr_mnn_toy": {
        "label":        r"Ph MiNNLOps + Py8 (gg+qq corr.) + $\eta_t$",
        "legend_label": r"+ $\eta_t$ [corr.]",
        "color": "#2ca02c", "marker": "o", "solid": True,  "no_err": False,
    },
}

# Legend: one entry per configuration + two style indicators for hollow/solid circles.
# NRC nominal is a triangle and self-explanatory; the style indicators cover circles only.
LEGEND_CONFIG_KEYS = [
    "hvq",
    "hvq_fuks", "hvq_fuks_ms", "hvq_toy",
    "nrc_nominal",
]

LEGEND_CONFIG_KEYS_MNN = [
    "mnn",
    "mnn_fuks", "mnn_fuks_ms", "mnn_toy",
    "nrc_nominal",
]

# ── X-groups ───────────────────────────────────────────────────────────────────
# Each group is (hollow_key, solid_key|None).
# Pairs share the same x position; solid_key=None → single marker.

ALL_GROUPS = [
    ("hvq",        "hvq_corr"),
    ("hvq_fuks",   "corr_hvq_fuks"),
    ("hvq_fuks_ms","corr_hvq_fuks_ms"),
    ("hvq_toy",    "corr_hvq_toy"),
    ("nrc_nominal", None),
]

NO_TOP_GROUPS = [
    ("hvq",      "hvq_corr"),
]

WITH_TOP_GROUPS = [
    ("hvq_fuks",   "corr_hvq_fuks"),
    ("hvq_fuks_ms","corr_hvq_fuks_ms"),
    ("hvq_toy",    "corr_hvq_toy"),
    ("nrc_nominal", None),
]

ALL_GROUPS_MNN = [
    ("mnn",        "mnn_corr"),
    ("mnn_fuks",   "corr_mnn_fuks"),
    ("mnn_fuks_ms","corr_mnn_fuks_ms"),
    ("mnn_toy",    "corr_mnn_toy"),
    ("nrc_nominal", None),
]

NO_TOP_GROUPS_MNN = [
    ("mnn", "mnn_corr"),
]

WITH_TOP_GROUPS_MNN = [
    ("mnn_fuks",   "corr_mnn_fuks"),
    ("mnn_fuks_ms","corr_mnn_fuks_ms"),
    ("mnn_toy",    "corr_mnn_toy"),
    ("nrc_nominal", None),
]

PANEL_GROUPS_HVQ = [ALL_GROUPS, NO_TOP_GROUPS,     WITH_TOP_GROUPS,     ALL_GROUPS]
PANEL_GROUPS_MNN = [ALL_GROUPS_MNN, NO_TOP_GROUPS_MNN, WITH_TOP_GROUPS_MNN, ALL_GROUPS_MNN]

# ── Measurements ──────────────────────────────────────────────────────────────

MEASUREMENTS = [
    {
        "D": -0.537, "err_hi": 0.019, "err_lo": 0.019,
        "m_lo": 340.0, "m_hi": 380.0, "bz_max": None,
        "ref": "ATLAS dilepton, arXiv:2311.07288",
        "sel_tex":   r"340 < $m_{t\bar{t}}$ < 380 GeV",
        "sel_plain": "340 < m_tt < 380 GeV",
        "groups": ALL_GROUPS,
    },
    {
        "D": -0.491, "err_hi": 0.026, "err_lo": 0.025,
        "m_lo": 0.0, "m_hi": 400.0, "bz_max": 0.9,
        "ref": "CMS dilepton, arXiv:2406.03976",
        "sel_tex":   r"$m_{t\bar{t}}$ < 400 GeV,  $\beta_z$ < 0.9",
        "sel_plain": "m_tt < 400 GeV, beta_z < 0.9",
        "groups": NO_TOP_GROUPS,
    },
    {
        "D": -0.480, "err_hi": 0.026, "err_lo": 0.029,
        "m_lo": 0.0, "m_hi": 400.0, "bz_max": 0.9,
        "ref": "CMS dilepton, arXiv:2406.03976",
        "sel_tex":   r"$m_{t\bar{t}}$ < 400 GeV,  $\beta_z$ < 0.9",
        "sel_plain": "m_tt < 400 GeV, beta_z < 0.9",
        "groups": WITH_TOP_GROUPS,
    },
    {
        "D": -0.382, "err_hi": 0.030, "err_lo": 0.030,
        "m_lo": 300.0, "m_hi": 400.0, "bz_max": None,
        "ref": "CMS lepton+jets, arXiv:2409.11067",
        "sel_tex":   r"300 < $m_{t\bar{t}}$ < 400 GeV",
        "sel_plain": "300 < m_tt < 400 GeV",
        "groups": ALL_GROUPS,
    },
]


def _panel_keys(meas: dict) -> set[str]:
    keys: set[str] = set()
    for h, s in meas["groups"]:
        keys.add(h)
        if s:
            keys.add(s)
    return keys


# ── Data loading ───────────────────────────────────────────────────────────────

_CACHE: dict[str, pd.DataFrame] = {}

_PATHS = {
    "hvq":      DATA_DIR / "cache_hvq_afterFSR.parquet",
    "minnlops": DATA_DIR / "cache_minnlops_afterFSR.parquet",
    "nrc":      DATA_DIR / "cache_nrc_afterFSR.parquet",
    "fuks":     DATA_DIR / "cache_toponium_Fuks_afterFSR.parquet",
    "toy":      DATA_DIR / "cache_toponium_Toy_afterFSR.parquet",
    "singlet":  DATA_DIR / "cache_toponium_Singlet_afterFSR.parquet",
}


def _load(name: str) -> pd.DataFrame:
    if name not in _CACHE:
        df = pd.read_parquet(_PATHS[name])
        df[M_COL] = df[M_COL] / 1000.0   # MeV → GeV
        _CACHE[name] = df
    return _CACHE[name]


# ── Integrated D computation ───────────────────────────────────────────────────

def _sel_mask(df: pd.DataFrame, m_lo: float, m_hi: float,
              bz_max: float | None) -> np.ndarray:
    sel = (df[M_COL].to_numpy() >= m_lo) & (df[M_COL].to_numpy() <= m_hi)
    if bz_max is not None:
        sel &= df[BZ_COL].to_numpy() < bz_max
    return sel


def _xs_arrays(df: pd.DataFrame, w_col: str, xs: float,
               sel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalise to unit area (all events), scale by xs, then apply selection."""
    w_all = df[w_col].to_numpy()
    w_xs  = w_all / w_all.sum() * xs
    return df[COS_COL].to_numpy()[sel], w_xs[sel]


def _D_combine(cos_list: list[np.ndarray], w_list: list[np.ndarray],
               signs: list[int] | None = None) -> tuple[float, float]:
    """
    Integrated D from one or more (cos, w) arrays.
    signs: +1 add / -1 subtract each sample; variance always adds.
    """
    if signs is None:
        signs = [1] * len(cos_list)
    WC = sum(s * float(np.sum(w * c)) for s, w, c in zip(signs, w_list, cos_list))
    W  = sum(s * float(np.sum(w))     for s, w    in zip(signs, w_list))
    if W == 0.0:
        return np.nan, np.nan
    mean_cos = WC / W
    wc2 = sum(float(np.sum(w**2 * (c - mean_cos)**2)) for w, c in zip(w_list, cos_list))
    return -3.0 * mean_cos, 3.0 * np.sqrt(max(wc2 / W**2, 0.0))


def _corrected_sample(name: str, xs: float, m_lo: float, m_hi: float,
                      bz_max: float | None) -> tuple[np.ndarray, np.ndarray]:
    """XS-normalised sample with gg+qq correction below threshold (requires PDFinfo columns)."""
    df  = _load(name)
    sel = _sel_mask(df, m_lo, m_hi, bz_max)
    m_s   = df[M_COL].to_numpy()[sel]
    cos_s = df[COS_COL].to_numpy()[sel].copy()
    w_all = df[W_NOM].to_numpy()
    w_xs  = w_all[sel] / w_all.sum() * xs
    id1 = np.abs(df["PDFinfo_PDGID1"].to_numpy()[sel])
    id2 = np.abs(df["PDFinfo_PDGID2"].to_numpy()[sel])
    below = m_s < THRESHOLD
    cos_s[below & (id1 == 21) & (id2 == 21)] =  1.0 / 3.0
    cos_s[below & (id1 != 21) & (id2 != 21)] = -1.0 / 9.0
    return cos_s, w_xs


def compute_D(key: str, m_lo: float, m_hi: float,
              bz_max: float | None) -> tuple[float, float]:

    if key == "hvq":
        df  = _load("hvq")
        sel = _sel_mask(df, m_lo, m_hi, bz_max)
        cos, w = _xs_arrays(df, W_NOM, XS_HVQ, sel)
        return _D_combine([cos], [w])

    elif key == "hvq_corr":
        cos_c, w_c = _corrected_sample("hvq", XS_HVQ, m_lo, m_hi, bz_max)
        return _D_combine([cos_c], [w_c])

    elif key == "hvq_fuks":
        df_hvq, df_fuks = _load("hvq"), _load("fuks")
        sel_hvq  = _sel_mask(df_hvq,  m_lo, m_hi, bz_max)
        sel_fuks = _sel_mask(df_fuks, m_lo, m_hi, bz_max)
        cos_hvq,  w_hvq  = _xs_arrays(df_hvq,  W_NOM, XS_HVQ,      sel_hvq)
        cos_fuks, w_fuks = _xs_arrays(df_fuks, W_NOM, XS_TOPONIUM, sel_fuks)
        return _D_combine([cos_hvq, cos_fuks], [w_hvq, w_fuks])

    elif key == "hvq_toy":
        df_hvq, df_toy = _load("hvq"), _load("toy")
        sel_hvq = _sel_mask(df_hvq, m_lo, m_hi, bz_max)
        sel_toy = _sel_mask(df_toy, m_lo, m_hi, bz_max)
        cos_hvq, w_hvq = _xs_arrays(df_hvq, W_NOM, XS_HVQ,      sel_hvq)
        cos_toy, w_toy = _xs_arrays(df_toy, W_NOM, XS_TOPONIUM, sel_toy)
        return _D_combine([cos_hvq, cos_toy], [w_hvq, w_toy])

    elif key == "hvq_fuks_ms":
        df_hvq, df_fuks = _load("hvq"), _load("fuks")
        df_singlet      = _load("singlet")
        sel_hvq     = _sel_mask(df_hvq,     m_lo, m_hi, bz_max)
        sel_fuks    = _sel_mask(df_fuks,    m_lo, m_hi, bz_max)
        sel_singlet = _sel_mask(df_singlet, m_lo, m_hi, bz_max)
        cos_hvq,     w_hvq     = _xs_arrays(df_hvq,     W_NOM, XS_HVQ,      sel_hvq)
        cos_fuks,    w_fuks    = _xs_arrays(df_fuks,    W_NOM, XS_TOPONIUM, sel_fuks)
        cos_singlet, w_singlet = _xs_arrays(df_singlet, W_NOM, XS_SINGLET,  sel_singlet)
        return _D_combine(
            [cos_hvq, cos_fuks, cos_singlet],
            [w_hvq,   w_fuks,   w_singlet],
            signs=[1, 1, -1],
        )

    elif key == "corr_hvq_fuks":
        cos_c, w_c = _corrected_sample("hvq", XS_HVQ, m_lo, m_hi, bz_max)
        df_fuks  = _load("fuks")
        sel_fuks = _sel_mask(df_fuks, m_lo, m_hi, bz_max)
        cos_fuks, w_fuks = _xs_arrays(df_fuks, W_NOM, XS_TOPONIUM, sel_fuks)
        return _D_combine([cos_c, cos_fuks], [w_c, w_fuks])

    elif key == "corr_hvq_toy":
        cos_c, w_c = _corrected_sample("hvq", XS_HVQ, m_lo, m_hi, bz_max)
        df_toy  = _load("toy")
        sel_toy = _sel_mask(df_toy, m_lo, m_hi, bz_max)
        cos_toy, w_toy = _xs_arrays(df_toy, W_NOM, XS_TOPONIUM, sel_toy)
        return _D_combine([cos_c, cos_toy], [w_c, w_toy])

    elif key == "corr_hvq_fuks_ms":
        cos_c, w_c = _corrected_sample("hvq", XS_HVQ, m_lo, m_hi, bz_max)
        df_fuks    = _load("fuks")
        df_singlet = _load("singlet")
        sel_fuks    = _sel_mask(df_fuks,    m_lo, m_hi, bz_max)
        sel_singlet = _sel_mask(df_singlet, m_lo, m_hi, bz_max)
        cos_fuks,    w_fuks    = _xs_arrays(df_fuks,    W_NOM, XS_TOPONIUM, sel_fuks)
        cos_singlet, w_singlet = _xs_arrays(df_singlet, W_NOM, XS_SINGLET,  sel_singlet)
        return _D_combine(
            [cos_c, cos_fuks, cos_singlet],
            [w_c,   w_fuks,   w_singlet],
            signs=[1, 1, -1],
        )

    elif key == "nrc_nominal":
        df  = _load("nrc")
        sel = _sel_mask(df, m_lo, m_hi, bz_max)
        cos, w = _xs_arrays(df, W_NOM, XS_HVQ, sel)
        return _D_combine([cos], [w])

    # ── MiNNLOps variant ───────────────────────────────────────────────────────
    elif key == "mnn":
        df  = _load("minnlops")
        sel = _sel_mask(df, m_lo, m_hi, bz_max)
        cos, w = _xs_arrays(df, W_NOM, XS_HVQ, sel)
        return _D_combine([cos], [w])

    elif key == "mnn_corr":
        cos_c, w_c = _corrected_sample("minnlops", XS_HVQ, m_lo, m_hi, bz_max)
        return _D_combine([cos_c], [w_c])

    elif key == "mnn_fuks":
        df_mnn  = _load("minnlops"); df_fuks = _load("fuks")
        sel_mnn  = _sel_mask(df_mnn,  m_lo, m_hi, bz_max)
        sel_fuks = _sel_mask(df_fuks, m_lo, m_hi, bz_max)
        cos_mnn,  w_mnn  = _xs_arrays(df_mnn,  W_NOM, XS_HVQ,      sel_mnn)
        cos_fuks, w_fuks = _xs_arrays(df_fuks, W_NOM, XS_TOPONIUM, sel_fuks)
        return _D_combine([cos_mnn, cos_fuks], [w_mnn, w_fuks])

    elif key == "mnn_toy":
        df_mnn = _load("minnlops"); df_toy = _load("toy")
        sel_mnn = _sel_mask(df_mnn, m_lo, m_hi, bz_max)
        sel_toy = _sel_mask(df_toy, m_lo, m_hi, bz_max)
        cos_mnn, w_mnn = _xs_arrays(df_mnn, W_NOM, XS_HVQ,      sel_mnn)
        cos_toy, w_toy = _xs_arrays(df_toy, W_NOM, XS_TOPONIUM, sel_toy)
        return _D_combine([cos_mnn, cos_toy], [w_mnn, w_toy])

    elif key == "mnn_fuks_ms":
        df_mnn = _load("minnlops"); df_fuks = _load("fuks"); df_singlet = _load("singlet")
        sel_mnn     = _sel_mask(df_mnn,     m_lo, m_hi, bz_max)
        sel_fuks    = _sel_mask(df_fuks,    m_lo, m_hi, bz_max)
        sel_singlet = _sel_mask(df_singlet, m_lo, m_hi, bz_max)
        cos_mnn,     w_mnn     = _xs_arrays(df_mnn,     W_NOM, XS_HVQ,      sel_mnn)
        cos_fuks,    w_fuks    = _xs_arrays(df_fuks,    W_NOM, XS_TOPONIUM, sel_fuks)
        cos_singlet, w_singlet = _xs_arrays(df_singlet, W_NOM, XS_SINGLET,  sel_singlet)
        return _D_combine([cos_mnn, cos_fuks, cos_singlet],
                          [w_mnn,   w_fuks,   w_singlet], signs=[1, 1, -1])

    elif key == "corr_mnn_fuks":
        cos_c, w_c = _corrected_sample("minnlops", XS_HVQ, m_lo, m_hi, bz_max)
        df_fuks  = _load("fuks")
        sel_fuks = _sel_mask(df_fuks, m_lo, m_hi, bz_max)
        cos_fuks, w_fuks = _xs_arrays(df_fuks, W_NOM, XS_TOPONIUM, sel_fuks)
        return _D_combine([cos_c, cos_fuks], [w_c, w_fuks])

    elif key == "corr_mnn_toy":
        cos_c, w_c = _corrected_sample("minnlops", XS_HVQ, m_lo, m_hi, bz_max)
        df_toy  = _load("toy")
        sel_toy = _sel_mask(df_toy, m_lo, m_hi, bz_max)
        cos_toy, w_toy = _xs_arrays(df_toy, W_NOM, XS_TOPONIUM, sel_toy)
        return _D_combine([cos_c, cos_toy], [w_c, w_toy])

    elif key == "corr_mnn_fuks_ms":
        cos_c, w_c = _corrected_sample("minnlops", XS_HVQ, m_lo, m_hi, bz_max)
        df_fuks = _load("fuks"); df_singlet = _load("singlet")
        sel_fuks    = _sel_mask(df_fuks,    m_lo, m_hi, bz_max)
        sel_singlet = _sel_mask(df_singlet, m_lo, m_hi, bz_max)
        cos_fuks,    w_fuks    = _xs_arrays(df_fuks,    W_NOM, XS_TOPONIUM, sel_fuks)
        cos_singlet, w_singlet = _xs_arrays(df_singlet, W_NOM, XS_SINGLET,  sel_singlet)
        return _D_combine([cos_c, cos_fuks, cos_singlet],
                          [w_c,   w_fuks,   w_singlet], signs=[1, 1, -1])

    else:
        raise ValueError(f"Unknown key: {key!r}")


# ── Terminal table ─────────────────────────────────────────────────────────────

def print_table(results: dict, panel_groups: list | None = None) -> None:
    if panel_groups is None:
        panel_groups = [m["groups"] for m in MEASUREMENTS]
    for pi, meas in enumerate(MEASUREMENTS):
        print(f"\nPanel {pi + 1}:  {meas['ref']}   [{meas['sel_plain']}]")
        print(f"  {'Measurement':55s}  D = {meas['D']:+.3f}  "
              f"+{meas['err_hi']:.3f} / -{meas['err_lo']:.3f}")
        print(f"  {'─' * 75}")
        for hollow_key, solid_key in panel_groups[pi]:
            for k, style in [(hollow_key, "hollow"), (solid_key, "solid ")]:
                if k is None:
                    continue
                D_val, err = results[(pi, k)]
                lbl = SETUPS[k]["label"]
                print(f"  [{style}] {lbl:55s}  D = {D_val:+.3f} ± {err:.3f}")
    print()


# ── Plot ───────────────────────────────────────────────────────────────────────

def _errorbar_kw(key: str) -> dict:
    s = SETUPS[key]
    return dict(
        color=s["color"],
        marker=s["marker"],
        ms=6.5,
        mfc=s["color"] if s["solid"] else "none",
        mew=1.5,
        capsize=0,
        lw=0,
        elinewidth=1.2,
        zorder=3,
    )


def make_plot(results: dict,
              panel_groups: list | None = None,
              legend_config_keys: list | None = None,
              outfile: str = "compare_D_measurements.pdf") -> None:
    if panel_groups is None:
        panel_groups = [m["groups"] for m in MEASUREMENTS]
    if legend_config_keys is None:
        legend_config_keys = LEGEND_CONFIG_KEYS
    n_panels = len(MEASUREMENTS)
    n_shown  = [len(g) for g in panel_groups]

    fig = plt.figure(figsize=(10, 5.5))
    gs  = gridspec.GridSpec(
        1, n_panels,
        width_ratios=n_shown,
        figure=fig,
        wspace=0.03,
    )
    ax0  = fig.add_subplot(gs[0])
    axes = [ax0] + [fig.add_subplot(gs[i], sharey=ax0) for i in range(1, n_panels)]

    # Dynamic y range
    all_D = [m["D"] for m in MEASUREMENTS]
    for (pi, key), (D_val, err) in results.items():
        if not np.isnan(D_val):
            all_D += [D_val - err, D_val + err]
    ylo = min(all_D) - 0.08
    yhi = max(all_D) + 0.08

    for pi, (ax, meas) in enumerate(zip(axes, MEASUREMENTS)):
        groups = panel_groups[pi]
        n      = len(groups)

        # Measurement band
        ax.axhspan(meas["D"] - meas["err_lo"],
                   meas["D"] + meas["err_hi"],
                   color="silver", alpha=0.55, zorder=0)
        ax.axhline(meas["D"], color="#555555", lw=1.2, zorder=1)

        # Predictions
        for xi, (hollow_key, solid_key) in enumerate(groups):
            for k in (hollow_key, solid_key):
                if k is None:
                    continue
                D_val, err = results[(pi, k)]
                s = SETUPS[k]
                if s["no_err"]:
                    # Marker only — no error bar
                    ax.plot(xi, D_val,
                            marker=s["marker"], ms=8,
                            color=s["color"],
                            mfc=s["color"] if s["solid"] else "none",
                            mew=1.5, ls="", zorder=4)
                else:
                    ax.errorbar(xi, D_val, yerr=err, **_errorbar_kw(k))

        # Axis styling
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(ylo, yhi)
        ax.axhline(0, color="k", lw=0.5, ls=":", zorder=0)
        ax.set_xticks([])

        if pi == 0:
            ax.set_ylabel(r"$D = -3\,\mathrm{Tr}[C]$", fontsize=12)
        else:
            ax.tick_params(labelleft=False, left=False)

        # Hide all spines; border rectangle drawn in figure coords after layout
        for side in ("top", "bottom", "left", "right"):
            ax.spines[side].set_visible(False)
        # Interior separators between pi=0/1 and pi=2/3 only (not pi=1/2)
        if pi in (0, 2):
            ax.spines["right"].set_visible(True)
            ax.spines["right"].set_bounds(ylo, -0.40)

    # Spin@Threshold'26
    axes[0].text(0.04, 0.97, "Spin@Threshold'26",
                 transform=axes[0].transAxes, ha="left", va="top",
                 fontsize=11, fontweight="bold")

    # ── Legend (inside the plot, spanning panels 1–2) ────────────────────────
    def _proxy(key):
        s = SETUPS[key]
        return Line2D([0], [0],
                      color=s["color"], marker=s["marker"], ms=7,
                      mfc=s["color"] if s["solid"] else "none",
                      mew=1.5, lw=0, ls="")

    leg_handles = [_proxy(k) for k in legend_config_keys]
    leg_labels  = [SETUPS[k]["legend_label"] for k in legend_config_keys]

    # Style indicators: hollow circle = uncorrected, solid circle = gg+qq corrected
    leg_handles += [
        Line2D([0], [0], color="grey", marker="o", ms=7,
               mfc="none", mew=1.5, lw=0, ls=""),
        Line2D([0], [0], color="grey", marker="o", ms=7,
               mfc="grey", mew=0,   lw=0, ls=""),
    ]
    leg_labels += ["uncorrected", "gg+qq corrected"]

    # Figure-level legend anchored just below the Spin@Threshold'26 label in axes[0].
    # Using fig.legend() ensures it is never clipped by adjacent panels.
    fig.legend(
        leg_handles, leg_labels,
        loc="upper left",
        bbox_to_anchor=(0.04, 0.89),
        bbox_transform=axes[0].transAxes,
        ncol=2, fontsize=8.5,
        frameon=False,
        handlelength=0.8, handletextpad=0.4, columnspacing=1.0,
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.subplots_adjust(top=0.94, bottom=0.26, left=0.08, right=0.99)

    # ── Continuous border rectangle around all panels ─────────────────────────
    ppos = [ax.get_position() for ax in axes]
    fig.add_artist(MplRect(
        (ppos[0].x0, ppos[0].y0),
        ppos[-1].x1 - ppos[0].x0,
        ppos[0].y1 - ppos[0].y0,
        linewidth=0.8, edgecolor="black", facecolor="none",
        transform=fig.transFigure, clip_on=False, zorder=10,
    ))
    LH   = 0.040

    def ftxt(xc, y, text, **kw):
        fig.text(xc, y, text, ha="center", va="top", **kw)

    for pi in [0, 3]:
        xc   = (ppos[pi].x0 + ppos[pi].x1) / 2
        ybot = ppos[pi].y0 - 0.010
        ftxt(xc, ybot,      MEASUREMENTS[pi]["ref"],
             fontsize=8, style="italic", color="#444444")
        ftxt(xc, ybot - LH, MEASUREMENTS[pi]["sel_tex"], fontsize=8)

    ybot      = ppos[1].y0 - 0.010
    xc_shared = (ppos[1].x0 + ppos[2].x1) / 2
    ftxt(xc_shared, ybot,      MEASUREMENTS[1]["ref"],
         fontsize=8, style="italic", color="#444444")
    ftxt(xc_shared, ybot - LH, MEASUREMENTS[1]["sel_tex"], fontsize=8)

    # "without/with η_t" inside panels 1 & 2, horizontally centred in each panel,
    # at a fixed data-y so they are vertically aligned (shared y-axis).
    y_eta = -0.53
    for ax_i, label in [(axes[1], r"without $\eta_t$"), (axes[2], r"with $\eta_t$")]:
        tr = blended_transform_factory(ax_i.transAxes, ax_i.transData)
        ax_i.text(0.5, y_eta, label, transform=tr,
                  ha="center", va="top", fontsize=8.5, color="#444444")

    # ── Save ──────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / outfile
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def _compute_for_variant(panel_groups: list) -> dict:
    results: dict[tuple[int, str], tuple[float, float]] = {}
    for pi, (meas, groups) in enumerate(zip(MEASUREMENTS, panel_groups)):
        seen: set[str] = set()
        for hollow_key, solid_key in groups:
            for key in (hollow_key, solid_key):
                if key is None or key in seen:
                    continue
                seen.add(key)
                results[(pi, key)] = compute_D(key, meas["m_lo"], meas["m_hi"], meas["bz_max"])
    return results


def main() -> None:
    print("Loading data …")
    for name in ["hvq", "minnlops", "nrc", "fuks", "toy", "singlet"]:
        _load(name)

    print("\n── Ph hvq variant ────────────────────────────────────────────────────────")
    results_hvq = _compute_for_variant(PANEL_GROUPS_HVQ)
    print_table(results_hvq, PANEL_GROUPS_HVQ)
    make_plot(results_hvq,
              panel_groups=PANEL_GROUPS_HVQ,
              legend_config_keys=LEGEND_CONFIG_KEYS,
              outfile="compare_D_measurements.pdf")

    print("\n── Ph MiNNLOps variant ───────────────────────────────────────────────────")
    results_mnn = _compute_for_variant(PANEL_GROUPS_MNN)
    print_table(results_mnn, PANEL_GROUPS_MNN)
    make_plot(results_mnn,
              panel_groups=PANEL_GROUPS_MNN,
              legend_config_keys=LEGEND_CONFIG_KEYS_MNN,
              outfile="compare_D_measurements_minnlops.pdf")


if __name__ == "__main__":
    main()
