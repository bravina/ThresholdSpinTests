# Spin@Threshold'26 — ttbar spin correlation analysis

Analysis code producing publication-quality plots for the Spin@Threshold'26 study of top quark pair spin correlations near the ttbar production threshold.

## Setup

Requires [uv](https://github.com/astral-sh/uv). Install dependencies and run:

```bash
uv run python analysis.py
```

Plots are written as 300 dpi PDFs to `plots/`.

## Inputs

Parquet cache files in `input_data/`, one per sample:

| File | Description |
|------|-------------|
| `cache_hvq_afterFSR.parquet` | Powheg hvq + Pythia 8 (baseline) |
| `cache_amcatnlo_afterFSR.parquet` | aMC@NLO + Pythia 8 |
| `cache_nrc_afterFSR.parquet` | Powheg hvq with NRC weight variations (nominal, NRCi0, Coulomb band) |
| `cache_toponium_Fuks_afterFSR.parquet` | Toponium signal — GFRW model |
| `cache_toponium_Toy_afterFSR.parquet` | Toponium signal — toy η_t model |
| `cache_MadGraph_LO_lhe.parquet` | MadGraph LO (parton-level, from LHE) |

All samples use after-FSR kinematics. NRC files cover the full phase space (mttbar up to ~6 TeV); events outside 300–500 GeV contribute to normalisation but are not plotted.

## Cross sections

Used to set the relative normalisation when combining NRCi0 with the toponium signal:

- Powheg hvq (NRCi0): **87.87 pb** (total, full phase space)
- Toponium signal (Fuks and Toy): **0.675 pb** each (total, full phase space)

The NRC nominal cross section is derived as `87.87 × sum(w_nom) / sum(w_i0)`.

## Plots

| # | Output file | Description |
|---|-------------|-------------|
| 1 | `plot1_mttbar_generators.pdf` | mttbar shape (300–500 GeV), log scale: hvq vs aMC@NLO vs MadGraph |
| 2 | `plot2_D_generators.pdf` | D vs mttbar: same three generators |
| 3 | `plot3_D_channels.pdf` | D per production channel (gg / qq / qg), three generators |
| 4 | `plot4_D_corrections_diff.pdf` | D vs mttbar for hvq: no correction vs gg-corrected vs gg+qq-corrected below threshold |
| 5 | `plot5_D_corrections_cum.pdf` | Cumulative D (integrated from 300 GeV) for the same three corrections |
| 6 | `plot6_mttbar_nrc.pdf` | dσ/dm near threshold: NRCi0 vs NRC nominal with Coulomb scale variation band |
| 7 | `plot7_mttbar_toponium.pdf` | dσ/dm near threshold: NRCi0 baseline + NRC nominal + NRCi0+Fuks + NRCi0+Toy, with Coulomb band |
| 8 | `plot8_D_toponium.pdf` | D vs mttbar: NRC nominal vs NRCi0+Fuks vs NRCi0+Toy |

Plots 6 and 7 show absolute dσ/dm [pb/GeV]; all others show normalised shapes or D.

## Observable

The spin correlation observable is:

```
D = -3 Tr[C]
```

computed per mttbar bin as `-3 × Σ(w·cosφ) / Σ(w)`, where φ is the helicity-basis opening angle between the decay leptons and cosφ is read from `spin_afterFSR_cos_phi_NOSYS`.

Limiting values: D = −1 for pure gg production (maximally correlated), D = +1/3 for pure qq production.
