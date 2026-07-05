#!/usr/bin/env python3
"""
ROOT → parquet converter for Spin@Threshold'26 analysis.

Reads truth ntuples for all five MC samples and writes compact parquet
caches containing only the branches needed by analysis.py, plus the
derived branch beta_z = |(pz_top + pz_tbar) / (E_top + E_tbar)|.

Usage:
    uv run python make_cache.py [--sample SAMPLE] [--no-cache] [--fsr FSR]

    --sample SAMPLE   process only this sample: hvq, amcatnlo, toy, fuks, nrc
                      (default: all)
    --no-cache        force rebuild even if the cache is newer than the ROOT files
    --fsr FSR         afterFSR or beforeFSR  (default: afterFSR)
"""

from __future__ import annotations

import argparse
import glob
import multiprocessing
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import uproot
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
#  Global settings
# ─────────────────────────────────────────────────────────────────────────────

TREE_NAME      = "truth"
CHUNK_SIZE     = 1_000_000   # events per uproot chunk
N_FILE_WORKERS = 8           # files processed in parallel (one process each)
N_IO_THREADS   = 4           # uproot decompression/interpretation threads per worker
DATA_DIR       = Path("input_data")

# ─────────────────────────────────────────────────────────────────────────────
#  Top 4-vector branch names (used only to compute beta_z; not saved to cache)
#
#  Branches: Ttbar_MC_t_{fsr}_pt, _eta, _phi, _m  (and tbar equivalents).
#  4-momentum convention: pt [MeV], eta, phi, m [MeV].
#  E and pz are derived:
#    pz = pt * sinh(eta)
#    E  = sqrt( (pt * cosh(eta))^2 + m^2 )
# ─────────────────────────────────────────────────────────────────────────────

def _top_branches(fsr: str) -> list[str]:
    return [
        f"Ttbar_MC_t_{fsr}_pt",
        f"Ttbar_MC_t_{fsr}_eta",
        f"Ttbar_MC_t_{fsr}_m",
        f"Ttbar_MC_tbar_{fsr}_pt",
        f"Ttbar_MC_tbar_{fsr}_eta",
        f"Ttbar_MC_tbar_{fsr}_m",
    ]


def _compute_beta_z(chunk: dict, fsr: str) -> np.ndarray:
    """
    beta_z = |(pz_top + pz_tbar)| / (E_top + E_tbar)

    From pt/eta/m storage:
      pz = pt * sinh(eta)
      E  = sqrt( (pt * cosh(eta))^2 + m^2 )

    Units [MeV] cancel in the ratio, so beta_z is dimensionless.
    """
    pt_t    = chunk[f"Ttbar_MC_t_{fsr}_pt"]
    eta_t   = chunk[f"Ttbar_MC_t_{fsr}_eta"]
    m_t     = chunk[f"Ttbar_MC_t_{fsr}_m"]
    pt_tb   = chunk[f"Ttbar_MC_tbar_{fsr}_pt"]
    eta_tb  = chunk[f"Ttbar_MC_tbar_{fsr}_eta"]
    m_tb    = chunk[f"Ttbar_MC_tbar_{fsr}_m"]

    # overflow="ignore": sentinel eta values (e.g. -999) cause sinh/cosh to
    # overflow to inf; both pz and E then become inf, their ratio becomes nan,
    # which is the correct result for unphysical events.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        pz_t  = pt_t  * np.sinh(eta_t)
        pz_tb = pt_tb * np.sinh(eta_tb)
        E_t   = np.sqrt((pt_t  * np.cosh(eta_t))**2  + m_t**2)
        E_tb  = np.sqrt((pt_tb * np.cosh(eta_tb))**2 + m_tb**2)
        denom = E_t + E_tb
        bz = np.where(denom > 0, np.abs(pz_t + pz_tb) / denom, np.nan)
    return bz

# ─────────────────────────────────────────────────────────────────────────────
#  Per-sample configuration
# ─────────────────────────────────────────────────────────────────────────────

NRC_WEIGHTS = [
    "weight_mc_GEN_ithreshold0",
    "weight_mc_GEN_ithreshold33_coulombScaleFact05",
    "weight_mc_GEN_ithreshold33_coulombScaleFact2",
]

SAMPLES: dict[str, dict] = {
    "hvq": {
        "input_dir": (
            "user.ravinab.410472.PhPy8EG.DAOD_PHYSLITE.e6348_s3681_r13167_p7018.ThresholdSpinTestFinalV1_output"
        ),
        "cache": DATA_DIR / "cache_hvq_afterFSR.parquet",
        "extra_branches": ["PDFinfo_PDGID1", "PDFinfo_PDGID2"],
        "nrc_weights": False,
    },
    "madspin": {
        "input_dir": (
            "user.ravinab.603013.PhPy8EG.DAOD_PHYS.e8557_s3797_r13144_p6266.ThresholdSpinTestFinalV1_output"
        ),
        "cache": DATA_DIR / "cache_madspin_afterFSR.parquet",
        "extra_branches": ["PDFinfo_PDGID1", "PDFinfo_PDGID2"],
        "nrc_weights": False,
    },
    "amcatnlo": {
        "input_dir": (
            "user.ravinab.521294.aMCatNLO_ttbar_dilep.aMCatNLO_ttbar_dilep.TRUTH3_EXT0_EXT0.ThresholdSpinTestFinalV1_output"
        ),
        "cache": DATA_DIR / "cache_amcatnlo_afterFSR.parquet",
        "extra_branches": ["PDFinfo_PDGID1", "PDFinfo_PDGID2"],
        "nrc_weights": False,
    },
    "toy": {
        "input_dir": (
            "user.ravinab.521385.MGPy8EG.DAOD_PHYSLITE.e8588_s4231_r13167_p6697.ThresholdSpinTestFinalV1_output"
        ),
        "cache": DATA_DIR / "cache_toponium_Toy_afterFSR.parquet",
        "extra_branches": [],
        "nrc_weights": False,
    },
    "fuks": {
        "input_dir": (
            "user.ravinab.802380.Py8EG_Toponium_2L.DAOD_PHYSLITE.e8562_s4231_r13167_p6697.ThresholdSpinTestFinalV1_output"
        ),
        "cache": DATA_DIR / "cache_toponium_Fuks_afterFSR.parquet",
        "extra_branches": [],
        "nrc_weights": False,
    },
    "singlet": {
        "input_dir": (
            "user.ravinab.802381.Py8EG.DAOD_PHYSLITE.e8562_s4231_r13167_p6697.ThresholdSpinTestFinalV1_output"
        ),
        "cache": DATA_DIR / "cache_toponium_Singlet_afterFSR.parquet",
        "extra_branches": [],
        "nrc_weights": False,
    },
    "nrc": {
        "input_dir": (
            "user.ravinab.ravinab.Powheg.ravinab.Powheg_hvq_dilep_NRC_Pythia8_TRUTH3_EXT0.ThresholdSpinTestFinalV1_output"
        ),
        "cache": DATA_DIR / "cache_nrc_afterFSR.parquet",
        "extra_branches": [],
        "nrc_weights": True,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_root_files(directory: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(directory, "*.root")))
    if not files:
        raise FileNotFoundError(f"No .root files found in {directory!r}")
    return files


def cache_is_fresh(cache_path: Path, root_files: list[str]) -> bool:
    if not cache_path.exists():
        return False
    cache_mtime = cache_path.stat().st_mtime
    return all(cache_mtime >= os.path.getmtime(p) for p in root_files)


def _branches_for_sample(name: str, fsr: str) -> tuple[list[str], list[str]]:
    """
    Return (save_branches, load_branches).

    save_branches: columns written to the parquet file.
    load_branches: superset — includes temporary 4-vector branches for beta_z.
    """
    cfg = SAMPLES[name]
    save = (
        ["weight_mc_NOSYS"]
        + (NRC_WEIGHTS if cfg["nrc_weights"] else [])
        + [f"spin_{fsr}_cos_phi_NOSYS", f"spin_{fsr}_m_ttbar_NOSYS"]
        + cfg["extra_branches"]
        + ["beta_z"]           # computed, not in ROOT file
    )
    load = (
        ["weight_mc_NOSYS"]
        + (NRC_WEIGHTS if cfg["nrc_weights"] else [])
        + [f"spin_{fsr}_cos_phi_NOSYS", f"spin_{fsr}_m_ttbar_NOSYS"]
        + cfg["extra_branches"]
        + _top_branches(fsr)   # needed only for beta_z, not saved
    )
    return save, list(dict.fromkeys(load))   # deduplicate, preserve order


# ─────────────────────────────────────────────────────────────────────────────
#  Core reader
# ─────────────────────────────────────────────────────────────────────────────

def _read_file_to_parquet(job: tuple) -> tuple[str, int, int]:
    """
    Worker function: read one ROOT file, compute beta_z, drop invalid events,
    write to a temp parquet file.  Returns (temp_path, n_rows, n_dropped).
    Must be a top-level function so multiprocessing can pickle it.
    """
    path, tree_name, read_cols, save_cols, fsr, temp_path = job

    acc: dict[str, list] = {c: [] for c in read_cols}
    acc["beta_z"] = []

    with uproot.open(f"{path}:{tree_name}") as tree:
        for chunk in tree.iterate(
            read_cols,
            library="np",
            step_size=CHUNK_SIZE,
            decompression_executor=uproot.ThreadPoolExecutor(N_IO_THREADS),
            interpretation_executor=uproot.ThreadPoolExecutor(N_IO_THREADS),
        ):
            for col in read_cols:
                acc[col].append(chunk[col])
            acc["beta_z"].append(_compute_beta_z(chunk, fsr))

    arrays = {col: np.concatenate(acc[col]) for col in read_cols}
    arrays["beta_z"] = np.concatenate(acc["beta_z"])

    df = pd.DataFrame({col: arrays[col] for col in save_cols})
    n_before = len(df)
    df = df[df["beta_z"].notna()].reset_index(drop=True)
    n_dropped = n_before - len(df)

    df.to_parquet(temp_path, index=False, compression="snappy")
    return temp_path, len(df), n_dropped


def read_sample(name: str, fsr: str, no_cache: bool) -> None:
    cfg        = SAMPLES[name]
    cache_path = Path(str(cfg["cache"]).replace("afterFSR", fsr))
    input_dir  = cfg["input_dir"]

    print(f"\n{'='*60}")
    print(f"  Sample : {name}")
    print(f"  FSR    : {fsr}")
    print(f"  Input  : {input_dir}")
    print(f"  Cache  : {cache_path}")
    print(f"{'='*60}")

    root_files = find_root_files(input_dir)
    print(f"  Found {len(root_files)} ROOT file(s)")

    if not no_cache and cache_is_fresh(cache_path, root_files):
        print("  Cache is up to date — skipping.")
        return

    save_cols, load_cols = _branches_for_sample(name, fsr)
    read_cols = [c for c in load_cols if c != "beta_z"]

    n_workers = min(N_FILE_WORKERS, len(root_files))

    with tempfile.TemporaryDirectory() as tmp_dir:
        jobs = [
            (path, TREE_NAME, read_cols, save_cols, fsr,
             os.path.join(tmp_dir, f"part_{i:04d}.parquet"))
            for i, path in enumerate(root_files)
        ]

        print(f"  Processing {len(root_files)} file(s) with {n_workers} worker(s) …")
        with multiprocessing.Pool(processes=n_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(_read_file_to_parquet, jobs),
                total=len(jobs), desc="  Files", unit="file",
            ))

        total_rows    = sum(r[1] for r in results)
        total_dropped = sum(r[2] for r in results)
        temp_paths    = sorted(r[0] for r in results)  # sorted for deterministic order

        if total_dropped:
            print(f"  {total_rows:,} rows kept, {total_dropped:,} dropped "
                  f"(invalid top reconstruction)")
        else:
            print(f"  {total_rows:,} rows")

        print("  Merging and writing cache …", end=" ", flush=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tables = [pq.read_table(p) for p in temp_paths]
        pq.write_table(pa.concat_tables(tables), str(cache_path), compression="snappy")

    size_mb = cache_path.stat().st_size / 1e6
    print(f"{size_mb:.0f} MB  →  {cache_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", choices=list(SAMPLES), default=None,
                        help="Process a single sample (default: all)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force rebuild even if cache is up to date")
    parser.add_argument("--fsr", default="afterFSR",
                        choices=["afterFSR", "beforeFSR"],
                        help="FSR setting (default: afterFSR)")
    args = parser.parse_args()

    targets = [args.sample] if args.sample else list(SAMPLES)

    for name in targets:
        try:
            read_sample(name, args.fsr, args.no_cache)
        except FileNotFoundError as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            print(f"  Skipping {name}.", file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    main()
