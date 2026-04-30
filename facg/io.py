"""
Flexible file I/O for FACG.

- Reads multiple formats: text/dat/csv, excel, fits table (no header required).
- Output files use descriptive names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_timeseries(
    filepath: str | Path,
    time_col: int = 0,
    data_col: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a two-column (or more) time-series file.

    Parameters
    ----------
    filepath : path-like
        Any readable text file with at least *time_col* and *data_col*
        numeric columns.
    time_col, data_col : int
        0-based column indices.

    Returns
    -------
    t, x : 1-D float64 arrays
        Time stamps and observed values, sorted by time.
    """
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"Input file not found: {filepath}")

    ext = filepath.suffix.lower()
    max_col = max(time_col, data_col)

    if ext == '.csv':
        import pandas as pd
        df = pd.read_csv(filepath, comment='#', header=None)
        if df.shape[1] <= max_col:
            raise ValueError(f"File has {df.shape[1]} columns but column {max_col} requested.")
        t = df.iloc[:, time_col].to_numpy().astype(np.float64)
        x = df.iloc[:, data_col].to_numpy().astype(np.float64)
        
    elif ext in ['.xls', '.xlsx']:
        import pandas as pd
        df = pd.read_excel(filepath, header=None)
        if df.shape[1] <= max_col:
            raise ValueError(f"File has {df.shape[1]} columns but column {max_col} requested.")
        t = df.iloc[:, time_col].to_numpy().astype(np.float64)
        x = df.iloc[:, data_col].to_numpy().astype(np.float64)
        
    elif ext in ['.fits', '.fit']:
        from astropy.io import fits
        with fits.open(filepath) as hdul:
            table_hdu = None
            for hdu in hdul:
                if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                    table_hdu = hdu
                    break
            if table_hdu is None:
                raise ValueError(f"No table found in FITS file {filepath.name}")
            cols = table_hdu.data.columns.names
            if len(cols) <= max_col:
                raise ValueError(f"FITS table has {len(cols)} columns but column {max_col} requested.")
            t = table_hdu.data[cols[time_col]].astype(np.float64)
            x = table_hdu.data[cols[data_col]].astype(np.float64)
            
    else:
        data = np.loadtxt(filepath)
        if data.ndim == 1:
            raise ValueError(f"Expected at least 2 columns, got 1 in {filepath.name}")
        if data.shape[1] <= max_col:
            raise ValueError(f"File has {data.shape[1]} columns but column {max_col} requested.")
        t = data[:, time_col].astype(np.float64)
        x = data[:, data_col].astype(np.float64)

    # Sort by time (SigSpec does this too)
    order = np.argsort(t)
    return t[order], x[order]


# ---------------------------------------------------------------------------
# Writing helpers
# ---------------------------------------------------------------------------

def _header(label: str, cfg_summary: str = "") -> str:
    """Build a standard FACG header comment block."""
    lines = [
        f"# FACG — {label}",
        f"# {cfg_summary}" if cfg_summary else "",
    ]
    return "\n".join(ln for ln in lines if ln) + "\n"


def write_result(
    outdir: Path,
    results: list[dict],
    stem: str = "result",
) -> Path:
    """Write the main frequency result table.

    Columns: amp  freq  phase  rms  sig  csig
    """
    outpath = outdir / f"{stem}.dat"
    with open(outpath, "w") as f:
        f.write("# amp                 freq                  phase"
                "                 rms                   sig                   csig\n")
        for r in results:
            f.write(
                f"{r['amp']:20.12f} {r['freq']:20.12f} "
                f"{r['phase']:20.12f} {r['rms']:20.12f} "
                f"{r['sig']:20.12f} {r['csig']:20.12f}\n"
            )
    return outpath


def write_spectrum(
    outdir: Path,
    freqs: np.ndarray,
    sig_spectrum: np.ndarray,
    amp_spectrum: np.ndarray,
    phase_spectrum: np.ndarray,
    iteration: int,
    stem: str = "spectrum",
) -> Path:
    """Write the significance / amplitude spectrum at a given iteration.

    Columns: freq  sig  amplitude  phase
    """
    if iteration < 0:
        outpath = outdir / f"{stem}_final.dat"
    else:
        outpath = outdir / f"{stem}_{iteration:04d}.dat"
    data = np.column_stack([freqs, sig_spectrum, amp_spectrum, phase_spectrum])
    np.savetxt(outpath, data, fmt="%24.16e",
               header="freq                    sig                     "
                      "amplitude               phase")
    return outpath


def write_residuals(
    outdir: Path,
    t: np.ndarray,
    residuals: np.ndarray,
    iteration: int,
    stem: str = "residuals",
) -> Path:
    """Write a residual time-series."""
    if iteration < 0:
        outpath = outdir / f"{stem}_final.dat"
    else:
        outpath = outdir / f"{stem}_{iteration:04d}.dat"
    data = np.column_stack([t, residuals])
    np.savetxt(outpath, data, fmt="%24.16e",
               header="time                    residual")
    return outpath


def write_phase_diagram(
    outdir: Path,
    phase: np.ndarray,
    data: np.ndarray,
    freq: float,
    iteration: int,
) -> Path:
    """Write a folded phase diagram for a single frequency."""
    outpath = outdir / f"phase_{iteration:04d}_f{freq:.6f}.dat"
    out = np.column_stack([phase, data])
    np.savetxt(outpath, out, fmt="%24.16e",
               header=f"phase                   data   (fold freq={freq:.12f})")
    return outpath
