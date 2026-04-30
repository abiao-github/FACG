"""
Configuration dataclass for FACG analysis.

All user-tuneable parameters live here, with sensible defaults that match
the SigSpec conventions.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional


@dataclasses.dataclass
class FACGConfig:
    """Complete set of parameters controlling a FACG frequency analysis run.

    Parameters
    ----------
    input_file : str or Path
        Path to the input time-series file (whitespace-delimited, no header,
        col-0 = time, col-1 = observable).
    output_dir : str or Path or None
        Directory to write output files.  When *None* a sub-directory
<<<<<<< HEAD
        ``<stem>_facg/`` next to the input file is created automatically.
=======
        ``<stem>/`` next to the input file is created automatically.
>>>>>>> 9960956 (Initial commit for FACG package)
    time_col : int
        0-based column index for the time axis (default 0).
    data_col : int
        0-based column index for the data axis (default 1).

    Frequency grid
    ~~~~~~~~~~~~~~
    freq_low : float or None
        Lower frequency limit (cycles/time-unit).
        Default *None* → Rayleigh resolution ``1 / T``.
    freq_high : float or None
        Upper frequency limit.
        Default *None* → Nyquist frequency ``0.5 / median(Δt)``.
    nyquist_coeff : float
        Nyquist coefficient (default 0.5).
    oversampling : float
        Oversampling ratio applied to Rayleigh resolution (default 20.0,
        same as SigSpec).

    Stopping criteria
    ~~~~~~~~~~~~~~~~~
    sig_limit : float
        Significance threshold – iteration stops when the highest
        significance drops below this value (default 5.0).
    csig_limit : float
        Cumulative significance limit (default 0 = disabled).
    max_iter : int
        Maximum number of prewhitening iterations (default 999).

    Output control
    ~~~~~~~~~~~~~~
    write_spectrum : bool
        Write residual amplitude spectrum after each iteration (default True).
    write_residuals : bool
        Write residual time-series after each iteration (default True).
    write_phase_diagram : bool
        Write folded phase diagrams (default False).
    plot : bool
        Generate summary plots (time series, fitted signal, amplitude
        spectrum, residual) and save as PNG.  Default False — plotting
        is skipped to save time; enable with ``--plot`` on the CLI.
    quiet : bool
        Suppress progress output (default False).
    """

    # --- I/O ---------------------------------------------------------------
    input_file: str = ""
    output_dir: Optional[str] = None
    time_col: int = 0
    data_col: int = 1

    # --- Frequency grid ----------------------------------------------------
    freq_low: Optional[float] = None
    freq_high: Optional[float] = None
    nyquist_coeff: float = 0.5
    oversampling: float = 20.0

    # --- Stopping criteria -------------------------------------------------
    sig_limit: float = 5.0
    csig_limit: float = 0.0
    max_iter: int = 999

    # --- Output control ----------------------------------------------------
    write_spectrum: bool = True
    write_residuals: bool = True
    write_phase_diagram: bool = False
    plot: bool = False
    quiet: bool = False

    # -----------------------------------------------------------------------
    # Derived helpers (not set by user)
    # -----------------------------------------------------------------------
    def resolve_output_dir(self) -> Path:
        """Return the concrete output directory, creating it if needed."""
        if self.output_dir is not None:
            p = Path(self.output_dir)
        else:
            inp = Path(self.input_file)
<<<<<<< HEAD
            p = inp.parent / f"{inp.stem}_facg"
=======
            p = inp.parent / inp.stem
>>>>>>> 9960956 (Initial commit for FACG package)
        p.mkdir(parents=True, exist_ok=True)
        return p
