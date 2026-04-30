"""
Iterative prewhitening cascade — the main FACG analysis loop.

This mirrors the ``SigSpec_Cascade`` function in *SigSpec.c*:

1. Compute the significance spectrum on the current residual.
2. Find the frequency with the highest significance.
3. Refine the frequency via bisection search.
4. Globally re-optimise *all* detected frequencies/amplitudes/phases.
5. Subtract the optimised model → new residual.
6. Repeat until ``sig < sig_limit`` or ``max_iter`` reached.

Throughout the loop, intermediate results / spectra / residuals are
written to disk according to the configuration.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from facg.backend import backend_name
from facg.config import FACGConfig
from facg.io import (
    read_timeseries,
    write_result,
    write_spectrum,
    write_residuals,
    write_phase_diagram,
)
from facg.spectrum import (
    compute_significance_spectrum,
    make_freq_grid,
    refine_frequency,
)
from facg.optimizer import global_optimize


# ------------------------------------------------------------------
# Cumulative significance  (SigSpec_CSig)
# ------------------------------------------------------------------

def _cumulative_sig(csig_prev: float, sig_new: float) -> float:
    """Combine cumulative significance with a new detection.

    Implements the same formula as ``SigSpec_CSig`` in *SigSpec.c*:

        csig = -log10(1 - (1 - 10^{-csig_prev}) * (1 - 10^{-sig_new}))

    For very large significances the formula simplifies to:
        csig ≈ csig_prev + sig_new  (independent detections add in log-space).
    """
    if csig_prev <= 0:
        return sig_new
    # For large values, 10^{-sig} underflows → use the additive
    # approximation that is exact in the limit of independent events
    if csig_prev > 300 or sig_new > 300:
        return csig_prev + sig_new
    try:
        p_prev = 1.0 - 10.0 ** (-csig_prev)  # P(prev detections real)
        p_new = 1.0 - 10.0 ** (-sig_new)      # P(new detection real)
        p_comb = p_prev * p_new
        if p_comb >= 1.0:
            return csig_prev + sig_new
        return -np.log10(1.0 - p_comb)
    except Exception:
        return csig_prev + sig_new


# ------------------------------------------------------------------
# Main analysis driver
# ------------------------------------------------------------------

def run_analysis(cfg: FACGConfig) -> list[dict]:
    """Execute a full FACG frequency analysis.

    Parameters
    ----------
    cfg : FACGConfig
        Fully populated configuration object.

    Returns
    -------
    results : list of dict
        One entry per detected frequency, each containing keys:
        ``freq``, ``sig``, ``amp``, ``phase``, ``rms``, ``csig``.
    """
    total_t0 = time.time()

    # --- Preamble ---------------------------------------------------
    outdir = cfg.resolve_output_dir()
    if not cfg.quiet:
        print("=" * 65)
        print("  FACG — Frequency Analysis of CPU and GPU mixed computing")
        print(f"  Backend : {backend_name()}")
        print(f"  Input   : {cfg.input_file}")
        print(f"  Output  : {outdir}")
        print("=" * 65)

    # --- Load data --------------------------------------------------
    t, x_orig = read_timeseries(
        cfg.input_file, cfg.time_col, cfg.data_col
    )
    N = len(t)
    if not cfg.quiet:
        print(f"  Data points   : {N}")

    # --- Zero-mean --------------------------------------------------
    mean_x = np.mean(x_orig)
    x_orig = x_orig - mean_x

    # --- Frequency grid ---------------------------------------------
    freq_grid, info = make_freq_grid(
        t,
        freq_low=cfg.freq_low,
        freq_high=cfg.freq_high,
        nyquist_coeff=cfg.nyquist_coeff,
        oversampling=cfg.oversampling,
    )
    if not cfg.quiet:
        print(f"  Time base     : {info['time_base']:.6f}")
        print(f"  Rayleigh res  : {info['rayleigh']:.9f}")
        print(f"  Freq step     : {info['freq_step']:.9f}")
        print(f"  Freq range    : [{info['freq_low']:.6f}, {info['freq_high']:.6f}]")
        print(f"  Nyquist coeff : {info['nyquist']:.6f}")
        print(f"  Oversampling  : {info['oversampling']:.1f}")
        print(f"  # frequencies : {info['n_freq']}")
        print(f"  sig threshold : {cfg.sig_limit}")
        print("-" * 65)
        sys.stdout.flush()

    # --- Iterative prewhitening cascade -----------------------------
    residual = x_orig.copy()
    all_params = np.array([], dtype=np.float64)  # [f1,A1,φ1, f2,A2,φ2, ...]
    results: list[dict] = []
    csig = 0.0

    for it in range(1, cfg.max_iter + 1):
        iter_t0 = time.time()
        rms = np.std(residual, ddof=0)
        if rms == 0:
            if not cfg.quiet:
                print("  *** residual rms = 0 – stopping.")
            break

        # 1) Significance spectrum on residual
        sig_spec, amp_spec, ph_spec = compute_significance_spectrum(
            t, residual, freq_grid, rms=rms,
        )

        # 2) Find highest-significance frequency
        idx_max = int(np.argmax(sig_spec))
        f0 = freq_grid[idx_max]
        sig0 = sig_spec[idx_max]

        if sig0 < cfg.sig_limit:
            if not cfg.quiet:
                print(
                    f"  iter {it:4d}: max sig = {sig0:.4f} < {cfg.sig_limit} "
                    f"– stopping."
                )
            # Write final spectrum
            if cfg.write_spectrum:
                write_spectrum(outdir, freq_grid, sig_spec, amp_spec, ph_spec,
                               iteration=-1, stem=Path(cfg.input_file).stem)
            break

        # 3) Refine frequency
        f_ref, sig_ref, amp_ref, ph_ref = refine_frequency(
            t, residual, f0, rms,
            search_range=info["freq_step"],
        )

        # 4) Append to parameter vector & global optimise
        new_params = np.append(all_params, [f_ref, amp_ref / 2.0, ph_ref])
        try:
            opt_params, opt_residual = global_optimize(
                t, x_orig, new_params,
            )
        except Exception as exc:
            if not cfg.quiet:
                print(f"  iter {it:4d}: optimiser failed ({exc}) – "
                      f"using unoptimised params.")
            opt_params = new_params
            n_sig = len(opt_params) // 3
            model = np.zeros_like(t)
            for k in range(n_sig):
                fk = opt_params[3 * k]
                Ak = opt_params[3 * k + 1]
                pk = opt_params[3 * k + 2]
                model += Ak * np.sin(2.0 * np.pi * fk * t + pk)
            opt_residual = x_orig - model

        all_params = opt_params
        residual = opt_residual
        rms_new = np.std(residual, ddof=0)

        # Re-evaluate significance at the refined frequency on the new residual
        # For the result table we use the significance from the DFT scan
        csig = _cumulative_sig(csig, sig_ref)

        # Extract optimised parameters for the latest signal
        n_sig = len(all_params) // 3
        f_opt = all_params[3 * (n_sig - 1)]
        A_opt = all_params[3 * (n_sig - 1) + 1]
        p_opt = all_params[3 * (n_sig - 1) + 2]

        result_entry = dict(
            freq=float(f_opt),
            sig=float(sig_ref),
            amp=float(abs(A_opt)),
            phase=float(p_opt),
            rms=float(rms_new),
            csig=float(csig),
        )
        results.append(result_entry)

        iter_dt = time.time() - iter_t0
        if not cfg.quiet:
            print(
<<<<<<< HEAD
                f"  iter {it:4d}: amp {abs(A_opt):12.9f}  "
                f"freq {f_opt:14.9f}  "
                f"phase {p_opt:7.3f}  "
                f"rms {rms_new:12.9f}  "
                f"sig {sig_ref:10.4f}  "
=======
                f"  iter {it:4d}: freq {f_opt:14.9f}  "
                f"sig {sig_ref:10.4f}  "
                f"amp {abs(A_opt):12.9f}  "
                f"phase {p_opt:7.3f}  "
                f"rms {rms_new:12.9f}  "
>>>>>>> 9960956 (Initial commit for FACG package)
                f"csig {csig:10.4f}  "
                f"[{iter_dt:.2f}s]"
            )
            sys.stdout.flush()

        # 5) Intermediate outputs
        if cfg.write_spectrum:
            write_spectrum(outdir, freq_grid, sig_spec, amp_spec, ph_spec,
                           iteration=it,
                           stem=Path(cfg.input_file).stem)
        if cfg.write_residuals:
            write_residuals(outdir, t, residual, iteration=it,
                            stem=Path(cfg.input_file).stem)
        if cfg.write_phase_diagram:
            folded_phase = np.mod(t * f_opt, 1.0)
            write_phase_diagram(outdir, folded_phase, residual, f_opt, it)

        # Check cumulative sig limit
        if cfg.csig_limit > 0 and csig < cfg.csig_limit:
            if not cfg.quiet:
                print(f"  csig {csig:.4f} < {cfg.csig_limit} – stopping.")
            break
    else:
        # max_iter reached
        if not cfg.quiet:
            print(f"  max iterations ({cfg.max_iter}) reached – stopping.")

    # --- Final outputs ---------------------------------------------------
    stem = Path(cfg.input_file).stem
    write_result(outdir, results, stem=stem)
    write_residuals(outdir, t, residual, iteration=-1, stem=stem)

    # Write final spectrum
    rms_final = np.std(residual, ddof=0)
    sig_spec_f = amp_spec_f = ph_spec_f = None
    if rms_final > 0:
        sig_spec_f, amp_spec_f, ph_spec_f = compute_significance_spectrum(
            t, residual, freq_grid, rms=rms_final,
        )
        if cfg.write_spectrum:
            write_spectrum(outdir, freq_grid, sig_spec_f, amp_spec_f,
                           ph_spec_f, iteration=-1, stem=stem)

    # --- Summary plots (optional) ----------------------------------------
    if cfg.plot and len(results) > 0:
        if not cfg.quiet:
            print("  Generating summary plot ...")
        _generate_summary_plot(
            outdir, stem, t, x_orig, mean_x, all_params,
            residual, freq_grid, amp_spec_f, sig_spec_f, results,
        )
        if not cfg.quiet:
            print(f"  Plot saved: {outdir / f'{stem}_summary.png'}")

    total_dt = time.time() - total_t0
    if not cfg.quiet:
        print("-" * 65)
        print(f"  Detected frequencies : {len(results)}")
        print(f"  Total elapsed time   : {total_dt:.3f} s")
        print("=" * 65)

    return results


# ------------------------------------------------------------------
# Summary plot
# ------------------------------------------------------------------

def _generate_summary_plot(
    outdir: Path,
    stem: str,
    t: np.ndarray,
    x_orig: np.ndarray,
    mean_x: float,
    all_params: np.ndarray,
    residual: np.ndarray,
    freq_grid: np.ndarray,
    amp_spec: np.ndarray | None,
    sig_spec: np.ndarray | None,
    results: list[dict],
) -> None:
    """Generate a 4-panel summary plot and save as PNG."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(12, 14))
    fig.suptitle(f"FACG Summary — {stem}", fontsize=14, fontweight="bold")

    # Reconstruct fitted signal
    n_sig = len(all_params) // 3
    fitted = np.zeros_like(t)
    for k in range(n_sig):
        fk = all_params[3 * k]
        Ak = all_params[3 * k + 1]
        pk = all_params[3 * k + 2]
        fitted += Ak * np.sin(2.0 * np.pi * fk * t + pk)

    # Panel 1: Original data + fitted model
    ax = axes[0]
    ax.plot(t, x_orig + mean_x, "k.", markersize=1, alpha=0.4, label="Data")
    ax.plot(t, fitted + mean_x, "r-", linewidth=0.8, alpha=0.8,
            label=f"Model ({n_sig} freq)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Flux / Magnitude")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Time Series + Multi-Sine Fit")
    ax.grid(True, alpha=0.3)

    # Panel 2: Residual
    ax = axes[1]
    ax.plot(t, residual, "k.", markersize=1, alpha=0.4)
    rms_val = np.std(residual, ddof=0)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Residual")
    ax.set_title(f"Residual (RMS = {rms_val:.2e})")
    ax.grid(True, alpha=0.3)

    # Panel 3: Amplitude spectrum of residual
    ax = axes[2]
    if amp_spec is not None:
        ax.plot(freq_grid, amp_spec, "k-", linewidth=0.5)
    ax.set_xlabel("Frequency (cycles / time unit)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Residual Amplitude Spectrum")
    ax.grid(True, alpha=0.3)
    # Mark detected frequencies
    for r in results:
        ax.axvline(r["freq"], color="red", linewidth=0.6, alpha=0.5)

    # Panel 4: Significance spectrum of residual
    ax = axes[3]
    if sig_spec is not None:
        ax.plot(freq_grid, sig_spec, "k-", linewidth=0.5)
    ax.axhline(5.0, color="red", linewidth=0.8, linestyle="--",
               label="sig = 5")
    ax.set_xlabel("Frequency (cycles / time unit)")
    ax.set_ylabel("Significance (SigSpec)")
    ax.set_title("Residual Significance Spectrum")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / f"{stem}_summary.png", dpi=150)
    plt.close(fig)
