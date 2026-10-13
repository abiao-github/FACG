"""
迭代预白化流程：FACG 分析的主控模块。

本模块按以下步骤重复提取显著频率，与 SigSpec 的 ``SigSpec_Cascade`` 流程
相对应：

1. 计算当前残差的显著性频谱并定位最高峰；
2. 细化峰值频率，联合优化全部已检测信号的频率、振幅和相位；
3. 根据拟合模型更新残差并记录本轮结果；
4. 达到显著性阈值、累积显著性阈值或最大迭代次数时停止。

迭代期间的频谱、残差、相位折叠数据和绘图均由配置项控制。
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
    """将本轮显著性合并到已有的累积显著性中。

    计算方式与 SigSpec 的 ``SigSpec_CSig`` 一致。通过按较小显著性因子化虚警
    概率，避免直接计算极小概率时的浮点下溢。

    参数为之前的累积显著性和新检测的显著性；若没有正的历史累积值，则直接
    返回新显著性。
    """
    if csig_prev <= 0:
        return sig_new

    m = min(csig_prev, sig_new)
    M = max(csig_prev, sig_new)
    try:
        # 1.0 + 10^{-(M-m)} - 10^{-M}
        term = 1.0 + 10.0 ** -(M - m) - 10.0 ** -M
        return m - np.log10(term)
    except Exception:
        return m


# ------------------------------------------------------------------
# Main analysis driver
# ------------------------------------------------------------------

def run_analysis(cfg: FACGConfig) -> list[dict]:
    """根据配置执行一次完整的 FACG 频率分析。

    Parameters
    ----------
    cfg : FACGConfig
        输入输出、频率网格、停止条件和可选输出设置。

    Returns
    -------
    results : list of dict
        每项记录一次检测结果，包含 ``freq``（频率）、``sig``（显著性）、
        ``amp``（振幅）、``phase``（相位）、``rms``（残差标准差）和
        ``csig``（累积显著性）。分析过程中按配置写出结果文件，输入或计算错误
        会由相关读取/计算函数抛出。
    """
    total_t0 = time.time()

    # --- Preamble ---------------------------------------------------
    outdir = cfg.resolve_output_dir()

    # --- Load data --------------------------------------------------
    t, x_orig = read_timeseries(
        cfg.input_file, cfg.time_col, cfg.data_col
    )
    N = len(t)

    # Calculate orig rms before zero-mean for output
    rms_orig = np.std(x_orig, ddof=0)

    # --- Zero-mean --------------------------------------------------
    mean_x = np.mean(x_orig)
    x_orig = x_orig - mean_x

    # --- Frequency grid ---------------------------------------------
    freq_grid, info = make_freq_grid(
        t,
        freq_low=cfg.freq_low,
        freq_high=cfg.freq_high,
        freq_step=cfg.freq_step,
        nyquist_coeff=cfg.nyquist_coeff,
        oversampling=cfg.oversampling,
    )
    if not cfg.quiet:
        stem = Path(cfg.input_file).stem
        print("\n*** time series properties *********************************\n")
        print(f"{stem} points {N}, time base {info['time_base']:g}, rms dev {rms_orig:g}\n")
        print("*** preparing to run FACG **********************************\n")
        print(f"{'Rayleigh frequency resolution':<42}{info['rayleigh']:.16f}")
        print(f"{'oversampling ratio':<42}{info['oversampling']:.16f}")
        print(f"{'frequency spacing':<42}{info['freq_step']:.16f}")
        print(f"{'lower frequency limit':<42}{info['freq_low']:.16f}")
        print(f"{'upper frequency limit':<42}{info['freq_high']:.16f}")
        print(f"{'Nyquist coefficient':<42}{cfg.nyquist_coeff:.16f}")
        print(f"{'number of frequencies':<38}{info['n_freq']}\n")
        print("*** running FACG *******************************************\n")
        print(f"{stem}                                ")
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

        # 3) Refine frequency to get true peak properties
        f_ref, sig_ref, amp_ref, ph_ref = refine_frequency(
            t, residual, f0, rms,
            search_range=info["freq_step"],
        )

        if sig_ref < cfg.sig_limit:
            csig = _cumulative_sig(csig, sig_ref)
            
            # Record the sub-threshold peak to match SigSpec output behavior
            result_entry = dict(
                freq=float(f_ref),
                sig=float(sig_ref),
                amp=float(abs(amp_ref)),
                phase=float(ph_ref),
                rms=float(rms),
                csig=float(csig),
            )
            results.append(result_entry)

            if not cfg.quiet:
                print(f"{it:4d} freq {f_ref:g}  sig {sig_ref:g}  rms {rms:g}  csig {csig:g}                ")

            # Write final spectrum
            if cfg.write_spectrum:
                write_spectrum(outdir, freq_grid, sig_spec, amp_spec, ph_spec,
                               iteration=-1, stem=Path(cfg.input_file).stem)
            break

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
            rms=float(rms),
            csig=float(csig),
        )
        results.append(result_entry)

        if not cfg.quiet:
            print(f"{it:4d} freq {f_opt:g}  sig {sig_ref:g}  rms {rms:g}  csig {csig:g}                ")
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
        if cfg.csig_limit > 0 and csig >= cfg.csig_limit:
            if not cfg.quiet:
                print(f"  csig {csig:.4f} >= {cfg.csig_limit} "
                      f"(limit reached) – stopping.")
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
        print(f"  Total elapsed time   : {total_dt:.3f} s")

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
    """生成并保存四联汇总图，展示拟合效果和最终残差频谱。

    四个面板分别绘制原始数据及多正弦模型、残差、残差振幅频谱和残差显著性
    频谱；图像保存为输出目录下的 ``<stem>_summary.png``。使用 Matplotlib 的
    非交互式 Agg 后端，适合命令行批处理。
    """
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
