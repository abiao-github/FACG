"""
FACG 频谱分析核心算法，支持可选 GPU 加速。

本模块批量计算给定频率网格上的 SigSpec 显著性、振幅和相位，并提供单频率
显著性评估、峰值频率细化以及标准频率网格构造功能。显著性遵循 Reegen
(2007, A&A 467, 1353) 的解析虚警概率公式。公开计算函数接收 NumPy 数组，
设备数据传输由 :mod:`facg.backend` 负责。
"""

from __future__ import annotations

import numpy as np
from facg.backend import xp, to_device, to_host, use_gpu

# log10(e) – used in the SigSpec significance formula
_LG_E = np.log10(np.e)


# ===================================================================
# 1. Batch significance spectrum  (GPU-accelerated)
# ===================================================================

def compute_significance_spectrum(
    t: np.ndarray,
    x: np.ndarray,
    freq_grid: np.ndarray,
    rms: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算频率网格上每个频点的 SigSpec 显著性、振幅和相位。

    显著性根据 Reegen (2007) 的采样剖面解析公式计算。频率维度会分块处理，
    以限制显存占用；有 GPU 时矩阵运算在设备执行，输出统一转换回 NumPy。

    Parameters
    ----------
    t, x:
        主机内存中的时间戳和观测值数组。
    freq_grid:
        要评估的频率网格。
    rms:
        数据标准差；若未提供则由 ``x`` 计算。零值会替换为极小正数以避免除零。

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        三个形状均为 ``(M,)`` 的数组，依次为显著性、振幅和相位（弧度）。
    """
    N = len(t)
    if rms is None:
        rms = np.std(x, ddof=0)
    if rms == 0:
        rms = 1e-30

    t_d = to_device(t)
    x_d = to_device(x)
    f_d = to_device(freq_grid)
    M = len(freq_grid)

    sig_out = xp.zeros(M, dtype=xp.float64)
    amp_out = xp.zeros(M, dtype=xp.float64)
    phase_out = xp.zeros(M, dtype=xp.float64)

    # Process in chunks to limit GPU memory usage
    CHUNK = 20_000
    for start in range(0, M, CHUNK):
        end = min(start + CHUNK, M)
        f_chunk = f_d[start:end]
        phi = 2.0 * xp.pi * f_chunk[:, None] * t_d[None, :]  # (chunk, N)

        cos_phi = xp.cos(phi)
        sin_phi = xp.sin(phi)

        # Fourier coefficients a, b
        a = xp.sum(x_d[None, :] * cos_phi, axis=1) / N
        b = xp.sum(x_d[None, :] * sin_phi, axis=1) / N

        # Sampling-profile tilt angle θ₀
        # Use double-angle: cos(2φ) = 2cos²φ - 1, sin(2φ) = 2sinφcosφ
        cos2_phi = 2.0 * cos_phi * cos_phi - 1.0
        sin2_phi = 2.0 * sin_phi * cos_phi
        S2 = xp.sum(sin2_phi, axis=1)
        C2 = xp.sum(cos2_phi, axis=1)
        theta0 = 0.5 * xp.arctan2(S2, C2)
        theta0 = xp.where(theta0 < 0, theta0 + xp.pi, theta0)

        # Sampling-profile axes α₀, β₀
        cos_shifted = xp.cos(phi - theta0[:, None])
        cos2_sum = xp.sum(cos_shifted ** 2, axis=1)
        sin2_sum = N - cos2_sum
        axis1 = 2.0 * xp.abs(cos2_sum) / N
        axis2 = 2.0 * xp.abs(sin2_sum) / N

        # Assign a0 (minor) and b0 (major); swap + shift θ₀ if needed
        swap = axis1 > axis2
        a0 = xp.where(swap, axis2, axis1)
        b0 = xp.where(swap, axis1, axis2)
        theta0 = xp.where(
            swap,
            xp.where(theta0 < 0.5 * xp.pi,
                      theta0 + 0.5 * xp.pi,
                      theta0 - 0.5 * xp.pi),
            theta0,
        )

        a0 = xp.maximum(a0, 1e-30)
        b0 = xp.maximum(b0, 1e-30)

        # Significance
        proj_a = a * xp.cos(theta0) + b * xp.sin(theta0)
        proj_b = a * xp.sin(theta0) - b * xp.cos(theta0)
        sig_chunk = (N * _LG_E / rms ** 2 *
                     (proj_a ** 2 / a0 + proj_b ** 2 / b0))

        sig_out[start:end] = sig_chunk
        amp_out[start:end] = 2.0 * xp.sqrt(a ** 2 + b ** 2)
        phase_out[start:end] = xp.arctan2(b, a)

    return to_host(sig_out), to_host(amp_out), to_host(phase_out)


# ===================================================================
# 2. Fast single-frequency sig (optimised for refinement loop)
# ===================================================================

def _sig_single(
    t: np.ndarray,
    x: np.ndarray,
    freq: float,
    rms: float,
    N: int,
) -> float:
    """用纯 NumPy 计算单个频率的 SigSpec 显著性。

    此轻量实现供频率细化循环反复调用，避免每次评估都产生 GPU 数据传输或
    分块调度开销。``N`` 为采样点数，``rms`` 为当前数据的标准差。
    """
    phi = 2.0 * np.pi * freq * t
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    a = np.dot(x, cos_phi) / N
    b = np.dot(x, sin_phi) / N

    S2 = 2.0 * np.dot(sin_phi, cos_phi)
    C2 = np.sum(2.0 * cos_phi * cos_phi - 1.0)
    theta0 = 0.5 * np.arctan2(S2, C2)
    if theta0 < 0:
        theta0 += np.pi

    cos_shifted = np.cos(phi - theta0)
    cos2_sum = np.dot(cos_shifted, cos_shifted)
    sin2_sum = N - cos2_sum
    axis1 = 2.0 * abs(cos2_sum) / N
    axis2 = 2.0 * abs(sin2_sum) / N

    if axis1 > axis2:
        a0, b0 = axis2, axis1
        theta0 = theta0 + 0.5 * np.pi if theta0 < 0.5 * np.pi else theta0 - 0.5 * np.pi
    else:
        a0, b0 = axis1, axis2

    a0 = max(a0, 1e-30)
    b0 = max(b0, 1e-30)

    proj_a = a * np.cos(theta0) + b * np.sin(theta0)
    proj_b = a * np.sin(theta0) - b * np.cos(theta0)
    return N * _LG_E / rms ** 2 * (proj_a ** 2 / a0 + proj_b ** 2 / b0)


def sig_at_freq(
    t: np.ndarray,
    x: np.ndarray,
    freq: float,
    rms: float,
) -> tuple[float, float, float]:
    """计算指定频率处的显著性、振幅和相位。

    根据离散傅里叶系数求得振幅和相位，并通过单频率快速实现计算显著性。
    返回值依次为显著性、振幅和相位（弧度）。
    """
    N = len(t)
    phi = 2.0 * np.pi * freq * t
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    a = np.dot(x, cos_phi) / N
    b = np.dot(x, sin_phi) / N
    amp = 2.0 * np.sqrt(a ** 2 + b ** 2)
    phase = np.arctan2(b, a)
    sig = _sig_single(t, x, freq, rms, N)
    return sig, amp, phase


# ===================================================================
# 3. Frequency refinement — bisection search  (SigSpec_MaxSig)
# ===================================================================

def refine_frequency(
    t: np.ndarray,
    x: np.ndarray,
    f0: float,
    rms: float,
    search_range: float | None = None,
    tol: float = 1e-12,
    max_refine_iter: int = 80,
) -> tuple[float, float, float, float]:
    """在初始频率附近搜索显著性峰值并细化频率估计。

    搜索区间按二分方式逐步缩小，行为对应 SigSpec 的 ``SigSpec_MaxSig``。
    达到容差或最大迭代次数后，使用最终频率重新计算完整峰值属性。

    Parameters
    ----------
    t, x:
        当前时间戳和数据（通常为残差）。
    f0:
        频谱扫描得到的初始峰值频率。
    rms:
        当前数据的标准差。
    search_range:
        搜索初始半宽；未指定时取时间跨度倒数。
    tol:
        搜索半宽的收敛容差。
    max_refine_iter:
        细化循环的最大次数，用作终止保护。

    Returns
    -------
    tuple[float, float, float, float]
        细化后的频率、显著性、振幅和相位（弧度）。
    """
    if search_range is None:
        T = t[-1] - t[0]
        search_range = 1.0 / T

    N = len(t)
    r = search_range
    if f0 <= r:
        r = f0 * 0.5 if f0 > 0 else search_range

    f_prev = max(f0 - r, 1e-15)
    f_curr = f0
    f_next = f0 + r

    s_prev = _sig_single(t, x, f_prev, rms, N)
    s_curr = _sig_single(t, x, f_curr, rms, N)
    s_next = _sig_single(t, x, f_next, rms, N)

    for _ in range(max_refine_iter):
        r *= 0.5
        if r < tol:
            break

        if s_curr >= s_prev and s_curr >= s_next:
            f_prev = max(f_curr - r, 1e-15)
            s_prev = _sig_single(t, x, f_prev, rms, N)
            f_next = f_curr + r
            s_next = _sig_single(t, x, f_next, rms, N)
        elif s_prev > s_next:
            f_next = f_curr
            s_next = s_curr
            f_curr = max(f_curr - r, 1e-15)
            s_curr = _sig_single(t, x, f_curr, rms, N)
        else:
            f_prev = f_curr
            s_prev = s_curr
            f_curr = f_curr + r
            s_curr = _sig_single(t, x, f_curr, rms, N)

    # Final full evaluation
    sig_final, amp_final, phase_final = sig_at_freq(t, x, f_curr, rms)
    return f_curr, sig_final, amp_final, phase_final


# ===================================================================
# 4. Build a standard frequency grid  (mirrors IniFile_Calculate)
# ===================================================================

def make_freq_grid(
    t: np.ndarray,
    freq_low: float | None = None,
    freq_high: float | None = None,
    freq_step: float | None = None,
    nyquist_coeff: float = 0.5,
    oversampling: float = 20.0,
) -> tuple[np.ndarray, dict]:
    """根据采样时间和用户设置构造待评估的频率网格。

    默认步长为 Rayleigh 分辨率除以过采样倍数，下限默认取该步长，上限默认
    取 ``nyquist_coeff / median(|Δt|)``。函数不修改输入时间数组。

    参数
    ----
    t:
        按时间排序的采样时间数组。
    freq_low, freq_high, freq_step:
        可选的频率下限、上限和网格间距；未提供时使用上述默认值。
    nyquist_coeff:
        Nyquist 上限系数。
    oversampling:
        Rayleigh 分辨率的过采样倍数。

    Returns
    -------
    tuple[np.ndarray, dict]
        频率网格以及包含 Rayleigh 分辨率、Nyquist 频率、步长、上下限、
        点数、时间跨度和中位采样间隔等信息的字典。
    """
    T = t[-1] - t[0]
    dt_med = np.median(np.abs(np.diff(t)))

    rayleigh = 1.0 / T
    nyquist = nyquist_coeff / dt_med
    fs = freq_step if freq_step is not None else rayleigh / oversampling

    fl = freq_low if freq_low is not None else fs
    fh = freq_high if freq_high is not None else nyquist

    # Enforce strictly positive start, but respect explicit freq_low
    if freq_low is None and fl < fs:
        fl = fs
    elif fl <= 0:
        fl = 1e-15

    n_freq = int(np.ceil((fh - fl) / fs)) + 1
    grid = fl + np.arange(n_freq) * fs

    info = dict(
        rayleigh=rayleigh,
        nyquist=nyquist,
        freq_step=fs,
        freq_low=fl,
        freq_high=fh,
        n_freq=n_freq,
        oversampling=oversampling,
        time_base=T,
        median_dt=dt_med,
    )
    return grid, info
