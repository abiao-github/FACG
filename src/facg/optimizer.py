"""
FACG 多正弦模型的全局参数优化器。

每轮预白化后，对原始（已去均值）数据中的所有已检测频率、振幅和相位执行
联合最小二乘拟合，以避免逐个拟合造成的参数偏差。

多正弦模型及其解析 Jacobian 可向量化计算。可用 GPU 时由设备计算模型和
Jacobian，再交由 SciPy 的 Levenberg–Marquardt 求解器更新参数；否则完全在
CPU 上计算。

模型形式为 ``y(t) = Σ A_k sin(2π f_k t + φ_k)``；频率、振幅、相位的
偏导数构成可直接计算的解析 Jacobian。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from facg.backend import xp, to_device, to_host, use_gpu


# ------------------------------------------------------------------
# Multi-sine model
# ------------------------------------------------------------------

def _model_cpu(t: np.ndarray, params: np.ndarray) -> np.ndarray:
    """在 CPU 上根据参数向量计算多正弦模型值。

    参数向量按 ``[f1, A1, φ1, f2, A2, φ2, ...]`` 排列，输出数组与时间数组
    ``t`` 等长。
    """
    n_sig = len(params) // 3
    y = np.zeros_like(t)
    for k in range(n_sig):
        f, A, phi = params[3 * k], params[3 * k + 1], params[3 * k + 2]
        y += A * np.sin(2.0 * np.pi * f * t + phi)
    return y


def _model_gpu(t_d, params: np.ndarray):
    """在当前设备上计算多正弦模型，并返回设备数组。

    ``t_d`` 应已位于目标设备；参数向量在函数内传输一次。参数排列与
    :func:`_model_cpu` 相同。
    """
    n_sig = len(params) // 3
    p_d = to_device(params)
    N = len(t_d)
    y = xp.zeros(N, dtype=xp.float64)
    for k in range(n_sig):
        f = p_d[3 * k]
        A = p_d[3 * k + 1]
        phi = p_d[3 * k + 2]
        y += A * xp.sin(2.0 * xp.pi * f * t_d + phi)
    return y


# ------------------------------------------------------------------
# Jacobian (GPU-accelerated)
# ------------------------------------------------------------------

def _jacobian_gpu(t_d, params: np.ndarray):
    """在设备上计算多正弦模型的解析 Jacobian，并将结果转回 NumPy。

    返回矩阵形状为 ``(N, 3K)``，每组三列依次对应一个信号的频率、振幅和
    相位偏导数，列顺序与参数向量一致。
    """
    n_sig = len(params) // 3
    p_d = to_device(params)
    N = len(t_d)
    J = xp.empty((N, 3 * n_sig), dtype=xp.float64)
    for k in range(n_sig):
        f = p_d[3 * k]
        A = p_d[3 * k + 1]
        phi = p_d[3 * k + 2]
        phase = 2.0 * xp.pi * f * t_d + phi
        sin_p = xp.sin(phase)
        cos_p = xp.cos(phase)
        J[:, 3 * k]     = 2.0 * xp.pi * t_d * A * cos_p   # ∂y/∂f
        J[:, 3 * k + 1] = sin_p                             # ∂y/∂A
        J[:, 3 * k + 2] = A * cos_p                         # ∂y/∂φ
    return to_host(J)


def _jacobian_cpu(t: np.ndarray, params: np.ndarray) -> np.ndarray:
    """使用 NumPy 计算解析 Jacobian，作为无 GPU 时的回退实现。

    返回 ``(N, 3K)`` 矩阵，每组三列对应一个信号参数的偏导数。
    """
    n_sig = len(params) // 3
    N = len(t)
    J = np.empty((N, 3 * n_sig), dtype=np.float64)
    for k in range(n_sig):
        f, A, phi = params[3 * k], params[3 * k + 1], params[3 * k + 2]
        phase = 2.0 * np.pi * f * t + phi
        sin_p = np.sin(phase)
        cos_p = np.cos(phase)
        J[:, 3 * k]     = 2.0 * np.pi * t * A * cos_p
        J[:, 3 * k + 1] = sin_p
        J[:, 3 * k + 2] = A * cos_p
    return J


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def global_optimize(
    t: np.ndarray,
    x_orig: np.ndarray,
    params0: np.ndarray,
    max_nfev: int = 5000,
    ftol: float = 1e-12,
    xtol: float = 1e-12,
    gtol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """通过 Levenberg–Marquardt 联合优化全部信号参数。

    Parameters
    ----------
    t:
        ``(N,)`` 时间戳数组。
    x_orig:
        ``(N,)`` 原始去均值数据，而非某轮预白化后的残差。
    params0:
        ``(3K,)`` 初始参数向量，顺序为 ``[f1, A1, φ1, ...]``。
    max_nfev:
        求解器允许的最大函数评估次数。
    ftol, xtol, gtol:
        分别控制目标函数、参数变化和梯度的收敛容差。

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        优化后的参数向量，以及 ``x_orig - model(t, params_opt)`` 残差。
        当没有待优化参数时，返回参数和数据的副本。
    """
    if len(params0) == 0:
        return params0.copy(), x_orig.copy()

    # Pre-transfer time array to device once
    if use_gpu():
        t_d = to_device(t)
    else:
        t_d = t

    def _residual_fun(p):
        """计算给定参数下的数据减去模型值，供最小二乘求解器使用。"""
        if use_gpu():
            model = to_host(_model_gpu(t_d, p))
        else:
            model = _model_cpu(t, p)
        return x_orig - model

    def _jac_fun(p):
        """返回残差对参数的 Jacobian；因残差为数据减模型，需对模型导数取负。"""
        if use_gpu():
            J = _jacobian_gpu(t_d, p)
        else:
            J = _jacobian_cpu(t, p)
        return -J  # because residual = data - model

    result = least_squares(
        _residual_fun,
        params0,
        jac=_jac_fun,
        method="lm",
        max_nfev=max_nfev,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
    )

    return result.x, result.fun
