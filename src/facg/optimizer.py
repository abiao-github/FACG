"""
GPU-accelerated multi-sine global optimiser.

After each prewhitening iteration SigSpec (and FACG) perform a
*simultaneous* least-squares fit of **all** detected frequencies,
amplitudes and phases to the *original* data.  This is the most
CPU-intensive part of the pipeline.

GPU acceleration strategy
~~~~~~~~~~~~~~~~~~~~~~~~~
The Jacobian matrix of the multi-sine model

    y(t) = Σ_k  A_k · sin(2π f_k t + φ_k)

has analytic partial derivatives that are trivially vectorisable:

    ∂y/∂f_k  = 2π t A_k cos(2π f_k t + φ_k)
    ∂y/∂A_k  = sin(2π f_k t + φ_k)
    ∂y/∂φ_k  = A_k cos(2π f_k t + φ_k)

We compute the full Jacobian on the GPU in one shot, then run
Levenberg–Marquardt on the CPU using ``scipy.optimize.least_squares``
with the GPU-computed Jacobian.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from facg.backend import xp, to_device, to_host, use_gpu


# ------------------------------------------------------------------
# Multi-sine model
# ------------------------------------------------------------------

def _model_cpu(t: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Evaluate  y = Σ A_k sin(2π f_k t + φ_k)  on the CPU."""
    n_sig = len(params) // 3
    y = np.zeros_like(t)
    for k in range(n_sig):
        f, A, phi = params[3 * k], params[3 * k + 1], params[3 * k + 2]
        y += A * np.sin(2.0 * np.pi * f * t + phi)
    return y


def _model_gpu(t_d, params: np.ndarray):
    """Evaluate the multi-sine model on the GPU.

    Returns a *device* array.
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
    """Compute the (N, 3K) Jacobian on the GPU; return as NumPy."""
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
    """CPU fallback for the Jacobian."""
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
    """Perform a global Levenberg–Marquardt optimisation of all
    frequency / amplitude / phase parameters simultaneously.

    Parameters
    ----------
    t : (N,) time stamps.
    x_orig : (N,) *original* (un-prewhitened) data.
    params0 : (3K,) initial guess [f1, A1, φ1, f2, A2, φ2, ...].
    max_nfev : maximum function evaluations.
    ftol, xtol, gtol : convergence tolerances.

    Returns
    -------
    params_opt : (3K,) optimised parameters.
    residuals : (N,) final residuals  x_orig − model(t, params_opt).
    """
    if len(params0) == 0:
        return params0.copy(), x_orig.copy()

    # Pre-transfer time array to device once
    if use_gpu():
        t_d = to_device(t)
    else:
        t_d = t

    def _residual_fun(p):
        if use_gpu():
            model = to_host(_model_gpu(t_d, p))
        else:
            model = _model_cpu(t, p)
        return x_orig - model

    def _jac_fun(p):
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
