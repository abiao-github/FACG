"""
Unified CPU/GPU computing backend.

Provides a transparent wrapper: when CuPy is available and a CUDA GPU is
detected the module exposes CuPy; otherwise it falls back to NumPy.

All heavy-lifting functions in FACG use ``xp`` from this module so that
the rest of the code is backend-agnostic.
"""

import sys
import numpy as np

# ---------------------------------------------------------------------------
# Attempt to import CuPy; fall back gracefully
# ---------------------------------------------------------------------------
_USE_GPU = False
_GPU_INFO = ""
_GPU_WARN = ""

try:
    import cupy as cp

    # Quick sanity check – allocate a tiny array on the device.
    cp.array([1.0])
    _USE_GPU = True
    dev = cp.cuda.Device()
    _GPU_INFO = (
        f"GPU acceleration enabled: {dev.name.decode()}, "
        f"CuPy {cp.__version__}, CUDA"
    )
except ImportError:
    cp = None
    _GPU_WARN = (
        "CuPy not installed — running on CPU only.\n"
        "  To enable GPU acceleration, install CuPy for your CUDA version:\n"
        "    pip install cupy-cuda12x      # for CUDA 12.x\n"
        "    pip install cupy-cuda11x      # for CUDA 11.x\n"
        "  See https://docs.cupy.dev/en/stable/install.html"
    )
except Exception as _exc:
    cp = None
    _GPU_WARN = (
        f"CuPy found but GPU initialisation failed: {_exc}\n"
        "  Falling back to CPU (NumPy)."
    )

# Public "array library" handle – use this everywhere instead of np/cp.
xp = cp if _USE_GPU else np


def use_gpu() -> bool:
    """Return *True* if GPU acceleration is active."""
    return _USE_GPU


def to_device(arr: np.ndarray):
    """Move a NumPy array to the current device (GPU if available)."""
    if _USE_GPU:
        return cp.asarray(arr)
    return arr


def to_host(arr) -> np.ndarray:
    """Move an array back to host (CPU) memory."""
    if _USE_GPU and hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def backend_name() -> str:
    """Human-readable backend description."""
    if _USE_GPU:
        dev = cp.cuda.Device()
        return f"GPU (CuPy {cp.__version__}, {dev.name.decode()}, CUDA)"
    return f"CPU (NumPy {np.__version__})"


def print_backend_status(file=sys.stderr) -> None:
    """Print a one-time status message about the computing backend.

    Called automatically during ``facg`` startup so the user always
    knows whether GPU acceleration is active.
    """
    if _USE_GPU:
        print(f"  ✓ {_GPU_INFO}", file=file)
    elif _GPU_WARN:
        print(f"  ⚠ {_GPU_WARN}", file=file)
