"""
Unified CPU/GPU computing backend.

<<<<<<< HEAD
Provides a transparent wrapper: when CuPy is available and a CUDA GPU is
detected the module exposes CuPy; otherwise it falls back to NumPy.
=======
Provides a transparent wrapper:
- On systems with an NVIDIA GPU and CuPy, it uses CUDA.
- On Apple Silicon systems with PyTorch, it uses the Metal (MPS) backend.
- Otherwise, it falls back to NumPy on the CPU.
>>>>>>> 9960956 (Initial commit for FACG package)

All heavy-lifting functions in FACG use ``xp`` from this module so that
the rest of the code is backend-agnostic.
"""

import sys
import numpy as np
<<<<<<< HEAD

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

=======
import platform


# ---------------------------------------------------------------------------
# Backend Abstraction Layer to smooth over API differences (e.g. axis/dim)
# ---------------------------------------------------------------------------
class _BackendWrapper:
    """A wrapper to unify calls to numpy, cupy, and pytorch."""
    def __init__(self, lib, is_torch=False):
        self._lib = lib
        self._is_torch = is_torch

    def __getattr__(self, name):
        # MPS does not support float64, use float32 instead
        if self._is_torch and name == "float64":
            return getattr(self._lib, "float32")
        return getattr(self._lib, name)

    @property
    def pi(self):
        # torch.pi is available from 1.7, but np.pi is safer
        return np.pi

    def sum(self, *args, **kwargs):
        if self._is_torch and 'axis' in kwargs:
            kwargs['dim'] = kwargs.pop('axis')
        return self._lib.sum(*args, **kwargs)

    def zeros(self, *args, **kwargs):
        if self._is_torch:
            kwargs['device'] = 'mps'
        return self._lib.zeros(*args, **kwargs)

    def empty(self, *args, **kwargs):
        if self._is_torch:
            kwargs['device'] = 'mps'
        return self._lib.empty(*args, **kwargs)

    def maximum(self, x1, x2):
        if self._is_torch:
            if not isinstance(x2, self._lib.Tensor):
                return self._lib.clamp(x1, min=x2)
            if not isinstance(x1, self._lib.Tensor):
                return self._lib.clamp(x2, min=x1)
        return self._lib.maximum(x1, x2)


# ---------------------------------------------------------------------------
# Global backend state variables
# ---------------------------------------------------------------------------
_USE_GPU = False
_USE_MPS = False
_BACKEND_INFO = ""
_BACKEND_WARN = ""
_BACKEND_SIMPLE_NAME = ""
xp = _BackendWrapper(np)  # Default to numpy before initialization
torch = None
cp = None
_BACKEND_INITIALIZED = False


def initialize_backend(force_cpu: bool = False):
    """
    Detect and initialize the computing backend.

    This function sets the global `xp` object to the best available backend.
    It is called automatically on module import, but can be called again
    to force a specific backend (e.g., force CPU).

    Order of preference: CuPy (CUDA) > PyTorch (MPS) > NumPy (CPU).

    Parameters
    ----------
    force_cpu : bool
        If True, bypasses GPU detection and forces the NumPy backend.
    """
    global xp, _USE_GPU, _USE_MPS, _BACKEND_INFO, _BACKEND_WARN
    global _BACKEND_SIMPLE_NAME, torch, cp, _BACKEND_INITIALIZED

    # Reset state
    _USE_GPU = False
    _USE_MPS = False
    _BACKEND_INFO = ""
    _BACKEND_WARN = ""
    _BACKEND_SIMPLE_NAME = ""
    torch = None
    cp = None

    # --- Force CPU mode if requested ---
    if force_cpu:
        xp._lib = np
        xp._is_torch = False
        _BACKEND_INFO = f"CPU (NumPy {np.__version__}) [FORCED]"
        _BACKEND_SIMPLE_NAME = "CPU"
        _BACKEND_INITIALIZED = True
        return

    # --- Auto-detection logic ---
    backend_lib = None
    is_torch = False

    # 1. Try CUDA (CuPy)
    try:
        import cupy as cp_lib
        cp_lib.array([1.0])
        cp = cp_lib
        _USE_GPU = True
        dev = cp.cuda.Device()
        _BACKEND_INFO = f"GPU: {dev.name.decode()} (CuPy {cp.__version__}, CUDA)"
        _BACKEND_SIMPLE_NAME = "GPU (CUDA)"
        backend_lib = cp
        is_torch = False
    except (ImportError, Exception):
        cp = None

    # 2. If no CUDA, try Apple Metal (PyTorch-MPS)
    if not _USE_GPU:
        try:
            import torch as torch_lib
            if torch_lib.backends.mps.is_available():
                torch_lib.tensor([1.0], device="mps")
                torch = torch_lib
                _USE_MPS = True
                _BACKEND_INFO = f"GPU: Apple Metal (PyTorch {torch.__version__})"
                _BACKEND_SIMPLE_NAME = "GPU (Metal)"
                backend_lib = torch
                is_torch = True
            else:
                torch = None
        except ImportError:
            torch = None

    # 3. Fallback to CPU and set warnings
    if backend_lib is None:
        backend_lib = np
        is_torch = False
        _BACKEND_INFO = f"CPU (NumPy {np.__version__})"
        _BACKEND_SIMPLE_NAME = "CPU"
        
        if platform.system() == "Darwin":
            _BACKEND_WARN = (
                "PyTorch not installed (or no Metal support) — running on CPU only.\n"
                "  To enable Apple Metal GPU acceleration, install PyTorch:\n"
                "    pip install torch"
            )
        else:
            _BACKEND_WARN = (
                "CuPy not installed — running on CPU only.\n"
                "  To enable GPU acceleration, install CuPy for your CUDA version:\n"
                "    pip install cupy-cuda12x      # for CUDA 12.x\n"
                "    pip install cupy-cuda11x      # for CUDA 11.x\n"
                "  See https://docs.cupy.dev/en/stable/install.html"
            )

    xp._lib = backend_lib
    xp._is_torch = is_torch
    _BACKEND_INITIALIZED = True

def use_gpu() -> bool:
    """Return *True* if CUDA GPU acceleration is active."""
    return _USE_GPU

def use_mps() -> bool:
    """Return *True* if MPS acceleration is active (Apple Metal)."""
    return _USE_MPS
>>>>>>> 9960956 (Initial commit for FACG package)

def to_device(arr: np.ndarray):
    """Move a NumPy array to the current device (GPU if available)."""
    if _USE_GPU:
        return cp.asarray(arr)
<<<<<<< HEAD
=======
    if _USE_MPS:
        # torch is guaranteed to be the pytorch module if _USE_MPS is True
        # MPS requires float32 for floating-point operations
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        return torch.from_numpy(arr.copy()).to("mps")
>>>>>>> 9960956 (Initial commit for FACG package)
    return arr


def to_host(arr) -> np.ndarray:
    """Move an array back to host (CPU) memory."""
<<<<<<< HEAD
    if _USE_GPU and hasattr(arr, "get"):
        return arr.get()
=======
    if _USE_GPU:
        if hasattr(arr, "get"):
            return arr.get()
    if _USE_MPS:
        cpu_arr = arr.cpu().numpy()
        # Restore float64 precision expected by NumPy routines downstream
        if cpu_arr.dtype == np.float32:
            cpu_arr = cpu_arr.astype(np.float64)
        return cpu_arr
>>>>>>> 9960956 (Initial commit for FACG package)
    return np.asarray(arr)


def backend_name() -> str:
    """Human-readable backend description."""
<<<<<<< HEAD
    if _USE_GPU:
        dev = cp.cuda.Device()
        return f"GPU (CuPy {cp.__version__}, {dev.name.decode()}, CUDA)"
    return f"CPU (NumPy {np.__version__})"
=======
    return _BACKEND_SIMPLE_NAME
>>>>>>> 9960956 (Initial commit for FACG package)


def print_backend_status(file=sys.stderr) -> None:
    """Print a one-time status message about the computing backend.
<<<<<<< HEAD

    Called automatically during ``facg`` startup so the user always
    knows whether GPU acceleration is active.
    """
    if _USE_GPU:
        print(f"  ✓ {_GPU_INFO}", file=file)
    elif _GPU_WARN:
        print(f"  ⚠ {_GPU_WARN}", file=file)
=======
    """
    if _USE_GPU or _USE_MPS:
        print(f"  ✓ {_BACKEND_INFO}", file=file)
    elif _BACKEND_WARN:
        print(f"  ⚠ {_BACKEND_WARN}", file=file)


# --- Auto-initialize on first import ---
# This ensures that the backend is ready for users of the Python API.
# The CLI entry point can override this by calling initialize_backend() again.
initialize_backend()
>>>>>>> 9960956 (Initial commit for FACG package)
