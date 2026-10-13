"""
FACG 的 CPU/GPU 统一计算后端。

优先使用 CuPy/CUDA（NVIDIA）、其次使用 PyTorch/MPS（Apple Silicon），
若加速后端不可用则回退到 CPU 上的 NumPy。

模块提供兼容不同数组库的 ``xp`` 包装对象，以及主机与设备间的数据转换函数，
使上层计算代码尽量不依赖具体后端。
"""

import sys
import numpy as np
import platform
import subprocess
import shutil


# ---------------------------------------------------------------------------
# Backend Abstraction Layer to smooth over API differences (e.g. axis/dim)
# ---------------------------------------------------------------------------
class _BackendWrapper:
    """适配 NumPy、CuPy 和 PyTorch 常用数组操作差异的轻量包装器。"""

    def __init__(self, lib, is_torch=False):
        """记录底层数组库及其是否为 PyTorch。

        参数
        ----
        lib:
            NumPy、CuPy 或 PyTorch 模块对象。
        is_torch:
            底层模块为 PyTorch 时设为 True，以启用参数名和设备兼容处理。
        """
        self._lib = lib
        self._is_torch = is_torch

    def __getattr__(self, name):
        """将未显式适配的属性和函数转交给底层数组库。

        PyTorch MPS 的浮点运算使用 float32，因此请求 ``float64`` 时返回
        ``torch.float32``；其他名称按原样转发。
        """
        # MPS does not support float64, use float32 instead
        if self._is_torch and name == "float64":
            return getattr(self._lib, "float32")
        return getattr(self._lib, name)

    @property
    def pi(self):
        """返回高精度常量 π，避免依赖后端的常量实现差异。"""
        # torch.pi is available from 1.7, but np.pi is safer
        return np.pi

    def sum(self, *args, **kwargs):
        """调用后端求和，并将通用参数 ``axis`` 适配为 PyTorch 的 ``dim``。"""
        if self._is_torch and 'axis' in kwargs:
            kwargs['dim'] = kwargs.pop('axis')
        return self._lib.sum(*args, **kwargs)

    def zeros(self, *args, **kwargs):
        """创建零值数组；使用 PyTorch 时自动将其放置在 MPS 设备上。"""
        if self._is_torch:
            kwargs['device'] = 'mps'
        return self._lib.zeros(*args, **kwargs)

    def empty(self, *args, **kwargs):
        """创建未初始化数组；使用 PyTorch 时自动将其放置在 MPS 设备上。"""
        if self._is_torch:
            kwargs['device'] = 'mps'
        return self._lib.empty(*args, **kwargs)

    def maximum(self, x1, x2):
        """逐元素取较大值，并处理 PyTorch 不接受标量张量混用的情形。"""
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
_MISSING_DEPS_PROMPT = False
xp = _BackendWrapper(np)  # Default to numpy before initialization
torch = None
cp = None
_BACKEND_INITIALIZED = False


def _detect_hardware():
    """在不导入 GPU 数组库的情况下粗略识别可用 GPU 硬件类型。

    返回 ``"apple"``、``"nvidia"`` 或 ``"none"``。检测结果用于决定尝试
    哪种加速后端，并不代表驱动或对应 Python 依赖已成功初始化。
    """
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "apple"
        
    if shutil.which("nvidia-smi") is not None:
        return "nvidia"
        
    if platform.system() == "Linux" and shutil.which("lspci"):
        try:
            out = subprocess.check_output(["lspci"], stderr=subprocess.DEVNULL, text=True).lower()
            if "nvidia" in out and ("vga" in out or "3d controller" in out):
                return "nvidia"
        except Exception:
            pass
            
    return "none"


def initialize_backend(force_cpu: bool = False):
    """检测并初始化全局计算后端。

    初始化过程会重置后端状态，按 CUDA、Apple Metal、NumPy 的优先级探测，
    并更新 ``xp`` 及状态描述。模块导入时会自动调用一次；命令行也可再次调用
    并强制使用 CPU。

    参数
    ----
    force_cpu:
        为 True 时跳过 GPU 探测并强制选择 NumPy。
    """
    global xp, _USE_GPU, _USE_MPS, _BACKEND_INFO, _BACKEND_WARN
    global _BACKEND_SIMPLE_NAME, _MISSING_DEPS_PROMPT, torch, cp, _BACKEND_INITIALIZED

    # Reset state
    _USE_GPU = False
    _USE_MPS = False
    _BACKEND_INFO = ""
    _BACKEND_WARN = ""
    _BACKEND_SIMPLE_NAME = ""
    _MISSING_DEPS_PROMPT = False
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
    hw_type = _detect_hardware()
    backend_lib = None
    is_torch = False

    # 1. Try CUDA (CuPy)
    if hw_type in ("nvidia", "none"):
        try:
            import cupy as cp_lib
            cp_lib.array([1.0])
            
            gpu_name = "CUDA GPU"
            try:
                dev_id = cp_lib.cuda.Device().id
                props = cp_lib.cuda.runtime.getDeviceProperties(dev_id)
                name_val = props.get('name', b'CUDA GPU')
                gpu_name = name_val.decode('utf-8') if isinstance(name_val, bytes) else str(name_val)
            except Exception:
                pass
                
            _BACKEND_INFO = f"GPU: {gpu_name} (CuPy {cp_lib.__version__}, CUDA)"
            _BACKEND_SIMPLE_NAME = "GPU (CUDA)"
            cp = cp_lib
            _USE_GPU = True
            backend_lib = cp
            is_torch = False
        except ImportError:
            cp = None
            _USE_GPU = False
            if hw_type == "nvidia":
                _MISSING_DEPS_PROMPT = True
                _BACKEND_WARN = (
                    "NVIDIA GPU hardware detected, but CuPy is not installed.\n"
                    "  To enable GPU acceleration, install CuPy for your CUDA version:\n"
                    "    pip install cupy-cuda12x      # for CUDA 12.x\n"
                    "    pip install cupy-cuda11x      # for CUDA 11.x\n"
                    "  See https://docs.cupy.dev/en/stable/install.html"
                )
        except Exception as e:
            cp = None
            _USE_GPU = False
            if hw_type == "nvidia":
                _MISSING_DEPS_PROMPT = True
                _BACKEND_WARN = (
                    f"NVIDIA GPU hardware detected, but CuPy failed to initialize:\n"
                    f"  {e}\n"
                    f"  Please check your NVIDIA drivers and CuPy installation."
                )

    # 2. If no CUDA, try Apple Metal (PyTorch-MPS)
    if not _USE_GPU:
        if hw_type in ("apple", "none"):
            try:
                import torch as torch_lib
                if torch_lib.backends.mps.is_available():
                    torch_lib.tensor([1.0], device="mps")
                    _BACKEND_INFO = f"GPU: Apple Metal (PyTorch {torch_lib.__version__})"
                    _BACKEND_SIMPLE_NAME = "GPU (Metal)"
                    torch = torch_lib
                    _USE_MPS = True
                    backend_lib = torch
                    is_torch = True
                else:
                    torch = None
                    if hw_type == "apple":
                        _MISSING_DEPS_PROMPT = True
                        _BACKEND_WARN = "Apple Silicon detected, but PyTorch MPS (Metal) backend is unavailable."
            except ImportError:
                torch = None
                _USE_MPS = False
                if hw_type == "apple":
                    _MISSING_DEPS_PROMPT = True
                    _BACKEND_WARN = (
                        "Apple Silicon detected, but PyTorch is not installed.\n"
                        "  To enable Apple Metal GPU acceleration, install PyTorch:\n"
                        "    pip install torch"
                    )
            except Exception as e:
                torch = None
                _USE_MPS = False
                if hw_type == "apple":
                    _MISSING_DEPS_PROMPT = True
                    _BACKEND_WARN = (
                        f"Apple Silicon detected, but Metal backend failed to initialize:\n"
                        f"  {e}\n"
                        f"  Please check your PyTorch installation."
                    )

    # 3. Fallback to CPU and set warnings
    if backend_lib is None:
        backend_lib = np
        is_torch = False
        _BACKEND_INFO = f"CPU (NumPy {np.__version__})"
        _BACKEND_SIMPLE_NAME = "CPU"
        
        if not _BACKEND_WARN:
            if platform.system() == "Darwin":
                _BACKEND_WARN = "No Apple Metal acceleration found — running on CPU."
            else:
                _BACKEND_WARN = "No CUDA acceleration found — running on CPU."

    xp._lib = backend_lib
    xp._is_torch = is_torch
    _BACKEND_INITIALIZED = True

def use_gpu() -> bool:
    """判断当前是否已启用 NVIDIA CUDA 加速后端。"""
    return _USE_GPU

def use_mps() -> bool:
    """判断当前是否已启用 Apple Metal（MPS）加速后端。"""
    return _USE_MPS

def to_device(arr: np.ndarray):
    """将 NumPy 数组转换为当前后端可计算的数组。

    CUDA 使用 CuPy 数组；MPS 将数组复制到 Metal 设备，并在需要时将
    float64 降为 float32；CPU 模式则原样返回输入数组。
    """
    if _USE_GPU:
        return cp.asarray(arr)
    if _USE_MPS:
        # torch is guaranteed to be the pytorch module if _USE_MPS is True
        # MPS requires float32 for floating-point operations
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        return torch.from_numpy(arr.copy()).to("mps")
    return arr


def to_host(arr) -> np.ndarray:
    """将设备数组复制回主机并转换为 NumPy 数组。

    MPS 计算结果会从 float32 恢复为 NumPy 后续处理所需的 float64；
    CPU 输入通过 ``numpy.asarray`` 规范为 NumPy 数组。
    """
    if _USE_GPU:
        if hasattr(arr, "get"):
            return arr.get()
    if _USE_MPS:
        cpu_arr = arr.cpu().numpy()
        # Restore float64 precision expected by NumPy routines downstream
        if cpu_arr.dtype == np.float32:
            cpu_arr = cpu_arr.astype(np.float64)
        return cpu_arr
    return np.asarray(arr)


def backend_name() -> str:
    """返回当前计算后端的简短可读名称，例如 ``CPU`` 或 ``GPU (CUDA)``。"""
    return _BACKEND_SIMPLE_NAME
    
def requires_dependency_prompt() -> bool:
    """判断是否检测到 GPU 硬件但加速依赖缺失或初始化失败。"""
    return _MISSING_DEPS_PROMPT


def print_backend_status(file=sys.stderr) -> None:
    """向指定输出流打印后端状态，并在 CPU 回退时附带可用警告。

    参数
    ----
    file:
        输出流对象，默认使用标准错误流。
    """
    print(f"  ✓ {_BACKEND_INFO}", file=file)
    if not (_USE_GPU or _USE_MPS) and _BACKEND_WARN:
        print(f"  ⚠ {_BACKEND_WARN}", file=file)


# --- Auto-initialize on first import ---
# This ensures that the backend is ready for users of the Python API.
# The CLI entry point can override this by calling initialize_backend() again.
initialize_backend()
