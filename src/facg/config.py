"""
FACG 分析配置的数据类定义。

集中声明用户可调整的分析参数及默认值。频率网格、停止条件、输入输出
和绘图行为均由 :class:`FACGConfig` 管理，默认设置尽量与 SigSpec 约定一致。
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional


@dataclasses.dataclass
class FACGConfig:
    """描述一次 FACG 频率分析所需的全部可配置参数。

    Parameters
    ----------
    input_file : str or Path
        输入时间序列文件路径。支持空白分隔且无表头的文本格式，默认第 0 列
        为时间、第 1 列为观测值；其他格式和列号由读取模块处理。
    output_dir : str or Path or None
        结果文件输出目录。为 ``None`` 时，在输入文件旁自动创建以文件主名命名
        的子目录。
    time_col : int
        时间数据所在的零起始列号，默认为 0。
    data_col : int
        观测值所在的零起始列号，默认为 1。

    Frequency grid
    ~~~~~~~~~~~~~~
    freq_low : float or None
        频率下限，单位为每时间单位的周期数。为 ``None`` 时使用
        ``rayleigh / oversampling``。
    freq_high : float or None
        频率上限。为 ``None`` 时根据中位采样间隔采用 Nyquist 频率
        ``0.5 / median(Δt)``。
    freq_step : float or None
        频率网格步长。为 ``None`` 时使用 ``rayleigh / oversampling``。
    nyquist_coeff : float
        计算 Nyquist 上限时使用的系数，默认为 0.5。
    oversampling : float
        应用于 Rayleigh 分辨率的过采样倍数，默认为 20.0，与 SigSpec 相同。

    Stopping criteria
    ~~~~~~~~~~~~~~~~~
    sig_limit : float
        显著性阈值。当最高峰显著性低于此值时停止迭代，默认为 5.0。
    csig_limit : float
        累积显著性停止阈值，默认为 0（禁用该停止条件）。
    max_iter : int
        预白化迭代的最大次数，默认为 999。

    Output control
    ~~~~~~~~~~~~~~
    write_spectrum : bool
        是否在每轮迭代后写出残差显著性/振幅频谱，默认为 True。
    write_residuals : bool
        是否在每轮迭代后写出残差时间序列，默认为 True。
    write_phase_diagram : bool
        是否为检测到的信号写出折叠相位图数据，默认为 False。
    plot : bool
        是否生成包含时间序列、拟合信号、振幅频谱和残差的汇总 PNG 图。
        默认为 False；命令行可用 ``--plot`` 启用。
    quiet : bool
        是否抑制运行进度信息，默认为 False。
    """

    # --- I/O ---------------------------------------------------------------
    input_file: str = ""
    output_dir: Optional[str] = None
    time_col: int = 0
    data_col: int = 1

    # --- Frequency grid ----------------------------------------------------
    freq_low: Optional[float] = None
    freq_high: Optional[float] = None
    freq_step: Optional[float] = None
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
        """解析并创建本次分析的输出目录。

        若配置了 ``output_dir``，直接使用该路径；否则在输入文件所在目录下
        以输入文件主名作为目录名。目录不存在时会递归创建。

        Returns
        -------
        Path
            已确保存在的输出目录路径。
        """
        if self.output_dir is not None:
            p = Path(self.output_dir)
        else:
            inp = Path(self.input_file)
            p = inp.parent / inp.stem
        p.mkdir(parents=True, exist_ok=True)
        return p
