"""
FACG — CPU 与 GPU 混合计算的频率分析工具。

本包提供受 SigSpec（Reegen 2007）启发的迭代预白化频率分析功能，
并通过统一计算后端在可用时利用 GPU 加速。
"""

__version__ = "0.1.0"
__author__ = "Niu Hubiao"

from facg.config import FACGConfig
from facg.prewhiten import run_analysis
