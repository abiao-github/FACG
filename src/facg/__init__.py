"""
FACG — Frequency Analysis of CPU and GPU mixed computing.

A GPU-accelerated iterative prewhitening frequency analysis tool
inspired by SigSpec (Reegen 2007).
"""

__version__ = "0.1.0"
__author__ = "Niu Hubiao"

from facg.config import FACGConfig
from facg.prewhiten import run_analysis
