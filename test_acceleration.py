#!/usr/bin/env python3
"""
FACG 硬件加速性能对比测试脚本。
用于展示纯 CPU 模式与 GPU (CUDA/Metal) 模式下的计算速度差异。
"""

import os
import time
import numpy as np
import tempfile

from facg import backend
from facg.config import FACGConfig
from facg.prewhiten import run_analysis


def generate_mock_data(filepath, N=10000, T=100.0):
    """生成包含两个正弦波信号和高斯噪声的模拟时间序列数据。"""
    print(f"正在生成 {N} 个数据点的模拟数据 (时间跨度 T={T})...")
    t = np.sort(np.random.uniform(0, T, N))
    
    # 注入两个已知频率的信号
    f1, A1, phi1 = 12.345, 0.05, 1.2
    f2, A2, phi2 = 45.678, 0.03, 3.4
    signal = A1 * np.sin(2 * np.pi * f1 * t + phi1) + \
             A2 * np.sin(2 * np.pi * f2 * t + phi2)
    
    # 添加高斯白噪声
    noise = np.random.normal(0, 0.1, N)
    data = signal + noise
    
    # 写入临时文件
    np.savetxt(filepath, np.column_stack([t, data]), fmt="%.6f")
    return filepath


def run_test(filepath, force_cpu):
    """使用指定的后端运行 FACG 分析并返回耗时。"""
    # 动态切换后端：强制 CPU 或 自动检测可用 GPU
    backend.initialize_backend(force_cpu=force_cpu)
    
    backend_name = backend.backend_name()
    print(f"\n▶ 开始测试后端: [{backend_name}]")
    
    # 配置 FACG：只迭代 3 次，计算到较高频率以增加计算量，关闭文件输出以测算纯算法时间
    cfg = FACGConfig(
        input_file=filepath,
        freq_high=50.0,       # 频率上限
        oversampling=20.0,    # 过采样率
        max_iter=3,           # 仅提取前 3 个信号
        write_spectrum=False, # 禁用中间文件写入
        write_residuals=False,
        quiet=True            # 禁用内部打印
    )
    
    start_time = time.time()
    run_analysis(cfg)
    end_time = time.time()
    
    elapsed = end_time - start_time
    print(f"  ✓ [{backend_name}] 计算完成，耗时: {elapsed:.3f} 秒")
    return elapsed


if __name__ == "__main__":
    # 确保当前环境能够检测到 GPU
    backend.initialize_backend(force_cpu=False)
    if not (backend.use_gpu() or backend.use_mps()):
        print("⚠ 警告: 当前环境未检测到 CUDA (CuPy) 或 Metal (PyTorch)。")
        print("  程序只能运行两次 CPU 测试，您将看不到加速效果。请先确保已安装加速库。")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = os.path.join(tmpdir, "mock_data.dat")
        generate_mock_data(data_file)
        
        time_gpu = run_test(data_file, force_cpu=False)
        time_cpu = run_test(data_file, force_cpu=True)
        
        print("\n" + "="*50)
        print("🚀 性能对比总结")
        print("="*50)
        print(f"  纯 CPU 模式耗时 : {time_cpu:.3f} 秒")
        print(f"  GPU 加速模式耗时: {time_gpu:.3f} 秒")
        if time_gpu > 0:
            print(f"  🌟 加速比       : {time_cpu / time_gpu:.1f} 倍 !!!")
        print("="*50)
        print("注意: 实际加速比取决于数据量(N)和频率网格大小。数据越多，GPU 优势越明显。")