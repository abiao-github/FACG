"""生成 FACG 性能测试使用的模拟时间序列及其真实参数记录。

测试数据采用非均匀时间采样，由若干正弦信号叠加高斯噪声构成；
固定随机种子以便在相同参数下复现数据。
"""

import os
import numpy as np

def generate_single_file(filename, n_points, time_span, signals, noise_level=0.1):
    """生成并保存一份由多正弦信号和噪声组成的测试数据。

    参数
    ----
    filename:
        输出文件路径。
    n_points:
        随机生成的采样点数量。
    time_span:
        采样时间范围，时间戳在 ``[0, time_span)`` 内均匀抽取后排序。
    signals:
        信号参数序列，每项依次为频率、振幅和相位（弧度）。
    noise_level:
        加性高斯噪声的标准差，默认为 0.1。

    结果以两列文本格式写入文件，并打印生成概要。
    """
    t = np.sort(np.random.uniform(0, time_span, n_points))
    y = np.zeros_like(t)
    for freq, amp, phase in signals:
        y += amp * np.sin(2 * np.pi * freq * t + phase)
    y += np.random.normal(0, noise_level, n_points)
    
    np.savetxt(filename, np.column_stack([t, y]), fmt="%14.6f %14.6f")
    print(f"✅ 生成完毕: {filename} (点数: {n_points}, 信号数: {len(signals)}, 时间跨度: {time_span})")

def generate_all_test_data(output_dir=".", num_files=3, num_signals=5):
    """批量生成性能对比数据集并保存信号真实参数。

    参数
    ----
    output_dir:
        测试数据和参数记录文件的输出目录，默认为当前目录。
    num_files:
        要生成的数据文件数量。小于 1 时仍生成一份文件。
    num_signals:
        每份数据注入的正弦信号数量。小于 1 时仍注入一个信号。

    函数会固定 NumPy 随机种子，以便复现实验；除各个数据文件外，还会生成
    ``frequency_record.log``，按振幅降序记录每个文件的频率、振幅和相位。
    """
    print(f"正在生成 FACG 与 SigSpec 性能对比测试数据 ({num_files} 个文件, 每个注入 {num_signals} 个频率)...\n")
    
    np.random.seed(42)
    all_signals = []
    for i in range(1, max(1, num_files) + 1):
        n_points = np.random.randint(5000, 10000)
        time_span = np.random.uniform(80.0, 300.0)
        noise_level = np.random.uniform(0.3, 0.8)
        
        signals = []
        for _ in range(max(1, num_signals)):
            signals.append((
                np.random.uniform(0.1, 40.0),  # freq
                np.random.uniform(0.2, 1.5),   # amp
                np.random.uniform(0, 2 * np.pi) # phase
            ))
        
        filename = f"FACG_Test_{i:02d}.dat"
        generate_single_file(os.path.join(output_dir, filename), n_points, time_span, signals, noise_level)
        all_signals.append((filename, signals))

    # ==========================================
    # 写入真实信号记录文件 (Ground Truth)
    # ==========================================
    record_path = os.path.join(output_dir, "frequency_record.log")
    with open(record_path, "w", encoding="utf-8") as f_out:
        f_out.write("FACG/SigSpec 模拟测试数据真实参数记录\n")
        f_out.write("=======================================\n\n")
        
        for filename, signals in all_signals:
            f_out.write(f"=== {filename} ===\n")
            f_out.write(f"{'Frequency':>14} {'Amplitude':>14} {'Phase(rad)':>14}\n")
            # 按振幅从大到小排序，方便与软件提取的结果（通常按振幅递减）逐行对比
            for freq, amp, phase in sorted(signals, key=lambda x: x[1], reverse=True):
                f_out.write(f"{freq:14.6f} {amp:14.6f} {phase:14.6f}\n")
            f_out.write("\n")

    print("\n🎉 生成结束。")
    print(f"📄 真实信号参数已记录至: {record_path}")
    print("建议先使用 FACG 运行（极速出结果），再使用 SigSpec 运行以体验性能差距。")