import os
import numpy as np

def generate_single_file(filename, n_points, time_span, signals, noise_level=0.1):
    """生成单份非均匀采样的时间序列测试数据。"""
    t = np.sort(np.random.uniform(0, time_span, n_points))
    y = np.zeros_like(t)
    for freq, amp, phase in signals:
        y += amp * np.sin(2 * np.pi * freq * t + phase)
    y += np.random.normal(0, noise_level, n_points)
    
    np.savetxt(filename, np.column_stack([t, y]), fmt="%14.6f %14.6f")
    print(f"✅ 生成完毕: {filename} (点数: {n_points}, 信号数: {len(signals)}, 时间跨度: {time_span})")

def generate_all_test_data(output_dir=".", num_files=3, num_signals=5):
    """
    生成用于 SigSpec 与 FACG 性能对比的测试数据。
    用户可指定生成的文件数量以及每个文件注入的真实频率信号数量。
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