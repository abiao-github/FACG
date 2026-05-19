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

def generate_all_test_data(output_dir="."):
    """
    生成三份用于 SigSpec 与 FACG 性能对比的中等规模测试数据。
    将会在指定的目录下生成 FACG_Test_01.dat, FACG_Test_02.dat, FACG_Test_03.dat。
    """
    print("正在生成 FACG 与 SigSpec 性能对比测试数据 (SigSpec 预估处理时间约为 1-2 分钟)...\n")
    
    # 数据 1: 常规混合信号
    # 中等规模: 6000 点, 6 个信号。
    signals_1 = [
        (1.23, 1.0, 0.5), (3.45, 0.8, 1.2), (5.67, 0.6, 2.1), 
        (10.1, 0.5, 0.8), (15.5, 0.4, 1.8), (22.8, 0.3, 2.5)
    ]
    generate_single_file(os.path.join(output_dir, "FACG_Test_01.dat"), 6000, 120, signals_1, 0.5)

    # 数据 2: 长时间基线导致频率网格密集
    # 中等规模: 7000 点, 5 个信号, 时间基线 250。
    signals_2 = [
        (0.12345, 1.5, 0.0), (1.23456, 1.0, 1.0), (4.56789, 0.8, 2.0), 
        (8.76, 0.7, 1.5), (12.9, 0.6, 0.1)
    ]
    generate_single_file(os.path.join(output_dir, "FACG_Test_02.dat"), 7000, 250, signals_2, 0.8)

    # 数据 3: 信号数量较多，测试迭代与多正弦联合优化的矩阵计算
    # 中等规模: 8000 点, 7 个随机信号。
    np.random.seed(42)
    signals_3 = []
    for _ in range(7):
        signals_3.append((np.random.uniform(0.5, 30.0), np.random.uniform(0.2, 1.0), np.random.uniform(0, 2*np.pi)))
    generate_single_file(os.path.join(output_dir, "FACG_Test_03.dat"), 8000, 100, signals_3, 0.4)

    # ==========================================
    # 写入真实信号记录文件 (Ground Truth)
    # ==========================================
    record_path = os.path.join(output_dir, "frequency_record.txt")
    with open(record_path, "w", encoding="utf-8") as f_out:
        f_out.write("FACG/SigSpec 模拟测试数据真实参数记录\n")
        f_out.write("=======================================\n\n")
        
        def write_record(filename, signals):
            f_out.write(f"=== {filename} ===\n")
            f_out.write(f"{'Frequency':>14} {'Amplitude':>14} {'Phase(rad)':>14}\n")
            # 按振幅从大到小排序，方便与软件提取的结果（通常按振幅递减）逐行对比
            for freq, amp, phase in sorted(signals, key=lambda x: x[1], reverse=True):
                f_out.write(f"{freq:14.6f} {amp:14.6f} {phase:14.6f}\n")
            f_out.write("\n")

        write_record("FACG_Test_01.dat", signals_1)
        write_record("FACG_Test_02.dat", signals_2)
        write_record("FACG_Test_03.dat", signals_3)

    print("\n🎉 生成结束。")
    print(f"📄 真实信号参数已记录至: {record_path}")
    print("建议先使用 FACG 运行（极速出结果），再使用 SigSpec 运行以体验性能差距。")