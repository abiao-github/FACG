#!/usr/bin/env python3
"""
FACG 命令行程序入口。

支持单文件或多文件分析、通配符匹配、配置文件默认值、测试数据生成以及
可选汇总图。命令行参数优先于 ``facg.conf`` 中的同名配置。

示例::

    facg data.dat
    facg data.dat --sig-limit 6 --oversampling 40 --output-dir ./results
    facg file1.dat file2.dat --plot
    facg --help
"""

from __future__ import annotations

import argparse
import sys
import glob
import time
from pathlib import Path

from facg.config import FACGConfig
from facg.prewhiten import run_analysis
from facg import backend
from facg.config_io import CONFIG_FILENAME, generate_default_config, load_config_for_argparse
from facg.testdata import generate_all_test_data


def _build_parser() -> argparse.ArgumentParser:
    """创建并返回 FACG 命令行参数解析器。

    参数分组覆盖输入文件、频率网格、迭代停止条件、结果输出、后端选择及
    辅助数据生成等选项。配置文件默认值会在解析命令行之前注入。

    返回
    ----
    argparse.ArgumentParser
        配置完成的参数解析器。
    """
    p = argparse.ArgumentParser(
        prog="facg",
        description=(
            "FACG — Frequency Analysis of CPU and GPU mixed computing.\n"
            "GPU-accelerated iterative prewhitening frequency analysis "
            "inspired by SigSpec (Reegen 2007)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "files",
        nargs="*",
        help="One or more input time-series files (whitespace-delimited, "
             "no header, any legal filename). Supports wildcards (e.g., *.dat). "
             "If omitted, processes all *.dat and *.txt files in the current directory.",
    )
    p.add_argument(
        "--time-col", type=int, default=0,
        help="0-based column index for time (default: 0).",
    )
    p.add_argument(
        "--data-col", type=int, default=1,
        help="0-based column index for data (default: 1).",
    )
    p.add_argument(
        "--sig-limit", type=float, default=5.0,
        help="Significance threshold (default: 5.0).",
    )
    p.add_argument(
        "--csig-limit", type=float, default=0.0,
        help="Cumulative significance threshold (default: 0 = disabled).",
    )
    p.add_argument(
        "--max-iter", type=int, default=999,
        help="Maximum prewhitening iterations (default: 999).",
    )
    p.add_argument(
        "--oversampling", type=float, default=20.0,
        help="Oversampling ratio (default: 20.0).",
    )
    p.add_argument(
        "--nyquist-coeff", type=float, default=0.5,
        help="Nyquist coefficient (default: 0.5).",
    )
    p.add_argument(
        "--freq-low", type=float, default=None,
        help="Lower frequency limit (default: Rayleigh resolution).",
    )
    p.add_argument(
        "--freq-high", type=float, default=None,
        help="Upper frequency limit (default: Nyquist frequency).",
    )
    p.add_argument(
        "--freq-step", type=float, default=None,
        help="Frequency step (default: Rayleigh resolution / oversampling).",
    )
    p.add_argument(
        "-o", "--output-dir", type=str, default=None,
        help="Output directory (default: <inputstem>/ next to input).",
    )
    p.add_argument(
        "--no-spectrum", action="store_true",
        help="Do not write intermediate spectrum files.",
    )
    p.add_argument(
        "--no-residuals", action="store_true",
        help="Do not write intermediate residual files.",
    )
    p.add_argument(
        "--phase-diagrams", action="store_true",
        help="Write folded phase diagrams for each detection.",
    )
    p.add_argument(
        "--plot", action="store_true",
        help="Generate summary plots (time series, spectrum, residuals) "
             "and save as PNG. Disabled by default to save time.",
    )
    p.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suppress progress output.",
    )
    p.add_argument(
        "--cpu", "--CPU", dest="force_cpu", action="store_true",
        help="Force CPU-only mode (NumPy), even if a GPU is available.",
    )
    p.add_argument(
        "--config", action="store_true",
        help="Generate a default 'facg.conf' file in the current directory and exit.",
    )
    p.add_argument(
        "--testdata", action="store_true",
        help="Generate benchmark datasets in the current directory for performance comparison and exit.",
    )
    p.add_argument(
        "--files-num", type=int, default=3,
        help="Number of test files to generate with --testdata (default: 3).",
    )
    p.add_argument(
        "--signals-num", type=int, default=5,
        help="Number of signals per test file with --testdata (default: 5).",
    )
    p.add_argument(
        "-V", "--version", action="version",
        version="%(prog)s 0.1.0",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """解析命令行参数并依次处理指定的输入文件。

    参数
    ----
    argv:
        可选的参数字符串列表。未提供时读取 ``sys.argv``，便于 CLI 调用和
        程序化测试。

    返回
    ----
    int
        进程退出状态码；输入文件列表为空等命令行级错误返回非零值，分析完成
        后返回 0。单个文件处理异常会打印错误并继续处理其余文件。
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = _build_parser()

    # Load config file and set as defaults before parsing CLI args
    config_defaults = load_config_for_argparse(CONFIG_FILENAME)
    if config_defaults:
        print(f"✓ Loaded settings from '{CONFIG_FILENAME}'")
    parser.set_defaults(**config_defaults)

    args = parser.parse_args(argv)

    if args.config:
        generate_default_config()
        return 0
    if args.testdata:
        generate_all_test_data(num_files=args.files_num, num_signals=args.signals_num)
        return 0

    # Initialize backend based on CLI flag. This must be done before
    # any other part of the code that uses the backend is called.
    backend.initialize_backend(force_cpu=args.force_cpu)

    # --- File handling: default to local files or expand wildcards ---
    files_to_process = []
    if not args.files:
        # No files given: default to *.dat and *.txt in the current directory.
        cwd = Path.cwd()
        files_to_process.extend(sorted(p.as_posix() for p in cwd.glob("*.dat")))
        files_to_process.extend(sorted(p.as_posix() for p in cwd.glob("*.txt")))
        if not files_to_process:
            print(
                "ERROR: No input files specified and no *.dat or *.txt files "
                "found in the current directory.",
                file=sys.stderr,
            )
            parser.print_help(sys.stderr)
            return 1
    else:
        # Files or patterns given: expand any wildcards.
        for pattern in args.files:
            matched_files = sorted(glob.glob(pattern))
            if not matched_files:
                print(f"WARNING: No file(s) found matching: {pattern}", file=sys.stderr)
            files_to_process.extend(matched_files)

    # Remove duplicates while preserving order
    unique_files = list(dict.fromkeys(files_to_process))

    if not unique_files:
        print("ERROR: No input files to process.", file=sys.stderr)
        return 1

    # Print GPU/CPU backend status on every invocation
    if not args.quiet:
        backend.print_backend_status(file=sys.stdout)

    total_t0 = time.time()

    for filepath in unique_files:
        fpath = Path(filepath)
        if not fpath.is_file():
            print(f"ERROR: file not found: {filepath}", file=sys.stderr)
            continue

        cfg = FACGConfig(
            input_file=str(fpath),
            output_dir=args.output_dir,
            time_col=args.time_col,
            data_col=args.data_col,
            freq_low=args.freq_low,
            freq_high=args.freq_high,
            freq_step=args.freq_step,
            nyquist_coeff=args.nyquist_coeff,
            oversampling=args.oversampling,
            sig_limit=args.sig_limit,
            csig_limit=args.csig_limit,
            max_iter=args.max_iter,
            write_spectrum=not args.no_spectrum,
            write_residuals=not args.no_residuals,
            write_phase_diagram=args.phase_diagrams,
            plot=args.plot,
            quiet=args.quiet,
        )
        try:
            run_analysis(cfg)
        except Exception as exc:
            print(f"ERROR processing '{filepath}': {exc}", file=sys.stderr)

    total_dt = time.time() - total_t0
    if not args.quiet:
        print()
        print(f"===== Total processing time: {total_dt:.3f} s =====")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
