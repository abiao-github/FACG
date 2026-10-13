"""
FACG 的时间序列输入与结果文件输出。

读取端支持文本、DAT、CSV、Excel 和 FITS 表格，并根据列号提取时间与观测值；
写入端负责统一结果表、频谱、残差以及折叠相位数据的文件格式。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_timeseries(
    filepath: str | Path,
    time_col: int = 0,
    data_col: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """读取时间序列文件，并按时间从小到大返回两列数据。

    Parameters
    ----------
    filepath : path-like
        待读取文件路径。根据扩展名选择文本、CSV、Excel 或 FITS 表格读取方式。
    time_col, data_col : int
        时间列和观测值列的零起始索引，两列必须包含可转换为浮点数的数据。

    Returns
    -------
    t, x : 1-D float64 arrays
        已排序的时间戳和观测值数组。若文件不存在、格式无法解析或指定列缺失，
        会抛出相应异常。
    """
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"Input file not found: {filepath}")

    ext = filepath.suffix.lower()
    max_col = max(time_col, data_col)

    if ext == '.csv':
        import pandas as pd
        df = pd.read_csv(filepath, comment='#', header=None)
        if df.shape[1] <= max_col:
            raise ValueError(f"File has {df.shape[1]} columns but column {max_col} requested.")
        t = df.iloc[:, time_col].to_numpy().astype(np.float64)
        x = df.iloc[:, data_col].to_numpy().astype(np.float64)
        
    elif ext in ['.xls', '.xlsx']:
        import pandas as pd
        df = pd.read_excel(filepath, header=None)
        if df.shape[1] <= max_col:
            raise ValueError(f"File has {df.shape[1]} columns but column {max_col} requested.")
        t = df.iloc[:, time_col].to_numpy().astype(np.float64)
        x = df.iloc[:, data_col].to_numpy().astype(np.float64)
        
    elif ext in ['.fits', '.fit']:
        from astropy.io import fits
        with fits.open(filepath) as hdul:
            table_hdu = None
            for hdu in hdul:
                if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                    table_hdu = hdu
                    break
            if table_hdu is None:
                raise ValueError(f"No table found in FITS file {filepath.name}")
            cols = table_hdu.data.columns.names
            if len(cols) <= max_col:
                raise ValueError(f"FITS table has {len(cols)} columns but column {max_col} requested.")
            t = table_hdu.data[cols[time_col]].astype(np.float64)
            x = table_hdu.data[cols[data_col]].astype(np.float64)
            
    else:
        try:
            data = np.loadtxt(filepath)
        except ValueError as exc:
            raise ValueError(f"Could not parse numeric data from '{filepath.name}'. Is it a valid time-series file? Details: {exc}")
        if data.ndim == 1:
            raise ValueError(f"Expected at least 2 columns, got 1 in {filepath.name}")
        if data.shape[1] <= max_col:
            raise ValueError(f"File has {data.shape[1]} columns but column {max_col} requested.")
        t = data[:, time_col].astype(np.float64)
        x = data[:, data_col].astype(np.float64)

    # Sort by time (SigSpec does this too)
    order = np.argsort(t)
    return t[order], x[order]


# ---------------------------------------------------------------------------
# Writing helpers
# ---------------------------------------------------------------------------

def _header(label: str, cfg_summary: str = "") -> str:
    """构造输出数据文件使用的统一 FACG 注释头。

    参数
    ----
    label:
        当前输出内容的名称，用于标识 FACG 文件类型。
    cfg_summary:
        可选的配置摘要；为空时只写入名称行。

    返回
    ----
    str
        以换行符结尾的注释文本块。
    """
    lines = [
        f"# FACG — {label}",
        f"# {cfg_summary}" if cfg_summary else "",
    ]
    return "\n".join(ln for ln in lines if ln) + "\n"


def write_result(
    outdir: Path,
    results: list[dict],
    stem: str = "result",
) -> Path:
    """将每轮检测到的频率参数写入主结果表。

    输出列依次为频率、显著性、振幅、相位、当前残差标准差和累积显著性。
    文件保存为 ``<stem>.dat``，返回实际输出路径。
    """
    outpath = outdir / f"{stem}.dat"
    with open(outpath, "w") as f:
        f.write("# freq                 sig                  amp                  "
                "phase                rms                  csig\n")
        for r in results:
            f.write(
                f"{r['freq']:20.12f} {r['sig']:20.12f} "
                f"{r['amp']:20.12f} {r['phase']:20.12f} "
                f"{r['rms']:20.12f} {r['csig']:20.12f}\n"
            )
    return outpath


def write_spectrum(
    outdir: Path,
    freqs: np.ndarray,
    sig_spectrum: np.ndarray,
    amp_spectrum: np.ndarray,
    phase_spectrum: np.ndarray,
    iteration: int,
    stem: str = "spectrum",
) -> Path:
    """写出指定迭代轮次的显著性、振幅与相位频谱。

    迭代编号为负数时文件名使用 ``_final``，否则使用四位补零的轮次编号。
    输出列依次为频率、显著性、振幅和相位。
    """
    if iteration < 0:
        outpath = outdir / f"{stem}_final.dat"
    else:
        outpath = outdir / f"{stem}_{iteration:04d}.dat"
    data = np.column_stack([freqs, sig_spectrum, amp_spectrum, phase_spectrum])
    np.savetxt(outpath, data, fmt="%24.16e",
               header="freq                    sig                     "
                      "amplitude               phase")
    return outpath


def write_residuals(
    outdir: Path,
    t: np.ndarray,
    residuals: np.ndarray,
    iteration: int,
    stem: str = "residuals",
) -> Path:
    """将时间戳及对应残差写入文本文件。

    迭代编号为负数时写入最终残差文件，否则在文件名中标记当前轮次；
    返回写出文件的路径。
    """
    if iteration < 0:
        outpath = outdir / f"{stem}_final.dat"
    else:
        outpath = outdir / f"{stem}_{iteration:04d}.dat"
    data = np.column_stack([t, residuals])
    np.savetxt(outpath, data, fmt="%24.16e",
               header="time                    residual")
    return outpath


def write_phase_diagram(
    outdir: Path,
    phase: np.ndarray,
    data: np.ndarray,
    freq: float,
    iteration: int,
) -> Path:
    """写出单个频率对应的折叠相位与数据值。

    输出文件名包含迭代轮次和频率，文件两列分别为相位和传入的数据值；
    返回写出文件的路径。
    """
    outpath = outdir / f"phase_{iteration:04d}_f{freq:.6f}.dat"
    out = np.column_stack([phase, data])
    np.savetxt(outpath, out, fmt="%24.16e",
               header=f"phase                   data   (fold freq={freq:.12f})")
    return outpath
