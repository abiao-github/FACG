"""
负责生成和读取 ``facg.conf`` 配置文件。

配置文件中填写的值会转换为与命令行参数 ``dest`` 同名的键，供 CLI
设置解析器默认值；命令行显式提供的参数仍可覆盖这些默认设置。
"""

import configparser
from pathlib import Path
from facg.config import FACGConfig

CONFIG_FILENAME = "facg.conf"


def generate_default_config(filepath=CONFIG_FILENAME):
    """在指定路径生成带说明的默认配置文件。

    参数
    ----
    filepath:
        目标文件路径，默认为当前目录中的 ``facg.conf``。写入成功或失败时
        会向标准输出打印提示信息。
    """
    defaults = FACGConfig()

    content = f"""# FACG Configuration File
#
# Settings in this file are used as defaults when you run 'facg'.
# Command-line flags (e.g., --sig-limit 6) will always override these settings.
# To use a built-in default for a numeric/text value, leave it blank (e.g., freq_low = ).
# For boolean flags, use 'True' or 'False'.

[FrequencyGrid]
# Lower frequency limit (cycles/time-unit).
# Default: Frequency step (rayleigh / oversampling).
freq_low = 

# Upper frequency limit.
# Default: Nyquist frequency (0.5 / median_delta_t).
freq_high = 

# Frequency step (spacing).
# Default: rayleigh / oversampling.
freq_step = 

# Nyquist coefficient.
nyquist_coeff = {defaults.nyquist_coeff}

# Oversampling ratio applied to Rayleigh resolution.
oversampling = {defaults.oversampling}


[StoppingCriteria]
# Stop when peak significance drops below this value.
sig_limit = {defaults.sig_limit}

# Stop when cumulative significance rises above this value (0 = disabled).
csig_limit = {defaults.csig_limit}

# Maximum number of prewhitening iterations.
max_iter = {defaults.max_iter}


[IO]
# 0-based column index for the time axis.
time_col = {defaults.time_col}

# 0-based column index for the data axis.
data_col = {defaults.data_col}

# Directory to write output files.
# Default: <input_stem>/ next to the input file.
output_dir = 

# Write residual amplitude spectrum after each iteration (True/False).
write_spectrum = {defaults.write_spectrum}

# Write residual time-series after each iteration (True/False).
write_residuals = {defaults.write_residuals}

# Write folded phase diagrams (True/False).
write_phase_diagram = {defaults.write_phase_diagram}

# Generate summary plots and save as PNG (True/False).
plot = {defaults.plot}
"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Default configuration file '{filepath}' created in the current directory.")
    except IOError as e:
        print(f"ERROR: Could not write configuration file to '{filepath}': {e}")


def load_config_for_argparse(filepath=CONFIG_FILENAME):
    """读取配置文件并转换为 argparse 可使用的默认参数字典。

    参数
    ----
    filepath:
        配置文件路径，默认读取当前目录的 ``facg.conf``。

    返回
    ----
    dict
        键与命令行解析器的参数 ``dest`` 对应。文件不存在或某个配置项为空时，
        对应值不会覆盖解析器的内置默认值；非法值会给出警告并忽略。
    """
    config_path = Path(filepath)
    if not config_path.is_file():
        return {}

    parser = configparser.ConfigParser()
    parser.read(config_path)

    settings = {}

    def set_if_present(section, key, type_func, dest_key=None):
        """读取并转换一个非空配置项，转换失败时保留程序默认值。

        ``type_func`` 可以是常规类型转换器，也可以是 ``bool`` 或
        ``inv_bool`` 这两个标记；后者用于映射命令行中的正向/反向开关。
        """
        if parser.has_option(section, key):
            val_str = parser.get(section, key)
            if val_str is not None and val_str.strip() != '':
                try:
                    if type_func == 'bool':
                        settings[dest_key or key] = parser.getboolean(section, key)
                    elif type_func == 'inv_bool':
                        settings[dest_key or key] = not parser.getboolean(section, key)
                    else:
                        settings[dest_key or key] = type_func(val_str)
                except (ValueError, TypeError):
                    print(f"Warning: Invalid value '{val_str}' for '{key}' in config file. Ignoring.")

    # Map config values to argparse destinations
    set_if_present('FrequencyGrid', 'freq_low', float)
    set_if_present('FrequencyGrid', 'freq_high', float)
    set_if_present('FrequencyGrid', 'freq_step', float)
    set_if_present('FrequencyGrid', 'nyquist_coeff', float)
    set_if_present('FrequencyGrid', 'oversampling', float)
    set_if_present('StoppingCriteria', 'sig_limit', float)
    set_if_present('StoppingCriteria', 'csig_limit', float)
    set_if_present('StoppingCriteria', 'max_iter', int)
    set_if_present('IO', 'time_col', int)
    set_if_present('IO', 'data_col', int)
    set_if_present('IO', 'output_dir', str)
    set_if_present('IO', 'write_phase_diagram', 'bool', 'phase_diagrams')
    set_if_present('IO', 'plot', 'bool')
    set_if_present('IO', 'write_spectrum', 'inv_bool', 'no_spectrum')
    set_if_present('IO', 'write_residuals', 'inv_bool', 'no_residuals')

    return settings