"""
Handles reading and writing of the 'facg.conf' configuration file.
"""

import configparser
from pathlib import Path
from facg.config import FACGConfig

CONFIG_FILENAME = "facg.conf"


def generate_default_config(filepath=CONFIG_FILENAME):
    """Generates a default configuration file with comments."""
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
    """
    Loads configuration from a file and returns a dictionary with keys
    matching argparse 'dest' names.
    """
    config_path = Path(filepath)
    if not config_path.is_file():
        return {}

    parser = configparser.ConfigParser()
    parser.read(config_path)

    settings = {}

    def set_if_present(section, key, type_func, dest_key=None):
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