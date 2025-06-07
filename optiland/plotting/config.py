"""Manages plotting configurations for the Optiland library.

This module provides a flexible way to manage plot settings,
allowing users to customize various aspects of their visualizations.
It supports default configurations, runtime modifications, and
will eventually support saving/loading configurations from files.
"""

import copy
import json
import os

from . import exceptions

_DEFAULT_CONFIG = {
    "figure.figsize": (10, 6),
    "font.size_title": 16,
    "font.size_label": 14,
    "font.size_legend": 12,
    "lines.linewidth": 1.5,
    "lines.markersize": 6.0,
    "image.cmap": "viridis",
    "plot.show_on_draw": True,
    "plot.return_fig_ax": False,
}

_current_config = copy.deepcopy(_DEFAULT_CONFIG)


def get_config(key: str):
    """Retrieves a configuration value.

    Args:
      key: A dot-separated string representing the configuration key
          (e.g., 'figure.figsize', 'font.size_title').

    Returns:
      The value associated with the given key.

    Raises:
      exceptions.ConfigurationError: If the key is not found.

    """
    if key not in _current_config:
        raise exceptions.ConfigurationError(f"Configuration key '{key}' not found.")
    return _current_config[key]


def set_config(key: str, value):
    """Sets a configuration value.

    The key must exist in the default configuration options.
    Basic type checking is performed for some known keys.

    Args:
      key: A dot-separated string representing the configuration key
          (e.g., 'figure.figsize', 'font.size_title').
      value: The new value to set for the configuration key.

    Raises:
      exceptions.ConfigurationError: If the key is not a valid configuration
          option (i.e., not present in `_DEFAULT_CONFIG`), or if the
          provided value has a type inconsistent with the default value's type.

    """
    if key not in _DEFAULT_CONFIG:
        raise exceptions.ConfigurationError(
            f"Invalid configuration key '{key}'. It does not exist in the "
            f"default options. Available keys: {list(_DEFAULT_CONFIG.keys())}",
        )

    default_value = _DEFAULT_CONFIG[key]
    if not isinstance(value, type(default_value)):
        # Special handling for tuples like figure.figsize where elements can be int or float
        if (
            key == "figure.figsize"
            and isinstance(default_value, tuple)
            and isinstance(value, tuple)
        ):
            if not (
                len(value) == len(default_value)
                and all(isinstance(v, (int, float)) for v in value)
                and all(isinstance(dv, (int, float)) for dv in default_value)
            ):  # Ensure default is also numbers
                raise exceptions.ConfigurationError(
                    f"Invalid type for configuration key '{key}'. Expected a tuple of "
                    f"{len(default_value)} numbers (int or float), but got {type(value)} with value {value}.",
                )
        # Special handling for float values that can also accept integers
        elif isinstance(default_value, float) and isinstance(value, int):
            value = float(value)  # Convert int to float
        else:
            raise exceptions.ConfigurationError(
                f"Invalid type for configuration key '{key}'. Expected type "
                f"'{type(default_value).__name__}', but got type "
                f"'{type(value).__name__}' for value '{value}'.",
            )

    # Further validation for specific complex types if needed, e.g. tuple contents
    if key == "figure.figsize":
        if not (
            isinstance(value, tuple)
            and len(value) == 2
            and all(isinstance(v, (int, float)) and v > 0 for v in value)
        ):
            raise exceptions.ConfigurationError(
                f"'{key}' must be a tuple of two positive numbers (int or float).",
            )

    _current_config[key] = value


def reset_config():
    """Resets the current configuration to the default values.

    This function restores all configuration options to their original
    state as defined in `_DEFAULT_CONFIG`.
    """
    global _current_config
    _current_config = copy.deepcopy(_DEFAULT_CONFIG)


def save_config(filepath: str):
    """Saves the current configuration to a file.

    Saves the current plotting configuration to a JSON file.

    Args:
      filepath: The path to the file where the configuration should be saved.
                The directory structure will be created if it doesn't exist.

    Raises:
      exceptions.ConfigurationError: If an error occurs during file writing
                                     or JSON serialization.

    """
    try:
        dir_name = os.path.dirname(filepath)
        if dir_name:  # Ensure directory exists if filepath includes one
            os.makedirs(dir_name, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(_current_config, f, indent=2, sort_keys=True)
    except (OSError, json.JSONEncodeError) as e:
        raise exceptions.ConfigurationError(
            f"Failed to save configuration to '{filepath}': {e}",
        ) from e


def load_config(filepath: str):
    """Loads plotting configuration from a JSON file and updates current settings.

    Only valid keys present in `_DEFAULT_CONFIG` will be loaded.
    Values will be type-checked against the defaults; mismatched types will
    cause a ConfigurationError or be skipped with a warning (currently raises error).

    Args:
      filepath: The path to the file from which to load the configuration.

    Raises:
      exceptions.ConfigurationError: If the file is not found, is not valid JSON,
                                     or if an error occurs during reading or
                                     applying the configuration.

    """
    global _current_config
    try:
        with open(filepath) as f:
            loaded_data = json.load(f)
    except FileNotFoundError:
        raise exceptions.ConfigurationError(
            f"Configuration file not found: '{filepath}'",
        )
    except (OSError, json.JSONDecodeError) as e:
        raise exceptions.ConfigurationError(
            f"Failed to load or parse configuration from '{filepath}': {e}",
        ) from e

    for key, value in loaded_data.items():
        if key in _DEFAULT_CONFIG:
            try:
                # Use set_config to ensure validation of the loaded value
                set_config(key, value)
            except exceptions.ConfigurationError as e:
                # Option: print a warning and skip, or re-raise. For now, re-raise.
                # print(f"Warning: Skipping invalid configuration item from file: {key}={value}. Reason: {e}")
                raise exceptions.ConfigurationError(
                    f"Invalid configuration for key '{key}' in '{filepath}': {e}",
                ) from e
        # else:
        # Optionally, log or print a warning for unknown keys from the config file
        # print(f"Warning: Unknown configuration key '{key}' found in '{filepath}'. Ignoring.")
