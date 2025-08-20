"""Provides utility functions for GUI plotting and analysis parameter inspection.

This module contains helper functions for configuring Matplotlib styles for GUI
embedding and for dynamically inspecting the parameters of analysis classes to
auto-generate settings interfaces.

Author: Manuel Fragata Mendes, 2025
"""

from __future__ import annotations

import inspect

import matplotlib


def apply_gui_matplotlib_styles(theme="light"):
    """Applies Matplotlib rcParams for GUI embedding with theme awareness.

    This function configures Matplotlib's runtime configuration parameters (`rcParams`)
    to ensure that plots embedded in the GUI have a consistent and readable style.
    It adjusts colors and sizes for either a "light" or "dark" theme.

    Args:
        theme (str): The theme to apply, either "light" or "dark". Defaults to "light".
    """
    base_style = {
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 12,
        "axes.titlepad": 4.0,
        "axes.labelpad": 3.0,
        "figure.autolayout": True,
    }

    if theme == "dark":
        dark_style = {
            "figure.facecolor": "#2a2a2a",
            "axes.facecolor": "#2a2a2a",
            "axes.edgecolor": "#bbbbbb",
            "axes.labelcolor": "#bbbbbb",
            "xtick.color": "#bbbbbb",
            "ytick.color": "#bbbbbb",
            "grid.color": "#555555",
            "text.color": "#bbbbbb",
            "legend.facecolor": "#3c3c3c",
            "legend.edgecolor": "#555555",
        }
        base_style.update(dark_style)
    else:
        matplotlib.rcdefaults()

    matplotlib.rcParams.update(base_style)


def get_analysis_parameters(analysis_class):
    """Inspects an analysis class's __init__ to find its configurable parameters.

    This function uses Python's `inspect` module to determine the parameters of
    the constructor of a given analysis class. It filters out standard parameters
    like 'self' and 'optic' to return a dictionary of parameters that can be
    configured by the user in the GUI.

    Args:
        analysis_class: The analysis class to inspect.

    Returns:
        dict: A dictionary where keys are parameter names and values are dicts
              containing the parameter's 'default' value and 'annotation'.
              Returns an empty dictionary if inspection fails.
    """
    if not analysis_class:
        return {}

    params = {}
    try:
        sig = inspect.signature(analysis_class.__init__)
        for param_name, param_obj in sig.parameters.items():
            if param_name in ["self", "optic", "optical_system"]:
                continue
            params[param_name] = {
                "default": (
                    param_obj.default
                    if param_obj.default is not inspect.Parameter.empty
                    else None
                ),
                "annotation": (
                    param_obj.annotation
                    if param_obj.annotation is not inspect.Parameter.empty
                    else None
                ),
            }
    except (ValueError, TypeError) as e:
        print(
            f"Warning: Could not inspect parameters for {analysis_class.__name__}: {e}"
        )
    return params


def handle_matplotlib_scroll_zoom(event):
    """Handles mouse wheel scrolling for zooming on a Matplotlib axes."""
    if not event.inaxes:
        return

    ax = event.inaxes
    scale_factor = 1.1 if event.step < 0 else 1 / 1.1

    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()

    xdata, ydata = event.xdata, event.ydata

    new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
    new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

    rel_x = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
    rel_y = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

    ax.set_xlim([xdata - new_width * (1 - rel_x), xdata + new_width * rel_x])
    ax.set_ylim([ydata - new_height * (1 - rel_y), ydata + new_height * rel_y])
    ax.figure.canvas.draw_idle()
