import inspect

import matplotlib


def apply_gui_matplotlib_styles(theme="light"):
    """
    Applies Matplotlib rcParams for GUI embedding, with theme awareness.
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
    print(f"Applied {theme}-specific Matplotlib styles.")


def get_analysis_parameters(analysis_class):
    """
    Inspects the __init__ method of an analysis class and returns
    a list of its parameters, excluding 'self', 'optic', 'optical_system'.
    This can be used to dynamically generate settings UI.
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
                "default": param_obj.default
                if param_obj.default is not inspect.Parameter.empty
                else None,
                "annotation": param_obj.annotation
                if param_obj.annotation is not inspect.Parameter.empty
                else None,
            }
    except (ValueError, TypeError) as e:
        print(
            f"Warning: Could not inspect parameters for {analysis_class.__name__}: {e}"
        )
    return params
