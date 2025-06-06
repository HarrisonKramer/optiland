# optiland_gui/gui_plot_utils.py
import matplotlib
import inspect

def apply_gui_matplotlib_styles():
    """
    Applies a set of Matplotlib rcParams suitable for GUI embedding,
    focusing on smaller font sizes for better integration.
    """
    style_params = {
        'font.size': 8,  # General base font size
        'axes.titlesize': 10, # Fontsize of the axes title (individual subplots)
        'axes.labelsize': 8,  # Fontsize of the x and y labels
        'xtick.labelsize': 7, # Fontsize of the x tick labels
        'ytick.labelsize': 7, # Fontsize of the y tick labels
        'legend.fontsize': 7, # Fontsize of the legend
        'figure.titlesize': 12, # Fontsize of the figure suptitle (if used)
        # Reduce padding around titles and labels if elements are still too large
        'axes.titlepad': 4.0, # Default is 6.0
        'axes.labelpad': 3.0, # Default is 4.0
    }
    matplotlib.rcParams.update(style_params)
    print("Applied GUI-specific Matplotlib styles.")

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
                "default": param_obj.default if param_obj.default is not inspect.Parameter.empty else None,
                "annotation": param_obj.annotation if param_obj.annotation is not inspect.Parameter.empty else None,
                # Add more info if needed, e.g., from type hints or docstrings
            }
    except (ValueError, TypeError) as e:
        print(f"Warning: Could not inspect parameters for {analysis_class.__name__}: {e}")
    return params

