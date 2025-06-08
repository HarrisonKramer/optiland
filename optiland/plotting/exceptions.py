"""Custom exceptions for the Optiland plotting library.

This module defines a set of custom exceptions to provide more specific
error information related to plotting operations, configuration, and themes.
"""


class PlottingError(Exception):
    """Base class for all custom exceptions in the Optiland plotting module.

    Attributes:
      message (str): Human-readable description of the error.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidPlotDataError(PlottingError):
    """Raised when the input data for a plot is invalid or inappropriate.

    This could occur if data is not in the expected format (e.g., not
    array-like for line plots), has mismatched dimensions, or contains
    unsuitable values.
    """

    def __init__(self, message: str = "Invalid data provided for plotting."):
        super().__init__(message)


class ThemeNotFoundError(PlottingError):
    """Raised when a specified plotting theme is not found.

    This exception is used when attempting to apply or retrieve a theme
    that has not been defined in the `themes.THEMES` dictionary.
    """

    def __init__(self, theme_name: str, available_themes: list[str] = None):
        message = f"Theme '{theme_name}' not found."
        if available_themes:
            message += f" Available themes are: {available_themes}"
        super().__init__(message)
        self.theme_name = theme_name
        self.available_themes = available_themes


class ConfigurationError(PlottingError):
    """Raised for general errors related to plotting configuration.

    This can include issues such as invalid configuration keys, values
    of the wrong type for a specific configuration setting, or problems
    encountered while loading or saving configurations.
    """

    def __init__(self, message: str = "A configuration error occurred."):
        super().__init__(message)
