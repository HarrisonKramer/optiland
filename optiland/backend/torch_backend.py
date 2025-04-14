"""
Torch Backend Module

This module provides a backend for numerical operations using PyTorch. It
implements an interface similar to the NumPy backend, enabling seamless switching
between them. It also allows global configuration options including device
selection (CPU or CUDA), gradient calculation control, and float precision.

Kramer Harrison, 2025
"""

import contextlib

try:
    import torch
except ImportError as err:
    torch = None
    raise ImportError("PyTorch is not installed.") from err

# Global variables for backend configuration.
_current_device = "cpu"
_current_precision = torch.float32  # Default precision
_lib = torch  # Alias for torch library


class GradMode:
    """Class to control the gradient calculation globally.

    Attributes:
        requires_grad (bool): Flag indicating whether gradient computation is enabled.
    """

    def __init__(self):
        """Initializes GradMode with gradient calculation disabled."""
        self.requires_grad = False

    def enable(self):
        """Enable gradient calculation."""
        self.requires_grad = True

    def disable(self):
        """Disable gradient calculation."""
        self.requires_grad = False

    @contextlib.contextmanager
    def temporary_enable(self):
        """Context manager to temporarily enable gradient calculation."""
        old_state = self.requires_grad
        self.requires_grad = True
        try:
            yield
        finally:
            self.requires_grad = old_state


# Global instance for controlling gradient mode.
grad_mode = GradMode()


def set_device(device: str) -> None:
    """Set the global device for PyTorch tensors.

    Args:
        device (str): The device to be used, either 'cpu' or 'cuda'.

    Raises:
        ValueError: If the device is not 'cpu' or 'cuda', or if 'cuda' is requested
            but not available.
    """
    global _current_device
    if device not in ["cpu", "cuda"]:
        raise ValueError("Device must be 'cpu' or 'cuda'.")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this system.")
    _current_device = device


def get_device() -> str:
    """Get the current global device setting.

    Returns:
        str: Current device ('cpu' or 'cuda').
    """
    return _current_device


def set_precision(precision: str) -> None:
    """Set the global floating point precision for tensor operations.

    Args:
        precision (str): String representing the precision, either
            'float32' or 'float64'.

    Raises:
        ValueError: If the provided precision is not supported.
    """
    global _current_precision
    if precision == "float32":
        _current_precision = torch.float32
    elif precision == "float64":
        _current_precision = torch.float64
    else:
        raise ValueError("Precision must be 'float32' or 'float64'.")


def get_precision() -> torch.dtype:
    """Get the current floating point precision.

    Returns:
        torch.dtype: The current floating point data type
            (torch.float32 or torch.float64).
    """
    return _current_precision
