"""
Torch Backend Module

This module provides a backend for numerical operations using PyTorch. It
implements an interface similar to the NumPy backend, enabling seamless switching
between them. It also allows global configuration options including device
selection (CPU or CUDA), gradient calculation control, and float precision.

Kramer Harrison, 2025
"""

import contextlib

import numpy as np

try:
    import torch
    import torch.nn.functional as F
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


def array(x):
    """Create an array/tensor."""
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(
        x,
        device=get_device(),
        dtype=get_precision(),
        requires_grad=grad_mode.requires_grad,
    )


def is_array_like(x):
    """Check if the input is array-like."""
    return isinstance(x, (torch.Tensor, np.ndarray, list, tuple))


def zeros(shape):
    """Create an array/tensor filled with zeros."""
    return torch.zeros(
        shape,
        device=get_device(),
        dtype=get_precision(),
        requires_grad=grad_mode.requires_grad,
    )


def zeros_like(x):
    """Create an array/tensor filled with zeros with the same shape as x."""
    return torch.zeros_like(
        array(x),
        device=get_device(),
        dtype=get_precision(),
        requires_grad=grad_mode.requires_grad,
    )


def ones(shape):
    """Create an array/tensor filled with ones."""
    return torch.ones(
        shape,
        device=get_device(),
        dtype=get_precision(),
        requires_grad=grad_mode.requires_grad,
    )


def ones_like(x):
    """Create an array/tensor filled with ones with the same shape as x."""
    return torch.ones_like(
        array(x),
        device=get_device(),
        dtype=get_precision(),
        requires_grad=grad_mode.requires_grad,
    )


def full(shape, fill_value):
    """Create an array/tensor filled with fill_value."""
    return torch.full(
        shape,
        fill_value,
        device=get_device(),
        dtype=get_precision(),
        requires_grad=grad_mode.requires_grad,
    )


def full_like(x, fill_value):
    """
    Create an array/tensor filled with fill_value with the same shape as x.
    """
    if isinstance(fill_value, torch.Tensor):
        fill_value = fill_value.item()
    return torch.full_like(
        x,
        fill_value,
        device=get_device(),
        dtype=get_precision(),
        requires_grad=grad_mode.requires_grad,
    )


def linspace(start, stop, num=50):
    """Create an array/tensor of evenly spaced values."""
    return torch.linspace(
        start,
        stop,
        num,
        device=get_device(),
        dtype=get_precision(),
        requires_grad=grad_mode.requires_grad,
    )


def from_matrix(matrix):
    raise NotImplementedError(
        "from_matrix is not implemented for torch. Please use the NumPy backend."
    )


def from_euler(euler):
    raise NotImplementedError("from_euler is not implemented for torch.")


def copy(x):
    return x.clone()


def polyfit(x, y, degree):
    X = torch.stack([x**i for i in range(degree + 1)], dim=1)
    coeffs, _ = torch.lstsq(y.unsqueeze(1), X)
    return coeffs[: degree + 1].squeeze()


def polyval(coeffs, x):
    return sum(c * x**i for i, c in enumerate(reversed(coeffs)))


def load(filename):
    data = np.load(filename)
    return array(data)


def hstack(arrays):
    return torch.cat(arrays, dim=1)


def vstack(arrays):
    return torch.cat(arrays, dim=0)


def interp(x, xp, fp):
    """
    Mimics numpy.interp for 1D linear interpolation in PyTorch.

    Args:
        x (torch.Tensor): Points to interpolate.
        xp (torch.Tensor): Known x-coordinates.
        fp (torch.Tensor): Known y-coordinates.

    Returns:
        torch.Tensor: Interpolated values.
    """
    # Ensure tensors are float for arithmetic operations
    x = torch.as_tensor(x, dtype=get_precision(), device=get_device())
    xp = torch.as_tensor(xp, dtype=get_precision(), device=get_device())
    fp = torch.as_tensor(fp, dtype=get_precision(), device=get_device())

    # Sort xp and fp based on xp
    sorted_indices = torch.argsort(xp)
    xp = xp[sorted_indices]
    fp = fp[sorted_indices]

    # Clip x to be within the range of xp
    x_clipped = torch.clip(x, xp[0], xp[-1])

    # Find indices where each x would be inserted to maintain order
    indices = torch.searchsorted(xp, x_clipped, right=True)
    indices = torch.clamp(indices, 1, len(xp) - 1)

    # Get the x-coordinates and y-coordinates for interpolation
    x0 = xp[indices - 1]
    x1 = xp[indices]
    y0 = fp[indices - 1]
    y1 = fp[indices]

    # Linear interpolation formula
    interpolated = y0 + (y1 - y0) * (x_clipped - x0) / (x1 - x0)
    return interpolated


def atleast_1d(x):
    x = torch.as_tensor(x, dtype=get_precision())
    if x.ndim == 0:  # Scalar -> (1,)
        return x.unsqueeze(0)
    return x  # Already 1D or higher


def atleast_2d(x):
    x = torch.as_tensor(x, dtype=get_precision())
    if x.ndim == 0:  # Scalar -> (1, 1)
        return x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 1:  # 1D array -> (1, N)
        return x.unsqueeze(0)
    return x  # Already 2D or higher


def as_array_1d(data):
    """Force conversion to a 1D tensor."""
    if isinstance(data, (int, float)):
        return array([data])
    elif isinstance(data, (list, tuple)):
        return array(data)
    elif is_array_like(data):
        return data.reshape(-1)
    else:
        raise ValueError(
            "Unsupported input type: expected scalar, list, tuple, or array-like."
        )


def size(x):
    return torch.numel(x)


def default_rng(seed=None):
    if seed is None:
        seed = torch.initial_seed()
    return torch.Generator(device=get_device()).manual_seed(seed)


def random_uniform(low=0.0, high=1.0, size=None, generator=None):
    if size is None:
        size = 1
    if generator is None:
        return torch.empty(size, device=get_device()).uniform_(low, high)
    else:
        return torch.empty(size, device=get_device()).uniform_(
            low, high, generator=generator
        )


def random_normal(loc=0.0, scale=1.0, size=None, generator=None):
    if size is None:
        size = 1
    if generator is None:
        return torch.randn(size, device=get_device()) * scale + loc
    else:
        return torch.randn(size, device=get_device(), generator=generator) * scale + loc


def repeat(x, repeats):
    return torch.repeat_interleave(x, repeats)


def flip(x):
    return torch.flip(x, dims=(0,))


def meshgrid(*arrays):
    return torch.meshgrid(*arrays, indexing="ij")


def matrix_vector_multiply_and_squeeze(p, E):
    return torch.matmul(p, E.unsqueeze(2)).squeeze(2)


def roll(x, shift, axis=None):
    return torch.roll(x, shift, dims=axis)


def reshape(x, shape):
    return x.view(shape)


def nearest_nd_interpolator(points, values, Hx, Hy):
    """Manual nearest neighbor in PyTorch"""
    query = torch.tensor([Hx, Hy], dtype=points.dtype, device=points.device)
    diffs = points - query  # shape: (N, 2)
    dists = torch.sum(diffs**2, dim=-1)
    idx = torch.argmin(dists)
    return values[idx]


def all(x):
    if isinstance(x, bool):
        return x
    return torch.all(x).item()


def radians(x):
    return torch.deg2rad(x)


def newaxis():
    return None


def cast(x):
    return x.to(dtype=get_precision(), device=get_device())


def stack(xs, axis=0):
    return torch.stack([cast(x) for x in xs], dim=axis)


def broadcast_to(x, shape):
    return x.expand(shape)


def cross(a, b):
    return torch.linalg.cross(a, b)


def unsqueeze_last(x):
    return x.unsqueeze(-1)


def eye(x):
    return torch.eye(x, device=get_device(), dtype=get_precision())


def mult_p_E(p, E):
    # Used only for electric field multiplication in polarized_rays.py
    p = p.to(torch.complex128)
    return torch.squeeze(torch.matmul(p, E.unsqueeze(2)), axis=2)


def to_complex(x):
    return x.to(torch.complex128)


def matmul(a, b):
    dtype = torch.promote_types(a.dtype, b.dtype)
    return torch.matmul(a.to(dtype), b.to(dtype))


def batched_chain_matmul3(a, b, c):
    dtype = torch.promote_types(torch.promote_types(a.dtype, b.dtype), c.dtype)
    return torch.matmul(torch.matmul(a.to(dtype), b.to(dtype)), c.to(dtype))


def isscalar(x):
    return torch.is_tensor(x) and x.dim() == 0


def pad(tensor, pad_width, mode="constant", constant_values=0):
    """
    Mimics numpy.pad for 2D tensors in PyTorch with limited support.

    Args:
        tensor (Tensor): 2D PyTorch tensor to pad.
        pad_width (tuple of tuple): ((pad_top, pad_bottom), (pad_left, pad_right))
        mode (str): Only 'constant' mode is supported.
        constant_values (int or float): Fill value for constant padding.

    Returns:
        Padded 2D tensor.
    """
    if mode != "constant":
        raise NotImplementedError("Only mode='constant' is supported in torch backend")

    if not isinstance(pad_width, (tuple, list)) or len(pad_width) != 2:
        raise ValueError("pad_width must be a tuple of two tuples for 2D input")

    (pad_top, pad_bottom), (pad_left, pad_right) = pad_width

    # PyTorch expects (pad_left, pad_right, pad_top, pad_bottom)
    padding = (pad_left, pad_right, pad_top, pad_bottom)

    return F.pad(tensor, padding, mode="constant", value=constant_values)
