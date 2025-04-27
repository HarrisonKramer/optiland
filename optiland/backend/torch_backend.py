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
    # ensure x is a torch tensor (wrap Python scalars)
    x_tensor = array(x)
    # unwrap fill_value if it's a tensor
    if isinstance(fill_value, torch.Tensor):
        fill_value = fill_value.item()
    return torch.full_like(
        x_tensor,
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
    """
    Vectorized nearest‐neighbor lookup in PyTorch, working for Hx,Hy of any shape
    without breaking autograd.
    """
    # Make sure we actually got values, not None
    if Hx is None or Hy is None:
        raise ValueError(
            f"nearest_nd_interpolator requires both Hx and Hy, got Hx={Hx!r}, Hy={Hy!r}"
        )

    # lift Python scalars or numpy arrays into torch.Tensors on the right device/dtype
    Hx = array(Hx)
    Hy = array(Hy)
    # Make sure Hx and Hy have the same shape – broadcast if necessary
    if Hx.shape != Hy.shape:  # <‑‑ NEW
        Hx, Hy = torch.broadcast_tensors(Hx, Hy)  # <‑‑ NEW
    # 1) pack queries into a flat (K,2) tensor
    q = torch.stack([Hx, Hy], dim=-1)  # (...,2)
    orig_shape = q.shape[:-1]
    q_flat = q.reshape(-1, 2)  # (K,2)

    # 2) compute all pairwise distances against the M field‐points
    pts = points.to(dtype=q.dtype, device=q.device)  # ensure same dtype/device
    dists = torch.cdist(q_flat, pts)  # (K, M)

    # 3) find nearest index for each query
    idx = dists.argmin(dim=1)  # (K,)

    # 4) gather values at those indices
    #    flatten values to (M, P) so we can index; P is remaining dims
    vals_flat = values.view(points.shape[0], -1)  # (M, P)
    out_flat = vals_flat[idx]  # (K, P)

    # 5) reshape back to original query shape + any extra dims
    out = out_flat.view(*orig_shape, *out_flat.shape[1:])
    # if values was one‑dimensional, squeeze that last axis
    if out.shape[-1] == 1:
        out = out.squeeze(-1)
    return out


def all(x):
    """Backend‐agnostic “all”: accept Python bool, NumPy arrays, or Tensors."""
    # Python bool → leave as is
    if isinstance(x, bool):
        return x
    # Anything else → lift into a torch.Tensor on the right device/dtype
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=_current_precision, device=_current_device)
    # Now we can safely call torch.all
    return torch.all(x).item()


def radians(x):
    """Convert degrees→radians, accepting both Python scalars and tensors."""
    # if it's not already a torch.Tensor, cast it into one
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=_current_precision, device=_current_device)
    return torch.deg2rad(x)


def deg2rad(x):
    """Convert degrees to radians, accepting Python scalars or tensors."""
    return radians(x)


def newaxis():
    return None


def cast(x):
    """Ensure x is a torch.Tensor on the right device/dtype."""
    if not isinstance(x, torch.Tensor):
        # lift Python scalars or numpy scalars into a tensor
        return torch.tensor(x, device=get_device(), dtype=get_precision())
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
    # cast E to complex so matmul(p:complex128, E:complex128) works
    try:
        E_c = E.to(torch.complex128)
    except Exception:
        E_c = torch.tensor(E, device=get_device(), dtype=torch.complex128)
    return torch.squeeze(torch.matmul(p, E_c.unsqueeze(2)), axis=2)


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


def sqrt(x):
    return _lib.sqrt(array(x))


def sin(x):
    return _lib.sin(array(x))


def cos(x):
    return _lib.cos(array(x))


def exp(x):
    return _lib.exp(array(x))


def max(x):
    """Backend‐agnostic max: returns a Python float when used on a torch.Tensor."""
    if isinstance(x, torch.Tensor):
        # detach → CPU → reduce → item()
        return x.detach().cpu().max().item()
    # fall back to numpy or builtin for lists/ndarrays
    return np.max(x)


def min(x):
    """Same for min."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().min().item()
    return np.min(x)


def mean(x, axis=None, keepdims=False):
    """
    Backend-agnostic mean: accepts Python scalars, NumPy arrays, or Tensors.
    Handles NaN values by ignoring them (similar to np.nanmean).

    Args:
        x: Input tensor
        axis: Axis along which to compute mean (None for overall mean)
        keepdims: Whether to keep the reduced dimensions

    Returns:
        Mean value as a tensor
    """
    # If not a tensor, convert to tensor
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=get_precision(), device=get_device())

    # Handle NaN values by replacing them with zeros and creating a mask
    mask = ~torch.isnan(x)
    # Count non-NaN elements
    count = mask.sum(dim=axis, keepdim=keepdims).to(x.dtype)
    # Sum non-NaN elements (replace NaNs with 0)
    x_masked = torch.where(mask, x, torch.tensor(0.0, dtype=x.dtype, device=x.device))
    sum_val = x_masked.sum(dim=axis, keepdim=keepdims)

    # Compute mean (avoiding division by zero)
    result = torch.where(
        count > 0,
        sum_val / count,
        torch.tensor(float("nan"), dtype=x.dtype, device=x.device),
    )

    return result


def vectorize(pyfunc):
    """
    simple elementwise mapper for Torch.
    Takes a Python scalar→scalar function and returns a new function
    that applies it over every element of a 1D tensor, preserving the shape.
    """

    def wrapped(x):
        # flatten to 1D
        flat = x.reshape(-1)
        # call your python function on each element (xi is a 0‑dim tensor)
        mapped = [pyfunc(xi) for xi in flat]
        # stack back into a tensor
        out = torch.stack(
            [
                m
                if isinstance(m, torch.Tensor)
                else torch.tensor(m, dtype=get_precision(), device=get_device())
                for m in mapped
            ],
            dim=0,
        )
        return out.view(x.shape)

    return wrapped


def tile(x, dims):
    if isinstance(dims, int):
        return torch.tile(x, dims=(dims,))
    return torch.tile(x, dims)


def isinf(x):
    """
    Torch‐backend‐agnostic “is infinity” test:
    accepts Python scalars, NumPy arrays, or Tensors.
    """
    import torch

    # lift non‐Tensor into a Tensor on the right device/dtype
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=get_precision(), device=get_device())
    return torch.isinf(x)


def array_equal(a, b):
    """
    Backend‑agnostic equivalent of numpy.array_equal for Torch.

    Returns
    -------
    bool
        True iff `a` and `b` have the same shape and every element is equal.
    """
    a_t = cast(a)
    b_t = cast(b)
    if a_t.shape != b_t.shape:
        return False
    return torch.eq(a_t, b_t).all().item()
