"""
Torch Backend Module

Functionalities are grouped into the following categories:
- Configuration
- Tensor Creation
- Array Utilities
- Random Number Generation
- Shape and Indexing Operations
- Mathematical Operations
- Linear Algebra
- Interpolation
- Polynomial Operations
- Padding
- Vectorization
- Miscellaneous Utilities

This module provides a backend for numerical operations using PyTorch.

Kramer Harrison, 2025
"""

import contextlib

import numpy as np
import torch
import torch.nn.functional as F

_lib = torch  # Alias for torch library


# --------------------------
# Configuration
# --------------------------
class GradMode:
    """Control global gradient computation."""

    def __init__(self):
        self.requires_grad = False

    def enable(self):
        self.requires_grad = True

    def disable(self):
        self.requires_grad = False

    @contextlib.contextmanager
    def temporary_enable(self):
        old = self.requires_grad
        self.requires_grad = True
        try:
            yield
        finally:
            self.requires_grad = old


class _Config:
    def __init__(self):
        self.device = "cpu"
        self.precision = torch.float32
        self.grad_mode = GradMode()

    def set_device(self, device: str) -> None:
        if device not in ("cpu", "cuda"):
            raise ValueError("Device must be 'cpu' or 'cuda'.")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available.")
        self.device = device

    def get_device(self) -> str:
        return self.device

    def set_precision(self, precision: str) -> None:
        if precision == "float32":
            self.precision = torch.float32
        elif precision == "float64":
            self.precision = torch.float64
        else:
            raise ValueError("Precision must be 'float32' or 'float64'.")

    def get_precision(self) -> torch.dtype:
        return self.precision


_config = _Config()


# Public API for configuration
def set_device(device: str) -> None:
    return _config.set_device(device)


def get_device() -> str:
    return _config.get_device()


def set_precision(precision: str) -> None:
    return _config.set_precision(precision)


def get_precision() -> torch.dtype:
    return _config.get_precision()


# Global gradient control
grad_mode = _config.grad_mode


# --------------------------
# Array Creation
# --------------------------
def array(x):
    """Create a tensor with current device, precision, and grad settings."""
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(
        x,
        device=get_device(),
        dtype=get_precision(),
        requires_grad=grad_mode.requires_grad,
    )


def zeros(shape):
    return torch.zeros(
        shape,
        device=get_device(),
        dtype=get_precision(),
        requires_grad=grad_mode.requires_grad,
    )


def ones(shape):
    return torch.ones(
        shape,
        device=get_device(),
        dtype=get_precision(),
        requires_grad=grad_mode.requires_grad,
    )


def full(shape, fill_value):
    return torch.full(
        shape,
        fill_value,
        device=get_device(),
        dtype=get_precision(),
        requires_grad=grad_mode.requires_grad,
    )


def linspace(start, stop, num=50):
    return torch.linspace(
        start,
        stop,
        num,
        device=get_device(),
        dtype=get_precision(),
        requires_grad=grad_mode.requires_grad,
    )


def arange(*args, step=1):
    if len(args) == 1:
        start = 0
        end = args[0]
    elif len(args) == 2:
        start, end = args

    if isinstance(start, torch.Tensor):
        start = start.item()
    if isinstance(end, torch.Tensor):
        end = end.item()
    if isinstance(step, torch.Tensor):
        step = step.item()

    return torch.arange(
        start,
        end,
        step,
        device=get_device(),
        dtype=get_precision(),
        requires_grad=grad_mode.requires_grad,
    )


def zeros_like(x):
    return torch.zeros_like(
        array(x),
        device=get_device(),
        dtype=get_precision(),
        requires_grad=grad_mode.requires_grad,
    )


def ones_like(x):
    return torch.ones_like(
        array(x),
        device=get_device(),
        dtype=get_precision(),
        requires_grad=grad_mode.requires_grad,
    )


def full_like(x, fill_value):
    x_t = array(x)
    val = fill_value.item() if isinstance(fill_value, torch.Tensor) else fill_value
    return torch.full_like(
        x_t,
        val,
        device=get_device(),
        dtype=get_precision(),
        requires_grad=grad_mode.requires_grad,
    )


def load(filename: str):
    data = np.load(filename)
    return array(data)


# --------------------------
# Array Utilities
# --------------------------
def cast(x):
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, device=get_device(), dtype=get_precision())
    return x.to(device=get_device(), dtype=get_precision())


def copy(x):
    return x.clone()


def is_array_like(x):
    return isinstance(x, (torch.Tensor, np.ndarray, list, tuple))


def size(x):
    return torch.numel(x)


def newaxis():
    return None


def array_equal(a, b):
    return torch.equal(a, b)


# --------------------------
# Shape and Indexing
# --------------------------
def reshape(x, shape):
    return x.view(shape)


def stack(xs, axis=0):
    return torch.stack([cast(x) for x in xs], dim=axis)


def broadcast_to(x, shape):
    return x.expand(shape)


def repeat(x, repeats):
    return torch.repeat_interleave(x, repeats)


def flip(x):
    return torch.flip(x, dims=(0,))


def meshgrid(*arrays):
    return torch.meshgrid(*arrays, indexing="xy")


def roll(x, shift, axis=None):
    return torch.roll(x, shifts=shift, dims=axis)


def unsqueeze_last(x):
    return x.unsqueeze(-1)


def tile(x, dims):
    return torch.tile(x, dims if isinstance(dims, (tuple, list)) else (dims,))


def isscalar(x):
    return torch.is_tensor(x) and x.dim() == 0


# --------------------------
# Random Number Generation
# --------------------------
def default_rng(seed=None):
    if seed is None:
        seed = torch.initial_seed()
    return torch.Generator(device=get_device()).manual_seed(seed)


def random_uniform(low=0.0, high=1.0, size=None, generator=None):
    size = size or 1
    gen_args = {"generator": generator} if generator else {}
    return torch.empty(size, device=get_device(), dtype=get_precision()).uniform_(
        low, high, **gen_args
    )


def random_normal(loc=0.0, scale=1.0, size=None, generator=None):
    size = size or 1
    gen_args = {"generator": generator} if generator else {}
    return (
        torch.randn(size, device=get_device(), dtype=get_precision(), **gen_args)
        * scale
        + loc
    )


# --------------------------
# Mathematical Operations
# --------------------------
def sqrt(x):
    return torch.sqrt(array(x))


def sin(x):
    return torch.sin(array(x))


def cos(x):
    return torch.cos(array(x))


def exp(x):
    return torch.exp(array(x))


def log2(x):
    return torch.log2(array(x))


def abs(x):
    return torch.abs(array(x))


def radians(x):
    t = array(x)
    return torch.deg2rad(t)


def degrees(x):
    t = array(x)
    return torch.rad2deg(t)


deg2rad = radians
rad2deg = degrees


def max(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().max().item()
    return np.max(x)


def min(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().min().item()
    return np.min(x)


def nanmax(input_tensor, axis=None, keepdim=False):
    nan_mask = torch.isnan(input_tensor)
    replaced = input_tensor.clone()
    replaced[nan_mask] = float("-inf")
    if axis is not None:
        result, _ = torch.max(replaced, dim=axis, keepdim=keepdim)
    else:
        result = torch.max(replaced)
    return result


def mean(x, axis=None, keepdims=False):
    x = array(x)
    mask = ~torch.isnan(x)
    cnt = mask.sum(dim=axis, keepdim=keepdims).to(x.dtype)
    x0 = torch.where(mask, x, torch.tensor(0.0, dtype=x.dtype, device=x.device))
    s = x0.sum(dim=axis, keepdim=keepdims)
    return torch.where(
        cnt > 0, s / cnt, torch.tensor(float("nan"), dtype=x.dtype, device=x.device)
    )


def all(x):
    if isinstance(x, bool):
        return x
    t = torch.as_tensor(x, dtype=_config.precision, device=_config.device)
    return torch.all(t).item()


def factorial(n):
    return torch.lgamma(array(n + 1)).exp()


# --------------------------
# Linear Algebra
# --------------------------
def matmul(a, b):
    dtype = torch.promote_types(a.dtype, b.dtype)
    return torch.matmul(a.to(dtype), b.to(dtype))


def batched_chain_matmul3(a, b, c):
    dtype = torch.promote_types(torch.promote_types(a.dtype, b.dtype), c.dtype)
    return torch.matmul(torch.matmul(a.to(dtype), b.to(dtype)), c.to(dtype))


def cross(a, b):
    return torch.linalg.cross(a, b)


def matrix_vector_multiply_and_squeeze(p, E):
    return torch.matmul(p, E.unsqueeze(2)).squeeze(2)


def to_complex(x):
    return x.to(torch.complex128)


def mult_p_E(p, E):
    p_c = p.to(torch.complex128)
    try:
        E_c = E.to(torch.complex128)
    except Exception:
        E_c = torch.tensor(E, device=get_device(), dtype=torch.complex128)
    return torch.squeeze(torch.matmul(p_c, E_c.unsqueeze(2)), dim=2)


# --------------------------
# Interpolation
# --------------------------
def interp(x, xp, fp):
    x = torch.as_tensor(x, dtype=get_precision(), device=get_device())
    xp = torch.as_tensor(xp, dtype=get_precision(), device=get_device())
    fp = torch.as_tensor(fp, dtype=get_precision(), device=get_device())
    sorted_indices = torch.argsort(xp)
    xp = xp[sorted_indices]
    fp = fp[sorted_indices]
    x_clipped = torch.clip(x, xp[0], xp[-1])
    indices = torch.searchsorted(xp, x_clipped, right=True)
    indices = torch.clamp(indices, 1, len(xp) - 1)
    x0 = xp[indices - 1]
    x1 = xp[indices]
    y0 = fp[indices - 1]
    y1 = fp[indices]
    interpolated = y0 + (y1 - y0) * (x_clipped - x0) / (x1 - x0)
    return interpolated


def nearest_nd_interpolator(points, values, Hx, Hy):
    if Hx is None or Hy is None:
        raise ValueError("Hx and Hy must be provided")
    Hx, Hy = array(Hx), array(Hy)
    Hx, Hy = torch.broadcast_tensors(Hx, Hy)
    q_flat = torch.stack([Hx, Hy], dim=-1).reshape(-1, 2)
    d = torch.cdist(q_flat, points.to(dtype=q_flat.dtype, device=q_flat.device))
    idx = d.argmin(dim=1)
    vals = values.view(points.shape[0], -1)
    out = vals[idx].view(*Hx.shape, -1)
    return out.squeeze(-1) if out.shape[-1] == 1 else out


# --------------------------
# Polynomial Operations
# --------------------------
def polyfit(x, y, degree):
    X = torch.stack([x**i for i in range(degree + 1)], dim=1)
    coeffs, _ = torch.lstsq(y.unsqueeze(1), X)
    return coeffs[: degree + 1].squeeze()


def polyval(coeffs, x):
    return sum(c * x**i for i, c in enumerate(reversed(coeffs)))


# --------------------------
# Padding
# --------------------------
def pad(tensor, pad_width, mode="constant", constant_values=0):
    if mode != "constant":
        raise NotImplementedError("Only constant mode supported")
    (pt, pb), (pl, pr) = pad_width
    return F.pad(tensor, (pl, pr, pt, pb), mode="constant", value=constant_values)


# --------------------------
# Vectorization
# --------------------------
def vectorize(pyfunc):
    def wrapped(x):
        flat = x.reshape(-1)
        mapped = [pyfunc(xi) for xi in flat]
        out = torch.stack(
            [
                m
                if isinstance(m, torch.Tensor)
                else torch.tensor(m, dtype=get_precision(), device=get_device())
                for m in mapped
            ]
        )
        return out.view(x.shape)

    return wrapped


# --------------------------
# Conversion and Utilities
# --------------------------
def atleast_1d(x):
    t = torch.as_tensor(x, dtype=get_precision())
    return t.unsqueeze(0) if t.ndim == 0 else t


def atleast_2d(x):
    t = torch.as_tensor(x, dtype=get_precision())
    if t.ndim == 0:
        return t.unsqueeze(0).unsqueeze(0)
    if t.ndim == 1:
        return t.unsqueeze(0)
    return t


def as_array_1d(data):
    if isinstance(data, (int, float)):
        return array([data])
    if isinstance(data, (list, tuple)):
        return array(data)
    if is_array_like(data):
        return data.reshape(-1)
    raise ValueError("Unsupported type for as_array_1d")


def eye(n):
    return torch.eye(n, device=get_device(), dtype=get_precision())


# --------------------------
# Error State Context
# --------------------------
@contextlib.contextmanager
def errstate(**kwargs):
    yield


# --------------------------
# Miscellaneous Utilities
# --------------------------
def path_contains_points(
    vertices: torch.Tensor, points: torch.Tensor
) -> torch.BoolTensor:
    """
    Vectorized ray‐crossing algorithm.
    vertices: (N,2) tensor of polygon verts in order (closed implicitly: last→first).
    points:   (M,2) tensor of query points.
    returns:  BoolTensor of shape (M,) indicating inside‐ness.
    """
    # split into x/y
    vx, vy = vertices[:, 0], vertices[:, 1]
    px, py = points[:, 0].unsqueeze(1), points[:, 1].unsqueeze(1)  # (M,1)

    # roll vertices to get edge endpoints
    vx_next = torch.roll(vx, -1)
    vy_next = torch.roll(vy, -1)

    # test if ray crosses the edge in the y‐direction
    # mask: (M, N) True where edge straddles the horizontal ray from point
    cond = (vy > py) != (vy_next > py)

    # compute intersection point’s x coordinate
    slope = (vx_next - vx) / (vy_next - vy)
    x_int = vx + slope * (py - vy)

    # does the intersection lie to the right of the point?
    cross = cond & (px < x_int)

    # count crossings per point (sum over edges) and take parity
    inside = torch.sum(cross, dim=1) % 2 == 1
    return inside


# --------------------------
# Exported Symbols
# --------------------------
__all__ = [
    # Config
    "set_device",
    "get_device",
    "set_precision",
    "get_precision",
    "grad_mode",
    # Creation
    "array",
    "zeros",
    "ones",
    "full",
    "linspace",
    "arange",
    "zeros_like",
    "ones_like",
    "full_like",
    "load",
    # Utilities
    "cast",
    "copy",
    "is_array_like",
    "size",
    "newaxis",
    # Shape
    "reshape",
    "stack",
    "broadcast_to",
    "repeat",
    "flip",
    "meshgrid",
    "roll",
    "unsqueeze_last",
    "tile",
    # Random
    "default_rng",
    "random_uniform",
    "random_normal",
    # Math
    "sqrt",
    "sin",
    "cos",
    "exp",
    "log2",
    "abs",
    "radians",
    "deg2rad",
    "max",
    "min",
    "mean",
    "all",
    # Linear Algebra
    "matmul",
    "batched_chain_matmul3",
    "cross",
    "matrix_vector_multiply_and_squeeze",
    "mult_p_E",
    "to_complex",
    # Interpolation
    "interp",
    "nearest_nd_interpolator",
    # Polynomial
    "polyfit",
    "polyval",
    # Padding
    "pad",
    # Vectorization
    "vectorize",
    # Conversion
    "atleast_1d",
    "atleast_2d",
    "as_array_1d",
    "eye",
    # Error State
    "errstate",
]
