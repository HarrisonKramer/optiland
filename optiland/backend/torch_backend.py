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

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from numpy.typing import ArrayLike
    from torch import Generator as TorchGenerator
    from torch import Tensor

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
        self.device: Literal["cpu", "cuda"] = "cpu"
        self.precision = torch.float32
        self.grad_mode = GradMode()

    def set_device(self, device: Literal["cpu", "cuda"]) -> None:
        if device not in ("cpu", "cuda"):
            raise ValueError("Device must be 'cpu' or 'cuda'.")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available.")
        self.device = device

    def get_device(self) -> Literal["cpu", "cuda"]:
        return self.device

    def set_precision(self, precision: Literal["float32", "float64"]) -> None:
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
def set_device(device: Literal["cpu", "cuda"]) -> None:
    return _config.set_device(device)


def get_device() -> Literal["cpu", "cuda"]:
    return _config.get_device()


def set_precision(precision: Literal["float32", "float64"]) -> None:
    return _config.set_precision(precision)


def get_precision() -> torch.dtype:
    return _config.get_precision()


# Global gradient control
grad_mode: GradMode = _config.grad_mode


# --------------------------
# Array Creation
# --------------------------
def array(x: ArrayLike) -> Tensor:
    """Create a tensor with current device, precision, and grad settings."""
    if isinstance(x, torch.Tensor):
        return x

    # to avoid slow conversion, if data is a list/tuple of numpy arrays,
    # convert it to a single multi-dimensional numpy array first
    if isinstance(x, (list | tuple)) and len(x) > 0 and isinstance(x[0], np.ndarray):
        x = np.array(x)

    return torch.tensor(
        x,
        device=get_device(),
        dtype=get_precision(),
        requires_grad=grad_mode.requires_grad,
    )


def zeros(shape: Sequence[int]) -> Tensor:
    return torch.zeros(
        shape,
        device=get_device(),
        dtype=get_precision(),
        requires_grad=grad_mode.requires_grad,
    )


def ones(shape: Sequence[int]) -> Tensor:
    return torch.ones(
        shape,
        device=get_device(),
        dtype=get_precision(),
        requires_grad=grad_mode.requires_grad,
    )


def full(shape: Sequence[int], fill_value: float) -> Tensor:
    return torch.full(
        shape,
        fill_value,
        device=get_device(),
        dtype=get_precision(),
        requires_grad=grad_mode.requires_grad,
    )


def linspace(start: float, stop: float, num: int = 50) -> Tensor:
    return torch.linspace(
        start,
        stop,
        num,
        device=get_device(),
        dtype=get_precision(),
        requires_grad=grad_mode.requires_grad,
    )


def arange(*args: float | Tensor, step: float | Tensor = 1) -> Tensor:
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


def zeros_like(x: ArrayLike) -> Tensor:
    return torch.zeros_like(
        array(x),
        device=get_device(),
        dtype=get_precision(),
        requires_grad=grad_mode.requires_grad,
    )


def ones_like(x: ArrayLike) -> Tensor:
    return torch.ones_like(
        array(x),
        device=get_device(),
        dtype=get_precision(),
        requires_grad=grad_mode.requires_grad,
    )


def full_like(x: ArrayLike, fill_value: float | Tensor) -> Tensor:
    x_t = array(x)
    val = fill_value.item() if isinstance(fill_value, torch.Tensor) else fill_value
    return torch.full_like(
        x_t,
        val,
        device=get_device(),
        dtype=get_precision(),
        requires_grad=grad_mode.requires_grad,
    )


def load(filename: str) -> Tensor:
    data = np.load(filename)
    return array(data)


# --------------------------
# Array Utilities
# --------------------------
def cast(x: ArrayLike) -> Tensor:
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, device=get_device(), dtype=get_precision())
    return x.to(device=get_device(), dtype=get_precision())


def copy(x: Tensor) -> Tensor:
    return x.clone()


def is_array_like(x: Any) -> bool:
    return isinstance(x, torch.Tensor | np.ndarray | list | tuple)


def size(x: Tensor) -> int:
    return torch.numel(x)


def newaxis() -> None:
    return None


def array_equal(a: Tensor, b: Tensor) -> bool:
    return torch.equal(a, b)


def shape(tensor: Tensor) -> tuple[int, ...]:
    """Returns the shape of a tensor."""
    return tensor.shape


# --------------------------
# Shape and Indexing
# --------------------------
def reshape(x: Tensor, shape: Sequence[int]) -> Tensor:
    return x.view(shape)


def stack(xs: Sequence[ArrayLike], axis: int = 0) -> Tensor:
    return torch.stack([cast(x) for x in xs], dim=axis)


def broadcast_to(x: Tensor, shape: Sequence[int]) -> Tensor:
    return x.expand(shape)


def repeat(x: Tensor, repeats: int) -> Tensor:
    return torch.repeat_interleave(x, repeats)


def flip(x: Tensor) -> Tensor:
    return torch.flip(x, dims=(0,))


def meshgrid(*arrays: Tensor) -> tuple[Tensor, ...]:
    return torch.meshgrid(*arrays, indexing="xy")


def roll(
    x: Tensor, shift: int | Sequence[int], axis: int | tuple[int, ...] = ()
) -> Tensor:
    return torch.roll(x, shifts=shift, dims=axis)


def unsqueeze_last(x: Tensor) -> Tensor:
    return x.unsqueeze(-1)


def tile(x: Tensor, dims: int | list[int] | tuple[int, ...]) -> Tensor:
    return torch.tile(x, dims if isinstance(dims, list | tuple) else (dims,))


def isscalar(x: ArrayLike | Tensor) -> bool:
    return torch.is_tensor(x) and x.dim() == 0


# --------------------------
# Random Number Generation
# --------------------------
def default_rng(seed: int | None = None) -> TorchGenerator:
    if seed is None:
        seed = torch.initial_seed()
    return torch.Generator(device=get_device()).manual_seed(seed)


def random_uniform(
    low: float = 0.0,
    high: float = 1.0,
    size: int | None = None,
    generator: TorchGenerator | None = None,
) -> Tensor:
    size = size or 1
    gen_args = {"generator": generator} if generator else {}
    return torch.empty(size, device=get_device(), dtype=get_precision()).uniform_(
        low, high, **gen_args
    )


def rand(*size: int) -> Tensor:
    """
    Returns a tensor filled with random numbers from a uniform distribution
    on the interval [0, 1).
    If no size is provided, returns a single random number as a 1-element tensor.
    """
    if not size:
        size = (1,)
    return torch.rand(
        size,
        device=get_device(),
        dtype=get_precision(),
        requires_grad=grad_mode.requires_grad,
    )


def random_normal(
    loc: float = 0.0,
    scale: float = 1.0,
    size: Sequence[int] | None = None,
    generator: TorchGenerator | None = None,
) -> Tensor:
    size = size or (1,)
    gen_args = {"generator": generator} if generator else {}
    return (
        torch.randn(size, device=get_device(), dtype=get_precision(), **gen_args)
        * scale
        + loc
    )


# --------------------------
# Mathematical Operations
# --------------------------
def sqrt(x: ArrayLike) -> Tensor:
    return torch.sqrt(array(x))


def sin(x: ArrayLike) -> Tensor:
    return torch.sin(array(x))


def power(x: ArrayLike, y: ArrayLike) -> Tensor:
    return torch.pow(array(x), array(y))


def cos(x: ArrayLike) -> Tensor:
    return torch.cos(array(x))


def exp(x: ArrayLike) -> Tensor:
    return torch.exp(array(x))


def log2(x: ArrayLike) -> Tensor:
    return torch.log2(array(x))


def abs(x: ArrayLike) -> Tensor:
    return torch.abs(array(x))


def radians(x: ArrayLike) -> Tensor:
    t = array(x)
    return torch.deg2rad(t)


def degrees(x: ArrayLike) -> Tensor:
    t = array(x)
    return torch.rad2deg(t)


deg2rad = radians
rad2deg = degrees


def max(x: ArrayLike) -> int | float:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().max().item()
    return np.max(x)


def min(x: ArrayLike) -> int | float:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().min().item()
    return np.min(x)


def maximum(a: ArrayLike, b: ArrayLike) -> Tensor:
    return torch.maximum(array(a), array(b))


def nanmax(
    input_tensor: Tensor, axis: int | None = None, keepdim: bool = False
) -> Tensor:
    nan_mask = torch.isnan(input_tensor)
    replaced = input_tensor.clone()
    replaced[nan_mask] = float("-inf")
    if axis is not None:
        result, _ = torch.max(replaced, dim=axis, keepdim=keepdim)
    else:
        result = torch.max(replaced)
    return result


def mean(x: ArrayLike, axis: int | None = None, keepdims: bool = False) -> Tensor:
    x = array(x)
    mask = ~torch.isnan(x)
    cnt = mask.sum(dim=axis, keepdim=keepdims).to(x.dtype)
    x0 = torch.where(mask, x, torch.tensor(0.0, dtype=x.dtype, device=x.device))
    s = x0.sum(dim=axis, keepdim=keepdims)
    return torch.where(
        cnt > 0, s / cnt, torch.tensor(float("nan"), dtype=x.dtype, device=x.device)
    )


def all(x: bool | ArrayLike) -> bool:
    if isinstance(x, bool):
        return x
    t = torch.as_tensor(x, dtype=_config.precision, device=_config.device)
    return bool(torch.all(t).item())


def factorial(n: ArrayLike) -> Tensor:
    return torch.lgamma(array(n + 1)).exp()


def histogram2d(
    x: Tensor, y: Tensor, bins: Sequence[Tensor], weights: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    if not isinstance(bins, (list | tuple)) or len(bins) != 2:
        raise ValueError("`bins` must be a list or tuple of two edge tensors.")

    x_edges, y_edges = bins[0], bins[1]
    nx = x_edges.numel() - 1
    ny = y_edges.numel() - 1

    # Find which bin each point belongs to
    # torch.searchsorted with right=False gives the insertion index
    # For histogram, we want the bin index (insertion_index - 1)
    x_bin_indices = torch.searchsorted(x_edges, x, right=False) - 1
    y_bin_indices = torch.searchsorted(y_edges, y, right=False) - 1

    # Clamp to valid bin range [0, n-1]
    x_bin_indices = torch.clamp(x_bin_indices, 0, nx - 1)
    y_bin_indices = torch.clamp(y_bin_indices, 0, ny - 1)

    # Create mask for points within the histogram bounds (inclusive of edges)
    mask = (
        (x >= x_edges[0]) & (x <= x_edges[-1]) & (y >= y_edges[0]) & (y <= y_edges[-1])
    )

    if weights is None:
        weights = torch.ones_like(x)

    # Apply mask to get valid points only
    valid_x_indices = x_bin_indices[mask]
    valid_y_indices = y_bin_indices[mask]
    valid_weights = weights[mask]

    # Convert 2D bin indices to 1D linear indices
    linear_indices = (valid_y_indices * nx + valid_x_indices).long()

    # Create flattened histogram and accumulate weights
    hist_flat = torch.zeros(nx * ny, device=x.device, dtype=valid_weights.dtype)
    hist_flat.index_add_(0, linear_indices, valid_weights)

    # Reshape to 2D form (note: .T transposes to get the right orientation)
    hist = hist_flat.reshape(ny, nx).T

    return hist, x_edges, y_edges


def get_bilinear_weights(
    coords: Tensor, bin_edges: Sequence[Tensor]
) -> tuple[Tensor, Tensor]:
    """
    Calculates differentiable bilinear interpolation weights.
    This version operates on pixel centers to avoid boundary indexing errors.
    Inspiration from the paper:
    "Differentiable design of a double-freeform lens with
    multi-level radial basis functions for extended
    source irradiance tailoring"
    https://doi.org/10.1364/OPTICA.520485
    """
    x_edges, y_edges = bin_edges
    x, y = coords[:, 0].contiguous(), coords[:, 1].contiguous()

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    ix = torch.searchsorted(x_centers, x, right=True) - 1
    iy = torch.searchsorted(y_centers, y, right=True) - 1
    # Clamp indices to the valid range for pixel centers.
    # The max index is len(centers) - 2 to allow access to ix+1.
    ix = torch.clamp(ix, 0, len(x_centers) - 2)
    iy = torch.clamp(iy, 0, len(y_centers) - 2)

    x0, x1 = x_centers[ix], x_centers[ix + 1]
    y0, y1 = y_centers[iy], y_centers[iy + 1]

    # bilinear interpolation weights
    wx = (x - x0) / (x1 - x0 + 1e-9)
    wy = (y - y0) / (y1 - y0 + 1e-9)

    # wEights for the four pixels
    w00 = (1 - wx) * (1 - wy)
    w01 = (1 - wx) * wy
    w10 = wx * (1 - wy)
    w11 = wx * wy

    # stack the indices of the four pixels for each ray
    all_indices = torch.stack(
        [
            torch.stack([ix, iy], dim=1),
            torch.stack([ix, iy + 1], dim=1),
            torch.stack([ix + 1, iy], dim=1),
            torch.stack([ix + 1, iy + 1], dim=1),
        ],
        dim=1,
    )

    all_weights = torch.stack([w00, w01, w10, w11], dim=1)

    return all_indices, all_weights


def copy_to(source: Tensor, destination: Tensor) -> None:
    """
    Performs an in-place copy from source to destination tensor.
    This version safely handles tensors that require gradients by
    modifying their underlying data.
    """
    if destination.requires_grad:
        destination.data.copy_(source)
    else:
        destination.copy_(source)


# --------------------------
# Linear Algebra
# --------------------------
def matmul(a: Tensor, b: Tensor) -> Tensor:
    dtype = torch.promote_types(a.dtype, b.dtype)
    return torch.matmul(a.to(dtype), b.to(dtype))


def batched_chain_matmul3(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    dtype = torch.promote_types(torch.promote_types(a.dtype, b.dtype), c.dtype)
    return torch.matmul(torch.matmul(a.to(dtype), b.to(dtype)), c.to(dtype))


def cross(
    a: Tensor,
    b: Tensor,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int | None = None,
) -> Tensor:
    """A NumPy-compatible cross product for PyTorch."""
    if axis is not None:
        axisa, axisb, axisc = (axis,) * 3

    # Move the specified axes to the end for `torch.linalg.cross`
    a_moved = torch.movedim(a, axisa, -1)
    b_moved = torch.movedim(b, axisb, -1)

    # Compute the cross product along the last dimension
    c = torch.linalg.cross(a_moved, b_moved, dim=-1)

    # Move the result axis to the specified position
    return torch.movedim(c, -1, axisc)


def matrix_vector_multiply_and_squeeze(p: Tensor, E: Tensor) -> Tensor:
    return torch.matmul(p, E.unsqueeze(2)).squeeze(2)


def to_complex(x: Tensor) -> Tensor:
    return x.to(torch.complex128)


def mult_p_E(p: Tensor, E: Tensor) -> Tensor:
    p_c = p.to(torch.complex128)
    try:
        E_c = E.to(torch.complex128)
    except Exception:
        E_c = torch.tensor(E, device=get_device(), dtype=torch.complex128)
    return torch.squeeze(torch.matmul(p_c, E_c.unsqueeze(2)), dim=2)


# --------------------------
# Interpolation
# --------------------------
def interp(x: ArrayLike, xp: ArrayLike, fp: ArrayLike) -> Tensor:
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


def nearest_nd_interpolator(
    points: Tensor, values: Tensor, Hx: Tensor, Hy: Tensor
) -> Tensor:
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
def polyfit(x: Tensor, y: Tensor, degree: int) -> Tensor:
    X = torch.stack([x**i for i in range(degree + 1)], dim=1)
    coeffs, _ = torch.linalg.lstsq(y.unsqueeze(1), X)
    return coeffs[: degree + 1].squeeze()


def polyval(coeffs: Sequence[float], x: float | Tensor) -> float | Tensor:
    return sum(c * x**i for i, c in enumerate(reversed(coeffs)))


# --------------------------
# Padding
# --------------------------
def pad(
    tensor: Tensor,
    pad_width: tuple[tuple[int, int], tuple[int, int]],
    mode: Literal["constant"] = "constant",
    constant_values: float | None = 0,
):
    if mode != "constant":
        raise NotImplementedError("Only constant mode supported")
    (pt, pb), (pl, pr) = pad_width
    return F.pad(tensor, (pl, pr, pt, pb), mode="constant", value=constant_values)


# --------------------------
# Vectorization
# --------------------------
def vectorize(pyfunc: Callable[..., Any]) -> Callable[[Tensor], Tensor]:
    def wrapped(x: Tensor) -> Tensor:
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
def atleast_1d(x: ArrayLike) -> Tensor:
    t = torch.as_tensor(x, dtype=get_precision())
    return t.unsqueeze(0) if t.ndim == 0 else t


def atleast_2d(x: ArrayLike) -> Tensor:
    t = torch.as_tensor(x, dtype=get_precision())
    if t.ndim == 0:
        return t.unsqueeze(0).unsqueeze(0)
    if t.ndim == 1:
        return t.unsqueeze(0)
    return t


def as_array_1d(data: ArrayLike) -> Tensor:
    if isinstance(data, int | float):
        return array([data])
    if isinstance(data, list | tuple):
        return array(data)
    if is_array_like(data):
        return array(data).reshape(-1)
    raise ValueError("Unsupported type for as_array_1d")


def eye(n: int) -> Tensor:
    return torch.eye(n, device=get_device(), dtype=get_precision())


# --------------------------
# Error State Context
# --------------------------
@contextlib.contextmanager
def errstate(**kwargs: Any) -> Generator[Any, Any, Any]:
    yield


# --------------------------
# Miscellaneous Utilities
# --------------------------
def path_contains_points(vertices: Tensor, points: Tensor) -> torch.BoolTensor:
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
    "copy_to",
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
    "shape",
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
    "maximum",
    "mean",
    "all",
    "histogram2d",
    "get_bilinear_weights",
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
