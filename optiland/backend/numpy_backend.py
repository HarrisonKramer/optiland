"""
This module provides a backend for numerical operations using NumPy and SciPy.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from matplotlib.path import Path
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve as _fftconvolve
from scipy.spatial.transform import Rotation as R
from scipy.special import gamma

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.random import Generator

# Link to the underlying library
_lib = np


def arange_indices(start, stop=None, step=1) -> NDArray:
    """Create an array of indices (int64)."""
    return np.arange(start, stop, step, dtype=np.int64)


def cast(x: ArrayLike) -> NDArray:
    """Cast an array to the current precision (float)."""
    return np.array(x, dtype=float)


def get_precision():
    return 64


ScalarOrArray = TypeVar("ScalarOrArray", float, NDArray)


def array(x: ArrayLike) -> NDArray:
    """Create an array/tensor"""
    return np.array(x)


def transpose(x: ArrayLike, axes: Sequence[int] | None = None) -> NDArray:
    return np.transpose(x, axes)


def is_array_like(x: NDArray | list | tuple) -> bool:
    """Check if x is array-like"""
    return isinstance(x, np.ndarray | list | tuple)


def atleast_1d(x: ArrayLike) -> NDArray:
    return np.atleast_1d(x).astype(float)


def as_array_1d(data: float | list | tuple | NDArray) -> NDArray:
    """Force conversion to a 1D array"""
    if isinstance(data, int | float):
        return array([data])
    elif isinstance(data, list | tuple):
        return array(data)
    elif is_array_like(data):
        return data.reshape(-1)
    else:
        raise ValueError(
            "Unsupported input type: expected scalar, list, tuple, or array-like."
        )


def ravel(x: NDArray) -> NDArray:
    return np.ravel(x).astype(float)


def from_matrix(matrix: NDArray) -> R:
    return R.from_matrix(matrix)


def from_euler(euler: NDArray) -> R:
    return R.from_euler("xyz", euler)


def default_rng(seed: int | None = None) -> Generator:
    return np.random.default_rng(seed)


def random_uniform(
    low: ScalarOrArray = 0.0,
    high: ScalarOrArray = 1.0,
    size: int | tuple[int, ...] | None = None,
    generator: Generator | None = None,
) -> ScalarOrArray:
    if generator is None:
        generator = np.random.default_rng()
    return generator.uniform(low, high, size)


def rand(*size: int) -> NDArray:
    """
    Returns an array of random numbers from a uniform distribution on the
    interval [0, 1).
    If no size is provided, returns a single random number.
    """
    return np.random.rand(*size) if size else np.random.rand()


def random_normal(
    loc: ScalarOrArray = 0.0,
    scale: ScalarOrArray = 1.0,
    size: int | tuple[int, ...] | None = None,
    generator: Generator | None = None,
) -> ScalarOrArray:
    if generator is None:
        generator = np.random.default_rng()
    return generator.normal(loc, scale, size)


def matrix_vector_multiply_and_squeeze(
    p: NDArray, E: NDArray, backend: Literal["numpy"] = "numpy"
) -> NDArray:
    return np.squeeze(np.matmul(p, E[:, :, np.newaxis]), axis=2)


def nearest_nd_interpolator(
    points: NDArray, values: NDArray, x: ScalarOrArray, y: ScalarOrArray
) -> NDArray:
    interpolator = NearestNDInterpolator(points, values)
    result = interpolator(x, y)
    return result


def unsqueeze_last(x: NDArray) -> NDArray:
    return x[:, np.newaxis]


def mult_p_E(p: NDArray, E: NDArray) -> NDArray:
    # Used only for electric field multiplication in polarized_rays.py
    return np.squeeze(np.matmul(p, E[:, :, np.newaxis]), axis=2)


def to_complex(x: NDArray) -> NDArray:
    return x.astype(np.complex128) if np.isrealobj(x) else x


def batched_chain_matmul3(a: NDArray, b: NDArray, c: NDArray) -> NDArray:
    dtype = np.result_type(a, b, c)
    return np.matmul(np.matmul(a.astype(dtype), b.astype(dtype)), c.astype(dtype))


def factorial(n: ScalarOrArray) -> ScalarOrArray:
    return gamma(n + 1)


def path_contains_points(vertices: NDArray, points: NDArray) -> NDArray:
    path = Path(vertices)
    mask = path.contains_points(points)
    return np.asarray(mask, dtype=bool)


def lstsq(a: ArrayLike, b: ArrayLike) -> NDArray:
    """
    Computes the least-squares solution to a linear matrix equation.

    Args:
        a (ArrayLike): Left-hand side matrix (M, N).
        b (ArrayLike): Right-hand side matrix (M,) or (M, K).

    Returns:
        NDArray: Least-squares solution (N,) or (N, K).
    """
    return np.linalg.lstsq(a, b, rcond=None)[0]


def fftconvolve(
    in1: ArrayLike, in2: ArrayLike, mode: Literal["full", "valid", "same"] = "full"
) -> NDArray:
    """Numpy/Scipy implementation of FFT-based convolution."""
    a = array(in1)
    b = array(in2)
    return _fftconvolve(a, b, mode=mode)


def grid_sample(
    input, grid, mode="bilinear", padding_mode="zeros", align_corners=False
):
    """
    Numpy implementation of torch.nn.functional.grid_sample.

    Args:
        input (np.ndarray): Input tensor of shape (N, C, H_in, W_in).
        grid (np.ndarray): Grid of shape (N, H_out, W_out, 2).
                           Last dim is (x, y). Coordinates in range [-1, 1].
        mode (str): Interpolation mode ('bilinear', 'nearest').
        padding_mode (str): Padding mode ('zeros', 'border', 'reflection').
        align_corners (bool): Whether to align corners.

    Returns:
        np.ndarray: Output tensor of shape (N, C, H_out, W_out).
    """
    N, C, H_in, W_in = input.shape
    _N, H_out, W_out, _ = grid.shape
    if N != _N:
        raise ValueError("Input and grid must have same batch size")

    # Map grid coordinates [-1, 1] to pixel coordinates
    # grid[..., 0] is x (columns), grid[..., 1] is y (rows)
    x = grid[..., 0]
    y = grid[..., 1]

    if align_corners:
        # -1 -> 0, 1 -> W-1
        x_pix = ((x + 1) / 2) * (W_in - 1)
        y_pix = ((y + 1) / 2) * (H_in - 1)
    else:
        # Standard pytorch align_corners=False:
        # x = (x_norm + 1) * W / 2 - 0.5
        x_pix = ((x + 1) * W_in / 2) - 0.5
        y_pix = ((y + 1) * H_in / 2) - 0.5

    output = _lib.zeros((N, C, H_out, W_out), dtype=input.dtype)

    # Scipy order: 0=nearest, 1=bilinear
    order = 0 if mode == "nearest" else 1

    # Scipy mode: 'constant' (zeros), 'nearest' (border), 'reflect' (reflection)
    scipy_mode = "constant"
    cval = 0.0
    if padding_mode == "border":
        scipy_mode = "nearest"
    elif padding_mode == "reflection":
        scipy_mode = "reflect"

    for n in range(N):
        for c in range(C):
            # Coordinates must be (y, x) for map_coordinates
            coords = _lib.stack((y_pix[n], x_pix[n]))
            output[n, c] = map_coordinates(
                input[n, c], coords, order=order, mode=scipy_mode, cval=cval
            )

    return output
