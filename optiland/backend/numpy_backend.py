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
from scipy.spatial.transform import Rotation as R
from scipy.special import gamma

if TYPE_CHECKING:
    from numpy.random import Generator

# Link to the underlying library
_lib = np


ScalarOrArray = TypeVar("ScalarOrArray", float, NDArray)


def array(x: ArrayLike) -> NDArray:
    """Create an array/tensor"""
    return np.array(x, dtype=float)


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
