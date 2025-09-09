"""
This module provides a backend for numerical operations using NumPy and SciPy.

Kramer Harrison, 2024
"""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial.transform import Rotation as R
from scipy.special import gamma

# Link to the underlying library
_lib = np


def array(x):
    """Create an array/tensor"""
    return np.array(x, dtype=float)


def is_array_like(x):
    """Check if x is array-like"""
    return isinstance(x, np.ndarray | list | tuple)


def atleast_1d(x):
    return np.atleast_1d(x).astype(float)


def as_array_1d(data):
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


def ravel(x):
    return np.ravel(x).astype(float)


def from_matrix(matrix):
    return R.from_matrix(matrix)


def from_euler(euler):
    return R.from_euler("xyz", euler)


def default_rng(seed=None):
    return np.random.default_rng(seed)


def random_uniform(low=0.0, high=1.0, size=None, generator=None):
    if generator is None:
        generator = np.random.default_rng()
    return generator.uniform(low, high, size)


def random_normal(loc=0.0, scale=1.0, size=None, generator=None):
    if generator is None:
        generator = np.random.default_rng()
    return generator.normal(loc, scale, size)


def sobol_sampler(dim, num_samples, scramble=True, seed=None):
    """
    Generate quasi-random samples using Sobol sequences.

    Args:
        dim (int): Dimension of the samples
        num_samples (int): Number of samples to generate
        scramble (bool): Whether to scramble the sequence
        seed (int): Random seed for scrambling

    Returns:
        np.ndarray: Samples of shape (num_samples, dim)
    """
    try:
        from scipy.stats import qmc
    except ImportError as exc:
        raise ImportError(
            "scipy is required for Sobol sampling with numpy backend"
        ) from exc

    # Ensure num_samples is a power of 2 for best Sobol performance
    if num_samples > 0:
        num_samples_pow2 = 1 << (num_samples - 1).bit_length()
    else:
        num_samples_pow2 = num_samples

    sobol = qmc.Sobol(d=dim, scramble=scramble, seed=seed)
    samples = sobol.random(n=num_samples_pow2)
    return samples.astype(np.float32)


def erfinv(x):
    """
    Inverse error function.

    Args:
        x: Input array

    Returns:
        np.ndarray: Inverse error function of x
    """
    try:
        from scipy.special import erfinv as scipy_erfinv
    except ImportError as exc:
        raise ImportError("scipy is required for erfinv with numpy backend") from exc

    return scipy_erfinv(np.asarray(x))


def matrix_vector_multiply_and_squeeze(p, E, backend="numpy"):
    return np.squeeze(np.matmul(p, E[:, :, np.newaxis]), axis=2)


def nearest_nd_interpolator(points, values, x, y):
    interpolator = NearestNDInterpolator(points, values)
    result = interpolator(x, y)
    return result


def unsqueeze_last(x):
    return x[:, np.newaxis]


def mult_p_E(p, E):
    # Used only for electric field multiplication in polarized_rays.py
    return np.squeeze(np.matmul(p, E[:, :, np.newaxis]), axis=2)


def to_complex(x):
    return x.astype(np.complex128) if np.isrealobj(x) else x


def batched_chain_matmul3(a, b, c):
    dtype = np.result_type(a, b, c)
    return np.matmul(np.matmul(a.astype(dtype), b.astype(dtype)), c.astype(dtype))


def factorial(n):
    return gamma(n + 1)


def path_contains_points(vertices: np.ndarray, points: np.ndarray) -> np.ndarray:
    path = Path(vertices)
    mask = path.contains_points(points)
    return np.asarray(mask, dtype=bool)
