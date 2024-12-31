import numpy as np
from scipy.spatial.transform import Rotation as R


# Link to the underlying library
_lib = np


def array(x):
    """Create an array/tensor."""
    return np.asarray(x, dtype=float)


def from_matrix(matrix):
    return R.from_matrix(matrix)


def from_euler(euler):
    return R.from_euler('xyz', euler)


def default_rng(seed=None):
    return np.random.default_rng(seed)


def random_uniform(low=0.0, high=1.0, size=None, generator=None):
    return generator.uniform(low, high, size)