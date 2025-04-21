import numpy as np
import optiland.backend as be


def assert_allclose(a, b, rtol=1.e-5, atol=1.e-8):
    """Assert that two arrays or tensors are element-wise equal within
    tolerance.
    """
    # pick a default only if the user didn’t pass one
    if atol is None:
        backend = be.get_backend()
        if backend == 'torch':
            atol = 1e-4
        else:
            atol = 1e-7

    a = be.to_numpy(a)
    b = be.to_numpy(b)
    assert np.allclose(a, b, rtol=rtol, atol=atol)
