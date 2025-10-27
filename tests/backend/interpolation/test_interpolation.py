"""Tests for the backend-agnostic spline interpolator."""
import pytest
import numpy as np

from optiland import backend as be
from optiland.backend.interpolation import get_spline_interpolator
from tests.utils import assert_allclose


def test_spline_interpolator(set_test_backend):
    """Tests that the spline interpolator works for all backends."""
    x = be.linspace(-3, 3, 10)
    y = be.linspace(-3, 3, 10)
    xx, yy = be.meshgrid(x, y)
    grid = be.sin(xx**2 + yy**2)

    interpolator = get_spline_interpolator(x, y, grid)

    # Test interpolation
    x_test = be.array([0.5, 1.5])
    y_test = be.array([0.5, 1.5])
    result = interpolator.ev(y_test, x_test)
    assert result.shape == (2,)

    # Test derivatives
    result_dx = interpolator.ev(y_test, x_test, dx=1)
    assert result_dx.shape == (2,)

    result_dy = interpolator.ev(y_test, x_test, dy=1)
    assert result_dy.shape == (2,)


def test_spline_interpolator_inplace_update(set_test_backend):
    """Tests that the spline interpolator can be updated in-place."""
    if be.get_backend() == "numpy":
        pytest.skip("In-place update test is for torch backend only.")

    x = be.linspace(-3, 3, 10)
    y = be.linspace(-3, 3, 10)
    xx, yy = be.meshgrid(x, y)
    grid = be.sin(xx**2 + yy**2).detach()
    grid.requires_grad = True

    interpolator = get_spline_interpolator(x, y, grid)

    x_test = be.array([0.5, 1.5])
    y_test = be.array([0.5, 1.5])

    # First interpolation
    result1 = interpolator.ev(y_test, x_test)
    loss1 = be.sum(result1)
    loss1.backward()
    assert grid.grad is not None

    # Update grid and re-interpolate
    with be.no_grad():
        grid_new = be.cos(xx**2 + yy**2).detach()
        grid_new.requires_grad = True
    interpolator.grid = grid_new

    result2 = interpolator.ev(y_test, x_test)
    loss2 = be.sum(result2)
    loss2.backward()

    # Check that results are different and new grad is computed
    assert not be.allclose(result1, result2)
    assert grid_new.grad is not None
