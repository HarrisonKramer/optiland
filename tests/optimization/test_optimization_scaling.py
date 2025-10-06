# tests/optimization/test_optimization_scaling.py
"""
Tests for the variable scaler classes in optiland.optimization.scaling.

These scalers are used to transform optimization variables into a more
uniform and well-behaved space, which can improve the performance and
stability of optimization algorithms.
"""
import pytest
import optiland.backend as be
from optiland.optimization.scaling.identity import IdentityScaler
from optiland.optimization.scaling.linear import LinearScaler
from optiland.optimization.scaling.reciprocal import ReciprocalScaler
from optiland.optimization.scaling.power import PowerScaler
from optiland.optimization.scaling.log import LogScaler
from ..utils import assert_allclose


class TestIdentityScaler:
    """
    Tests the IdentityScaler, which performs no scaling.
    """

    def test_scale(self, set_test_backend):
        """Tests that the scale method returns the input value unchanged."""
        scaler = IdentityScaler()
        assert_allclose(scaler.scale(10.0), 10.0)

    def test_inverse_scale(self, set_test_backend):
        """Tests that the inverse_scale method returns the input value unchanged."""
        scaler = IdentityScaler()
        assert_allclose(scaler.inverse_scale(10.0), 10.0)


class TestLinearScaler:
    """
    Tests the LinearScaler, which applies a linear transformation
    (y = factor * x + offset).
    """

    def test_scale(self, set_test_backend):
        """Tests the forward linear scaling transformation."""
        scaler = LinearScaler(factor=2.0, offset=1.0)
        assert_allclose(scaler.scale(10.0), 21.0)

    def test_inverse_scale(self, set_test_backend):
        """Tests the inverse linear scaling transformation."""
        scaler = LinearScaler(factor=2.0, offset=1.0)
        assert_allclose(scaler.inverse_scale(21.0), 10.0)


class TestReciprocalScaler:
    """
    Tests the ReciprocalScaler, which applies a reciprocal transformation
    (y = 1 / x).
    """

    def test_scale(self, set_test_backend):
        """Tests the forward reciprocal transformation."""
        scaler = ReciprocalScaler()
        assert_allclose(scaler.scale(2.0), 0.5)

    def test_inverse_scale(self, set_test_backend):
        """Tests the inverse reciprocal transformation."""
        scaler = ReciprocalScaler()
        assert_allclose(scaler.inverse_scale(0.5), 2.0)

    def test_scale_zero(self, set_test_backend):
        """Tests that scaling zero results in infinity."""
        scaler = ReciprocalScaler()
        assert scaler.scale(0.0) == be.inf

    def test_inverse_scale_zero(self, set_test_backend):
        """Tests that inverse scaling zero results in infinity."""
        scaler = ReciprocalScaler()
        assert scaler.inverse_scale(0.0) == be.inf


class TestPowerScaler:
    """
    Tests the PowerScaler, which raises the input value to a given power
    (y = x^power).
    """

    def test_scale(self, set_test_backend):
        """Tests the forward power transformation."""
        scaler = PowerScaler(power=2.0)
        assert_allclose(scaler.scale(10.0), 100.0)

    def test_inverse_scale(self, set_test_backend):
        """Tests the inverse power transformation."""
        scaler = PowerScaler(power=2.0)
        assert_allclose(scaler.inverse_scale(100.0), 10.0)


class TestLogScaler:
    """
    Tests the LogScaler, which applies a logarithmic transformation
    (y = log_base(x)).
    """

    def test_scale(self, set_test_backend):
        """Tests the forward logarithmic transformation."""
        scaler = LogScaler(base=10.0)
        assert_allclose(scaler.scale(100.0), 2.0)

    def test_inverse_scale(self, set_test_backend):
        """Tests the inverse logarithmic transformation."""
        scaler = LogScaler(base=10.0)
        assert_allclose(scaler.inverse_scale(2.0), 100.0)