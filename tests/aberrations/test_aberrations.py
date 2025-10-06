# tests/aberrations/test_aberrations.py
"""
Tests for the Aberrations class in optiland.aberrations.

This file verifies the calculation of Seidel and third-order aberrations for
various sample optical systems.
"""
import optiland.backend as be
import pytest

from optiland.aberrations import Aberrations
from optiland.optic import Optic
from optiland.samples.objectives import DoubleGauss
from optiland.samples.simple import Edmund_49_847, SingletStopSurf2
from ..utils import assert_allclose


@pytest.fixture
def double_gauss():
    """Provides a DoubleGauss lens instance for testing."""
    return DoubleGauss()


@pytest.fixture
def edmund_singlet():
    """Provides an Edmund Optics 49-847 singlet instance for testing."""
    return Edmund_49_847()


@pytest.fixture
def singlet_stop_surf_two():
    """Provides a singlet lens with the stop on surface 2."""
    return SingletStopSurf2()


@pytest.fixture
def simple_singlet():
    """Provides a simple singlet lens with a single field and wavelength."""
    lens = Optic()
    lens.add_surface(index=0, radius=be.inf, thickness=be.inf)
    lens.add_surface(index=1, thickness=7, radius=19.93, is_stop=True, material="N-SF11")
    lens.add_surface(index=2, thickness=21.48)
    lens.add_surface(index=3)
    lens.set_aperture(aperture_type="EPD", value=20.0)
    lens.set_field_type(field_type="angle")
    lens.add_field(y=0)
    lens.add_wavelength(value=0.55, is_primary=True)
    lens.update_paraxial()
    lens.image_solve()
    return lens


class TestDoubleGaussAberrations:
    """Tests aberration calculations for the Double Gauss lens system."""

    def test_init(self, set_test_backend, double_gauss):
        """Tests the initialization of the Aberrations class."""
        aberrations = Aberrations(double_gauss)
        assert aberrations.optic == double_gauss

    def test_seidels(self, set_test_backend, double_gauss):
        """Tests the calculation of the five Seidel aberration sums."""
        S = double_gauss.aberrations.seidels()
        assert_allclose(S, [-0.0039, 0.0003, 0.0034, -0.0162, -0.0464], atol=1e-4)

    def test_third_order(self, set_test_backend, double_gauss):
        """
        Tests the comprehensive `third_order` method, which returns all
        third-order aberration coefficients.
        """
        data = double_gauss.aberrations.third_order()
        TSC, SC, CC, TCC, TAC, AC, TPC, PC, DC, TAchC, LchC, TchC, S = data
        assert_allclose(be.sum(SC), -0.19647, atol=1e-5)
        assert_allclose(be.sum(CC), 0.001977, atol=1e-5)
        assert_allclose(be.sum(AC), 0.171195, atol=1e-5)
        assert_allclose(be.sum(PC), -0.813237, atol=1e-5)
        assert_allclose(be.sum(DC), -0.232420, atol=1e-5)
        assert_allclose(be.sum(LchC), 0.295705, atol=1e-5)
        assert_allclose(be.sum(TchC), -0.018043, atol=1e-5)
        assert_allclose(S[0], -0.003929, atol=1e-5)

    def test_third_order_all_functions(self, set_test_backend, double_gauss):
        """
        Tests the individual methods for each third-order aberration type.
        """
        assert_allclose(be.sum(double_gauss.aberrations.SC()), -0.19647, atol=1e-5)
        assert_allclose(be.sum(double_gauss.aberrations.CC()), 0.001977, atol=1e-5)
        assert_allclose(be.sum(double_gauss.aberrations.AC()), 0.171195, atol=1e-5)
        assert_allclose(be.sum(double_gauss.aberrations.PC()), -0.813237, atol=1e-5)
        assert_allclose(be.sum(double_gauss.aberrations.DC()), -0.232420, atol=1e-5)
        assert_allclose(be.sum(double_gauss.aberrations.LchC()), 0.295705, atol=1e-5)
        assert_allclose(be.sum(double_gauss.aberrations.TchC()), -0.018043, atol=1e-5)


class TestEdmundSinglet:
    """Tests aberration calculations for the Edmund Optics 49-847 singlet."""

    def test_init(self, set_test_backend, edmund_singlet):
        """Tests the initialization of the Aberrations class."""
        aberrations = Aberrations(edmund_singlet)
        assert aberrations.optic == edmund_singlet

    def test_seidels(self, set_test_backend, edmund_singlet):
        """Tests the calculation of the five Seidel aberration sums."""
        S = edmund_singlet.aberrations.seidels()
        assert_allclose(S[0], -1.730769, atol=1e-5)
        assert_allclose(S[1], 0.142537, atol=1e-5)
        assert_allclose(S[2], -0.352955, atol=1e-5)
        assert_allclose(S[3], -0.221200, atol=1e-5)
        assert_allclose(S[4], -0.020854, atol=1e-5)

    def test_third_order(self, set_test_backend, edmund_singlet):
        """
        Tests the comprehensive `third_order` method for the singlet.
        """
        data = edmund_singlet.aberrations.third_order()
        TSC, SC, CC, TCC, TAC, AC, TPC, PC, DC, TAchC, LchC, TchC, S = data
        assert_allclose(be.sum(SC), -3.46088, atol=1e-5)
        assert_allclose(be.sum(CC), 0.14252, atol=1e-5)
        assert_allclose(be.sum(AC), -0.70577, atol=1e-5)
        assert_allclose(be.sum(PC), -0.44231, atol=1e-5)
        assert_allclose(be.sum(DC), -0.02085, atol=1e-5)
        assert_allclose(be.sum(LchC), -0.98941, atol=1e-5)
        assert_allclose(be.sum(TchC), 0.0, atol=1e-5)

    def test_third_order_all_functions(self, set_test_backend, edmund_singlet):
        """
        Tests the individual aberration methods for the singlet.
        """
        assert_allclose(be.sum(edmund_singlet.aberrations.SC()), -3.46088, atol=1e-5)
        assert_allclose(be.sum(edmund_singlet.aberrations.CC()), 0.14252, atol=1e-5)
        assert_allclose(be.sum(edmund_singlet.aberrations.AC()), -0.70577, atol=1e-5)
        assert_allclose(be.sum(edmund_singlet.aberrations.PC()), -0.44231, atol=1e-5)
        assert_allclose(be.sum(edmund_singlet.aberrations.DC()), -0.02085, atol=1e-5)
        assert_allclose(be.sum(edmund_singlet.aberrations.LchC()), -0.98941, atol=1e-5)
        assert_allclose(be.sum(edmund_singlet.aberrations.TchC()), 0.0, atol=1e-5)


class TestSingletStopTwo:
    """Tests aberration calculations for a singlet with the stop on surface 2."""

    def test_init(self, set_test_backend, singlet_stop_surf_two):
        """Tests the initialization of the Aberrations class."""
        aberrations = Aberrations(singlet_stop_surf_two)
        assert aberrations.optic == singlet_stop_surf_two

    def test_seidels(self, set_test_backend, singlet_stop_surf_two):
        """Tests the calculation of the five Seidel aberration sums."""
        S = singlet_stop_surf_two.aberrations.seidels()
        assert_allclose(S[0], -0.032605, atol=1e-5)
        assert_allclose(S[1], -0.000438, atol=1e-5)
        assert_allclose(S[2], -0.011424, atol=1e-5)
        assert_allclose(S[3], -0.006920, atol=1e-5)
        assert_allclose(S[4], 0.001654, atol=1e-5)

    def test_third_order(self, set_test_backend, singlet_stop_surf_two):
        """
        Tests the comprehensive `third_order` method for this singlet.
        """
        data = singlet_stop_surf_two.aberrations.third_order()
        TSC, SC, CC, TCC, TAC, AC, TPC, PC, DC, TAchC, LchC, TchC, S = data
        assert_allclose(be.sum(SC), -1.07493, atol=1e-5)
        assert_allclose(be.sum(CC), -0.00178, atol=1e-5)
        assert_allclose(be.sum(AC), -0.37665, atol=1e-5)
        assert_allclose(be.sum(PC), -0.22814, atol=1e-5)
        assert_allclose(be.sum(DC), 0.00671, atol=1e-5)
        assert_allclose(be.sum(LchC), -1.85642, atol=1e-5)
        assert_allclose(be.sum(TchC), 0.01112, atol=1e-5)

    def test_third_order_all_functions(self, set_test_backend, singlet_stop_surf_two):
        """
        Tests the individual aberration methods for this singlet.
        """
        assert_allclose(be.sum(singlet_stop_surf_two.aberrations.SC()), -1.07493, atol=1e-5)
        assert_allclose(be.sum(singlet_stop_surf_two.aberrations.CC()), -0.00178, atol=1e-5)
        assert_allclose(be.sum(singlet_stop_surf_two.aberrations.AC()), -0.37665, atol=1e-5)
        assert_allclose(be.sum(singlet_stop_surf_two.aberrations.PC()), -0.22814, atol=1e-5)
        assert_allclose(be.sum(singlet_stop_surf_two.aberrations.DC()), 0.00671, atol=1e-5)
        assert_allclose(be.sum(singlet_stop_surf_two.aberrations.LchC()), -1.85642, atol=1e-5)
        assert_allclose(be.sum(singlet_stop_surf_two.aberrations.TchC()), 0.01112, atol=1e-5)


class TestSimpleSinglet:
    """Tests aberration calculations for a simple on-axis singlet."""

    def test_on_axis_seidels_are_not_zero(self, set_test_backend, simple_singlet):
        """
        Tests that for an on-axis field, only spherical aberration (S[0]) is
        non-zero.
        """
        S = simple_singlet.aberrations.seidels()
        assert not be.isclose(S[0], be.array(0.0))
        assert_allclose(S[0], -0.675281089)
        assert_allclose(S[1:], [0, 0, 0, 0], atol=1e-8)