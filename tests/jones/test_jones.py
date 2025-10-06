# tests/jones/test_jones.py
"""
Tests for various Jones matrix components in optiland.jones.

This file verifies the functionality of Jones matrices for representing
different optical components that affect polarization, such as Fresnel
surfaces and various types of polarizers.
"""
import optiland.backend as be
import pytest
import numpy as np
from optiland import jones, materials
from optiland.rays import RealRays
from ..utils import assert_allclose


class TestJonesFresnel:
    """
    Tests the JonesFresnel class, which calculates the Jones matrix for
    reflection and transmission at an interface between two media based on the
    Fresnel equations.
    """

    def test_normal_incidence(self, set_test_backend):
        """
        Tests the Fresnel Jones matrix at normal incidence (AOI = 0).
        """
        rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        material_pre = materials.IdealMaterial(n=1.0)
        material_post = materials.IdealMaterial(n=1.5)
        jones_fresnel = jones.JonesFresnel(material_pre, material_post)

        # Test reflection matrix
        jones_matrix_refl = jones_fresnel.calculate_matrix(rays, reflect=True, aoi=be.array([0.0]))
        assert jones_matrix_refl.shape == (1, 3, 3)
        assert jones_matrix_refl[0, 0, 0] == -0.2
        assert jones_matrix_refl[0, 1, 1] == -0.2

        # Test transmission matrix
        jones_matrix_trans = jones_fresnel.calculate_matrix(rays, reflect=False, aoi=be.array([0.0]))
        assert jones_matrix_trans.shape == (1, 3, 3)
        assert jones_matrix_trans[0, 0, 0] == 0.8
        assert jones_matrix_trans[0, 1, 1] == 0.8

    def test_non_normal_incidence(self, set_test_backend):
        """
        Tests the Fresnel Jones matrix at a non-normal angle of incidence,
        using a complex refractive index.
        """
        rays = RealRays(1.0, 2.0, 3.0, 0.1, -0.25, np.sqrt(1 - 0.1**2 - (-0.25)**2))
        material_pre = materials.IdealMaterial(n=1.0, k=0.0)
        material_post = materials.IdealMaterial(n=1.5, k=0.1)
        jones_fresnel = jones.JonesFresnel(material_pre, material_post)

        # Test reflection with an arbitrary AOI
        jones_matrix_refl = jones_fresnel.calculate_matrix(rays, reflect=True, aoi=be.array([0.2]))
        assert jones_matrix_refl.shape == (1, 3, 3)
        assert_allclose(be.real(jones_matrix_refl[0, 0, 0]), -0.205411, atol=1e-5)
        assert_allclose(be.real(jones_matrix_refl[0, 1, 1]), -0.194576, atol=1e-5)

        # Test transmission
        jones_matrix_trans = jones_fresnel.calculate_matrix(rays, reflect=False, aoi=be.array([0.2]))
        assert jones_matrix_trans.shape == (1, 3, 3)
        assert_allclose(be.real(jones_matrix_trans[0, 0, 0]), 0.794588, atol=1e-5)
        assert_allclose(be.real(jones_matrix_trans[0, 1, 1]), 0.796384, atol=1e-5)


def test_jones_polarizer_H(set_test_backend):
    """Tests the Jones matrix for a horizontal linear polarizer."""
    rays = RealRays(x=[1.0, 2.0], y=[3.0, 0.0], z=[0.0, 1.0], L=0, M=0, N=1)
    jones_matrix = jones.JonesPolarizerH().calculate_matrix(rays)
    assert jones_matrix.shape == (2, 3, 3)
    assert jones_matrix[0, 0, 0] == 1.0
    assert jones_matrix[0, 1, 1] == 0.0


def test_jones_polarizer_V(set_test_backend):
    """Tests the Jones matrix for a vertical linear polarizer."""
    rays = RealRays(x=[1.0, 2.0], y=[3.0, 0.0], z=[0.0, 1.0], L=0, M=0, N=1)
    jones_matrix = jones.JonesPolarizerV().calculate_matrix(rays)
    assert jones_matrix.shape == (2, 3, 3)
    assert jones_matrix[0, 0, 0] == 0.0
    assert jones_matrix[0, 1, 1] == 1.0


def test_jones_polarizer_L45(set_test_backend):
    """Tests the Jones matrix for a linear polarizer at +45 degrees."""
    rays = RealRays(1, 2, 3, 0, 0, 1, 1, 1)
    jones_matrix = jones.JonesPolarizerL45().calculate_matrix(rays)
    assert jones_matrix.shape == (1, 3, 3)
    assert jones_matrix[:, 0, 0] == 0.5
    assert jones_matrix[:, 0, 1] == 0.5


def test_jones_polarizer_L135(set_test_backend):
    """Tests the Jones matrix for a linear polarizer at +135 degrees."""
    rays = RealRays(1, 2, 3, 0, 0, 1, 1, 1)
    jones_matrix = jones.JonesPolarizerL135().calculate_matrix(rays)
    assert jones_matrix.shape == (1, 3, 3)
    assert jones_matrix[:, 0, 0] == 0.5
    assert jones_matrix[:, 0, 1] == -0.5


def test_jones_polarizer_rcp(set_test_backend):
    """Tests the Jones matrix for a right-circular polarizer."""
    rays = RealRays(1, 2, 3, 0, 0, 1, 1, 1)
    jones_matrix = jones.JonesPolarizerRCP().calculate_matrix(rays)
    assert jones_matrix.shape == (1, 3, 3)
    assert jones_matrix[0, 0, 1] == 1j * 0.5


def test_jones_polarizer_lcp(set_test_backend):
    """Tests the Jones matrix for a left-circular polarizer."""
    rays = RealRays(1, 2, 3, 0, 0, 1, 1, 1)
    jones_matrix = jones.JonesPolarizerLCP().calculate_matrix(rays)
    assert jones_matrix.shape == (1, 3, 3)
    assert jones_matrix[0, 0, 1] == -1j * 0.5


def test_jones_linear_diattenuator(set_test_backend):
    """Tests the Jones matrix for a linear diattenuator."""
    rays = RealRays(1, 2, 3, 0, 0, 1, 1, 1)
    jones_diattenuator = jones.JonesLinearDiattenuator(t_min=0.2, t_max=1.0, theta=0.5)
    jones_matrix = jones_diattenuator.calculate_matrix(rays)
    assert jones_matrix.shape == (1, 3, 3)
    assert_allclose(jones_matrix[0, 0, 0], 0.81612, atol=1e-5)


def test_jones_linear_retarder(set_test_backend):
    """Tests the Jones matrix for a linear retarder."""
    rays = RealRays(1, 2, 3, 0, 0, 1, 1, 1)
    jones_retarder = jones.JonesLinearRetarder(retardance=0.5, theta=0.5)
    jones_matrix = jones_retarder.calculate_matrix(rays)
    assert jones_matrix.shape == (1, 3, 3)
    assert_allclose(be.real(jones_matrix[0, 0, 0]), 0.96891, atol=1e-5)
    assert_allclose(be.imag(jones_matrix[0, 0, 0]), -0.13367, atol=1e-5)


def test_jones_quarter_wave_retarder(set_test_backend):
    """Tests the Jones matrix for a quarter-wave plate."""
    rays = RealRays(1, 2, 3, 0, 0, 1, 1, 1)
    jones_retarder = jones.JonesQuarterWaveRetarder(theta=0.5)
    jones_matrix = jones_retarder.calculate_matrix(rays)
    assert jones_matrix.shape == (1, 3, 3)
    assert_allclose(be.real(jones_matrix[0, 0, 0]), 0.70710, atol=1e-5)
    assert_allclose(be.imag(jones_matrix[0, 0, 0]), -0.38205, atol=1e-5)


def test_jones_half_wave_retarder(set_test_backend):
    """Tests the Jones matrix for a half-wave plate."""
    rays = RealRays(1, 2, 3, 0, 0, 1, 1, 1)
    jones_retarder = jones.JonesHalfWaveRetarder(theta=0.5)
    jones_matrix = jones_retarder.calculate_matrix(rays)
    assert jones_matrix.shape == (1, 3, 3)
    assert_allclose(be.imag(jones_matrix[0, 0, 0]), -0.54030, atol=1e-5)
    assert_allclose(be.imag(jones_matrix[0, 1, 1]), 0.54030, atol=1e-5)