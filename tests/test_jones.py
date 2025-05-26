import optiland.backend as be
import pytest
import numpy as np
from optiland import jones, materials
from optiland.rays import RealRays
from .utils import assert_allclose


class TestJonesFresnel:
    def test_normal_incidence(self, set_test_backend):
        rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)

        material_pre = materials.IdealMaterial(n=1.0)
        material_post = materials.IdealMaterial(n=1.5)
        jones_fresnel = jones.JonesFresnel(material_pre, material_post)

        # Test reflection calculation
        aoi = be.array([0.0])
        jones_matrix = jones_fresnel.calculate_matrix(rays, reflect=True, aoi=aoi)

        assert jones_matrix.shape == (1, 3, 3)
        assert jones_matrix[0, 0, 0] == -0.2
        assert jones_matrix[0, 1, 1] == -0.2
        assert jones_matrix[0, 2, 2] == -1.0

        # Test transmission calculation
        jones_matrix = jones_fresnel.calculate_matrix(rays, reflect=False, aoi=aoi)
        assert jones_matrix.shape == (1, 3, 3)
        assert jones_matrix[0, 0, 0] == 0.8
        assert jones_matrix[0, 1, 1] == 0.8
        assert jones_matrix[0, 2, 2] == 1.0

    def test_non_normal_incidence(self, set_test_backend):
        L = 0.1
        M = -0.25
        N = np.sqrt(1 - L**2 - M**2)
        rays = RealRays(1.0, 2.0, 3.0, L, M, N, 1.0, 1.0)

        material_pre = materials.IdealMaterial(n=1.0, k=0.0)
        material_post = materials.IdealMaterial(n=1.5, k=0.1)
        jones_fresnel = jones.JonesFresnel(material_pre, material_post)

        # Test reflection calculation
        aoi = be.array([0.2])  # use arbitrary angle of incidence
        jones_matrix = jones_fresnel.calculate_matrix(rays, reflect=True, aoi=aoi)

        assert jones_matrix.shape == (1, 3, 3)
        assert_allclose(be.real(jones_matrix[0, 0, 0]), -0.20541108217641596)
        assert_allclose(be.real(jones_matrix[0, 1, 1]), -0.19457669033430527)
        assert_allclose(be.real(jones_matrix[0, 2, 2]), -1.0)

        # Test transmission calculation
        jones_matrix = jones_fresnel.calculate_matrix(rays, reflect=False, aoi=aoi)
        assert jones_matrix.shape == (1, 3, 3)
        assert_allclose(be.real(jones_matrix[0, 0, 0]), 0.7945889178235841)
        assert_allclose(be.real(jones_matrix[0, 1, 1]), 0.7963844602228702)
        assert_allclose(be.real(jones_matrix[0, 2, 2]), 1.0)


def test_jones_polarizer_H(set_test_backend):
    rays = RealRays(
        [1.0, 2.0],
        [3.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
    )

    jones_polarizer = jones.JonesPolarizerH()
    jones_matrix = jones_polarizer.calculate_matrix(rays)

    assert jones_matrix.shape == (2, 3, 3)
    assert jones_matrix[0, 0, 0] == 1.0
    assert jones_matrix[0, 1, 1] == 0.0
    assert jones_matrix[0, 2, 2] == 1.0
    assert jones_matrix[1, 0, 0] == 1.0
    assert jones_matrix[1, 1, 1] == 0.0
    assert jones_matrix[1, 2, 2] == 1.0


def test_jones_polarizer_V(set_test_backend):
    rays = RealRays(
        [1.0, 2.0],
        [3.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
    )

    jones_polarizer = jones.JonesPolarizerV()
    jones_matrix = jones_polarizer.calculate_matrix(rays)

    assert jones_matrix.shape == (2, 3, 3)
    assert jones_matrix[0, 0, 0] == 0.0
    assert jones_matrix[0, 1, 1] == 1.0
    assert jones_matrix[0, 2, 2] == 1.0
    assert jones_matrix[1, 0, 0] == 0.0
    assert jones_matrix[1, 1, 1] == 1.0
    assert jones_matrix[1, 2, 2] == 1.0


def test_jones_polarizer_L45(set_test_backend):
    rays = RealRays(1, 2, 3, 0, 0, 1, 1, 1)

    jones_polarizer = jones.JonesPolarizerL45()
    jones_matrix = jones_polarizer.calculate_matrix(rays)

    assert jones_matrix.shape == (1, 3, 3)
    assert jones_matrix[:, 0, 0] == 0.5
    assert jones_matrix[:, 0, 1] == 0.5
    assert jones_matrix[:, 1, 0] == 0.5
    assert jones_matrix[:, 1, 1] == 0.5
    assert jones_matrix[:, 2, 2] == 1.0


def test_jones_polarizer_L135(set_test_backend):
    rays = RealRays(1, 2, 3, 0, 0, 1, 1, 1)

    jones_polarizer = jones.JonesPolarizerL135()
    jones_matrix = jones_polarizer.calculate_matrix(rays)

    assert jones_matrix.shape == (1, 3, 3)
    assert jones_matrix[:, 0, 0] == 0.5
    assert jones_matrix[:, 0, 1] == -0.5
    assert jones_matrix[:, 1, 0] == -0.5
    assert jones_matrix[:, 1, 1] == 0.5
    assert jones_matrix[:, 2, 2] == 1.0


def test_jones_polarizer_rcp(set_test_backend):
    rays = RealRays(1, 2, 3, 0, 0, 1, 1, 1)

    jones_polarizer = jones.JonesPolarizerRCP()
    jones_matrix = jones_polarizer.calculate_matrix(rays)

    assert jones_matrix.shape == (1, 3, 3)
    assert jones_matrix[0, 0, 0] == 0.5
    assert jones_matrix[0, 0, 1] == 1j * 0.5
    assert jones_matrix[0, 1, 0] == -1j * 0.5
    assert jones_matrix[0, 1, 1] == 0.5
    assert jones_matrix[0, 2, 2] == 1


def test_jones_polarizer_lcp(set_test_backend):
    rays = RealRays(1, 2, 3, 0, 0, 1, 1, 1)

    jones_polarizer = jones.JonesPolarizerLCP()
    jones_matrix = jones_polarizer.calculate_matrix(rays)

    assert jones_matrix.shape == (1, 3, 3)
    assert jones_matrix[0, 0, 0] == 0.5
    assert jones_matrix[0, 0, 1] == -1j * 0.5
    assert jones_matrix[0, 1, 0] == 1j * 0.5
    assert jones_matrix[0, 1, 1] == 0.5
    assert jones_matrix[0, 2, 2] == 1


def test_jones_linear_diattenuator(set_test_backend):
    rays = RealRays(1, 2, 3, 0, 0, 1, 1, 1)

    # Test with t_min = 0.0, t_max = 1.0, theta = 0.0
    jones_diattenuator = jones.JonesLinearDiattenuator(t_min=0.0, t_max=1.0, theta=0.0)
    jones_matrix = jones_diattenuator.calculate_matrix(rays)
    assert jones_matrix.shape == (1, 3, 3)
    assert jones_matrix[0, 0, 0] == 1.0
    assert jones_matrix[0, 0, 1] == 1.0
    assert jones_matrix[0, 1, 0] == 1.0
    assert jones_matrix[0, 2, 2] == 1.0

    # Test with t_min = 0.0, t_max = 0.5, theta = 0.0
    jones_diattenuator = jones.JonesLinearDiattenuator(t_min=0.0, t_max=0.5, theta=0.0)
    jones_matrix = jones_diattenuator.calculate_matrix(rays)
    assert jones_matrix.shape == (1, 3, 3)
    assert jones_matrix[0, 0, 0] == 0.5
    assert jones_matrix[0, 0, 1] == 0.5
    assert jones_matrix[0, 1, 0] == 0.5
    assert jones_matrix[0, 2, 2] == 1.0

    # Test with t_min = 0.2, t_max = 1.0, theta = 0.5
    jones_diattenuator = jones.JonesLinearDiattenuator(t_min=0.2, t_max=1.0, theta=0.5)
    jones_matrix = jones_diattenuator.calculate_matrix(rays)
    assert jones_matrix.shape == (1, 3, 3)
    assert jones_matrix[0, 0, 0] == 0.8161209223472559
    assert jones_matrix[0, 0, 1] == 0.9158529015192103
    assert jones_matrix[0, 1, 0] == 0.9158529015192103
    assert jones_matrix[0, 2, 2] == 1.0


def test_jones_linear_retarder(set_test_backend):
    rays = RealRays(1, 2, 3, 0, 0, 1, 1, 1)

    # Test with retardance = 0.0, theta = 0.0
    jones_retarder = jones.JonesLinearRetarder(retardance=0.0, theta=0.0)
    jones_matrix = jones_retarder.calculate_matrix(rays)
    assert jones_matrix.shape == (1, 3, 3)
    assert jones_matrix[0, 0, 0] == 1.0
    assert jones_matrix[0, 1, 1] == 1.0
    assert jones_matrix[0, 2, 2] == 1.0

    # Test with retardance = 0.5, theta = 0.0
    jones_retarder = jones.JonesLinearRetarder(retardance=0.5, theta=0.0)
    jones_matrix = jones_retarder.calculate_matrix(rays)
    assert jones_matrix.shape == (1, 3, 3)
    assert_allclose(be.real(jones_matrix[0, 0, 0]), 0.9689124217106447)
    assert_allclose(be.imag(jones_matrix[0, 0, 0]), -0.2474039592545229)
    assert_allclose(be.real(jones_matrix[0, 1, 1]), 0.9689124217106447)
    assert_allclose(be.imag(jones_matrix[0, 1, 1]), 0.24740395925452294)
    assert jones_matrix[0, 2, 2] == 1.0

    # Test with retardance = 0.5, theta = 0.5
    jones_retarder = jones.JonesLinearRetarder(retardance=0.5, theta=0.5)
    jones_matrix = jones_retarder.calculate_matrix(rays)
    assert jones_matrix.shape == (1, 3, 3)
    assert_allclose(be.real(jones_matrix[0, 0, 0]), 0.9689124217106448)
    assert_allclose(be.imag(jones_matrix[0, 0, 0]), -0.1336729296661260)
    assert_allclose(be.real(jones_matrix[0, 1, 1]), 0.9689124217106448)
    assert_allclose(be.imag(jones_matrix[0, 1, 1]), 0.13367292966612604)
    assert jones_matrix[0, 2, 2] == 1.0


def test_jones_quarter_wave_retarder(set_test_backend):
    rays = RealRays(1, 2, 3, 0, 0, 1, 1, 1)

    # Test with theta = 0.0
    jones_retarder = jones.JonesQuarterWaveRetarder(theta=0.0)
    jones_matrix = jones_retarder.calculate_matrix(rays)
    assert jones_matrix.shape == (1, 3, 3)
    assert_allclose(be.real(jones_matrix[0, 0, 0]), 0.7071067811865476)
    assert_allclose(be.imag(jones_matrix[0, 0, 0]), -0.7071067811865476)
    assert_allclose(be.real(jones_matrix[0, 1, 1]), 0.7071067811865476)
    assert_allclose(be.imag(jones_matrix[0, 1, 1]), 0.7071067811865476)
    assert jones_matrix[0, 2, 2] == 1.0

    # Test with theta = 0.5
    jones_retarder = jones.JonesQuarterWaveRetarder(theta=0.5)
    jones_matrix = jones_retarder.calculate_matrix(rays)
    assert jones_matrix.shape == (1, 3, 3)
    assert_allclose(be.real(jones_matrix[0, 0, 0]), 0.7071067811865476)
    assert_allclose(be.imag(jones_matrix[0, 0, 0]), -0.3820514243700897)
    assert_allclose(be.real(jones_matrix[0, 1, 1]), 0.7071067811865476)
    assert_allclose(be.imag(jones_matrix[0, 1, 1]), 0.38205142437008976)
    assert jones_matrix[0, 2, 2] == 1.0


def test_jones_half_wave_retarder(set_test_backend):
    rays = RealRays(1, 2, 3, 0, 0, 1, 1, 1)

    # Test with theta = 0.0
    jones_retarder = jones.JonesHalfWaveRetarder(theta=0.0)
    jones_matrix = jones_retarder.calculate_matrix(rays)
    assert jones_matrix.shape == (1, 3, 3)
    assert_allclose(be.real(jones_matrix[0, 0, 0]), 0.0)
    assert_allclose(be.imag(jones_matrix[0, 0, 0]), -1.0)
    assert_allclose(be.real(jones_matrix[0, 1, 1]), 0.0)
    assert_allclose(be.imag(jones_matrix[0, 1, 1]), 1.0)
    assert jones_matrix[0, 2, 2] == 1.0

    # Test with theta = 0.5
    jones_retarder = jones.JonesHalfWaveRetarder(theta=0.5)
    jones_matrix = jones_retarder.calculate_matrix(rays)
    assert jones_matrix.shape == (1, 3, 3)
    assert_allclose(be.real(jones_matrix[0, 0, 0]), 0.0)
    assert_allclose(be.imag(jones_matrix[0, 0, 0]), -0.5403023058681398)
    assert_allclose(be.real(jones_matrix[0, 1, 1]), 0.0)
    assert_allclose(be.imag(jones_matrix[0, 1, 1]), 0.5403023058681398)
    assert jones_matrix[0, 2, 2] == 1.0
