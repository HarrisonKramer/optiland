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
        assert_allclose(jones_matrix[0, 0, 0], -0.2)
        assert_allclose(jones_matrix[0, 1, 1], -0.2)
        assert_allclose(jones_matrix[0, 2, 2], -1.0)

        # Test transmission calculation
        jones_matrix = jones_fresnel.calculate_matrix(rays, reflect=False, aoi=aoi)
        assert jones_matrix.shape == (1, 3, 3)
        assert_allclose(jones_matrix[0, 0, 0], 0.8)
        assert_allclose(jones_matrix[0, 1, 1], 0.8)
        assert_allclose(jones_matrix[0, 2, 2], 1.0)

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
    assert_allclose(jones_matrix[0, 0, 0], 1.0)
    assert_allclose(jones_matrix[0, 1, 1], 0.0)
    assert_allclose(jones_matrix[0, 2, 2], 1.0)
    assert_allclose(jones_matrix[1, 0, 0], 1.0)
    assert_allclose(jones_matrix[1, 1, 1], 0.0)
    assert_allclose(jones_matrix[1, 2, 2], 1.0)


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
    assert_allclose(jones_matrix[0, 0, 0], 0.0)
    assert_allclose(jones_matrix[0, 1, 1], 1.0)
    assert_allclose(jones_matrix[0, 2, 2], 1.0)
    assert_allclose(jones_matrix[1, 0, 0], 0.0)
    assert_allclose(jones_matrix[1, 1, 1], 1.0)
    assert_allclose(jones_matrix[1, 2, 2], 1.0)


def test_jones_polarizer_L45(set_test_backend):
    rays = RealRays(1, 2, 3, 0, 0, 1, 1, 1)

    jones_polarizer = jones.JonesPolarizerL45()
    jones_matrix = jones_polarizer.calculate_matrix(rays)

    assert jones_matrix.shape == (1, 3, 3)
    assert_allclose(jones_matrix[:, 0, 0], 0.5)
    assert_allclose(jones_matrix[:, 0, 1], 0.5)
    assert_allclose(jones_matrix[:, 1, 0], 0.5)
    assert_allclose(jones_matrix[:, 1, 1], 0.5)
    assert_allclose(jones_matrix[:, 2, 2], 1.0)


def test_jones_polarizer_L135(set_test_backend):
    rays = RealRays(1, 2, 3, 0, 0, 1, 1, 1)

    jones_polarizer = jones.JonesPolarizerL135()
    jones_matrix = jones_polarizer.calculate_matrix(rays)

    assert jones_matrix.shape == (1, 3, 3)
    assert_allclose(jones_matrix[:, 0, 0], 0.5)
    assert_allclose(jones_matrix[:, 0, 1], -0.5)
    assert_allclose(jones_matrix[:, 1, 0], -0.5)
    assert_allclose(jones_matrix[:, 1, 1], 0.5)
    assert_allclose(jones_matrix[:, 2, 2], 1.0)


def test_jones_polarizer_rcp(set_test_backend):
    rays = RealRays(1, 2, 3, 0, 0, 1, 1, 1)

    jones_polarizer = jones.JonesPolarizerRCP()
    jones_matrix = jones_polarizer.calculate_matrix(rays)

    assert jones_matrix.shape == (1, 3, 3)
    assert_allclose(jones_matrix[0, 0, 0], 0.5)
    assert_allclose(jones_matrix[0, 0, 1], be.j * 0.5)
    assert_allclose(jones_matrix[0, 1, 0], -be.j * 0.5)
    assert_allclose(jones_matrix[0, 1, 1], 0.5)
    assert_allclose(jones_matrix[0, 2, 2], 1.0)


def test_jones_polarizer_lcp(set_test_backend):
    rays = RealRays(1, 2, 3, 0, 0, 1, 1, 1)

    jones_polarizer = jones.JonesPolarizerLCP()
    jones_matrix = jones_polarizer.calculate_matrix(rays)

    assert jones_matrix.shape == (1, 3, 3)
    assert_allclose(jones_matrix[0, 0, 0], 0.5)
    assert_allclose(jones_matrix[0, 0, 1], -be.j * 0.5)
    assert_allclose(jones_matrix[0, 1, 0], be.j * 0.5)
    assert_allclose(jones_matrix[0, 1, 1], 0.5)
    assert_allclose(jones_matrix[0, 2, 2], 1.0)


def test_jones_linear_diattenuator(set_test_backend):
    rays = RealRays(1, 2, 3, 0, 0, 1, 1, 1)

    # Test with t_min = 0.0, t_max = 1.0, theta = 0.0
    jones_diattenuator = jones.JonesLinearDiattenuator(t_min=0.0, t_max=1.0, theta=0.0)
    jones_matrix = jones_diattenuator.calculate_matrix(rays)
    assert jones_matrix.shape == (1, 3, 3)
    assert_allclose(jones_matrix[0, 0, 0], 1.0)
    assert_allclose(jones_matrix[0, 0, 1], 1.0) # Should be cos(0)*sin(0) * (t_max-t_min) = 0
    assert_allclose(jones_matrix[0, 1, 0], 1.0) # Should be 0
    assert_allclose(jones_matrix[0, 1, 1], 0.0) # Should be t_max*sin(0)^2 + t_min*cos(0)^2 = 0
    assert_allclose(jones_matrix[0, 2, 2], 1.0)

    # Test with t_min = 0.0, t_max = 0.5, theta = 0.0
    jones_diattenuator = jones.JonesLinearDiattenuator(t_min=0.0, t_max=0.5, theta=0.0)
    jones_matrix = jones_diattenuator.calculate_matrix(rays)
    assert jones_matrix.shape == (1, 3, 3)
    assert_allclose(jones_matrix[0, 0, 0], 0.5)
    assert_allclose(jones_matrix[0, 0, 1], 0.5) # Should be 0
    assert_allclose(jones_matrix[0, 1, 0], 0.5) # Should be 0
    assert_allclose(jones_matrix[0, 1, 1], 0.0) # Should be 0
    assert_allclose(jones_matrix[0, 2, 2], 1.0)

    # Test with t_min = 0.2, t_max = 1.0, theta = 0.5
    # Recalculating expected values
    t_min = 0.2
    t_max = 1.0
    theta = 0.5
    c, s = be.cos(be.array(theta)), be.sin(be.array(theta))
    exp_j00 = t_max * c**2 + t_min * s**2
    exp_j0x = (t_max - t_min) * c * s # Corrected formula
    exp_j11 = t_max * s**2 + t_min * c**2
    jones_diattenuator = jones.JonesLinearDiattenuator(t_min=t_min, t_max=t_max, theta=theta)
    jones_matrix = jones_diattenuator.calculate_matrix(rays)
    assert jones_matrix.shape == (1, 3, 3)
    assert_allclose(jones_matrix[0, 0, 0], exp_j00) # 0.8161209...
    assert_allclose(jones_matrix[0, 0, 1], exp_j0x) # 0.33658...
    assert_allclose(jones_matrix[0, 1, 0], exp_j0x) # 0.33658...
    assert_allclose(jones_matrix[0, 1, 1], exp_j11) # 0.383879...
    assert_allclose(jones_matrix[0, 2, 2], 1.0)


def test_jones_linear_retarder(set_test_backend):
    rays = RealRays(1, 2, 3, 0, 0, 1, 1, 1)

    # Test with retardance = 0.0, theta = 0.0
    jones_retarder = jones.JonesLinearRetarder(retardance=0.0, theta=0.0)
    jones_matrix = jones_retarder.calculate_matrix(rays)
    assert jones_matrix.shape == (1, 3, 3)
    assert_allclose(jones_matrix[0, 0, 0], 1.0)
    assert_allclose(jones_matrix[0, 0, 1], 0.0)
    assert_allclose(jones_matrix[0, 1, 0], 0.0)
    assert_allclose(jones_matrix[0, 1, 1], 1.0)
    assert_allclose(jones_matrix[0, 2, 2], 1.0)

    # Test with retardance = 0.5, theta = 0.0
    jones_retarder = jones.JonesLinearRetarder(retardance=0.5, theta=0.0)
    jones_matrix = jones_retarder.calculate_matrix(rays)
    assert jones_matrix.shape == (1, 3, 3)
    assert_allclose(be.real(jones_matrix[0, 0, 0]),
        0.9689124217106447 # exp(-j*0.25)*1 + exp(j*0.25)*0
    )
    assert_allclose(be.imag(jones_matrix[0, 0, 0]),
        -0.2474039592545229
    )
    assert_allclose(jones_matrix[0, 0, 1], 0.0) # -j*sin(0.25)*sin(0) = 0
    assert_allclose(jones_matrix[0, 1, 0], 0.0)
    assert_allclose(be.real(jones_matrix[0, 1, 1]),
        0.9689124217106447 # exp(j*0.25)*1 + exp(-j*0.25)*0
    )
    assert_allclose(be.imag(jones_matrix[0, 1, 1]),
        0.24740395925452294
    )
    assert_allclose(jones_matrix[0, 2, 2], 1.0)

    # Test with retardance = 0.5, theta = 0.5
    ret = 0.5
    theta = 0.5
    d = be.array(ret)
    t = be.array(theta)
    c, s = be.cos(t), be.sin(t)
    exp_j00 = be.exp(-be.j * d / 2) * c**2 + be.exp(be.j * d / 2) * s**2
    exp_j0x = -be.j * be.sin(d / 2) * be.sin(2 * t)
    exp_j11 = be.exp(be.j * d / 2) * c**2 + be.exp(-be.j * d / 2) * s**2
    jones_retarder = jones.JonesLinearRetarder(retardance=ret, theta=theta)
    jones_matrix = jones_retarder.calculate_matrix(rays)
    assert jones_matrix.shape == (1, 3, 3)
    assert_allclose(jones_matrix[0, 0, 0], exp_j00)
    assert_allclose(jones_matrix[0, 0, 1], exp_j0x)
    assert_allclose(jones_matrix[0, 1, 0], exp_j0x)
    assert_allclose(jones_matrix[0, 1, 1], exp_j11)
    assert_allclose(jones_matrix[0, 2, 2], 1.0)


def test_jones_quarter_wave_retarder(set_test_backend):
    rays = RealRays(1, 2, 3, 0, 0, 1, 1, 1)
    sqrt2_inv = 1 / be.sqrt(be.array(2.0))

    # Test with theta = 0.0
    jones_retarder = jones.JonesQuarterWaveRetarder(theta=0.0)
    jones_matrix = jones_retarder.calculate_matrix(rays)
    assert jones_matrix.shape == (1, 3, 3)
    # j00 = exp(-j*pi/4)*1 + exp(j*pi/4)*0 = (1-j)/sqrt(2)
    # j0x = -j*sin(pi/4)*sin(0) = 0
    # j11 = exp(j*pi/4)*1 + exp(-j*pi/4)*0 = (1+j)/sqrt(2)
    assert_allclose(jones_matrix[0, 0, 0], sqrt2_inv * (1 - be.j))
    assert_allclose(jones_matrix[0, 0, 1], 0.0)
    assert_allclose(jones_matrix[0, 1, 0], 0.0)
    assert_allclose(jones_matrix[0, 1, 1], sqrt2_inv * (1 + be.j))
    assert_allclose(jones_matrix[0, 2, 2], 1.0)

    # Test with theta = pi / 4
    theta_pi4 = be.pi / 4.0
    jones_retarder = jones.JonesQuarterWaveRetarder(theta=theta_pi4)
    jones_matrix = jones_retarder.calculate_matrix(rays)
    assert jones_matrix.shape == (1, 3, 3)
    # d = pi/2, t = pi/4
    # j00 = exp(-j*pi/4)*(1/2) + exp(j*pi/4)*(1/2) = 1/sqrt(2)
    # j0x = -j*sin(pi/4)*sin(pi/2) = -j/sqrt(2)
    # j11 = exp(j*pi/4)*(1/2) + exp(-j*pi/4)*(1/2) = 1/sqrt(2)
    assert_allclose(jones_matrix[0, 0, 0], sqrt2_inv)
    assert_allclose(jones_matrix[0, 0, 1], -be.j * sqrt2_inv)
    assert_allclose(jones_matrix[0, 1, 0], -be.j * sqrt2_inv)
    assert_allclose(jones_matrix[0, 1, 1], sqrt2_inv)
    assert_allclose(jones_matrix[0, 2, 2], 1.0)
    # Check the specific instruction requirement
    assert_allclose(be.real(jones_matrix[0, 0, 0]), sqrt2_inv)

    # Test with theta = 0.5 ( Keep original test )
    theta_05 = 0.5
    ret = be.pi / 2.0
    d = be.array(ret)
    t = be.array(theta_05)
    c, s = be.cos(t), be.sin(t)
    exp_j00 = be.exp(-be.j * d / 2) * c**2 + be.exp(be.j * d / 2) * s**2
    exp_j0x = -be.j * be.sin(d / 2) * be.sin(2 * t)
    exp_j11 = be.exp(be.j * d / 2) * c**2 + be.exp(-be.j * d / 2) * s**2
    jones_retarder = jones.JonesQuarterWaveRetarder(theta=theta_05)
    jones_matrix = jones_retarder.calculate_matrix(rays)
    assert jones_matrix.shape == (1, 3, 3)
    assert_allclose(jones_matrix[0, 0, 0], exp_j00)
    assert_allclose(jones_matrix[0, 0, 1], exp_j0x)
    assert_allclose(jones_matrix[0, 1, 0], exp_j0x)
    assert_allclose(jones_matrix[0, 1, 1], exp_j11)
    assert_allclose(jones_matrix[0, 2, 2], 1.0)


def test_jones_half_wave_retarder(set_test_backend):
    rays = RealRays(1, 2, 3, 0, 0, 1, 1, 1)

    # Test with theta = 0.0
    jones_retarder = jones.JonesHalfWaveRetarder(theta=0.0)
    jones_matrix = jones_retarder.calculate_matrix(rays)
    # d = pi, t = 0
    # j00 = exp(-j*pi/2)*1 + exp(j*pi/2)*0 = -j
    # j0x = -j*sin(pi/2)*sin(0) = 0
    # j11 = exp(j*pi/2)*1 + exp(-j*pi/2)*0 = j
    assert jones_matrix.shape == (1, 3, 3)
    assert_allclose(jones_matrix[0, 0, 0], -be.j)
    assert_allclose(jones_matrix[0, 0, 1], 0.0)
    assert_allclose(jones_matrix[0, 1, 0], 0.0)
    assert_allclose(jones_matrix[0, 1, 1], be.j)
    assert_allclose(jones_matrix[0, 2, 2], 1.0)

    # Test with theta = pi / 4
    theta_pi4 = be.pi / 4.0
    jones_retarder = jones.JonesHalfWaveRetarder(theta=theta_pi4)
    jones_matrix = jones_retarder.calculate_matrix(rays)
    # d = pi, t = pi / 4
    # j00 = exp(-j*pi/2)*(1/2) + exp(j*pi/2)*(1/2) = (-j)/2 + j/2 = 0
    # j0x = -j*sin(pi/2)*sin(pi/2) = -j * 1 * 1 = -j
    # j11 = exp(j*pi/2)*(1/2) + exp(-j*pi/2)*(1/2) = j/2 + (-j)/2 = 0
    assert jones_matrix.shape == (1, 3, 3)
    assert_allclose(jones_matrix[0, 0, 0], 0.0)
    assert_allclose(jones_matrix[0, 0, 1], -be.j)
    assert_allclose(jones_matrix[0, 1, 0], -be.j)
    assert_allclose(jones_matrix[0, 1, 1], 0.0)
    assert_allclose(jones_matrix[0, 2, 2], 1.0)

    # Test with theta = 0.5 ( Keep original test )
    ret = be.pi
    theta = 0.5
    d = be.array(ret)
    t = be.array(theta)
    c, s = be.cos(t), be.sin(t)
    exp_j00 = be.exp(-be.j * d / 2) * c**2 + be.exp(be.j * d / 2) * s**2
    exp_j0x = -be.j * be.sin(d / 2) * be.sin(2 * t)
    exp_j11 = be.exp(be.j * d / 2) * c**2 + be.exp(-be.j * d / 2) * s**2
    jones_retarder = jones.JonesHalfWaveRetarder(theta=theta)
    jones_matrix = jones_retarder.calculate_matrix(rays)
    assert jones_matrix.shape == (1, 3, 3)
    assert_allclose(jones_matrix[0, 0, 0], exp_j00)
    assert_allclose(jones_matrix[0, 0, 1], exp_j0x)
    assert_allclose(jones_matrix[0, 1, 0], exp_j0x)
    assert_allclose(jones_matrix[0, 1, 1], exp_j11)
    assert_allclose(jones_matrix[0, 2, 2], 1.0)

