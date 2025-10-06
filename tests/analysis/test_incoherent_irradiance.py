# tests/analysis/test_incoherent_irradiance.py
"""
Tests for the IncoherentIrradiance and RadiantIntensity analysis tools.
"""
from unittest.mock import patch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import optiland.backend as be
from optiland import analysis
from optiland.optic import Optic
from optiland.physical_apertures import RectangularAperture
from optiland.rays import RealRays
from ..utils import assert_allclose

matplotlib.use("Agg")  # use non-interactive backend for testing


@pytest.fixture
def test_system_irradiance_v1():
    """Provides a simple system for basic irradiance testing."""
    optic = Optic()
    optic.add_surface(index=0, thickness=be.inf)
    optic.add_surface(index=1, thickness=0, is_stop=True)
    optic.add_surface(index=2, thickness=10)
    detector_size = RectangularAperture(x_max=2.5, x_min=-2.5, y_max=2.5, y_min=-2.5)
    optic.add_surface(index=3, aperture=detector_size)
    optic.add_wavelength(0.55)
    optic.set_field_type("angle")
    optic.add_field(y=0)
    optic.set_aperture("EPD", 5.0)
    return optic


@pytest.fixture
def perfect_mirror_system():
    """Provides a perfect parabolic mirror system that focuses to a point."""
    optic = Optic()
    optic.add_surface(index=0, thickness=be.inf)
    optic.add_surface(index=1, thickness=50)
    optic.add_surface(index=2, thickness=-25, radius=-50, conic=-1.0, material="mirror", is_stop=True)
    detector_size = RectangularAperture(x_max=2.5, x_min=-2.5, y_max=2.5, y_min=-2.5)
    optic.add_surface(index=3, aperture=detector_size)
    optic.add_wavelength(0.55)
    optic.set_field_type("angle")
    optic.add_field(y=0)
    optic.set_aperture("EPD", 5.0)
    return optic


def _create_square_grid_rays(num_rays_edge, min_coord, max_coord, wavelength_val=0.55):
    """Helper to create a grid of rays for testing."""
    x_rays = be.linspace(min_coord, max_coord, num_rays_edge)
    x, y = be.meshgrid(x_rays, x_rays)
    x_flat, y_flat = x.flatten(), y.flatten()
    num_rays = x_flat.shape[0]
    return RealRays(x=x_flat, y=y_flat, z=0, L=0, M=0, N=1, intensity=1, wavelength=wavelength_val)


def _apply_gaussian_apodization(x, y, sigma_x, sigma_y, peak_intensity=1.0):
    """Helper to apply a Gaussian intensity profile to a set of rays."""
    exponent = -(((x**2) / (2 * sigma_x**2)) + ((y**2) / (2 * sigma_y**2)))
    return peak_intensity * be.exp(exponent)


class TestIncoherentIrradiance:
    """
    Tests the IncoherentIrradiance analysis tool, which simulates the
    energy distribution on a detector surface.
    """

    def test_irradiance_v1_uniform_and_user_defined_rays(self, set_test_backend, test_system_irradiance_v1):
        """
        Tests irradiance calculation for both a uniform random ray distribution
        and a user-defined grid of rays.
        """
        irr_uniform = analysis.IncoherentIrradiance(test_system_irradiance_v1, num_rays=500, distribution="uniform", res=(5, 5))
        irr_map_uniform, _, _ = irr_uniform.data[0][0]
        assert be.sum(irr_map_uniform) > 0

        user_rays = _create_square_grid_rays(num_rays_edge=5, min_coord=-2.25, max_coord=2.25)
        irr_user = analysis.IncoherentIrradiance(test_system_irradiance_v1, res=(5, 5), user_initial_rays=user_rays)
        irr_map_user, _, _ = irr_user.data[0][0]
        pixel_area = 1.0  # (5mm / 5px) * (5mm / 5px)
        total_power_on_detector = be.sum(irr_map_user) * pixel_area
        initial_total_power = be.sum(user_rays.i)
        assert_allclose(total_power_on_detector, initial_total_power, atol=1e-5)

    def test_irradiance_gaussian_apodization(self, set_test_backend, test_system_irradiance_v1):
        """
        Tests irradiance calculation with a Gaussian apodized source.
        """
        res = (50, 50)
        user_rays = _create_square_grid_rays(num_rays_edge=100, min_coord=-2.5, max_coord=2.5)
        user_rays.i = _apply_gaussian_apodization(user_rays.x, user_rays.y, sigma_x=0.5, sigma_y=0.5)
        irr_apodized = analysis.IncoherentIrradiance(test_system_irradiance_v1, res=res, user_initial_rays=user_rays)
        irr_map, _, _ = irr_apodized.data[0][0]
        center_val = irr_map[res[0] // 2, res[1] // 2]
        assert be.to_numpy(center_val) > be.to_numpy(irr_map[0, 0])
        assert_allclose(center_val, be.max(irr_map), rtol=1e-3)

    def test_irradiance_perfect_mirror_focus(self, set_test_backend, perfect_mirror_system):
        """
        Tests that a perfect parabolic mirror focuses all rays to a single point,
        and that the bilinear interpolation correctly distributes the energy to
        the four adjacent pixels.
        """
        res = (21, 21)
        user_rays = _create_square_grid_rays(num_rays_edge=51, min_coord=-2.5, max_coord=2.5)
        irr_perfect = analysis.IncoherentIrradiance(perfect_mirror_system, res=res, user_initial_rays=user_rays)
        irr_map, _, _ = irr_perfect.data[0][0]
        total_sum = be.to_numpy(be.sum(irr_map))
        center_x, center_y = res[0] // 2, res[1] // 2
        central_four_sum = be.to_numpy(irr_map[center_x-1:center_x+1, center_y-1:center_y+1].sum())
        assert_allclose(central_four_sum, total_sum, atol=1e-5)

    def test_irradiance_plot_cross_section(self, set_test_backend, test_system_irradiance_v1):
        """Tests the cross-section plotting functionality."""
        irr = analysis.IncoherentIrradiance(test_system_irradiance_v1, res=(50, 50), user_initial_rays=_create_square_grid_rays(100, -2.5, 2.5))
        fig, _ = irr.view(cross_section=("cross-x", 25))
        assert fig is not None
        plt.close(fig)

    def test_detector_surface_no_aperture(self, set_test_backend):
        """
        Tests that an error is raised if the specified detector surface has no
        physical aperture defined.
        """
        optic = Optic()
        optic.add_surface(index=0, thickness=10)
        optic.add_surface(index=1) # No aperture
        with pytest.raises(ValueError, match="Detector surface has no physical aperture"):
            analysis.IncoherentIrradiance(optic, num_rays=5, res=(5, 5))

    def test_irradiance_autodiff(self, set_test_backend):
        """Tests that the irradiance calculation is differentiable with torch."""
        if be.get_backend() != "torch":
            pytest.skip("Autodiff test only runs for torch backend")

        be.grad_mode.enable()
        optic = Optic()
        optic.add_surface(index=0, thickness=be.inf)
        radius_tensor = be.array(20.0, requires_grad=True)
        optic.add_surface(index=1, thickness=7, radius=radius_tensor, is_stop=True, material="bk7")
        optic.add_surface(index=2, thickness=10)
        optic.add_surface(index=3, aperture=RectangularAperture(-2.5, 2.5, -2.5, 2.5))
        optic.add_wavelength(0.55)
        optic.set_aperture("EPD", 5.0)

        irr_analysis = analysis.IncoherentIrradiance(optic, user_initial_rays=_create_square_grid_rays(10, -2.5, 2.5), res=(10, 10))
        irr_map, _, _ = irr_analysis.data[0][0]
        loss = be.sum(irr_map**2)
        loss.backward()

        grad = optic.surface_group.surfaces[1].geometry.radius.grad
        assert grad is not None
        assert be.to_numpy(grad) != 0

    def test_view_normalize_true_peak_zero(self, set_test_backend, test_system_irradiance_v1):
        """
        Tests that normalization in the view method handles a case where the
        peak irradiance is zero without errors.
        """
        irr = analysis.IncoherentIrradiance(test_system_irradiance_v1, num_rays=1, res=(5, 5))
        dummy_edges = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
        irr.data = [[(be.zeros((5, 5)), dummy_edges, dummy_edges)]]
        fig, _ = irr.view(normalize=True)
        assert fig is not None
        plt.close(fig)