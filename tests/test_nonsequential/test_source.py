"""Tests for PointSource."""

from __future__ import annotations

import numpy as np

import optiland.backend as be
from optiland.nonsequential.source import PointSource
from tests.utils import assert_allclose


class TestPointSource:
    """Tests for the PointSource class."""

    def test_collimated_source(self, set_test_backend):
        """Collimated source (half_angle=0) should emit parallel rays."""
        src = PointSource(
            position=(0, 0, 0),
            direction=(0, 0, 1),
            half_angle=0.0,
            wavelength=0.55,
        )
        rays = src.generate_rays(100)

        assert_allclose(rays.L, be.zeros(100), atol=1e-12)
        assert_allclose(rays.M, be.zeros(100), atol=1e-12)
        assert_allclose(rays.N, be.ones(100), atol=1e-12)

    def test_source_position(self, set_test_backend):
        """Rays should originate from the source position."""
        src = PointSource(
            position=(1, 2, 3),
            direction=(0, 0, 1),
            half_angle=0.0,
        )
        rays = src.generate_rays(10)

        assert_allclose(rays.x, be.full(10, 1.0))
        assert_allclose(rays.y, be.full(10, 2.0))
        assert_allclose(rays.z, be.full(10, 3.0))

    def test_cone_directions_normalized(self, set_test_backend):
        """All generated rays should have unit direction vectors."""
        src = PointSource(
            position=(0, 0, 0),
            direction=(0, 0, 1),
            half_angle=np.radians(30),
        )
        rays = src.generate_rays(1000)

        mag = be.sqrt(rays.L**2 + rays.M**2 + rays.N**2)
        assert_allclose(mag, be.ones(1000), atol=1e-10)

    def test_cone_angle_bounds(self, set_test_backend):
        """Rays should be within the specified half-angle cone."""
        half_angle = np.radians(20)
        src = PointSource(
            position=(0, 0, 0),
            direction=(0, 0, 1),
            half_angle=half_angle,
        )
        rays = src.generate_rays(10000)

        # cos(angle) = dot(ray_dir, cone_axis) = N for z-axis cone
        cos_angles = be.to_numpy(rays.N)
        min_cos = np.cos(half_angle)

        # All rays should have cos(angle) >= cos(half_angle)
        assert np.all(cos_angles >= min_cos - 1e-10)

    def test_wavelength_assignment(self, set_test_backend):
        """Rays should have the source wavelength."""
        src = PointSource(wavelength=0.633)
        rays = src.generate_rays(10)

        assert_allclose(rays.wavelength, be.full(10, 0.633))

    def test_initial_intensity(self, set_test_backend):
        """Rays should start with unit intensity."""
        src = PointSource()
        rays = src.generate_rays(10)

        assert_allclose(rays.intensity, be.ones(10))

    def test_off_axis_direction(self, set_test_backend):
        """Source with non-z direction should emit in correct direction."""
        src = PointSource(
            direction=(1, 0, 0),
            half_angle=0.0,
        )
        rays = src.generate_rays(10)

        assert_allclose(rays.L, be.ones(10), atol=1e-12)
        assert_allclose(rays.M, be.zeros(10), atol=1e-12)
        assert_allclose(rays.N, be.zeros(10), atol=1e-12)

    def test_path_recording_flag(self, set_test_backend):
        """Record paths flag should propagate to ray pool."""
        src = PointSource(record_paths=True)
        rays = src.generate_rays(5)
        assert rays.path_history is not None

        src2 = PointSource(record_paths=False)
        rays2 = src2.generate_rays(5)
        assert rays2.path_history is None
