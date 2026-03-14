"""Tests for NSQRayPool."""

from __future__ import annotations

import optiland.backend as be
from optiland.nonsequential.ray_data import NSQRayPool
from tests.utils import assert_allclose


class TestNSQRayPool:
    """Tests for the NSQRayPool class."""

    def _make_pool(self, n=5, record_paths=False):
        """Create a simple ray pool for testing."""
        return NSQRayPool(
            x=be.zeros(n),
            y=be.zeros(n),
            z=be.zeros(n),
            L=be.zeros(n),
            M=be.zeros(n),
            N=be.ones(n),
            intensity=be.ones(n),
            wavelength=be.full(n, 0.55),
            record_paths=record_paths,
        )

    def test_initial_state(self, set_test_backend):
        """All rays should start active."""
        pool = self._make_pool(5)
        assert pool.n_active == 5
        assert be.all(pool.active)

    def test_deactivate(self, set_test_backend):
        """Deactivate should mark rays inactive."""
        pool = self._make_pool(5)
        mask = be.array([1.0, 0.0, 1.0, 0.0, 0.0]) > 0.5
        pool.deactivate(mask)
        assert pool.n_active == 3

    def test_propagate(self, set_test_backend):
        """Propagation should advance position along direction."""
        pool = self._make_pool(3)
        distances = be.array([1.0, 2.0, 3.0])
        pool.propagate(distances)

        assert_allclose(pool.z, [1.0, 2.0, 3.0])
        assert_allclose(pool.x, [0.0, 0.0, 0.0])

    def test_propagate_respects_active(self, set_test_backend):
        """Inactive rays should not be propagated."""
        pool = self._make_pool(3)
        pool.deactivate(be.array([0.0, 1.0, 0.0]) > 0.5)
        distances = be.array([1.0, 2.0, 3.0])
        pool.propagate(distances)

        assert_allclose(pool.z, [1.0, 0.0, 3.0])

    def test_update_directions(self, set_test_backend):
        """Direction update should only apply where mask is True."""
        pool = self._make_pool(3)
        mask = be.array([1.0, 0.0, 1.0]) > 0.5
        pool.update_directions(
            be.array([1.0, 0.0, 0.0]),
            be.array([0.0, 0.0, 0.0]),
            be.array([0.0, 0.0, 1.0]),
            mask,
        )

        assert_allclose(pool.L, [1.0, 0.0, 0.0])
        assert_allclose(pool.N, [0.0, 1.0, 1.0])

    def test_apply_intensity(self, set_test_backend):
        """Intensity should be multiplied only where mask is True."""
        pool = self._make_pool(3)
        mask = be.array([1.0, 1.0, 0.0]) > 0.5
        pool.apply_intensity(be.array([0.5, 0.5, 0.5]), mask)

        assert_allclose(pool.intensity, [0.5, 0.5, 1.0])

    def test_path_recording(self, set_test_backend):
        """Path recording should capture state snapshots."""
        pool = self._make_pool(3, record_paths=True)
        surface_ids = be.array([0.0, 1.0, 0.0])
        pool.record_path_point(surface_ids)

        assert pool.path_history is not None
        assert len(pool.path_history) == 1

    def test_no_path_recording(self, set_test_backend):
        """Without record_paths, path_history should be None."""
        pool = self._make_pool(3, record_paths=False)
        pool.record_path_point(be.array([0.0, 0.0, 0.0]))
        assert pool.path_history is None
