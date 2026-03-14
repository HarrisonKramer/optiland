"""Tests for DetectorData."""

from __future__ import annotations

import optiland.backend as be
from optiland.nonsequential.detector import DetectorData
from optiland.nonsequential.ray_data import NSQRayPool
from tests.utils import assert_allclose


class TestDetectorData:
    """Tests for the DetectorData class."""

    def _make_pool(self, n=5):
        """Create a simple ray pool for testing."""
        return NSQRayPool(
            x=be.array([float(i) for i in range(n)]),
            y=be.zeros(n),
            z=be.zeros(n),
            L=be.zeros(n),
            M=be.zeros(n),
            N=be.ones(n),
            intensity=be.ones(n),
            wavelength=be.full(n, 0.55),
        )

    def test_empty_detector(self, set_test_backend):
        """Empty detector should have zero hits."""
        det = DetectorData()
        assert det.n_hits == 0

    def test_record_and_count(self, set_test_backend):
        """Recording rays should increase hit count."""
        det = DetectorData()
        pool = self._make_pool(5)
        mask = be.array([1.0, 0.0, 1.0, 0.0, 1.0]) > 0.5
        det.record(pool, mask)
        assert det.n_hits == 3

    def test_accumulation(self, set_test_backend):
        """Multiple record calls should accumulate."""
        det = DetectorData()
        pool = self._make_pool(3)
        mask_all = be.ones(3) > 0.5
        det.record(pool, mask_all)
        det.record(pool, mask_all)
        assert det.n_hits == 6

    def test_get_positions(self, set_test_backend):
        """Positions should concatenate correctly."""
        det = DetectorData()
        pool = self._make_pool(3)
        mask = be.array([1.0, 1.0, 0.0]) > 0.5
        det.record(pool, mask)

        x, y, z = det.get_positions()
        assert_allclose(x, [0.0, 1.0])

    def test_get_intensities(self, set_test_backend):
        """Intensities should concatenate correctly."""
        det = DetectorData()
        pool = self._make_pool(3)
        mask = be.ones(3) > 0.5
        det.record(pool, mask)

        intensities = det.get_intensities()
        assert_allclose(intensities, be.ones(3))

    def test_reset(self, set_test_backend):
        """Reset should clear all data."""
        det = DetectorData()
        pool = self._make_pool(3)
        det.record(pool, be.ones(3) > 0.5)
        assert det.n_hits == 3

        det.reset()
        assert det.n_hits == 0

    def test_empty_mask_records_nothing(self, set_test_backend):
        """Recording with all-False mask should not add hits."""
        det = DetectorData()
        pool = self._make_pool(3)
        det.record(pool, be.zeros(3) > 0.5)
        assert det.n_hits == 0
