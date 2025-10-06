# tests/distribution/test_distribution.py
"""
Tests for the main Distribution class in optiland.distribution.
"""
import pytest

import optiland.backend as be
from optiland import distribution
from ..utils import assert_allclose


class TestDistribution:
    """
    Tests the high-level distribution creation and point generation functionality.
    """

    def test_random(self, set_test_backend):
        """
        Tests the 'random' distribution to ensure it generates the correct
        number of points within the unit circle.
        """
        dist = distribution.create_distribution("random")
        dist.generate_points(num_points=100)
        assert len(dist.x) == 100
        assert len(dist.y) == 100
        assert be.all(dist.x**2 + dist.y**2 <= 1)

    def test_grid(self, set_test_backend):
        """
        Tests the 'grid' (uniform) distribution. A 10x10 grid is generated,
        and only points inside the unit circle are kept.
        """
        dist = distribution.create_distribution("uniform")
        dist.generate_points(num_points=10)
        assert len(dist.x) == 81
        assert len(dist.y) == 81
        assert be.all(dist.x**2 + dist.y**2 <= 1)

    def test_hexapolar(self, set_test_backend):
        """
        Tests the 'hexapolar' distribution. For 3 rings, it should generate
        37 points.
        """
        dist = distribution.create_distribution("hexapolar")
        dist.generate_points(num_rings=3)
        assert len(dist.x) == 37
        assert len(dist.y) == 37
        assert be.all(dist.x**2 + dist.y**2 <= 1)

    def test_invalid_distribution(self, set_test_backend):
        """
        Tests that creating an unrecognized distribution type raises a ValueError.
        """
        with pytest.raises(ValueError):
            distribution.create_distribution("invalid_type")

    def test_central_obstruction(self, set_test_backend):
        """
        Tests that a central obstruction correctly excludes points from the
        center of the pupil.
        """
        # Instantiate RandomDistribution directly to test this constructor argument
        dist = distribution.RandomDistribution(central_obstruction=0.5)
        dist.generate_points(num_points=1000)
        assert len(dist.x) > 0
        assert be.all(dist.x**2 + dist.y**2 >= 0.5**2)

    def test_no_rays(self, set_test_backend):
        """
        Tests that generating 0 points results in empty coordinate arrays.
        """
        dist = distribution.create_distribution("random")
        dist.generate_points(num_points=0)
        assert len(dist.x) == 0
        assert len(dist.y) == 0

    def test_single_ray(self, set_test_backend):
        """
        Tests that generating 1 point places it at the origin (0, 0).
        """
        dist = distribution.create_distribution("random")
        dist.generate_points(num_points=1)
        assert len(dist.x) == 1
        assert_allclose(dist.x[0], 0.0)
        assert_allclose(dist.y[0], 0.0)

    # The following tests are commented out as the feature of creating a
    # distribution directly from a list of points appears to have been removed
    # from the high-level API.

    # def test_list_distribution(self, set_test_backend):
    #     x = be.asarray([0.1, 0.2, 0.3])
    #     y = be.asarray([0.4, 0.5, 0.6])
    #     dist = Distribution(distribution=[x, y])
    #     assert be.all(dist.x == x)
    #     assert be.all(dist.y == y)

    # def test_invalid_list_distribution(self, set_test_backend):
    #     with pytest.raises(ValueError):
    #         Distribution(distribution=[1, 2, 3])

    # def test_list_distribution_with_num_rays(self, set_test_backend):
    #     x = be.asarray([0.1, 0.2])
    #     y = be.asarray([0.4, 0.5])
    #     dist = Distribution(num_rays=100, distribution=[x, y])
    #     assert len(dist.x) == 2
    #     assert len(dist.y) == 2