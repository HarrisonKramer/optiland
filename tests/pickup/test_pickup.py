# tests/pickup/test_pickup.py
"""
Tests for the Pickup and PickupManager classes in optiland.pickup.
"""
import pytest

from optiland.optic import Optic
from optiland.pickup import Pickup, PickupManager
from ..utils import assert_allclose


class TestPickup:
    """
    Tests the functionality of the individual Pickup class, which represents
    a single constraint between two surface properties.
    """

    @pytest.fixture
    def setup_optic(self):
        """
        Provides a basic Optic instance with two surfaces for testing pickups.
        """
        optic = Optic()
        optic.add_surface(index=0, radius=10, thickness=5, conic=-0.5)
        optic.add_surface(index=1, radius=20, thickness=10)
        return optic

    def test_apply_radius(self, set_test_backend, setup_optic):
        """
        Tests that a radius pickup correctly applies the scaling and offset.
        """
        pickup = Pickup(setup_optic, 0, "radius", 1, scale=2, offset=3)
        pickup.apply()
        # Expected: target_radius = 10 * 2 + 3 = 23
        assert_allclose(setup_optic.surface_group.surfaces[1].geometry.radius, 23.0)

    def test_apply_conic(self, set_test_backend, setup_optic):
        """
        Tests that a conic constant pickup correctly applies the scaling and
        offset.
        """
        pickup = Pickup(setup_optic, 0, "conic", 1, scale=2, offset=3)
        pickup.apply()
        # Expected: target_conic = -0.5 * 2 + 3 = 2.0
        assert_allclose(setup_optic.surface_group.surfaces[1].geometry.k, 2.0)

    def test_apply_thickness(self, set_test_backend, setup_optic):
        """
        Tests that a thickness pickup correctly applies the scaling and offset.
        """
        pickup = Pickup(setup_optic, 0, "thickness", 1, scale=1, offset=0)
        pickup.apply()
        # Expected: target_thickness = source_thickness = 5
        assert_allclose(setup_optic.surface_group.get_thickness(1), 5.0)

    def test_invalid_attr_type(self, set_test_backend, setup_optic):
        """
        Tests that attempting to apply a pickup with an invalid attribute type
        raises a ValueError.
        """
        pickup = Pickup(setup_optic, 0, "invalid_attr", 1)
        with pytest.raises(ValueError):
            pickup.apply()

    def test_to_dict(self, set_test_backend, setup_optic):
        """
        Tests the serialization of a Pickup instance to a dictionary.
        """
        pickup = Pickup(setup_optic, 0, "radius", 1, scale=2, offset=3)
        d = pickup.to_dict()
        expected = {
            "source_surface_idx": 0,
            "attr_type": "radius",
            "target_surface_idx": 1,
            "scale": 2,
            "offset": 3,
        }
        assert d == expected

    def test_from_dict(self, set_test_backend, setup_optic):
        """
        Tests the deserialization of a Pickup instance from a dictionary.
        """
        d = {
            "source_surface_idx": 0,
            "attr_type": "radius",
            "target_surface_idx": 1,
            "scale": 2,
            "offset": 3,
        }
        pickup = Pickup.from_dict(d, optic=setup_optic)
        assert pickup.source_surface_idx == 0
        assert pickup.attr_type == "radius"
        assert pickup.target_surface_idx == 1
        assert pickup.scale == 2
        assert pickup.offset == 3


class TestPickupManager:
    """
    Tests the PickupManager class, which manages a collection of Pickups for
    an optical system.
    """

    @pytest.fixture
    def setup_manager(self):
        """
        Provides an Optic instance and an associated PickupManager for testing.
        """
        optic = Optic()
        optic.add_surface(index=0, radius=10, thickness=5)
        optic.add_surface(index=1, radius=20, thickness=10)
        optic.add_surface(index=2, radius=30, thickness=15)
        return optic, optic.pickups

    def test_add_pickup(self, set_test_backend, setup_manager):
        """Tests adding a new pickup to the manager."""
        _, manager = setup_manager
        manager.add(0, "radius", 1, scale=2, offset=3)
        assert len(manager) == 1
        assert manager[0].attr_type == "radius"

    def test_apply_pickups(self, set_test_backend, setup_manager):
        """
        Tests that `apply_all` correctly executes all pickups in the manager.
        """
        optic, manager = setup_manager
        manager.add(0, "radius", 1, scale=2, offset=3)
        manager.apply()
        # Expected: radius of surf 1 = 10 * 2 + 3 = 23
        assert_allclose(optic.surface_group.surfaces[1].geometry.radius, 23.0)

    def test_clear_pickups(self, set_test_backend, setup_manager):
        """Tests clearing all pickups from the manager."""
        _, manager = setup_manager
        manager.add(0, "radius", 1)
        manager.clear()
        assert len(manager) == 0

    def test_to_dict(self, set_test_backend, setup_manager):
        """Tests serializing the entire PickupManager to a list of dicts."""
        _, manager = setup_manager
        manager.add(0, "radius", 1, scale=2, offset=3)
        d = manager.to_dict()
        assert len(d) == 1
        assert d[0]["attr_type"] == "radius"
        assert d[0]["scale"] == 2

    def test_from_dict(self, set_test_backend, setup_manager):
        """Tests deserializing a PickupManager from a list of dicts."""
        optic, _ = setup_manager
        d = [{"source_surface_idx": 0, "attr_type": "radius", "target_surface_idx": 1, "scale": 2, "offset": 3}]
        manager = PickupManager.from_dict(d, optic)
        assert len(manager) == 1
        assert manager[0].attr_type == "radius"