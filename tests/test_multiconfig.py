
import pytest
from optiland.optic import Optic
from optiland.multiconfig.multi_configuration import MultiConfiguration


@pytest.fixture
def base_optic():
    optic = Optic()
    # Create a simple triplet-like structure
    # Surfaces: 0 (Obj), 1 (Lens1), 2 (Air), 3 (Lens2), 4 (Img)
    optic.add_surface(radius=100, thickness=10, material='air', index=0)
    optic.add_surface(radius=-100, thickness=5, material='air', index=1)
    optic.add_surface(radius=50, thickness=10, material='air', index=2)
    optic.add_surface(radius=-50, thickness=0, material='air', index=3) # Image
    return optic


def test_multiconfig_initial_linking(base_optic):
    """Verify that a new configuration is correctly linked to the base."""
    mc = MultiConfiguration(base_optic)
    config1 = mc.add_configuration()

    # Verify initial link by modifying base
    base_optic.set_radius(200, 1)
    base_optic.update()
    config1.update()

    r1_base = base_optic.surface_group.surfaces[1].geometry.radius
    r1_conf = config1.surface_group.surfaces[1].geometry.radius
    
    assert r1_base == 200
    assert r1_conf == 200


def test_multiconfig_break_link_logic(base_optic):
    """Verify that we can manually break the link (simulating optimization variable addition)."""
    mc = MultiConfiguration(base_optic)
    config1 = mc.add_configuration()

    # 1. Identify and remove the pickup for Surface 1, Radius
    surface_number = 1
    variable_type = 'radius'
    
    pickups_to_remove = []
    for pickup in config1.pickups.pickups:
        if (
            pickup.target_surface_idx == surface_number
            and pickup.attr_type == variable_type
        ):
            pickups_to_remove.append(pickup)
    
    assert len(pickups_to_remove) > 0, "Default pickup not found."

    for pickup in pickups_to_remove:
        config1.pickups.pickups.remove(pickup)

    # 2. Modify base, verify config1 does NOT change
    base_optic.set_radius(300, 1)
    base_optic.update()
    config1.update()

    r1_base = base_optic.surface_group.surfaces[1].geometry.radius
    r1_conf = config1.surface_group.surfaces[1].geometry.radius

    assert r1_base == 300
    assert r1_conf != 300  # Should remain at previous value (likely original or last set)
