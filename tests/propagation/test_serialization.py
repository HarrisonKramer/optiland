"""Unit tests for serialization of propagation models."""

import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.standard import StandardGeometry
from optiland.materials.ideal import IdealMaterial
from optiland.optic.optic import Optic
from optiland.surfaces.standard_surface import Surface
from optiland.propagation.grin import GrinPropagation


def test_optic_serialization_with_default_homogeneous_model():
    """Verify serialization works with the default Homogeneous model."""
    optic = Optic()
    optic.add_surface(
        Surface(
            geometry=StandardGeometry(coordinate_system=CoordinateSystem(), radius=100),
            material_pre=IdealMaterial(n=1.0),
            material_post=IdealMaterial(n=1.5),
        )
    )

    d = optic.to_dict()

    # Check that the propagation model is serialized
    pm_data = d["surface_group"]["surfaces"][0]["material_pre"]["propagation_model"]
    assert pm_data["class"] == "HomogeneousPropagation"

    # Check that it can be deserialized
    optic_from_dict = Optic.from_dict(d)
    assert (
        optic_from_dict.surface_group.surfaces[0]
        .material_pre.propagation_model.__class__.__name__
        == "HomogeneousPropagation"
    )


def test_optic_serialization_with_grin_model():
    """Verify serialization works with a non-default Grin model."""
    # Create a material with a GrinPropagation model
    grin_material = IdealMaterial(n=1.0)
    grin_material.propagation_model = GrinPropagation()

    optic = Optic()
    optic.add_surface(
        Surface(
            geometry=StandardGeometry(coordinate_system=CoordinateSystem(), radius=100),
            material_pre=grin_material,
            material_post=IdealMaterial(n=1.5),
        )
    )

    d = optic.to_dict()

    # Check that the propagation model is serialized correctly
    pm_data = d["surface_group"]["surfaces"][0]["material_pre"]["propagation_model"]
    assert pm_data["class"] == "GrinPropagation"

    # Check that it can be deserialized correctly
    optic_from_dict = Optic.from_dict(d)
    assert (
        optic_from_dict.surface_group.surfaces[0]
        .material_pre.propagation_model.__class__.__name__
        == "GrinPropagation"
    )
