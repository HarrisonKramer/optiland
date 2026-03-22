"""Build and visualize simple GRIN systems with Optiland APIs.

This example shows the normal Optiland workflow:

1. Build an `Optic`
2. Insert a GRIN slab as the medium between two surfaces
3. Visualize the system with `optic.draw()`
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from optiland.materials import GradientMaterial
from optiland.optic import Optic


def build_radial_grin_slab() -> tuple[Optic, float]:
    """Create a simple optic containing a radial GRIN slab."""
    grin_thickness = 8.0
    grin_material = GradientMaterial(
        n0=1.62,
        nr2=-0.03,  # Higher index on axis -> focusing behavior
    )

    optic = Optic()
    optic.add_surface(index=0, radius=np.inf, thickness=np.inf, comment="Object")
    optic.add_surface(
        index=1,
        radius=np.inf,
        thickness=grin_thickness,
        material=grin_material,
        is_stop=True,
        comment="GRIN entrance plane",
    )
    optic.add_surface(
        index=2,
        radius=np.inf,
        thickness=20.0,
        material="air",
        comment="GRIN exit plane",
    )
    optic.add_surface(index=3, radius=np.inf, comment="Image")

    optic.set_aperture(aperture_type="EPD", value=3.0)
    optic.set_field_type(field_type="angle")
    optic.add_field(y=0.0)
    optic.add_wavelength(value=0.587, is_primary=True)

    return optic, grin_thickness


def build_axial_grin_slab() -> tuple[Optic, float]:
    """Create an axial-GRIN slab for an obliquely incident chief ray."""
    grin_thickness = 10.0
    grin_material = GradientMaterial(
        n0=2.0,
        nz1=-0.3,
        step_size=0.001,  # Use a smaller step size for better accuracy with oblique rays
    )

    optic = Optic()
    optic.add_surface(index=0, radius=np.inf, thickness=np.inf, comment="Object")
    optic.add_surface(
        index=1,
        radius=np.inf,
        thickness=grin_thickness,
        material=grin_material,
        is_stop=True,
        comment="Axial GRIN entrance plane",
    )
    optic.add_surface(
        index=2,
        radius=np.inf,
        thickness=20.0,
        material="air",
        comment="Axial GRIN exit plane",
    )
    optic.add_surface(index=3, radius=np.inf, comment="Image")

    optic.set_aperture(aperture_type="EPD", value=4.0)
    optic.set_field_type(field_type="angle")

    # L = 0.3 in the manual GRIN reflection script corresponds to
    # arcsin(0.3) ~= 17.46 degrees. Put the field angle in X so the
    # oblique ray is visible in the XZ projection.
    optic.add_field(x=float(np.degrees(np.arcsin(0.3))), y=0.0)
    optic.add_wavelength(value=0.55, is_primary=True)

    return optic, grin_thickness


def visualize_grin_system():
    """Draw the GRIN system using Optiland's built-in visualization."""
    optic, _ = build_radial_grin_slab()

    optic.draw(
        num_rays=9,
        distribution="line_y",
        projection="YZ",
        title="Radial GRIN slab (YZ view)",
        reference="chief",
    )

    plt.show()


def visualize_axial_grin_system():
    """Draw an oblique ray through a z-gradient GRIN slab."""
    optic, _ = build_axial_grin_slab()

    optic.draw(
        distribution=None,
        projection="XZ",
        title="Axial GRIN slab (XZ view, oblique chief ray)",
        reference="chief",
    )

    # If you need to draw onto an existing matplotlib subplot, use
    # OpticViewer(optic).view(ax=your_axis, ...) instead of optic.draw().

    plt.show()


if __name__ == "__main__":
    visualize_grin_system()
    visualize_axial_grin_system()
