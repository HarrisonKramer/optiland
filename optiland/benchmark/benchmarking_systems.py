"""
This module contains the definitions of various benchmarking systems to be
used in the ray tracing performance speed up analysis.

NOTE: Here we are not concerned with optimization performance. This will
be handled in a separate issue. Therefore, the presented systems are
already optimized versions.

List of systems:

- Aspheric Singlet (Odd-Asphere): 1 field, 1 wavelength
- 5 element Mobile Imaging: 3 fields, 3 wavelengths
    -> patent: US06476982-1
- Catadioptric Microscope: 3 fields, 1 wavelength
- Toroidal Singlet: 1 field, 3 wavelengths


- Beam Shaping System: 1 field, 1 wavelength
    -> Even Asphere
    -> Forbes Qbfs

"""

from __future__ import annotations

import optiland.backend as be
from optiland.materials import Material
from optiland.optic import Optic


class AsphericSinglet:
    @staticmethod
    def create_system():
        optic = Optic(name="Aspheric Singlet (Odd-Asphere)")

        # LDE data
        optic.add_surface(index=0, radius=be.inf, thickness=be.inf)
        optic.add_surface(
            index=1,
            thickness=7,
            radius=20.0,
            is_stop=True,
            material="N-SF11",
            surface_type="even_asphere",
            conic=0.0,
            coefficients=[-2.248851e-4, -4.690412e-6, -6.404376e-8],
        )  # <-- coefficients for asphere
        optic.add_surface(index=2, thickness=21.56201105)
        optic.add_surface(index=3)

        # Aperture
        optic.set_aperture(aperture_type="EPD", value=20.0)

        # Fields
        optic.set_field_type(field_type="angle")
        optic.add_field(y=0)

        # Wavelengths
        optic.add_wavelength(value=0.587, is_primary=True)

        return optic


class MobileImagingSystem:
    """Defines a 5-element mobile imaging system based on US patent US06476982-1."""

    @staticmethod
    def create_system():
        optic = Optic(name="Mobile Imaging System")

        PMMA = Material(name="ACRYLIC")
        H_ZLAF55C = Material(name="H-ZLAF55C", reference="cdgm")
        H_ZF50 = Material(name="H-ZF50", reference="cdgm")

        # LDE data
        optic.add_surface(index=0, radius=be.inf, thickness=be.inf)
        optic.add_surface(index=1, radius=be.inf, thickness=0.2, material="air")
        optic.add_surface(
            index=2,
            radius=be.inf,
            thickness=0.46,
            material="air",
            is_stop=True,
        )
        optic.add_surface(index=3, radius=3.663, thickness=1.56, material=H_ZLAF55C)
        optic.add_surface(index=4, radius=-3.704, thickness=0.81, material=H_ZF50)
        optic.add_surface(index=5, radius=4.756, thickness=0.64, material="air")
        optic.add_surface(
            index=6,
            radius=-3.704,
            thickness=1.21,
            material=PMMA,
            surface_type="even_asphere",
            conic=-23.8604,
            coefficients=[0.0, -4.09849e-002, 6.16592e-003],
        )
        optic.add_surface(
            index=7,
            radius=-2.503,
            thickness=0.03,
            material="air",
            surface_type="even_asphere",
            conic=-7.38406,
            coefficients=[0.0, -3.81281e-002, 5.04072e-003, 6.07261e-004, 6.88523e-005],
        )
        optic.add_surface(
            index=8,
            radius=4.599,
            thickness=1.24,
            material=PMMA,
            surface_type="even_asphere",
            conic=2.25228,
            coefficients=[
                0.0,
                -3.53667e-002,
                2.03259e-003,
                6.85038e-005,
                -3.86813e-005,
            ],
        )
        optic.add_surface(
            index=9,
            radius=3.36,
            thickness=0.69,
            material="air",
            surface_type="even_asphere",
            conic=-11.5784,
            coefficients=[
                0.0,
                -4.98743e-003,
                -2.64256e-003,
                3.74355e-004,
                -2.29586e-005,
            ],
        )
        optic.add_surface(index=10, radius=be.inf, thickness=0.5, material="N-BK7")
        optic.add_surface(index=11, radius=be.inf, thickness=0.93, material="air")
        optic.add_surface(index=12)  # img

        # Aperture
        optic.set_aperture(aperture_type="EPD", value=2.85)

        # Fields
        optic.set_field_type(field_type="angle")
        optic.add_field(y=0)
        optic.add_field(y=22)
        optic.add_field(y=31)

        # Wavelengths
        optic.add_wavelength(value=0.4861)
        optic.add_wavelength(value=0.5876, is_primary=True)
        optic.add_wavelength(value=0.6563)

        return optic


syst = MobileImagingSystem.create_system()

syst.draw(num_rays=11)
