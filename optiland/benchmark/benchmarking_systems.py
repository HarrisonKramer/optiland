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
from optiland.materials import IdealMaterial, Material
from optiland.optic import Optic
from optiland.physical_apertures import RadialAperture


def run_ray_trace(optic: Optic, num_rays: int):
    """
    Traces rays for all fields and wavelengths defined in the optic.
    This function leverages vectorization by tracing all fields simultaneously
    for each wavelength.
    """
    # Get all field coordinates as arrays
    field_coords = optic.fields.get_field_coords()
    if not field_coords:
        return

    Hx_all = be.array([fc[0] for fc in field_coords])
    Hy_all = be.array([fc[1] for fc in field_coords])

    # Loop through each wavelength and trace all fields at once
    for wvl in optic.wavelengths.get_wavelengths():
        optic.trace(
            Hx=Hx_all,
            Hy=Hy_all,
            wavelength=wvl,
            num_rays=num_rays,
            distribution="random",
        )


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

        PMMA = Material("PMMA")
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
            aperture=5.0,
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
            aperture=5.0,
        )
        optic.add_surface(
            index=10,
            radius=be.inf,
            thickness=0.5,
            material="N-BK7",
            aperture=3.193879680378425e000 * 2.0,
        )
        optic.add_surface(
            index=11,
            radius=be.inf,
            thickness=0.93,
            material="air",
            aperture=3.193879680378425e000 * 2.0,
        )
        optic.add_surface(index=12)  # img

        # Aperture
        optic.set_aperture(aperture_type="EPD", value=1.85)

        # Fields
        optic.set_field_type(field_type="angle")
        optic.add_field(y=0)
        optic.add_field(y=20, vy=0.2)
        optic.add_field(y=26, vy=0.3)

        # Wavelengths
        optic.add_wavelength(value=0.4861)
        optic.add_wavelength(value=0.5876, is_primary=True)
        optic.add_wavelength(value=0.6563)

        return optic


class CatadiopticMicroscope:
    @staticmethod
    def create_system():
        optic = Optic(name="Catadioptic Microscope")
        silica = IdealMaterial(n=1.4980)
        CaF2 = IdealMaterial(n=1.4610)

        optic.add_surface(index=0, radius=be.inf, thickness=be.inf)
        optic.add_surface(index=1, radius=1.72300, thickness=0.0350, material=silica)
        optic.add_surface(index=2, radius=2.90631, thickness=0.9731)
        optic.add_surface(
            index=3,
            radius=0.17783,
            thickness=-0.4995,
            material="mirror",
            is_stop=True,
        )

        obscuration = RadialAperture(r_max=be.inf, r_min=0.15)
        optic.add_surface(
            index=4,
            radius=0.72913,
            thickness=0.5776,
            material="mirror",
            aperture=obscuration,
        )

        optic.add_surface(index=5, radius=2.66996, thickness=0.0427, material=CaF2)
        optic.add_surface(index=6, radius=0.48467, thickness=0.005)
        optic.add_surface(index=7, radius=0.23767, thickness=0.1861, material=CaF2)
        optic.add_surface(index=8, radius=8.64894, thickness=0.005)
        optic.add_surface(index=9, radius=7.25758, thickness=0.0588, material=silica)
        optic.add_surface(index=10, radius=0.44435, thickness=0.0771)
        optic.add_surface(index=11, thickness=0.0071, material=silica)
        optic.add_surface(index=12)

        optic.set_aperture(aperture_type="imageFNO", value=0.7)

        optic.set_field_type(field_type="angle")
        optic.add_field(y=0)
        optic.add_field(y=2.8)
        optic.add_field(y=4)

        optic.add_wavelength(value=0.27, is_primary=True)

        optic.update_paraxial()

        # scale from inches to mm
        optic.scale_system(25.4)

        return optic


class ToroidalSinglet:
    @staticmethod
    def create_system():
        optic = Optic(name="Toroidal Singlet")
        optic.add_surface(index=0, radius=be.inf, thickness=be.inf)
        optic.add_surface(
            index=1,
            thickness=7,
            radius_x=20,  # <- radius: x radius of rotation.
            radius_y=40,
            is_stop=True,
            material="N-BK7",
            surface_type="toroidal",
            conic=-0.13,
            toroidal_coeffs_poly_y=[0.0, 1.2e-6, -1.5e-8],
        )  # <-- aspheric terms are possible
        optic.add_surface(index=2, thickness=65)
        optic.add_surface(index=3)

        # Aperture
        optic.set_aperture(aperture_type="EPD", value=20.0)

        # Fields
        optic.set_field_type(field_type="angle")
        optic.add_field(y=0)

        # WAvelengths
        optic.add_wavelength(value=0.440, is_primary=True)
        optic.add_wavelength(value=0.587, is_primary=True)
        optic.add_wavelength(value=0.656, is_primary=True)
        return optic


class BeamShaperAsphere:
    @staticmethod
    def create_system():
        optic = Optic(name="Beam Shaping System")

        # LDE data
        optic.add_surface(index=0, thickness=be.inf)
        optic.add_surface(
            index=1,
            thickness=20.0,
            is_stop=True,
        )

        optic.add_surface(
            index=2,
            surface_type="even_asphere",
            radius=5.1253,
            conic=-0.9743,
            thickness=15.0,
            material="N-BK7",
            coefficients=[
                0.0,
                -7.7523e-04,
                7.2050e-06,
                -5.1227e-08,
                2.4998e-10,
                -7.8623e-13,
                1.4274e-15,
                -1.1331e-18,
            ],
        )

        optic.add_surface(
            index=3,
            surface_type="standard",
            radius=be.inf,
            thickness=70.0,
            material="air",
        )

        optic.add_surface(
            index=4,
        )

        # Aperture
        optic.set_aperture(aperture_type="EPD", value=30.0)
        # Wavelengths
        optic.add_wavelength(value=550.0, is_primary=True, unit="nm")
        # Fields
        optic.set_field_type(field_type="angle")
        optic.add_field(y=0.0)
        return optic


class BeamShaperForbesQbfs:
    @staticmethod
    def create_system():
        optic = Optic(name="Beam Shaping System with Forbes Qbfs")

        # LDE data
        optic.add_surface(index=0, thickness=be.inf)
        optic.add_surface(
            index=1,
            thickness=20.0,
            is_stop=True,
        )

        optic.add_surface(
            index=2,
            surface_type="forbes_qbfs",
            radius=5.6068,
            conic=-1.9163,
            thickness=15.0,
            material="N-BK7",
            radial_terms={
                0: 5.5386,
                1: -3.6235,
                2: 2.6929,
                3: -1.2980,
                4: 7.5341e-1,
                5: -2.0739e-1,
                6: 8.0129e-2,
            },
            forbes_norm_radius=18.7500,
        )

        optic.add_surface(
            index=3,
            surface_type="standard",
            radius=be.inf,
            thickness=70.0,
            material="air",
        )

        optic.add_surface(
            index=4,
        )

        # Aperture
        optic.set_aperture(aperture_type="EPD", value=30.0)
        # Wavelengths
        optic.add_wavelength(value=550.0, is_primary=True, unit="nm")
        # Fields
        optic.set_field_type(field_type="angle")
        optic.add_field(y=0.0)

        return optic
