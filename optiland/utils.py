"""Optiland Utilities Module

This module provides utility functions for optical system analysis, including
the calculation of the working F-number (F/#) of an optical system.

Kramer Harrison, 2025
"""

from __future__ import annotations

import optiland.backend as be


def get_working_FNO(optic, field, wavelength):
    """Calculates the working F-number of the optical system for the
    single defined field point and given wavelength.

    Args:
        optic (Optic): The optic object.
        field (tuple): The field at which to calculate the F/#.
        wavelength (float): The wavelength at which to calculate the F/#.

    Algorithm:
        1. Retrieve the defined given wavelength and field coordinates.
        2. Determine the image-space refractive index 'n' at the given wavelength.
        3. Trace four marginal rays (top, bottom, left, right) at the pupil edges,
            as well as the chief ray.
        4. Compute the angle between each marginal ray and the chief ray.
        4. Calculate the average of the squared numerical apertures from all traced
            marginal rays.
        5. Compute the working F-number as 1 / (2 * sqrt(average_NA_squared)).
        6. Cap the calculated F/# at 10,000 if it exceeds this value.

    Returns:
        float: The working F-number.
    """
    MAX_FNUM = 10000.0

    Hx, Hy = field

    n = optic.image_surface.material_post.n(wavelength)
    Px = be.array([0, 0, 0, 1, -1])
    Py = be.array([0, 1, -1, 0, 0])

    rays = optic.trace_generic(Hx=Hx, Hy=Hy, Px=Px, Py=Py, wavelength=wavelength)

    L0, M0, N0 = rays.L[0], rays.M[0], rays.N[0]
    L, M, N = rays.L[1:], rays.M[1:], rays.N[1:]
    dot = L0 * L + M0 * M + N0 * N
    dot = be.clip(dot, -1.0, 1.0)
    angles = be.arccos(dot)

    numerical_apertures_squared = (n * be.sin(angles)) ** 2
    avg_NA_squared = be.mean(be.array(numerical_apertures_squared))

    fno = be.inf if avg_NA_squared <= 0 else 1 / (2 * be.sqrt(avg_NA_squared))

    if fno > MAX_FNUM:
        fno = MAX_FNUM

    if be.isnan(fno):
        raise ValueError("Working F/# could not be calculated due to raytrace errors.")

    return fno
