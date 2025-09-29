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
        5. Compute the working F-number as 1 / (2 * be.sqrt(average_NA_squared)).
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


def resolve_wavelengths(optic, wavelengths):
    """Resolves wavelength input into a list of wavelength values.

    Args:
        optic (Optic): The optic object.
        wavelengths (str or list): The wavelengths to resolve.
            Can be 'all', 'primary', or a list of wavelength values.

    Returns:
        list: A list of wavelength values.
    """
    if isinstance(wavelengths, str):
        if wavelengths == "all":
            return optic.wavelengths.get_wavelengths()
        elif wavelengths == "primary":
            return [optic.primary_wavelength]
        else:
            raise ValueError("Invalid wavelength string. Must be 'all' or 'primary'.")
    elif isinstance(wavelengths, list):
        return wavelengths
    else:
        raise TypeError("Wavelengths must be a string ('all', 'primary') or a list.")


def resolve_fields(optic, fields):
    """Resolves field input into a list of field coordinates.

    Args:
        optic (Optic): The optic object.
        fields (str or list): The fields to resolve.
            Can be 'all' or a list of field coordinates.

    Returns:
        list: A list of field coordinates.
    """
    if isinstance(fields, str):
        if fields == "all":
            return optic.fields.get_field_coords()
        else:
            raise ValueError("Invalid field string. Must be 'all'.")
    elif isinstance(fields, list):
        return fields
    else:
        raise TypeError("Fields must be a string ('all') or a list.")


def resolve_wavelength(optic, wavelength):
    """Resolves a single wavelength input into a float value.

    Args:
        optic (Optic): The optic object.
        wavelength (str or float or int): The wavelength to resolve.
            Can be 'primary' or a numerical value.

    Returns:
        float: A single wavelength value.
    """
    if isinstance(wavelength, str):
        if wavelength == "primary":
            return optic.primary_wavelength
        else:
            raise ValueError(
                "Invalid wavelength string. For a single wavelength, it must be "
                "'primary'."
            )
    elif isinstance(wavelength, int | float):
        return float(wavelength)
    else:
        raise TypeError("Wavelength must be a string ('primary') or a number.")
