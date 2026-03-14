"""Standalone refraction and reflection functions.

These functions implement the vectorial forms of Snell's law and the law of
reflection. They are used by both the sequential and non-sequential ray
tracers.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be

if TYPE_CHECKING:
    from optiland._types import BEArray


def refract(
    L: BEArray,
    M: BEArray,
    N: BEArray,
    nx: BEArray,
    ny: BEArray,
    nz: BEArray,
    n1: BEArray | float,
    n2: BEArray | float,
) -> tuple[BEArray, BEArray, BEArray, BEArray]:
    """Apply Snell's law in vector form.

    The surface normal must already be aligned so that the dot product of
    the incident direction and the normal is positive (i.e. the normal
    points into the hemisphere of the incident ray).

    Args:
        L: x-direction cosines of incident rays.
        M: y-direction cosines of incident rays.
        N: z-direction cosines of incident rays.
        nx: x-components of aligned surface normals.
        ny: y-components of aligned surface normals.
        nz: z-components of aligned surface normals.
        n1: Refractive index of the incident medium.
        n2: Refractive index of the refracted medium.

    Returns:
        L_new: x-direction cosines after refraction.
        M_new: y-direction cosines after refraction.
        N_new: z-direction cosines after refraction.
        tir_mask: Boolean mask where total internal reflection occurred.
    """
    u = n1 / n2
    dot = L * nx + M * ny + N * nz

    discriminant = 1 - u**2 * (1 - dot**2)

    with be.errstate(invalid="ignore"):
        root = be.sqrt(discriminant)

    L_new = u * L + nx * root - u * nx * dot
    M_new = u * M + ny * root - u * ny * dot
    N_new = u * N + nz * root - u * nz * dot

    tir_mask = discriminant < 0

    return L_new, M_new, N_new, tir_mask


def reflect(
    L: BEArray,
    M: BEArray,
    N: BEArray,
    nx: BEArray,
    ny: BEArray,
    nz: BEArray,
) -> tuple[BEArray, BEArray, BEArray]:
    """Apply specular reflection law.

    The surface normal must already be aligned so that the dot product of
    the incident direction and the normal is positive.

    Args:
        L: x-direction cosines of incident rays.
        M: y-direction cosines of incident rays.
        N: z-direction cosines of incident rays.
        nx: x-components of aligned surface normals.
        ny: y-components of aligned surface normals.
        nz: z-components of aligned surface normals.

    Returns:
        L_new: x-direction cosines after reflection.
        M_new: y-direction cosines after reflection.
        N_new: z-direction cosines after reflection.
    """
    dot = L * nx + M * ny + N * nz

    L_new = L - 2 * dot * nx
    M_new = M - 2 * dot * ny
    N_new = N - 2 * dot * nz

    return L_new, M_new, N_new
