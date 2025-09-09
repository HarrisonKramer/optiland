"""Thin film optics core functions.

This provides core functions for thin film optics calculations using the
transfer matrix method (TMM).

Corentin Nannini, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import optiland.backend as be

if TYPE_CHECKING:
    from optiland.materials import BaseMaterial
    from optiland.thin_film import ThinFilmStack

Array: TypeAlias = Any  # be.ndarray
PolSP = Literal["s", "p"]


def _complex_index(material: BaseMaterial, wavelength_um: float | Array) -> Array:
    n = material.n(wavelength_um)
    k = material.k(wavelength_um)
    n = be.atleast_1d(n)
    k = be.atleast_1d(k)
    return be.asarray(n, dtype=be.complex128) + 1j * be.asarray(k, dtype=be.complex128)


def _snell_cos(n0, theta0, n):
    """Transmitted angle cosine with forward-branch selection.
    Calculation is following 'Thin-Film Optical Filters, Fifth Edition, Macleod,
    Hugh Angus CRC Press, Ch2.6
    """
    nr = n.real
    k = n.imag
    return be.sqrt(nr**2 - k**2 - (n0 * be.sin(theta0)) ** 2 - 2j * nr * k) / n


def _admittance(n: complex, cos_t: complex, pol: PolSP):
    """Admittance η = sqrt(ε/μ) * n * cos(θ) for s and p polarizations.
    Calculation is following 'Thin-Film Optical Filters, Fifth Edition, Macleod,
    Hugh Angus CRC Press, Ch2.6
    """
    sqrt_eps_mu = 0.002654418729832701370374020517935  # S
    eta_s = sqrt_eps_mu * n * cos_t

    if pol == "s":
        return eta_s
    elif pol == "p":
        eta_p = sqrt_eps_mu**2 * (n.real - 1j * n.imag) ** 2 / eta_s
        return eta_p
    else:
        raise ValueError("Invalid polarization state")


def _tmm_coh(stack: ThinFilmStack, wavelength_um, theta0_rad, pol: PolSP):
    """
    Compute the reflection and transmission coefficients for
    a thin film stack. Based on Abelès Matrix.
    Calculation is vectorized over wavelength and angle of incidence.

    Ref :
    - Chap 13. Polarized Light and Optical Systems, Russell
        A. Chipman, Wai-Sze Tiffany Lam, and Garam Young
    - F. Abelès, Researches sur la propagation des ondes électromagnétiques
        sinusoïdales dans les milieus stratifies.
        Applications aux couches minces, Ann. Phys. Paris,
        12ième Series 5 (1950): 596–640.
    - Chap 2. Thin-Film Optical Filters, Fifth Edition, Macleod, Hugh Angus CRC Press
    """
    n0 = _complex_index(stack.incident_material, wavelength_um)
    ns = _complex_index(stack.substrate_material, wavelength_um)
    cos0 = _snell_cos(n0, theta0_rad, n0)
    coss = _snell_cos(n0, theta0_rad, ns)
    eta0 = _admittance(n0, cos0, pol)
    etas = _admittance(ns, coss, pol)

    # Id initial matrix
    A = be.ones_like(eta0, dtype=be.complex128)
    B = be.zeros_like(eta0, dtype=be.complex128)
    C = be.zeros_like(eta0, dtype=be.complex128)
    D = be.ones_like(eta0, dtype=be.complex128)

    for layer in stack.layers:
        n_l = layer.n_complex(wavelength_um)
        cos_l = _snell_cos(n0, theta0_rad, n_l)
        eta_l = _admittance(n_l, cos_l, pol)
        delta = layer.phase_thickness(wavelength_um, cos_l, n_l)
        c = be.cos(delta)
        s = be.sin(delta)
        i = 1j
        mA = c
        mB = i * (s / eta_l)
        mC = i * (eta_l * s)
        mD = c
        A, B, C, D = A * mA + B * mC, A * mB + B * mD, C * mA + D * mC, C * mB + D * mD

    denom = eta0 * (A + etas * B) + C + etas * D
    denom = be.where(be.abs(denom) == 0, 1e-30 + 0j, denom)

    r = (eta0 * A + eta0 * etas * B - C - etas * D) / denom
    t = be.conj((2 * eta0) / denom)

    R = (r * be.conj(r)).real
    T = (t * be.conj(t)).real * etas.real / eta0.real
    abso = 1 - R - T
    # Absorption can also be obtained with :
    # abso = (1 - R) * (1 - etas.real / ((A + etas * B) * (C + etas * D).conj()).real)
    return r, t, R, T, abso
