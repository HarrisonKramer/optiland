"""
This module defines the ZernikeOPD class.

Kramer Harrison, 2024
"""

from __future__ import annotations

from optiland.zernike import ZernikeFit

from .opd import OPD


class ZernikeOPD(ZernikeFit, OPD):
    """Represents a Zernike Optical Path Difference (OPD) calculation.

    This class inherits from both the ZernikeFit and OPD classes. It first
    generates the OPD map(s), then fits Zernike polynomials to the map(s).

    Args:
        optic (object): The optic object representing the optical system.
        field (tuple): The field used for the calculation.
        wavelength (float): The wavelength of light used in the calculation.
        num_rings (int, optional): The number of rings used in the Zernike
            calculation. Default is 15.
        zernike_type (str, optional): The type of Zernike polynomials used.
            Default is 'fringe'. See zernike module for more information.
        num_terms (int, optional): The number of Zernike terms used in the
            calculation. Default is 37.
        strategy (str): The calculation strategy to use. Supported options are
            "chief_ray" and "centroid_sphere". Defaults to "chief_ray".
        remove_tilt (bool): If True, removes tilt and piston from the OPD data.
            Defaults to False.
        **kwargs: Additional keyword arguments passed to the strategy.

    """

    def __init__(
        self,
        optic,
        field,
        wavelength,
        num_rings=15,
        zernike_type="fringe",
        num_terms=37,
        strategy="chief_ray",
        remove_tilt=False,
        **kwargs,
    ):
        OPD.__init__(
            self,
            optic=optic,
            field=field,
            wavelength=wavelength,
            num_rays=num_rings,
            distribution="hexapolar",
            strategy=strategy,
            remove_tilt=remove_tilt,
            **kwargs,
        )

        x = self.distribution.x
        y = self.distribution.y

        data = self.get_data(self.fields[0], self.wavelengths[0])
        z = data.opd

        mask = data.intensity > 0
        x = x[mask]
        y = y[mask]
        z = z[mask]

        ZernikeFit.__init__(self, x, y, z, zernike_type, num_terms)
