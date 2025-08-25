"""Lens Operands Module

This module provides a class that calculates various physical constraints
for an optical system to be used in conjunction with the optimization module to
optimize systems.

"""

from __future__ import annotations

import optiland.backend as be


class LensOperand:
    """A class that provides static methods for performing phyical constraints
    calculations on an optic

    Methods:
        edge_thickness: Calculates the edge thickness between two surfaces.
                        note that this can be in glass or air.
    """

    @staticmethod
    def edge_thickness(optic, surface_number):
        """Calculates the edge thickness between two surfaces.

        Args:
            optic: The optic object.
            surface_number: The number of the first surface.

        Returns:
            The edge thickness between the two surfaces.
        """

        surface1 = optic.surface_group.surfaces[surface_number]
        surface2 = optic.surface_group.surfaces[surface_number + 1]
        semi_apt_1 = surface1.semi_aperture
        semi_apt_2 = surface2.semi_aperture
        semi_apt_min = semi_apt_1

        if semi_apt_1 != semi_apt_2:
            # in case of different semi-diameter, take the minimum
            semi_apt_min = be.minimum(be.array(semi_apt_1), be.array(semi_apt_2))

        sag1 = surface1.geometry.sag(y=semi_apt_min)
        sag2 = surface2.geometry.sag(y=semi_apt_min)

        thickness = optic.surface_group.get_thickness(surface_number)

        edge_thickness = thickness - sag1 + sag2
        return edge_thickness
