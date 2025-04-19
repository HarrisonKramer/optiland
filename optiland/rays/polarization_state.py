"""Polarization State

This module contains the PolarizationState class, which represents the
polarization state of a light ray or ray bundle. This class may be used
to define the polarization state of the light rays in an optical system.

Kramer Harrison, 2024
"""

from typing import Optional

import optiland.backend as be


class PolarizationState:
    """Represents the polarization state of a light ray.

    Attributes:
        is_polarized (bool): Indicates whether the state is polarized.
        Ex (Optional[float]): Electric field component in the x-direction.
        Ey (Optional[float]): Electric field component in the y-direction.
        phase_x (Optional[float]): Phase of the x-component of the electric
            field.
        phase_y (Optional[float]): Phase of the y-component of the electric
            field.

    """

    def __init__(
        self,
        is_polarized: bool = False,
        Ex: Optional[float] = None,
        Ey: Optional[float] = None,
        phase_x: Optional[float] = None,
        phase_y: Optional[float] = None,
    ):
        if is_polarized:
            if None in [Ex, Ey, phase_x, phase_y]:
                raise ValueError(
                    "All parameters must be provided for a polarized state.",
                )
        elif not all(var is None for var in [Ex, Ey, phase_x, phase_y]):
            raise ValueError(
                "Ex, Ey, phase_x, and phase_y must be None for a non-polarized state.",
            )

        self.is_polarized = is_polarized
        self.Ex = be.array(Ex) if Ex is not None else None
        self.Ey = be.array(Ey) if Ey is not None else None
        self.phase_x = be.array(phase_x) if phase_x is not None else None
        self.phase_y = be.array(phase_y) if phase_y is not None else None

        if self.Ex is not None and self.Ey is not None:
            mag = be.sqrt(self.Ex**2 + self.Ey**2)
            self.Ex = self.Ex / mag
            self.Ey = self.Ey / mag

    def __str__(self):
        """Returns a string representation of the polarization state.

        Returns:
            str: The string representation of the polarization state.

        """
        if self.is_polarized:
            return (
                f"Polarized Light: Ex: {self.Ex}, Ey: {self.Ey}, "
                f"Phase x: {self.phase_x}, Phase y: {self.phase_y}"
            )
        return "Unpolarized Light"

    def __repr__(self):
        """Returns a string representation of the polarization state.

        Returns:
            str: The string representation of the polarization state.

        """
        return self.__str__()


def create_polarization(pol_type: str):
    """Create a polarization state based on the given polarization type.

    Args:
        pol_type (str): The type of polarization. Must be one of the following:
            - 'unpolarized' for unpolarized light
            - 'H' for horizontal polarization
            - 'V' for vertical polarization
            - 'L+45' for linear polarization at +45 degrees
            - 'L-45' for linear polarization at -45 degrees
            - 'RCP' for right circular polarization
            - 'LCP' for left circular polarization

    Returns:
        PolarizationState: The created polarization state.

    Raises:
        ValueError: If an invalid polarization type is provided.

    """
    if pol_type == "unpolarized":
        return PolarizationState(is_polarized=False)
    if pol_type == "H":
        Ex = 1
        Ey = 0
        phase_x = 0
        phase_y = 0
    elif pol_type == "V":
        Ex = 0
        Ey = 1
        phase_x = 0
        phase_y = 0
    elif pol_type == "L+45":
        Ex = 1
        Ey = 1
        phase_x = 0
        phase_y = 0
    elif pol_type == "L-45":
        Ex = 1
        Ey = -1
        phase_x = 0
        phase_y = 0
    elif pol_type == "RCP":
        Ex = be.sqrt(2) / 2
        Ey = be.sqrt(2) / 2
        phase_x = 0
        phase_y = -be.pi / 2
    elif pol_type == "LCP":
        Ex = be.sqrt(2) / 2
        Ey = be.sqrt(2) / 2
        phase_x = 0
        phase_y = be.pi / 2
    else:
        raise ValueError(
            "Invalid polarization type. Must be H, V, L+45, L-45, RCP or LCP.",
        )
    return PolarizationState(
        is_polarized=True,
        Ex=Ex,
        Ey=Ey,
        phase_x=phase_x,
        phase_y=phase_y,
    )
