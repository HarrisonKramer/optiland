"""Base Phase

This module defines the `BasePhase` class, which serves as the abstract base
class for all phase functions that can be applied to a surface in an optical
system.

The `BasePhase` class provides a common interface for phase calculations,
ensuring that all phase models can be used interchangeably within the
`DiffractiveInteractionModel`. It defines an abstract `phase_calc` method that
must be implemented by all subclasses. This method calculates the effect of the
phase function on the direction and optical path difference of a ray.

The module also includes a registry for all `BasePhase` subclasses, which
enables the dynamic instantiation of phase models from a dictionary
representation. This is used for serialization and deserialization of optical
systems.

Hhsoj, 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Dict, Type

if TYPE_CHECKING:
    from optiland.rays import RealRays
    from optiland._types import BEArray


class BasePhase(ABC):
    """Represents an abstract phase function that can be added to a surface.

    This class is the abstract base class for all phase functions. It defines
    the common interface for phase calculations and serialization.

    """

    _registry: ClassVar[Dict[str, Type[BasePhase]]] = {}

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses."""
        super().__init_subclass__(**kwargs)
        BasePhase._registry[cls.__name__] = cls

    @abstractmethod
    def phase_calc(
        self,
        rays: RealRays,
        nx: BEArray,
        ny: BEArray,
        nz: BEArray,
        n1: BEArray,
        n2: BEArray,
    ) -> tuple[BEArray, BEArray, BEArray, BEArray]:
        """Calculates the effect of the phase function on the rays.

        Args:
            rays (RealRays): The rays incident on the surface.
            nx (BEArray): The x-component of the surface normal.
            ny (BEArray): The y-component of the surface normal.
            nz (BEArray): The z-component of the surface normal.
            n1 (BEArray): The refractive index of the medium before the surface.
            n2 (BEArray): The refractive index of the medium after the surface.

        Returns:
            tuple[BEArray, BEArray, BEArray, BEArray]: A tuple containing the
            new x, y, and z direction cosines (L, M, N) and the optical path
            difference (OPD) to be added to the rays.

        """
        raise NotImplementedError

    @abstractmethod
    def efficiency(self, rays: RealRays) -> BEArray:
        """Calculates the diffraction efficiency of the phase function.

        Args:
            rays (RealRays): The rays incident on the surface.

        Returns:
            BEArray: The diffraction efficiency for each ray.

        """
        raise NotImplementedError

    def to_dict(self) -> dict:
        """Converts the phase object to a dictionary.

        Returns:
            dict: The dictionary representation of the phase object.

        """
        return {"type": self.__class__.__name__}

    @classmethod
    def from_dict(cls, data: dict) -> "BasePhase":
        """Creates a phase object from a dictionary.

        Args:
            data (dict): A dictionary containing the phase data.

        Returns:
            BasePhase: An instance of a specific phase subclass.

        Raises:
            ValueError: If the phase type specified in the dictionary is unknown.

        """
        phase_type = data.get("type")
        if phase_type not in cls._registry:
            raise ValueError(f"Unknown phase type: {phase_type}")

        return cls._registry[phase_type].from_dict(data)
