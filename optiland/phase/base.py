"""Phase

Phase that gets added to any surface defiend in Geometry:



Hhsoj, 2025
"""

from __future__ import annotations


class BasePhase:
    """Represents a phase function that can be added to a Geometry
    BasePhase is reserved for expansion of the Phase class

    """

    def __str__(self):
        return "Base"

    def phase_calc(self, rays, nx, ny, nz, n1, n2):
        """Calculate the surface normal of the geometry at the given points.

        Args:
            rays (RealRays): The rays, positioned at the surface, for which to
                calculate the surface normals.

        Returns:
            tuple[be.ndarray, be.ndarray, be.ndarray]: The l, m, n and opd
            applied to input L,m,n, opd due to the phase function from DOE.

        """

        return l, m, n, opd

    def efficiency(self, rays):
        """applies the efficiency to the ray based on the order and the fabrication of the doe deisgn"""

        return d_eff

    def to_dict(self):
        """Convert the phase to a dictionary.

        Returns:
            dict: The dictionary representation of the phase.

        """
        return {"type": self.__class__.__name__, "cs": self.cs.to_dict()}

    @classmethod
    def from_dict(cls, data):
        """Create a phase from a dictionary.

        Args:
            data (dict): A dictionary containing the phase data, including
                its 'type' and coordinate system 'cs'.

        Returns:
            BasePhase: An instance of a specific phase subclass created
            from the dictionary data.

        """
        phase_type = data.get("type")
        if phase_type not in cls._registry:
            raise ValueError(f"Unknown phase type: {phase_type_type}")

        # Delegate to the correct subclass's from_dict
        return cls._registry[phase_type_type].from_dict(data)
