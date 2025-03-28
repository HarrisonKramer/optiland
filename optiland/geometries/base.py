"""Base Geometry

The base geometry class for all geometries.
Geometries are used to define the shape of optical elements.

Kramer Harrison, 2024
"""

from abc import ABC, abstractmethod


class BaseGeometry(ABC):
    """Base geometry for all geometries.

    Args:
        cs (CoordinateSystem): The coordinate system of the geometry.

    """

    _registry = {}

    def __init__(self, coordinate_system):
        self.cs = coordinate_system

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses."""
        super().__init_subclass__(**kwargs)
        BaseGeometry._registry[cls.__name__] = cls

    def __str__(self):
        return f"{self.__class__.__name__}"  # pragma: no cover

    @abstractmethod
    def sag(self, x=0, y=0):
        """Calculate the surface sag of the geometry.

        Args:
            x (float or np.ndarray, optional): The x-coordinate. Defaults to 0.
            y (float or np.ndarray, optional): The y-coordinate. Defaults to 0.

        Returns:
            Union[float, np.ndarray]: The surface sag of the geometry.

        """
        # pragma: no cover

    @abstractmethod
    def distance(self, rays):
        """Find the propagation distance to the geometry.

        Args:
            rays (RealRays): The rays to calculate the distance for.

        Returns:
            np.ndarray: The propagation distance to the geometry.

        """
        # pragma: no cover

    @abstractmethod
    def surface_normal(self, rays):
        """Find the surface normal of the geometry at the given ray positions.

        Args:
            rays (RealRays): The rays position at which to calculate the
                surface normal.

        Returns:
            np.ndarray: The surface normals of the geometry at the given
                ray positions.

        """
        # pragma: no cover

    def localize(self, rays):
        """Convert rays from the global coordinate system to the local
        coordinate system.

        Args:
            rays (RealRays): The rays to convert.

        """
        self.cs.localize(rays)

    def globalize(self, rays):
        """Convert rays from the local coordinate system to the global
        coordinate system.

        Args:
            rays (RealRays): The rays to convert.

        """
        self.cs.globalize(rays)

    def to_dict(self):
        """Convert the geometry to a dictionary.

        Returns:
            dict: The dictionary representation of the geometry.

        """
        return {"type": self.__class__.__name__, "cs": self.cs.to_dict()}

    @classmethod
    def from_dict(cls, data):
        """Create a geometry from a dictionary.

        Args:
            data (dict): The dictionary representation of the geometry.

        Returns:
            BaseGeometry: The geometry.

        """
        geometry_type = data.get("type")
        if geometry_type not in cls._registry:
            raise ValueError(f"Unknown geometry type: {geometry_type}")

        # Delegate to the correct subclass's from_dict
        return cls._registry[geometry_type].from_dict(data)
