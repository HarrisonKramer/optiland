from abc import ABC, abstractmethod


class BaseGeometry(ABC):
    """Base geometry for all geometries.

    Args:
        cs (CoordinateSystem): The coordinate system of the geometry.
    """

    def __init__(self, coordinate_system):
        self.cs = coordinate_system

    @abstractmethod
    def sag(self, x=0, y=0):
        """Calculate the surface sag of the geometry.

        Args:
            x (float or np.ndarray, optional): The x-coordinate. Defaults to 0.
            y (float or np.ndarray, optional): The y-coordinate. Defaults to 0.

        Returns:
            Union[float, np.ndarray]: The surface sag of the geometry.
        """
        pass  # pragma: no cover

    @abstractmethod
    def distance(self, rays):
        """Find the propagation distance to the geometry.

        Args:
            rays (RealRays): The rays to calculate the distance for.

        Returns:
            np.ndarray: The propagation distance to the geometry.
        """
        pass  # pragma: no cover

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
        pass  # pragma: no cover

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
