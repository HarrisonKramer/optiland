"""Physical Apertures Module

This module provides classes to define physical apertures on optical surfaces.
For example, a telescope primary mirror with a center hole can be represented
by a radial aperture that clips rays based on their distance from the origin.

Kramer Harrison, 2024
"""

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


class BaseAperture(ABC):
    """
    Base class for physical apertures.

    Methods:
        clip(RealRays): Clips the given rays based on the aperture's shape.
    """
    _registry = {}

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses."""
        super().__init_subclass__(**kwargs)
        BaseAperture._registry[cls.__name__] = cls

    @abstractmethod
    def contains(self, x, y):
        """
        Checks if the given point is inside the aperture.

        Args:
            x (np.ndarray): The x-coordinate of the point.
            y (np.ndarray): The y-coordinate of the point.

        Returns:
            np.ndarray: Boolean array indicating if the point is inside the
                aperture
        """
        pass  # pragma: no cover

    def clip(self, rays):
        """
        Clips the given rays based on the aperture's shape.

        Args:
            rays (RealRays): List of rays to be clipped.

        Returns:
            list: List of clipped rays.
        """
        inside = self.contains(rays.x, rays.y)
        rays.clip(~inside)

    @abstractmethod
    def scale(self, scale_factor):
        """
        Scales the aperture by the given factor.

        Args:
            scale_factor (float): The factor by which to scale the aperture.
        """
        pass  # pragma: no cover

    def to_dict(self):
        """
        Convert the aperture to a dictionary.

        Returns:
            dict: The dictionary representation of the aperture.
        """
        return {
            'type': self.__class__.__name__
        }

    @classmethod
    def from_dict(cls, data):
        """
        Create an aperture from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the aperture.

        Returns:
            BaseAperture: The aperture object.
        """
        aperture_type = data['type']
        return cls._registry[aperture_type].from_dict(data)

    def view(self, x_min, x_max, y_min, y_max, nx=256, ny=256,
             ax=None, **kwargs):
        """
        Visualize the aperture.

        Args:
            x_min (float): The minimum x-coordinate of the plot.
            x_max (float): The maximum x-coordinate of the plot.
            y_min (float): The minimum y-coordinate of the plot.
            y_max (float): The maximum y-coordinate of the plot.
            nx (int): The number of points in the x-direction.
            ny (int): The number of points in the y-direction.
            ax (matplotlib.axes.Axes): The axes to plot on.
            **kwargs: Additional keyword arguments to pass to the plot
                function.
        """
        if ax is None:
            fig, ax = plt.subplots()
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y)
        Z = self.contains(X, Y)
        ax.contourf(X, Y, Z, **kwargs)
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_aspect('equal')
        plt.show()

    def __or__(self, other):
        """Union: a point is inside if it is in either region."""
        return UnionAperture(self, other)

    def __add__(self, other):
        """Alternative operator for union."""
        return self.__or__(other)

    def __and__(self, other):
        """Intersection: a point is inside if it is in both regions."""
        return IntersectionAperture(self, other)

    def __sub__(self, other):
        """Difference: a point is allowed if it is in self but not in other."""
        return DifferenceAperture(self, other)


class BaseBooleanAperture(BaseAperture):
    """
    Base class for boolean operations on apertures.

    Args:
        a (BaseAperture): The first aperture.
        b (BaseAperture): The second aperture.
    """
    def __init__(self, a: BaseAperture, b: BaseAperture):
        self.a = a
        self.b = b

    @abstractmethod
    def contains(self, x, y):
        """
        Checks if the given point is inside the aperture.

        Args:
            x (np.ndarray): The x-coordinate of the point.
            y (np.ndarray): The y-coordinate of the point.

        Returns:
            np.ndarray: Boolean array indicating if the point is inside the
                aperture
        """
        pass  # pragma: no cover

    def scale(self, scale_factor):
        """
        Scales the aperture by the given factor.

        Args:
            scale_factor (float): The factor by which to scale the aperture.
        """
        self.a.scale(scale_factor)
        self.b.scale(scale_factor)

    def to_dict(self):
        """
        Convert the aperture to a dictionary.

        Returns:
            dict: The dictionary representation of the aperture.
        """
        data = super().to_dict()
        data.update({
            'a': self.a.to_dict(),
            'b': self.b.to_dict()
        })
        return data

    @classmethod
    def from_dict(cls, data):
        """
        Create an aperture from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the aperture.

        Returns:
            BaseBooleanAperture: The aperture object.
        """
        a = BaseAperture.from_dict(data['a'])
        b = BaseAperture.from_dict(data['b'])
        return cls(a, b)


class UnionAperture(BaseBooleanAperture):
    """
    Class for union of two apertures.

    Args:
        a (BaseAperture): The first aperture.
        b (BaseAperture): The second aperture.
    """
    def __init__(self, a: BaseAperture, b: BaseAperture):
        super().__init__(a, b)

    def contains(self, x, y):
        """
        Checks if the given point is inside either aperture.

        Args:
            x (np.ndarray): The x-coordinate of the point.
            y (np.ndarray): The y-coordinate of the point.

        Returns:
            np.ndarray: Boolean array indicating if the point is inside the
                aperture
        """
        return np.logical_or(self.a.contains(x, y), self.b.contains(x, y))


class IntersectionAperture(BaseBooleanAperture):
    """
    Class for intersection of two apertures.

    Args:
        a (BaseAperture): The first aperture.
        b (BaseAperture): The second aperture.
    """
    def __init__(self, a: BaseAperture, b: BaseAperture):
        super().__init__(a, b)

    def contains(self, x, y):
        """
        Checks if the given point is inside the aperture.

        Args:
            x (np.ndarray): The x-coordinate of the point.
            y (np.ndarray): The y-coordinate of the point.

        Returns:
            np.ndarray: Boolean array indicating if the point is inside the
                aperture
        """
        return np.logical_and(self.a.contains(x, y), self.b.contains(x, y))


class DifferenceAperture(BaseBooleanAperture):
    """
    Class for difference of two apertures.

    Args:
        a (BaseAperture): The first aperture.
        b (BaseAperture): The second aperture.
    """
    def __init__(self, a: BaseAperture, b: BaseAperture):
        super().__init__(a, b)

    def contains(self, x, y):
        """
        Checks if the given point is inside the aperture.

        Args:
            x (np.ndarray): The x-coordinate of the point.
            y (np.ndarray): The y-coordinate of the point.

        Returns:
            np.ndarray: Boolean array indicating if the point is inside the
                aperture
        """
        return np.logical_and(self.a.contains(x, y),
                              np.logical_not(self.b.contains(x, y)))
