"""Physical Apertures Base Module

This module contains the base classes for physical apertures. The BaseAperture
class is an abstract base class that defines the interface for physical
apertures. The BaseBooleanAperture class is an abstract base class for boolean
operations on apertures. The UnionAperture, IntersectionAperture, and
DifferenceAperture classes are concrete classes that implement the union,
intersection, and difference of two apertures, respectively.

Kramer Harrison, 2024
"""

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np


class BaseAperture(ABC):
    """Base class for physical apertures.

    Methods:
        clip(RealRays): Clips the given rays based on the aperture's shape.

    """

    _registry = {}

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses."""
        super().__init_subclass__(**kwargs)
        BaseAperture._registry[cls.__name__] = cls

    @property
    @abstractmethod
    def extent(self):
        """Returns the extent of the aperture.

        Returns:
            tuple: The extent of the aperture in the x and y directions.

        """
        # pragma: no cover

    @abstractmethod
    def contains(self, x, y):
        """Checks if the given point is inside the aperture.

        Args:
            x (np.ndarray): The x-coordinate of the point.
            y (np.ndarray): The y-coordinate of the point.

        Returns:
            np.ndarray: Boolean array indicating if the point is inside the
                aperture

        """
        # pragma: no cover

    def clip(self, rays):
        """Clips the given rays based on the aperture's shape.

        Args:
            rays (RealRays): List of rays to be clipped.

        Returns:
            list: List of clipped rays.

        """
        inside = self.contains(rays.x, rays.y)
        rays.clip(~inside)

    @abstractmethod
    def scale(self, scale_factor):
        """Scales the aperture by the given factor.

        Args:
            scale_factor (float): The factor by which to scale the aperture.

        """
        # pragma: no cover

    def to_dict(self):
        """Convert the aperture to a dictionary.

        Returns:
            dict: The dictionary representation of the aperture.

        """
        return {"type": self.__class__.__name__}

    @classmethod
    def from_dict(cls, data):
        """Create an aperture from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the aperture.

        Returns:
            BaseAperture: The aperture object.

        """
        aperture_type = data["type"]
        return cls._registry[aperture_type].from_dict(data)

    def view(self, nx=256, ny=256, ax=None, buffer=1.1, **kwargs):
        """Visualize the aperture.

        Args:
            nx (int): The number of points in the x-direction.
            ny (int): The number of points in the y-direction.
            ax (matplotlib.axes.Axes): The axes to plot on.
            buffer (float): The buffer around the aperture.
            **kwargs: Additional keyword arguments to pass to the plot
                function.

        """
        x_min, x_max, y_min, y_max = self.extent
        x_min *= buffer
        x_max *= buffer
        y_min *= buffer
        y_max *= buffer
        if ax is None:
            fig, ax = plt.subplots()
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y)
        Z = self.contains(X, Y)
        ax.contourf(X, Y, Z, **kwargs)
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_aspect("equal")

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
    """Base class for boolean operations on apertures.

    Args:
        a (BaseAperture): The first aperture.
        b (BaseAperture): The second aperture.

    """

    def __init__(self, a: BaseAperture, b: BaseAperture):
        self.a = a
        self.b = b

    @property
    def extent(self):
        """Returns the extent of the aperture.

        Returns:
            tuple: The extent of the aperture in the x and y directions.

        """
        a_extent = self.a.extent
        b_extent = self.b.extent
        x_min = min(a_extent[0], b_extent[0])
        x_max = max(a_extent[1], b_extent[1])
        y_min = min(a_extent[2], b_extent[2])
        y_max = max(a_extent[3], b_extent[3])
        return x_min, x_max, y_min, y_max

    @abstractmethod
    def contains(self, x, y):
        """Checks if the given point is inside the aperture.

        Args:
            x (np.ndarray): The x-coordinate of the point.
            y (np.ndarray): The y-coordinate of the point.

        Returns:
            np.ndarray: Boolean array indicating if the point is inside the
                aperture

        """
        # pragma: no cover

    def scale(self, scale_factor):
        """Scales the aperture by the given factor.

        Args:
            scale_factor (float): The factor by which to scale the aperture.

        """
        self.a.scale(scale_factor)
        self.b.scale(scale_factor)

    def to_dict(self):
        """Convert the aperture to a dictionary.

        Returns:
            dict: The dictionary representation of the aperture.

        """
        data = super().to_dict()
        data.update({"a": self.a.to_dict(), "b": self.b.to_dict()})
        return data

    @classmethod
    def from_dict(cls, data):
        """Create an aperture from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the aperture.

        Returns:
            BaseBooleanAperture: The aperture object.

        """
        a = BaseAperture.from_dict(data["a"])
        b = BaseAperture.from_dict(data["b"])
        return cls(a, b)


class UnionAperture(BaseBooleanAperture):
    """Class for union of two apertures.

    Args:
        a (BaseAperture): The first aperture.
        b (BaseAperture): The second aperture.

    """

    def __init__(self, a: BaseAperture, b: BaseAperture):
        super().__init__(a, b)

    def contains(self, x, y):
        """Checks if the given point is inside either aperture.

        Args:
            x (np.ndarray): The x-coordinate of the point.
            y (np.ndarray): The y-coordinate of the point.

        Returns:
            np.ndarray: Boolean array indicating if the point is inside the
                aperture

        """
        return np.logical_or(self.a.contains(x, y), self.b.contains(x, y))


class IntersectionAperture(BaseBooleanAperture):
    """Class for intersection of two apertures.

    Args:
        a (BaseAperture): The first aperture.
        b (BaseAperture): The second aperture.

    """

    def __init__(self, a: BaseAperture, b: BaseAperture):
        super().__init__(a, b)

    def contains(self, x, y):
        """Checks if the given point is inside the aperture.

        Args:
            x (np.ndarray): The x-coordinate of the point.
            y (np.ndarray): The y-coordinate of the point.

        Returns:
            np.ndarray: Boolean array indicating if the point is inside the
                aperture

        """
        return np.logical_and(self.a.contains(x, y), self.b.contains(x, y))


class DifferenceAperture(BaseBooleanAperture):
    """Class for difference of two apertures.

    Args:
        a (BaseAperture): The first aperture.
        b (BaseAperture): The second aperture.

    """

    def __init__(self, a: BaseAperture, b: BaseAperture):
        super().__init__(a, b)

    def contains(self, x, y):
        """Checks if the given point is inside the aperture.

        Args:
            x (np.ndarray): The x-coordinate of the point.
            y (np.ndarray): The y-coordinate of the point.

        Returns:
            np.ndarray: Boolean array indicating if the point is inside the
                aperture

        """
        return np.logical_and(
            self.a.contains(x, y),
            np.logical_not(self.b.contains(x, y)),
        )
