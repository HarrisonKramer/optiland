import numpy as np
from optiland.rays import RealRays


def transform(x, y, z, surface, is_global=True):
    """
    Transforms the coordinates (x, y, z) based on the surface geometry.

    Args:
        x (numpy.ndarray): The x-coordinates of the points.
        y (numpy.ndarray): The y-coordinates of the points.
        z (numpy.ndarray): The z-coordinates of the points.
        surface (Surface): The surface object that contains the geometry for
            transformation.
        is_global (bool, optional): If True, localize the points to the
            surface geometry. If False, globalize the points. Defaults to True.
    Returns:
        tuple: Transformed x, y, and z coordinates as numpy arrays.
    """
    t = np.zeros(x.shape[0])
    points = RealRays(x, y, z, t, t, t, t, t)
    if is_global:
        surface.geometry.localize(points)
    else:
        surface.geometry.globalize(points)

    return points.x, points.y, points.z
