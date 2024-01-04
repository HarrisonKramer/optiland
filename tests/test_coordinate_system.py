import pytest
import numpy as np
from optiland import coordinate_system, rays


def generate_real_rays():
    x = np.zeros(1)
    y = np.zeros(1)
    z = np.zeros(1)

    L = np.zeros(1)
    M = np.zeros(1)
    N = np.ones(1)

    energy = np.ones(1)
    wavelength = np.ones(1)

    r = rays.RealRays(x, y, z, L, M, N,
                      energy, wavelength)
    return r


def generate_input_data():
    """helper function to generate several coordinate systems
    and target values"""
    x = [0, -5, 0, 0, 2]
    y = [0, 0, 3.2, 0, -23.3]
    z = [0, 0, 0, 63.5, 5.2]
    rx = [0, np.pi/2, 0, 0, -np.pi]
    ry = [0, 0, -np.pi/2, 0, np.pi/2]
    rz = [0, 0, 0, -np.pi, 3*np.pi/2]

    cs = []
    for k in range(len(x)):
        new_cs = coordinate_system.CoordinateSystem(x[k], y[k], z[k],
                                                    rx[k], ry[k], rz[k])
        cs.append(new_cs)

    for k in range(1, len(x)):
        ref_cs = coordinate_system.CoordinateSystem(x[k-1], y[k-1], z[k-1],
                                                    rx[k-1], ry[k-1], rz[k-1])
        new_cs = coordinate_system.CoordinateSystem(x[k], y[k], z[k],
                                                    rx[k], ry[k], rz[k],
                                                    reference_cs=ref_cs)
        cs.append(new_cs)

    # find expected position/rotations of simple ray along z axis


# @pytest.mark.parametrize('cs', generate_input_data())
def test_localize_rays_simple():
    # r = generate_real_rays_simple()
    pass
