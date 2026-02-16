import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import optiland.backend as be
from optiland.colorimetry import core
from optiland.colorimetry.plotting import plot_cie_1931_chromaticity_diagram
from tests.utils import assert_allclose


matplotlib.use("Agg")  # use non-interactive backend for testing


def _as_float(value):
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def test_spectrum_to_xyz_white_reflectance():
    wavelengths = list(range(380, 781, 5))
    values = [1.0] * len(wavelengths)

    X, Y, Z = core.spectrum_to_xyz(wavelengths=wavelengths, values=values)
    x, y, _ = core.xyz_to_xyY(X, Y, Z)
    r, g, b = core.xyz_to_srgb(X, Y, Z)

    assert_allclose(Y, 100.0, rtol=1e-6, atol=1e-6)
    assert abs(_as_float(x) - 0.3270924866130143) < 1e-6
    assert abs(_as_float(y) - 0.33730940791054975) < 1e-6
    assert _as_float(r) == 255
    assert _as_float(g) == 252
    assert _as_float(b) == 243


def test_spectrum_to_xyz_requires_visible_range():
    wavelengths = list(range(420, 701, 10))
    values = [1.0] * len(wavelengths)

    with pytest.raises(ValueError, match="does not cover"):
        core.spectrum_to_xyz(wavelengths=wavelengths, values=values)


def test_spectrum_to_xyz_illuminant_length_check():
    wavelengths = list(range(380, 781, 5))
    values = [1.0] * len(wavelengths)
    illuminant = [1.0] * (len(wavelengths) - 1)

    with pytest.raises(ValueError, match="Illuminant must have the same size"):
        core.spectrum_to_xyz(
            wavelengths=wavelengths,
            values=values,
            illuminant=illuminant,
        )


def test_xyz_to_xyY_vectorized_inputs():
    xyz = be.array(
        [
            [41.24, 21.26, 1.93],
            [35.76, 71.52, 11.92],
        ]
    )

    x, y, Y = core.xyz_to_xyY(xyz)

    xy_sum = xyz[:, 0] + xyz[:, 1] + xyz[:, 2]
    expected_x = xyz[:, 0] / xy_sum
    expected_y = xyz[:, 1] / xy_sum

    assert_allclose(x, expected_x)
    assert_allclose(y, expected_y)
    assert_allclose(Y, xyz[:, 1])


def test_xyz_to_xyY_handles_zero_sum():
    x, y, Y = core.xyz_to_xyY(0.0, 0.0, 0.0)

    assert _as_float(Y) == 0.0
    assert abs(_as_float(x) - 0.3127) < 1e-6
    assert abs(_as_float(y) - 0.3290) < 1e-6


def test_xyz_to_srgb_white_point():
    # D65 white point in XYZ (scaled for Y=100)
    X, Y, Z = 95.047, 100.0, 108.883
    r, g, b = core.xyz_to_srgb(X, Y, Z)

    r_val = _as_float(r)
    g_val = _as_float(g)
    b_val = _as_float(b)

    assert r_val == 255
    assert g_val == 254
    assert b_val == 254


def test_plot_cie_1931_diagram_no_legend():
    fig, ax = plot_cie_1931_chromaticity_diagram(show_legend=False)
    assert fig is not None
    assert ax is not None
    assert ax.get_legend() is None
    plt.close(fig)


def test_plot_cie_1931_diagram_with_legend():
    fig, ax = plot_cie_1931_chromaticity_diagram(color="contour", show_legend=True)
    assert fig is not None
    assert ax is not None
    assert ax.get_legend() is not None
    plt.close(fig)
