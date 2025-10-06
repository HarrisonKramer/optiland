# tests/geometries/test_zernike.py
"""
Tests for the ZernikePolynomialGeometry class in optiland.geometries.
"""
from contextlib import nullcontext as does_not_raise
import numpy as np
import pytest

import optiland.backend as be
from optiland import geometries
from optiland.coordinate_system import CoordinateSystem
from ..utils import assert_allclose, assert_array_equal


class TestZernikeGeometry:
    """
    Tests for the ZernikePolynomialGeometry class, which represents a surface
    with a departure defined by Zernike polynomials.
    """

    def coefficients_dict_to_list(
        self, coefficients: dict[int, float], zernike_type: str
    ) -> list[float]:
        """
        Converts a dictionary of Zernike coefficients to a list, which is the
        format expected by the geometry class.
        """
        start = 0 if zernike_type == "standard" else 1
        return [
            coefficients.get(i, 0.0)
            for i in range(start, max(coefficients.keys()) + 1)
        ]

    def create_geometry(
        self,
        coefficients,
        norm_radius: float = 10,
        zernike_type: str = "fringe",
        radius=22,
        conic=0.0,
    ) -> geometries.ZernikePolynomialGeometry:
        """
        Helper function to create a ZernikePolynomialGeometry instance for
        testing.
        """
        cs = CoordinateSystem()
        return geometries.ZernikePolynomialGeometry(
            cs,
            radius=radius,
            conic=conic,
            coefficients=coefficients,
            norm_radius=norm_radius,
            zernike_type=zernike_type,
        )

    @pytest.mark.parametrize(
        "norm_radius, expectation",
        [
            (10, does_not_raise()),
            (
                0,
                pytest.raises(
                    ValueError, match="Normalization radius must be positive"
                ),
            ),
            (
                -5,
                pytest.raises(
                    ValueError, match="Normalization radius must be positive"
                ),
            ),
        ],
    )
    def test_init(self, set_test_backend, norm_radius, expectation):
        """
        Tests the initialization of the geometry, including validation of the
        normalization radius.
        """
        coefficients = list(range(10))
        with expectation:
            self.create_geometry(coefficients, norm_radius=norm_radius)

    def test_get_coefficients(self, set_test_backend):
        """
        Tests that the coefficients property returns the correct list of
        coefficients.
        """
        coefficients = list(range(10))
        geometry = self.create_geometry(coefficients)
        assert list(geometry.coefficients) == coefficients
        assert_array_equal(geometry.coefficients, geometry.zernike.coeffs)

    def test_set_coefficients(self, set_test_backend):
        """
        Tests that the coefficients can be updated after initialization.
        """
        coefficients = list(range(10))
        geometry = self.create_geometry(coefficients)
        old_zernike = geometry.zernike
        new_coefficients = list(range(20))
        geometry.coefficients = new_coefficients
        assert list(geometry.coefficients) == new_coefficients
        assert geometry.zernike != old_zernike
        assert len(geometry.zernike.indices) == len(new_coefficients)

    def test_str(self, set_test_backend):
        """
        Tests the string representation of the ZernikePolynomialGeometry.
        """
        coefficients = be.array([[0.0, 1e-2, -2e-3], [0, 0, 0], [0, 0, 0]])
        geometry = self.create_geometry(coefficients)
        assert str(geometry) == "Zernike Polynomial"

    # fmt: off
    REFERENCE_SAG = {
        "standard": np.array(
            [
                [14.5393493 ,  9.06299698,  5.6901489 ,  3.82524018,  3.04804724,
                    3.1079359 ,  3.92161984,  5.57316059,  8.31620907, 12.57975751],
                [10.42095581,  5.89794049,  3.18020179,  1.69752586,  1.05162725,
                    1.01183632,  1.5133748 ,  2.65735586,  4.71250841,  8.11948982],
                [ 7.67790733,  3.9395579 ,  1.76960277,  0.62031594,  0.11396659,
                    0.03927432,  0.34996893,  1.16479049,  2.76892944,  5.61757143],
                [ 5.88121164,  2.74719186,  1.0031662 ,  0.12216224, -0.25386995,
                    -0.31719339, -0.09557018,  0.54773847,  1.91525828,  4.47843744],
                [ 4.7772178 ,  2.05211344,  0.59570162, -0.09946097, -0.37237837,
                    -0.39657086, -0.18129248,  0.42846898,  1.7532095 ,  4.28186429],
                [ 4.28186429,  1.7532095 ,  0.42846898, -0.18129248, -0.39657086,
                    -0.37237837, -0.09946097,  0.59570162,  2.05211344,  4.7772178 ],
                [ 4.47843744,  1.91525828,  0.54773847, -0.09557018, -0.31719339,
                    -0.25386995,  0.12216224,  1.0031662 ,  2.74719186,  5.88121164],
                [ 5.61757143,  2.76892944,  1.16479049,  0.34996893,  0.03927432,
                    0.11396659,  0.62031594,  1.76960277,  3.9395579 ,  7.67790733],
                [ 8.11948982,  4.71250841,  2.65735586,  1.5133748 ,  1.01183632,
                    1.05162725,  1.69752586,  3.18020179,  5.89794049, 10.42095581],
                [12.57975751,  8.31620907,  5.57316059,  3.92161984,  3.1079359 ,
                    3.04804724,  3.82524018,  5.6901489 ,  9.06299698, 14.5393493 ]
            ]
         ),
         "fringe": np.array(
            [
                [ 8.84770045,  6.57121389,  4.91876121,  3.83164416,  3.24019577,
                    3.05802842,  3.17979231,  3.48117547,  3.8211453 ,  4.04770045],
                [ 5.98800333,  4.17563074,  2.88912966,  2.08054662,  1.69404966,
                    1.66161558,  1.90130565,  2.31726561,  2.80144997,  3.23793475],
                [ 4.17536504,  2.63007023,  1.54424187,  0.87780814,  0.58738851,
                    0.62274898,  0.92536187,  1.42840587,  2.05820619,  2.73777931],
                [ 3.20777585,  1.78878893,  0.7852545 ,  0.16324809, -0.10903309,
                    -0.06440575,  0.26201352,  0.83280823,  1.60954796,  2.555924  ],
                [ 2.88526343,  1.50466796,  0.51001549, -0.12769556, -0.42932525,
                    -0.41054762, -0.08306822,  0.54537595,  1.47223388,  2.70309608],
                [ 3.01064404,  1.63140338,  0.61602236, -0.06059874, -0.40891508,
                    -0.42769271, -0.10522609,  0.5806619 ,  1.66383746,  3.1928114 ],
                [ 3.39378408,  2.02828494,  1.00548516,  0.30316579, -0.07926121,
                    -0.12388855,  0.20440036,  0.95793143,  2.20752591,  4.04563593],
                [ 3.85810311,  2.56669913,  1.59182947,  0.91293151,  0.53864933,
                    0.50328887,  0.86537779,  1.70766547,  3.13856318,  5.29568884],
                [ 4.24931851,  3.13835661,  2.30763971,  1.73652725,  1.44202168,
                    1.47445576,  1.91576822,  2.87950375,  4.51253737,  6.9993871 ],
                [ 4.44770045,  3.66610796,  3.11470695,  2.76991577,  2.6557117 ,
                    2.83787905,  3.42176762,  4.55229268,  6.41617655,  9.24770045]
            ]
        ),
    }
    # fmt: on

    @pytest.mark.parametrize(
        "zernike_type, coefficients",
        [
            ("standard", {4: 0.5, 3: 0.2, 5: 0.3, 10: 0.1, 12: 0.2}),
            ("noll", {4: 0.5, 5: 0.2, 6: 0.3, 11: 0.2, 15: 0.1}),
            ("fringe", {4: 0.5, 6: 0.2, 11: 0.3, 13: 0.2, 27: 0.1}),
        ],
    )
    def test_sag(
        self, set_test_backend, zernike_type: str, coefficients: dict[int, float]
    ):
        """
        Tests the sag calculation for a Zernike polynomial surface against
        pre-computed reference values for different Zernike conventions.
        """
        geometry = self.create_geometry(
            coefficients=self.coefficients_dict_to_list(coefficients, zernike_type),
            zernike_type=zernike_type,
        )
        x = y = be.linspace(-10, 10, 10)
        X, Y = be.meshgrid(x, y)
        sag = geometry.sag(X, Y)
        reference = self.REFERENCE_SAG.get(zernike_type, self.REFERENCE_SAG["standard"])
        assert_allclose(sag, reference, atol=1e-6)

    @pytest.mark.parametrize(
        "x, y, norm_radius, expectation",
        [
            ([-1, -0.5, 0, 0.5, 1], [0, 0, 0, 0, 0], 1, does_not_raise()),
            ([0, 0, 0, 0, 0], [-1, -0.5, 0, 0.5, 1], 1, does_not_raise()),
            (
                -1.1,
                0,
                1,
                pytest.raises(
                    ValueError, match="Zernike coordinates must be normalized"
                ),
            ),
            (
                0,
                -1.1,
                1,
                pytest.raises(
                    ValueError, match="Zernike coordinates must be normalized"
                ),
            ),
            (
                1.1,
                0,
                1,
                pytest.raises(
                    ValueError, match="Zernike coordinates must be normalized"
                ),
            ),
            (
                0,
                1.1,
                1,
                pytest.raises(
                    ValueError, match="Zernike coordinates must be normalized"
                ),
            ),
        ],
    )
    def test_validate_inputs(self, set_test_backend, x, y, norm_radius, expectation):
        """
        Tests that the geometry raises a ValueError for coordinates outside
        the unit circle normalization radius.
        """
        geometry = self.create_geometry([], norm_radius=norm_radius)
        with expectation:
            geometry._validate_inputs(x, y)

    def test_to_dict(self, set_test_backend):
        """
        Tests the serialization of a ZernikePolynomialGeometry instance to a
        dictionary.
        """
        geometry = self.create_geometry(
            coefficients=[0.5, 0.2, 0.3, 0.1, 0.2],
            zernike_type="standard",
            norm_radius=1.0,
        )
        geometry_dict = geometry.to_dict()
        assert geometry_dict["coefficients"] == [0.5, 0.2, 0.3, 0.1, 0.2]
        assert geometry_dict["zernike_type"] == "standard"
        assert geometry_dict["norm_radius"] == 1.0

    def test_from_dict(self, set_test_backend):
        """
        Tests the deserialization of a ZernikePolynomialGeometry instance
        from a dictionary.
        """
        geometry = self.create_geometry(
            coefficients=[0.5, 0.2, 0.3, 0.1, 0.2],
            zernike_type="standard",
            norm_radius=1.0,
        )
        geometry_dict = geometry.to_dict()
        new_geometry = geometries.ZernikePolynomialGeometry.from_dict(geometry_dict)
        assert all(new_geometry.coefficients == geometry.coefficients)
        assert new_geometry.zernike_type == geometry.zernike_type
        assert new_geometry.norm_radius == geometry.norm_radius