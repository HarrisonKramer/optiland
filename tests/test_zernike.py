import pytest
import numpy as np
from optiland import zernike


class TestZernikeStandard:
    def test_get_term(self):
        z = zernike.ZernikeStandard()
        coeff = 1.0
        n = 3
        m = -2
        r = 0.5
        phi = np.pi / 4

        term = z.get_term(coeff, n, m, r, phi)

        assert term == pytest.approx(-1.0606601717798214)

    def test_terms(self):
        z = zernike.ZernikeStandard(coeffs=[0.2, 0.8, 0.4, -0.8, -0.1])
        r = np.array([0.1, 0.2, 0.3])
        phi = np.array([0.0, np.pi / 2, np.pi])

        terms = z.terms(r, phi)
        assert np.allclose(terms[0], np.array([0.2, 0.2, 0.2]))
        assert np.allclose(terms[1], np.array([0.0, -0.32, 0.0]))
        assert np.allclose(terms[2], np.array([0.08,  0.0, -0.24]))
        assert np.allclose(terms[3], np.array([0.0, 0.0, 0.0]))
        assert np.allclose(terms[4], np.array([0.16974098, 0.15934867,
                                               0.14202817]))

    def test_poly(self):
        z = zernike.ZernikeStandard(coeffs=[0.5, -0.5, 0.3, -0.8, 1.0])
        r = np.array([0.1, 0.2, 0.3])
        phi = np.array([0.0, np.pi / 2, np.pi])

        poly = z.poly(r, phi)
        assert np.allclose(poly, np.array([-1.13740979, -0.89348674,
                                           -1.10028166]))

    def test_radial_term(self):
        z = zernike.ZernikeStandard()
        n = 3
        m = -2
        r = 0.5

        radial_term = z._radial_term(n, m, r)
        assert radial_term == pytest.approx(0.375)

    def test_azimuthal_term(self):
        z = zernike.ZernikeStandard()
        m = -2
        phi = np.pi / 4.5

        azimuthal_term = z._azimuthal_term(m, phi)
        assert azimuthal_term == pytest.approx(-0.984807753012208)

    def test_norm_constant(self):
        z = zernike.ZernikeStandard()
        n = 3
        m = -2

        norm_constant = z._norm_constant(n, m)
        assert norm_constant == pytest.approx(2.8284271247461903)

    def test_generate_indices(self):
        z = zernike.ZernikeStandard()

        indices = z._generate_indices()

        assert indices == [(0, 0), (1, -1), (1, 1), (2, -2), (2, 0),
                           (2, 2), (3, -3), (3, -1), (3, 1), (3, 3),
                           (4, -4), (4, -2), (4, 0), (4, 2), (4, 4),
                           (5, -5), (5, -3), (5, -1), (5, 1), (5, 3),
                           (5, 5), (6, -6), (6, -4), (6, -2), (6, 0),
                           (6, 2), (6, 4), (6, 6), (7, -7), (7, -5),
                           (7, -3), (7, -1), (7, 1), (7, 3), (7, 5),
                           (7, 7), (8, -8), (8, -6), (8, -4), (8, -2),
                           (8, 0), (8, 2), (8, 4), (8, 6), (8, 8),
                           (9, -9), (9, -7), (9, -5), (9, -3), (9, -1),
                           (9, 1), (9, 3), (9, 5), (9, 7), (9, 9),
                           (10, -10), (10, -8), (10, -6), (10, -4), (10, -2),
                           (10, 0), (10, 2), (10, 4), (10, 6), (10, 8),
                           (10, 10), (11, -11), (11, -9), (11, -7), (11, -5),
                           (11, -3), (11, -1), (11, 1), (11, 3), (11, 5),
                           (11, 7), (11, 9), (11, 11), (12, -12), (12, -10),
                           (12, -8), (12, -6), (12, -4), (12, -2), (12, 0),
                           (12, 2), (12, 4), (12, 6), (12, 8), (12, 10),
                           (12, 12), (13, -13), (13, -11), (13, -9), (13, -7),
                           (13, -5), (13, -3), (13, -1), (13, 1), (13, 3),
                           (13, 5), (13, 7), (13, 9), (13, 11), (13, 13),
                           (14, -14), (14, -12), (14, -10), (14, -8), (14, -6),
                           (14, -4), (14, -2), (14, 0), (14, 2), (14, 4),
                           (14, 6), (14, 8), (14, 10), (14, 12), (14, 14)]
