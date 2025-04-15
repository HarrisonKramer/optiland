from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import optiland.backend as be
import pytest

from optiland import zernike

matplotlib.use("Agg")  # use non-interactive backend for testing


class TestZernikeStandard:
    def test_get_term(self):
        z = zernike.ZernikeStandard()
        coeff = 1.0
        n = 3
        m = -2
        r = 0.5
        phi = be.pi / 4

        term = z.get_term(coeff, n, m, r, phi)

        assert term == pytest.approx(1.0606601717798214)

    def test_invalid_num_terms(self):
        with pytest.raises(ValueError):
            zernike.ZernikeStandard(coeffs=[0 for _ in range(125)])

    def test_terms(self):
        z = zernike.ZernikeStandard(coeffs=[0.2, 0.8, 0.4, -0.8, -0.1])
        r = be.array([0.1, 0.2, 0.3])
        phi = be.array([0.0, be.pi / 2, be.pi])

        terms = z.terms(r, phi)
        assert be.allclose(terms[0], be.array([0.2, 0.2, 0.2]))
        assert be.allclose(terms[1], be.array([0.0, 0.32, 0.0]))
        assert be.allclose(terms[2], be.array([0.08, 0.0, -0.24]))
        assert be.allclose(terms[3], be.array([0.0, 0.0, 0.0]))
        assert be.allclose(terms[4], be.array([0.16974098, 0.15934867, 0.14202817]))

    def test_poly(self):
        z = zernike.ZernikeStandard(coeffs=[0.5, -0.5, 0.3, -0.8, 1.0])
        r = be.array([0.1, 0.2, 0.3])
        phi = be.array([0.0, be.pi / 2, be.pi])

        poly = z.poly(r, phi)
        assert be.allclose(poly, be.array([-1.13740979, -1.29348674, -1.10028166]))

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
        phi = be.pi / 4.5

        azimuthal_term = z._azimuthal_term(m, phi)
        assert azimuthal_term == pytest.approx(0.984807753012208)

    def test_norm_constant(self):
        z = zernike.ZernikeStandard()
        n = 3
        m = -2

        norm_constant = z._norm_constant(n, m)
        assert norm_constant == pytest.approx(2.8284271247461903)

    def test_generate_indices(self):
        z = zernike.ZernikeStandard()

        indices = z._generate_indices()

        assert indices == [
            (0, 0),
            (1, -1),
            (1, 1),
            (2, -2),
            (2, 0),
            (2, 2),
            (3, -3),
            (3, -1),
            (3, 1),
            (3, 3),
            (4, -4),
            (4, -2),
            (4, 0),
            (4, 2),
            (4, 4),
            (5, -5),
            (5, -3),
            (5, -1),
            (5, 1),
            (5, 3),
            (5, 5),
            (6, -6),
            (6, -4),
            (6, -2),
            (6, 0),
            (6, 2),
            (6, 4),
            (6, 6),
            (7, -7),
            (7, -5),
            (7, -3),
            (7, -1),
            (7, 1),
            (7, 3),
            (7, 5),
            (7, 7),
            (8, -8),
            (8, -6),
            (8, -4),
            (8, -2),
            (8, 0),
            (8, 2),
            (8, 4),
            (8, 6),
            (8, 8),
            (9, -9),
            (9, -7),
            (9, -5),
            (9, -3),
            (9, -1),
            (9, 1),
            (9, 3),
            (9, 5),
            (9, 7),
            (9, 9),
            (10, -10),
            (10, -8),
            (10, -6),
            (10, -4),
            (10, -2),
            (10, 0),
            (10, 2),
            (10, 4),
            (10, 6),
            (10, 8),
            (10, 10),
            (11, -11),
            (11, -9),
            (11, -7),
            (11, -5),
            (11, -3),
            (11, -1),
            (11, 1),
            (11, 3),
            (11, 5),
            (11, 7),
            (11, 9),
            (11, 11),
            (12, -12),
            (12, -10),
            (12, -8),
            (12, -6),
            (12, -4),
            (12, -2),
            (12, 0),
            (12, 2),
            (12, 4),
            (12, 6),
            (12, 8),
            (12, 10),
            (12, 12),
            (13, -13),
            (13, -11),
            (13, -9),
            (13, -7),
            (13, -5),
            (13, -3),
            (13, -1),
            (13, 1),
            (13, 3),
            (13, 5),
            (13, 7),
            (13, 9),
            (13, 11),
            (13, 13),
            (14, -14),
            (14, -12),
            (14, -10),
            (14, -8),
            (14, -6),
            (14, -4),
            (14, -2),
            (14, 0),
            (14, 2),
            (14, 4),
            (14, 6),
            (14, 8),
            (14, 10),
            (14, 12),
            (14, 14),
        ]


class TestZernikeFringe:
    def test_get_term(self):
        z = zernike.ZernikeFringe()
        coeff = 1.0
        n = 3
        m = -2
        r = 0.5
        phi = be.pi / 4

        term = z.get_term(coeff, n, m, r, phi)

        assert term == pytest.approx(0.375)

    def test_terms(self):
        z = zernike.ZernikeFringe(coeffs=[0.2, 0.8, 0.4, -0.8, -0.1])
        r = be.array([0.1, 0.2, 0.3])
        phi = be.array([0.0, be.pi / 2, be.pi])

        terms = z.terms(r, phi)
        assert be.allclose(terms[0], be.array([0.2, 0.2, 0.2]))
        assert be.allclose(terms[1], be.array([0.08, 0.0, -0.24]))
        assert be.allclose(terms[2], be.array([0.0, 0.08, 0.0]))
        assert be.allclose(terms[3], be.array([0.784, 0.736, 0.656]))
        assert be.allclose(terms[4], be.array([-0.001, 0.004, -0.009]))

    def test_poly(self):
        z = zernike.ZernikeFringe(coeffs=[0.5, -0.5, 0.3, -0.8, 1.0])
        r = be.array([0.1, 0.2, 0.3])
        phi = be.array([0.0, be.pi / 2, be.pi])

        poly = z.poly(r, phi)
        assert be.allclose(poly, be.array([1.244, 1.256, 1.396]))

    def test_radial_term(self):
        z = zernike.ZernikeFringe()
        n = 3
        m = -2
        r = 0.5

        radial_term = z._radial_term(n, m, r)
        assert radial_term == pytest.approx(0.375)

    def test_azimuthal_term(self):
        z = zernike.ZernikeFringe()
        m = -2
        phi = be.pi / 4.5

        azimuthal_term = z._azimuthal_term(m, phi)
        assert azimuthal_term == pytest.approx(0.984807753012208)

    def test_norm_constant(self):
        z = zernike.ZernikeFringe()
        n = 3
        m = -2

        norm_constant = z._norm_constant(n, m)
        assert norm_constant == pytest.approx(1)

    def test_generate_indices(self):
        z = zernike.ZernikeFringe()

        indices = z._generate_indices()

        assert indices == [
            (0, 0),
            (1, 1),
            (1, -1),
            (2, 0),
            (2, 2),
            (2, -2),
            (3, 1),
            (3, -1),
            (4, 0),
            (3, 3),
            (3, -3),
            (4, 2),
            (4, -2),
            (5, 1),
            (5, -1),
            (6, 0),
            (4, 4),
            (4, -4),
            (5, 3),
            (5, -3),
            (6, 2),
            (6, -2),
            (7, 1),
            (7, -1),
            (8, 0),
            (5, 5),
            (5, -5),
            (6, 4),
            (6, -4),
            (7, 3),
            (7, -3),
            (8, 2),
            (8, -2),
            (9, 1),
            (9, -1),
            (10, 0),
            (6, 6),
            (6, -6),
            (7, 5),
            (7, -5),
            (8, 4),
            (8, -4),
            (9, 3),
            (9, -3),
            (10, 2),
            (10, -2),
            (11, 1),
            (11, -1),
            (12, 0),
            (7, 7),
            (7, -7),
            (8, 6),
            (8, -6),
            (9, 5),
            (9, -5),
            (10, 4),
            (10, -4),
            (11, 3),
            (11, -3),
            (12, 2),
            (12, -2),
            (13, 1),
            (13, -1),
            (14, 0),
            (8, 8),
            (8, -8),
            (9, 7),
            (9, -7),
            (10, 6),
            (10, -6),
            (11, 5),
            (11, -5),
            (12, 4),
            (12, -4),
            (13, 3),
            (13, -3),
            (14, 2),
            (14, -2),
            (15, 1),
            (15, -1),
            (16, 0),
            (9, 9),
            (9, -9),
            (10, 8),
            (10, -8),
            (11, 7),
            (11, -7),
            (12, 6),
            (12, -6),
            (13, 5),
            (13, -5),
            (14, 4),
            (14, -4),
            (15, 3),
            (15, -3),
            (16, 2),
            (16, -2),
            (17, 1),
            (17, -1),
            (18, 0),
            (10, 10),
            (10, -10),
            (11, 9),
            (11, -9),
            (12, 8),
            (12, -8),
            (13, 7),
            (13, -7),
            (14, 6),
            (14, -6),
            (15, 5),
            (15, -5),
            (16, 4),
            (16, -4),
            (17, 3),
            (17, -3),
            (18, 2),
            (18, -2),
            (19, 1),
            (19, -1),
        ]


class TestZernikeNoll:
    def test_get_term(self):
        z = zernike.ZernikeNoll()
        coeff = 1.0
        n = 3
        m = -2
        r = 0.5
        phi = be.pi / 4

        term = z.get_term(coeff, n, m, r, phi)

        assert term == pytest.approx(1.0606601717798214)

    def test_terms(self):
        z = zernike.ZernikeNoll(coeffs=[0.2, 0.8, 0.4, -0.8, -0.1])
        r = be.array([0.1, 0.2, 0.3])
        phi = be.array([0.0, be.pi / 2, be.pi])

        terms = z.terms(r, phi)
        assert be.allclose(terms[0], be.array([0.2, 0.2, 0.2]))
        assert be.allclose(terms[1], be.array([0.16, 0.0, -0.48]))
        assert be.allclose(terms[2], be.array([0.0, 0.16, 0.0]))
        assert be.allclose(terms[3], be.array([1.35792783, 1.27478939, 1.13622533]))
        assert be.allclose(terms[4], be.array([0.0, 0.0, 0.0]))

    def test_poly(self):
        z = zernike.ZernikeNoll(coeffs=[0.5, -0.5, 0.3, -0.8, 1.0])
        r = be.array([0.1, 0.2, 0.3])
        phi = be.array([0.0, be.pi / 2, be.pi])

        poly = z.poly(r, phi)
        assert be.allclose(poly, be.array([1.75792783, 1.89478939, 1.93622533]))

    def test_radial_term(self):
        z = zernike.ZernikeNoll()
        n = 3
        m = -2
        r = 0.5

        radial_term = z._radial_term(n, m, r)
        assert radial_term == pytest.approx(0.375)

    def test_azimuthal_term(self):
        z = zernike.ZernikeNoll()
        m = -2
        phi = be.pi / 4.5

        azimuthal_term = z._azimuthal_term(m, phi)
        assert azimuthal_term == pytest.approx(0.984807753012208)

    def test_norm_constant(self):
        z = zernike.ZernikeNoll()
        n = 3
        m = -2

        norm_constant = z._norm_constant(n, m)
        assert norm_constant == pytest.approx(2.8284271247461903)

    def test_generate_indices(self):
        z = zernike.ZernikeNoll()

        indices = z._generate_indices()

        assert indices == [
            (0, 0),
            (1, 1),
            (1, -1),
            (2, 0),
            (2, -2),
            (2, 2),
            (3, -1),
            (3, 1),
            (3, -3),
            (3, 3),
            (4, 0),
            (4, 2),
            (4, -2),
            (4, 4),
            (4, -4),
            (5, 1),
            (5, -1),
            (5, 3),
            (5, -3),
            (5, 5),
            (5, -5),
            (6, 0),
            (6, -2),
            (6, 2),
            (6, -4),
            (6, 4),
            (6, -6),
            (6, 6),
            (7, -1),
            (7, 1),
            (7, -3),
            (7, 3),
            (7, -5),
            (7, 5),
            (7, -7),
            (7, 7),
            (8, 0),
            (8, 2),
            (8, -2),
            (8, 4),
            (8, -4),
            (8, 6),
            (8, -6),
            (8, 8),
            (8, -8),
            (9, 1),
            (9, -1),
            (9, 3),
            (9, -3),
            (9, 5),
            (9, -5),
            (9, 7),
            (9, -7),
            (9, 9),
            (9, -9),
            (10, 0),
            (10, -2),
            (10, 2),
            (10, -4),
            (10, 4),
            (10, -6),
            (10, 6),
            (10, -8),
            (10, 8),
            (10, -10),
            (10, 10),
            (11, -1),
            (11, 1),
            (11, -3),
            (11, 3),
            (11, -5),
            (11, 5),
            (11, -7),
            (11, 7),
            (11, -9),
            (11, 9),
            (11, -11),
            (11, 11),
            (12, 0),
            (12, 2),
            (12, -2),
            (12, 4),
            (12, -4),
            (12, 6),
            (12, -6),
            (12, 8),
            (12, -8),
            (12, 10),
            (12, -10),
            (12, 12),
            (12, -12),
            (13, 1),
            (13, -1),
            (13, 3),
            (13, -3),
            (13, 5),
            (13, -5),
            (13, 7),
            (13, -7),
            (13, 9),
            (13, -9),
            (13, 11),
            (13, -11),
            (13, 13),
            (13, -13),
            (14, 0),
            (14, -2),
            (14, 2),
            (14, -4),
            (14, 4),
            (14, -6),
            (14, 6),
            (14, -8),
            (14, 8),
            (14, -10),
            (14, 10),
            (14, -12),
            (14, 12),
            (14, -14),
            (14, 14),
        ]


class TestZernikeFit:
    def setup_method(self):
        coeffs = [0.1, 0.2, 0.8, -0.2, 0.4, 0.6, 0.0, 0.0, 0.5, 0.1, -0.3]

        z = zernike.ZernikeStandard(coeffs=coeffs)

        x = be.linspace(-1, 1, 10)
        x, y = be.meshgrid(x, x)
        x = be.ravel(x)
        y = be.ravel(y)
        r = be.sqrt(x**2 + y**2)
        cond = r <= 1
        x = x[cond]
        y = y[cond]
        r = r[cond]
        theta = be.arctan2(y, x)

        self.x = x
        self.y = y
        self.z = z.poly(r, theta)

    def test_init(self):
        zernike_fit = zernike.ZernikeFit(self.x, self.y, self.z)
        assert be.allclose(zernike_fit.x, self.x)
        assert be.allclose(zernike_fit.y, self.y)
        assert be.allclose(zernike_fit.z, self.z)
        assert zernike_fit.type == "fringe"
        assert zernike_fit.num_terms == 36

    def test_init_standard(self):
        zernike_fit_standard = zernike.ZernikeFit(
            self.x,
            self.y,
            self.z,
            zernike_type="standard",
        )
        assert be.allclose(zernike_fit_standard.x, self.x)
        assert be.allclose(zernike_fit_standard.y, self.y)
        assert be.allclose(zernike_fit_standard.z, self.z)
        assert zernike_fit_standard.type == "standard"
        assert zernike_fit_standard.num_terms == 36

    def test_init_noll(self):
        zernike_fit_noll = zernike.ZernikeFit(
            self.x,
            self.y,
            self.z,
            zernike_type="noll",
        )
        assert be.allclose(zernike_fit_noll.x, self.x)
        assert be.allclose(zernike_fit_noll.y, self.y)
        assert be.allclose(zernike_fit_noll.z, self.z)
        assert zernike_fit_noll.type == "noll"
        assert zernike_fit_noll.num_terms == 36

    def test_invalid_zernike_type(self):
        with pytest.raises(ValueError):
            zernike.ZernikeFit(self.x, self.y, self.z, zernike_type="invalid")

    def test_coeffs(self):
        zernike_fit = zernike.ZernikeFit(self.x, self.y, self.z)
        assert len(zernike_fit.coeffs) == 36

        arr = be.array([0.1, 1.6, 0.4, 0.69282032, 1.46969385])
        assert be.allclose(zernike_fit.coeffs[:5], arr)

    def test_coeffs_standard(self):
        zernike_fit_standard = zernike.ZernikeFit(
            self.x,
            self.y,
            self.z,
            zernike_type="standard",
        )
        assert len(zernike_fit_standard.coeffs) == 36

        arr = be.array([0.1, 0.2, 0.8, -0.2, 0.4])
        assert be.allclose(zernike_fit_standard.coeffs[:5], arr)

    def test_coeffs_noll(self):
        zernike_fit_noll = zernike.ZernikeFit(
            self.x,
            self.y,
            self.z,
            zernike_type="noll",
        )
        assert len(zernike_fit_noll.coeffs) == 36

        arr = be.array([0.1, 0.8, 0.2, 0.4, -0.2])
        assert be.allclose(zernike_fit_noll.coeffs[:5], arr)

    def test_invalid_view_projection(self):
        zernike_fit = zernike.ZernikeFit(self.x, self.y, self.z)
        with pytest.raises(ValueError):
            zernike_fit.view(projection="invalid")

    @patch("matplotlib.pyplot.show")
    def test_view_standard(self, mock_show):
        zernike_fit_standard = zernike.ZernikeFit(
            self.x,
            self.y,
            self.z,
            zernike_type="standard",
        )
        zernike_fit_standard.view(projection="2d")
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_standard_3d(self, mock_show):
        zernike_fit_standard = zernike.ZernikeFit(
            self.x,
            self.y,
            self.z,
            zernike_type="standard",
        )
        zernike_fit_standard.view(projection="3d")
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_noll(self, mock_show):
        zernike_fit_noll = zernike.ZernikeFit(
            self.x,
            self.y,
            self.z,
            zernike_type="noll",
        )
        zernike_fit_noll.view(projection="2d")
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_noll_3d(self, mock_show):
        zernike_fit_noll = zernike.ZernikeFit(
            self.x,
            self.y,
            self.z,
            zernike_type="noll",
        )
        zernike_fit_noll.view(projection="3d")
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_residual_standard(self, mock_show):
        zernike_fit_standard = zernike.ZernikeFit(
            self.x,
            self.y,
            self.z,
            zernike_type="standard",
        )
        zernike_fit_standard.view_residual()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_residual_noll(self, mock_show):
        zernike_fit_noll = zernike.ZernikeFit(
            self.x,
            self.y,
            self.z,
            zernike_type="noll",
        )
        zernike_fit_noll.view_residual()
        mock_show.assert_called_once()
        plt.close()

    def test_objective(self):
        zernike_fit = zernike.ZernikeFit(self.x, self.y, self.z)
        coeffs = [0.1 for _ in range(zernike_fit.num_terms)]
        objective = zernike_fit._objective(coeffs)
        assert len(objective) == len(self.z)

        arr = be.array([1.30233667, 0.85157615, 0.88422155, 0.96304606, 0.58096649])
        assert be.allclose(objective[:5], arr)

    def test_objective_standard(self):
        zernike_fit_standard = zernike.ZernikeFit(
            self.x,
            self.y,
            self.z,
            zernike_type="standard",
        )
        coeffs = [0.1 for _ in range(zernike_fit_standard.num_terms)]
        objective = zernike_fit_standard._objective(coeffs)
        assert len(objective) == len(self.z)

        arr = be.array([1.0484952, 0.93610002, 1.09385253, 0.79558155, 0.41807518])
        assert be.allclose(objective[:5], arr)

    def test_objective_noll(self):
        zernike_fit_noll = zernike.ZernikeFit(
            self.x,
            self.y,
            self.z,
            zernike_type="noll",
        )
        coeffs = [0.1 for _ in range(zernike_fit_noll.num_terms)]
        objective = zernike_fit_noll._objective(coeffs)
        assert len(objective) == len(self.z)

        arr = be.array([1.0484952, 0.93610002, 1.09385253, 0.79558155, 0.41807518])
        assert be.allclose(objective[:5], arr)

    @patch("matplotlib.pyplot.show")
    def test_view(self, mock_show):
        zernike_fit = zernike.ZernikeFit(self.x, self.y, self.z)
        zernike_fit.view(projection="2d")
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_3d(self, mock_show):
        zernike_fit = zernike.ZernikeFit(self.x, self.y, self.z)
        zernike_fit.view(projection="3d")
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_residual(self, mock_show):
        zernike_fit = zernike.ZernikeFit(self.x, self.y, self.z)
        zernike_fit.view_residual()
        mock_show.assert_called_once()
        plt.close()
