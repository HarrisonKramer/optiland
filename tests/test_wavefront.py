from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import optiland.backend as be
import pytest

from optiland import distribution, wavefront
from optiland.samples.eyepieces import EyepieceErfle
from optiland.samples.objectives import CookeTriplet, DoubleGauss
from tests.utils import assert_allclose

matplotlib.use("Agg")  # use non-interactive backend for testing


class TestWavefront:
    @pytest.mark.parametrize("OpticClass", [CookeTriplet, DoubleGauss, EyepieceErfle])
    def test_wavefront_initialization(self, OpticClass, set_test_backend):
        optic = OpticClass()
        w = wavefront.Wavefront(optic)
        assert w.num_rays == 12
        assert w.fields == optic.fields.get_field_coords()
        assert w.wavelengths == optic.wavelengths.get_wavelengths()
        assert isinstance(w.distribution, distribution.HexagonalDistribution)

    def test_wavefront_init_custom(self, set_test_backend):
        optic = DoubleGauss()
        w = wavefront.Wavefront(
            optic,
            num_rays=100,
            distribution="random",
            wavelengths="primary",
        )
        assert w.num_rays == 100
        assert isinstance(w.distribution, distribution.RandomDistribution)
        assert w.wavelengths == [optic.primary_wavelength]

    def test_generate_data(self, set_test_backend):
        optic = EyepieceErfle()
        w = wavefront.Wavefront(optic)
        w._generate_data()
        assert isinstance(w.data, dict)
        assert isinstance(w.data[((0.0, 0.0), 0.5876)], wavefront.WavefrontData)
        assert isinstance(w.data[((0.0, 0.0), 0.5876)].intensity, be.ndarray)
        assert isinstance(w.data[((0.0, 0.7), 0.5876)].opd, be.ndarray)
        assert isinstance(w.data[((0.0, 0.0), 0.5876)].pupil_x, be.ndarray)
        assert (
            be.size(w.data[((0.0, 1.0), 0.6563)].opd) == 469
        )  # num points in the pupil

    def test_trace_chief_ray(self, set_test_backend):
        optic = DoubleGauss()
        w = wavefront.Wavefront(optic)
        w._trace_chief_ray((0, 0), 0.55)
        assert be.all(optic.surface_group.y == 0)

    def test_get_reference_sphere(self, set_test_backend):
        optic = DoubleGauss()
        w = wavefront.Wavefront(optic)
        w._trace_chief_ray((0, 0), 0.55)
        xc, yc, zc, R = w._get_reference_sphere(pupil_z=100)
        assert be.allclose(xc, be.array([0.0]))
        assert be.allclose(yc, be.array([0.0]))
        assert be.allclose(zc, be.array([139.454938]))
        assert be.allclose(R, be.array([39.454938]))

    def test_get_reference_sphere_error(self, set_test_backend):
        optic = DoubleGauss()
        w = wavefront.Wavefront(optic)
        optic.trace(Hx=0, Hy=0, wavelength=0.55)
        # fails when >1 rays traced in the pupil
        with pytest.raises(ValueError):
            w._get_reference_sphere(pupil_z=100)

    def test_get_path_length(self, set_test_backend):
        optic = CookeTriplet()
        w = wavefront.Wavefront(optic)
        w._trace_chief_ray((0, 0), 0.55)
        xc, yc, zc, R = w._get_reference_sphere(pupil_z=100)
        path_length, _ = w._get_path_length(xc, yc, zc, R, 0.55)
        assert be.allclose(path_length, be.array([34.84418309]))

    def test_correct_tilt(self, set_test_backend):
        optic = DoubleGauss()
        w = wavefront.Wavefront(optic)
        opd = be.linspace(5, 100, be.size(w.distribution.x))
        corrected_opd = w._correct_tilt((0, 1), opd, x=None, y=None)
        assert_allclose(corrected_opd[0], 2.5806903748015824)
        assert_allclose(corrected_opd[10], 5.013823175582515)
        assert_allclose(corrected_opd[100], 24.08949048654609)
        assert_allclose(corrected_opd[111], 24.699015344473096)
        assert_allclose(corrected_opd[242], 52.123070395591235)

    def test_opd_image_to_xp(self, set_test_backend):
        optic = DoubleGauss()
        w = wavefront.Wavefront(optic)
        w._trace_chief_ray((0, 0), 0.55)
        xc, yc, zc, R = w._get_reference_sphere(pupil_z=100)
        t = w._opd_image_to_xp(xc, yc, zc, R, 0.55)
        assert be.allclose(t, be.array([39.454938]))

    def test_opd_image_to_xp_masking_logic(self, set_test_backend):
        # Create a mock optic and wavefront object
        optic_mock = type('OpticMock', (), {})()

        # Mock fields
        fields_mock = type('FieldsMock', (), {})()
        def get_field_coords():
            return [((0.0, 0.0))] # Dummy field
        fields_mock.get_field_coords = get_field_coords
        fields_mock.max_field = 1.0 # Add max_field
        optic_mock.fields = fields_mock
        optic_mock.field_type = "angle" # Add field_type

        # Mock wavelengths
        wavelengths_mock = type('WavelengthsMock', (), {})()
        def get_wavelengths():
            return [0.55] # Dummy wavelength
        wavelengths_mock.get_wavelengths = get_wavelengths
        optic_mock.wavelengths = wavelengths_mock
        optic_mock.primary_wavelength = 0.55

        # Mock paraxial attributes
        paraxial_mock = type('ParaxialMock', (), {})()
        def XPL():
            return 0.0 # Dummy XPL
        paraxial_mock.XPL = XPL
        def EPD():
            return 10.0 # Dummy EPD value
        paraxial_mock.EPD = EPD
        optic_mock.paraxial = paraxial_mock

        # Mock surface_group attributes
        surface_group_mock = type('SurfaceGroupMock', (), {})()
        surface_group_mock.positions = be.array([0.0]) # Dummy positions
        # Initialize L, M, N which are needed by _opd_image_to_xp during Wavefront init
        # These will be (num_surfaces, num_rays_in_chief_ray_trace=1)
        surface_group_mock.L = be.array([[0.0]])
        surface_group_mock.M = be.array([[0.0]])
        surface_group_mock.N = be.array([[-1.0]]) # Pointing along -Z
        optic_mock.surface_group = surface_group_mock

        # Mock image_surface and its material_post
        optic_mock.image_surface = type('ImageSurfaceMock', (), {})()
        optic_mock.image_surface.material_post = type('MaterialPostMock', (), {})()
        def mock_n(wavelength):
            return be.array(1.5) # Example refractive index
        optic_mock.image_surface.material_post.n = mock_n

        # Initialize Wavefront with the mocked optic
        # The Wavefront constructor will call methods on the mocked optic.
        # We need to ensure all accessed attributes are present.
        # The _generate_data call in Wavefront.__init__ is complex.
        # For this specific test, we only care about _opd_image_to_xp,
        # so we can try to bypass full data generation if it causes issues,
        # or ensure all its dependencies are minimally mocked.

        # For _generate_data:
        # pupil_z = self.optic.paraxial.XPL() + self.optic.surface_group.positions[-1] (mocked)
        # self._trace_chief_ray(field, wl) -> optic.trace_generic(*field, Px=0.0, Py=0.0, wavelength=wavelength)
        #   This will update optic.surface_group.x, y, z, L, M, N, opd etc.
        #   We are setting these directly later for _opd_image_to_xp, but trace_generic might be called first.
        # Let's mock trace_generic
        def mock_trace_generic(*args, **kwargs):
            # The actual trace_generic modifies optic_mock.surface_group attributes.
            # We will set these manually for the test of _opd_image_to_xp.
            # However, _get_reference_sphere needs x,y,z to be set by trace_generic for chief ray.
            # And _get_path_length needs opd to be set.
            # These are expected to be 2D arrays: (num_surfaces, num_rays)
            # For chief ray, num_rays = 1. We'll assume 1 surface for simplicity of mock.
            optic_mock.surface_group.x = be.array([[0.0]])
            optic_mock.surface_group.y = be.array([[0.0]])
            # zc is set to 100.0 in the test's common parameters later.
            # R is 50.0. Let chief ray intersect at zc for simplicity of path length calcs during init.
            optic_mock.surface_group.z = be.array([[100.0]])
            optic_mock.surface_group.opd = be.array([[0.0]])
            # Also ensure L,M,N are set by trace_generic for chief ray
            optic_mock.surface_group.L = be.array([[0.0]])
            optic_mock.surface_group.M = be.array([[0.0]])
            optic_mock.surface_group.N = be.array([[-1.0]]) # Ray traveling along -Z
            return None
        optic_mock.trace_generic = mock_trace_generic

        # Mock optic.trace for the _generate_field_data call in Wavefront.__init__
        # This uses self.distribution, which is Hexapolar with num_rays=12 by default (469 points)
        default_num_dist_points = 3 * 12 * (12 + 1) + 1 # 469 for default hexapolar num_rays=12

        def mock_trace(*args, **kwargs):
            # This trace is for the full pupil sampling, not the chief ray.
            # It needs to populate surface_group attributes for these rays.
            # Shape: (num_surfaces, num_rays_in_distribution) - assume 1 surface for mock
            optic_mock.surface_group.x = be.zeros((1, default_num_dist_points))
            optic_mock.surface_group.y = be.zeros((1, default_num_dist_points))
            optic_mock.surface_group.z = be.ones((1, default_num_dist_points)) * 100.0 # Dummy z
            optic_mock.surface_group.L = be.zeros((1, default_num_dist_points))
            optic_mock.surface_group.M = be.zeros((1, default_num_dist_points))
            optic_mock.surface_group.N = -be.ones((1, default_num_dist_points)) # Pointing along -Z
            optic_mock.surface_group.opd = be.zeros((1, default_num_dist_points))
            optic_mock.surface_group.intensity = be.ones((1, default_num_dist_points))

            # Return a "rays" object with x, y, z, L, M, N attributes (1D arrays for rays at exit pupil)
            rays_obj = type('RaysMock', (), {})()
            rays_obj.x = optic_mock.surface_group.x[-1,:]
            rays_obj.y = optic_mock.surface_group.y[-1,:]
            rays_obj.z = optic_mock.surface_group.z[-1,:]
            rays_obj.L = optic_mock.surface_group.L[-1,:]
            rays_obj.M = optic_mock.surface_group.M[-1,:]
            rays_obj.N = optic_mock.surface_group.N[-1,:]
            return rays_obj
        optic_mock.trace = mock_trace


        wf = wavefront.Wavefront(optic_mock)
        # wf.optic will be optic_mock due to constructor.
        # The self.distribution will be initialized by Wavefront constructor.
        # For the actual test of _opd_image_to_xp, we override surface_group attributes later.

        # Common parameters
        xc = be.array([0.0])
        yc = be.array([0.0])
        zc = be.array([100.0])
        R = be.array([50.0])
        wavelength = 0.55

        # Case 1: Mask is all False (all t >= 0 initially)
        # We need d >= 0. Let d = 4. So b^2 - 4ac = 4.
        # Let a = 1. Then b^2 - 4c = 4.
        # Let b = 4. Then 16 - 4c = 4 => 4c = 12 => c = 3.
        # t = (-b - sqrt(d)) / (2a) = (-4 - 2) / 2 = -3. This would make mask True.
        # We want t >= 0. So, (-b - sqrt(d)) / (2a) >= 0.
        # If a > 0, then -b - sqrt(d) >= 0 => -b >= sqrt(d). This requires b to be negative and |b| >= sqrt(d).
        # Let a = be.array([1.0, 1.0])
        # Let d = be.array([4.0, 4.0]) # so sqrt(d) = 2.0
        # Let b = be.array([-4.0, -5.0])
        # Then t = (-(-4) - 2) / 2 = (4-2)/2 = 1.0
        # And t = (-(-5) - 2) / 2 = (5-2)/2 = 1.5
        # Both are >= 0, so mask should be all False.

        # For _opd_image_to_xp, 'a' is L**2 + M**2 + N**2 (always positive)
        # 'b' is 2 * (L * (xr - xc) + M * (yr - yc) + N * (zr - zc))
        # 'c' is (xr**2 + yr**2 + zr**2 - 2 * (xr * xc + yr * yc + zr * zc) + xc**2 + yc**2 + zc**2 - R**2)

        # Case 1: mask all False (initial t >= 0)
        # L, M, N define 'a' and part of 'b'
        # xr, yr, zr define part of 'b' and 'c'
        # These are for the actual test cases, not the Wavefront init.
        # Shape is (num_rays_for_test_case) - this is a slight abuse as SG usually stores (num_surfaces, num_rays)
        # But _opd_image_to_xp uses [-1, :] which effectively takes the last surface's rays.
        # So we mock them as 2D arrays representing rays data at the last surface (1 surface, 2 rays).
        optic_mock.surface_group.L = be.array([[0.0, 0.0]])
        optic_mock.surface_group.M = be.array([[0.0, 0.0]])
        optic_mock.surface_group.N = be.array([[-1.0, -1.0]]) # So a = 1 for these test rays

        optic_mock.surface_group.x = be.array([[0.0, 0.0]]) # xr for test rays
        optic_mock.surface_group.y = be.array([[0.0, 0.0]]) # yr for test rays
        # zr values that ensure c >= 0 and b < 0
        # c = (zr-zc)^2 - R^2 (simplified for x=y=xc=yc=0)
        # Let zr-zc = 60 (so zr=160). c = 60^2 - 50^2 = 3600 - 2500 = 1100 > 0
        # b = 2 * N * (zr-zc). If N=-1, zr-zc=60, then b = 2*(-1)*(60) = -120 < 0
        # Let zr-zc = 70 (so zr=170). c = 70^2 - 50^2 = 4900 - 2500 = 2400 > 0
        # b = 2*(-1)*(70) = -140 < 0
        optic_mock.surface_group.z = be.array([[160.0, 170.0]]) # zr for test rays

        # With xc=0, yc=0, zc=100, R=50:
        # For point 1 (zr=140):
        # a = 1
        # b = 2 * (-1 * (140-100)) = 2 * (-40) = -80
        # c = 140^2 - 2*(140*100) + 100^2 - 50^2 = 19600 - 28000 + 10000 - 2500 = 100
        # d = b^2 - 4ac = (-80)^2 - 4*1*100 = 6400 - 400 = 6000
        # sqrt(d) approx 77.46
        # t = (-(-80) - 77.46) / 2 = (80 - 77.46) / 2 = 2.54 / 2 = 1.27 (>=0)
        # For point 2 (zr=145):
        # a = 1
        # b = 2 * (-1 * (145-100)) = 2 * (-45) = -90
        # c = 145^2 - 2*(145*100) + 100^2 - 50^2 = 21025 - 29000 + 10000 - 2500 = -475
        # d = b^2 - 4ac = (-90)^2 - 4*1*(-475) = 8100 + 1900 = 10000
        # sqrt(d) = 100
        # t = (-(-90) - 100) / 2 = (90 - 100) / 2 = -10 / 2 = -5. This will make mask True.
        # This setup will result in a mixed mask. Let's adjust for all False first.

        # Revised calculations based on N_calc = -N_surface_group
        # N_surface_group is [[-1.0, -1.0]], so N_calc for formula is [[1.0, 1.0]]
        # Common for all test cases here:
        optic_mock.surface_group.x = be.array([[0.0, 0.0]])
        optic_mock.surface_group.y = be.array([[0.0, 0.0]])
        optic_mock.surface_group.L = be.array([[0.0, 0.0]])
        optic_mock.surface_group.M = be.array([[0.0, 0.0]])
        optic_mock.surface_group.N = be.array([[-1.0, -1.0]]) # This N is on surface_group

        # Case 1: Mask all False. We need t_sol1 = (-b - sqrt(d)) / (2a) >= 0.
        # This requires b to be negative (given N_calc=1, zr-zc < 0).
        # Point 1: zr=40 (zr-zc=-60). b = 2*1*(-60)=-120. c=(-60)^2-50^2=1100. d=10000. t_sol1=(120-100)/2=10.
        # Point 2: zr=30 (zr-zc=-70). b = 2*1*(-70)=-140. c=(-70)^2-50^2=2400. d=10000. t_sol1=(140-100)/2=20.
        optic_mock.surface_group.z = be.array([[40.0, 30.0]])
        t_case1 = wf._opd_image_to_xp(xc, yc, zc, R, wavelength)
        expected_t_case1 = be.array([10.0, 20.0]) * 1.5
        assert_allclose(t_case1, expected_t_case1, atol=1e-5)

        # Case 2: Mask all True. We need t_sol1 < 0.
        # This requires b to be positive (given N_calc=1, zr-zc > 0).
        # Then t_sol2 = (-b + sqrt(d)) / (2a) is chosen.
        # Point 1: zr=160 (zr-zc=60). b=120. c=1100. d=10000. t_sol1=(-120-100)/2=-110 (<0, mask=True)
        #          t_sol2=(-120+100)/2 = -10.
        # Point 2: zr=170 (zr-zc=70). b=140. c=2400. d=10000. t_sol1=(-140-100)/2=-120 (<0, mask=True)
        #          t_sol2=(-140+100)/2 = -20.
        optic_mock.surface_group.z = be.array([[160.0, 170.0]])
        t_case2 = wf._opd_image_to_xp(xc, yc, zc, R, wavelength)
        expected_t_case2 = be.array([-10.0, -20.0]) * 1.5
        assert_allclose(t_case2, expected_t_case2, atol=1e-5)

        # Case 3: Mask is mixed.
        # Point 1 (Mask False): zr=40. t_sol1=10. Output: 10.0 * 1.5 = 15.0
        # Point 2 (Mask True): zr=160. t_sol1=-110 (masked). t_sol2=-10. Output: -10.0 * 1.5 = -15.0
        optic_mock.surface_group.z = be.array([[40.0, 160.0]])
        t_case3 = wf._opd_image_to_xp(xc, yc, zc, R, wavelength)
        expected_t_case3 = be.array([10.0, -10.0]) * 1.5
        assert_allclose(t_case3, expected_t_case3, atol=1e-5)


class TestOPDFan:
    def test_opd_fan_initialization(self, set_test_backend):
        optic = DoubleGauss()
        opd_fan = wavefront.OPDFan(optic)
        assert opd_fan.num_rays == 100
        assert opd_fan.fields == optic.fields.get_field_coords()
        assert opd_fan.wavelengths == optic.wavelengths.get_wavelengths()
        assert isinstance(opd_fan.distribution, distribution.CrossDistribution)
        arr = be.linspace(-1, 1, opd_fan.num_rays)
        assert be.all(opd_fan.pupil_coord == arr)

    @patch("matplotlib.pyplot.show")
    def test_opd_fan_view(self, moch_show, set_test_backend):
        optic = DoubleGauss()
        opd_fan = wavefront.OPDFan(optic)
        opd_fan.view()
        moch_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_opd_fan_view_large(self, moch_show, set_test_backend):
        optic = DoubleGauss()
        opd_fan = wavefront.OPDFan(optic)
        opd_fan.view(figsize=(20, 20))
        moch_show.assert_called_once()
        plt.close()


class TestOPD:
    def test_opd_initialization(self, set_test_backend):
        optic = EyepieceErfle()
        opd = wavefront.OPD(optic, (0, 1), 0.55)
        assert opd.num_rays == 15
        assert opd.fields == [(0, 1)]
        assert opd.wavelengths == [0.55]
        assert isinstance(opd.distribution, distribution.HexagonalDistribution)

    @patch("matplotlib.pyplot.show")
    def test_opd_view(self, moch_show, set_test_backend):
        optic = DoubleGauss()
        opd = wavefront.OPD(optic, (0, 1), 0.55)
        opd.view()
        moch_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_opd_view_large(self, moch_show, set_test_backend):
        optic = DoubleGauss()
        opd = wavefront.OPD(optic, (0, 1), 0.55)
        opd.view(figsize=(20, 20))
        moch_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_opd_view_3d(self, moch_show, set_test_backend):
        optic = DoubleGauss()
        opd = wavefront.OPD(optic, (0, 1), 0.55)
        opd.view(projection="3d")
        moch_show.assert_called_once()
        plt.close()

    def test_old_invalid_projection(self, set_test_backend):
        optic = EyepieceErfle()
        opd = wavefront.OPD(optic, (0, 1), 0.55)
        with pytest.raises(ValueError):
            opd.view(projection="invalid")

    def test_opd_rms(self, set_test_backend):
        optic = CookeTriplet()
        opd = wavefront.OPD(optic, (0, 1), 0.55)
        rms = opd.rms()
        assert_allclose(rms, 0.9709788038168692)


class TestZernikeOPD:
    def test_zernike_opd_initialization(self, set_test_backend):
        optic = DoubleGauss()
        zernike_opd = wavefront.ZernikeOPD(optic, (0, 1), 0.55)
        assert zernike_opd.num_rays == 15
        assert zernike_opd.fields == [(0, 1)]
        assert zernike_opd.wavelengths == [0.55]
        assert isinstance(zernike_opd.distribution, distribution.HexagonalDistribution)
        assert be.allclose(zernike_opd.x, zernike_opd.distribution.x)
        assert be.allclose(zernike_opd.y, zernike_opd.distribution.y)
        assert be.allclose(zernike_opd.z, zernike_opd.data[((0, 1), 0.55)].opd)
        assert zernike_opd.zernike_type == "fringe"
        assert zernike_opd.num_terms == 37

    @patch("matplotlib.pyplot.show")
    def test_zernike_opd_view(self, moch_show, set_test_backend):
        optic = DoubleGauss()
        zernike_opd = wavefront.ZernikeOPD(optic, (0, 1), 0.55)
        zernike_opd.view()
        moch_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_zernike_opd_view_large(self, moch_show, set_test_backend):
        optic = DoubleGauss()
        zernike_opd = wavefront.ZernikeOPD(optic, (0, 1), 0.55)
        zernike_opd.view(figsize=(20, 20))
        moch_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_zernike_opd_view_3d(self, moch_show, set_test_backend):
        optic = DoubleGauss()
        zernike_opd = wavefront.ZernikeOPD(optic, (0, 1), 0.55)
        zernike_opd.view(projection="3d")
        moch_show.assert_called_once()
        plt.close()

    def test_zernike_opd_rms(self, set_test_backend):
        optic = CookeTriplet()
        zernike_opd = wavefront.ZernikeOPD(optic, (0, 1), 0.55)
        rms = zernike_opd.rms()
        assert_allclose(rms, 0.9709788038168692)

    def test_zernike_opd_fit(self, set_test_backend):
        optic = CookeTriplet()
        zernike_opd = wavefront.ZernikeOPD(optic, (0, 1), 0.55)
        c = zernike_opd.zernike.coeffs
        assert_allclose(c[0], 0.8430890395012354)
        assert_allclose(c[1], 6.863699034904449e-13)
        assert_allclose(c[2], 0.14504379704525455)
        assert_allclose(c[6], -1.160298338689596e-13)
        assert_allclose(c[24], -0.0007283668376039182)

    def test_zernike_xy_symmetry(self, set_test_backend):
        optic = CookeTriplet()
        zernike_opd0 = wavefront.ZernikeOPD(optic, (0, 1), 0.55)
        c0 = zernike_opd0.zernike.coeffs

        # swap x and y fields
        optic.fields.fields[0].x = 0
        optic.fields.fields[0].y = 0
        optic.fields.fields[1].x = 14
        optic.fields.fields[1].y = 0
        optic.fields.fields[2].x = 20
        optic.fields.fields[2].y = 0

        # run at max y field (should be the same field)
        zernike_opd1 = wavefront.ZernikeOPD(optic, (0, 1), 0.55)
        c1 = zernike_opd1.zernike.coeffs
        assert be.allclose(c0, c1)

    def test_zernike_xy_axis_swap(self, set_test_backend):
        optic = CookeTriplet()
        zernike_opd0 = wavefront.ZernikeOPD(optic, (0, 1), 0.55)
        c0 = zernike_opd0.zernike.coeffs

        # swap x and y fields
        optic.fields.fields[0].x = 0
        optic.fields.fields[0].y = 0
        optic.fields.fields[1].x = 0
        optic.fields.fields[1].y = 14
        optic.fields.fields[2].x = 0
        optic.fields.fields[2].y = 20

        # run at max x field
        zernike_opd1 = wavefront.ZernikeOPD(optic, (1, 0), 0.55)
        c1 = zernike_opd1.zernike.coeffs

        # x and y tilts swapped
        assert be.isclose(c0[1], c1[2])
        assert be.isclose(c0[2], c1[1])
