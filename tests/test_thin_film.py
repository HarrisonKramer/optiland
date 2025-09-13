import optiland.backend as be
import pytest
import numpy as np
import matplotlib.pyplot as plt

from optiland.thin_film import Layer, ThinFilmStack, SpectralAnalyzer
from optiland.materials import Material, IdealMaterial
from .utils import assert_allclose

# Import physical constants for testing unit conversions
from optiland.thin_film.analysis import SPEED_OF_LIGHT, PLANCK_EV


@pytest.fixture
def air():
    return IdealMaterial(n=1.0)


@pytest.fixture
def glass():
    return IdealMaterial(n=1.52)


@pytest.fixture
def sio2():
    return Material("SiO2", reference="Gao")


@pytest.fixture
def tio2():
    return Material("TiO2", reference="Zhukovsky")


@pytest.fixture
def bk7():
    return Material("N-BK7", reference="SCHOTT")


@pytest.fixture
def simple_stack(air, glass):
    """Simple stack with no layers (air-glass interface)."""
    return ThinFilmStack(incident_material=air, substrate_material=glass)


@pytest.fixture
def single_layer_stack(air, glass, sio2):
    """Stack with one SiO2 layer."""
    stack = ThinFilmStack(incident_material=air, substrate_material=glass)
    stack.add_layer_nm(sio2, 100.0, name="SiO2")
    return stack


@pytest.fixture
def multilayer_stack(air, bk7, sio2, tio2):
    """Multilayer stack similar to notebook example."""
    stack = ThinFilmStack(
        incident_material=air, substrate_material=bk7, reference_wl_um=0.6
    )
    for i in range(3):  # Smaller stack for faster tests
        stack.add_layer_qwot(material=tio2, qwot_thickness=1.0, name=f"TiO2_{i}")
        stack.add_layer_qwot(material=sio2, qwot_thickness=1.0, name=f"SiO2_{i}")
    return stack


class TestImports:
    """Test that all components can be imported."""

    def test_layer_import(self):
        assert Layer is not None

    def test_stack_import(self):
        assert ThinFilmStack is not None

    def test_analyzer_import(self):
        assert SpectralAnalyzer is not None


class TestLayer:
    """Test Layer class functionality."""

    def test_layer_creation(self, set_test_backend, sio2):
        layer = Layer(material=sio2, thickness_um=0.1, name="SiO2")
        assert layer.material == sio2
        assert layer.thickness_um == 0.1
        assert layer.name == "SiO2"

    def test_layer_creation_without_name(self, set_test_backend, sio2):
        layer = Layer(material=sio2, thickness_um=0.1)
        assert layer.material == sio2
        assert layer.thickness_um == 0.1
        assert layer.name is None


class TestThinFilmStackBasic:
    """Test basic ThinFilmStack functionality."""

    def test_stack_creation(self, set_test_backend, air, glass):
        stack = ThinFilmStack(incident_material=air, substrate_material=glass)
        assert stack.incident_material == air
        assert stack.substrate_material == glass
        assert len(stack.layers) == 0
        assert len(stack) == 0

    def test_stack_with_reference(self, set_test_backend, air, glass):
        stack = ThinFilmStack(
            incident_material=air,
            substrate_material=glass,
            reference_wl_um=0.55,
            reference_AOI_deg=30.0,
        )
        assert stack.reference_wl_um == 0.55
        assert stack.reference_AOI_deg == 30.0

    def test_stack_repr(self, set_test_backend, single_layer_stack):
        repr_str = repr(single_layer_stack)
        assert "ThinFilmStack" in repr_str
        assert "1 layers" in repr_str


class TestThinFilmStackLayerManipulation:
    """Test layer addition methods."""

    def test_add_layer_um(self, set_test_backend, air, glass, sio2):
        stack = ThinFilmStack(incident_material=air, substrate_material=glass)
        stack.add_layer(sio2, 0.1, "SiO2")

        assert len(stack) == 1
        assert stack.layers[0].material == sio2
        assert stack.layers[0].thickness_um == 0.1
        assert stack.layers[0].name == "SiO2"

    def test_add_layer_nm(self, set_test_backend, air, glass, sio2):
        stack = ThinFilmStack(incident_material=air, substrate_material=glass)
        stack.add_layer_nm(sio2, 100.0, "SiO2")

        assert len(stack) == 1
        assert stack.layers[0].thickness_um == 0.1  # 100 nm = 0.1 µm

    def test_add_layer_qwot(self, set_test_backend, air, glass, sio2):
        stack = ThinFilmStack(
            incident_material=air, substrate_material=glass, reference_wl_um=0.55
        )
        stack.add_layer_qwot(sio2, 1.0, "SiO2_QWOT")

        assert len(stack) == 1
        # QWOT thickness should be λ/(4n) where n is refractive index of SiO2
        n_sio2 = float(sio2.n(0.55))
        expected_thickness = 0.55 / (4 * n_sio2)
        assert_allclose(stack.layers[0].thickness_um, expected_thickness, rtol=1e-10)

    def test_add_layer_qwot_no_reference_fails(
        self, set_test_backend, air, glass, sio2
    ):
        stack = ThinFilmStack(incident_material=air, substrate_material=glass)
        with pytest.raises(ValueError, match="reference_wl_um must be set"):
            stack.add_layer_qwot(sio2, 1.0, "SiO2_QWOT")

    def test_chaining_add_layer(self, set_test_backend, air, glass, sio2, tio2):
        stack = ThinFilmStack(incident_material=air, substrate_material=glass)
        result = stack.add_layer(sio2, 0.1).add_layer(tio2, 0.05)

        assert result is stack  # Should return self for chaining
        assert len(stack) == 2


class TestThinFilmStackCalculations:
    """Test optical calculations."""

    def test_simple_interface_normal_incidence(self, set_test_backend, simple_stack):
        """Test air-glass interface at normal incidence."""
        wavelength_um = 0.55
        aoi_rad = 0.0

        # Test all polarizations
        for pol in ["s", "p", "u"]:
            result = simple_stack.compute_rtRTA(wavelength_um, aoi_rad, pol)

            # Check that we have all expected keys
            assert "r" in result
            assert "t" in result
            assert "R" in result
            assert "T" in result
            assert "A" in result

            # For lossless interface, A should be 0
            assert_allclose(result["A"], 0.0, atol=1e-10)

            # Energy conservation: R + T + A = 1
            rta_sum = result["R"] + result["T"] + result["A"]
            assert_allclose(rta_sum, 1.0, rtol=1e-10)

    def test_nm_deg_interface(self, set_test_backend, simple_stack):
        """Test that nm/deg interface gives same results as um/rad."""
        wavelength_nm = 550.0
        wavelength_um = 0.55
        aoi_deg = 30.0
        aoi_rad = be.deg2rad(30.0)

        result_um_rad = simple_stack.compute_rtRTA(wavelength_um, aoi_rad, "s")
        result_nm_deg = simple_stack.compute_rtRAT_nm_deg(wavelength_nm, aoi_deg, "s")

        for key in ["R", "T", "A"]:
            assert_allclose(result_um_rad[key], result_nm_deg[key], rtol=1e-10)

    def test_convenience_methods(self, set_test_backend, simple_stack):
        """Test convenience getter methods."""
        wavelength_um = 0.55
        aoi_rad = 0.0

        # Test um/rad methods
        R = simple_stack.reflectance(wavelength_um, aoi_rad, "s")
        T = simple_stack.transmittance(wavelength_um, aoi_rad, "s")
        A = simple_stack.absorptance(wavelength_um, aoi_rad, "s")

        # Test nm/deg methods
        R_nm = simple_stack.reflectance_nm_deg(550.0, 0.0, "s")
        T_nm = simple_stack.transmittance_nm_deg(550.0, 0.0, "s")
        A_nm = simple_stack.absorptance_nm_deg(550.0, 0.0, "s")

        assert_allclose(R, R_nm, rtol=1e-10)
        assert_allclose(T, T_nm, rtol=1e-10)
        assert_allclose(A, A_nm, rtol=1e-10)

        # Test RTA tuple methods
        R2, T2, A2 = simple_stack.RTA(wavelength_um, aoi_rad, "s")
        R3, T3, A3 = simple_stack.RTA_nm_deg(550.0, 0.0, "s")

        assert_allclose(R, R2, rtol=1e-10)
        assert_allclose(T, T2, rtol=1e-10)
        assert_allclose(A, A2, rtol=1e-10)
        assert_allclose(R2, R3, rtol=1e-10)
        assert_allclose(T2, T3, rtol=1e-10)
        assert_allclose(A2, A3, rtol=1e-10)

    def test_array_inputs_wavelength(self, set_test_backend, simple_stack):
        """Test calculations with wavelength arrays."""
        wavelengths_um = be.linspace(0.4, 0.8, 5)
        aoi_rad = 0.0

        result = simple_stack.compute_rtRTA(wavelengths_um, aoi_rad, "s")

        # Check output shapes
        expected_shape = (5, 1)  # (Nλ, Nθ)
        assert result["R"].shape == expected_shape
        assert result["T"].shape == expected_shape
        assert result["A"].shape == expected_shape

    def test_array_inputs_angle(self, set_test_backend, simple_stack):
        """Test calculations with angle arrays."""
        wavelength_um = 0.55
        aoi_rads = be.linspace(0.0, be.deg2rad(60), 5)

        result = simple_stack.compute_rtRTA(wavelength_um, aoi_rads, "s")

        # Check output shapes
        expected_shape = (1, 5)  # (Nλ, Nθ)
        assert result["R"].shape == expected_shape

    def test_array_inputs_both(self, set_test_backend, simple_stack):
        """Test calculations with both wavelength and angle arrays."""
        wavelengths_um = be.linspace(0.4, 0.8, 3)
        aoi_rads = be.linspace(0.0, be.deg2rad(60), 4)

        result = simple_stack.compute_rtRTA(wavelengths_um, aoi_rads, "s")

        # Check output shapes
        expected_shape = (3, 4)  # (Nλ, Nθ)
        assert result["R"].shape == expected_shape

    def test_polarization_modes(self, set_test_backend, simple_stack):
        """Test all polarization modes."""
        wavelength_um = 0.55
        aoi_rad = be.deg2rad(45)  # Non-normal incidence to see polarization effects

        result_s = simple_stack.compute_rtRTA(wavelength_um, aoi_rad, "s")
        result_p = simple_stack.compute_rtRTA(wavelength_um, aoi_rad, "p")
        result_u = simple_stack.compute_rtRTA(wavelength_um, aoi_rad, "u")

        # At 45° incidence, s and p should give different results
        assert not be.allclose(result_s["R"], result_p["R"])

        # Unpolarized should be average of s and p powers
        expected_R_u = 0.5 * (result_s["R"] + result_p["R"])
        expected_T_u = 0.5 * (result_s["T"] + result_p["T"])
        expected_A_u = 0.5 * (result_s["A"] + result_p["A"])

        assert_allclose(result_u["R"], expected_R_u, rtol=1e-10)
        assert_allclose(result_u["T"], expected_T_u, rtol=1e-10)
        assert_allclose(result_u["A"], expected_A_u, rtol=1e-10)

    def test_invalid_polarization(self, set_test_backend, simple_stack):
        """Test that invalid polarization raises error."""
        with pytest.raises(ValueError, match="polarization must be"):
            simple_stack.compute_rtRTA(0.55, 0.0, "invalid")


class TestThinFilmStackUnitHelpers:
    """Test unit conversion helper methods."""

    def test_to_um_from_nm(self, set_test_backend):
        wavelength_nm = 550.0
        result = ThinFilmStack._to_um(wavelength_nm, assume_nm=True)
        assert_allclose(result, 0.55)

    def test_to_um_already_um(self, set_test_backend):
        wavelength_um = 0.55
        result = ThinFilmStack._to_um(wavelength_um, assume_nm=False)
        assert_allclose(result, 0.55)

    def test_deg_to_rad(self, set_test_backend):
        angle_deg = 30.0
        result = ThinFilmStack._deg_to_rad(angle_deg)
        expected = be.deg2rad(30.0)
        assert_allclose(result, expected)


class TestThinFilmStackVisualization:
    """Test plotting methods."""

    def test_plot_structure_simple(self, set_test_backend, simple_stack):
        """Test structure plotting for simple stack."""
        fig, ax = simple_stack.plot_structure()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_structure_multilayer(self, set_test_backend, multilayer_stack):
        """Test structure plotting for multilayer stack."""
        fig, ax = multilayer_stack.plot_structure()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_structure_with_ax(self, set_test_backend, simple_stack):
        """Test structure plotting with provided axes."""
        fig, ax = plt.subplots()
        returned_fig, returned_ax = simple_stack.plot_structure(ax=ax)
        assert returned_fig is fig
        assert returned_ax is ax
        plt.close(fig)

    def test_plot_structure_thickness(self, set_test_backend, multilayer_stack):
        """Test thickness plotting."""
        fig, ax = multilayer_stack.plot_structure_thickness()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_structure_thickness_with_ax(self, set_test_backend, multilayer_stack):
        """Test thickness plotting with provided axes."""
        fig, ax = plt.subplots()
        returned_fig, returned_ax = multilayer_stack.plot_structure_thickness(ax=ax)
        assert returned_fig is fig
        assert returned_ax is ax
        plt.close(fig)


class TestSpectralAnalyzerUnitConversions:
    """Test unit conversion methods in SpectralAnalyzer."""

    def test_wavelength_unit_conversions(self, set_test_backend, simple_stack):
        """Test wavelength unit conversions with specific known values."""
        analyzer = SpectralAnalyzer(simple_stack)

        # Test wavelength: 550 nm = 0.55 μm
        wl_nm = 550.0
        wl_um = 0.55

        # Test conversions
        assert_allclose(
            analyzer._convert_to_wavelength_um(wl_nm, "nm"), wl_um, rtol=1e-10
        )
        assert_allclose(
            analyzer._convert_to_wavelength_um(wl_um, "um"), wl_um, rtol=1e-10
        )

        # Test frequency conversion: λ = c/ν
        frequency_hz = SPEED_OF_LIGHT / (wl_um * 1e-6)  # Convert μm to m
        converted_wl = analyzer._convert_to_wavelength_um(frequency_hz, "frequency")
        assert_allclose(converted_wl, wl_um, rtol=1e-10)

        # Test energy conversion: E = hc/λ
        energy_ev = (PLANCK_EV * SPEED_OF_LIGHT) / (wl_um * 1e-6)  # Convert μm to m
        converted_wl = analyzer._convert_to_wavelength_um(energy_ev, "energy")
        assert_allclose(converted_wl, wl_um, rtol=1e-10)

        # Test wavenumber conversion: k = 1/λ (cm⁻¹)
        wavenumber_cm_inv = 1e4 / wl_um  # Convert μm to cm, then invert
        converted_wl = analyzer._convert_to_wavelength_um(
            wavenumber_cm_inv, "wavenumber"
        )
        assert_allclose(converted_wl, wl_um, rtol=1e-10)

    def test_relative_wavenumber_conversion(self, set_test_backend, air, glass):
        """Test relative wavenumber conversion."""
        stack = ThinFilmStack(
            incident_material=air, substrate_material=glass, reference_wl_um=0.55
        )
        analyzer = SpectralAnalyzer(stack)

        # Relative wavenumber: k_rel = λ_ref / λ
        reference_wl = 0.55
        target_wl = 0.6
        k_rel = reference_wl / target_wl

        converted_wl = analyzer._convert_to_wavelength_um(k_rel, "relative_wavenumber")
        assert_allclose(converted_wl, target_wl, rtol=1e-10)

    def test_relative_wavenumber_no_reference_fails(
        self, set_test_backend, simple_stack
    ):
        """Test that relative wavenumber fails without reference wavelength."""
        analyzer = SpectralAnalyzer(simple_stack)

        with pytest.raises(ValueError, match="reference_wl_um must be set"):
            analyzer._convert_to_wavelength_um(1.0, "relative_wavenumber")

    def test_invalid_wavelength_unit_fails(self, set_test_backend, simple_stack):
        """Test that invalid wavelength unit raises error."""
        analyzer = SpectralAnalyzer(simple_stack)

        with pytest.raises(ValueError, match="Unknown wavelength unit"):
            analyzer._convert_to_wavelength_um(550.0, "invalid_unit")

    def test_angle_unit_conversions(self, set_test_backend, simple_stack):
        """Test angle unit conversions."""
        analyzer = SpectralAnalyzer(simple_stack)

        # Test 30 degrees to radians
        angle_deg = 30.0
        angle_rad = be.deg2rad(30.0)

        converted_rad = analyzer._convert_angle_to_radians(angle_deg, "deg")
        assert_allclose(converted_rad, angle_rad, rtol=1e-10)

        # Test radians to radians (no conversion)
        converted_rad2 = analyzer._convert_angle_to_radians(angle_rad, "rad")
        assert_allclose(converted_rad2, angle_rad, rtol=1e-10)

    def test_invalid_angle_unit_fails(self, set_test_backend, simple_stack):
        """Test that invalid angle unit raises error."""
        analyzer = SpectralAnalyzer(simple_stack)

        with pytest.raises(ValueError, match="Unknown angle unit"):
            analyzer._convert_angle_to_radians(30.0, "invalid_unit")

    def test_wavelength_axis_labels(self, set_test_backend, simple_stack):
        """Test wavelength axis label generation."""
        analyzer = SpectralAnalyzer(simple_stack)

        expected_labels = {
            "um": r"$\lambda$ ($\mu$m)",
            "nm": r"$\lambda$ (nm)",
            "frequency": r"$\nu$ (Hz)",
            "energy": r"$E$ (eV)",
            "wavenumber": r"$k$ (cm$^{-1}$)",
            "relative_wavenumber": r"$k/k_{\mathrm{ref}}$",
        }

        for unit, expected_label in expected_labels.items():
            assert analyzer._get_wavelength_axis_label(unit) == expected_label

    def test_wavelength_plotting_conversions(self, set_test_backend, simple_stack):
        """Test wavelength conversion for plotting axes."""
        analyzer = SpectralAnalyzer(simple_stack)

        wl_um = be.array([0.5, 0.55, 0.6])

        # Test nm conversion
        wl_nm = analyzer._convert_wavelength_for_plotting(wl_um, "nm")
        expected_nm = wl_um * 1000.0
        assert_allclose(wl_nm, expected_nm, rtol=1e-10)

        # Test frequency conversion
        wl_freq = analyzer._convert_wavelength_for_plotting(wl_um, "frequency")
        expected_freq = SPEED_OF_LIGHT / (wl_um * 1e-6)
        assert_allclose(wl_freq, expected_freq, rtol=1e-10)

        # Test energy conversion
        wl_energy = analyzer._convert_wavelength_for_plotting(wl_um, "energy")
        expected_energy = (PLANCK_EV * SPEED_OF_LIGHT) / (wl_um * 1e-6)
        assert_allclose(wl_energy, expected_energy, rtol=1e-10)

        # Test wavenumber conversion
        wl_wavenumber = analyzer._convert_wavelength_for_plotting(wl_um, "wavenumber")
        expected_wavenumber = 1e4 / wl_um
        assert_allclose(wl_wavenumber, expected_wavenumber, rtol=1e-10)


class TestSpectralAnalyzer:
    """Test SpectralAnalyzer functionality."""

    def test_analyzer_creation(self, set_test_backend, simple_stack):
        analyzer = SpectralAnalyzer(simple_stack)
        assert analyzer.stack is simple_stack

    def test_plot_vs_wavelength(self, set_test_backend, simple_stack):
        """Test plotting R vs wavelength."""
        analyzer = SpectralAnalyzer(simple_stack)
        wavelengths_um = be.linspace(0.4, 0.8, 10)

        fig, ax = analyzer.wavelength_view(wavelengths_um, aoi=0.0, to_plot="R")
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_vs_angle(self, set_test_backend, simple_stack):
        """Test plotting R vs angle."""
        analyzer = SpectralAnalyzer(simple_stack)
        angles_deg = be.linspace(0, 80, 10)

        fig, ax = analyzer.angular_view(angles_deg, wavelength=0.55, to_plot="R")
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_2d_map(self, set_test_backend, simple_stack):
        """Test 2D plotting (wavelength vs angle)."""
        analyzer = SpectralAnalyzer(simple_stack)
        wavelengths_um = be.linspace(0.4, 0.8, 5)
        angles_deg = be.linspace(0, 60, 4)

        fig, axs = analyzer.map_view(wavelengths_um, aoi_values=angles_deg, to_plot="R")
        assert isinstance(fig, plt.Figure)
        assert isinstance(axs, plt.Axes)
        plt.close(fig)

    def test_plot_multiple_quantities(self, set_test_backend, simple_stack):
        """Test plotting multiple quantities."""
        analyzer = SpectralAnalyzer(simple_stack)
        wavelengths_um = be.linspace(0.4, 0.8, 10)

        fig, ax = analyzer.wavelength_view(wavelengths_um, to_plot=["R", "T", "A"])
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_with_provided_ax(self, set_test_backend, simple_stack):
        """Test plotting with provided axes."""
        analyzer = SpectralAnalyzer(simple_stack)
        wavelengths_um = be.linspace(0.4, 0.8, 10)

        fig, ax = plt.subplots()
        returned_fig, returned_ax = analyzer.wavelength_view(wavelengths_um, ax=ax)
        assert returned_fig is fig
        assert returned_ax is ax
        plt.close(fig)

    def test_plot_invalid_quantity(self, set_test_backend, simple_stack):
        """Test that invalid quantity raises error."""
        analyzer = SpectralAnalyzer(simple_stack)
        wavelengths_um = be.linspace(0.4, 0.8, 10)

        with pytest.raises(ValueError, match="to_plot must be"):
            analyzer.wavelength_view(wavelengths_um, to_plot="invalid")

    def test_wavelength_view_different_units(self, set_test_backend, simple_stack):
        """Test wavelength_view with different wavelength units."""
        analyzer = SpectralAnalyzer(simple_stack)

        # Define wavelength range in different units
        wl_um = be.linspace(0.4, 0.8, 5)
        wl_nm = wl_um * 1000.0
        wl_freq = SPEED_OF_LIGHT / (wl_um * 1e-6)
        wl_energy = (PLANCK_EV * SPEED_OF_LIGHT) / (wl_um * 1e-6)
        wl_wavenumber = 1e4 / wl_um

        # Test each unit and verify same results
        result_um = analyzer.stack.compute_rtRTA(wl_um, 0.0, "s")

        # Test nm
        fig, ax = analyzer.wavelength_view(wl_nm, wavelength_unit="nm", to_plot="R")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        # Test frequency
        fig, ax = analyzer.wavelength_view(
            wl_freq, wavelength_unit="frequency", to_plot="R"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        # Test energy
        fig, ax = analyzer.wavelength_view(
            wl_energy, wavelength_unit="energy", to_plot="R"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        # Test wavenumber
        fig, ax = analyzer.wavelength_view(
            wl_wavenumber, wavelength_unit="wavenumber", to_plot="R"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_wavelength_view_relative_wavenumber(self, set_test_backend, air, glass):
        """Test wavelength_view with relative wavenumber unit."""
        stack = ThinFilmStack(
            incident_material=air, substrate_material=glass, reference_wl_um=0.55
        )
        analyzer = SpectralAnalyzer(stack)

        # Define relative wavenumber range (around 1.0 for reference wavelength)
        k_rel = be.linspace(0.8, 1.2, 5)

        fig, ax = analyzer.wavelength_view(
            k_rel, wavelength_unit="relative_wavenumber", to_plot="R"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_wavelength_view_different_angles(self, set_test_backend, simple_stack):
        """Test wavelength_view with different angle units."""
        analyzer = SpectralAnalyzer(simple_stack)
        wavelengths_um = be.linspace(0.4, 0.8, 5)

        # Test degrees
        fig, ax = analyzer.wavelength_view(
            wavelengths_um, aoi=30.0, aoi_unit="deg", to_plot="R"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        # Test radians
        fig, ax = analyzer.wavelength_view(
            wavelengths_um, aoi=be.deg2rad(30.0), aoi_unit="rad", to_plot="R"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_wavelength_view_multiple_polarizations(
        self, set_test_backend, simple_stack
    ):
        """Test wavelength_view with different polarizations."""
        analyzer = SpectralAnalyzer(simple_stack)
        wavelengths_um = be.linspace(0.4, 0.8, 5)

        for pol in ["s", "p", "u"]:
            fig, ax = analyzer.wavelength_view(
                wavelengths_um, polarization=pol, to_plot="R"
            )
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_angular_view_different_units(self, set_test_backend, simple_stack):
        """Test angular_view with different wavelength and angle units."""
        analyzer = SpectralAnalyzer(simple_stack)
        angles_deg = be.linspace(0, 60, 5)
        angles_rad = be.deg2rad(angles_deg)

        # Test degrees
        fig, ax = analyzer.angular_view(
            angles_deg,
            aoi_unit="deg",
            wavelength=550,
            wavelength_unit="nm",
            to_plot="R",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        # Test radians
        fig, ax = analyzer.angular_view(
            angles_rad,
            aoi_unit="rad",
            wavelength=0.55,
            wavelength_unit="um",
            to_plot="R",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_angular_view_wavelength_units(self, set_test_backend, simple_stack):
        """Test angular_view with different wavelength units."""
        analyzer = SpectralAnalyzer(simple_stack)
        angles_deg = be.linspace(0, 60, 5)

        # Define wavelength in different units (all equivalent to 550 nm)
        test_cases = [
            (550.0, "nm"),
            (0.55, "um"),
            (SPEED_OF_LIGHT / (0.55e-6), "frequency"),
            ((PLANCK_EV * SPEED_OF_LIGHT) / (0.55e-6), "energy"),
            (1e4 / 0.55, "wavenumber"),
        ]

        for wavelength, unit in test_cases:
            fig, ax = analyzer.angular_view(
                angles_deg, wavelength=wavelength, wavelength_unit=unit, to_plot="R"
            )
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_angular_view_multiple_quantities(self, set_test_backend, simple_stack):
        """Test angular_view with multiple quantities and polarizations."""
        analyzer = SpectralAnalyzer(simple_stack)
        angles_deg = be.linspace(0, 60, 5)

        # Test multiple quantities
        fig, ax = analyzer.angular_view(angles_deg, to_plot=["R", "T", "A"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        # Test different polarizations
        for pol in ["s", "p", "u"]:
            fig, ax = analyzer.angular_view(angles_deg, polarization=pol, to_plot="R")
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_map_view_different_units(self, set_test_backend, simple_stack):
        """Test map_view with different wavelength and angle units."""
        analyzer = SpectralAnalyzer(simple_stack)

        wavelengths_nm = be.linspace(400, 800, 3)
        wavelengths_um = wavelengths_nm / 1000.0
        angles_deg = be.linspace(0, 60, 3)
        angles_rad = be.deg2rad(angles_deg)

        # Test nm and degrees
        fig, ax = analyzer.map_view(
            wavelengths_nm,
            wavelength_unit="nm",
            aoi_values=angles_deg,
            aoi_unit="deg",
            to_plot="R",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        # Test um and radians
        fig, ax = analyzer.map_view(
            wavelengths_um,
            wavelength_unit="um",
            aoi_values=angles_rad,
            aoi_unit="rad",
            to_plot="R",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_map_view_wavelength_units(self, set_test_backend, simple_stack):
        """Test map_view with different wavelength units."""
        analyzer = SpectralAnalyzer(simple_stack)
        angles_deg = be.linspace(0, 60, 3)

        # Test with frequency units
        wl_um = be.linspace(0.4, 0.8, 3)
        wl_freq = SPEED_OF_LIGHT / (wl_um * 1e-6)
        fig, ax = analyzer.map_view(
            wl_freq, wavelength_unit="frequency", aoi_values=angles_deg, to_plot="R"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        # Test with energy units
        wl_energy = (PLANCK_EV * SPEED_OF_LIGHT) / (wl_um * 1e-6)
        fig, ax = analyzer.map_view(
            wl_energy, wavelength_unit="energy", aoi_values=angles_deg, to_plot="R"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        # Test with wavenumber units
        wl_wavenumber = 1e4 / wl_um
        fig, ax = analyzer.map_view(
            wl_wavenumber,
            wavelength_unit="wavenumber",
            aoi_values=angles_deg,
            to_plot="R",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_map_view_multiple_quantities(self, set_test_backend, simple_stack):
        """Test map_view with multiple quantities."""
        analyzer = SpectralAnalyzer(simple_stack)
        wavelengths_um = be.linspace(0.4, 0.8, 3)
        angles_deg = be.linspace(0, 60, 3)

        # Test multiple quantities - should return array of axes
        fig, axs = analyzer.map_view(
            wavelengths_um, aoi_values=angles_deg, to_plot=["R", "T", "A"]
        )
        assert isinstance(fig, plt.Figure)
        assert hasattr(axs, "__len__")  # axs should be array-like
        assert len(axs) == 3
        plt.close(fig)

    def test_map_view_default_angles(self, set_test_backend, simple_stack):
        """Test map_view with default angle range."""
        analyzer = SpectralAnalyzer(simple_stack)
        wavelengths_um = be.linspace(0.4, 0.8, 3)

        # Test without providing aoi_values (should use default)
        fig, ax = analyzer.map_view(wavelengths_um, to_plot="R")
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_spectral_analyzer_error_cases(self, set_test_backend, simple_stack):
        """Test error cases for SpectralAnalyzer methods."""
        analyzer = SpectralAnalyzer(simple_stack)
        wavelengths_um = be.linspace(0.4, 0.8, 5)
        angles_deg = be.linspace(0, 60, 5)

        # Test invalid wavelength unit in wavelength_view
        with pytest.raises(ValueError, match="Unknown wavelength unit"):
            analyzer.wavelength_view(wavelengths_um, wavelength_unit="invalid_unit")

        # Test invalid angle unit in wavelength_view
        with pytest.raises(ValueError, match="Unknown angle unit"):
            analyzer.wavelength_view(wavelengths_um, aoi_unit="invalid_unit")

        # Test invalid wavelength unit in angular_view
        with pytest.raises(ValueError, match="Unknown wavelength unit"):
            analyzer.angular_view(angles_deg, wavelength_unit="invalid_unit")

        # Test invalid angle unit in angular_view
        with pytest.raises(ValueError, match="Unknown angle unit"):
            analyzer.angular_view(angles_deg, aoi_unit="invalid_unit")

        # Test invalid wavelength unit in map_view
        with pytest.raises(ValueError, match="Unknown wavelength unit"):
            analyzer.map_view(wavelengths_um, wavelength_unit="invalid_unit")

        # Test invalid angle unit in map_view
        with pytest.raises(ValueError, match="Unknown angle unit"):
            analyzer.map_view(
                wavelengths_um, aoi_values=angles_deg, aoi_unit="invalid_unit"
            )

        # Test invalid quantity in all methods
        with pytest.raises(ValueError, match="to_plot must be"):
            analyzer.wavelength_view(wavelengths_um, to_plot="invalid")

        with pytest.raises(ValueError, match="to_plot must be"):
            analyzer.angular_view(angles_deg, to_plot="invalid")

        with pytest.raises(ValueError, match="to_plot must be"):
            analyzer.map_view(wavelengths_um, to_plot="invalid")

    def test_relative_wavenumber_errors(self, set_test_backend, simple_stack):
        """Test relative wavenumber errors in plotting methods."""
        analyzer = SpectralAnalyzer(simple_stack)
        k_rel = be.linspace(0.8, 1.2, 5)

        # Test wavelength_view with relative_wavenumber but no reference
        with pytest.raises(ValueError, match="reference_wl_um must be set"):
            analyzer.wavelength_view(k_rel, wavelength_unit="relative_wavenumber")

        # Test angular_view with relative_wavenumber but no reference
        with pytest.raises(ValueError, match="reference_wl_um must be set"):
            analyzer.angular_view(
                be.linspace(0, 60, 5),
                wavelength=1.0,
                wavelength_unit="relative_wavenumber",
            )

        # Test map_view with relative_wavenumber but no reference
        with pytest.raises(ValueError, match="reference_wl_um must be set"):
            analyzer.map_view(k_rel, wavelength_unit="relative_wavenumber")

    def test_consistency_across_units(self, set_test_backend, simple_stack):
        """Test that different units give consistent results."""
        analyzer = SpectralAnalyzer(simple_stack)

        # Define equivalent wavelengths in different units
        wl_um = 0.55
        wl_nm = wl_um * 1000.0
        wl_freq = SPEED_OF_LIGHT / (wl_um * 1e-6)
        wl_energy = (PLANCK_EV * SPEED_OF_LIGHT) / (wl_um * 1e-6)
        wl_wavenumber = 1e4 / wl_um

        # Get results for each unit using angular_view
        angles_deg = be.linspace(0, 60, 5)

        result_um = analyzer.stack.compute_rtRTA(wl_um, be.deg2rad(angles_deg), "s")

        # Test consistency - all should convert to the same wavelength internally
        # We can't easily test the plotting results directly, but we can test the conversion methods
        assert_allclose(
            analyzer._convert_to_wavelength_um(wl_nm, "nm"), wl_um, rtol=1e-10
        )
        assert_allclose(
            analyzer._convert_to_wavelength_um(wl_freq, "frequency"), wl_um, rtol=1e-10
        )
        assert_allclose(
            analyzer._convert_to_wavelength_um(wl_energy, "energy"), wl_um, rtol=1e-10
        )
        assert_allclose(
            analyzer._convert_to_wavelength_um(wl_wavenumber, "wavenumber"),
            wl_um,
            rtol=1e-10,
        )


class TestThinFilmStackComplex:
    """Test more complex scenarios."""

    def test_multilayer_energy_conservation(self, set_test_backend, multilayer_stack):
        """Test energy conservation for multilayer stack."""
        wavelengths_um = be.linspace(0.4, 0.8, 5)
        aoi_deg = 0.0

        for pol in ["s", "p", "u"]:
            result = multilayer_stack.compute_rtRAT_nm_deg(
                wavelengths_um * 1000, aoi_deg, pol
            )

            # Energy conservation: R + T + A = 1 (within numerical precision)
            rta_sum = result["R"] + result["T"] + result["A"]
            assert_allclose(rta_sum, 1.0, rtol=1e-8)

    def test_physical_constraints(self, set_test_backend, multilayer_stack):
        """Test that R, T, A are between 0 and 1."""
        wavelengths_um = be.linspace(0.4, 0.8, 5)
        angles_deg = be.linspace(0, 80, 5)

        result = multilayer_stack.compute_rtRAT_nm_deg(
            wavelengths_um * 1000, angles_deg, "u"
        )

        for quantity in ["R", "T", "A"]:
            values = result[quantity]
            assert be.all(values >= 0), f"{quantity} has negative values"
            assert be.all(values <= 1), f"{quantity} has values > 1"
