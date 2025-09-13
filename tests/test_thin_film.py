import optiland.backend as be
import pytest
import numpy as np
import matplotlib.pyplot as plt

from optiland.thin_film import Layer, ThinFilmStack, SpectralAnalyzer
from optiland.materials import Material, IdealMaterial
from .utils import assert_allclose


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


class TestSpectralAnalyzer:
    """Test SpectralAnalyzer functionality."""

    def test_analyzer_creation(self, set_test_backend, simple_stack):
        analyzer = SpectralAnalyzer(simple_stack)
        assert analyzer.stack is simple_stack

    def test_plot_vs_wavelength(self, set_test_backend, simple_stack):
        """Test plotting R vs wavelength."""
        analyzer = SpectralAnalyzer(simple_stack)
        wavelengths_um = be.linspace(0.4, 0.8, 10)

        fig, ax = analyzer.view(wavelengths_um, aoi_deg=0.0, to_plot="R")
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_vs_angle(self, set_test_backend, simple_stack):
        """Test plotting R vs angle."""
        analyzer = SpectralAnalyzer(simple_stack)
        angles_deg = be.linspace(0, 80, 10)

        fig, ax = analyzer.view(0.55, aoi_deg=angles_deg, to_plot="R")
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_2d_map(self, set_test_backend, simple_stack):
        """Test 2D plotting (wavelength vs angle)."""
        analyzer = SpectralAnalyzer(simple_stack)
        wavelengths_um = be.linspace(0.4, 0.8, 5)
        angles_deg = be.linspace(0, 60, 4)

        fig, axs = analyzer.view(wavelengths_um, aoi_deg=angles_deg, to_plot="R")
        assert isinstance(fig, plt.Figure)
        assert isinstance(axs, list)
        plt.close(fig)

    def test_plot_multiple_quantities(self, set_test_backend, simple_stack):
        """Test plotting multiple quantities."""
        analyzer = SpectralAnalyzer(simple_stack)
        wavelengths_um = be.linspace(0.4, 0.8, 10)

        fig, ax = analyzer.view(wavelengths_um, to_plot=["R", "T", "A"])
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_with_provided_ax(self, set_test_backend, simple_stack):
        """Test plotting with provided axes."""
        analyzer = SpectralAnalyzer(simple_stack)
        wavelengths_um = be.linspace(0.4, 0.8, 10)

        fig, ax = plt.subplots()
        returned_fig, returned_ax = analyzer.view(wavelengths_um, ax=ax)
        assert returned_fig is fig
        assert returned_ax is ax
        plt.close(fig)

    def test_plot_invalid_quantity(self, set_test_backend, simple_stack):
        """Test that invalid quantity raises error."""
        analyzer = SpectralAnalyzer(simple_stack)
        wavelengths_um = be.linspace(0.4, 0.8, 10)

        with pytest.raises(ValueError, match="to_plot must be"):
            analyzer.view(wavelengths_um, to_plot="invalid")

    def test_plot_scalar_inputs_fails(self, set_test_backend, simple_stack):
        """Test that both scalar inputs raises error."""
        analyzer = SpectralAnalyzer(simple_stack)

        with pytest.raises(
            ValueError,
            match="At least one of wavelength_um or aoi_deg must be an array",
        ):
            analyzer.view(0.55, aoi_deg=0.0)


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
