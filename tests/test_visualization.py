# import pkg_resources
from importlib import resources
from unittest.mock import patch, MagicMock

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pytest

import optiland.backend as be
from optiland import fields
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import BaseGeometry, EvenAsphere
from optiland.materials import AbbeMaterial, BaseMaterial, IdealMaterial, MaterialFile
from optiland.optic import Optic
from optiland.samples.objectives import ReverseTelephoto, TessarLens
from optiland.samples.simple import Edmund_49_847
from optiland.samples.telescopes import HubbleTelescope
from optiland.visualization.base import BaseViewer
from optiland.visualization.system import OpticViewer, OpticViewer3D
from optiland.visualization.system.system import OpticalSystem
from optiland.visualization.system.lens import Lens2D, Lens3D
from optiland.visualization.info import LensInfoViewer
from optiland.visualization.analysis import SurfaceSagViewer

matplotlib.use("Agg")  # use non-interactive backend for testing


class InvalidGeometry(BaseGeometry):
    def __init__(self, coordinate_system=CoordinateSystem):
        super().__init__(coordinate_system())
        self.radius = be.inf

    def sag(self, x=0, y=0):
        return 0

    def distance(self, rays):
        return 0

    def surface_normal(self, rays):
        return 0

    def flip(self):
        pass


class InvalidMaterial(BaseMaterial):
    def __init__(self):
        super().__init__()
        self.index = -42

    def _calculate_n(self, wavelength):
        return -42

    def _calculate_k(self, wavelength):
        return -42


class TestBaseViewer:
    """Tests for the abstract BaseViewer class."""

    def test_instantiating_subclass_without_view_raises_error(self, set_test_backend):
        """Verify that a subclass of BaseViewer must implement the view method."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            # This class is abstract because it doesn't implement 'view'
            class IncompleteViewer(BaseViewer):
                pass

            IncompleteViewer(optic=None)


class TestOpticViewer:
    def test_init(self):
        lens = TessarLens()
        viewer = OpticViewer(lens)
        assert viewer.optic == lens

    def test_view(self, set_test_backend):
        lens = ReverseTelephoto()
        viewer = OpticViewer(lens)
        fig, ax = viewer.view()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_view_from_optic(self, set_test_backend):
        lens = ReverseTelephoto()
        fig, ax = lens.draw()
        assert fig is not None  # verify figure creation
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_view_bonded_lens(self, set_test_backend):
        lens = TessarLens()
        fig, ax = lens.draw()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_view_reflective_lens(self, set_test_backend):
        lens = HubbleTelescope()
        fig, ax = lens.draw()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_view_single_field(self, set_test_backend):
        lens = ReverseTelephoto()
        lens.fields = fields.FieldGroup()
        lens.set_field_type(field_type="angle")
        lens.add_field(y=0)
        fig, ax = lens.draw()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_reference_chief_and_bundle(self, set_test_backend):
        lens = ReverseTelephoto()
        fig, ax = lens.draw(reference="chief")
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_reference_marginal_and_bundle(self, set_test_backend):
        lens = ReverseTelephoto()
        fig, ax = lens.draw(reference="marginal")
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_invalid_reference(self, set_test_backend):
        lens = ReverseTelephoto()
        viewer = OpticViewer(lens)
        with pytest.raises(ValueError):
            viewer.view(reference="invalid")

    def test_reference_chief_only(self, set_test_backend):
        lens = ReverseTelephoto()
        fig, ax = lens.draw(reference="chief", distribution=None)
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_reference_marginal_only(self, set_test_backend):
        lens = ReverseTelephoto()
        fig, ax = lens.draw(reference="marginal", distribution=None)
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_plot_content_is_generated(self, set_test_backend):
        """Verify that rays and lens polygons are actually added to the plot."""
        lens = TessarLens()
        fig, ax = lens.draw(num_rays=5)
        assert len(ax.get_lines()) > 0, "No ray lines were drawn"
        assert len(ax.patches) > 0, "No lens patches were drawn"
        plt.close(fig)

    def test_view_with_custom_plot_parameters(self, set_test_backend):
        lens = ReverseTelephoto()
        viewer = OpticViewer(lens)
        custom_title = "Custom Test Title"
        custom_xlim = (-10, 100)
        custom_ylim = (-25, 25)

        fig, ax = viewer.view(title=custom_title, xlim=custom_xlim, ylim=custom_ylim)

        assert ax.get_title() == custom_title
        assert ax.get_xlim() == custom_xlim
        assert ax.get_ylim() == custom_ylim
        plt.close(fig)

    def test_view_all_wavelengths(self, set_test_backend):
        lens = ReverseTelephoto()
        # Add a second wavelength to test "all"
        lens.add_wavelength(value=0.65)
        viewer = OpticViewer(lens)
        fig, ax = viewer.view(wavelengths="all")
        assert fig is not None
        assert ax is not None
        assert len(ax.get_lines()) > 0  # Ensure rays were drawn
        plt.close(fig)


class TestOpticViewer3D:
    def test_init(self, set_test_backend):
        lens = TessarLens()
        viewer = OpticViewer3D(lens)
        assert viewer.optic == lens

    def test_view(self, set_test_backend):
        lens = ReverseTelephoto()
        viewer = OpticViewer3D(lens)
        with (
            patch.object(viewer.iren, "Start") as mock_start,
            patch.object(viewer.ren_win, "Render") as mock_render,
        ):
            viewer.view()
            mock_start.assert_called_once()
            mock_render.assert_called()

    def test_view_asymmetric(self, set_test_backend):
        lens = ReverseTelephoto()
        lens.surface_group.surfaces[1].geometry.is_symmetric = False
        viewer = OpticViewer3D(lens)
        with (
            patch.object(viewer.iren, "Start") as mock_start,
            patch.object(viewer.ren_win, "Render") as mock_render,
        ):
            viewer.view()
            mock_start.assert_called_once()
            mock_render.assert_called()

    def test_view_bonded_lens(self, set_test_backend):
        lens = TessarLens()
        viewer = OpticViewer3D(lens)
        with (
            patch.object(viewer.iren, "Start") as mock_start,
            patch.object(viewer.ren_win, "Render") as mock_render,
        ):
            viewer.view()
            mock_start.assert_called_once()
            mock_render.assert_called()

    def test_view_reflective_lens(self, set_test_backend):
        lens = HubbleTelescope()
        viewer = OpticViewer3D(lens)
        with (
            patch.object(viewer.iren, "Start") as mock_start,
            patch.object(viewer.ren_win, "Render") as mock_render,
        ):
            viewer.view()
            mock_start.assert_called_once()
            mock_render.assert_called()

    def test_view_single_field(self, set_test_backend):
        lens = ReverseTelephoto()
        lens.fields = fields.FieldGroup()
        lens.set_field_type(field_type="angle")
        lens.add_field(y=0)
        viewer = OpticViewer3D(lens)
        with (
            patch.object(viewer.iren, "Start") as mock_start,
            patch.object(viewer.ren_win, "Render") as mock_render,
        ):
            viewer.view()
            mock_start.assert_called_once()
            mock_render.assert_called()

    def test_non_symmetric(self, set_test_backend):
        lens = ReverseTelephoto()
        lens.surface_group.surfaces[1].geometry.is_symmetric = False
        viewer = OpticViewer3D(lens)
        viewer.system._identify_components()
        c = viewer.system.components[0]
        assert not c.is_symmetric

    def test_view_toroidal(self, set_test_backend):
        cylindrical_lens = Optic()
        cylindrical_lens.add_surface(index=0, radius=be.inf, thickness=be.inf)
        cylindrical_lens.add_surface(
            index=1,
            thickness=7,
            radius_x=20,  # <- radius: x radius of rotation.
            radius_y=25,
            is_stop=True,
            material="N-BK7",
            surface_type="toroidal",
            conic=0.0,
            coefficients=[],
        )
        cylindrical_lens.add_surface(index=2, thickness=65)
        cylindrical_lens.add_surface(index=3)
        cylindrical_lens.set_aperture(aperture_type="EPD", value=20.0)
        cylindrical_lens.set_field_type(field_type="angle")
        cylindrical_lens.add_field(y=0)
        cylindrical_lens.add_wavelength(value=0.587, is_primary=True)

        viewer = OpticViewer3D(cylindrical_lens)

        with (
            patch.object(viewer.iren, "Start") as mock_start,
            patch.object(viewer.ren_win, "Render") as mock_render,
        ):
            viewer.view()
            mock_start.assert_called_once()
            mock_render.assert_called()

    def test_view_non_symmetric(self, set_test_backend):
        lens = ReverseTelephoto()
        lens.surface_group.surfaces[1].geometry.is_symmetric = False
        viewer = OpticViewer3D(lens)
        viewer.system._identify_components()
        with (
            patch.object(viewer.iren, "Start") as mock_start,
            patch.object(viewer.ren_win, "Render") as mock_render,
        ):
            viewer.view(reference="chief")
            mock_start.assert_called_once()
            mock_render.assert_called()

    def test_reference_chief_and_bundle(self, set_test_backend):
        lens = ReverseTelephoto()
        viewer = OpticViewer3D(lens)
        with (
            patch.object(viewer.iren, "Start") as mock_start,
            patch.object(viewer.ren_win, "Render") as mock_render,
        ):
            viewer.view(reference="chief")
            mock_start.assert_called_once()
            mock_render.assert_called()

    def test_reference_marginal_and_bundle(self, set_test_backend):
        lens = ReverseTelephoto()
        viewer = OpticViewer3D(lens)
        with (
            patch.object(viewer.iren, "Start") as mock_start,
            patch.object(viewer.ren_win, "Render") as mock_render,
        ):
            viewer.view(reference="marginal")
            mock_start.assert_called_once()
            mock_render.assert_called()

    def test_invalid_reference(self, set_test_backend):
        lens = ReverseTelephoto()
        viewer = OpticViewer3D(lens)
        with pytest.raises(ValueError):
            viewer.view(reference="invalid")

    def test_reference_chief_only(self, set_test_backend):
        lens = ReverseTelephoto()
        viewer = OpticViewer3D(lens)
        with (
            patch.object(viewer.iren, "Start") as mock_start,
            patch.object(viewer.ren_win, "Render") as mock_render,
        ):
            viewer.view(reference="chief", distribution=None)
            mock_start.assert_called_once()
            mock_render.assert_called()

    def test_reference_marginal_only(self, set_test_backend):
        lens = ReverseTelephoto()
        viewer = OpticViewer3D(lens)
        with (
            patch.object(viewer.iren, "Start") as mock_start,
            patch.object(viewer.ren_win, "Render") as mock_render,
        ):
            viewer.view(reference="marginal", distribution=None)
            mock_start.assert_called_once()
            mock_render.assert_called()

    @pytest.mark.parametrize(
        "dark_mode, bg1, bg2",
        [
            (True, (0.13, 0.15, 0.19), (0.195, 0.21, 0.24)),
            (False, (0.8, 0.9, 1.0), (0.4, 0.5, 0.6)),
        ],
    )
    @patch("optiland.visualization.system.optic_viewer_3d.vtk")
    def test_view_sets_background_color_for_theme(
        self, mock_vtk, dark_mode, bg1, bg2, set_test_backend
    ):
        """Test that view() correctly sets the background for both themes."""
        lens = TessarLens()
        viewer = OpticViewer3D(lens)

        mock_renderer_instance = mock_vtk.vtkRenderer.return_value
        viewer.view(dark_mode=dark_mode)

        mock_renderer_instance.SetBackground.assert_called_with(*bg1)
        mock_renderer_instance.SetBackground2.assert_called_with(*bg2)


class TestLensInfoViewer:
    def test_view_standard(self, capsys, set_test_backend):
        lens = TessarLens()
        viewer = LensInfoViewer(lens)
        viewer.view()
        captured = capsys.readouterr()
        assert "Type" in captured.out
        assert "Comment" in captured.out
        assert "Radius" in captured.out
        assert "Thickness" in captured.out
        assert "Material" in captured.out
        assert "Conic" in captured.out
        assert "Semi-aperture" in captured.out

    def test_view_from_optic(self, capsys, set_test_backend):
        lens = TessarLens()
        lens.info()
        captured = capsys.readouterr()
        assert "Type" in captured.out
        assert "Comment" in captured.out
        assert "Radius" in captured.out
        assert "Thickness" in captured.out
        assert "Material" in captured.out
        assert "Conic" in captured.out
        assert "Semi-aperture" in captured.out

    def test_view_plano_convex(self, capsys, set_test_backend):
        lens = Edmund_49_847()
        viewer = LensInfoViewer(lens)
        viewer.view()
        captured = capsys.readouterr()
        assert "Type" in captured.out
        assert "Comment" in captured.out
        assert "Radius" in captured.out
        assert "Thickness" in captured.out
        assert "Material" in captured.out
        assert "Conic" in captured.out
        assert "Semi-aperture" in captured.out

    def test_invalid_geometry(self, set_test_backend):
        lens = ReverseTelephoto()
        lens.surface_group.surfaces[2].geometry = InvalidGeometry()
        viewer = LensInfoViewer(lens)
        with pytest.raises(ValueError):
            viewer.view()

    def test_view_reflective_lens(self, capsys, set_test_backend):
        lens = HubbleTelescope()
        viewer = LensInfoViewer(lens)
        viewer.view()
        captured = capsys.readouterr()
        assert "Type" in captured.out
        assert "Comment" in captured.out
        assert "Radius" in captured.out
        assert "Thickness" in captured.out
        assert "Material" in captured.out
        assert "Conic" in captured.out
        assert "Semi-aperture" in captured.out
        assert "Mirror" in captured.out

    def test_view_asphere(self, capsys, set_test_backend):
        lens = ReverseTelephoto()
        asphere_geo = EvenAsphere(CoordinateSystem(), 100, coefficients=[0.1, 0.3, 1.2])
        lens.surface_group.surfaces[2].geometry = asphere_geo
        viewer = LensInfoViewer(lens)
        viewer.view()
        captured = capsys.readouterr()
        assert "Type" in captured.out
        assert "Comment" in captured.out
        assert "Radius" in captured.out
        assert "Thickness" in captured.out
        assert "Material" in captured.out
        assert "Conic" in captured.out
        assert "Semi-aperture" in captured.out
        assert "Even Asphere" in captured.out
        assert "c0" in captured.out
        assert "c1" in captured.out
        assert "c2" in captured.out

    def test_view_material_file(self, capsys, set_test_backend):
        lens = ReverseTelephoto()
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/glass/hoya/LAC9.yml"
            ),
        )
        mat = MaterialFile(filename)
        lens.surface_group.surfaces[2].material_post = mat
        viewer = LensInfoViewer(lens)
        viewer.view()
        captured = capsys.readouterr()
        assert "Type" in captured.out
        assert "Comment" in captured.out
        assert "Radius" in captured.out
        assert "Thickness" in captured.out
        assert "Material" in captured.out
        assert "Conic" in captured.out
        assert "Semi-aperture" in captured.out

    def test_view_ideal_material(self, capsys, set_test_backend):
        lens = ReverseTelephoto()
        mat = IdealMaterial(1.5)
        lens.surface_group.surfaces[2].material_post = mat
        viewer = LensInfoViewer(lens)
        viewer.view()
        captured = capsys.readouterr()
        assert "Type" in captured.out
        assert "Comment" in captured.out
        assert "Radius" in captured.out
        assert "Thickness" in captured.out
        assert "Material" in captured.out
        assert "Conic" in captured.out
        assert "Semi-aperture" in captured.out

    def test_view_invalid_material(self, set_test_backend):
        lens = ReverseTelephoto()
        lens.surface_group.surfaces[2].material_post = InvalidMaterial()
        viewer = LensInfoViewer(lens)
        with pytest.raises(ValueError):
            viewer.view()

    def test_view_abbe_material(self, set_test_backend):
        lens = ReverseTelephoto()
        lens.surface_group.surfaces[2].material_post = AbbeMaterial(1.5, 60)
        viewer = LensInfoViewer(lens)
        viewer.view()


class TestSurfaceSagViewer:
    """Tests for the new SurfaceSagViewer."""

    def test_view_with_cylindrical_lens(self, set_test_backend):
        """Test the sag viewer with a biconic (cylindrical) lens."""
        lens = Optic(name="Cylindrical Test Lens")
        lens.add_surface(index=0, thickness=be.inf)
        lens.add_surface(index=1, surface_type="biconic", radius_y=-50, thickness=5)
        lens.add_surface(index=2, radius=be.inf)

        viewer = SurfaceSagViewer(lens)
        viewer.view(surface_index=1)
        # Check that a plot was created by inspecting the active figure
        assert plt.gcf() is not None
        plt.close()

    def test_view_with_custom_cross_section(self, set_test_backend):
        """Test the sag viewer with non-default cross-sections."""
        lens = Optic(name="Cylindrical Test Lens")
        lens.add_surface(index=0, thickness=be.inf)
        lens.add_surface(index=1, surface_type="biconic", radius_y=-50, thickness=5)
        lens.add_surface(index=2, radius=be.inf)

        viewer = SurfaceSagViewer(lens)
        viewer.view(surface_index=1, y_cross_section=1.5, x_cross_section=-1.5)
        assert plt.gcf() is not None
        plt.close()


@pytest.mark.parametrize("projection, lens_class", [("2d", Lens2D), ("3d", Lens3D)])
def test_mangin_mirror_visualization(projection, lens_class, set_test_backend):
    """Test that a Mangin mirror is visualized as a single lens component."""
    # Create a simple Mangin mirror
    mangin_mirror = Optic(name="Mangin Mirror")
    mangin_mirror.add_wavelength(value=0.55, is_primary=True)
    mangin_mirror.add_surface(index=0, radius=be.inf, thickness=be.inf)  # Object
    mangin_mirror.add_surface(index=1, radius=-100, thickness=+5, material="N-BK7")   # Front surface
    mangin_mirror.add_surface(index=2, radius=-100, thickness=-5, material="mirror", is_stop=True)  # Back surface (reflective)
    mangin_mirror.add_surface(index=3, radius=-100, thickness=-50, material="N-BK7")   # Front surface
    mangin_mirror.add_surface(index=4, radius=be.inf)  # Image
    mangin_mirror.set_field_type("angle")
    mangin_mirror.add_field(y=0)
    mangin_mirror.set_aperture(aperture_type="EPD", value=25)
    mangin_mirror.add_wavelength(value=0.65, is_primary=True)

    # Dummy rays object for OpticalSystem
    class DummyRays:
        def __init__(self, optic):
            self.r_extent = [15] * optic.surface_group.num_surfaces

    optical_system = OpticalSystem(
        mangin_mirror, DummyRays(mangin_mirror), projection=projection
    )
    optical_system._identify_components()

    # Verify the components were identified correctly
    # Two "lens" (really just same lens, superimposed) and one image plane.
    assert (
        len(optical_system.components) == 3
    ), "Expected two components: the lens and the image plane."

    lens_components = [
        c for c in optical_system.components if isinstance(c, lens_class)
    ]
    
    # Ensure "two" lenses are found - really, this is the same lens, superimposed.
    assert len(lens_components) == 2

    # Ensure that the identified lens consists of two surfaces.
    assert (
        len(lens_components[0].surfaces) == 2
    ), "The Mangin mirror component should be made of two surfaces."
