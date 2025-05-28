# import pkg_resources
from importlib import resources
from unittest.mock import patch, MagicMock # Added MagicMock

import matplotlib
import matplotlib.pyplot as plt
import optiland.backend as be
import pytest

from optiland import fields
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import BaseGeometry, EvenAsphere, Sphere # Added Sphere
from optiland.materials import AbbeMaterial, BaseMaterial, IdealMaterial, MaterialFile
from optiland.optic import Optic # Added Optic
from optiland.samples.objectives import ReverseTelephoto, TessarLens
from optiland.samples.simple import Edmund_49_847
from optiland.samples.telescopes import HubbleTelescope
from optiland.surface import Surface # Added Surface
from optiland.visualization import LensInfoViewer, OpticViewer, OpticViewer3D
from optiland.visualization.lens import Lens3D # Added Lens3D

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


class InvalidMaterial(BaseMaterial):
    def __init__(self):
        super().__init__()
        self.index = -42

    def n(self, wavelength):
        return -42

    def k(self, wavelength):
        return -42


class TestOpticViewer:
    def test_init(self):
        lens = TessarLens()
        viewer = OpticViewer(lens)
        assert viewer.optic == lens

    @patch("matplotlib.pyplot.show")
    def test_view(self, mock_show, set_test_backend):
        lens = ReverseTelephoto()
        viewer = OpticViewer(lens)
        viewer.view()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_from_optic(self, mock_show, set_test_backend):
        lens = ReverseTelephoto()
        lens.draw()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_bonded_lens(self, mock_show, set_test_backend):
        lens = TessarLens()
        viewer = OpticViewer(lens)
        viewer.view()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_reflective_lens(self, mock_show, set_test_backend):
        lens = HubbleTelescope()
        viewer = OpticViewer(lens)
        viewer.view()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_single_field(self, mock_show, set_test_backend):
        lens = ReverseTelephoto()
        lens.fields = fields.FieldGroup()
        lens.set_field_type(field_type="angle")
        lens.add_field(y=0)
        viewer = OpticViewer(lens)
        viewer.view()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_reference_chief_and_bundle(self, mock_show, set_test_backend):
        lens = ReverseTelephoto()
        viewer = OpticViewer(lens)
        viewer.view(reference="chief")
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_reference_marginal_and_bundle(self, mock_show, set_test_backend):
        lens = ReverseTelephoto()
        viewer = OpticViewer(lens)
        viewer.view(reference="marginal")
        mock_show.assert_called_once()
        plt.close()

    def test_invalid_reference(self, set_test_backend):
        lens = ReverseTelephoto()
        viewer = OpticViewer(lens)
        with pytest.raises(ValueError):
            viewer.view(reference="invalid")

    @patch("matplotlib.pyplot.show")
    def test_reference_chief_only(self, mock_show, set_test_backend):
        lens = ReverseTelephoto()
        viewer = OpticViewer(lens)
        viewer.view(reference="chief", distribution=None)
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_reference_marginal_only(self, mock_show, set_test_backend):
        lens = ReverseTelephoto()
        viewer = OpticViewer(lens)
        viewer.view(reference="marginal", distribution=None)
        mock_show.assert_called_once()
        plt.close()


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

        lens = ReverseTelephoto()
        lens.surface_group.surfaces[1].geometry.cs.rx = 0.1
        viewer = OpticViewer3D(lens)
        viewer.system._identify_components()
        c = viewer.system.components[0]
        assert not c.is_symmetric

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

    def test_lens3d_uses_surface3d_get_surface_for_plotting(self, set_test_backend):
        # 1. Create a simple Optic with two surfaces
        s1_geom = Sphere(radius=50)
        s1 = Surface(geometry=s1_geom, material_post=IdealMaterial(1.5))
        s1.name = "S1"
        s1.semi_aperture = 10.0

        s2_geom = Sphere(radius=-50)
        # Tilting s2 makes the overall lens system asymmetrically handled by older logic,
        # but with the refactor, even symmetric lenses use Surface3D.get_surface().
        # This tilt ensures it would have gone through _plot_surfaces path before,
        # and now correctly uses the new path.
        s2_geom.cs.rx = 0.1 
        s2 = Surface(geometry=s2_geom, material_post=IdealMaterial(1.0)) # material_post is air
        s2.name = "S2"
        s2.semi_aperture = 10.0

        lens_optic = Optic(surface_group=[s1, s2])

        # 2. Instantiate OpticViewer3D and find Lens3D component
        viewer = OpticViewer3D(lens_optic)
        viewer.system._identify_components()

        lens_3d_component = None
        for comp in viewer.system.components:
            if isinstance(comp, Lens3D):
                lens_3d_component = comp
                break
        assert lens_3d_component is not None, "Lens3D component not found in OpticViewer3D system"
        assert len(lens_3d_component.plotting_surfaces_3d) == 2, \
            f"Expected 2 plotting surfaces, found {len(lens_3d_component.plotting_surfaces_3d)}"

        # 3. Patch Surface3D.get_surface and call viewer.view()
        # Path to the method to patch.
        # Surface3D is imported in optiland.visualization.lens as `from .surface import Surface3D`
        # So the path from the perspective of tests/test_visualization.py is:
        target_path_get_surface = "optiland.visualization.lens.Surface3D.get_surface"
        # If Surface3D was imported directly into lens.py like `from optiland.visualization.surface import Surface3D`
        # then the path would be `optiland.visualization.surface.Surface3D.get_surface`
        # The current import style in lens.py makes it `optiland.visualization.lens.Surface3D`

        with patch(target_path_get_surface) as mock_get_surface, \
             patch.object(viewer.iren, "Start") as mock_iren_start, \
             patch.object(viewer.ren_win, "Render") as mock_ren_render:

            # mock_get_surface needs to return a mock actor that can be added to the renderer
            mock_get_surface.return_value = MagicMock() 

            viewer.view() # This triggers Lens3D.plot()

            # --- Assertions ---
            # Verify mock_get_surface was called for each Surface3D object
            assert mock_get_surface.call_count == len(lens_3d_component.plotting_surfaces_3d), \
                f"Expected {len(lens_3d_component.plotting_surfaces_3d)} calls to get_surface, " \
                f"got {mock_get_surface.call_count}"

            # Verify that the method was called on the correct Surface3D instances
            actual_called_instances = [c.args[0] for c in mock_get_surface.call_args_list]
            
            for s3d_obj in lens_3d_component.plotting_surfaces_3d:
                assert s3d_obj in actual_called_instances, \
                    f"Surface3D object ({s3d_obj}) from plotting_surfaces_3d was not found among " \
                    f"instances calling get_surface ({actual_called_instances})."
            
            # Verify that the VTK pipeline methods were also called
            mock_iren_start.assert_called_once()
            mock_ren_render.assert_called()


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
        asphere_geo = EvenAsphere(CoordinateSystem(), 100)
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
