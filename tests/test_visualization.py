# import pkg_resources
from importlib import resources
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
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
from optiland.visualization import LensInfoViewer, OpticViewer, OpticViewer3D

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

    def test_view_toroidal(self, set_test_backend):
        cylindrical_lens = Optic()
        cylindrical_lens.add_surface(index=0, radius=be.inf, thickness=be.inf)
        cylindrical_lens.add_surface(
            index=1,
            thickness=7,
            radius=20,  # <- radius: x radius of rotation.
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
