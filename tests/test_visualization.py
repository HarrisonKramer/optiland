# import pkg_resources
from importlib import resources
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from optiland import fields
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import BaseGeometry, EvenAsphere
from optiland.materials import AbbeMaterial, BaseMaterial, IdealMaterial, MaterialFile
from optiland.samples.objectives import ReverseTelephoto, TessarLens
from optiland.samples.simple import Edmund_49_847
from optiland.samples.telescopes import HubbleTelescope
from optiland.visualization import LensInfoViewer, OpticViewer, OpticViewer3D

matplotlib.use("Agg")  # use non-interactive backend for testing


class InvalidGeometry(BaseGeometry):
    def __init__(self, coordinate_system=CoordinateSystem):
        super().__init__(coordinate_system())
        self.radius = np.inf

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
    def test_view(self, mock_show):
        lens = ReverseTelephoto()
        viewer = OpticViewer(lens)
        viewer.view()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_from_optic(self, mock_show):
        lens = ReverseTelephoto()
        lens.draw()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_bonded_lens(self, mock_show):
        lens = TessarLens()
        viewer = OpticViewer(lens)
        viewer.view()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_reflective_lens(self, mock_show):
        lens = HubbleTelescope()
        viewer = OpticViewer(lens)
        viewer.view()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_view_single_field(self, mock_show):
        lens = ReverseTelephoto()
        lens.fields = fields.FieldGroup()
        lens.set_field_type(field_type="angle")
        lens.add_field(y=0)
        viewer = OpticViewer(lens)
        viewer.view()
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_reference_chief_and_bundle(self, mock_show):
        lens = ReverseTelephoto()
        viewer = OpticViewer(lens)
        viewer.view(reference="chief")
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_reference_marginal_and_bundle(self, mock_show):
        lens = ReverseTelephoto()
        viewer = OpticViewer(lens)
        viewer.view(reference="marginal")
        mock_show.assert_called_once()
        plt.close()

    def test_invalid_reference(self):
        lens = ReverseTelephoto()
        viewer = OpticViewer(lens)
        with pytest.raises(ValueError):
            viewer.view(reference="invalid")

    @patch("matplotlib.pyplot.show")
    def test_reference_chief_only(self, mock_show):
        lens = ReverseTelephoto()
        viewer = OpticViewer(lens)
        viewer.view(reference="chief", distribution=None)
        mock_show.assert_called_once()
        plt.close()

    @patch("matplotlib.pyplot.show")
    def test_reference_marginal_only(self, mock_show):
        lens = ReverseTelephoto()
        viewer = OpticViewer(lens)
        viewer.view(reference="marginal", distribution=None)
        mock_show.assert_called_once()
        plt.close()


class TestOpticViewer3D:
    def test_init(self):
        lens = TessarLens()
        viewer = OpticViewer3D(lens)
        assert viewer.optic == lens

    def test_view(self):
        lens = ReverseTelephoto()
        viewer = OpticViewer3D(lens)
        with (
            patch.object(viewer.iren, "Start") as mock_start,
            patch.object(viewer.ren_win, "Render") as mock_render,
        ):
            viewer.view()
            mock_start.assert_called_once()
            mock_render.assert_called()

    def test_view_asymmetric(self):
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

    def test_view_bonded_lens(self):
        lens = TessarLens()
        viewer = OpticViewer3D(lens)
        with (
            patch.object(viewer.iren, "Start") as mock_start,
            patch.object(viewer.ren_win, "Render") as mock_render,
        ):
            viewer.view()
            mock_start.assert_called_once()
            mock_render.assert_called()

    def test_view_reflective_lens(self):
        lens = HubbleTelescope()
        viewer = OpticViewer3D(lens)
        with (
            patch.object(viewer.iren, "Start") as mock_start,
            patch.object(viewer.ren_win, "Render") as mock_render,
        ):
            viewer.view()
            mock_start.assert_called_once()
            mock_render.assert_called()

    def test_view_single_field(self):
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

    def test_non_symmetric(self):
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

    def test_view_non_symmetric(self):
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

    def test_reference_chief_and_bundle(self):
        lens = ReverseTelephoto()
        viewer = OpticViewer3D(lens)
        with (
            patch.object(viewer.iren, "Start") as mock_start,
            patch.object(viewer.ren_win, "Render") as mock_render,
        ):
            viewer.view(reference="chief")
            mock_start.assert_called_once()
            mock_render.assert_called()

    def test_reference_marginal_and_bundle(self):
        lens = ReverseTelephoto()
        viewer = OpticViewer3D(lens)
        with (
            patch.object(viewer.iren, "Start") as mock_start,
            patch.object(viewer.ren_win, "Render") as mock_render,
        ):
            viewer.view(reference="marginal")
            mock_start.assert_called_once()
            mock_render.assert_called()

    def test_invalid_reference(self):
        lens = ReverseTelephoto()
        viewer = OpticViewer3D(lens)
        with pytest.raises(ValueError):
            viewer.view(reference="invalid")

    def test_reference_chief_only(self):
        lens = ReverseTelephoto()
        viewer = OpticViewer3D(lens)
        with (
            patch.object(viewer.iren, "Start") as mock_start,
            patch.object(viewer.ren_win, "Render") as mock_render,
        ):
            viewer.view(reference="chief", distribution=None)
            mock_start.assert_called_once()
            mock_render.assert_called()

    def test_reference_marginal_only(self):
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
    def test_view_standard(self, capsys):
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

    def test_view_from_optic(self, capsys):
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

    def test_view_plano_convex(self, capsys):
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

    def test_invalid_geometry(self):
        lens = ReverseTelephoto()
        lens.surface_group.surfaces[2].geometry = InvalidGeometry()
        viewer = LensInfoViewer(lens)
        with pytest.raises(ValueError):
            viewer.view()

    def test_view_reflective_lens(self, capsys):
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

    def test_view_asphere(self, capsys):
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

    def test_view_material_file(self, capsys):
        lens = ReverseTelephoto()
        filename = str(
            resources.files("optiland.database").joinpath(
                "data-nk/glass/hoya/LAC9.yml"
            ),
        )
        # filename = pkg_resources.resource_filename(
        #    'optiland.database', 'data-nk/glass/hoya/LAC9.yml'
        # )
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

    def test_view_ideal_material(self, capsys):
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

    def test_view_invalid_material(self):
        lens = ReverseTelephoto()
        lens.surface_group.surfaces[2].material_post = InvalidMaterial()
        viewer = LensInfoViewer(lens)
        with pytest.raises(ValueError):
            viewer.view()

    def test_view_abbe_material(self):
        lens = ReverseTelephoto()
        lens.surface_group.surfaces[2].material_post = AbbeMaterial(1.5, 60)
        viewer = LensInfoViewer(lens)
        viewer.view()
