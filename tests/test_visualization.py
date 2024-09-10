import os
from unittest.mock import patch
import pytest
import numpy as np
from optiland.visualization import LensViewer, LensViewer3D, LensInfoViewer
from optiland.samples.objectives import (
    TessarLens,
    ReverseTelephoto
)
from optiland.samples.simple import Edmund_49_847
from optiland.samples.telescopes import HubbleTelescope
from optiland import fields
from optiland.geometries import BaseGeometry, EvenAsphere
from optiland.coordinate_system import CoordinateSystem
from optiland.materials import BaseMaterial, MaterialFile, IdealMaterial
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # use non-interactive backend for testing


class InvalidGeometry(BaseGeometry):
    def __init__(self, coordinate_system=CoordinateSystem()):
        super().__init__(coordinate_system)
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


class TestLensViewer:
    def test_init(self):
        lens = TessarLens()
        viewer = LensViewer(lens)
        assert viewer.optic == lens
        assert np.array_equal(viewer._real_ray_extent, np.zeros(10))

    @patch('matplotlib.pyplot.show')
    def test_view(self, mock_show):
        lens = ReverseTelephoto()
        viewer = LensViewer(lens)
        viewer.view()
        mock_show.assert_called_once()
        plt.close()

    @patch('matplotlib.pyplot.show')
    def test_view_bonded_lens(self, mock_show):
        lens = TessarLens()
        viewer = LensViewer(lens)
        viewer.view()
        mock_show.assert_called_once()
        plt.close()

    @patch('matplotlib.pyplot.show')
    def test_view_reflective_lens(self, mock_show):
        lens = HubbleTelescope()
        viewer = LensViewer(lens)
        viewer.view()
        mock_show.assert_called_once()
        plt.close()

    @patch('matplotlib.pyplot.show')
    def test_view_single_field(self, mock_show):
        lens = ReverseTelephoto()
        lens.fields = fields.FieldGroup()
        lens.set_field_type(field_type='angle')
        lens.add_field(y=0)
        viewer = LensViewer(lens)
        viewer.view()
        mock_show.assert_called_once()
        plt.close()


class TestLensViewer3D:
    def test_init(self):
        lens = TessarLens()
        viewer = LensViewer3D(lens)
        assert viewer.optic == lens
        assert np.array_equal(viewer._real_ray_extent, np.zeros(10))

    def test_view(self):
        lens = ReverseTelephoto()
        viewer = LensViewer3D(lens)
        with patch.object(viewer.iren, 'Start') as mock_start, \
             patch.object(viewer.renWin, 'Render') as mock_render:

            viewer.view()
            mock_start.assert_called_once()
            mock_render.assert_called()

    def test_view_bonded_lens(self):
        lens = TessarLens()
        viewer = LensViewer3D(lens)
        with patch.object(viewer.iren, 'Start') as mock_start, \
             patch.object(viewer.renWin, 'Render') as mock_render:

            viewer.view()
            mock_start.assert_called_once()
            mock_render.assert_called()

    def test_view_reflective_lens(self):
        lens = HubbleTelescope()
        viewer = LensViewer3D(lens)
        with patch.object(viewer.iren, 'Start') as mock_start, \
             patch.object(viewer.renWin, 'Render') as mock_render:

            viewer.view()
            mock_start.assert_called_once()
            mock_render.assert_called()

    def test_view_single_field(self):
        lens = ReverseTelephoto()
        lens.fields = fields.FieldGroup()
        lens.set_field_type(field_type='angle')
        lens.add_field(y=0)
        viewer = LensViewer3D(lens)
        with patch.object(viewer.iren, 'Start') as mock_start, \
             patch.object(viewer.renWin, 'Render') as mock_render:

            viewer.view()
            mock_start.assert_called_once()
            mock_render.assert_called()


class TestLensInfoViewer:
    def test_view_standard(self, capsys):
        lens = TessarLens()
        viewer = LensInfoViewer(lens)
        viewer.view()
        captured = capsys.readouterr()
        assert 'Type' in captured.out
        assert 'Radius' in captured.out
        assert 'Thickness' in captured.out
        assert 'Material' in captured.out
        assert 'Conic' in captured.out
        assert 'Semi-aperture' in captured.out

    def test_view_plano_convex(self, capsys):
        lens = Edmund_49_847()
        viewer = LensInfoViewer(lens)
        viewer.view()
        captured = capsys.readouterr()
        assert 'Type' in captured.out
        assert 'Radius' in captured.out
        assert 'Thickness' in captured.out
        assert 'Material' in captured.out
        assert 'Conic' in captured.out
        assert 'Semi-aperture' in captured.out

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
        assert 'Type' in captured.out
        assert 'Radius' in captured.out
        assert 'Thickness' in captured.out
        assert 'Material' in captured.out
        assert 'Conic' in captured.out
        assert 'Semi-aperture' in captured.out
        assert 'Mirror' in captured.out

    def test_view_asphere(self, capsys):
        lens = ReverseTelephoto()
        asphere_geo = EvenAsphere(CoordinateSystem(), 100)
        lens.surface_group.surfaces[2].geometry = asphere_geo
        viewer = LensInfoViewer(lens)
        viewer.view()
        captured = capsys.readouterr()
        assert 'Type' in captured.out
        assert 'Radius' in captured.out
        assert 'Thickness' in captured.out
        assert 'Material' in captured.out
        assert 'Conic' in captured.out
        assert 'Semi-aperture' in captured.out
        assert 'Even Asphere' in captured.out

    def test_view_material_file(self, capsys):
        lens = ReverseTelephoto()
        filename = os.path.join(os.path.dirname(__file__),
                                '../database/data-nk/glass/hoya/', 'LAC9.yml')
        mat = MaterialFile(filename)
        lens.surface_group.surfaces[2].material_post = mat
        viewer = LensInfoViewer(lens)
        viewer.view()
        captured = capsys.readouterr()
        assert 'Type' in captured.out
        assert 'Radius' in captured.out
        assert 'Thickness' in captured.out
        assert 'Material' in captured.out
        assert 'Conic' in captured.out
        assert 'Semi-aperture' in captured.out

    def test_view_ideal_material(self, capsys):
        lens = ReverseTelephoto()
        mat = IdealMaterial(1.5)
        lens.surface_group.surfaces[2].material_post = mat
        viewer = LensInfoViewer(lens)
        viewer.view()
        captured = capsys.readouterr()
        assert 'Type' in captured.out
        assert 'Radius' in captured.out
        assert 'Thickness' in captured.out
        assert 'Material' in captured.out
        assert 'Conic' in captured.out
        assert 'Semi-aperture' in captured.out

    def test_view_invalid_material(self):
        lens = ReverseTelephoto()
        lens.surface_group.surfaces[2].material_post = InvalidMaterial()
        viewer = LensInfoViewer(lens)
        with pytest.raises(ValueError):
            viewer.view()
