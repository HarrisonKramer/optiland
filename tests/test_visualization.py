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

    @patch('optiland.visualization.LensViewer3D.view')
    def test_view(self, mock_view):
        lens = ReverseTelephoto()
        viewer = LensViewer3D(lens)
        viewer.view()
        mock_view.assert_called_once()

    @patch('optiland.visualization.LensViewer3D.view')
    def test_view_bonded_lens(self, mock_view):
        lens = TessarLens()
        viewer = LensViewer3D(lens)
        viewer.view()
        mock_view.assert_called_once()
        plt.close()

    @patch('optiland.visualization.LensViewer3D.view')
    def test_view_reflective_lens(self, mock_view):
        lens = HubbleTelescope()
        viewer = LensViewer3D(lens)
        viewer.view()
        mock_view.assert_called_once()
        plt.close()

    @patch('optiland.visualization.LensViewer3D.view')
    def test_view_single_field(self, mock_view):
        lens = ReverseTelephoto()
        lens.fields = fields.FieldGroup()
        lens.set_field_type(field_type='angle')
        lens.add_field(y=0)
        viewer = LensViewer3D(lens)
        viewer.view()
        mock_view.assert_called_once()
        plt.close()


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
