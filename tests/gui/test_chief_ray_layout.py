"""Chief identity, physical traces, exact counts and redraw-only appearance."""

from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest
from matplotlib.colors import to_rgba
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

import optiland.backend as be
from optiland.physical_apertures import RadialAperture
from optiland.visualization.system.ray_path import (
    neutral_reference_mask,
    physical_ray_path,
)
from optiland.visualization.system.rays import Rays2D
from optiland_gui.optiland_connector import OptilandConnector
from optiland_gui.services.job_records import OpticSnapshot
from optiland_gui.services.layout_rays import layout_pupil, trace_layout_rays
from optiland_gui.services.layout_tasks import prepare_2d, prepare_3d
from optiland_gui.surface_interaction import SurfaceInteractionState
from optiland_gui.viewer_panel import MatplotlibViewer
from optiland_gui.widgets.ray_count import RayCountSpinBox
from tests.gui.test_calculation_jobs import wait_for


def progress(*args):
    pass


@pytest.mark.parametrize(
    "distribution,requested,total",
    [
        ("line_x", 1, 1),
        ("line_y", 1, 1),
        ("line_y", 2, 3),
        ("line_x", 4, 5),
        ("line_y", 100, 101),
        ("random", 4, 4),
        ("hexapolar", 1, 1),
        ("hexapolar", 8, 19),
        ("ring", 25, 25),
    ],
)
def test_counts_center_and_symmetry(set_test_backend, distribution, requested, total):
    x, y, chief = layout_pupil(requested, distribution)
    points = np.column_stack((be.to_numpy(x), be.to_numpy(y)))
    assert points.shape == (total, 2)
    np.testing.assert_array_equal(points[chief], [0, 0])
    assert np.count_nonzero(np.linalg.norm(points, axis=1) < 1e-14) == 1
    if distribution in ("line_x", "line_y", "hexapolar"):
        for point in points:
            assert np.any(np.linalg.norm(points + point, axis=1) < 1e-12)


def test_off_axis_chiefs_match_independent_trace_and_preserve_snapshot(
    set_test_backend, minimal_optic
):
    minimal_optic.surfaces[0].geometry.cs.z = -10.0
    minimal_optic.fields.set_type("angle")
    minimal_optic.fields.add(y=2.0)
    minimal_optic.surfaces[1].geometry.cs.y = 0.1
    snapshot = OpticSnapshot.capture(minimal_optic)
    before = json.dumps(minimal_optic.to_dict(), sort_keys=True)
    optic = snapshot.restore()
    paths = trace_layout_rays(Rays2D(optic), 5, "line_y", progress, threading.Event())
    chiefs = [path for path in paths if path["chief"]]
    assert len(paths) == 10 and len(chiefs) == 2
    assert [path["color_index"] for path in chiefs] == [0, 1]
    for path in chiefs:
        optic.trace_generic(
            *path["field"], Px=0, Py=0, wavelength=optic.primary_wavelength
        )
        points = np.column_stack(
            [be.to_numpy(getattr(optic.surfaces, axis))[:, 0] for axis in "xyz"]
        )
        expected = physical_ray_path(
            points,
            be.to_numpy(optic.surfaces.intensity)[:, 0],
            neutral_reference_mask(list(optic.surfaces)),
        )
        np.testing.assert_allclose(path["xyz"], expected, atol=1e-10)
    assert abs(chiefs[-1]["xyz"][1, 1] - chiefs[-1]["xyz"][0, 1]) > 0.1
    assert json.dumps(minimal_optic.to_dict(), sort_keys=True) == before


def test_one_ray_keeps_lens_outlines_and_image_marker(minimal_optic):
    snapshot = OpticSnapshot.capture(minimal_optic)
    data = prepare_2d(
        snapshot, {"num_rays": 1, "distribution": "line_y"}, progress, threading.Event()
    )
    rays = [p for p in data["primitives"] if p["role"] == "ray"]
    assert len(rays) == 1 and rays[0]["chief"]
    assert data["extent"][1] > 4
    bodies = [p for p in data["primitives"] if p["role"] == "lens"]
    assert bodies and np.ptp(bodies[0]["xy"][:, 1]) > 8
    assert data["extent_sources"][3] == "schematic"


def test_blocked_chief_ends_at_first_blocker(minimal_optic):
    minimal_optic.surfaces[2].aperture = RadialAperture(r_max=10, r_min=0.5)
    data = prepare_2d(
        OpticSnapshot.capture(minimal_optic),
        {"num_rays": 3, "distribution": "line_y"},
        progress,
        threading.Event(),
    )
    chief = next(p for p in data["primitives"] if p["role"] == "ray" and p["chief"])
    assert chief["xy"][-1, 0] == pytest.approx(5)
    assert np.nanmax(chief["xy"][:, 0]) < 50


def test_mirror_chief_keeps_physical_fold(set_test_backend):
    from tests.visualization.system.test_physical_ray_paths import _folded_optic

    optic = _folded_optic()
    optic.set_aperture("EPD", 2)
    optic.fields.set_type("angle")
    optic.fields.add(y=0)
    optic.wavelengths.add(0.55, is_primary=True)
    optic.surfaces.stop_index = 2
    paths = trace_layout_rays(Rays2D(optic), 3, "line_y", progress, threading.Event())
    chief = next(p["xyz"] for p in paths if p["chief"])
    np.testing.assert_allclose(chief, [[0, 0, -5], [0, 0, 0], [0, -5, 0]], atol=1e-12)


def test_native_actor_data_includes_center_as_ordinary_3d_ray(minimal_optic):
    data = prepare_3d(
        OpticSnapshot.capture(minimal_optic), {}, progress, threading.Event()
    )
    rays = [mesh for mesh in data["meshes"] if mesh["role"] == "ray"]
    assert len(rays) == 1  # chief batches with the ordinary, equal-style rays
    mesh = rays[0]
    assert mesh["linewidth"] == 1 and mesh["opacity"] == 1
    offsets, cells = mesh["cells"]["lines"]
    segments = [
        mesh["points"][cells[a:b]]
        for a, b in zip(offsets[:-1], offsets[1:], strict=True)
    ]
    assert any(np.allclose(segment[:, :2], 0) for segment in segments)


def test_count_widget_steps_normalizes_and_uses_actual_hex_totals(qapp):
    spin = RayCountSpinBox()
    spin.show()
    try:
        assert spin.minimum() == 1
        QTest.keyClick(spin, Qt.Key_Up)
        assert spin.value() == 3
        QTest.keyClick(spin, Qt.Key_Down)
        assert spin.value() == 1
        spin.lineEdit().selectAll()
        QTest.keyClicks(spin.lineEdit(), "4")
        QTest.keyClick(spin, Qt.Key_Return)
        assert spin.value() == 5
        spin.set_distribution("hexapolar")
        assert spin.value() == 7
        QTest.keyClick(spin, Qt.Key_Up)
        assert spin.value() == 19
        spin.set_distribution("random")
        QTest.keyClick(spin, Qt.Key_Down)
        assert spin.value() == 18
    finally:
        spin.close()


def test_checkbox_restyles_without_jobs_and_latest_choice_survives_result(
    qapp, minimal_optic, monkeypatch
):
    connector = OptilandConnector()
    connector.load_optic_from_object(minimal_optic)
    viewer = MatplotlibViewer(connector)
    state = SurfaceInteractionState(viewer)
    state.sync_document(connector.get_optic())
    viewer.set_interaction_state(state)
    viewer.resize(850, 550)
    viewer.show()
    jobs = connector.calculation_jobs
    try:
        wait_for(qapp, lambda: viewer.layout_job.data is not None, timeout=30)
        viewer.ax.set_xlim(0, 35)
        viewer.ax.set_ylim(-4, 4)
        viewer.preserve_zoom = True
        state.set_selected_indices([1])
        serial, token = jobs._serial, connector.document_state.edit_token
        old_data = viewer.layout_job.data
        line, normal_color, normal_width, order = viewer._chief_ray_artists[0]
        assert to_rgba(line.get_color()) == to_rgba("#2196F3")
        assert line.get_zorder() > order and line.get_linewidth() > normal_width
        trace = MagicMock(side_effect=AssertionError("Display toggle retraced"))
        monkeypatch.setattr(connector.get_optic(), "trace_generic", trace)
        viewer.chief_ray_checkbox.setChecked(False)
        qapp.processEvents()
        assert line.get_color() == normal_color and line.get_zorder() == order
        assert jobs._serial == serial and connector.document_state.edit_token == token
        assert viewer.layout_job.data is old_data
        assert viewer.ax.get_xlim() == (0, 35) and viewer.ax.get_ylim() == (-4, 4)
        assert state.selected_surfaces == (connector.get_optic().surfaces[1],)
        trace.assert_not_called()
        monkeypatch.undo()
        viewer.num_rays_spinbox.setValue(4)
        viewer.plot_optic()
        assert viewer.num_rays_spinbox.value() == 5
        viewer.chief_ray_checkbox.setChecked(True)
        wait_for(qapp, lambda: viewer.layout_job.data is not old_data, timeout=30)
        assert (
            len([p for p in viewer.layout_job.data["primitives"] if p["role"] == "ray"])
            == 5
        )
        line = viewer._chief_ray_artists[0][0]
        assert to_rgba(line.get_color()) == to_rgba("#2196F3")
        viewer.update_theme("light")
        assert to_rgba(viewer._chief_ray_artists[0][0].get_color()) == to_rgba(
            "#005FCC"
        )
        assert jobs._serial == serial + 1
        assert viewer.ax.get_xlim() == (0, 35)
    finally:
        viewer.hide()
        jobs.shutdown()
        wait_for(qapp, lambda: jobs._process is None)
        viewer.deleteLater()
