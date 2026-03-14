"""Tests for non-sequential visualization.

Covers trace_with_paths(), NSQSurfaceAdapter, NSQRays2D,
NSQSceneRenderer2D, and NSQViewer.
"""

from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import optiland.backend as be  # noqa: E402
from optiland.coordinate_system import CoordinateSystem  # noqa: E402
from optiland.geometries.plane import Plane  # noqa: E402
from optiland.materials import IdealMaterial  # noqa: E402
from optiland.nonsequential import (  # noqa: E402
    NSQSurface,
    NonSequentialScene,
    PointSource,
)
from optiland.visualization.nonsequential.adapter import NSQSurfaceAdapter  # noqa: E402
from optiland.visualization.nonsequential.rays import NSQRays2D  # noqa: E402
from optiland.visualization.nonsequential.scene_renderer import (  # noqa: E402
    NSQSceneRenderer2D,
)
from optiland.visualization.nonsequential.nsq_viewer import NSQViewer  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_flat_mirror_scene() -> tuple[NonSequentialScene, NSQSurface]:
    """Return a minimal scene: point source → flat mirror."""
    scene = NonSequentialScene(max_interactions=10)
    air = IdealMaterial(n=1.0)

    cs = CoordinateSystem(x=0, y=0, z=50)
    mirror = NSQSurface(
        geometry=Plane(cs),
        material_front=air,
        material_back=air,
        is_reflective=True,
        label="mirror",
    )
    scene.add_surface(mirror)

    source = PointSource(
        position=(0, 0, 0),
        direction=(0, 0, 1),
        half_angle=np.radians(5),
        wavelength=0.55,
    )
    scene.add_source(source)
    return scene, mirror


# ---------------------------------------------------------------------------
# trace_with_paths
# ---------------------------------------------------------------------------


class TestTraceWithPaths:
    def test_returns_one_pool_per_source(self, set_test_backend):
        scene, _ = _make_flat_mirror_scene()
        pools = scene.trace_with_paths(n_rays=20)
        assert len(pools) == len(scene._sources)

    def test_pool_has_path_history(self, set_test_backend):
        scene, _ = _make_flat_mirror_scene()
        pools = scene.trace_with_paths(n_rays=20)
        for pool in pools:
            assert pool.path_history is not None

    def test_sentinel_step_has_surface_id_minus_one(self, set_test_backend):
        scene, _ = _make_flat_mirror_scene()
        pools = scene.trace_with_paths(n_rays=20)
        for pool in pools:
            step0 = pool.path_history[0]
            sids = be.to_numpy(step0["surface_ids"])
            assert np.all(sids == -1), "Sentinel step must have surface_ids = -1"

    def test_path_history_has_at_least_two_steps_for_mirror_hit(
        self, set_test_backend
    ):
        scene, _ = _make_flat_mirror_scene()
        pools = scene.trace_with_paths(n_rays=20)
        for pool in pools:
            # Step 0 = source, ≥1 more steps from mirror hit
            assert len(pool.path_history) >= 2

    def test_record_paths_restored_on_source(self, set_test_backend):
        scene, _ = _make_flat_mirror_scene()
        source = scene._sources[0]
        original = source.record_paths
        scene.trace_with_paths(n_rays=10)
        assert source.record_paths == original

    def test_record_paths_restored_on_exception(self, set_test_backend):
        """record_paths must be restored even if tracing raises."""
        scene, _ = _make_flat_mirror_scene()
        source = scene._sources[0]
        original = source.record_paths

        # Patch tracer to raise
        import unittest.mock as mock

        with mock.patch.object(
            scene._tracer, "trace", side_effect=RuntimeError("boom")
        ):
            with pytest.raises(RuntimeError):
                scene.trace_with_paths(n_rays=10)

        assert source.record_paths == original

    def test_does_not_modify_detector_data(self, set_test_backend):
        """trace_with_paths must not accumulate data in _detector_data."""
        scene = NonSequentialScene(max_interactions=10)
        air = IdealMaterial(n=1.0)
        det_cs = CoordinateSystem(x=0, y=0, z=100)
        detector = NSQSurface(
            geometry=Plane(det_cs),
            material_front=air,
            material_back=air,
            is_detector=True,
        )
        scene.add_surface(detector)
        scene.add_source(
            PointSource(position=(0, 0, 0), direction=(0, 0, 1), half_angle=0.0)
        )

        scene.trace_with_paths(n_rays=20)
        # _detector_data should still be empty for detector (no-op dict passed)
        assert detector.surface_id in scene._detector_data
        assert scene._detector_data[detector.surface_id].n_hits == 0

    def test_source_without_record_paths_emits_warning(self, set_test_backend):
        """Sources lacking record_paths attr should produce a warning."""

        class NoRecordPathsSource:
            def generate_rays(self, n_rays):
                from optiland.nonsequential.ray_data import NSQRayPool

                return NSQRayPool(
                    x=be.zeros(n_rays),
                    y=be.zeros(n_rays),
                    z=be.zeros(n_rays),
                    L=be.zeros(n_rays),
                    M=be.zeros(n_rays),
                    N=be.ones(n_rays),
                    intensity=be.ones(n_rays),
                    wavelength=be.full(n_rays, 0.55),
                    record_paths=False,
                )

        scene = NonSequentialScene(max_interactions=2)
        scene.add_source(NoRecordPathsSource())

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pools = scene.trace_with_paths(n_rays=5)
        assert any("record_paths" in str(warning.message) for warning in w)
        assert len(pools) == 1
        assert pools[0].path_history is None


# ---------------------------------------------------------------------------
# NSQSurfaceAdapter
# ---------------------------------------------------------------------------


class TestNSQSurfaceAdapter:
    def test_geometry_forwarded(self, set_test_backend):
        air = IdealMaterial(n=1.0)
        cs = CoordinateSystem(x=0, y=0, z=10)
        surf = NSQSurface(geometry=Plane(cs), material_front=air, material_back=air)
        surf.surface_id = 0
        adapter = NSQSurfaceAdapter(surf)
        assert adapter.geometry is surf.geometry

    def test_aperture_is_none(self, set_test_backend):
        air = IdealMaterial(n=1.0)
        cs = CoordinateSystem(x=0, y=0, z=10)
        surf = NSQSurface(geometry=Plane(cs), material_front=air, material_back=air)
        surf.surface_id = 0
        adapter = NSQSurfaceAdapter(surf)
        assert adapter.aperture is None

    def test_comment_uses_label_when_set(self, set_test_backend):
        air = IdealMaterial(n=1.0)
        cs = CoordinateSystem(x=0, y=0, z=10)
        surf = NSQSurface(
            geometry=Plane(cs), material_front=air, material_back=air, label="lens"
        )
        surf.surface_id = 3
        adapter = NSQSurfaceAdapter(surf)
        assert adapter.comment == "lens"

    def test_comment_falls_back_to_surface_id(self, set_test_backend):
        air = IdealMaterial(n=1.0)
        cs = CoordinateSystem(x=0, y=0, z=10)
        surf = NSQSurface(geometry=Plane(cs), material_front=air, material_back=air)
        surf.surface_id = 7
        adapter = NSQSurfaceAdapter(surf)
        assert "7" in adapter.comment


# ---------------------------------------------------------------------------
# NSQRays2D
# ---------------------------------------------------------------------------


class TestNSQRays2D:
    def test_plot_produces_artists(self, set_test_backend):
        scene, _ = _make_flat_mirror_scene()
        pools = scene.trace_with_paths(n_rays=20)
        fig, ax = plt.subplots()
        before = len(ax.lines)
        NSQRays2D(scene).plot(ax, pools, num_rays=5)
        after = len(ax.lines)
        plt.close(fig)
        assert after > before, "NSQRays2D.plot() should add line artists"

    def test_plot_with_empty_pools_does_not_crash(self, set_test_backend):
        scene, _ = _make_flat_mirror_scene()
        fig, ax = plt.subplots()
        NSQRays2D(scene).plot(ax, [])
        plt.close(fig)

    def test_plot_skips_pool_with_none_history(self, set_test_backend):
        from optiland.nonsequential.ray_data import NSQRayPool

        scene, _ = _make_flat_mirror_scene()
        pool = NSQRayPool(
            x=be.zeros(3),
            y=be.zeros(3),
            z=be.zeros(3),
            L=be.zeros(3),
            M=be.zeros(3),
            N=be.ones(3),
            intensity=be.ones(3),
            wavelength=be.full(3, 0.55),
            record_paths=False,
        )
        assert pool.path_history is None
        fig, ax = plt.subplots()
        NSQRays2D(scene).plot(ax, [pool])  # should not crash
        plt.close(fig)


# ---------------------------------------------------------------------------
# NSQSceneRenderer2D
# ---------------------------------------------------------------------------


class TestNSQSceneRenderer2D:
    def test_plot_does_not_crash(self, set_test_backend):
        scene, _ = _make_flat_mirror_scene()
        pools = scene.trace_with_paths(n_rays=20)
        fig, ax = plt.subplots()
        NSQSceneRenderer2D(scene, pools).plot(ax)
        plt.close(fig)

    def test_unhit_surface_uses_default_extent(self, set_test_backend):
        """A surface never hit by any ray gets default_extent as fallback."""
        scene = NonSequentialScene(max_interactions=5)
        air = IdealMaterial(n=1.0)

        cs = CoordinateSystem(x=0, y=0, z=50)
        surf = NSQSurface(
            geometry=Plane(cs),
            material_front=air,
            material_back=air,
            label="surf",
        )
        scene.add_surface(surf)
        scene.add_source(
            PointSource(position=(0, 0, 0), direction=(0, 0, 1), half_angle=0.0)
        )

        # Pass empty pools — no rays → surface has no extent in map
        renderer = NSQSceneRenderer2D(scene, pools=[], default_extent=3.14)
        extent_map = renderer._build_extent_map()
        assert surf.surface_id not in extent_map

        # Plot should not crash (uses default_extent as fallback)
        fig, ax = plt.subplots()
        renderer.plot(ax)
        plt.close(fig)


# ---------------------------------------------------------------------------
# NSQViewer
# ---------------------------------------------------------------------------


class TestNSQViewer:
    def test_view_returns_fig_and_ax(self, set_test_backend):
        scene, _ = _make_flat_mirror_scene()
        viewer = NSQViewer(scene, default_extent=10.0)
        fig, ax = viewer.view(n_rays=20, num_display_rays=5)
        plt.close(fig)
        assert hasattr(fig, "savefig"), "Expected a matplotlib Figure"
        assert hasattr(ax, "plot"), "Expected a matplotlib Axes"

    def test_view_with_existing_ax(self, set_test_backend):
        scene, _ = _make_flat_mirror_scene()
        fig_pre, ax_pre = plt.subplots()
        viewer = NSQViewer(scene, default_extent=10.0)
        fig_out, ax_out = viewer.view(n_rays=20, num_display_rays=5, ax=ax_pre)
        plt.close(fig_pre)
        assert ax_out is ax_pre
        assert fig_out is fig_pre

    def test_view_xz_projection(self, set_test_backend):
        scene, _ = _make_flat_mirror_scene()
        viewer = NSQViewer(scene, default_extent=10.0)
        fig, ax = viewer.view(n_rays=10, num_display_rays=5, projection="XZ")
        plt.close(fig)

    def test_view_xy_projection(self, set_test_backend):
        scene, _ = _make_flat_mirror_scene()
        viewer = NSQViewer(scene, default_extent=10.0)
        fig, ax = viewer.view(n_rays=10, num_display_rays=5, projection="XY")
        plt.close(fig)

    def test_view_multiple_sources(self, set_test_backend):
        scene, _ = _make_flat_mirror_scene()
        # Add a second source
        scene.add_source(
            PointSource(
                position=(0, 1, 0),
                direction=(0, 0, 1),
                half_angle=np.radians(3),
            )
        )
        viewer = NSQViewer(scene, default_extent=10.0)
        fig, ax = viewer.view(n_rays=15, num_display_rays=5)
        plt.close(fig)
