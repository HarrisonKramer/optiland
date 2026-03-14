"""Tests for standalone refract() and reflect() physics kernels."""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.physics.interaction import reflect, refract
from tests.utils import assert_allclose


class TestRefract:
    """Tests for the refract() function."""

    def test_normal_incidence(self, set_test_backend):
        """Ray along surface normal should pass straight through."""
        L = be.array([0.0])
        M = be.array([0.0])
        N = be.array([1.0])
        nx = be.array([0.0])
        ny = be.array([0.0])
        nz = be.array([1.0])
        n1, n2 = 1.0, 1.5

        L_new, M_new, N_new, tir = refract(L, M, N, nx, ny, nz, n1, n2)

        assert_allclose(L_new, [0.0], atol=1e-12)
        assert_allclose(M_new, [0.0], atol=1e-12)
        assert_allclose(N_new, [1.0], atol=1e-12)
        assert not be.any(tir)

    def test_snells_law_45_degrees(self, set_test_backend):
        """Verify Snell's law at 45 degrees incidence."""
        theta_i = np.radians(45)
        L = be.array([np.sin(theta_i)])
        M = be.array([0.0])
        N = be.array([np.cos(theta_i)])
        nx = be.array([0.0])
        ny = be.array([0.0])
        nz = be.array([1.0])
        n1, n2 = 1.0, 1.5

        L_new, M_new, N_new, tir = refract(L, M, N, nx, ny, nz, n1, n2)

        # sin(theta_t) = (n1/n2) * sin(theta_i)
        expected_sin_t = (n1 / n2) * np.sin(theta_i)
        actual_sin_t = float(be.to_numpy(L_new)[0])
        assert_allclose(actual_sin_t, expected_sin_t, atol=1e-10)
        assert not be.any(tir)

    def test_total_internal_reflection(self, set_test_backend):
        """TIR should be flagged when angle exceeds critical angle."""
        # Critical angle for n1=1.5, n2=1.0: arcsin(1/1.5) ≈ 41.8°
        theta_i = np.radians(50)  # Beyond critical angle
        L = be.array([np.sin(theta_i)])
        M = be.array([0.0])
        N = be.array([np.cos(theta_i)])
        nx = be.array([0.0])
        ny = be.array([0.0])
        nz = be.array([1.0])
        n1, n2 = 1.5, 1.0

        _, _, _, tir = refract(L, M, N, nx, ny, nz, n1, n2)

        assert be.all(tir)

    def test_no_tir_below_critical_angle(self, set_test_backend):
        """No TIR below critical angle."""
        theta_i = np.radians(30)  # Below critical angle
        L = be.array([np.sin(theta_i)])
        M = be.array([0.0])
        N = be.array([np.cos(theta_i)])
        nx = be.array([0.0])
        ny = be.array([0.0])
        nz = be.array([1.0])
        n1, n2 = 1.5, 1.0

        _, _, _, tir = refract(L, M, N, nx, ny, nz, n1, n2)

        assert not be.any(tir)

    def test_vectorized(self, set_test_backend):
        """Multiple rays at different angles should be handled correctly."""
        angles = be.array([0.0, np.radians(30), np.radians(60)])
        L = be.sin(angles)
        M = be.zeros(3)
        N = be.cos(angles)
        nx = be.zeros(3)
        ny = be.zeros(3)
        nz = be.ones(3)

        L_new, M_new, N_new, tir = refract(L, M, N, nx, ny, nz, 1.0, 1.5)

        # All should refract (no TIR for air→glass)
        assert not be.any(tir)
        # Direction should be normalized (approximately)
        mag = be.sqrt(L_new**2 + M_new**2 + N_new**2)
        assert_allclose(mag, be.ones(3), atol=1e-10)


class TestReflect:
    """Tests for the reflect() function."""

    def test_normal_incidence(self, set_test_backend):
        """Ray along normal should reverse direction."""
        L = be.array([0.0])
        M = be.array([0.0])
        N = be.array([1.0])
        nx = be.array([0.0])
        ny = be.array([0.0])
        nz = be.array([1.0])

        L_new, M_new, N_new = reflect(L, M, N, nx, ny, nz)

        assert_allclose(L_new, [0.0], atol=1e-12)
        assert_allclose(M_new, [0.0], atol=1e-12)
        assert_allclose(N_new, [-1.0], atol=1e-12)

    def test_45_degree_mirror(self, set_test_backend):
        """Ray at 45° on flat mirror should deflect 90°."""
        theta = np.radians(45)
        L = be.array([np.sin(theta)])
        M = be.array([0.0])
        N = be.array([np.cos(theta)])
        nx = be.array([0.0])
        ny = be.array([0.0])
        nz = be.array([1.0])

        L_new, M_new, N_new = reflect(L, M, N, nx, ny, nz)

        # Reflected ray should have same L but reversed N
        assert_allclose(L_new, [np.sin(theta)], atol=1e-12)
        assert_allclose(N_new, [-np.cos(theta)], atol=1e-12)

    def test_preserves_normalization(self, set_test_backend):
        """Reflected direction should remain normalized."""
        theta = np.radians(30)
        L = be.array([np.sin(theta)])
        M = be.array([0.0])
        N = be.array([np.cos(theta)])
        nx = be.array([0.0])
        ny = be.array([0.0])
        nz = be.array([1.0])

        L_new, M_new, N_new = reflect(L, M, N, nx, ny, nz)

        mag = be.sqrt(L_new**2 + M_new**2 + N_new**2)
        assert_allclose(mag, [1.0], atol=1e-12)

    def test_vectorized(self, set_test_backend):
        """Multiple rays reflected simultaneously."""
        n = 5
        L = be.zeros(n)
        M = be.zeros(n)
        N = be.ones(n)
        nx = be.zeros(n)
        ny = be.zeros(n)
        nz = be.ones(n)

        L_new, M_new, N_new = reflect(L, M, N, nx, ny, nz)

        assert_allclose(N_new, -be.ones(n), atol=1e-12)


class TestRealRaysRegression:
    """Verify RealRays.refract/reflect still produce correct results."""

    def test_realrays_refract_matches_kernel(self, set_test_backend):
        """RealRays.refract should produce same results as kernel."""
        from optiland.rays import RealRays

        theta = np.radians(30)
        x = be.array([0.0])
        y = be.array([0.0])
        z = be.array([0.0])
        L = be.array([np.sin(theta)])
        M = be.array([0.0])
        N = be.array([np.cos(theta)])

        rays = RealRays(x, y, z, L, M, N, be.ones(1), be.ones(1))

        nx = be.array([0.0])
        ny = be.array([0.0])
        nz = be.array([1.0])

        rays.refract(nx, ny, nz, 1.0, 1.5)

        # Compare with kernel
        L_k, M_k, N_k, _ = refract(
            be.array([np.sin(theta)]),
            be.array([0.0]),
            be.array([np.cos(theta)]),
            nx, ny, nz, 1.0, 1.5,
        )

        assert_allclose(rays.L, L_k, atol=1e-12)
        assert_allclose(rays.M, M_k, atol=1e-12)
        assert_allclose(rays.N, N_k, atol=1e-12)

    def test_realrays_reflect_matches_kernel(self, set_test_backend):
        """RealRays.reflect should produce same results as kernel."""
        from optiland.rays import RealRays

        theta = np.radians(30)
        x = be.array([0.0])
        y = be.array([0.0])
        z = be.array([0.0])
        L = be.array([np.sin(theta)])
        M = be.array([0.0])
        N = be.array([np.cos(theta)])

        rays = RealRays(x, y, z, L, M, N, be.ones(1), be.ones(1))

        nx = be.array([0.0])
        ny = be.array([0.0])
        nz = be.array([1.0])

        rays.reflect(nx, ny, nz)

        L_k, M_k, N_k = reflect(
            be.array([np.sin(theta)]),
            be.array([0.0]),
            be.array([np.cos(theta)]),
            nx, ny, nz,
        )

        assert_allclose(rays.L, L_k, atol=1e-12)
        assert_allclose(rays.M, M_k, atol=1e-12)
        assert_allclose(rays.N, N_k, atol=1e-12)
