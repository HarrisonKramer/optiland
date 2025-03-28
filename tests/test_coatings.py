from copy import deepcopy

import numpy as np
import pytest

from optiland import coatings, materials, rays


@pytest.fixture
def rays_parallel():
    num = 10
    x = np.linspace(0, 10, num)
    y = np.linspace(10, 20, num)
    z = np.linspace(-5, 10, num)
    L = np.zeros(num)
    M = np.zeros(num)
    N = np.ones(num)
    intensity = np.ones(num)
    wavelength = np.ones(num)

    r = rays.RealRays(x, y, z, L, M, N, intensity, wavelength)
    r.L0 = r.L
    r.M0 = r.M
    r.N0 = r.N
    return r


@pytest.fixture
def rays_parallel_polarized():
    num = 10
    x = np.linspace(0, 10, num)
    y = np.linspace(10, 20, num)
    z = np.linspace(-5, 10, num)
    L = np.zeros(num)
    M = np.zeros(num)
    N = np.ones(num)
    intensity = np.ones(num)
    wavelength = np.ones(num)

    r = rays.PolarizedRays(x, y, z, L, M, N, intensity, wavelength)
    r.L0 = r.L
    r.M0 = r.M
    r.N0 = r.N
    return r


@pytest.fixture
def rays_non_parallel():
    num = 10
    x = np.linspace(0, 10, num)
    y = np.linspace(10, 20, num)
    z = np.linspace(-5, 10, num)
    L = np.linspace(0, 0.5, num)
    M = np.linspace(-0.3, 0.4, num)
    N = np.sqrt(1 - L**2 - M**2)
    intensity = np.ones(num)
    wavelength = np.ones(num)

    r = rays.RealRays(x, y, z, L, M, N, intensity, wavelength)
    r.L0 = r.L
    r.M0 = r.M
    r.N0 = r.N
    return r


class TestSimpleCoating:
    def test_interact_reflect(self, rays_parallel):
        coating = coatings.SimpleCoating(transmittance=0.8, reflectance=0.1)

        rays_before = deepcopy(rays_parallel)
        rays_after = coating.interact(rays_parallel, reflect=True)

        assert np.all(rays_after.x == rays_before.x)
        assert np.all(rays_after.y == rays_before.y)
        assert np.all(rays_after.z == rays_before.z)
        assert np.all(rays_after.L == rays_before.L)
        assert np.all(rays_after.M == rays_before.M)
        assert np.all(rays_after.N == rays_before.N)
        assert np.all(rays_after.i == 0.1 * rays_before.i)
        assert np.all(rays_after.w == rays_before.w)

    def test_interact_transmit(self, rays_parallel):
        coating = coatings.SimpleCoating(transmittance=0.3, reflectance=0.5)

        rays_before = deepcopy(rays_parallel)
        rays_after = coating.interact(rays_parallel, reflect=False)

        assert np.all(rays_after.x == rays_before.x)
        assert np.all(rays_after.y == rays_before.y)
        assert np.all(rays_after.z == rays_before.z)
        assert np.all(rays_after.L == rays_before.L)
        assert np.all(rays_after.M == rays_before.M)
        assert np.all(rays_after.N == rays_before.N)
        assert np.all(rays_after.i == 0.3 * rays_before.i)
        assert np.all(rays_after.w == rays_before.w)

    def test_compute_aoi(self, rays_parallel):
        coating = coatings.SimpleCoating(transmittance=0.3, reflectance=0.5)

        nx = np.zeros_like(rays_parallel.x)
        ny = np.zeros_like(rays_parallel.y)
        nz = np.ones_like(rays_parallel.z)

        aoi = coating._compute_aoi(rays_parallel, nx, ny, nz)

        assert np.all(aoi == 0)

    def test_to_dict(self):
        coating = coatings.SimpleCoating(transmittance=0.3, reflectance=0.5)
        assert coating.to_dict() == {
            "type": "SimpleCoating",
            "transmittance": 0.3,
            "reflectance": 0.5,
        }


class TestFresnelCoating:
    def test_reflect(self, rays_parallel_polarized):
        mat1 = materials.IdealMaterial(n=1.0)
        mat2 = materials.IdealMaterial(n=1.5)
        coating = coatings.FresnelCoating(mat1, mat2)
        state = rays.PolarizationState(is_polarized=False)

        nx = np.zeros_like(rays_parallel_polarized.x)
        ny = np.zeros_like(rays_parallel_polarized.y)
        nz = np.ones_like(rays_parallel_polarized.z)

        rays_before = deepcopy(rays_parallel_polarized)
        rays_after = coating.reflect(rays_parallel_polarized, nx, ny, nz)
        rays_after.update_intensity(state)

        R = ((1.5 - 1.0) / (1.5 + 1.0)) ** 2
        assert np.all(rays_after.i == R * rays_before.i)

    def test_transmit(self, rays_parallel_polarized):
        mat1 = materials.IdealMaterial(n=1.0)
        mat2 = materials.IdealMaterial(n=1.5)
        coating = coatings.FresnelCoating(mat1, mat2)
        state = rays.PolarizationState(is_polarized=False)

        nx = np.zeros_like(rays_parallel_polarized.x)
        ny = np.zeros_like(rays_parallel_polarized.y)
        nz = np.ones_like(rays_parallel_polarized.z)

        rays_before = deepcopy(rays_parallel_polarized)
        rays_after = coating.transmit(rays_parallel_polarized, nx, ny, nz)
        rays_after.update_intensity(state)

        R = ((1.5 - 1.0) / (1.5 + 1.0)) ** 2
        assert np.allclose(rays_after.i * 1.5, (1 - R) * rays_before.i, atol=1e-9)

    def test_to_dict(self):
        mat1 = materials.IdealMaterial(n=1.0)
        mat2 = materials.IdealMaterial(n=1.5)
        coating = coatings.FresnelCoating(mat1, mat2)
        assert coating.to_dict() == {
            "type": "FresnelCoating",
            "material_pre": mat1.to_dict(),
            "material_post": mat2.to_dict(),
        }

    def test_from_dict(self):
        mat1 = materials.IdealMaterial(n=1.0)
        mat2 = materials.IdealMaterial(n=1.5)
        coating = coatings.FresnelCoating(mat1, mat2)
        coating_dict = coating.to_dict()
        coating2 = coatings.FresnelCoating.from_dict(coating_dict)
        assert coating2.to_dict() == coating.to_dict()
