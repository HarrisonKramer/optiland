import numpy as np
from optiland.rays import RealRays
from optiland import jones, materials


class TestJonesFresnel:
    def test_normal_incidence(self):
        rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)

        material_pre = materials.IdealMaterial(n=1.0)
        material_post = materials.IdealMaterial(n=1.5)
        jones_fresnel = jones.JonesFresnel(material_pre, material_post)

        # Test reflection calculation
        aoi = np.array([0.0])
        jones_matrix = jones_fresnel.calculate_matrix(rays, reflect=True,
                                                      aoi=aoi)

        assert jones_matrix.shape == (1, 3, 3)
        assert jones_matrix[0, 0, 0] == -0.2
        assert jones_matrix[0, 1, 1] == -0.2
        assert jones_matrix[0, 2, 2] == -1.0

        # Test transmission calculation
        jones_matrix = jones_fresnel.calculate_matrix(rays, reflect=False,
                                                      aoi=aoi)
        assert jones_matrix.shape == (1, 3, 3)
        assert jones_matrix[0, 0, 0] == 0.8
        assert jones_matrix[0, 1, 1] == 0.8
        assert jones_matrix[0, 2, 2] == 1.0

    def test_non_normal_incidence(self):
        L = 0.1
        M = -0.25
        N = np.sqrt(1 - L**2 - M**2)
        rays = RealRays(1.0, 2.0, 3.0, L, M, N, 1.0, 1.0)

        material_pre = materials.IdealMaterial(n=1.0, k=0.0)
        material_post = materials.IdealMaterial(n=1.5, k=0.1)
        jones_fresnel = jones.JonesFresnel(material_pre, material_post)

        # Test reflection calculation
        aoi = np.array([0.2])  # use arbitrary angle of incidence
        jones_matrix = jones_fresnel.calculate_matrix(rays, reflect=True,
                                                      aoi=aoi)

        assert jones_matrix.shape == (1, 3, 3)
        assert jones_matrix[0, 0, 0] == -0.20541108217641596
        assert jones_matrix[0, 1, 1] == -0.19457669033430527
        assert jones_matrix[0, 2, 2] == -1.0

        # Test transmission calculation
        jones_matrix = jones_fresnel.calculate_matrix(rays, reflect=False,
                                                      aoi=aoi)
        assert jones_matrix.shape == (1, 3, 3)
        assert jones_matrix[0, 0, 0] == 0.7945889178235841
        assert jones_matrix[0, 1, 1] == 0.7963844602228702
        assert jones_matrix[0, 2, 2] == 1.0


def test_jones_polarizer_h():
    rays = RealRays([1.0, 2.0], [3.0, 0.0], [0.0, 1.0], [0.0, 0.0],
                    [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0])

    jones_polarizer = jones.JonesPolarizerH()
    jones_matrix = jones_polarizer.calculate_matrix(rays)

    assert jones_matrix.shape == (2, 3, 3)
    assert jones_matrix[0, 0, 0] == 1.0
    assert jones_matrix[0, 1, 1] == 0.0
    assert jones_matrix[0, 2, 2] == 1.0
    assert jones_matrix[1, 0, 0] == 1.0
    assert jones_matrix[1, 1, 1] == 0.0
    assert jones_matrix[1, 2, 2] == 1.0


def test_jones_polarizer_v():
    rays = RealRays([1.0, 2.0], [3.0, 0.0], [0.0, 1.0], [0.0, 0.0],
                    [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0])

    jones_polarizer = jones.JonesPolarizerV()
    jones_matrix = jones_polarizer.calculate_matrix(rays)

    assert jones_matrix.shape == (2, 3, 3)
    assert jones_matrix[0, 0, 0] == 0.0
    assert jones_matrix[0, 1, 1] == 1.0
    assert jones_matrix[0, 2, 2] == 1.0
    assert jones_matrix[1, 0, 0] == 0.0
    assert jones_matrix[1, 1, 1] == 1.0
    assert jones_matrix[1, 2, 2] == 1.0
