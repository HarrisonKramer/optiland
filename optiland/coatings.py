import numpy as np
from optiland.rays import RealRays


class BaseCoating:

    # TODO - finalize and verify correctness
    def interact(self, rays: RealRays, aoi: np.ndarray,
                 L0: np.ndarray, M0: np.ndarray, N0: np.ndarray):
        # merge k-vector components into matrix for speed
        k0 = np.array([L0, M0, N0]).T
        k1 = np.array([rays.L, rays.M, rays.N]).T

        # find s-component
        s = np.cross(k0, k1)
        mag = np.linalg.norm(s, axis=1)
        s /= mag[:, np.newaxis]

        # find p-component pre and post surface
        p0 = np.cross(k0, s)
        p1 = np.cross(k1, s)

        # othogonal transformation matrices
        o_in = np.stack((s, p0, k0), axis=1)
        o_out = np.stack((s, p1, k1), axis=2)

        # get jones matrix for each ray
        j = self._jones_matrix(rays)

        # compute polarization matrix for surface
        p = np.einsum('nij,njk,nkl->nil', o_out, j, o_in)

        # singular values of p represent rs and rp transmission on this surface
        singular_values = np.linalg.svd(p, compute_uv=False)

        # scale ray energies
        energy_scale = 0.5 * (np.abs(singular_values[1])**2 +
                              np.abs(singular_values[2])**2)
        rays.e *= energy_scale

        # update polarization matrices of rays
        rays.p = np.matmul(p, rays.p)

    def _jones_matrix(self, rays: RealRays):
        return np.tile(np.eye(2), (rays.x.size, 1, 1))


class Fresnel(BaseCoating):

    def __init__(self, material_pre, material_post):
        self.material_pre = material_pre
        self.material_post = material_post

    def _jones_matrix(self, rays: RealRays):
        pass
