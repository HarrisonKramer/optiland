import numpy as np
from optiland import wavefront
from optiland.distribution import GaussianQuadrature


class RayOperand:
    """
    A class that provides static methods for performing ray tracing
        calculations on an optic.

    Methods:
        x_intercept: Calculates the x-coordinate of the intercept point on a
            specific surface.
        y_intercept: Calculates the y-coordinate of the intercept point on a
            specific surface.
        z_intercept: Calculates the z-coordinate of the intercept point on a
            specific surface.
        L: Calculates the direction cosine L of the ray on a specific surface.
        M: Calculates the direction cosine M of the ray on a specific surface.
        N: Calculates the direction cosine N of the ray on a specific surface.
        rms_spot_size: Calculates the root mean square (RMS) spot size on a
            specific surface.
        OPD_difference: Calculates the optical path difference (OPD)
            difference for a given ray distribution.
    """

    @staticmethod
    def x_intercept(optic, surface_number, Hx, Hy, Px, Py, wavelength):
        """
        Calculates the x-coordinate of the intercept point on a specific
            surface.

        Parameters:
            optic: The optic object.
            surface_number: The number of the surface.
            Hx: The x-coordinate of the incoming ray direction.
            Hy: The y-coordinate of the incoming ray direction.
            Px: The x-coordinate of the point on the surface.
            Py: The y-coordinate of the point on the surface.
            wavelength: The wavelength of the ray.

        Returns:
            The x-coordinate of the intercept point.
        """
        optic.trace_generic(Hx, Hy, Px, Py, wavelength)
        return optic.surface_group.x[surface_number, 0]

    @staticmethod
    def y_intercept(optic, surface_number, Hx, Hy, Px, Py, wavelength):
        """
        Calculates the y-coordinate of the intercept point on a specific
            surface.

        Parameters:
            optic: The optic object.
            surface_number: The number of the surface.
            Hx: The x-coordinate of the incoming ray direction.
            Hy: The y-coordinate of the incoming ray direction.
            Px: The x-coordinate of the point on the surface.
            Py: The y-coordinate of the point on the surface.
            wavelength: The wavelength of the ray.

        Returns:
            The y-coordinate of the intercept point.
        """
        optic.trace_generic(Hx, Hy, Px, Py, wavelength)
        return optic.surface_group.y[surface_number, 0]

    @staticmethod
    def z_intercept(optic, surface_number, Hx, Hy, Px, Py, wavelength):
        """
        Calculates the z-coordinate of the intercept point on a specific
            surface.

        Parameters:
            optic: The optic object.
            surface_number: The number of the surface.
            Hx: The x-coordinate of the incoming ray direction.
            Hy: The y-coordinate of the incoming ray direction.
            Px: The x-coordinate of the point on the surface.
            Py: The y-coordinate of the point on the surface.
            wavelength: The wavelength of the ray.

        Returns:
            The z-coordinate of the intercept point.
        """
        optic.trace_generic(Hx, Hy, Px, Py, wavelength)
        return optic.surface_group.z[surface_number, 0]

    @staticmethod
    def L(optic, surface_number, Hx, Hy, Px, Py, wavelength):
        """
        Calculates the direction cosine L of the ray on a specific surface.

        Parameters:
            optic: The optic object.
            surface_number: The number of the surface.
            Hx: The x-coordinate of the incoming ray direction.
            Hy: The y-coordinate of the incoming ray direction.
            Px: The x-coordinate of the point on the surface.
            Py: The y-coordinate of the point on the surface.
            wavelength: The wavelength of the ray.

        Returns:
            The direction cosine L of the ray.
        """
        optic.trace_generic(Hx, Hy, Px, Py, wavelength)
        return optic.surface_group.L[surface_number, 0]

    @staticmethod
    def M(optic, surface_number, Hx, Hy, Px, Py, wavelength):
        """
        Calculates the direction cosine M of the ray on a specific surface.

        Parameters:
            optic: The optic object.
            surface_number: The number of the surface.
            Hx: The x-coordinate of the incoming ray direction.
            Hy: The y-coordinate of the incoming ray direction.
            Px: The x-coordinate of the point on the surface.
            Py: The y-coordinate of the point on the surface.
            wavelength: The wavelength of the ray.

        Returns:
            The direction cosine M of the ray.
        """
        optic.trace_generic(Hx, Hy, Px, Py, wavelength)
        return optic.surface_group.M[surface_number, 0]

    @staticmethod
    def N(optic, surface_number, Hx, Hy, Px, Py, wavelength):
        """
        Calculates the direction cosine N of the ray on a specific surface.

        Parameters:
            optic: The optic object.
            surface_number: The number of the surface.
            Hx: The x-coordinate of the incoming ray direction.
            Hy: The y-coordinate of the incoming ray direction.
            Px: The x-coordinate of the point on the surface.
            Py: The y-coordinate of the point on the surface.
            wavelength: The wavelength of the ray.

        Returns:
            The direction cosine N of the ray.
        """
        optic.trace_generic(Hx, Hy, Px, Py, wavelength)
        return optic.surface_group.N[surface_number, 0]

    @staticmethod
    def rms_spot_size(optic, surface_number, Hx, Hy, num_rays, wavelength,
                      distribution='hexapolar'):
        """
        Calculates the root mean square (RMS) spot size on a specific surface.

        Parameters:
            optic: The optic object.
            surface_number: The number of the surface.
            Hx: The x-coordinate of the incoming ray direction.
            Hy: The y-coordinate of the incoming ray direction.
            num_rays: The number of rays to trace.
            wavelength: The wavelength of the rays.
            distribution: The distribution of the rays. Default is 'hexapolar'.

        Returns:
            The RMS spot size on the specified surface.
        """
        if wavelength == 'all':
            x = []
            y = []
            for wave in optic.wavelengths.get_wavelengths():
                optic.trace(Hx, Hy, wave, num_rays, distribution)
                x.append(optic.surface_group.x[surface_number, :].flatten())
                y.append(optic.surface_group.y[surface_number, :].flatten())
            wave_idx = optic.wavelengths.primary_index
            mean_x = np.mean(x[wave_idx])
            mean_y = np.mean(y[wave_idx])
            r2 = [(x[i] - mean_x)**2 + (y[i] - mean_y)**2
                  for i in range(len(x))]
            return np.sqrt(np.mean(np.concatenate(r2)))
        else:
            optic.trace(Hx, Hy, wavelength, num_rays, distribution)
            x = optic.surface_group.x[surface_number, :].flatten()
            y = optic.surface_group.y[surface_number, :].flatten()
            r2 = (x - np.mean(x))**2 + (y - np.mean(y))**2
            return np.sqrt(np.mean(r2))

    @staticmethod
    def OPD_difference(optic, Hx, Hy, num_rays, wavelength,
                       distribution='gaussian_quad'):
        """
        Calculates the mean optical path difference (OPD) difference for a
            given ray distribution.

        Parameters:
            optic: The optic object.
            Hx: The x-coordinate of the incoming ray direction.
            Hy: The y-coordinate of the incoming ray direction.
            num_rays: The number of rays to trace.
            wavelength: The wavelength of the rays.
            distribution: The distribution of the rays.
                Default is 'gaussian_quad'.

        Returns:
            The OPD difference for the given ray distribution.
        """
        weights = 1.0

        if distribution == 'gaussian_quad':
            if Hx == Hy == 0:
                distribution = GaussianQuadrature(is_symmetric=True)
                weights = distribution.get_weights(num_rays)
            else:
                distribution = GaussianQuadrature(is_symmetric=False)
                weights = np.repeat(distribution.get_weights(num_rays), 3)

            distribution.generate_points(num_rings=num_rays)

        wf = wavefront.Wavefront(optic, [(Hx, Hy)], [wavelength], num_rays,
                                 distribution)
        delta = (wf.data[0][0][0] - np.mean(wf.data[0][0][0])) * weights
        return np.mean(np.abs(delta))