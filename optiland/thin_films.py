import numpy as np


class ThinFilmLayer:
    def __init__(self, thickness, material):
        self.thickness = thickness
        self.material = material

    def characteristic_matrix(self, wavelength, theta, pol):
        """Calculate the characteristic matrix of the layer."""
        n = self.material.n(wavelength) + 1j * self.material.k(wavelength)
        admittance = self._admittance(pol, n)
        phase = self._phase_thickness(wavelength, theta, n)
        return np.array([[np.cos(phase), 1j / admittance * np.sin(phase)],
                         [1j * admittance * np.sin(phase), np.cos(phase)]])

    def _phase_thickness(self, wavelength, theta, n):
        """Calculate the phase thickness of the layer."""
        return 2 * np.pi * n * self.thickness * np.cos(theta) / wavelength

    def _admittance(self, pol, theta, n):
        """Calculate the characteristic admittance of the layer."""
        scale = 0.0026544187288907494
        if pol == 's':
            return scale * n * np.cos(theta)
        elif pol == 'p':
            return scale * n / np.cos(theta)


class ThinFilmStack:
    def __init__(self, incident_material, substrate_material):
        self.inc_mat = incident_material
        self.sub_mat = substrate_material
        self.layers = []

    def grow(self, layer):
        """Grow a layer on the stack."""
        self.layers.append(layer)

    def reflectance(self, wavelength, aoi, pol):
        """Calculate the reflectance of the stack."""
        pass

    def jones_matrix(self, wavelength, aoi, pol):
        """Calculate the Jones matrix of the stack."""
        pass

    def _characteristic_matrix(self, wavelength, aoi, pol):
        theta = self._compute_thetas(wavelength, aoi)
        m = np.eye(2)
        for k, layer in enumerate(self.layers):
            m_layer = layer._characteristic_matrix(wavelength, theta[k], pol)
            m = np.dot(m, m_layer)
        return m

    def _compute_thetas(self, wavelength, aoi):
        """Compute angles through all layers."""
        n0 = self.inc_mat.n(wavelength)
        n = np.array([[layer.material.n(wavelength) for layer in self.layers]])
        return np.arcsin(n0 / n * np.sin(aoi))

    def _admittance(self, pol):
        pass
