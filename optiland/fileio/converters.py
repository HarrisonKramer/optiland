from optiland.optic import Optic


class ZemaxToOpticConverter:
    """Converts Zemax data into an Optic object.

    Args:
        zemax_data (dict): The Zemax data to be converted. This is the data
            extracted by the ZemaxFileReader.

    Attributes:
        data (dict): The Zemax data to be converted.
        optic (Optic): The Optic object based on the Zemax data.

    Methods:
        convert(): Converts the configuration of the file handler into an
            Optic object.
    """

    def __init__(self, zemax_data):
        self.data = zemax_data
        self.optic = None

    def convert(self):
        """
        Converts the configuration of the file handler into an Optic object.

        Returns:
            Optic: The Optic object based on the Zemax data.
        """
        self.optic = Optic()
        self._configure_surfaces()
        self._configure_aperture()
        self._configure_fields()
        self._configure_wavelengths()
        return self.optic

    def _configure_surfaces(self):
        """
        Configures the surfaces for the optic.
        """
        for idx, surf_data in self.data['surfaces'].items():
            self._configure_surface(idx, surf_data)
        self.optic.add_surface(index=len(self.data['surfaces']))

    def _configure_surface(self, index, data):
        """
        Configures a surface for the optic.
        """
        coefficients = self._configure_surface_coefficients(data)
        self.optic.add_surface(index=index,
                               surface_type=data['type'],
                               radius=data['radius'],
                               conic=data['conic'],
                               thickness=data['thickness'],
                               is_stop=data['is_stop'],
                               material=data['material'],
                               coefficients=coefficients)

    def _configure_surface_coefficients(self, data):
        """
        Configures the coefficients for the surface. This is None for standard
        surfaces.
        """
        surf_type = data['type']
        if surf_type == 'standard':
            return None
        elif surf_type == 'even_asphere':
            coefficients = []
            for k in range(8):
                coefficients.append(data[f'param_{k}'])
            return coefficients
        else:
            raise ValueError('Unsupported surface type.')

    def _configure_aperture(self):
        """
        Configures the aperture for the optic.
        """
        aperture_data = self.data['aperture']
        ap_type, value = next(iter(aperture_data.items()))
        self.optic.set_aperture(aperture_type=ap_type, value=value)

    def _configure_fields(self):
        """
        Configure the fields for the optic.
        """
        self.optic.set_field_type(field_type=self.data['fields']['type'])
        for fx, fy in zip(self.data['fields']['x'], self.data['fields']['y']):
            self.optic.add_field(x=fx, y=fy)

    def _configure_wavelengths(self):
        """
        Configure the wavelengths for the optic.
        """
        primary_idx = self.data['wavelengths']['primary_index']
        for idx, value in enumerate(self.data['wavelengths']['data']):
            self.optic.add_wavelength(value=value,
                                      is_primary=(idx == primary_idx))