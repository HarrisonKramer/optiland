"""Converters between data formats.

This module contains classes that convert data between different formats,
including Zemax and Optiland (.json) files.

Kramer Harrison, 2024
"""

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
        """Converts the configuration of the file handler into an Optic object.

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
        """Configures the surfaces for the optic."""
        for idx, surf_data in self.data["surfaces"].items():
            self._configure_surface(idx, surf_data)
        self.optic.add_surface(index=len(self.data["surfaces"]))

    def _configure_surface(self, index, data):
        """Configures a surface for the optic."""
        coefficients = self._configure_surface_coefficients(data)
        self.optic.add_surface(
            index=index,
            surface_type=data["type"],
            radius=data["radius"],
            conic=data["conic"],
            thickness=data["thickness"],
            is_stop=data["is_stop"],
            material=data["material"],
            coefficients=coefficients,
        )

    def _configure_surface_coefficients(self, data):
        """Configures the coefficients for the surface. This is None for standard
        surfaces.
        """
        surf_type = data["type"]
        if surf_type == "standard":
            return None
        if surf_type in ["even_asphere", "odd_asphere"]:
            coefficients = []
            for k in range(8):
                coefficients.append(data[f"param_{k}"])
            return coefficients
        raise ValueError("Unsupported surface type.")

    def _configure_aperture(self):
        """Configures the aperture for the optic."""
        aperture_data = self.data["aperture"]
        ap_type, value = next(iter(aperture_data.items()))
        self.optic.set_aperture(aperture_type=ap_type, value=value)

    def _configure_fields(self):
        """Configure the fields for the optic."""
        self.optic.set_field_type(field_type=self.data["fields"]["type"])

        field_x = self.data["fields"]["x"]
        field_y = self.data["fields"]["y"]

        try:
            vig_x = self.data["fields"]["vignette_compress_x"]
            vig_y = self.data["fields"]["vignette_compress_y"]
        except KeyError:
            vig_x = [0.0] * len(field_x)
            vig_y = [0.0] * len(field_y)

        try:
            dx = self.data["fields"]["vignette_decenter_x"]
            dy = self.data["fields"]["vignette_decenter_y"]

            # TODO: Implement decentering.
            if any(dx) or any(dy):
                print("Warning: Vignette decentering is not supported.")
        except KeyError:
            pass

        for k in range(len(field_x)):
            self.optic.add_field(x=field_x[k], y=field_y[k], vx=vig_x[k], vy=vig_y[k])

    def _configure_wavelengths(self):
        """Configure the wavelengths for the optic."""
        primary_idx = self.data["wavelengths"]["primary_index"]
        for idx, value in enumerate(self.data["wavelengths"]["data"]):
            self.optic.add_wavelength(value=value, is_primary=(idx == primary_idx))
