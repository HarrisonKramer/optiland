"""Converters between data formats.

This module contains classes that convert data between different formats,
including Zemax and Optiland (.json) files.

Kramer Harrison, 2024
"""

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.optic import Optic


class ZemaxToOpticConverter:
    """Converts Zemax data into an Optic object.

    Args:
        zemax_data (dict): The Zemax data to be converted. This is typically
            the data extracted by `ZemaxFileReader`.

    Attributes:
        data (dict): The Zemax data to be converted.
        optic (Optic): The Optic object based on the Zemax data.
        current_cs (CoordinateSystem): Running, cumulative coordinate system
            that tracks the decentres / tilts defined by any
            preceding coordinate_break surface in the Zemax file.

    Methods:
        convert(): Converts the configuration of the file handler into an
            Optic object.

    """

    def __init__(self, zemax_data: dict):
        self.data = zemax_data
        self.optic = None
        self.current_cs = CoordinateSystem()

    def convert(self):
        """Converts the configuration of the file handler into an Optic object.

        Returns:
            Optic: The configured `Optic` object.

        """
        self.optic = Optic()
        self._configure_surfaces()
        self._configure_aperture()
        self._configure_fields()
        self._configure_wavelengths()
        return self.optic

    def _configure_surfaces(self):
        """Configures the surfaces for the optic."""
        has_cb = any(
            sd.get("type") == "coordinate_break"
            for sd in self.data["surfaces"].values()
        )

        if not has_cb:
            for idx, surf_data in self.data["surfaces"].items():
                self._configure_surface(idx, surf_data)
            self.optic.add_surface(index=len(self.data["surfaces"]))
            return

        # in case there are "Coordinate Break" surfaces
        surf_idx = 0

        for idx in sorted(self.data["surfaces"].keys(), key=int):
            surf = self.data["surfaces"][idx]

            # CB: we update the CS only, no added surface
            if surf.get("type") == "coordinate_break":
                dx = float(surf.get("param_0", 0.0))
                dy = float(surf.get("param_1", 0.0))
                dz = float(surf.get("thickness", 0.0))  # CB 'thickness'
                rx = be.deg2rad(surf.get("param_2", 0.0))
                ry = be.deg2rad(surf.get("param_3", 0.0))
                rz = be.deg2rad(surf.get("param_4", 0.0))
                # there is another param: order. implement later

                # chain a new cs
                # first apply rotations and translations
                cs_rot_decs = CoordinateSystem(
                    x=dx, y=dy, z=0.0, rx=rx, ry=ry, rz=rz, reference_cs=self.current_cs
                )
                # then apply the coordinate break's thickness as a translation
                # along the Z-axis of the new cs.
                self.current_cs = CoordinateSystem(
                    x=0.0, y=0.0, z=dz, reference_cs=cs_rot_decs
                )

                continue

            # now, the usual surfaces from the file
            # transform into global CS, then append to optic
            translation, _ = self.current_cs.get_effective_transform()
            rx_, ry_, rz_ = self.current_cs.get_effective_rotation_euler()
            coeffs = self._configure_surface_coefficients(surf)
            thickness = surf.get("thickness", 0.0)

            # special care now, we ramify. for object surface,
            # if DISZ == inf, then keep infinity
            if be.isinf(float(thickness)):
                self.optic.add_surface(
                    index=surf_idx,
                    surface_type=surf["type"],
                    radius=surf.get("radius"),
                    conic=surf.get("conic"),
                    thickness=thickness,
                    is_stop=surf.get("is_stop", False),
                    material=surf.get("material"),
                    coefficients=coeffs,
                    rx=float(rx_),
                    ry=float(ry_),
                    rz=float(rz_),
                )
            else:  # normal surface, we pass no thickness argument
                self.optic.add_surface(
                    index=surf_idx,
                    surface_type=surf["type"],
                    radius=surf.get("radius"),
                    conic=surf.get("conic"),
                    is_stop=surf.get("is_stop", False),
                    material=surf.get("material"),
                    coefficients=coeffs,
                    x=float(translation[0]),
                    y=float(translation[1]),
                    z=float(translation[2]),
                    rx=float(rx_),
                    ry=float(ry_),
                    rz=float(rz_),
                )
            surf_idx = surf_idx + 1

            # we need to advance the cs by the surface thickness
            # if it is finite
            dt = surf.get("thickness", 0.0)
            if not be.isinf(dt):
                self.current_cs = CoordinateSystem(
                    x=0.0, y=0.0, z=dt, reference_cs=self.current_cs
                )

        # image surface specific
        translation, _ = self.current_cs.get_effective_transform()
        rx_, ry_, rz_ = self.current_cs.get_effective_rotation_euler()
        self.optic.add_surface(
            index=surf_idx,
            x=translation[0],
            y=translation[1],
            z=translation[2],
            rx=rx_,
            ry=ry_,
            rz=rz_,
        )

    def _configure_surface(self, index: int, data: dict):
        """Configures a single surface for the optic.

        Args:
            index (int): The index of the surface to configure.
            data (dict): The data for the surface.
        """
        coefficients = self._configure_surface_coefficients(data)
        extra_params = {}
        if data["type"] == "coordinate_break":
            # map the zmx PARM values to the actual decenters and rotations
            extra_params["dx"] = data.get("param_0", 0.0)
            extra_params["dy"] = data.get("param_1", 0.0)
            # convert degrees to radians
            extra_params["rx"] = be.deg2rad(be.array(data.get("param_2", 0.0)))
            extra_params["ry"] = be.deg2rad(be.array(data.get("param_3", 0.0)))
            extra_params["rz"] = be.deg2rad(be.array(data.get("param_4", 0.0)))
            extra_params["order_flag"] = data.get("param_5", 0.0)
        self.optic.add_surface(
            index=index,
            surface_type=data["type"],
            radius=data["radius"],
            conic=data["conic"],
            thickness=data["thickness"],
            is_stop=data["is_stop"],
            material=data["material"],
            coefficients=coefficients,
            **extra_params,
        )

    def _configure_surface_coefficients(self, data: dict):
        """Configures the aspheric coefficients for a surface.

        Returns None for standard or coordinate_break surfaces.

        Args:
            data (dict): The surface data dictionary.

        Returns:
            list[float] or None: A list of coefficient values, or None.

        Raises:
            ValueError: If the surface type is unsupported for coefficients.
        """
        surf_type = data["type"]
        if surf_type == "standard" or surf_type == "coordinate_break":
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
