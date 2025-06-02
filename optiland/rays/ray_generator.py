"""Ray Generator

This module contains the RayGenerator class, which is used to generate rays
for tracing through an optical system.

Kramer Harrison, 2024
"""

import optiland.backend as be
from optiland.rays.polarized_rays import PolarizedRays
from optiland.rays.real_rays import RealRays


class RayGenerator:
    """Generates rays for optical system analysis.

    This class takes an optical system definition (`optic`) and provides
    methods to generate initial ray data (positions and directions) based on
    field points and pupil coordinates. It handles various configurations like
    telecentricity and objects at infinity.

    Attributes:
        optic (Optic): An instance of the `Optic` class, representing the
            optical system for which rays are generated.
    """

    def __init__(self, optic):
        """Initializes a RayGenerator object.

        Args:
            optic (Optic): The optical system (`Optic` instance) for which this
                generator will create rays.
        """
        self.optic = optic

    def generate_rays(
        self,
        Hx: float,
        Hy: float,
        Px: float | be.Tensor,
        Py: float | be.Tensor,
        wavelength: float,
    ) -> RealRays | PolarizedRays:
        """Generates rays for tracing based on field and pupil coordinates.

        This method calculates the initial positions (x0, y0, z0) and direction
        cosines (L, M, N) for a set of rays. It considers vignetting,
        telecentricity, and the object's location (finite or infinite).

        Args:
            Hx (float): Normalized x-coordinate of the field point, ranging
                from -1 to 1.
            Hy (float): Normalized y-coordinate of the field point, ranging
                from -1 to 1.
            Px (float | be.Tensor): Normalized x-coordinate(s) in the pupil,
                typically ranging from -1 to 1. Can be a scalar or a tensor
                for multiple rays.
            Py (float | be.Tensor): Normalized y-coordinate(s) in the pupil,
                typically ranging from -1 to 1. Can be a scalar or a tensor
                for multiple rays.
            wavelength (float): The wavelength of the rays in micrometers.

        Returns:
            RealRays | PolarizedRays: An instance of `RealRays` or
            `PolarizedRays` (if `self.optic.polarization` is not "ignore")
            containing the generated ray data.

        Raises:
            ValueError:
                - If `self.optic.obj_space_telecentric` is True and
                  `self.optic.field_type` is "angle".
                - If `self.optic.obj_space_telecentric` is True and
                  `self.optic.aperture.ap_type` is "EPD" or "imageFNO".
                - If `self.optic.polarization` is "ignore" but the optic's
                  surface group uses polarization-dependent coatings.
        """
        vxf, vyf = self.optic.fields.get_vig_factor(Hx, Hy)
        vx = 1 - be.array(vxf)
        vy = 1 - be.array(vyf)
        x0, y0, z0 = self._get_ray_origins(Hx, Hy, Px, Py, vx, vy)

        x1, y1, z1 = self._calculate_initial_target_points(Px, Py, x0, y0, z0, vx, vy)

        mag = be.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
        L = (x1 - x0) / mag
        M = (y1 - y0) / mag
        N = (z1 - z0) / mag

        intensity = be.ones_like(x1)
        wavelength = be.ones_like(x1) * wavelength

        if self.optic.polarization == "ignore":
            if self.optic.surface_group.uses_polarization:
                raise ValueError(
                    "Polarization must be set when surfaces have "
                    "polarization-dependent coatings.",
                )
            return RealRays(x0, y0, z0, L, M, N, intensity, wavelength)
        return PolarizedRays(x0, y0, z0, L, M, N, intensity, wavelength)

    def _get_ray_origins(
        self,
        Hx: float,
        Hy: float,
        Px: float | be.Tensor,
        Py: float | be.Tensor,
        vx: float | be.Tensor,
        vy: float | be.Tensor,
    ) -> tuple[be.Tensor, be.Tensor, be.Tensor]:
        """Calculates the initial positions (origins) for rays.

        This method determines the (x0, y0, z0) coordinates where rays start,
        based on the field coordinates, pupil coordinates, vignetting factors,
        and object surface characteristics (e.g., whether it's at infinity).

        Args:
            Hx (float): Normalized x-coordinate of the field point.
            Hy (float): Normalized y-coordinate of the field point.
            Px (float | be.Tensor): Normalized x-coordinate(s) in the pupil.
            Py (float | be.Tensor): Normalized y-coordinate(s) in the pupil.
            vx (float | be.Tensor): Vignetting factor in the x-direction.
                Applied to pupil coordinates for object at infinity.
            vy (float | be.Tensor): Vignetting factor in the y-direction.
                Applied to pupil coordinates for object at infinity.

        Returns:
            tuple[be.Tensor, be.Tensor, be.Tensor]: A tuple (x0, y0, z0)
            containing the tensor coordinates of the ray origins.

        Raises:
            ValueError:
                - If `self.optic.field_type` is "object_height" and the object
                  is at infinity.
                - If `self.optic.obj_space_telecentric` is True and the object
                  is at infinity.
        """
        obj = self.optic.object_surface
        max_field = self.optic.fields.max_field
        field_x = max_field * Hx
        field_y = max_field * Hy

        if obj.is_infinite:
            x0, y0, z0 = self._get_ray_origins_infinite_object(
                field_x, field_y, Px, Py, vx, vy
            )
        else:
            x0, y0, z0 = self._get_ray_origins_finite_object(field_x, field_y, Px, Py)

        return x0, y0, z0

    def _calculate_initial_target_points(
        self,
        Px: float | be.Tensor,
        Py: float | be.Tensor,
        x0: be.Tensor,
        y0: be.Tensor,
        z0: be.Tensor,
        vx: be.Tensor,
        vy: be.Tensor,
    ) -> tuple[be.Tensor, be.Tensor, be.Tensor]:
        """Calculates initial target points (x1, y1, z1) for rays.

        These points are typically on the entrance pupil or a similar reference
        plane, depending on whether the object space is telecentric.

        Args:
            Px (float | be.Tensor): Normalized x-coordinate(s) in the pupil.
            Py (float | be.Tensor): Normalized y-coordinate(s) in the pupil.
            x0 (be.Tensor): X-coordinates of the ray origins.
            y0 (be.Tensor): Y-coordinates of the ray origins.
            z0 (be.Tensor): Z-coordinates of the ray origins.
            vx (be.Tensor): Vignetting factor in the x-direction.
            vy (be.Tensor): Vignetting factor in the y-direction.

        Returns:
            tuple[be.Tensor, be.Tensor, be.Tensor]: Target coordinates (x1, y1, z1).

        Raises:
            ValueError: If telecentric conditions are not met.
        """
        if self.optic.obj_space_telecentric:
            if self.optic.field_type == "angle":
                raise ValueError(
                    'Field type cannot be "angle" for telecentric object space.',
                )
            if self.optic.aperture.ap_type == "EPD":
                raise ValueError(
                    'Aperture type cannot be "EPD" for telecentric object space.',
                )
            if self.optic.aperture.ap_type == "imageFNO":
                raise ValueError(
                    'Aperture type cannot be "imageFNO" for telecentric object space.',
                )

            sin = self.optic.aperture.value
            # Ensure z0 is correctly broadcastable if it's a scalar and Px is an array
            z0_broadcast = be.full_like(Px, z0) if be.ndim(z0) == 0 else z0
            z = be.sqrt(1 - sin**2) / sin + z0_broadcast
            z1 = be.full_like(Px, z)
            x1 = Px * vx + x0
            y1 = Py * vy + y0
        else:
            EPL = self.optic.paraxial.EPL()
            EPD = self.optic.paraxial.EPD()

            x1 = Px * EPD * vx / 2
            y1 = Py * EPD * vy / 2
            z1 = be.full_like(Px, EPL)
        return x1, y1, z1

    def _get_ray_origins_infinite_object(
        self,
        field_x: float,
        field_y: float,
        Px: float | be.Tensor,
        Py: float | be.Tensor,
        vx: be.Tensor,
        vy: be.Tensor,
    ) -> tuple[be.Tensor, be.Tensor, be.Tensor]:
        """Calculates ray origins for an object at infinity.

        Args:
            field_x (float): Actual field coordinate in x (e.g., angle in degrees).
            field_y (float): Actual field coordinate in y (e.g., angle in degrees).
            Px (float | be.Tensor): Normalized x-coordinate(s) in the pupil.
            Py (float | be.Tensor): Normalized y-coordinate(s) in the pupil.
            vx (be.Tensor): Vignetting factor in the x-direction.
            vy (be.Tensor): Vignetting factor in the y-direction.

        Returns:
            tuple[be.Tensor, be.Tensor, be.Tensor]: Ray origin coordinates (x0, y0, z0).

        Raises:
            ValueError: If conditions for infinite object are not met.
        """
        if self.optic.field_type == "object_height":
            raise ValueError(
                'Field type cannot be "object_height" for an object at infinity.',
            )
        if self.optic.obj_space_telecentric:
            raise ValueError(
                "Object space cannot be telecentric for an object at infinity.",
            )
        EPL = self.optic.paraxial.EPL()
        EPD = self.optic.paraxial.EPD()

        offset = self._get_starting_z_offset()

        # x, y, z positions of ray starting points
        x = -be.tan(be.radians(field_x)) * (offset + EPL)
        y = -be.tan(be.radians(field_y)) * (offset + EPL)
        z = self.optic.surface_group.positions[1] - offset

        x0 = Px * EPD / 2 * vx + x
        y0 = Py * EPD / 2 * vy + y
        z0 = be.full_like(Px, z)
        return x0, y0, z0

    def _get_ray_origins_finite_object(
        self, field_x: float, field_y: float, Px: float | be.Tensor, Py: float | be.Tensor
    ) -> tuple[be.Tensor, be.Tensor, be.Tensor]:
        """Calculates ray origins for a finite object.

        Args:
            field_x (float): Actual field coordinate in x (e.g., object height).
            field_y (float): Actual field coordinate in y (e.g., object height).
            Px (float | be.Tensor): Normalized x-coordinate(s) in the pupil.
            Py (float | be.Tensor): Normalized y-coordinate(s) in the pupil.
                                   (Px, Py are used for be.full_like to match shape)
        Returns:
            tuple[be.Tensor, be.Tensor, be.Tensor]: Ray origin coordinates (x0, y0, z0).
        """
        obj = self.optic.object_surface
        if self.optic.field_type == "object_height":
            x = field_x
            y = field_y
            z = obj.geometry.sag(x, y) + obj.geometry.cs.z

        elif self.optic.field_type == "angle":
            EPL = self.optic.paraxial.EPL()
            z = self.optic.surface_group.positions[0]
            x = -be.tan(be.radians(field_x)) * (EPL - z)
            y = -be.tan(be.radians(field_y)) * (EPL - z)

        # Px is used here to ensure x0, y0, z0 have the same shape as Px, Py
        # if Px, Py are arrays (for multiple rays from the same field point).
        x0 = be.full_like(Px, x)
        y0 = be.full_like(Px, y)
        z0 = be.full_like(Px, z)
        return x0, y0, z0

    def _get_starting_z_offset(self):
        """Calculate the starting ray z-coordinate offset for systems with an
        object at infinity. This is relative to the first surface of the optic.

        This method chooses a starting point that is equivalent to the entrance
        pupil diameter of the optic.

        Returns:
            float: The z-coordinate offset relative to the first surface.

        """
        # Ensure z is an array, even if only one relevant position.
        # This handles cases where surface_group.positions might be short.
        relevant_positions = self.optic.surface_group.positions[1:-1]
        if not relevant_positions.any(): # Check if the slice is empty
             # If no intermediate surfaces, offset is relative to the first surface's EPD.
             # Or, handle as an error or specific default if this case isn't expected.
             # For now, let's assume EPD itself is a sensible offset if no internal surfaces.
            min_z = 0.0 # Or some other appropriate default for an empty list
        else:
            min_z = be.min(relevant_positions)

        offset = self.optic.paraxial.EPD()
        return offset - min_z
