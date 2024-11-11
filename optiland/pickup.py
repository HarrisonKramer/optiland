class Pickup:
    """A class representing a pickup on an optic surface

    Args:
        optic (Optic): The optic object on which the pickup operation is
            performed.
        source_surface_idx (int): The index of the source surface in the
            optic's surface group.
        attr_type (str): The type of attribute to be picked up ('radius',
            'conic', or 'thickness').
        target_surface_idx (int): The index of the target surface in the
            optic's surface group.
        scale (float, optional): The scaling factor applied to the picked up
            value. Defaults to 1.
        offset (float, optional): The offset added to the picked up value.
            Defaults to 0.

    Methods:
        apply(): Applies the pickup operation by scaling and offsetting the
            picked up value and setting it on the target surface.

    Raises:
        ValueError: If an invalid source attribute is specified.
    """

    def __init__(self, optic, source_surface_idx, attr_type,
                 target_surface_idx, scale=1, offset=0):
        self.optic = optic
        self.source_surface_idx = source_surface_idx
        self.attr_type = attr_type
        self.target_surface_idx = target_surface_idx
        self.scale = scale
        self.offset = offset

    def apply(self):
        """
        Updates the target surface based on the source surface attribute.

        This method calculates the new value by multiplying the current value
        by the scale factor and adding the offset. The new value is then set
        on the target surface.
        """
        old_value = self._get_value()
        new_value = self.scale * old_value + self.offset
        self._set_value(new_value)

    def _get_value(self):
        """
        Returns the value of the source surface attribute.

        Returns:
            The value of the attribute.

        Raises:
            ValueError: If the source attribute is invalid.
        """
        surface = self.optic.surface_group.surfaces[self.source_surface_idx]
        if self.attr_type == 'radius':
            return surface.geometry.radius
        elif self.attr_type == 'conic':
            return surface.geometry.k
        elif self.attr_type == 'thickness':
            return (
                self.optic.surface_group.get_thickness(self.source_surface_idx)
            )
        else:
            raise ValueError('Invalid source attribute')

    def _set_value(self, value):
        """
        Sets the value of the target surface attribute.

        Parameters:
            value (float): The value to set for the attribute.

        Raises:
            ValueError: If the source attribute is invalid.
        """
        if self.attr_type == 'radius':
            self.optic.set_radius(value, self.target_surface_idx)
        elif self.attr_type == 'conic':
            self.optic.set_conic(value, self.target_surface_idx)
        elif self.attr_type == 'thickness':
            self.optic.set_thickness(value, self.target_surface_idx)
        else:
            raise ValueError('Invalid source attribute')
