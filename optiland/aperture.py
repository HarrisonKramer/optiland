"""Optiland Aperture Module

This module provides a class for representing apertures in lens systems.

Kramer Harrison, 2024
"""


class Aperture:
    """
    Represents an aperture used in optics.

    Attributes:
        ap_type (str): The type of the aperture. Must be one of 'EPD',
            'imageFNO', or 'objectNA'.
        value (float): The value associated with the aperture.
        object_space_telecentric (bool): Indicates whether the lens is
            telecentric in object space.

    Raises:
        ValueError: If the aperture type is not one of 'EPD', 'imageFNO',
            or 'objectNA', or if the aperture type is 'EPD' or 'imageFNO'
            and the lens is telecentric in object space.

    """

    def __init__(self, aperture_type, value, object_space_telecentric=False):
        if aperture_type not in ['EPD', 'imageFNO', 'objectNA']:
            raise ValueError('Aperture type must be "EPD", "imageFNO", '
                             '"objectNA"')

        if aperture_type in ['EPD', 'imageFNO'] and object_space_telecentric:
            raise ValueError('Cannot set aperture type to "EPD" or "imageFNO" '
                             'if lens is telecentric in object space.')

        self.ap_type = aperture_type
        self.value = value
        self.object_space_telecentric = object_space_telecentric
