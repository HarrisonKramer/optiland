"""Aperture Module

This module provides a class for representing apertures in lens systems.
Aperture options include entrance pupil diameter (EPD), image space F-number
(imageFNO), and object space numerical aperture (objectNA), or float by
stop size (float_by_stop_size). If the `float_by_stop_size` is chosen,
the input provided is the diameter of the stop surface.

Kramer Harrison, 2024
"""


class Aperture:
    """Represents an aperture used in optics.

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
        if aperture_type not in ["EPD", "imageFNO", "objectNA", "float_by_stop_size"]:
            raise ValueError(
                'Aperture type must be "EPD", "imageFNO", '
                '"objectNA", "float_by_stop_size"'
            )

        if aperture_type in ["EPD", "imageFNO"] and object_space_telecentric:
            raise ValueError(
                'Cannot set aperture type to "EPD" or "imageFNO" '
                "if lens is telecentric in object space.",
            )

        self.ap_type = aperture_type
        self.value = value
        self.object_space_telecentric = object_space_telecentric

    def to_dict(self):
        """Get a dictionary representation of the aperture.

        Returns:
            dict: A dictionary containing the aperture data.

        """
        return {
            "type": self.ap_type,
            "value": self.value,
            "object_space_telecentric": self.object_space_telecentric,
        }

    @classmethod
    def from_dict(cls, data):
        """Create an Aperture instance from a dictionary representation.

        Args:
            data (dict): A dictionary containing the aperture data.

        Returns:
            Aperture: A new Aperture instance created from the data.

        """
        required_keys = {"type", "value"}
        if not required_keys.issubset(data):
            missing = required_keys - data.keys()
            raise ValueError(f"Missing required keys: {missing}")

        return cls(
            aperture_type=data["type"],
            value=data["value"],
            object_space_telecentric=data.get("object_space_telecentric", False),
        )
