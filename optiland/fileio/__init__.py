"""This package handles file input/output operations for Optiland,
including loading and saving Optiland's native JSON format and
importing Zemax (.zmx) files."""

from .converters import ZemaxToOpticConverter
from .optiland_handler import (
    load_obj_from_json,
    load_optiland_file,
    save_obj_to_json,
    save_optiland_file,
)
from .zemax_handler import ZemaxFileReader, load_zemax_file

__all__ = [
    # From converters.py
    "ZemaxToOpticConverter",
    # From optiland_handler.py
    "load_obj_from_json",
    "save_obj_to_json",
    "load_optiland_file",
    "save_optiland_file",
    # From zemax_handler.py
    "load_zemax_file",
    "ZemaxFileReader",
]
