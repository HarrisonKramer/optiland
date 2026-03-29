from __future__ import annotations

import os
import sys

# Add project root to sys.path
sys.path.append(os.getcwd())

from optiland.optic.optic import Optic
from optiland.optimization.operand.operand import Operand


def test_primary_wavelength():
    optic = Optic()
    optic.add_wavelength(0.55)

    # This is what failed before
    input_data = {
        "optic": optic,
        "surface_number": 1,
        "Hx": 0.0,
        "Hy": 0.1,
        "num_rays": 6,
        "wavelength": 0.55,
    }
    op = Operand(operand_type="rms_spot_size", input_data=input_data)
    print(f"Value with float wavelength: {op.value}")

    try:
        input_data_str = {
            "optic": optic,
            "surface_number": 1,
            "Hx": 0.0,
            "Hy": 0.1,
            "num_rays": 6,
            "wavelength": "primary",
        }
        op_str = Operand(operand_type="rms_spot_size", input_data=input_data_str)
        print(f"Value with 'primary' wavelength: {op_str.value}")
    except Exception as e:
        print(f"Caught expected failure with 'primary': {e}")


if __name__ == "__main__":
    test_primary_wavelength()
