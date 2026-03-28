"""Shared fixtures for optiland_gui tests."""

from __future__ import annotations

import pytest

import optiland.backend as be


@pytest.fixture(scope="session")
def qapp():
    """Session-scoped QApplication required by PySide6 signal/thread machinery."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture()
def minimal_optic():
    """A 4-surface singlet (object, lens front, lens back/stop, image).

    EPD = 10 mm, single on-axis angle field (y=0°), wavelength 0.55 µm primary.
    """
    from optiland.optic import Optic

    optic = Optic()
    optic.add_surface(index=0, radius=be.inf, thickness=be.inf)
    optic.add_surface(
        index=1,
        radius=50.0,
        thickness=5.0,
        material="N-BK7",
        is_stop=False,
    )
    optic.add_surface(
        index=2,
        radius=-50.0,
        thickness=45.0,
        is_stop=True,
    )
    optic.add_surface(index=3, radius=be.inf, thickness=0.0)

    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.set_field_type("angle")
    optic.add_field(y=0.0)
    optic.add_wavelength(value=0.55, is_primary=True)
    optic.update()
    return optic
