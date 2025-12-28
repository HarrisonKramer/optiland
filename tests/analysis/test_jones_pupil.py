
import pytest
import numpy as np
import matplotlib.pyplot as plt
import optiland.backend as be
from optiland.samples.objectives import CookeTriplet
from optiland.analysis.jones_pupil import JonesPupil
from optiland.rays import PolarizationState

@pytest.fixture
def optic():
    optic = CookeTriplet()
    optic.set_polarization("ignore") # Default state for test
    return optic

def test_jones_pupil_initialization(optic):
    jp = JonesPupil(optic)
    assert jp.optic == optic
    assert jp.grid_size == 33
    assert jp.fields is not None
    assert jp.wavelengths is not None

def test_jones_pupil_generate_data(optic):
    jp = JonesPupil(optic, grid_size=5)
    data = jp.data

    # Check structure: list of fields -> list of wavelengths -> dict
    assert isinstance(data, list)
    assert len(data) == len(jp.fields)
    assert isinstance(data[0], list)
    assert len(data[0]) == len(jp.wavelengths)

    single_data = data[0][0]
    assert "Px" in single_data
    assert "Py" in single_data
    assert "J" in single_data

    # Check shapes
    num_rays = 5 * 5
    assert single_data["Px"].shape == (num_rays,)
    assert single_data["J"].shape == (num_rays, 2, 2)

    # Check values (should be roughly identity for a simple lens near center)
    # The Cooke Triplet is paraxial-ish.
    # Jxx should be close to 1 (amplitude) and Jxy close to 0.

    J = single_data["J"]
    Jxx = J[:, 0, 0]
    Jxy = J[:, 0, 1]

    # Center ray (index 12 for 5x5 grid)
    center_idx = num_rays // 2

    # Use backend agnostic checks
    val_xx = be.to_numpy(Jxx[center_idx])
    val_xy = be.to_numpy(Jxy[center_idx])

    # Amplitude near 1 (accounting for Fresnel losses, maybe < 1)
    # Since polarization was "ignore" initially, and we set it for trace,
    # the coating behaviour depends on if default coatings are applied.
    # Optiland defaults to no coatings (transmittance = 1)?
    # Or Fresnel? The default material behavior might include Fresnel.
    # If not, Jxx ~ 1.

    # Let's check magnitude
    assert np.abs(val_xx) > 0.5
    assert np.abs(val_xy) < 0.1

def test_jones_pupil_polarization_handling(optic):
    # Ensure it works even if optic polarization is 'ignore'
    optic.set_polarization("ignore")
    jp = JonesPupil(optic, grid_size=3)
    # Trace happens in __init__

    # Ensure optic state is restored
    assert optic.polarization == "ignore"

    # Data should be valid
    assert jp.data is not None

def test_jones_pupil_view(optic):
    jp = JonesPupil(optic, grid_size=5)
    fig, axs = jp.view()

    assert fig is not None
    # 8 data axes + 8 colorbars = 16 axes in total
    assert len(axs) == 16

    # Clean up
    plt.close(fig)

def test_view_no_fields(optic):
    optic.fields.fields.clear()
    jp = JonesPupil(optic)
    fig, axs = jp.view()
    assert fig is None
    assert axs is None
