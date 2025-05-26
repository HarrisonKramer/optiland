"""Tests for PSF classes in optiland.psf.

This module includes tests for HuygensPSF and FFTPSF, focusing on
instantiation, PSF properties, Strehl ratio, and visualization method execution.
Tests are parameterized to run with both numpy and torch backends.
"""
import os
import pytest
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for tests to prevent GUI popups
import matplotlib.pyplot as plt

import optiland.backend as be
from optiland.psf.huygens_fresnel import HuygensPSF
from optiland.psf.fft import FFTPSF # Import FFTPSF
from optiland.optic.optic import Optic
from optiland.surfaces.standard_surface import Surface
from optiland.surfaces.object_surface import ObjectSurface
from optiland.geometries.plane import Plane
from optiland.geometries.standard import Standard as StandardGeometry
from optiland.materials.ideal import IdealMaterial
from optiland.coordinate_system import CoordinateSystem

# Environment variable to simulate conditions where torch tests should be skipped
SKIP_TORCH_TESTS = os.getenv('SKIP_TORCH_TESTS', 'false').lower() == 'true'

@pytest.fixture(scope="function")
def set_backend_fixture(request):
    """Fixture to set the backend for parameterized tests."""
    backend_name = request.param
    original_backend = be.get_current_backend_name()

    if backend_name == "torch":
        if SKIP_TORCH_TESTS:
            pytest.skip("Skipping torch backend tests due to SKIP_TORCH_TESTS environment variable.")
        try:
            import torch 
        except ImportError:
            pytest.skip("Skipping torch backend tests: PyTorch not installed.")
            
    try:
        be.set_backend(backend_name)
        yield backend_name 
    except ImportError: 
        pytest.skip(f"Backend '{backend_name}' not available or import failed.")
    finally:
        be.set_backend(original_backend)


@pytest.fixture
def simple_optic_fixture():
    """Provides a basic Optic instance: Object at inf, Stop, Planar Image."""
    air = IdealMaterial(n=1.0)
    cs_origin = CoordinateSystem() 

    obj_z = -be.inf if be.get_current_backend_name() == "numpy" else float('-inf')
    obj_cs = CoordinateSystem(z=obj_z) 
    obj_geometry = Plane(coordinate_system=obj_cs)
    obj_geometry.thickness = 10.0 
    obj_surface = ObjectSurface(geometry=obj_geometry, material_post=air)

    stop_geometry = Plane(coordinate_system=cs_origin) 
    stop_geometry.thickness = 100.0 
    stop_surface = Surface(
        geometry=stop_geometry,
        material_pre=air,
        material_post=air,
        is_stop=True,
        aperture=10.0 
    )

    image_geometry = Plane(coordinate_system=cs_origin) 
    image_surface = Surface(
        geometry=image_geometry,
        material_pre=air,
        material_post=air
    )

    optic_system = Optic(name="SimpleTestOptic")
    optic_system.add_surface(new_surface=obj_surface)
    optic_system.add_surface(new_surface=stop_surface)
    optic_system.add_surface(new_surface=image_surface)
    
    optic_system.set_aperture(aperture_type="EPD", value=10.0) 
    optic_system.set_field_type("angle") 
    optic_system.add_field(y=0.0) 
    optic_system.add_wavelength(value=0.55, is_primary=True) 
    optic_system.update() 

    return optic_system

@pytest.fixture
def curved_image_optic_fixture(simple_optic_fixture):
    """Modifies simple_optic_fixture to have a curved image surface."""
    optic = simple_optic_fixture
    
    if len(optic.surface_group.surfaces) > 0 and optic.image_surface == optic.surface_group.surfaces[-1]:
         optic.surface_group.surfaces.pop() 

    air = IdealMaterial(n=1.0)
    curved_image_cs = CoordinateSystem() 
    curved_image_geometry = StandardGeometry(radius=-200.0, conic=0.0, coordinate_system=curved_image_cs) 
    
    new_image_surface = Surface(
        geometry=curved_image_geometry,
        material_pre=air,
        material_post=air 
    )
    optic.add_surface(new_surface=new_image_surface)
    optic.update()
    return optic


# Determine backend parameters for tests
backend_params_for_tests = ["numpy"]
if not SKIP_TORCH_TESTS:
    try:
        import torch
        if torch.__version__: 
             backend_params_for_tests.append("torch")
    except ImportError:
        pass 

# --- HuygensPSF Tests ---

@pytest.mark.parametrize("set_backend_fixture", backend_params_for_tests, indirect=True)
def test_huygens_psf_instantiation_defaults(set_backend_fixture, simple_optic_fixture):
    optic = simple_optic_fixture
    field_point = (0.0, 0.0) 
    wavelength_um = 0.55  
    test_num_rays = 16 
    test_image_size = 16

    psf_calculator = HuygensPSF(
        optic=optic,
        field=field_point,
        wavelength=wavelength_um,
        num_rays=test_num_rays,
        image_size=test_image_size
    )
    assert isinstance(psf_calculator, HuygensPSF)
    assert hasattr(psf_calculator, 'psf')
    assert be.is_tensor(psf_calculator.psf)
    assert psf_calculator.psf.shape == (test_image_size, test_image_size)
    plt.close('all')

@pytest.mark.parametrize("set_backend_fixture", backend_params_for_tests, indirect=True)
@pytest.mark.parametrize("image_size, num_rays_pupil", [(16, 16), (24, 20)])
def test_huygens_psf_instantiation_custom(set_backend_fixture, simple_optic_fixture, image_size, num_rays_pupil):
    optic = simple_optic_fixture
    field_point = (0.0, 0.0)
    wavelength_um = 0.55

    psf_calculator = HuygensPSF(
        optic=optic,
        field=field_point,
        wavelength=wavelength_um,
        image_size=image_size,
        num_rays=num_rays_pupil
    )
    assert isinstance(psf_calculator, HuygensPSF)
    assert hasattr(psf_calculator, 'psf')
    assert be.is_tensor(psf_calculator.psf)
    assert psf_calculator.psf.shape == (image_size, image_size)
    plt.close('all')

@pytest.mark.parametrize("set_backend_fixture", backend_params_for_tests, indirect=True)
def test_huygens_psf_attributes_and_peak(set_backend_fixture, simple_optic_fixture):
    optic = simple_optic_fixture
    field_point = (0.0, 0.0) 
    wavelength_um = 0.55
    test_image_size = 24 
    test_num_rays = 20   

    psf_calculator = HuygensPSF(
        optic=optic,
        field=field_point,
        wavelength=wavelength_um,
        image_size=test_image_size,
        num_rays=test_num_rays
    )
    psf_array = psf_calculator.psf
    assert be.is_tensor(psf_array)
    assert be.all(psf_array >= -1e-9)
    peak_indices = be.unravel_index(be.argmax(psf_array), psf_array.shape)
    center_coord = test_image_size // 2
    peak_idx_0_np = be.to_numpy(peak_indices[0]).item()
    peak_idx_1_np = be.to_numpy(peak_indices[1]).item()
    assert abs(peak_idx_0_np - center_coord) <= 2
    assert abs(peak_idx_1_np - center_coord) <= 2
    plt.close('all')

@pytest.mark.parametrize("set_backend_fixture", backend_params_for_tests, indirect=True)
def test_huygens_psf_strehl_ratio(set_backend_fixture, simple_optic_fixture):
    optic = simple_optic_fixture
    field_point = (0.0, 0.0)
    wavelength_um = 0.55
    test_image_size = 20 
    test_num_rays = 16   
    psf_calculator = HuygensPSF(
        optic=optic,
        field=field_point,
        wavelength=wavelength_um,
        image_size=test_image_size,
        num_rays=test_num_rays
    )
    strehl = psf_calculator.strehl_ratio()
    assert 0.95 <= be.to_numpy(strehl) <= 1.05
    plt.close('all')

@pytest.mark.parametrize("set_backend_fixture", backend_params_for_tests, indirect=True)
@pytest.mark.parametrize("projection_type", ["2d", "3d"])
def test_huygens_psf_view_execution(set_backend_fixture, simple_optic_fixture, projection_type):
    optic = simple_optic_fixture
    field_point = (0.0, 0.0)
    wavelength_um = 0.55
    test_image_size = 16 
    test_num_rays = 16   
    psf_calculator = HuygensPSF(
        optic=optic,
        field=field_point,
        wavelength=wavelength_um,
        image_size=test_image_size,
        num_rays=test_num_rays
    )
    try:
        psf_calculator.view(projection=projection_type, log=True)
        psf_calculator.view(projection=projection_type, log=False)
    except Exception as e:
        pytest.fail(f"HuygensPSF.view(projection='{projection_type}') raised an exception: {e}")
    finally:
        plt.close('all')

@pytest.mark.parametrize("set_backend_fixture", backend_params_for_tests, indirect=True)
def test_huygens_psf_with_curved_image_surface(set_backend_fixture, curved_image_optic_fixture):
    optic = curved_image_optic_fixture
    field_point = (0.0, 0.0) 
    wavelength_um = 0.55  
    test_num_rays = 16 
    test_image_size = 16
    try:
        psf_calculator = HuygensPSF(
            optic=optic,
            field=field_point,
            wavelength=wavelength_um,
            num_rays=test_num_rays,
            image_size=test_image_size
        )
        assert isinstance(psf_calculator, HuygensPSF)
        assert hasattr(psf_calculator, 'psf')
        assert be.is_tensor(psf_calculator.psf)
        assert psf_calculator.psf.shape == (test_image_size, test_image_size)
    except Exception as e:
        pytest.fail(f"HuygensPSF with curved image surface raised an exception: {e}")
    finally:
        plt.close('all')

# --- FFTPSF Tests ---

@pytest.mark.parametrize("set_backend_fixture", backend_params_for_tests, indirect=True)
@pytest.mark.parametrize("num_rays, grid_size", [(16, 32), (24, 48)])
def test_fftpsf_instantiation_and_pupils(set_backend_fixture, simple_optic_fixture, num_rays, grid_size):
    """Test FFTPSF instantiation and _generate_pupils method."""
    optic = simple_optic_fixture
    field_point = (0.0, 0.0)
    wavelength_um = 0.55

    fft_psf = FFTPSF(
        optic=optic,
        field=field_point,
        wavelength=wavelength_um,
        num_rays=num_rays,
        grid_size=grid_size
    )
    assert isinstance(fft_psf, FFTPSF)
    assert hasattr(fft_psf, 'pupils')
    assert isinstance(fft_psf.pupils, list)
    assert len(fft_psf.pupils) > 0 # Should have at least one pupil for the one wavelength
    
    for pupil_array in fft_psf.pupils:
        assert be.is_tensor(pupil_array)
        assert pupil_array.shape == (num_rays, num_rays)
        assert be.is_complex_dtype(pupil_array), f"Pupil dtype is {pupil_array.dtype}"
    plt.close('all')

@pytest.mark.parametrize("set_backend_fixture", backend_params_for_tests, indirect=True)
@pytest.mark.parametrize("num_rays, grid_size", [(16, 32), (24, 48)])
def test_fftpsf_psf_attributes_and_peak(set_backend_fixture, simple_optic_fixture, num_rays, grid_size):
    """Test FFTPSF's computed PSF attributes and peak location."""
    optic = simple_optic_fixture
    field_point = (0.0, 0.0)
    wavelength_um = 0.55

    fft_psf = FFTPSF(
        optic=optic,
        field=field_point,
        wavelength=wavelength_um,
        num_rays=num_rays,
        grid_size=grid_size
    )
    
    assert hasattr(fft_psf, 'psf')
    psf_array = fft_psf.psf
    assert be.is_tensor(psf_array)
    assert psf_array.shape == (grid_size, grid_size)
    assert be.all(psf_array >= -1e-9) # Allow for tiny float precision noise

    # Peak location for on-axis field should be at grid_size // 2
    peak_indices = be.unravel_index(be.argmax(psf_array), psf_array.shape)
    center_coord = grid_size // 2
    
    peak_idx_0_np = be.to_numpy(peak_indices[0]).item()
    peak_idx_1_np = be.to_numpy(peak_indices[1]).item()

    assert abs(peak_idx_0_np - center_coord) <= 1 # FFT result should be well-centered
    assert abs(peak_idx_1_np - center_coord) <= 1
    plt.close('all')

@pytest.mark.parametrize("set_backend_fixture", backend_params_for_tests, indirect=True)
def test_fftpsf_strehl_ratio(set_backend_fixture, simple_optic_fixture):
    """Test FFTPSF's Strehl ratio."""
    optic = simple_optic_fixture
    field_point = (0.0, 0.0)
    wavelength_um = 0.55
    # Use parameters that should result in a reasonably well-sampled PSF
    num_rays = 32 
    grid_size = 64 

    fft_psf = FFTPSF(
        optic=optic,
        field=field_point,
        wavelength=wavelength_um,
        num_rays=num_rays,
        grid_size=grid_size
    )
    strehl = fft_psf.strehl_ratio() 
    # For a simple aperture (diffraction limited), Strehl should be close to 1.0
    assert 0.95 <= be.to_numpy(strehl) <= 1.05, \
        f"Strehl ratio expected around 1.0 for simple optic, got {be.to_numpy(strehl)}"
    plt.close('all')

@pytest.mark.parametrize("set_backend_fixture", backend_params_for_tests, indirect=True)
def test_fftpsf_get_psf_units(set_backend_fixture, simple_optic_fixture):
    """Test FFTPSF's _get_psf_units method indirectly via view or directly."""
    optic = simple_optic_fixture
    field_point = (0.0, 0.0)
    wavelength_um = 0.55
    num_rays = 16
    grid_size = 32

    fft_psf = FFTPSF(
        optic=optic,
        field=field_point,
        wavelength=wavelength_um,
        num_rays=num_rays,
        grid_size=grid_size
    )
    
    # Create a dummy image array for _get_psf_units, matching shape of a zoomed PSF
    # _get_psf_units is called by view() with a potentially zoomed/cropped psf.
    # We test it with an array of the same shape as the non-zoomed, non-cropped psf for simplicity.
    dummy_psf_for_shape_ref = fft_psf.psf 
    
    x_extent, y_extent = fft_psf._get_psf_units(dummy_psf_for_shape_ref)
    assert isinstance(x_extent, (float, be.get_default_dtype_np()))
    assert isinstance(y_extent, (float, be.get_default_dtype_np()))
    assert x_extent > 0
    assert y_extent > 0
    plt.close('all')

@pytest.mark.parametrize("set_backend_fixture", backend_params_for_tests, indirect=True)
@pytest.mark.parametrize("projection_type", ["2d", "3d"])
def test_fftpsf_view_execution(set_backend_fixture, simple_optic_fixture, projection_type):
    """Test that FFTPSF.view() executes without errors."""
    optic = simple_optic_fixture
    field_point = (0.0, 0.0)
    wavelength_um = 0.55
    num_rays = 16
    grid_size = 32

    fft_psf = FFTPSF(
        optic=optic,
        field=field_point,
        wavelength=wavelength_um,
        num_rays=num_rays,
        grid_size=grid_size
    )
    try:
        fft_psf.view(projection=projection_type, log=True)
        fft_psf.view(projection=projection_type, log=False)
    except Exception as e:
        pytest.fail(f"FFTPSF.view(projection='{projection_type}') raised an exception: {e}")
    finally:
        plt.close('all')
