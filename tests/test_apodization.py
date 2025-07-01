import pytest
import optiland.backend as be
from optiland.apodization import UniformApodization, GaussianApodization, BaseApodization


def test_uniform_apodization_get_intensity():
    apod = UniformApodization()
    # Test with scalar inputs (though Px, Py are expected as arrays)
    # The function should handle them if backend array creation is robust
    px = be.array([0.5])
    py = be.array([0.5])
    intensity = apod.apply(px, py)
    assert be.all(intensity == be.array([1.0])), "Intensity should be 1.0 for UniformApodization"

    # Test with array inputs
    px_array = be.array([-1.0, 0.0, 1.0])
    py_array = be.array([-1.0, 0.0, 1.0])
    intensity_array = apod.apply(px_array, py_array)
    assert be.all(intensity_array == be.ones_like(px_array)), "Intensity should be all ones for UniformApodization with array inputs"


def test_gaussian_apodization_apply():
    sigma = 0.5
    apod = GaussianApodization(sigma=sigma)
    px = be.array([0.0])
    py = be.array([0.0])
    # Expected: exp(-(0^2 + 0^2) / (2 * 0.5^2)) = exp(0) = 1.0
    intensity_center = apod.apply(px, py)
    assert be.allclose(intensity_center, be.array([1.0])), "Intensity at center should be 1.0 for GaussianApodization"

    px_edge = be.array([sigma])
    py_edge = be.array([0.0])
    # Expected: exp(-(sigma^2 + 0^2) / (2 * sigma^2)) = exp(-1/2)
    intensity_edge = apod.apply(px_edge, py_edge)
    expected_intensity_edge = be.exp(be.array([-0.5]))
    assert be.allclose(intensity_edge, expected_intensity_edge), "Intensity at sigma distance not as expected"

    # Test with array inputs
    px_array = be.array([-sigma, 0.0, sigma])
    py_array = be.array([0.0, sigma, 0.0])
    # Expected: exp(-([-s,0,s]^2 + [0,s,0]^2) / (2s^2)) = exp(-([s^2,s^2,s^2])/(2s^2)) = exp(-0.5)
    intensity_array = apod.apply(px_array, py_array)
    expected_intensity_array = be.exp(be.array([-0.5, -0.5, -0.5]))
    assert be.allclose(intensity_array, expected_intensity_array), "Intensity array not as expected for GaussianApodization"


def test_apodization_to_dict_uniform():
    apod = UniformApodization()
    data = apod.to_dict()
    assert data == {"type": "UniformApodization"}


def test_apodization_to_dict_gaussian():
    sigma = 0.75
    apod = GaussianApodization(sigma=sigma)
    data = apod.to_dict()
    assert data == {"type": "GaussianApodization", "sigma": sigma}


def test_apodization_from_dict_uniform():
    data = {"type": "UniformApodization"}
    apod = BaseApodization.from_dict(data)
    assert isinstance(apod, UniformApodization)


def test_apodization_from_dict_gaussian():
    sigma = 0.6
    data = {"type": "GaussianApodization", "sigma": sigma}
    apod = BaseApodization.from_dict(data)
    assert isinstance(apod, GaussianApodization)
    assert apod.sigma == sigma


def test_apodization_from_dict_gaussian_default_sigma():
    data = {"type": "GaussianApodization"} # Missing sigma
    apod = BaseApodization.from_dict(data)
    assert isinstance(apod, GaussianApodization)
    assert apod.sigma == 1.0 # Default sigma


def test_apodization_from_dict_unknown():
    data = {"type": "UnknownApodization"}
    with pytest.raises(ValueError):
        BaseApodization.from_dict(data)
