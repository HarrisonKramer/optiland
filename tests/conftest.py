import pytest

try:
    import optiland.backend as be
    numpy_available = True
except ModuleNotFoundError as e:
    if 'numpy' in str(e):
        numpy_available = False
    else:
        raise # Reraise if it's a different ModuleNotFoundError

if numpy_available:
    @pytest.fixture(params=be.list_available_backends(), ids=lambda b: f"backend={b}")
    def set_test_backend(request):
        """Fixture to set the backend for each test and ensure proper device configuration."""
        backend_name = request.param
        be.set_backend(backend_name)

        if backend_name == "torch":
            be.set_device("cpu")  # Use CPU for tests
            be.grad_mode.enable()  # Enable gradient tracking
            be.set_precision("float64")  # Set precision to float64 for tests

        yield

        # Reset the backend to numpy after the test
        be.set_backend("numpy")
else:
    # If numpy (and thus optiland.backend) is not available,
    # define a dummy fixture so tests that don't use it can still run.
    # Tests that *do* require this fixture will fail if it's not properly set up,
    # which will be a signal to fix the numpy installation.
    @pytest.fixture
    def set_test_backend(request):
        print("Warning: numpy not found, optiland.backend functionality is unavailable. Skipping backend tests.")
        yield
