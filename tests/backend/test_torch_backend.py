# tests/backend/test_torch_backend.py
"""
Tests for the configuration and state management of the PyTorch backend.
"""
import pytest
from optiland import backend as be
from optiland.backend import torch_backend

# Attempt importing torch and skip all tests if not installed.
torch = pytest.importorskip("torch", reason="torch is not installed")


@pytest.fixture(autouse=True)
def restore_backend_config():
    """
    A fixture that runs for every test in this module to ensure a clean state.
    It saves the original backend configuration (backend name, device,
    precision, grad mode) before each test and restores it afterward.
    """
    original_backend = be.get_backend()
    original_device = torch_backend.get_device()
    original_precision = torch_backend.get_precision()
    original_grad = torch_backend.grad_mode.requires_grad
    yield
    # Restore the original state after the test completes
    be.set_backend(original_backend)
    torch_backend.set_device(original_device)
    torch_backend.set_precision(str(original_precision).split('.')[-1])
    if original_grad:
        torch_backend.grad_mode.enable()
    else:
        torch_backend.grad_mode.disable()


def test_set_and_get_device_cpu():
    """
    Tests that the device can be successfully set to 'cpu' and retrieved.
    """
    torch_backend.set_device("cpu")
    assert torch_backend.get_device() == "cpu"


def test_set_and_get_device_cuda():
    """
    Tests that the device can be set to 'cuda' if available, otherwise
    it should raise a ValueError.
    """
    if torch.cuda.is_available():
        torch_backend.set_device("cuda")
        assert torch_backend.get_device() == "cuda"
    else:
        with pytest.raises(ValueError, match="CUDA is not available on this system."):
            torch_backend.set_device("cuda")


def test_set_device_invalid():
    """
    Tests that attempting to set an unsupported device raises a ValueError.
    """
    with pytest.raises(ValueError, match="Unsupported device"):
        torch_backend.set_device("tpu")


def test_set_and_get_precision_float32():
    """
    Tests that the global floating-point precision can be set to 'float32'.
    """
    torch_backend.set_precision("float32")
    assert torch_backend.get_precision() == torch.float32


def test_set_and_get_precision_float64():
    """
    Tests that the global floating-point precision can be set to 'float64'.
    """
    torch_backend.set_precision("float64")
    assert torch_backend.get_precision() == torch.float64


def test_set_precision_invalid():
    """
    Tests that attempting to set an unsupported precision raises a ValueError.
    """
    with pytest.raises(ValueError, match="Unsupported precision"):
        torch_backend.set_precision("float16")


def test_grad_mode_enable_disable():
    """
    Tests the functions for globally enabling and disabling gradient tracking.
    """
    torch_backend.grad_mode.disable()
    assert not torch_backend.grad_mode.requires_grad
    torch_backend.grad_mode.enable()
    assert torch_backend.grad_mode.requires_grad


def test_grad_mode_temporary_enable():
    """
    Tests the context manager for temporarily enabling gradient tracking,
    ensuring it reverts to the original state upon exit.
    """
    torch_backend.grad_mode.disable()
    initial_state = torch_backend.grad_mode.requires_grad
    assert not initial_state
    with torch_backend.grad_mode.temporary_enable():
        assert torch_backend.grad_mode.requires_grad
    assert torch_backend.grad_mode.requires_grad == initial_state