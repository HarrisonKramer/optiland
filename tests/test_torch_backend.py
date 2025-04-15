import pytest
import optiland.backend as be
from optiland.backend import torch_backend

# Attempt importing torch and skip all tests if not installed.
# Note that some users may not have torch installed, so this avoids unnecessary errors.
torch = pytest.importorskip("torch", reason="torch is not installed")


@pytest.fixture(autouse=True)
def restore_backend_config():
    """Automatically restore global state after each test."""
    original_backend = be.get_backend()
    original_device = torch_backend.get_device()
    original_precision = torch_backend.get_precision()
    original_grad = torch_backend.grad_mode.requires_grad

    yield  # run the test

    # Restore the original state
    be.set_backend(original_backend)
    torch_backend.set_device(original_device)
    if original_precision == torch.float32:
        torch_backend.set_precision("float32")
    elif original_precision == torch.float64:
        torch_backend.set_precision("float64")
    if original_grad:
        torch_backend.grad_mode.enable()
    else:
        torch_backend.grad_mode.disable()


def test_set_and_get_device_cpu():
    # Set device to cpu and then retrieve it.
    torch_backend.set_device("cpu")
    assert torch_backend.get_device() == "cpu"


def test_set_and_get_device_cuda():
    # If CUDA is available, test setting to cuda.
    if torch.cuda.is_available():
        torch_backend.set_device("cuda")
        assert torch_backend.get_device() == "cuda"
    else:
        with pytest.raises(ValueError):  # CUDA not available
            torch_backend.set_device("cuda")


def test_set_device_invalid():
    # Test that an invalid device option raises ValueError.
    with pytest.raises(ValueError):
        torch_backend.set_device("tpu")


def test_set_and_get_precision_float32():
    torch_backend.set_precision("float32")
    assert torch_backend.get_precision() == torch.float32


def test_set_and_get_precision_float64():
    torch_backend.set_precision("float64")
    assert torch_backend.get_precision() == torch.float64


def test_set_precision_invalid():
    # Test that setting an unsupported precision raises ValueError.
    with pytest.raises(ValueError):
        torch_backend.set_precision("float16")


def test_grad_mode_enable_disable():
    # First ensure grad mode is disabled.
    torch_backend.grad_mode.disable()
    assert torch_backend.grad_mode.requires_grad is False

    # Enable grad mode and verify.
    torch_backend.grad_mode.enable()
    assert torch_backend.grad_mode.requires_grad is True

    # Disable and check again.
    torch_backend.grad_mode.disable()
    assert torch_backend.grad_mode.requires_grad is False


def test_grad_mode_temporary_enable():
    # Start with gradient mode disabled.
    torch_backend.grad_mode.disable()
    initial_state = torch_backend.grad_mode.requires_grad
    assert initial_state is False

    # Inside the context, grad mode should be enabled.
    with torch_backend.grad_mode.temporary_enable():
        assert torch_backend.grad_mode.requires_grad is True

    # After the context, it should revert to the original state.
    assert torch_backend.grad_mode.requires_grad is initial_state
