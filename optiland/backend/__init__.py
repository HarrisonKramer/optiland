"""
Backend Module

This module provides a unified interface for performing numerical operations
using either NumPy or PyTorch as the backend. The default backend is NumPy,
but it can be switched to PyTorch using the `set_backend` function. Other
backends may be added in the future.

Kramer Harrison, 2025
"""

import numpy as np

from optiland.backend import numpy_backend
from optiland.backend.utils import to_numpy  # noqa: F401

try:
    import torch

    from optiland.backend import torch_backend

    _torch_available = True
except ImportError:
    torch = None
    torch_backend = None
    _torch_available = False

# Registry for available backends.
_backends = {
    "numpy": numpy_backend,
}

# Add torch_backend to the registry only if it is available.
if _torch_available:
    _backends["torch"] = torch_backend

# Default backend
_current_backend = "numpy"

ndarray = (np.ndarray, torch.Tensor) if _torch_available is not None else np.ndarray


def set_backend(name: str):
    """Set the current backend.

    Args:
        name (str): The name of the backend. Must be one of the available backends.

    Raises:
        ValueError: If the backend name is not registered.
    """
    global _current_backend
    if name not in _backends:
        raise ValueError(
            f'Unknown backend "{name}". Available backends: {list_available_backends()}'
        )
    _current_backend = name


def get_backend():
    """Get the name of the current backend."""
    return _current_backend


def list_available_backends():
    """Return a list of all registered backend names."""
    return list(_backends.keys())


def __getattr__(name):
    """Dynamically retrieve attributes (functions/constants) from the current backend.

    When a user accesses an attribute (e.g., be.sin or be.pi), this function
    tries to retrieve the attribute from the current backend module. If not found,
    it attempts to retrieve the attribute from an optional `_lib` attribute of the
    backend module (allowing for a two-level backend organization).

    Raises:
        AttributeError: If the attribute is not found in the current backend.
    """
    if name in globals():
        return globals()[name]

    backend = _backends[_current_backend]

    # Direct attribute lookup in the backend module.
    try:
        return getattr(backend, name)
    except AttributeError:
        pass

    # Fallback: check the _lib submodule if it exists.
    lib = getattr(backend, "_lib", None)
    if lib:
        try:
            return getattr(lib, name)
        except AttributeError:
            pass

    raise AttributeError(
        f"The '{_current_backend}' backend (module {backend.__name__}) "
        f"has no attribute '{name}'."
    )


def __dir__():
    """
    Extend the module's directory listing to include attributes available in the
    current backend.
    """
    backend = _backends[_current_backend]
    # Combine the standard globals with the backend's attributes.
    return sorted(list(globals().keys()) + dir(backend))
