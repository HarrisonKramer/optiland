"""
Utility functions for working with different backends and switching between them.

Kramer Harrison, 2024-2025
"""

import importlib
import sys
import numpy as np

# --- Backend Implementations ---
import optiland.backend.numpy_backend as numpy_backend
if importlib.util.find_spec("torch"):
    import optiland.backend.torch_backend as torch_backend
else:
    torch_backend = None # Indicate torch is not available


# --- Conversion Utilities ---
def torch_to_numpy(obj):
    if torch_backend and importlib.util.find_spec("torch"):
        import torch # Import locally within function if torch is available
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
    raise TypeError # Raise TypeError if not a torch tensor or torch is unavailable


CONVERTERS = [torch_to_numpy]


def to_numpy(obj):
    """Converts input scalar or array to NumPy array, regardless of backend."""
    if isinstance(obj, np.ndarray):
        return obj
    # Handle standard Python/NumPy numeric types including complex
    elif isinstance(obj, (int, float, complex, np.number)):
        # np.array handles complex numbers correctly
        return np.array(obj)
    elif isinstance(obj, list):
        processed_elements = []
        for item in obj:
            converted = to_numpy(item)
            # Extract scalar value if it's a 0-dim or 1-element array
            # np.ndarray.item() works for both real and complex numbers
            if isinstance(converted, np.ndarray) and converted.size == 1:
                processed_elements.append(converted.item())
            # Handle if it was already converted to a Python/Numpy scalar (including complex)
            elif isinstance(converted, (int, float, complex, np.number)):
                processed_elements.append(converted)
            else:
                raise TypeError(
                    f"List element conversion resulted in non-scalar or unsupported "
                    f"type: {type(converted)}"
                )
        # Determine dtype based on elements (handles mixed real/complex)
        return np.array(processed_elements)

    # Try backend-specific converters (e.g., for torch tensors)
    for converter in CONVERTERS:
        try:
            return converter(obj)
        except TypeError:
            continue

    # If no conversion worked
    raise TypeError(f"Unsupported object type for to_numpy: {type(obj)}")


# --- Backend Switching Logic ---

# Dynamically gather functions/constants from backend modules
def _gather_backend_exports(module, module_name):
    exports = {}
    # Prefer __all__ for explicit exports, otherwise gather public attributes
    export_list = []
    if hasattr(module, '__all__') and module.__all__: # Check if __all__ exists and is not empty
        export_list = module.__all__
    else:
         # Fallback: gather non-private attributes if __all__ is missing or empty
        export_list = [name for name in dir(module) if not name.startswith('_')]
        # print(f"Warning: __all__ not defined or empty for {module_name}. Using dir().") # Optional debug

    for name in export_list:
        if hasattr(module, name):
            exports[name] = getattr(module, name)
        else:
            print(f"Warning: '{name}' listed in {module_name} exports but not found in module.")

    # Ensure _lib is always included
    if hasattr(module, '_lib') and '_lib' not in exports:
        exports['_lib'] = getattr(module, '_lib')

    return exports

_FUNCS = {
    'numpy': _gather_backend_exports(numpy_backend, 'numpy_backend'),
}

if torch_backend:
    _FUNCS['torch'] = _gather_backend_exports(torch_backend, 'torch_backend')
    # Ensure 'j' from torch_backend is explicitly added if missing (e.g., if __all__ wasn't updated)
    if 'j' not in _FUNCS['torch'] and hasattr(torch_backend, 'j'):
         _FUNCS['torch']['j'] = torch_backend.j


# Ensure 'j' from numpy_backend is explicitly added if missing
if 'j' not in _FUNCS['numpy'] and hasattr(numpy_backend, 'j'):
    _FUNCS['numpy']['j'] = numpy_backend.j


_current_backend_name = 'numpy' # Default

# --- Update __all__ for utils module ---
_BASE_ALL = ['set_backend', 'to_numpy', 'be', 'CONVERTERS', 'torch_to_numpy']
__all__ = [] # Initialize __all__; will be populated by _update_utils_all

def _update_utils_all():
    """Dynamically updates the __all__ list of this utils module."""
    global __all__
    # Start with base exports
    current_all = set(_BASE_ALL)
    # Add exports from the currently active backend using _FUNCS
    if _current_backend_name in _FUNCS:
        current_all.update(_FUNCS[_current_backend_name].keys())

    # Assign the sorted list to __all__
    __all__[:] = sorted(list(current_all)) # Modify in-place

# Class to act as a namespace for the current backend's functions
class BackendModule:
    def __init__(self, name):
        # Initialize attributes to None first to prevent AttributeError during initial _update
        # for key in _FUNCS.get(name, {}).keys():
        #      setattr(self, key, None)
        self._update(name)

    def _update(self, name):
        global _current_backend_name
        if name not in _FUNCS:
             # If trying to set torch but it's not available
             if name == 'torch' and not torch_backend:
                 raise ImportError("Torch backend is not available. Please install PyTorch.")
             raise ValueError(f"Unsupported backend: {name}")

        _current_backend_name = name

        # --- Attribute Management ---
        # 1. Get all attributes currently in the instance's dict (excluding internal ones)
        current_public_attrs = {attr for attr in self.__dict__ if not attr.startswith('_')}
        # 2. Get all attributes the *new* backend should have
        new_backend_attrs = set(_FUNCS[name].keys())
        # 3. Find attributes to remove (present in current, not in new)
        attrs_to_remove = current_public_attrs - new_backend_attrs
        # 4. Find attributes to add/update (present in new)
        attrs_to_add_update = new_backend_attrs

        # Remove old attributes
        for attr in attrs_to_remove:
            try:
                delattr(self, attr)
            except AttributeError:
                 pass # Ignore if already gone

        # Add/Update new attributes
        for attr in attrs_to_add_update:
            setattr(self, attr, _FUNCS[name][attr])
        # --- End Attribute Management ---


        # Update the __all__ list for the utils module *after* 'be' instance is updated
        _update_utils_all()

    def __repr__(self):
        return f"<BackendModule: {_current_backend_name}>"

# Global instance 'be' holding the current backend's functions
be = BackendModule(_current_backend_name)

def set_backend(name: str):
    """Sets the active numerical backend ('numpy' or 'torch')."""
    global _current_backend_name # Ensure we modify the global name tracker
    if name == _current_backend_name:
        return # No change needed
    if name == 'torch' and not torch_backend:
         raise ImportError("Cannot set backend to 'torch'. PyTorch is not installed.")
    if name not in _FUNCS:
         raise ValueError(f"Unsupported backend: {name}")

    # Update the 'be' instance which triggers __all__ update
    be._update(name)
    # _current_backend_name is now updated inside be._update()


# Initial update of __all__ based on the default backend ('numpy')
_update_utils_all()
