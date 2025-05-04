.. _configurable_backend:

Configurable Backend
====================

Optiland’s core computations are routed through a unified backend abstraction layer, allowing seamless use of either NumPy or PyTorch. All numerical operations, including array creation, linear algebra, trigonometric functions, and more, are automatically dispatched to the selected backend without changing your code. This design ensures:

- Consistent API across backends
- Easy GPU acceleration via PyTorch
- Full compatibility with ML/DL workflows
- Simplified maintenance and extensibility

Importing and Selecting a Backend
----------------------------------

Begin by importing the backend module instead of `numpy` or `torch` directly:

.. code:: python

   import optiland.backend as be

By default, Optiland uses the NumPy backend. To switch to PyTorch:

.. code:: python

   # for torch with CUDA support, it must be installed manually, e.g.:
   #   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
   be.set_backend("torch")

To explicitly revert to NumPy at any time:

.. code:: python

   be.set_backend("numpy")

It is generally recommended to restart the Python kernel after switching backends to avoid unexpected behavior.

NumPy Backend
-------------

The NumPy backend provides pure-CPU execution and leverages Numba JIT compilation for hotspots. Common functions (e.g., `sum`, `cos`, `sin`, `dot`, etc.) are automatically proxied from NumPy into `optiland.backend`:

.. code:: python

   # under the hood, be.sum == numpy.sum, etc.
   x = be.arange(0.0, 1.0, 0.1)    # array creation
   y = be.sin(x)                   # trigonometric
   total = be.sum(y)               # reduction

Because of automatic patching, you do not need to re‑implement any basic routines in `numpy_backend.py`.

PyTorch Backend
---------------

The PyTorch backend routes operations through `torch.*` and adds configurable device, precision, and gradient settings:

- **Device management**  
  Select CPU or CUDA:

  .. code:: python

     be.set_device("cuda")   # or "cpu"

- **Precision control**  
  Choose between single- and double‑precision floats:

  .. code:: python

     be.set_precision("float32")   # or "float64"

- **Global gradient mode**  
  Enable or disable autograd across all PyTorch operations:

  .. code:: python

     be.grad_mode.enable()    # all ops will track gradients
     be.grad_mode.disable()   # disable gradient tracking

Example usage:

.. code:: python

   import optiland.backend as be

   be.set_backend("torch")
   be.set_device("cuda")
   be.set_precision("float64")
   be.grad_mode.enable()

   # now every be.* call (e.g. be.matmul, be.exp) uses torch.cuda.FloatTensor,
   # with gradient support enabled.

Adding New Functionality
------------------------

When extending Optiland, always import operations from `optiland.backend` to ensure compatibility across both backends:

.. code:: python

   from optiland.backend import array, dot, sqrt

   def my_custom_metric(x, y):
       return dot(x, y) / sqrt(dot(x, x) * dot(y, y))

If you define a function that relies on a backend-specific feature, add it to both `numpy_backend.py` and `torch_backend.py`, following the existing patterns.

Backend Implementation Details
------------------------------

- **Dynamic function patching**  
  A metaprogramming layer inspects NumPy’s namespace and re‑exports core ufuncs and functions into the NumPy backend module automatically.
- **Explicit registration for PyTorch**  
  The PyTorch backend manually maps dozens of functions (including tensor creation, indexing, linear algebra, etc.), as well as defines custom functions not directly available in PyTorch.
- **Error handling**  
  If you call a backend operation that isn’t yet implemented in the active backend, Optiland will raise a clear `AttributeError` pointing to the missing function.

Best Practices
--------------

- **Use `be.*` everywhere**. Never import `np` or `torch` directly in Optiland modules - you’ll break backend neutrality. There are exceptions, but they are rare.
- **Test on both backends**. Our CI includes pytest fixtures that run the full test suite under both NumPy and PyTorch modes. If you add a new feature, follow existing testing patterns to ensure it works on both backends.
- **Document backend-specific behavior**. If a function has different characteristics, note it in the docstring.

Troubleshooting
---------------

- **“Module ‘torch’ not found”**  
  Ensure you have installed a PyTorch build with appropriate CUDA support (if you wish to use GPU).
- **Precision mismatches**  
  Verify `be.get_precision()` matches your expectations before heavy computations. PyTorch is strict about tensor types and will raise errors if you mix types.
- **Device mismatches**  
  Check `be.get_device()` and confirm your tensors reside on the correct device.

Potential Future Backends
-------------------------

- Support for both **JAX** and/or **CuPy** backends are under consideration. If you have experience with these libraries and would like to contribute, please reach out!

.. note::
   For new contributions or questions about the backend layer, please open an issue or pull request on our GitHub repository. We welcome feedback and improvements!
