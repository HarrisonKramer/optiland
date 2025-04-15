System Requirements
===================

Optiland is a Python-based library, and it is compatible with Python versions **3.9 through 3.13**. 

Operating System and Hardware
-----------------------------

Python is a cross-platform language, and Optiland is expected to work on Windows, macOS, and Linux systems.
Testing is performed on Linux (Ubuntu latest) and Windows 11. 

The library does not impose specific hardware requirements. However, for large-scale simulations or complex optimization tasks, we recommend using a machine with:

- A multi-core processor
- At least 8 GB of RAM
- A dedicated GPU (optional, if future extensions leverage GPU-based computations)

Dependencies
------------

The following Python libraries are required to use Optiland. These dependencies are automatically installed when you install Optiland via `pip`:

- `numpy` (for vectorized ray tracing)
- `scipy` (for optimization routines)
- `matplotlib` (for basic visualization)
- `vtk` (for advanced 3D visualization)
- `pandas` (for data manipulation)
- `pyyaml` (for reading and writing YAML files)
- `tabulate` (for tabular data formatting)
- `numba` (for just-in-time compilation)
- `requests` (for HTTP requests)
- `seaborn` (for statistical data visualization)

You can view the complete list of dependencies in the `pyproject.toml` file of the repository.
