Optiland File Format
====================

The Optiland file format is a **JSON-based format** designed for saving and loading optical systems in a human-readable
and easily extensible way. This format captures the full state of an `Optic` instance, including all of its attributes and
associated data.

Overview
--------

Optiland leverages the flexibility of JSON to represent optical systems in a nested structure.
This structure corresponds to the internal organization of the `Optic` class and its associated components.

Key Features
------------

- **Human-Readable**: JSON is a plain-text format, making it easy to inspect and modify files manually when needed.
- **Comprehensive**: The format includes all properties of the `Optic` instance, such as:
  - Wavelengths & polarization state
  - Fields
  - Aperture
  - Surface groups, including all surfaces and their respective properties
  - Pickups and solves
- **Interoperability**: Optiland files can be generated from Zemax `.zmx` files, enabling easy migration of designs into Optiland.
- **Extensible**: Additional properties or components can be serialized without altering the core structure, allowing the format to evolve alongside Optiland.

Core Functionality
------------------

Serialization and deserialization of Optiland files are managed through the `optiland.fileio` module. This module provides two primary functions:

- **save_optiland_file(optic, filepath)**: Saves an `Optic` instance to a JSON file.
- **load_optiland_file(filepath)**: Loads an `Optic` instance from a JSON file.

These functions ensure consistency between the internal state of the `Optic` instance and its representation in the JSON format.

Implementation
--------------

The Optiland file format is implemented through the recursive use of `to_dict` and `from_dict` methods in the `Optic` class and its
associated components. These methods convert the internal structure of an `Optic` instance into a nested dictionary format that can be
directly serialized to JSON.

Example Workflow
----------------

1. **Saving a File**:

.. code:: python

   from optiland.fileio import save_optiland_file

   # assumes an Optic instance named 'optic' is defined
   save_optiland_file(optic, "example_design.json")

2. **Loading a File**:

.. code:: python

   from optiland.fileio import load_optiland_file

   optic = load_optiland_file("example_design.json")

Interoperability with Zemax
---------------------------

Optiland can generate its file format from Zemax .zmx files. This feature simplifies transitioning optical
designs from Zemax to Optiland, preserving the integrity of system properties. Not all Zemax features are supported in
the Optiland file format, including some geometry and field types. However, the core optical properties are maintained
during the conversion process. If you experience any unexpected issues during the conversion process, please kindly report
them as issues on the Optiland GitHub repository.

Future Extensions
-----------------

The JSON-based format allows for straightforward extensions. As Optiland evolves, new attributes and components can be
incorporated into the format without breaking compatibility with existing files.
