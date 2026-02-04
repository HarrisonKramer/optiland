Materials
=========

This section provides a comprehensive overview of the materials available in Optiland.
A material is a set of properties that define the optical behavior of a medium. Specifically in Optiland, a material defines the refractive
index and extinction coefficient of a medium at any wavelength. A material may be described simply by refractive index and abbe number, or it may be a more complex
model based on Sellmeier, or other, coefficients. Optiland provides a wide range of materials, which are outlined here.

Optiland includes a database of common materials based on refrativeindex.info. Any material in this database
can be accessed by name via the Material class.

Abbe Material Model
-------------------

The :class:`~optiland.materials.abbe.AbbeMaterial` class utilizes a robust, data-driven statistical dispersion model to resolve the ambiguity of the two-parameter ($n_d, V_d$) definition. While standard approximations like the "Normal Line" rule often fail for anomalous glasses, Optiland's model is derived from a principal component analysis (PCA) of over 1,000 commercial optical glasses.

The model construction involves:
1.  **Dimensionality Reduction:** Using PCA to quantify the effective degrees of freedom in standard optical glasses.
2.  **Basis Selection:** Applying Sparse Regression (LassoLarsIC) to the Buchdahl dispersion formula to identify the minimum set of coefficients required for accurate spectral reconstruction.

This approach allows for accurate refractive index prediction across the visible and near-infrared spectrum, even for glasses that deviate from the standard "normal line".

For a detailed walkthrough of the model derivation and validation, please refer to the :doc:`Abbe Material Model Building <../references/AbbeMaterial_Model_Building>` notebook.

.. autosummary::
   :toctree: materials/
   :caption: Material Modules

   materials.abbe
   materials.ideal
   materials.material_file
   materials.material
