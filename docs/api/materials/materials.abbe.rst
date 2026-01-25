materials.abbe
==============

.. automodule:: materials.abbe

   
   .. rubric:: Classes

   .. autosummary::
   
      AbbeMaterial

Model Details
-------------

The `AbbeMaterial` class supports two underlying models:

1.  **"polynomial" (Legacy)**: Uses a polynomial fit to Schott glass data. This is the default in versions < 0.7.0.
2.  **"buchdahl" (New)**: Uses a Buchdahl 3-term model with coefficients derived from LASSO regression on a large glass dataset. This model is more accurate and will become the default in v0.7.0.

For a detailed explanation of the derivation of these models, please refer to the following notebook:

*   :doc:`../../references/AbbeMaterial_Model_Building`
   