This learning guide breaks down Optiland into a series of tutorials that cover the basic concepts and functionalities of the package. Each tutorial is a self-contained Jupyter notebook that demonstrates how to perform a specific task using Optiland. The tutorials are designed to be followed sequentially, starting with an introduction to Optiland and progressing to more advanced topics.


1. Introduction to Optiland
---------------------------

.. toctree::
    :maxdepth: 1

   examples/Tutorial_1a_Optiland_for_Beginners
   examples/Tutorial_1b_Lens_Properties
   examples/Tutorial_1c_Save_and_Load_Files
   examples/Tutorial_1d_Material_Database
   examples/Tutorial_1e_Non_Rotationally_Symmetric_Systems


2. Real Raytracing & Analysis
------------------------------

.. toctree::
    :maxdepth: 1

   examples/Tutorial_2a_Tracing_&_Analyzing_Rays
   examples/Tutorial_2b_Tilting_&_Decentering_Components
   examples/Tutorial_2c_Monte_Carlo_Raytracing
   examples/Tutorial_2d_Raytracing_Aspheres


3. Aberrations
--------------

.. toctree::
    :maxdepth: 1

    examples/Tutorial_3a_Common_Aberration_Analyses
    examples/Tutorial_3b_First_&_Third_Order_Aberrations
    examples/Tutorial_3c_Chromatic_Aberrations


4. OPD, PSF, & MTF
------------------

.. toctree::
    :maxdepth: 1

    examples/Tutorial_4a_Optical_Path_Difference_Calculation
    examples/Tutorial_4b_PSF_&_MTF_Calculation
    examples/Tutorial_4c_Zernike_Decomposition


5. Optimization
---------------

.. toctree::
    :maxdepth: 1

    examples/Tutorial_5a_Simple_Optimization
    examples/Tutorial_5b_Advanced_Optimization
    examples/Tutorial_5c_Optimization_Case_Study
    examples/Tutorial_5d_User_Defined_Optimization


6. Coatings & Polarization
--------------------------

.. toctree::
    :maxdepth: 1

    examples/Tutorial_6a_Introduction_to_Coatings
    examples/Tutorial_6b_Introduction_to_Polarization


7. Advanced Optical Design
--------------------------

.. toctree::
    :maxdepth: 1

    examples/Tutorial_7a_Lithographic_Projection_System
    examples/Tutorial_7b_Surface_Roughness_&_Scattering
    examples/Tutorial_7c_Freeform_Surfaces
    examples/Tutorial_7d_Three_Mirror_Anastigmat


8. Tolerancing
--------------

.. toctree::
    :maxdepth: 1

    examples/Tutorial_8a_Tolerancing_Sensitivity_Analysis
    examples/Tutorial_8b_Monte_Carlo_Analysis


9. Lens Catalogue Integration
-----------------------------

.. toctree::
    :maxdepth: 1

    examples/Tutorial_9a_Edmund_Optics_Catalogue
    examples/Tutorial_9b_Thorlabs_Catalogue

10. Extending Optiland
----------------------

.. toctree::
    :maxdepth: 1

    examples/Tutorial_10a_Custom_Surface_Types
    examples/Tutorial_10b_Custom_Coating_Types
    examples/Tutorial_10c_Custom_Optimization_Algorithm

11. Machine Learning in Optical Design
--------------------------------------

These examples demonstrate how Optiland can be used in conjunction with machine learning to solve optical design problems.
Concepts in machine and deep learning are also covered.
Note that these are hosted on the `LensAI repository <https://github.com/HarrisonKramer/LensAI>`_.

.. toctree::
    :maxdepth: 1

    examples/Singlet_RF_Model_RMS_Spot_Size
    examples/Ray_Path_Failure_Classification_Model
    examples/Double_Gauss_Surrogate_Model
    examples/SR_GAN_for_wavefront_data
    examples/RL_aspheric_singlet
    examples/Misalignment_Prediction_Cooke_Triplet
