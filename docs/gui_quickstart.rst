.. _gui_quickstart:

Optiland GUI Quickstart
=======================

Welcome to the Optiland Graphical User Interface (GUI)! This guide will help you get started with the basic features for interactive optical design and analysis.

What can the GUI do?
--------------------

The Optiland GUI provides a user-friendly way to:

*   Create new optical systems from scratch.
*   Load existing lens files (Optiland JSON format, and import from other formats like Zemax .zmx).
*   Visually inspect optical systems in 2D and 3D.
*   Edit system parameters, including surface data (radius, thickness, material, etc.), aperture, field points, and wavelengths.
*   Run various optical analyses (e.g., ray fans, spot diagrams, MTF).
*   View analysis results graphically.

Launching the GUI
-----------------

You can launch the Optiland GUI in a couple of ways:

1.  **From the command line (recommended for most users):**
    Open your terminal or console and type:

    .. code-block:: bash

       optiland

2.  **From Python (useful for development or troubleshooting):**
    Open your terminal or console and type:

    .. code-block:: bash

       python -m optiland_gui.run_gui

Main Interface Components
-------------------------

When you first open the Optiland GUI, you'll see a main window containing several panels. Here's a brief overview:

.. image:: _static/gui_overview.png
   :alt: Optiland GUI Overview
   :align: center
   :width: 600px

   *Placeholder for a general overview screenshot of the GUI.*

*   **Main Window**: Contains the main menu bar (File, Edit, View, Tools, Help), toolbars for quick actions, and manages the different panels.
*   **Lens Editor Panel**: This is where you view and modify the surface-by-surface data of your optical system, such as radius, thickness, material, conic constants, and semi-diameters. Changes made here are reflected in other panels.

    .. image:: _static/gui_lens_editor.png
       :alt: Lens Editor Panel
       :align: center
       :width: 400px

       *Placeholder for Lens Editor screenshot.*

*   **Viewer Panel**: This panel provides visual representations of your optical system.
    *   **2D View**: Shows a 2D cross-section of the lens, with options to display rays.
    *   **3D View**: Renders a 3D model of the system (if VTK is installed and working).

    .. image:: _static/gui_viewer_panel.png
       :alt: Viewer Panel (2D/3D)
       :align: center
       :width: 400px

       *Placeholder for Viewer Panel screenshot.*

*   **Analysis Panel**: Allows you to select, configure, and run various optical analyses. Results are typically displayed as plots within this panel.

    .. image:: _static/gui_analysis_panel.png
       :alt: Analysis Panel
       :align: center
       :width: 400px

       *Placeholder for Analysis Panel screenshot.*

*   **System Properties Panel**: Manage system-wide settings that are not tied to individual surfaces. This includes:
    *   **Aperture**: Define the system aperture (e.g., Entrance Pupil Diameter, F-number).
    *   **Fields**: Set up field points for analysis.
    *   **Wavelengths**: Define the wavelengths and their weights for calculations.

    .. image:: _static/gui_system_properties.png
       :alt: System Properties Panel
       :align: center
       :width: 400px

       *Placeholder for System Properties screenshot.*

*   **Sidebar**: Located on the left, it provides quick navigation to show/hide the main panels like Lens Editor, Viewer, Analysis, etc.
*   **Python Terminal** (View > Python Terminal): An embedded IPython terminal for advanced users who want to interact with the optical system programmatically using Optiland's Python API.

Getting Started: Basic Actions
------------------------------

Let's try a few basic operations.

### 1. Opening an Existing Lens File

Optiland supports its native JSON format (`.json`) and can import other formats. Many sample files are also provided.

*   Go to the menu: **File > Open Sample > Cooke Triplet**.
*   The Cooke Triplet lens system will load, and you should see its data in the Lens Editor and a 2D/3D representation in the Viewer Panel.

### 2. Viewing a Raytrace

With the Cooke Triplet loaded:

*   In the **Viewer Panel**, ensure the **2D View** tab is selected.
*   You may see some default rays already traced. If not, or to customize:
    *   Look for ray tracing controls within the 2D View tab (e.g., a "Trace Rays" button or options to select number of rays and field points).
    *   Click to trace or update the ray display.
*   Switch to the **3D View** tab in the Viewer Panel to see the lens and rays in 3D. You can rotate, pan, and zoom this view.

    .. image:: _static/gui_raytrace_example.png
       :alt: Example of a Raytrace in the GUI
       :align: center
       :width: 450px

       *Placeholder for a raytrace example screenshot.*

### 3. Changing a Surface Parameter

Let's modify a surface and see the update:

*   In the **Lens Editor Panel**, find the row for **Surface 2** (the second optical surface).
*   Click on the cell containing its **Radius** value.
*   Change the value (e.g., from -435.76 to -300) and press **Enter**.
*   Observe how the 2D and 3D views in the **Viewer Panel** update to reflect this change. You might also see changes in analysis results if any were open.

### 4. Creating a New System

*   To start a new, empty optical system, go to **File > New**.
*   You can then add surfaces using the buttons or context menus in the Lens Editor and define system properties.

Explore Further
---------------

This quickstart covered only the very basics. The Optiland GUI has many more features for detailed optical design and analysis. We encourage you to explore the menus, right-click options in different panels, and consult the other sections of the Optiland documentation for more in-depth information on specific functionalities.
