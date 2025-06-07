Visualization Framework
=======================

The visualization framework in Optiland provides robust tools for visualizing optical systems and plotting analysis data.
It is modular, extensible, and designed for easy integration into workflows.

Optiland offers two main categories of visualization tools:

1.  **System Layout Visualization**: For rendering 2D and 3D views of the optical system itself, including lenses, mirrors, and ray paths through these components.
2.  **Generic Plotting Utilities**: For creating standard 2D and 3D plots like line charts, scatter plots, image plots, etc., which are heavily used by the analysis modules.

System Layout Visualization
---------------------------

This part of the framework allows users to analyze system components, ray paths, and overall design layouts interactively.

- **2D System Visualization**: Built with Matplotlib, this mode is ideal for quick visual representations of optical systems and ray paths.
- **3D System Visualization**: Powered by VTK, this mode provides detailed, interactive views of optical systems in three dimensions. A dark mode is available for enhanced viewing aesthetics.

**Core Classes for System Layout:**

- **OpticViewer**: Handles 2D visualization of an `Optic` instance and its components. This is the primary class for generating 2D system layout plots.

- **OpticViewer3D**: Handles 3D visualization of an `Optic` instance and its components, enabling interactive exploration of the system layout in three dimensions.

- **OpticalSystem**: Orchestrates the plotting of system components and rays for layout views. It identifies system elements such as lenses and mirrors and delegates rendering to specialized component classes.

- **Rays2D and Rays3D**: Responsible for plotting ray paths in 2D and 3D system layout views, respectively. These classes are called by the `OpticalSystem` to handle ray visualization.

Component Classes for System Layout
-----------------------------------

Each optical component, such as lenses or mirrors, has specialized classes for 2D and 3D system layout visualization:

- **Lens2D** and **Lens3D**: Deal with rendering lenses in their respective projections, taking into account geometry, thickness, and positioning, etc.

- **Surface2D** and **Surface3D**: Handle rendering of surfaces or mirrors, accounting for their orientations and coordinate systems.

Workflow for System Layout Visualization
----------------------------------------

The typical workflow for visualizing an optical system's layout is as follows:

1. **Initialize Viewer**: Create an instance of `OpticViewer` or `OpticViewer3D` and provide an `Optic` instance.

2. **Customize if Needed**: While users typically interact with these high-level viewers, the framework supports fine-grained customization by extending component classes or altering plotting parameters.

3. **Render the Plot**: Generate the desired system layout visualization, whether in 2D or 3D, using the `view` method of the viewer. For 3D plots, additional options like dark mode can be applied for enhanced aesthetics.

General Purpose Plotting with Plotter
-------------------------------------

For general-purpose 2D and 3D plotting needs, such as line charts, scatter plots, image displays (heatmaps), surface plots, etc., Optiland provides the `optiland.plotting.Plotter` class. This is the primary interface used by most analysis modules (e.g., Ray Fan, Spot Diagram, MTF plots) and is recommended for developers needing to create custom visualizations of data.

Key features of the `optiland.plotting.Plotter` and its associated modules (`optiland.plotting.config`, `optiland.plotting.themes`, `optiland.plotting.plot_configs`):

*   **Versatile Static Methods**: `Plotter` offers a suite of static methods like `Plotter.plot_line()`, `Plotter.plot_scatter()`, `Plotter.plot_image()`, `Plotter.plot_subplots()`, and their 3D counterparts.
*   **Theming**: Apply consistent visual styles across all plots using predefined or custom themes managed by `optiland.plotting.themes`.
*   **Global Configuration**: Customize default plot appearances (e.g., font sizes, figure size) system-wide via `optiland.plotting.config`.
*   **Legend Customization**: Use the `LegendConfig` TypedDict from `optiland.plotting.plot_configs` to pass detailed legend styling overrides to `Plotter` methods.
*   **Flexible Figure/Axes Handling**: Plotting methods can either display plots directly or return the Matplotlib `Figure` and `Axes` objects for further customization, controlled by a `return_fig_ax` parameter or global settings.

**Workflow for General Plotting:**

1.  **Import Plotter**: `from optiland.plotting import Plotter`.
2.  **Prepare Data**: Gather your data (e.g., from analysis results, calculations).
3.  **Call Plotter Method**: Use the appropriate static method on `Plotter`, e.g., `Plotter.plot_line(x_data, y_data, title="My Data")`.
4.  **Customize (Optional)**:
    *   Set a global theme: `themes.set_active_theme('dark')`.
    *   Modify global configurations: `config.set_config('font.size_title', 18)`.
    *   Pass a `LegendConfig` for specific legend styling.
    *   Request `fig, ax` objects by passing `return_fig_ax=True` for further manual adjustments.

Refer to the API documentation for `optiland.plotting.core.Plotter` and `optiland.plotting.plot_configs.LegendConfig` for detailed usage of each plotting method and configuration options.

Dark Mode for VTK (System Layout)
---------------------------------

The `OpticViewer3D` class includes an optional **dark mode** for 3D plots, offering a modern and visually appealing interface for exploring optical systems.

.. note::
   The image in the :ref:`developers_guide` shows an example of the dark mode 3D visualization output for a lithographic lens system.
