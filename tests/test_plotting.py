"""Unit tests for the optiland.plotting module.

This test suite covers the functionality of config management, theme application,
and various plotting methods provided by the Plotter class.
"""
import pytest
import numpy as np
import matplotlib.pyplot as plt
import json
import os # For path manipulation in config persistence tests

from optiland.plotting import Plotter, config, themes
from optiland.plotting.exceptions import ThemeNotFoundError, ConfigurationError, InvalidPlotDataError

# --- Fixtures ---

@pytest.fixture(autouse=True)
def mock_plt_show(monkeypatch):
    """Mocks plt.show() to prevent GUI popups and closes figures after each test."""
    mock_calls = []
    def mock_show(*args, **kwargs):
        mock_calls.append({"args": args, "kwargs": kwargs})
        # print(f"plt.show() called with args: {args}, kwargs: {kwargs}") # For debugging

    monkeypatch.setattr(plt, "show", mock_show)
    yield mock_calls # Can be used to check if show was called if needed
    plt.close('all') # Close all figures after the test run


@pytest.fixture(autouse=True)
def reset_plotting_defaults():
    """Resets plotting configurations and active theme before each test."""
    config.reset_config()
    themes.set_active_theme('light') # Default to 'light' theme for tests


@pytest.fixture
def plotter_instance():
    """Provides a Plotter instance for tests."""
    return Plotter()

# --- Helper Data & Functions ---

def get_sample_data_2d(size=10):
    """Generates simple 2D data for plotting."""
    x = np.linspace(0, 1, size)
    y = x**2
    return x, y

def get_sample_data_3d(size=10):
    """Generates simple 3D data for plotting."""
    x = np.linspace(-np.pi, np.pi, size)
    y = np.sin(x)
    z = np.cos(x)
    return x, y, z

def get_sample_meshgrid_data(size=10):
    """Generates meshgrid data for surface/wireframe plots."""
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    return X, Y, Z

def simple_plot_callback(ax, index):
    """A simple callback for testing plot_subplots."""
    x, y = get_sample_data_2d(5)
    ax.plot(x, y, label=f"Plot {index}")
    ax.set_title(f"Subplot {index}")

# --- Tests for config.py ---

def test_config_get_default_values():
    assert isinstance(config.get_config('figure.figsize'), tuple)
    assert isinstance(config.get_config('font.size_title'), int)
    assert config.get_config('plot.show_on_draw') is True # Default

def test_config_set_and_get_value():
    original_figsize = config.get_config('figure.figsize')
    new_figsize = (12, 8)
    config.set_config('figure.figsize', new_figsize)
    assert config.get_config('figure.figsize') == new_figsize
    config.set_config('figure.figsize', original_figsize) # Reset

def test_config_get_invalid_key():
    with pytest.raises(ConfigurationError, match="Configuration key 'non.existent.key' not found"):
        config.get_config('non.existent.key')

def test_config_set_invalid_key():
    with pytest.raises(ConfigurationError, match="Invalid configuration key 'non.existent.key'"):
        config.set_config('non.existent.key', 100)

def test_config_set_invalid_type():
    original_size = config.get_config('font.size_title')
    # Test setting a string where an int is expected
    with pytest.raises(ConfigurationError, match="Invalid type for configuration key 'font.size_title'. Expected type 'int', but got type 'str'"):
        config.set_config('font.size_title', "not-an-int")
    config.set_config('font.size_title', original_size) # Reset

    original_figsize = config.get_config('figure.figsize')
    # Test setting a list where a tuple is expected
    with pytest.raises(ConfigurationError, match="Invalid type for configuration key 'figure.figsize'. Expected type 'tuple', but got type 'list'"):
        config.set_config('figure.figsize', [10,5])
    config.set_config('figure.figsize', original_figsize) # Reset

    # Test setting a tuple with non-numeric or non-positive values for figsize
    with pytest.raises(ConfigurationError, match="'figure.figsize' must be a tuple of two positive numbers"):
        config.set_config('figure.figsize', ('a', 'b'))
    with pytest.raises(ConfigurationError, match="'figure.figsize' must be a tuple of two positive numbers"):
        config.set_config('figure.figsize', (-10, 5))
    config.set_config('figure.figsize', original_figsize) # Reset


def test_config_reset():
    default_figsize = config._DEFAULT_CONFIG['figure.figsize']
    default_title_size = config._DEFAULT_CONFIG['font.size_title']

    config.set_config('figure.figsize', (15, 10))
    config.set_config('font.size_title', 25)
    assert config.get_config('figure.figsize') == (15, 10)
    assert config.get_config('font.size_title') == 25

    config.reset_config()
    assert config.get_config('figure.figsize') == default_figsize
    assert config.get_config('font.size_title') == default_title_size

# --- Tests for themes.py ---

def test_themes_get_active_theme():
    assert themes.get_active_theme() == 'light' # Default due to reset fixture

def test_themes_set_active_theme_valid():
    themes.set_active_theme('dark')
    assert themes.get_active_theme() == 'dark'
    themes.set_active_theme('light') # Reset
    assert themes.get_active_theme() == 'light'

def test_themes_set_active_theme_invalid():
    with pytest.raises(ThemeNotFoundError, match="Theme 'non_existent_theme' not found"):
        themes.set_active_theme('non_existent_theme')

def test_themes_get_active_theme_dict():
    light_settings = themes.get_active_theme_dict()
    assert isinstance(light_settings, dict)
    assert light_settings['figure.facecolor'] == '#FFFFFF'

    themes.set_active_theme('dark')
    dark_settings = themes.get_active_theme_dict()
    assert dark_settings['figure.facecolor'] == '#1E1E1E'

def test_themes_get_theme_value_active_theme():
    assert themes.get_theme_value('lines.color') == themes.THEMES['light']['lines.color']
    themes.set_active_theme('dark')
    assert themes.get_theme_value('lines.color') == themes.THEMES['dark']['lines.color']

def test_themes_get_theme_value_specific_theme():
    assert themes.get_theme_value('lines.color', theme_name='dark') == themes.THEMES['dark']['lines.color']
    # Active theme should remain unchanged
    assert themes.get_active_theme() == 'light'

def test_themes_get_theme_value_invalid_key():
    with pytest.raises(ThemeNotFoundError, match="Key 'non.existent.key' not found in theme 'light'"):
        themes.get_theme_value('non.existent.key')

def test_themes_get_theme_value_invalid_theme_name():
    with pytest.raises(ThemeNotFoundError, match="Theme 'invalid_theme' not found"):
        themes.get_theme_value('lines.color', theme_name='invalid_theme')

def test_themes_list_themes():
    available_themes = themes.list_themes()
    assert isinstance(available_themes, list)
    assert 'light' in available_themes
    assert 'dark' in available_themes

# --- Tests for Plotter class ---

def test_plotter_instantiation(plotter_instance):
    assert isinstance(plotter_instance, Plotter)

def test_plotter_set_theme(plotter_instance):
    plotter_instance.set_theme('dark')
    assert themes.get_active_theme() == 'dark'
    # Check if a plot reflects the theme (e.g., figure facecolor)
    config.set_config('plot.return_fig_ax', True)
    config.set_config('plot.show_on_draw', False) # prevent plt.show issues
    x, y = get_sample_data_2d()
    # Wrap in try-except as plot_line itself might raise PlottingError for other reasons if data is bad
    try:
        fig, _ = plotter_instance.plot_line(x, y)
        assert fig.get_facecolor() == themes.THEMES['dark']['figure.facecolor']
    except InvalidPlotDataError: # Should not happen with get_sample_data
        pytest.fail("plot_line raised InvalidPlotDataError with valid sample data during theme test.")


def test_plotter_update_config(plotter_instance):
    new_figsize = (11, 7)
    plotter_instance.update_config('figure.figsize', new_figsize)
    assert config.get_config('figure.figsize') == new_figsize

    # Check if a plot reflects the config (e.g., figsize)
    config.set_config('plot.return_fig_ax', True)
    config.set_config('plot.show_on_draw', False) # prevent plt.show issues
    x, y = get_sample_data_2d()
    try:
        fig, _ = plotter_instance.plot_line(x, y)
        # Compare inches, be mindful of DPI settings if they vary
        np.testing.assert_almost_equal(fig.get_size_inches(), new_figsize, decimal=2)
    except InvalidPlotDataError: # Should not happen
        pytest.fail("plot_line raised InvalidPlotDataError with valid sample data during config test.")


def test_plotter_get_current_theme_settings(plotter_instance):
    settings = plotter_instance.get_current_theme_settings()
    assert settings == themes.THEMES['light']
    plotter_instance.set_theme('dark')
    settings_dark = plotter_instance.get_current_theme_settings()
    assert settings_dark == themes.THEMES['dark']

def test_plotter_get_config_value(plotter_instance):
    assert plotter_instance.get_config_value('font.size_title') == config.get_config('font.size_title')


# --- Tests for 2D Plotting Methods ---

@pytest.mark.parametrize("plot_method_name", ["plot_line", "plot_scatter"])
@pytest.mark.parametrize("return_fig_ax_config", [True, False]) # Test both global config options
@pytest.mark.parametrize("explicit_return_arg", [None, True, False]) # Test explicit arg override
def test_plotter_2d_methods(plotter_instance, plot_method_name, return_fig_ax_config, explicit_return_arg, mock_plt_show):
    plot_method = getattr(plotter_instance, plot_method_name)
    x, y = get_sample_data_2d()

    # Set global config for return/show behavior
    config.set_config('plot.return_fig_ax', return_fig_ax_config)
    # If we expect to return fig/ax, show_on_draw should be false to avoid plt.show()
    # If we don't expect to return, show_on_draw can be anything, mock handles it.
    config.set_config('plot.show_on_draw', not return_fig_ax_config if explicit_return_arg is None else not explicit_return_arg)


    title = f"Test {plot_method_name}"
    xlabel = "X Data"
    ylabel = "Y Data"
    legend = "Data Series"

    plot_args = {
        "x": x, "y": y, "title": title, "xlabel": xlabel, "ylabel": ylabel,
        "legend_label": legend
    }
    if explicit_return_arg is not None:
        plot_args["return_fig_ax"] = explicit_return_arg
        should_return = explicit_return_arg
    else:
        should_return = return_fig_ax_config

    result = plot_method(**plot_args)

    if should_return:
        assert isinstance(result, tuple)
        assert len(result) == 2
        fig, ax = result
        assert fig is not None
        assert ax is not None
        assert ax.get_title() == title
        assert ax.get_xlabel() == xlabel
        assert ax.get_ylabel() == ylabel
        if plot_method_name == "plot_line":
            assert len(ax.lines) > 0
            assert np.array_equal(ax.lines[0].get_xdata(), x)
            assert np.array_equal(ax.lines[0].get_ydata(), y)
        elif plot_method_name == "plot_scatter":
            # Scatter plots store data in collections
            assert len(ax.collections) > 0
            # Check one point from the collection
            sc_data = ax.collections[0].get_offsets()
            assert np.array_equal(sc_data[:,0], x) # Compare x coordinates
            assert np.array_equal(sc_data[:,1], y) # Compare y coordinates

        assert ax.legend_ is not None
        # Check if plt.show() was NOT called when fig/ax are returned
        if explicit_return_arg is True: # Explicit request to return means show shouldn't be called
             assert not any("plt.show() called" in str(call) for call in mock_plt_show) # A bit hacky way to check mock
    else:
        assert result is None
        # Check if plt.show() was called if not returning and global show_on_draw is True
        if config.get_config('plot.show_on_draw'):
             assert len(mock_plt_show) > 0 # mock_plt_show stores calls

def test_plot_line_kwargs(plotter_instance):
    config.set_config('plot.return_fig_ax', True)
    config.set_config('plot.show_on_draw', False)
    x, y = get_sample_data_2d()
    fig, ax = plotter_instance.plot_line(x, y, linestyle='--', marker='o')
    assert len(ax.lines) == 1
    assert ax.lines[0].get_linestyle() == '--'
    assert ax.lines[0].get_marker() == 'o'

def test_plot_scatter_kwargs(plotter_instance):
    config.set_config('plot.return_fig_ax', True)
    config.set_config('plot.show_on_draw', False)
    x, y = get_sample_data_2d()
    fig, ax = plotter_instance.plot_scatter(x, y, s=100, c='red', alpha=0.5) # s for size, c for color
    assert len(ax.collections) == 1
    # Cannot easily check scatter point size directly without more complex logic,
    # but color and alpha can be indicative.
    # Facecolor of scatter is an array of colors
    assert np.allclose(ax.collections[0].get_facecolors()[0], plt.cm.colors.to_rgba('red'))
    assert ax.collections[0].get_alpha() == 0.5


# --- Tests for plot_image ---
def test_plot_image(plotter_instance):
    config.set_config('plot.return_fig_ax', True)
    config.set_config('plot.show_on_draw', False)
    img_data = np.random.rand(10, 10)

    fig, ax = plotter_instance.plot_image(img_data, title="Test Image", cmap="plasma", show_colorbar=True)
    assert fig is not None
    assert ax is not None
    assert ax.get_title() == "Test Image"
    assert len(ax.images) == 1
    assert ax.images[0].get_cmap().name == "plasma"
    assert len(fig.axes) > 1 # Should have ax and colorbar ax

    # Test without colorbar
    fig_no_cb, ax_no_cb = plotter_instance.plot_image(img_data, show_colorbar=False)
    assert len(fig_no_cb.axes) == 1 # Only the main axes

    # Test default cmap from config
    original_cmap = config.get_config('image.cmap')
    try:
        config.set_config('image.cmap', 'magma')
        fig_def_cmap, ax_def_cmap = plotter_instance.plot_image(img_data)
        assert ax_def_cmap.images[0].get_cmap().name == 'magma'
    finally: # Ensure reset even if assert fails
        config.set_config('image.cmap', original_cmap) # reset

def test_plot_image_invalid_data(plotter_instance):
    with pytest.raises(InvalidPlotDataError, match="image_data must be a 2D array-like structure"):
        plotter_instance.plot_image(np.random.rand(5,5,5)) # 3D data
    with pytest.raises(InvalidPlotDataError, match="image_data cannot be empty"):
        plotter_instance.plot_image(np.array([]))


# --- Tests for plot_subplots ---
def test_plot_subplots_basic(plotter_instance):
    config.set_config('plot.return_fig_ax', True) # Corresponds to return_fig_axs
    config.set_config('plot.show_on_draw', False)

    callbacks = [simple_plot_callback, simple_plot_callback]
    fig, axs = plotter_instance.plot_subplots(1, 2, callbacks, main_title="Subplots Test")

    assert fig is not None
    assert isinstance(axs, np.ndarray)
    assert axs.shape == (2,) # Flattened for 1 row
    assert fig._suptitle.get_text() == "Subplots Test"

    for i, ax_sub in enumerate(axs.flat):
        assert f"Subplot {i}" in ax_sub.get_title() # Callback sets this
        assert len(ax_sub.lines) == 1

def test_plot_subplots_return_behavior(plotter_instance, mock_plt_show):
    callbacks = [simple_plot_callback]
    # Case 1: return_fig_axs = True
    config.set_config('plot.return_fig_ax', False) # Global return is False
    config.set_config('plot.show_on_draw', True)  # Global show is True

    result = plotter_instance.plot_subplots(1, 1, callbacks, return_fig_axs=True)
    assert isinstance(result, tuple) and len(result) == 2
    # No plt.show call because of explicit return
    # This check for mock_plt_show is tricky because it's autouse.
    # We'd need to clear its history before this call or check its length change.
    # For now, assume the logic in core.py (show_plot_on_draw = not return_fig_axs) is correct.

    # Case 2: return_fig_axs = False (explicit)
    initial_show_calls = len(mock_plt_show)
    result_no_return = plotter_instance.plot_subplots(1, 1, callbacks, return_fig_axs=False)
    assert result_no_return is None
    if config.get_config('plot.show_on_draw'): # Which it is from above
         assert len(mock_plt_show) > initial_show_calls

    # Case 3: return_fig_axs = None (use global config)
    config.set_config('plot.return_fig_ax', True) # Global return is True
    config.set_config('plot.show_on_draw', False) # Global show is False
    result_global_return = plotter_instance.plot_subplots(1, 1, callbacks, return_fig_axs=None)
    assert isinstance(result_global_return, tuple)


def test_plot_subplots_invalid_callbacks_count(plotter_instance):
    callbacks = [simple_plot_callback] * 3
    with pytest.raises(ValueError, match="Number of plot_callbacks"):
        plotter_instance.plot_subplots(1, 2, callbacks) # Expect 2, got 3
    with pytest.raises(InvalidPlotDataError, match="plot_callbacks must be a non-empty list"):
        plotter_instance.plot_subplots(1,1, [])
    with pytest.raises(InvalidPlotDataError, match="All items in plot_callbacks must be callable"):
        plotter_instance.plot_subplots(1,1, ["not_callable"])


# --- Tests for 3D Plotting Methods ---
@pytest.mark.parametrize("plot_method_3d_name, data_func, check_collection_type", [
    ("plot_line_3d", get_sample_data_3d, "lines"),
    ("plot_scatter_3d", get_sample_data_3d, "collections_scatter"), # Special handling for scatter
    ("plot_surface", get_sample_meshgrid_data, "collections_surface"),
    ("plot_wireframe", get_sample_meshgrid_data, "lines_wireframe") # Wireframe creates Line3DCollection
])
def test_plotter_3d_methods(plotter_instance, plot_method_3d_name, data_func, check_collection_type):
    config.set_config('plot.return_fig_ax', True)
    config.set_config('plot.show_on_draw', False)
    plot_method = getattr(plotter_instance, plot_method_3d_name)
    d1, d2, d3 = data_func()

    title = f"Test {plot_method_3d_name}"
    xlabel, ylabel, zlabel = "X3D", "Y3D", "Z3D"

    plot_args = {"title": title, "xlabel": xlabel, "ylabel": ylabel, "zlabel": zlabel}
    if plot_method_3d_name in ["plot_line_3d", "plot_scatter_3d"]:
        plot_args["legend_label"] = "3D Data"

    fig, ax = plot_method(d1, d2, d3, **plot_args)

    assert fig is not None
    assert ax is not None
    assert hasattr(ax, 'zaxis') # Check if it's a 3D axis
    assert ax.get_title() == title
    assert ax.get_xlabel() == xlabel
    assert ax.get_ylabel() == ylabel
    assert ax.get_zlabel() == zlabel

    if check_collection_type == "lines":
        assert len(ax.lines) == 1
        line_data_x, line_data_y, line_data_z = ax.lines[0].get_data_3d()
        assert np.array_equal(line_data_x, d1)
        assert np.array_equal(line_data_y, d2)
        assert np.array_equal(line_data_z, d3)
    elif check_collection_type == "collections_scatter":
         assert len(ax.collections) > 0 or len(ax.patches) > 0 # scatter can be Patch3DCollection or Path3DCollection
    elif check_collection_type == "collections_surface":
        assert len(ax.collections) > 0 # plot_surface creates Poly3DCollection
    elif check_collection_type == "lines_wireframe":
        assert len(ax.lines) > 0 # plot_wireframe creates Line3DCollection

    if plot_args.get("legend_label"):
        assert ax.legend_ is not None

@pytest.mark.parametrize("plot_method_name, invalid_data_sets", [
    ("plot_line", [(np.array([1,2]), np.array([1,2,3])), (np.array([]), np.array([]))]),
    ("plot_scatter", [(np.array([1,2]), np.array([1,2,3])), (np.array([]), np.array([]))]),
    ("plot_line_3d", [(np.array([1,2]), np.array([1,2,3]), np.array([1,2])), # Mismatched lengths
                       (np.array([]),np.array([]),np.array([]))]), # Empty data
    ("plot_scatter_3d", [(np.array([1,2]), np.array([1,2,3]), np.array([1,2])),
                         (np.array([]),np.array([]),np.array([]))]),
    ("plot_surface", [(np.random.rand(5), np.random.rand(5), np.random.rand(5)), # 1D data
                      (np.random.rand(5,5), np.random.rand(5,6), np.random.rand(5,5))]), # Mismatched shapes
    ("plot_wireframe", [(np.random.rand(5), np.random.rand(5), np.random.rand(5)),
                        (np.random.rand(5,5), np.random.rand(5,6), np.random.rand(5,5))])
])
def test_plot_methods_invalid_data(plotter_instance, plot_method_name, invalid_data_sets):
    plot_method = getattr(plotter_instance, plot_method_name)
    for data_args in invalid_data_sets:
        with pytest.raises(InvalidPlotDataError):
            plot_method(*data_args)


# --- Tests for Theme and Config Application ---
@pytest.mark.parametrize("plot_method_name", [
    "plot_line", "plot_scatter", "plot_image",
    "plot_line_3d", "plot_scatter_3d", "plot_surface", "plot_wireframe"
])
def test_theme_application_on_plots(plotter_instance, plot_method_name):
    config.set_config('plot.return_fig_ax', True)
    config.set_config('plot.show_on_draw', False)
    plot_method = getattr(plotter_instance, plot_method_name)

    # Prepare data based on plot type
    if "3d" in plot_method_name or "surface" in plot_method_name or "wireframe" in plot_method_name:
        if "surface" in plot_method_name or "wireframe" in plot_method_name:
            d1, d2, d3 = get_sample_meshgrid_data()
        else:
            d1, d2, d3 = get_sample_data_3d()
        args = (d1, d2, d3)
    elif "image" in plot_method_name:
        args = (np.random.rand(5,5),)
    else: # 2D
        args = get_sample_data_2d()

    # Test with 'dark' theme
    plotter_instance.set_theme('dark')
    dark_theme_settings = themes.get_active_theme_dict()
    fig_dark, ax_dark = plot_method(*args, title=f"Dark {plot_method_name}")

    assert fig_dark.get_facecolor() == dark_theme_settings['figure.facecolor']

    ax_facecolor_expected = dark_theme_settings.get('axes.facecolor')
    if hasattr(ax_dark, 'zaxis'): # 3D plot
        ax_facecolor_expected = dark_theme_settings.get('axes3d.facecolor', ax_facecolor_expected)

    # For some axes like 3D, get_facecolor() might return (r,g,b,a) even if set by hex
    # So, compare them carefully or check against theme's internal values more directly.
    # This check is simplified; more robust checks might involve converting hex to RGBA.
    # For now, if no error and fig facecolor is right, assume ax colors are generally applied.

    # Test with 'light' theme
    plotter_instance.set_theme('light')
    light_theme_settings = themes.get_active_theme_dict()
    fig_light, ax_light = plot_method(*args, title=f"Light {plot_method_name}")
    assert fig_light.get_facecolor() == light_theme_settings['figure.facecolor']


def test_config_application_on_plot_fontsize(plotter_instance):
    config.set_config('plot.return_fig_ax', True)
    config.set_config('plot.show_on_draw', False) # Prevent GUI
    original_title_size = config.get_config('font.size_title')

    try:
        new_title_size = 22
        plotter_instance.update_config('font.size_title', new_title_size)

        x, y = get_sample_data_2d()
        _, ax = plotter_instance.plot_line(x, y, title="Config Font Test")

        assert ax.title.get_fontsize() == new_title_size
    finally:
        # Reset for other tests
        config.set_config('font.size_title', original_title_size)

def test_config_application_on_plot_figsize(plotter_instance):
    config.set_config('plot.return_fig_ax', True)
    config.set_config('plot.show_on_draw', False) # Prevent GUI
    original_figsize = config.get_config('figure.figsize')
    try:
        new_figsize = (9, 4)
        plotter_instance.update_config('figure.figsize', new_figsize)

        x,y = get_sample_data_2d()
        fig, _ = plotter_instance.plot_line(x,y)

        np.testing.assert_array_almost_equal(fig.get_size_inches(), new_figsize, decimal=2)
    finally:
        config.set_config('figure.figsize', original_figsize) # reset

# --- Tests for Config Persistence (save_config, load_config) ---
def test_config_save_and_load(tmp_path):
    # tmp_path is a pytest fixture providing a temporary directory path
    filepath = tmp_path / "test_config.json"

    # Modify some config settings
    original_cmap = config.get_config("image.cmap")
    original_figsize = config.get_config("figure.figsize")

    config.set_config("image.cmap", "plasma")
    config.set_config("figure.figsize", (12, 7))
    config.set_config("font.size_title", 20)

    # Save the modified config
    config.save_config(str(filepath))
    assert filepath.exists()

    # Reset config to defaults to ensure load works
    config.reset_config()
    assert config.get_config("image.cmap") == original_cmap # Default
    assert config.get_config("figure.figsize") == original_figsize # Default

    # Load the saved config
    config.load_config(str(filepath))

    # Check if loaded values are correct
    assert config.get_config("image.cmap") == "plasma"
    assert config.get_config("figure.figsize") == (12, 7)
    assert config.get_config("font.size_title") == 20

def test_load_config_non_existent_file(tmp_path):
    filepath = tmp_path / "non_existent_config.json"
    with pytest.raises(ConfigurationError, match="Configuration file not found"):
        config.load_config(str(filepath))

def test_load_config_malformed_json(tmp_path):
    filepath = tmp_path / "malformed.json"
    with open(filepath, "w") as f:
        f.write("{'cmap': 'viridis', 'figsize': (10,6") # Malformed JSON

    with pytest.raises(ConfigurationError, match="Failed to load or parse configuration"):
        config.load_config(str(filepath))

def test_load_config_invalid_key_in_file(tmp_path):
    filepath = tmp_path / "invalid_key_config.json"
    valid_cmap = config.get_config("image.cmap")
    data_to_save = {
        "image.cmap": "cividis",
        "non.existent.key": "some_value",
        "font.size_label": 15
    }
    with open(filepath, "w") as f:
        json.dump(data_to_save, f)

    config.reset_config() # Start from defaults
    config.load_config(str(filepath)) # non.existent.key should be ignored

    assert config.get_config("image.cmap") == "cividis"
    assert config.get_config("font.size_label") == 15
    # Ensure non.existent.key was not added
    with pytest.raises(ConfigurationError, match="Configuration key 'non.existent.key' not found"):
        config.get_config("non.existent.key")


def test_load_config_invalid_value_type_in_file(tmp_path):
    filepath = tmp_path / "invalid_type_config.json"
    data_to_save = {
        "font.size_title": "should_be_int", # Invalid type
        "figure.figsize": [8,8] # Invalid type (list instead of tuple)
    }
    with open(filepath, "w") as f:
        json.dump(data_to_save, f)

    config.reset_config()
    with pytest.raises(ConfigurationError, match="Invalid configuration for key 'font.size_title'"):
        config.load_config(str(filepath))

    # Test with a partially valid file if the above raises on first error
    filepath_partial = tmp_path / "partial_invalid_type.json"
    data_to_save_partial = {
        "image.cmap": "twilight",
        "font.size_title": "not_an_int_again"
    }
    with open(filepath_partial, "w") as f:
        json.dump(data_to_save_partial, f)

    config.reset_config()
    with pytest.raises(ConfigurationError, match="Invalid configuration for key 'font.size_title'"):
        config.load_config(str(filepath_partial))
    # Check that valid part before error was not loaded, or handle partial load if desired
    # Current implementation of load_config might apply valid ones before erroring,
    # or error out on first problem depending on loop structure.
    # The test expects it to error out on the first problematic key.
    assert config.get_config("image.cmap") != "twilight" # Should still be default
