"""Unit tests for the optiland.plotting module.

This test suite covers the functionality of config management, theme application,
and various plotting methods provided by the Plotter class.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from optiland.plotting import Plotter, config, themes
from optiland.plotting.exceptions import (
    ConfigurationError,
    InvalidPlotDataError,
    ThemeNotFoundError,
)

# --- Fixtures ---


@pytest.fixture(autouse=True)
def mock_plt_show(monkeypatch):
    """Mocks plt.show() to prevent GUI popups and closes figures after each test."""
    mock_calls = []

    def mock_show(*args, **kwargs):
        mock_calls.append({"args": args, "kwargs": kwargs})
        # print(f"plt.show() called with args: {args}, kwargs: {kwargs}") # For debugging # noqa: E501

    monkeypatch.setattr(plt, "show", mock_show)
    yield mock_calls  # Can be used to check if show was called if needed
    plt.close("all")  # Close all figures after the test run


@pytest.fixture(autouse=True)
def reset_plotting_defaults():
    """Resets plotting configurations and active theme before each test."""
    config.reset_config()
    themes.set_active_theme("light")  # Default to 'light' theme for tests


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
    assert isinstance(config.get_config("figure.figsize"), tuple)
    assert isinstance(config.get_config("font.size_title"), int)
    assert config.get_config("plot.show_on_draw") is True  # Default


def test_config_set_and_get_value():
    original_figsize = config.get_config("figure.figsize")
    new_figsize = (12, 8)
    config.set_config("figure.figsize", new_figsize)
    assert config.get_config("figure.figsize") == new_figsize
    config.set_config("figure.figsize", original_figsize)  # Reset


def test_config_get_invalid_key():
    with pytest.raises(
        ConfigurationError,
        match="Configuration key 'non.existent.key' not found",
    ):
        config.get_config("non.existent.key")


def test_config_set_invalid_key():
    with pytest.raises(
        ConfigurationError,
        match="Invalid configuration key 'non.existent.key'",
    ):
        config.set_config("non.existent.key", 100)


def test_config_set_invalid_type():
    original_size = config.get_config("font.size_title")
    # Test setting a string where an int is expected
    with pytest.raises(
        ConfigurationError,
        match=(
            "Invalid type for configuration key 'font.size_title'. "
            "Expected type 'int', but got type 'str'"
        ),
    ):
        config.set_config("font.size_title", "not-an-int")
    config.set_config("font.size_title", original_size)  # Reset

    original_figsize = config.get_config("figure.figsize")
    # Test setting a list where a tuple is expected
    with pytest.raises(
        ConfigurationError,
        match=(
            "Invalid type for configuration key 'figure.figsize'. "
            "Expected type 'tuple', but got type 'list'"
        ),
    ):
        config.set_config("figure.figsize", [10, 5])
    config.set_config("figure.figsize", original_figsize)  # Reset

    # Test setting a tuple with non-numeric or non-positive values for figsize
    with pytest.raises(
        ConfigurationError,
        match="'figure.figsize' must be a tuple of two positive numbers",
    ):
        config.set_config("figure.figsize", ("a", "b"))
    with pytest.raises(
        ConfigurationError,
        match="'figure.figsize' must be a tuple of two positive numbers",
    ):
        config.set_config("figure.figsize", (-10, 5))
    config.set_config("figure.figsize", original_figsize)  # Reset


def test_config_reset():
    default_figsize = config._DEFAULT_CONFIG["figure.figsize"]
    default_title_size = config._DEFAULT_CONFIG["font.size_title"]

    config.set_config("figure.figsize", (15, 10))
    config.set_config("font.size_title", 25)
    assert config.get_config("figure.figsize") == (15, 10)
    assert config.get_config("font.size_title") == 25

    config.reset_config()
    assert config.get_config("figure.figsize") == default_figsize
    assert config.get_config("font.size_title") == default_title_size


# --- Tests for themes.py ---


def test_themes_get_active_theme():
    assert themes.get_active_theme() == "light"  # Default due to reset fixture


def test_themes_set_active_theme_valid():
    themes.set_active_theme("dark")
    assert themes.get_active_theme() == "dark"
    themes.set_active_theme("light")  # Reset
    assert themes.get_active_theme() == "light"


def test_themes_set_active_theme_invalid():
    with pytest.raises(
        ThemeNotFoundError,
        match="Theme 'non_existent_theme' not found",
    ):
        themes.set_active_theme("non_existent_theme")


def test_themes_get_active_theme_dict():
    light_settings = themes.get_active_theme_dict()
    assert isinstance(light_settings, dict)
    assert light_settings["figure.facecolor"] == "#FFFFFF"

    themes.set_active_theme("dark")
    dark_settings = themes.get_active_theme_dict()
    assert dark_settings["figure.facecolor"] == "#1E1E1E"


def test_themes_get_theme_value_active_theme():
    assert (
        themes.get_theme_value("lines.color") == themes.THEMES["light"]["lines.color"]
    )
    themes.set_active_theme("dark")
    assert themes.get_theme_value("lines.color") == themes.THEMES["dark"]["lines.color"]


def test_themes_get_theme_value_specific_theme():
    assert (
        themes.get_theme_value("lines.color", theme_name="dark")
        == themes.THEMES["dark"]["lines.color"]
    )
    # Active theme should remain unchanged
    assert themes.get_active_theme() == "light"


def test_themes_get_theme_value_invalid_key():
    with pytest.raises(
        ThemeNotFoundError,
        match="Key 'non.existent.key' not found in theme 'light'",
    ):
        themes.get_theme_value("non.existent.key")


def test_themes_get_theme_value_invalid_theme_name():
    with pytest.raises(ThemeNotFoundError, match="Theme 'invalid_theme' not found"):
        themes.get_theme_value("lines.color", theme_name="invalid_theme")


def test_themes_list_themes():
    available_themes = themes.list_themes()
    assert isinstance(available_themes, list)
    assert "light" in available_themes
    assert "dark" in available_themes


# --- Tests for Plotter class ---


def test_plotter_instantiation(plotter_instance):
    assert isinstance(plotter_instance, Plotter)


def test_plotter_set_theme_instance_method(plotter_instance):  # Renamed for clarity
    plotter_instance.set_theme("dark")
    assert themes.get_active_theme() == "dark"
    # Check if a plot reflects the theme (e.g., figure facecolor)
    config.set_config("plot.return_fig_ax", True)
    config.set_config("plot.show_on_draw", False)  # prevent plt.show issues
    x, y = get_sample_data_2d()
    try:
        # Using static call here as instance methods for plotting are just wrappers now
        fig, _ = Plotter.plot_line(x, y)
        assert fig.get_facecolor() == themes.THEMES["dark"]["figure.facecolor"]
    except InvalidPlotDataError:
        pytest.fail(
            "plot_line raised InvalidPlotDataError with valid sample data "
            "during theme test."
        )


def test_plotter_update_config_instance_method(plotter_instance):  # Renamed for clarity
    new_figsize = (11, 7)
    plotter_instance.update_config("figure.figsize", new_figsize)
    assert config.get_config("figure.figsize") == new_figsize

    # Check if a plot reflects the config (e.g., figsize)
    config.set_config("plot.return_fig_ax", True)
    config.set_config("plot.show_on_draw", False)  # prevent plt.show issues
    x, y = get_sample_data_2d()
    try:
        # Using static call here
        fig, _ = Plotter.plot_line(x, y)
        np.testing.assert_almost_equal(fig.get_size_inches(), new_figsize, decimal=2)
    except InvalidPlotDataError:
        pytest.fail(
            "plot_line raised InvalidPlotDataError with valid sample data "
            "during config test."
        )


def test_plotter_get_current_theme_settings(plotter_instance):
    settings = plotter_instance.get_current_theme_settings()
    assert settings == themes.THEMES["light"]
    plotter_instance.set_theme("dark")
    settings_dark = plotter_instance.get_current_theme_settings()
    assert settings_dark == themes.THEMES["dark"]


def test_plotter_get_config_value(plotter_instance):
    assert plotter_instance.get_config_value("font.size_title") == config.get_config(
        "font.size_title",
    )


# --- Tests for 2D Plotting Methods ---


@pytest.mark.parametrize("plot_method_name", ["plot_line", "plot_scatter"])
@pytest.mark.parametrize("return_fig_ax_config", [True, False])
@pytest.mark.parametrize("explicit_return_arg", [None, True, False])
@pytest.mark.parametrize(
    "use_static_call",
    [True, False],
)  # Test static and instance calls
def test_plotter_2d_methods(
    plotter_instance,
    plot_method_name,
    return_fig_ax_config,
    explicit_return_arg,
    use_static_call,
    mock_plt_show,
):
    if use_static_call:
        plot_method = getattr(Plotter, plot_method_name)
    else:
        plot_method = getattr(plotter_instance, plot_method_name)

    x, y = get_sample_data_2d()

    config.set_config("plot.return_fig_ax", return_fig_ax_config)
    # If we expect to return fig/ax, show_on_draw should be false to avoid plt.show()
    # If we don't expect to return, show_on_draw can be anything, mock handles it.
    config.set_config(
        "plot.show_on_draw",
        not return_fig_ax_config
        if explicit_return_arg is None
        else not explicit_return_arg,
    )

    title = f"Test {plot_method_name}{' Static' if use_static_call else ' Instance'}"
    xlabel = "X Data"
    ylabel = "Y Data"
    legend = "Data Series"

    plot_args = {
        "x": x,
        "y": y,
        "title": title,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "legend_label": legend,
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
            assert len(ax.collections) > 0
            sc_data = ax.collections[0].get_offsets()
            assert np.array_equal(sc_data[:, 0], x)
            assert np.array_equal(sc_data[:, 1], y)

        assert ax.legend_ is not None
        if explicit_return_arg is True:
            assert not mock_plt_show  # Check that plt.show was not called
    else:
        assert result is None
        if config.get_config("plot.show_on_draw"):
            assert mock_plt_show  # Check that plt.show was called


@pytest.mark.parametrize("use_static_call", [True, False])
def test_plot_line_kwargs(plotter_instance, use_static_call):
    config.set_config("plot.return_fig_ax", True)
    config.set_config("plot.show_on_draw", False)
    x, y = get_sample_data_2d()

    plot_method = Plotter.plot_line if use_static_call else plotter_instance.plot_line
    fig, ax = plot_method(x, y, linestyle="--", marker="o")

    assert len(ax.lines) == 1
    assert ax.lines[0].get_linestyle() == "--"
    assert ax.lines[0].get_marker() == "o"


@pytest.mark.parametrize("use_static_call", [True, False])
def test_plot_scatter_kwargs(plotter_instance, use_static_call):
    config.set_config("plot.return_fig_ax", True)
    config.set_config("plot.show_on_draw", False)
    x, y = get_sample_data_2d()

    plot_method = (
        Plotter.plot_scatter if use_static_call else plotter_instance.plot_scatter
    )
    fig, ax = plot_method(x, y, s=100, c="red", alpha=0.5)

    assert len(ax.collections) == 1
    assert np.allclose(
        ax.collections[0].get_facecolors()[0],
        plt.cm.colors.to_rgba("red"),
    )
    assert ax.collections[0].get_alpha() == 0.5


# --- Tests for plot_image (static and instance) ---
@pytest.mark.parametrize("use_static_call", [True, False])
def test_plot_image(plotter_instance, use_static_call):
    config.set_config("plot.return_fig_ax", True)
    config.set_config("plot.show_on_draw", False)
    img_data = np.random.rand(10, 10)

    plot_method = Plotter.plot_image if use_static_call else plotter_instance.plot_image
    fig, ax = plot_method(
        img_data,
        title="Test Image",
        cmap="plasma",
        show_colorbar=True,
    )

    assert fig is not None
    assert ax is not None
    assert ax.get_title() == "Test Image"
    assert len(ax.images) == 1
    assert ax.images[0].get_cmap().name == "plasma"
    assert len(fig.axes) > 1

    fig_no_cb, ax_no_cb = plot_method(img_data, show_colorbar=False)
    assert len(fig_no_cb.axes) == 1

    original_cmap = config.get_config("image.cmap")
    try:
        config.set_config("image.cmap", "magma")
        fig_def_cmap, ax_def_cmap = plot_method(img_data)
        assert ax_def_cmap.images[0].get_cmap().name == "magma"
    finally:
        config.set_config("image.cmap", original_cmap)


@pytest.mark.parametrize("use_static_call", [True, False])
def test_plot_image_invalid_data(plotter_instance, use_static_call):
    plot_method = Plotter.plot_image if use_static_call else plotter_instance.plot_image
    with pytest.raises(
        InvalidPlotDataError,
        match="image_data must be a 2D array-like structure",
    ):
        plot_method(np.random.rand(5, 5, 5))
    with pytest.raises(InvalidPlotDataError, match="image_data cannot be empty"):
        plot_method(np.array([]))


# --- Tests for plot_subplots (static and instance) ---
@pytest.mark.parametrize("use_static_call", [True, False])
def test_plot_subplots_basic(plotter_instance, use_static_call):
    config.set_config("plot.return_fig_ax", True)
    config.set_config("plot.show_on_draw", False)

    plot_method = (
        Plotter.plot_subplots if use_static_call else plotter_instance.plot_subplots
    )
    callbacks = [simple_plot_callback, simple_plot_callback]
    fig, axs = plot_method(1, 2, callbacks, main_title="Subplots Test")

    assert fig is not None
    assert isinstance(axs, np.ndarray)
    assert axs.shape == (2,)
    assert fig._suptitle.get_text() == "Subplots Test"

    for i, ax_sub in enumerate(axs.flat):
        assert f"Subplot {i}" in ax_sub.get_title()
        assert len(ax_sub.lines) == 1


@pytest.mark.parametrize("use_static_call", [True, False])
def test_plot_subplots_return_behavior(
    plotter_instance,
    use_static_call,
    mock_plt_show,
):
    plot_method = (
        Plotter.plot_subplots if use_static_call else plotter_instance.plot_subplots
    )
    callbacks = [simple_plot_callback]

    config.set_config("plot.return_fig_ax", False)
    config.set_config("plot.show_on_draw", True)

    result = plot_method(1, 1, callbacks, return_fig_axs=True)
    assert isinstance(result, tuple) and len(result) == 2

    mock_plt_show.clear()  # Clear mock calls before next check
    result_no_return = plot_method(1, 1, callbacks, return_fig_axs=False)
    assert result_no_return is None
    if config.get_config("plot.show_on_draw"):
        assert len(mock_plt_show) > 0

    config.set_config("plot.return_fig_ax", True)
    config.set_config("plot.show_on_draw", False)
    result_global_return = plot_method(1, 1, callbacks, return_fig_axs=None)
    assert isinstance(result_global_return, tuple)


@pytest.mark.parametrize("use_static_call", [True, False])
def test_plot_subplots_invalid_callbacks_count(plotter_instance, use_static_call):
    plot_method = (
        Plotter.plot_subplots if use_static_call else plotter_instance.plot_subplots
    )
    callbacks = [simple_plot_callback] * 3
    with pytest.raises(ValueError, match="Number of plot_callbacks"):
        plot_method(1, 2, callbacks)
    with pytest.raises(
        InvalidPlotDataError,
        match="plot_callbacks must be a non-empty list",
    ):
        plot_method(1, 1, [])
    with pytest.raises(
        InvalidPlotDataError,
        match="All items in plot_callbacks must be callable",
    ):
        plot_method(1, 1, ["not_callable"])


# --- Tests for 3D Plotting Methods ---
@pytest.mark.parametrize(
    "plot_method_3d_name, data_func, check_collection_type",
    [
        ("plot_line_3d", get_sample_data_3d, "lines"),
        ("plot_scatter_3d", get_sample_data_3d, "collections_scatter"),
        ("plot_surface", get_sample_meshgrid_data, "collections_surface"),
        ("plot_wireframe", get_sample_meshgrid_data, "lines_wireframe"),
    ],
)
@pytest.mark.parametrize("use_static_call", [True, False])
def test_plotter_3d_methods(
    plotter_instance,
    plot_method_3d_name,
    data_func,
    check_collection_type,
    use_static_call,
):
    config.set_config("plot.return_fig_ax", True)
    config.set_config("plot.show_on_draw", False)

    if use_static_call:
        plot_method = getattr(Plotter, plot_method_3d_name)
    else:
        plot_method = getattr(plotter_instance, plot_method_3d_name)

    d1, d2, d3 = data_func()

    title = f"Test {plot_method_3d_name}{' Static' if use_static_call else ' Instance'}"
    xlabel, ylabel, zlabel = "X3D", "Y3D", "Z3D"

    plot_args = {"title": title, "xlabel": xlabel, "ylabel": ylabel, "zlabel": zlabel}
    if plot_method_3d_name in ["plot_line_3d", "plot_scatter_3d"]:
        plot_args["legend_label"] = "3D Data"

    fig, ax = plot_method(d1, d2, d3, **plot_args)

    assert fig is not None
    assert ax is not None
    assert hasattr(ax, "zaxis")  # Check if it's a 3D axis
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
        assert (
            len(ax.collections) > 0 or len(ax.patches) > 0
        )  # scatter can be Patch3DCollection or Path3DCollection
    elif check_collection_type == "collections_surface":
        assert len(ax.collections) > 0  # plot_surface creates Poly3DCollection
    elif check_collection_type == "lines_wireframe":
        assert len(ax.lines) > 0  # plot_wireframe creates Line3DCollection

    if plot_args.get("legend_label"):
        assert ax.legend_ is not None


@pytest.mark.parametrize(
    "plot_method_name, invalid_data_sets",
    [
        (
            "plot_line",
            [(np.array([1, 2]), np.array([1, 2, 3])), (np.array([]), np.array([]))],
        ),
        (
            "plot_scatter",
            [(np.array([1, 2]), np.array([1, 2, 3])), (np.array([]), np.array([]))],
        ),
        (
            "plot_line_3d",
            [
                (
                    np.array([1, 2]),
                    np.array([1, 2, 3]),
                    np.array([1, 2]),
                ),  # Mismatched lengths
                (np.array([]), np.array([]), np.array([])),
            ],
        ),  # Empty data
        (
            "plot_scatter_3d",
            [
                (np.array([1, 2]), np.array([1, 2, 3]), np.array([1, 2])),
                (np.array([]), np.array([]), np.array([])),
            ],
        ),
        (
            "plot_surface",
            [
                (np.random.rand(5), np.random.rand(5), np.random.rand(5)),  # 1D data
                (np.random.rand(5, 5), np.random.rand(5, 6), np.random.rand(5, 5)),
            ],
        ),  # Mismatched shapes
        (
            "plot_wireframe",
            [
                (np.random.rand(5), np.random.rand(5), np.random.rand(5)),
                (np.random.rand(5, 5), np.random.rand(5, 6), np.random.rand(5, 5)),
            ],
        ),
    ],
)
@pytest.mark.parametrize("use_static_call", [True, False])
def test_plot_methods_invalid_data(
    plotter_instance,
    plot_method_name,
    invalid_data_sets,
    use_static_call,
):
    if use_static_call:
        plot_method = getattr(Plotter, plot_method_name)
    else:
        plot_method = getattr(plotter_instance, plot_method_name)

    for data_args in invalid_data_sets:
        with pytest.raises(InvalidPlotDataError):
            plot_method(*data_args)


# --- Tests for Theme and Config Application ---
@pytest.mark.parametrize(
    "plot_method_name",
    [
        "plot_line",
        "plot_scatter",
        "plot_image",
        "plot_line_3d",
        "plot_scatter_3d",
        "plot_surface",
        "plot_wireframe",
    ],
)
@pytest.mark.parametrize("use_static_call", [True, False])
def test_theme_application_on_plots(
    plotter_instance,
    plot_method_name,
    use_static_call,
):
    config.set_config("plot.return_fig_ax", True)
    config.set_config("plot.show_on_draw", False)

    if use_static_call:
        plot_method = getattr(Plotter, plot_method_name)
    else:
        plot_method = getattr(plotter_instance, plot_method_name)

    if (
        "3d" in plot_method_name
        or "surface" in plot_method_name
        or "wireframe" in plot_method_name
    ):
        if "surface" in plot_method_name or "wireframe" in plot_method_name:
            d1, d2, d3 = get_sample_meshgrid_data()
        else:
            d1, d2, d3 = get_sample_data_3d()
        args = (d1, d2, d3)
    elif "image" in plot_method_name:
        args = (np.random.rand(5, 5),)
    else:  # 2D
        args = get_sample_data_2d()

    # Test with 'dark' theme
    # Use themes module directly if testing static calls primarily,
    # or plotter_instance if testing instance's effect on global state.
    active_theme_setter = (
        plotter_instance.set_theme if not use_static_call else themes.set_active_theme
    )

    active_theme_setter("dark")
    dark_theme_settings = themes.get_active_theme_dict()
    fig_dark, ax_dark = plot_method(*args, title=f"Dark {plot_method_name}")
    assert fig_dark.get_facecolor() == dark_theme_settings["figure.facecolor"]
    # Further checks for ax properties can be added here

    # Test with 'light' theme
    active_theme_setter("light")
    light_theme_settings = themes.get_active_theme_dict()
    fig_light, ax_light = plot_method(*args, title=f"Light {plot_method_name}")
    assert fig_light.get_facecolor() == light_theme_settings["figure.facecolor"]


@pytest.mark.parametrize("use_static_call", [True, False])
def test_config_application_on_plot_fontsize(plotter_instance, use_static_call):
    config.set_config("plot.return_fig_ax", True)
    config.set_config("plot.show_on_draw", False)
    original_title_size = config.get_config("font.size_title")

    active_config_setter = (
        plotter_instance.update_config if not use_static_call else config.set_config
    )

    try:
        new_title_size = 22
        active_config_setter("font.size_title", new_title_size)

        x, y = get_sample_data_2d()
        plot_method = (
            Plotter.plot_line if use_static_call else plotter_instance.plot_line
        )
        _, ax = plot_method(x, y, title="Config Font Test")
        assert ax.title.get_fontsize() == new_title_size
    finally:
        config.set_config("font.size_title", original_title_size)


@pytest.mark.parametrize("use_static_call", [True, False])
def test_config_application_on_plot_figsize(plotter_instance, use_static_call):
    config.set_config("plot.return_fig_ax", True)
    config.set_config("plot.show_on_draw", False)
    original_figsize = config.get_config("figure.figsize")

    active_config_setter = (
        plotter_instance.update_config if not use_static_call else config.set_config
    )

    try:
        new_figsize = (9, 4)
        active_config_setter("figure.figsize", new_figsize)

        x, y = get_sample_data_2d()
        plot_method = (
            Plotter.plot_line if use_static_call else plotter_instance.plot_line
        )
        fig, _ = plot_method(x, y)
        np.testing.assert_array_almost_equal(
            fig.get_size_inches(),
            new_figsize,
            decimal=2,
        )
    finally:
        config.set_config("figure.figsize", original_figsize)


# --- Tests for Config Persistence (save_config, load_config) ---
# These test the config module directly, Plotter instance not strictly needed.
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
    assert config.get_config("image.cmap") == original_cmap  # Default
    assert config.get_config("figure.figsize") == original_figsize  # Default

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
        f.write("{'cmap': 'viridis', 'figsize': (10,6")  # Malformed JSON

    with pytest.raises(
        ConfigurationError,
        match="Failed to load or parse configuration",
    ):
        config.load_config(str(filepath))


def test_load_config_invalid_key_in_file(tmp_path):
    filepath = tmp_path / "invalid_key_config.json"
    # valid_cmap = config.get_config("image.cmap") # F841
    data_to_save = {
        "image.cmap": "cividis",
        "non.existent.key": "some_value",
        "font.size_label": 15,
    }
    with open(filepath, "w") as f:
        json.dump(data_to_save, f)

    config.reset_config()  # Start from defaults
    config.load_config(str(filepath))  # non.existent.key should be ignored

    assert config.get_config("image.cmap") == "cividis"
    assert config.get_config("font.size_label") == 15
    # Ensure non.existent.key was not added
    with pytest.raises(
        ConfigurationError,
        match="Configuration key 'non.existent.key' not found",
    ):
        config.get_config("non.existent.key")


def test_load_config_invalid_value_type_in_file(tmp_path):
    filepath = tmp_path / "invalid_type_config.json"
    data_to_save = {
        "font.size_title": "should_be_int",  # Invalid type
        "figure.figsize": [8, 8],  # Invalid type (list instead of tuple)
    }
    with open(filepath, "w") as f:
        json.dump(data_to_save, f)

    config.reset_config()
    with pytest.raises(
        ConfigurationError,
        match="Invalid configuration for key 'font.size_title'",
    ):
        config.load_config(str(filepath))

    # Test with a partially valid file if the above raises on first error
    filepath_partial = tmp_path / "partial_invalid_type.json"
    data_to_save_partial = {
        "image.cmap": "twilight",
        "font.size_title": "not_an_int_again",
    }
    with open(filepath_partial, "w") as f:
        json.dump(data_to_save_partial, f)

    config.reset_config()
    with pytest.raises(
        ConfigurationError,
        match="Invalid configuration for key 'font.size_title'",
    ):
        config.load_config(str(filepath_partial))
    # Check that valid part before error was not loaded,
    # or handle partial load if desired.
    # Current implementation of load_config might apply valid ones before erroring,
    # or error out on first problem depending on loop structure.
    # The test expects it to error out on the first problematic key.
    assert config.get_config("image.cmap") != "twilight"  # Should still be default


# --- Tests for Plotting on Existing Axes ---


@pytest.mark.parametrize(
    "plot_method_name, data_func, is_3d",
    [
        ("plot_line", get_sample_data_2d, False),
        ("plot_scatter", get_sample_data_2d, False),
        (
            "plot_image",
            lambda: (np.random.rand(5, 5),),
            False,
        ),  # plot_image takes one data arg
        ("plot_line_3d", get_sample_data_3d, True),
        ("plot_scatter_3d", get_sample_data_3d, True),
        ("plot_surface", get_sample_meshgrid_data, True),
        ("plot_wireframe", get_sample_meshgrid_data, True),
    ],
)
def test_plotting_on_provided_ax(plot_method_name, data_func, is_3d):
    config.set_config("plot.return_fig_ax", True)  # Ensure fig, ax are returned
    config.set_config("plot.show_on_draw", False)  # Prevent GUI popups

    plot_method_static = getattr(Plotter, plot_method_name)
    data_args = data_func()

    # Create a figure and axes beforehand
    pre_fig = plt.figure()
    if is_3d:
        pre_ax = pre_fig.add_subplot(111, projection="3d")
    else:
        pre_ax = pre_fig.add_subplot(111)

    initial_ax_children = len(pre_ax.get_children())

    # Call the static plotting method with the pre-existing ax
    # For plot_image, data_args is a tuple with one element
    returned_fig, returned_ax = plot_method_static(
        *data_args,
        ax=pre_ax,
        title=f"Test on pre_ax: {plot_method_name}",
    )

    assert returned_fig is pre_fig, (
        "Figure returned should be the same as the one from the pre-existing axes."
    )
    assert returned_ax is pre_ax, (
        "Axes returned should be the same as the pre-existing axes."
    )

    # Check if the pre-existing ax was actually used for plotting
    if plot_method_name == "plot_line" or plot_method_name == "plot_wireframe":
        assert len(returned_ax.lines) > 0, "No lines were plotted on the provided axes."
    elif (
        plot_method_name == "plot_scatter"
        or plot_method_name == "plot_scatter_3d"
        or plot_method_name == "plot_surface"
    ):
        assert len(returned_ax.collections) > 0, (
            "No collections (scatter/surface) were plotted."
        )
    elif plot_method_name == "plot_image":
        assert len(returned_ax.images) > 0, "No image was plotted."

    # More generic check: number of children of the axes should have increased
    assert len(returned_ax.get_children()) > initial_ax_children, (
        "Plotting did not add elements to the provided axes."
    )

    assert returned_ax.get_title() == f"Test on pre_ax: {plot_method_name}"


def test_plot_subplots_does_not_accept_ax_kwarg(plotter_instance):
    """Confirms plot_subplots does not accept 'ax' or 'axs' as it creates its own."""
    callbacks = [simple_plot_callback]
    with pytest.raises(
        TypeError,
    ) as excinfo:  # Matplotlib's subplots raises TypeError for unexpected kwargs
        Plotter.plot_subplots(1, 1, callbacks, ax="some_ax_obj")
    assert "unexpected keyword argument 'ax'" in str(excinfo.value).lower()

    with pytest.raises(TypeError) as excinfo_axs:
        Plotter.plot_subplots(1, 1, callbacks, axs="some_axs_obj")
    assert "unexpected keyword argument 'axs'" in str(excinfo_axs.value).lower()


def test_plot_line_3d_with_invalid_ax_type():
    """Test that 3D plots raise ValueError if a 2D ax is passed."""
    x, y, z = get_sample_data_3d()
    fig, ax_2d = plt.subplots()
    with pytest.raises(
        ValueError,
        match="Provided 'ax' for 3D plot must be a 3D Axes object.",
    ):
        Plotter.plot_line_3d(x, y, z, ax=ax_2d)


# --- Tests for Color Cycle Functionality ---


@pytest.mark.parametrize("plot_method_name", ["plot_line", "plot_scatter"])
@pytest.mark.parametrize("theme_name", ["light", "dark"])
def test_color_cycle_application_on_same_axes(
    plot_method_name,
    theme_name,
    mock_plt_show,
):
    """Tests that sequential plots on same axes use different colors from cycle."""
    config.set_config("plot.return_fig_ax", True)
    config.set_config("plot.show_on_draw", False)
    themes.set_active_theme(theme_name)

    plot_method = getattr(Plotter, plot_method_name)
    expected_colors = themes.THEMES[theme_name]["axes.prop_cycle"].by_key()["color"]

    x, y1 = get_sample_data_2d(size=5)
    _, y2 = get_sample_data_2d(size=5)  # Different y data for second plot

    fig, ax = plt.subplots()

    # First plot
    plot_method(x, y1, ax=ax, legend_label="Series 1")
    # Second plot
    plot_method(x, y2, ax=ax, legend_label="Series 2")

    if plot_method_name == "plot_line":
        assert len(ax.lines) == 2
        color1 = plt.colors.to_rgba(ax.lines[0].get_color())
        color2 = plt.colors.to_rgba(ax.lines[1].get_color())
    elif plot_method_name == "plot_scatter":
        assert len(ax.collections) == 2
        # get_facecolor() returns an array of colors, even if it's just one.
        color1 = plt.colors.to_rgba(ax.collections[0].get_facecolors()[0])
        color2 = plt.colors.to_rgba(ax.collections[1].get_facecolors()[0])
    else:
        pytest.fail(
            f"Unsupported plot_method_name for color cycle test: {plot_method_name}",
        )

    expected_rgba1 = plt.colors.to_rgba(expected_colors[0])
    expected_rgba2 = plt.colors.to_rgba(expected_colors[1])

    np.testing.assert_almost_equal(
        color1,
        expected_rgba1,
        decimal=4,
        err_msg=f"First plot color mismatch for {theme_name} theme.",
    )
    np.testing.assert_almost_equal(
        color2,
        expected_rgba2,
        decimal=4,
        err_msg=f"Second plot color mismatch for {theme_name} theme.",
    )
    assert color1 != color2, "Sequential plots should have different colors."


@pytest.mark.parametrize("plot_method_name", ["plot_line", "plot_scatter"])
@pytest.mark.parametrize("theme_name", ["light", "dark"])
def test_color_cycle_override(plot_method_name, theme_name, mock_plt_show):
    """Tests that providing an explicit color overrides the color cycle."""
    config.set_config("plot.return_fig_ax", True)
    config.set_config("plot.show_on_draw", False)
    themes.set_active_theme(theme_name)

    plot_method = getattr(Plotter, plot_method_name)
    expected_colors = themes.THEMES[theme_name]["axes.prop_cycle"].by_key()["color"]
    override_color_str = "red"
    override_color_rgba = plt.colors.to_rgba(override_color_str)

    x, y1 = get_sample_data_2d(size=5)
    _, y2 = get_sample_data_2d(size=5)
    _, y3 = get_sample_data_2d(size=5)

    fig, ax = plt.subplots()

    # Plot 1 (uses cycler)
    plot_method(x, y1, ax=ax, legend_label="Series 1")
    # Plot 2 (override color)
    plot_method(
        x,
        y2,
        ax=ax,
        color=override_color_str,
        legend_label="Series 2 Override",
    )
    # Plot 3 (uses next in cycler)
    plot_method(x, y3, ax=ax, legend_label="Series 3")

    if plot_method_name == "plot_line":
        assert len(ax.lines) == 3
        color1 = plt.colors.to_rgba(ax.lines[0].get_color())
        color2 = plt.colors.to_rgba(ax.lines[1].get_color())
        color3 = plt.colors.to_rgba(ax.lines[2].get_color())
    elif plot_method_name == "plot_scatter":
        assert len(ax.collections) == 3
        color1 = plt.colors.to_rgba(ax.collections[0].get_facecolors()[0])
        color2 = plt.colors.to_rgba(ax.collections[1].get_facecolors()[0])
        color3 = plt.colors.to_rgba(ax.collections[2].get_facecolors()[0])
    else:
        pytest.fail(
            f"Unsupported plot_method_name for color override test: {plot_method_name}",
        )

    expected_rgba1 = plt.colors.to_rgba(expected_colors[0])
    expected_rgba_next_cycler = plt.colors.to_rgba(
        expected_colors[1],
    )  # Color for the third plot

    np.testing.assert_almost_equal(
        color1,
        expected_rgba1,
        decimal=4,
        err_msg="First plot (cycler) color mismatch.",
    )
    np.testing.assert_almost_equal(
        color2,
        override_color_rgba,
        decimal=4,
        err_msg="Second plot (override) color mismatch.",
    )
    np.testing.assert_almost_equal(
        color3,
        expected_rgba_next_cycler,
        decimal=4,
        err_msg="Third plot (cycler after override) color mismatch.",
    )


# --- Tests for Legend Configuration ---


def test_legend_defaults_plot_line(plotter_instance):
    """Test default legend properties for plot_line."""
    config.set_config("plot.return_fig_ax", True)
    x, y = get_sample_data_2d()
    _, ax = Plotter.plot_line(x, y, legend_label="Test Line")

    assert ax.legend_ is not None, "Legend should be displayed by default."
    assert ax.legend_.get_visible() is True

    # Check against config defaults
    assert ax.legend_.get_loc() == config.get_config("legend.loc")
    expected_title = config.get_config("legend.title")
    # Matplotlib legend title is an empty string if None, so handle this
    assert (ax.legend_.get_title().get_text() == expected_title) or \
           (expected_title is None and ax.legend_.get_title().get_text() == "")
    assert ax.legend_.get_frame().get_visible() == config.get_config("legend.frameon")
    assert ax.legend_.shadow == config.get_config("legend.shadow")
    assert ax.legend_.fancybox == config.get_config("legend.fancybox")
    # _ncol is internal, but often used. Let's assume it's stable for testing.
    assert ax.legend_._ncol == config.get_config("legend.ncol")
    # bbox_to_anchor needs careful comparison as it can be a Bbox object
    assert ax.legend_.get_bbox_to_anchor() is None  # Default is None


def test_legend_defaults_plot_scatter(plotter_instance):
    """Test default legend properties for plot_scatter."""
    config.set_config("plot.return_fig_ax", True)
    x, y = get_sample_data_2d()
    _, ax = Plotter.plot_scatter(x, y, legend_label="Test Scatter")

    assert ax.legend_ is not None, "Legend should be displayed by default."
    assert ax.legend_.get_visible() is True
    assert ax.legend_.get_loc() == config.get_config("legend.loc")
    expected_title = config.get_config("legend.title")
    assert (ax.legend_.get_title().get_text() == expected_title) or \
           (expected_title is None and ax.legend_.get_title().get_text() == "")
    assert ax.legend_.get_frame().get_visible() == config.get_config("legend.frameon")


@pytest.mark.parametrize(
    "plot_method_name", ["plot_line", "plot_scatter"]
)
def test_legend_override_parameters(plotter_instance, plot_method_name):
    """Test overriding legend parameters for 2D plots."""
    config.set_config("plot.return_fig_ax", True)
    plot_method = getattr(Plotter, plot_method_name)
    x, y = get_sample_data_2d()

    # Test show_legend = False
    _, ax_no_legend = plot_method(x, y, legend_label="No Show", show_legend=False)
    assert ax_no_legend.legend_ is None or not ax_no_legend.legend_.get_visible(), \
        "Legend should not be visible when show_legend=False."

    # Test other parameters
    test_title = "My Custom Legend"
    test_loc = "upper left"
    test_ncol = 2
    test_bbox = (0.5, 0.5)

    _, ax = plot_method(
        x,
        y,
        legend_label="Override Test",
        show_legend=True,
        legend_loc=test_loc,
        legend_title=test_title,
        legend_frameon=False,
        legend_shadow=True,
        legend_fancybox=False,
        legend_ncol=test_ncol,
        legend_bbox_to_anchor=test_bbox,
    )

    assert ax.legend_ is not None, "Legend should be visible."
    assert ax.legend_.get_loc() == test_loc
    assert ax.legend_.get_title().get_text() == test_title
    assert ax.legend_.get_frame().get_visible() is False
    assert ax.legend_.shadow is True
    assert ax.legend_.fancybox is False
    assert ax.legend_._ncol == test_ncol
    # Check bbox_to_anchor more carefully
    bbox_obj = ax.legend_.get_bbox_to_anchor()
    assert bbox_obj is not None
    # Depending on matplotlib version, it might be a Bbox instance or tuple
    if hasattr(bbox_obj, "x0"): # Bbox object
        assert bbox_obj.x0 == test_bbox[0]
        assert bbox_obj.y0 == test_bbox[1]
    else: # Tuple
        assert bbox_obj == test_bbox


def test_legend_global_config_interaction(plotter_instance):
    """Test that legend parameters use global config if not specified."""
    config.set_config("plot.return_fig_ax", True)
    original_loc = config.get_config("legend.loc")
    original_title = config.get_config("legend.title")

    try:
        config.set_config("legend.loc", "center")
        config.set_config("legend.title", "Global Title")
        x, y = get_sample_data_2d()
        _, ax = Plotter.plot_line(x, y, legend_label="Global Config Test")

        assert ax.legend_ is not None
        assert ax.legend_.get_loc() == "center"
        assert ax.legend_.get_title().get_text() == "Global Title"
    finally:
        # Test fixture reset_plotting_defaults will handle this,
        # but explicit reset is also fine for clarity.
        config.set_config("legend.loc", original_loc)
        config.set_config("legend.title", original_title)


def test_legend_theme_influence(plotter_instance):
    """Test theme's influence on legend text and frame colors."""
    config.set_config("plot.return_fig_ax", True)
    themes.set_active_theme("dark")  # Dark theme has distinct colors
    dark_theme_settings = themes.get_active_theme_dict()

    x, y = get_sample_data_2d()
    _, ax = Plotter.plot_line(x, y, legend_label="Theme Test")

    assert ax.legend_ is not None
    legend_text = ax.legend_.get_texts()[0]
    expected_text_color = dark_theme_settings.get(
        "legend.labelcolor", dark_theme_settings.get("text.color")
    )
    assert plt.colors.to_rgba(legend_text.get_color()) == plt.colors.to_rgba(
        expected_text_color,
    )

    legend_frame = ax.legend_.get_frame()
    expected_face_color = dark_theme_settings.get("legend.facecolor", "white")
    expected_edge_color = dark_theme_settings.get("legend.edgecolor", "black")

    assert plt.colors.to_rgba(
        legend_frame.get_facecolor(),
    ) == plt.colors.to_rgba(expected_face_color)
    assert plt.colors.to_rgba(
        legend_frame.get_edgecolor(),
    ) == plt.colors.to_rgba(expected_edge_color)


def test_legend_in_plot_subplots_callback(plotter_instance):
    """Test legend configuration within a plot_subplots callback."""
    config.set_config("plot.return_fig_ax", True) # For plot_subplots

    def subplot_with_legend_callback(ax, index):
        x, y = get_sample_data_2d(5)
        Plotter.plot_line(
            x,
            y,
            ax=ax,
            legend_label=f"Subplot {index} Line",
            legend_title=f"Legend for {index}",
            legend_loc="upper right",
            show_legend=True,
        )
        ax.set_title(f"Subplot {index}") # Plotter does not set title if ax is given

    _, axs = Plotter.plot_subplots(
        1, 1, [subplot_with_legend_callback], return_fig_axs=True
    )
    ax_sub = axs[0] if isinstance(axs, np.ndarray) else axs


    assert ax_sub.legend_ is not None, "Legend should be present in subplot."
    assert ax_sub.legend_.get_title().get_text() == "Legend for 0"
    assert ax_sub.legend_.get_loc() == "upper right"
    assert len(ax_sub.legend_.get_texts()) == 1
    assert ax_sub.legend_.get_texts()[0].get_text() == "Subplot 0 Line"


def test_legend_edge_cases(plotter_instance):
    """Test edge cases for legend display."""
    config.set_config("plot.return_fig_ax", True)
    x, y = get_sample_data_2d()

    # Case 1: legend_label=None, show_legend=True (explicitly)
    _, ax1 = Plotter.plot_line(x, y, legend_label=None, show_legend=True)
    # Behavior: legend might be created but have no elements, or not created.
    # Matplotlib typically doesn't show a legend if there are no labeled artists.
    assert ax1.legend_ is None or not ax1.legend_.get_texts(), \
        "Legend should not appear or be empty if legend_label is None."

    # Case 2: legend_label=None, show_legend=None (config default is True)
    config.set_config("legend.show", True)
    _, ax2 = Plotter.plot_line(x, y, legend_label=None)
    assert ax2.legend_ is None or not ax2.legend_.get_texts(), \
        "Legend should not appear or be empty if label is None, even if config says show."

    # Case 3: No data points, but legend_label is provided
    # Plotter methods might raise InvalidPlotDataError before legend logic
    with pytest.raises(InvalidPlotDataError):
        Plotter.plot_line(np.array([]), np.array([]), legend_label="Empty Data")

    # Case 4: Multiple plots on same axes, one with legend_label, one without
    fig, ax3 = plt.subplots()
    Plotter.plot_line(x, y, ax=ax3, legend_label="Series A")
    Plotter.plot_line(x, 0.5 * y, ax=ax3, legend_label=None) # No label for this one
    assert ax3.legend_ is not None
    assert len(ax3.legend_.get_texts()) == 1 # Only "Series A" should be in legend
    assert ax3.legend_.get_texts()[0].get_text() == "Series A"
