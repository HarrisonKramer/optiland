"""Interaction Manager Module

This module provides the InteractionManager class for handling user interactions
with Matplotlib-based visualizations of optical systems.

"""

from __future__ import annotations

from matplotlib.widgets import Button

from optiland.visualization.info.providers import INFO_PROVIDER_REGISTRY
from optiland.visualization.system.lens import Lens2D
from optiland.visualization.system.surface import Surface2D


class InteractionManager:
    """Manages user interactions for optical system visualizations.

    This class connects to a Matplotlib figure's event loop to handle
    mouse events, such as hovering and clicking, on plotted artists.

    Args:
        fig (matplotlib.figure.Figure): The Matplotlib figure to connect to.
        ax (matplotlib.axes.Axes): The Matplotlib axes containing the artists.

    """

    def __init__(self, fig, ax, tooltip_format=None):
        self.fig = fig
        self.ax = ax
        self.artist_registry = {}
        self.active_artist = None
        self.original_props = {}

        # Tooltip setup
        self.tooltip = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )
        self.tooltip.set_visible(False)
        self.tooltip_format = tooltip_format or self.default_tooltip_format

        self.info_panel = None

        self.connect()

    def register_artist(self, artist, optiland_object):
        """Registers a Matplotlib artist with its corresponding Optiland object."""
        self.artist_registry[artist] = optiland_object

    def connect(self):
        """Connects to the Matplotlib event loop."""
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_hover)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    def on_hover(self, event):
        """Handles hover events to show tooltips and highlight artists."""
        if event.inaxes != self.ax:
            self.clear_hover_effects()
            return

        found_artist = None
        for artist, _ in self.artist_registry.items():
            if hasattr(artist, "get_xdata"):  # It's a line
                xdata, ydata = artist.get_xdata(), artist.get_ydata()
                x, y = event.xdata, event.ydata
                if x is None or y is None:
                    continue
                d = ((xdata - x) ** 2 + (ydata - y) ** 2) ** 0.5
                if d.min() < 0.1:  # 0.1 data units tolerance
                    found_artist = artist
                    break
            elif artist.contains(event)[0]:
                found_artist = artist
                break

        if found_artist:
            if found_artist != self.active_artist:
                self.clear_hover_effects()
                self.highlight_artist(found_artist)
                self.show_tooltip(found_artist, event)
                self.active_artist = found_artist
        else:
            self.clear_hover_effects()

    def on_click(self, event):
        """Handles click events to show detailed information."""
        if event.inaxes != self.ax:
            return

        for artist, optiland_object in self.artist_registry.items():
            contains, _ = artist.contains(event)
            if contains:
                self.show_info_panel(optiland_object)
                break

    def highlight_artist(self, artist):
        """Highlights the given artist."""
        if hasattr(artist, "get_linewidth"):
            self.original_props[artist] = {"linewidth": artist.get_linewidth()}
            artist.set_linewidth(artist.get_linewidth() * 2)
            self.fig.canvas.draw_idle()

    def show_tooltip(self, artist, event):
        """Shows a tooltip for the given artist."""
        optiland_object = self.artist_registry[artist]
        tooltip_text = self.tooltip_format(optiland_object)

        self.tooltip.set_text(tooltip_text)
        self.tooltip.xy = (event.xdata, event.ydata)
        self.tooltip.set_visible(True)
        self.fig.canvas.draw_idle()

    def clear_hover_effects(self):
        """Clears any active hover effects."""
        if self.active_artist and hasattr(self.active_artist, "get_linewidth"):
            if self.active_artist in self.original_props:
                self.active_artist.set_linewidth(
                    self.original_props[self.active_artist]["linewidth"]
                )
                del self.original_props[self.active_artist]
            self.active_artist = None
            self.tooltip.set_visible(False)
            self.fig.canvas.draw_idle()

    def default_tooltip_format(self, optiland_object):
        """Default formatter for the tooltip text."""
        if isinstance(optiland_object, Surface2D):
            return "Surface"
        elif isinstance(optiland_object, Lens2D):
            return "Lens"
        else:
            return "Ray Bundle"

    def show_info_panel(self, optiland_object):
        """Shows an information panel for the given object."""
        if self.info_panel:
            self.close_info_panel()

        info_text = self.get_info_text(optiland_object)

        self.info_panel = self.fig.add_axes([0.7, 0.7, 0.25, 0.25])
        self.info_panel.set_xticks([])
        self.info_panel.set_yticks([])
        self.info_panel.text(
            0.05,
            0.95,
            info_text,
            transform=self.info_panel.transAxes,
            fontsize=10,
            verticalalignment="top",
        )

        close_ax = self.fig.add_axes([0.9, 0.9, 0.05, 0.05])
        close_button = Button(close_ax, "X")
        cid = close_button.on_clicked(self.close_info_panel)
        self.info_panel.button = close_button
        self.info_panel.cid = cid

        self.fig.canvas.draw_idle()

    def close_info_panel(self, event=None):
        """Closes the information panel."""
        if self.info_panel:
            self.info_panel.button.disconnect(self.info_panel.cid)

            # Manually release the mouse grabber
            if self.fig.canvas.mouse_grabber == self.info_panel.button.ax:
                self.fig.canvas.release_mouse(self.info_panel.button.ax)

            self.fig.delaxes(self.info_panel)
            self.fig.delaxes(self.info_panel.button.ax)
            self.info_panel = None
            self.fig.canvas.draw_idle()

    def get_info_text(self, optiland_object):
        """Gets the detailed information text for the given object."""
        obj_type = type(optiland_object).__name__
        if obj_type in INFO_PROVIDER_REGISTRY:
            provider = INFO_PROVIDER_REGISTRY[obj_type]
            return provider.get_info(optiland_object)
        elif isinstance(optiland_object, tuple):  # Ray bundle
            provider = INFO_PROVIDER_REGISTRY["RayBundle"]
            return provider.get_info(optiland_object)
        else:
            return "No information available."
