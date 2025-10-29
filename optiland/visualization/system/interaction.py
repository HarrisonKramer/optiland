"""Interaction Manager Module.

This module provides the InteractionManager class for handling user interactions
with Matplotlib-based visualizations of optical systems.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.axes

from optiland.visualization.info.providers import INFO_PROVIDER_REGISTRY
from optiland.visualization.system.lens import Lens2D
from optiland.visualization.system.surface import Surface2D

if TYPE_CHECKING:
    import matplotlib.figure
    from matplotlib.backend_bases import Event


class InteractionManager:
    """Manages user interactions for optical system visualizations.

    This class connects to a Matplotlib figure's event loop to handle
    mouse events, such as hovering and clicking, on plotted artists.

    Attributes:
        fig (matplotlib.figure.Figure): The Matplotlib figure to connect to.
        ax (matplotlib.axes.Axes): The Matplotlib axes containing the artists.
        artist_registry (dict): Maps Matplotlib artists to Optiland objects.
        active_artist (matplotlib.artist.Artist | None): The artist currently
            under the mouse cursor.
        original_props (dict): Stores original properties of artists before
            highlighting, to allow restoration.
        tooltip (matplotlib.text.Annotation): The annotation object used for
            displaying tooltips.
        tooltip_format (callable): A function that takes an Optiland object
            and returns a string for its tooltip.
        info_panel (matplotlib.axes.Axes | None): The axes object for the
            detailed information panel, if currently displayed.
        cids (list[int]): A list of connection IDs for Matplotlib events.
        info_panel_cid (int): The connection ID for the info panel's click event.
    """

    def __init__(
        self,
        fig: matplotlib.figure.Figure,
        ax: matplotlib.axes.Axes,
        tooltip_format: callable | None = None,
    ):
        """Initializes the InteractionManager.

        Args:
            fig (matplotlib.figure.Figure): The Matplotlib figure to connect to.
            ax (matplotlib.axes.Axes): The Matplotlib axes containing the artists.
            tooltip_format (callable | None, optional): A function to format
                tooltip text. If None, `default_tooltip_format` is used.
        """
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
            bbox={"boxstyle": "round", "fc": "w"},
            arrowprops={"arrowstyle": "->"},
        )
        self.tooltip.set_visible(False)
        self.tooltip_format = tooltip_format or self.default_tooltip_format

        self.info_panel = None
        self.info_panel_cid = None
        self.cids = []
        self.connect()

    def register_artist(
        self, artist: matplotlib.artist.Artist, optiland_object: object
    ):
        """Registers a Matplotlib artist with its corresponding Optiland object.

        Args:
            artist (matplotlib.artist.Artist): The Matplotlib artist (e.g.,
                Line2D, Patch) to be tracked.
            optiland_object (object): The Optiland object (e.g., Surface2D,
                Lens2D) that the artist represents.
        """
        self.artist_registry[artist] = optiland_object

    def connect(self):
        """Connects to the Matplotlib event loop.

        This method connects the `on_hover` and `on_click` methods to the
        figure's canvas events.
        """
        if not self.cids:
            cid_hover = self.fig.canvas.mpl_connect(
                "motion_notify_event", self.on_hover
            )
            cid_click = self.fig.canvas.mpl_connect("button_press_event", self.on_click)
            self.cids.extend([cid_hover, cid_click])

    def disconnect(self):
        """Disconnects from the Matplotlib event loop.

        This is useful for temporarily disabling interactions, e.g., when an
        info panel is open.
        """
        for cid in self.cids:
            self.fig.canvas.mpl_disconnect(cid)
        self.cids = []

    def on_hover(self, event: Event):
        """Handles mouse hover events.

        Identifies the artist under the cursor, highlights it, and displays
        a tooltip.

        Args:
            event (matplotlib.backend_bases.Event): The motion notify event.
        """
        if event.inaxes != self.ax:
            if self.active_artist:
                self.clear_hover_effects()
            return

        found_artist = None
        # Prioritize surfaces over other artists for hover detection
        artists = sorted(
            self.artist_registry.keys(),
            key=lambda a: isinstance(self.artist_registry[a], Surface2D),
            reverse=True,
        )
        for artist in artists:
            contains, _ = artist.contains(event)
            if contains:
                found_artist = artist
                break

        if self.active_artist != found_artist:
            if self.active_artist:
                self.clear_hover_effects()

            if found_artist:
                self.active_artist = found_artist
                self.highlight_artist(found_artist)
                self.show_tooltip(found_artist, event)

    def on_click(self, event: Event):
        """Handles mouse click events.

        Identifies the clicked artist and shows a detailed info panel for its
        corresponding Optiland object.

        Args:
            event (matplotlib.backend_bases.Event): The button press event.
        """
        if event.inaxes != self.ax:
            return

        for artist, optiland_object in self.artist_registry.items():
            contains, _ = artist.contains(event)
            if contains:
                self.show_info_panel(optiland_object)
                break

    def highlight_artist(self, artist: matplotlib.artist.Artist):
        """Highlights the given artist by increasing its linewidth.

        If the artist's object has a 'bundle_id', all artists in that
        bundle are highlighted.

        Args:
            artist (matplotlib.artist.Artist): The artist to highlight.
        """
        obj = self.artist_registry.get(artist)
        if obj is None:
            return

        if hasattr(obj, "bundle_id"):
            for art, o in self.artist_registry.items():
                if hasattr(o, "bundle_id") and o.bundle_id == obj.bundle_id:
                    self.original_props[art] = {"linewidth": art.get_linewidth()}
                    art.set_linewidth(art.get_linewidth() * 2)
        elif hasattr(artist, "get_linewidth"):
            self.original_props[artist] = {"linewidth": artist.get_linewidth()}
            artist.set_linewidth(artist.get_linewidth() * 2)
        self.fig.canvas.draw_idle()

    def show_tooltip(self, artist: matplotlib.artist.Artist, event: Event):
        """Shows a tooltip for the given artist at the event location.

        Args:
            artist (matplotlib.artist.Artist): The artist to show a tooltip for.
            event (matplotlib.backend_bases.Event): The mouse event, used to
                position the tooltip.
        """
        optiland_object = self.artist_registry[artist]
        tooltip_text = self.tooltip_format(optiland_object)

        self.tooltip.set_text(tooltip_text)
        self.tooltip.xy = (event.xdata, event.ydata)
        self.tooltip.set_visible(True)
        self.fig.canvas.draw_idle()

    def clear_hover_effects(self):
        """Clears any active hover effects.

        Restores the original properties of the active artist and hides
        the tooltip.
        """
        if self.active_artist:
            obj = self.artist_registry.get(self.active_artist)
            if obj and hasattr(obj, "bundle_id"):
                for art, o in self.artist_registry.items():
                    if (
                        hasattr(o, "bundle_id")
                        and o.bundle_id == obj.bundle_id
                        and art in self.original_props
                    ):
                        art.set_linewidth(self.original_props[art]["linewidth"])
                        del self.original_props[art]
            elif self.active_artist in self.original_props and hasattr(
                self.active_artist, "get_linewidth"
            ):
                self.active_artist.set_linewidth(
                    self.original_props[self.active_artist]["linewidth"]
                )
                del self.original_props[self.active_artist]

            self.active_artist = None
            self.tooltip.set_visible(False)
            self.fig.canvas.draw_idle()

    def default_tooltip_format(self, optiland_object: object) -> str:
        """Default formatter for the tooltip text.

        Args:
            optiland_object (object): The Optiland object.

        Returns:
            str: The formatted tooltip text.
        """
        if isinstance(optiland_object, Surface2D):
            return "Surface"
        elif isinstance(optiland_object, Lens2D):
            return "Lens"
        else:
            # Assume it's a ray bundle if not a known type
            return "Ray Bundle"

    def show_info_panel(self, optiland_object: object):
        """Shows a detailed information panel for the given object.

        This disconnects the main plot events and creates a new axes
        for the info panel.

        Args:
            optiland_object (object): The Optiland object to display info for.
        """
        if self.info_panel:
            self.close_info_panel()

        self.disconnect()  # Disconnect main plot events

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
            bbox={"boxstyle": "round", "fc": "w", "alpha": 0.9},
        )

        # Draw a manual "X" and set up a custom click handler
        self.info_panel.text(
            0.95,
            0.95,
            "X",
            transform=self.info_panel.transAxes,
            ha="right",
            va="top",
            fontsize=12,
            color="red",
            fontweight="bold",
        )
        self.info_panel_cid = self.fig.canvas.mpl_connect(
            "button_press_event", self.on_info_panel_click
        )

        self.fig.canvas.draw_idle()

    def on_info_panel_click(self, event: Event):
        """Handles click events on the info panel.

        Specifically looks for clicks on the 'X' to close the panel.

        Args:
            event (matplotlib.backend_bases.Event): The button press event.
        """
        # Close the panel if it exists, the click is inside it,
        # and is in the top-right corner
        if (
            self.info_panel
            and event.inaxes == self.info_panel
            and event.xdata is not None
            and event.ydata is not None
            and event.xdata > 0.9
            and event.ydata > 0.9
        ):
            # Click in info panel top-right corner (approximating the 'X') -> close
            self.close_info_panel()

    def close_info_panel(self, event: Event | None = None):
        """Closes the information panel and reconnects main plot events.

        Args:
            event (matplotlib.backend_bases.Event | None, optional): The event
                that triggered the close. Not currently used.
        """
        if self.info_panel:
            if self.info_panel_cid:
                self.fig.canvas.mpl_disconnect(self.info_panel_cid)
                self.info_panel_cid = None
            self.fig.delaxes(self.info_panel)
            self.info_panel = None
            self.fig.canvas.draw_idle()
            self.connect()  # Reconnect main plot events

    def get_info_text(self, optiland_object: object) -> str:
        """Gets the detailed information text for the given object.

        Uses the `INFO_PROVIDER_REGISTRY` to find the correct info provider.

        Args:
            optiland_object (object): The Optiland object.

        Returns:
            str: The detailed information text for the object.
        """
        # Import locally to avoid circular dependencies
        from optiland.visualization.system.ray_bundle import RayBundle

        if isinstance(optiland_object, RayBundle):
            provider = INFO_PROVIDER_REGISTRY["RayBundle"]
            return provider.get_info(optiland_object)

        obj_type = type(optiland_object).__name__
        if obj_type in INFO_PROVIDER_REGISTRY:
            provider = INFO_PROVIDER_REGISTRY[obj_type]
            return provider.get_info(optiland_object)
        else:
            return "No information available."
