"""Interaction Manager Module

This module provides the InteractionManager class for handling user interactions
with Matplotlib-based visualizations of optical systems.

"""

from __future__ import annotations

from threading import Timer

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

    def __init__(self, fig, ax, optic, tooltip_format=None):
        self.fig = fig
        self.ax = ax
        self.optic = optic
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
        )
        self.tooltip.set_visible(False)
        self.tooltip_format = tooltip_format or self.default_tooltip_format

        self.info_panel = None
        self.cids = []
        self.hover_timer = None
        self.last_hover_time = 0
        self.hover_delay = 0.7  # seconds
        self.connect()

    def register_artist(self, artist, optiland_object):
        """Registers a Matplotlib artist with its corresponding Optiland object."""
        self.artist_registry[artist] = optiland_object

    def connect(self):
        """Connects to the Matplotlib event loop."""
        if not self.cids:
            cid_hover = self.fig.canvas.mpl_connect(
                "motion_notify_event", self.on_hover
            )
            self.cids.extend([cid_hover])

    def disconnect(self):
        """Disconnects from the Matplotlib event loop."""
        for cid in self.cids:
            self.fig.canvas.mpl_disconnect(cid)
        self.cids = []

    def on_hover(self, event):
        """Handles hover events to show tooltips and highlight artists."""
        if event.inaxes != self.ax:
            if self.active_artist:
                self.clear_hover_effects()
            return

        found_artist = None
        # Prioritize surfaces over other artists
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
            if self.hover_timer:
                self.hover_timer.cancel()

            if found_artist:
                self.active_artist = found_artist
                self.hover_timer = Timer(
                    self.hover_delay, self.show_tooltip, args=[found_artist, event]
                )
                self.hover_timer.start()

    # TODO: Re-enable pop-up box functionality in a future update.
    # The following methods are temporarily disabled.
    # def on_click(self, event):
    # def show_info_panel(self, optiland_object):
    # def on_info_panel_click(self, event):
    # def close_info_panel(self, event=None):

    def highlight_artist(self, artist):
        """Highlights the given artist."""
        obj = self.artist_registry[artist]
        if hasattr(obj, "bundle_id"):
            for art, o in self.artist_registry.items():
                if hasattr(o, "bundle_id") and o.bundle_id == obj.bundle_id:
                    self.original_props[art] = {"linewidth": art.get_linewidth()}
                    art.set_linewidth(art.get_linewidth() * 2)
        elif hasattr(artist, "get_linewidth"):
            self.original_props[artist] = {"linewidth": artist.get_linewidth()}
            artist.set_linewidth(artist.get_linewidth() * 2)
        self.fig.canvas.draw_idle()

    def show_tooltip(self, artist, event):
        """Shows a tooltip for the given artist."""
        self.highlight_artist(artist)
        optiland_object = self.artist_registry[artist]
        tooltip_text = self.tooltip_format(optiland_object)

        self.tooltip.set_text(tooltip_text)
        self.tooltip.xy = (event.xdata, event.ydata)
        self.tooltip.set_visible(True)
        self.fig.canvas.draw_idle()

    def clear_hover_effects(self):
        """Clears any active hover effects."""
        if self.hover_timer:
            self.hover_timer.cancel()
        if self.active_artist:
            obj = self.artist_registry[self.active_artist]
            if hasattr(obj, "bundle_id"):
                for art, o in self.artist_registry.items():
                    if (
                        hasattr(o, "bundle_id")
                        and o.bundle_id == obj.bundle_id
                        and art in self.original_props
                    ):
                        art.set_linewidth(self.original_props[art]["linewidth"])
                        del self.original_props[art]
            elif (
                hasattr(self.active_artist, "get_linewidth")
                and self.active_artist in self.original_props
            ):
                self.active_artist.set_linewidth(
                    self.original_props[self.active_artist]["linewidth"]
                )
                del self.original_props[self.active_artist]
            self.active_artist = None
            self.tooltip.set_visible(False)
            self.fig.canvas.draw_idle()

    def default_tooltip_format(self, optiland_object):
        """Default formatter for the tooltip text."""
        from optiland.visualization.info.providers import SurfaceInfoProvider
        from optiland.visualization.system.ray_bundle import RayBundle

        if isinstance(optiland_object, Surface2D):
            provider = SurfaceInfoProvider(self.optic.surface_group)
            return provider.get_info(optiland_object)
        elif isinstance(optiland_object, RayBundle):
            provider = INFO_PROVIDER_REGISTRY["RayBundle"]
            return provider.get_info(optiland_object)
        elif isinstance(optiland_object, Lens2D):
            provider = INFO_PROVIDER_REGISTRY["Lens2D"]
            return provider.get_info(optiland_object)
        else:
            return "No information available."

    def show_info_panel(self, optiland_object):
        """Shows an information panel for the given object."""
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
        )
        self.info_panel_cid = self.fig.canvas.mpl_connect(
            "button_press_event", self.on_info_panel_click
        )

        self.fig.canvas.draw_idle()

    def on_info_panel_click(self, event):
        """Handles click events on the info panel."""
        if (
            self.info_panel
            and event.inaxes == self.info_panel
            and event.x > 0.9
            and event.y > 0.9
        ):
            self.close_info_panel()

    def close_info_panel(self, event=None):
        """Closes the information panel."""
        if self.info_panel:
            self.fig.canvas.mpl_disconnect(self.info_panel_cid)
            self.fig.delaxes(self.info_panel)
            self.info_panel = None
            self.fig.canvas.draw_idle()
            self.connect()  # Reconnect main plot events

    def get_info_text(self, optiland_object):
        """Gets the detailed information text for the given object."""
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
