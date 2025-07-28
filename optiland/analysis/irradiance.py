"""Irradiance Analysis

This module implements the necessary logic for the
irradiance analysis in a given optical system.
*note*: for now we only take of the incoherent irradiance.

The analysis is analogous to the SpotDiagram except that
instead of plotting the landing position of individual rays,
we accumulate their power on a detector and express the result
in W/mm^2.


Manuel Fragata Mendes, 2025
"""

import matplotlib.pyplot as plt
import numpy as _np  # Use _np for the binning. Later extend to
from matplotlib.colors import Colormap

# other backends
import optiland.backend as be

from .base import BaseAnalysis


class IncoherentIrradiance(BaseAnalysis):
    """Compute and visualise incoherent irradiance on the detector surface.
    For simplification, we assume that the detector surface = image surface.

     Attributes:
     ---
     optic : optiland.optic.Optic
         Reference to the optical system - must already define fields, wavelengths
         and, critically, a physical aperture on the chosen detector surface.
     res : tuple[int, int]
         Requested pixel count along (x,y) of the irradiance grid.
     px_size : tuple[float, float] | None
         Physical pixel pitch (dx,dy) in mm.  If ``None`` the pitch is
         derived from the surface aperture and `res`.
     num_rays : int
         Number of real rays launched for every (field,wavelength) pair.
     fields, wavelengths : tuple | "all"
         Convenience selectors that work exactly like those in
         `SpotDiagram` - default is to analyse all of them.
     detector_surface : int
         Index into `optic.surface_group.surfaces` that designates the detector
         plane to analyse (default=`-1`->image surface).
     data : list[list[be.ndarray]]
         2-D irradiance arrays for every (field,wvl) - outer index is field,
         inner index is wavelength.  Each array has shape
         (res[0],res[1]) with X as the row index so that
         ``irr_data[f][w][i,j]`` refers to X=i, Y=j.
     user_initial_rays : RealRays | None
         Optional user-provided initial rays (at the source/object plane)
         to be traced through the whole optical system.

     Methods
     ---
     view(figsize=(6,5), cmap="inferno") → plt.Figure, np.ndarray
         Display a false-colour irradiance map or cross-section plots for the
         current irradiance data.  If `cross_section` is specified, a 1D
         cross-section is plotted instead of a 2D map.
         If `normalize` is True, the irradiance maps are normalised to their
         peak value, otherwise absolute values are used.

     peak_irradiance() → list[list[float]]
         Return the maximum pixel value for every (field,wvl) pair.
    """

    def __init__(
        self,
        optic,
        num_rays: int = 5,
        res=(128, 128),
        px_size: float = None,
        detector_surface: int = -1,
        *,
        fields="all",
        wavelengths="all",
        distribution: str = "random",
        user_initial_rays=None,
    ):
        if fields == "all":
            self.fields = optic.fields.get_field_coords()
        else:
            self.fields = tuple(fields)

        self.num_rays = num_rays
        self.npix_x, self.npix_y = res
        self.px_size = (
            None if px_size is None else (float(px_size[0]), float(px_size[1]))
        )
        self.detector_surface = int(detector_surface)
        self.user_initial_rays = user_initial_rays
        self.distribution = distribution

        # The detector surface must have a physical aperture
        surf = optic.surface_group.surfaces[self.detector_surface]
        if surf.aperture is None:
            raise ValueError(
                "Detector surface has no physical aperture - set one "
                "(e.g. RectangularAperture) so that the detector size is defined."
            )

        super().__init__(optic, wavelengths)

    def view(
        self,
        fig_to_plot_on: plt.Figure = None,
        figsize: tuple = (6, 5),
        cmap: str | Colormap = "inferno",
        cross_section: tuple[str, int] = None,
        *,
        normalize: bool = True,
    ):
        """
        Display a false-colour irradiance map or cross-section plots for the current
        irradiance data.

        args :
            fig_to_plot_on : plt.Figure, optional
                Existing matplotlib Figure to plot on. If None, a new figure is created.
                Default is None.
            figsize : tuple, optional
                Size of each subplot as (width, height) in inches. Default is (6, 5).
            cmap : str or Colormap, optional
                Colormap to use for the irradiance map. Default is "inferno".
            cross_section : tuple[str, int], optional
                If provided, plot a cross-section instead of a 2D map. Should be a tuple
                of ('cross-x' or 'cross-y', index), where index is the slice index along
                the specified axis.
                If None, a 2D irradiance map is plotted. Default is None.
            normalize : bool, optional
                If True, normalize irradiance maps to their peak value. If False, use
                absolute values.
                Default is True.

        Returns :
            fig : matplotlib.figure.Figure
                The matplotlib Figure object containing the plot(s).
            axs : numpy.ndarray or None
                Array of Axes objects for the subplots, or None if plotting on an
                existing figure.

        Notes
        -----
        - If no valid irradiance data is available, the method prints a warning
        and returns None.
        - If `cross_section` is invalid or not provided, a 2D irradiance map is
        shown.
        - The method supports plotting multiple fields and wavelengths as a grid
        of subplots.
        - Colorbars and axis labels are automatically added to each subplot.
        """
        is_gui_embedding = fig_to_plot_on is not None

        if not self.data:  # Changed from self.irr_data
            print("No irradiance data to display.")
            return

        plot_cross_section_requested = False
        valid_cross_section_request = False
        cs_axis_type = None
        cs_slice_idx = -1

        if cross_section is not None:
            if isinstance(cross_section, tuple) and len(cross_section) == 2:
                axis_type_in, slice_idx_in = cross_section
                if (
                    isinstance(axis_type_in, str)
                    and axis_type_in.lower() in ["cross-x", "cross-y"]
                    and (isinstance(slice_idx_in, int) or slice_idx_in is None)
                ):
                    plot_cross_section_requested = True
                    valid_cross_section_request = True
                    cs_axis_type = axis_type_in.lower()
                    cs_slice_idx = slice_idx_in if slice_idx_in is not None else -1
                    cross_section_title = self._get_cross_section_title(
                        cs_axis_type,
                        cs_slice_idx,
                        normalize=normalize,
                    )
                else:
                    print(
                        "[IncoherentIrradiance] Warning: Invalid cross_section_info "
                        "format. Expected ('cross-x' or 'cross-y', int). Defaulting "
                        "to 2D plot."
                    )
            else:
                print(
                    "[IncoherentIrradiance] Warning: Invalid cross_section_info type. "
                    "Expected tuple. Defaulting to 2D plot."
                )

        # logic for vmin_plot, vmax_plot calculation

        all_irr_values_list = []
        n_fields = len(self.fields)
        n_wavelengths = len(self.wavelengths)

        if is_gui_embedding:
            fig = fig_to_plot_on
            fig.set_size_inches(figsize[0] * n_wavelengths, figsize[1] * n_fields)
            fig.clear()
            axs = fig.subplots(
                nrows=n_fields,
                ncols=n_wavelengths,
                sharex=True,
                sharey=True,
                squeeze=False,
            )
        else:
            fig, axs = plt.subplots(
                nrows=n_fields,
                ncols=n_wavelengths,
                figsize=(figsize[0] * n_wavelengths, figsize[1] * n_fields),
                squeeze=False,
                tight_layout=True,
                sharex=True,
                sharey=True,
            )

        main_title = "Irradiance Analysis"

        for field_block_idx, field_block in enumerate(
            self.data
        ):  # Changed from self.irr_data
            if not field_block:
                print(f"Warning: Field block {field_block_idx} is empty.")
                continue
            for entry_idx, entry in enumerate(field_block):
                if entry is None:
                    print(
                        f"Warning: Entry {entry_idx} in field block {field_block_idx} "
                        "is None."
                    )
                    continue
                irr_map, x_edges, y_edges = entry  # self.data stores tuples
                if irr_map is None:
                    print(
                        f"Warning: Irradiance map in entry {entry_idx}, "
                        f"field block {field_block_idx} is None."
                    )
                    continue
                all_irr_values_list.append(be.to_numpy(irr_map).flatten())

        if not all_irr_values_list:
            print("No valid irradiance map data found to plot.")
            return

        if not normalize:
            all_irr_values = _np.concatenate(all_irr_values_list)
            if len(all_irr_values) == 0:
                print(
                    "No valid irradiance values to plot after concatenation "
                    "(all maps might be empty)."
                )
                return
            vmin_plot, vmax_plot = _np.min(all_irr_values), _np.max(all_irr_values)
            if vmin_plot == vmax_plot:  # Handle case where all values are the same
                vmin_plot -= 0.1 * abs(vmin_plot) if vmin_plot != 0 else 0.1
                vmax_plot += 0.1 * abs(vmax_plot) if vmax_plot != 0 else 0.1
                if vmin_plot == vmax_plot:  # Still same (e.g. all zeros)
                    vmin_plot, vmax_plot = 0.0, 1.0  # Default range
        else:
            vmin_plot, vmax_plot = 0.0, 1.0  # Normalized range

        for f_idx, field_block in enumerate(self.data):  # Changed from self.irr_data
            for w_idx, entry_data in enumerate(field_block):  # entry_data is the tuple
                irr_map, x_edges, y_edges = entry_data
                if normalize:
                    peak_val = self.peak_irradiance()[f_idx][w_idx]
                    if peak_val > 0:
                        irr_map = irr_map / peak_val
                # title string stuff
                title_str_base = ""
                if self.user_initial_rays is not None:
                    # Original code uses self.fields[f_idx][0] for label
                    field_label_info_val = (
                        self.fields[f_idx][0]
                        if isinstance(self.fields[f_idx], tuple)
                        else self.fields[f_idx]
                    )
                    title_str_base = f"(User Rays: {field_label_info_val})"
                else:
                    field_coord = self.fields[f_idx]
                    wavelength_val = self.wavelengths[w_idx]
                    title_str_base = (
                        f"Field {f_idx} {field_coord}, "
                        f"$\\lambda_{w_idx}$ = {wavelength_val:.3f} µm"
                    )

                # call helper cross section
                if plot_cross_section_requested and valid_cross_section_request:
                    self._plot_cross_section(
                        axs[f_idx, w_idx],
                        irr_map,
                        x_edges,
                        y_edges,
                        cs_axis_type,
                        cs_slice_idx,
                        title_str_base,
                        normalize,
                    )

                else:
                    # 2D plotting stuff
                    im = axs[f_idx, w_idx].imshow(
                        be.to_numpy(irr_map).T,
                        aspect="auto",
                        origin="lower",
                        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                        cmap=cmap,
                        vmin=vmin_plot,
                        vmax=vmax_plot,
                    )
                    cbar_lbl = (
                        "Normalized Irradiance"
                        if normalize
                        else "Irradiance (W/mm$^2$)"
                    )
                    fig.colorbar(im, ax=axs[f_idx, w_idx], label=cbar_lbl)
                    axs[f_idx, w_idx].set_xlabel("X (mm)")
                    axs[f_idx, w_idx].set_ylabel("Y (mm)")
                    axs[f_idx, w_idx].set_title(title_str_base)
                    axs[f_idx, w_idx].set_aspect("equal")

        if plot_cross_section_requested and valid_cross_section_request:
            main_title += cross_section_title
        fig.suptitle(main_title)

        if is_gui_embedding and hasattr(fig, "canvas"):
            fig.canvas.draw_idle()

        return fig, axs

    # --- helper functions ---

    def peak_irradiance(self):
        """Maximum pixel value for each (field,wvl) pair."""
        return [
            [be.max(irr) for irr, *_ in fblock] for fblock in self.data
        ]  # Changed from self.irr_data

    def _plot_cross_section(
        self,
        ax: plt.Axes | None,
        irr_map_be,
        x_edges,
        y_edges,
        axis_type: str,
        slice_idx: int,
        title_prefix: str,
        normalize: bool = True,
    ) -> None:
        """Helper method to plot a 1D cross-section of the irradiance map."""
        irr_map_np = be.to_numpy(irr_map_be)

        # Calculate pixel centers
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        if axis_type == "cross-x":  # cross section along X, data varies with Y
            if slice_idx < 0:
                slice_idx = irr_map_np.shape[0] // 2  # middle pixel
            if not (0 <= slice_idx < irr_map_np.shape[0]):
                print(
                    f"[IncoherentIrradiance] Warning: X-slice index {slice_idx} is out "
                    f"of bounds for map shape {irr_map_np.shape}. Skipping plot."
                )
                return

            data_to_plot = irr_map_np[slice_idx, :]
            coords_to_plot_against = y_centers
            ax.set_xlabel("Y (mm)")

        elif axis_type == "cross-y":  # cross section along Y, data varies with X
            if slice_idx < 0:
                slice_idx = irr_map_np.shape[1] // 2  # middle pixel
            if not (0 <= slice_idx < irr_map_np.shape[1]):
                print(
                    f"[IncoherentIrradiance] Warning: Y-slice index {slice_idx} is "
                    f"out of bounds for map shape {irr_map_np.shape}. Skipping plot."
                )
                return

            data_to_plot = irr_map_np[:, slice_idx]
            coords_to_plot_against = x_centers
            ax.set_xlabel("X (mm)")
        else:
            return

        if normalize:
            peak_val = data_to_plot.max()
            if peak_val > 0:
                data_to_plot = data_to_plot / peak_val

        ax.plot(coords_to_plot_against, data_to_plot, linestyle="-")
        ylbl = "Normalized Irradiance" if normalize else "Irradiance (W/mm$^2$)"
        ax.set_ylabel(ylbl)
        ax.set_title(title_prefix)
        ax.grid(True)

    def _get_cross_section_title(
        self,
        axis_type: str,
        slice_idx: int,
        normalize: bool = True,
    ) -> str:
        """
        Generate a descriptive title for a cross-section plot of an irradiance map.

        Parameters
        ----------
        data : Any
            The data structure containing the irradiance map and its axis edges.
            (Note: This parameter is not used directly in the function body.)
        axis_type : str
            The type of cross-section to plot. Must be either "cross-x" (cross-section
              along X, varies with Y)
            or "cross-y" (cross-section along Y, varies with X).
        slice_idx : int
            The index of the slice to use for the cross-section. If negative, the
            middle slice is used.
        normalize : bool, optional
            Whether to indicate normalization in the title. Default is True.

        Returns
        -------
        str
            A formatted string suitable as a plot title, indicating the axis,
            position, index, and normalization status.
            Returns an empty string if the slice index is out of bounds.
        """

        irr_map_be = self.data[0][0][0]  # Get the first irradiance map
        x_edges = self.data[0][0][1]  # Get the x_edges
        y_edges = self.data[0][0][2]  # Get the y_edges

        irr_map_np = be.to_numpy(irr_map_be)

        # Calculate pixel centers
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        cross_section_title = ""

        if axis_type == "cross-x":  # cross section along X, data varies with Y
            if slice_idx < 0:
                slice_idx = irr_map_np.shape[0] // 2  # middle pixel
            if not (0 <= slice_idx < irr_map_np.shape[0]):
                return cross_section_title

            cross_section_title += (
                f" - $X$-Cross-section at $X$ ≈ {x_centers[slice_idx]:.2f} mm "
            )
            cross_section_title += f"(index {slice_idx}/{irr_map_np.shape[0]})"

        elif axis_type == "cross-y":  # cross section along Y, data varies with X
            if slice_idx < 0:
                slice_idx = irr_map_np.shape[1] // 2  # middle pixel
            if not (0 <= slice_idx < irr_map_np.shape[1]):
                return cross_section_title

            cross_section_title += (
                f" - $Y$-Cross-section at $Y$ ≈ {y_centers[slice_idx]:.2f} "
            )
            cross_section_title += f"mm (index {slice_idx}/{irr_map_np.shape[1]})"

        if normalize:
            cross_section_title += " (normalized)"

        return cross_section_title

    # --- data generation functions ---

    def _generate_data(self):  # Signature changed
        data = []
        # Use self.fields, self.wavelengths, self.distribution, self.user_initial_rays
        for field in self.fields:
            f_block = []
            for (
                wl
            ) in self.wavelengths:  # self.wavelengths is now a list from BaseAnalysis
                f_block.append(
                    self._generate_field_data(
                        field, wl, self.distribution, self.user_initial_rays
                    )
                )
            data.append(f_block)
        return data

    def _generate_field_data(
        self, field, wavelength, distribution, user_initial_rays
    ):  # Signature unchanged
        """Trace rays and bin their power into the pixels of the detector."""
        # Uses self.num_rays internally
        if user_initial_rays is None:
            Hx, Hy = field
            self.optic.trace(Hx, Hy, wavelength, self.num_rays, distribution)
        else:
            self.optic.surface_group.trace(user_initial_rays)

        # get ray coords on detector surface
        surf = self.optic.surface_group.surfaces[self.detector_surface]
        x_g, y_g, z_g = surf.x, surf.y, surf.z
        power = surf.intensity

        from optiland.visualization.system.utils import transform

        x_local, y_local, _ = transform(x_g, y_g, z_g, surf, is_global=True)
        x_np = be.to_numpy(x_local)
        y_np = be.to_numpy(y_local)
        power_np = be.to_numpy(power)

        valid = power_np > 0.0
        x_np, y_np, power_np = x_np[valid], y_np[valid], power_np[valid]

        # get the physical siize of the detector
        x_min, x_max, y_min, y_max = surf.aperture.extent
        if self.px_size is None:
            x_edges = _np.linspace(x_min, x_max, self.npix_x + 1, dtype=float)
            y_edges = _np.linspace(y_min, y_max, self.npix_y + 1, dtype=float)
            pixel_area = (x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0])
        else:
            dx, dy = self.px_size
            x_edges = _np.arange(x_min, x_max + 0.5 * dx, dx, dtype=float)
            y_edges = _np.arange(y_min, y_max + 0.5 * dy, dy, dtype=float)
            pixel_area = dx * dy
            # if the pitch supplied by the user gives a different res
            # than the one requested warn once
            exp_nx = len(x_edges) - 1
            exp_ny = len(y_edges) - 1
            if (exp_nx, exp_ny) != (self.npix_x, self.npix_y):
                print(
                    f"[IncoherentIrradiance] Warning: res parameter ignored - "
                    f"derived from px_size instead → ({exp_nx},{exp_ny}) pixels"
                )
                self.npix_x, self.npix_y = exp_nx, exp_ny

        # 2d binning with numpy histogram
        hist, _, _ = _np.histogram2d(
            x_np, y_np, bins=[x_edges, y_edges], weights=power_np
        )
        irr = hist / pixel_area
        return be.array(irr), x_edges, y_edges
