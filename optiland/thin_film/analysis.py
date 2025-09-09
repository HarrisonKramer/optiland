from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import optiland.backend as be

if TYPE_CHECKING:
    from .stack import ThinFilmStack

import matplotlib.pyplot as plt

Pol = Literal["s", "p", "u"]
PlotType = Literal["R", "T", "A"]
Array: TypeAlias = Any  # be.ndarray


class SpectralAnalyzer:
    """Class for analyzing thin film stacks optical response (R/T/A).

    Attributes:
        stack (ThinFilmStack): The thin film stack to be analyzed.
    """

    def __init__(self, stack: ThinFilmStack) -> None:
        """Initialize the SpectralAnalyzer with a ThinFilmStack.

        Args:
            stack (ThinFilmStack): The thin film stack to be analyzed.
        """
        self.stack = stack

    def plot(
        self,
        wavelength_um: float | Array,
        aoi_deg: float | Array = 0.0,
        polarization: Pol = "u",
        to_plot: PlotType | list[PlotType] = "R",
        ax: plt.Axes = None,
    ) -> plt.Figure:
        """Plot R/T/A vs wavelength and/or AOI for given polarization.

        This method adapts older implementations where `self` was the ThinFilmStack.
        Internally it calls the stack's compute_rtRAT_nm_deg method.

        Args:
            wavelength_um: Wavelength(s) in micrometers (scalar or array).
            aoi_deg: Angle(s) of incidence in degrees (scalar or array), default 0.
            polarization: 's', 'p' or 'u' (unpolarized averages powers of s and p),
                default 'u'.
            to_plot: 'R', 'T', 'A' or list of these, default 'R'.
            ax: Optional matplotlib Axes to plot on. If None, a new figure
                and axes are created.
        Returns:
            fig, ax|axs: The matplotlib Figure and Axes object(s) containing the plot.
        """

        wl_array = be.atleast_1d(wavelength_um) * 1000.0  # convert to nm
        aoi_array = be.atleast_1d(aoi_deg)

        # Use the underlying ThinFilmStack to compute R/T/A (expects nm, deg)
        rta_data = self.stack.compute_rtRAT_nm_deg(wl_array, aoi_array, polarization)

        if isinstance(to_plot, str):
            to_plot = [to_plot]

        # Case 1: wavelength is array, AOI is scalar
        if len(wl_array) > 1 and len(aoi_array) == 1:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.figure
            for quantity in to_plot:
                if quantity not in ("R", "T", "A"):
                    raise ValueError("to_plot must be 'R', 'T', 'A' or a list of these")
                ax.plot(
                    wl_array,
                    rta_data[quantity].flatten(),
                    label=f"{quantity}, {polarization}-pol, AOI={float(aoi_deg)}°",
                )
            ax.set_xlabel("$\\lambda$ (nm)")
            ax.set_ylabel("Power fraction")
            ax.set_xlim(float(wl_array.min()), float(wl_array.max()))
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend()
            return fig, ax

        # Case 2: AOI is array, wavelength is scalar
        elif len(aoi_array) > 1 and len(wl_array) == 1:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.figure
            for quantity in to_plot:
                if quantity not in ("R", "T", "A"):
                    raise ValueError("to_plot must be 'R', 'T', 'A' or a list of these")
                ax.plot(
                    aoi_array,
                    rta_data[quantity].flatten(),
                    label=f"{quantity}, {polarization}-pol,"
                    + f" $\\lambda$={float(wl_array[0])} nm",
                )
            ax.set_xlabel("AOI (°)")
            ax.set_ylabel("Power fraction")
            ax.set_xlim(float(aoi_array.min()), float(aoi_array.max()))
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend()
            return fig, ax

        # Case 3: Both are arrays - 2D plot using pcolormesh
        elif len(wl_array) > 1 and len(aoi_array) > 1:
            WL, AOI = be.meshgrid(wl_array, aoi_array, indexing="ij")

            # If multiple quantities requested, create one subplot per quantity
            fig, axs = plt.subplots(len(to_plot), 1, figsize=(6, 4 * len(to_plot)))
            if len(to_plot) == 1:
                axs = [axs]
            for ax_idx, quantity in enumerate(to_plot):
                if quantity not in ("R", "T", "A"):
                    raise ValueError("to_plot must be 'R', 'T', 'A' or a list of these")
                ax_i = axs[ax_idx]
                im = ax_i.pcolormesh(WL, AOI, rta_data[quantity], shading="auto")
                ax_i.set_xlabel("$\\lambda$ (nm)")
                ax_i.set_ylabel("AOI (°)")
                ax_i.set_title(f"{quantity}, {polarization}-pol")
                fig.colorbar(im, ax=ax_i, label="Power fraction")
            fig.tight_layout()
            return fig, axs

        # Case 4: Both are scalars - single point plot
        else:
            raise ValueError(
                "At least one of wavelength_um or aoi_deg must be an array"
            )
