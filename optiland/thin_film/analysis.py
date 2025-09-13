"""Thin film analysis class.

This provides core functions for thin film optics calculations using the
transfer matrix method (TMM).

Corentin Nannini, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import optiland.backend as be

if TYPE_CHECKING:
    from .stack import ThinFilmStack

import matplotlib.pyplot as plt

# Physical constants
SPEED_OF_LIGHT = 299792458.0  # m/s
PLANCK_CONSTANT = 6.62607015e-34  # J⋅s
ELEMENTARY_CHARGE = 1.602176634e-19  # C
PLANCK_EV = PLANCK_CONSTANT / ELEMENTARY_CHARGE  # eV⋅s ≈ 4.135667696e-15

# Type definitions
Pol = Literal["s", "p", "u"]
PlotType = Literal["R", "T", "A"]
Array: TypeAlias = Any  # be.ndarray
WavelengthUnit = Literal[
    "um", "nm", "frequency", "energy", "wavenumber", "relative_wavenumber"
]
AngleUnit = Literal["deg", "rad"]


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

    def _convert_to_wavelength_um(
        self, values: float | Array, unit: WavelengthUnit
    ) -> Array:
        """Convert input values to wavelength in micrometers.

        Args:
            values: Input values in the specified unit
            unit: Unit of the input values

        Returns:
            Wavelength values in micrometers

        Raises:
            ValueError: If relative_wavenumber is requested but no reference
            wavelength is set
        """
        values = be.atleast_1d(values)

        if unit == "um":
            return values
        elif unit == "nm":
            return values / 1000.0
        elif unit == "frequency":  # Hz to um
            # λ (m) = c / ν, then convert to um
            return (SPEED_OF_LIGHT / values) * 1e6
        elif unit == "energy":  # eV to um
            # E = hf = hc/λ, so λ = hc/E
            # λ (m) = (h * c) / E, then convert to um
            return (PLANCK_EV * SPEED_OF_LIGHT / values) * 1e6
        elif unit == "wavenumber":  # cm⁻¹ to um
            # k = 1/λ (cm⁻¹), so λ (cm) = 1/k, then convert to um
            return 1e4 / values
        elif unit == "relative_wavenumber":
            if self.stack.reference_wl_um is None:
                raise ValueError("reference_wl_um must be set for relative_wavenumber")
            # k_rel = k / k_ref, so k = k_rel * k_ref = k_rel / λ_ref
            # λ = 1/k = λ_ref / k_rel
            return self.stack.reference_wl_um / values
        else:
            raise ValueError(f"Unknown wavelength unit: {unit}")

    def _get_wavelength_axis_label(self, unit: WavelengthUnit) -> str:
        """Get the appropriate axis label for wavelength unit."""
        labels = {
            "um": r"$\lambda$ ($\mu$m)",
            "nm": r"$\lambda$ (nm)",
            "frequency": r"$\nu$ (Hz)",
            "energy": r"$E$ (eV)",
            "wavenumber": r"$k$ (cm$^{-1}$)",
            "relative_wavenumber": r"$k/k_{\mathrm{ref}}$",
        }
        return labels[unit]

    def _convert_wavelength_for_plotting(
        self, wavelength_um: Array, unit: WavelengthUnit
    ) -> Array:
        """Convert wavelength in um to the desired unit for plotting."""
        if unit == "um":
            return wavelength_um
        elif unit == "nm":
            return wavelength_um * 1000.0
        elif unit == "frequency":  # um to Hz
            return SPEED_OF_LIGHT / (wavelength_um * 1e-6)
        elif unit == "energy":  # um to eV
            return (PLANCK_EV * SPEED_OF_LIGHT) / (wavelength_um * 1e-6)
        elif unit == "wavenumber":  # um to cm⁻¹
            return 1e4 / wavelength_um
        elif unit == "relative_wavenumber":
            if self.stack.reference_wl_um is None:
                raise ValueError("reference_wl_um must be set for relative_wavenumber")
            return self.stack.reference_wl_um / wavelength_um
        else:
            raise ValueError(f"Unknown wavelength unit: {unit}")

    def _convert_angle_to_radians(
        self, angles: float | Array, unit: AngleUnit
    ) -> Array:
        """Convert angles to radians."""
        angles = be.atleast_1d(angles)
        if unit == "rad":
            return angles
        elif unit == "deg":
            return be.deg2rad(angles)
        else:
            raise ValueError(f"Unknown angle unit: {unit}")

    def wavelength_view(
        self,
        wavelength_values: Array,
        wavelength_unit: WavelengthUnit = "um",
        aoi: float = 0.0,
        aoi_unit: AngleUnit = "deg",
        polarization: Pol = "u",
        to_plot: PlotType | list[PlotType] = "R",
        ax: plt.Axes = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot R/T/A vs wavelength (or equivalent units).

        Args:
            wavelength_values: Wavelength values in the specified unit
            wavelength_unit: Unit of wavelength values ('um', 'nm', 'frequency',
            'energy', 'wavenumber', 'relative_wavenumber')
            aoi: Angle of incidence (scalar)
            aoi_unit: Unit of the angle ('deg' or 'rad')
            polarization: Polarization type
            to_plot: Quantity(ies) to plot
            ax: Optional matplotlib Axes

        Returns:
            Tuple of (figure, axes)
        """
        # Convert inputs
        wl_um = self._convert_to_wavelength_um(wavelength_values, wavelength_unit)
        aoi_rad = float(self._convert_angle_to_radians(aoi, aoi_unit))

        if isinstance(to_plot, str):
            to_plot = [to_plot]

        # Compute R/T/A
        rta_data = self.stack.compute_rtRTA(wl_um, aoi_rad, polarization)

        # Convert wavelength back for plotting x-axis
        x_values = self._convert_wavelength_for_plotting(wl_um, wavelength_unit)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        for quantity in to_plot:
            if quantity not in ("R", "T", "A"):
                raise ValueError("to_plot must be 'R', 'T', 'A' or a list of these")
            ax.plot(
                x_values,
                rta_data[quantity].flatten(),
                label=f"{quantity}, {polarization}-pol, AOI={aoi}{aoi_unit}",
            )

        ax.set_xlabel(self._get_wavelength_axis_label(wavelength_unit))
        ax.set_ylabel("Power fraction")
        ax.set_xlim(float(x_values.min()), float(x_values.max()))
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()

        return fig, ax

    def angular_view(
        self,
        aoi_values: Array,
        aoi_unit: AngleUnit = "deg",
        wavelength: float = 0.55,
        wavelength_unit: WavelengthUnit = "um",
        polarization: Pol = "u",
        to_plot: PlotType | list[PlotType] = "R",
        ax: plt.Axes = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot R/T/A vs angle of incidence.

        Args:
            aoi_values: Angle of incidence values in the specified unit
            aoi_unit: Unit of the angle values ('deg' or 'rad')
            wavelength: Wavelength value (scalar)
            wavelength_unit: Unit of the wavelength ('um', 'nm', 'frequency',
            'energy', 'wavenumber', 'relative_wavenumber')
            polarization: Polarization type
            to_plot: Quantity(ies) to plot
            ax: Optional matplotlib Axes

        Returns:
            Tuple of (figure, axes)
        """
        # Convert inputs
        aoi_rad = self._convert_angle_to_radians(aoi_values, aoi_unit)
        wl_um = float(self._convert_to_wavelength_um(wavelength, wavelength_unit))

        if isinstance(to_plot, str):
            to_plot = [to_plot]

        # Compute R/T/A
        rta_data = self.stack.compute_rtRTA(wl_um, aoi_rad, polarization)

        # Convert angles back for plotting x-axis
        x_values = be.atleast_1d(aoi_values)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        for quantity in to_plot:
            if quantity not in ("R", "T", "A"):
                raise ValueError("to_plot must be 'R', 'T', 'A' or a list of these")
            ax.plot(
                x_values,
                rta_data[quantity].flatten(),
                label=f"{quantity}, {polarization}-pol, "
                + f"{
                    self._get_wavelength_axis_label(wavelength_unit)
                    .split('(')[0]
                    .strip()
                }={wavelength}{wavelength_unit}",
            )

        xlabel = r"AOI (°)" if aoi_unit == "deg" else r"AOI (rad)"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Power fraction")
        ax.set_xlim(float(x_values.min()), float(x_values.max()))
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()

        return fig, ax

    def map_view(
        self,
        wavelength_values: float | Array,
        wavelength_unit: WavelengthUnit = "um",
        aoi_values: float | Array = None,
        aoi_unit: AngleUnit = "deg",
        polarization: Pol = "u",
        to_plot: PlotType | list[PlotType] = "R",
        fig: plt.Figure = None,
        axs: plt.Axes | list[plt.Axes] = None,
    ) -> tuple[plt.Figure, plt.Axes | list[plt.Axes]]:
        """Plot 2D maps of R/T/A vs wavelength and angle of incidence.

        Args:
            wavelength_values: Wavelength values in the specified unit
            wavelength_unit: Unit of wavelength values ('um', 'nm', 'frequency',
            'energy', 'wavenumber', 'relative_wavenumber')
            aoi_values: Angle of incidence values in the specified unit
            aoi_unit: Unit of the angle values ('deg' or 'rad')
            polarization: Polarization type
            to_plot: Quantity(ies) to plot
            fig: Optional matplotlib Figure
            axs: Optional matplotlib Axes (single or list)

        Returns:
            Tuple of (figure, axes or list of axes)
        """
        # Default AOI range if not provided
        if aoi_values is None:
            aoi_values = (
                be.linspace(0, 80, 81)
                if aoi_unit == "deg"
                else be.linspace(0, be.deg2rad(80), 81)
            )

        # Convert inputs
        wl_um = self._convert_to_wavelength_um(wavelength_values, wavelength_unit)
        aoi_rad = self._convert_angle_to_radians(aoi_values, aoi_unit)

        if isinstance(to_plot, str):
            to_plot = [to_plot]

        # Compute R/T/A on 2D grid
        rta_data = self.stack.compute_rtRTA(wl_um, aoi_rad, polarization)

        # Convert back for plotting axes
        wl_plot = self._convert_wavelength_for_plotting(wl_um, wavelength_unit)
        aoi_plot = be.atleast_1d(aoi_values)

        # Create meshgrid for plotting
        WL, AOI = be.meshgrid(wl_plot, aoi_plot, indexing="ij")

        # Create figure and axes
        if fig is None or axs is None:
            fig, axs = plt.subplots(len(to_plot), 1, figsize=(8, 4 * len(to_plot)))
            if len(to_plot) == 1:
                axs = [axs]
        else:
            if len(to_plot) == 1 and not isinstance(axs, list):
                axs = [axs]

        # Plot each quantity
        for ax_idx, quantity in enumerate(to_plot):
            if quantity not in ("R", "T", "A"):
                raise ValueError("to_plot must be 'R', 'T', 'A' or a list of these")

            ax_i = axs[ax_idx]
            im = ax_i.pcolormesh(
                WL, AOI, rta_data[quantity], shading="auto", vmin=0, vmax=1
            )

            ax_i.set_xlabel(self._get_wavelength_axis_label(wavelength_unit))
            ylabel = r"AOI (°)" if aoi_unit == "deg" else r"AOI (rad)"
            ax_i.set_ylabel(ylabel)
            ax_i.set_title(f"{quantity}, {polarization}-pol")

            # Add colorbar
            cbar = fig.colorbar(im, ax=ax_i, label="Power fraction")
            cbar.set_label("Power fraction")

        fig.tight_layout()

        # Return single axes if only one plot, otherwise return list
        return fig, axs[0] if len(to_plot) == 1 else axs
