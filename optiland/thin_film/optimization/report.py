"""Thin Film Optimization Report Module

This module contains classes for generating detailed reports of thin film
optimization results, including before/after comparisons and performance
analysis.

Corentin Nannini, 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import pandas as pd

import optiland.backend as be

if TYPE_CHECKING:
    from .optimizer import ThinFilmOptimizer


@dataclass
class OptimizationResult:
    """Enhanced optimization result with reporting capabilities."""

    original_result: Any
    report: ThinFilmReport
    optimizer: ThinFilmOptimizer

    def __getattr__(self, name):
        """Delegate attribute access to original result."""
        return getattr(self.original_result, name)


class ThinFilmReport:
    """Generates detailed reports for thin film optimization results.

    This class provides methods to analyze and visualize the results of
    thin film optimization, including before/after comparisons and
    performance metrics.

    Args:
        optimizer: The ThinFilmOptimizer that was used.
        result: The optimization result object.
    """

    def __init__(self, optimizer: ThinFilmOptimizer, result: Any):
        self.optimizer = optimizer
        self.result = result
        self.stack = optimizer.stack

        # Store optimization data
        self._initial_thicknesses = optimizer._initial_thicknesses.copy()
        self._final_thicknesses = [layer.thickness_um for layer in self.stack.layers]

    def summary_table(self) -> pd.DataFrame:
        """Generate a summary table of optimization variables.

        Returns:
            DataFrame with columns: Variable, Initial, Final, Change, Unit
        """
        data = []

        for _i, var in enumerate(self.optimizer.variables):
            layer_idx = var.layer_index
            initial_nm = self._initial_thicknesses[layer_idx] * 1000
            final_nm = self._final_thicknesses[layer_idx] * 1000
            change_nm = final_nm - initial_nm
            change_pct = (change_nm / initial_nm) * 100 if initial_nm != 0 else 0

            data.append(
                {
                    "Variable": f"Layer {layer_idx} thickness",
                    "Initial": f"{initial_nm:.1f}",
                    "Final": f"{final_nm:.1f}",
                    "Change": f"{change_nm:+.1f} ({change_pct:+.1f}%)",
                    "Unit": "nm",
                }
            )

        return pd.DataFrame(data)

    def performance_table(self) -> pd.DataFrame:
        """Generate a performance table showing target achievement.

        Returns:
            DataFrame with columns: Target, Initial, Final, Target_Value,
            Target_Type, Weight, Achieved
        """
        data = []

        # Reset to initial state to compute initial values
        for _i, thickness_um in enumerate(self._initial_thicknesses):
            if _i < len(self.stack.layers):
                self.stack.layers[_i].update_thickness(thickness_um)

        for _i, target in enumerate(self.optimizer.targets):
            # Get target info
            prop = target.property
            wl_nm = target.wavelength_nm
            aoi_deg = target.aoi_deg
            polarization = target.polarization
            target_value = target.value
            target_type = target.target_type
            weight = target.weight

            # Compute initial value
            if prop == "R":
                initial_val = float(
                    self.stack.reflectance_nm_deg(wl_nm, aoi_deg, polarization)
                )
            elif prop == "T":
                initial_val = float(
                    self.stack.transmittance_nm_deg(wl_nm, aoi_deg, polarization)
                )
            elif prop == "A":
                initial_val = float(
                    self.stack.absorptance_nm_deg(wl_nm, aoi_deg, polarization)
                )

            # Restore final state
            for j, thickness_um in enumerate(self._final_thicknesses):
                if j < len(self.stack.layers):
                    self.stack.layers[j].update_thickness(thickness_um)

            # Compute final value
            if prop == "R":
                final_val = float(
                    self.stack.reflectance_nm_deg(wl_nm, aoi_deg, polarization)
                )
            elif prop == "T":
                final_val = float(
                    self.stack.transmittance_nm_deg(wl_nm, aoi_deg, polarization)
                )
            elif prop == "A":
                final_val = float(
                    self.stack.absorptance_nm_deg(wl_nm, aoi_deg, polarization)
                )

            # Check if target achieved
            if target_type == "equal":
                achieved = abs(final_val - target_value) < target.tolerance
            elif target_type == "below":
                achieved = final_val <= target_value
            elif target_type == "over":
                achieved = final_val >= target_value
            else:
                achieved = False

            # Format wavelength string
            if isinstance(wl_nm, list):
                wl_str = f"{min(wl_nm):.0f}-{max(wl_nm):.0f} nm"
            else:
                wl_str = f"{wl_nm:.0f} nm"

            data.append(
                {
                    "Target": f"{prop} at {wl_str} ({polarization}, {aoi_deg:.1f}°)",
                    "Initial": f"{initial_val:.4f}",
                    "Final": f"{final_val:.4f}",
                    "Target_Value": f"{target_value:.4f}",
                    "Target_Type": target_type,
                    "Weight": f"{weight:.2f}",
                    "Achieved": "✓" if achieved else "✗",
                }
            )

        return pd.DataFrame(data)

    def convergence_plot(self, ax: plt.Axes = None) -> tuple[plt.Figure, plt.Axes]:
        """Plot the convergence of the optimization.

        Args:
            ax: Matplotlib axes to plot on. If None, creates new figure.

        Returns:
            Tuple of (figure, axes).
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        # Try to extract convergence data from result
        if hasattr(self.result, "fun_history"):
            # Some optimizers provide function history
            fun_values = self.result.fun_history
            iterations = range(len(fun_values))
            ax.plot(iterations, fun_values, "b-", linewidth=2)
        elif hasattr(self.result, "nit") and hasattr(self.result, "fun"):
            # Basic info: just show final value
            ax.axhline(
                y=self.result.fun,
                color="b",
                linestyle="-",
                linewidth=2,
                label=f"Final merit function: {self.result.fun:.6f}",
            )
            ax.set_xlim(0, self.result.nit)
            ax.legend()
        else:
            # No convergence data available
            ax.text(
                0.5,
                0.5,
                "Convergence data not available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Merit Function Value")
        ax.set_title("Optimization Convergence")
        ax.grid(True, alpha=0.3)

        return fig, ax

    def comparison_plot(
        self,
        wavelength_range_nm: tuple[float, float] = (400, 800),
        num_points: int = 100,
        properties: list[str] = ("R", "T"),
        polarization: str = "u",
        aoi_deg: float = 0.0,
        ax: plt.Axes = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot spectral comparison before and after optimization.

        Args:
            wavelength_range_nm: Wavelength range in nm. Defaults to (400, 800).
            num_points: Number of wavelength points. Defaults to 100.
            properties: List of properties to plot ('R', 'T', 'A').
                Defaults to ['R', 'T'].
            polarization: Polarization state. Defaults to 'u'.
            aoi_deg: Angle of incidence in degrees. Defaults to 0.0.
            ax: Matplotlib axes to plot on. If None, creates new figure.

        Returns:
            Tuple of (figure, axes).
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        # Generate wavelength array
        wl_nm = be.linspace(wavelength_range_nm[0], wavelength_range_nm[1], num_points)

        # Color and style maps
        color_map = {"R": "red", "T": "blue", "A": "green"}

        for prop in properties:
            # Compute initial spectrum
            # Reset to initial state
            for i, thickness_um in enumerate(self._initial_thicknesses):
                if i < len(self.stack.layers):
                    self.stack.layers[i].update_thickness(thickness_um)

            if prop == "R":
                initial_spectrum = self.stack.reflectance_nm_deg(
                    wl_nm, aoi_deg, polarization
                )
            elif prop == "T":
                initial_spectrum = self.stack.transmittance_nm_deg(
                    wl_nm, aoi_deg, polarization
                )
            elif prop == "A":
                initial_spectrum = self.stack.absorptance_nm_deg(
                    wl_nm, aoi_deg, polarization
                )

            # Restore final state
            for i, thickness_um in enumerate(self._final_thicknesses):
                if i < len(self.stack.layers):
                    self.stack.layers[i].update_thickness(thickness_um)

            # Compute final spectrum
            if prop == "R":
                final_spectrum = self.stack.reflectance_nm_deg(
                    wl_nm, aoi_deg, polarization
                )
            elif prop == "T":
                final_spectrum = self.stack.transmittance_nm_deg(
                    wl_nm, aoi_deg, polarization
                )
            elif prop == "A":
                final_spectrum = self.stack.absorptance_nm_deg(
                    wl_nm, aoi_deg, polarization
                )

            # Plot
            color = color_map.get(prop, "black")
            ax.plot(
                wl_nm,
                initial_spectrum.flatten(),
                "--",
                color=color,
                alpha=0.7,
                label=f"{prop} initial",
                linewidth=1,
            )
            ax.plot(
                wl_nm,
                final_spectrum.flatten(),
                "-",
                color=color,
                label=f"{prop} optimized",
                linewidth=2,
            )

        # Add target indicators
        for target in self.optimizer.targets:
            if target.property in properties:
                target_wl = target.wavelength_nm
                target_val = target.value
                color = color_map.get(target.property, "black")

                if isinstance(target_wl, int | float):
                    # Single wavelength target
                    ax.axvline(x=target_wl, color=color, linestyle=":", alpha=0.5)
                    ax.axhline(y=target_val, color=color, linestyle=":", alpha=0.5)
                    ax.plot(
                        target_wl,
                        target_val,
                        "o",
                        color=color,
                        markersize=8,
                        markerfacecolor="white",
                        markeredgewidth=2,
                    )

        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Optical Property")
        ax.set_title(
            f"Spectral Response Comparison (pol={polarization}, AOI={aoi_deg:.1f}°)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(wavelength_range_nm)
        ax.set_ylim(0, 1)

        return fig, ax

    def show_all(self) -> None:
        """Display all available reports and plots."""
        print("=== Thin Film Optimization Report ===\n")

        # Summary table
        print("Variables Summary:")
        print(self.summary_table().to_string(index=False))
        print()

        # Performance table
        print("Performance Summary:")
        print(self.performance_table().to_string(index=False))
        print()

        # Create plots
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 6))

        # Convergence plot
        self.convergence_plot(ax=ax1)

        # Comparison plot
        self.comparison_plot(ax=ax2)

        plt.tight_layout()
        plt.show()

    def export_results(self, filename_prefix: str = "thin_film_optimization") -> None:
        """Export results to files.

        Args:
            filename_prefix: Prefix for output files.
        """
        # Export tables
        self.summary_table().to_csv(f"{filename_prefix}_summary.csv", index=False)
        self.performance_table().to_csv(
            f"{filename_prefix}_performance.csv", index=False
        )

        # Export plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        self.convergence_plot(ax=axes[0])
        self.comparison_plot(ax=axes[1])
        plt.tight_layout()
        plt.savefig(f"{filename_prefix}_plots.png", dpi=300, bbox_inches="tight")
        plt.close()

        print("Results exported:")
        print(f"  - {filename_prefix}_summary.csv")
        print(f"  - {filename_prefix}_performance.csv")
        print(f"  - {filename_prefix}_plots.png")
