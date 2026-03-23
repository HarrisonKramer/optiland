"""Thin Film Optimizer Module

This module contains the ThinFilmOptimizer class, which provides a high-level
interface for optimizing thin film stacks. It creates its own optimization
framework specifically designed for thin film applications.

Corentin Nannini, 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.optimize import minimize

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from .operand import (
    OptimizationTarget,
    SpectralOptimizationOperand,
    ThinFilmCustomOperand,
    ThinFilmOperandManager,
    ThinFilmOperandPlotter,
    thin_film_operand_registry,
)
from .operand.core import ThinFilmEvaluationContext
from .variable.layer_thickness import LayerThicknessVariable

if TYPE_CHECKING:
    from optiland.thin_film import ThinFilmStack

# Type aliases
OpticalProperty = Literal["R", "T", "A"]
TargetType = Literal["equal", "below", "over"]
OptimizerMethod = Literal["L-BFGS-B", "TNC", "SLSQP"]


@dataclass
class VariableInfo:
    """Information about an optimization variable."""

    variable: LayerThicknessVariable
    min_val: float | None
    max_val: float | None
    layer_index: int


@dataclass
class StackSnapshot:
    """Minimal stack snapshot kept for reset and future tolerance workflows."""

    thicknesses_um: list[float]

    @classmethod
    def capture(cls, stack: ThinFilmStack) -> StackSnapshot:
        return cls([layer.thickness_um for layer in stack.layers])

    def restore(self, stack: ThinFilmStack) -> None:
        for layer, thickness_um in zip(stack.layers, self.thicknesses_um, strict=False):
            layer.update_thickness(thickness_um)


class ThinFilmOptimizer:
    """High-level interface for optimizing thin film stacks.

    This class provides a fluent API for setting up and running optimizations
    on thin film stacks. It handles the conversion between different units
    and provides its own optimization framework.
    """

    def __init__(self, stack: ThinFilmStack):
        """Initialize the optimizer.

        Args:
            stack: The thin film stack to optimize.
        """
        self.stack = stack
        self.variables: list[VariableInfo] = []
        self.operands = ThinFilmOperandManager()
        self.targets = self.operands.operands
        self.result = None

        # Store initial state for reporting
        self._initial_thicknesses = [layer.thickness_um for layer in stack.layers]
        self._initial_snapshot = StackSnapshot.capture(stack)

    def __repr__(self) -> str:
        """String representation of the optimizer."""
        return (
            f"<ThinFilmOptimizer: {len(self.stack.layers)} layers, "
            f"{len(self.variables)} variables, {len(self.targets)} targets>"
        )

    @staticmethod
    def register_operand(name: str, func, overwrite: bool = False) -> None:
        """Register a custom operand metric function."""
        thin_film_operand_registry.register(name, func, overwrite=overwrite)

    def add_variable(
        self,
        layer_index: int,
        min_nm: float | None = None,
        max_nm: float | None = None,
        apply_scaling: bool = True,
    ) -> ThinFilmOptimizer:
        """Add a layer thickness as an optimization variable.

        Args:
            layer_index: Index of the layer to vary (0-based).
            min_nm: Minimum thickness in nanometers. Defaults to None (no bound).
            max_nm: Maximum thickness in nanometers. Defaults to None (no bound).
            apply_scaling: Whether to apply scaling for optimization. Defaults to True.

        Returns:
            self for method chaining.
        """
        if layer_index < 0 or layer_index >= len(self.stack.layers):
            raise ValueError(f"layer_index {layer_index} is out of range")

        # Create the variable
        variable = LayerThicknessVariable(
            stack=self.stack, layer_index=layer_index, apply_scaling=apply_scaling
        )

        # Set bounds if provided (convert nm to μm for internal use)
        # Ensure minimum thickness is always positive (at least 0.01 nm = 0.000001 μm)
        min_val = min_nm / 1000.0 if min_nm is not None else None
        if min_val is not None and min_val <= 0:
            min_val = 0.000001  # Force minimum to 0.01 nm

        max_val = max_nm / 1000.0 if max_nm is not None else None
        if max_val is not None and max_val <= 0:
            max_val = 1.0  # Force reasonable maximum if negative

        # Ensure max > min if both are specified
        if min_val is not None and max_val is not None and max_val <= min_val:
            max_val = min_val + 0.1  # Add 100 nm minimum range

        # Apply scaling to bounds if needed
        if apply_scaling and min_val is not None:
            min_val = variable.scale(min_val)
        if apply_scaling and max_val is not None:
            max_val = variable.scale(max_val)

        # Store variable info
        var_info = VariableInfo(
            variable=variable, min_val=min_val, max_val=max_val, layer_index=layer_index
        )
        self.variables.append(var_info)

        return self

    def add_operand(
        self,
        property: str | None = None,
        wavelength_nm: float | list[float] | None = None,
        target_type: TargetType | None = None,
        value: float | list[float] | None = None,
        weight: float = 1.0,
        aoi_deg: float | list[float] = 0.0,
        polarization: str = "u",
        tolerance: float = 1e-6,
        target: float | None = None,
        min_val: float | None = None,
        max_val: float | None = None,
        input_data: dict[str, Any] | None = None,
        label: str | None = None,
        operand_type: str | None = None,
    ) -> ThinFilmOptimizer:
        """Add an optimization operand.

        Args:
            property: Operand key. Built-ins are 'R', 'T', and 'A'. Any
                registered operand key is also accepted.
            operand_type: Alias of property. Useful for explicit custom calls.
            wavelength_nm: Wavelength(s) in nanometers for spectral R/T/A
                operands. Can be scalar or array.
            target_type: Type of target ('equal', 'below', 'over') for spectral
                R/T/A operands.
            value: Target value(s) for spectral R/T/A operands. Can be scalar
                or array for interpolation.
            weight: Weight for this operand. Defaults to 1.0.
            aoi_deg: Angle(s) of incidence in degrees for spectral R/T/A
                operands. Can be scalar or array. Defaults to 0.0.
            polarization: Polarization state ('s', 'p', 'u') for spectral R/T/A
                operands. Defaults to 'u'.
            tolerance: Tolerance for 'equal' spectral targets. Defaults to 1e-6.
            target: Equality target for custom registered operands.
            min_val: Lower inequality bound for custom registered operands.
            max_val: Upper inequality bound for custom registered operands.
            input_data: Input dictionary for custom registered operands.
            label: Optional display label for custom registered operands.

        Returns:
            self for method chaining.

        Raises:
            ValueError: If inputs are invalid for the chosen operand mode.

        Examples:
            Built-in spectral operand at a single wavelength and angle:

            >>> optimizer.add_operand(
            ...     property="R",
            ...     wavelength_nm=550.0,
            ...     target_type="below",
            ...     value=0.05,
            ...     aoi_deg=0.0,
            ...     polarization="s",
            ... )

            Built-in spectral operand over a wavelength range:

            >>> optimizer.add_operand(
            ...     property="T",
            ...     wavelength_nm=[450.0, 550.0, 650.0],
            ...     target_type="equal",
            ...     value=[0.90, 0.95, 0.90],
            ...     aoi_deg=0.0,
            ...     polarization="u",
            ... )

            Built-in angular operand at fixed wavelength:

            >>> optimizer.add_operand(
            ...     property="R",
            ...     wavelength_nm=550.0,
            ...     target_type="over",
            ...     value=[0.95, 0.90, 0.85],
            ...     aoi_deg=[0.0, 30.0, 60.0],
            ...     polarization="p",
            ... )
        """
        operand_name = property if property is not None else operand_type
        if (
            property is not None
            and operand_type is not None
            and property != operand_type
        ):
            raise ValueError("property and operand_type must match when both set")
        if operand_name is None:
            raise ValueError("property or operand_type must be provided")

        is_builtin_spectral = operand_name in ["R", "T", "A"]

        if is_builtin_spectral:
            if wavelength_nm is None:
                raise ValueError("wavelength_nm is required for R/T/A operands")
            if target_type is None:
                raise ValueError("target_type is required for R/T/A operands")
            if value is None:
                raise ValueError("value is required for R/T/A operands")
            if target_type not in ["equal", "below", "over"]:
                raise ValueError(
                    f"Invalid target_type '{target_type}'. Must be "
                    "'equal', 'below', 'over'"
                )

            # Check that wavelength_nm and aoi_deg are not both arrays
            is_wl_array = isinstance(wavelength_nm, list | np.ndarray)
            is_aoi_array = isinstance(aoi_deg, list | np.ndarray)

            if is_wl_array and is_aoi_array:
                raise ValueError(
                    "Cannot specify both wavelength_nm and aoi_deg as arrays "
                    "simultaneously. Use one as array and the other as scalar."
                )

            # Validate value array dimensions
            is_value_array = isinstance(value, list | np.ndarray)
            if is_value_array:
                if is_wl_array and len(value) != len(wavelength_nm):
                    raise ValueError(
                        f"Length of value array ({len(value)}) must match "
                        f"length of wavelength_nm array ({len(wavelength_nm)})"
                    )
                if is_aoi_array and len(value) != len(aoi_deg):
                    raise ValueError(
                        f"Length of value array ({len(value)}) must match "
                        f"length of aoi_deg array ({len(aoi_deg)})"
                    )

            operand = SpectralOptimizationOperand(
                property=operand_name,
                wavelength_nm=wavelength_nm,
                target_type=target_type,
                value=value,
                weight=weight,
                aoi_deg=aoi_deg,
                polarization=polarization,
                tolerance=tolerance,
            )
            self.operands.add(operand)
            return self

        if operand_name not in thin_film_operand_registry:
            raise ValueError(
                f"Invalid property '{operand_name}'. Must be 'R', 'T', 'A' "
                "or a registered operand name."
            )

        if target_type is not None:
            raise ValueError("target_type is only valid for built-in R/T/A operands")
        if wavelength_nm is not None:
            raise ValueError("wavelength_nm is only valid for built-in R/T/A operands")
        if value is not None:
            raise ValueError("value is only valid for built-in R/T/A operands")
        if target is not None and (min_val is not None or max_val is not None):
            raise ValueError(
                "Custom operand cannot mix equality and inequality targets"
            )

        operand = ThinFilmCustomOperand(
            operand_type=operand_name,
            target=target,
            min_val=min_val,
            max_val=max_val,
            weight=weight,
            input_data=input_data,
            label=label,
        )
        self.operands.add(operand)
        return self

    def add_angular_operand(
        self,
        property: str,
        wavelength_nm: float,
        aoi_deg_range: list[float],
        target_type: str,
        value: float | list[float],
        weight: float = 1.0,
        polarization: str = "s",
    ) -> ThinFilmOptimizer:
        """Convenience method to add an angular operand with multiple AOI values.

        Args:
            property: Property to optimize ("R", "T", "A").
            wavelength_nm: Single wavelength value in nm.
            aoi_deg_range: List of angles of incidence in degrees.
            target_type: Type of target ("equal", "over", "below").
            value: Target value(s). Single value or list matching
                aoi_deg_range length.
            weight: Operand weight for optimization. Defaults to 1.0.
            polarization: Polarization state ("s", "p", "u"). Defaults to "s".

        Returns:
            ThinFilmOptimizer: self for method chaining.

        Example:
            >>> optimizer.add_angular_operand(
            ...     property="R",
            ...     wavelength_nm=550.0,
            ...     aoi_deg_range=[0.0, 20.0, 40.0, 60.0],
            ...     target_type="below",
            ...     value=[0.08, 0.10, 0.14, 0.20],
            ...     polarization="s",
            ... )
        """
        return self.add_operand(
            property=property,
            wavelength_nm=wavelength_nm,
            target_type=target_type,
            value=value,
            weight=weight,
            aoi_deg=aoi_deg_range,
            polarization=polarization,
        )

    def add_interpolated_operand(
        self,
        property: str,
        wavelength_nm: list[float],
        target_type: str,
        value: list[float],
        weight: float = 1.0,
        aoi_deg: float = 0.0,
        polarization: str = "s",
    ) -> ThinFilmOptimizer:
        """Convenience method to add an interpolated spectral operand.

        Args:
            property: Property to optimize ("R", "T", "A").
            wavelength_nm: List of wavelength values in nm.
            target_type: Type of target ("equal", "over", "below").
            value: List of target values matching wavelength_nm length.
            weight: Operand weight for optimization. Defaults to 1.0.
            aoi_deg: Angle of incidence in degrees. Defaults to 0.0.
            polarization: Polarization state ("s", "p", "u"). Defaults to "s".

        Returns:
            ThinFilmOptimizer: self for method chaining.

        Example:
            >>> optimizer.add_interpolated_operand(
            ...     property="T",
            ...     wavelength_nm=[450.0, 550.0, 650.0],
            ...     target_type="equal",
            ...     value=[0.88, 0.94, 0.89],
            ...     aoi_deg=0.0,
            ...     polarization="u",
            ... )
        """
        return self.add_operand(
            property=property,
            wavelength_nm=wavelength_nm,
            target_type=target_type,
            value=value,
            weight=weight,
            aoi_deg=aoi_deg,
            polarization=polarization,
        )

    def _interpolate_target_value(
        self,
        target: OptimizationTarget,
        current_wl: float | None = None,
        current_aoi: float | None = None,
    ) -> float:
        """Interpolate target value based on current wavelength or AOI.

        Args:
            target: The optimization target.
            current_wl: Current wavelength for interpolation (when aoi_deg is array).
            current_aoi: Current AOI for interpolation (when wavelength_nm is array).

        Returns:
            Interpolated target value.
        """
        return target.interpolate_target_value(
            current_wl=current_wl,
            current_aoi=current_aoi,
        )

    def _evaluation_context(self) -> ThinFilmEvaluationContext:
        return ThinFilmEvaluationContext(stack=self.stack)

    def _merit_function(self, x: np.ndarray) -> float:
        """Evaluate the merit function.

        Args:
            x: Array of variable values in optimization space.

        Returns:
            Merit function value (sum of weighted squared residuals).
        """
        # Update variables
        for i, var_info in enumerate(self.variables):
            var_info.variable.update_value(x[i])
        return self.sum_squared()

    def fun_array(self) -> np.ndarray:
        """Array of operand weighted deltas for the current stack state."""
        context = self._evaluation_context()
        terms = [operand.fun(context) for operand in self.operands]
        if not terms:
            return np.array([0.0])
        return np.asarray(terms, dtype=float)

    def sum_squared(self) -> float:
        """Calculate the sum of squared operand deltas."""
        values = self.fun_array()
        return float(np.sum(values**2))

    def rss(self) -> float:
        """Root sum of squares of the current merit function."""
        return float(np.sqrt(self.sum_squared()))

    def optimize(
        self,
        method: OptimizerMethod = "L-BFGS-B",
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = False,
        **kwargs,
    ) -> dict:
        """Run the optimization.

        Args:
            method: Optimization method to use. Defaults to "L-BFGS-B". See
            scipy.optimize.minimize for options.
            max_iterations: Maximum number of iterations. Defaults to 100.
            tolerance: Convergence tolerance. Defaults to 1e-6.
            verbose: Whether to print optimization progress. Defaults to False.
            **kwargs: Additional keyword arguments for the optimizer.

        Returns:
            dict: Optimization results including success status, final merit,
                  iterations, and thickness changes.

        Raises:
            ValueError: If no variables or targets are defined.
        """
        if not self.variables:
            raise ValueError("No variables defined. Use add_variable() first.")
        if not self.targets:
            raise ValueError("No operands defined. Use add_operand() first.")

        # Get initial values and bounds
        x0 = np.array([var.variable.get_value() for var in self.variables])
        bounds = [(var.min_val, var.max_val) for var in self.variables]

        # Store initial merit
        initial_merit = self._merit_function(x0)

        # Run optimization
        # Note: 'disp' and 'iprint' are deprecated in scipy 1.18+
        # Only include them in kwargs if explicitly provided by user
        options = {
            "maxiter": max_iterations,
            "ftol": tolerance,
        }

        # Add user-provided kwargs, but avoid deprecated scipy options
        for key, value in kwargs.items():
            if key not in ("disp", "iprint"):
                options[key] = value

        result = minimize(
            self._merit_function, x0, method=method, bounds=bounds, options=options
        )

        # Store result
        self.result = result

        # Compute final statistics
        final_merit = result.fun
        thickness_changes = {}

        for _i, var_info in enumerate(self.variables):
            initial_thickness = self._initial_thicknesses[var_info.layer_index]
            final_thickness = self.stack.layers[var_info.layer_index].thickness_um
            thickness_changes[var_info.layer_index] = {
                "initial_nm": initial_thickness * 1000,
                "final_nm": final_thickness * 1000,
                "change_nm": (final_thickness - initial_thickness) * 1000,
                "change_percent": (
                    (final_thickness - initial_thickness) / initial_thickness
                )
                * 100,
            }

        return {
            "success": result.success,
            "message": result.message,
            "initial_merit": initial_merit,
            "final_merit": final_merit,
            "improvement": initial_merit - final_merit,
            "iterations": result.nit,
            "function_evaluations": result.nfev,
            "thickness_changes": thickness_changes,
            "optimization_result": result,
        }

    def reset(self) -> ThinFilmOptimizer:
        """Reset the stack to its initial state.

        Returns:
            self for method chaining.
        """
        for i, initial_thickness in enumerate(self._initial_thicknesses):
            self.stack.layers[i].thickness_um = initial_thickness

        return self

    def get_current_performance(self) -> dict[str, Any]:
        """Get current performance metrics for all targets.

        Returns:
            dict: Current values for all targets.
        """
        performance = {}

        context = self._evaluation_context()
        for i, operand in enumerate(self.operands):
            performance[f"target_{i}"] = operand.performance_data(context)

        return performance

    def add_spectral_operand(
        self,
        property: OpticalProperty,
        wavelengths_nm: list[float],
        target_type: TargetType,
        value: float,
        weight: float = 1.0,
        weights: list[float] | None = None,  # For weighted spectral operands
        aoi_deg: float = 0.0,
        polarization: str = "u",
        tolerance: float = 1e-6,
    ) -> ThinFilmOptimizer:
        """Add a spectral optimization operand (convenience method).

        Args:
            property: Optical property to target ('R', 'T', or 'A').
            wavelengths_nm: List of wavelengths in nanometers.
            target_type: Type of target ('equal', 'below', 'over').
            value: Target value.
            weight: Weight for this operand. Defaults to 1.0.
            weights: Per-wavelength weights (alternative to weight). Defaults to None.
            aoi_deg: Angle of incidence in degrees. Defaults to 0.0.
            polarization: Polarization state ('s', 'p', 'u'). Defaults to 'u'.
            tolerance: Tolerance for 'equal' targets. Defaults to 1e-6.

        Returns:
            self for method chaining.
        """
        # Use per-wavelength weights if provided
        final_weight = weight
        if weights is not None:
            # For now, just use the average weight
            # In a more sophisticated implementation, we could create separate operands
            final_weight = sum(weights) / len(weights)

        return self.add_operand(
            property=property,
            wavelength_nm=wavelengths_nm,
            target_type=target_type,
            value=value,
            weight=final_weight,
            aoi_deg=aoi_deg,
            polarization=polarization,
            tolerance=tolerance,
        )

    def plot_operands(
        self,
        ax,
        plot_type: Literal["wavelength", "angle"] = "wavelength",
        wavelength_range_nm: tuple[float, float] | None = None,
        angle_range_deg: tuple[float, float] | None = None,
        num_points: int = 100,
        fixed_wavelength_nm: float = 550.0,
        fixed_angle_deg: float = 0.0,
    ) -> None:
        """Plot optimization operands on the provided axes (lightweight version).

        Args:
            ax: Matplotlib axes to plot on.
            plot_type: Type of plot - "wavelength" or "angle".
            wavelength_range_nm: Wavelength range for plotting (min, max) in nm.
            angle_range_deg: Angle range for plotting (min, max) in degrees.
            num_points: Number of points for plotting smooth curves.
            fixed_wavelength_nm: Fixed wavelength when plotting vs angle.
            fixed_angle_deg: Fixed angle when plotting vs wavelength.
        """
        if plt is None:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install matplotlib"
            )

        plotter = ThinFilmOperandPlotter(self.operands)
        plotter.plot(
            ax,
            plot_type=plot_type,
            wavelength_range_nm=wavelength_range_nm,
            angle_range_deg=angle_range_deg,
            num_points=num_points,
        )

    def info(self) -> None:
        """Display information about the optimizer state in tabular format."""
        try:
            from tabulate import tabulate
        except ImportError:
            # Fallback to manual formatting if tabulate not available
            tabulate = None

        print("ThinFilm Optimizer Information")
        print("=" * 50)

        # Summary table
        summary_data = [
            ["Stack layers", len(self.stack.layers)],
            ["Variables", len(self.variables)],
            ["Targets", len(self.targets)],
        ]

        if tabulate:
            print(
                tabulate(summary_data, headers=["Property", "Count"], tablefmt="grid")
            )
        else:
            print(f"{'Property':<15} {'Count':<10}")
            print("-" * 25)
            for prop, count in summary_data:
                print(f"{prop:<15} {count:<10}")
        print()

        # Variables table
        if self.variables:
            print("Variables:")
            var_data = []
            for i, var_info in enumerate(self.variables):
                layer = self.stack.layers[var_info.layer_index]
                thickness_nm = layer.thickness_um * 1000

                # Handle bounds correctly with scaling
                if var_info.min_val is not None:
                    # If scaling was applied, inverse scale to get real μm value,
                    # then convert to nm
                    if var_info.variable.apply_scaling:
                        min_um = var_info.variable.inverse_scale(var_info.min_val)
                        min_bound = f"{min_um * 1000:.1f}"
                    else:
                        min_bound = f"{var_info.min_val * 1000:.1f}"
                else:
                    min_bound = "None"

                if var_info.max_val is not None:
                    # If scaling was applied, inverse scale to get real μm value,
                    # then convert to nm
                    if var_info.variable.apply_scaling:
                        max_um = var_info.variable.inverse_scale(var_info.max_val)
                        max_bound = f"{max_um * 1000:.1f}"
                    else:
                        max_bound = f"{var_info.max_val * 1000:.1f}"
                else:
                    max_bound = "None"

                var_data.append(
                    [
                        i,
                        var_info.layer_index,
                        f"{thickness_nm:.1f}",
                        min_bound,
                        max_bound,
                    ]
                )

            headers = ["ID", "Layer", "Thickness (nm)", "Min (nm)", "Max (nm)"]
            if tabulate:
                print(tabulate(var_data, headers=headers, tablefmt="grid"))
            else:
                print(
                    f"{'ID':<4} {'Layer':<6} {'Thickness (nm)':<15} "
                    f"{'Min (nm)':<10} {'Max (nm)':<10}"
                )
                print("-" * 60)
                for row in var_data:
                    print(
                        f"{row[0]:<4} {row[1]:<6} {row[2]:<15} "
                        f"{row[3]:<10} {row[4]:<10}"
                    )
            print()

        # Targets table
        if self.targets:
            print("Targets:")
            target_data = []
            for i, target in enumerate(self.targets):
                if not isinstance(target, SpectralOptimizationOperand):
                    target_data.append(
                        [
                            i,
                            getattr(
                                target,
                                "display_name",
                                getattr(target, "operand_type", "custom"),
                            ),
                            "custom",
                            getattr(target, "target", ""),
                            "-",
                            "-",
                            f"{target.weight:.1f}",
                            "-",
                        ]
                    )
                    continue

                # Handle different parameter types for display
                if isinstance(target.wavelength_nm, list | np.ndarray):
                    wl_str = (
                        f"{len(target.wavelength_nm)} λ "
                        f"({min(target.wavelength_nm):.0f}-{max(target.wavelength_nm):.0f})"
                    )
                else:
                    wl_str = f"{target.wavelength_nm:.0f}"

                if isinstance(target.aoi_deg, list | np.ndarray):
                    aoi_str = (
                        f"{len(target.aoi_deg)} θ "
                        f"({min(target.aoi_deg):.0f}-{max(target.aoi_deg):.0f}°)"
                    )
                else:
                    aoi_str = f"{target.aoi_deg:.1f}°"

                if isinstance(target.value, list | np.ndarray):
                    value_str = (
                        f"interp ({min(target.value):.3f}-{max(target.value):.3f})"
                    )
                else:
                    value_str = f"{float(target.value):.3f}"

                target_data.append(
                    [
                        i,
                        target.property,
                        target.target_type,
                        value_str,
                        wl_str,
                        aoi_str,
                        f"{target.weight:.1f}",
                        target.polarization,
                    ]
                )

            headers = [
                "ID",
                "Prop",
                "Type",
                "Value",
                "Wavelength",
                "AOI",
                "Weight",
                "Pol",
            ]
            if tabulate:
                print(tabulate(target_data, headers=headers, tablefmt="grid"))
            else:
                print(
                    f"{'ID':<3} {'Prop':<4} {'Type':<6} {'Value':<18} "
                    f"{'Wavelength':<12} {'AOI':<12} {'Weight':<7} {'Pol':<3}"
                )
                print("-" * 80)
                for row in target_data:
                    print(
                        f"{row[0]:<3} {row[1]:<4} {row[2]:<6} {row[3]:<18} "
                        f"{row[4]:<12} {row[5]:<12} {row[6]:<7} {row[7]:<3}"
                    )
            print()

        # Optimization results table
        if self.result:
            print("Last Optimization Result:")
            result_data = [
                ["Success", "Yes" if self.result.success else "No"],
                ["Merit Function", f"{self.result.fun:.6f}"],
                ["Iterations", f"{self.result.nit}"],
                ["Method", getattr(self.result, "method", "N/A")],
            ]

            if tabulate:
                print(
                    tabulate(result_data, headers=["Metric", "Value"], tablefmt="grid")
                )
            else:
                print(f"{'Metric':<15} {'Value':<15}")
                print("-" * 30)
                for metric, value in result_data:
                    print(f"{metric:<15} {value:<15}")
            print()
