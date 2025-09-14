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

from .operand.thin_film import ThinFilmOperand
from .variable.layer_thickness import LayerThicknessVariable

if TYPE_CHECKING:
    from optiland.thin_film import ThinFilmStack

# Type aliases
OpticalProperty = Literal["R", "T", "A"]
TargetType = Literal["equal", "below", "over"]
OptimizerMethod = Literal["L-BFGS-B", "TNC", "SLSQP"]


@dataclass
class OptimizationTarget:
    """Represents an optimization target."""

    property: OpticalProperty
    wavelength_nm: float | list[float]
    target_type: TargetType
    value: float
    weight: float
    aoi_deg: float
    polarization: str
    tolerance: float


@dataclass
class VariableInfo:
    """Information about an optimization variable."""

    variable: LayerThicknessVariable
    min_val: float | None
    max_val: float | None
    layer_index: int


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
        self.targets: list[OptimizationTarget] = []
        self.result = None

        # Store initial state for reporting
        self._initial_thicknesses = [layer.thickness_um for layer in stack.layers]

    def add_thickness_variable(
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
        min_val = min_nm / 1000.0 if min_nm is not None else None
        max_val = max_nm / 1000.0 if max_nm is not None else None

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

    def add_target(
        self,
        property: OpticalProperty,
        wavelength_nm: float | list[float],
        target_type: TargetType,
        value: float,
        weight: float = 1.0,
        aoi_deg: float = 0.0,
        polarization: str = "u",
        tolerance: float = 1e-6,
    ) -> ThinFilmOptimizer:
        """Add an optimization target.

        Args:
            property: Optical property to target ('R', 'T', or 'A').
            wavelength_nm: Wavelength(s) in nanometers.
            target_type: Type of target ('equal', 'below', 'over').
            value: Target value.
            weight: Weight for this target. Defaults to 1.0.
            aoi_deg: Angle of incidence in degrees. Defaults to 0.0.
            polarization: Polarization state ('s', 'p', 'u'). Defaults to 'u'.
            tolerance: Tolerance for 'equal' targets. Defaults to 1e-6.

        Returns:
            self for method chaining.

        Raises:
            ValueError: If property or target_type is invalid.
        """
        # Validation
        if property not in ["R", "T", "A"]:
            raise ValueError(
                f"Invalid property '{property}'. Must be 'R', 'T', or 'A'."
            )
        if target_type not in ["equal", "below", "over"]:
            raise ValueError(
                f"Invalid target_type '{target_type}'. Must be 'equal', 'below', 'over'"
            )

        # Create target object
        target = OptimizationTarget(
            property=property,
            wavelength_nm=wavelength_nm,
            target_type=target_type,
            value=value,
            weight=weight,
            aoi_deg=aoi_deg,
            polarization=polarization,
            tolerance=tolerance,
        )

        self.targets.append(target)
        return self

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

        merit = 0.0

        # Evaluate all targets
        for target in self.targets:
            if target.property == "R":
                if isinstance(target.wavelength_nm, list):
                    # Generate uniform weights for wavelength list
                    weights = [1.0] * len(target.wavelength_nm)
                    current_value = ThinFilmOperand.reflectance_weighted(
                        self.stack,
                        target.wavelength_nm,
                        weights,
                        target.aoi_deg,
                        target.polarization,
                    )
                else:
                    current_value = ThinFilmOperand.reflectance(
                        self.stack,
                        target.wavelength_nm,
                        target.aoi_deg,
                        target.polarization,
                    )
            elif target.property == "T":
                if isinstance(target.wavelength_nm, list):
                    # Generate uniform weights for wavelength list
                    weights = [1.0] * len(target.wavelength_nm)
                    current_value = ThinFilmOperand.transmittance_weighted(
                        self.stack,
                        target.wavelength_nm,
                        weights,
                        target.aoi_deg,
                        target.polarization,
                    )
                else:
                    current_value = ThinFilmOperand.transmittance(
                        self.stack,
                        target.wavelength_nm,
                        target.aoi_deg,
                        target.polarization,
                    )
            elif target.property == "A":
                if isinstance(target.wavelength_nm, list):
                    # Generate uniform weights for wavelength list
                    weights = [1.0] * len(target.wavelength_nm)
                    current_value = ThinFilmOperand.absorptance_weighted(
                        self.stack,
                        target.wavelength_nm,
                        weights,
                        target.aoi_deg,
                        target.polarization,
                    )
                else:
                    current_value = ThinFilmOperand.absorptance(
                        self.stack,
                        target.wavelength_nm,
                        target.aoi_deg,
                        target.polarization,
                    )

            # Compute residual based on target type
            if target.target_type == "equal":
                residual = current_value - target.value
            elif target.target_type == "below":
                residual = max(0, current_value - target.value)
            elif target.target_type == "over":
                residual = max(0, target.value - current_value)

            # Add weighted squared residual to merit
            merit += target.weight * residual**2

        # Ensure we return a scalar float
        if hasattr(merit, "item"):
            return merit.item()
        return float(merit)

    def optimize(
        self,
        method: OptimizerMethod = "L-BFGS-B",
        max_iterations: int = 100,
        maxiter: int | None = None,  # For backward compatibility
        tolerance: float = 1e-6,
        verbose: bool = False,
        disp: bool | None = None,  # For backward compatibility
        generate_report: bool = False,
    ) -> dict:
        """Run the optimization.

        Args:
            method: Optimization method to use. Defaults to "L-BFGS-B".
            max_iterations: Maximum number of iterations. Defaults to 100.
            maxiter: Alternative name for max_iterations (backward compatibility).
            tolerance: Convergence tolerance. Defaults to 1e-6.
            verbose: Whether to print optimization progress. Defaults to False.
            disp: Alternative name for verbose (backward compatibility).
            generate_report: Whether to generate a report. Defaults to False.

        Returns:
            dict: Optimization results including success status, final merit,
                  iterations, and thickness changes.

        Raises:
            ValueError: If no variables or targets are defined.
        """
        if not self.variables:
            raise ValueError(
                "No variables defined. Use add_thickness_variable() first."
            )
        if not self.targets:
            raise ValueError("No targets defined. Use add_target() first.")

        # Handle backward compatibility
        if maxiter is not None:
            max_iterations = maxiter
        if disp is not None:
            verbose = disp

        # Get initial values and bounds
        x0 = np.array([var.variable.get_value() for var in self.variables])
        bounds = [(var.min_val, var.max_val) for var in self.variables]

        # Store initial merit
        initial_merit = self._merit_function(x0)

        # Run optimization
        options = {"maxiter": max_iterations, "ftol": tolerance, "disp": verbose}

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

        for i, target in enumerate(self.targets):
            if target.property == "R":
                if isinstance(target.wavelength_nm, list):
                    # Generate uniform weights for wavelength list
                    weights = [1.0] * len(target.wavelength_nm)
                    current_value = ThinFilmOperand.reflectance_weighted(
                        self.stack,
                        target.wavelength_nm,
                        weights,
                        target.aoi_deg,
                        target.polarization,
                    )
                else:
                    current_value = ThinFilmOperand.reflectance(
                        self.stack,
                        target.wavelength_nm,
                        target.aoi_deg,
                        target.polarization,
                    )
            elif target.property == "T":
                if isinstance(target.wavelength_nm, list):
                    # Generate uniform weights for wavelength list
                    weights = [1.0] * len(target.wavelength_nm)
                    current_value = ThinFilmOperand.transmittance_weighted(
                        self.stack,
                        target.wavelength_nm,
                        weights,
                        target.aoi_deg,
                        target.polarization,
                    )
                else:
                    current_value = ThinFilmOperand.transmittance(
                        self.stack,
                        target.wavelength_nm,
                        target.aoi_deg,
                        target.polarization,
                    )
            elif target.property == "A":
                if isinstance(target.wavelength_nm, list):
                    # Generate uniform weights for wavelength list
                    weights = [1.0] * len(target.wavelength_nm)
                    current_value = ThinFilmOperand.absorptance_weighted(
                        self.stack,
                        target.wavelength_nm,
                        weights,
                        target.aoi_deg,
                        target.polarization,
                    )
                else:
                    current_value = ThinFilmOperand.absorptance(
                        self.stack,
                        target.wavelength_nm,
                        target.aoi_deg,
                        target.polarization,
                    )

            performance[f"target_{i}"] = {
                "property": target.property,
                "wavelength_nm": target.wavelength_nm,
                "target_type": target.target_type,
                "target_value": target.value,
                "current_value": current_value,
                "difference": current_value - target.value,
                "weight": target.weight,
            }

        return performance

    def add_spectral_target(
        self,
        property: OpticalProperty,
        wavelengths_nm: list[float],
        target_type: TargetType,
        value: float,
        weight: float = 1.0,
        weights: list[float] | None = None,  # For weighted spectral targets
        aoi_deg: float = 0.0,
        polarization: str = "u",
        tolerance: float = 1e-6,
    ) -> ThinFilmOptimizer:
        """Add a spectral optimization target (convenience method).

        Args:
            property: Optical property to target ('R', 'T', or 'A').
            wavelengths_nm: List of wavelengths in nanometers.
            target_type: Type of target ('equal', 'below', 'over').
            value: Target value.
            weight: Weight for this target. Defaults to 1.0.
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
            # In a more sophisticated implementation, we could create separate targets
            final_weight = sum(weights) / len(weights)

        return self.add_target(
            property=property,
            wavelength_nm=wavelengths_nm,
            target_type=target_type,
            value=value,
            weight=final_weight,
            aoi_deg=aoi_deg,
            polarization=polarization,
            tolerance=tolerance,
        )

    def info(self) -> None:
        """Print information about the optimizer state."""
        print("ThinFilm Optimizer Information:")
        print(f"  Stack: {len(self.stack.layers)} layers")
        print(f"  Variables: {len(self.variables)}")
        print(f"  Targets: {len(self.targets)}")
        print()

        if self.variables:
            print("Variables:")
            for i, var_info in enumerate(self.variables):
                layer = self.stack.layers[var_info.layer_index]
                thickness_nm = layer.thickness_um * 1000
                bounds_nm = (
                    var_info.min_val * 1000 if var_info.min_val else None,
                    var_info.max_val * 1000 if var_info.max_val else None,
                )
                print(
                    f"  {i}: Layer {var_info.layer_index} thickness = "
                    + f"{thickness_nm:.1f} nm, bounds = {bounds_nm}"
                )
            print()

        if self.targets:
            print("Targets:")
            for i, target in enumerate(self.targets):
                wl_str = (
                    f"{target.wavelength_nm}"
                    if isinstance(target.wavelength_nm, int | float)
                    else f"{len(target.wavelength_nm)} wavelengths"
                )
                print(
                    f"  {i}: {target.property} {target.target_type} {target.value}"
                    + f" at {wl_str} nm (weight={target.weight})"
                )
            print()

        if self.result:
            print("Last optimization result:")
            print(f"  Success: {self.result.success}")
            print(f"  Merit: {self.result.fun:.6f}")
            print(f"  Iterations: {self.result.nit}")
            print()
