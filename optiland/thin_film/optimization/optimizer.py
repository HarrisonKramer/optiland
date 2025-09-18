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
from scipy.interpolate import interp1d
from scipy.optimize import minimize

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

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
    value: float | list[float]
    weight: float
    aoi_deg: float | list[float]
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
        # Ensure minimum thickness is always positive (at least 1 nm = 0.001 μm)
        min_val = min_nm / 1000.0 if min_nm is not None else None
        if min_val is not None and min_val <= 0:
            min_val = 0.001  # Force minimum to 1 nm

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

    def add_target(
        self,
        property: OpticalProperty,
        wavelength_nm: float | list[float],
        target_type: TargetType,
        value: float | list[float],
        weight: float = 1.0,
        aoi_deg: float | list[float] = 0.0,
        polarization: str = "u",
        tolerance: float = 1e-6,
    ) -> ThinFilmOptimizer:
        """Add an optimization target.

        Args:
            property: Optical property to target ('R', 'T', or 'A').
            wavelength_nm: Wavelength(s) in nanometers. Can be scalar or array.
            target_type: Type of target ('equal', 'below', 'over').
            value: Target value(s). Can be scalar or array for interpolation.
            weight: Weight for this target. Defaults to 1.0.
            aoi_deg: Angle(s) of incidence in degrees. Can be scalar or array.
                Defaults to 0.0.
            polarization: Polarization state ('s', 'p', 'u'). Defaults to 'u'.
            tolerance: Tolerance for 'equal' targets. Defaults to 1e-6.

        Returns:
            self for method chaining.

        Raises:
            ValueError: If property, target_type is invalid, or both wavelength_nm
                and aoi_deg are arrays.
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
            elif is_aoi_array and len(value) != len(aoi_deg):
                raise ValueError(
                    f"Length of value array ({len(value)}) must match "
                    f"length of aoi_deg array ({len(aoi_deg)})"
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

    def add_angular_target(
        self,
        property: str,
        wavelength_nm: float,
        aoi_deg_range: list[float],
        target_type: str,
        value: float | list[float],
        weight: float = 1.0,
        polarization: str = "s",
    ) -> ThinFilmOptimizer:
        """
        Convenience method to add an angular target with multiple AOI values.

        Parameters:
        -----------
        property : str
            Property to optimize ("R", "T", "A")
        wavelength_nm : float
            Single wavelength value in nm
        aoi_deg_range : list[float]
            List of angles of incidence in degrees
        target_type : str
            Type of target ("equal", "over", "below")
        value : float or list[float]
            Target value(s) - single value or list matching aoi_deg_range length
        weight : float, optional
            Target weight for optimization
        polarization : str, optional
            Polarization state ("s", "p", "u")
        """
        return self.add_target(
            property=property,
            wavelength_nm=wavelength_nm,
            target_type=target_type,
            value=value,
            weight=weight,
            aoi_deg=aoi_deg_range,
            polarization=polarization,
        )

    def add_interpolated_target(
        self,
        property: str,
        wavelength_nm: list[float],
        target_type: str,
        value: list[float],
        weight: float = 1.0,
        aoi_deg: float = 0.0,
        polarization: str = "s",
    ) -> ThinFilmOptimizer:
        """
        Convenience method to add an interpolated spectral target.

        Parameters:
        -----------
        property : str
            Property to optimize ("R", "T", "A")
        wavelength_nm : list[float]
            List of wavelength values in nm
        target_type : str
            Type of target ("equal", "over", "below")
        value : list[float]
            List of target values matching wavelength_nm length
        weight : float, optional
            Target weight for optimization
        aoi_deg : float, optional
            Angle of incidence in degrees
        polarization : str, optional
            Polarization state ("s", "p", "u")
        """
        return self.add_target(
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
        # If value is scalar, return as-is
        if isinstance(target.value, int | float):
            return float(target.value)

        # If value is array, interpolate
        value_array = np.array(target.value)

        # Determine interpolation axis
        if isinstance(target.wavelength_nm, list | np.ndarray):
            # Interpolate along wavelength
            if current_wl is None:
                raise ValueError(
                    "current_wl must be provided for wavelength interpolation"
                )
            wl_array = np.array(target.wavelength_nm)
            if len(value_array) != len(wl_array):
                raise ValueError("Value and wavelength arrays must have same length")

            # Use linear interpolation, with extrapolation for out-of-bounds
            interp_func = interp1d(
                wl_array,
                value_array,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            return float(interp_func(current_wl))

        elif isinstance(target.aoi_deg, list | np.ndarray):
            # Interpolate along AOI
            if current_aoi is None:
                raise ValueError("current_aoi must be provided for AOI interpolation")
            aoi_array = np.array(target.aoi_deg)
            if len(value_array) != len(aoi_array):
                raise ValueError("Value and AOI arrays must have same length")

            # Use linear interpolation, with extrapolation for out-of-bounds
            interp_func = interp1d(
                aoi_array,
                value_array,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            return float(interp_func(current_aoi))

        else:
            # Neither wavelength nor AOI is array, but value is array - take first value
            return float(value_array[0])

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
            # Handle different combinations of array/scalar parameters
            if isinstance(target.wavelength_nm, list | np.ndarray):
                # Wavelength is array, AOI should be scalar
                aoi_deg = (
                    float(target.aoi_deg)
                    if not isinstance(target.aoi_deg, list | np.ndarray)
                    else target.aoi_deg[0]
                )

                # Calculate property for each wavelength
                total_residual = 0.0
                wavelengths = np.array(target.wavelength_nm)

                for wl in wavelengths:
                    # Get interpolated target value for this wavelength
                    target_value = self._interpolate_target_value(target, current_wl=wl)

                    # Calculate current property value
                    if target.property == "R":
                        current_value = ThinFilmOperand.reflectance(
                            self.stack, wl, aoi_deg, target.polarization
                        )
                    elif target.property == "T":
                        current_value = ThinFilmOperand.transmittance(
                            self.stack, wl, aoi_deg, target.polarization
                        )
                    elif target.property == "A":
                        current_value = ThinFilmOperand.absorptance(
                            self.stack, wl, aoi_deg, target.polarization
                        )

                    # Compute residual based on target type
                    if target.target_type == "equal":
                        residual = current_value - target_value
                    elif target.target_type == "below":
                        residual = max(0, current_value - target_value)
                    elif target.target_type == "over":
                        residual = max(0, target_value - current_value)

                    total_residual += residual**2

                # Average over wavelengths and apply weight
                merit += target.weight * total_residual / len(wavelengths)

            elif isinstance(target.aoi_deg, list | np.ndarray):
                # AOI is array, wavelength should be scalar
                wavelength_nm = (
                    float(target.wavelength_nm)
                    if not isinstance(target.wavelength_nm, list | np.ndarray)
                    else target.wavelength_nm[0]
                )

                # Calculate property for each AOI
                total_residual = 0.0
                aoi_angles = np.array(target.aoi_deg)

                for aoi in aoi_angles:
                    # Get interpolated target value for this AOI
                    target_value = self._interpolate_target_value(
                        target, current_aoi=aoi
                    )

                    # Calculate current property value
                    if target.property == "R":
                        current_value = ThinFilmOperand.reflectance(
                            self.stack, wavelength_nm, aoi, target.polarization
                        )
                    elif target.property == "T":
                        current_value = ThinFilmOperand.transmittance(
                            self.stack, wavelength_nm, aoi, target.polarization
                        )
                    elif target.property == "A":
                        current_value = ThinFilmOperand.absorptance(
                            self.stack, wavelength_nm, aoi, target.polarization
                        )

                    # Compute residual based on target type
                    if target.target_type == "equal":
                        residual = current_value - target_value
                    elif target.target_type == "below":
                        residual = max(0, current_value - target_value)
                    elif target.target_type == "over":
                        residual = max(0, target_value - current_value)

                    total_residual += residual**2

                # Average over AOIs and apply weight
                merit += target.weight * total_residual / len(aoi_angles)

            else:
                # Both wavelength and AOI are scalars
                wavelength_nm = float(target.wavelength_nm)
                aoi_deg = float(target.aoi_deg)
                target_value = self._interpolate_target_value(target)

                # Calculate current property value
                if target.property == "R":
                    current_value = ThinFilmOperand.reflectance(
                        self.stack, wavelength_nm, aoi_deg, target.polarization
                    )
                elif target.property == "T":
                    current_value = ThinFilmOperand.transmittance(
                        self.stack, wavelength_nm, aoi_deg, target.polarization
                    )
                elif target.property == "A":
                    current_value = ThinFilmOperand.absorptance(
                        self.stack, wavelength_nm, aoi_deg, target.polarization
                    )

                # Compute residual based on target type
                if target.target_type == "equal":
                    residual = current_value - target_value
                elif target.target_type == "below":
                    residual = max(0, current_value - target_value)
                elif target.target_type == "over":
                    residual = max(0, target_value - current_value)

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
            # Handle different parameter combinations
            if isinstance(target.wavelength_nm, list | np.ndarray):
                # Wavelength array case
                aoi_deg = (
                    float(target.aoi_deg)
                    if not isinstance(target.aoi_deg, list | np.ndarray)
                    else target.aoi_deg[0]
                )
                wavelengths = np.array(target.wavelength_nm)

                current_values = []
                target_values = []

                for wl in wavelengths:
                    target_val = self._interpolate_target_value(target, current_wl=wl)
                    target_values.append(target_val)

                    if target.property == "R":
                        current_val = ThinFilmOperand.reflectance(
                            self.stack, wl, aoi_deg, target.polarization
                        )
                    elif target.property == "T":
                        current_val = ThinFilmOperand.transmittance(
                            self.stack, wl, aoi_deg, target.polarization
                        )
                    elif target.property == "A":
                        current_val = ThinFilmOperand.absorptance(
                            self.stack, wl, aoi_deg, target.polarization
                        )
                    current_values.append(current_val)

                performance[f"target_{i}"] = {
                    "property": target.property,
                    "wavelength_nm": target.wavelength_nm,
                    "aoi_deg": aoi_deg,
                    "target_type": target.target_type,
                    "target_values": target_values,
                    "current_values": current_values,
                    "differences": [
                        c - t
                        for c, t in zip(current_values, target_values, strict=False)
                    ],
                    "weight": target.weight,
                }

            elif isinstance(target.aoi_deg, list | np.ndarray):
                # AOI array case
                wavelength_nm = (
                    float(target.wavelength_nm)
                    if not isinstance(target.wavelength_nm, list | np.ndarray)
                    else target.wavelength_nm[0]
                )
                aoi_angles = np.array(target.aoi_deg)

                current_values = []
                target_values = []

                for aoi in aoi_angles:
                    target_val = self._interpolate_target_value(target, current_aoi=aoi)
                    target_values.append(target_val)

                    if target.property == "R":
                        current_val = ThinFilmOperand.reflectance(
                            self.stack, wavelength_nm, aoi, target.polarization
                        )
                    elif target.property == "T":
                        current_val = ThinFilmOperand.transmittance(
                            self.stack, wavelength_nm, aoi, target.polarization
                        )
                    elif target.property == "A":
                        current_val = ThinFilmOperand.absorptance(
                            self.stack, wavelength_nm, aoi, target.polarization
                        )
                    current_values.append(current_val)

                performance[f"target_{i}"] = {
                    "property": target.property,
                    "wavelength_nm": wavelength_nm,
                    "aoi_deg": target.aoi_deg,
                    "target_type": target.target_type,
                    "target_values": target_values,
                    "current_values": current_values,
                    "differences": [
                        c - t
                        for c, t in zip(current_values, target_values, strict=False)
                    ],
                    "weight": target.weight,
                }

            else:
                # Scalar case
                wavelength_nm = float(target.wavelength_nm)
                aoi_deg = float(target.aoi_deg)
                target_value = self._interpolate_target_value(target)

                if target.property == "R":
                    current_value = ThinFilmOperand.reflectance(
                        self.stack, wavelength_nm, aoi_deg, target.polarization
                    )
                elif target.property == "T":
                    current_value = ThinFilmOperand.transmittance(
                        self.stack, wavelength_nm, aoi_deg, target.polarization
                    )
                elif target.property == "A":
                    current_value = ThinFilmOperand.absorptance(
                        self.stack, wavelength_nm, aoi_deg, target.polarization
                    )

                performance[f"target_{i}"] = {
                    "property": target.property,
                    "wavelength_nm": wavelength_nm,
                    "aoi_deg": aoi_deg,
                    "target_type": target.target_type,
                    "target_value": target_value,
                    "current_value": current_value,
                    "difference": current_value - target_value,
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

    def plot_targets(
        self,
        ax,
        plot_type: Literal["wavelength", "angle"] = "wavelength",
        wavelength_range_nm: tuple[float, float] | None = None,
        angle_range_deg: tuple[float, float] | None = None,
        num_points: int = 100,
        fixed_wavelength_nm: float = 550.0,
        fixed_angle_deg: float = 0.0,
    ) -> None:
        """Plot optimization targets on the provided axes (lightweight version).

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

        if not self.targets:
            return

        # Determine plotting range
        if plot_type == "wavelength":
            if wavelength_range_nm is None:
                wl_values = []
                for target in self.targets:
                    if isinstance(target.wavelength_nm, list | np.ndarray):
                        wl_values.extend(target.wavelength_nm)
                    else:
                        wl_values.append(target.wavelength_nm)

                if wl_values:
                    margin = (max(wl_values) - min(wl_values)) * 0.1
                    wavelength_range_nm = (
                        min(wl_values) - margin,
                        max(wl_values) + margin,
                    )
                else:
                    wavelength_range_nm = (400, 800)

            x_values = np.linspace(
                wavelength_range_nm[0], wavelength_range_nm[1], num_points
            )

        elif plot_type == "angle":
            if angle_range_deg is None:
                angle_values = []
                for target in self.targets:
                    if isinstance(target.aoi_deg, list | np.ndarray):
                        angle_values.extend(target.aoi_deg)
                    else:
                        angle_values.append(target.aoi_deg)

                if angle_values:
                    margin = (max(angle_values) - min(angle_values)) * 0.1
                    angle_range_deg = (
                        min(angle_values) - margin,
                        max(angle_values) + margin,
                    )
                else:
                    angle_range_deg = (0, 80)

            x_values = np.linspace(angle_range_deg[0], angle_range_deg[1], num_points)
        else:
            raise ValueError(
                f"Invalid plot_type '{plot_type}'. Must be 'wavelength' or 'angle'."
            )

        # Color and style maps
        color_map = {"R": "red", "T": "blue", "A": "green"}
        target_styles = {"equal": "-", "below": "--", "over": ":"}

        # Plot each target
        for _i, target in enumerate(self.targets):
            color = color_map.get(target.property, "black")
            style = target_styles.get(target.target_type, "-")

            if plot_type == "wavelength":
                # Handle wavelength-dependent targets
                if isinstance(target.wavelength_nm, list | np.ndarray):
                    wl_array = np.array(target.wavelength_nm)

                    if isinstance(target.value, list | np.ndarray):
                        value_array = np.array(target.value)
                        interp_func = interp1d(
                            wl_array,
                            value_array,
                            kind="linear",
                            bounds_error=False,
                            fill_value="extrapolate",
                        )
                        y_target = interp_func(x_values)
                    else:
                        y_target = np.full_like(x_values, target.value)

                    label = f"{target.property} {target.target_type}"
                    ax.plot(
                        x_values, y_target, linestyle=style, color=color, label=label
                    )

                elif not isinstance(target.aoi_deg, list | np.ndarray):
                    # Single wavelength target
                    if (
                        wavelength_range_nm[0]
                        <= target.wavelength_nm
                        <= wavelength_range_nm[1]
                    ):
                        label = f"{target.property} @ {target.wavelength_nm}nm"
                        ax.axvline(
                            target.wavelength_nm,
                            color=color,
                            linestyle=style,
                            label=label,
                        )

            elif plot_type == "angle":
                # Handle angle-dependent targets
                if isinstance(target.aoi_deg, list | np.ndarray):
                    angle_array = np.array(target.aoi_deg)

                    if isinstance(target.value, list | np.ndarray):
                        value_array = np.array(target.value)
                        interp_func = interp1d(
                            angle_array,
                            value_array,
                            kind="linear",
                            bounds_error=False,
                            fill_value="extrapolate",
                        )
                        y_target = interp_func(x_values)
                    else:
                        y_target = np.full_like(x_values, target.value)

                    label = f"{target.property} {target.target_type}"
                    ax.plot(
                        x_values, y_target, linestyle=style, color=color, label=label
                    )

                elif not isinstance(target.wavelength_nm, list | np.ndarray):
                    # Single angle target
                    if angle_range_deg[0] <= target.aoi_deg <= angle_range_deg[1]:
                        label = f"{target.property} @ {target.aoi_deg}°"
                        ax.axvline(
                            target.aoi_deg, color=color, linestyle=style, label=label
                        )

        # Add legend
        ax.legend()

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
                    value_str = f"{target.value:.3f}"

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
