"""Batched Ray Evaluator Module

This module provides the BatchedRayEvaluator class which dramatically accelerates
optimization by eliminating redundant ray tracing through intelligent batching
of operand evaluations.

The evaluator analyzes optimization problems to create efficient "trace jobs" that
batch multiple operands together, then evaluates all operands using pre-traced
data for maximum performance.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.optimization.operand.ray import RayOperand

if TYPE_CHECKING:
    from optiland.optimization.problem import OptimizationProblem


class TraceJob:
    """Represents a single trace operation that can satisfy multiple operands."""

    def __init__(self, job_id: str, job_type: str, optic, **params):
        self.job_id = job_id
        self.job_type = job_type  # "trace_generic" or "trace"
        self.optic = optic
        self.params = params
        self.operand_indices = []  # Which operands use this job
        self.ray_mappings = []  # How each operand maps to rays in this job

    def add_operand(self, operand_idx: int, ray_indices: list[int] | slice):
        """Add an operand that will use this trace job."""
        self.operand_indices.append(operand_idx)
        self.ray_mappings.append(ray_indices)

    def execute(self) -> Any:
        """Execute the trace job and return a copy of the surface_group."""
        if self.job_type == "trace_generic":
            self.optic.trace_generic(**self.params)
        elif self.job_type == "trace":
            self.optic.trace(**self.params)
        elif self.job_type == "vectorized_trace":
            # Execute vectorized ray tracing with multiple rays in a single call
            self._execute_vectorized_trace()
        else:
            raise ValueError(f"Unknown job type: {self.job_type}")
        return self.optic.surface_group

    def _execute_vectorized_trace(self):
        """Execute vectorized ray tracing for multiple rays at once."""
        import optiland.backend as be

        ray_data = self.params.get("ray_batch", [])
        if not ray_data:
            return

        # Vectorize the input by creating arrays from the list of ray parameters
        Hx_batch = be.array([ray["Hx"] for ray in ray_data])
        Hy_batch = be.array([ray["Hy"] for ray in ray_data])
        Px_batch = be.array([ray["Px"] for ray in ray_data])
        Py_batch = be.array([ray["Py"] for ray in ray_data])
        wavelength_batch = be.array([ray["wavelength"] for ray in ray_data])

        # Single vectorized call to the ray tracer
        self.optic.trace_generic(
            Hx_batch, Hy_batch, Px_batch, Py_batch, wavelength_batch
        )


class BatchedRayEvaluator:
    """High-performance evaluator for optimization problems using batched ray tracing.

    This evaluator analyzes the optimization problem to identify opportunities for
    batching ray traces, then provides a single evaluate() method that computes
    all operand values efficiently.
    """

    def __init__(self, problem: OptimizationProblem):
        """Initialize the evaluator by analyzing the problem structure.

        Args:
            problem: The optimization problem to analyze and evaluate.
        """
        self.problem = problem
        self.trace_jobs: list[TraceJob] = []
        # (job_idx, ray_mapping, calc_func)
        self.operand_map: list[tuple[int, int, Any]] = []

        # Ray operand types that we can batch
        self.ray_operand_types = {
            "real_x_intercept",
            "real_y_intercept",
            "real_z_intercept",
            "real_x_intercept_lcs",
            "real_y_intercept_lcs",
            "real_z_intercept_lcs",
            "real_L",
            "real_M",
            "real_N",
            "AOI",
            "rms_spot_size",
        }

        self._analyze_problem()

    def _analyze_problem(self):
        """Analyze the problem to create optimal vectorized trace jobs."""
        # Group operands by optic for vectorized tracing
        # Key: optic_id, Value: {operand_indices, ray_data}
        optic_groups = defaultdict(lambda: {"operands": [], "rays": []})
        trace_groups = defaultdict(list)  # For rms_spot_size operands
        non_ray_operands = []  # Operand indices for non-ray operands

        for i, operand in enumerate(self.problem.operands):
            if operand.operand_type not in self.ray_operand_types:
                non_ray_operands.append(i)
                continue

            optic = operand.input_data["optic"]
            optic_id = id(optic)

            if operand.operand_type == "rms_spot_size":
                # Uses optic.trace() - handle separately for now
                Hx = operand.input_data["Hx"]
                Hy = operand.input_data["Hy"]
                wavelength = operand.input_data["wavelength"]
                num_rays = operand.input_data["num_rays"]
                distribution = operand.input_data.get("distribution", "hexapolar")

                # Skip multi-wavelength case for now
                if wavelength == "all":
                    non_ray_operands.append(i)
                    continue

                trace_key = (
                    optic_id,
                    Hx,
                    Hy,
                    wavelength,
                    num_rays,
                    distribution,
                    "trace",
                )
                trace_groups[trace_key].append(i)
            else:
                # Group ray operands by optic for vectorized tracing
                ray_data = {
                    "Hx": operand.input_data["Hx"],
                    "Hy": operand.input_data["Hy"],
                    "Px": operand.input_data["Px"],
                    "Py": operand.input_data["Py"],
                    "wavelength": operand.input_data["wavelength"],
                }

                optic_groups[optic_id]["operands"].append(i)
                optic_groups[optic_id]["rays"].append(ray_data)
                optic_groups[optic_id]["optic"] = optic  # Store optic reference

        job_idx = 0

        # Create vectorized trace jobs for ray operands
        for _optic_id, group_data in optic_groups.items():
            operand_indices = group_data["operands"]
            ray_data = group_data["rays"]
            optic = group_data["optic"]

            if not operand_indices:
                continue

            job_id = f"vectorized_trace_{job_idx}"
            job = TraceJob(
                job_id=job_id,
                job_type="vectorized_trace",
                optic=optic,
                ray_batch=ray_data,
            )

            # Map each operand to its corresponding ray index in the batch
            for local_ray_idx, operand_idx in enumerate(operand_indices):
                job.add_operand(operand_idx, [local_ray_idx])

                # Create operand mapping
                operand = self.problem.operands[operand_idx]
                calc_func_name = (
                    operand.operand_type.replace("real_", "") + "_from_traced"
                )
                calc_func = getattr(RayOperand, calc_func_name)

                self.operand_map.append((job_idx, [local_ray_idx], calc_func))

            self.trace_jobs.append(job)
            job_idx += 1

        # Create trace jobs for rms_spot_size operands (keep existing logic)
        for trace_key, operand_indices in trace_groups.items():
            optic_id, Hx, Hy, wavelength, num_rays, distribution, job_type = trace_key

            # Find the optic object from the first operand
            optic = self.problem.operands[operand_indices[0]].input_data["optic"]

            job_id = f"trace_{job_idx}"
            job = TraceJob(
                job_id=job_id,
                job_type="trace",
                optic=optic,
                Hx=Hx,
                Hy=Hy,
                wavelength=wavelength,
                num_rays=num_rays,
                distribution=distribution,
            )

            # All these operands use all rays
            for operand_idx in operand_indices:
                job.add_operand(operand_idx, slice(None))

                # Create operand mapping - rms_spot_size uses slice(None) for all rays
                calc_func = RayOperand.rms_spot_size_from_traced
                self.operand_map.append((job_idx, slice(None), calc_func))

            self.trace_jobs.append(job)
            job_idx += 1

        # Handle non-ray operands - they evaluate directly without batching
        for _operand_idx in non_ray_operands:
            self.operand_map.append((None, None, None))  # Marker for direct evaluation

    def evaluate(self, variables_vector: list | Any) -> float | Any:
        """Evaluate the merit function efficiently using batched ray tracing.

        Args:
            variables_vector: The current values of optimization variables.
                Can be a list (for SciPy) or torch Parameters (for PyTorch).

        Returns:
            The sum of squared weighted operand deltas. Returns a torch.Tensor
            if using PyTorch backend, or a float if using NumPy backend.
        """
        try:
            # (new operands may have been added)
            if len(self.operand_map) != len(self.problem.operands):
                self._analyze_problem()

            # Update all variables to their new values
            is_iterable = hasattr(variables_vector, "__iter__")
            if is_iterable and not be.is_array_like(variables_vector):
                # Handle list of torch.nn.Parameter objects or plain list
                for i, var in enumerate(self.problem.variables):
                    if hasattr(variables_vector[i], "data"):
                        # torch.nn.Parameter
                        var.variable.update_value(variables_vector[i])
                    else:
                        # Plain value
                        var.update(be.array(variables_vector[i]))
            else:
                # Handle numpy array or single torch tensor
                for i, var in enumerate(self.problem.variables):
                    # Check if it's already a tensor to avoid unnecessary conversion
                    if be.is_array_like(variables_vector[i]):
                        var.update(variables_vector[i])
                    else:
                        var.update(be.array(variables_vector[i]))

            # Update optics
            self.problem.update_optics()

            # Execute all trace jobs and store results
            traced_results = []
            job_success = []  # Track which jobs succeeded
            for job in self.trace_jobs:
                try:
                    result = job.execute()
                    traced_results.append(result)
                    job_success.append(True)
                except Exception as e:
                    # If tracing fails, add a placeholder and mark as failed
                    print(f"TraceJob failed: {e}")
                    traced_results.append(None)
                    job_success.append(False)

            # Compute all operand values using the traced data
            operand_values = []

            for i, operand in enumerate(self.problem.operands):
                try:
                    job_idx, ray_indices, calc_func = self.operand_map[i]

                    if job_idx is None:
                        # Non-ray operand
                        from optiland.optimization.operand.operand import (
                            operand_registry,
                        )

                        metric_function = operand_registry.get(operand.operand_type)
                        value = metric_function(**operand.input_data)
                    else:
                        # Ray operand - use traced data
                        if job_idx >= len(traced_results) or not job_success[job_idx]:
                            # Job failed or index out of bounds
                            raise RuntimeError(
                                f"TraceJob {job_idx} failed or not found"
                            )

                        traced_data = traced_results[job_idx]
                        if traced_data is None:
                            raise RuntimeError(f"TraceJob {job_idx} returned None")

                        value = calc_func(traced_data, operand.input_data, ray_indices)

                    # Calculate operand contribution (weight * delta)
                    if operand.target is not None:
                        delta = value - operand.target
                    elif operand.min_val is not None or operand.max_val is not None:
                        lower_penalty = (
                            be.maximum(be.array(0.0), operand.min_val - value)
                            if operand.min_val is not None
                            else be.array(0.0)
                        )
                        upper_penalty = (
                            be.maximum(be.array(0.0), value - operand.max_val)
                            if operand.max_val is not None
                            else be.array(0.0)
                        )
                        delta = lower_penalty + upper_penalty
                    else:
                        raise ValueError(f"Operand {i} has no target or bounds defined")

                    weighted_delta = operand.weight * delta
                    operand_values.append(weighted_delta)

                except Exception as e:
                    # If operand evaluation fails, add high penalty
                    print(f"Operand {i} evaluation failed: {e}")
                    operand_values.append(be.array(1e10))

            # Sum all squared operand contributions
            if not operand_values:
                return be.array(0.0)

            squared_values = [val**2 for val in operand_values]
            if len(squared_values) > 1:
                total = be.sum(be.stack(squared_values))
            else:
                total = be.sum(squared_values[0])

            return total

        except Exception as e:
            print(f"BatchedRayEvaluator.evaluate() failed: {e}")
            import traceback

            traceback.print_exc()
            return be.array(1e10) if be.get_backend() == "torch" else 1e10
