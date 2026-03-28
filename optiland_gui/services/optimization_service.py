"""Full OptimizationService for the Optiland GUI.

Manages variable/operand lists, builds ``OptimizationProblem`` objects, and
runs the optimizer in a ``QThread`` so the main thread stays responsive.
"""

from __future__ import annotations

import json
import logging

from PySide6.QtCore import QObject, QThread, Signal, Slot

logger = logging.getLogger(__name__)


class _OptimizationWorker(QObject):
    """Runs ``optimizer.optimize()`` on a background ``QThread``.

    Signals:
        finished (str): Summary string emitted when optimisation completes.
        error (str): Error message emitted if ``optimize()`` raises.
        progress (int): Iteration counter emitted via the callback (where
            the optimizer supports one).
    """

    finished = Signal(str)
    error = Signal(str)
    progress = Signal(int)

    def __init__(self, optimizer: object, kwargs: dict) -> None:
        super().__init__()
        self._optimizer = optimizer
        self._kwargs = kwargs
        self._iteration_count = 0
        self._cancelled = False

    def request_cancel(self) -> None:
        """Signal that the run should be aborted at the next callback."""
        self._cancelled = True

    def _make_callback(self):
        """Return a callback function that forwards progress signals."""

        def callback(*args, **kwargs):  # noqa: ANN002, ANN003
            if self._cancelled:
                return True  # Truthy return stops some scipy optimisers
            self._iteration_count += 1
            self.progress.emit(self._iteration_count)

        return callback

    @Slot()
    def run(self) -> None:
        """Execute the optimiser.  Called by QThread.started."""
        import inspect

        try:
            initial_merit = float(self._optimizer.problem.rss())
        except Exception:
            initial_merit = float("nan")

        try:
            run_kwargs = dict(self._kwargs)
            sig = inspect.signature(self._optimizer.optimize)
            if "callback" in sig.parameters:
                run_kwargs.setdefault("callback", self._make_callback())

            self._optimizer.optimize(**run_kwargs)

            try:
                final_merit = float(self._optimizer.problem.rss())
            except Exception:
                final_merit = float("nan")

            summary = (
                f"Optimization complete.\n"
                f"Iterations: {self._iteration_count}\n"
                f"Initial merit: {initial_merit:.6f}\n"
                f"Final merit:   {final_merit:.6f}"
            )
            self.finished.emit(summary)
        except Exception as exc:
            logger.exception("Optimization worker error")
            self.error.emit(str(exc))


class OptimizationService:
    """Manages optimization variables, operands, and threaded execution.

    Args:
        connector: The :class:`~optiland_gui.optiland_connector.OptilandConnector`
            instance that owns this service.
    """

    # ------------------------------------------------------------------
    # Catalog constants
    # ------------------------------------------------------------------

    OPERAND_CATEGORIES: dict[str, list[str]] = {
        "Paraxial": [
            "f1",
            "f2",
            "F1",
            "F2",
            "P1",
            "P2",
            "N1",
            "N2",
            "EPD",
            "EPL",
            "XPD",
            "XPL",
            "magnification",
            "total_track",
        ],
        "Aberration": [
            "seidel",
            "TSC",
            "SC",
            "CC",
            "TCC",
            "TAC",
            "AC",
            "TPC",
            "PC",
            "DC",
            "TAchC",
            "LchC",
            "TchC",
            "TSC_sum",
            "SC_sum",
            "CC_sum",
            "TCC_sum",
            "TAC_sum",
            "AC_sum",
            "TPC_sum",
            "PC_sum",
            "DC_sum",
            "TAchC_sum",
            "LchC_sum",
            "TchC_sum",
        ],
        "Ray": [
            "real_x_intercept",
            "real_y_intercept",
            "real_z_intercept",
            "real_x_intercept_lcs",
            "real_y_intercept_lcs",
            "real_z_intercept_lcs",
            "clearance",
            "real_L",
            "real_M",
            "real_N",
            "rms_spot_size",
            "OPD_difference",
            "AOI",
        ],
        "Lens": ["edge_thickness"],
    }

    COMMON_VARIABLE_TYPES: list[tuple[str, str]] = [
        ("Radius", "radius"),
        ("Thickness", "thickness"),
        ("Conic", "conic"),
        ("Asphere Coeff", "asphere_coeff"),
        ("Index", "index"),
        ("Tilt", "tilt"),
        ("Decenter", "decenter"),
    ]

    # Default extra input_data (JSON string, optic excluded) per operand type.
    _DEFAULT_INPUT_DATA: dict[str, str] = {
        "clearance": '{"surface_number": 1}',
        "edge_thickness": '{"surface_number": 1}',
        "real_x_intercept": '{"surface_number": 1}',
        "real_y_intercept": '{"surface_number": 1}',
        "real_z_intercept": '{"surface_number": 1}',
        "real_x_intercept_lcs": '{"surface_number": 1}',
        "real_y_intercept_lcs": '{"surface_number": 1}',
        "real_z_intercept_lcs": '{"surface_number": 1}',
        "real_L": '{"surface_number": 1}',
        "real_M": '{"surface_number": 1}',
        "real_N": '{"surface_number": 1}',
        "AOI": '{"surface_number": 1}',
        "seidel": '{"seidel_number": 0, "surface_number": 1}',
    }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, connector: object) -> None:
        self._connector = connector
        self._variables: list[dict] = []
        self._operands: list[dict] = []
        self._thread: QThread | None = None
        self._worker: _OptimizationWorker | None = None

    # ------------------------------------------------------------------
    # Variable management
    # ------------------------------------------------------------------

    def add_variable(self, var_dict: dict) -> None:
        """Append a variable descriptor.

        Args:
            var_dict: Required keys: ``surface_number`` (int), ``type`` (str).
                Optional keys: ``min_val`` (float|None), ``max_val``
                (float|None), ``coeff_number`` (int|None).
        """
        self._variables.append(dict(var_dict))

    def remove_variable(self, index: int) -> None:
        """Remove a variable by its list index.

        Args:
            index: Zero-based index.
        """
        if 0 <= index < len(self._variables):
            self._variables.pop(index)

    def get_variables(self) -> list[dict]:
        """Return a shallow copy of the variable list."""
        return list(self._variables)

    def clear_variables(self) -> None:
        """Remove all registered variables."""
        self._variables.clear()

    def get_variable_current_value(self, var_dict: dict) -> float | None:
        """Read the current physical value for a variable from the live optic.

        Args:
            var_dict: A variable descriptor dict.

        Returns:
            The current float value, or ``None`` if retrieval fails.
        """
        optic = self._connector._optic
        if optic is None:
            return None
        try:
            from optiland.optimization.variable.variable import Variable as _Var

            extra: dict = {}
            if var_dict.get("coeff_number") is not None:
                extra["coeff_number"] = var_dict["coeff_number"]
            v = _Var(
                optic,
                var_dict["type"],
                surface_number=var_dict["surface_number"],
                **extra,
            )
            return float(v.variable.get_value())
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Operand management
    # ------------------------------------------------------------------

    def add_operand(self, op_dict: dict) -> None:
        """Append an operand descriptor.

        Args:
            op_dict: Required keys: ``type`` (str), ``category`` (str).
                Optional keys: ``target`` (float|None), ``min_val``
                (float|None), ``max_val`` (float|None), ``weight`` (float),
                ``input_data_str`` (str — JSON without optic).
        """
        self._operands.append(dict(op_dict))

    def remove_operand(self, index: int) -> None:
        """Remove an operand by its list index.

        Args:
            index: Zero-based index.
        """
        if 0 <= index < len(self._operands):
            self._operands.pop(index)

    def get_operands(self) -> list[dict]:
        """Return a shallow copy of the operand list."""
        return list(self._operands)

    def clear_operands(self) -> None:
        """Remove all registered operands."""
        self._operands.clear()

    def get_default_input_data_str(self, op_type: str) -> str:
        """Return the default extra-parameter JSON string for an operand type.

        Args:
            op_type: Operand type key.

        Returns:
            A JSON string (optic excluded); defaults to ``"{}"``.
        """
        return self._DEFAULT_INPUT_DATA.get(op_type, "{}")

    # ------------------------------------------------------------------
    # Problem construction
    # ------------------------------------------------------------------

    def build_problem(self, optic: object) -> object:
        """Build an :class:`~optiland.optimization.OptimizationProblem`.

        Injects the live optic into each operand's ``input_data`` before adding
        to the problem.

        Args:
            optic: The :class:`~optiland.optic.Optic` instance to optimise.

        Returns:
            A configured ``OptimizationProblem``.
        """
        from optiland.optimization import OptimizationProblem

        problem = OptimizationProblem()

        for vd in self._variables:
            extra: dict = {}
            if vd.get("coeff_number") is not None:
                extra["coeff_number"] = vd["coeff_number"]
            try:
                problem.add_variable(
                    optic,
                    vd["type"],
                    surface_number=vd["surface_number"],
                    min_val=vd.get("min_val"),
                    max_val=vd.get("max_val"),
                    **extra,
                )
            except Exception as exc:
                logger.warning(
                    "OptimizationService: skipping variable %s surface %s: %s",
                    vd.get("type"),
                    vd.get("surface_number"),
                    exc,
                )

        for od in self._operands:
            try:
                extra_data = json.loads(od.get("input_data_str") or "{}")
            except json.JSONDecodeError:
                extra_data = {}
            input_data = {"optic": optic, **extra_data}
            try:
                problem.add_operand(
                    operand_type=od["type"],
                    target=od.get("target"),
                    min_val=od.get("min_val"),
                    max_val=od.get("max_val"),
                    weight=od.get("weight", 1.0),
                    input_data=input_data,
                )
            except Exception as exc:
                logger.warning(
                    "OptimizationService: skipping operand %s: %s",
                    od.get("type"),
                    exc,
                )

        return problem

    # ------------------------------------------------------------------
    # Optimizer catalog
    # ------------------------------------------------------------------

    @staticmethod
    def get_optimizer_catalog() -> list[tuple[str, type]]:
        """Return available optimizer classes as ``(display_name, cls)`` tuples.

        Returns:
            A list of ``(name, class)`` pairs for the standard SciPy optimisers.
        """
        from optiland.optimization.optimizer.scipy import (
            SHGO,
            BasinHopping,
            DifferentialEvolution,
            DualAnnealing,
            LeastSquares,
            OptimizerGeneric,
        )

        catalog = [("Generic (scipy.minimize)", OptimizerGeneric)]
        scipy_methods = [
            "Nelder-Mead",
            "Powell",
            "CG",
            "BFGS",
            "L-BFGS-B",
            "TNC",
            "COBYLA",
            "SLSQP",
            "trust-constr",
        ]
        for method in scipy_methods:

            def make_cls(m=method):
                class ScipyMethod(OptimizerGeneric):
                    def optimize(
                        self, maxiter=1000, disp=True, tol=1e-3, callback=None
                    ):
                        return super().optimize(
                            method=m,
                            maxiter=maxiter,
                            disp=disp,
                            tol=tol,
                            callback=callback,
                        )

                return ScipyMethod

            catalog.append((method, make_cls()))

        catalog.extend(
            [
                ("Least Squares", LeastSquares),
                ("Dual Annealing", DualAnnealing),
                ("Differential Evolution", DifferentialEvolution),
                ("SHGO", SHGO),
                ("Basin Hopping", BasinHopping),
            ]
        )

        return catalog

    # ------------------------------------------------------------------
    # Threaded execution
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """``True`` while an optimisation run is in progress."""
        return self._thread is not None and self._thread.isRunning()

    def run(
        self,
        optimizer_cls: type,
        optimizer_kwargs: dict,
        on_progress: object | None = None,
        on_finished: object | None = None,
        on_error: object | None = None,
    ) -> None:
        """Capture undo state, build problem, then start the optimizer thread.

        Args:
            optimizer_cls: Optimizer class (e.g., ``LeastSquares``).
            optimizer_kwargs: Forwarded to ``optimizer.optimize()``.
            on_progress: Optional ``(iteration_count: int) -> None`` callback.
            on_finished: Optional ``(summary: str) -> None`` callback.
            on_error: Optional ``(message: str) -> None`` callback.
        """
        if self.is_running:
            return

        optic = self._connector._optic
        if optic is None:
            return

        # Capture undo checkpoint before modifying optic
        self._connector._undo_redo_manager.add_state(
            self._connector._capture_optic_state()
        )

        try:
            problem = self.build_problem(optic)
            optimizer = optimizer_cls(problem)
        except Exception as exc:
            if on_error:
                on_error(f"Failed to build optimisation problem: {exc}")
            return

        self._thread = QThread()
        self._worker = _OptimizationWorker(optimizer, optimizer_kwargs)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)

        if on_progress is not None:
            self._worker.progress.connect(on_progress)
        if on_finished is not None:
            self._worker.finished.connect(on_finished)
        if on_error is not None:
            self._worker.error.connect(on_error)

        self._thread.start()

    def stop(self) -> None:
        """Request cancellation of an in-progress run."""
        if self._worker is not None:
            self._worker.request_cancel()
