from __future__ import annotations

import gc
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display

import optiland.backend as be


class LiveOptimizationPlotter:
    def __init__(self, optimizer: Any) -> None:
        self.optimizer = optimizer
        self.history: list[float] = []

        backend = matplotlib.get_backend().lower()
        self._is_inline_backend = "inline" in backend

        self._system_fig: plt.Figure | None = None
        self._system_ax: Any = None
        self._merit_fig: plt.Figure | None = None
        self._merit_ax: Any = None
        self._merit_line: Any = None

        self._optical_system_handle: Any = None
        self._merit_function_handle: Any = None

        if not self._is_inline_backend:
            plt.ion()

    def initialize(self) -> None:
        """Create persistent figures once so CLI backends keep the windows alive."""
        if self._system_fig is None:
            self._system_fig, self._system_ax = plt.subplots()
            self._system_fig.canvas.manager.set_window_title("Optimization")

        if self._merit_fig is None:
            self._merit_fig, self._merit_ax = plt.subplots()
            (self._merit_line,) = self._merit_ax.plot([], [])
            self._merit_ax.set_yscale("log")
            self._merit_ax.set_title("Merit function value")
            self._merit_ax.set_xlabel("Iteration")
            self._merit_ax.set_ylabel("Merit function value")
            self._merit_ax.grid(alpha=0.25)
            self._merit_fig.canvas.manager.set_window_title("Merit function")

        if not self._is_inline_backend:
            assert self._system_fig is not None
            assert self._merit_fig is not None
            self._system_fig.show()
            self._merit_fig.show()
            self._position_windows()
            plt.pause(0.001)

    def update(self) -> None:
        """Update the optical system plot and the merit function plot."""
        self.initialize()

        assert self._system_fig is not None
        assert self._system_ax is not None
        assert self._merit_fig is not None
        assert self._merit_ax is not None
        assert self._merit_line is not None

        self._redraw_optical_system()
        self._update_merit_history()
        self._render()

    def _position_windows(self) -> None:
        """Place optics and merit windows side by side."""
        try:
            mgr1 = self._system_fig.canvas.manager
            mgr2 = self._merit_fig.canvas.manager

            # Qt backend
            if hasattr(mgr1.window, "setGeometry"):
                mgr1.window.setGeometry(50, 50, 800, 600)
                mgr2.window.setGeometry(900, 50, 800, 600)

        except Exception:
            pass

    def _redraw_optical_system(self) -> None:
        """Redraw the optical system into the persistent axes."""
        assert self._system_fig is not None
        assert self._system_ax is not None

        self._system_ax.clear()

        self.optimizer.problem.variables[0].optic.draw(
            title="Optimization",
            ax=self._system_ax,
        )

        self._system_fig.tight_layout()

    def _update_merit_history(self) -> None:
        f_val = self.optimizer.problem.sum_squared()
        f_val_float = float(
            be.to_numpy(f_val).item() if hasattr(be.to_numpy(f_val), "item") else f_val
        )
        self.history.append(f_val_float)

        self._merit_line.set_data(range(len(self.history)), self.history)
        self._merit_ax.relim()
        self._merit_ax.autoscale_view()
        self._merit_fig.tight_layout()

    def _render(self) -> None:
        assert self._system_fig is not None
        assert self._merit_fig is not None

        if self._is_inline_backend:
            if self._optical_system_handle is None:
                self._optical_system_handle = display(self._system_fig, display_id=True)
            else:
                self._optical_system_handle.update(self._system_fig)

            if self._merit_function_handle is None:
                self._merit_function_handle = display(self._merit_fig, display_id=True)
            else:
                self._merit_function_handle.update(self._merit_fig)
        else:
            self._system_fig.canvas.draw_idle()
            self._system_fig.canvas.flush_events()
            self._merit_fig.canvas.draw_idle()
            self._merit_fig.canvas.flush_events()
            plt.pause(0.001)

        gc.collect()

    def finalize(self) -> None:
        if self._is_inline_backend:
            if self._merit_fig is not None:
                plt.close(self._merit_fig)
            if self._system_fig is not None:
                plt.close(self._system_fig)
            return
        plt.ioff()
        plt.show()
