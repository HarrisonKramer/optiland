from __future__ import annotations

import warnings
from math import pi
from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.utils import resolve_wavelengths

if TYPE_CHECKING:
    from optiland.optic import Optic


class BaseWavePropagator:
    def __init__(self, optic: Optic):
        self.optic = optic

    def _build_propagator(self, num_points: int, dx: float):
        from optiland.wavepropagation.asm import AngularSpectrumPropagator as Propagator

        return Propagator(num_points, dx)

    def compute_surface_phase(self, surf, X, Y, wl_arr):
        interaction = getattr(surf, "interaction_model", None)
        if interaction is None:
            raise ValueError("Surface has no interaction_model")

        interaction_type = getattr(interaction, "interaction_type", None)
        if interaction_type not in ("phase", "refractive_reflective"):
            raise NotImplementedError(
                f"Unsupported interaction_type: {interaction_type}"
            )

        if interaction_type == "phase":
            phase_profile = getattr(interaction, "phase_profile", None)
            if phase_profile is None:
                raise ValueError("Phase surface has no phase_profile")
            phi = be.array(phase_profile.get_phase(X, Y, wl_arr))
        else:
            n = surf.material_post.n(wl_arr) if hasattr(surf, "material_post") else 1.0
            sag = be.array(surf.geometry.sag(X, Y))
            phi = (
                (2 * pi / (wl_arr * 1e-3))[:, None, None]
                * (n - 1.0)[:, None, None]
                * sag[None, :, :]
            )

        if getattr(phi, "ndim", None) == 2:
            phi = phi[None]

        dphi_x = be.max(be.abs(phi[:, 1:] - phi[:, :-1]))
        dphi_y = be.max(be.abs(phi[:, :, 1:] - phi[:, :, :-1]))
        dphi_max = be.max(be.array([dphi_x, dphi_y]))

        if dphi_max > be.pi:
            warnings.warn(
                f"Phase under-sampled: phase jump > π between pixels. Max phase jump = {dphi_max:.2f} rad.",
                RuntimeWarning,
                stacklevel=2,
            )

        phase = be.exp(1j * phi)
        if interaction_type != "phase":
            phase = phase.conj()
        return phase

    def create_input_field(self, X, Y, wl_arr, field, w0: float | str | None = "auto"):
        angle_x, angle_y = field

        k = (2 * pi / (wl_arr * 1e-3))[:, None, None]
        ax = be.array(float(angle_x))
        ay = be.array(float(angle_y))

        phase = k * (X[None] * be.sin(ax) + Y[None] * be.sin(ay))
        field = be.exp(1j * phase)

        if w0 is not None:
            if w0 == "auto":
                dx = float(X[0, 1] - X[0, 0])
                w0 = ((X.shape[0] * dx) / 2) / 2
            r2 = X**2 + Y**2
            field = field * be.exp(-(r2 / (w0**2)))[None]

        return field

    def compute_field(
        self,
        z_target: float,
        num_points: int,
        dx: float,
        field: list[tuple[float, float]] | tuple[float, float] | None = None,
        wavelengths: str | float | list = "primary",
        beam_waist: float | str | None = "auto",
    ):
        if field is None:
            fields = [(0.0, 0.0)]
        elif isinstance(field, tuple) and len(field) == 2:
            fields = [field]
        else:
            fields = list(field)

        F = len(fields)
        if F == 0:
            raise ValueError("No fields provided.")

        wavelengths_resolved = resolve_wavelengths(self.optic, wavelengths)
        wl_arr = be.array([float(w) for w in wavelengths_resolved])
        W = int(wl_arr.shape[0])
        if W == 0:
            raise ValueError("No wavelengths resolved.")

        x = be.linspace(-(num_points // 2) * dx, (num_points // 2) * dx, num_points)
        y = be.copy(x)
        Y, X = be.meshgrid(y, x, indexing="ij")

        field_arr = be.zeros((F, W, num_points, num_points)) + 0j
        for i, f in enumerate(fields):
            field_arr[i] = self.create_input_field(
                X=X, Y=Y, wl_arr=wl_arr, field=f, w0=beam_waist
            )

        propagator = self._build_propagator(num_points, dx)
        current_z = 0.0
        wl_batch = wl_arr.repeat(F)

        def propagate(field_arr, dist: float):
            flat = field_arr.reshape(F * W, num_points, num_points)
            flat = propagator(flat[:, None], dist, wl_batch)[:, 0]
            return flat.reshape(F, W, num_points, num_points)

        for surf in self.optic.surface_group.surfaces:
            phase = self.compute_surface_phase(surf, X, Y, wl_arr)
            field_arr = field_arr * phase[None]

            if getattr(surf, "aperture", None):
                aperture = be.array(surf.aperture.contains(X, Y))
                field_arr = field_arr * aperture[None, None]

            t = surf.thickness if surf.thickness != float("inf") else 0.0

            if current_z + t >= z_target:
                remaining = z_target - current_z
                if remaining > 0:
                    field_arr = propagate(field_arr, remaining)
                return field_arr

            if t > 0:
                field_arr = propagate(field_arr, t)
                current_z += t

        if z_target > current_z:
            field_arr = propagate(field_arr, z_target - current_z)

        return field_arr
