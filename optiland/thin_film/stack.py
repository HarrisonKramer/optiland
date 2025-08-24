from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import optiland.backend as be
from optiland.materials import IdealMaterial

if TYPE_CHECKING:
    from optiland.materials import BaseMaterial
import re

import matplotlib.pyplot as plt

from .layer import Layer

Pol = Literal["s", "p", "u"]


@dataclass
class ThinFilmStack:
    """Multilayer thin-film stack with inlined TMM calculations.

    This class encapsulates both the stack structure (incident/substrate, layers)
    and the numerical Transfer Matrix Method (TMM) to compute complex amplitude
    coefficients (r, t) and power coefficients (R, T, A) for s, p and unpolarized
    cases.

    Units and conventions:
    - Wavelength in microns (µm) internally; convenience helpers accept nm.
    - AOI in radians internally; convenience helpers accept degrees.
    - Layers are ordered from the incident side to the substrate side.

    Parameters
    ----------
    incident : BaseMaterial
        Incident medium (e.g., air).
    substrate : BaseMaterial
        Substrate medium (e.g., glass).
    layers : list[Layer], optional
        Ordered layers between incident and substrate, default empty.

    Examples
    --------
    >>> from optiland.materials import IdealMaterial
    >>> air, glass = IdealMaterial(1.0), IdealMaterial(1.52)
    >>> tf = ThinFilmStack(incident=air, substrate=glass)
    >>> # 100 nm SiO2 on glass
    >>> sio2 = IdealMaterial(1.46)
    >>> tf.add_layer_nm(sio2, 100.0)
    >>> R = tf.reflectance_nm_deg([550.0], [0.0], polarization="s")
    >>> T = tf.transmittance_nm_deg([550.0], [0.0], polarization="s")
    >>> A = tf.absorptance_nm_deg([550.0], [0.0], polarization="s")
    """

    incident: BaseMaterial
    substrate: BaseMaterial
    layers: list[Layer] = field(default_factory=list)

    # ----- structure helpers -----
    def add_layer(
        self, material: BaseMaterial, thickness_um: float, name: str | None = None
    ) -> ThinFilmStack:
        """Append a layer to the stack.

        Args:
            material: Optiland material providing n(λ), k(λ).
            thickness_um: Thickness in microns (µm).
            name: Optional label.

        Returns:
            self for chaining.
        """
        self.layers.append(Layer(material, thickness_um, name))
        return self

    def add_layer_nm(
        self, material: BaseMaterial, thickness_nm: float, name: str | None = None
    ) -> ThinFilmStack:
        """Append a layer, thickness in nm.

        Args:
            material: Optiland material providing n(λ), k(λ).
            thickness_nm: Thickness in nanometers.
            name: Optional label.
        """
        return self.add_layer(material, thickness_nm / 1000.0, name)

    # ----- units helpers -----
    @staticmethod
    def _to_um(wavelength_um_or_nm: float | Any, assume_nm: bool = False):
        arr = be.atleast_1d(wavelength_um_or_nm)
        return arr / 1000.0 if assume_nm else arr

    @staticmethod
    def _deg_to_rad(angle_deg: float | Any):
        return be.atleast_1d(angle_deg) * (be.pi / 180.0)

    # ----- public API: coefficients -----
    def coefficients(
        self,
        wavelength_um: float | Any,
        aoi_rad: float | Any = 0.0,
        polarization: Pol = "s",
    ) -> dict[str, Any]:
        """Compute complex and power coefficients over λ×θ grids.

        Args:
            wavelength_um: Wavelength(s) in microns (scalar or array). Use helpers
            for nm.
            aoi_rad: Angle(s) of incidence in radians (scalar or array). Use helpers
            for degrees.
            polarization: 's', 'p' or 'u' (unpolarized averages powers of s and p).

        Returns:
            Dict with keys 'r','t','R','T','A'. Shapes are (Nλ, Nθ).
        """
        wl = be.atleast_1d(wavelength_um)
        th = be.atleast_1d(aoi_rad)
        if polarization in ("s", "p"):
            r, t, R, T, A = _tmm_coh(self, wl[:, None], th[None, :], polarization)

            return {"r": r, "t": t, "R": R, "T": T, "A": A}
        elif polarization == "u":
            rs, ts, Rs, Ts, As = _tmm_coh(self, wl[:, None], th[None, :], "s")
            rp, tp, Rp, Tp, Ap = _tmm_coh(self, wl[:, None], th[None, :], "p")
            R = 0.5 * (Rs + Rp)
            T = 0.5 * (Ts + Tp)
            A = 0.5 * (As + Ap)
            # Return s-amplitudes for reference; intensities are averaged
            return {"r": rs, "t": ts, "R": R, "T": T, "A": A}
        else:
            raise ValueError("polarization must be 's', 'p' or 'u'")

    def coefficients_nm_deg(
        self,
        wavelength_nm: float | Any,
        aoi_deg: float | Any = 0.0,
        polarization: Pol = "s",
    ) -> dict[str, Any]:
        """Same as coefficients() but inputs in nm and degrees."""
        wl_um = self._to_um(wavelength_nm, assume_nm=True)
        th_rad = self._deg_to_rad(aoi_deg)
        return self.coefficients(wl_um, th_rad, polarization)

    # ----- convenience getters -----
    def reflectance(self, wavelength_um, aoi_rad=0.0, polarization: Pol = "s"):
        return self.coefficients(wavelength_um, aoi_rad, polarization)["R"]

    def transmittance(self, wavelength_um, aoi_rad=0.0, polarization: Pol = "s"):
        return self.coefficients(wavelength_um, aoi_rad, polarization)["T"]

    def absorptance(self, wavelength_um, aoi_rad=0.0, polarization: Pol = "s"):
        return self.coefficients(wavelength_um, aoi_rad, polarization)["A"]

    def reflectance_nm_deg(self, wavelength_nm, aoi_deg=0.0, polarization: Pol = "s"):
        return self.coefficients_nm_deg(wavelength_nm, aoi_deg, polarization)["R"]

    def transmittance_nm_deg(self, wavelength_nm, aoi_deg=0.0, polarization: Pol = "s"):
        return self.coefficients_nm_deg(wavelength_nm, aoi_deg, polarization)["T"]

    def absorptance_nm_deg(self, wavelength_nm, aoi_deg=0.0, polarization: Pol = "s"):
        return self.coefficients_nm_deg(wavelength_nm, aoi_deg, polarization)["A"]

    def __len__(self):
        return len(self.layers)

    def __repr__(self):
        parts = [layer.name or f"Layer({i})" for i, layer in enumerate(self.layers)]
        return f"ThinFilmStack({len(self.layers)} layers: " + " -> ".join(parts) + ")"

    def plot_structure(self, ax: plt.Axes = None) -> tuple[plt.Figure, plt.Axes]:
        if ax is None:
            fig, ax = plt.subplots()
        import matplotlib.colors as mcolors

        color_cycle = list(mcolors.TABLEAU_COLORS.values())

        def _get_name(obj):
            name = getattr(obj, "name", "") or ""
            if isinstance(obj, IdealMaterial):
                name = f"$n$ = {obj.index[0]}"
            return name

        def _add_rect(y, height, color, label, text=None):
            ax.add_patch(
                plt.Rectangle(
                    (0, y),
                    1,
                    height,
                    color=color,
                    label=label,
                    alpha=0.7,
                    edgecolor="k",
                )
            )
            if text is not None:
                ax.text(
                    0.5,
                    y + height / 2,
                    text,
                    ha="center",
                    va="center",
                    fontsize=10,
                    rotation=0,
                )

        material_names = (
            [_get_name(self.incident)]
            + [_get_name(layer.material) for layer in self.layers]
            + [_get_name(self.substrate)]
        )
        unique_materials = {
            name: color_cycle[i % len(color_cycle)]
            for i, name in enumerate(dict.fromkeys(material_names))
        }
        total_layer_thickness = sum(layer.thickness_um for layer in self.layers)

        incident_thickness = 0.08 * total_layer_thickness
        substrate_thickness = 0.08 * total_layer_thickness
        y = -substrate_thickness

        # Substrate (bottom, negative y)
        _add_rect(
            y,
            substrate_thickness,
            unique_materials[_get_name(self.substrate)],
            label=_get_name(self.substrate),
            text=_get_name(self.substrate),
        )
        y = 0

        # Layers (middle, positive y)
        for _, layer in enumerate(self.layers):
            color = unique_materials[_get_name(layer.material)]
            label = layer.name or _get_name(layer.material)
            if label:
                label = re.sub(r"\d+", lambda m: str(int(m.group())), label)
            _add_rect(
                y,
                layer.thickness_um,
                color,
                label=label,
                text=None,
            )
            y += layer.thickness_um

        # Incident medium (top)
        _add_rect(
            y,
            incident_thickness,
            unique_materials[_get_name(self.incident)],
            label=_get_name(self.incident),
            text=_get_name(self.incident),
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(-substrate_thickness, y + incident_thickness)
        ax.set_ylabel("Thickness (µm)")
        ax.set_xticks([])
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles, strict=False))
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="center left",
            bbox_to_anchor=(1.05, 0.5),
            borderaxespad=0.0,
            ncol=1,
        )
        fig = ax.figure
        return fig, ax

    def plot_structure_thickness(self, ax: plt.Axes = None, max_xticks: int = 10):
        if ax is None:
            fig, ax = plt.subplots()
        import matplotlib.colors as mcolors

        ax.grid(True, alpha=0.3)
        color_cycle = list(mcolors.TABLEAU_COLORS.values())

        def _get_name(obj):
            name = getattr(obj, "name", "") or ""
            if isinstance(obj, IdealMaterial):
                name = f"$n$ = {obj.index[0]}"
            return name

        material_names = [_get_name(layer.material) for layer in self.layers]
        unique_materials = {
            name: color_cycle[i % len(color_cycle)]
            for i, name in enumerate(dict.fromkeys(material_names))
        }
        colors = [unique_materials[_get_name(layer.material)] for layer in self.layers]
        thicknesses_nm = [layer.thickness_um * 1000 for layer in self.layers]
        labels = [layer.name or _get_name(layer.material) for layer in self.layers]

        indices = list(range(len(self.layers)))
        bars = ax.bar(
            indices,
            thicknesses_nm,
            color=colors,
            edgecolor=None,
            alpha=0.7,
            width=1,
        )
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Thickness (nm)")

        # Legend
        by_label = {}
        for bar, label in zip(bars, labels, strict=False):
            if label not in by_label:
                by_label[label] = bar
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="center left",
            bbox_to_anchor=(1.05, 0.5),
            borderaxespad=0.0,
            ncol=1,
        )
        ax.set_xlim(0.5, len(self.layers) - 0.5)
        fig = ax.figure
        return fig, ax


# -------------- Internal TMM core --------------
PolSP = Literal["s", "p"]


def _complex_index(material: BaseMaterial, wavelength_um):
    n = material.n(wavelength_um)
    k = material.k(wavelength_um)
    n = be.atleast_1d(n)
    k = be.atleast_1d(k)
    return be.asarray(n, dtype=be.complex128) + 1j * be.asarray(k, dtype=be.complex128)


def _snell_cos(n0, theta0, n):
    """Transmitted angle cosine with forward-branch selection.
    Calculation is following 'Thin-Film Optical Filters, Fifth Edition, Macleod,
      Hugh Angus CRC Press, Ch2.6
    """
    nr = n.real
    k = n.imag
    return be.sqrt(nr**2 - k**2 - (n0 * be.sin(theta0)) ** 2 - 2j * nr * k) / n


def _admittance(n: complex, cos_t: complex, pol: PolSP):
    """Admittance η = sqrt(ε/μ) * n * cos(θ) for s and p polarizations.
    Calculation is following 'Thin-Film Optical Filters, Fifth Edition, Macleod,
      Hugh Angus CRC Press, Ch2.6
    """
    sqrt_eps_mu = 0.002654418729832701370374020517935  # S
    eta_s = sqrt_eps_mu * n * cos_t

    if pol == "s":
        return eta_s
    elif pol == "p":
        eta_p = sqrt_eps_mu**2 * (n.real - 1j * n.imag) ** 2 / eta_s
        return eta_p
    else:
        raise ValueError("Invalid polarization state")


def _tmm_coh(stack: ThinFilmStack, wavelength_um, theta0_rad, pol: PolSP):
    """
    Compute the reflection and transmission coefficients for
    a thin film stack. Base on Abelès Matrix.

    Ref :
    - Chap 13. Polarized Light and Optical Systems, Russell
    A. Chipman, Wai-Sze Tiffany Lam, and Garam Young
    - F. Abelès, Researches sur la propagation des ondes électromagnétiques
    sinusoïdales dans les milieus stratifies.
    Applications aux couches minces, Ann. Phys. Paris,
    12ième Series 5 (1950): 596–640.
    - Chap 2. Thin-Film Optical Filters, Fifth Edition, Macleod, Hugh Angus CRC Press
    """
    n0 = _complex_index(stack.incident, wavelength_um)
    ns = _complex_index(stack.substrate, wavelength_um)
    cos0 = _snell_cos(n0, theta0_rad, n0)
    coss = _snell_cos(n0, theta0_rad, ns)
    eta0 = _admittance(n0, cos0, pol)
    etas = _admittance(ns, coss, pol)

    # Id initial matrix
    A = be.ones_like(eta0, dtype=be.complex128)
    B = be.zeros_like(eta0, dtype=be.complex128)
    C = be.zeros_like(eta0, dtype=be.complex128)
    D = be.ones_like(eta0, dtype=be.complex128)

    for layer in stack.layers:
        n_l = layer.n_complex(wavelength_um)
        cos_l = _snell_cos(n0, theta0_rad, n_l)
        eta_l = _admittance(n_l, cos_l, pol)
        delta = layer.phase_thickness(wavelength_um, cos_l, n_l)
        c = be.cos(delta)
        s = be.sin(delta)
        i = 1j
        mA = c
        mB = i * (s / eta_l)
        mC = i * (eta_l * s)
        mD = c
        A, B, C, D = A * mA + B * mC, A * mB + B * mD, C * mA + D * mC, C * mB + D * mD

    denom = eta0 * (A + etas * B) + C + etas * D
    denom = be.where(be.abs(denom) == 0, 1e-30 + 0j, denom)

    r = (eta0 * A + eta0 * etas * B - C - etas * D) / denom
    t = be.conj((2 * eta0) / denom)

    R = (r * be.conj(r)).real
    T = (t * be.conj(t)).real * etas.real / eta0.real
    abso = 1 - R - T
    # Absorption can also be obtained with :
    # abso = (1 - R) * (1 - etas.real / ((A + etas * B) * (C + etas * D).conj()).real)
    return r, t, R, T, abso
