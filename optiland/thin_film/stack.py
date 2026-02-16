"""Thin film optics stack class with inlined TMM.

This class encapsulates both the stack structure (incident/substrate, layers)
and the numerical Transfer Matrix Method (TMM) to compute complex amplitude
coefficients (r, t) and power coefficients (R, T, A) for s, p and unpolarized
cases.

Corentin Nannini, 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import optiland.backend as be
from optiland.materials import IdealMaterial

from .core import _tmm_coh
from .layer import Layer

if TYPE_CHECKING:
    from optiland.materials import BaseMaterial
import re

import matplotlib.pyplot as plt

Pol = Literal["s", "p", "u"]
PlotType = Literal["R", "T", "A"]
Array: TypeAlias = Any  # be.ndarray


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
    reference_wl_um : float | None, optional
        Reference wavelength for thickness quarter-wave calculations, by default None.
    reference_AOI_deg : float | None, optional
        Reference angle of incidence in degrees for thickness quarter-wave
        calculations, by default 0 degrees (normal incidence).

    Examples
    --------
    >>> from optiland.materials import IdealMaterial, Material
    >>> from optiland.thin_film import ThinFilmStack
    >>> air, glass = IdealMaterial(1.0), IdealMaterial(1.52)
    >>> tf = ThinFilmStack(incident_material=air, substrate_material=glass)
    >>> # 100 nm SiO2 on glass
    >>> SiO2 = Material("SiO2", reference="Gao")
    >>> tf.add_layer_nm(SiO2, 100.0)
    >>> R = tf.reflectance_nm_deg([550.0], [0.0], polarization="s")
    >>> T = tf.transmittance_nm_deg([550.0], [0.0], polarization="s")
    >>> A = tf.absorptance_nm_deg([550.0], [0.0], polarization="s")
    """

    incident_material: BaseMaterial
    substrate_material: BaseMaterial
    layers: list[Layer] = field(default_factory=list)
    reference_wl_um: float | None = None
    reference_AOI_deg: float | None = 0

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

    def add_layer_qwot(
        self,
        material: BaseMaterial,
        qwot_thickness: float = 1.0,
        name: str | None = None,
    ) -> ThinFilmStack:
        """Append a quarter-wave optical thickness (QWOT) layer at the reference
        wavelength and angle of incidence.

        Args:
            material: Optiland material providing n(λ), k(λ).
            name: Optional label.

        Raises:
            ValueError: If reference_wl_um is not set.
        """
        if self.reference_wl_um is None:
            raise ValueError("reference_wl_um must be set for adding QWOT layer")
        wl_um = self.reference_wl_um
        th_rad = 0.0
        if self.reference_AOI_deg is not None:
            th_rad = be.deg2rad(self.reference_AOI_deg)
        n = float(be.atleast_1d(material.n(wl_um))[0])  # to ensure scalar float
        thickness_um = qwot_thickness * wl_um / (4 * n * be.cos(th_rad))
        return self.add_layer(thickness_um=thickness_um, material=material, name=name)

    # ----- units helpers -----
    @staticmethod
    def _to_um(wavelength_um_or_nm: float | Array, assume_nm: bool = False):
        arr = be.atleast_1d(wavelength_um_or_nm)
        return arr / 1000.0 if assume_nm else arr

    @staticmethod
    def _deg_to_rad(angle_deg: float | Array):
        return be.atleast_1d(angle_deg) * (be.pi / 180.0)

    # ----- public API: coefficients -----
    def compute_rtRTA(
        self,
        wavelength_um: float | Array,
        aoi_rad: float | Array = 0.0,
        polarization: Pol = "u",
    ) -> dict[str, Any]:
        """Compute complex and power coefficients over λ×θ grids.

        Args:
            wavelength_um: Wavelength(s) in microns (scalar or array). Use helpers
            for nm.
            aoi_rad: Angle(s) of incidence in radians (scalar or array). Use helpers
            for degrees.
            polarization: 's', 'p' or 'u' (unpolarized averages powers of s and p).
                default 'u'.

        Returns:
            Dict with keys 'r','t','R','T','A'. Shapes are (Nλ, Nθ).

        Note:
        - For unpolarized 'u', r, t are s-polarization amplitudes; R, T, A are
        averaged powers.
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

    def compute_rtRAT_nm_deg(
        self,
        wavelength_nm: float | Array,
        aoi_deg: float | Array = 0.0,
        polarization: Pol = "u",
    ) -> dict[str, float | Array]:
        """Same as coefficients() but inputs in nm and degrees."""
        wl_um = self._to_um(wavelength_nm, assume_nm=True)
        th_rad = self._deg_to_rad(aoi_deg)
        return self.compute_rtRTA(wl_um, th_rad, polarization)

    # ----- convenience getters -----
    def reflectance(
        self,
        wavelength_um: float | Array,
        aoi_rad: float | Array = 0.0,
        polarization: Pol = "u",
    ) -> Array:
        return self.compute_rtRTA(wavelength_um, aoi_rad, polarization)["R"]

    def transmittance(
        self,
        wavelength_um: float | Array,
        aoi_rad: float | Array = 0.0,
        polarization: Pol = "u",
    ) -> Array:
        return self.compute_rtRTA(wavelength_um, aoi_rad, polarization)["T"]

    def absorptance(
        self,
        wavelength_um: float | Array,
        aoi_rad: float | Array = 0.0,
        polarization: Pol = "u",
    ) -> Array:
        return self.compute_rtRTA(wavelength_um, aoi_rad, polarization)["A"]

    def reflectance_nm_deg(
        self,
        wavelength_nm: float | Array,
        aoi_deg: float | Array = 0.0,
        polarization: Pol = "u",
    ) -> Array:
        return self.compute_rtRAT_nm_deg(wavelength_nm, aoi_deg, polarization)["R"]

    def transmittance_nm_deg(
        self,
        wavelength_nm: float | Array,
        aoi_deg: float | Array = 0.0,
        polarization: Pol = "u",
    ) -> Array:
        return self.compute_rtRAT_nm_deg(wavelength_nm, aoi_deg, polarization)["T"]

    def absorptance_nm_deg(
        self,
        wavelength_nm: float | Array,
        aoi_deg: float | Array = 0.0,
        polarization: Pol = "u",
    ) -> Array:
        return self.compute_rtRAT_nm_deg(wavelength_nm, aoi_deg, polarization)["A"]

    def RTA(
        self,
        wavelength_um: float | Array,
        aoi_rad: float | Array = 0.0,
        polarization: Pol = "u",
    ) -> tuple[Array, Array, Array]:
        """Return (R, T, A) for given wavelength(s) in µm and AOI(s) in radians."""
        rta_data = self.compute_rtRTA(wavelength_um, aoi_rad, polarization)
        return (
            rta_data["R"],
            rta_data["T"],
            rta_data["A"],
        )

    def RTA_nm_deg(
        self,
        wavelength_nm: float | Array,
        aoi_deg: float | Array = 0.0,
        polarization: Pol = "u",
    ) -> tuple[Array, Array, Array]:
        """Return (R, T, A) for given wavelength(s) in nm and AOI(s) in degrees."""
        rta_data = self.compute_rtRAT_nm_deg(wavelength_nm, aoi_deg, polarization)
        return (
            rta_data["R"],
            rta_data["T"],
            rta_data["A"],
        )

    def __len__(self):
        return len(self.layers)

    def __repr__(self):
        parts = [layer.name or f"Layer({i})" for i, layer in enumerate(self.layers)]
        return f"ThinFilmStack({len(self.layers)} layers: " + " -> ".join(parts) + ")"

    def plot_structure(self, ax: plt.Axes = None) -> tuple[plt.Figure, plt.Axes]:
        """Plots a schematic representation of the thin film stack structure.
        This method visualizes the stack as a series of colored rectangles, each
        representing a material layer, the substrate, and the incident medium.
        Each rectangle's height corresponds to the physical thickness of the
        layer (in micrometers), and colors are assigned uniquely to each material.
        The substrate is plotted at the bottom, followed by the stack layers,
        and the incident medium at the top. Material names or refractive indices
        are used as labels in the legend.
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot the structure. If None, a new figure and
            axes are created.
        Returns
        -------
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
            The matplotlib Figure and Axes objects containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        import matplotlib.colors as mcolors

        color_cycle = list(mcolors.TABLEAU_COLORS.values())

        def _get_name(obj):
            """
            Get the name of a material or layer, or its refractive index if it's an
            IdealMaterial. Because IdealMaterial may not have a name, we use its
            refractive index for labeling.
            """
            name = getattr(obj, "name", "") or ""
            if isinstance(obj, IdealMaterial):
                name = f"$n$ = {obj.index[0]}"
            return name

        def _add_rect(y, height, color, label, text=None):
            ax.add_patch(
                plt.Rectangle((0, y), 1, height, color=color, label=label, alpha=0.7)
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
            [_get_name(self.incident_material)]
            + [_get_name(layer.material) for layer in self.layers]
            + [_get_name(self.substrate_material)]
        )
        unique_materials = {
            name: color_cycle[i % len(color_cycle)]
            for i, name in enumerate(dict.fromkeys(material_names))
        }
        total_layer_thickness = sum(layer.thickness_um for layer in self.layers)

        # Ensure minimum thickness for visualization (avoid singular ylim
        # on empty stacks)
        if total_layer_thickness == 0:
            total_layer_thickness = 1.0

        incident_thickness = 0.08 * total_layer_thickness
        substrate_thickness = 0.08 * total_layer_thickness
        y = -substrate_thickness

        # Substrate (bottom, negative y)
        _add_rect(
            y,
            substrate_thickness,
            unique_materials[_get_name(self.substrate_material)],
            label=_get_name(self.substrate_material),
            text=_get_name(self.substrate_material),
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
            unique_materials[_get_name(self.incident_material)],
            label=_get_name(self.incident_material),
            text=_get_name(self.incident_material),
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

    def plot_structure_thickness(
        self, ax: plt.Axes = None
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plots the thickness of each layer in the thin film stack as a bar chart.
        Each bar represents a layer, with its height corresponding to the layer's
        thickness in nanometers.
        Bars are colored according to the material of each layer, and a legend is
        provided to identify materials.
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The matplotlib Axes object to plot on. If None, a new figure and axes
            will be created.
        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib Figure object containing the plot.
        ax : matplotlib.axes.Axes
            The matplotlib Axes object containing the plot.
        """

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
