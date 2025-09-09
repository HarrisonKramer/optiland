from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import optiland.backend as be
from optiland.materials import IdealMaterial

if TYPE_CHECKING:
    from optiland.materials import BaseMaterial
import re

import matplotlib.pyplot as plt

from .layer import Layer

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

        Returns:
            self for chaining.
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

    def coefficients_nm_deg(
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
        return self.coefficients_nm_deg(wavelength_nm, aoi_deg, polarization)["R"]

    def transmittance_nm_deg(
        self,
        wavelength_nm: float | Array,
        aoi_deg: float | Array = 0.0,
        polarization: Pol = "u",
    ) -> Array:
        return self.coefficients_nm_deg(wavelength_nm, aoi_deg, polarization)["T"]

    def absorptance_nm_deg(
        self,
        wavelength_nm: float | Array,
        aoi_deg: float | Array = 0.0,
        polarization: Pol = "u",
    ) -> Array:
        return self.coefficients_nm_deg(wavelength_nm, aoi_deg, polarization)["A"]

    def rtRTA(
        self,
        wavelength_um: float | Array,
        aoi_rad: float | Array = 0.0,
        polarization: Pol = "u",
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Return (r, t, R, T, A) for given wavelength(s) in µm and AOI(s)
        in radians."""
        rta_data = self.compute_rtRTA(wavelength_um, aoi_rad, polarization)
        return (
            rta_data["r"],
            rta_data["t"],
            rta_data["R"],
            rta_data["T"],
            rta_data["A"],
        )

    def rtRTA_nm_deg(
        self,
        wavelength_nm: float | Array,
        aoi_deg: float | Array = 0.0,
        polarization: Pol = "u",
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Return (r, t, R, T, A) for given wavelength(s) in nm and
        AOI(s) in degrees."""
        rta_data = self.coefficients_nm_deg(wavelength_nm, aoi_deg, polarization)
        return (
            rta_data["r"],
            rta_data["t"],
            rta_data["R"],
            rta_data["T"],
            rta_data["A"],
        )

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
        rta_data = self.coefficients_nm_deg(wavelength_nm, aoi_deg, polarization)
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

    def plot(
        self,
        wavelength_um: float | Array,
        aoi_deg: float | Array = 0.0,
        polarization: Pol = "u",
        to_plot: PlotType | list[PlotType] = "R",
        ax: plt.Axes = None,
    ) -> plt.Figure:
        """Plot R/T/A vs wavelength and/or AOI for given polarization.
        Args:
            wavelength_um: Wavelength(s) in micrometers (scalar or array).
            aoi_deg: Angle(s) of incidence in degrees (scalar or array), default 0.
            polarization: 's', 'p' or 'u' (unpolarized averages powers of s and p),
            default 'u'.
            to_plot: 'R', 'T', 'A' or list of these, default 'R'.
            ax: Optional matplotlib Axes to plot on. If None, a new figure
            and axes are created.
        Returns:
            fig: The matplotlib Figure object containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()

        wl_array = be.atleast_1d(wavelength_um) * 1000.0  # convert to nm
        aoi_array = be.atleast_1d(aoi_deg)

        rta_data = self.coefficients_nm_deg(wl_array, aoi_array, polarization)

        if isinstance(to_plot, str):
            to_plot = [to_plot]

        # Case 1: wavelength is array, AOI is scalar
        if len(wl_array) > 1 and len(aoi_array) == 1:
            for quantity in to_plot:
                if quantity not in ("R", "T", "A"):
                    raise ValueError("to_plot must be 'R', 'T', 'A' or a list of these")
                ax.plot(
                    wl_array,
                    rta_data[quantity].flatten(),
                    label=f"{quantity}, {polarization}-pol, AOI={aoi_deg}°",
                )
            ax.set_xlabel("$\lambda$ (nm)")
            ax.set_ylabel("Power fraction")
            ax.set_xlim(wl_array.min(), wl_array.max())
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Case 2: AOI is array, wavelength is scalar
        elif len(aoi_array) > 1 and len(wl_array) == 1:
            for quantity in to_plot:
                if quantity not in ("R", "T", "A"):
                    raise ValueError("to_plot must be 'R', 'T', 'A' or a list of these")
                ax.plot(
                    aoi_array,
                    rta_data[quantity].flatten(),
                    label=f"{quantity}, {polarization}-pol, λ={wavelength_um}nm",
                )
            ax.set_xlabel("AOI (°)")
            ax.set_ylabel("Power fraction")
            ax.set_xlim(aoi_array.min(), aoi_array.max())
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend()
        # Case 3: Both are arrays - 2D plot using pcolormesh
        elif len(wl_array) > 1 and len(aoi_array) > 1:
            if isinstance(to_plot, str):
                to_plot = [to_plot]

            WL, AOI = be.meshgrid(wl_array, aoi_array, indexing="ij")

            # If multiple quantities requested, create one subplot per quantity
            if len(to_plot) > 1:
                fig, axs = plt.subplots(len(to_plot), 1, figsize=(6, 4 * len(to_plot)))
                if len(to_plot) == 1:
                    axs = [axs]
                for ax_idx, quantity in enumerate(to_plot):
                    if quantity not in ("R", "T", "A"):
                        raise ValueError(
                            "to_plot must be 'R', 'T', 'A' or a list of these"
                        )
                    ax_i = axs[ax_idx]
                    im = ax_i.pcolormesh(WL, AOI, rta_data[quantity], shading="auto")
                    ax_i.set_xlabel("$\\lambda$ (nm)")
                    ax_i.set_ylabel("AOI (°)")
                    ax_i.set_title(f"{quantity}, {polarization}-pol")
                    fig.colorbar(im, ax=ax_i, label="Power fraction")
            else:
                # Single quantity: honor provided ax or create one
                quantity = to_plot[0]
                if quantity not in ("R", "T", "A"):
                    raise ValueError("to_plot must be 'R', 'T', 'A' or a list of these")
                if ax is None:
                    fig, ax = plt.subplots()
                im = ax.pcolormesh(WL, AOI, rta_data[quantity], shading="auto")
                ax.set_xlabel("$\\lambda$ (nm)")
                ax.set_ylabel("AOI (°)")
                ax.set_title(f"{quantity}, {polarization}-pol")
                fig.colorbar(im, ax=ax, label="Power fraction")

            # Ensure fig is defined for return
            if "fig" not in locals():
                fig = ax.figure
            return fig

        # Case 4: Both are scalars - single point plot
        else:
            raise ValueError(
                "At least one of wavelength_nm or aoi_deg must be an array"
            )


# -------------- Internal TMM core --------------
PolSP = Literal["s", "p"]


def _complex_index(material: BaseMaterial, wavelength_um: float | Array) -> Array:
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
    n0 = _complex_index(stack.incident_material, wavelength_um)
    ns = _complex_index(stack.substrate_material, wavelength_um)
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
