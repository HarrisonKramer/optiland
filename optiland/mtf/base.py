"""Base Modulation Transfer Function (FFTMTF) Module.

This module contains the abstract base class for MTF calculations
based on the PSF. This includes, e.g., the FFT-based method
and the Huygen-Fresnel-based method.

Kramer Harrison, 2025
"""

import abc

import matplotlib.pyplot as plt

import optiland.backend as be


class BaseMTF(abc.ABC):
    """Base class for MTF computations based on a PSF calculation.

    Attributes:
        optic: The optical system.
        fields: Original field point specification (e.g., "all" or list).
        wavelength: Original wavelength specification (e.g., "primary" or value).
        resolved_fields: List of actual field coordinates (Hx, Hy) to be used.
        resolved_wavelength: Actual wavelength value (in µm) to be used.
    """

    def __init__(
        self,
        optic,
        fields,
        wavelength,
        strategy="chief_ray",
        remove_tilt=False,
        **kwargs,
    ):
        """Initializes BaseMTF and resolves field/wavelength values.

        Args:
            optic: The optical system.
            fields: The field points for MTF calculation. Can be "all" to
                use all fields from the optic, or a list of field coordinates.
            wavelength: The wavelength for MTF calculation. Can be "primary"
                to use the optic's primary wavelength, or a specific
                wavelength value (typically in µm).
            strategy (str): The calculation strategy to use. Supported options are
                "chief_ray" and "centroid_sphere". Defaults to "chief_ray".
            remove_tilt (bool): If True, removes tilt and piston from the OPD data.
                Defaults to False.
            **kwargs: Additional keyword arguments passed to the strategy.
        """
        self.optic = optic
        self.fields = fields
        self.wavelength = wavelength
        self.strategy = strategy
        self.remove_tilt = remove_tilt
        self.strategy_kwargs = kwargs

        if fields == "all":
            self.resolved_fields = optic.fields.get_field_coords()
        else:
            self.resolved_fields = fields

        if wavelength == "primary":
            self.resolved_wavelength = optic.primary_wavelength
        else:
            self.resolved_wavelength = wavelength

        self._calculate_psf()
        self.mtf = self._generate_mtf_data()

    @abc.abstractmethod
    def _generate_mtf_data(self):
        """Generates and returns MTF data."""
        pass

    @abc.abstractmethod
    def _plot_field_mtf(self, ax, field_index, mtf_field_data, color):
        """Plots the MTF data for a single field on the given axes.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axes object.
            field_index (int): The index of the current field.
            mtf_field_data (any): The MTF data for this specific field.
                                Subclasses will define its structure.
            color (str): The color to use for plotting this field.
        """
        pass

    @abc.abstractmethod
    def _calculate_psf(self):
        """Calculates and potentially stores the Point Spread Function."""
        pass

    def view(
        self,
        fig_to_plot_on: plt.Figure = None,
        figsize: tuple[float, float] = (12, 4),
        add_reference: bool = False,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Visualizes the Modulation Transfer Function (MTF).

        This method sets up the plot and iterates through field data,
        calling `_plot_field_mtf` for each field's specific plotting.

        Subclasses must ensure `self.mtf`, `self.freq`, and `self.max_freq`
        are populated before calling this method. `self.resolved_fields`
        (from __init__) is also used.

        Args:
            fig_to_plot_on (plt.Figure, optional): The figure to plot on.
                If None, a new figure will be created. Defaults to None.
            figsize (tuple, optional): The size of the figure.
                Defaults to (12, 4).
            add_reference (bool, optional): Whether to add the diffraction
                limit reference line. Defaults to False.
        Returns:
            tuple: A tuple containing the figure and axes objects.
        """
        is_gui_embedding = fig_to_plot_on is not None

        if is_gui_embedding:
            current_fig = fig_to_plot_on
            current_fig.clear()
            ax = current_fig.add_subplot(111)
        else:
            current_fig, ax = plt.subplots(figsize=figsize)

        for k, field_mtf_item in enumerate(self.mtf):
            self._plot_field_mtf(ax, k, field_mtf_item, color=f"C{k}")

        if add_reference:
            ratio = be.clip(self.freq / self.max_freq, 0.0, 1.0)
            phi = be.arccos(ratio)
            diff_limited_mtf = (2 / be.pi) * (phi - be.cos(phi) * be.sin(phi))

            ax.plot(
                be.to_numpy(self.freq),
                be.to_numpy(diff_limited_mtf),
                "k--",
                label="Diffraction Limit",
            )

        ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
        ax.set_xlim([0, be.to_numpy(self.max_freq)])
        ax.set_ylim([0, 1])
        ax.set_xlabel("Frequency (cycles/mm)", labelpad=10)
        ax.set_ylabel("Modulation", labelpad=10)
        current_fig.tight_layout()
        ax.grid(alpha=0.25)
        if is_gui_embedding and hasattr(current_fig, "canvas"):
            current_fig.canvas.draw_idle()
        return current_fig, ax
