"""
Metrics Module for Optiland Reporting.

This module provides stateless service classes responsible for calculating
numerical performance data for optical systems. It adheres to SOLID principles,
separating calculation logic from visualization.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.analysis.spot_diagram import SpotDiagram
from optiland.wavefront.wavefront import Wavefront

if TYPE_CHECKING:
    from optiland.optic import Optic


class FirstOrderMetrics:
    """Calculates first-order properties of the optical system."""

    @staticmethod
    def calculate(optic: Optic) -> dict[str, Any]:
        """Calculates all first-order metrics.

        Args:
            optic: The optical system to analyze.

        Returns:
            A dictionary containing EFL, F/#, EPD, Magnification, etc.
        """
        paraxial = optic.paraxial
        metrics = {}

        try:
            metrics["EFL"] = float(be.to_numpy(paraxial.f2()))
        except Exception:
            metrics["EFL"] = float("nan")

        try:
            metrics["F/#"] = float(be.to_numpy(paraxial.FNO()))
        except Exception:
            metrics["F/#"] = float("nan")

        try:
            metrics["EPD"] = float(be.to_numpy(paraxial.EPD()))
        except Exception:
            metrics["EPD"] = float("nan")

        try:
            metrics["EPL"] = float(be.to_numpy(paraxial.EPL()))
        except Exception:
            metrics["EPL"] = float("nan")

        try:
            metrics["XPD"] = float(be.to_numpy(paraxial.XPD()))
        except Exception:
            metrics["XPD"] = float("nan")

        try:
            metrics["XPL"] = float(be.to_numpy(paraxial.XPL()))
        except Exception:
            metrics["XPL"] = float("nan")

        try:
            metrics["Magnification"] = float(
                be.to_numpy(paraxial.magnification())
            )
        except Exception:
            metrics["Magnification"] = float("nan")

        metrics["Total Track"] = float(optic.total_track)

        return metrics


class ImageQualityMetrics:
    """Calculates image quality metrics based on spot diagrams."""

    @staticmethod
    def calculate(
        optic: Optic, fields: str | list = "all", wavelength="primary"
    ) -> dict[str, Any]:
        """Calculates spot diagram metrics.

        Args:
            optic: The optical system.
            fields: Fields to analyze.
            wavelength: Wavelength to analyze.

        Returns:
            Dictionary with RMS Radius, Geometric Radius, Centroid for each field.
        """
        wls = "primary" if wavelength == "primary" else [wavelength]
        spot = SpotDiagram(optic, fields=fields, wavelengths=wls)
        results = {}

        rms_radii = spot.rms_spot_radius()
        geo_radii = spot.geometric_spot_radius()
        centroids = spot.centroid()

        for i, field in enumerate(spot.fields):
            # rms_radii is list of lists [field][wavelength]
            # We assume single wavelength analysis for metrics report per call
            # or we take the first one if multiple are implicitly present
            rms = rms_radii[i][0]
            geo = geo_radii[i][0]
            cx, cy = centroids[i]

            results[f"Field {i+1}"] = {
                "Field Coordinate": field,
                "RMS Radius": float(be.to_numpy(rms)),
                "Geo Radius": float(be.to_numpy(geo)),
                "Centroid X": float(be.to_numpy(cx)),
                "Centroid Y": float(be.to_numpy(cy)),
            }

        return results


class WavefrontMetrics:
    """Calculates wavefront error metrics."""

    @staticmethod
    def calculate(
        optic: Optic, fields: str | list = "all", wavelength="primary"
    ) -> dict[str, Any]:
        """Calculates wavefront metrics.

        Args:
            optic: The optical system.
            fields: Fields to analyze.
            wavelength: Wavelength to analyze.

        Returns:
            Dictionary with RMS Wavefront Error, Strehl Ratio, P-V.
        """
        wls = "primary" if wavelength == "primary" else [wavelength]
        wf = Wavefront(optic, fields=fields, wavelengths=wls)
        results = {}

        # wf.data is dict keyed by (field, wl) -> WavefrontData

        for key, data in wf.data.items():
            field, _ = key
            opd = data.opd

            # Remove NaN values (vignetted rays)
            opd = opd[~be.isnan(opd)]

            if opd.size == 0:
                metrics = {
                    "RMS WFE": float("nan"),
                    "P-V WFE": float("nan"),
                    "Strehl": 0.0,
                }
            else:
                rms = be.std(opd)
                pv = be.max(opd) - be.min(opd)
                # Maréchal approximation
                strehl = be.exp(-(2 * be.pi * rms) ** 2)

                metrics = {
                    "RMS WFE": float(be.to_numpy(rms)),
                    "P-V WFE": float(be.to_numpy(pv)),
                    "Strehl": float(be.to_numpy(strehl)),
                }

            results[str(field)] = metrics

        return results


class SeidelMetrics:
    """Calculates Seidel aberration coefficients."""

    @staticmethod
    def calculate(optic: Optic) -> dict[str, float]:
        """Calculates Summed Seidel Coefficients.

        Args:
            optic: The optical system.

        Returns:
            Dictionary of Seidel coefficients.
        """
        try:
            # helper to get sum of seidels
            S = optic.aberrations.seidels()
            # S is array [Spherical, Coma, Astigmatism, Petzval, Distortion]

            return {
                "Spherical (S1)": float(be.to_numpy(S[0])),
                "Coma (S2)": float(be.to_numpy(S[1])),
                "Astigmatism (S3)": float(be.to_numpy(S[2])),
                "Petzval (S4)": float(be.to_numpy(S[3])),
                "Distortion (S5)": float(be.to_numpy(S[4])),
            }
        except Exception:
            return {
                "Spherical (S1)": float("nan"),
                "Coma (S2)": float("nan"),
                "Astigmatism (S3)": float("nan"),
                "Petzval (S4)": float("nan"),
                "Distortion (S5)": float("nan"),
            }


class MaterialMetrics:
    """Summarizes material usage in the system."""

    @staticmethod
    def calculate(optic: Optic) -> list[dict[str, Any]]:
        """Extracts material info for each surface.

        Args:
            optic: The optical system.

        Returns:
            List of dictionaries describing materials.
        """
        materials = []
        for i, surface in enumerate(optic.surface_group.surfaces):
            # Skip image surface if needed, but it might have material property
            mat_name = "Air"
            if hasattr(surface, "material_post"):
                # Check if it is really a material object
                if hasattr(surface.material_post, "name"):
                    mat_name = surface.material_post.name
                else:
                    mat_name = str(surface.material_post)

            n_val = 1.0
            vd_val = float("nan")

            if hasattr(surface, "material_post"):
                with contextlib.suppress(Exception):
                    n_val = float(
                        be.to_numpy(
                            surface.material_post.n(optic.primary_wavelength)
                        )
                    )

                if hasattr(surface.material_post, "Vd"):
                    with contextlib.suppress(Exception):
                        vd_val = float(be.to_numpy(surface.material_post.Vd))

            materials.append({
                "Surface": i,
                "Material": mat_name,
                "Refractive Index": n_val,
                "Abbe Number": vd_val,
            })
        return materials
