"""Abbe Material

This module defines a material based on the refractive index at the Fraunhofer
d-line (587.56 nm) and the Abbe number. It provides two models for calculating
the refractive index:
1.  "polynomial": A polynomial fit to glass data (Legacy).
2.  "buchdahl": A Buchdahl 3-term model with LASSO-derived coefficients (New).

Kramer Harrison, 2024
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from importlib import resources

import optiland.backend as be
from optiland.materials.base import BaseMaterial


class AbbeModel(ABC):
    """Abstract base class for Abbe number based material models."""

    @abstractmethod
    def predict_n(self, wavelength: float | be.ndarray) -> float | be.ndarray:
        """Predicts the refractive index at a given wavelength."""
        pass

    @abstractmethod
    def predict_k(self, wavelength: float | be.ndarray) -> float | be.ndarray:
        """Predicts the extinction coefficient at a given wavelength."""
        pass


class AbbePolynomialModel(AbbeModel):
    """Legacy polynomial model for Abbe materials.

    This model uses a polynomial fit to glass data from the Schott catalog.
    It corresponds to the original implementation of AbbeMaterial.
    """

    def __init__(self, index: float, abbe: float):
        self.index = be.array([index])
        self.abbe = be.array([abbe])
        self._p = self._get_coefficients()

    def predict_n(self, wavelength: float | be.ndarray) -> float | be.ndarray:
        wavelength = be.array(wavelength)
        if be.any(wavelength < 0.380) or be.any(wavelength > 0.750):
            # This legacy check is preserved
            raise ValueError("Wavelength out of range for this model.")
        return be.atleast_1d(be.polyval(self._p, wavelength))

    def predict_k(self, wavelength: float | be.ndarray) -> float | be.ndarray:
        return be.zeros_like(0)

    def _get_coefficients(self):
        # Polynomial fit to the refractive index data
        X_poly = be.ravel(
            be.array(
                [
                    self.index,
                    self.abbe,
                    self.index**2,
                    self.abbe**2,
                    self.index**3,
                    self.abbe**3,
                ]
            )
        )

        coefficients_file = str(
            resources.files("optiland.database").joinpath(
                "glass_model_coefficients.npy",
            ),
        )
        coefficients = be.load(coefficients_file)
        return be.matmul(X_poly, coefficients)


class AbbeBuchdahlModel(AbbeModel):
    """Buchdahl 3-term model with LASSO-derived coefficients.

    This model predicts refractive index using the Buchdahl coordinate $\omega$:
    $$ \omega = \frac{\lambda - \lambda_D}{1 + \alpha(\lambda - \lambda_D)} $$
    $$ n(\lambda) = n_d + v_1 \omega + v_2 \omega^2 + v_3 \omega^3 $$

    where $\lambda_D = 0.5875618$ $\mu m$ and $\alpha = 2.5$.
    The coefficients $v_1, v_2, v_3$ are predicted from $n_d$ and $V_d$ using a
    linear regression model derived via LASSO.
    """

    # Buchdahl constants
    WAVE_D = 0.5875618
    ALPHA = 2.5

    # Coefficients for v1 prediction
    # Features: [1, 1/V, 1/V^2, n, n^2, n/V]
    # v1 = c0 + c1/V + c2/V^2 + c3*n + c4*n^2 + c5*n/V
    V1_COEFFS = [
        0.004160,  # Intercept
        4.462559,  # 1/V
        2.326660,  # 1/V^2
        0.002330,  # n
        -0.003697,  # n^2
        -4.697604,  # n/V
    ]

    # Coefficients for v2 prediction
    # Features: [1, 1/V, 1/V^2, n^2, n/V] (Note: 'n' term was zeroed out by LASSO)
    # v2 = c0 + c1/V + c2/V^2 + c3*n^2 + c4*n/V
    V2_COEFFS = [
        0.066434,  # Intercept
        -7.636396,  # 1/V
        12.597434,  # 1/V^2
        -0.037014,  # n^2
        5.551013,  # n/V
    ]

    # Coefficients for v3 prediction
    # Features: [1, 1/V, 1/V^2, n^2, n/V] (Note: 'n' term was zeroed out by LASSO)
    # v3 = c0 + c1/V + c2/V^2 + c3*n^2 + c4*n/V
    V3_COEFFS = [
        -0.032218,  # Intercept
        2.230357,  # 1/V
        -103.318994,  # 1/V^2
        -0.009654,  # n^2
        1.934983,  # n/V
    ]

    def __init__(self, index: float, abbe: float):
        self.nd = float(index)
        self.vd = float(abbe)
        self.v1, self.v2, self.v3 = self._calculate_buchdahl_coefficients()

    def _calculate_buchdahl_coefficients(self):
        nd = self.nd
        vd = self.vd
        inv_v = 1.0 / vd
        inv_v2 = 1.0 / (vd**2)
        nd_sq = nd**2
        nd_div_v = nd / vd

        # Calculate v1
        # Terms: [1, 1/V, 1/V^2, n, n^2, n/V]
        c = self.V1_COEFFS
        v1 = (
            c[0]
            + c[1] * inv_v
            + c[2] * inv_v2
            + c[3] * nd
            + c[4] * nd_sq
            + c[5] * nd_div_v
        )

        # Calculate v2
        # Terms: [1, 1/V, 1/V^2, n^2, n/V]
        c = self.V2_COEFFS
        v2 = c[0] + c[1] * inv_v + c[2] * inv_v2 + c[3] * nd_sq + c[4] * nd_div_v

        # Calculate v3
        # Terms: [1, 1/V, 1/V^2, n^2, n/V]
        c = self.V3_COEFFS
        v3 = c[0] + c[1] * inv_v + c[2] * inv_v2 + c[3] * nd_sq + c[4] * nd_div_v

        return v1, v2, v3

    def predict_n(self, wavelength: float | be.ndarray) -> float | be.ndarray:
        wavelength = be.array(wavelength)

        # Calculate Buchdahl coordinate omega
        # omega = (lambda - lambda_d) / (1 + alpha * (lambda - lambda_d))
        d_lambda = wavelength - self.WAVE_D
        omega = d_lambda / (1 + self.ALPHA * d_lambda)

        # Buchdahl polynomial: n = nd + v1*w + v2*w^2 + v3*w^3
        n_pred = self.nd + self.v1 * omega + self.v2 * (omega**2) + self.v3 * (omega**3)

        return be.atleast_1d(n_pred)

    def predict_k(self, wavelength: float | be.ndarray) -> float | be.ndarray:
        return be.zeros_like(0)


class AbbeMaterial(BaseMaterial):
    """Represents a material based on the refractive index and Abbe number.

    This class serves as a wrapper around specific model implementations.
    Currently supported models:
    - 'polynomial': The legacy polynomial fit (Default, deprecated).
    - 'buchdahl': The new Buchdahl 3-term model (Recommended).

    Args:
        n (float): The refractive index of the material at 587.56 nm (n_d).
        abbe (float): The Abbe number of the material (V_d).
        model (str, optional): The model to use. Defaults to "polynomial".
            Valid options are "polynomial" and "buchdahl".

    Attributes:
        index (float): The refractive index of the material at 587.56 nm.
        abbe (float): The Abbe number of the material.
        model_name (str): The name of the model being used.
        model (AbbeModel): The underlying model instance.

    """

    def __init__(self, n, abbe, model=None):
        super().__init__()
        self.index = n
        self.abbe = abbe

        if model is None:
            warnings.warn(
                "The default model for AbbeMaterial will change from 'polynomial' "
                "to 'buchdahl' in v0.7.0. The 'buchdahl' model offers improved "
                "accuracy. To silence this warning, specify `model='polynomial'` "
                "explicitly if you intended to use the legacy model, or switch to "
                "`model='buchdahl'`.",
                FutureWarning,
                stacklevel=2,
            )
            model = "polynomial"

        self.model_name = model

        if model == "polynomial":
            self.model = AbbePolynomialModel(n, abbe)
        elif model == "buchdahl":
            self.model = AbbeBuchdahlModel(n, abbe)
        else:
            raise ValueError(
                f"Unknown model: {model}. Valid options: 'polynomial', 'buchdahl'"
            )

    def _calculate_n(self, wavelength, **kwargs):
        """Returns the refractive index of the material."""
        return self.model.predict_n(wavelength)

    def _calculate_k(self, wavelength, **kwargs):
        """Returns the extinction coefficient of the material."""
        return self.model.predict_k(wavelength)

    def to_dict(self):
        """Returns a dictionary representation of the material."""
        material_dict = super().to_dict()
        material_dict.update(
            {
                "index": float(self.index),
                "abbe": float(self.abbe),
                "model": self.model_name,
            }
        )
        return material_dict

    @classmethod
    def from_dict(cls, data):
        """Creates a material from a dictionary representation."""
        required_keys = ["index", "abbe"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")

        model = data.get("model")
        return cls(data["index"], data["abbe"], model=model)
