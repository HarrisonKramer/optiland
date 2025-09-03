""" Polynomial regression model """
from __future__ import annotations

from itertools import combinations_with_replacement

from numpy.typing import NDArray

import optiland.backend as be


class PolynomialRegression:
    """
    A simple polynomial regression model.

    This model fits a polynomial of a specified degree to the input data (X)
    to predict the output data (y). It supports multi-output regression by
    fitting an independent model for each output dimension.

    Parameters
    ----------
    degree : int, optional
        The degree of the polynomial to fit, by default 2.
    """

    def __init__(self, degree: int = 2):
        self.degree = degree
        self.weights = None  # This will be a backend tensor

    def _generate_features(self, X: NDArray) -> NDArray:
        n_samples, n_features = X.shape

        features = [be.ones((n_samples, 1))]

        for d in range(1, self.degree + 1):
            for items in combinations_with_replacement(range(n_features), d):
                p = X[:, items[0] : items[0] + 1]
                for i in range(1, len(items)):
                    p = p * X[:, items[i] : items[i] + 1]
                features.append(p)

        return be.concatenate(features, axis=1)

    def fit(self, X: NDArray, y: NDArray):
        """
        Fit the polynomial regression model to the data using a direct solver.
        """
        X_poly = self._generate_features(X)

        # w = (X^T X)^-1 X^T y
        XTX = be.matmul(be.transpose(X_poly), X_poly)

        # Add a small regularization term to avoid singularity
        reg = be.eye(XTX.shape[0]) * 1e-9

        XTX_inv = be.linalg.inv(XTX + reg)
        XTy = be.matmul(be.transpose(X_poly), y)

        self.weights = be.matmul(XTX_inv, XTy)

    def predict(self, X: NDArray) -> NDArray:
        """
        Predict using the fitted polynomial regression model.

        Parameters
        ----------
        X : NDArray
            The input samples for which to predict, shape (n_samples, n_features).

        Returns
        -------
        NDArray
            The predicted values, shape (n_samples, n_outputs).
        """
        if self.weights is None:
            raise RuntimeError("The model must be fitted before prediction.")

        X_poly = self._generate_features(X)

        predictions = be.matmul(X_poly, self.weights)

        return predictions
