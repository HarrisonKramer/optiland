"""
PSF Utilities Module

This module contains utility functions for processing point spread function (PSF) data.
"""

import optiland.backend as be


def remove_tilt(x, y, opd, weights, remove_piston=False, ridge=1e-12):
    """
    Removes piston and tilt from 1D OPD data using weighted least squares.

    Args:
        x (array-like): Exit pupil x coordinates, shape (N,).
        y (array-like): Exit pupil y coordinates, shape (N,).
        opd (array-like): OPD values, shape (N,).
        weights (array-like): Weights per point, shape (N,). Zero weight masks a point.
        remove_piston (bool, optional): If True, removes piston term as well as tilt.
            Defaults to False.
        ridge (float, optional): Small diagonal regularization for stability.
            Defaults to 1e-12.

    Returns:
        opd_detrended (array-like): OPD with piston and tilt removed, shape (N,).
        coeffs (array-like): [piston, tilt_x, tilt_y] coefficients, shape (3,).
    """
    # weighted design matrix
    one = be.ones_like(x)
    X = be.stack([one, x, y], axis=1)  # (N,3)

    # apply sqrt(weights) to each column
    W = be.sqrt(weights)[:, None]
    Xw = X * W
    yw = opd * be.sqrt(weights)

    XT_X = be.matmul(Xw.T, Xw) + ridge * be.eye(3, dtype=opd.dtype)
    XT_y = be.matmul(Xw.T, yw)

    # solve for coefficients
    coeffs = be.linalg.solve(XT_X, XT_y)

    if not remove_piston:
        coeffs = coeffs.clone() if hasattr(coeffs, "clone") else coeffs.copy()
        coeffs[0] = 0.0

    # subtract fitted plane
    fitted = X @ coeffs
    opd_detrended = opd - fitted

    return opd_detrended
