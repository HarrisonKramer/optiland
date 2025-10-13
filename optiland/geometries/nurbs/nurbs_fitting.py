"""This module defines functions for NURBS surface fitting.

This module defines functions used to find the NURBS parameters that
approximate a given set of points. Most of the code is derived from the
nurbs-geomdl package written by Onur R. Bingol <contact@onurbingol.net>.

Matteo Taccola, 2025
"""

from __future__ import annotations

from scipy.linalg import lu_factor, lu_solve

import optiland.backend as be

from .nurbs_basis_functions import basis_function, basis_function_one


def approximate_surface(points, size_u, size_v, degree_u, degree_v, **kwargs):
    """Approximates a surface using a least-squares method.

    This algorithm interpolates the corner control points and approximates the
    remaining control points. Please refer to Algorithm A9.7 of The NURBS Book
    (2nd Edition), pp.422-423 for details.

    Args:
        points: The data points.
        size_u: The number of data points on the u-direction, :math:`r`.
        size_v: The number of data points on the v-direction, :math:`s`.
        degree_u: The degree of the output surface for the u-direction.
        degree_v: The degree of the output surface for the v-direction.
        **kwargs: Keyword arguments.

    Returns:
        The approximated B-Spline surface.
    """
    use_centripetal = kwargs.get("centripetal", False)
    num_cpts_u = kwargs.get("ctrlpts_size_u", size_u - 1)
    num_cpts_v = kwargs.get("ctrlpts_size_v", size_v - 1)

    dim = len(points[0])

    uk, vl = compute_params_surface(points, size_u, size_v, use_centripetal)

    kv_u = compute_knot_vector(degree_u, size_u, num_cpts_u, uk)
    kv_v = compute_knot_vector(degree_v, size_v, num_cpts_v, vl)

    matrix_nu = []
    for i in range(1, size_u - 1):
        m_temp = []
        for j in range(1, num_cpts_u - 1):
            m_temp.append(basis_function_one(degree_u, kv_u, j, uk[i]))
        matrix_nu.append(m_temp)
    matrix_nu = be.asarray(matrix_nu)
    matrix_ntu = matrix_nu.T
    matrix_ntnu = be.matmul(matrix_ntu, matrix_nu)
    lu, piv = lu_factor(matrix_ntnu)

    ctrlpts_tmp = [[0.0 for _ in range(dim)] for _ in range(num_cpts_u * size_v)]
    for j in range(size_v):
        ctrlpts_tmp[j + (size_v * 0)] = list(points[j + (size_v * 0)])
        ctrlpts_tmp[j + (size_v * (num_cpts_u - 1))] = list(
            points[j + (size_v * (size_u - 1))]
        )
        pt0 = points[j + (size_v * 0)]
        ptm = points[j + (size_v * (size_u - 1))]
        rku = []
        for i in range(1, size_u - 1):
            ptk = points[j + (size_v * i)]
            n0p = basis_function_one(degree_u, kv_u, 0, uk[i])
            nnp = basis_function_one(degree_u, kv_u, num_cpts_u - 1, uk[i])
            elem2 = [c * n0p for c in pt0]
            elem3 = [c * nnp for c in ptm]
            rku.append([a - b - c for a, b, c in zip(ptk, elem2, elem3, strict=False)])
        ru = [[0.0 for _ in range(dim)] for _ in range(num_cpts_u - 2)]
        for i in range(1, num_cpts_u - 1):
            ru_tmp = []
            for idx, pt in enumerate(rku):
                ru_tmp.append(
                    [p * basis_function_one(degree_u, kv_u, i, uk[idx + 1]) for p in pt]
                )
            for d in range(dim):
                for idx in range(len(ru_tmp)):
                    ru[i - 1][d] += ru_tmp[idx][d]
        for d in range(dim):
            b = [pt[d] for pt in ru]
            x = lu_solve((lu, piv), b)
            for i in range(1, num_cpts_u - 1):
                ctrlpts_tmp[j + (size_v * i)][d] = x[i - 1]

    matrix_nv = []
    for i in range(1, size_v - 1):
        m_temp = []
        for j in range(1, num_cpts_v - 1):
            m_temp.append(basis_function_one(degree_v, kv_v, j, vl[i]))
        matrix_nv.append(m_temp)
    matrix_nv = be.asarray(matrix_nv)
    matrix_ntv = matrix_nv.T
    matrix_ntnv = be.matmul(matrix_ntv, matrix_nv)
    lu, piv = lu_factor(matrix_ntnv)

    ctrlpts = [[0.0 for _ in range(dim)] for _ in range(num_cpts_u * num_cpts_v)]
    for i in range(num_cpts_u):
        ctrlpts[0 + (num_cpts_v * i)] = list(ctrlpts_tmp[0 + (size_v * i)])
        ctrlpts[num_cpts_v - 1 + (num_cpts_v * i)] = list(
            ctrlpts_tmp[size_v - 1 + (size_v * i)]
        )
        pt0 = ctrlpts_tmp[0 + (size_v * i)]
        ptm = ctrlpts_tmp[size_v - 1 + (size_v * i)]
        rkv = []
        for j in range(1, size_v - 1):
            ptk = ctrlpts_tmp[j + (size_v * i)]
            n0p = basis_function_one(degree_v, kv_v, 0, vl[j])
            nnp = basis_function_one(degree_v, kv_v, num_cpts_v - 1, vl[j])
            elem2 = [c * n0p for c in pt0]
            elem3 = [c * nnp for c in ptm]
            rkv.append([a - b - c for a, b, c in zip(ptk, elem2, elem3, strict=False)])
        rv = [[0.0 for _ in range(dim)] for _ in range(num_cpts_v - 2)]
        for j in range(1, num_cpts_v - 1):
            rv_tmp = []
            for idx, pt in enumerate(rkv):
                rv_tmp.append(
                    [p * basis_function_one(degree_v, kv_v, j, vl[idx + 1]) for p in pt]
                )
            for d in range(dim):
                for idx in range(len(rv_tmp)):
                    rv[j - 1][d] += rv_tmp[idx][d]
        for d in range(dim):
            b = [pt[d] for pt in rv]
            x = lu_solve((lu, piv), b)
            for j in range(1, num_cpts_v - 1):
                ctrlpts[j + (num_cpts_v * i)][d] = x[j - 1]

    return ctrlpts, degree_u, degree_v, num_cpts_u, num_cpts_v, kv_u, kv_v


def compute_knot_vector(degree, num_dpts, num_cpts, params):
    """Computes a knot vector.

    This function computes a knot vector, ensuring that every knot span has
    at least one :math:`\\overline{u}_{k}`. Please refer to Equations 9.68
    and 9.69 on The NURBS Book (2nd Edition), p.412 for details.

    Args:
        degree: The degree.
        num_dpts: The number of data points.
        num_cpts: The number of control points.
        params: A list of parameters, :math:`\\overline{u}_{k}`.

    Returns:
        The knot vector.
    """
    kv = [0.0 for _ in range(degree + 1)]

    d = float(num_dpts) / float(num_cpts - degree)
    for j in range(1, num_cpts - degree):
        i = int(j * d)
        alpha = (j * d) - i
        temp_kv = ((1.0 - alpha) * params[i - 1]) + (alpha * params[i])
        kv.append(temp_kv)

    kv += [1.0 for _ in range(degree + 1)]

    return kv


def compute_params_curve(points, centripetal=False):
    """Computes :math:`\\overline{u}_{k}` for curves.

    Please refer to Equations 9.4 and 9.5 for chord length
    parametrization, and Equation 9.6 for the centripetal method on The NURBS
    Book (2nd Edition), pp.364-365.

    Args:
        points: The data points.
        centripetal: Activates centripetal parametrization method.

    Returns:
        The parameter array, :math:`\\overline{u}_{k}`.
    """
    if not isinstance(points, list | tuple):
        raise TypeError("Data points must be a list or a tuple")

    num_points = len(points)

    cds = [0.0 for _ in range(num_points + 1)]
    cds[-1] = 1.0
    for i in range(1, num_points):
        distance = be.linalg.norm(be.asarray(points[i]) - be.asarray(points[i - 1]))
        cds[i] = be.sqrt(distance) if centripetal else distance

    d = sum(cds[1:-1])

    uk = [0.0 for _ in range(num_points)]
    for i in range(num_points):
        uk[i] = sum(cds[0 : i + 1]) / d

    return uk


def compute_params_surface(points, size_u, size_v, centripetal=False):
    """Computes :math:`\\overline{u}_{k}` and :math:`\\overline{u}_{l}`.

    The data points array has a row size of ``size_v`` and a column size of
    ``size_u`` and it is 1-dimensional. Please refer to The NURBS Book (2nd
    Edition), pp.366-367 for details on how to compute the parameter arrays
    for global surface interpolation.

    Please note that this function is not a direct implementation of
    Algorithm A9.3, which can be found in The NURBS Book (2nd Edition),
    pp.377-378. However, the output is the same.

    Args:
        points: The data points.
        size_u: The number of points on the u-direction.
        size_v: The number of points on the v-direction.
        centripetal: Activates centripetal parametrization method.

    Returns:
        A tuple of the parameter arrays.
    """
    uk = [0.0 for _ in range(size_u)]

    uk_temp = []
    for v in range(size_v):
        pts_u = [points[v + (size_v * u)] for u in range(size_u)]
        uk_temp += compute_params_curve(pts_u, centripetal)

    for u in range(size_u):
        knots_v = [uk_temp[u + (size_u * v)] for v in range(size_v)]
        uk[u] = sum(knots_v) / size_v

    vl = [0.0 for _ in range(size_v)]

    vl_temp = []
    for u in range(size_u):
        pts_v = [points[v + (size_v * u)] for v in range(size_v)]
        vl_temp += compute_params_curve(pts_v, centripetal)

    for v in range(size_v):
        knots_u = [vl_temp[v + (size_v * u)] for u in range(size_u)]
        vl[v] = sum(knots_u) / size_u

    return uk, vl


def _build_coeff_matrix(degree, knotvector, params, points):
    """Builds the coefficient matrix for global interpolation.

    This function only uses data points to build the coefficient matrix.
    Please refer to The NURBS Book (2nd Edition), pp364-370 for details.

    Args:
        degree: The degree.
        knotvector: The knot vector.
        params: A list of parameters.
        points: The data points.

    Returns:
        The coefficient matrix.
    """
    num_points = len(points)

    matrix_a = [[0.0 for _ in range(num_points)] for _ in range(num_points)]
    for i in range(num_points):
        span = degree + 1
        while span < num_points and knotvector[span] <= params[i]:
            span += 1

        span = span - 1
        matrix_a[i][span - degree : span + 1] = basis_function(
            degree, knotvector, span, params[i]
        )

    return matrix_a
