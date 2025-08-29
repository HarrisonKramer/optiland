import optiland.backend as be
import numba as nb


# -------------------------------------------------------------------------------------------------------------------- #
# Compute the basis polynomials
# -------------------------------------------------------------------------------------------------------------------- #
@nb.jit(nopython=True, cache=True)
def compute_basis_polynomials(n, p, U, u, return_degree=None):

    """ Evaluate the n-th B-Spline basis polynomials of degree ´p´ for the input u-parametrization

    The basis polynomials are computed from their definition by implementing equation 2.5 directly

    Parameters
    ----------
    n : integer
        Highest index of the basis polynomials (n+1 basis polynomials)

    p : integer
        Degree of the basis polynomials

    U : ndarray with shape (r+1=n+p+2,)
        Knot vector of the basis polynomials
        Set the multiplicity of the first and last entries equal to ´p+1´ to obtain a clamped spline

    u : scalar or ndarray with shape (Nu,)
        Parameter used to evaluate the basis polynomials (real or complex!)

    return_degree : int
        Degree of the returned basis polynomials

    Returns
    -------
    N : ndarray with shape (n+1, Nu)
        Array containing the basis polynomials of order ´p´ evaluated at ´u´
        The first dimension of ´N´ spans the n-th polynomials
        The second dimension of ´N´ spans the ´u´ parametrization sample points

    """

    # # Check the degree of the basis basis polynomials (not possible to compile with Numba yet)
    # if p < 0: raise Exception('The degree of the basis polynomials cannot be negative')
    #
    # # Check the number of basis basis polynomials (not possible to compile with Numba yet)
    # if p > n: raise Exception('The degree of the basis polynomials must be equal or lower than the number of basis polynomials')

    # Number of points where the polynomials are evaluated (vectorized computations)
    u = be.asarray(u * 1.0)     # Convert to array of floats
    Nu = u.size

    # Number of basis polynomials at the current step of the recursion
    m = n + p + 1

    # Initialize the array of basis polynomials
    N = be.zeros((p + 1, m, Nu), dtype=u.dtype)

    # First step of the recursion formula (p = 0)
    # The case when point_index==n and u==1 is an special case. See the NURBS book section 2.5 and algorithm A2.1
    # The be.real() operator is used such that the function extends to complex u-parameter as well
    for i in range(m):
        # N[0, i, :] = 0.0 + 1.0 * (u >= U[i]) * (u < U[i + 1])
        # N[0, i, :] = 0.0 + 1.0 * (u >= U[i]) * (u < U[i + 1]) + 1.00 * (be.logical_and(u == 1, i == n))
        N[0, i, :] = 0.0 + 1.0 * (be.real(u) >= U[i]) * (be.real(u) < U[i + 1]) + 1.00 * (be.logical_and(be.real(u) == U[-1], i == n))

    # Second and next steps of the recursion formula (p = 1, 2, ...)
    for k in range(1, p + 1):

        # Update the number of basis polynomials
        m = m - 1

        # Compute the basis polynomials using the de Boor recursion formula
        for i in range(m):

            # Compute first factor (avoid division by zero by convention)
            if (U[i + k] - U[i]) == 0:
                n1 = be.zeros((Nu,), dtype=u.dtype)
            else:
                n1 = (u - U[i]) / (U[i + k] - U[i]) * N[k - 1, i, :]

            # Compute second factor (avoid division by zero by convention)
            if (U[i + k + 1] - U[i + 1]) == 0:
                n2 = be.zeros((Nu,), dtype=u.dtype)
            else:
                n2 = (U[i + k + 1] - u) / (U[i + k + 1] - U[i + 1]) * N[k - 1, i + 1, :]

            # Compute basis polynomial (recursion formula 2.5)
            N[k, i, ...] = n1 + n2

    # Get the n+1 basis polynomials of the desired degree
    N = N[p, 0:n+1, :] if return_degree is None else N[return_degree, 0:n+1, :]


    # # Numba assertions
    # assert p >= 0    # The degree of the basis polynomials cannot be negative
    # assert p <= n    # The degree of the basis polynomials must be equal or lower than the number of basis polynomials

    return N



# -------------------------------------------------------------------------------------------------------------------- #
# Compute the derivatives of the basis polynomials
# -------------------------------------------------------------------------------------------------------------------- #
@nb.jit(nopython=True, cache=True)
def compute_basis_polynomials_derivatives(n, p, U, u, derivative_order):

    """ Evaluate the derivative of the n-th B-Spline basis polynomials of degree ´p´ for the input u-parametrization

    The basis polynomials derivatives are computed recursively by implementing equations 2.7 and 2.9 directly

    Parameters
    ----------
    n : integer
        Highest index of the basis polynomials (n+1 basis polynomials)

    p : integer
        Degree of the original basis polynomials

    U : ndarray with shape (r+1=n+p+2,)
        Knot vector of the basis polynomials
        Set the multiplicity of the first and last entries equal to ´p+1´ to obtain a clamped spline

    u : scalar or ndarray with shape (Nu,)
        Parameter used to evaluate the basis polynomials

    derivative_order : scalar
        Order of the basis polynomial derivatives

    Returns
    -------
    N_ders : ndarray with shape (n+1, Nu)
        Array containing the basis spline polynomials derivatives evaluated at ´u´
        The first dimension of ´N´ spans the n-th polynomials
        The second dimension of ´N´ spans the ´u´ parametrization sample points

    """

    # Check the number of derivative order (not possible to compile with Numba yet)
    # if derivative_order > p: raise Exception('The derivative order is higher than the degree of the basis polynomials')
    if derivative_order > p: print('The derivative order is higher than the degree of the basis polynomials')

    # Number of points where the polynomials are evaluated (vectorized computations)
    u = be.asarray(u * 1.0)     # Convert to array of floats
    Nu = u.size

    # Compute the basis polynomials to the right hand side of equation 2.9 recursively down to the zeroth derivative
    # Each new call reduces the degree of the basis polynomials by one
    if derivative_order >= 1:
        derivative_order -= 1
        N = compute_basis_polynomials_derivatives(n, p - 1, U, u, derivative_order)
    elif derivative_order == 0:
        N = compute_basis_polynomials(n, p, U, u)
        return N
    else:
        print('Oooopps, something went wrong in compute_basis_polynomials_derivatives()')
        N = compute_basis_polynomials(n, p, U, u)
        return N

    # Initialize the array of basis polynomial derivatives
    N_ders = be.zeros((n + 1, Nu), dtype=u.dtype)

    # Compute the derivatives of the (0, 1, ..., n) basis polynomials using equations 2.7 and 2.9
    for i in range(n + 1):

        # Compute first factor (avoid division by zero by convention)
        if (U[i + p] - U[i]) == 0:
            n1 = be.zeros(Nu, dtype=u.dtype)
        else:
            n1 = p * N[i, :] / (U[i + p] - U[i])

        # Compute second factor (avoid division by zero by convention)
        if (U[i + p + 1] - U[i + 1]) == 0:
            n2 = be.zeros(Nu, dtype=u.dtype)
        else:
            n2 = p * N[i + 1, :] / (U[i + p + 1] - U[i + 1])

        # Compute the derivative of the current basis polynomials
        N_ders[i, :] = n1 - n2

    return N_ders

def basis_function(degree, knot_vector, span, knot):
    """Computes the non-vanishing basis functions for a single parameter.

    Implementation of Algorithm A2.2 from The NURBS Book by Piegl & Tiller.
    Uses recurrence to compute the basis functions, also known as Cox - de
    Boor recursion formula.

    :param degree: degree, :math:`p`
    :type degree: int
    :param knot_vector: knot vector, :math:`U`
    :type knot_vector: list, tuple
    :param span: knot span, :math:`i`
    :type span: int
    :param knot: knot or parameter, :math:`u`
    :type knot: float
    :return: basis functions
    :rtype: list
    """
    left = [0.0 for _ in range(degree + 1)]
    right = [0.0 for _ in range(degree + 1)]
    N = [1.0 for _ in range(degree + 1)]  # N[0] = 1.0 by definition

    for j in range(1, degree + 1):
        left[j] = knot - knot_vector[span + 1 - j]
        right[j] = knot_vector[span + j] - knot
        saved = 0.0
        for r in range(0, j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved

    return N

def basis_function_one(degree, knot_vector, span, knot):
    """Computes the value of a basis function for a single parameter.

    Implementation of Algorithm 2.4 from The NURBS Book by Piegl & Tiller.

    :param degree: degree, :math:`p`
    :type degree: int
    :param knot_vector: knot vector
    :type knot_vector: list, tuple
    :param span: knot span, :math:`i`
    :type span: int
    :param knot: knot or parameter, :math:`u`
    :type knot: float
    :return: basis function, :math:`N_{i,p}`
    :rtype: float
    """
    # Special case at boundaries
    if (
        (span == 0 and knot == knot_vector[0])
        or (span == len(knot_vector) - degree - 2)
        and knot == knot_vector[len(knot_vector) - 1]
    ):
        return 1.0

    # Knot is outside of span range
    if knot < knot_vector[span] or knot >= knot_vector[span + degree + 1]:
        return 0.0

    N = [0.0 for _ in range(degree + span + 1)]

    # Initialize the zeroth degree basis functions
    for j in range(0, degree + 1):
        if knot_vector[span + j] <= knot < knot_vector[span + j + 1]:
            N[j] = 1.0

    # Computing triangular table of basis functions
    for k in range(1, degree + 1):
        # Detecting zeros saves computations
        saved = 0.0
        if N[0] != 0.0:
            saved = ((knot - knot_vector[span]) * N[0]) / (knot_vector[span + k] - knot_vector[span])

        for j in range(0, degree - k + 1):
            Uleft = knot_vector[span + j + 1]
            Uright = knot_vector[span + j + k + 1]

            # Zero detection
            if N[j + 1] == 0.0:
                N[j] = saved
                saved = 0.0
            else:
                temp = N[j + 1] / (Uright - Uleft)
                N[j] = saved + (Uright - knot) * temp
                saved = (knot - Uleft) * temp

    return N[0]
