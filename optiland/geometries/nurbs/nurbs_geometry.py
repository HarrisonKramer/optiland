"""NURBS for Optical Surfaces

This module provides implementations of NURBS for optical surfaces.
The geometric implementation of the surface is mostly based on the work of Roberto Agromayor
(https://github.com/turbo-sim/nurbspy)
The surface fitting is mostly based on the work of Onur R. Bingol
(https://github.com/orbingol/NURBS-Python)
The ray intersection with the NURBS and the computation of the sag are based on my understanding 
of the algorithms described in the paper 'Practical ray tracing of trimmed nurbs surfaces'
  
Matteo Taccola, 2025
"""

import optiland.backend as be
from optiland.geometries.base import BaseGeometry
from scipy.special import binom
from .nurbs_basis_functions  import compute_basis_polynomials, compute_basis_polynomials_derivatives
from .nurbs_fitting import approximate_surface

class NurbsGeometry(BaseGeometry):

    """ 
    Create a NURBS (Non-Uniform Rational Basis Spline) geometry

    Parameters
    ----------
    control_points : ndarray with shape (ndim, n+1, m+1)
        Array containing the coordinates of the control points
        The first dimension of ´P´ spans the coordinates of the control points (any number of dimensions)
        The second dimension of ´P´ spans the u-direction control points (0, 1, ..., n)
        The third dimension of ´P´ spans the v-direction control points (0, 1, ..., m)

    weights : ndarray with shape (n+1, m+1)
        Array containing the weight of the control points
        The first dimension of ´W´ spans the u-direction control points weights (0, 1, ..., n)
        The second dimension of ´W´ spans the v-direction control points weights (0, 1, ..., m)

    u_degree : int
        Degree of the u-basis polynomials

    v_degree : int
        Degree of the v-basis polynomials

    u_knots : ndarray with shape (r+1=n+p+2,)
        Knot vector in the u-direction
        Set the multiplicity of the first and last entries equal to ´p+1´ to obtain a clamped spline

    v_knots : ndarray with shape (s+1=m+q+2,)
        Knot vector in the v-direction
        Set the multiplicity of the first and last entries equal to ´q+1´ to obtain a clamped spline

    radius : float
        Radius of curvature of the base conic. (ignored if control points are passed)
    
    conic : float
        Conic constant of the base conic. (ignored if control points are passed)

    nurbs_norm_x :    float (default None)
    nurbs_norm_y :    float (default None)
    x_center :  float (default 0.0)
    y_center :  float (default 0.0)     
        These parameters define the rectangular area for the fit of the surface with the NURBS 
    
    n_points_u : int (default 4) 
    n_points_v : int (default 4)
        Define grid size of control points (n_points_u x n_points_v). 
        Default value is updated in case control points are passed.
        
    tol : float, optional
        Tolerance for Newton-Raphson iteration, by default 1e-10.
    max_iter : int, optional
        Maximum number of iterations for Newton-Raphson, by default 100.        

    Notes
    -----
    The class can be used to represent polynomial and rational Bézier, B-Spline and NURBS surfaces
    The type of surface depends on the initialization arguments

        - Polynomial Bézier:  Provide the array of control points
        - Rational Bézier:    Provide the arrays of control points and weights
        - B-Spline:           Provide the array of control points, (u,v) degrees and (u,v) knot vectors
        - NURBS:              Provide the arrays of control points and weights, (u,v) degrees and (u,v) knot vectors

    References
    ----------
    The NURBS Book. See references to equations and algorithms throughout the code
    L. Piegl and W. Tiller
    Springer, second edition

    Curves and Surfaces for CADGD. See references to equations in the source code
    G. Farin
    Morgan Kaufmann Publishers, fifth edition

    All references correspond to The NURBS book unless it is explicitly stated that they come from Farin's book

    """

    def __init__(self, coordinate_system, radius=be.inf, conic=0.0, nurbs_norm_x=None, nurbs_norm_y=None,
                 x_center = 0.0, y_center = 0.0, control_points=None, weights=None, 
                 u_degree=None, v_degree=None, u_knots=None, v_knots=None, 
                 n_points_u = 4, n_points_v = 4, tol=1e-10, max_iter=100):

        super().__init__(coordinate_system)
        self.P = control_points
        self.W = weights
        self.p = u_degree
        self.q = v_degree
        self.U = u_knots
        self.V = v_knots       
        self.nurbs_norm_x = nurbs_norm_x
        self.nurbs_norm_y = nurbs_norm_y
        self.x_center = x_center
        self.y_center = y_center
        self.radius = radius
        self.k = conic
        self.tol = tol
        self.max_iter = max_iter
        
        self.is_symmetric = False #TO BE CHECKED. If I put false I get an error calling draw3D 

        # If control points are not provided the NURBS is obtained as fit of standard surface 
        if control_points is None and weights is None and u_degree is None and v_degree is None \
                and u_knots is None and v_knots is None:

            self.is_fitted = True
            self.ndim = 3 
            self.P_size_u = n_points_u + 1 
            self.P_size_v = n_points_v + 1

        # Polynomial Bezier surface initialization
        elif weights is None and u_degree is None and u_knots is None and v_degree is None and v_knots is None:

            self.is_fitted = False
            # Set the surface type flag
            self.surface_type = 'Bezier'

            # Set the number of dimensions of the problem
            self.ndim = be.shape(control_points)[0]

            # Maximum index of the control points (counting from zero)
            n = be.shape(control_points)[1] - 1
            m = be.shape(control_points)[2] - 1
            self.P_size_u = n + 1
            self.P_size_v = m + 1
            # Define the weight of the control points
            weights = be.ones((n + 1, m + 1), dtype=control_points.dtype)

            # Define the order of the basis polynomials
            u_degree = n
            v_degree = m

            # Define the knot vectors (clamped spline)
            u_knots = be.concatenate((be.zeros(u_degree), be.linspace(0, 1, n - u_degree + 2), be.ones(u_degree)))
            v_knots = be.concatenate((be.zeros(v_degree), be.linspace(0, 1, m - v_degree + 2), be.ones(v_degree)))


        # Rational Bezier surface initialization
        elif u_degree is None and u_knots is None and v_degree is None and v_knots is None:

            self.is_fitted = False

            # Set the surface type flag
            self.surface_type = 'R-Bezier'

            # Set the number of dimensions of the problem
            self.ndim = be.shape(control_points)[0]

            # Maximum index of the control points (counting from zero)
            n = be.shape(control_points)[1] - 1
            m = be.shape(control_points)[2] - 1
            self.P_size_u = n + 1
            self.P_size_v = m + 1

            # Define the order of the basis polynomials
            u_degree = n
            v_degree = m

            # Define the knot vectors (clamped spline)
            u_knots = be.concatenate((be.zeros(u_degree), be.linspace(0, 1, n - u_degree + 2), be.ones(u_degree)))
            v_knots = be.concatenate((be.zeros(v_degree), be.linspace(0, 1, m - v_degree + 2), be.ones(v_degree)))


        # B-Spline surface initialization (both degree and knot vector are provided)
        elif weights is None and u_knots is not None and v_knots is not None:

            self.is_fitted = False

            # Set the surface type flag
            self.surface_type = 'B-Spline'

            # Set the number of dimensions of the problem
            self.ndim = be.shape(control_points)[0]

            # Maximum index of the control points (counting from zero)
            n = be.shape(control_points)[1] - 1
            m = be.shape(control_points)[2] - 1
            self.P_size_u = n + 1
            self.P_size_v = m + 1

            # Define the weight of the control points
            weights = be.ones((n + 1, m + 1), dtype=control_points.dtype)


        # B-Spline surface initialization (degree is given but the knot vector is not provided)
        elif weights is None and u_knots is None and v_knots is None:

            self.is_fitted = False

            # Set the surface type flag
            self.surface_type = 'B-Spline'

            # Set the number of dimensions of the problem
            self.ndim = be.shape(control_points)[0]

            # Maximum index of the control points (counting from zero)
            n = be.shape(control_points)[1] - 1
            m = be.shape(control_points)[2] - 1
            self.P_size_u = n + 1
            self.P_size_v = m + 1

            # Define the knot vectors (clamped spline)
            u_knots = be.concatenate((be.zeros(u_degree), be.linspace(0, 1, n - u_degree + 2), be.ones(u_degree)))
            v_knots = be.concatenate((be.zeros(v_degree), be.linspace(0, 1, m - v_degree + 2), be.ones(v_degree)))

            # Define the weight of the control points
            weights = be.ones((n + 1, m + 1), dtype=control_points.dtype)

        # NURBS surface initialization
        else:
            self.is_fitted = False

            # Set the surface type flag
            self.surface_type = 'NURBS'

            if u_knots is None and v_knots is None:

                # Maximum index of the control points (counting from zero)
                n = be.shape(control_points)[1] - 1
                m = be.shape(control_points)[2] - 1
                self.P_size_u = n + 1
                self.P_size_v = m + 1

                # Define the knot vectors (clamped spline)
                u_knots = be.concatenate((be.zeros(u_degree), be.linspace(0, 1, n - u_degree + 2), be.ones(u_degree)))
                v_knots = be.concatenate((be.zeros(v_degree), be.linspace(0, 1, m - v_degree + 2), be.ones(v_degree)))

            # Set the number of dimensions of the problem
            self.ndim = be.shape(control_points)[0]

    def flip(self):
        """Flip the geometry.

        Changes the sign of the radius abd the control points z coordinate
        """
        self.radius = -self.radius
        self.P[2,:,:] = -self.P[2,:,:]

    # ---------------------------------------------------------------------------------------------------------------- #
    # Compute NURBS surface coordinates
    # ---------------------------------------------------------------------------------------------------------------- #
    def get_value(self, u, v):

        """ Evaluate the coordinates of the surface corresponding to the (u,v) parametrization

        Parameters
        ----------
        u : scalar or ndarray with shape (N,)
            u-parameter used to evaluate the surface

        v : scalar or ndarray with shape (N,)
            v-parameter used to evaluate the surface

        Returns
        -------
        S : ndarray with shape (ndim, N)
            Array containing the coordinates of the surface
            The first dimension of ´S´ spans the ´(x,y,z)´ coordinates
            The second dimension of ´S´ spans the (u,v) parametrization sample points

        """

        # Check that u and v have the same size
        if be.isscalar(u) and be.isscalar(v): pass
        elif u.size == v.size: pass
        else: raise Exception('u and v must have the same size')
        
        if u.ndim > 1:
            a,b = u.shape
            u = be.ravel(u)
            v = be.ravel(v)
            # Evaluate the NURBS surface for the input (u,v) parametrization
            S = self.compute_nurbs_coordinates(self.P, self.W, self.p, self.q, self.U, self.V, u, v)
            S = be.reshape(S,(self.ndim,a,b))            
        else:    
            # Evaluate the NURBS surface for the input (u,v) parametrization
            S = self.compute_nurbs_coordinates(self.P, self.W, self.p, self.q, self.U, self.V, u, v)

        return S


    @staticmethod
    def compute_nurbs_coordinates(P, W, p, q, U, V, u, v):

        """ Evaluate the coordinates of the NURBS surface corresponding to the (u,v) parametrization

        This function computes the coordinates of the NURBS surface in homogeneous space using equation 4.15 and then
        maps the coordinates to ordinary space using the perspective map given by equation 1.16. See algorithm A4.3

        Parameters
        ----------
        P : ndarray with shape (ndim, n+1, m+1)
            Array containing the coordinates of the control points
            The first dimension of ´P´ spans the coordinates of the control points (any number of dimensions)
            The second dimension of ´P´ spans the u-direction control points (0, 1, ..., n)
            The third dimension of ´P´ spans the v-direction control points (0, 1, ..., m)

        W : ndarray with shape (n+1, m+1)
            Array containing the weight of the control points
            The first dimension of ´W´ spans the u-direction control points weights (0, 1, ..., n)
            The second dimension of ´W´ spans the v-direction control points weights (0, 1, ..., m)

        p : int
            Degree of the u-basis polynomials

        q : int
            Degree of the v-basis polynomials

        U : ndarray with shape (r+1=n+p+2,)
            Knot vector in the u-direction
            Set the multiplicity of the first and last entries equal to ´p+1´ to obtain a clamped spline

        V : ndarray with shape (s+1=m+q+2,)
            Knot vector in the v-direction
            Set the multiplicity of the first and last entries equal to ´q+1´ to obtain a clamped spline

        u : scalar or ndarray with shape (N,)
            u-parameter used to evaluate the surface

        v : scalar or ndarray with shape (N,)
            v-parameter used to evaluate the surface

        Returns
        -------
        S : ndarray with shape (ndim, N)
            Array containing the NURBS surface coordinates
            The first dimension of ´S´ spans the ´(x,y,z)´ coordinates
            The second dimension of ´S´ spans the (u,v) parametrization sample points

        """

        # Check the shape of the input parameters
        if P.ndim > 3:           raise Exception('P must be an array of shape (ndim, n+1, m+1)')
        if W.ndim > 2:           raise Exception('W must be an array of shape (n+1, m+1)')
        if not be.isscalar(p):   raise Exception('p must be an scalar')
        if not be.isscalar(q):   raise Exception('q must be an scalar')
        if U.ndim > 1:           raise Exception('U must be an array of shape (r+1=n+p+2,)')
        if V.ndim > 1:           raise Exception('V must be an array of shape (s+1=m+q+2,)')
        if be.isscalar(u):       u = be.asarray(u)
        elif u.ndim > 1:         raise Exception('u must be a scalar or an array of shape (N,)')
        if be.isscalar(v):       v = be.asarray(v)
        elif u.ndim > 1:         raise Exception('v must be a scalar or an array of shape (N,)')

        # Shape of the array of control points
        n_dim, nn, mm = be.shape(P)

        # Highest index of the control points (counting from zero)
        n = nn - 1
        m = mm - 1

        # Compute the B-Spline basis polynomials
        N_basis_u = compute_basis_polynomials(n, p, U, u)  # shape (n+1, N)
        N_basis_v = compute_basis_polynomials(m, q, V, v)  # shape (m+1, N)

        # Map the control points to homogeneous space | P_w = (x*w,y*w,z*w,w)
        P_w = be.concatenate((P * W[be.newaxis, :], W[be.newaxis, :]), axis=0)

        # Compute the coordinates of the NURBS surface in homogeneous space
        # This implementation is vectorized to increase speed
        A = be.dot(P_w, N_basis_v)                                      # shape (ndim+1, n+1, N)
        B = be.repeat(N_basis_u[be.newaxis], repeats=n_dim+1, axis=0)   # shape (ndim+1, n+1, N)
        S_w = be.sum(A*B, axis=1)                                       # shape (ndim+1, N)

        # Map the coordinates back to the ordinary space
        S = S_w[0:-1,:]/S_w[-1, :]

        return S


    @staticmethod
    def compute_bspline_coordinates(P, p, q, U, V, u, v):

        """ Evaluate the coordinates of the B-Spline surface corresponding to the (u,v) parametrization

        This function computes the coordinates of a B-Spline surface as given by equation 3.11. See algorithm A3.5

        Parameters
        ----------
        P : ndarray with shape (ndim, n+1, m+1)
            Array containing the coordinates of the control points
            The first dimension of ´P´ spans the coordinates of the control points (any number of dimensions)
            The second dimension of ´P´ spans the u-direction control points (0, 1, ..., n)
            The third dimension of ´P´ spans the v-direction control points (0, 1, ..., m)

        p : int
            Degree of the u-basis polynomials

        q : int
            Degree of the v-basis polynomials

        U : ndarray with shape (r+1=n+p+2,)
            Knot vector in the u-direction
            Set the multiplicity of the first and last entries equal to ´p+1´ to obtain a clamped spline

        V : ndarray with shape (s+1=m+q+2,)
            Knot vector in the v-direction
            Set the multiplicity of the first and last entries equal to ´q+1´ to obtain a clamped spline

        u : scalar or ndarray with shape (N,)
            u-parameter used to evaluate the surface

        v : scalar or ndarray with shape (N,)
            v-parameter used to evaluate the surface

        Returns
        -------
        S : ndarray with shape (ndim, N)
            Array containing the NURBS surface coordinates
            The first dimension of ´S´ spans the ´(x,y,z)´ coordinates
            The second dimension of ´S´ spans the (u,v) parametrization sample points

        """

        # Check the shape of the input parameters
        if P.ndim > 3:           raise Exception('P must be an array of shape (ndim, n+1, m+1)')
        if not be.isscalar(p):   raise Exception('p must be an scalar')
        if not be.isscalar(q):   raise Exception('q must be an scalar')
        if U.ndim > 1:           raise Exception('U must be an array of shape (r+1=n+p+2,)')
        if V.ndim > 1:           raise Exception('V must be an array of shape (s+1=m+q+2,)')
        if be.isscalar(u):       u = be.asarray(u)
        elif u.ndim > 1:         raise Exception('u must be a scalar or an array of shape (N,)')
        if be.isscalar(v):       v = be.asarray(v)
        elif u.ndim > 1:         raise Exception('v must be a scalar or an array of shape (N,)')

        # Shape of the array of control points
        n_dim, nn, mm = be.shape(P)

        # Highest index of the control points (counting from zero)
        n = nn - 1
        m = mm - 1

        # Compute the B-Spline basis polynomials
        N_basis_u = compute_basis_polynomials(n, p, U, u)  # shape (n+1, N)
        N_basis_v = compute_basis_polynomials(m, q, V, v)  # shape (m+1, N)

        # Compute the coordinates of the B-Spline surface
        # This implementation is vectorized to increase speed
        A = be.dot(P, N_basis_v)                                        # shape (ndim, n+1, N)
        B = be.repeat(N_basis_u[be.newaxis], repeats=n_dim, axis=0)     # shape (ndim, n+1, N)
        S = be.sum(A*B,axis=1)                                          # shape (ndim, N)

        return S


    # ---------------------------------------------------------------------------------------------------------------- #
    # Compute the derivatives of the surface
    # ---------------------------------------------------------------------------------------------------------------- #
    def get_derivative(self, u, v, order_u, order_v):

        """ Evaluate the derivative of the surface for the input u-parametrization

        Parameters
        ----------
        u : scalar or ndarray with shape (N,)
            u-parameter used to evaluate the surface

        v : scalar or ndarray with shape (N,)
            v-parameter used to evaluate the surface

        order_u : int
            Order of the partial derivative in the u-direction

        order_v : int
            Order of the partial derivative in the v-direction

        Returns
        -------
        dS : ndarray with shape (ndim, N)
            Array containing the derivative of the desired order
            The first dimension of ´dC´ spans the ´(x,y,z)´ coordinates
            The second dimension of ´dC´ spans the ´u´ parametrization sample points

        """

        if u.ndim > 1:
            a,b = u.shape
            u = be.ravel(u)
            v = be.ravel(v)
            # Compute the array of surface derivatives up to the input (u,v) orders and slice the desired values
            dS = self.compute_nurbs_derivatives(self.P, self.W, self.p, self.q, self.U, self.V, u, v, order_u, order_v)[order_u, order_v, ...]
            dS = be.reshape(dS,(self.ndim,a,b))            
        else:    
            # Compute the array of surface derivatives up to the input (u,v) orders and slice the desired values
            dS = self.compute_nurbs_derivatives(self.P, self.W, self.p, self.q, self.U, self.V, u, v, order_u, order_v)[order_u, order_v, ...]



        return dS


    def compute_nurbs_derivatives(self, P, W, p, q, U, V, u, v, up_to_order_u, up_to_order_v):

        """ Compute the derivatives of a NURBS surface in ordinary space up to to the desired orders

        This function computes the analytic derivatives of the NURBS surface in ordinary space using equation 4.20 and
        the derivatives of the NURBS surface in homogeneous space obtained from compute_bspline_derivatives()

        The derivatives are computed recursively in a fashion similar to algorithm A4.4

        Parameters
        ----------
        P : ndarray with shape (ndim, n+1, m+1)
            Array containing the coordinates of the control points
            The first dimension of ´P´ spans the coordinates of the control points (any number of dimensions)
            The second dimension of ´P´ spans the u-direction control points (0, 1, ..., n)
            The third dimension of ´P´ spans the v-direction control points (0, 1, ..., m)

        W : ndarray with shape (n+1, m+1)
            Array containing the weight of the control points
            The first dimension of ´W´ spans the u-direction control points weights (0, 1, ..., n)
            The second dimension of ´W´ spans the v-direction control points weights (0, 1, ..., m)

        p : int
            Degree of the u-basis polynomials

        q : int
            Degree of the v-basis polynomials

        U : ndarray with shape (r+1=n+p+2,)
            Knot vector in the u-direction
            Set the multiplicity of the first and last entries equal to ´p+1´ to obtain a clamped spline

        V : ndarray with shape (s+1=m+q+2,)
            Knot vector in the v-direction
            Set the multiplicity of the first and last entries equal to ´q+1´ to obtain a clamped spline

        u : scalar or ndarray with shape (N,)
            u-parameter used to evaluate the surface

        v : scalar or ndarray with shape (N,)
            v-parameter used to evaluate the surface

        up_to_order_u : integer
            Order of the highest derivative in the u-direction

        up_to_order_v : integer
            Order of the highest derivative in the v-direction

        Returns
        -------
        nurbs_derivatives: ndarray of shape (up_to_order_u+1, up_to_order_v+1, ndim, Nu)
            The first dimension spans the order of the u-derivatives (0, 1, 2, ...)
            The second dimension spans the order of the v-derivatives (0, 1, 2, ...)
            The third dimension spans the coordinates (x,y,z,...)
            The fourth dimension spans (u,v) parametrization sample points

        """

        # Set the data type used to initialize arrays (set `complex` if an argument is complex and `float` if not)
        u, v = be.asarray(u), be.asarray(v)

        # Map the control points to homogeneous space | P_w = (x*w,y*w,z*w,w)
        P_w = be.concatenate((P * W[be.newaxis, :], W[be.newaxis, :]), axis=0)

        # Compute the derivatives of the NURBS surface in homogeneous space
        bspline_derivatives = self.compute_bspline_derivatives(P_w, p, q, U, V, u, v, up_to_order_u, up_to_order_v)
        A_ders = bspline_derivatives[:, :, 0:-1, :]
        w_ders = bspline_derivatives[:, :, [-1], :]

        # Initialize array of derivatives
        n_dim, N = be.shape(P)[0], be.size(u)
        nurbs_derivatives = be.zeros((up_to_order_u+1, up_to_order_v+1, n_dim, N))

        # Compute the derivatives of up to the desired order
        # See algorithm A4.4 from the NURBS book
        for k in range(up_to_order_u+1):
            for L in range(up_to_order_v+1):

                # Update the numerator of equation 4.20 recursively
                temp_numerator = A_ders[[k], [L], ...]

                # Summation j=0 and point_index=1:k
                for i in range(1, k + 1):
                    temp_numerator -= binom(k, i)*w_ders[[i], [0], ...]*nurbs_derivatives[[k-i], [L], ...]

                # Summation point_index=0 and j=1:L
                for j in range(1, L + 1):
                    temp_numerator -= binom(L, j)*w_ders[[0], [j], ...]*nurbs_derivatives[[k], [L-j], ...]

                # Summation point_index=1:k and j=1:L
                for i in range(1, k+1):
                    for j in range(1, L+1):
                        temp_numerator -= binom(k, i) * binom(L, j)* w_ders[[i], [j], ...] * nurbs_derivatives[[k-i], [L-j], ...]

                # Compute the (k,L)-th order NURBS surface partial derivative
                nurbs_derivatives[k, L, ...] = temp_numerator/w_ders[[0], [0], ...]

        return nurbs_derivatives


    @staticmethod
    def compute_bspline_derivatives(P, p, q, U, V, u, v, up_to_order_u, up_to_order_v):

        """ Compute the derivatives of a B-Spline (or NURBS surface in homogeneous space) up to orders
        `derivative_order_u` and `derivative_order_v`

        This function computes the analytic derivatives of a B-Spline surface using equation 3.17. See algorithm A3.6

        Parameters
        ----------
        P : ndarray with shape (ndim, n+1, m+1)
            Array containing the coordinates of the control points
            The first dimension of ´P´ spans the coordinates of the control points (any number of dimensions)
            The second dimension of ´P´ spans the u-direction control points (0, 1, ..., n)
            The third dimension of ´P´ spans the v-direction control points (0, 1, ..., m)

        p : int
            Degree of the u-basis polynomials

        q : int
            Degree of the v-basis polynomials

        U : ndarray with shape (r+1=n+p+2,)
            Knot vector in the u-direction
            Set the multiplicity of the first and last entries equal to ´p+1´ to obtain a clamped spline

        V : ndarray with shape (s+1=m+q+2,)
            Knot vector in the v-direction
            Set the multiplicity of the first and last entries equal to ´q+1´ to obtain a clamped spline

        u : scalar or ndarray with shape (N,)
            u-parameter used to evaluate the surface

        v : scalar or ndarray with shape (N,)
            v-parameter used to evaluate the surface

        up_to_order_u : integer
            Order of the highest derivative in the u-direction

        up_to_order_v : integer
            Order of the highest derivative in the v-direction

        Returns
        -------
        bspline_derivatives: ndarray of shape (up_to_order_u+1, up_to_order_v+1, ndim, Nu)
            The first dimension spans the order of the u-derivatives (0, 1, 2, ...)
            The second dimension spans the order of the v-derivatives (0, 1, 2, ...)
            The third dimension spans the coordinates (x,y,z,...)
            The fourth dimension spans (u,v) parametrization sample points

        """

        # Set the data type used to initialize arrays (set `complex` if an argument is complex and `float` if not)
        u = be.asarray(u)

        # Set the B-Spline coordinates as the zero-th derivatives
        n_dim, N = be.shape(P)[0], be.size(u)
        bspline_derivatives = be.zeros((up_to_order_u+1, up_to_order_v+1, n_dim, N))

        # Compute the derivatives of up to the desired order
        # See algorithm A3.2 from the NURBS book
        for order_u in range(min(p, up_to_order_u) + 1):
            for order_v in range(min(q, up_to_order_v) + 1):

                # Highest index of the control points (counting from zero)
                n = be.shape(P)[1] - 1
                m = be.shape(P)[2] - 1

                # Compute the B-Spline basis polynomials
                N_basis_u = compute_basis_polynomials_derivatives(n, p, U, u, order_u)
                N_basis_v = compute_basis_polynomials_derivatives(m, q, V, v, order_v)

                # Compute the coordinates of the B-Spline surface
                # This implementation is vectorized to increase speed
                A = be.dot(P, N_basis_v)                                                # shape (ndim, n+1, N)
                B = be.repeat(N_basis_u[be.newaxis], repeats=n_dim, axis=0)             # shape (ndim, n+1, N)
                bspline_derivatives[order_u, order_v, :, :] = be.sum(A * B, axis=1)     # shape (ndim, N)

        # Note that derivatives with order higher than `p` and `q` are not computed and are be zero from initialization
        # These zero-derivatives are required to compute the higher order derivatives of rational surfaces

        return bspline_derivatives


    # ---------------------------------------------------------------------------------------------------------------- #
    # Compute the unitary normal vectors
    # ---------------------------------------------------------------------------------------------------------------- #
    def get_normals(self, u, v):

        """ Evaluate the unitary vectors normal to the surface the input (u,v) parametrization

        The definition of the unitary normal vector is given in section 19.2 (Farin's textbook)

        Parameters
        ----------
        u : scalar or ndarray with shape (N,)
            Scalar or array containing the u-parameter used to evaluate the normals

        v : scalar or ndarray with shape (N,)
            Scalar or array containing the v-parameter used to evaluate the normals

        Returns
        -------
        normals : ndarray with shape (ndim, N)
            Array containing the unitary vectors normal to the surface

        """

        # Compute 2 vectors tangent to the surface
        S_u = self.get_derivative(u, v, order_u=1, order_v=0)
        S_v = self.get_derivative(u, v, order_u=0, order_v=1)

        # Compute the normal vector as the cross product of the tangent vectors and normalize it
        normals = be.cross(S_u, S_v, axisa=0, axisb=0, axisc=0)
        normals = normals/be.sum(normals ** 2, axis=0) ** (1 / 2)

        return normals

    def _corr_general(self,u,v,d1,d2,N1,N2):
        """
        Define the correction step for the update of u,v coordinates
        
        See paper Practical ray tracing of trimmed NURBS surface - section 2.3

        Parameters
        ----------
        u : TYPE scalar or ndarray with shape (N,)
            DESCRIPTION. Scalar or array containing the u-parameter
        v : TYPE scalar or ndarray with shape (N,)
            DESCRIPTION. Scalar or array containing the v-parameter
        d1 : TYPE scalar or ndarray with shape (N,)
            DESCRIPTION. Contains minus dot product of N1 rows with ray (x,y,z) 
        d2 : TYPE scalar or ndarray with shape (N,)
            DESCRIPTION. Contains minus dot product of N2 rows with ray (x,y,z)
        N1 : TYPE ndarray with shape (N,3)
            DESCRIPTION. Each row of N1 is a normal vector of a plane
        N2 : TYPE ndarray with shape (N,3)
            DESCRIPTION. Each row of N2 is a normal vector of a plane. Intersection of
            planes defined by N1[i,:] and N2[i,:] define a specific ray

        Returns
        -------
        correction : TYPE ndarray with shape (2,N)
            DESCRIPTION. First line contains correction steps for u parameters
            second line contains correction steps for v parameters                
        residual : TYPE scalar 
            DESCRIPTION.Maximum distance between all rays and surface intersection

        """
        S_uv = self.get_value(u,v).T
        #r is the distance between the point on the surface S-uv and the ray
        #r need to be be minimized (r=0 means that the intersection of the ray with the surface is found)
        r = be.array([be.sum(N1 * S_uv, axis=1)+d1,be.sum(N2 * S_uv, axis=1)+d2]).reshape(2,-1)
        # Compute Jacobian matrix and its inversion
        _,Np = r.shape
        a = be.sum(N1 * self.get_derivative(u, v, order_u=1, order_v=0).T, axis = 1)
        b = be.sum(N1 * self.get_derivative(u, v, order_u=0, order_v=1).T, axis = 1)
        c = be.sum(N2 * self.get_derivative(u, v, order_u=1, order_v=0).T, axis = 1)
        d = be.sum(N2 * self.get_derivative(u, v, order_u=0, order_v=1).T, axis = 1) 
        
        J = be.vstack((a,b,c,d)).T.reshape((Np,2,2))
        # Linear algebra property used to compute Jacobian inversion from its determinant
        adj = be.zeros((Np,2,2))
        adj[:,0,0] = J[:,1,1]
        adj[:,0,1] = -J[:,0,1]
        adj[:,1,0] = -J[:,1,0]
        adj[:,1,1] = J[:,0,0]
        
        detJ = be.linalg.det(J)
        detJ = detJ[: , be.newaxis,be.newaxis]
        invj = adj/detJ
        
        correction = be.einsum('ijk,ki->ji',invj,r)
        
        residual = be.max(be.abs(r))
        
        return correction,residual

    def _corr(self,u,v,d1,d2):
        '''
        _corr is a specific version of _corr_general that compute the intersection
        between rays and NURBS surface when rays direction is along Z axis.
        This is used to compute surface sag.

        Parameters
        ----------
        u : TYPE
            DESCRIPTION.
        v : TYPE
            DESCRIPTION.
        d1 : TYPE
            DESCRIPTION.
        d2 : TYPE
            DESCRIPTION.

        Returns
        -------
        correction : TYPE
            DESCRIPTION.
        residual : TYPE
            DESCRIPTION.

        '''
        S_uv = self.get_value(u,v)
        r = be.array([S_uv[1,:]+d1,S_uv[0,:]+d2]).reshape(2,-1)
        _,Np = r.shape
        a = self.get_derivative(u, v, order_u=1, order_v=0)[1,:]
        b = self.get_derivative(u, v, order_u=0, order_v=1)[1,:]
        c = self.get_derivative(u, v, order_u=1, order_v=0)[0,:]
        d = self.get_derivative(u, v, order_u=0, order_v=1)[0,:]    

        J = be.vstack((a,b,c,d)).T.reshape((Np,2,2))
        adj = be.zeros((Np,2,2))
        adj[:,0,0] = J[:,1,1]
        adj[:,0,1] = -J[:,0,1]
        adj[:,1,0] = -J[:,1,0]
        adj[:,1,1] = J[:,0,0]
        
        detJ = be.linalg.det(J)
        detJ = detJ[: , be.newaxis,be.newaxis]
        invj = adj/detJ
        
        correction = be.einsum('ijk,ki->ji',invj,r)

        residual = be.max(be.abs(r))
        return correction,residual
    
    def sag(self, x=0, y=0):
        '''
        Compute surface sag for specific x,y from the z coordinate of the 
        intersection point between the ray with direction (0,0,1) and passing from
        (x,y,0) and the NURBS surface. 

        Parameters
        ----------
        x : TYPE, optional
            DESCRIPTION. The default is 0.
        y : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        shape = x.shape
        u = be.zeros(x.size)
        v = be.zeros(x.size)        
        for _ in range(self.max_iter):
            correction,residual = self._corr(u,v,-be.ravel(y),-be.ravel(x))
            u = u - correction[0,:]
            v = v - correction[1,:]
            u[be.logical_or(u < 0.0, v < 0.0)] = be.random.rand()
            v[be.logical_or(u < 0.0, v < 0.0)] = be.random.rand()
            u[be.logical_or(u > 1.0, v > 1.0)] = be.random.rand()
            v[be.logical_or(u > 1.0, v > 1.0)] = be.random.rand()

            if residual < self.tol:
                break        
        return self.get_value(u,v)[2,:].reshape(shape)
    
    def distance(self, rays):
        """Find the propagation distance to the geometry for the given rays.
        The approach is described in the paper "Practical ray tracing of 
        trimmed NURBS surfaces" from William Martin etc.

        Args:
            rays (RealRays): The rays for which to calculate the distance.

        Returns:
            be.ndarray: An array of distances from each ray's current position
            to its intersection point with the geometry.

        """
        #The ray is expressed as the intersection of two planes with normals N1 and N2
        #N1,N2 and d (ray direction) are 3 ortogonal vectors
        N1x = be.zeros_like(rays.x)
        N1y = be.zeros_like(rays.x)
        N1z = be.zeros_like(rays.x)
        mask = be.logical_and(rays.L > rays.M, rays.L > rays.N)
        
        N1x[mask] = rays.M[mask]/be.sqrt(rays.L[mask]**2+rays.M[mask]**2)
        N1y[mask] = -rays.L[mask]/be.sqrt(rays.L[mask]**2+rays.M[mask]**2)
        N1y[~mask] = rays.N[~mask]/be.sqrt(rays.N[~mask]**2+rays.M[~mask]**2)
        N1z[~mask] = -rays.M[~mask]/be.sqrt(rays.N[~mask]**2+rays.M[~mask]**2)
        

        N1 = be.column_stack((N1x,N1y,N1z))
        
        #rays directions 
        d = be.column_stack((rays.L,rays.M,rays.N))
        
        N2 = be.cross(N1,d)
        
        #ray position        
        P0 = be.column_stack((rays.x,rays.y,rays.z))
        
        d1 = -be.sum(N1 * P0, axis=1)
        d2 = -be.sum(N2 * P0, axis=1)
        
        #As a starting guess it assume that all u,v parameters are zero
        #Maybe a more clever way to define a better initial guess of u,v parameters can be found 
        u = be.zeros_like(rays.x)
        v = be.zeros_like(u)

        for _ in range(self.max_iter):
            correction,residual = self._corr_general(u,v,d1,d2,N1,N2)
            u = u - correction[0,:]
            v = v - correction[1,:]
            #if u,v assume values outside the range [0,1] take another random starting point
            u[be.logical_or(u < 0.0, v < 0.0)] = be.random.rand()
            v[be.logical_or(u < 0.0, v < 0.0)] = be.random.rand()
            u[be.logical_or(u > 1.0, v > 1.0)] = be.random.rand()
            v[be.logical_or(u > 1.0, v > 1.0)] = be.random.rand()
            if residual < self.tol:
                break
            
        t = be.sqrt(be.sum((self.get_value(u,v).T-P0)**2,axis=1))        

        return t
    
    def surface_normal(self, rays):
        '''
        Compute surface normal

        Parameters
        ----------
        rays : TYPE
            DESCRIPTION.

        Returns
        -------
        nx : TYPE
            DESCRIPTION.
        ny : TYPE
            DESCRIPTION.
        nz : TYPE
            DESCRIPTION.

        '''
        x = rays.x
        y = rays.y
        #compute u,v parameters of surface intersections with rays (x,y,0),(0,0,1)
        u = be.zeros_like(x)
        v = be.zeros_like(u)        
        for _ in range(self.max_iter):
            correction,residual = self._corr(u,v,-y,-x)
            u = u - correction[0,:]
            v = v - correction[1,:]
            u[be.logical_or(u < 0.0, v < 0.0)] = be.random.rand()
            v[be.logical_or(u < 0.0, v < 0.0)] = be.random.rand()
            u[be.logical_or(u > 1.0, v > 1.0)] = be.random.rand()
            v[be.logical_or(u > 1.0, v > 1.0)] = be.random.rand()
            if residual < self.tol:
                break        
        n = self.get_normals(u, v)
        nx = n[0,:]
        ny = n[1,:]
        nz = n[2,:]

        return nx, ny, nz   

    def __str__(self) -> str:
        return "NURBS"        

    def fit_surface(self):
        '''
        This function handle the NURBS surface approximation calling specific functions
        depending on the nature of the surface that we want to fit. For the time being
        standard surface and plane surface are implemented.

        Returns
        -------
        None.

        '''
        if be.isinf(self.radius):
            self._plane_surface()
        else:
            self._standard_surface()
        
    def _standard_surface(self): 
        '''
        Generate NURBS parameters based on the surface approximation with a standard surface

        Returns
        -------
        None.

        '''
        
        radius = self.radius
        k = self.k
        nurbs_norm_x = self.nurbs_norm_x
        nurbs_norm_y = self.nurbs_norm_y
        xc = self.x_center
        yc = self.y_center
        P_size_u = self.P_size_u
        P_size_v = self.P_size_v
        P_ndim = self.ndim
        
        x = be.linspace(xc - nurbs_norm_x,xc + nurbs_norm_x,P_size_u)
        y = be.linspace(yc - nurbs_norm_y,yc + nurbs_norm_y,P_size_v)
        x,y = be.meshgrid(x,y)
        r2 = x**2 + y**2
        z = r2 / (radius * (1 + be.sqrt(1 - (1 + k) * r2 / radius**2)))
        points = be.zeros((P_ndim, P_size_u, P_size_v))
        points[0,:,:] = x.T
        points[1,:,:] = y.T
        points[2,:,:] = z.T
        
        # Flatten array of surface points and convert to list (input to fitting function)
        xp = (points.reshape(P_ndim,-1).T).tolist()
        
        u_degree = 3
        v_degree = 3
        
        # Do global surface approximation
        ctrlpts,u_degree,v_degree,num_cpts_u,num_cpts_v,kv_u,kv_v = approximate_surface(xp, P_size_u, P_size_v, u_degree, v_degree)
        
        self.P_size_u = num_cpts_u
        self.P_size_v = num_cpts_v
        
        ctrlpts = be.asarray(ctrlpts)
        ctrlpts = (ctrlpts.T).reshape((P_ndim,num_cpts_u,num_cpts_v))
        # Define the array of control point weights
        weights = be.ones((num_cpts_u,num_cpts_v))
        
        u_knots = be.asarray(kv_u)
        v_knots = be.asarray(kv_v)

        # Set the surface type flag
        self.surface_type = 'NURBS'

        self.P = ctrlpts
        self.W = weights
        self.p = u_degree
        self.q = v_degree
        self.U = u_knots
        self.V = v_knots       
            
    def _plane_surface(self):
        '''
        Generate plane surface

        Parameters: None
        ----------

        Returns
        -------
        control_points : TYPE
            DESCRIPTION.
        weights : TYPE
            DESCRIPTION.
        u_degree : TYPE
            DESCRIPTION.
        v_degree : TYPE
            DESCRIPTION.
        u_knots : TYPE
            DESCRIPTION.
        v_knots : TYPE
            DESCRIPTION.

        '''
        x = be.linspace(self.x_center - self.nurbs_norm_x,self.x_center + self.nurbs_norm_x,self.P_size_u)
        y = be.linspace(self.y_center - self.nurbs_norm_y,self.y_center + self.nurbs_norm_y,self.P_size_v)
        x,y = be.meshgrid(x,y)
        z = be.zeros_like(x)
        control_points = be.zeros((self.ndim, self.P_size_u, self.P_size_v))
        control_points[0,:,:] = x.T
        control_points[1,:,:] = y.T
        control_points[2,:,:] = z.T

        # Define the array of control point weights
        weights = be.ones((self.P_size_u, self.P_size_v))
        
        # Maximum index of the control points (counting from zero)
        n = be.shape(control_points)[1] - 1
        m = be.shape(control_points)[2] - 1
        
        # Define the order of the basis polynomials
        # Linear (p = 1), Quadratic (p = 2), Cubic (p = 3), etc.
        # Set p = n (number of control points minus one) to obtain a Bezier
        u_degree = 3
        v_degree = 3
        
        # Define the knot vectors (clamped spline)
        # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones.  In total r+1 points where r=n+p+1
        # q+1 zeros, m-p equispaced points between 0 and 1, and q+1 ones. In total s+1 points where s=m+q+1
        u_knots = be.concatenate((be.zeros(u_degree), be.linspace(0, 1, n - u_degree + 2), be.ones(u_degree)))
        v_knots = be.concatenate((be.zeros(v_degree), be.linspace(0, 1, m - v_degree + 2), be.ones(v_degree)))

        # Set the surface type flag
        self.surface_type = 'NURBS'
        self.P = control_points
        self.W = weights
        self.p = u_degree
        self.q = v_degree
        self.U = u_knots
        self.V = v_knots       
