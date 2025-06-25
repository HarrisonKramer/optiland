# optiland/geometries/forbes/qpoly.py
"""Tools for working with Q (Forbes) polynomials."""
from collections import defaultdict
from functools import lru_cache

import optiland.backend as be
from .jacobi import jacobi_sum_clenshaw_der

# These functions are used for coefficient generation and are scalar, so numpy is fine.
import numpy as np
from scipy import special


def kronecker(i, j):
    return 1 if i == j else 0


def gamma_func(n, m):
    if n == 1 and m == 2:
        return 3 / 8
    elif n == 1 and m > 2:
        mm1 = m - 1
        numerator = 2 * mm1 + 1
        denominator = 2 * (mm1 - 1)
        coef = numerator / denominator
        return coef * gamma_func(1, mm1)
    else:
        nm1 = n - 1
        num = (nm1 + 1) * (2 * m + 2 * nm1 - 1)
        den = (m + nm1 - 2) * (2 * nm1 + 1)
        coef = num / den
        return coef * gamma_func(nm1, m)


@lru_cache(1000)
def g_qbfs(n_minus_1):
    if n_minus_1 == 0:
        return - 1 / 2
    else:
        n_minus_2 = n_minus_1 - 1
        return - (1 + g_qbfs(n_minus_2) * h_qbfs(n_minus_2)) / f_qbfs(n_minus_1)


@lru_cache(1000)
def h_qbfs(n_minus_2):
    n = n_minus_2 + 2
    return -n * (n - 1) / (2 * f_qbfs(n_minus_2))


@lru_cache(1000)
def f_qbfs(n):
    if n == 0:
        return 2
    elif n == 1:
        return np.sqrt(19) / 2
    else:
        term1 = n * (n + 1) + 3
        term2 = g_qbfs(n - 1) ** 2
        term3 = h_qbfs(n - 2) ** 2
        return np.sqrt(term1 - term2 - term3)


def change_basis_Qbfs_to_Pn(cs):
    bs = be.empty_like(be.array(cs))
    M = len(bs) - 1
    if M < 0:
        return bs
    fM = f_qbfs(M)
    bs[M] = cs[M] / fM
    if M == 0:
        return bs

    g = g_qbfs(M - 1)
    f = f_qbfs(M - 1)
    bs[M - 1] = (cs[M - 1] - g * bs[M]) / f
    for i in range(M - 2, -1, -1):
        g = g_qbfs(i)
        h = h_qbfs(i)
        f = f_qbfs(i)
        bs[i] = (cs[i] - g * bs[i + 1] - h * bs[i + 2]) / f

    return bs


def _initialize_alphas_q(cs, x, alphas, j=0):
    if alphas is None:
        shape = (len(cs), *be.shape(x)) if hasattr(x, 'shape') else (len(cs),)
        if j != 0:
            shape = (j + 1, *shape)
        if be.get_backend() == 'torch':
            alphas = be.zeros(shape)
            alphas.requires_grad = False
        else:
            alphas = be.zeros(shape)
    return alphas

def _clenshaw_qbfs_functional(bs, usq):
    """
    Pure-functional Clenshaw that returns (S, alpha0, alpha1).
    This version is fixed to handle broadcasting correctly, resolving the stack error,
    and returns the 3 values expected by the caller.
    """
    M = len(bs) - 1
    prefix = 2 - 4 * usq

    if M < 0:
        zeros = be.zeros_like(usq)
        return zeros, zeros, zeros

    # FIX: Ensure b_curr and b_next are broadcast to the shape of usq from the start.
    b_curr = bs[M] + usq * 0
    b_next = be.zeros_like(b_curr)

    # Loop from M-1 down to 0 for the recurrence
    for n in range(M - 1, -1, -1):
        b_new = bs[n] + prefix * b_curr - b_next
        b_next, b_curr = b_curr, b_new

    alpha0, alpha1 = b_curr, b_next
    S = 2 * (alpha0 + alpha1) if M > 0 else 2 * alpha0
    
    # This now returns the BARE polynomial sum S, and the final alpha values
    # The prefactor is applied by the caller
    return S, alpha0, alpha1

def clenshaw_qbfs(cs, usq, alphas=None):
    if be.get_backend() == "torch":
        bs = change_basis_Qbfs_to_Pn(cs)
        # Unpacking now works because _clenshaw_qbfs_functional returns 3 values
        S, alpha0, alpha1 = _clenshaw_qbfs_functional(bs, usq)
        
        # If the caller (clenshaw_qbfs_der) supplied a buffer, fill it
        if alphas is not None:
            M = len(bs) - 1
            # Construct the list of tensors to stack. All elements are now guaranteed
            # to be tensors of the correct shape, fixing the stack error.
            if M == 0:
                fill = [alpha0]
            elif M > 0:
                fill = [alpha0, alpha1] + [be.zeros_like(alpha0)] * (M - 1)
            else: # M < 0
                fill = []

            if fill:
                alphas[...] = be.stack(fill)
        
        # Apply the Forbes pre-factor here, as in the original NumPy path
        return usq * (1 - usq) * S

    # ─ NumPy backend – keep original fast in-place path ─
    x = usq
    bs = change_basis_Qbfs_to_Pn(cs)
    alphas = _initialize_alphas_q(cs, x, alphas, j=0)
    M = len(bs) - 1
    if M < 0:
        return 0.0
    prefix = 2 - 4 * x
    alphas[M] = bs[M]
    if M > 0:
        alphas[M - 1] = bs[M - 1] + prefix * alphas[M]
    for i in range(M - 2, -1, -1):
        alphas[i] = bs[i] + prefix * alphas[i + 1] - alphas[i + 2]
    S = 2 * (alphas[0] + alphas[1]) if M > 0 else 2 * alphas[0]
    return (x * (1 - x)) * S


def clenshaw_qbfs_der(cs, usq, j=1, alphas=None):
    x = usq
    M = len(cs) - 1
    if M < 0:
        return _initialize_alphas_q(cs, usq, alphas, j=j)
    prefix = 2 - 4 * x
    alphas = _initialize_alphas_q(cs, usq, alphas, j=j)
    clenshaw_qbfs(cs, usq, alphas[0])
    for jj in range(1, j + 1):
        if M - jj < 0: continue
        alphas[jj][M - j] = -4 * jj * alphas[jj - 1][M - j + 1]
        for n in range(M - 2, -1, -1):
            alphas[jj][n] = prefix * alphas[jj][n + 1] - alphas[jj][n + 2] - 4 * jj * alphas[jj - 1][n + 1]
    return alphas


def product_rule(u, v, du, dv):
    return u * dv + v * du


def compute_z_zprime_Qbfs(coefs, u, usq):
    alphas = clenshaw_qbfs_der(coefs, usq, j=1)
    S  = 2 * (alphas[0][0] + alphas[0][1]) if len(coefs) > 1 else 2 * alphas[0][0]
    # dS / d(u²)
    dS_dusq = (alphas[1][0] + alphas[1][1]) if len(coefs) > 1 else alphas[1][0]
    dS_dusq = dS_dusq * 4
    # convert to dS/du
    dS_du   = dS_dusq * 2 * u
    return S, dS_du


@lru_cache(4000)

def abc_q2d(n, m):
    D = (4 * n ** 2 - 1) * (m + n - 2) * (m + 2 * n - 3)
    if D == 0: D = 1e-99
    term1 = (2 * n - 1) * (m + 2 * n - 2)
    term2 = (4 * n * (m + n - 2) + (m - 3) * (2 * m - 1))
    A = (term1 * term2) / D
    num = -2 * (2 * n - 1) * (m + 2 * n - 3) * (m + 2 * n - 2) * (m + 2 * n - 1)
    B = num / D
    num = n * (2 * n - 3) * (m + 2 * n - 1) * (2 * m + 2 * n - 3)
    C = num / D
    return A, B, C


@lru_cache(4000)
def G_q2d(n, m):
    if n == 0:
        num = special.factorial2(2 * m - 1)
        den = 2 ** (m + 1) * special.factorial(m - 1)
        return num / den
    elif n > 0 and m == 1:
        t1num = (2 * n ** 2 - 1) * (n ** 2 - 1)
        t1den = 8 * (4 * n ** 2 - 1)
        term1 = -t1num / t1den
        term2 = 1 / 24 * kronecker(n, 1)
        return term1 - term2
    else:
        nt1 = 2 * n * (m + n - 1) - m
        nt2 = (n + 1) * (2 * m + 2 * n - 1)
        num = nt1 * nt2
        dt1 = (m + 2 * n - 2) * (m + 2 * n - 1)
        dt2 = (m + 2 * n) * (2 * n + 1)
        den = dt1 * dt2
        term1 = -num / den
        return term1 * gamma_func(n, m)


@lru_cache(4000)
def F_q2d(n, m):
    if n == 0 and m == 1:
        return 0.25
    if n == 0:
        num = m ** 2 * special.factorial2(2 * m - 3)
        den = 2 ** (m + 1) * special.factorial(m - 1)
        return num / den
    elif n > 0 and m == 1:
        t1num = 4 * (n - 1) ** 2 * n ** 2 + 1
        t1den = 8 * (2 * n - 1) ** 2
        term1 = t1num / t1den
        term2 = 11 / 32 * kronecker(n, 1)
        return term1 + term2
    else:
        Chi = m + n - 2
        nt1 = 2 * n * Chi * (3 - 5 * m + 4 * n * Chi)
        nt2 = m ** 2 * (3 - m + 4 * n * Chi)
        num = nt1 + nt2
        dt1 = (m + 2 * n - 3) * (m + 2 * n - 2)
        dt2 = (m + 2 * n - 1) * (2 * n - 1)
        den = dt1 * dt2
        term1 = num / den
        return term1 * gamma_func(n, m)


@lru_cache(4000)
def g_q2d(n, m):
    return G_q2d(n, m) / f_q2d(n, m)


@lru_cache(4000)
def f_q2d(n, m):
    if n == 0:
        return np.sqrt(F_q2d(n=0, m=m))
    else:
        return np.sqrt(F_q2d(n, m) - g_q2d(n - 1, m) ** 2)


def change_of_basis_Q2d_to_Pnm(cns, m):
    if m < 0:
        m = -m
    cs = cns
    ds = be.empty_like(be.array(cs))
    N = len(cs) - 1
    if N < 0:
        return ds
    ds[N] = cs[N] / f_q2d(N, m)
    for n in range(N - 1, -1, -1):
        ds[n] = (cs[n] - g_q2d(n, m) * ds[n + 1]) / f_q2d(n, m)

    return ds


@lru_cache(4000)
def abc_q2d_clenshaw(n, m):
    if m == 1:
        if n == 0: return 2, -1, 0
        if n == 1: return -4 / 3, -8 / 3, -11 / 3
        if n == 2: return 9 / 5, -24 / 5, 0
    if m == 2 and n == 0: return 3, -2, 0
    if m == 3 and n == 0: return 5, -4, 0
    return abc_q2d(n, m)


def clenshaw_q2d(cns, m, usq, alphas=None):
    x = usq
    ds = change_of_basis_Q2d_to_Pnm(cns, m)
    alphas = _initialize_alphas_q(ds, x, alphas, j=0)
    N = len(ds) - 1
    if N < 0:
        return alphas

    alphas[N] = ds[N]
    if N == 0:
        return alphas

    A, B, _ = abc_q2d_clenshaw(N - 1, m)
    alphas[N - 1] = ds[N - 1] + (A + B * x) * alphas[N]
    for n in range(N - 2, -1, -1):
        A, B, _ = abc_q2d_clenshaw(n, m)
        _, _, C = abc_q2d_clenshaw(n + 1, m)
        alphas[n] = ds[n] + (A + B * x) * alphas[n + 1] - C * alphas[n + 2]

    return alphas


def clenshaw_q2d_der(cns, m, usq, j=1, alphas=None):
    cs = cns
    x = usq
    N = len(cs) - 1
    alphas = _initialize_alphas_q(cs, x, alphas, j=j)
    if N < 0:
        return alphas
        
    clenshaw_q2d(cs, m, x, alphas[0])
    for jj in range(1, j + 1):
        if N - jj < 0: continue
        _, b, _ = abc_q2d_clenshaw(N - jj, m)
        alphas[jj][N - jj] = j * b * alphas[jj - 1][N - jj + 1]
        for n in range(N - jj - 1, -1, -1):
            a, b, _ = abc_q2d_clenshaw(n, m)
            _, _, c = abc_q2d_clenshaw(n + 1, m)
            alphas[jj][n] = jj * b * alphas[jj - 1][n + 1] + (a + b * x) * alphas[jj][n + 1] - c * alphas[jj][n + 2]

    return alphas


def compute_z_zprime_Q2d(cm0, ams, bms, u, t):
    usq = u * u
    z = be.zeros_like(u)
    dr = be.zeros_like(u)
    dt = be.zeros_like(u)

    if cm0 is not None and len(cm0) > 0:
        zm0, zprimem0 = compute_z_zprime_Qbfs(cm0, u, usq)
        z = z + zm0
        dr = dr + zprimem0

    m = 0
    for a_coef, b_coef in zip(ams, bms):
        m = m + 1
        if not a_coef and not b_coef:
            continue

        alphas_a = clenshaw_q2d_der(a_coef, m, usq, j=1) if a_coef else None
        alphas_b = clenshaw_q2d_der(b_coef, m, usq, j=1) if b_coef else None

        Sa, Sb, Sprimea, Sprimeb = 0, 0, 0, 0
        if alphas_a is not None:
            Sa = 0.5 * alphas_a[0][0]
            Sprimea = 0.5 * alphas_a[1][0]
            if m == 1 and len(a_coef) - 1 > 2:
                Sa = Sa - 2 / 5 * alphas_a[0][3]
                Sprimea = Sprimea - 2 / 5 * alphas_a[1][3]
        
        if alphas_b is not None:
            Sb = 0.5 * alphas_b[0][0]
            Sprimeb = 0.5 * alphas_b[1][0]
            if m == 1 and len(b_coef) - 1 > 2:
                Sb = Sb - 2 / 5 * alphas_b[0][3]
                Sprimeb = Sprimeb - 2 / 5 * alphas_b[1][3]

        um = u ** m
        cost = be.cos(m * t)
        sint = be.sin(m * t)

        kernel = cost * Sa + sint * Sb
        z = z + um * kernel

        umm1 = u ** (m - 1) if m > 0 else be.ones_like(u)
        twousq = 2 * usq
        aterm = cost * (twousq * Sprimea + m * Sa)
        bterm = sint * (twousq * Sprimeb + m * Sb)
        dr = dr + umm1 * (aterm + bterm)
        dt = dt + m * um * (-Sa * sint + Sb * cost)

    return z, dr, dt


def Q2d_nm_c_to_a_b(nms, coefs):
    def factory():
        return []

    def expand_and_copy(cs, N):
        cs2 = [None] * (N + 1)
        for i, cc in enumerate(cs):
            cs2[i] = cc
        return cs2

    cms = []
    ac = defaultdict(factory)
    bc = defaultdict(factory)

    for (n, m), c in zip(nms, coefs):
        if m == 0:
            if len(cms) < n + 1:
                cms = expand_and_copy(cms, n)
            cms[n] = c
        elif m > 0:
            if len(ac[m]) < n + 1:
                ac[m] = expand_and_copy(ac[m], n)
            ac[m][n] = c
        else:
            m = -m
            if len(bc[m]) < n + 1:
                bc[m] = expand_and_copy(bc[m], n)
            bc[m][n] = c

    for i, c in enumerate(cms):
        if c is None:
            cms[i] = 0
    for k in ac:
        for i, c in enumerate(ac[k]):
            if ac[k][i] is None:
                ac[k][i] = 0
    for k in bc:
        for i, c in enumerate(bc[k]):
            if bc[k][i] is None:
                bc[k][i] = 0

    max_m_a = max(list(ac.keys())) if ac else 0
    max_m_b = max(list(bc.keys())) if bc else 0
    max_m = max(max_m_a, max_m_b)
    ac_ret = []
    bc_ret = []
    for i in range(1, max_m + 1):
        ac_ret.append(ac[i])
        bc_ret.append(bc[i])
    return cms, ac_ret, bc_ret