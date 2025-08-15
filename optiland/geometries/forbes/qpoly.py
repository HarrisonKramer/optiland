"""
Tools for working with Q (Forbes) polynomials.


code adapted in its majority from the prysm package - (https://github.com/brandondube/prysm)
Manuel Fragata Mendes, 2025

Copyright notice:
Copyright (c) 2017 Brandon Dube
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

from scipy import special

import optiland.backend as be


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
        return -1 / 2
    else:
        n_minus_2 = n_minus_1 - 1
        return -(1 + g_qbfs(n_minus_2) * h_qbfs(n_minus_2)) / f_qbfs(n_minus_1)


@lru_cache(1000)
def h_qbfs(n_minus_2):
    n = n_minus_2 + 2
    return -n * (n - 1) / (2 * f_qbfs(n_minus_2))


@lru_cache(1000)
def f_qbfs(n):
    if n == 0:
        return 2
    elif n == 1:
        return be.sqrt(19) / 2
    else:
        term1 = n * (n + 1) + 3
        term2 = g_qbfs(n - 1) ** 2
        term3 = h_qbfs(n - 2) ** 2
        return be.sqrt(term1 - term2 - term3)


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
        shape = (len(cs), *be.shape(x)) if hasattr(x, "shape") else (len(cs),)
        if j != 0:
            shape = (j + 1, *shape)
        if be.get_backend() == "torch":
            alphas = be.zeros(shape)
            alphas.requires_grad = False
        else:
            alphas = be.zeros(shape)
    return alphas


def _clenshaw_qbfs_functional(bs, usq):
    """
    Pure-functional Clenshaw that returns (S, alpha0, alpha1).
    This version is fixed to handle broadcasting correctly.
    """
    M = len(bs) - 1
    prefix = 2 - 4 * usq

    if M < 0:
        zeros = be.zeros_like(usq)
        return zeros, zeros, zeros

    b_curr = bs[M] + usq * 0
    b_next = be.zeros_like(b_curr)

    for n in range(M - 1, -1, -1):
        b_new = bs[n] + prefix * b_curr - b_next
        b_next, b_curr = b_curr, b_new

    alpha0, alpha1 = b_curr, b_next
    S = 2 * (alpha0 + alpha1) if M > 0 else 2 * alpha0
    return S, alpha0, alpha1


def clenshaw_qbfs(cs, usq, alphas=None):
    """
    Computes the sum of Q-BFS polynomials.
    NOTE: This function returns the raw polynomial sum. The u^2(1-u^2) prefactor
    is applied by the geometry's sag method.
    """
    if be.get_backend() == "torch":
        bs = change_basis_Qbfs_to_Pn(cs)
        S, alpha0, _ = _clenshaw_qbfs_functional(bs, usq)

        if alphas is not None:
            M = len(bs) - 1
            fill = []
            if M >= 0:
                prefix = 2 - 4 * usq
                alphas_list = [be.zeros_like(alpha0) for _ in range(M + 1)]
                alphas_list[M] = bs[M] + usq * 0
                if M > 0:
                    alphas_list[M - 1] = bs[M - 1] + prefix * alphas_list[M]
                for i in range(M - 2, -1, -1):
                    alphas_list[i] = (
                        bs[i] + prefix * alphas_list[i + 1] - alphas_list[i + 2]
                    )
                fill = alphas_list
            if fill:
                alphas[...] = be.stack(fill)
        return S

    # ─ NumPy backend – keep original fast in-place path ─
    x = usq
    bs = change_basis_Qbfs_to_Pn(cs)
    alphas = _initialize_alphas_q(cs, x, alphas, j=0)
    M = len(bs) - 1
    if M < 0:
        return be.zeros_like(x) if hasattr(x, "shape") else 0.0

    prefix = 2 - 4 * x
    alphas[M] = bs[M]
    if M > 0:
        alphas[M - 1] = bs[M - 1] + prefix * alphas[M]
    for i in range(M - 2, -1, -1):
        alphas[i] = bs[i] + prefix * alphas[i + 1] - alphas[i + 2]

    S = 2 * (alphas[0] + alphas[1]) if M > 0 else 2 * alphas[0]
    return S


def _clenshaw_qbfs_der_functional(cs, usq, j=1):
    """Pure-functional Clenshaw for Q-BFS derivatives (PyTorch backend)."""
    M = len(cs) - 1
    if M < 0:
        shape = (
            (j + 1, len(cs), *be.shape(usq))
            if hasattr(usq, "shape")
            else (j + 1, len(cs))
        )
        return be.zeros(shape)

    prefix = 2 - 4 * usq
    bs = change_basis_Qbfs_to_Pn(cs)

    # j=0 part (the polynomial recurrence values)
    alphas_j0_list = [be.zeros_like(usq) for _ in range(M + 1)]
    if M >= 0:
        alphas_j0_list[M] = bs[M] + usq * 0
    if M >= 1:
        alphas_j0_list[M - 1] = bs[M - 1] + prefix * alphas_j0_list[M]
    for i in range(M - 2, -1, -1):
        alphas_j0_list[i] = (
            bs[i] + prefix * alphas_j0_list[i + 1] - alphas_j0_list[i + 2]
        )

    all_alphas_tensors = [be.stack(alphas_j0_list)]

    prev_alphas_j_list = alphas_j0_list
    for jj in range(1, j + 1):
        alphas_jj_list = [be.zeros_like(usq) for _ in range(M + 1)]
        if M - jj >= 0:
            alphas_jj_list[M - jj] = -4 * jj * prev_alphas_j_list[M - jj + 1]
        if M - jj - 1 >= 0:
            alphas_jj_list[M - jj - 1] = (
                prefix * alphas_jj_list[M - jj] - 4 * jj * prev_alphas_j_list[M - jj]
            )
        for n in range(M - jj - 2, -1, -1):
            alphas_jj_list[n] = (
                prefix * alphas_jj_list[n + 1]
                - alphas_jj_list[n + 2]
                - 4 * jj * prev_alphas_j_list[n + 1]
            )

        all_alphas_tensors.append(be.stack(alphas_jj_list))
        prev_alphas_j_list = alphas_jj_list

    return be.stack(all_alphas_tensors)


def clenshaw_qbfs_der(cs, usq, j=1, alphas=None):
    """Computes derivatives of Q-BFS polynomials using Clenshaw's method."""
    if be.get_backend() == "torch":
        return _clenshaw_qbfs_der_functional(cs, usq, j)

    # --- NumPy backend – keep original fast in-place path ---
    x = usq
    M = len(cs) - 1
    if M < 0:
        return _initialize_alphas_q(cs, usq, alphas, j=j)

    prefix = 2 - 4 * x
    alphas = _initialize_alphas_q(cs, usq, alphas, j=j)

    # Populate alphas[0] using the in-place buffer argument of clenshaw_qbfs
    clenshaw_qbfs(cs, usq, alphas=alphas[0])

    for jj in range(1, j + 1):
        if M - jj < 0:
            continue
        alphas[jj][M - jj] = -4 * jj * alphas[jj - 1][M - jj + 1]
        if M - jj - 1 >= 0:
            alphas[jj][M - jj - 1] = (
                prefix * alphas[jj][M - jj] - 4 * jj * alphas[jj - 1][M - jj]
            )
        for n in range(M - jj - 2, -1, -1):
            alphas[jj][n] = (
                prefix * alphas[jj][n + 1]
                - alphas[jj][n + 2]
                - 4 * jj * alphas[jj - 1][n + 1]
            )

    return alphas


def compute_z_zprime_Qbfs(coefs, u, usq):
    """
    Computes the raw Q-BFS polynomial sum and its derivative w.r.t. u.
    This version contains the corrected derivative calculation.
    """
    if coefs is None or len(coefs) == 0:
        zeros = be.zeros_like(u)
        return zeros, zeros

    alphas = clenshaw_qbfs_der(coefs, usq, j=1)

    # The sum S is 2*(alpha_0 + alpha_1)
    # The derivative S' w.r.t. usq is 2*(alpha_0' + alpha_1')
    if len(coefs) > 1:
        S = 2 * (alphas[0][0] + alphas[0][1])
        dS_dusq = 2 * (alphas[1][0] + alphas[1][1])
    else:
        S = 2 * alphas[0][0]
        dS_dusq = 2 * alphas[1][0]

    # dS/du = dS/dusq * d(usq)/du
    # d(usq)/du = d(u^2)/du = 2*u
    dS_du = dS_dusq * 2 * u
    return S, dS_du


@lru_cache(4000)
def abc_q2d(n, m):
    D = (4 * n**2 - 1) * (m + n - 2) * (m + 2 * n - 3)
    if D == 0:
        D = 1e-99
    term1 = (2 * n - 1) * (m + 2 * n - 2)
    term2 = 4 * n * (m + n - 2) + (m - 3) * (2 * m - 1)
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
        t1num = (2 * n**2 - 1) * (n**2 - 1)
        t1den = 8 * (4 * n**2 - 1)
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
        num = m**2 * special.factorial2(2 * m - 3)
        den = 2 ** (m + 1) * special.factorial(m - 1)
        return num / den
    elif n > 0 and m == 1:
        t1num = 4 * (n - 1) ** 2 * n**2 + 1
        t1den = 8 * (2 * n - 1) ** 2
        term1 = t1num / t1den
        term2 = 11 / 32 * kronecker(n, 1)
        return term1 + term2
    else:
        Chi = m + n - 2
        nt1 = 2 * n * Chi * (3 - 5 * m + 4 * n * Chi)
        nt2 = m**2 * (3 - m + 4 * n * Chi)
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
        return be.sqrt(F_q2d(n=0, m=m))
    else:
        return be.sqrt(F_q2d(n, m) - g_q2d(n - 1, m) ** 2)


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
        if n == 0:
            return 2, -1, 0
        if n == 1:
            return -4 / 3, -8 / 3, -11 / 3
        if n == 2:
            return 9 / 5, -24 / 5, 0
    if m == 2 and n == 0:
        return 3, -2, 0
    if m == 3 and n == 0:
        return 5, -4, 0
    return abc_q2d(n, m)


def _clenshaw_q2d_functional(ds, m, usq):
    """Pure-functional Clenshaw for Q2D polynomials."""
    x = usq
    N = len(ds) - 1
    if N < 0:
        return []

    all_alphas = [be.zeros_like(x) for _ in range(N + 1)]

    if N >= 0:
        all_alphas[N] = ds[N] + x * 0

    if N >= 1:
        A, B, _ = abc_q2d_clenshaw(N - 1, m)
        all_alphas[N - 1] = ds[N - 1] + (A + B * x) * all_alphas[N]

    for n in range(N - 2, -1, -1):
        A, B, _ = abc_q2d_clenshaw(n, m)
        _, _, C = abc_q2d_clenshaw(n + 1, m)
        all_alphas[n] = ds[n] + (A + B * x) * all_alphas[n + 1] - C * all_alphas[n + 2]

    return all_alphas


def clenshaw_q2d(cns, m, usq, alphas=None):
    if be.get_backend() == "torch":
        ds = change_of_basis_Q2d_to_Pnm(cns, m)
        all_alphas_list = _clenshaw_q2d_functional(ds, m, usq)

        if alphas is not None and all_alphas_list:
            alphas[...] = be.stack(all_alphas_list)
        return alphas

    # --- NumPy backend – keep original fast in-place path ---
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


def _clenshaw_q2d_der_functional(cns, m, usq, j=1):
    """Pure-functional Clenshaw for Q-2D derivatives (PyTorch backend)."""
    N = len(cns) - 1
    x = usq

    if N < 0:
        shape = (
            (j + 1, len(cns), *be.shape(usq))
            if hasattr(usq, "shape")
            else (j + 1, len(cns))
        )
        return be.zeros(shape)

    ds = change_of_basis_Q2d_to_Pnm(cns, m)

    # j=0 part
    alphas_j0_list = _clenshaw_q2d_functional(ds, m, usq)
    all_alphas_tensors = [be.stack(alphas_j0_list)]

    prev_alphas_j_list = alphas_j0_list
    for jj in range(1, j + 1):
        alphas_jj_list = [be.zeros_like(x) for _ in range(N + 1)]
        if N - jj >= 0:
            _, b, _ = abc_q2d_clenshaw(N - jj, m)
            alphas_jj_list[N - jj] = jj * b * prev_alphas_j_list[N - jj + 1]
            for n in range(N - jj - 1, -1, -1):
                a, b, _ = abc_q2d_clenshaw(n, m)
                _, _, c = abc_q2d_clenshaw(n + 1, m)
                alphas_jj_list[n] = (
                    jj * b * prev_alphas_j_list[n + 1]
                    + (a + b * x) * alphas_jj_list[n + 1]
                    - c * alphas_jj_list[n + 2]
                )

        all_alphas_tensors.append(be.stack(alphas_jj_list))
        prev_alphas_j_list = alphas_jj_list

    return be.stack(all_alphas_tensors)


def clenshaw_q2d_der(cns, m, usq, j=1, alphas=None):
    """Computes derivatives of Q-2D polynomials using Clenshaw's method."""
    if be.get_backend() == "torch":
        return _clenshaw_q2d_der_functional(cns, m, usq, j)

    # --- NumPy backend – keep original fast in-place path ---
    cs = cns
    x = usq
    N = len(cs) - 1
    alphas = _initialize_alphas_q(cs, x, alphas, j=j)
    if N < 0:
        return alphas

    clenshaw_q2d(cs, m, x, alphas[0])
    for jj in range(1, j + 1):
        if N - jj < 0:
            continue
        _, b, _ = abc_q2d_clenshaw(N - jj, m)
        alphas[jj][N - jj] = jj * b * alphas[jj - 1][N - jj + 1]
        for n in range(N - jj - 1, -1, -1):
            a, b, _ = abc_q2d_clenshaw(n, m)
            _, _, c = abc_q2d_clenshaw(n + 1, m)
            alphas[jj][n] = (
                jj * b * alphas[jj - 1][n + 1]
                + (a + b * x) * alphas[jj][n + 1]
                - c * alphas[jj][n + 2]
            )

    return alphas


def compute_z_zprime_Q2d(cm0, ams, bms, u, t):
    """
    Computes the polynomial sum components for a Q2D surface.
    """
    usq = u * u
    zeros = be.zeros_like(u)

    # --- m=0 component ---
    poly_sum_m0, d_poly_sum_m0_du = zeros, zeros
    if cm0 is not None and any(c != 0 for c in cm0):
        poly_sum_m0, d_poly_sum_m0_du = compute_z_zprime_Qbfs(cm0, u, usq)

    # --- m>0 components ---
    poly_sum_m_gt0 = be.zeros_like(u)
    dr_m_gt0 = be.zeros_like(u)
    dt_m_gt0 = be.zeros_like(u)

    m = 0
    for a_coef, b_coef in zip(ams, bms, strict=False):
        m = m + 1
        has_a = a_coef is not None and any(c != 0 for c in a_coef)
        has_b = b_coef is not None and any(c != 0 for c in b_coef)
        if not has_a and not has_b:
            continue

        alphas_a = clenshaw_q2d_der(a_coef, m, usq, j=1) if has_a else None
        alphas_b = clenshaw_q2d_der(b_coef, m, usq, j=1) if has_b else None

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

        um = u**m
        cost = be.cos(m * t)
        sint = be.sin(m * t)

        kernel = cost * Sa + sint * Sb
        poly_sum_m_gt0 = poly_sum_m_gt0 + um * kernel

        umm1 = u ** (m - 1) if m > 0 else be.ones_like(u)
        twousq = 2 * usq

        aterm = cost * (twousq * Sprimea + m * Sa)
        bterm = sint * (twousq * Sprimeb + m * Sb)
        dr_m_gt0 = dr_m_gt0 + umm1 * (aterm + bterm)
        dt_m_gt0 = dt_m_gt0 + m * um * (-Sa * sint + Sb * cost)

    return poly_sum_m0, d_poly_sum_m0_du, poly_sum_m_gt0, dr_m_gt0, dt_m_gt0


def Q2d_nm_c_to_a_b(nms, coefs):
    def factory():
        return []

    def expand_and_copy(cs, N):
        cs2 = [0.0] * (N + 1)
        for i, cc in enumerate(cs):
            cs2[i] = cc
        return cs2

    cms = []
    ac = defaultdict(factory)
    bc = defaultdict(factory)

    for (n, m), c in zip(nms, coefs, strict=False):
        if m == 0:
            if len(cms) < n + 1:
                cms = expand_and_copy(cms, n)
            cms[n] = c
        elif m > 0:
            if len(ac[m]) < n + 1:
                ac[m] = expand_and_copy(ac[m], n)
            ac[m][n] = c
        else:  # m < 0
            m_abs = -m
            if len(bc[m_abs]) < n + 1:
                bc[m_abs] = expand_and_copy(bc[m_abs], n)
            bc[m_abs][n] = c

    # Fill in missing zero coefficients
    max_n_cm0 = max([n for (n, m) in nms if m == 0] or [-1])
    if len(cms) < max_n_cm0 + 1:
        cms = expand_and_copy(cms, max_n_cm0)

    max_m_a = max([m for (n, m) in nms if m > 0] or [0])
    max_m_b = max([abs(m) for (n, m) in nms if m < 0] or [0])
    max_m = max(max_m_a, max_m_b)

    ac_ret = []
    bc_ret = []
    for i in range(1, max_m + 1):
        max_n_a = max([n for (n, m) in nms if m == i] or [-1])
        max_n_b = max([n for (n, m) in nms if m == -i] or [-1])

        a_list = ac.get(i, [])
        if len(a_list) < max_n_a + 1:
            a_list = expand_and_copy(a_list, max_n_a)
        ac_ret.append(a_list)

        b_list = bc.get(i, [])
        if len(b_list) < max_n_b + 1:
            b_list = expand_and_copy(b_list, max_n_b)
        bc_ret.append(b_list)

    return cms, ac_ret, bc_ret
