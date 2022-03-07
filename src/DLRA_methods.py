from typing import Union
import numpy as np
import numpy.linalg as la
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp
import copy

from IVPs.General import GeneralIVP
from IVPs.Lyapunov import LyapunovIVP
from IVPs.Riccati import RiccatiIVP
from RK_table import RK_rule
import classical_solvers
from low_rank_toolbox.svd import SVD
from low_rank_toolbox import svd


def scipy_dlra(problem: GeneralIVP,
               t_span: tuple,
               Y0: SVD) -> SVD:
    """ DLRA solved by scipy for theoretical purposes """
    # INITIALISATION
    shape = Y0.shape
    rank = Y0.rank
    y0 = Y0.todense().flatten()
    projected_vec_ode = problem.projected_vec_ode
    
    # DEFINE DLRA ODE
    def dlra_ode(t: tuple, y: ArrayLike) -> ArrayLike:
        Y = y.reshape(shape)
        Y_svd = svd.truncated_svd(Y, rank)
        DY = problem.ode_F(t, Y)
        PDY = Y_svd.project_onto_tangent_space(DY).truncate(rank)
        pdy = PDY.todense().flatten()
        return pdy
    
    # SOLVE BY SCIPY
    sol = solve_ivp(dlra_ode, t_span, y0, 'RK45', atol=1e-13, rtol=1e-13)
    Y1 = np.reshape(sol.y[:, -1], shape)
    Y1_svd = svd.truncated_svd(rank)
    return Y1_svd
    

def KSL1(problem: GeneralIVP,
         t_span: tuple,
         Y0: SVD) -> SVD:
    """
    Projector Splitting Integrator (KSL scheme) for DLRA, order 1.

    See Lubich & Osedelets 2013
    """
    # INTIALISATION
    U0, S0, V0 = Y0.U, Y0.S, Y0.Vt.T

    # K SUBSTEP
    K0 = U0.dot(S0)
    problem.select_ode(K0, 'K', (V0,))
    K1 = classical_solvers.solve_one_step(problem, t_span, K0)
    U1, S_hat = la.qr(K1, mode='reduced')

    # S SUBSTEP
    problem.select_ode(S_hat, 'minus_S', (U1, V0))
    S0_tilde = classical_solvers.solve_one_step(problem, t_span, S_hat)

    # L SUBSTEP
    L0 = V0.dot(S0_tilde.T)
    problem.select_ode(L0, 'L', (U1,))
    L1 = classical_solvers.solve_one_step(problem, t_span, L0)
    V1, S1t = la.qr(L1, mode='reduced')
    S1 = S1t.T

    # SOLUTION
    Y1 = SVD(U1, S1, V1.T)
    return Y1


def KSL2(problem: GeneralIVP,
         t_span: tuple,
         Y0: SVD) -> SVD:
    """
    Projector Splitting Integrator (KSL scheme) for DLRA, order 2.

    See Lubich & Osedelets 2013
    """
    # INTIALISATION
    t0 = t_span[0]
    t1 = t_span[1]
    t_half = (t0 + t1) / 2
    U0, S0, V0 = Y0.U, Y0.S, Y0.Vt.T

    # FORWARD PASS
    K0 = U0.dot(S0)
    problem.select_ode(K0, 'K', (V0,))
    K1 = classical_solvers.solve_one_step(problem, (t0, t_half), K0)
    U1, S_hat = la.qr(K1, mode='reduced')

    problem.select_ode(S_hat, 'minus_S', (U1, V0))
    S1 = classical_solvers.solve_one_step(problem, (t0, t_half), S_hat)

    ## DOUBLE (III)
    L0 = V0.dot(S1.T)
    problem.select_ode(L0, 'L', (U1,))
    L2 = classical_solvers.solve_one_step(problem, (t0, t1), L0)
    V2, St_hat = la.qr(L2, mode='reduced')

    # BACKWARD PASS
    problem.select_ode(St_hat.T, 'minus_S', (U1, V2))
    S1_bis = classical_solvers.solve_one_step(problem, (t_half, t1), St_hat.T)

    K1 = U1.dot(S1_bis)
    problem.select_ode(K1, 'K', (V2,))
    K2 = classical_solvers.solve_one_step(problem, (t_half, t1), K1)
    U2, S2 = la.qr(K2, mode='reduced')

    # SOLUTION
    Y1 = SVD(U2, S2, V2.T)
    return Y1


def unconventional(problem: GeneralIVP,
                   t_span: tuple,
                   Y0: SVD) -> SVD:
    """
    Unconventional integrator for DLRA, order 1.

    See Ceruti & Lubich 2020.
    """
    # INITIALIZATION
    U0, S0, V0 = Y0.U, Y0.S, Y0.Vt.T

    # K SUBSTEP
    K0 = U0.dot(S0)
    problem.select_ode(K0, 'K', (V0,))
    K1 = classical_solvers.solve_one_step(problem, t_span, K0)
    (U1, _) = la.qr(K1, mode='reduced')
    M = U1.T.dot(U0)

    # L SUBSTEP
    L0 = V0.dot(S0.T)
    problem.select_ode(L0, 'L', (U0,))
    L1 = classical_solvers.solve_one_step(problem, t_span, L0)
    V1, _ = la.qr(L1, mode='reduced')
    N = V1.T.dot(V0)

    # S SUBSTEP
    S0 = M.dot(S0.dot(N.T))
    problem.select_ode(S0, 'S', (U1, V1))
    S1 = classical_solvers.solve_one_step(problem, t_span, S0)

    # SOLUTION
    Y1 = SVD(U1, S1, V1)

    return Y1


def PRK(problem: GeneralIVP,
        t_span: tuple,
        Y0: SVD,
        order: int = 2) -> SVD:
    """
    Projected Runge Kutta integrators for DLRA.

    See Kieri and Vandereycken 2019.
    """
    # CREATE VARIABLES
    a, b, c = RK_rule(order)
    h = t_span[1] - t_span[0]
    r = Y0.rank
    eta = np.empty(order, dtype=object)
    kappa = np.empty(order, dtype=object)

    # PRK METHOD
    eta[0] = Y0
    for j in range(order):
        if j > 0:
            full_eta =  (Y0 + h * np.sum([a[j, i] * kappa[i]for i in np.arange(0, j)]))
            eta[j] = full_eta.truncate(r)
        kappa[j] = problem.project_ode_F(eta[j])
    Y1 = (Y0 + h * np.sum([b[i] * kappa[i] for i in range(order)])).truncate(r)
    return Y1


def strang_splitting(problem: Union[LyapunovIVP, RiccatiIVP],
                     t_span: tuple,
                     Y0: SVD,
                     order: int = 2) -> SVD:
    """Strang splitting for a stiff problem; see Ostermann et al. 2019.
    The problem has form $X' = AX + XA^* + G(t,X)$ where the non linear part $G$ is non stiff.

    Args:
        problem (GeneralIVP): problem to solve
        t_span (tuple): time interval
        Y0 (SVD): initial value
        order (int, optional): Order of the method, Lie-Trotter or Strang splitting. Defaults to 2.
    """
    # INTEGRATOR OF STIFF PART : X' = A X + X A^T
    h = t_span[1] - t_span[0]
    A = problem.A

    # LIE-TROTTER SPLITTING
    if order == 1:
        Y_half = problem.stiff_solution((0, h), Y0)
        try:
            Y1 = problem.non_stiff_solution((0, h), Y_half)
        except:
            # WORKS FOR RICCATI ONLY RIGHT NOW
            PB_nonstiff = copy.deepcopy(problem)
            PB_nonstiff.A = A * 0
            Y1 = KSL2(PB_nonstiff, t_span, Y_half)
        return Y1

    # STRANG SPLITTING
    if order == 2:
        Y_one = problem.stiff_solution((0, h/2), Y0)
        try:
            Y_two = problem.non_stiff_solution((0, h), Y_one)
        except:
            # WORKS FOR RICCATI ONLY RIGHT NOW
            PB_nonstiff = copy.deepcopy(problem)
            PB_nonstiff.A = A * 0
            Y_two = KSL2(PB_nonstiff, (0, h), Y_one)
        Y1 = problem.stiff_solution((0, h/2), Y_two)
        return Y1

    else:
        raise ValueError('Choose order 1 or 2')
