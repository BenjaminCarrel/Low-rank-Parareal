
import numpy as np
import numpy.linalg as la
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp

from ivps.general import GeneralIVP
from ivps.sylvester import SylvesterIVP
import classical_solvers
from low_rank_toolbox.svd import SVD
from low_rank_toolbox import svd
from ivps.lyapunov import solve_sparse_low_rank_lyapunov


def scipy_dlra(problem: GeneralIVP,
               t_span: tuple,
               Y0: SVD,
               scipy_method: str = 'RK45') -> SVD:
    """DLRA solved by scipy for theoretical purposes

    Args:
        problem (GeneralIVP): problem to solve
        t_span (tuple): initial and final time
        Y0 (SVD): initial value in SVD format
    """
    # INITIALISATION
    rank = Y0.rank
    shape = Y0.shape
    y0 = Y0.todense().flatten()

    # DEFINE DLRA ODE
    def dlra_ode(t: tuple, y: ArrayLike) -> ArrayLike:
        Y = y.reshape(shape)
        Y_svd = svd.truncated_svd(Y, rank)
        DY = problem.ode_F(t, Y)
        PDY = Y_svd.project_onto_tangent_space(DY)
        pdy = PDY.todense().flatten()
        return pdy

    # SOLVE BY SCIPY
    sol = solve_ivp(dlra_ode, t_span, y0, method=scipy_method, atol=1e-13, rtol=1e-13)
    Y1 = np.reshape(sol.y[:, -1], shape)
    Y1_svd = svd.truncated_svd(Y1, rank)
    return Y1_svd

# EXACT LINEAR DLRA
def exact_dlra_linear(problem: SylvesterIVP,
                 t_span: tuple,
                 Y0: SVD) -> SVD:
    """Exact solver of DLRA for linear problems. 

    Args:
        problem (SylvesterIVP): Problem to solve
        t_span (tuple): Time interval
        Y0 (SVD): Initial value in SVD format
    """
    # INITIALIZATION
    rank = Y0.rank
    h = t_span[1] - t_span[0]
    A = problem.A
    spluA = problem.spluA

    # COMPUTE THE FIRST TERM
    TERM1 = problem.stiff_solution((0, h), Y0)

    # COMPUTE PROJECTION
    PGY0 = problem.non_linear_field(h, Y0)

    # COMPUTE SECOND TERM
    RHS = problem.stiff_solution((0, h), PGY0) - PGY0
    TERM2 = solve_sparse_low_rank_lyapunov(A, RHS, spluA, tol=1e-12, max_iter=10)

    # ASSEMBLE AND TRUNCATE
    Z = TERM1 + TERM2
    Y1 = Z.truncate(rank, inplace=False)
    return Y1


def KSL(problem: GeneralIVP,
        t_span: tuple,
        Y0: SVD,
        order: int = 2) -> SVD:
    """Projector splitting integrator, also called KSL.
    See Lubich & Osedelets 2013

    Args:
        problem (GeneralIVP): problem to solve.
        t_span (tuple): initial and final time.
        Y0 (SVD): initial value.
        order (int, optional): order of the scheme. Defaults to 2.
    """
    if order == 1:
        Y1 = KSL1(problem, t_span, Y0)

    elif order == 2:
        Y1 = KSL2(problem, t_span, Y0)
    
    else:
        raise ValueError('KSL of higher order than 2 is not implemented')
    
    return Y1

def KSL1(problem: GeneralIVP,
        t_span: tuple,
        Y0: SVD,
        scipy_method: str = 'optimal') -> SVD:
    """Projector splitting integrator of order 1.

    Args:
        problem (GeneralIVP): problem to solve
        t_span (tuple): one step time interval
        Y0 (SVD): initial value
    """
    # INITIALISATION
    U0, S0, V0 = Y0.U, Y0.S, Y0.Vt.T
    # K-STEP
    K0 = U0.dot(S0)
    problem.select_ode(K0, 'K', (V0,))
    K1 = classical_solvers.solve_one_step(problem, t_span, K0, scipy_method)
    U1, S_hat = la.qr(K1, mode='reduced')
    # S-STEP
    problem.select_ode(S_hat, 'minus_S', (U1, V0))
    S0_tilde = classical_solvers.solve_one_step(problem, t_span, S_hat, scipy_method)
    # L-STEP
    L0 = V0.dot(S0_tilde.T)
    problem.select_ode(L0, 'L', (U1,))
    L1 = classical_solvers.solve_one_step(problem, t_span, L0, scipy_method)
    V1, S1t = la.qr(L1, mode='reduced')
    S1 = S1t.T
    # SOLUTION
    Y1 = SVD(U1, S1, V1.T)
    return Y1

def KSL2(problem: GeneralIVP,
        t_span: tuple,
        Y0: SVD,
        scipy_method: str = 'optimal') -> SVD:
    """Projector splitting integrator of order 2.

    Args:
        problem (GeneralIVP): problem to solve
        t_span (tuple): one step time interval
        Y0 (SVD): initial value
    """
    # INITIALISATION
    U0, S0, V0 = Y0.U, Y0.S, Y0.Vt.T
    t0, t1 = t_span[0], t_span[1]
    t_half = (t1 - t0) / 2
    # FORWARD PASS
    K0 = U0.dot(S0)
    problem.select_ode(K0, 'K', (V0,))
    K1 = classical_solvers.solve_one_step(problem, (t0, t0+t_half), K0, scipy_method)
    U1, S_hat = la.qr(K1, mode='reduced')
    problem.select_ode(S_hat, 'minus_S', (U1, V0))
    S1 = classical_solvers.solve_one_step(problem, (t0, t0+t_half), S_hat, scipy_method)
    ## DOUBLE (III)
    L0 = V0.dot(S1.T)
    problem.select_ode(L0, 'L', (U1,))
    L2 = classical_solvers.solve_one_step(problem, (t0, t1), L0, scipy_method)
    V2, St_hat = la.qr(L2, mode='reduced')
    # BACKWARD PASS
    problem.select_ode(St_hat.T, 'minus_S', (U1, V2))
    S1_bis = classical_solvers.solve_one_step(problem, (t0+t_half, t1), St_hat.T, scipy_method)
    K1 = U1.dot(S1_bis)
    problem.select_ode(K1, 'K', (V2,))
    K2 = classical_solvers.solve_one_step(problem, (t0+t_half, t1), K1, scipy_method)
    U2, S2 = la.qr(K2, mode='reduced')
    # SOLUTION
    Y1 = SVD(U2, S2, V2.T)
    return Y1
