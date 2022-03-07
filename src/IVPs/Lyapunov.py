# %% IMPORTATIONS
import numpy as np
import numpy.linalg as la
import scipy as sp
import scipy.linalg as sla
import scipy.sparse.linalg as spala
from scipy.sparse import spmatrix
import classical_solvers
from numpy.typing import ArrayLike
from IVPs.Sylvester import SylvesterIVP
from low_rank_toolbox import svd
from low_rank_toolbox.svd import SVD
from typing import Union
import krylov

# %% CLASS LYAPUNOV
class LyapunovIVP(SylvesterIVP):
    """
    Subclass of SylvesterProblem. Specific for the Lyapunov equation.

    Lyapunov equation :
        .. math::
            \dot{Ys}(t) = A Ys(t) + Ys(t) A + C C^T.
    The initial value is given by :math:`Ys(t_0) = Y_0`
    """

    is_closed_form_available = True

    def __init__(self,
                 t_span: tuple,
                 Y0: SVD,
                 A: ArrayLike,
                 C: ArrayLike):
        # LYAPUNOV IS SYLVESTER WITH B=A
        SylvesterIVP.__init__(self, t_span, Y0, A, A, C)

    def closed_form_full_ode(self, t_span: tuple, X0: Union[ArrayLike, SVD]) -> Union[ArrayLike, SVD]:
        "Closed form solution computed more efficiently for Lyapunov"
        return closed_form_invertible_diff_lyapunov(t_span, X0, self.A, self.CCt, invA=self.spluA)
# %% LYAPUNOV RELATED METHODS
def solve_small_lyapunov(A: ArrayLike, RHS: ArrayLike) -> ArrayLike:
    "Shortcut for solve_lyapunov which is 2x faster than solve_sylvester"
    return sla.solve_lyapunov(A, RHS)


def solve_sparse_low_rank_lyapunov(A: spmatrix,
                                   RHS: SVD,
                                   invA: object = None,
                                   tol: float = 1e-8,
                                   max_iter: int = 100) -> SVD:
    """Low-rank solver for sylvester equation, with large matrix A. Find X such that AX + XA = RHS
    The method is based on Krylov Space for finding the solution with a criterion; see Simoncini.

    Args:
        A (spmatrix): matrix of shape (m,m)
        RHS (SVD): low-rank matrix of shape (n,n)
        invA (object, optional): inverse object of matrix A. Defaults to None.
        tol (float, optional): tolerance for solution. Defaults to 1e-8.
        max_iter (int, optional): maximal number of iterations. Defaults to 100.

    Returns:
        X (SVD): computed solution
    """
    # INITIALIZATION
    normA = spala.norm(A)
    normRHS = RHS.norm()
    KS = krylov.KrylovSpace(A, RHS._matrices[0], invA, inverted=True)
    V = KS.Q

    # SOLVE SMALL PROJECTED SYLVESTER IN LOOP
    for k in np.arange(1, max_iter):
        # SOLVE PROJECTED SYLVESTER Ak Y + Y Bk = Ck
        Ak = V.T.dot(A.dot(V))
        Ck = (RHS.dot(V)).dot(V.T, side="opposite")  # Vt @ RHS @ W
        Yk = sla.solve_lyapunov(Ak, Ck.full())

        # CHECK CONVERGENCE
        Xk = SVD(V, Yk, V.T)
        AXk = Xk.dot_sparse(A, side="opposite")
        XkA = Xk.dot_sparse(A)
        # computation of crit could be more efficient, but SVD so its OK.
        crit = svd.add_svd([RHS, -AXk, -XkA]).norm() / \
            (2 * normA * la.norm(Yk) + normRHS)
        # print(crit)
        if crit < tol:
            return Xk
        else:
            KS.compute_next_basis()
            V = KS.Q

    print('No convergence before max_iter')
    X = SVD(V, Yk, V.T)
    return X


def closed_form_invertible_diff_lyapunov(t_span: tuple, X0: SVD, A: spmatrix, C: SVD, invA: object = None) -> SVD:
    """Closed form of the differential Lyapunov equation.

    Args:
        A (spmatrix): sparse matrix of shape (n,n)
        RHS (SVD): low-rank matrix of shape (n,n)
        invA (object, optional): inverse object of matrix A. Defaults to None.
    """
    t0, t1 = t_span
    h = t1 - t0
    EX = X0.expm_multiply(A, h, side='right').expm_multiply(A, h, side='left')
    EC = C.expm_multiply(A, h, side='right').expm_multiply(A, h, side='left')
    Z = solve_sparse_low_rank_lyapunov(A, EC - C)
    X1 = EX + Z
    return X1

# %% MAKE PROBLEMS
def make_heat_problem(t_span: tuple = (0.01, 2.01),
                       n: int = 100,
                       initial_rank: int = 21,
                       source_rank: int = 5) -> LyapunovIVP:
    """
    Stiff version of Lyapunov time with tridiagonal matrix.
    Initial X0 is random with 10^-i singular values.

    Equation is: X' = AX + XA + C C^T
    """
    # 1D Laplacian as stencil 1/dx^2 [1 -2 1] in csc format
    x = np.linspace(-1, 1, num=n)
    dx = x[1] - x[0]
    ones = np.ones(n)
    data = 1 / (dx ** 2) * np.array([ones, -2 * ones, ones])
    diags = np.array([-1, 0, 1])
    A = sp.sparse.spdiags(data, diags, n, n, format="csc")

    # X0 with decaying singular values
    np.random.seed(1111)
    # if initial_rank=20, numerical initial_rank is about 16
    G0 = np.random.rand(n, initial_rank)
    H0 = np.random.rand(n, initial_rank)
    U0, _ = sp.linalg.qr(G0, mode="economic")
    V0, _ = sp.linalg.qr(H0, mode="economic")
    s0 = np.logspace(0, -20, initial_rank)
    S0 = np.diag(s0)
    # x = np.sin(x.reshape(50,1))
    # y = np.cos(x.reshape(50,1))
    # X0 = x.dot(y.T)
    # U0, S0, V0 = np.linalg.svd(X0, full_matrices=False)
    X0 = SVD(U0, S0, V0.T)

    # C with decaying singular values
    np.random.seed(2222)
    C = np.random.rand(n, source_rank)
    U1, _ = sp.linalg.qr(C, mode="economic")
    s1 = np.logspace(0, -20, source_rank)
    S1 = np.diag(s1)
    C = U1.dot(S1)

    # Create a Lyapunov Problem
    stiff_problem = LyapunovIVP(t_span, X0, A, C)

    # Change initial value
    if t_span[0] != 0:
        new_X0 = classical_solvers.solve_one_step(stiff_problem, (0, t_span[0]), X0)
        # u, s, vh = np.linalg.svd(new_X0, full_matrices=False)
        # new_X0 = SVD(u, s, vh)
        stiff_problem = LyapunovIVP(t_span, new_X0, A, C)

    return stiff_problem


def make_nonstiff_problem(t_span: tuple = (0, 1),
                          n: int = 50,
                          initial_rank: int = 20,
                          source_rank: int = 5) -> LyapunovIVP:
    """
    Non-stiff version of Lyapunov time with tridiagonal matrix.
    Initial X0 is random with 10^-i singular values.

    Equation is: X' = AX + XA + C C^T
    """
    # tridiagonal matrix [1/2 -1 1/2] in csc format
    ones = np.ones(n)
    data = (-0.5) * np.array([ones, -2 * ones, ones])
    diags = np.array([-1, 0, 1])
    A = sp.sparse.spdiags(data, diags, n, n, format="csc")

    # X0 with decaying singular values
    np.random.seed(1111)
    # if initial_rank=20, numerical initial_rank is about 16
    G0 = np.random.rand(n, initial_rank)
    H0 = np.random.rand(n, initial_rank)
    U0, _ = sp.linalg.qr(G0, mode="economic")
    V0, _ = sp.linalg.qr(H0, mode="economic")
    s0 = np.sqrt(np.logspace(-1, -21, initial_rank))
    X0 = SVD(U0, s0, V0.T)

    # RHS matrix C
    np.random.seed(2222)
    C = np.random.rand(n, source_rank)
    U1, _ = sp.linalg.qr(C, mode="economic")
    s1 = np.logspace(-1, -21, source_rank)
    S1 = np.diag(np.sqrt(s1))
    C = U1.dot(S1)

    # Create a Lyapunov Problem
    nonstiff_problem = LyapunovIVP(t_span, X0, A, C)

    # Change initial value
    if t_span[0] != 0:
        new_X0 = classical_solvers.solve_one_step(
            nonstiff_problem, (0, t_span[0]), X0.full())
        u, s, vh = np.linalg.svd(new_X0, full_matrices=False)
        new_X0 = SVD(u, s, vh)
        nonstiff_problem = LyapunovIVP(t_span, new_X0, A, C)

    return nonstiff_problem


#%% BOUNDS
def theoretical_bound(t: float, 
                      A: spmatrix, 
                      C: ArrayLike, 
                      X0: SVD, 
                      rho: int, 
                      r: int, 
                      r0: int) -> float:
    """Theoretical bound for the best rank rho approximation of the Lyapunov solution

    Args:
        t (float): time
        A (spmatrix): matrix A of the problem
        C (ArrayLike): matrix C of the problem
        X0 (ArrayLike): initial value
        rho (int): rank of approximation
        r (int): rank approximation of the source term
        r0 (int): rank approximation of the initial value
    """
    # extract data
    ell = 2 * np.max(la.eigvals(A.todense()))
    kappa_A = la.cond(A.todense(), p=2)
    CCt = C.dot(C.T)
    s_C = la.svd(CCt, full_matrices=True, compute_uv=False)
    norm_CCt = la.norm(CCt, ord=2)
    s_X0 = la.svd(X0.todense(), full_matrices=True, compute_uv=False)
    
    
    # first term
    sigma0 = s_X0[r0]
    term1 = np.exp(ell*t)*sigma0
    
    # second term
    sigmaC = s_C[r]
    term2 = (np.exp(t*ell)-1)/ell * (4 * np.exp(-np.pi**2 * rho/np.log(4*kappa_A))*norm_CCt + sigmaC)
    
    return term1 + term2