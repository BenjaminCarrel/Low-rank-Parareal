# %% IMPORTATIONS
import numpy as np
import numpy.linalg as la
import scipy as sp
from scipy.sparse import spmatrix
import classical_solvers
from numpy.typing import ArrayLike
from IVPs.General import GeneralIVP
from low_rank_toolbox import svd
from low_rank_toolbox.low_rank_matrix import LowRankMatrix


# %% COOKIE CLASS
class CookieIVP(GeneralIVP):
    r"""Subclass of GeneralIVP. Specific to the Cookie problem.

    Cookie equation :
    .. math::
        \dot{Ys}(t) = -A_0 Ys(t) - A_1 Ys(t) C1 + B B^T.
    The initial value is given by :math:`Ys(t_0) = Y_0`
    """
    # ATTRIBUTES
    is_closed_form_available = True

    # %% INIT FUNCTION
    def __init__(
        self,
        t_span: tuple,
        Y0: svd.SVD,
        A0: ArrayLike,
        A1: ArrayLike,
        C1: ArrayLike,
        B: ArrayLike,
    ):
        # SPECIFIC PROPERTIES
        self.A0 = A0
        self.A1 = A1
        self.C1 = C1
        self.B = B
        self.one = np.ones((C1.shape[0], 1))

        # PRECOMPUTATIONS
        (self.L_C1, self.Q_C1) = sp.linalg.eigh(C1)

        # INITIALIZATION
        GeneralIVP.__init__(self, None, t_span, Y0)

    # %% SELECTION OF ODE
    def select_ode(self,
                   initial_value: svd.SVD,
                   ode_type: str,
                   mats_UV: tuple = ()):
        "Make precomputations to accelerate the solution of the selected ode"
        # SELECT ODE
        GeneralIVP.select_ode(self, initial_value, ode_type, mats_UV)
        A0, A1, B, C1, one = self.A0, self.A1, self.B, self.C1, self.one
        # PRECOMPUTATIONS
        if ode_type == "F":
            self.D = B.dot(one.T)
        elif ode_type == "K":
            (V,) = mats_UV
            self.V = V
            self.VtC1V = np.linalg.multi_dot([V.T, C1, V])
            self.D = np.linalg.multi_dot([B, one.T, V])
        elif ode_type == "L":
            (U,) = mats_UV
            self.U = U
            self.UtA0tU = np.linalg.multi_dot([A0.dot(U).T, U])
            self.UtA1tU = np.linalg.multi_dot([A1.dot(U).T, U])
            self.D = np.linalg.multi_dot([U.T, B, one.T])
        else:
            U, V = mats_UV
            self.U, self.V = U, V
            self.UtA0tU = U.T.dot(A0.T.dot(U))
            self.UtA1tU = U.T.dot(A1.T.dot(U))
            self.VtC1V = V.T.dot(C1.dot(V))
            self.D = np.linalg.multi_dot([U.T, B, one.T, V])

    # %% VECTOR FIELDS
    def ode_F(self, t: float, Y: ArrayLike) -> ArrayLike:
        dY = -self.A0.dot(Y) - self.A1.dot(Y.dot(self.C1)) + \
            self.B.dot(self.one.T)
        return dY

    def ode_K(self, t: float, K: ArrayLike) -> ArrayLike:
        dK = -self.A0.dot(K) - self.A1.dot(K.dot(self.VtC1V)) + self.D
        return dK

    def ode_L(self, t: float, L: ArrayLike) -> ArrayLike:
        dLt = -self.UtA0tU.dot(L.T) - \
            self.UtA1tU.dot(L.T.dot(self.C1)) + self.D
        return dLt.T

    def ode_S(self, t: float, S: ArrayLike) -> ArrayLike:
        dS = -self.UtA0tU.dot(S) - self.UtA1tU.dot(S.dot(self.VtC1V)) + self.D
        return dS

    def project_ode_F(self, X: svd.SVD) -> svd.SVD:
        "Compute P(X)[F(X)]"
        A0, A1, B, C1 = self.A0, self.A1, self.B, self.C1
        U, S, Vt = X.U, X.S, X.Vt
        UUtA1XC1 = X.dot(C1).dot_sparse(A1, side='opposite').dot(
            U.T, side='opposite').dot(U, side='opposite')
        UUtA1XC1VVt = X.dot(C1).dot(Vt.T).dot(Vt).dot_sparse(
            A1, side='opposite').dot(U.T, side='opposite').dot(U, side='opposite')
        A0X = X.dot_sparse(A0, side='opposite')
        A1XC1VVt = X.dot(C1).dot(Vt.T).dot(Vt).dot_sparse(A0, side='opposite')
        UUtBBt = svd.SVD(U, U.T.dot(B), B.T)
        UUtBBtVVt = svd.SVD(U, np.linalg.multi_dot([U.T, B, B.T, Vt.T]), Vt)
        BBtVVt = svd.SVD(B, B.T.dot(Vt.T), Vt)
        list_of_SVD = [-UUtA1XC1, UUtBBt, UUtA1XC1VVt, -
                       UUtBBtVVt, - A0X, -A1XC1VVt, BBtVVt]
        return svd.add_svd(list_of_SVD)

    # %% CLOSED FORM SOLUTION

    def closed_form_solution(self,
                             t_span: tuple,
                             initial_value: svd.SVD) -> ArrayLike:
        if self.current_ode_type == "F":
            A = -self.A0
            B = -self.A1
            (L, Q) = (self.L_C1, self.Q_C1)
            D = self.D
        elif self.current_ode_type == "K":
            A = -self.A0
            B = -self.A1
            (L, Q) = sp.linalg.eigh(self.VtC1V)
            D = self.D
        elif self.current_ode_type == "L":
            A = -self.UtA0tU
            B = -self.UtA1tU
            (L, Q) = (self.L_C1, self.Q_C1)
            D = self.D
        elif self.current_ode_type == "S":
            A = -self.UtA0tU
            B = -self.UtA1tU
            (L, Q) = sp.linalg.eigh(self.VtC1V)
            D = self.D
        elif self.current_ode_type == "minus_S":
            A = self.UtA0tU
            B = self.UtA1tU
            (L, Q) = sp.linalg.eigh(self.VtC1V)
            D = -self.D

        if self.current_ode_type == "L":
            L0t = initial_value.T
            L1t = closed_form_cookie(t_span, L0t, A, B, Q, L, D)
            return L1t.T

        sol = closed_form_cookie(t_span, initial_value, A, B, Q, L, D)
        return sol


# %% CLOSED FORM COOKIE
def closed_form_cookie(t_span: tuple,
                       X0: svd.SVD,
                       A: ArrayLike,
                       B: ArrayLike,
                       Q: ArrayLike,
                       L: ArrayLike,
                       D: ArrayLike, ) -> ArrayLike:
    """
    Return the solution of the problem:
    X' = AX + BXC + D
    where C = Q * L * Q^T (C is supposed to be symmetric)
    X(t_0) = X0
    See my master thesis for details.

    Parameters
    ----------
    t_span: tuple
        initial and final ts (t0,t1)
    X0: ndarray
        initial value at ts t0
    A, B, Q, L, D: ndarray
        matrices of the problem as described above

    Returns
    -------
    X1: ndarray
        solution at ts t1
    """

    # Creation of variables
    h = t_span[1] - t_span[0]
    n, p = D.shape
    m1 = np.zeros(n * p)
    m2 = np.zeros(n * p)
    m3 = np.zeros(n * p)
    
    # Transform in dense for safety... TODO: update this
    if isinstance(D, LowRankMatrix):
        D = D.todense()
    if isinstance(X0, LowRankMatrix):
        X0 = X0.todense()

    # Efficient computation of the solution
    D_tilde = D.dot(Q)
    X0_tilde = X0.dot(Q)
    for i in range(p):
        Ai = A + L[i] * B
        if isinstance(Ai, spmatrix):
            m1[i * n: (i + 1) * n] = sp.sparse.linalg.spsolve(Ai, D_tilde[:, i])
        else:
            m1[i * n: (i + 1) * n] = la.solve(Ai, D_tilde[:, i]).reshape(-1)
        m2[i * n: (i + 1) * n] = sp.sparse.linalg.expm_multiply(
            Ai, m1[i * n: (i + 1) * n], start=0, stop=h, num=2
        )[-1]
        m3[i * n: (i + 1) * n] = sp.sparse.linalg.expm_multiply(
            Ai, X0_tilde[:, i], start=0, stop=h, num=2
        )[-1]

    M1 = m1.reshape(D.shape, order="F")
    M2 = m2.reshape(D.shape, order="F")
    M3 = m3.reshape(D.shape, order="F")
    X1 = (M3 + M2 - M1).dot(Q.T)
    return X1


# %% MAKE PROBLEM
def make_cookie_problem(t_span: tuple, p:int = 101, c0: int = 0, cp:int = 100) -> CookieIVP:
    """
    Return a cookie problem.

    Equation is: Ys' = -A0 Ys - A1 Ys C1 + bb^T
    """
    # IMPORT DATA
    import scipy.io as spio
    problem_cookie = spio.loadmat('~/src/IVPs/data/parametric_cookie_2x2_1580.mat')

    # DEFINE MATRICES
    A_all = problem_cookie["A"][0]  # all the A
    A0 = A_all[0]
    A1 = A_all[1] + A_all[2] + A_all[3] + A_all[4]
    b = problem_cookie["b"]

    # PARAMETERS
    c1 = np.linspace(c0, cp, num=p)
    C1 = np.diag(c1)

    # INITIAL VALUE
    X0 = np.zeros((1580, p))
    # for k in np.arange(1580):
    #     X0[k] = np.sin(2 * np.pi * k / (1580)) + np.cos(2 * np.pi * k / (1580))
    X0 = svd.truncated_svd(X0, k=50)

    # Create problem
    cookie_problem = CookieIVP(t_span, X0, A0, A1, C1, b)

    # Change initial value
    if t_span[0] != 0:
        new_X0 = classical_solvers.solve_one_step(
            cookie_problem, (0, t_span[0]), X0.full())
        u, s, vh = np.linalg.svd(new_X0, full_matrices=False)
        new_X0 = svd.SVD(u, s, vh)
        cookie_problem = CookieIVP(t_span, new_X0, A0, A1, C1, b)

    return cookie_problem
