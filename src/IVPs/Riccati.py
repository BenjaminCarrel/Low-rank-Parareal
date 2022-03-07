# %% IMPORTATIONS
import numpy as np
import scipy as sp
import classical_solvers
from numpy.typing import ArrayLike
from IVPs.General import GeneralIVP
from low_rank_toolbox import svd

# %% CLASS RICCATI
class RiccatiIVP(GeneralIVP):
    r"""Subclass of GeneralProblem. Specific to the Riccati equation

    Riccati equation :
        .. math::
            \dot{Q}(t) = A^T Q(t) + Q(t) A + C^T C - Q(t) B B^T Q(t)
    The initial value is given by :math:`Q(t_0) = Q_0`
    """
    # ATTRIBUTES
    _is_closed_form_available = False

    # %% INIT FUNCTION
    def __init__(self,
                 t_span: tuple,
                 Y0: svd.SVD,
                 A: ArrayLike,
                 B: ArrayLike,
                 C: ArrayLike):
        # SPECIFIC PROPERTIES
        self.A = A.tocsc()
        self.B = B
        self.C = C
        # INITIALIZATION
        GeneralIVP.__init__(self, None, t_span, Y0)

    # %% SELECTION OF ODE
    def select_ode(self,
                   initial_value: ArrayLike,
                   ode_type: str,
                   mats_UV: tuple = ()):
        " Select the current ode, and precompute specific data for the Lyapunov ODEs "
        # SELECT ODE
        GeneralIVP.select_ode(self, initial_value, ode_type, mats_UV)
        A, B, C = self.A, self.B, self.C

        # PRECOMPUTATIONS
        if ode_type == "F":
            None
        elif ode_type == "K":
            (V,) = mats_UV
            self.VtAV = V.T.dot(A.dot(V))
            self.CtCV = np.linalg.multi_dot([C.T, C, V])
            self.VtBBt = np.linalg.multi_dot([V.T, B, B.T])
        elif ode_type == "L":
            (U,) = mats_UV
            self.UtAU = U.T.dot(A.dot(U))
            self.CtCU = np.linalg.multi_dot([C.T, C, U])
            self.UtBBt = np.linalg.multi_dot([U.T, B, B.T])
        else:
            U, V = mats_UV
            self.UtAU = U.T.dot(A.T.dot(U))
            self.VtAV = V.T.dot(A.dot(V))
            self.UtCtCV = np.linalg.multi_dot([U.T, C.T, C, V])
            self.VtBBtU = np.linalg.multi_dot([V.T, B, B.T, U])

    @property
    def spluA(self):
        return sp.sparse.linalg.splu(self.A)

    @property
    def spluB(self):
        return sp.sparse.linalg.splu(self.A)

    # %% VECTOR FIELDS

    def ode_F(self, t: float, X: ArrayLike) -> ArrayLike:
        AX = self.A.T.dot(X)
        XA = (self.A.T.dot(X.T)).T
        CC = self.C.T.dot(self.C)
        XBBX = X.dot(self.B).dot(self.B.T.dot(X))
        dX = AX + XA + CC - XBBX
        return dX

    def ode_K(self, t: float, K: ArrayLike) -> ArrayLike:
        dK = self.A.T.dot(K) + K.dot(self.VtAV) + \
            self.CtCV - K.dot(self.VtBBt.dot(K))
        return dK

    def ode_L(self, t: float, L: ArrayLike) -> ArrayLike:
        dL = L.dot(self.UtAU) + self.A.T.dot(L) + \
            self.CtCU - L.dot(self.UtBBt.dot(L))
        return dL

    def ode_S(self, t: float, S: ArrayLike) -> ArrayLike:
        dS = (
            self.UtAU.dot(S)
            + S.dot(self.VtAV)
            + self.UtCtCV
            - S.dot(self.VtBBtU.dot(S))
        )
        return dS

    def linear_field(self, t: float, X: svd.SVD) -> svd.SVD:
        A = self.A
        AtX = X.dot_sparse(A.T, side='opposite')
        XA = X.dot_sparse(A, side='usual')

    def non_linear_field(self, t: float, X: ArrayLike) -> ArrayLike:
        CtC = svd.SVD(self.C.T, np.eye(self.C.shape[0]), self.C)
        QBBtQ = svd.SVD(X.dot(self.B, dense_output=True), np.eye(
            self.B.shape[1]), X.dot(self.B.T, side='opposite', dense_output=True))
        return CtC - QBBtQ

    def project_ode_F(self, X: svd.SVD) -> svd.SVD:
        "Compute P(X)[F(X)]"
        A, B, C = self.A, self.B, self.C
        U, S, Vt = X.U, X.S, X.Vt
        AtX = X.dot_sparse(A.T, side='opposite')
        XA = X.dot_sparse(A, side='usual')
        UUtCtC = svd.SVD(U, U.T.dot(C.T), C)
        UUtCtCVVt = svd.SVD(U, np.linalg.multi_dot([U.T, C.T, C, Vt.T]), Vt)
        CtCVVt = svd.SVD(C.T, C.dot(Vt.T), Vt)
        XBBtX = svd.SVD(U, np.linalg.multi_dot([S, Vt, B, B.T, U, S]), Vt)
        list_of_SVD = [AtX, XA, UUtCtC, -UUtCtCVVt, CtCVVt, -XBBtX]
        return svd.add_svd(list_of_SVD)

    # %% SOLUTIONS
    def stiff_solution(self, t_span: tuple, initial_value: svd.SVD) -> svd.SVD:
        "Solution of the stiff part of the ODE"
        A = self.A
        h = t_span[1] - t_span[0]
        X_tilde = initial_value.expm_multiply(A.T, h, side='left')
        X1 = X_tilde.expm_multiply(A, h, side='right')
        return X1

# %% MAKE PROBLEMS
def make_riccati_ostermann(t_span: tuple = (0, 0.1),
                           m: int = 50,
                           q: int = 5,
                           initial_rank: int = 20) -> RiccatiIVP:
    """
    Create data for the problem proposed in the paper
    "Convergence of a Lie-Trotter splitting for stiff matrix differential equation" by Ostermann et al.

    Equation is Q' = A^T Q + Q A + C^T C - Q B B^T Q
    """
    # Parameters
    nb = int((q - 1) / 2)
    lam = 1
    x = np.zeros(m)
    x_minus = np.zeros(m)
    x_plus = np.zeros(m)
    for j in np.arange(m):
        x[j] = (j + 1) / (m + 1)
        x_minus[j] = (j + 1) / (m + 1) - 1 / (2 * m + 2)
        x_plus[j] = (j + 1) / (m + 1) + 1 / (2 * m + 2)

    # Construction of A : discrete of d_x(alpha(xs) d_x) - lam*I
    alpha_minus = 2 + np.cos(2 * np.pi * x_minus)
    alpha_plus = 2 + np.cos(2 * np.pi * x_plus)
    data = (m + 1) ** 2 * \
        np.array([alpha_plus, -alpha_minus - alpha_plus, alpha_minus])
    D = sp.sparse.spdiags(data, diags=[-1, 0, 1], m=m, n=m)
    Lam = lam * sp.sparse.eye(m)
    A = D - Lam
    # A = A.todense()

    # Construction of B
    B = np.eye(m)

    # Construction of C
    ones = np.ones((1, m))
    e = np.zeros((nb, m))
    f = np.zeros((nb, m))
    for k in np.arange(nb):
        e[k] = np.sqrt(2) * np.cos(2 * np.pi * k * x)
        f[k] = np.sqrt(2) * np.sin(2 * np.pi * k * x)
    C = np.concatenate((ones, e, f))

    # Initial value
    np.random.seed(1111)
    G0 = np.random.rand(m, initial_rank)  # numerical initial_rank is about 16
    H0 = np.random.rand(m, initial_rank)
    U0, _ = sp.linalg.qr(G0, mode="economic")
    V0, _ = sp.linalg.qr(H0, mode="economic")
    s0 = np.logspace(-1, -21, initial_rank)
    s0 = np.sqrt(np.logspace(-1, -21, initial_rank))
    # X0 = np.zeros((m,m))
    # U0, s0, V0 = sp.linalg.svd(X0)
    X0 = svd.SVD(U0, s0, V0.T)

    # Create the problem
    riccati_ostermann = RiccatiIVP(t_span, X0, A, B, C)

    # Change initial value
    if t_span[0] != 0:
        new_X0 = classical_solvers.solve_one_step(
            riccati_ostermann, (0, t_span[0]), X0.full())
        u, s, vh = np.linalg.svd(new_X0, full_matrices=False)
        new_X0 = svd.SVD(u, s, vh)
        riccati_ostermann = RiccatiIVP(t_span, new_X0, A, B, C)

    return riccati_ostermann

# TODO: make riccati rail working


def make_riccati_rail(t_span: tuple,
                      path_to_data: str,
                      initial_rank: int = 20) -> RiccatiIVP:
    """
    Riccati problem.
    Data come from the Oberwolfach Model Reduction Benchmark Collection hosted at MORWiki[https://modelreduction.org]

    Equation is Q' = A^T Q + Q A + C^T C - Q B B^T Q
    """
    # IMPORT DATA
    rail = sp.io.loadmat(path_to_data, squeeze_me=True)

    # EXTRACT MATRICES
    A = rail["A"]
    B = rail["B"]
    C = rail["C"]
    (n, n) = A.shape

    # Q0 with decaying singular values
    np.random.seed(1111)
    G0 = np.random.rand(n, initial_rank)  # numerical initial_rank is about 16
    H0 = np.random.rand(n, initial_rank)
    U0, _ = sp.linalg.qr(G0, mode="economic")
    V0, _ = sp.linalg.qr(H0, mode="economic")
    s0 = np.sqrt(np.logspace(-1, -21, initial_rank))
    Q0 = svd.SVD(U0, s0, V0.T, full_matrices=False)
    S0 = np.diag(np.sqrt(s0))

    # Create Riccati Problem
    riccati_rail = RiccatiIVP(t_span, Q0, A, B, C)

    return riccati_rail
