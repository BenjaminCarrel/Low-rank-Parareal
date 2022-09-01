# %% IMPORTATIONS
import numpy as np
import scipy as sp
from scipy.sparse import spmatrix
import classical_solvers
from numpy.typing import ArrayLike
from ivps.general import GeneralIVP
from low_rank_toolbox import svd
from low_rank_toolbox.low_rank_matrix import LowRankMatrix
from low_rank_toolbox.svd import SVD

# %% CLASS RICCATI
class RiccatiIVP(GeneralIVP):
    r"""Subclass of generalProblem. Specific to the riccati equation

    Riccati differential equation : Q'(t) = A^T Q(t) + Q(t) A + C - Q(t) D Q(t)
    Initial value: Q(t_0) = Q0
    
    A is a sparse matrix
    Q0, C and D are low-rank matrices
    C and D are symmetric matrices
    """
    # ATTRIBUTES
    _is_closed_form_available = False
    name = 'Riccati'
    is_stiff = False

    # %% INIT FUNCTION
    def __init__(self,
                 t_span: tuple,
                 Y0: SVD,
                 A: spmatrix,
                 C: SVD,
                 D: SVD):
        # SPECIFIC PROPERTIES
        self.A = A.tocsc()
        self.B = A.tocsc()
        self.C = C
        self.D = D
        
        # INITIALIZATION
        GeneralIVP.__init__(self, None, t_span, Y0)
        
        # SOME PREPROCESSING
        if isinstance(A, spmatrix):
            self.A = A.tocsc()
        else:
            self.A = A
        self.B = self.A

    @property
    def spluA(self):
        "LU cannot be copied (pickle)... Need to fix this somehow"
        return sp.sparse.linalg.splu(self.A)
    
    @property
    def spluB(self):
        "LU cannot be copied (pickle)... Need to fix this somehow"
        return sp.sparse.linalg.splu(self.B)
        
    def copy(self):
        "Copy the problem"
        return RiccatiIVP(self.t_span, self.Y0, self.A, self.C, self.D)

    # %% SELECTION OF ODE
    def select_ode(self,
                   initial_value: ArrayLike,
                   ode_type: str,
                   mats_UV: tuple = ()):
        " Select the current ode, and pre-compute specific data for the lyapunov ODEs "
        # SELECT ODE
        GeneralIVP.select_ode(self, initial_value, ode_type, mats_UV)
        A, D, C = self.A, self.D, self.C

        # PRE-COMPUTATIONS
        if ode_type == "F":
            None
        elif ode_type == "K":
            (V,) = mats_UV
            self.VtAV = V.T.dot(A.dot(V))
            self.CV = C.dot(V, dense_output=True)
            self.VtD = D.dot(V.T, side='opposite', dense_output=True)
        elif ode_type == "L":
            (U,) = mats_UV
            self.UtAU = U.T.dot(A.dot(U))
            self.CU = C.dot(U, dense_output=True)
            self.UtD = D.dot(U.T, side='opposite', dense_output=True)
        else:
            U, V = mats_UV
            self.UtAU = U.T.dot(A.T.dot(U))
            self.VtAV = V.T.dot(A.dot(V))
            self.UtCV = C.dot(U.T, side='opposite').dot(V, dense_output=True)
            self.VtDU = D.dot(V.T, side='opposite').dot(U, dense_output=True)

    # %% VECTOR FIELDS
    def ode_F(self, t: float, X: ArrayLike) -> ArrayLike:
        AX = self.A.T.dot(X)
        XA = (self.A.T.dot(X.T)).T
        C = self.C.todense()
        XDX = self.D.dot(X, side='opposite').dot(X, dense_output=True)
        dX = AX + XA + C - XDX
        return dX

    def ode_K(self, t: float, K: ArrayLike) -> ArrayLike:
        dK = self.A.T.dot(K) + K.dot(self.VtAV) + \
            self.CV - K.dot(self.VtD.dot(K))
        return dK

    def ode_L(self, t: float, L: ArrayLike) -> ArrayLike:
        dL = L.dot(self.UtAU) + self.A.T.dot(L) + \
            self.CU - L.dot(self.UtD.dot(L))
        return dL

    def ode_S(self, t: float, S: ArrayLike) -> ArrayLike:
        dS = (
            self.UtAU.dot(S)
            + S.dot(self.VtAV)
            + self.UtCV
            - S.dot(self.VtDU.dot(S))
        )
        return dS

    def linear_field(self, t: float, Y: SVD) -> SVD:
        A = self.A
        AtY = Y.dot_sparse(A.T, side='opposite')
        YA = Y.dot_sparse(A, side='usual')
        return AtY + YA

    def non_linear_field(self, t: float, Y: SVD) -> SVD:
        C = self.C
        XDX = self.D.dot(Y, side='opposite').dot(Y)
        return C - XDX

    # def projected_ode_F(self, t: float, Y: svd.SVD) -> svd.SVD:
    #     "Compute P(X)[F(X)]"
    #     # SHORTCUTS
    #     A, D, C = self.A, self.D, self.C
    #     U, S, V = Y.U, Y.S, Y.Vt.T
        
    #     # STEP 1 : FACTORIZATION
    #     AtUS = A.T.dot(U.dot(S))
    #     CtCV = np.linalg.multi_dot([C.T, C, V])
    #     USVtDDtUS = np.linalg.multi_dot([U, S, V.T, D, D.T, U, S])
    #     M1 = np.column_stack([U, AtUS + CtCV - USVtDDtUS])
    #     SVtA = A.T.dot(V.dot(S.T)).T
    #     UtCtC = np.linalg.multi_dot([U.T, C.T, C])
    #     UtCtCVVt = np.linalg.multi_dot([UtCtC, V, V.T])
    #     M2 = np.row_stack([SVtA + UtCtC - UtCtCVVt, V.T])
        
    #     # STEP 2 : DOUBLE QR
    #     Q1, R1 = np.linalg.qr(M1, mode='reduced')
    #     Q2, R2 = np.linalg.qr(M2.T, mode='reduced')
    #     return svd.SVD(Q1, R1.dot(R2.T), Q2.T)


    # %% SOLUTIONS
    def stiff_solution(self, t_span: tuple, initial_value: SVD) -> SVD:
        "Solution of the stiff part of the ODE"
        A = self.A
        h = t_span[1] - t_span[0]
        Y_tilde = initial_value.expm_multiply(A.T, h, side='left')
        Y1 = Y_tilde.expm_multiply(A, h, side='right')
        return Y1

# %% MAKE PROBLEMS
def make_riccati_ostermann(t_span: tuple = (0, 0.1),
                           m: int = 50,
                           q: int = 5,
                           initial_rank: int = 20) -> RiccatiIVP:
    """
    Create data for the problem proposed in the paper
    "Convergence of a Lie-Trotter splitting for stiff matrix differential equation" by Ostermann et al.

    Equation is Q' = A^T Q + Q A + C^T C - Q D D^T Q
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

    # Construction of D
    d = np.eye(m)
    D = svd.truncated_svd(d.dot(d.T))

    # Construction of C
    ones = np.ones((1, m))
    e = np.zeros((nb, m))
    f = np.zeros((nb, m))
    for k in np.arange(nb):
        e[k] = np.sqrt(2) * np.cos(2 * np.pi * k * x)
        f[k] = np.sqrt(2) * np.sin(2 * np.pi * k * x)
    c = np.concatenate((ones, e, f))
    C = svd.truncated_svd(c.T.dot(c))

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
    X0 = SVD(U0, s0, V0.T)

    # Create the problem
    riccati_ostermann = RiccatiIVP(t_span, X0, A, C, D)

    # Change initial value
    if t_span[0] != 0:
        X0_tilde = classical_solvers.solve_one_step(
            riccati_ostermann, (0, t_span[0]), X0.full())
        new_X0 = svd.truncated_svd(X0_tilde)
        shifted_t_span = (0, t_span[1] - t_span[0])
        riccati_ostermann = RiccatiIVP(shifted_t_span, new_X0, A, C, D)
        
    # Additional properties
    riccati_ostermann.is_stiff = True

    return riccati_ostermann

# TODO: make riccati rail working
def make_riccati_rail(t_span: tuple = (0, 0.1),
                      filename: str = 'rail_371.mat',
                      initial_rank: int = 20) -> RiccatiIVP:
    """
    Make the riccati problem with the steel profile rail data.
    The data come from the Oberwolfach Model Reduction Benchmark Collection hosted at MORWiki[https://modelreduction.org]

    Equation is Q' = A^T Q + Q A + C^T C - Q B B^T Q
    They use the notation B for D.
    """
    # IMPORT DATA
    import scipy.io as spio
    import sys
    rail = spio.loadmat(sys.path[-1]+'/ivps/data/rails/'+filename, squeeze_me=True)

    # EXTRACT MATRICES
    A = rail["A"].tocsc()
    qb, rb = np.linalg.qr(rail["B"], mode='reduced')
    B = SVD(qb, rb.dot(rb.T), qb.T)
    qc, rc = np.linalg.qr(rail["C"].T, mode='reduced')
    C = SVD(qc, rc.dot(rc.T), qc.T)
    (n, n) = A.shape
    
     # Initial value
    np.random.seed(1111)
    G0 = np.random.rand(n, initial_rank)  # numerical initial_rank is about 16
    H0 = np.random.rand(n, initial_rank)
    U0, _ = sp.linalg.qr(G0, mode="economic")
    V0, _ = sp.linalg.qr(H0, mode="economic")
    s0 = np.logspace(-1, -21, initial_rank)
    s0 = np.sqrt(np.logspace(-1, -21, initial_rank))
    # X0 = np.zeros((m,m))
    # U0, s0, V0 = sp.linalg.svd(X0)
    X0 = SVD(U0, s0, V0.T)

    # Create riccati Problem
    riccati_rail = RiccatiIVP(t_span, X0, A, C, B)
    
    # Change initial value
    if t_span[0] != 0:
        X0_tilde = classical_solvers.solve_one_step(
            riccati_rail, (0, t_span[0]), X0.full())
        new_X0 = svd.truncated_svd(X0_tilde)
        shifted_t_span = (0, t_span[1] - t_span[0])
        riccati_rail = RiccatiIVP(shifted_t_span, new_X0, A, C, B)

    return riccati_rail

# %%
