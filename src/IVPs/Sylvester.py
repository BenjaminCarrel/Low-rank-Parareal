# %% IMPORTATIONS
import numpy as np
import numpy.linalg as la
import scipy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as spala
import scipy.linalg as sla
import krylov
from scipy.sparse import spmatrix
from numpy.typing import ArrayLike
from IVPs.General import GeneralIVP
from low_rank_toolbox import svd
from low_rank_toolbox.svd import SVD
from low_rank_toolbox import low_rank_matrix
from low_rank_toolbox.low_rank_matrix import LowRankMatrix
from typing import Union, Optional


# %% CLASS SYLVESTER
class SylvesterIVP(GeneralIVP):
    """
    Subclass of GeneralProblem. Specific for the Sylvester equation.

    Sylvester equation :
        .. math::
            \dot{Ys}(t) = A Ys(t) + Ys(t) B + C C^T.
    The initial value is given by :math:`Ys(t_0) = Y_0`
    """

    is_closed_form_available = True

    # %% INIT FUNCTION
    def __init__(self,
                 t_span: tuple,
                 Y0: SVD,
                 A: ArrayLike,
                 B: ArrayLike,
                 C: ArrayLike):
        # SPECIFIC PROPERTIES
        self.A = A.tocsc()
        self.B = B.tocsc()
        self.C = C
        # INITIALIZATION
        GeneralIVP.__init__(self, None, t_span, Y0)

        # SOME SPECIFIC PREPROCESSING
        # computed only once so its ok
        UC, RC = la.qr(self.C, mode='reduced')
        SC = RC.dot(RC.T)
        self.CCt = SVD(UC, SC, UC.T)
        self.spluA = sp.sparse.linalg.splu(self.A)
        self.spluB = sp.sparse.linalg.splu(self.B)

    # %% SELECTION OF ODE
    def select_ode(self,
                   initial_value: SVD,
                   ode_type: str,
                   mats_UV: tuple = ()):
        """
        Select the current ode, and precompute specific data for the Lyapunov ODEs.
        """
        # SELECT ODE
        GeneralIVP.select_ode(self, initial_value, ode_type, mats_UV)
        A, B, C = self.A, self.B, self.C
        # PREPROCESSING
        if ode_type == "F":
            None
        elif ode_type == "K":
            (V,) = mats_UV
            self.VtBV = np.linalg.multi_dot([V.T, B.dot(V)])
            self.CCtV = np.linalg.multi_dot([C, C.T, V])
        elif ode_type == "L":
            (U,) = mats_UV
            self.UtAU = np.linalg.multi_dot([U.T, A.dot(U)])
            self.CCtU = np.linalg.multi_dot([C, C.T, U])
        else:
            U, V = mats_UV
            self.UtAU = np.linalg.multi_dot([U.T, A.dot(U)])
            self.VtBV = np.linalg.multi_dot([V.T, B.dot(V)])
            self.UtCCtV = np.linalg.multi_dot([U.T, C, C.T, V])

    # %% VECTOR FIELDS
    def ode_F(self, t: float, X: ArrayLike) -> ArrayLike:
        dY = self.A.dot(X) + (self.B.T.dot(X.T)).T + self.C.dot(self.C.T)
        return dY

    def ode_K(self, t: float, K: ArrayLike) -> ArrayLike:
        dK = self.A.dot(K) + K.dot(self.VtBV) + self.CCtV
        return dK

    def ode_L(self, t: float, L: ArrayLike) -> ArrayLike:
        dL = L.dot(self.UtAU) + self.B.dot(L) + self.CCtU
        return dL

    def ode_S(self, t: float, S: ArrayLike) -> ArrayLike:
        dS = self.UtAU.dot(S) + S.dot(self.VtBV) + self.UtCCtV
        return dS

    def linear_field(self, t: float, Y: LowRankMatrix) -> LowRankMatrix:
        "Linear field of the equation"
        AX = Y.dot_sparse(self.A, side='opposite')
        XB = Y.dot_sparse(self.B, side='usual')
        return AX + XB

    def non_linear_field(self, t: float, Y: LowRankMatrix) -> LowRankMatrix:
        "Non linear field of the equation"
        return self.CCt

    def low_rank_ode_F(self, t: float, Y: LowRankMatrix) -> LowRankMatrix:
        dY = self.linear_field(t, Y) + self.non_linear_field(t, Y)
        return dY

    def project_ode_F(self, Y: LowRankMatrix) -> LowRankMatrix:
        "Compute P(Y)[F(Y)]"
        A, B, C = self.A, self.B, self.C
        U, Vt = Y.U, Y.Vt
        AY = Y.dot_sparse(A, side='opposite')
        YB = Y.dot_sparse(B, side='usual')
        UUtCCt = self.CCt.dot(U.T, side='opposite').dot(U, side='opposite')
        UUtCCtVVt = UUtCCt.dot(Vt.T).dot(Vt)
        CCtVVt = self.CCt.dot(Vt.T).dot(Vt)
        PFX = svd.add_svd([AY, YB, UUtCCt, -UUtCCtVVt, CCtVVt])
        # PFX = AX + XB + UUtCCt - UUtCCtVVt + CCtVVt # this option may be is faster for very large n and k
        return PFX

    # %% CLOSED FORM SOLUTIONS
    def closed_form_solution(self, t_span: tuple, XKSL0: Union[ArrayLike, SVD]) -> Union[ArrayLike, SVD]:
        "Closed form solution of the current selected ode"
        if self.current_ode_type=='F':
            XKSL1 = self.closed_form_full_ode(t_span, XKSL0)
        else:
            XKSL1 = self.closed_form_projected_ode(t_span, XKSL0)
        return XKSL1

    def closed_form_full_ode(self, t_span: tuple, X0: SVD) -> SVD:
        """Closed form solution of the full ode. Efficient implementation for low-rank matrices.

        Args:
            t_span (tuple): time interval (t0, t1)
            X0 (Union[ArrayLike, SVD]): low-rank matrix of shape (m,n)
        """
        return closed_form_invertible_diff_sylvester(t_span, X0, self.A, self.B, self.CCt)
    
    def closed_form_projected_ode(self, t_span: tuple, KSL0: ArrayLike) -> ArrayLike:
        """Closed form solution of the projected ode

        Args:
            t_span (tuple): time interval (t0,t1)
            Z0 (ArrayLike): initial value, matrix of shape (n,k) or (k,k). Typically K0, S0, L0 for KSL.
        """
        if self.current_ode_type == "K":
            A = self.A
            B = self.VtBV
            C = self.CCtV
        elif self.current_ode_type == "L":
            A = self.B
            B = self.UtAU
            C = self.CCtU
        elif self.current_ode_type == "S":
            A = self.UtAU
            B = self.VtBV
            C = self.UtCCtV
        elif self.current_ode_type == "minus_S":
            A = -self.UtAU
            B = -self.VtBV
            C = -self.UtCCtV
        KSL1 = closed_form_invertible_diff_sylvester(t_span, KSL0, A, B, C)
        return KSL1
    
    
    def stiff_solution(self, t_span: tuple, initial_value: LowRankMatrix) -> LowRankMatrix:
        "Solution of the stiff part of the ODE"
        A = self.A
        B = self.B
        h = t_span[1] - t_span[0]
        X_tilde = initial_value.expm_multiply(A, h, side='left')
        X1 = X_tilde.expm_multiply(B, h, side='right')
        return X1

    def non_stiff_solution(self, t_span: tuple, initial_value: SVD) -> SVD:
        "Solution of the non stiff part of the ODE"
        h = t_span[1] - t_span[0]
        CCt = self.non_linear_field(0, initial_value, output='SVD')
        return h * CCt + initial_value


# %% SYLVESTER RELATED METHODS

def solve_small_sylvester(A: ArrayLike, B: ArrayLike, RHS: ArrayLike) -> ArrayLike:
    "Shortcut for solve_sylvester function from scipy. Find X such that AX + XB = RHS"
    return sla.solve_sylvester(A, B, RHS)

def solve_sparse_low_rank_sylvester(A: spmatrix,
                                    B: spmatrix,
                                    RHS: SVD,
                                    invA: Optional[object] = None,
                                    invB: Optional[object] = None,
                                    tol: float = 1e-12,
                                    max_iter: int = 100) -> SVD:
    """Low-rank solver for sylvester equation, with large matrices A and B. Find X such that AX + XB = RHS
    The method is based on Krylov Space for finding the solution with a criterion; see Simoncini.

    Args:
        A (spmatrix): matrix of shape (m,m)
        B (spmatrix): matrix of shape (n,n)
        RHS (LowRankMatrix): low-rank matrix of shape (m,n)

    Returns:
        X (LowRankMatrix): computed solution
    """

    # INITIALIZATION
    normA = spala.norm(A)
    normB = spala.norm(B)
    normRHS = RHS.norm()
    left_space = krylov.KrylovSpace(A, RHS._matrices[0], invA, inverted=True)
    right_space = krylov.KrylovSpace(B, RHS._matrices[-1].T, invB, inverted=True)
    V = left_space.Q
    W = right_space.Q

    # SOLVE SMALL PROJECTED SYLVESTER IN LOOP
    for k in np.arange(1, max_iter):
        # SOLVE PROJECTED SYLVESTER Ak Y + Y Bk = Ck
        Ak = V.T.dot(A.dot(V))
        Bk = W.T.dot(B.dot(W))
        Ck = (RHS.dot(W)).dot(V.T, side="opposite")  # Vt @ RHS @ W
        Yk = sla.solve_sylvester(Ak, Bk, Ck.full())

        # CHECK CONVERGENCE
        Xk = SVD(V, Yk, W.T)
        AXk = Xk.dot_sparse(A, side="opposite")
        XkB = Xk.dot_sparse(B)
        # computation of crit could be more efficient, but SVD so its OK.
        crit = svd.add_svd([RHS, -AXk, -XkB]).norm() / \
            ((normA + normB) * la.norm(Yk) + normRHS)
        if crit < tol:
            return Xk.truncate(tol=np.finfo(float).eps) # truncate up to machine precision since the criterion overestimate the error
        else:
            left_space.compute_next_basis()
            V = left_space.Q
            right_space.compute_next_basis()
            W = right_space.Q

    print('No convergence before max_iter')
    X = SVD(V, Yk, W.T)
    return X

def solve_sylvester_large_A_small_B(A: spmatrix,
                                    B: ArrayLike,
                                    C: ArrayLike) -> ArrayLike:
    "Efficient for large sparse A and small dense B (symmetric). See Simoncini for details."
    S, W = la.eigh(B)
    C_hat = C.dot(W)
    X_hat = np.zeros(C.shape)
    I = sparse.eye(*A.shape)
    for i in np.arange(len(S)):
            X_hat[:, i] = spala.spsolve(A + S[i] * I, C_hat[:, i])
    # COMPUTE SVD OF THE SOLUTION X_hat @ W.T
    # if isinstance(C, LowRankMatrix):
    #     U, R1 = la.qr(X_hat, mode='reduced')
    #     V, R2 = la.qr(W, mode='reduced')
    #     S = R1.dot(R2.T)
    #     X = SVD(U,S,V.T)
    # else:
    X = X_hat @ W.T
    return X

def solve_sylvester(A: Union[ArrayLike, spmatrix],
                    B: Union[ArrayLike, spmatrix],
                    RHS: Union[ArrayLike, SVD],
                    invA: object = None,
                    invB: object = None) -> Union[ArrayLike, SVD]:
    """Solve the Sylvester equation AX + XB = RHS.
    Automagically choose the adapted method for solving the equation efficiently.

    Args:
        A (Union[ArrayLike, spmatrix]): matrix of shape (n,n)
        B (Union[ArrayLike, spmatrix]): matrix of shape (m,m)
        RHS (SVD): matrix of shape (m,n)
        invA (object, optional): inverse object of matrix A. Defaults to None.
        invB (object, optional): inverse object of matrix B. Defaults to None.
    """
    # SEPARATE ALL CASES
    if isinstance(A, spmatrix):
        if isinstance(B, spmatrix):
            # A AND B SPARSE, X IS LOW-RANK
            X = solve_sparse_low_rank_sylvester(A, B, RHS, invA, invB)
        else:
            # A SPARSE AND B SMALL, RHS IS DENSE -> X IS DENSE
            X = solve_sylvester_large_A_small_B(A, B, RHS)
    else:
        if isinstance(B, spmatrix):
            # NEVER USED SO FAR
            X = NotImplementedError
        else:
            # A AND B SMALL, RHS IS DENSE -> X IS DENSE
            X = solve_small_sylvester(A, B, RHS)
    return X

def closed_form_invertible_diff_sylvester(t_span: tuple,
                                          X0: LowRankMatrix,
                                          A: Union[ArrayLike, spmatrix],
                                          B: Union[ArrayLike, spmatrix],
                                          C: Union[ArrayLike, SVD]) -> Union[ArrayLike, SVD]:
    """Closed form solution of the Sylvester differential equation.
    X' = AX + XB + C
    X(t0) = X0
    
    The solution is:
    X(t) = exp(A(t-t0)) (X0 + Z) exp(B(t-t0)) - Z
    where AZ + ZB = C

    Args:
        t_span (tuple): time interval (t0, t1)
        X0 (LowRankMatrix): matrix of shape (m,n)
        A (Union[ArrayLike, spmatrix]): matrix of shape (m,m)
        B (Union[ArrayLike, spmatrix]): matrix of shape (n,n)
        C (SVD): low-rank matrix of shape (m,n)

    Returns:
        X1 (LowRankMatrix): solution at time t1 which is a low-rank matrix of shape (m,n)
    """
    # INITIALIZATION
    (t0, t1) = t_span
    h = t1 - t0
    
    # SOLVE SYLVESTER SYSTEM
    Z = solve_sylvester(A, B, C)
    M = X0 + Z
    
    # COMPUTE MATRIX EXPONENTIAL
    if isinstance(M, LowRankMatrix):
        N = M.expm_multiply(A, h, side='left')
        Y = N.expm_multiply(B, h, side='right')
    else:
        N = spala.expm_multiply(A, M, start=0, stop=h, endpoint=True, num=2)[-1]
        Y = spala.expm_multiply(B.T, N.T, start=0, stop=h, endpoint=True, num=2)[-1].T
    
    # ASSEMBLE SOLUTION
    X1 =  Y - Z
    return X1

# %%
