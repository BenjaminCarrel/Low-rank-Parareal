# %% IMPORTATIONS
import warnings
import numpy as np
import numpy.linalg as la
import scipy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as spala
import scipy.linalg as sla
import krylov
from scipy.sparse import spmatrix
from numpy.typing import ArrayLike
from ivps.general import GeneralIVP
from low_rank_toolbox import svd
from low_rank_toolbox.svd import SVD
from low_rank_toolbox import low_rank_matrix
from low_rank_toolbox.low_rank_matrix import LowRankMatrix
from typing import Union, Optional


# %% CLASS SYLVESTER
class SylvesterIVP(GeneralIVP):
    """
    Subclass of generalProblem. Specific for the sylvester equation.

    Sylvester differential equation : Y'(t) = A Y(t) + Y(t) B + C.
    Initial value given by Y(t_0) = Y0.
    
    Y0 and C are low-rank matrices.
    """
    # ATTRIBUTES
    is_closed_form_available = True
    name = 'Sylvester'
    is_stiff = False

    def __init__(self,
                 t_span: tuple,
                 Y0: SVD,
                 A: Union[ArrayLike, spmatrix],
                 B: Union[ArrayLike, spmatrix],
                 C: SVD):
        """Initialize a Sylvester problem of the form Y' = AY + BY + C

        Args:
            t_span (tuple): time interval
            Y0 (Union[ArrayLike, LowRankMatrix]): (low-rank) initial value
            A (Union[ArrayLike, spmatrix]): (sparse) matrix A of the problem
            B (Union[ArrayLike, spmatrix]): (sparse) matrix B of the problem
            C (Union[ArrayLike, LowRankMatrix]): (low-rank) matrix C of the problem
        """
        # SANITY CHECK
        if not isinstance(Y0, SVD):
            warnings.warn('Initial value not low-rank -> converted into low-rank matrix')
            Y0 = svd.truncated_svd(Y0)
        if not isinstance(C, SVD):
            warnings.warn('Source not low-rank -> converted into low-rank matrix')
            C = svd.truncated_svd(C)
        
        # SPECIFIC PROPERTIES
        self.A, self.B, self.C = A, B, C
        
        # INITIALIZATION
        GeneralIVP.__init__(self, None, t_span, Y0)

        # PRE-PROCESSING
        if isinstance(A, spmatrix):
            self.A = A.tocsc()
        #     self.spluA = sp.sparse.linalg.splu(self.A)
        # else:
        #     self.spluA = None
        if isinstance(B, spmatrix):
            self.B = B.tocsc()
        #     self.spluB = sp.sparse.linalg.splu(self.B)
        # else:
        #     self.spluB = None
        
    @property
    def spluA(self):
        "LU cannot be copied (pickle)... Need to fix this somehow"
        return sp.sparse.linalg.splu(self.A)

    def spluB(self):
        return sp.sparse.linalg.splu(self.B)
        
    def select_ode(self,
                   initial_value: SVD,
                   ode_type: str,
                   mats_UV: tuple = ()):
        """
        Select the current ode, and pre-compute specific data for the lyapunov ODEs.
        """
        # SELECT ODE
        GeneralIVP.select_ode(self, initial_value, ode_type, mats_UV)
        A, B, C = self.A, self.B, self.C
        # PREPROCESSING
        if ode_type == "F":
            None
        elif ode_type == "K":
            (V,) = mats_UV
            self.VtBV = V.T.dot(B.dot(V))
            self.CV = C.dot(V).todense()
        elif ode_type == "L":
            (U,) = mats_UV
            self.UtAU = U.T.dot(A.dot(U))
            self.CU = C.dot(U).todense()
        else:
            U, V = mats_UV
            self.UtAU = U.T.dot(A.dot(U))
            self.VtBV = V.T.dot(B.dot(V))
            self.UtCV = C.dot(V).dot(U.T, side='opposite').todense()

    ## VECTOR FIELDS
    def ode_F(self, t: float, X: ArrayLike) -> ArrayLike:
        dY = self.A.dot(X) + (self.B.T.dot(X.T)).T + self.C.todense()
        return dY

    def ode_K(self, t: float, K: ArrayLike) -> ArrayLike:
        dK = self.A.dot(K) + K.dot(self.VtBV) + self.CV
        return dK

    def ode_L(self, t: float, L: ArrayLike) -> ArrayLike:
        dL = L.dot(self.UtAU) + self.B.dot(L) + self.CU
        return dL

    def ode_S(self, t: float, S: ArrayLike) -> ArrayLike:
        dS = self.UtAU.dot(S) + S.dot(self.VtBV) + self.UtCV
        return dS

    def linear_field(self, t: float, Y: SVD) -> SVD:
        "Linear field of the equation"
        AX = Y.dot_sparse(self.A, side='opposite')
        XB = Y.dot_sparse(self.B, side='usual')
        return AX + XB

    def non_linear_field(self, t: float, Y: SVD) -> SVD:
        "Non linear field of the equation"
        return self.C

    def projected_ode_F(self, t: float, Y: SVD) -> SVD:
        "Compute P(Y)[F(Y)]"
        # VARIABLES
        A, B, C = self.A, self.B, self.C
        U, S, V = Y.U, Y.S, Y.Vt.T
        
        # STEP 1 : FACTORIZATION
        AUS = A.dot(U.dot(S))
        CV = C.dot(V).todense()
        UUtCV = np.linalg.multi_dot([U, U.T, CV])
        M1 = np.column_stack([AUS - UUtCV + CV, U])
        SVtB = B.T.dot(V.dot(S.T)).T
        UtC = C.dot(U.T, side='opposite').todense()
        M2 = np.row_stack([V.T, SVtB + UtC])
        
        # STEP 2 : DOUBLE QR
        Q1, R1 = la.qr(M1, mode='reduced')
        Q2, R2 = la.qr(M2.T, mode='reduced')
        return SVD(Q1, R1.dot(R2.T), Q2.T)


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
        return closed_form_invertible_diff_sylvester(t_span, X0, self.A, self.B, self.C)
    
    def closed_form_projected_ode(self, t_span: tuple, KSL0: ArrayLike) -> ArrayLike:
        """Closed form solution of the projected ode

        Args:
            t_span (tuple): time interval (t0,t1)
            Z0 (ArrayLike): initial value, matrix of shape (n,k) or (k,k). Typically K0, S0, L0 for KSL.
        """
        if self.current_ode_type == "K":
            A = self.A
            B = self.VtBV
            C = self.CV
        elif self.current_ode_type == "L":
            A = self.B
            B = self.UtAU
            C = self.CU
        elif self.current_ode_type == "S":
            A = self.UtAU
            B = self.VtBV
            C = self.UtCV
        elif self.current_ode_type == "minus_S":
            A = -self.UtAU
            B = -self.VtBV
            C = -self.UtCV
        KSL1 = closed_form_invertible_diff_sylvester(t_span, KSL0, A, B, C)
        return KSL1
    
    
    def stiff_solution(self, t_span: tuple, Y0: LowRankMatrix) -> LowRankMatrix:
        "Solution of the stiff part of the ODE"
        A = self.A
        B = self.B
        h = t_span[1] - t_span[0]
        Y_tilde = Y0.expm_multiply(A, h, side='left')
        Y1 = Y_tilde.expm_multiply(B, h, side='right')
        return Y1

    def non_stiff_solution(self, t_span: tuple, Y0: SVD) -> SVD:
        "Solution of the non stiff part of the ODE"
        h = t_span[1] - t_span[0]
        return h * self.C + Y0


# %% SYLVESTER RELATED METHODS

def solve_small_sylvester(A: ArrayLike, B: ArrayLike, RHS: ArrayLike) -> ArrayLike:
    "Shortcut for solve_sylvester function from scipy. Find X such that AX + XB = RHS"
    if isinstance(RHS, LowRankMatrix):
        RHS = RHS.todense()
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
    left_space = krylov.KrylovSpace(A, RHS._matrices[0], invA)
    right_space = krylov.KrylovSpace(B, RHS._matrices[-1].T, invB)
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
    """Solve the sylvester equation AX + XB = RHS.
    Automatically choose the adapted method for solving the equation efficiently.

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
    """Closed form solution of the sylvester differential equation.
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
