from __future__ import annotations

import copy
from typing import Union, List, Optional, Tuple
from low_rank_toolbox.low_rank_matrix import LowRankMatrix
import numpy as np
import numpy.linalg as la
from scipy.linalg import block_diag
from numpy.typing import ArrayLike
from scipy.sparse import spmatrix
import low_rank_toolbox



# %% SVD CLASS
class SVD(LowRankMatrix):
    """
    Singular Value Decomposition (SVD)
    X = U @ S @ V.T
    where U, V are orthonormal and S is non-singular (not necessarly diagonal)
    If X is low-rank, it is much more efficient to store only U, S, V.
    Behaves like a numpy ndarray but preserve the low-rank structure.
    """

    _format = "SVD"

    # %% BASICS
    def __init__(self, U: ArrayLike, S: ArrayLike, Vt: ArrayLike):
        """
        Create a low-rank matrix stored by its SVD
        :param U: ArrayLike (m,r)
        :param S: ArrayLike (r,r)
        :param Vt: ArrayLike (r,n)
        """
        # Check if S is 1D or 2D
        if not len(S.shape) == 2:
            S = np.diag(S)
        # Check if input is V or Vt
        (r, n) = Vt.shape
        if r > n:
            Vt = Vt.T
        # INITIALIZE
        super().__init__(U, S, Vt)

    # Alias for U, S, Vt
    U = LowRankMatrix.create_matrix_alias(0)
    S = LowRankMatrix.create_matrix_alias(1)
    Vt = LowRankMatrix.create_matrix_alias(2)

    @classmethod
    def from_full(cls,
                  matrix: ArrayLike,
                  r: int = None,
                  tol: float = None) -> SVD:
        "Initialize from full (m,n) matrix"
        U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
        return cls(U, s, Vt).truncate(r, tol)
    
    @property
    def is_symmetric(self) -> bool:
        return np.allclose(self.U, self.Vt.T)
        

    # %% STANDARD OPERATIONS
    def __add__(self, other: Union[LowRankMatrix, SVD]) -> Union[SVD, ArrayLike]:
        if isinstance(other, SVD):
            return add_svd([self, other])
        else:
            return super().__add__(other)
        

    def dot(self,
            other: Union[SVD, LowRankMatrix, ArrayLike],
            side: str = 'usual',
            dense_output: bool = False) -> Union[SVD, LowRankMatrix, ArrayLike]:
        """Matrix multiplication between SVD and other

        Args:
            other (Union[SVD, LowRankMatrix, ArrayLike]): Matrix to multiply with.
            side (str, optional): Can be 'usual' or 'opposite'. Defaults to 'usual'.
            dense_output (bool, optional): Return a dense matrix as output. Defaults to False.
        """
        if isinstance(other, SVD) and dense_output == False:
            if side == 'usual':
                return dot_svd(self, other)
            elif side == 'opposite':
                return dot_svd(other, self)
            else:
                raise ValueError(
                    'Incorrect side. Choose "usual" or "opposite".')
        else:
            return super().dot(other, side, dense_output)

    # %% SPECIFIC OPERATIONS
    def truncate(self,
                 r: int = None,
                 tol: float = None,
                 inplace: bool = False) -> SVD:
        """
        Truncate the SVD to rank r and tolerance tol (rank is prior to tol)
        :param r: int
            rank of truncation
        :param tol: float
            tolerance of truncation
        :param inplace: bool
            modify the matrix itself or not
        """
        # Both argument are None -> do nothing
        if r is None and tol is None:
            return self
        # Tol is not None -> find the rank of tolerance
        if tol is not None:
            sing_vals = self.sing_vals
            r_tol = np.sum([sing_vals > sing_vals[0] * tol])
            r_tol = max(1, r_tol)
            if r is None:
                r = r_tol
            else:
                r = min(r, r_tol)
        # Copy or not the matrix
        if inplace is True:
            new_mat = self
        else:
            new_mat = self.copy()
        # Truncate
        new_mat.U = new_mat.U[:, :r]
        new_mat.S = new_mat.S[:r, :r]
        new_mat.Vt = new_mat.Vt[:r, :]
        return new_mat

    def truncate_perpendicular(self, 
                      r: int = None,
                      tol: float = None,
                      inplace: bool = False) -> SVD:
        """Perpendicular truncation of SVD. Rank is prior to tolerance.

        Args:
            r (int, optional): Rank of truncation. Defaults to None.
            tol (float, optional): Tolerance truncation. Defaults to None.
            inplace (bool, optional): Modify the matrix itself or not. Defaults to False.

        Returns:
            SVD: SVD containing the n-r last singular values
        """
        # Both argument are None -> do nothing
        if r is None and tol is None:
            return self
        # Tol is not None -> find the rank of tolerance
        if tol is not None:
            sing_vals = self.sing_vals
            r_tol = np.sum([sing_vals > sing_vals[0] * tol])
            r_tol = max(1, r_tol)
            if r is None:
                r = r_tol
            else:
                r = min(r, r_tol)
        # Copy or not the matrix
        if inplace is True:
            new_mat = self
        else:
            new_mat = self.copy()
        # Truncate
        new_mat.U = new_mat.U[:, r:]
        new_mat.S = new_mat.S[r:, r:]
        new_mat.Vt = new_mat.Vt[r:, :]
        return new_mat
        
        
    
    def project_onto_tangent_space(self, other: Union[ArrayLike, LowRankMatrix]) -> SVD:
        "Projection of other onto the tangent space at self"
        M1 = copy.deepcopy(other)
        M2 = copy.deepcopy(other)
        M3 = copy.deepcopy(other)
        M1._matrices[0] = self.U.dot(self.U.T.dot(other))
        M2._matrices[0] = M1._matrices[0]
        M2._matrices[-1] = self.Vt.dot(self.Vt.T.dot(other.T)).T
        M3._matrices[-1] = M2._matrices[-1]
        return add_svd([M1, -M2, M3])

    def norm(self) -> float:
        """Calculate Frobenius norm"""
        if self.is_uv_orthogonal:
            return np.linalg.norm(self.S, ord='fro')
        else:
            return super().norm()

    @property
    def is_S_diagonal(self) -> bool:
        return np.all(self.S == np.diag(np.diagonal(self.S)))

    @property
    def is_uv_orthogonal(self) -> bool:
        norm_u = np.linalg.norm(self.U, 2)
        norm_v = np.linalg.norm(self.Vt, 2)
        if (norm_u - 1) < 1e-12 and (norm_v - 1) < 1e-12:
            return True
        else:
            return False

    @property
    def sing_vals(self) -> Optional[ArrayLike]:
        "Return the singular values"
        if self.is_S_diagonal:
            sing_vals = np.diagonal(self.S)
        else:
            sing_vals = np.linalg.svd(self.S, compute_uv=False)
        return sing_vals

    @property
    def K(self) -> ArrayLike:
        return self.U @ self.S

    @property
    def L(self):
        return self.V @ self.S.T



# %% SVD RELATED METHODS
def add_svd(list_of_SVD: List[SVD]) -> SVD:
    "Efficient addition of several SVDs"
    L = len(list_of_SVD)
    list_of_U = np.empty(L, dtype=object)
    list_of_S = np.empty(L, dtype=object)
    list_of_V = np.empty(L, dtype=object)
    for k in range(L):
        list_of_U[k] = list_of_SVD[k].U
        list_of_S[k] = list_of_SVD[k].S
        list_of_V[k] = list_of_SVD[k].Vt.T

    # First, concatenate matrices
    left = np.concatenate(list_of_U, axis=1)
    mid = block_diag(*list_of_S)
    right = np.concatenate(list_of_V, axis=1)

    # Second, do two QRs and SVD in the middle
    Q1, R1 = np.linalg.qr(left, mode="reduced")
    Q2, R2 = np.linalg.qr(right, mode="reduced")
    middle = np.linalg.multi_dot([R1, mid, R2.T])
    # U, S, Vt = np.linalg.svd(middle, full_matrices=False)
    # U = Q1.dot(U)
    # Vt = Vt.dot(Q2.T)
    # output = SVD(U, S, Vt)
    USVt = truncated_svd(middle)
    output = USVt.dot(Q1, side='opposite').dot(Q2.T)
    return output


def dot_svd(left_matrix: SVD, right_matrix: SVD) -> SVD:
    "Efficient multiplication between two SVDs"
    middle = [left_matrix.S, left_matrix.Vt, right_matrix.U, right_matrix.S]
    M = np.linalg.multi_dot(middle)
    (u, S, vt) = np.linalg.svd(M, full_matrices=False)
    U = left_matrix.U.dot(u)
    Vt = vt.dot(right_matrix.Vt)
    return SVD(U, S, Vt)


def full_svd(X: ArrayLike) -> SVD:
    "Shortcut for numpy's SVD"
    (U, s, Vt) = la.svd(X, full_matrices=True)
    return SVD(U, s, Vt)


def reduced_svd(X: ArrayLike) -> SVD:
    "Shortcut for numpy's SVD"
    (U, s, Vt) = la.svd(X, full_matrices=False)
    return SVD(U, s, Vt)

def truncated_svd(X: ArrayLike, k:int = None, tol: float = 1e-15) -> SVD:
    "Shortcut for numpy's SVD"
    (U, s, Vt) = la.svd(X, full_matrices=False)
    return SVD(U, s, Vt).truncate(k, tol)


def best_rank_approximation(X: ArrayLike, k: int):
    "Shortcut for SVD combined with truncation"
    X = reduced_svd(X)
    Xk = X.truncate(k)
    return Xk 


def randomized_SVD(X: Union[ArrayLike, spmatrix, LowRankMatrix], k: int, p: int) -> SVD:
    """Randomized SVD algorithm; see Halko, Martinsson and Tropp 2010.

    Args:
        X (Union[ArrayLike, tuple]): Matrix of interest
        k (int): Rank of approximation
        p (int): Oversampling parameter, improve the accuracy of the computed SVD

    Returns:
        SVD: Near-optimal best rank-k approximation of X
    """
    _, n = X.shape
    # Draw the random matrix
    np.random.seed(123)
    Omega = np.random.randn(n, k + p)
    Y = X.dot(Omega)
    # Randomized SVD routine
    Q, _ = la.qr(Y, mode='reduced')
    C = X.T.dot(Q).T

    Xk = truncated_svd(C, k)
    Xk.U = Q.dot(Xk.U)
    return Xk

def adaptive_randomized_SVD(X: Union[ArrayLike, spmatrix, LowRankMatrix], tol, prob_success):
    "Adaptive randomized SVD"
    # SEE EXERCISE 9 FROM BART LECTURE
    # TODO : IMPLEMENT IT
    return NotImplementedError


def simultaneous_power_iteration_tilde(A,X0,k,nmb_iter,tol,U_k=None):
    "See exercise 7 from Bart lecture"

def subspace_iteration(X: Union[ArrayLike, spmatrix, LowRankMatrix], k:int, nb_iter: int, side: str ='left'):
    """Subspace iteration for first k singular vectors

    Args:
        X (Union[ArrayLike, spmatrix, LowRankMatrix]): Matrix of interest
        k (int): Number of singular vectors
        nb_iter (int): number of iterations
        side (str): 'left' or 'right' singular vectors
    """
    if side == 'right':
        X = X.T
    
    # Draw an orthogonal random matrix
    np.random.seed(123)
    n, _ = X.shape
    M = np.random.randn(n, k)
    Q, _ = la.qr(M, mode='reduced')
    
    for _ in np.arange(nb_iter):
        Q, _ = la.qr(X.dot(X.T.dot(Q)), mode='reduced')
    return Q


# %% GENERATE RANDOM LOW-RANK MATRICES
def generate_low_rank_matrix(shape: tuple,
                             singular_values: list,
                             is_symmetric: bool = True):
    "Generate a low-rank matrix with an exponential decay"
    rank = len(singular_values)
    M1 = np.random.randn(shape[0], rank)
    U, _ = la.qr(M1, mode='reduced')
    if is_symmetric:
        V = U
    else:
        M2 = np.random.randn(shape[1], rank)
        V, _ = la.qr(M2, mode='reduced')

    S = np.diag(singular_values)
    return SVD(U, S, V.T)
    
