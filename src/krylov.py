# %% IMPORTATIONS
from re import X
from types import NoneType
from typing import List, Tuple, Union, Optional

import numpy as np
import scipy.linalg as la
from scipy.sparse import spmatrix
import scipy.sparse.linalg as spala
from low_rank_toolbox import svd
from numpy.typing import ArrayLike


# %% KRYLOV CLASSES
class KrylovVector:

    def __init__(self, A: ArrayLike, v: ArrayLike) -> None:
        """Initialize a Krylov Space where v is a vector

        Args:
            A (ArrayLike): Matrix of shape (n,n)
            v (ArrayLike): Vector of shape (n,)
        """
        # STORE DATA
        self.A = A
        v = v.flatten()
        self.v = v
        self.n = v.shape[0]

        # CHECK FOR SYMMETRIC MATRIX
        if not abs(A-A.T).nnz:
            self.symmetric = True
        else:
            self.symmetric = False

        # FIRST BASIS
        Q = np.zeros((self.n, 1))
        Q[:, 0] = v/la.norm(v)
        self.Q = Q
        if self.symmetric:
            # LANCZOS INITIALIZATION
            self.u = A.dot(Q[:, 0])
            self.alpha = [Q[:, 0].T.dot(self.u)]
            self.beta = []

    @property
    def size(self):
        return self.Q.shape[1]

    def compute_next_basis(self):
        "Compute the next basis of the Krylov space"
        # INITIALIZE
        A = self.A
        k = self.Q.shape[1]
        Q = np.zeros((self.n, k+1))
        Q[:, :k] = self.Q

        # SEPARATE TWO CASES
        if self.symmetric:
            # LANCZOS ALGORITHM
            u = A.dot(Q[:, k-1])
            alpha = Q[:, k-1].T.dot(u)
            u = u - alpha * Q[:, k-1]
            if k > 1:
                u = u - self.beta[k-2] * Q[:, k-2]
            beta = la.norm(u)
            Q[:, k] = u / beta

            # UPDATE DATA
            self.Q = Q
            self.alpha = self.alpha + [alpha]
            self.beta = self.beta + [beta]
            self.u = u

        else:
            # INITIALIZE
            H = np.zeros((k+1, k))
            if k > 1:
                H[:k, :k-1] = self.H

            # ARNOLDI ALGORITHM
            u = A.dot(Q[:, k-1])
            for i in range(k):
                H[i, k-1] = Q[:, i].T.dot(u)
                u = u - H[i, k-1] * Q[:, i]
            H[k, k-1] = la.norm(u)
            if H[k, k-1] < 1e-15:
                print('Warning: H[k,k-1] < 1e-15.')
            Q[:, k] = u/H[k, k-1]

            # UPDATE DATA
            self.Q = Q
            self.H = H
            self.u = u


class KrylovInvertedVector:

    def __init__(self, A: ArrayLike, v: ArrayLike, invA: Optional[object] = None) -> None:
        """Initialize a Krylov Inverted Space where v is a vector

        Args:
            A (ArrayLike): Matrix of shape (n,n)
            v (ArrayLike): Vector of shape (n,)
            invA (optional, object): Object inverse of A, containing a .solve method for faster computations
        """
        # STORE DATA
        self.A = A
        self.invA = invA
        v = v.flatten()
        self.v = v
        self.n = v.shape[0]

        # CHECK FOR SYMMETRIC MATRIX
        if not abs(A-A.T).nnz:
            self.symmetric = True
        else:
            self.symmetric = False

        # FIRST BASIS
        v = self.invdot(v)
        Q = np.zeros((self.n, 1))
        Q[:, 0] = v/la.norm(v)
        self.Q = Q
        if self.symmetric:
            # LANCZOS INITIALIZATION
            self.u = self.invdot(Q[:, 0])
            self.alpha = [Q[:, 0].T.dot(self.u)]
            self.beta = []

    def invdot(self, b: ArrayLike) -> ArrayLike:
        "Solve Ax = b"
        A = self.A
        if self.invA == None:
            if isinstance(A, spmatrix):
                x = spala.spsolve(A, b)
            else:
                x = la.solve(A, b)
        else:
            x = self.invA.solve(b)
        return x

    def compute_next_basis(self):
        "Compute the next basis of the Krylov space"
        # INITIALIZE
        A = self.A
        k = self.Q.shape[1]
        Q = np.zeros((self.n, k+1))
        Q[:, :k] = self.Q

        # SEPARATE TWO CASES
        if self.symmetric:
            # LANCZOS ALGORITHM
            u = self.invdot(Q[:, k-1])
            alpha = Q[:, k-1].T.dot(u)
            u = u - alpha * Q[:, k-1]
            if k > 1:
                u = u - self.beta[k-2] * Q[:, k-2]
            beta = la.norm(u)
            Q[:, k] = u / beta

            # UPDATE DATA
            self.Q = Q
            self.alpha = self.alpha + [alpha]
            self.beta = self.beta + [beta]
            self.u = u

        else:
            # INITIALIZE
            H = np.zeros((k+1, k))
            if k > 1:
                H[:k, :k-1] = self.H

            # ARNOLDI ALGORITHM
            u = self.invdot(Q[:, k-1])
            for i in range(k):
                H[i, k-1] = Q[:, i].T.dot(u)
                u = u - H[i, k-1] * Q[:, i]
            H[k, k-1] = la.norm(u)
            if H[k, k-1] < 1e-15:
                print('Warning: H[k,k-1] < 1e-15.')
            Q[:, k] = u/H[k, k-1]

            # UPDATE DATA
            self.Q = Q
            self.H = H
            self.u = u


class KrylovMatrix:

    def __init__(self, A: ArrayLike, V: ArrayLike) -> None:
        """Initialize a Krylov Space where V is a (tall) matrix

        Args:
            A (ArrayLike): Matrix of shape (n,n)
            v (ArrayLike): Vector of shape (n,r)
        """
        # STORE DATA
        self.A = A
        self.V = V
        self.n = A.shape[0]
        self.r = V.shape[1]
        self.size = 1

        # CHECK FOR SYMMETRIC MATRIX
        if not abs(A-A.T).nnz:
            self.symmetric = True
        else:
            self.symmetric = False

        # FIRST BASIS
        Q = la.orth(V)
        self.Q = Q
        # self.H = H # Storing H is useless except if you need its data

    def compute_next_basis(self):
        "Compute the next basis of the Krylov space"
        # INITIALIZE
        A = self.A
        r = self.r
        self.size = self.size+1
        m = self.size
        Q = np.zeros((self.n, m*r), dtype=A.dtype)
        Q[:, :(m-1)*r] = self.Q
        H = np.empty(m, dtype=object)

        # BLOCK LANCZOS : COULDN'T FIND GOOD REFERENCE SO BLOCK ARNOLDI IS ENOUGH NOW
        # BLOCK ARNOLDI ALGORITHM
        Wj = A.dot(Q[:, (m-2)*r:(m-1)*r])
        for i in np.arange(m-1):
            H[i] = Q[:, i*r:(i+1)*r].T.dot(Wj)
            Wj = Wj - Q[:, i*r:(i+1)*r].dot(H[i])
        Q[:, (m-1)*r:m*r], _ = la.qr(Wj, mode='economic')

        # UPDATE DATA
        self.Q = Q


class KrylovInvertedMatrix:

    def __init__(self, A: ArrayLike, V: ArrayLike, invA: Optional[object] = None) -> None:
        """Initialize a Krylov Inverted Space where V is a (tall) matrix

        Args:
            A (ArrayLike): Matrix of shape (n,n)
            v (ArrayLike): Vector of shape (n,r)
        """
        # STORE DATA
        self.A = A
        self.invA = invA
        self.V = V
        self.n = A.shape[0]
        self.r = V.shape[1]
        self.size = 1

        # CHECK FOR SYMMETRIC MATRIX
        if not abs(A-A.T).nnz:
            self.symmetric = True
        else:
            self.symmetric = False

        # FIRST BASIS
        V = self.invdot(V)
        Q = la.orth(V)
        self.Q = Q
        # self.H = H # Storing H is useless except if you need its data

    def invdot(self, b: ArrayLike) -> ArrayLike:
        "Solve Ax = b"
        A = self.A
        if self.invA == None:
            if isinstance(A, spmatrix):
                x = spala.spsolve(A, b)
            else:
                x = la.solve(A, b)
        else:
            x = self.invA.solve(b)
        return x

    def compute_next_basis(self):
        "Compute the next basis of the Krylov space"
        # INITIALIZE
        A = self.A
        r = self.r
        self.size = self.size+1
        m = self.size
        Q = np.zeros((self.n, m*r), dtype=A.dtype)
        Q[:, :(m-1)*r] = self.Q
        H = np.empty(m, dtype=object)

        # BLOCK LANCZOS : COULDN'T FIND GOOD REFERENCE SO BLOCK ARNOLDI IS ENOUGH NOW
        # BLOCK ARNOLDI ALGORITHM
        Wj = self.invdot(Q[:, (m-2)*r:(m-1)*r])
        for i in np.arange(m-1):
            H[i] = Q[:, i*r:(i+1)*r].T.dot(Wj)
            Wj = Wj - Q[:, i*r:(i+1)*r].dot(H[i])
        Q[:, (m-1)*r:m*r], _ = la.qr(Wj, mode='economic')

        # UPDATE DATA
        self.Q = Q


class KrylovSpace:

    def __init__(self, A: ArrayLike, V: ArrayLike, invA: Optional[object] = None) -> None:
        """Krylov Space, potentially inverted.
        Supports matrix or vector V.
        Krylov Space : K(A,V) = span(V, AV, ..., A^{k-1} V)
        Inverted Krylov Space : K(A,V) = span(V, A^{-1}V, AV, A^{-2}V, ..., A^{k-1}V, A^{-k}V)

        Args:
            A (ArrayLike): Matrix of shape (n,n)
            V (ArrayLike): Matrix or vector of shape (n,r)
            invA (Optional[object], optional): Preprocessed inverse of the matrix A for faster computations. Defaults to None.
            inverted (bool, optional): Use the inverted space or not. Defaults to False.

        Returns:
            _type_: _description_
        """
        # STORE DATA
        self.A = A
        self.V = V
        self.n = V.shape[0]
        self.r = V.shape[1]
        self.invA = invA
        if isinstance(invA, NoneType):
            self.inverted = False
        else:
            self.inverted = True

        # INITIALIZE KRYLOV SPACES
        self.size = 1
        self.Q_precomputed = np.zeros((0, 0))
        KS_inv = None
        if len(V.shape) == 2:
            if V.shape[1] > 1:
                KS = KrylovMatrix(A, V)
            if self.inverted:
                KS_inv = KrylovInvertedMatrix(A, V, invA)
        else:
            KS = KrylovVector(A, V)
            if self.inverted:
                KS_inv = KrylovInvertedVector(A, V, invA)
        self.KS = KS
        self.KS_inv = KS_inv

    def __repr__(self):
        return ("Krylov Space with properties: \n"
                f"Shape of A: {self.A.shape} with format {type(self.A)} \n"
                f"Shape of V: {self.V.shape} \n"
                f"Inverted space is included: {self.inverted} \n"
                f"Inverse is precomputed with: {self.invA} \n"
                f"Current size is {self.size} so shape of Q is {self.Q.shape}")

    @property
    def Q(self):
        "Orthogonal matrix of the Krylov space, efficient"
        # No inverted space case (easy)
        if not self.inverted:
            return self.KS.Q

        # Inverted space : we have to do a QR
        elif self.Q_precomputed.shape[1] != 2 * self.r * self.size:
            Q = la.orth(np.column_stack([self.KS.Q, self.KS_inv.Q]))
            self.Q_precomputed = Q
        return self.Q_precomputed

    def compute_next_basis(self):
        "Compute the next basis of the (two) Krylov Space(s)"
        self.size = self.size + 1
        # if 2*self.r*self.size >= self.n: print('Reached maximal size')
        self.KS.compute_next_basis()
        if self.inverted:
            self.KS_inv.compute_next_basis()


# %% KRYLOV RELATED METHODS
def Arnoldi(A: ArrayLike, v: ArrayLike, k: int) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Arnoldi algorithm. Computes orthogonal basis of a Krylov space. See Martin J. Gander lecture.

    Args:
        A (ArrayLike): Matrix of shape (n,n)
        v (ArrayLike): Vector of shape (n,)
        k (int): Size of the Krylov space to compute.

    Returns:
        Q (ArrayLike): Orthogonal matrix such that span(v, Av, ..., A**{k-1}v) = span(q[:,1], ..., q[:,k])
        H (ArrayLike): Additional data
        u (ArrayLike): Additional data
    """
    # CREATE VARIABLES
    n = A.shape[0]
    Q = np.zeros((n, k))
    H = np.zeros((k+1, k))

    # Arnoldi Algorithm, see Iterative methods by Martin J. Gander
    Q[:, 0] = v / la.norm(v)
    for j in range(k):
        u = A.dot(Q[:, j])
        for i in range(j+1):
            H[i, j] = Q[:, i].T.dot(u)
            u = u - H[i, j] * Q[:, i]
        if j < k-1:
            H[j+1, j] = la.norm(u)
            if H[j+1, j] < 1e-15:
                print('Lucky breakdown.')
                break
            Q[:, j+1] = u/H[j+1, j]
    return Q, H, u


def Arnoldi_inverted(invA: object, v: ArrayLike, k: int) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Arnoldi algorithm. Computes orthogonal basis of a Krylov space. See Martin J. Gander lecture.

    Args:
        invA (ArrayLike): object of inverse matrix A with solve function
        v (ArrayLike): Vector of shape (n,)
        k (int): Size of the Krylov space to compute.

    Returns:
        Q (ArrayLike): Orthogonal matrix such that span(v, Av, ..., A**{k-1}v) = span(q[:,1], ..., q[:,k])
        H (ArrayLike): Additional data
        u (ArrayLike): Additional data
    """
    # CREATE VARIABLES
    n = v.shape[0]
    Q = np.zeros((n, k))
    H = np.zeros((k+1, k))

    # Arnoldi Algorithm, see Iterative methods by Martin J. Gander
    Q[:, 0] = v / la.norm(v)
    for j in range(k):
        u = invA.solve(Q[:, j])
        for i in range(j+1):
            H[i, j] = Q[:, i].T.dot(u)
            u = u - H[i, j] * Q[:, i]
        if j < k-1:
            H[j+1, j] = la.norm(u)
            if H[j+1, j] < 1e-15:
                print('Lucky breakdown.')
                break
            Q[:, j+1] = u/H[j+1, j]
    return Q, H, u


def block_Arnoldi(A: ArrayLike, V: ArrayLike, m: int) -> Tuple[ArrayLike, ArrayLike]:
    """Block Arnoldi algorithm. See Saad, Algorithm 6.23

    Args:
        A (ArrayLike): Matrix of shape (n,n)
        V (ArrayLike): Matrix of shape (n,r)
        m (int): number of iterations

    Returns:
        Q (ArrayLike): Orthogonal matrix
        H (ArrayLike): Additional data
    """
    # CREATE VARIABLE
    n = A.shape[0]
    r = V.shape[1]
    Q = np.zeros((n, m*r))
    H = np.empty((m, m-1), dtype=object)

    # Arnoldi Algorithm
    Q[:, :r] = la.orth(V)
    for j in range(m-1):
        Wj = A.dot(Q[:, j*r:(j+1)*r])
        for i in range(j+1):
            H[i, j] = Q[:, i*r:(i+1)*r].T.dot(Wj)
            Wj = Wj - Q[:, i*r:(i+1)*r].dot(H[i, j])
        Q[:, (j+1)*r:(j+2)*r] = la.orth(Wj)
    return Q, H


def block_Arnoldi_inverted(invA: object, V: ArrayLike, m: int) -> Tuple[ArrayLike, ArrayLike]:
    """Block Arnoldi algorithm. See Saad, Algorithm 6.23

    Args:
        invA (ArrayLike): inverse object of matrix A with a solve function
        V (ArrayLike): Matrix of shape (n,r)
        m (int): number of iterations

    Returns:
        Q (ArrayLike): Orthogonal matrix
        H (ArrayLike): Additional data
    """
    # CREATE VARIABLE
    (n, r) = V.shape
    Q = np.zeros((n, m*r))
    H = np.empty((m, m-1), dtype=object)

    # Arnoldi Algorithm
    Q[:, :r] = la.orth(V)
    for j in range(m-1):
        Wj = invA.solve(Q[:, j*r:(j+1)*r])
        for i in range(j+1):
            H[i, j] = Q[:, i*r:(i+1)*r].T.dot(Wj)
            Wj = Wj - Q[:, i*r:(i+1)*r].dot(H[i, j])
        Q[:, (j+1)*r:(j+2)*r], H[j+1, j] = la.qr(Wj, mode='reduced')
    return Q, H


def Lanczos(A: ArrayLike, v: ArrayLike, k: int) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Lanczos algorithm. Computes orthogonal basis of a Krylov space when A is symmetric.

    Args:
        A (ArrayLike): Symmetric matrix of size (n,n)
        v (ArrayLike): Vector of size (n,)
        k (int): Size of the Krylov space to compute.

    Returns:
        Q (ArrayLike): Orthogonal matrix such that span(v, Av, ..., A**{k-1}v) = span(q[:,1], ..., q[:,k])
        alpha (ArrayLike): Additional data
        beta (ArrayLike): Additional data
        u (ArrayLike): Additional data
    """
    # CREATE VARIABLES
    n = A.shape[0]
    Q = np.zeros((n, k))
    alpha = np.zeros(k)
    beta = np.zeros(k-1)

    # Lanczos Algorithm, see Iterative methods by Martin J. Gander
    Q[:, 0] = v / la.norm(v)
    for j in range(k):
        u = A.dot(Q[:, j])
        alpha[j] = Q[:, j].T.dot(u)
        u = u - alpha[j] * Q[:, j]
        if j > 0:
            u = u - beta[j-1] * Q[:, j-1]
        if j < k-1:
            beta[j] = la.norm(u)
            Q[:, j+1] = u / beta[j]
    return Q, alpha, beta, u


def Lanczos_inverted(invA: object, v: ArrayLike, k: int) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Lanczos algorithm. Computes orthogonal basis of a Krylov space when A is symmetric.

    Args:
        invA (object): object inverse of matrix A with solve function
        v (ArrayLike): Vector of size (n,)
        k (int): Size of the Krylov space to compute.

    Returns:
        Q (ArrayLike): Orthogonal matrix such that span(v, Av, ..., A**{k-1}v) = span(q[:,1], ..., q[:,k])
        alpha (ArrayLike): Additional data
        beta (ArrayLike): Additional data
        u (ArrayLike): Additional data
    """
    # CREATE VARIABLES
    n = v.shape[0]
    Q = np.zeros((n, k))
    alpha = np.zeros(k)
    beta = np.zeros(k-1)

    # Lanczos Algorithm, see Iterative methods by Martin J. Gander
    Q[:, 0] = v / la.norm(v)
    for j in range(k):
        u = invA.solve(Q[:, j])
        alpha[j] = Q[:, j].T.dot(u)
        u = u - alpha[j] * Q[:, j]
        if j > 0:
            u = u - beta[j-1] * Q[:, j-1]
        if j < k-1:
            beta[j] = la.norm(u)
            Q[:, j+1] = u / beta[j]
    return Q, alpha, beta, u


def Givens():
    return NotImplementedError


def Householder():
    return NotImplementedError


def modified_gram_schmidt():
    return NotImplementedError


# %% SHORTCUTS
def compute_krylov_space_basis(k: int, A: ArrayLike, V: ArrayLike, invA: object = None):
    """Compute the basis of Krylov space of size k

    Args:
        k (int): size of krylov space
        A (ArrayLike): Matrix of shape (n,n)
        V (ArrayLike): Matrix of shape(n,r)
        invA (object, optional): Inverse of matrix A with solve function. Defaults to None.
    """
    KS = KrylovSpace(A, V, invA)
    for _ in range(k-1):
        KS.compute_next_basis()
    return KS.Q