from __future__ import annotations
from copy import deepcopy
import numpy as np
from typing import List, Optional, Sequence, Tuple, Type, Union
from numpy.typing import ArrayLike
import scipy.sparse.linalg as spala


class LowRankMatrix:
    """
    Meta class for dealing with low rank matrices in different formats.

    Do not use this class directly, but rather use its subclasses.

    We always decompose a matrix as a product of smaller matrices. These smaller
    matrices are stored in ``self._matrices``.
    """

    _format = "generic"

    def __init__(
        self,
        *matrices: Sequence[ArrayLike],
        **extra_data,
    ):
        # Convert so values can be changed.
        self._matrices = list(matrices)
        self._extra_data = extra_data

    @property
    def rank(self) -> int:
        return min(min(M.shape) for M in self._matrices)

    @property
    def length(self) -> int:
        "Number of matrices"
        return len(self._matrices)

    @property
    def shape(self) -> tuple:
        return (self._matrices[0].shape[0], self._matrices[-1].shape[-1])

    @property
    def deepshape(self) -> tuple:
        return tuple(
            M.shape[0] for M in self._matrices if len(M.shape) == 2
        ) + (self._matrices[-1].shape[-1],)

    @property
    def dtype(self):
        return self._matrices[0].dtype

    def __repr__(self) -> str:
        return (
            f"{self.shape} low-rank matrix rank {self.rank}"
            f" and type {self.__class__._format}."
        )

    def copy(self) -> LowRankMatrix:
        return deepcopy(self)

    def __add__(self, other: Union[LowRankMatrix, ArrayLike]) -> LowRankMatrix:
        """Generic addition method"""
        # other is a LowRankMatrix
        if isinstance(other, LowRankMatrix):
            sum = self.full() + other.full()
        else:
            sum = self.full() + other
        return sum

    def __imul__(self, other: float) -> LowRankMatrix:
        self._matrices[0] *= other
        return self

    def __mul__(self, other: float) -> LowRankMatrix:
        """Scalar multiplication"""
        new_mat = self.copy()
        new_mat.__imul__(other)
        return new_mat

    __rmul__ = __mul__

    def __neg__(self) -> LowRankMatrix:
        return -1 * self

    def __sub__(self, other: LowRankMatrix) -> LowRankMatrix:
        return self + (-1) * other

    def convert(self, format: Type[LowRankMatrix]) -> LowRankMatrix:
        """Convert the low rank matrix to a different format. Generic method,
        this may not be the fastest in every situation."""
        return format.from_full(self.full())

    @classmethod
    def from_full(cls, matrix: ArrayLike):
        """Convert a full matrix to a low rank matrix. Needs to be implemented
        separately by every subclass."""
        raise NotImplementedError

    def full(self) -> ArrayLike:
        " Multiply all factors in optimal order "
        return np.linalg.multi_dot(self._matrices)

    def todense(self) -> ArrayLike:
        return self.full()

    #%% MATRIX MULTIPLICATIONS
    def dot(self,
            other: Union[LowRankMatrix, ArrayLike],
            side: str = 'usual',
            dense_output: bool = False) -> Union[ArrayLike, LowRankMatrix]:
        """Matrix and vector multiplication

        Args:
            other (Union[LowRankMatrix, ArrayLike]): Matrix to multiply with
            side (str, optional): Can be 'usual' or 'opposite'. Defaults to 'usual'.
            dense_output (bool, optional): Return a dense matrix if True. Defaults to False.
        """
        # MATRIX-VECTOR CASE
        if len(other.shape) == 1:
            dense_output = True

        if dense_output:
            if side == 'usual':
                return np.linalg.multi_dot(self._matrices + [other])
            elif side == 'opposite':
                return np.linalg.multi_dot([other] + self._matrices)
            else:
                raise ValueError('Incorrect side. Choose "usual" or "opposite".')

        new_matrix = self.copy()
        if side == 'usual':
            new_matrix._matrices[-1] = self._matrices[-1].dot(other)
            return new_matrix
        elif side == 'opposite':
            new_matrix._matrices[0] = other.dot(self._matrices[0])
            return new_matrix
        else:
            raise ValueError('Incorrect side. Choose "usual" or "opposite".')

    __matmul__ = dot

    def dot_sparse(self,
                   sparse_other: ArrayLike,
                   side: str = 'usual',
                   dense_output: bool = False) -> Union[ArrayLike, LowRankMatrix]:
        """
        Efficient sparse multiplication
        usual: output = matrix @ sparse_other
        opposite: output = sparse_other @ matrix
        """
        sparse_other = sparse_other.tocsc() # sanity check
        new_matrix = self.copy()
        if side == 'usual':
            new_matrix._matrices[-1] = (sparse_other.T.dot(new_matrix._matrices[-1].T)).T
        elif side == 'opposite':
            new_matrix._matrices[0] = sparse_other.dot(new_matrix._matrices[0])
        else:
            raise ValueError('incorrect side')
        if dense_output:
            return new_matrix.todense()
        return new_matrix

    def expm_multiply(self,
                      A: ArrayLike,
                      h: float,
                      side: str = 'left',
                      dense_output: bool = False) -> Union[ArrayLike, LowRankMatrix]:
        """
        Efficient exponential matrix multiplication
        left: output = exp(h*A) @ matrix
        right: output = matrix @ exp(h*A)
        """
        A = A.tocsc()  # sanity check
        new_matrix = self.copy()
        if side == 'left':
            new_matrix._matrices[0] = spala.expm_multiply(A, new_matrix._matrices[0], start=0, stop=h, num=2, endpoint=True)[-1]
        elif side == 'right':
            new_matrix._matrices[-1] = spala.expm_multiply(A.T, new_matrix._matrices[-1].T, start=0, stop=h, num=2, endpoint=True)[-1].T
        else:
            raise ValueError('incorrect side')
        if dense_output:
            return new_matrix.todense()
        return new_matrix

    @property
    def norm(self) -> float:
        """Default implementation, overload this for some subclasses"""
        return np.trace(self.T.dot(self, dense_output=True))

    @property
    def T(self):
        # reverse order of matrices and transpose each element
        return self.__class__(
            *[np.transpose(M) for M in self._matrices[::-1]], **self._extra_data
        )
        

    def gather(self, indices: ArrayLike) -> ArrayLike:
        """Access entries of the full matrix indexed by ``indices``.

        This is faster and more memory-efficient than forming full matrix. Very
        useful for e.g. matrix completion tasks, or estimating reconstruction
        error on large matrices.
        """
        raise NotImplementedError

    @staticmethod
    def create_matrix_alias(index: int) -> property:
        def getter(self) -> ArrayLike:
            return self._matrices[index]

        def setter(self, value: ArrayLike):
            self._matrices[index] = value

        return property(getter, setter)

    @staticmethod
    def create_data_alias(key: "str") -> property:
        def getter(self) -> ArrayLike:
            return self._extra_data[key]

        def setter(self, value: ArrayLike):
            self._extra_data[key] = value

        return property(getter, setter)
