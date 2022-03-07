# %% IMPORTATIONS
import numpy as np
from numpy.typing import ArrayLike
from typing import Union
from low_rank_toolbox import svd

# %% CLASS GENERAL


class GeneralIVP:
    r"""General IVP class. Contains useful features for dealing with IVPs.

    An general initial value problem (IVP) described by
    .. math::
        d/dt Ys(t) = F(t,Ys(t))
        Ys(t0) = Y0
    """
    # ATTRIBUTES
    is_closed_form_available = False

    # %% INIT FUNCTION
    def __init__(self,
                 F: callable,
                 t_span: tuple,
                 Y0: svd.SVD):
        # PROBLEM PARAMETERS
        self.F = F
        self.t_span = t_span
        self.Y0 = Y0

        # DEFAULT ODE
        self.select_ode(Y0, "F")

    def __repr__(self) -> str:
        return (f"{type(self)} problem with following properties \n"
                f"Shape: {self.Y_shape} \n"
                f"Initial rank: {self.rank} \n"
                f"Time interval: {self.t_span} \n"
                f"Current ODE type: {self.current_ode_type} \n"
                f"Closed form solution available: {self.is_closed_form_available}")

    # %% PROPERTIES
    @property
    def rank(self):
        return self.Y0.rank

    @property
    def Y_shape(self):
        return self.Y0.shape

    @property
    def ode(self):
        "Current ode"
        return self.valid_odes[self.current_ode_type]

    @property
    def vec_ode(self):
        "Current ode vectorized"
        shape = self.current_ode_shape

        def fun_vec_ode(t, y):
            Y = np.reshape(y, shape)
            dY = self.ode(t, Y)
            return dY.flatten()

        return fun_vec_ode

    # %% GENERAL VECTOR FIELDS (NO NEED TO OVERWRITE THIS)
    def select_ode(self,
                   initial_value: svd.SVD,
                   ode_type: str,
                   mats_UV: tuple = ()):
        """
        Select the current ODE that will be integrated using any of the integrate methods.
        Parameters
        ----------
        ode_type: str
            Can be F, K, S, L, minus_S.
        mats_UV: tuple
            Depending on type, you need to supply the orthonormal matrices U, V in mats_UV. Example: (U,)
        """
        # CHECK ODE TYPE
        self.valid_odes = {'F': self.ode_F,
                           'K': self.ode_K,
                           'S': self.ode_S,
                           'L': self.ode_L,
                           'minus_S': self.ode_minus_S}
        if ode_type not in self.valid_odes.keys():
            raise ValueError("Select an ODE of type {}".format(
                self.valid_odes.keys()))

        # SET CURRENT ODE
        self.current_ode_initial_value = initial_value
        self.current_ode_shape = initial_value.shape
        self.current_ode_type = ode_type
        self.current_ode_mats_UV = mats_UV

    def ode_F(self, t: float, Y: ArrayLike) -> ArrayLike:
        return self.F(t, Y)

    def ode_K(self, t: float, K: ArrayLike) -> ArrayLike:
        (V,) = self.current_ode_mats_UV
        KV = K.dot(V.T)
        dK = self.F(t, KV).dot(V)
        return dK

    def ode_L(self, t: float, L: ArrayLike) -> ArrayLike:
        (U,) = self.current_ode_mats_UV
        UL = U.dot(L.T)
        dL = self.F(t, UL).T.dot(U)
        return dL

    def ode_S(self, t: float, S: ArrayLike) -> ArrayLike:
        (U, V) = self.current_ode_mats_UV
        USVt = np.linalg.multi_dot([U, S, V.T])
        dS = np.linalg.multi_dot([U.T, self.F(t, USVt), V])
        return dS

    def ode_minus_S(self, t: float, S: ArrayLike) -> ArrayLike:
        dS = -self.ode_S(t, S)
        return dS

    #%% LOW-RANK FIELDS (NO NEED TO OVERWRITE)
    def projected_ode_F(self, t:float,  Y: svd.SVD) -> svd.SVD:
        "Compute P(X)[F(X)]."
        DY = self.low_rank_ode_F(t, Y)
        PDY = Y.project_onto_tangent_space(DY)
        return PDY
        
    #%% LOW-RANK FIELDS (NEED TO OVERWRITE)
    def linear_field(self, t: float, X: svd.SVD) -> svd.SVD:
        "Linear field of the problem at time t."
        "Overwrite this for each new problem."
        raise ValueError('Not Implemented for a general problem')

    def non_linear_field(self, t: float, X: svd.SVD) -> svd.SVD:
        "Non-linear field of the problem at time t."
        "Overwrite this for each new problem."
        raise ValueError('Not Implemented for a general problem')

    # %% CLOSED FORM SOLUTIONS (NEED TO OVERWRITE)
    def closed_form_solution(self,
                             t_span: tuple,
                             initial_value: Union[ArrayLike, svd.SVD]) -> Union[ArrayLike, svd.SVD]:
        if self._is_closed_form_available:
            raise ValueError('Not Implemented for general problem')
        else:
            raise ValueError('Closed form solution not available.')

    def stiff_solution(self, t_span: tuple, initial_value: svd.SVD) -> svd.SVD:
        "Solution of the stiff part of the ODE"
        "Overwrite this for each new problem."
        raise ValueError('Not Implemented for this problem')

    def non_stiff_solution(self, t_span: tuple, initial_value: svd.SVD) -> svd.SVD:
        "Solution of the non stiff part of the ODE"
        raise ValueError('Not Implemented for this problem')
