# %% IMPORTATIONS
from types import NoneType
import numpy as np
from numpy.typing import ArrayLike
from typing import Union
from low_rank_toolbox.svd import SVD
from matplotlib import pyplot as plt, cm

# %% CLASS GENERAL IVP
class GeneralIVP:
    r"""general IVP class. Contains useful features for dealing with ivps.

    An general initial value problem (IVP) described by
    .. math::
        d/dt Ys(t) = F(t,Ys(t))
        Ys(t0) = Y0
    """
    # ATTRIBUTES
    is_closed_form_available = False
    dtype = np.float64
    name = 'General'
    is_stiff = False
    

    # %% INIT FUNCTION
    def __init__(self,
                 F: callable,
                 t_span: tuple,
                 Y0: SVD):
        # PROBLEM PARAMETERS
        self.F = F
        self.t_span = t_span
        self.Y0 = Y0

        # DEFAULT ODE
        self.select_ode(Y0, "F")

    def __repr__(self) -> str:
        all_sing_vals = np.zeros(min(self.Y_shape))
        all_sing_vals[:self.initial_rank] = self.Y0.sing_vals
        plt.figure()
        plt.semilogy(all_sing_vals, 'o')
        plt.title('singular values of the initial value')
        plt.grid()
        plt.show()
        return (f"{self.name} problem with following properties \n"
                f"Shape: {self.Y_shape} \n"
                f"Initial rank: {self.initial_rank} \n"
                f"Time interval: {self.t_span} \n"
                f"Current ODE type: {self.current_ode_type} \n"
                f"Closed form solution available: {self.is_closed_form_available}")
        
    def copy(self):
        "Copy the general problem. Specific to a problem."
        return GeneralIVP(self.F, self.t_span, self.Y0)
        

    # %% PROPERTIES
    @property
    def initial_rank(self):
        return self.Y0.rank

    @property
    def Y_shape(self):
        return self.Y0.shape
    
    @property
    def x_space(self):
        return np.linspace(-1, 1, self.Y_shape[0])
    
    @property
    def y_space(self):
        return np.linspace(-1, 1, self.Y_shape[1])
    
    @property
    def meshgrid(self):
        return np.meshgrid(self.x_space, self.y_space)

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
                   initial_value: SVD,
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
    
    

    #%% LOW-RANK FIELDS
    def linear_field(self, t: float, Y: SVD) -> SVD:
        "Linear field of the ODE. Specific to a problem."
        return NotImplementedError

    def non_linear_field(self, t: float, Y: SVD) -> SVD:
        "Non-linear field of the ODE. Specific to a problem."
        return NotImplementedError
    
    def low_rank_ode_F(self, t: float, Y: SVD) -> SVD:
        "Full ODE written in low-rank format."
        dY = self.linear_field(t, Y) + self.non_linear_field(t, Y)
        return dY
    
    def projected_ode_F(self, t:float,  Y: SVD) -> SVD:
        "Compute P(X)[F(X)] in low-rank format."
        FY = self.low_rank_ode_F(t, Y)
        PFY = Y.project_onto_tangent_space(FY)
        return PFY

    # %% CLOSED FORM SOLUTIONS
    def closed_form_solution(self,
                             t_span: tuple,
                             initial_value: Union[ArrayLike, SVD]) -> Union[ArrayLike, SVD]:
        "Closed form solution of the ODE. Specific to a problem."
        return NotImplementedError

    def stiff_solution(self, t_span: tuple, initial_value: SVD) -> SVD:
        "Solution of the stiff part of the ODE. Specific to a problem."
        return NotImplementedError

    def non_stiff_solution(self, t_span: tuple, initial_value: SVD) -> SVD:
        "Solution of the non stiff part of the ODE. Specific to a problem."
        return NotImplementedError
    
    #%% PLOT
    def plot(self, 
             solution: Union[ArrayLike, SVD], 
             title=None, 
             do_save: bool = False, 
             filename: str = None):
        "doc TODO"
        
        # PROCESS DATA
        X, Y = self.meshgrid
        if isinstance(solution, SVD):
            solution = solution.todense()
            
        # MAKE FIGURE
        fig = plt.figure(clear=True)
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, solution, cmap=cm.coolwarm)
        if title is not None:
            plt.title(title, y=1.03, fontsize=16)
        plt.tight_layout()
        
        if do_save:
            if isinstance(filename, NoneType):
                filename = f'figures/{self.name}/plot'
            fig.savefig(filename) 
        
        plt.show()
        return fig
