# %% IMPORTATIONS
import numpy as np
import numpy.linalg as la
import scipy as sp
from typing import Union
from scipy.sparse import spmatrix
import classical_solvers
from numpy.typing import ArrayLike
from ivps.general import GeneralIVP
from low_rank_toolbox import svd
from low_rank_toolbox.svd import SVD
from low_rank_toolbox.low_rank_matrix import LowRankMatrix
from matplotlib import pyplot as plt, cm
import scipy.io as sio
import matplotlib as mpl
import matplotlib.animation as animation
from types import NoneType


# %% COOKIE CLASS
class CookieIVP(GeneralIVP):
    r"""Subclass of GeneralIVP. Specific to the cookie problem.

    Cookie differential equation: Y'(t) = - A0 Y(t) - A1 Y(t) C1 + b 1^T
    Initial value: Y(t0) = Y0

    A0 and A1 are sparse matrices
    C1 is a diagonal matrix
    Y0 and B are low-rank matrices
    """
    # ATTRIBUTES
    is_closed_form_available = True
    name = 'Cookie'
    is_stiff = True

    # %% INIT FUNCTION
    def __init__(self,
                 t_span: tuple,
                 Y0: SVD,
                 A0: spmatrix,
                 A1: spmatrix,
                 C1: ArrayLike,
                 b: ArrayLike):
        # SPECIFIC PROPERTIES
        self.A0 = A0
        self.A1 = A1
        self.C1 = C1
        self.b = b
        self.one = np.ones((C1.shape[0], 1))

        # PRE-PROCESSING
        (self.L_C1, self.Q_C1) = sp.linalg.eigh(C1)
        self.B = svd.truncated_svd(b.dot(self.one.T))

        # INITIALIZATION
        GeneralIVP.__init__(self, None, t_span, Y0)
        
    def copy(self):
        "Copy the problem"
        return CookieIVP(self.t_span, self.Y0, self.A0, self.A1, self.C1, self.b)

    # %% SELECTION OF ODE
    def select_ode(self,
                   initial_value: svd.SVD,
                   ode_type: str,
                   mats_UV: tuple = ()):
        "Make precomputations to accelerate the solution of the selected ode"
        # SELECT ODE
        GeneralIVP.select_ode(self, initial_value, ode_type, mats_UV)
        A0, A1, b, C1, one = self.A0, self.A1, self.b, self.C1, self.one
        # PRECOMPUTATIONS
        if ode_type == "F":
            self.D = b.dot(one.T)
        elif ode_type == "K":
            (V,) = mats_UV
            self.V = V
            self.VtC1V = np.linalg.multi_dot([V.T, C1, V])
            self.D = np.linalg.multi_dot([b, one.T, V])
        elif ode_type == "L":
            (U,) = mats_UV
            self.U = U
            self.UtA0tU = np.linalg.multi_dot([A0.dot(U).T, U])
            self.UtA1tU = np.linalg.multi_dot([A1.dot(U).T, U])
            self.D = np.linalg.multi_dot([U.T, b, one.T])
        else:
            U, V = mats_UV
            self.U, self.V = U, V
            self.UtA0tU = U.T.dot(A0.T.dot(U))
            self.UtA1tU = U.T.dot(A1.T.dot(U))
            self.VtC1V = V.T.dot(C1.dot(V))
            self.D = np.linalg.multi_dot([U.T, b, one.T, V])

    # %% VECTOR FIELDS
    def ode_F(self, t: float, X: ArrayLike) -> ArrayLike:
        dY = -self.A0.dot(X) - self.A1.dot(X.dot(self.C1)) + \
            self.b.dot(self.one.T)
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

    def linear_field(self, t: float, Y: SVD) -> SVD:
        linear_part = - Y.dot_sparse(self.A0, side='opposite') - \
            Y.dot_sparse(self.A1, side='opposite').dot(self.C1)
        return linear_part

    def non_linear_field(self, t: float, Y: SVD) -> SVD:
        return self.B

    # def projected_ode_F(self, t: float, Y: SVD) -> SVD:
    #     "Compute P(X)[F(X)]"
    #     # SHORTCUTS
    #     A0, A1, b, C1, one = self.A0, self.A1, self.b, self.C1, self.one
    #     U, S, V = Y.U, Y.S, Y.Vt.T

    #     # STEP 1 : FACTORIZATION
    #     A0US = A0.dot(U.dot(S))
    #     A1USVtC1V = A1.dot(np.linalg.multi_dot([U, S, V.T, C1, V]))
    #     B1tV = np.linalg.multi_dot([b, one.T, V])
    #     UUtA1USVtC1V = np.linalg.multi_dot([U, U.T, A1USVtC1V])
    #     M1 = np.column_stack([-A0US - A1USVtC1V + B1tV + UUtA1USVtC1V, U])
    #     UtB1t = np.linalg.multi_dot([U.T, b, one.T])
    #     UtB1tVVt = np.linalg.multi_dot([UtB1t, V, V.T])
    #     M2 = np.row_stack([V.T, UtB1t - UtB1tVVt])

    #     # STEP 2 : DOUBLE QR
    #     Q1, R1 = np.linalg.qr(M1, mode='reduced')
    #     Q2, R2 = np.linalg.qr(M2.T, mode='reduced')
    #     return svd.SVD(Q1, R1.dot(R2.T), Q2.T)

    # %% CLOSED FORM SOLUTION

    def closed_form_solution(self,
                             t_span: tuple,
                             initial_value: SVD) -> ArrayLike:
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

    def plot(self,
             x_space: ArrayLike,
             y_space: ArrayLike,
             solution: Union[ArrayLike, SVD], 
             title=None, 
             do_save: bool = False, 
             filename: str = None):
            "doc TODO"
            # PROCESS SOLUTION
            if isinstance(solution, SVD):
                solution = solution.todense()
                
            # IMPORT THE MESH
            import scipy.io as spio
            import sys
            problem_cookie = spio.loadmat(sys.path[-1]+'/ivps/data/parametric_cookie_2x2_1580.mat')
            # Construct a triangulation. The triangles are indexed by 1,2, ... in MATLAB hence -1
            FreeDofs = problem_cookie['FreeDofs'][0]
            # we need floating points, not integers
            U_bd = problem_cookie['U_bd'].astype(np.float64).reshape(-1)
            mesh_elements = problem_cookie['Mesh']['Elements'][0][0]
            mesh_coordinates = problem_cookie['Mesh']['Coordinates'][0][0]
            tri = mpl.tri.Triangulation(
                x=mesh_coordinates[:, 0], y=mesh_coordinates[:, 1], triangles=mesh_elements - 1)
            # Plot only the triangles
            # to have bigger plots
            fig, ax = plt.subplots(figsize=(8, 8), dpi=80,
                                facecolor='w', edgecolor='k')
            ax.set_aspect('equal')
            ax.triplot(tri, lw=0.2, color='grey')
            ax.set_title('Triangulation')

            # Combine solution with boundary solution.
            def add_boundary(y):
                u = U_bd.copy()  # just to be sure, copy
                u[FreeDofs - 1] = y  # MATLAB indices start with 1
                return u

            def plot_solution(y, title=None):
                u = add_boundary(y)

                # Now plot. Countourf seems to be the standard choice.
                fig, ax = plt.subplots(figsize=(8, 8), dpi=80,
                                    facecolor='w', edgecolor='k')
                ax.set_aspect('equal')
                ax.triplot(tri, lw=0.1, color='white')  # wireframe of mesh
                tcf = ax.tricontourf(tri, u, 30)  # automatic N levels
                # hardcoded after finding that max is 1.1779
                tcf = ax.tricontourf(tri, u, levels=np.linspace(0, 1.18, 100))
                fig.colorbar(tcf)
                if title is not None:
                    ax.set_title(title)

            plot_solution(solution, title)
            plt.show()
            
            # Save the plot
            if do_save:
                if isinstance(filename, NoneType):
                    filename = f'figures/Cookie/plot'
                fig.savefig(filename)
                
            return fig

# %% CLOSED FORM COOKIE
def closed_form_cookie(t_span: tuple,
                       X0: Union[ArrayLike, SVD],
                       A: ArrayLike,
                       B: ArrayLike,
                       Q: ArrayLike,
                       L: ArrayLike,
                       D: ArrayLike) -> ArrayLike:
    """
    Return the solution of the problem:
    X' = AX + BXC + D
    where C = Q * L * Q^T (C is supposed to be symmetric)
    X(t_0) = X0
    See my master thesis for details.

    Parameters
    ----------
    t_span: tuple
        initial and final time (t0,t1)
    X0: ndarray
        initial value at ts t0
    A, B, Q, L, D: ndarray
        matrices of the problem as described above

    Returns
    -------
    X1: ndarray
        solution at time t1
    """

    # Creation of variables
    h = t_span[1] - t_span[0]
    n, p = D.shape
    m1 = np.zeros(n * p)
    m2 = np.zeros(n * p)
    m3 = np.zeros(n * p)

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
def make_cookie_problem(t_span: tuple, p: int = 101, c0: int = 0, cp: int = 100) -> CookieIVP:
    """
    Return a cookie problem.

    Equation is: Ys' = -A0 Ys - A1 Ys C1 + b 1^T
    """
    # IMPORT DATA (RELATIVE PATH)
    import scipy.io as spio
    import sys
    problem_cookie = spio.loadmat(
        sys.path[-1]+'/ivps/data/parametric_cookie_2x2_1580.mat')

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
        new_X0 = classical_solvers.solve_one_step(cookie_problem, (0, t_span[0]), X0.full())
        new_X0 = svd.truncated_svd(new_X0)
        shifted_t_span = (0, t_span[1] - t_span[0])
        cookie_problem = CookieIVP(shifted_t_span, new_X0, A0, A1, C1, b)

    return cookie_problem




def animation_cookie(times, solutions):
    """
    Animation template for the cookie problem.
    Reference: https://stackoverflow.com/questions/16915966/using-matplotlib-animate-to-animate-a-contour-plot-in-python
    Beware - this might take a while even 10 mins.
    """
    # Import data
    problem_cookie = sio.loadmat('resources/parametric_cookie_2x2_1580.mat')
    # Construct a triangulation. The triangles are indexed by 1,2, ... in MATLAB hence -1
    FreeDofs = problem_cookie['FreeDofs'][0]
    U_bd = problem_cookie['U_bd'].astype(np.float64).reshape(-1)  # we need floating points, not integers
    mesh_elements = problem_cookie['Mesh']['Elements'][0][0]
    mesh_coordinates = problem_cookie['Mesh']['Coordinates'][0][0]
    tri = mpl.tri.Triangulation(x=mesh_coordinates[:, 0], y=mesh_coordinates[:, 1], triangles=mesh_elements - 1)

    # Combine solution with boundary solution.
    def add_boundary(y):
        u = U_bd.copy()  # just to be sure, copy
        u[FreeDofs - 1] = y  # MATLAB indices start with 1
        return u

    def cookies_data(i):  # function returns data to plot
        y = solutions[i]
        y_bd = add_boundary(y)
        return y_bd

    # FIRST IMAGE
    fig = plt.figure(figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.axes(xlim=(0, 4), ylim=(0, 4), xlabel='x', ylabel='y')
    ax.triplot(tri, lw=0.2, color='white')
    tcf = ax.tricontourf(tri, cookies_data(0)) # , levels=color_levels
    fig.colorbar(tcf)

    def animate(i):  # animation function
        global tcf
        u = cookies_data(i)
        for c in tcf.collections:
            c.remove()  # removes only the contours, leaves the rest intact
        ax.triplot(tri, lw=0.01, color='white')
        tcf = ax.tricontourf(tri, u) # , levels=color_levels
        plt.title('time = {}'.format(times[i]))
        return tcf

    anim = animation.FuncAnimation(fig, animate, frames=Nt, repeat=False)
    anim.save('figures/animation_cookie.gif')
    return anim