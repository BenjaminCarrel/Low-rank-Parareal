
import warnings
import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp
from low_rank_toolbox import svd

from IVPs.General import GeneralIVP
from RK_table import RK_rule
from typing import Union
from low_rank_toolbox.low_rank_matrix import LowRankMatrix
from low_rank_toolbox.svd import SVD


# SOLUTION BY SCIPY
def solve_by_scipy(problem: GeneralIVP,
                   t_span: tuple,
                   X0: ArrayLike) -> ArrayLike:
    """
    Solve the selected ODE with scipy.
    """
    # SANITY CHECK
    if isinstance(X0, LowRankMatrix):
            warnings.warn('Efficiency issue: tried to use scipy with a low-rank matrix -> converted to dense matrix')
            X0 = X0.todense()
            
    # RESHAPE INITIAL VALUE
    shape = X0.shape
    X0 = X0.flatten()
    vec_ode = problem.vec_ode
    
    # COMPUTE SOLUTION
    sol = solve_ivp(vec_ode, t_span, X0, 'RK45', atol=1e-13, rtol=1e-13)
    X1 = np.reshape(sol.y[:, -1], shape)
    return X1

# SOLUTION BY CLOSED FORM
def solve_by_closed_form(problem: GeneralIVP,
                         t_span: tuple,
                         X0: Union[ArrayLike, LowRankMatrix]) -> Union[ArrayLike, LowRankMatrix]:
    """
    Solve the selected ODE with closed form.
    """
    # COMPUTE FINAL TIME
    X1 = problem.closed_form_solution(t_span, X0)

    return X1

# SOLUTION BY OPTIMAL
def solve_by_optimal(problem: GeneralIVP,
                     t_span: tuple,
                     X0: Union[ArrayLike, SVD]) -> Union[ArrayLike, SVD]:
    "Optimal solver is closed form if available. Note that scipy may be faster for small problems."
    if problem.is_closed_form_available:
        sol = solve_by_closed_form(problem, t_span, X0)
    else:
        sol = solve_by_scipy(problem, t_span, X0)
    return sol

# BEST RANK SOLUTION
def solve_best_rank(problem: GeneralIVP,
                    t_span: tuple,
                    X0: Union[ArrayLike, LowRankMatrix],
                    rank: int) -> SVD:
    """
    Solve the full problem and then truncate the solution at each time of the grid.
    WARNING: if the solution is dense, very expensive.
    """
    # WARNING
    if np.min(problem.Y_shape) > 200:
        raise ValueError(
            'The problem is too big for solving the full problem and computing its SVD')
    # SOLVE
    full_sol = solve_by_optimal(problem, t_span, X0)
    # TRUNCATE SOLUTION
    if isinstance(full_sol, LowRankMatrix):
        best_rank_sol = full_sol.truncate(rank)
    else:
        best_rank_sol = svd.truncated_svd(full_sol, rank)
    return best_rank_sol


# RUNGE KUTTA METHODS
def solve_by_explicit_Runge_Kutta(problem: GeneralIVP,
                                  t_span: tuple,
                                  X0: ArrayLike,
                                  s: int = 4) -> ArrayLike:
    """Explicit Runge-Kutta methods

    Args:
        problem (GeneralIVP): Problem of interest. Solve the current type of ODE.
        t_span (tuple): Time interval
        X0 (ArrayLike): Initial value
        s (int, optional): Order of the method. Defaults to 4.
    """
    # VARIABLES
    a, b, c = RK_rule(s)
    shape = X0.shape
    eta = np.zeros((s, *shape))
    kappa = np.zeros((s, *shape))
    h = t_span[1] - t_span[0]

    # INITIALIZATION
    eta[0] = X0
    kappa[0] = problem.ode(t_span[0], X0)

    # RUNGE KUTTA LOOP
    for j in np.arange(0, s):
        eta[j] = X0 + h * np.sum(a[j, l] * kappa[l] for l in range(j))
        tj = t_span[0] + h * c[j]
        kappa[j] = problem.ode(tj, eta[j])
    X1 = X0 + h * sum(b[j] * kappa[j] for j in range(s))
    return X1
