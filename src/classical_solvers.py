import time
from types import NoneType
from typing import Union
import warnings

import numpy as np
from numpy.typing import ArrayLike
from low_rank_toolbox.low_rank_matrix import LowRankMatrix
from low_rank_toolbox import svd
from tqdm import tqdm

import classical_methods
from solution import Solution
from ivps.general import GeneralIVP

valid_classical_methods = {'scipy': classical_methods.solve_by_scipy,
                           'closed_form': classical_methods.solve_by_closed_form,
                           'optimal': classical_methods.solve_by_optimal,
                           'best_rank': classical_methods.solve_best_rank,
                           'explicit_RK': classical_methods.solve_by_explicit_Runge_Kutta}


def solve_one_step(problem: GeneralIVP,
                   t_span: tuple,
                   X0: Union[ArrayLike, LowRankMatrix],
                   method: str = 'optimal',
                   **args) -> Union[ArrayLike, LowRankMatrix]:
    "Solve one step of the selected ODE of the problem, with the selected method."
    solver = valid_classical_methods[method]
    solution = solver(problem, t_span, X0, **args)
    return solution


def solve(problem: GeneralIVP,
          t_span: tuple = None,
          X0: Union[ArrayLike, LowRankMatrix] = None,
          method: str = 'optimal',
          nb_t_steps: int = 100,
          monitoring: bool = False,
          **args) -> Solution:
    """Solve the given problem. The output is low-rank iff the input is low-rank.

    Args:
        problem (GeneralIVP): _description_
        t_span (tuple, optional): _description_. Defaults to None.
        X0 (Union[ArrayLike, LowRankMatrix], optional): _description_. Defaults to None.
        method (str, optional): _description_. Defaults to 'optimal'.
        nb_t_steps (int, optional): _description_. Defaults to 100.
        monitoring (bool, optional): _description_. Defaults to False.
    """
    # PROCESS TIME
    if isinstance(t_span, NoneType):
        t_span = problem.t_span
    t_disc = np.linspace(t_span[0], t_span[1], nb_t_steps + 1, endpoint=True)

    # PROCESS INITIAL VALUE
    if isinstance(X0, NoneType):
        X0 = problem.Y0

    # MONITORING
    if monitoring:
        loop = tqdm(np.arange(nb_t_steps), desc=f'Solving with {method}')
    else:
        loop = np.arange(nb_t_steps)

    # SANITY CHECK
    problem.select_ode(X0, 'F')

    # VARIABLES
    time_of_computation = np.zeros(nb_t_steps)
    X = np.empty(nb_t_steps+1, dtype=np.ndarray)
    X[0] = X0

    # SOLVE IN LOOP
    for j in loop:
        ts = t_disc[j:j+2]
        t0 = time.time()
        X[j+1] = solve_one_step(problem, ts, X[j], method, **args)
        time_of_computation[j] = time.time() - t0

    output = Solution(problem, t_disc, X, time_of_computation)
    return output
