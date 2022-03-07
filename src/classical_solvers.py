import time

import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm

import classical_methods
import solution
from IVPs.General import GeneralIVP

valid_classical_methods = {'scipy': classical_methods.solve_by_scipy,
               'closed_form': classical_methods.solve_by_closed_form,
               'optimal': classical_methods.solve_by_optimal,
               'best_rank': classical_methods.solve_best_rank,
               'explicit_RK': classical_methods.solve_by_explicit_Runge_Kutta}

def solve_one_step(problem: GeneralIVP, 
                   t_span: tuple, 
                   X0: ArrayLike, 
                   method: valid_classical_methods.keys() = 'optimal',
                   **args) -> ArrayLike:
    "Solve one step of the selected ODE of the problem, with the selected method. "
    solver = valid_classical_methods[method]
    solution = solver(problem, t_span, X0, **args)
    return solution

def solve(problem: GeneralIVP,
          t_span: tuple,
          X0: ArrayLike,
          method: valid_classical_methods.keys() = 'optimal',
          nb_t_steps: int = 100,
          monitoring: bool = False,
          **args) -> solution.Solution:
    """
    Solves the selected ODE with selected method.
    Return final time solution or a solution object.
    """
    # TIME DISCRET
    t_disc = np.linspace(t_span[0], t_span[1], nb_t_steps + 1)
    
    # MONITORING
    if monitoring:
        loop = tqdm(np.arange(nb_t_steps), desc=f'Solving with {method}')
    else:
        loop = np.arange(nb_t_steps)
        
    # SANITY CHECK
    problem.select_ode(X0, 'F')

    # VARIABLES
    time_of_computation = np.zeros(nb_t_steps)
    X = np.empty(nb_t_steps+1, dtype=object)
    X[0] = X0

    # SOLVE IN LOOP
    for j in loop:
        ts = t_disc[j:j+2]
        t0 = time.time()
        X[j+1] = solve_one_step(problem, ts, X[j], method, **args)
        time_of_computation[j] = time.time() - t0

    if nb_t_steps == 1:
        output = X[1]
    else:
        output = solution.Solution(problem, t_disc, X, time_of_computation)
    return output
