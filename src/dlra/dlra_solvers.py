from types import NoneType
from typing import Union
import numpy as np
import time
from tqdm import tqdm
from ivps.general import GeneralIVP
from solution import Solution
from low_rank_toolbox.low_rank_matrix import LowRankMatrix
from dlra import dlra_methods

valid_dlra_methods = {'scipy_dlra': dlra_methods.scipy_dlra,
                      'KSL': dlra_methods.KSL,
                      'KSL1': dlra_methods.KSL1,
                      'KSL2': dlra_methods.KSL2,
                      'exact_dlra_linear': dlra_methods.exact_dlra_linear}

def solve_DLRA_one_step(problem: GeneralIVP,
                        t_span: tuple,
                        Y0: LowRankMatrix,
                        DLRA_method: str = 'KSL',
                        **args) -> LowRankMatrix:
    """Solve one step of the DLRA.

    Args:
        problem (GeneralIVP): problem to solve.
        t_span (tuple): initial and final time.
        Y0 (LowRankMatrix): initial value in low-rank format.
        DLRA_method (str, optional): DLRA method to use. Defaults to 'KSL'.
    """
    DLRA_solver = valid_dlra_methods[DLRA_method]
    Y1 = DLRA_solver(problem, t_span, Y0, **args)
    return Y1


def solve_DLRA(problem: GeneralIVP,
               t_span: tuple = None,
               Y0: LowRankMatrix = None,
               rank: int = None, 
               DLRA_method: str = 'KSL',
               nb_t_steps: int = 100,
               monitoring: bool = False,
               return_final_value: bool = False,
               **args) -> Union[Solution, LowRankMatrix]:
    """Solve the DLRA.
    Give time interval, initial value, rank and method.
    If these values are not given, default values obtained from the problem definition are used.

    Args:
        problem (GeneralIVP): problem to solve
        t_span (tuple, optional): time interval. Defaults to None.
        Y0 (LowRankMatrix, optional): initial value in low-rank format. Defaults to None.
        rank (int, optional): rank for DLRA. Defaults to None.
        DLRA_method (str, optional): DLRA method to use. Defaults to 'KSL'.
        nb_t_steps (int, optional): number of time steps. Defaults to 100.
        monitoring (bool, optional): monitor the computations. Defaults to False.
        return_final_value (bool, optional): return only final value to save memory. Defaults to False.
        **arg (optional): additional arguments (order)
    """
    # PROCESS TIME
    if isinstance(t_span, NoneType):
        t_span = problem.t_span
    t_disc = np.linspace(t_span[0], t_span[1], nb_t_steps + 1, endpoint=True)
    
    # PROCESS INITIAL VALUE
    if isinstance(Y0, NoneType):
        Y0 = problem.Y0
        
    # PROCESS RANK
    if isinstance(rank, NoneType):
        rank = Y0.rank
    else:
        if Y0.rank != rank:
            raise ValueError('The rank of the initial value does not correspond to the rank of DLRA')
            # if rank < Y0.rank:
            #     Y0 = Y0.truncate(rank)
            # else:
            #     warnings.warn('The rank of the initial value is lower than the rank of DLRA. The rank of the initial value is kept.')
            #     rank = Y0.rank
    
    # MONITORING
    if monitoring:
        loop = tqdm(np.arange(nb_t_steps), desc=f'Solving with {DLRA_method} of rank {rank}')
    else:
        loop = np.arange(nb_t_steps)

    # RETURN ONLY FINAL VALUE
    if return_final_value:
        Y1 = Y0
        for j in loop:
            ts = t_disc[j:j+2]
            Y1 = solve_DLRA_one_step(problem, ts, Y1, DLRA_method, **args)
        sol = Y1
    
    # SAVE ALL INTERMEDIATE STEPS
    else:
        time_of_computation = np.zeros(nb_t_steps)
        Y = np.empty(nb_t_steps+1, dtype=object)
        Y[0] = Y0
        for j in loop:
            ts = t_disc[j:j+2]
            t0 = time.time()
            Y[j+1] = solve_DLRA_one_step(problem, ts, Y[j], DLRA_method, **args)
            time_of_computation[j] = time.time() - t0

        sol = Solution(problem, t_disc, Y, time_of_computation)
    return sol

    
    