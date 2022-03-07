import numpy as np
import time
from tqdm import tqdm
from IVPs.General import GeneralIVP
import solution
from low_rank_toolbox import svd
import DLRA_methods

valid_DLRA_methods = {'KSL1': DLRA_methods.KSL1,
                      'KSL2': DLRA_methods.KSL2,
                      'unconventional': DLRA_methods.unconventional,
                      'PRK': DLRA_methods.PRK,
                      'strang_splitting': DLRA_methods.strang_splitting}

def solve_DLRA_one_step(problem: GeneralIVP,
                        t_span: tuple,
                        Y0: svd.SVD,
                        DLRA_method: valid_DLRA_methods.keys() = 'KSL2',
                        **args) -> svd.SVD:
    "One step of DLRA with the selected method."
    DLRA_solver = valid_DLRA_methods[DLRA_method]
    Y1 = DLRA_solver(problem, t_span, Y0, **args)
    return Y1


def solve_DLRA(problem: GeneralIVP,
               t_span: tuple,
               Y0: svd.SVD,
               DLRA_method: valid_DLRA_methods.keys() = 'KSL2',
               nb_t_steps: int = 100,
               monitoring: bool = False,
               **args) -> solution.Solution:
    "Solve DLRA with the selected method. The rank is implicitly defined by Y0."
    # TIME DISCRET
    t_disc = np.linspace(t_span[0], t_span[1], nb_t_steps + 1)

    # MONITORING
    if monitoring:
        loop = tqdm(np.arange(nb_t_steps), desc=f'Solving with {DLRA_method}')
    else:
        loop = np.arange(nb_t_steps)

    # VARIABLES
    time_of_computation = np.zeros(nb_t_steps)
    Y = np.empty(nb_t_steps+1, dtype=object)
    Y[0] = Y0

    # SOLVE IN LOOP
    for j in loop:
        ts = t_disc[j:j+2]
        t0 = time.time()
        Y[j+1] = solve_DLRA_one_step(problem, ts, Y[j], DLRA_method, **args)
        time_of_computation[j] = time.time() - t0

    solutions = solution.Solution(problem, t_disc, Y, time_of_computation)
    return solutions
