#%% IMPORTATIONS
import numpy as np
import numpy.linalg as la
import time
import math

from matplotlib import pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from numpy.typing import ArrayLike
from typing import Tuple, Union

from low_rank_toolbox.low_rank_matrix import LowRankMatrix
from low_rank_toolbox import svd
from IVPs.General import GeneralIVP
import solution
import DLRA_solvers



#%% LOW-RANK PARAREAL METHODS
def solve_coarse(coarse_solver: callable,
                 problem: GeneralIVP,
                 t_span: tuple,
                 Y0: svd.SVD,
                 N: int,
                 nb_substeps: int = 1) -> solution.Solution:
    " Apply iteratively the coarse solver "
    # VARIABLES
    Ys = np.empty(N+1, dtype=object)
    Ys[0] = Y0
    ts = np.linspace(t_span[0], t_span[1], N + 1)

    # ITERATION
    time_of_computation = np.zeros(N+1)
    for k in tqdm(range(N), desc=f"Coarse solver ({nb_substeps} substeps)"):
        t_sub = ts[k: k + 2]
        t0 = time.time()
        Ys[k + 1] = coarse_solver(t_sub, Ys[k])
        time_of_computation[k] = time.time() - t0

    sol = solution.Solution(problem, ts, Ys, time_of_computation)
    return sol


def solve_fine(fine_solver: callable,
               problem: GeneralIVP,
               t_span: tuple,
               Y0: svd.SVD,
               N: int,
               nb_substeps: int = 1) -> solution.Solution:
    " Apply iteratively the fine solver "
    # VARIABLES
    Ys = np.empty(N+1, dtype=object)
    Ys[0] = Y0
    ts = np.linspace(t_span[0], t_span[1], N + 1)

    # ITERATION
    time_of_computation = np.zeros(N+1)
    for k in tqdm(range(N), desc=f"Fine solver ({nb_substeps} substeps)"):
        t_sub = ts[k: k + 2]
        t0 = time.time()
        Ys[k + 1] = fine_solver(t_sub, Ys[k])
        time_of_computation[k] = time.time() - t0

    sol = solution.Solution(problem, ts, Ys, time_of_computation)
    return sol


def Parareal(t_span: tuple,
             initial_value: Union[svd.SVD, ArrayLike],
             nb_steps: int,
             coarse_solver: callable,
             fine_solver: callable) -> solution.Solution:
    "The Parareal algorithm. No parallel implementation."

    # DATA CREATION
    N = nb_steps
    K = N
    ts = np.linspace(t_span[0], t_span[1], N + 1)
    Fnk = np.empty((N + 1, K + 1), dtype=object)
    Gnk = np.empty((N + 1, K + 1), dtype=object)
    Unk = np.empty((N + 1, K + 1), dtype=object)

    # INITIAL VALUE
    for k in range(K + 1):
        Unk[0, k] = initial_value

    # INITIAL ITERATION : COARSE INTEGRATION
    for n in range(N):
        Gnk[n + 1, 0] = coarse_solver(ts[n: n + 2], Unk[n, 0])
        Unk[n + 1, 0] = Gnk[n + 1, 0]

    # PARAREAL ITERATION
    for k in tqdm(range(K), desc="Parareal"):
        # SAVE COMPUTATIONS THANKS TO FINITE CONVERGENCE
        Unk[: k + 1, k + 1] = Unk[: k + 1, k]
        for n in np.arange(k, N):
            # Evaluation of fine solver
            Fnk[n + 1, k] = fine_solver(ts[n: n + 2], Unk[n, k])

            # Evaluation of coarse solver
            Gnk[n + 1, k + 1] = coarse_solver(ts[n: n + 2], Unk[n, k + 1])

            # Parareal scheme
            Unk[n + 1, k +
                1] = svd.add_svd([Fnk[n + 1, k], Gnk[n + 1, k + 1], - Gnk[n + 1, k]])

    return ts, Unk


def Low_Rank_Parareal(PB: GeneralIVP,
                      t_span: tuple,
                      Y0: svd.SVD,
                      N: int,
                      coarse_rank: int,
                      fine_rank: int,
                      nb_coarse_substeps: int = 1,
                      nb_fine_substeps: int = 1,
                      DLRA_method: str = 'KSL2') -> Tuple:
    """Low-rank Parareal algorithm

    Args:
        PB (GeneralIVP): problem of interest
        t_span (tuple): time interval (t0, t1)
        Y0 (svd.SVD): initial value low-rank matrix of shape (n,n)
        N (int): number of iteration
        coarse_rank (int): coarse rank for coarse solver
        fine_rank (int): fine rank for fine solver
        nb_coarse_substeps: (int, optional): number of substeps for the coarse solver. Defaults to 1.
        nb_fine_substeps (int, optional): number of substeps for the fine solver. Defaults to 1.
        DLRA_method (str, optional): DLRA method to use; see DLRA_methods.py. Defaults to 'KSL2'.
    """

    # DEFINE COARSE SOLVER AS DLRA WITH COARSE RANK
    def coarse_solver(t_span: tuple, Y0: svd.SVD):
        iv = Y0.truncate(coarse_rank, inplace=False)
        sol = DLRA_solvers.solve_DLRA(PB, t_span, iv, DLRA_method, nb_t_steps=nb_coarse_substeps)
        Y1 = sol.Ys[-1]
        return Y1
    
    # COARSE SOLUTION
    coarse_sol = solve_coarse(coarse_solver, PB, t_span, Y0, N, nb_coarse_substeps)

    # DEFINE FINE SOLVER AS DLRA WITH FINE RANK
    def fine_solver(t_span: tuple, Y0: svd.SVD):
        iv = Y0.truncate(fine_rank, inplace=False)
        sol = DLRA_solvers.solve_DLRA(PB, t_span, iv, DLRA_method, nb_t_steps=nb_fine_substeps)
        Y1 = sol.Ys[-1]
        return Y1
    
    # FINE SOLUTION
    fine_sol = solve_fine(fine_solver, PB, t_span, Y0, N, nb_fine_substeps)

    # LOW-RANK PARAREAL ROUTINE
    ## DATA CREATION
    ts = np.linspace(t_span[0], t_span[1], N + 1)
    Fnk = np.empty((N + 1, N + 1), dtype=object)
    Gnk = np.empty((N + 1, N + 1), dtype=object)
    Ynk = np.empty((N + 1, N + 1), dtype=object)
    
    ## INITIAL VALUE
    for k in range(N + 1):
        Ynk[0, k] = Y0
    
    ## INITIAL APPROXIMATION -> COARSE INTEGRATION
    for n in range(N):
        Gnk[n + 1, 0] = coarse_solver(ts[n: n + 2], Ynk[n, 0])
        Ynk[n + 1, 0] = Gnk[n + 1, 0]
        
    ## PARAREAL ITERATION
    for k in tqdm(range(N), desc=f"Low-rank Parareal"):
        # SAVE COMPUTATIONS THANKS TO FINITE CONVERGENCE
        Ynk[: k + 1, k + 1] = Ynk[: k + 1, k]
        
        for n in np.arange(k, N):
            # EVALUATION OF FINE SOLVER -> CAN BE DONE IN PARALLEL
            Fnk[n + 1, k] = fine_solver(ts[n: n + 2], Ynk[n, k])

            # EVALUATION OF COARSE SOLVER
            Gnk[n + 1, k + 1] = coarse_solver(ts[n: n + 2], Ynk[n, k + 1])

            # PARAREAL SCHEME
            Ynk[n + 1, k + 1] = svd.add_svd([Fnk[n + 1, k], Gnk[n + 1, k + 1], - Gnk[n + 1, k]])

    return ts, Ynk, coarse_sol, fine_sol


# %% PARAREAL ANIMATION
def parareal_animation(Ynk: ArrayLike, 
                       coarse_sol: solution.Solution, 
                       fine_sol: solution.Solution, 
                       reference_sol: solution.Solution, 
                       title: str = 'parareal_animation', 
                       coarse_name: str = 'Coarse solver', 
                       fine_name: str = 'Fine solver', 
                       do_save: bool = True):
    
    # PARAMETERS
    N = reference_sol.nb_t_steps
    ts = reference_sol.ts

    # SOLVER ERRORS
    coarse_error = coarse_sol.compute_errors(reference_sol)
    fine_error = fine_sol.compute_errors(reference_sol)
    
    # PARAREAL ERROR
    enk = compute_parareal_error(reference_sol, Ynk)
    
    # GLOBAL ERROR : MAX OVER TIME
    
    err_global_coarse = np.max(coarse_error) * np.ones(N)
    err_global_fine = np.max(fine_error) * np.ones(N)
    max_enk = np.max(enk, axis=0) * np.ones(N)

    # ANIMATION
    # Figure 1
    fig, (ax1, ax2) = plt.subplots(2, dpi=100, figsize=(10, 10))
    ax1.semilogy(ts, coarse_error, "r--", label=coarse_name)
    ax1.semilogy(ts, fine_error, "g--", label=fine_name)
    plot1 = ax1.semilogy(ts, enk[:, 0], "b", label="Parareal")
    ax1.legend(loc="right")
    ax1.grid()
    ax1.set(
        xlabel="ts",
        ylabel="L^2-error with exact solution",
        title=f"Parareal iteration 0/{N}",
    )

    # Figure 2
    it = np.arange(0, N)
    ax2.semilogy(it, err_global_coarse, "r--", label=coarse_name)
    ax2.semilogy(it, err_global_fine, "g--", label=fine_name)
    plot2 = ax2.semilogy(it[0:1], max_enk[0:1], "bo", label="Parareal")
    ax2.legend(loc="right")
    ax2.grid()
    ax2.set(xlabel="iterations", ylabel="Max ts error", title=title)
    plt.close()

    # Function of updating
    def update_plot(frame_number, data, plot1, plot2):
        ax1.lines[2].remove()
        plot1 = ax1.semilogy(ts, data[:, frame_number], "b")
        ax1.set_title(f"Parareal iteration {frame_number}/{N-1}")

        ax2.lines[2].remove()
        plot2 = ax2.semilogy(
            it[0: frame_number + 1], max_enk[0: frame_number + 1], "bo"
        )

    anim = animation.FuncAnimation(
        fig, update_plot, N, fargs=(enk, plot1, plot2))
    if do_save:
        anim.save(f"{title}.gif", writer=animation.PillowWriter(fps=5))

    plt.figure()
    plt.semilogy(it, err_global_coarse, "r--", label=coarse_name)
    plt.semilogy(it, err_global_fine, "g--", label=fine_name)
    plt.semilogy(it, max_enk, "bo", label='Low-rank Parareal')
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Max error over ts")
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.show()

    return anim, coarse_error, fine_error, enk


# %% PARAREAL ERRORS AND BOUNDS
def compute_parareal_error(reference_sol: solution.Solution, Unk: ArrayLike) -> ArrayLike:
    (N, K) = Unk.shape
    Enk = np.zeros(Unk.shape)
    for k in range(K):
        for n in range(N):
            if isinstance(reference_sol.Ys[n], LowRankMatrix):
                Enk[n, k] = (Unk[n][k] - reference_sol.Ys[n]).norm()
            else:
                Enk[n, k] = la.norm(Unk[n][k] - reference_sol.Ys[n])
    return Enk


def compute_max_error_over_time(
    exact_sol: solution.Solution,
    coarse_sol: solution.Solution,
    fine_sol: solution.Solution,
    Unk: ArrayLike,
) -> Tuple:
    (N, K) = Unk.shape
    err_para = np.zeros(N)
    for k in range(K):
        err = [la.norm(Unk[n][k] - exact_sol.Ys[n]) for n in range(N)]
        err_para[k] = np.max(err)

    err_coarse = np.max(coarse_sol.compute_errors(exact_sol)) * np.ones(N)
    err_fine = np.max(fine_sol.compute_errors(exact_sol)) * np.ones(N)
    return err_para, err_coarse, err_fine


def linear_bound(
    alpha: float, beta: float, gamma: float, kappa: float, nb_steps: int
) -> ArrayLike:
    k = np.arange(0, nb_steps+1)
    linear_bd = (alpha/(1-beta)) ** k * gamma + kappa/(1-alpha-beta)
    return linear_bd


def superlinear_bound(
    alpha: float, beta: float, gamma: float, kappa: float, nb_steps: int
) -> ArrayLike:
    iter = np.arange(0, nb_steps+1)
    superlinear_bd = np.zeros(nb_steps+1)
    n = nb_steps
    for k in iter+1:
        superlinear_bd[k-1] = alpha ** k / math.factorial(k-1) * np.prod(
            [float(n-j) for j in np.arange(2, k+1)]) / (1-beta) * gamma + kappa / (1-alpha-beta)
    return superlinear_bd


