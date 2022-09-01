#%% IMPORTATIONS
from types import NoneType
import numpy as np
import numpy.linalg as la
import time
import math

from matplotlib import pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from numpy.typing import ArrayLike
from typing import Tuple, Union
import joblib
from joblib import Parallel, delayed

from low_rank_toolbox.low_rank_matrix import LowRankMatrix
from low_rank_toolbox import svd
from low_rank_toolbox.svd import SVD
from ivps.general import GeneralIVP
import solution
from solution import Solution
from dlra import dlra_solvers
import classical_solvers



#%% PARAREAL CLASS
class Parareal:
    
    
    # ATTRIBUTES
    name = "Parareal"
    
    
    
    def __init__(self, 
                 problem: GeneralIVP,
                 nb_parareal_iter: int,
                 coarse_method: str, 
                 fine_method: str,
                 nb_coarse_substeps: int = 1,
                 nb_fine_substeps: int = 1,
                 reference_solution: Solution = None) -> None:
        """Initialise a Parareal object

        Args:
            problem (GeneralIVP): problem to solve
            nb_parareal_iter (int): number of parareal iterations
            coarse_method (str): coarse method to use, see classical_solvers.py
            fine_solver (str): fine method to use, see classical_solvers.py
            nb_coarse_substeps (int, optional): number of substeps done by the coarse solver. Defaults to 1.
            nb_fine_substeps (int, optional): number of substeps done by the fine solver. Defaults to 1.
            reference_solution (Solution, optional): reference solution of the problem. Defaults to None.
        """
        self.problem = problem
        self.coarse_method = coarse_method
        self.fine_method = fine_method
        self.nb_parareal_iter = nb_parareal_iter
        self.nb_coarse_substeps = nb_coarse_substeps
        self.nb_fine_substeps = nb_fine_substeps
        self.saved_reference_solution = reference_solution
        pass
    
    
    def __repr__(self):
        return (f'{self.name} method with following properties: \n' +
              f'Problem: {self.problem.name} of shape {self.problem.Y_shape} \n' + 
              f'Coarse solver: {self.coarse_method} with {self.nb_coarse_substeps} substep(s) \n' +
              f'Fine solver: {self.fine_method} with {self.nb_fine_substeps} substep(s) \n' +
              f'Number of parareal iterations: {self.nb_parareal_iter}')
        
    #%% PROPERTIES
    @property
    def coarse_label(self):
        return f"{self.coarse_method} [{self.nb_coarse_substeps} substep(s)]"
    
    @property
    def fine_label(self):
        return f"{self.fine_method} [{self.nb_fine_substeps} substep(s)]"
    
    @property
    def ts(self):
        "Time discretization of Parareal"
        return np.linspace(self.problem.t_span[0], self.problem.t_span[1], self.nb_parareal_iter+1, endpoint=True)
    
    @property
    def coarse_solution(self) -> Solution:
        "Return the coarse solution. Compute it if not already done."
        try:
            coarse_solution = self.saved_coarse_solution
        except:
            coarse_solution = self.compute_coarse_solution()
        return coarse_solution
    
    @property
    def fine_solution(self) -> Solution:
        "Return the fine solution. Compute it if not already done."
        try:
            fine_solution = self.saved_fine_solution
        except:
            fine_solution = self.compute_fine_solution()
        return fine_solution
    
    @property
    def reference_solution(self) -> Solution:
        "Return the reference solution. Compute it if not already done."
        if isinstance(self.saved_reference_solution, NoneType):
            self.compute_reference_solution()
        return self.saved_reference_solution
        
        
    #%% COARSE AND FINE SOLVERS ROUTINES
    def coarse_solver(self, t_span, X0):
        "Apply the number of coarse substeps of the coarse method, which is one step from the POV of Parareal"
        sol = classical_solvers.solve(self.problem, t_span, X0, method=self.coarse_method, nb_t_steps=self.nb_coarse_substeps, monitoring=False)
        X1 = sol.Ys[-1]
        return X1
        
    def compute_coarse_solution(self):
        "Compute the coarse solution with the number of parareal iterations"
        t_span = self.problem.t_span
        ts = np.linspace(t_span[0], t_span[1], self.nb_parareal_iter+1, endpoint=True)
        y0 = self.problem.Y0
        coarse_solution = np.empty(self.nb_parareal_iter+1, dtype=object)
        coarse_solution[0] = y0
        time_of_computation = np.zeros(self.nb_parareal_iter)
        
        # COARSE SOLVER LOOP OVER TIME
        for i in tqdm(np.arange(self.nb_parareal_iter), desc=self.coarse_label):
            t_subspan = ts[i:i+2]
            t0 = time.time()
            coarse_solution[i+1] = self.coarse_solver(t_subspan, coarse_solution[i])
            time_of_computation[i] = time.time() - t0
            
        # SAVE THE COARSE SOLUTION
        self.saved_coarse_solution = Solution(self.problem, ts, coarse_solution, time_of_computation)
            
        return self.saved_coarse_solution
    
    def fine_solver(self, t_span, X0):
        "Apply the number of fine substeps of the fine method, which is one step from the POV of Parareal"
        sol = classical_solvers.solve(self.problem, t_span, X0, method=self.fine_method, nb_t_steps=self.nb_fine_substeps, monitoring=False)
        X1 = sol.Ys[-1]
        return X1
    
    def compute_fine_solution(self):
        "Compute the coarse solution with the number of parareal iterations"
        t_span = self.problem.t_span
        ts = np.linspace(t_span[0], t_span[1], self.nb_parareal_iter+1, endpoint=True)
        y0 = self.problem.Y0
        fine_solution = np.empty(self.nb_parareal_iter+1, dtype=object)
        fine_solution[0] = y0
        time_of_computation = np.zeros(self.nb_parareal_iter)
        
        # FINE SOLVER LOOP OVER TIME
        for i in tqdm(np.arange(self.nb_parareal_iter), desc=self.fine_label):
            t_subspan = ts[i:i+2]
            t0 = time.time()
            fine_solution[i+1] = self.fine_solver(t_subspan, fine_solution[i])
            time_of_computation[i] = time.time() - t0
            
        # SAVE FINE SOLUTION
        self.saved_fine_solution = Solution(self.problem, ts, fine_solution, time_of_computation)
            
        return self.saved_fine_solution
    
    
    def compute_reference_solution(self, method: str = 'optimal', **args):
        "Compute the reference solution"
        self.saved_reference_solution = classical_solvers.solve(self.problem, method=method, nb_t_steps=self.nb_parareal_iter, monitoring=True, **args)
        return self.saved_reference_solution
    
    
    #%% PARAREAL ROUTINES
    def initial_approximation(self):
        "Compute the initial approximation. Can be overwritten for modified methods."
        t_span = self.problem.t_span
        N = self.nb_parareal_iter
        ts = np.linspace(t_span[0], t_span[1], N + 1, endpoint=True)
        Gn0 = np.empty(N + 1, dtype=object)
        Un0 = np.empty(N + 1, dtype=object)
        Un0[0] = self.problem.Y0
        for n in np.arange(N):
                Gn0[n + 1] = self.coarse_solver(ts[n: n + 2], Un0[n])
                Un0[n + 1] = Gn0[n + 1]
        return Gn0, Un0
    
    
    def compute_parareal(self, skip_confirmation: bool = False, do_parallel: bool = True):
        "Apply the Parareal algorithm. It can take from few minutes to several hours, so check the parameters before run."
    
        # COMPUTE COARSE AND FINE SOLUTIONS
        self.compute_coarse_solution()
        time_coarse = np.sum(self.coarse_solution.time_of_computation)
        self.compute_fine_solution()
        time_fine = np.sum(self.fine_solution.time_of_computation)
        self.compute_reference_solution()
        estimated_time = (self.nb_parareal_iter+1)/2 * (time_coarse + time_fine)
        
        
        print("Please check the parameters:")
        print(f"The problem to solve is {self.problem.name} of shape {self.problem.Y_shape}.")
        print(f"The coarse solver is {self.coarse_label}.")
        print(f"The fine solver is {self.fine_label}.")
        print(f"The total time of computation is estimated at {round(estimated_time)} seconds.")
        time.sleep(1)
        if skip_confirmation:
            answer = 'yes'
        else:
            answer = input(f"Do you want to continue? (yes/y or no/n)")
        if answer.lower() in ['y', 'yes', 'yeah', 'oui', 'ja', 'si', 'da']:
            print('Started Parareal computations.')
            
            # DATA CREATION
            t_span = self.problem.t_span
            initial_value = self.problem.Y0
            N = self.nb_parareal_iter
            K = N
            ts = np.linspace(t_span[0], t_span[1], N + 1, endpoint=True)
            Fnk = np.empty((N + 1, K + 1), dtype=object)
            Gnk = np.empty((N + 1, K + 1), dtype=object)
            Unk = np.empty((N + 1, K + 1), dtype=object)

            # INITIAL VALUE
            for k in np.arange(K + 1):
                Unk[0, k] = initial_value

            # INITIAL APPROXIMATION : COARSE INTEGRATION
            t0 = time.time()
            Gnk[:, 0], Unk[:, 0] = self.initial_approximation()

            # PARAREAL ITERATION
            for k in tqdm(np.arange(K), desc=f"{self.name} applied to {self.problem.name} {self.problem.Y_shape}"):
                # SAVE COMPUTATIONS THANKS TO FINITE CONVERGENCE
                Unk[: k + 1, k + 1] = Unk[: k + 1, k]

                # Evaluation of fine solver, done in parallel if possible
                if do_parallel:
                    nb_process = min(joblib.cpu_count(), N-k)
                    Fnk[k+1:, k] = Parallel(n_jobs=nb_process)(delayed(self.fine_solver)(ts[n: n + 2], Unk[n, k]) for n in np.arange(k, N))
                else:
                    for n in np.arange(k, N):
                        Fnk[n + 1, k] = self.fine_solver(ts[n: n + 2], Unk[n, k])

                # Sequential part
                for n in np.arange(k, N):
                    # Evaluation of coarse solver
                    Gnk[n + 1, k + 1] = self.coarse_solver(ts[n: n + 2], Unk[n, k + 1])

                    # Parareal scheme
                    Unk[n + 1, k + 1] = Fnk[n + 1, k] + Gnk[n + 1, k + 1] - Gnk[n + 1, k]
            
            # SAVE COMPUTATIONS
            self.Gnk = Gnk
            self.Fnk = Fnk
            # self.Unk = Unk
            self.parareal_solution = Unk
            self.saved_parareal_solution = Unk
            total_time = time.time() - t0
            print(f'Done. Total time of computation: {round(total_time)} seconds.')

            return self.parareal_solution

        else:
            print('Computations cancelled.')
        
        pass
        
    #%% ERRORS RELATED ROUTINES
    def compute_coarse_error(self) -> ArrayLike:
        "Error of the coarse solver"
        return self.coarse_solution.compute_relative_errors(self.reference_solution)
    
    def compute_fine_error(self) -> ArrayLike:
        "Error of the fine solver"
        return self.fine_solution.compute_relative_errors(self.reference_solution)
    
    def compute_parareal_error(self) -> ArrayLike:
        "Error of Parareal"
        (N, K) = self.parareal_solution.shape
        Enk = np.zeros((N, K))
        reference_solution: Solution = self.reference_solution
        parareal_solution: ArrayLike = self.parareal_solution
        for k in np.arange(K):
            for n in np.arange(N):
                if isinstance(reference_solution.Ys[n], LowRankMatrix) and isinstance(parareal_solution[n][k], LowRankMatrix):
                    Enk[n, k] = (parareal_solution[n][k] - reference_solution.Ys[n]).norm()/reference_solution.Ys[n].norm()
                else:
                    Enk[n, k] = la.norm(parareal_solution[n][k] - reference_solution.Ys[n])/la.norm(reference_solution.Ys[n])
        self.Enk = Enk   
        return Enk
    
    #%% PLOT ROUTINES
    def plot_coarse_and_fine_errors(self, title: str, do_save: bool = False, filename: str = None):
        """Plot the coarse and the fine errors

        Args:
            title (str): title of the plot
            do_save (bool, optional): save the plot, or not. Defaults to False.
            filename (str, optional): name of the file if saved (automatic to .pdf). Defaults to None.
        """
        # COMPUTE ERRORS
        ts = self.ts
        coarse_error = self.compute_coarse_error()
        fine_error = self.compute_fine_error()
        
        # PLOT
        fig = plt.figure()
        plt.semilogy(ts, coarse_error, label=self.coarse_label)
        plt.semilogy(ts, fine_error, label=self.fine_label)
        plt.grid()
        plt.legend()
        plt.title(title)
        plt.xlabel('time')
        plt.ylabel('error')
        plt.show()
        
        if do_save:
            fig.savefig(filename)
        
        return fig
        
        
        
    def animation_parareal(self, title: str, do_save: bool = False, filename: str = None) -> animation:
        """Generates an animation of Parareal

        Args:
            title (str): title of the plot
            do_save (bool, optional): save the animation, or not. Defaults to False.
            filename (str, optional): name of the file if saved (automatic to .gif). Defaults to None.
        """
        
        
        # PARAMETERS
        N = self.nb_parareal_iter+1
        ts = self.ts

        # COMPUTE ERRORS
        coarse_error = self.compute_coarse_error()
        fine_error = self.compute_fine_error()
        Enk = self.compute_parareal_error()
        
        # GLOBAL ERROR : MAX OVER TIME
        err_global_coarse = np.max(coarse_error) * np.ones(N)
        err_global_fine = np.max(fine_error) * np.ones(N)
        max_enk = np.max(Enk, axis=0) * np.ones(N)

        # ANIMATION
        # Figure 1
        fig, (ax1, ax2) = plt.subplots(2, dpi=100, figsize=(10, 10))
        ax1.semilogy(ts, coarse_error, "r--", label=self.coarse_label)
        ax1.semilogy(ts, fine_error, "g--", label=self.fine_label)
        plot1 = ax1.semilogy(ts, Enk[:, 0], "b", label=self.name)
        ax1.legend(loc="right")
        ax1.grid()
        ax1.set(
            xlabel="time",
            ylabel="L^2-error",
            title=f"{self.name} - iteration 0/{N}",
        )

        # Figure 2
        it = np.arange(0, N)
        ax2.semilogy(it, err_global_coarse, "r--", label=self.coarse_label)
        ax2.semilogy(it, err_global_fine, "g--", label=self.fine_label)
        plot2 = ax2.semilogy(it[0:1], max_enk[0:1], "bo", label=self.name)
        ax2.legend(loc="right")
        ax2.grid()
        ax2.set(
            xlabel="iterations", 
            ylabel="Max error over time", 
            title=f"{self.name} - {self.problem.name} {self.problem.Y_shape}")
        plt.close()

        # Function of updating
        def update_plot(frame_number, data, plot1, plot2):
            ax1.lines[2].remove()
            plot1 = ax1.semilogy(ts, data[:, frame_number], "b")
            ax1.set_title(f"{self.name} - iteration {frame_number}/{N-1}")

            ax2.lines[2].remove()
            plot2 = ax2.semilogy(
                it[0: frame_number + 1], max_enk[0: frame_number + 1], "bo"
            )

        anim = animation.FuncAnimation(
            fig, update_plot, N, fargs=(Enk, plot1, plot2))
        if do_save:
            anim.save(filename, writer=animation.PillowWriter(fps=5))

        plt.figure()
        plt.semilogy(it, err_global_coarse, "r--", label=self.coarse_label)
        plt.semilogy(it, err_global_fine, "g--", label=self.fine_label)
        plt.semilogy(it, max_enk, "bo", label=self.name)
        plt.legend()
        plt.xlabel("iterations")
        plt.ylabel("Max error over time")
        plt.title(f"{self.name} - {self.problem.name} {self.problem.Y_shape}")
        plt.grid()
        plt.tight_layout()
        plt.show()

        return anim
      
            
class LowRankParareal(Parareal):
    
    # ATTRIBUTES
    name = "Low-rank Parareal"
    
    def __init__(self, 
                 problem: GeneralIVP, 
                 coarse_rank: int, 
                 fine_rank: int, 
                 nb_parareal_iter: int, 
                 nb_coarse_substeps: int = 1,
                 coarse_dlra_method: str = 'scipy_dlra',
                 nb_fine_substeps: int = 1, 
                 fine_dlra_method: str = 'scipy_dlra',
                 reference_solution: Solution = None) -> None:
        """Initialize a low-rank Parareal object

        Args:
            problem (GeneralIVP): problem to solve
            coarse_rank (int): coarse rank $q$
            fine_rank (int): fine rank $r$
            nb_parareal_iter (int): number of parareal iterations
            nb_coarse_substeps (int, optional): number of substeps for the coarse solver. Defaults to 1.
            coarse_dlra_method (str, optional): name of the coarse DLRA method. Defaults to 'scipy_dlra'.
            nb_fine_substeps (int, optional): number of substeps for the fine solver. Defaults to 1.
            fine_dlra_method (str, optional): name of the fine DLRA method. Defaults to 'scipy_dlra'.
            reference_solution (Solution, optional): reference solution for computing the error. Defaults to None.
        """
        # INIT
        super().__init__(problem, nb_parareal_iter, None, None, nb_coarse_substeps, nb_fine_substeps, reference_solution)
        # SPECIFIC VALUES
        self.coarse_rank = coarse_rank
        self.coarse_method = coarse_dlra_method
        self.fine_rank = fine_rank
        self.fine_method = fine_dlra_method
        
    
    @property
    def coarse_label(self):
        return f"DLRA (rank {self.coarse_rank})"
    
    @property
    def fine_label(self):
        return f"DLRA (rank {self.fine_rank})"
    
    
    
    #%% COARSE AND FINE SOLVERS SPECIFIC TO LOW-RANK PARAREAL
    def coarse_solver(self, t_span: tuple, iv: SVD) -> SVD:
        """Low-rank Parareal's coarse solver is a truncation to the coarse rank and then DLRA of the coarse rank

        Args:
            t_span (tuple): time interval
            iv (SVD): initial value in SVD format
        """
        truncated_iv = iv.truncate(r=self.coarse_rank, inplace=False)
        Y1 = dlra_solvers.solve_DLRA(problem=self.problem, 
                                    t_span=t_span, 
                                    Y0=truncated_iv, 
                                    DLRA_method=self.coarse_method, 
                                    rank=self.coarse_rank, 
                                    nb_t_steps=self.nb_coarse_substeps, 
                                    monitoring=False,
                                    return_final_value=True)
        return Y1
    
    def fine_solver(self, t_span: tuple, iv: SVD) -> SVD:
        """Low-rank Parareal's fine solver is a truncation to the fine rank and then DLRA of the fine rank

        Args:
            t_span (tuple): time interval
            iv (SVD): initial value in SVD format
        """
        truncated_iv = iv.truncate(r=self.fine_rank, inplace=False)
        Y1 = dlra_solvers.solve_DLRA(problem=self.problem, 
                                    t_span=t_span, 
                                    Y0=truncated_iv, 
                                    DLRA_method=self.fine_method, 
                                    rank=self.fine_rank, 
                                    nb_t_steps=self.nb_fine_substeps, 
                                    monitoring=False,
                                    return_final_value=True)
        return Y1
        
    #%% PARAREAL ROUTINES SPECIFIC TO LOW-RANK PARAREAL
    def initial_approximation(self):
        "Adding random small matrices insures that the rank is large enough"
        t_span = self.problem.t_span
        N = self.nb_parareal_iter
        ts = np.linspace(t_span[0], t_span[1], N + 1, endpoint=True)
        Gn0 = np.empty(N + 1, dtype=object)
        Yn0 = np.empty(N + 1, dtype=object)
        Yn0[0] = self.problem.Y0
        
        sing_vals = np.logspace(-13, -14, self.fine_rank + self.coarse_rank)
        
        for n in range(N):
            Gn0[n + 1] = self.coarse_solver(ts[n: n + 2], Yn0[n])
            small_matrix = svd.generate_low_rank_matrix(self.problem.Y_shape, singular_values=sing_vals)
            Yn0[n + 1] = Gn0[n + 1] + small_matrix
        return Gn0, Yn0
    
    
class AdaptiveLowRankParareal(Parareal):


    # ATTRIBUTES
    name = "Adaptive low-rank Parareal"
    
    def __init__(self, problem: GeneralIVP, 
                 nb_parareal_iter: int, 
                 coarse_rank: int,
                 coarse_dlra_method: str, 
                 fine_tolerance: float,
                 fine_dlra_method: str, 
                 nb_coarse_substeps: int = 1, 
                 nb_fine_substeps: int = 1, 
                 reference_solution: Solution = None) -> None:
        """Initializes an adaptive low-rank Parareal object

        Args:
            problem (GeneralIVP): problem to solve
            nb_parareal_iter (int): number of parareal iterations
            coarse_rank (int): coarse rank
            coarse_dlra_method (str): coarse dlra method
            fine_tolerance (float): fine tolerance
            fine_dlra_method (str): fine dlra method
            nb_coarse_substeps (int, optional): number of coarse substeps. Defaults to 1.
            nb_fine_substeps (int, optional): number of fine substeps. Defaults to 1.
            reference_solution (Solution, optional): reference solution for the error. Defaults to None.
        """
        
        # INIT
        super().__init__(problem, nb_parareal_iter, None, None, nb_coarse_substeps, nb_fine_substeps, reference_solution)
        
        # SPECIFIC VALUES
        self.coarse_rank = coarse_rank
        self.coarse_tol = None
        self.coarse_method = coarse_dlra_method
        
        self.fine_tol = fine_tolerance
        self.fine_method = fine_dlra_method
        
        
    def switch_coarse_to_adaptive(self, coarse_tol: float) -> None:
        """Transform the method into an coarse adaptive formulation.
        Truncate the coarse solves to the given tolerance.

        Args:
            coarse_tol (float, optional): tolerance for the coarse truncation.
        """
        self.coarse_tol = coarse_tol
        self.coarse_label = f"DLRA (tol {coarse_tol})"
        print('The coarse rank is now adaptive')
        
    # PROPERTIES
    @property
    def coarse_label(self):
        if isinstance(self.coarse_tol, float):
            return "DLRA (tol {:.2e})".format(self.coarse_tol)
        else:
            return "DLRA (rank {:d})".format(self.coarse_rank)
        
    @property
    def fine_label(self):
        return "DLRA (tol {:.2e})".format(self.fine_tol)

        
     #%% COARSE AND FINE SOLVERS SPECIFIC TO LOW-RANK PARAREAL
    def coarse_solver(self, t_span: tuple, iv: SVD) -> SVD:
        """Low-rank Parareal's coarse solver is a truncation to the coarse rank and then DLRA of the coarse rank

        Args:
            t_span (tuple): time interval
            iv (SVD): initial value in SVD format
        """
        if isinstance(self.coarse_tol, float):
            truncated_iv = iv.truncate(tol=self.coarse_tol, inplace=False)
            rk = truncated_iv.rank
        else:
            truncated_iv = iv.truncate(r=self.coarse_rank, inplace=False)
            rk = self.coarse_rank
        Y1 = dlra_solvers.solve_DLRA(problem=self.problem, 
                                    t_span=t_span, 
                                    Y0=truncated_iv, 
                                    DLRA_method=self.coarse_method, 
                                    rank=rk, 
                                    nb_t_steps=self.nb_coarse_substeps, 
                                    monitoring=False,
                                    return_final_value=True)
        return Y1
    
    def fine_solver(self, t_span: tuple, iv: SVD) -> SVD:
        """Low-rank Parareal's fine solver is a truncation to the fine rank and then DLRA of the fine rank

        Args:
            t_span (tuple): time interval
            iv (SVD): initial value in SVD format
        """
        truncated_iv = iv.truncate(tol=self.fine_tol, inplace=False)
        rk = truncated_iv.rank
        Y1 = dlra_solvers.solve_DLRA(problem=self.problem, 
                                      t_span=t_span, 
                                      Y0=truncated_iv, 
                                      DLRA_method=self.fine_method, 
                                      rank=rk, 
                                      nb_t_steps=self.nb_fine_substeps, 
                                      monitoring=False,
                                      return_final_value=True)
        return Y1
    
    def initial_approximation(self):
        "Adding random small matrices insures that the rank is large enough"
        # VARIABLES
        t_span = self.problem.t_span
        N = self.nb_parareal_iter
        ts = np.linspace(t_span[0], t_span[1], N + 1, endpoint=True)
        Gn0 = np.empty(N + 1, dtype=object)
        Yn0 = np.empty(N + 1, dtype=object)
        Yn0[0] = self.problem.Y0
        
        # Initial approximation
        sing_vals = np.logspace(np.log10(self.fine_tol)+1, np.log10(self.fine_tol), self.problem.Y0.rank - self.coarse_rank)
        for n in range(N):
            Gn0[n + 1] = self.coarse_solver(ts[n: n + 2], Yn0[n])
            small_matrix = svd.generate_low_rank_matrix(self.problem.Y_shape, singular_values=sing_vals)
            Yn0[n + 1] = Gn0[n + 1] + small_matrix
        return Gn0, Yn0
        
    
    
        

#%% BOUNDS
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

#%% OLD FUNCTIONS, KEEP FOR RETRO-COMPATIBILITY
#%% PARAREAL 
def Parareal(t_span: tuple,
             initial_value: Union[SVD, ArrayLike],
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
            Unk[n + 1, k + 1] = svd.add_svd([Fnk[n + 1, k], Gnk[n + 1, k + 1], - Gnk[n + 1, k]])

    return ts, Unk

#%% LOW-RANK PARAREAL METHODS
def solve_coarse(coarse_solver: callable,
                 problem: GeneralIVP,
                 t_span: tuple,
                 Y0: SVD,
                 N: int,
                 nb_substeps: int = 1) -> solution.Solution:
    " Apply iteratively the coarse solver "
    # VARIABLES
    Ys = np.empty(N+1, dtype=object)
    Ys[0] = Y0
    ts = np.linspace(t_span[0], t_span[1], N + 1)

    # ITERATION
    time_of_computation = np.zeros(N+1)
    for k in tqdm(range(N), desc=f"Coarse solver [{nb_substeps} substep(s)]"):
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
    for k in tqdm(range(N), desc=f"Fine solver [{nb_substeps} substep(s)]"):
        t_sub = ts[k: k + 2]
        t0 = time.time()
        Ys[k + 1] = fine_solver(t_sub, Ys[k])
        time_of_computation[k] = time.time() - t0

    sol = solution.Solution(problem, ts, Ys, time_of_computation)
    return sol

def Low_Rank_Parareal(PB: GeneralIVP,
                      t_span: tuple,
                      Y0: SVD,
                      N: int,
                      coarse_rank: int,
                      fine_rank: int,
                      nb_coarse_substeps: int = 1,
                      nb_fine_substeps: int = 1,
                      DLRA_method: str = 'scipy_dlra') -> Tuple:
    """Low-rank Parareal algorithm. No parallel implementation.

    Args:
        PB (generalIVP): problem to solve
        t_span (tuple): time interval (t0, t1)
        Y0 (SVD): initial value in SVD format
        N (int): number of parareal iterations
        coarse_rank (int): rank of the coarse solver
        fine_rank (int): rank of the fine solver
        nb_coarse_substeps: (int, optional): number of substeps for the coarse solver. Defaults to 1.
        nb_fine_substeps (int, optional): number of substeps for the fine solver. Defaults to 1.
        DLRA_method (str, optional): DLRA method to use; see dlra_methods.py. Defaults to 'KSL2'.
    """

    # DEFINE COARSE SOLVER AS DLRA WITH COARSE RANK
    def coarse_solver(t_span: tuple, Y0: SVD):
        "Coarse solver shortcut for DLRA with checker"
        iv = Y0.truncate(coarse_rank, inplace=False) 
        sol = dlra_solvers.solve_DLRA(PB, t_span, iv, rank=coarse_rank, DLRA_method=DLRA_method, nb_t_steps=nb_coarse_substeps)
        Y1 = sol.Ys[-1]
        return Y1
    
    # COARSE SOLUTION
    coarse_sol = solve_coarse(coarse_solver, PB, t_span, Y0, N, nb_coarse_substeps)

    # DEFINE FINE SOLVER AS DLRA WITH FINE RANK
    def fine_solver(t_span: tuple, Y0: SVD):
        iv = Y0.truncate(fine_rank, inplace=False)
        sol = dlra_solvers.solve_DLRA(PB, t_span, iv, rank=fine_rank, DLRA_method=DLRA_method, nb_t_steps=nb_fine_substeps)
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
        # Ynk[n + 1, 0] = Gnk[n + 1, 0] 
        small_matrix = svd.generate_low_rank_matrix(Y0.shape, singular_values=np.logspace(1e-13, 1e-15, fine_rank + coarse_rank))
        Ynk[n + 1, 0] = Gnk[n + 1, 0] + small_matrix
        # Ynk[n + 1, 0] = svd.generate_low_rank_matrix(Y0.shape, singular_values=np.logspace(-8, -14, fine_rank), is_symmetric=True)
        
        
    ## PARAREAL ITERATION
    for k in tqdm(np.arange(N), desc="Low-rank Parareal"):
        # SAVE COMPUTATIONS THANKS TO FINITE CONVERGENCE
        Ynk[: k + 1, k + 1] = Ynk[: k + 1, k]
        
        for n in tqdm(np.arange(k, N), desc='Computing coarse and fine solvers', leave=False):
            # EVALUATION OF FINE SOLVER -> CAN BE DONE IN PARALLEL
            Fnk[n + 1, k] = fine_solver(ts[n: n + 2], Ynk[n, k])

            # EVALUATION OF COARSE SOLVER
            Gnk[n + 1, k + 1] = coarse_solver(ts[n: n + 2], Ynk[n, k + 1])

            # PARAREAL SCHEME
            Ynk[n + 1, k + 1] = svd.add_svd([Fnk[n + 1, k], Gnk[n + 1, k + 1], - Gnk[n + 1, k]])
    print('Done.')

    return ts, Ynk, coarse_sol, fine_sol

def Low_Rank_Parareal_adaptive(PB: GeneralIVP,
                      t_span: tuple,
                      Y0: SVD,
                      N: int,
                      coarse_rank: int,
                      fine_tol: int,
                      nb_coarse_substeps: int = 1,
                      nb_fine_substeps: int = 1,
                      DLRA_method: str = 'scipy_dlra') -> Tuple:
    """Adaptive version of the Low-rank Parareal algorithm. No parallel implementation.

    Args:
        PB (generalIVP): problem to solve
        t_span (tuple): time interval (t0, t1)
        Y0 (SVD): initial value in SVD format
        N (int): number of parareal iterations
        coarse_rank (int): rank of the coarse solver
        fine_tol (int): tolerance of the fine solver
        nb_coarse_substeps: (int, optional): number of substeps for the coarse solver. Defaults to 1.
        nb_fine_substeps (int, optional): number of substeps for the fine solver. Defaults to 1.
        DLRA_method (str, optional): DLRA method to use; see dlra_methods.py. Defaults to 'KSL2'.
    """

    # DEFINE COARSE SOLVER AS DLRA WITH COARSE RANK
    def coarse_solver(t_span: tuple, Y0: SVD):
        "Coarse solver shortcut for DLRA with checker"
        iv = Y0.truncate(coarse_rank, inplace=False) 
        sol = dlra_solvers.solve_DLRA(PB, t_span, iv, rank=coarse_rank, DLRA_method=DLRA_method, nb_t_steps=nb_coarse_substeps)
        Y1 = sol.Ys[-1]
        return Y1
    
    # COARSE SOLUTION
    coarse_sol = solve_coarse(coarse_solver, PB, t_span, Y0, N, nb_coarse_substeps)

    # DEFINE FINE SOLVER AS DLRA WITH FINE RANK
    def fine_solver(t_span: tuple, Y0: SVD):
        iv = Y0.truncate(tol=fine_tol, inplace=False)
        fine_rank = iv.rank
        sol = dlra_solvers.solve_DLRA(PB, t_span, iv, rank=fine_rank, DLRA_method=DLRA_method, nb_t_steps=nb_fine_substeps)
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
        # Ynk[n + 1, 0] = Gnk[n + 1, 0] 
        small_matrix = svd.generate_low_rank_matrix(Y0.shape, singular_values=np.logspace(-6, -9, Y0.rank))
        Ynk[n + 1, 0] = Gnk[n + 1, 0] + small_matrix
        # Ynk[n + 1, 0] = svd.generate_low_rank_matrix(Y0.shape, singular_values=np.logspace(-8, -14, fine_rank), is_symmetric=True)
        
        
    ## PARAREAL ITERATION
    for k in tqdm(np.arange(N), desc="Adaptive low-rank Parareal"):
        # SAVE COMPUTATIONS THANKS TO FINITE CONVERGENCE
        Ynk[: k + 1, k + 1] = Ynk[: k + 1, k]
        
        for n in tqdm(np.arange(k, N), desc='Computing coarse and fine solvers', leave=False):
            # EVALUATION OF FINE SOLVER -> CAN BE DONE IN PARALLEL
            Fnk[n + 1, k] = fine_solver(ts[n: n + 2], Ynk[n, k])

            # EVALUATION OF COARSE SOLVER
            Gnk[n + 1, k + 1] = coarse_solver(ts[n: n + 2], Ynk[n, k + 1])

            # PARAREAL SCHEME
            Ynk[n + 1, k + 1] = svd.add_svd([Fnk[n + 1, k], Gnk[n + 1, k + 1], - Gnk[n + 1, k]])
    print('Done.')

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
    coarse_error = coarse_sol.compute_relative_errors(reference_sol)
    fine_error = fine_sol.compute_relative_errors(reference_sol)
    
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
    plot1 = ax1.semilogy(ts, enk[:, 0], "b", label="Low-rank Parareal")
    ax1.legend(loc="right")
    ax1.grid()
    ax1.set(
        xlabel="time",
        ylabel="L^2-error",
        title=f"Low-rank Parareal iteration 0/{N}",
    )

    # Figure 2
    it = np.arange(0, N)
    ax2.semilogy(it, err_global_coarse, "r--", label=coarse_name)
    ax2.semilogy(it, err_global_fine, "g--", label=fine_name)
    plot2 = ax2.semilogy(it[0:1], max_enk[0:1], "bo", label="Low-rank Parareal")
    ax2.legend(loc="right")
    ax2.grid()
    ax2.set(
        xlabel="iterations", 
        ylabel="Max error over time", 
        title=f"{reference_sol.problem_name} shape {reference_sol.shape} - Low-rank Parareal"
    )
    plt.close()

    # Function of updating
    def update_plot(frame_number, data, plot1, plot2):
        ax1.lines[2].remove()
        plot1 = ax1.semilogy(ts, data[:, frame_number], "b")
        ax1.set_title(f"Low-rank Parareal iteration {frame_number}/{N-1}")

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
    plt.ylabel("Max error over time")
    plt.title(f"{reference_sol.problem_name} shape {reference_sol.shape} - Low-rank Parareal")
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
                Enk[n, k] = (Unk[n][k] - reference_sol.Ys[n]).norm()/reference_sol.Ys[n].norm()
            else:
                Enk[n, k] = la.norm(Unk[n][k] - reference_sol.Ys[n])/la.norm(reference_sol.Ys[n])
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

    err_coarse = np.max(coarse_sol.compute_relative_errors(exact_sol)) * np.ones(N)
    err_fine = np.max(fine_sol.compute_relative_errors(exact_sol)) * np.ones(N)
    return err_para, err_coarse, err_fine

