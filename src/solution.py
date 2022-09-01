"""
Created on Sun 10 Oct 2021
Written with python 3.9
Author : B. Carrel
"""
from types import NoneType
import numpy as np
import numpy.linalg as la
from ivps.general import GeneralIVP
from typing import Union
from numpy.typing import ArrayLike
from low_rank_toolbox.low_rank_matrix import LowRankMatrix
from low_rank_toolbox import svd
from low_rank_toolbox.svd import SVD
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from IPython.display import HTML
import plotting


class Solution:
    """
    Store the solution of a given problem. To be used in the class generalProblem
    """

    # %% BASIC FUNCTIONS
    def __init__(self,
                 problem: GeneralIVP,
                 ts: list,
                 Ys: Union[list, ArrayLike],
                 time_of_computation: Union[list, ArrayLike] = None):
        # STORE DATA
        self.ts = ts
        self.Ys = Ys
        self.problem = problem
        self.time_of_computation = time_of_computation
        self.problem_name = problem.name

    def __repr__(self):
        return (
            f"Problem: {self.problem_name} of shape {self.shape}\n"
            f"Number of time steps: {self.nb_t_steps} \n"
            f"Stepsize: h={self.stepsize} \n"
            f"Total time of computation: {round(np.sum(self.time_of_computation), 2)} seconds"
        )

    # %% PROPERTIES
    @property
    def shape(self) -> tuple:
        return self.Ys[0].shape

    @property
    def nb_t_steps(self) -> int:
        return len(self.ts)

    @property
    def stepsize(self) -> float:
        return self.ts[1] - self.ts[0]

    def copy(self):
        return Solution(self.problem, self.ts, self.Ys, self.time_of_computation)

    def convert_to_dense(self):
        old_sol = self.Ys
        for k, Y in enumerate(old_sol):
            if isinstance(Y, LowRankMatrix):
                self.Ys[k] = Y.todense()

    def convert_to_SVD(self):
        old_sol = self.Ys
        for k, Y in enumerate(old_sol):
            if not isinstance(Y, SVD):
                self.Ys[k] = svd.truncated_svd(Y)
                
    def plot(self, 
             time_index: str,
             title: str = None,
             do_save: bool = False,
             filename: str = None):
        "Plot the solution corresponding to the time index"
        # VARIABLES
        time = self.ts[time_index]
        solution = self.Ys[time_index]
        if isinstance(title, NoneType):
            title = f'{self.problem_name} - Solution at time t={round(self.ts[time_index], 2)}'
        return self.problem.plot(solution, title, do_save, filename)
    
    def plot_singular_values(self,
                             time_index: str,
                             title: str = None,
                             do_save: bool = False,
                             filename: str = None):
        "Plot the singular values of the solution corresponding to the time index"
        # VARIABLES
        solution = self.Ys[time_index]
        if isinstance(solution, SVD):
            sing_vals = solution.sing_vals
        else:
            sing_vals = la.svd(solution, compute_uv=False)
        index = np.arange(1, len(sing_vals) + 1)
        if isinstance(title, NoneType):
            title = f'{self.problem_name} - Singular values at time t={round(self.ts[time_index], 2)}'
            
        # PLOT
        fig = plt.figure(clear=True)
        plt.semilogy(index, sing_vals, 'o')
        plt.semilogy(index, sing_vals[0] * np.finfo(float).eps * np.ones(len(sing_vals)), 'k--')
        plt.title(title, fontsize=16)
        plt.xlim((index[0], index[-1]))
        plt.xlabel('index', fontsize=16)
        plt.ylim((1e-18, 1e2))
        plt.ylabel('value', fontsize=16)
        plt.tight_layout()
        plt.grid()
        plt.show()
        if do_save:
            fig.savefig(filename)
        return fig
    
    def plot_solution(self,
                      time_index: str,
                      title: str = None,
                      do_save: bool = False,
                      filename: str = None):
        "Plot the solution with the representation of the problem at the given time index"
        # VARIABLES
        solution = self.Ys[time_index]
        if isinstance(title, NoneType):
            title = f'{self.problem_name} - Solution at time t={round(self.ts[time_index], 2)}'
        
        # PLOT
        fig = self.problem.plot(solution, title, do_save, filename)
        return fig
        
                
    def animation_singular_values(self,
                                title: str = None,
                                do_save: bool = False) -> animation.FuncAnimation:
        "Return an animation of the singular values"
        return plotting.singular_values(self.ts, self.Ys, title, do_save)
    
    def animation_2D(self,
                     title: str = None,
                     do_save: bool = False):
        "Return an animation in 2D of the solution"
        return plotting.animation_2D(self.ts, self.Ys, title, do_save)
    
    def animation_sing_vals_and_sol(self,
                                    title: str = None,
                                    do_save: bool = False):
        "Return an animation of the singular values together with the problem's representation"
        
    

    # %% ERRORS
    def compute_errors(self, other) -> ArrayLike:
        "Compute error at each time step with an other solution."
        # SOME VERIFICATIONS
        checkers = (self.shape == other.shape,
                    self.nb_t_steps == other.nb_t_steps)
        if not all(checkers):
            raise ValueError('The two solutions do not corresponds')
        # COMPUTE ERRORS ITERATIVELY
        errors = np.zeros(self.nb_t_steps)
        for k in np.arange(self.nb_t_steps):
            if isinstance(other.Ys[k], LowRankMatrix):
                diff = other.Ys[k] - self.Ys[k]
            else:
                diff = self.Ys[k] - other.Ys[k]
            if isinstance(diff, LowRankMatrix):
                errors[k] = diff.norm()
            else:
                errors[k] = la.norm(diff)
        return errors

    def compute_relative_errors(self, ref_sol) -> ArrayLike:
        "Compute relative error at each time step with an other solution."
        # SOME VERIFICATIONS
        checkers = (self.shape == ref_sol.shape,
                    self.nb_t_steps == ref_sol.nb_t_steps)
        if not all(checkers):
            raise ValueError('The two solutions do not corresponds')
        # COMPUTE ERRORS ITERATIVELY
        errors = np.zeros(self.nb_t_steps)
        for k in np.arange(self.nb_t_steps):
            if isinstance(ref_sol.Ys[k], LowRankMatrix):
                normYs = ref_sol.Ys[k].norm()
                diff = ref_sol.Ys[k] - self.Ys[k]
            else:
                normYs = la.norm(ref_sol.Ys[k])
                diff = self.Ys[k] - ref_sol.Ys[k]
            if isinstance(diff, LowRankMatrix):
                errors[k] = diff.norm() / normYs
            else:
                errors[k] = la.norm(diff) / normYs
        return errors
    
    
    # %% SINGULAR VALUE

        