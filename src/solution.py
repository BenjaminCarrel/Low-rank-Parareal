"""
Created on Sun 10 Oct 2021
Written with python 3.9
Author : B. Carrel
"""
import numpy as np
import numpy.linalg as la
from IVPs.General import GeneralIVP
from typing import Union
from numpy.typing import ArrayLike
from low_rank_toolbox.low_rank_matrix import LowRankMatrix



class Solution:
    """
    Store the solution of a given problem. To be used in the class GeneralProblem
    """

    #%% BASIC FUNCTIONS
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

    def __repr__(self):
        return (
            f"Problem: {self.problem} \n"
            f"Initial shape: {self.initial_shape} \n"
            f"Final shape: {self.final_shape} \n"
            f"Number of time steps: {self.nb_t_steps} \n"
            f"Stepsize: {self.stepsize} \n"
            f"Total time of computation: {np.sum(self.time_of_computation)}"
        )

    #%% PROPERTIES
    @property
    def initial_shape(self) -> tuple:
        return self.Ys[0].shape

    @property
    def final_shape(self) -> tuple:
        return self.Ys[-1].shape

    @property
    def shape(self) -> tuple:
        return self.initial_shape

    @property
    def nb_t_steps(self) -> int:
        return len(self.ts)

    @property
    def stepsize(self) -> float:
        return self.ts[1] - self.ts[0]

    #%% ERRORS
    def compute_errors(self, other) -> ArrayLike:
        "Compute error at each ts step with an other solution."
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

