#%% IMPORTATIONS
import numpy as np
from numpy import linalg as la
from numpy.typing import ArrayLike
from scipy.sparse import spmatrix
from IVPs import Lyapunov
import classical_solvers
from matplotlib import pyplot as plt
from low_rank_toolbox.svd import SVD

#%% DEF BOUND
def theoretical_bound(t: float, 
                      A: spmatrix, 
                      C: ArrayLike, 
                      X0: SVD, 
                      rho: int, 
                      r: int, 
                      r0: int) -> float:
    """Theoretical bound for the best rank rho approximation of the Lyapunov solution

    Args:
        t (float): time
        A (spmatrix): matrix A of the problem
        C (ArrayLike): matrix C of the problem
        X0 (ArrayLike): initial value
        rho (int): rank of approximation
        r (int): rank approximation of the source term
        r0 (int): rank approximation of the initial value
    """
    # extract data
    ell = 2 * np.max(la.eigvals(A.todense()))
    kappa_A = la.cond(A.todense(), p=2)
    CCt = C.dot(C.T)
    s_C = la.svd(CCt, full_matrices=True, compute_uv=False)
    norm_CCt = la.norm(CCt, ord=2)
    s_X0 = la.svd(X0.todense(), full_matrices=True, compute_uv=False)
    
    
    # first term
    sigma0 = s_X0[r0]
    term1 = np.exp(ell*t)*sigma0
    
    # second term
    sigmaC = s_C[r]
    term2 = (np.exp(t*ell)-1)/ell * (4 * np.exp(-np.pi**2 * rho/np.log(4*kappa_A))*norm_CCt + sigmaC)
    
    return term1 + term2

def theoretical_exact_low_rank(t, A, C, X0, rho):
    CCt = C.dot(C.T)
    norm_CCt = la.norm(CCt, ord=2)
    ell = 2 * np.max(la.eigvals(A.todense()))
    kappa_A = la.cond(A.todense(), p=2)
    bd = (np.exp(t*ell)-1)/ell * 4 * np.exp(-np.pi**2 * rho/np.log(4*kappa_A)) * norm_CCt
    return bd

#%% MAKE HEAT LYAPUNOV PROBLEM
T = 1000
t_span = (0,T)
size = 100
PB = Lyapunov.make_heat_problem(t_span, n=size, initial_rank=41, source_rank=5)

#%% SOLVE IT
nb_steps = 100
reference_sol = classical_solvers.solve(PB, t_span, PB.Y0, nb_t_steps=nb_steps, monitoring=True)

#%% TRUE SINGULAR VALUES
true_sing_vals = reference_sol.Ys[-1].sing_vals

#%% BOUND ON SINGULAR VALUES AT FIXED TIME
index = np.arange(1,size+1)
bound = np.zeros(size)
for k in range(size):
    bound_values = []
    for r0 in range(k+1):
        for r in range(k+1-r0):
            for rho in range(k+1-r0):
                if r0+2*r*rho == k:
                    new_bd = theoretical_bound(T, PB.A, PB.C, PB.Y0, rho, r, r0)
                    bound_values.append(new_bd)
    bound[k] = np.min(bound_values)
    
#%% PLOT IT
plt.figure()
plt.semilogy(index, true_sing_vals, 'o', label='sing vals')
plt.semilogy(index, bound, '--', label='bound')
plt.grid()
plt.legend()
plt.show()


#%% COMPUTE BEST RANK APPROXIMATION
r0 = 32
r = 1
rho = 20
sol = np.empty(nb_steps, dtype=object)
best_rank_approximation = np.empty(nb_steps, dtype=object)
for j in range(nb_steps):
    best_rank_approximation[j] = reference_sol.Ys[j].truncate(r0 + 2*r*rho).todense()
    sol[j] = reference_sol.Ys[j].todense()
    
#%% COMPUTE TRUE ERROR AND BOUND
times = np.linspace(t_span[0], t_span[1], nb_steps)
true_error = np.zeros(nb_steps)
bound1 = np.zeros(nb_steps)
bound2 = np.zeros(nb_steps)
for j in range(nb_steps):
    true_error[j] = la.norm(sol[j] - best_rank_approximation[j], 2)
    bound1[j] = theoretical_bound(times[j], PB.A, PB.C, PB.Y0, rho, r, r0)
    bound2[j] = theoretical_exact_low_rank(times[j], PB.A, PB.C, PB.Y0, rho)
    
#%% PLOT
fig = plt.figure()
plt.semilogy(times, true_error, label=r'$|X(t)-X_k(t)|_2$')
plt.semilogy(times, bound1, '--', label='bound')
# plt.semilogy(times, bound2, '--', label='bound exact low-rank')
plt.legend()
plt.grid()
plt.show()


    
    




# %%
