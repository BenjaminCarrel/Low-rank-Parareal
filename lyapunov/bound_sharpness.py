#%% IMPORTATIONS
import matplotlib.pyplot as plt
from LRP import parareal
from IVPs import Lyapunov
import numpy as np
import numpy.linalg as la
import classical_solvers

#%% CREATE PROBLEM
t_span = (0.01, 2.01)
size = 100
PB = Lyapunov.make_heat_problem(t_span, size)

#%% REFERENCE SOLUTION
nb_t_steps = 10
reference_sol = classical_solvers.solve(PB, t_span, PB.Y0.full(), method='scipy', nb_t_steps=nb_t_steps, monitoring=True)


#%% LOW RANK PARAREAL
ts, Unk, coarse_sol, fine_sol = parareal.Low_Rank_Parareal(PB, t_span, PB.Y0, N=nb_t_steps, nb_coarse_substeps=40, nb_fine_substeps=40, coarse_rank=4, fine_rank=16, DLRA_method='KSL2')

#%% COMPUTE ERRORS AND BOUNDS
Enk = parareal.compute_parareal_error(reference_sol, Unk)
err_para, err_coarse, err_fine = parareal.compute_max_error_over_time(reference_sol, coarse_sol, fine_sol, Unk)

l = 2*np.max(la.eigvals(PB.A.todense()))
h = (t_span[1] - t_span[0]) / nb_t_steps
alpha = np.exp(l*h).real
beta = alpha
gamma = np.max(Enk[:, 0])
kappa = err_fine[0] * (1 - alpha - beta)
linear_bd = parareal.linear_bound(alpha, beta, gamma, kappa, nb_t_steps)
superlinear_bd = parareal.superlinear_bound(alpha, beta, gamma, kappa, nb_t_steps)

#%% PLOT ERROR AND BOUNDS
fig = plt.figure(clear=True)
it = np.arange(0, nb_t_steps+1)
plt.semilogy(it, err_para, 'kx', label='low-rank Parareal')
plt.semilogy(it, err_coarse, 'k-')
plt.semilogy(it, err_fine, 'k-')
plt.semilogy(it, linear_bd, '--', label='linear bound')
plt.semilogy(it, superlinear_bd, '--', label='superlinear bound')
plt.grid()
plt.xlim((0, nb_t_steps+1))
plt.title('Theoretical bound for low-rank parareal')
plt.xlabel('iteration')
plt.ylabel('max error over time')
plt.tight_layout()
plt.legend()
plt.show()
fig.savefig(f'figures/parareal_and_bounds.pdf')
# %%
