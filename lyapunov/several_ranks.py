#%% IMPORTATIONS
import matplotlib.pyplot as plt
from LRP import parareal
from IVPs import Lyapunov
import classical_solvers

import numpy as np

#%% CREATE PROBLEM
t_span = (0.01, 2.01)
n = 100
PB = Lyapunov.make_heat_problem(t_span, n)

#%% EXACT SOLUTION
nb_t_steps = 50
full_iv = PB.Y0.full()
reference_solution = classical_solvers.solve(PB, t_span, full_iv, nb_t_steps=nb_t_steps, monitoring=True, method='scipy')


#%% SEVERAL COARSE RANKS
all_coarse_ranks = (2, 4, 6)
linestyles = ('-o', '-x', '-+')
it = np.arange(0, nb_t_steps+1)
fig = plt.figure(clear=True)
for k in range(len(all_coarse_ranks)):
    # PARAREAL
    ts, Unk, coarse_sol, fine_sol = parareal.Low_Rank_Parareal(PB, t_span, PB.Y0, nb_t_steps, coarse_rank=all_coarse_ranks[k], fine_rank=16, nb_coarse_substeps=40, nb_fine_substeps=40)
    # ERROR
    Enk = parareal.compute_parareal_error(reference_solution, Unk)
    # PLOT
    err_para = np.max(Enk, axis=0)
    plt.semilogy(it, err_para, linestyles[k], label=f'q={all_coarse_ranks[k]}')

plt.grid()
plt.xlabel('iteration')
plt.ylabel('max error over time')
plt.tight_layout()
plt.legend()
plt.show()
fig.savefig(f'figures/lyapunov_several_coarse_ranks.pdf')

#%% SEVERAL FINE RANKS
all_fine_ranks = (12, 16, 20)
linestyles = ('-o', '-x', '-+')
it = np.arange(0, nb_t_steps+1)
fig = plt.figure(clear=True)
for k in range(len(all_fine_ranks)):
    # PARAREAL
    ts, Unk, coarse_sol, fine_sol = parareal.Low_Rank_Parareal(PB, t_span, PB.Y0, nb_t_steps, coarse_rank=4, fine_rank=all_fine_ranks[k], nb_coarse_substeps=40, nb_fine_substeps=40)
    # ERROR
    Enk = parareal.compute_parareal_error(reference_solution, Unk)
    # PLOT
    err_para = np.max(Enk, axis=0)
    plt.semilogy(it, err_para, linestyles[k], label=f'r={all_fine_ranks[k]}')

plt.grid()
plt.xlabel('iteration')
plt.ylabel('max error over time')
plt.tight_layout()
plt.legend()
plt.show()
fig.savefig(f'figures/lyapunov_several_fine_ranks.pdf')
# %%

# %%
