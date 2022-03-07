#%% IMPORTATIONS
import matplotlib.pyplot as plt
from LRP import parareal
from IVPs import Cookie
import classical_solvers
import numpy as np


#%% CREATE PROBLEM
t_span = (0.01, 0.11)
PB = Cookie.make_cookie_problem(t_span)

#%% REFERENCE SOLUTION
nb_t_steps = 50
iv = PB.Y0.copy()
reference_sol = classical_solvers.solve(PB, t_span, iv, nb_t_steps=nb_t_steps, monitoring=True)

#%% SEVERAL COARSE RANKS
all_coarse_ranks = (2, 4, 6)
linestyles = ('-o', '-x', '-+')
it = np.arange(0, nb_t_steps+1)
fig = plt.figure(clear=True)
for k in range(len(all_coarse_ranks)):
    # PARAREAL
    ts, Unk, _, _ = parareal.Low_Rank_Parareal(PB, t_span, iv, nb_t_steps, nb_coarse_substeps=2, nb_fine_substeps=2, coarse_rank=all_coarse_ranks[k], fine_rank=16)
    # ERROR
    Enk = parareal.compute_parareal_error(reference_sol, Unk)
    # PLOT
    err_para = np.max(Enk, axis=0)
    plt.semilogy(it, err_para, linestyles[k], label=f'q={all_coarse_ranks[k]}')

plt.grid()
plt.xlabel('iteration')
plt.ylabel('max error over time')
plt.tight_layout()
plt.legend()
plt.show()
fig.savefig(f'figures/cookie_several_coarse_ranks.pdf')

#%% SEVERAL FINE RANKS
all_fine_ranks = (14, 16, 18)
linestyles = ('-o', '-x', '-+')
it = np.arange(0, nb_t_steps+1)
fig = plt.figure(clear=True)
for k in range(len(all_fine_ranks)):
    # PARAREAL
    ts, Unk, _, _ = parareal.Low_Rank_Parareal(PB, t_span, iv, nb_t_steps, nb_coarse_substeps=2, nb_fine_substeps=2, coarse_rank=4, fine_rank=all_fine_ranks[k])
    # ERROR
    Enk = parareal.compute_parareal_error(reference_sol, Unk)
    # PLOT
    err_para = np.max(Enk, axis=0)
    plt.semilogy(it, err_para, linestyles[k], label=f'r={all_fine_ranks[k]}')

plt.grid()
plt.xlabel('iteration')
plt.ylabel('max error over time')
plt.tight_layout()
plt.legend()
plt.show()
fig.savefig(f'figures/cookie_several_fine_ranks.pdf')
# %%
