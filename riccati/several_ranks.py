#%% IMPORTATIONS
import matplotlib.pyplot as plt
from LRP import parareal
from IVPs import Riccati
import classical_solvers
import numpy as np

#%% CREATE PROBLEM
t_span = (0.01, 0.11)
m = 200 # DEFAULT IS 200
q = 9 #DEFAULT IS 9
PB = Riccati.make_riccati_ostermann(t_span, m, q, initial_rank=m)

#%% REFERENCE SOLUTION
nb_t_steps = 20
full_iv = PB.Y0.full()
exact_sol = classical_solvers.solve(PB, t_span, full_iv, nb_t_steps=nb_t_steps, monitoring=True, method='scipy')

#%% SEVERAL COARSE RANKS
all_coarse_ranks = (4, 6, 8)
linestyles = ('-o', '-x', '-+')
it = np.arange(0, nb_t_steps+1)
fig = plt.figure(clear=True)
for k in range(len(all_coarse_ranks)):
    # PARAREAL
    fine_r = 18
    ts, Unk, _, _ = parareal.Low_Rank_Parareal(PB, t_span, PB.Y0, nb_t_steps, nb_coarse_substeps=200, nb_fine_substeps=200, coarse_rank=all_coarse_ranks[k], fine_rank=fine_r, DLRA_method='KSL2')
    # ERROR
    Enk = parareal.compute_parareal_error(exact_sol, Unk)
    # PLOT
    err_para = np.max(Enk, axis=0)
    plt.semilogy(it, err_para, linestyles[k], label=f'q={all_coarse_ranks[k]}')

plt.grid()
plt.xlabel('iteration')
plt.ylabel('max error over time')
plt.tight_layout()
plt.legend()
plt.show()
fig.savefig(f'figures/riccati_several_coarse_ranks.pdf')

#%% SEVERAL FINE RANKS
all_fine_ranks = (14, 18, 22)
linestyles = ('-o', '-x', '-+')
it = np.arange(0, nb_t_steps+1)
fig = plt.figure(clear=True)
for k in range(len(all_fine_ranks)):
    # PARAREAL
    ts, Unk, _, _ = parareal.Low_Rank_Parareal(PB, t_span, PB.Y0, nb_t_steps, nb_coarse_substeps=200, nb_fine_substeps=200, coarse_rank=6, fine_rank=all_fine_ranks[k], DLRA_method='KSL2')
    # ERROR
    Enk = parareal.compute_parareal_error(exact_sol, Unk)
    # PLOT
    err_para = np.max(Enk, axis=0)
    plt.semilogy(it, err_para, linestyles[k], label=f'r={all_fine_ranks[k]}')

plt.grid()
plt.xlabel('iteration')
plt.ylabel('max error over time')
plt.tight_layout()
plt.legend()
plt.show()
fig.savefig(f'figures/riccati_several_fine_ranks.pdf')
# %%

# %%
