#%% IMPORTATIONS
import matplotlib.pyplot as plt
from LRP import parareal
from IVPs import Lyapunov
import classical_solvers
import numpy as np

#%% SEVERAL SIZES
all_size = (50, 100, 200)
t_span = (0.01, 2.01)
nb_t_steps = 50
it = np.arange(0, nb_t_steps+1)
fig = plt.figure(clear=True)
linestyles = ('o-','x-','+-')
for k in range(len(all_size)):
    # MAKE PROBLEM
    print(f'Size: {all_size[k]}')
    PB = Lyapunov.make_heat_problem(t_span, all_size[k])

    # COMPUTE EXACT SOLUTION
    full_iv = PB.Y0.full()
    reference_solution = classical_solvers.solve(PB, t_span, full_iv, nb_t_steps=nb_t_steps, monitoring=True, method='scipy')

    # COMPUTE LOW-RANK PARAREAL
    n_sub = int(10*(all_size[k]/50)**2) # hard coded,  scales quadratically with the size of the problem
    ts, Unk, _, _ = parareal.Low_Rank_Parareal(PB, t_span, PB.Y0, nb_t_steps, coarse_rank=4, fine_rank=16,
                                         nb_coarse_substeps=n_sub, nb_fine_substeps=n_sub, DLRA_method='KSL2')
    Enk = parareal.compute_parareal_error(reference_solution, Unk)
    max_Enk = np.max(Enk, axis=0)

    # PLOT
    plt.semilogy(it, max_Enk, linestyles[k], label=f'n={all_size[k]}')

plt.grid()
plt.xlabel('iteration')
plt.ylabel('max error over time')
plt.tight_layout()
plt.xlim((0,nb_t_steps+1))
plt.legend()
plt.show()
fig.savefig('figures/lyapunov_several_sizes.pdf')
# %%

# %%

# %%
