# %% IMPORTATIONS
import matplotlib.pyplot as plt
from LRP import parareal
from IVPs import Lyapunov
import classical_solvers
import numpy as np


# %% SEVERAL SIZES
size = 100
all_t_span = ((0.01, 1.01), (0.01, 2.01), (0.01, 4.01))
nb_t_steps = 50
it = np.arange(0, nb_t_steps+1)
fig = plt.figure(clear=True)
linestyles = ('o-', 'x-', '+-')
for k in range(len(all_t_span)):
    # MAKE PROBLEM
    print(f'Time: {all_t_span[k]}')
    PB = Lyapunov.make_heat_problem(all_t_span[k], size)

    # COMPUTE EXACT SOLUTION
    full_iv = PB.Y0.full()
    reference_solution = classical_solvers.solve(
        PB, all_t_span[k], full_iv, nb_t_steps=nb_t_steps, monitoring=True, method='scipy')

    # COMPUTE LOW-RANK PARAREAL
    ts, Unk, _, _ = parareal.Low_Rank_Parareal(
        PB, all_t_span[k], PB.Y0, nb_t_steps, coarse_rank=4, fine_rank=16, nb_coarse_substeps=80, nb_fine_substeps=80, DLRA_method='KSL2')
    Enk = parareal.compute_parareal_error(reference_solution, Unk)
    max_Enk = np.max(Enk, axis=0)

    # PLOT
    stepsize = all_t_span[k][-1] / nb_t_steps
    plt.semilogy(
        it, max_Enk, linestyles[k], label=f'$h = {round(stepsize, 2)}$')

plt.grid()
plt.xlabel('iteration')
plt.ylabel('max error over time')
plt.tight_layout()
plt.xlim((0, nb_t_steps+1))
plt.legend()
plt.show()
fig.savefig('figures/lyapunov_several_stepsizes.pdf')
# %%

# %%

# %%
