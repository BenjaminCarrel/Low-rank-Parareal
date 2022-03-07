#%% IMPORTATIONS
import matplotlib.pyplot as plt
from IPython.display import HTML
from IVPs import Lyapunov
import classical_solvers
import numpy as np
import scipy.linalg as la
import plotting

#%% CREATE PROBLEM
t_span = (0.01, 2.01)
n = 100
PB = Lyapunov.make_heat_problem(t_span, n)

#%% EXACT SOLUTION
nb_t_steps = 20
full_iv = PB.Y0.full()
reference_solution = classical_solvers.solve(PB, t_span, full_iv, nb_t_steps=nb_t_steps, monitoring=True, method='scipy')

#%% SINGULAR VALUES ANIMATION
anim_sing_vals = plotting.singular_values(reference_solution.ts, reference_solution.Ys, title='figures/animation_lyapunov_sing_vals.gif', do_save=True)
HTML(anim_sing_vals.to_jshtml())

#%% SINGULAR VALUES PLOTS
i_times = (0, 10, 20)
index = np.arange(1, n+1)

for i in i_times:
    t = reference_solution.ts[i]
    sing_vals = la.svd(reference_solution.Ys[i], compute_uv=False)

    fig = plt.figure(clear=True)
    plt.semilogy(index, sing_vals, 'o')
    plt.semilogy(index, sing_vals[0] * np.finfo(float).eps * np.ones(len(sing_vals)), 'k--')
    plt.title(f'Singular values at time t={round(t-0.01, 2)}', fontsize=16)
    plt.xlim((1,50))
    plt.xlabel('index', fontsize=16)
    plt.ylim((1e-21, 1e0))
    plt.ylabel('value', fontsize=16)
    plt.tight_layout()
    plt.grid()
    plt.show()
    fig.savefig(f'figures/lyapunov_sing_vals_t_{i}.pdf')

for i in i_times:
    t = reference_solution.ts[i]

    x = np.linspace(-1,1,n)
    X, Y = np.meshgrid(x, x)
    fig = plotting.plot_3D(X, Y, reference_solution.Ys[i], title=f'Heat distribution at time t={round(t-0.01, 2)}')
    fig.savefig(f'figures/lyapunov_3D_solution_t_{i}.pdf')

# %%

# %%
