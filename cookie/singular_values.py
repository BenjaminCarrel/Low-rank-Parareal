#%% IMPORTATIONS
import matplotlib.pyplot as plt
from IVPs import Cookie
import classical_solvers
import numpy as np
import scipy.linalg as la
import plotting
from IPython.display import HTML

#%% CREATE PROBLEM
t_span = (0.01, 0.11)
PB = Cookie.make_cookie_problem(t_span)

#%% EXACT SOLUTION
nb_t_steps = 20
full_iv = PB.Y0.full()
reference_sol = classical_solvers.solve(PB, t_span, full_iv, nb_t_steps=nb_t_steps, monitoring=True)

#%% SINGULAR VALUES ANIMATION
anim_sing_vals = plotting.singular_values(reference_sol.ts, reference_sol.Ys, title='figures/animation_cookie_sing_vals.gif', do_save=True)
HTML(anim_sing_vals.to_jshtml())

#%% SINGULAR VALUES PLOTS

i_times = (0, 10, 20)
index = np.arange(1, 102)

for i in i_times:
    t = reference_sol.ts[i]
    sing_vals = la.svd(reference_sol.Ys[i], compute_uv=False)

    fig = plt.figure(clear=True)
    plt.semilogy(index, sing_vals, 'o')
    plt.semilogy(index, sing_vals[0] * 1e-12 * np.ones(len(sing_vals)), 'k--')
    plt.title(f'Singular values at time t={round(t-0.01,2)}', fontsize=16)
    plt.xlim((1,50))
    plt.xlabel('index', fontsize=16)
    plt.ylabel('value', fontsize=16)
    plt.tight_layout()
    plt.grid()
    plt.show()
    fig.savefig(f'figures/cookie_sing_vals_t_{i}.pdf')

# %%

# %%
