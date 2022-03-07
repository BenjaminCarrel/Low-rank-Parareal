# %% IMPORTATIONS
import matplotlib.pyplot as plt
from LRP import parareal
from IVPs import Riccati
import classical_solvers
import DLRA_solvers
from IPython.display import HTML

# %% CREATE PROBLEM
t_span = (0.01, 0.11)
m = 200  # 200
q = 9
PB = Riccati.make_riccati_ostermann(t_span, m, q, initial_rank=m)


# %% REFERENCE SOLUTION
nb_t_steps = 4000
iv = PB.Y0
full_iv = iv.full()
reference_sol = classical_solvers.solve(PB, t_span, full_iv, nb_t_steps=nb_t_steps, monitoring=True, method='scipy')

#%% COARSE SOLUTION
coarse_q = 6
coarse_iv = iv.truncate(coarse_q, inplace=False)
dlra_meth = 'KSL2'
coarse_sol = DLRA_solvers.solve_DLRA(PB, t_span, coarse_iv, DLRA_method=dlra_meth, monitoring=True,  nb_t_steps=nb_t_steps)

#%% FINE SOLUTION
fine_r = 18
fine_iv = iv.truncate(fine_r, inplace=False)
dlra_meth = 'KSL2'
fine_sol = DLRA_solvers.solve_DLRA(PB, t_span, fine_iv, DLRA_method=dlra_meth, monitoring=True,  nb_t_steps=nb_t_steps)

#%% COMPARE SOLUTIONS
coarse_err = coarse_sol.compute_errors(reference_sol)
fine_err = fine_sol.compute_errors(reference_sol)
fig = plt.figure(2, dpi=150)
plt.semilogy(coarse_sol.ts, coarse_err, label=f'Coarse rank {coarse_q}')
plt.semilogy(fine_sol.ts, fine_err, label=f'Fine rank {fine_r}')
plt.grid()
plt.legend()
plt.xlabel('time')
plt.ylabel('error')
plt.show()
# %% LOW RANK PARAREAL
N = 20
reference_sol = classical_solvers.solve(PB, t_span, iv, method='optimal', nb_t_steps=N, monitoring=True)
ts, Unk, coarse_sol, fine_sol = parareal.Low_Rank_Parareal(
    PB, t_span, PB.Y0, N, coarse_rank=coarse_q, fine_rank=fine_r, nb_coarse_substeps=200, nb_fine_substeps=200, DLRA_method='KSL2')

# %% ANIMATION
anim, err_coarse, err_fine, err_para = parareal.parareal_animation(Unk, coarse_sol, fine_sol,
                                                                   reference_sol, title='figures/animation_riccati_parareal',
                                                                   coarse_name='coarse', fine_name='fine', do_save=True)
HTML(anim.to_jshtml())

# %% COARSE AND FINE SOLVERS
coarse_err = coarse_sol.compute_errors(reference_sol)
fine_err = fine_sol.compute_errors(reference_sol)
fig = plt.figure(2, dpi=150)
plt.semilogy(reference_sol.ts, coarse_err, label=f'Coarse rank {coarse_q}')
plt.semilogy(reference_sol.ts, fine_err, label=f'Fine rank {fine_r}')
plt.grid()
plt.legend()
plt.xlabel('time')
plt.ylabel('error')
plt.show()
# %%
