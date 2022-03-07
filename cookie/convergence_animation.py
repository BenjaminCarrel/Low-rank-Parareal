#%% IMPORTATIONS
from LRP import parareal
from IVPs import Cookie
import classical_solvers
import DLRA_solvers
from matplotlib import pyplot as plt
from IPython.display import HTML

#%% CREATE PROBLEM
t_span = (0.01, 0.11)
PB = Cookie.make_cookie_problem(t_span)

#%% REFERENCE SOLUTION
nb_t_steps = 100
iv = PB.Y0
reference_sol = classical_solvers.solve(PB, t_span, iv, nb_t_steps=nb_t_steps, monitoring=True, method='scipy')

#%% COARSE SOLUTION
coarse_q = 4
coarse_iv = iv.truncate(coarse_q, inplace=False)
dlra_meth = 'KSL2'
coarse_sol = DLRA_solvers.solve_DLRA(PB, t_span, coarse_iv, DLRA_method=dlra_meth, monitoring=True,  nb_t_steps=nb_t_steps)

#%% FINE SOLUTION
fine_r = 16
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

#%% LOW RANK PARAREAL
N = 50
reference_sol = classical_solvers.solve(PB, t_span, iv, method='optimal', nb_t_steps=N, monitoring=True)
ts, Unk, coarse_sol, fine_sol = parareal.Low_Rank_Parareal(PB, t_span, PB.Y0, N, coarse_rank=coarse_q, fine_rank=fine_r, nb_coarse_substeps=2, nb_fine_substeps=2)

#%% ANIMATION
anim, err_coarse, err_fine, err_para = parareal.parareal_animation(Unk, coarse_sol, fine_sol,
reference_sol, title='figures/animation_cookie_parareal',coarse_name='coarse', fine_name='fine', do_save=True)
HTML(anim.to_jshtml())

# %%
