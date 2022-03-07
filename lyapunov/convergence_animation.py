#%% IMPORTATIONS
from LRP import parareal
from IVPs import Lyapunov
import classical_solvers
import DLRA_solvers
from matplotlib import pyplot as plt
from IPython.display import HTML

#%% CREATE PROBLEM
t_span = (0.01, 2.01)
size = 100
PB = Lyapunov.make_heat_problem(t_span, size)
iv = PB.Y0.copy()


#%% REFERENCE SOLUTION
nb_t_steps = 50 * int(10*(size/50)**2)
reference_sol = classical_solvers.solve(PB, t_span, iv, method='optimal', nb_t_steps=nb_t_steps, monitoring=True)

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
nb_sub = int(10*(size/50)**2) # hard coded, scales quadratically with the size since stiff
reference_sol = classical_solvers.solve(PB, t_span, iv, method='optimal', nb_t_steps=N, monitoring=True)
ts, Unk, coarse_sol, fine_sol = parareal.Low_Rank_Parareal(PB, t_span, iv, N, coarse_rank=coarse_q, fine_rank=fine_r, nb_coarse_substeps=nb_sub, nb_fine_substeps=nb_sub, DLRA_method='KSL2')

#%% ANIMATION
anim, err_coarse, err_fine, err_para = parareal.parareal_animation(Unk, coarse_sol, fine_sol,reference_sol, title=f'figures/animation_parareal_lyapunov_size_{size}', coarse_name=f'coarse q={coarse_q}', fine_name=f'fine r={fine_r}', do_save=True)
HTML(anim.to_jshtml())

# %%
