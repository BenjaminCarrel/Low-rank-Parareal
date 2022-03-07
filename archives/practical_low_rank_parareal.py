# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] pycharm={"name": "#%% md\n"}
# # Multirank Parareal : using DLRA
# ### Author : Benjamin Carrel, PhD student of Bart Vandereycken
# ### University of Geneva, May 2021

# %% [markdown]
# ## 0. Setup
#
# Parareal iteration : $$U_{n+1}^{k+1} = F(U_n^k) + G(U_n^{k+1}) - G(U_n^k)$$
#
# Multirank Parareal iteration :
# $$U_{n+1}^{k+1} = F( \mathcal{T}_{r_2} (U_n^k) ) + G( \mathcal{T}_{r_1} (U_n^{k+1})) - G( \mathcal{T}_{r_1} (U_n^k))$$

# %% [markdown]
# ## 1. Python importations and useful functions

# %% pycharm={"name": "#%%\n"}
# IMPORTATIONS
from main.src.integrators import *
from problem_maker import *
from problems import *
from drafts.useful_functions import *
from SVD import *

# %% [markdown]
# ## 2. Setup the problem

# %% pycharm={"name": "#%%\n"}
# CHOOSE PROBLEM : lyapunov OR riccati
choice_of_problem = 'lyapunov'
# choice_of_problem = 'riccati'

# # LYAPUNOV STIFF PROBLEM
if choice_of_problem == 'lyapunov':
    n = 50
    t_span = (0,1)
    (A,b,X0) = make_stiff_problem(n)
    PB = LyapunovProblem(A, b, t_span, X0)
    PB_name = 'Lyapunov_stiff'
    l = 2*np.max(la.eigvals(A.todense())); l=l.real; print(f'l = {l.real}')
    q, r = 2, 20

# RICCATI OSTERMANN STIFF
elif choice_of_problem=='riccati':
    n=50; q=5
    t_span = (0,0.1)
    (A,B,C,X0) = make_riccati_ostermann(n,q)
    print(f'Maximal eigenvalue of A: {np.max(la.eigvals(A.todense()))}')
    rank = 20
    PB = RiccatiProblem(A, B, C, t_span, X0)
    PB_name = 'Riccati_Ostermann'
    q, r = 2, 30

# %% pycharm={"name": "#%%\n"}
# Create integrator
nb = 100
PB_int = IntegrateDLRA(PB, rank=r, nb_t_steps=nb,
                       dlra_method='KSL2', monitoring=True)

# Compute exact sol
ts = PB_int.ts(nb)
Y_sol = PB_int.solve(t_span, PB.Y0, t_eval=ts)
t = 1

# Plot singular values
x = (np.arange(1,n+1),)
y = (la.svdvals(Y_sol[-1]),)
fig = semilogy_colored(x, y, ('o',), (f'sing. values at time {t}',), xlabel='index', ylabel='value')


# %% pycharm={"name": "#%%\n"}
def compare_dlra_and_best(r):
    # Compute best initial_rank approx
    PB_int.problem.select_ode('F')
    Y_best = PB_int.solve_best_rank(t_span, PB.Y0, t_eval=ts, rank=r)
    err_best = compute_many_errors(Y_best, Y_sol)
    # Compute DLRA
    iv = truncate(PB.Y0, r)
    Y_dlra = PB_int.solve_dlra(t_span, iv)
    err_dlra = compute_many_errors(Y_dlra, Y_sol)
    # Plot
    plt.semilogy(ts, err_best, '--', label=f'best rank {r}')
    plt.semilogy(ts, err_dlra, label=f'dlra rank {r}')

fig = plt.figure(dpi=125, figsize=(7,5))
compare_dlra_and_best(r=q)
compare_dlra_and_best(r=r)

plt.grid()
plt.legend()
plt.show()

# %% [markdown] pycharm={"name": "#%% md\n"}
# ## 3. Practical Multirank Parareal

# %% pycharm={"name": "#%%\n"}
# Define subintegrator (less substeps)
PB_int = IntegrateDLRA(PB, rank=r, nb_t_steps=10,
                       dlra_method='KSL2', monitoring=False)
# Coarse solver using DLRA
def coarse_solver(t_span, Y0):
    # Truncate to initial_rank q
    Y0 = truncate(Y0, q)
    # Integrate
    Y = PB_int.solve_dlra(t_span, Y0)
    (U1,S1,V1) = Y[-1]
    Y1 = U1.dot(S1.dot(V1.T))
    return Y1

# Fine solver using DLRA
def fine_solver(t_span, Y0):
    # Truncate to initial_rank r
    Y0 = truncate(Y0, r)
    # Integrate
    Y = PB_int.solve_dlra(t_span, Y0)
    (U1,S1,V1) = Y[-1]
    Y1 = U1.dot(S1.dot(V1.T))
    return Y1


# %% pycharm={"name": "#%%\n"}
# Multirank Parareal iteration
N = 10
K = N
T0,T1 = t_span
initial_iteration = (PB.Y0,)


# Animation
title = f'{PB_name}_practical_multirank_q_{q}_r_{r}'
ts = np.linspace(T0, T1, N + 1)
sol = PB_int.solve(t_span, PB.Y0, t_eval=ts)
anim, err_coarse, err_fine, err_para = parareal_animator(fine_solver, coarse_solver, t_span, initial_iteration, N, sol,
                                                title=title, coarse_name=f'Coarse Solver q={q}', fine_name=f'Fine Solver r={r}',
                                                plot=True)
#HTML(anim.to_jshtml())


# %% pycharm={"name": "#%%\n"}
# Plot
it = np.arange(0,N+1)
x = (it, it, it)
y = (err_coarse, err_fine, err_para)
linestyles = ('-', '-', 'o-')
labels = ('coarse error', 'fine error', f'practical, q={q}, r={r}')
fig = semilogy_colored(x, y, linestyles=linestyles, labels=labels, xlabel='iteration $k$', ylabel='error', loc='lower left')


# %% [markdown]
# ## 4. Many ranks

# %% [markdown]
# In this part, we will apply the algorithm with many different ranks for the coarse initial_rank.
#
# WARNING: This imply heavy computations, so uncomment it only if needed.

# %% pycharm={"name": "#%%\n"}
# FINE SOLVER
def fine_solver(t_span, Y0):
    # Truncate to initial_rank r
    Y0 = truncate(Y0, r)
    # Integrate
    Y = PB_int.solve_dlra(t_span, Y0)
    (U1,S1,V1) = Y[-1]
    Y1 = U1.dot(S1.dot(V1.T))
    return Y1


# %% pycharm={"name": "#%%\n"}
# Define variables
all_q = [1,2,3,4,5,6,7,8]
err_para = np.zeros((N+1,len(all_q)))

# PLOT
fig = plt.figure(dpi=125, figsize=(7,5))
plt.semilogy(it, err_fine, '-')
plt.grid()
plt.xlabel('iteration $k$', fontsize=14)
plt.ylabel('error',fontsize=14)

# MANY COARSE SOLVERS
for i in range(len(all_q)):
    q = all_q[i]
    # Define coarse solver
    def coarse_solver(t_span, Y0):
        # Truncate to initial_rank q
        Y0 = truncate(Y0, q)
        # Integrate
        Y = PB_int.solve_dlra(t_span, Y0)
        (U1,S1,V1) = Y[-1]
        Y1 = U1.dot(S1.dot(V1.T))
        return Y1
    # Apply Parareal
    print(f'q={q}')
    ts, Unk = Parareal(fine_solver, coarse_solver, t_span, initial_iteration, N)
    # Compute errors
    for k in range(N+1):
        err_para[k,i] = np.max(la.norm(sol-Unk[:,k], axis=(1,2)))
    # Plot
    plt.semilogy(it, err_para[:,i], 'o-', label=f'q={q}')

plt.tight_layout()
plt.legend(loc='upper right', fontsize=14)
plt.show()

# %% pycharm={"name": "#%%\n"}
fig.savefig(f'figures/{PB_name}_practical_many_coarse_ranks.pdf')



# %% pycharm={"name": "#%%\n"}
