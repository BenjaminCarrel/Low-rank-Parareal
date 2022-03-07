# -*- coding: utf-8 -*-
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

# %% [markdown]
# # Multirank Parareal : using exact solvers
# ### Author : Benjamin Carrel, PhD student of Bart Vandereycken
# ### University of Geneva, Februrary 2021

# %% [markdown]
# ## 0. Setup
#
# Parareal iteration : $$U_{n+1}^{k+1} = F(U_n^k) + G(U_n^{k+1}) - G(U_n^k)$$
#
# Multirank Parareal iteration :
# $$U_{n+1}^{k+1} = F( \mathcal{T}_{r_2} (U_n^k) ) + G( \mathcal{T}_{r_1} (U_n^{k+1})) - G( \mathcal{T}_{r_1} (U_n^k))$$
#
# If one uses exact solvers (coarse and fine) for the original parareal iteration, we clearly converge in one iteration. What happen is less clear for the low_rank_parareal parareal iteration. Let's check what happen.

# %% [markdown]
# ## 1. Python importations and useful functions

# %%
# Import complete package

from problems import *
from main.src.integrators import *
from problem_maker import *
import matplotlib.pyplot as plt
import math

# %% [markdown]
# ## 2. Setup the problem
#
# Comment & uncomment to choose the problem.

# %%
# CHOOSE PROBLEM : lyapunov OR riccati
choice_of_problem = 'lyapunov'
#choice_of_problem = 'riccati'

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
    q, r= 2, 30

# # RICCATI BENCHMARK RAIL PROBLEM
# # Copy your absoluate path to data here
# t_span = (0,1)
# (A,B,C,Q0) = make_riccati_rail('/Users/benjamincarrel/switchdrive/PhD/GitLab/DLRA/python/src/data/rail_371.mat')
# PB = RiccatiProblem(A, B, C, t_span, Q0)
# B = B*0
# PB_name = 'Riccati_rail'

# %%
# Create integrator
PB_int = IntegrateClassic(PB)

# Compute exact sol
t_eval = np.array([0,0.05,0.1])
Y_sol = PB_int.solve(t_span, PB.Y0, t_eval)

# Plot singular values
it = np.arange(1,n+1)
x = (it, it, it, it)
y = (la.svdvals(Y_sol[0]), la.svdvals(Y_sol[1]), la.svdvals(Y_sol[2]), 2 ** -53 * np.ones(n))
linestlyes = ('o', 'xs', '+', '--')
labels = (f't={t_eval[0]}', f't={t_eval[1]}', f't={t_eval[2]}','$\epsilon_{machine}$')
fig = semilogy_colored(x, y, linestlyes, labels, xlabel='index', ylabel='value')
fig.savefig(f'figures/{PB_name}_singular_values.pdf')


# %% [markdown]
# ## 3. Idealized Multirank Parareal

# %%
# Define coarse solver as described above
def coarse_solver(t_span, Y0):
    # Truncate to initial_rank q
    Y0 = best_rank_approx(Y0, q)
    # Integrate
    Y1 = PB_int.solve(t_span, Y0)
    return Y1

# Define fine solver as described above
def fine_solver(t_span, Y0):
    # Truncate to initial_rank r
    Y0 = best_rank_approx(Y0, r)
    # Integrate
    Y1 = PB_int.solve(t_span, Y0)
    return Y1


# %% pycharm={"name": "#%%\n"}

# INITIAL ITERATION
initial_iteration = (PB.Y0,)
#initial_iteration = PB_int.solve_best_rank(t_span, PB.Y0, t_eval=ts, initial_rank=q)
#initial_iteration[0] = PB.Y0

# Multirank Parareal iteration
N = 10; K = N
ts = np.linspace(t_span[0], t_span[1], N+1)
ts, Unk = Parareal(fine_solver, coarse_solver, t_span, initial_iteration, N)

# %% pycharm={"name": "#%%\n"}
# Animation
title = f'{PB_name}_idealized_multirank_q_{q}_r_{r}'
sol = PB_int.solve(t_span, PB.Y0, t_eval=ts)
anim, err_coarse, err_fine, err_para = parareal_animator(fine_solver, coarse_solver, t_span, initial_iteration, N, sol,
             title=title, coarse_name='Coarse Solver', fine_name='Fine Solver', plot=True)
#HTML(anim.to_jshtml())

# %% [markdown]
# ## 4. Theoretical bounds

# %% [markdown]
# Using the exact solver as fine solver, we obtained the iteration (for an affine flow) :
# $$ U(t_{n+1}) - U_{n+1}^{k+1} = \tilde{\mathcal{T}}_q^{\perp} (U(t_n)) - \tilde{\mathcal{T}}_q^{\perp} (U_n^k) + \tilde{\mathcal{T}}_q (U(t_n)) - \tilde{\mathcal{T}}_q (U_n^k), $$
# where $\tilde{\mathcal{T}}_q = \Phi^h \circ \mathcal{T}_q$, and $\Phi^h(X)$ is the exact flow of the problem with initial value X.
#
# Let $\alpha$ be the Lipschitz constant of $\tilde{\mathcal{T}}_q^{\perp}$ and $\beta$ be the Lipschitz constant of $\tilde{\mathcal{T}}_q$. Then,
# $$ || E_{n+1}^{k+1} || \leq \alpha || E_n^k|| + \beta ||E_n^{k+1}||. $$
#
# Since $\tilde{\mathcal{T}}_q = \Phi^h \circ \mathcal{T}_q$, it is also possible to decompose this Lipschitz constant into two parts, ie,
# \begin{align}
# || \tilde{\mathcal{T}}_q (U) - \tilde{\mathcal{T}}_q (V) || 
# &\leq || \Phi^h( \mathcal{T}_q(U)) - \Phi^h(\mathcal{T}_q(V)) || \\
# &\leq L_{\Phi^h} ||\mathcal{T}_q(U)-\mathcal{T}_q(V)|| \\
# &\leq L_{\Phi^h} C_q ||U-V||
# \end{align}
# where $C_q$ is the Lipschitz constant of $\mathcal{T}_q$ and $L_{\Phi^h}$ is the Lipschitz constant of $\Phi^h$. It seems that the Lipschitz constant of $\mathcal{T}_q$ is $1$ since the derivative of $\mathcal{T}_q$ is the projection onto the tangent space.
#
# Idem for $\mathcal{T}_q^{\perp}$ and $C_q^{\perp}$.

# %% [markdown] pycharm={"name": "#%% md\n"}
# ### 4.1 Estimation of $\alpha$ and $\beta$
#
# Recall :
#
# $$ \alpha = \max \frac{|| \phi^h \circ T_q^{\perp}( U(t_n) ) - \phi^h \circ T_q^{\perp}( U_n^k ) ||}{|| U(t_n) - U_n^k ||} $$
#
# $$ \beta = \max \frac{|| \phi^h \circ T_q( U(t_n) ) - \phi^h \circ T_q( U_n^{k+1} ) ||}{|| U(t_n) - U_n^{k+1} ||} $$

# %% pycharm={"name": "#%%\n"}
def estimate_alpha_beta(sol, PB_int):
    all_alpha = np.zeros((N, K))
    all_beta = np.zeros((N, K))
    # LOOP ON N
    for nt in np.arange(1, N):
        # EXACT TERM
        ex = sol[nt]

        # ALPHA
        (U, S, V) = truncate_perp(ex, q)
        trp_ex = U.dot(S.dot(V.T))
        phi_trp_ex = PB_int.solve(t_span=ts[nt:nt + 2], initial_value=trp_ex)
        for k in np.arange(0, nt):
            Unk_val = Unk[nt, k]
            (U, S, V) = truncate_perp(Unk_val, q)
            trp_Unk = U.dot(S.dot(V.T))
            phi_trp_Unk = PB_int.solve(t_span=ts[nt:nt + 2], initial_value=trp_Unk)
            all_alpha[nt, k] = la.norm(phi_trp_Unk - phi_trp_ex) / la.norm(Unk_val - ex)

        # BETA
        (U, S, V) = truncate(ex, q)
        tr_ex = U.dot(S.dot(V.T))
        phi_tr_ex = PB_int.solve(t_span=ts[nt:nt + 2], initial_value=tr_ex)
        for k in np.arange(0, nt - 1):
            Unk_val = Unk[nt, k + 1]
            (U, S, V) = truncate(Unk_val, q)
            tr_Unk = U.dot(S.dot(V.T))
            phi_tr_Unk = PB_int.solve(t_span=ts[nt:nt + 2], initial_value=tr_Unk)
            all_beta[nt, k + 1] = la.norm(phi_tr_Unk - phi_tr_ex) / la.norm(Unk_val - ex)
    alpha = np.max(all_alpha)
    print(f'max alpha = {alpha}')
    beta = np.max(all_beta)
    print(f'max beta = {beta}')
    return alpha, beta

alpha, beta = estimate_alpha_beta(sol, PB_int)

# %% [markdown] pycharm={"name": "#%% md\n"}
# ### 4.2 Figures

# %% pycharm={"name": "#%%\n"}
# Convergence rate
# alpha = np.exp(l*h); beta = np.exp(l*h)
gamma = np.absolute(alpha / (1-beta))
print(f'Convergence rate : {gamma}')

# Compute errors
Enk = np.zeros((N+1,K+1))
for i in range(N+1):
    for k in range(N+1):
        Enk[i,k] = la.norm(sol[i]-Unk[i,k])

# Compute bounds
it = np.arange(0,K+1)
ini_err = err_para[0]; fin_err = err_para[-1]
linear = np.zeros(K+1); superlinear = np.zeros(K+1); optimal = np.zeros(K+1)
for k in it:
    linear[k] = gamma ** k * ini_err
    superlinear[k] = alpha ** k * np.product([N-l for l in range(k)]) / np.math.factorial(k+1) * ini_err
optimal[0] = ini_err
for k in it[1:]:
    optimal[k] = alpha ** k * sum([Enk[N-k-j,0] * math.comb(k+j-1,j) * beta**j for j in np.arange(0,N-k)])

# Plot
x = (it, it, it, it, it, it)
y = (err_coarse, err_fine, err_para, linear, superlinear, optimal)
linestyles = ('-', '-', 'o-', '--', '--', '--')
labels = (None, None, f'idealized, q={q}, r={r}', 'linear bound', 'superlinear bound', 'optimal bound')
fig = semilogy_colored(x, y, linestyles, labels, xlabel='iteration $k$', ylabel='error', loc='lower left')
fig.savefig(f'figures/{PB_name}_idealized_bounds.pdf')

# %% [markdown] pycharm={"name": "#%% md\n"}
# ## 5. Many ranks

# %% [markdown] pycharm={"name": "#%% md\n"}
# In this part, we will apply the algorithm with many different ranks for the coarse initial_rank.
#
# WARNING: This imply heavy computations, so uncomment it only if needed.

# %% pycharm={"name": "#%%\n"}
# FINE SOLVER
r = 20
def fine_solver(t_span, Y0):
    # Truncate to initial_rank r
    Y0 = best_rank_approx(Y0, r)
    # Integrate
    Y1 = PB_int.solve(t_span, Y0)
    return Y1


# %% pycharm={"name": "#%%\n"}
# Define variables
all_q = [1,2,3,4,5,6,7,8]
err_para = np.zeros((N+1,len(all_q)))

# PLOT
fig = plt.figure(dpi=150, figsize=(7,5))
plt.semilogy(it, err_fine, '-')
plt.grid()
plt.xlabel('iteration $k$', fontsize=14)
plt.ylabel('error', fontsize=14)

# MANY COARSE SOLVERS
for i in range(len(all_q)):
    q = all_q[i]
    # Define coarse solver
    def coarse_solver(t_span, Y0):
        # Truncate to initial_rank q
        Y0 = best_rank_approx(Y0, q)
        # Integrate
        Y1 = PB_int.solve(t_span, Y0)
        return Y1
    # Apply Parareal
    print(f'q={q}')
    ts, Unk = Parareal(fine_solver, coarse_solver, t_span, initial_iteration, N)
    # Compute errors
    for k in range(N+1):
        err_para[k,i] = np.max(la.norm(sol-Unk[:,k], axis=(1,2)))
    # Plot
    plt.plot(it, err_para[:,i], 'o-', label=f'q={q}')

plt.legend(loc='upper right', fontsize=14)
plt.tight_layout()
plt.show()

# %% pycharm={"name": "#%%\n"}
fig.savefig(f'figures/{PB_name}_idealized_many_coarse_ranks.pdf')


# %% [markdown] pycharm={"name": "#%% md\n"}
# ## 6. Many size of problem (Lyapunov stiff only!)

# %% pycharm={"name": "#%%\n"}
if PB_name == 'Lyapunov_stiff':
    # Parameters
    all_n = [50, 100, 150]
    nb = len(all_n)
    N=10
    err_para = np.zeros((N+1,nb))
    it = np.arange(0,N+1)
    fig = plt.figure()

    # Plot
    fig = plt.figure(dpi=125, figsize=(7,5))
    colors = ['r','b','g']
    #plt.semilogy(it, err_fine, '-')
    plt.xlabel('iteration $k$', fontsize=14)
    plt.ylabel('error', fontsize=14)

    # Solve and plot for many n
    for k in range(len(all_n)):
        n = all_n[k]
        t_span = (0,1)
        (A,b,X0) = make_stiff_problem(n)
        PB = LyapunovProblem(A, b, t_span, X0)
        PB_int = IntegrateClassic(PB)
        PB_name = 'Lyapunov_stiff'

        # COARSE AND FINE SOLVERS
        q = 2
        def coarse_solver(t_span, Y0):
            Y0 = best_rank_approx(Y0, q)
            Y1 = PB_int.solve(t_span, Y0)
            return Y1
        r = 20
        def fine_solver(t_span, Y0):
            Y0 = best_rank_approx(Y0, r)
            Y1 = PB_int.solve(t_span, Y0)
            return Y1

        # PARAREAL
        iv = PB.Y0
        initial_iteration = (iv,)
        ts, Unk = Parareal(fine_solver, coarse_solver, t_span, initial_iteration, N)

        # EXACT SOL
        sol = PB_int.solve(t_span, iv, t_eval=ts)

        # ERRORS
        for j in range(N+1):
            err_para[j,k] = np.max(la.norm(sol-Unk[:,j], axis=(1,2)))
        Enk = np.zeros((N+1,N+1))
        for i in range(N+1):
            for j in range(N+1):
                Enk[i,j] = la.norm(sol[i]-Unk[i,j])

        # BOUND
        alpha, beta = estimate_alpha_beta(sol, PB_int)
        ini_err = err_para[0,k]
        optimal = np.zeros(N+1)
        optimal[0] = ini_err
        for j in it[1:]:
            optimal[j] = alpha ** j * sum([Enk[N-j-i,0] * math.comb(j+i-1,i) * beta**i for i in np.arange(0,N-j)])

        # PLOT
        color = colors[k]
        plt.semilogy(it, err_para[:,k], 'o-', c=color, label=f'n={n}')
        plt.semilogy(it, optimal, '--', c=color, label=f'bound n={n}')

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.grid()
    plt.show()
    fig.savefig(f'figures/{PB_name}_idealized_many_sizes.pdf')

# %% [markdown] pycharm={"name": "#%% md\n"}
# ## 7. Plot convergence rate Riccati Ostermann

# %% pycharm={"name": "#%%\n"}
if PB_name == 'Riccati_Ostermann':
    # PLOT
    colors = ['r','b','g']
    fig = plt.figure(dpi=125, figsize=(7,5))
    plt.semilogy(it, err_fine, '-')
    err_para = np.zeros((N+1,3))

    for k in range(0,3):
        # COARSE AND FINE SOLVERS
        q = k+1
        def coarse_solver(t_span, Y0):
            # Truncate to initial_rank q
            Y0 = best_rank_approx(Y0, q)
            # Integrate
            Y1 = PB_int.solve(t_span, Y0)
            return Y1
        r = 30
        def fine_solver(t_span, Y0):
            # Truncate to initial_rank r
            Y0 = best_rank_approx(Y0, r)
            # Integrate
            Y1 = PB_int.solve(t_span, Y0)
            return Y1

        # PARAREAL
        iv = PB.Y0
        initial_iteration = (iv,)
        ts, Unk = Parareal(fine_solver, coarse_solver, t_span, initial_iteration, N)

        # EXACT SOL
        sol = PB_int.solve(t_span, iv, t_eval=ts)

        # ERRORS
        for j in range(N+1):
            err_para[j,k] = np.max(la.norm(sol-Unk[:,j], axis=(1,2)))

        # PLOT
        color = colors[k]
        plt.semilogy(it, err_para[:,k], 'o-', c=color, label=f'$q={q}$')
        plt.semilogy(it, err_para[0,k] * 0.25**(it*(k+1)), '--', c=color, label=f'order ${k+1}$')

    plt.tight_layout()
    plt.legend(loc='upper right', fontsize=14)
    plt.xlabel('iteration $k$', fontsize=14)
    plt.ylabel('error', fontsize=14)
    plt.ylim(1e-12, 1e0)
    plt.grid()
    plt.show()
    fig.savefig(f'figures/{PB_name}_convergence_rate.pdf')

# %% pycharm={"name": "#%%\n"}
