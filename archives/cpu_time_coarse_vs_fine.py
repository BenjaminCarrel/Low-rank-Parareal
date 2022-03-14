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
# # CPU TIME

# %% [markdown] pycharm={"name": "#%% md\n"}
# ### Importations

# %% pycharm={"name": "#%%\n"}
import timeit
from main.src.integrators import *
from problems import *
from problem_maker import *
from tqdm import tqdm
from matplotlib.pyplot import plot as plt


# %% [markdown] pycharm={"name": "#%% md\n"}
# ### TIME OF DLRA: size $n$
#
# Consider the nonstiff lyapunov problem.
# We fix the initial_rank of approximation $k$ and we vary the size of the problem $n$.
# We take the mean ts over many loops.

# %% pycharm={"name": "#%%\n"}
# Parameters
all_n = np.logspace(1, 5, num=20, dtype=int)
k = 8
n_step = 1
t_span = (0,0.001)
tn = np.zeros(len(all_n))

for j in tqdm(range(len(all_n))):
    n = all_n[j]
    # Create problem
    (A,b,X0) = make_nonstiff_problem(n)
    PB = LyapunovProblem(A, b, t_span, X0)
    # Time DLRA
    PB_int = IntegrateDLRA(PB, rank=k, nb_t_steps=n_step, dlra_method='KSL2', monitoring=False)
    #PB_int.classical_method = 'scipy'
    def to_time():
        PB_int.solve_dlra(t_span, PB_int.Y0)
    tn[j] = timeit.timeit(to_time, number=10)/10
print('Done.')

# %% pycharm={"name": "#%%\n"}
fig = plt.figure()
plt.loglog(all_n, tn, '-xs', label='KSL order 2')
plt.loglog(all_n, all_n/1000, '--', label='slope 1')
plt.xlabel('size $n$')
plt.ylabel('CPU ts')
plt.title(f'CPU ts for fixed initial_rank $k={k}$')
plt.legend()
plt.grid()
plt.show()
#fig.savefig('figures/CPU_time_fixed_rank.pdf')

# %% [markdown] pycharm={"name": "#%% md\n"}
# ### One step of DLRA varying $k$
#
# Consider the nonstiff lyapunov problem.
# We fix the size of the problem $n$ and we vary the initial_rank of approximation $k$.
# We take the mean ts over many loops.

# %% pycharm={"name": "#%%\n"}
# Parameters
n = 10000
all_k = np.logspace(0, 2.7, num=9, dtype=int)
n_step = 1
t_span = (0,0.001)
tk = np.zeros(len(all_k))

# Create problem
(A,b,X0) = make_nonstiff_problem(n, initial_rank=1000)
PB = LyapunovProblem(A, b, t_span, X0)

for j in tqdm(range(len(all_k))):
    k = all_k[j]
    # Time DLRA
    PB_int = IntegrateDLRA(PB, rank=k, nb_t_steps=n_step, dlra_method='KSL2', monitoring=False)
    #PB_int.classical_method = 'scipy'
    def to_time():
        PB_int.solve_dlra(t_span, PB_int.Y0)
    tk[j] = timeit.timeit(to_time, number=10)/10
print('Done.')

# %% pycharm={"name": "#%%\n"}
fig = plt.figure()
plt.loglog(all_k, tk, '-xs', label='KSL order 2')
plt.loglog(all_k, all_k**2/100, '--', label='slope 2')
plt.xlabel('initial_rank $k$')
plt.ylabel('CPU ts')
plt.title(f'CPU ts for fixed size $n={n}$')
plt.legend()
plt.grid()
plt.show()
#fig.savefig('figures/CPU_time_fixed_size.pdf')

# %% [markdown] pycharm={"name": "#%% md\n"}
# Note : for low initial_rank parareal, there are additionnal costs due to the truncations and the updates.

# %% [markdown] pycharm={"name": "#%% md\n"}
# ### Performance: low-initial_rank Parareal

# %% [markdown] pycharm={"name": "#%% md\n"}
# Based on the previous observations, we have:
# - Cost of the fine solver: $O(nr^2)$
# - Cost of the coarse solver: $O(nq^2)$
# - Cost of assembling: ignore?
#
# Hence, it is possible to plot the total cost of the low initial_rank parareal algorithm
# assuming we are using $N_{k}$ processors in parallel.

# %% pycharm={"name": "#%%\n"}
# PARAMETERS
N_iter = 10
n = 150
r = 20
q = 2

# ITERATIONS
it = np.arange(0,N_iter)

# COST FINE SOLVER
cost_fine = N_iter * n * r ** 2 * np.ones(N_iter)

# COST COARSE SOLVER
cost_coarse = N_iter * n * q ** 2 * np.ones(N_iter)

# COST LOW RANK PARAREAL WITH N PROCESSORS
cost_parareal = np.zeros(N_iter)
for k in np.arange(N_iter):
    cost_parareal[k] = k * n * r ** 2 + (k+1) * N_iter * n * q ** 2

# PLOT
fig = plt.figure()
plt.semilogy(it, cost_fine, '--', label=f'Fine solver $r={r}$')
plt.semilogy(it, cost_coarse, '--', label=f'Coarse solver $q={q}$')
plt.semilogy(it, cost_parareal, '-xs', label=f'Low initial_rank Parareal')
plt.xlabel('iteration $k$')
plt.ylabel('cost')
plt.title('Theoretical performance of low-initial_rank Parareal')
plt.grid()
plt.legend()
plt.show()
fig.savefig('figures/cost_performance.pdf')

# %% [markdown] pycharm={"name": "#%% md\n"}
# Now, we can do the same with cputime as we estimated them above.

# %% pycharm={"name": "#%%\n"}
## CPU TIME ESTIMATION
# CREATE PROBLEM
n = 1000
t_span = (0,0.00001)
(A,b,X0) = make_stiff_problem(n)
PB = LyapunovProblem(A, b, t_span, X0)

# FINE TIME
r = 20
n_fine_step = 1
PB_int = IntegrateDLRA(PB, rank=r, nb_t_steps=n_fine_step, dlra_method='KSL2', monitoring=False)
def to_time():
        PB_int.solve_dlra(t_span, PB_int.Y0)
one_time_fine = timeit.timeit(to_time, number=10)/10
time_fine = N_iter * one_time_fine * np.ones(N_iter)

# COARSE TIME
q = 2
n_coarse_step = 1
PB_int = IntegrateDLRA(PB, rank=q, nb_t_steps=n_coarse_step, dlra_method='KSL2', monitoring=False)
def to_time():
        PB_int.solve_dlra(t_span, PB_int.Y0)
one_time_coarse = timeit.timeit(to_time, number=10)/10
time_coarse = N_iter * one_time_coarse * np.ones(N_iter)

# ASSEMBLING TIME
# MAYBE LATER

# LOW RANK PARAREAL TIME
time_parareal = np.zeros(N_iter)
for k in np.arange(N_iter):
    time_parareal[k] = k * one_time_fine + (k+1) * N_iter * one_time_coarse

# PLOT
fig = plt.figure()
plt.semilogy(it, time_fine, '--', label=f'Fine solver $r={r}$')
plt.semilogy(it, time_coarse, '--', label=f'Coarse solver $q={q}$')
plt.semilogy(it, time_parareal, '-xs', label=f'Low initial_rank Parareal')
plt.xlabel('iterations')
plt.ylabel('CPU ts')
plt.title('Perfomance of low-initial_rank Parareal')
plt.grid()
plt.legend()
plt.show()
fig.savefig('figures/cpu_time_performance')


# %% pycharm={"name": "#%%\n"}
