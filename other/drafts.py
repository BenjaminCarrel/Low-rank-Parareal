#%% md

# Bonus: Animation of singular values

#%%

# Animation of the singular values
# Compute the singular values of the solution at each ts steps
nb_t_steps = 9
sing_val = np.zeros((n,nb_t_steps))
ts = np.linspace(T0,T1,nb_t_steps)
for j in tqdm(range(nb_t_steps)):
    sol_j = exact_sol(ts[j])
    (U,S,V) = la.svd(sol_j)
    sing_val[:,j] = S

#%%

# Animation
fig, ax = plt.subplots(1,dpi=100)
plot = ax.semilogy(sing_val[:,0], 'bo', label='Sing. Values')
ax.legend()
plt.grid()
ax.set(xlabel='Index', ylabel='Value', title=f'Lyapunov - Singular values at time t={ts[0]}')
plt.close()
def update_plot(frame_nb, data, plot):
    ax.lines[0].remove()
    plot = ax.semilogy(sing_val[:,frame_nb], 'bo')
    ax.set_title(f'Lyapunov - Singular values at time t={ts[frame_nb]}')

SingVals = animation.FuncAnimation(fig, update_plot, nb_t_steps, fargs=(sing_val, plot))
#SingVals.save(f'figures/singular_values_over_time.gif', writer='imagemagick', fps=5)
#HTML(SingVals.to_jshtml())

#%% md

# Drafts

#%%



# Matrix of Enk
matrix_Enk = np.zeros((N+1,K+1))
for n in range(N+1):
    matrix_Enk[n,0] = la.norm(sol[n]-Unk[n,0]) # Need E_n^0 (and E_0^k = 0)
for n in range(N):
    for k in range(K):
        matrix_Enk[n+1, k+1] = alpha * E[n,k] + beta * E[n,k+1]



# Inspect more precisely the iteration bound
h = T1/N
n = N-1
errors = np.zeros(K)
exact_iter_bound = np.zeros(K)
best_bound = np.zeros(K)
superlinear_bound = np.zeros(K)
for k in np.arange(0,K):
    errors[k] = la.norm(sol[n+1]-Unk[n+1,k+1])
    Enk = la.norm(sol[n]-Unk[n,k])
    Enk1 = la.norm(sol[n]-Unk[n,k+1])
    exact_iter_bound[k] = alpha * Enk + beta * Enk1
    superlinear_bound[k] = alpha**k * np.product([n-l for l in range(k+1)]) / np.math.factorial(k+1) * ini_err

# for the best formal sum
best_bound[0] = ini_err
for k in np.arange(1,K):
    best_bound[k] = alpha ** k * sum([math.comb(k+j-1,j) * beta**j * Enk[n-k-j][0] for j in np.arange(0,n-k)])

# Plot
it = np.arange(0,K)
plt.figure(dpi=100)
plt.semilogy(it, errors, label='$E_{n+1}^{k+1}$')
plt.semilogy(it, E[n,:-1], '--', label='Matrix of $E_n^k$')
plt.semilogy(it, exact_iter_bound, '--', label='$a*E_n^k + b*E_n^{k+1}$')
plt.semilogy(it, best_bound, '+', label='Exact with formal sum')
plt.semilogy(it, superlinear_bound, '-.', label='Superlinear bound')
plt.legend()
plt.grid()
plt.title('Idealized low initial_rank parareal parareal - bound estimation')
plt.xlabel('iteration k'); plt.ylabel(f'Error bounds with n = {n}')
plt.show()

#%%

# Convergence curve
n_h = 100
all_h = np.linspace(0,2,n_h)
lin_gamma = np.zeros(n_h)
for k in range(n_h):
    h = all_h[k]
    lin_gamma[k] = (1*np.exp(l*h) / (1-1*np.exp(l*h)))**2

# Plot
plt.figure(dpi=100)
plt.semilogy(all_h, lin_gamma, label='$\gamma(h)$')
plt.grid()
plt.legend()
plt.xlabel('h')
plt.ylabel('Convergence rate')

#%%



#%%
#%% md
# Drafts

#%%

# ts, Unk = low_rank_Parareal(fine_solver, coarse_solver, t_span, initial_iteration, N)

# # Compute exact sol
# lpint.problem.select_ode('F')
#
#
# # Compute errors
# err_para = np.zeros(N+1)
# for k in range(N+1):
#     err_k = np.zeros(N+1)
#     for j in range(N+1):
#         err_k = compute_errors(Unk[j][k], sol[j])
#     err_para[k] = np.max(err_k)
# err_fine = [err_para[-1] for _ in range(N+1)]
# err_coarse = [err_para[0] for _ in range(N+1)]