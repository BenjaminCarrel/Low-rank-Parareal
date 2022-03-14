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
#     display_name: 'Python 3.9.1 64-bit (''base'': conda)'
#     name: python391jvsc74a57bd0bf170c85b31d4838a35a5e905515c3cfda122e280d5fd842b19232f61c46ef8d
# ---

# %% [markdown]
# # Multirank Parareal
# ## Author : Benjamin Carrel, PhD student of Bart Vandereycken

# %% [markdown]
# ### 0. Problem
#
# The problem is to study the constant $C_r$ in the following inequality
# $$ || \mathcal{T}_r^{\perp} (A) - \mathcal{T}_r^{\perp} (A+hE) || \leq C_r h ||E|| $$
# when $h$ is small. Sometimes we denote $\tilde{A} = A(h) = A+hE$.
#
# Questions :
#
# 1.Do we have $C_r < 1$ ?
#
# 2.What assumptions do we need on $A$ ?
#
# 3.How $C_r$ variate with $r$ ?
#

# %% [markdown]
# ### 0.1 Python importations

# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import time as tm
from tqdm import tqdm
np.random.seed(1234)


# %% [markdown]
# ### 0.2 Useful functions

# %%
# Define some functions
def truncate_perp(X, r):
    (U,s,Vt) = np.linalg.svd(X)
    U = U[:,r:]
    V = Vt[r:,:].T
    S = np.diag(s[r:])
    return U.dot(S.dot(V.T))

def estimate_Cr(X, r, nb_est, h):
    # variables
    X_shape = X.shape
    errors = np.zeros(nb_est)
    # create nb_est random perturbations
    E = np.random.randn(X_shape[0], X_shape[1], nb_est)
    # truncation
    X_trunc_perp = truncate_perp(X, r)
    # compute errors
    for n in range(nb_est):
        X_tilde = X+h*E[:,:,n]
        X_tilde_trunc_perp = truncate_perp(X_tilde, r)
        errors[n] = la.norm(X_trunc_perp-X_tilde_trunc_perp, 2)/la.norm(X-X_tilde, 2)
    max_err = np.max(errors)
    return max_err


# %% [markdown]
# ### 1. Statistical approach

# %% [markdown]
# ### 1.1 Variating $h$
#
# We first study $C_r$ with a fixed initial_rank $r$ and we variate $h$.

# %%
# Parameters
N = 20
r = 8

# Generate A with exponential decay
A = np.diag([np.exp(-k) for k in range(N+1)])
A_trunc_perp = truncate_perp(A,r)

# Size of ||E||
nh = 200
all_h = np.logspace(-14,5,nh)

# Number of E
nb = 20
E = np.random.randn(N+1,N+1,nb)

# Compute errors and C_r
error = np.zeros((nb,nh))
truncated_error = np.zeros((nb,nh))
Crh = np.zeros((nb,nh))
for k in tqdm(range(nb)):
    for n in range(nh):
        h = all_h[n]
        Ah = A + h * E[:,:,k]
        Ah_trunc_perp = truncate_perp(Ah,r)
        error[k,n] = la.norm(A-Ah, 2)
        truncated_error[k,n] = la.norm(A_trunc_perp-Ah_trunc_perp, 2)
        Crh[k,n] = truncated_error[k,n]/error[k,n]

# %%
# Plot
plt.figure(dpi=100)  
plt.loglog(all_h,Crh[0,:],'g--', label='multiples $C_{'+str(r)+'}(h)$')
for k in range(nb-1):
    plt.loglog(all_h,Crh[k+1,:],'g--')
plt.loglog(all_h, np.max(Crh,axis=0),'b+', label='max $C_{'+str(r)+'}(h)$')
plt.loglog(all_h, np.ones(nh), 'k', label='$y=1$')
plt.legend()
plt.grid()
plt.ylabel('Value')
plt.xlabel('h')
plt.title('Assumption - Numerical estimation of $C_{'+str(r)+'}$')

# %% [markdown]
# It seems that, for a fixed initial_rank $r$ :
# - $C_r$ is smaller than 1.
# - $C_r$ decrease when $h$ is larger. (??)

# %% [markdown]
# ### 1.2 Variating $r$
#
# Similarly, we fix $h$, and we study $C_r$ as a function of the initial_rank $r$.

# %%
# parameters
N = 50

# Generate A and B
S_A = np.logspace(2,-14,N)
A = np.diag(S_A)
B = np.random.randn(N,N)
S_B = la.svdvals(B)

# variables
nb_est = 100
h = 10**-1
err_A = np.zeros(N)
err_B = np.zeros(N)
ranks = np.arange(1,N+1)

# estimate Cr
for r in tqdm(ranks-1):
    B = np.random.randn(N,r).dot(np.random.randn(r,N))
    err_A[r] = estimate_Cr(A,r,nb_est,h)
    err_B[r] = estimate_Cr(B,r,nb_est,h)

# %%
# plot
plt.figure(dpi=100)
plt.semilogy(ranks,err_A, label='Exponential decay')
plt.semilogy(ranks,err_B, label='Random but initial_rank exact r')
plt.semilogy(ranks,np.ones(N), 'k--')
plt.xlabel('initial_rank $r$'); plt.ylabel('$C_r$')
plt.legend()
plt.grid()
plt.title('Statistical estimation of $C_r$')


# %% [markdown]
# Observation : 
# - For exponential decay, it seems that $C_r < 1$ and $C_r$ is slowly decreasing in $r$.
# - For non exponential decay, it's not true.

# %% [markdown]
# ### 2. Theoretical bound
#
# One can find the following bound
# \begin{align}
# || \mathcal{T}_r^{\perp} (A) - \mathcal{T}_r^{\perp} (A+E) ||_2 &\leq ||Q||_2 ||U_{\perp}^T A||_2 + ||\tilde{Q}||_2 ||U_{\perp}^T A||_2 + ||(I-U_r U_r^{\perp})E||_2 + \mathcal{O} (||E||_2^2)\\
# &\leq (c \sigma_{r+1}(A) + \tilde{c} \sigma_{r+1}(A) +  \kappa_r) ||E||_2 + \mathcal{O} (||E||_2^2)
# \end{align}
# where $||Q|| = \mathcal{O}(||E||), ||\tilde{Q}|| = \mathcal{O}(||E||)$ and $$\kappa_r = \frac{||(I-U_r U_r^{\perp})E||_2}{||E||_2}.$$ Note that $U_r, U_{\perp}$ come from the SVD of $A$. 

# %% [markdown]
# ### 2.1 Numerical verification

# %%
def compute_bound(A,E,r):
    # svd of A
    (n,_) = A.shape
    (U,s,Vt) = la.svd(A)
    U_r = U[:,:r]
    V_r = Vt[:r,:].T
    S_r = np.diag(s[:r])
    U_perp = U[:,r:]
    V_perp = Vt[r:,:].T
    S_perp = np.diag(s[r:])
    # matrix of error
    E_rr = U_r.T @ E @ V_r
    E_perpr = U_perp.T @ E @ V_r
    E_rperp = U_r.T @ E @ V_perp
    E_perpperp = U_perp.T.dot(E.dot(V_perp))
    norm_E = la.norm(E,2)
    # mu and phi
    mu1 = S_perp @ E_rperp.T + E_perpr @ S_r
    mu1 = - mu1.reshape(-1)
    phi0 = np.kron(S_r @ S_r.T, np.eye(n-r)) - np.kron(np.eye(r), S_perp.dot(S_perp.T))
    mu2 = U_perp.T @ E @ E.T @ U_r
    mu2 = - mu2.reshape(-1)
    phi1 = np.kron(S_r @ E_rr.T + E_rr @ S_r, np.eye(n-r)) - np.kron(np.eye(r), S_perp @ E_perpperp.T + E_perpperp @ S_perp.T)
    # Q and Q_tilde
    vec_Q = la.solve(phi0, mu1)
    vec_Q_tilde = phi1.dot(vec_Q)
    vec_Q_tilde = mu2 - vec_Q_tilde
    vec_Q_tilde = la.solve(phi0, vec_Q_tilde)

    # compute norm kappa_r
    E_bis = E - U_r @ U_r.T @ E
    kappa_r = la.norm(E_bis,2)/norm_E

    # compute c and c_tilde
    Q = vec_Q.reshape(n-r,r)
    c = la.norm(Q,2) / norm_E
    Q_tilde = vec_Q_tilde.reshape(n-r,r)
    c_tilde = la.norm(Q_tilde,2) / norm_E

    bound = c*s[r] + c_tilde*s[r] + kappa_r

    return bound, s[r], c, c_tilde, kappa_r

def compute_Cr(A,E,r):
    # truncate perp
    A_trunc_perp = truncate_perp(A,r)
    A_tilde = A+E
    A_tilde_trunc_perp = truncate_perp(A+E,r)
    # compute norm
    norm_E = la.norm(E, 2)
    truncated_error = la.norm(A_trunc_perp - A_tilde_trunc_perp, 2)
    Cr = truncated_error/norm_E
    return Cr


# %%
# parameters
h = 10**-1
N = 20

# Generate random error E
E = h*np.random.randn(N,N)
# Generate A with exponential decay in sing val
s_A = np.logspace(2,-5,N)
A = np.diag(s_A)
A_tilde = A + E
# Generate random B (for comparison)
B = np.random.randn(N,r).dot(np.random.randn(r,N))
B_tilde = B + E


# Compute singular values
s_E = la.svdvals(E)
s_B = la.svdvals(B)
s_A_tilde = la.svdvals(A_tilde)
s_B_tilde = la.svdvals(B_tilde)

# Plot
index = np.arange(1,N+1)
plt.figure(dpi=100)
plt.semilogy(index, s_A, '+-', label='$A$')
plt.semilogy(index, s_B, '+-', label='$B$')
plt.semilogy(index, s_A_tilde, 'o', label='$A+E$')
plt.semilogy(index, s_B_tilde, 'o', label='$B+E$')
plt.semilogy(index, s_E, 'kx--', label='$E$')
plt.legend()
plt.title('Singular values')

# %%
# variables
ranks = np.arange(1,N)
A_Cr = np.zeros(N-1); A_bound = np.zeros(N-1)
A_c = np.zeros(N-1); A_c_tilde = np.zeros(N-1); A_kappa = np.zeros(N-1)
B_Cr = np.zeros(N-1); B_bound = np.zeros(N-1)
B_c = np.zeros(N-1); B_c_tilde = np.zeros(N-1); B_kappa = np.zeros(N-1)

# compute Cr
for i in tqdm(range(N-1)):
    r = ranks[i]
    A_Cr[i] = compute_Cr(A,E,r)
    B_Cr[i] = compute_Cr(B,E,r)

# compute bound for A
    bd, _, c, c_tilde, kappa = compute_bound(A,E,r)
    A_bound[i] = bd; A_c[i] = c
    A_c_tilde[i] = c_tilde; A_kappa[i] = kappa
# compute bound for B
    bd, _, c, c_tilde, kappa = compute_bound(B,E,r)
    B_bound[i] = bd; B_c[i] = c
    B_c_tilde[i] = c_tilde; B_kappa[i] = kappa

# %% [markdown]
# Recall : $C_r \leq (c + \tilde{c}) \times \sigma_{r+1}(A) + \kappa_r$

# %%
# plot
plt.figure(dpi=125)
plt.semilogy(ranks,A_Cr,'b+-',label='$A: C_r$')
plt.semilogy(ranks,A_bound,'b--',label='A: bound') # uncomment to show the bound
plt.semilogy(ranks,A_kappa,'bo-',label='$A: \kappa_r$')
#plt.semilogy(ranks,s_A[:-1],'xs-',label='sing. val. $A$') # uncomment to show the singular values of A
plt.semilogy(ranks,s_E[:-1],'kx-',label='sing. val. $E$')
plt.grid()
plt.legend()

# %% [markdown]
# Observation : 
# - The bound is not sharp.
# - The most important part seems to be $\kappa_r$ and not $c$ and $\tilde{c}$. (consider $\epsilon$-pseudoinvserse ?)

# %% [markdown]
#

# %%
plt.figure(dpi=125)
plt.semilogy(ranks,B_Cr,'r+-',label='$B: C_r$')
plt.semilogy(ranks,B_bound,'r--',label='B: bound')
plt.semilogy(ranks,B_kappa,'ro-',label='$B: \kappa_r$')
plt.semilogy(ranks,s_E[:-1],'kx-',label='sing. val. $E$')
plt.legend()
plt.grid()

# %% [markdown]
# Observation :
# - The bound is sharp.

# %% [markdown]
# ### 2.2 The constant $\kappa_r$
# Let's inspect $\kappa_r$ more precisely.

# %%
# Generate random A
N = 50
A = np.random.randn(N,N)
(U,s,Vt) = la.svd(A) 

# Generate random E
E = np.random.randn(N,N)
norm_E = la.norm(E,2)
E = E / norm_E # uncomment for normalizing

# variables
s_E = la.svdvals(E)
errors = np.zeros(N)

# compute errors
for r in range(N):
    Ur = U[:,:r]
    errors[r] = la.norm(E - Ur @ Ur.T @ E,2)

# plot
index = np.arange(1,N+1)
plt.figure(dpi=125)
plt.plot(index, s_E, 'k+-', label='$\sigma_r (E)$')
plt.plot(index, errors, 'r+-', label='$||(I-U_r U_r^T)E||_2$')
plt.plot(index, np.ones(N), 'k--', label='$||E||_2$')
plt.legend()
plt.xlabel('initial_rank $r$')
plt.grid()

# %%

# %% [markdown]
# ### Brouillon

# %%
E = np.random.randn(20,50)
s = la.svdvals(E)
plt.plot(s)

X = np.random.randn(50,50)
(U,s,Vt) = np.linalg.svd(X)
r=20
Ur = U[:,r+1:]
E_rest = Ur.T @ E
s_rest = la.svdvals(E_rest)
plt.plot(s_rest)

# %%
all_c_tilde

# %%
A_c_tilde

# %%
np.eye(15-1)

# %%
