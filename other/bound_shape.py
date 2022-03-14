#%% IMPORTATIONS
import numpy as np
import math
import scipy.linalg as la
import matplotlib.pyplot as plt

#%% BOUND DEFINITIONS
def function(alpha, beta, gamma, kappa, n , k):
    if n > k:
        part1 = gamma * alpha**k * np.sum([math.comb(l+k-1, l) * beta ** l for l in range(n-k)])
    else:
        part1 = 0
    part2 = kappa * np.sum([np.sum([math.comb(l+j, l) * alpha**j * beta **l for l in range(n-j)]) for j in range(k)])
    return part1 + part2

def superlinear(alpha, beta, gamma, kappa, n, k):
    return gamma * alpha ** k / math.factorial(k-1) * np.prod([float(n-j) for j in np.arange(2,k+1)]) / (1-beta) + kappa / (1 - alpha - beta)

def linear1(alpha, beta, gamma, kappa, n, k):
    return gamma * alpha**k * 1/(1 - beta)**k + kappa / (1 - alpha - beta)

def linear2(alpha, beta, gamma, kappa, n, k):
    return gamma * alpha**k * (1+ beta)**(n-1) + kappa / (1 - alpha - beta)

#%% COMPUTATIONS
n = 30
k = 30
alpha = 0.49
beta = 0.49
gamma = 1
kappa = 1e-15

fc = np.zeros(n)
suplin = np.zeros(n)
lin1 = np.zeros(n)
lin2 = np.zeros(n)
it = np.arange(1,n+1)
for k in it:
    fc[k-1] = function(alpha, beta, gamma, kappa, n, k)
    suplin[k-1] = superlinear(alpha, beta, gamma, kappa, n, k)
    lin1[k-1] = linear1(alpha, beta, gamma, kappa, n, k)
    lin2[k-1] = linear2(alpha, beta, gamma, kappa, n, k)


#%% PLOT
fig = plt.figure(dpi=100)
plt.rcParams['text.usetex'] = True
plt.semilogy(it, fc, '-1', label='Original equality')
plt.semilogy(it, suplin, '-x', label='Superlinear bound')
plt.semilogy(it, lin1, '-+', label=r'Linear bound $B_{n,k} = \alpha^k (1-\beta)^{-k}$')
plt.semilogy(it, lin2, '-o', label=r'Linear bound $B_{n,k} = \alpha^k (1+\beta)^{n-1}$')
plt.grid()
plt.legend(loc='lower left', fontsize=10)
plt.title(rf'$\alpha={alpha}, \beta={beta}$', fontsize=14)
plt.xlabel('Iteration $k$', fontsize=14)
plt.ylabel('Error $E_n^k$', fontsize=14)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
fig.savefig(f'figures/bounds_alpha={alpha}_beta={beta}_gamma={gamma}_kappa={kappa}.pdf')


# %%
