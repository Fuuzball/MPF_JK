import numpy as np
import time
rng = np.random.RandomState(1)

N = 100
D = 50
p = .5

X = rng.binomial(1, p, size=(N, D)) * 2 - 1


d = 4
dE_nmi = ((X@X.T)**d)[:, :, None] - ((X@X.T)[:, :, None] - 2 * X[:, None, :] * X[None, :, :])**d
dE = dE_nmi.sum(axis=0)
