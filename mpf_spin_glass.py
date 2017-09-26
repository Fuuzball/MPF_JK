import numpy as np
from scipy.signal import convolve
from scipy import optimize

np.set_printoptions(precision=3)

def get_rand_J(D):
    J = np.random.random((D, D))
    J = 0.5 * (J + J.T)
    J = J - np.diag(np.diagonal(J))
    return J

class MPF_Glass(object):
# Conventions: spins are symmetric: {-1,1}, energy is E = -0.5 x.T @ J @ X
# Conventions: spins are stacked as N x D matrices

    def __init__(self, X, J0=None, b0=None):
        self.X = X
        self.N, self.D = X.shape
        self.J_init = J0
        self.b_init = b0

    def energy(self, J, b):
        x = self.X
        return - 0.5 * (x @ J @ x.T) + x.T @ b

    def reshape_jb(self, Jb):
        D = self.D
        J_flat = Jb[:-D]
        J = J_flat.reshape((self.D, self.D))
        b = Jb[-D:]
        return J, b

    def K_dK(self, Jb):
        x = self.X
        J, b = self.reshape_jb(Jb)

        dE = 2 * x * (x @ J) - 2 * x * b[None, :]

        Knd = np.exp(-0.5 * dE)
        K = Knd.sum()

        dKdJ = -0.5 * (x.T @ (x * Knd) + (x * Knd).T @ x)
        dKdJ -= np.diag(np.diagonal(dKdJ))
        dKdb = (x * Knd).sum(axis=0)
        dK = np.hstack(
                (dKdJ.reshape(-1), dKdb)
                )
        return K, dK

    def learn_jb(self):
        Jb = np.zeros(self.D * (self.D + 1))

        min_out = optimize.fmin_l_bfgs_b(self.K_dK, Jb)
        estimate = min_out[0]

        return self.reshape_jb(estimate)

if __name__ == '__main__':
    D = 5 
    N = 100
    J = get_rand_J(D)
    b = np.random.random(D)
    #b = np.zeros_like(b)

    X = np.random.randint(2, size=(N, D)) * 2 - 1
    mpf = MPF_Glass(X)

    eps = 1E-3
    print(mpf.learn_jk())


