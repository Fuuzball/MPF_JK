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
        return - 0.5 * np.diagonal(x @ J @ x.T) + x @ b

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

class MPF_Glass_JK(MPF_Glass):

    def __init__(self, X, shape=None):
        super().__init__(X)

        if shape == None:
            W = int(np.sqrt(self.D))
            H = int(self.D / W)
            self.shape=(W, H)
        self.X_2d = X.reshape((self.N, *self.shape))

        self.M = np.ones((2,2))
        self.Q = []
        for n, x in enumerate(self.X_2d):
            XM = convolve(x, self.M, 'valid')
            Q = np.ones_like(XM)
            Q[np.abs(XM) == 2] = -1
            self.Q.append(Q)
        self.Q = np.array(self.Q)
        
    def dE4(self, K):
        dE = np.zeros_like(self.X_2d, dtype=float)
        for n in range(self.N):
            dE[n] = -2 * convolve(self.Q[n] * K, self.M, 'full')
        return dE
        
    def reshape_JbK(self, params):
        i_J = self.D**2
        i_b = i_J + self.D
        i_K = i_b + (self.shape[0] - 1) * (self.shape[1] - 1)
        J, b, K, _ =  np.split(params, (i_J, i_b, i_K))
        J = J.reshape((self.D, self.D))
        K = K.reshape((self.shape[0] - 1, self.shape[1] - 1))
        return J, b, K

    def K_dK(self, JbK):
        x = self.X
        J, b, K = self.reshape_JbK(JbK)

        dE = 2 * x * (x @ J) - 2 * x * b[None, :] # From J, b
        dE += self.dE4(K).reshape((self.N, -1))

        Knd = np.exp(-0.5 * dE)
        K = Knd.sum()

        dKdJ = -0.5 * (x.T @ (x * Knd) + (x * Knd).T @ x)
        dKdJ -= np.diag(np.diagonal(dKdJ))
        dKdb = (x * Knd).sum(axis=0)
        dKdK = 0
        for n, k in enumerate(Knd):
            k_2d = k.reshape(self.shape)
            dKdK += (self.Q[n] * convolve(k_2d, self.M, 'valid')).reshape(-1)
        
        dK = np.hstack(
                (dKdJ.reshape(-1), dKdb, dKdK.reshape(-1))
                )
        return K, dK

    def learn_jbk(self):
        Jbk = np.zeros(self.D * (self.D + 1) + (self.shape[0] - 1) * (self.shape[1] - 1))

        min_out = optimize.fmin_l_bfgs_b(self.K_dK, Jbk)
        estimate = min_out[0]

        return self.reshape_JbK(estimate)

if __name__ == '__main__':
    d = 4
    D = d**2
    N = 2
    J = get_rand_J(D)
    b = np.random.random(D)
    #b = np.zeros_like(b)
    np.random.seed(15)

    X = np.random.randint(2, size=(N, D)) * 2 - 1
    K = np.random.random((d-1,d-1)) - 0.5
    Kp = K.copy()
    eps = 0.0000001
    Kp[0,0] += eps

    mpf = MPF_Glass_JK(X)
    JbK = np.hstack((
        J.reshape(-1),
        b,
        K.reshape(-1)
        ))

    JbKp = np.hstack((
        J.reshape(-1),
        b,
        Kp.reshape(-1)
        ))

    Kmpf = mpf.K_dK(JbK)[0]
    Kmpfp = mpf.K_dK(JbKp)[0]
    print((Kmpfp - Kmpf)/eps)


    dkdj, dkdb, dkdk = mpf.reshape_JbK(mpf.K_dK(JbK)[1])
    print(mpf.learn_jbk()[2])


