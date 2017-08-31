import numpy as np
import matplotlib.pylab as plt
from scipy.signal import convolve

# Conventions: spins are symmetric: {-1,1}, J has vanishing diagonals, energy is E = 0.5 x.T @ J @ X

np.set_printoptions( precision = 3 )
epsilon = 1E-10
Seed = 15
if Seed:
    np.random.seed(Seed)

def get_rand_X(D, N = 1):
    if isinstance(D, int):
        Dx = Dy = D
    else:
        Dx, Dy = D

    X = np.floor(np.random.random( (N, Dx, Dy) ) * 2)
    return 2*X - 1

def sigmoid( x ):
    return 1 / ( 1 + np.exp( -x ) )

class JKIsingModel(object):

    def __init__(self, X, J = (0, 0, 0, 0), K = 0):
        self.X = X
        self.J = J
        self.K = K

        self.n_samples, self.D1, self.D2 = X.shape

    def get_coupling_matrix(self, J):
        """
        For given 1-4th NN coupling strength J, return 5 x 5 matrix to be convolved with spins
        """

        a = np.array([[4,1,0,1,4]])
        r2 = a + a.T
        W = np.zeros((5,5))
        W[r2 == 1] = J[0]
        W[r2 == 2] = J[1]
        W[r2 == 4] = J[2]
        W[r2 == 5] = J[3]
        return W

    def site_nn_correlation(self, W):
        """
        Get spin-spin correlation per site for given coupling matrix W
        """

        corr_mat = np.zeros_like(self.X)

        for i, x in enumerate(self.X):
            corr_mat[i] = convolve(x, W,  mode='same')
        
        return corr_mat * self.X

    def total_correlation(self):
        """
        Get 1-4th NN spin-spin correlation of the entire system
        """

        total_corr = np.zeros((self.n_samples, 4))
        for n in range(4):
            j = [0, 0, 0, 0]
            j[n] = 1
            W = self.get_coupling_matrix(j)
            C = self.site_nn_correlation(W)
            total_corr[:, n] = C.sum(axis=(1,2))

        return total_corr / 2

    def fourth_order_interaction(self):
        """
        Return the energy due to fourth order interaction
        """

        E4 = np.zeros(self.n_samples)
        M = np.ones((2,2))
        for n in range(self.n_samples):
            XM = convolve(self.X[n], M, 'valid')
            Q = np.ones_like(XM)
            Q[np.abs(XM) == 2] = -1
            E4[n] = Q.sum()

        return E4

    def energy(self):
        W = self.get_coupling_matrix(self.J)
        C = self.site_nn_correlation(W)
        # Energy due to second order interactions
        E = C.sum(axis=(1,2)) / 2 
        # Energy due to fourth order interactions
        E += self. K * self.fourth_order_interaction()

        return E

    def propose_flip(self, dx, dy, p):
        assert dx < self.D1
        assert dy < self.D2

        E0 = self.energy()
        self.X[:, dx, dy] *= -1
        dE = self.energy() - E0

        # Reject proposal if fails check
        fails = sigmoid(-dE) < p
        self.X[fails, dx, dy] *= -1


def local_sampling(D, N, J, K, burn_in, thin):
    n_sample_steps = burn_in + N * thin

    if isinstance(D, int):
        Dx = Dy = D
    else:
        Dx, Dy = D

    pRand = np.random.random(n_sample_steps)
    dxRand = np.random.randint(Dx, size = (n_sample_steps))
    dyRand = np.random.randint(Dy, size = (n_sample_steps))

    X = np.zeros((N, Dx, Dy))
    x = get_rand_X(D)
    S = JKIsingModel(x, J, K)
    S_next = JKIsingModel(x, J, K)

    for i in range(n_sample_steps):
        p = pRand[i]
        dx = dxRand[i]
        dy = dyRand[i]
        S.propose_flip(dx, dy, p)

        if i >= burn_in and (i - burn_in) % thin == 0:
            X[(i - burn_in) // thin ] = S.X[0]

    return X


X = get_rand_X(3, 5)
print(X)
jk_model = JKIsingModel(X, J=(1,0,0,0), K=1)

N = 60 #Number of samples
D = (5, 5) #Dimension
burn_in = 100 * D[0] * D[1]
thin = 10 * D[0] * D[1]
J = (1, 0, 0, 0)
K = 1

X_sampled = local_sampling(D, N, J, K, burn_in, thin)
:w
