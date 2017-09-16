import numpy as np
import random
from math import exp
from scipy.signal import convolve
from scipy import optimize
import time
import matplotlib.pylab as plt
from gibbs import GibbsSampler
from wolff import WolffSampler
from mpf_ising_jk import MPF_Estimator
np.set_printoptions(precision=3)
Seed = 15
if Seed:
    np.random.seed(Seed)

def sigmoid( x ):
    return 1 / ( 1 + np.exp( -x ) )

class EffectiveWolffSampler(WolffSampler):

    def __init__(self, D, J, J0, K0):
        WolffSampler.__init__(self, D, J)
        self.x = np.ones((D, D))
        self.J0 = J0
        self.K0 = K0

        self.W = self._W()

    def _W(self):
        J = self.J0
        J[0] -= self.J
        a = np.array([[4,1,0,1,4]])
        r2 = a + a.T
        W = np.zeros((5,5))
        W[r2 == 1] = J[0]
        W[r2 == 2] = J[1]
        W[r2 == 4] = J[2]
        W[r2 == 5] = J[3]

        return W
            
    def _H(self):
        H = convolve(self.x, self.W, 'same')
        return H / 2

    def delta_E_eff(self):
        E = -(self.x * self._H()).sum()
        M = np.ones((2,2))
        XM = convolve(self.x, M, 'valid')
        Q = np.ones_like(XM)
        Q[np.abs(XM) == 2] = -1
        E -= self.K0 * Q.sum()
        return E

    def sample_X(self, N, burn_in=None, thin=None, new_J=None, display_time=False): 
        D = self.D
        t0 = time.process_time()
        if new_J:
            self.J = new_J

        if (burn_in is None) or (thin is None):
            print('Using recommended burn in / thin...')
            burn_in_rec, thin_rec = self.get_recommended_burnin_thin()
            if burn_in is None:
                burn_in = burn_in_rec
                print('Burn in : ', burn_in)
            if thin is None:
                thin = thin_rec
                print('Thin : ', thin)


        n_sample_steps = burn_in + N * thin
        X = np.zeros((N, D, D)) 
        self.x = np.random.randint(2, size=(D, D)) * 2 - 1
        rand_p = np.random.random(n_sample_steps)

        reject = 0
        for i in range(n_sample_steps):
            x0 = np.copy(self.x)
            E0 = self.delta_E_eff()
            self.flip_random_cluster() 
            E = self.delta_E_eff()

            if exp(-(E - E0)) < rand_p[i]:
                reject +=1
                self.x = x0


            if i >= burn_in and (i - burn_in) % thin == 0:
                X[(i - burn_in) // thin] = self.x

        self.X = X
        print('rejection fraction', reject / n_sample_steps)

        if display_time:
            print('Sampling took {:.4f}s'.format(time.process_time() - t0))
        return X
D = 10
J = [0.2905666,  0.15444032,  -0.06169692,  -0.05076923]
J = [1.7, 0, 0, 0]
J_eff = 0.7
K = -0.06480178
K = 0
JK = np.hstack((J, K))

gibbs = GibbsSampler(D, J, K)
wolff = EffectiveWolffSampler(D, J_eff, J ,K)
#wolff.sample_X(1, 0, 1)
wolff.sample_X(15, 100, 100)
wolff.plot_sample()
plt.show()

if False:
    N = 100
    print('Sample paramters : {}'.format(JK))
    print('Sampling with local update rules...')
    X_local = gibbs.sample_X(N, 300, 300)
    mpf = MPF_Estimator(X_local)
    print('Estimated parameters : {}'.format(mpf.learn_jk()))
    C1 = mpf.total_second_corr()[0]
    E = mpf.energy(J, K)
    a = np.hstack((np.ones(N)[:, None], -C1[:, None]))
    E0, J_eff = np.linalg.lstsq(a, E)[0]


