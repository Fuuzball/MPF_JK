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
from mpf_local_higher_order import MPF_Glass_HLE
np.set_printoptions(precision=3)
Seed = 0
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

class SLMC_Wolff(object):

    def __init__(self, D, J, K, gibbs_params, wolff_params):
        self.D = D
        self.J = J
        self.K = K
        self.N_g, self.burn_g, self.thin_g = gibbs_params
        self.N_w, self.burn_w, self.thin_w = wolff_params
        JK = np.hstack((J, K)) 
        print('Sample paramters : {}'.format(JK))

    def sample_gibbs(self):
        """
            Using Gibbs sampling, get effective energy and effective coupling E0, J_eff
        """

        self.gibbs = GibbsSampler(self.D, self.J, self.K)
        print('Sampling with local update rules (Gibbs)...')
        self.X_gibbs = self.gibbs.sample_X(self.N_g, self.burn_g, self.thin_g)
        mpf = MPF_Estimator(self.X_gibbs)
        print('Estimated parameters : {}'.format(mpf.learn_jk()))

        C1 = mpf.total_second_corr()[0]
        E = mpf.energy(self.J, self.K)
        a = np.hstack((np.ones(self.N_g)[:, None], -C1[:, None]))
        self.E0, self.J_eff = np.linalg.lstsq(a, E)[0]
        print('Effective parameters -- E0 : {}, J_eff : {}'.format(self.E0, self.J_eff))

    def sample_wolff(self):
        """
            Given J_eff, do wolff sampling to return final X
        """

        self.wolff = EffectiveWolffSampler(self.D, self.J_eff, self.J, self.K)
        self.X_wolff = self.wolff.sample_X(self.N_w, self.burn_w, self.thin_w)
        mpf = MPF_Estimator(self.X_wolff)
        print('Estimated parameters : {}'.format(mpf.learn_jk()))

    def plot_gibbs(self):
        self.gibbs.plot_sample()

    def plot_wolff(self):
        self.wolff.plot_sample()

if __name__ == '__main__':

    D = 10
    J = [.4, 0, 0, 0]
    K = 0.30
    JK = np.hstack((J, K)) 

    N_local = 500
    gibbs_params = (300, 3000, 3000)
    wolff_params = (30, 300, 300)


    if False:
        gibbs = GibbsSampler(D, J, K)
        print('Sample paramters : {}'.format(JK))
        print('Sampling with local update rules (Gibbs)...')
        X_local = gibbs.sample_X(N_local, burn_in, thin)
        mpf = MPF_Estimator(X_local)
        print('Estimated parameters : {}'.format(mpf.learn_jk()))
        C1 = mpf.total_second_corr()[0]
        E = mpf.energy(J, K)
        a = np.hstack((np.ones(N_local)[:, None], -C1[:, None]))
        E0, J_eff = np.linalg.lstsq(a, E)[0]
        print('Effective parameters -- E0 : {}, J_eff : {}'.format(E0, J_eff))
        #gibbs.plot_sample()
        #plt.show()

    J_eff = 1.38

    if True:
        wolff = EffectiveWolffSampler(D, J_eff,J ,K)
        X = wolff.sample_X(60, 300, 300)
        mpf = MPF_Estimator(X)
        print('Estimated parameters : {}'.format(mpf.learn_jk()))
        wolff.plot_sample()
        plt.show()

