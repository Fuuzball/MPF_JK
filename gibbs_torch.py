import numpy as np
import random
from math import exp
from scipy.signal import convolve
from scipy import optimize
import time
import matplotlib.pylab as plt
import torch
from torch.autograd import Variable
from mpf_local_higher_order import MPF_Glass_HOLI, torch_double_var
import logging

logger = logging.getLogger(__name__)


def stack_X(X, ratio = 1.5, W = None, pad = 1):
    N, Dx, Dy = X.shape

    if W:
        H = int(np.ceil(N / W))
    else:
        W = int(np.ceil(np.sqrt(ratio * N)))
        H = int(np.ceil(N / W))

    if H * W > N:
        X = np.concatenate((X, np.zeros((H * W - N, Dx, Dy))))

    padX = np.pad(X, ((0,0), (pad,pad), (pad,pad)), 'constant', constant_values = 0.5)
    rows = []
    for i in range(H):
        rows.append(np.hstack((padX[i*W:(i+1)*W])))
    Xstack = np.vstack(rows)
    return Xstack

class GibbsSampler(object):

    def __init__(self, D, theta, EstClass, shape_2d=None):
        self.logger = logger
        self.logger.info('Initializing GibbsSampler')
        self.theta = theta
        self.D = D
        if shape_2d is None:
            W = int(np.sqrt(self.D))
            H = int(self.D / W)
            shape_2d = H, W
        self.shape_2d = shape_2d
        self.EstClass = EstClass

    def get_rand_spins(self):
        return np.random.randint(2, size=self.D) * 2 - 1

    def sample_X(self, N, burn_in, thin): 
        self.logger.info('Start sampling with N={}, burn_in={}, thin={}'.format(N, burn_in, thin))
        t0 = time.process_time()
        n_sample_steps = burn_in + N * thin
        
        rand_d = np.random.randint(self.D, size=n_sample_steps)
        rand_p = np.random.random(n_sample_steps)
        X = np.zeros((N, self.D)) 
        x = self.get_rand_spins()
        estimator = self.EstClass(x[None, :])

        for n in range(n_sample_steps):
            i = rand_d[n]
            dE = estimator.get_dE(self.theta)[0, i]
            self.logger.debug('-'*20 + 'dE' + '-'*20 + '\n{}'.format(dE))
            p = exp(-dE)
            if p > rand_p[n]:
                x[i] *= -1

            estimator.X = torch_double_var(x[None, :], False)
            
            if n >= burn_in and (n - burn_in) % thin == 0:
                k = (n - burn_in) // thin
                self.logger.info('Sample {} out of {}'.format(k, N))
                X[k] = x

        self.X = X

        self.logger.info('Sampling took {:.4f}s'.format(time.process_time() - t0))
        return X

    def plot_sample(self, fname=None, ratio = 1.5, W = None, pad = 1):

        X_stacked = stack_X(self.X.reshape((-1, *self.shape_2d)), ratio, W, pad)
        fig = plt.imshow(X_stacked, cmap='Greys_r')
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        if fname is None:
            logger.info('Showing plot...')
            plt.show()
        else:
            plt.imsave(fname, X_stacked, format='png', cmap='Greys')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    D = 10
    x = np.random.randint(2, size=(1, D**2)) * 2 - 1

    J = np.random.normal(size=(D**2, D**2))
    b = np.zeros(D**2)
    K = np.random.normal(size=(D-1, D-1))

    J = torch_double_var(J, False)
    b = torch_double_var(b, False)
    K = torch_double_var(K, False)

    estimator = MPF_Glass_HOLI(x)
    theta = (estimator.flatten_params(J, b, K))

    sampler = GibbsSampler(D**2, theta, MPF_Glass_HOLI)
    sampler.sample_X(10, 100, 500)
    sampler.plot_sample()

