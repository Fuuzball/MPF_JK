import numpy as np
import random
from math import exp
from scipy.signal import convolve
from scipy import optimize
import time
import matplotlib.pylab as plt
import torch
from torch.autograd import Variable
import logging
import matplotlib.pylab as plt
from holi_mpf_est import HOLIGlass

logger = logging.getLogger(__name__)

def torch_double_var(npy_arry, grad=False):
    return Variable(torch.from_numpy(npy_arry).double(), requires_grad=grad)

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

    def __init__(self, D, theta, estimator, shape_2d=None):
        self.logger = logger
        self.logger.info('Initializing GibbsSampler')
        self.theta = theta
        self.D = D
        if shape_2d is None:
            W = int(np.sqrt(self.D))
            H = int(self.D / W)
            shape_2d = H, W
        self.shape_2d = shape_2d
        self.estimator = estimator

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
        self.estimator.X = x[None, :]

        for n in range(n_sample_steps):
            i = rand_d[n]
            dE = self.estimator.get_dE(self.theta)[0, i]
            self.logger.debug('-'*20 + 'dE' + '-'*20 + '\n{}'.format(dE))
            p = exp(-dE)
            if p > rand_p[n]:
                x_2d = (self.estimator.X_2d.data.numpy())
                i_x = i // self.D**.5
                i_y = i % self.D**.5
                x[i] *= -1
                self.estimator.X = x[None, :]
            
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
        else:
            plt.imsave(fname, X_stacked, format='png', cmap='Greys')

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y:%H:%M:%S',
    level=logging.INFO)
    logging.getLogger('torch_lbfgs.py').setLevel(logging.DEBUG)
    D = 4**2

    rng = np.random.RandomState(seed=10)

    X = rng.randint(2, size=(1, D)) * 2 - 1
    params = ['J_glass', 'b']
    estimator = HOLIGlass(X, params=params, M=[])
    params = estimator.get_random_params()
    theta = estimator.flatten_params(params)

    sampler = GibbsSampler(D, theta, estimator)
    N = 2000
    burn_in = 1000
    thin = 200
    X = sampler.sample_X(N, burn_in, thin)
    estimator = HOLIGlass(X, params=params, M=[])
    params_est = estimator.learn()
    plt.imshow(params['J_glass'].data.numpy())
    plt.figure()
    plt.imshow(params_est['J_glass'])
    plt.show()
