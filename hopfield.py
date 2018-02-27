from gibbs_sampler import GibbsSampler
from holi_mpf_est import HOLIGlass
import numpy as np
import torch
import logging
import random
import time
import matplotlib.pylab as plt
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
X_mnist = mnist.data

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

class HopfieldNetwork(object):

    def __init__(self, D, theta, model):
        self.logger = logger
        self.logger.info('Initializing HopfieldNetwork')
        self.theta = theta
        self.D = D
        self.model = model
        self.shape_2d = model.shape_2d

    def is_local_min(self, X):
        self.model.X = X
        dE = self.model.get_dE(self.theta)
        return (dE >= 0).byte().all()

    def run_model(self, X, history=False): 
        t0 = time.process_time()
        self.model.X = X.copy()
        dE = self.model.get_dE(self.theta)
        is_min = (dE >= 0).byte().all()

        n_iter = 0
        self.X_history = [X.copy()[0]]
        while not is_min:
            n_iter += 1
            i = int((dE < 0).float().multinomial()) #Get random bit that decreases energy when flipped
            X[:, i] *= -1
            self.model.X = X
            dE = self.model.get_dE(self.theta)
            is_min = (dE >= 0).byte().all()
            self.X_history.append(X.copy()[0])

        self.X_history = np.array(self.X_history)

        self.logger.info('Hopfield network converged in {:.4f}s after {} iterations'.format(time.process_time() - t0, n_iter))
        return self.model.X_2d.data.numpy()[0]

    def plot_sample(self, fname=None, ratio = 1.5, W = None, pad = 1):

        X_stacked = stack_X(self.X_history.reshape((-1, *self.shape_2d)), ratio, W, pad)
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

    rng = np.random.RandomState(seed=10)

    D = 20**2
    X = rng.randint(2, size=(1, D)) * 2 - 1
    X = (X_mnist[0] > 10) * 1.
    X = X[None, :] * 2 - 1


    
    fit_params = ['J_glass', 'b']
    model = HOLIGlass(X, params=fit_params, M=[])
    params = model.get_random_params()
    theta = model.flatten_params(params)
    theta = model.learn(unflatten=False)

    noise = np.random.binomial(1, .1, size=X.shape).astype(np.bool)
    X_flip = X.copy()
    X_flip[noise] *= -1

    hopfield = HopfieldNetwork(D, theta, model)
    print(hopfield.is_local_min(X_flip))
    print(hopfield.is_local_min(X))
