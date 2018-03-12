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

    def __init__(self, theta, model, shape_2d=None):
        self.logger = logger
        self.logger.info('Initializing HopfieldNetwork')
        self.theta = theta
        self.D = model.D
        self.model = model
        if shape_2d == None:
            W = int(np.sqrt(self.D))
            H = int(self.D / W)
        else:
            H, W = shape
        self.H = H
        self.W = W
        self.shape_2d = (H, W)

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
        return self.model.X.data.numpy()[0]

    def plot_sample(self, fname=None, ratio = 1.5, W = None, pad = 1):

        print((-1, *self.shape_2d))
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

    #D = 20**2
    N = 2000
    #X = rng.randint(2, size=(1, D)) * 2 - 1
    X = (X_mnist[:N] > 20) * 1.
    def sample_mnist(n_samples, thres=20):
        N_X = X_mnist.shape[0]
        rand_idx = rng.choice(np.arange(N_X), n_samples, replace=False)
        return (X_mnist[rand_idx] > thres) * 2 -1
    X = X * 2 - 1
    X = sample_mnist(N)

    model = HOLIGlass(X, M=[], params=['J_glass'], use_cuda=True)
    theta = (X.T @ X).flatten()
    dE_model = model.get_dE(theta, to_numpy=True)
    J = X.T @ X
    J -= np.diag(np.diagonal(J))
    dE = -2 * X @ (J)
    print(dE_model)
    print(dE)
    print(np.where(dE_model != dE))


    assert False


    
    #model = HOLIGlass(X, params=fit_params, M=[], use_cuda=False)
    model = HOLIGlass(X, M=[], params=['J_glass'], use_cuda=True)
    print(model.X)
    #theta = model.learn(unflatten=False)
    def f(prop):
        print (prop['n_iter'], prop['abs_grad_sum'], prop['loss'])

    params = [
            {'call_back' :f, 'history_size' : 20,  'lr' : 1, 'max_iter': 100}
            ,{'call_back' :f, 'history_size' : 20,  'lr' : .1, 'max_iter': 200}
            #,{'call_back' :f, 'history_size' : 1000,  'lr' : 1, 'max_iter': 20}
            ]
    theta = model.learn(unflatten=False, theta0=1E-4, params=params)
    dE = (model.get_dE(theta, to_numpy=True))
    print(theta)
    print(dE)
    print((dE > 0).all(axis=1).sum()/N)


    #noise = np.random.binomial(1, .1, size=X.shape).astype(np.bool)
    #hopfield = HopfieldNetwork(theta, model)
    #hopfield.run_model(X[0][None, :])
    #plt.imshow((dE < 0).sum(axis= 0).reshape((28, 28)))
    #plt.show()
