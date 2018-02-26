from gibbs_sampler import GibbsSampler
from holi_mpf_est import HOLIGlass
import numpy as np
import torch
import logging
import random
import time

logger = logging.getLogger(__name__)
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
        self.model.X = X
        dE = self.model.get_dE(self.theta)
        is_min = (dE >= 0).byte().all()

        n_iter = 0
        while not is_min:
            n_iter += 1
            i = int((dE < 0).float().multinomial()) #Get random bit that decreases energy when flipped
            X[:, i] *= -1
            self.model.X = X
            dE = self.model.get_dE(self.theta)
            is_min = (dE >= 0).byte().all()


        self.logger.info('Hopfield network converged in {:.4f}s after {} iterations'.format(time.process_time() - t0, n_iter))
        return self.model.X_2d.data.numpy()[0]


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y:%H:%M:%S',
    level=logging.INFO)
    logging.getLogger('torch_lbfgs.py').setLevel(logging.DEBUG)

    rng = np.random.RandomState(seed=10)

    D = 4**2
    X = rng.randint(2, size=(1, D)) * 2 - 1
    params = ['J_glass', 'b']
    model = HOLIGlass(X, params=params, M=[])
    params = model.get_random_params()
    theta = model.flatten_params(params)

    hopfield = HopfieldNetwork(D, theta, model)
    X_min = hopfield.run_model(X)
    print(X_min)
