import numpy as np
import random

import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F

from mpf_local_higher_order import MPF_Glass, MPF_Glass_HLE

import time

class GibbsSampler(object):
    """
        Performs Gibbs sampling with transitions between spins one hamming distance apart

        Args:
           D : Dimension of spins to be sampled
           theta : Parameters of sampler
           model : The model of sampler. Must have a method model.get_dE(theta)
    """

    def __init__(self, D, theta, model):
        self.D = D
        self.theta = theta
        self.model = model
        
    def get_rand_spins(self):
        p = torch.ones(self.D)
        return p.random_(2) * 2 - 1

    def sample_X(self, N, burn_in, thin, display_time=False):
        t0 = time.process_time()

        n_sample_steps = burn_in + N * thin

        # Generate random dimensions and probs for later use
        steps_tensor = torch.ones(n_sample_steps)
        rand_d = steps_tensor.random_(self.D)
        rand_p = steps_tensor.uniform_()

        # Initialize tensor to store all samples
        X = torch.zeros((N, self.D))

        # Initialize first sample
        x = self.get_rand_spins()
        x = np.random.randint(2, size=(1, self.D)) * 2 - 1


        #for n in range(n_sample_steps):
        #    d = rand_d[n]
        print(self.model(x))
        print(self.model(x).get_dE())



gibbs = GibbsSampler(10, None, MPF_Glass)
gibbs.sample_X(10, 10, 10)
