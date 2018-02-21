from sklearn.datasets import fetch_mldata
from holi_mpf_est import HOLIGlass
import numpy as np
import matplotlib.pylab as plt
import logging
import pickle
from torch import optim

logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)

mnist = fetch_mldata('MNIST original')
X = mnist.data

N_X = X.shape[0]

n_samples = 1

rng = np.random.RandomState(seed=42)
rand_idx = rng.choice(np.arange(N_X), n_samples, replace=False)

X_sample = X[rand_idx]
print(X_sample.shape)

estimator = HOLIGlass(X_sample, params=['j_1', 'b'])
print(estimator.num_params)
params = estimator.learn(lr=1, max_iter=100)
print(params)
pickle.dump(params, open('params.p', 'wb'))

