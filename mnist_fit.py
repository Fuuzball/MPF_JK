from sklearn.datasets import fetch_mldata
from holi_mpf_est import HOLIGlass
import numpy as np
import matplotlib.pylab as plt
import logging
import pickle

logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)

mnist = fetch_mldata('MNIST original')
X = mnist.data

N_X = X.shape[0]

n_samples = 1000

rng = np.random.RandomState(seed=42)
rand_idx = rng.choice(np.arange(N_X), n_samples, replace=False)

X_sample = X[rand_idx]
print(X_sample.shape)

estimator = HOLIGlass(X_sample, params=['j_1', 'b'])
print(estimator.num_params)
params = estimator.learn(lr=.0010, max_iter=2)
params = estimator.learn(lr=10, max_iter=2)
pickle.dump(params, open('params.p', 'wb'))

