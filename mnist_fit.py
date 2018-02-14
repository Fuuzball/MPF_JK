from sklearn.datasets import fetch_mldata
from mpf_local_higher_order import HOLIGlass
import numpy as np
import matplotlib.pylab as plt
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)

mnist = fetch_mldata('MNIST original')
X = mnist.data

N_X = X.shape[0]

n_samples = 10

rng = np.random.RandomState(seed=42)
rand_idx = rng.choice(np.arange(N_X), n_samples, replace=False)

X_sample = X[rand_idx]
print(X_sample.shape)


estimator = HOLIGlass(X_sample, params=['J_glass', 'b'])
params = estimator.learn(disp=True)
print(params)
