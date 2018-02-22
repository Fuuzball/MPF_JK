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

n_samples = 10

rng = np.random.RandomState(seed=42)
rand_idx = rng.choice(np.arange(N_X), n_samples, replace=False)

X_sample = X[rand_idx]
print(X_sample.shape)

estimator = HOLIGlass(X_sample, params=['j_1', 'b'])
tol_grad = 1E-9
tol_change = tol_grad**2
params = estimator.learn(lr=.01, max_iter=100, history_size=1000, tolerance_grad=tol_grad, tolerance_change=tol_change)
params = estimator.learn_scipy()
plt.subplot(211)
plt.imshow(params['k_0'])
plt.subplot(212)
plt.imshow(params['b'].reshape((28, 28)))
plt.show()
#print(params)
#print(estimator.learn_scipy())
#pickle.dump(params, open('params.p', 'wb'))

