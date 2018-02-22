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

n_samples = 1000

M0 = np.ones((2, 2))
M1 = np.ones((2, 2))
M2 = np.ones((2, 2))
M3 = np.ones((2, 2))

M0[0, 0] = 0
M1[0, 1] = 0
M2[1, 0] = 0
M3[1, 1] = 0

M = [M0, M1, M2, M3]

if n_samples:
    rng = np.random.RandomState(seed=42)
    rand_idx = rng.choice(np.arange(N_X), n_samples, replace=False)
    X_sample = X[rand_idx]
else:
    X_sample = X
print(X_sample.shape)

estimator = HOLIGlass(X_sample, params=['j_1', 'b'], M=M)

def f(prop):
    print (prop['n_iter'], prop['abs_grad_sum'], prop['loss'])

est_params_lr_5 = {
        'lr' : 1,
        'max_iter' : 50,
        'tolerance_grad' : 1,
        'tolerance_change' : 1E-5,
        'call_back' : f
        }

est_params = {
        'lr' : .5,
        'max_iter' : 100,
        'tolerance_grad' : 1E-9,
        'tolerance_change' : 1E-18,
        'call_back' : f
        }

est_params_list = [est_params_lr_5, est_params]
est_params_list = [est_params_lr_5]
params = estimator.learn(params=est_params_list)
#print(params)
plt.subplot(221)
plt.imshow(params['k_0'])
plt.subplot(222)
plt.imshow(params['k_1'])
plt.subplot(223)
plt.imshow(params['k_2'])
plt.subplot(224)
plt.imshow(params['k_3'])
print(params)
plt.show()
#pickle.dump(params, open('params.p', 'wb'))

