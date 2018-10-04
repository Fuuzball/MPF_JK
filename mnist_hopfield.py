from gibbs_sampler import GibbsSampler
from holi_mpf_est import HOLIGlass
import numpy as np
import torch
import logging
import random
import time
import logging
import matplotlib.pylab as plt
from sklearn.datasets import fetch_mldata
import os

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
datefmt='%d-%m-%Y:%H:%M:%S',
level=logging.WARNING)
logging.getLogger('torch_lbfgs.py').setLevel(logging.DEBUG)

rng = np.random.RandomState(10)

mnist = fetch_mldata('MNIST original')
X_mnist = mnist.data

N_X = X_mnist.shape[0]

def sample_mnist(n_samples, thres=20):
    rand_idx = rng.choice(np.arange(N_X), n_samples, replace=False)
    return (X_mnist[rand_idx] > thres) * 2 -1


N_list = np.arange(1000, 3000, 10)
#N_list = np.arange(1, 101, 1)
#method = 'OPR_2'
#method = 'OPR_ho_4'
#N_list = [2000]
method = 'MPF_HOLI_4'
method = 'MPF_glass'
frac_min_arr = np.zeros_like(N_list, dtype=np.float)

p = .17
D = 28**2


time_stamp = str(int(time.time()))
dir_path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(dir_path, 'data', 'mnist_capacity', str(time_stamp))
os.mkdir(path)

def f(prop):
    print (prop['n_iter'], prop['abs_grad_sum'], prop['loss'])

def get_ho_OPR_dE(d, X):
    d = 4
    dE_nmi = ((X@X.T)**d)[:, :, None] - ((X@X.T)[:, :, None] - 2 * X[:, None, :] * X[None, :, :])**d
    return dE_nmi.sum(axis=0)

meta_data_path = os.path.join(path, 'meta_data')
npy_arr_path = os.path.join(path, 'frac_min.npy')
plot_path = os.path.join(path, 'min_frac_plot.png')


with open(meta_data_path, 'w') as f:
    f.write(f'N: {N_list}\n')
    f.write(f'method: {method}')


for n_i, N in enumerate(N_list):
    print(N)

    #X = rng.binomial(1, p, size=(N, D)) * 2 - 1
    X = sample_mnist(N)

    n_attempts = 3

    for i in range(n_attempts):
        try: 
            if method == 'MPF_glass':
                model = HOLIGlass(X, M=[], params=['J_glass', 'b'], use_cuda=True)
                theta = model.learn(unflatten=False, theta0=1E-2, params=[{'lr' : 1, 'max_iter' : 100}])
            elif method == 'MPF_HOLI_4':
                model = HOLIGlass(X, params=['J_glass', 'b'], use_cuda=True)
                theta = model.learn(unflatten=False, theta0=1E-2, params=[{'lr' : 1, 'max_iter' : 100}])
        except KeyboardInterrupt:
            raise
        except:
            logger.warning(f'Attempt {i+1} at fitting failed, trying again with different initialization...')
        else:
            if method == 'OPR_ho_4':
                dE = get_ho_OPR_dE(4, X)
            elif method =='OPR_2':
                J = X.T @ X
                J -= np.diag(np.diagonal(J))
                dE = 2 * X @ (J)
            else:
                dE = model.get_dE(theta, to_numpy=True)

            frac_min = (dE > 0).all(axis=1).sum()/N
            frac_min_arr[n_i] = frac_min
            print(frac_min)
            break
    else:
        logger.warning(f'All attempts have failed, moving on to next set of parameters recording result as -1')
        frac_min_arr[n_i] = -1

np.save(npy_arr_path, frac_min_arr)

print(frac_min_arr)
plt.plot(N_list, frac_min_arr)
plt.savefig(plot_path)
