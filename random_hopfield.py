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

rng = np.random.RandomState(1)

D_list = np.arange(1, 101)
N_list = np.arange(1, 161)
p = .3
frac_min_arr = np.zeros((len(D_list), len(N_list)))


time_stamp = str(int(time.time()))
dir_path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(dir_path, 'data', 'random_capacity', str(time_stamp))
os.mkdir(path)

def f(prop):
    print (prop['n_iter'], prop['abs_grad_sum'], prop['loss'])


meta_data_path = os.path.join(path, 'meta_data')
npy_arr_path = os.path.join(path, 'frac_min.npy')
plot_path = os.path.join(path, 'min_frac_plot.png')

with open(meta_data_path, 'w') as f:
    f.write(f'D: {D_list}\n')
    f.write(f'N: {N_list}\n')
    f.write(f'p :{p}')

for d_i, D in enumerate(D_list):
    for n_i, N in enumerate(N_list):
        print(D, N)

        #X = rng.randint(0, 2, size=(N, D)) * 2 - 1
        X = rng.binomial(1, p, size=(N, D)) * 2 - 1
        model = HOLIGlass(X, M=[], params=['J_glass'], use_cuda=False)

        n_attempts = 3

        for i in range(n_attempts):
            try: 
                theta = model.learn(unflatten=False, theta0=1E-2, params=[{'lr' : 1, 'max_iter' : 100}])
                #theta = (X.T @ X).reshape(-1)
            except KeyboardInterrupt:
                raise
            except:
                logger.warning(f'Attempt {i} at fitting failed, trying again with different initialization...')
            else:
                dE = model.get_dE(theta, to_numpy=True)
                frac_min = (dE > 0).all(axis=1).sum()/N
                frac_min_arr[d_i, n_i] = frac_min
                break
        else:
            logger.warning(f'All attempts have failed, moving on to next set of parameters recording result as -1')
            frac_min_arr[d_i, n_i] = -1



np.save(npy_arr_path, frac_min_arr)

plt.imshow(frac_min_arr.T, cmap='Greys_r', origin='lower')
plt.savefig(plot_path)
