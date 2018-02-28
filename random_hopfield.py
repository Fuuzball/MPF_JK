from gibbs_sampler import GibbsSampler
from holi_mpf_est import HOLIGlass
import numpy as np
import torch
import logging
import random
import time
import matplotlib.pylab as plt
from sklearn.datasets import fetch_mldata
import os

rng = np.random.RandomState(1)

D_list = np.arange(10, 110, 10)
N_list = np.arange(10, 160, 10)
frac_min_arr = np.zeros((len(D_list), len(N_list)))


time_stamp = str(int(time.time()))
dir_path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(dir_path, 'data', 'random_capacity', str(time_stamp))
os.mkdir(path)

meta_data_path = os.path.join(path, 'meta_data')
npy_arr_path = os.path.join(path, 'frac_min.npy')
plot_path = os.path.join(path, 'min_frac_plot.png')

with open(meta_data_path, 'w') as f:
    f.write(f'D: {D_list}\n')
    f.write(f'N: {N_list}')

for d_i, D in enumerate(D_list):
    for n_i, N in enumerate(N_list):
        print(D, N)

        X = rng.randint(0, 2, size=(N, D)) * 2 - 1

        model = HOLIGlass(X, M=[], params=['J_glass'])
        
        #theta = model.learn(unflatten=False, params=[{'lr' : .5, 'max_iter' : 100}])
        theta = (X.T @ X).reshape(-1)
        dE = model.get_dE(theta).data.numpy()
        frac_min = (dE > 0).all(axis=1).sum()/N
        frac_min_arr[d_i, n_i] = frac_min

np.save(npy_arr_path, frac_min_arr)

plt.imshow(frac_min_arr.T, cmap='Greys_r', origin='lower')
plt.savefig(plot_path)
