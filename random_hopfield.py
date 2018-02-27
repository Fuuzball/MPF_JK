from gibbs_sampler import GibbsSampler
from holi_mpf_est import HOLIGlass
import numpy as np
import torch
import logging
import random
import time
import matplotlib.pylab as plt
from sklearn.datasets import fetch_mldata

rng = np.random.RandomState(0)

D_list = np.arange(10, 110, 10)
N_list = np.arange(10, 160, 10)
frac_min_arr = np.zeros((len(D_list), len(N_list)))

f_name = './frac_min_arr_mpf.npy'
try:
    frac_min_arr = np.load(f_name)
except:
    for d_i, D in enumerate(D_list):
        for n_i, N in enumerate(N_list):

            X = rng.randint(0, 2, size=(N, D)) * 2 - 1

            model = HOLIGlass(X, M=[], params=['J_glass'])
            theta = model.learn(False)
            #theta = (X.T @ X).reshape(-1)
            dE = model.get_dE(theta).data.numpy()
            frac_min = (dE > 0).all(axis=1).sum()/N
            frac_min_arr[d_i, n_i] = frac_min
            print(D, N, frac_min)

np.save(f_name, frac_min_arr)

plt.imshow(frac_min_arr.T, cmap='Greys_r', origin='lower')
plt.show()
