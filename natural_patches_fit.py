import numpy as np
from mpf_ising_jk import MPF_Estimator
import time
import matplotlib.pylab as plt

t0 = time.time()
patches = np.load('./geisler/patches.npz')['arr_0']

means = patches.mean(axis=(1,2))
median_idx = np.nonzero(np.abs(means) < 0.1)[0]

n_trial = 10
jk_trials = np.zeros((n_trial, 5))

for i in range(n_trial):
    print(i)
    random_idx = np.random.choice(median_idx, 10000, replace=False) 
    patches_s = patches[random_idx]

    mpf_estimator = MPF_Estimator(patches_s)
    jk_trials[i] = mpf_estimator.learn_jk()
    print(jk_trials[i])

print('----'*10)
print(jk_trials.mean(axis=0))
print(np.var(jk_trials, axis=0))
