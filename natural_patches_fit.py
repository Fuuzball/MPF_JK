import numpy as np
from mpf_ising_jk import MPF_Estimator
import time
import matplotlib.pylab as plt

t0 = time.time()
patches = np.load('./geisler/patches.npz')['arr_0']
print(time.time() - t0)



patches_s = patches[
        np.random.choice(patches.shape[0], 10000, replace=False)
        ]

if False:
    mpf_estimator = MPF_Estimator(patches_s)
    JK = mpf_estimator.learn_jk()
    print(JK)

