import numpy as np
from mpf_ising_jk import MPF_Estimator
import time
import matplotlib.pylab as plt
np.random.seed(1)

def stack_X(X, ratio = 1.5, pad = 1):
    N, Dx, Dy = X.shape
    W = int(np.ceil(np.sqrt(ratio * N)))
    H = int(np.ceil(N / W))

    if H * W > N:
        X = np.concatenate((X, np.zeros((H * W - N, Dx, Dy))))

    padX = np.pad(X, ((0,0), (pad,pad), (pad,pad)), 'constant', constant_values = 0.5)
    rows = []
    for i in range(H):
        rows.append(np.hstack((padX[i*W:(i+1)*W])))
    Xstack = np.vstack(rows)
    return Xstack

t0 = time.time()
patches = np.load('./geisler/patches.npz')['arr_0']

means = patches.mean(axis=(1,2))
median_idx = np.nonzero(np.abs(means) < 1)[0]

if False:
    random_idx = np.random.choice(median_idx, 15, replace=False) 
    patches_s = patches[random_idx] * (-1)
    plt.imshow(stack_X(patches_s))
    plt.show()

if True:
    n_trial = 10
    #jk_trials = np.zeros((n_trial, 5))
    jk_trials = []

    for i in range(n_trial):
        print(i)
        random_idx = np.random.choice(median_idx, 1000, replace=False) 
        patches_s = patches[random_idx] * (-1)

        mpf_estimator = MPF_Estimator(patches_s, fit_params=[1,0,0,0,0])
        #jk_trials[i] = mpf_estimator.learn_jk()
        jk_trials.append(mpf_estimator.learn_jk())
        print(jk_trials[i])

    jk_trials = np.array(jk_trials)

    print('----'*10)
    print(jk_trials.mean(axis=0))
    print(np.var(jk_trials, axis=0))
