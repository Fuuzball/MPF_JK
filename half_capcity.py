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
import seaborn as sns

dir_path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(dir_path, 'data', 'random_capacity', 'OPR_d_4_p_03')
arr_path = os.path.join(path, 'frac_min.npy')
img_path = os.path.join(path,'thresholds.png')

frac_min_arr = np.load(arr_path)

threshold_1 = (np.argmax(frac_min_arr < 1., axis=1))
threshold_5 = (np.argmax(frac_min_arr < .5, axis=1))
threshold_01 = (np.argmax(frac_min_arr < .1, axis=1))

fig, ax = plt.subplots(1)

fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

fig.set_size_inches(4, 16)

ax.imshow(frac_min_arr.T, cmap='Greys_r', origin='lower')
#ax.axis('tight')
#ax.axis('off')
ax.plot(threshold_1, 'g-')
ax.plot(threshold_5, 'b-')
ax.plot(threshold_01, 'r-')

plt.savefig(img_path)
