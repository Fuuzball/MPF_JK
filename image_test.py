from scipy.io import loadmat
import matplotlib.pylab as plt
import numpy as np
import os

image_arr = np.array(loadmat('./geisler/cps201004281214.B1')['B'])
plt.imshow(image_arr[1000:1400, 1800:2200], cmap='Greys_r')
plt.show()
