import numpy as np 
import matplotlib.pylab as plt
from ising_sampler import SLMC_sample, stack_X

J = [-0.6905666,  -0.15444032,  0.06169692,  0.05076923]
K = -0.06480178

N = 2 #Number of samples
D = (18, 18) #Dimension
burn_in = int(10 * ( D[0] * D[1] ))
thin = int(10 * ( D[0] * D[1] ))

X = SLMC_sample(J, K, D, N, burn_in, thin)
plt.imshow(stack_X(X))
plt.show()

