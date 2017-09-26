import numpy as np
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original') 
import matplotlib.pylab as plt
from mpf_spin_glass import MPF_Glass

np.random.seed(0)

mnist_bin = np.array(mnist.data > 128, dtype=np.int8)
mnist_bin = mnist_bin * 2 -1

N = 50
N_mnist, D = mnist_bin.shape

idx = np.random.choice(N_mnist, N, replace=False)
mnist_train = mnist_bin[idx]
#print(mnist_train.shape)

mpf = MPF_Glass(mnist_train)
J, b = mpf.learn_jb()

X0 = np.random.randint(2, size=D)*2 - 1
X0 = mnist_train[2]
idx = np.random.choice(D, 200, replace=False)
X0[idx] *= -1
X = []

class HopfieldNet(object):

    def __init__(self, X0, J, b):
        self.N, self.D = X0.shape
        self.X0 = X0
        self.J = J
        self.b = b

    def run_network(self, history=True, max_iter=10):
        X = self.X0.copy()
        X_prev = X.copy()

        for n in range(max_iter):
            

for n in range(10):
    X.append(np.copy(X0))
    X_prev = X0.copy
    print(n)
    for i in np.random.permutation(D):
        X0[i] = 2 * ((J @ X0 - b)[i] > 0) -1
    if np.all(X_prev == X0):
        print('converged')
        break
        

for n in range(10):
    plt.subplot(2, 5, n+1)
    plt.imshow(X[n].reshape((28, 28)))


plt.show()
