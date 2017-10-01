import numpy as np
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original') 
import matplotlib.pylab as plt
from mpf_spin_glass import MPF_Glass
import time



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

class HopfieldNet(object):

    def __init__(self, X_train, shape=None):
        self.K, self.D = X_train.shape
        self.X_train = X_train
        self.X_w_bias = np.hstack(
                (X_train, np.ones((self.K, 1)))
                )
        self.shape = shape

    def set_Jb(self, J, b):
        self.J = J
        self.b = b

    def learn_Jb(self, method='mpf'):
        if method == 'mpf':
            mpf = MPF_Glass(self.X_w_bias)
            self.J, self.b = mpf.learn_jb()
            self.b = np.zeros_like(self.b)
        elif method == 'opr':
            self.J = self.X_w_bias.T @ self.X_w_bias #/ len(X_train)

    def run_network(self, X0=None, history=True, max_iter=10, output=True):
        #X = X0.copy()
        if X0 is None:
            X0 = np.random.randint(2, size=D) * 2 - 1
        X = np.hstack((X0, 1))

        if history:
            self.X_history = [X[:-1].copy()]
        converged = False

        for n in range(max_iter):
            X_prev = X.copy()
            for i in np.random.permutation(self.D):
                #X[i] = 2 * ( X @ self.J[:, i] - self.b[i] > 0 ) - 1
                X[i] = 2 * ( X @ self.J[:, i] > 0 ) - 1
                if history:
                    self.X_history.append(X[:-1].copy())
            if np.array_equal(X_prev, X):
                converged = True
                if output:
                    print('Converged after {} iterations'.format(n + 1))
                break

        if not converged:
            if output:
                print('Did not converge after {} iteration{}'.format(max_iter, 's' if max_iter > 1 else ''))

        if history:
            self.X_history = np.array(self.X_history)

        return X[:D]

    def corrupt_memory(self, frac=0.3):
        X = self.X_train.copy()

        n_bits = int(self.D * frac)
        for x in X:
            idx = np.random.choice(self.D, n_bits, replace=False)
            x[idx] *= -1

        return X

    def energy(self, X):
        return -0.5 * X @ self.J @ X.T + X @ self.b

    def is_local_min(self):
        #X_conv = self.X_train.copy()
        X_conv = self.X_w_bias.copy()
        for i in range(self.D):
            X_conv[:, i] = 2 * (X_conv @ self.J[:, i] > 0) - 1
        return np.all(X_conv == self.X_w_bias, axis=1)


if __name__ == '__main__':
    np.random.seed(0)

    mnist_bin = np.array(mnist.data > 128, dtype=np.int8)
    mnist_bin = mnist_bin * 2 -1

    N = 200

    N_mnist, D = mnist_bin.shape 
    idx = np.random.choice(N_mnist, N, replace=False)
    mnist_train = mnist_bin[idx]

    hopfield = HopfieldNet(mnist_train)
    X0 = hopfield.corrupt_memory(frac=0.2)[2]

    hopfield.learn_Jb(method='mpf')
    hopfield.run_network(X0, max_iter=10)

    if True:
        n_plot = 20
        end = int(len(hopfield.X_history)/3)
        d_frame = int(len(hopfield.X_history) / n_plot) + 1
        plt.imshow(
                stack_X(
                    hopfield.X_history[::int(D/5)].reshape((-1, 28, 28)), ratio=2
                )
                )

        #plt.imshow(hopfield.X_history[-1].reshape((28,28)))

        plt.show()
