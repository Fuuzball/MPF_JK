import numpy as np
import random
from math import exp
from scipy.signal import convolve
from scipy import optimize
import time
import matplotlib.pylab as plt

np.random.seed(0)

class GibbsSampler(object):
# Conventions: spins are symmetric: {-1,1}, energy is E = -0.5 x.T @ J @ X

    def __init__(self, D, J, K):
        self.D = D
        self.J = J
        self.K = K

        self.x = self.get_rand_spins()
        self.W = self.get_W(J)

    def get_rand_spins(self):
        return np.random.randint(2, size=(self.D, self.D)) * 2 - 1

    def get_W(self, J):
        a = np.array([[4,1,0,1,4]])
        r2 = a + a.T
        W = np.zeros((5,5))
        W[r2 == 1] = J[0]
        W[r2 == 2] = J[1]
        W[r2 == 4] = J[2]
        W[r2 == 5] = J[3]
        return W

    def get_dE2(self, i, j):
        X_padded = np.zeros((self.D + 4, self.D + 4))
        X_padded[2:-2, 2:-2] = self.x
        X_near = X_padded[i:i+5, j:j+5]
        return self.x[i,j ] * (self.W * X_near).sum()

    def get_dE4(self, i, j):
        dE = 0
        di = []
        dj = []
        if i > 0:
            di.append(i-1)
        if i < self.D - 1:
            di.append(i+1)
        if j > 0:
            dj.append(j-1)
        if j < self.D - 1:
            dj.append(j+1)


        for ii in di:
            for jj in dj:
                dE += 2 * self.K * self.x[i, j] * self.x[i, jj] * self.x[ii, j] * self.x[ii, jj] 
        return dE

    def get_dE(self, i, j):
        return self.get_dE2(i, j) + self.get_dE4(i, j)

    def sample_X(self, N, burn_in=None, thin=None, display_time=False): 
        t0 = time.process_time()

        n_sample_steps = burn_in + N * thin
        
        rand_d = np.random.randint(self.D, size=(n_sample_steps, 2))
        rand_p = np.random.random(n_sample_steps)
        X = np.zeros((N, self.D, self.D)) 
        self.x = self.get_rand_spins()

        for n in range(n_sample_steps):
            i, j = rand_d[n]
            dE = self.get_dE(i, j)
            p = exp(-dE)
            if p > rand_p[n]:
                self.x[i, j] *= -1
            
            if n >= burn_in and (n - burn_in) % thin == 0:
                X[(n - burn_in) // thin] = self.x

        self.X = X

        if display_time:
            print('Sampling took {:.4f}s'.format(time.process_time() - t0))
        return X

    def _get_fourth_order(self):
        dE = np.zeros_like(self.X)
        M = np.ones((2,2))
        for n, x in enumerate(self.X):
            XM = convolve(x, M, 'valid')
            Q = np.ones_like(XM)
            Q[np.abs(XM) == 2] = -1
            dE[n] = 2 * convolve(Q, M, 'full')

        return dE

    def plot_sample(self, fname='./data/sample.png', ratio = 1.5, W = None, pad = 1):

        X_stacked = stack_X(self.X, ratio, W, pad)
        fig = plt.imshow(stack_X(self.X, ratio, W, pad), cmap='Greys_r')
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.imsave(fname, X_stacked, format='png', cmap='Greys')

    def get_recommended_burnin_thin(self, max_iter=1000, plot=False, window_mult=1, thres=0.5):
        X = self.sample_X(max_iter, 0, 1)
        dist0 = ((X[0] != X).mean(axis=(1,2)))
        burn_in_rec = np.nonzero(dist0 > thres)[0][1]

        window = burn_in_rec * window_mult
        valid_start = np.arange(burn_in_rec, max_iter - window)
        dist = np.zeros((valid_start.shape[0], window))
        for i, t in enumerate(valid_start):
            dist[i] = ((X[t] != X).mean(axis=(1,2)))[t: t + window]  
        dist_mean = dist.mean(axis=0)
        if dist_mean.max() > thres:
            thin_rec = np.nonzero(dist_mean > thres)[0][1]
        else:
            thin_rec = None

        if plot:
            plt.figure()
            plt.subplot(211)
            plt.plot(np.arange(max_iter), dist0)
            plt.subplot(212)
            plt.plot(dist_mean)
            plt.show()

        return burn_in_rec, thin_rec

def stack_X(X, ratio = 1.5, W = None, pad = 1):
    N, Dx, Dy = X.shape

    if W:
        H = int(np.ceil(N / W))
    else:
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


if __name__ == '__main__':
    N = 4
    D = 18
    J = [0.6905666,  0.15444032,  -0.06169692,  -0.05076923]
    J = [-0.5, 0, 0, 0]
    K = 0
    #K = -0.06480178
    #J = [0.2905666,  0.15444032,  -0.06169692,  -0.05076923]


    burn_in = 0
    thin = 600

    gibbs = GibbsSampler(D, J, K)
    X = gibbs.sample_X(N, burn_in, thin)
    gibbs.plot_sample('./data/sample_m05.png', W=1)
    #plt.show()

