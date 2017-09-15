import time
import numpy as np
import random
import matplotlib.pylab as plt
from mpf_ising_jk import MPF_Estimator
#np.random.seed(42)


def sample_X(J, D, N, burn_in, thin):
    n_sample_steps = burn_in + N * thin
    Dx = D
    Dy = D

    X = np.zeros((N, Dx, Dy))
    x = np.random.randint(2, size=(Dx, Dy)) * 2 - 1

    for i in range(n_sample_steps):
        d =  np.random.randint(D, size=(2))
        d = tuple(d)
        x = flip_cluster(x, d, J)

        if i >= burn_in and (i - burn_in) % thin == 0:
            X[(i - burn_in) // thin] = x
    
    return X


def get_nn_indices(D, d):
    nn = []
    i, j = d
    if i > 0:
        nn.append((i - 1, j))
    if j > 0:
        nn.append((i, j - 1)) 
    if i < D - 1:
        nn.append((i + 1, j))
    if j < D - 1:
        nn.append((i, j + 1))
    return nn

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
            
def flip_cluster(X, d_init, J):
    visited = set([])
    # Initial site
    to_flip = {d_init}
    p = 1 - np.exp(-J)

    while to_flip:
        new_to_flip = set([])
        for d in to_flip:
            visited.add(d)
            nn = get_nn_indices(D, d)
            for n in nn:
                if ( X[n] == X[d] ) and ( n not in visited ) and (random.random() < p):
                    new_to_flip.add(n)
            X[d] *= -1
        visited.update(to_flip)
        to_flip = set(new_to_flip)

    return X

class WolffSampler(object):
    
    def __init__(self, J, D, N, burn_in, thin):
        self.J = J
        self.D = D
        self.N = N
        self.burn_in = burn_in
        self.thin = thin

        self.n_sample_steps = burn_in + N * thin 
        self.make_nn_dict()

    def sample_X(self, new_J=None): 

        if new_J:
            self.J = new_J
        self.x = np.random.randint(2, size=(self.D, self.D)) * 2 - 1
        X = np.zeros((N, D, D)) 
        for i in range(self.n_sample_steps):
            self.flip_random_cluster()

            if i >= self.burn_in and (i - self.burn_in) % self.thin == 0:
                X[(i - self.burn_in) // self.thin] = self.x

        return X

    def flip_random_cluster(self):
        d_init = self.get_rand_site()
        visited = set([])
        # Initial site
        to_flip = {d_init}
        p = 1 - np.exp(-self.J)

        while to_flip:
            new_to_flip = set([])
            for d in to_flip:
                spin = self.x[d]
                visited.add(d)
                for n in self.nn_dict[d]:
                    if ( self.x[n] == spin) and ( n not in visited ) and (random.random() < p):
                        new_to_flip.add(n)
                self.x[d] *= -1
            visited.update(to_flip)
            to_flip = set(new_to_flip)

    def get_rand_site(self):
        return (random.randint(0, self.D - 1), random.randint(0, self.D - 1))

    def make_nn_dict(self):
        self.nn_dict = {}
        for i in range(self.D):
            for j in range(self.D):
                nn = []
                if i > 0:
                    nn.append((i - 1, j))
                if j > 0:
                    nn.append((i, j - 1)) 
                if i < D - 1:
                    nn.append((i + 1, j))
                if j < D - 1:
                    nn.append((i, j + 1))
                self.nn_dict[(i, j)] = nn

if __name__ == '__main__':
    N = 20
    D = 20
    burn_in = 0
    thin = 1
    J = 1

    wolff = WolffSampler(J, D, N, burn_in, thin)
    t0 = time.process_time()
    X = wolff.sample_X()
    t1 = time.process_time()
    plt.imshow(stack_X(X))
    plt.show()



