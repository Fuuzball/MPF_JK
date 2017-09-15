import numpy as np
import matplotlib.pylab as plt
from mpf_ising_jk import MPF_Estimator
np.random.seed(5)


def sampleX(J, D, N, burn_in, thin):
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

def flip_cluster(X, d_init, J):
    visited = []
    # Initial site
    to_flip = [d_init]
    p = 1 - np.exp(-J)

    while to_flip:
        new_to_flip = []
        for d in to_flip:
            visited.append(d)
            nn = get_nn_indices(D, d)
            for n in nn:
                if ( X[n] == X[d] ) and ( n not in visited ) and (np.random.rand() < p):
                    new_to_flip.append(n)
            X[d] *= -1
        visited.extend(to_flip)
        to_flip = set(new_to_flip)

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

def stackX(X, ratio = 1.5, pad = 1):
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
            

if __name__ == '__main__':
    N = 100
    D = 50

    burn_in = 10 * D**2
    burn_in = 2
    thin = 10 * D**2
    thin = 2

    J_list = np.arange(0.75, 1.5, 0.05)
    n_J = J_list.shape[0]
    M_list = np.zeros_like(J_list)

    x = np.random.randint(2, size=(D, D)) * 2 - 1
    d =  np.random.randint(D, size=(2))
    d = tuple(d)
    flip_cluster(x, d, 0.2)




    if True:
        for i in range(n_J):
            J = 1/J_list[i]
            print(J)
            X = sampleX(J, D, N, burn_in, thin)
            M = X.mean(axis=(1,2))
            M_list[i] = np.abs(M).mean()

        plt.plot(J_list, M_list)
        plt.show()
