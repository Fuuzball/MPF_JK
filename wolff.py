import numpy as np
import matplotlib.pylab as plt
np.random.seed(5)


def sampleX(J, D, N, burn_in, thin):
    n_sample_steps = burn_in + N * thin
    Dx = D
    Dy = D

    X = np.zeros((N, Dx, Dy))
    x = np.random.randint(2, size=(Dx, Dy))

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
            X[d] = 1 - X[d]
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
    N = 5
    D = 20

    burn_in = 100 * D**2
    thin = 10 * D**2

    Jstar = np.log(1 + np.sqrt(2)) / 2
    J = Jstar
    X = sampleX(J, D, 6, 100, 100)
    X_stacked = stackX(X)
    plt.imshow(X_stacked)
    plt.show()
