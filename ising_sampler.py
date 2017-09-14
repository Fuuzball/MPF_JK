import numpy as np
import matplotlib.pylab as plt
import MPF_Ising as MPF
from mpf_ising_jk import MPF_Estimator
import time
from scipy.signal import convolve

# Conventions: spins are symmetric: {-1,1}, J has vanishing diagonals, energy is E = 0.5 x.T @ J @ X

epsilon = 1E-10
Seed = 15
if Seed:
    np.random.seed(Seed)

def get_rand_X(D, N = 1):
    if isinstance(D, int):
        Dx = Dy = D
    else:
        Dx, Dy = D

    X = np.floor(np.random.random( (N, Dx, Dy) ) * 2)
    return 2*X - 1

def sigmoid( x ):
    return 1 / ( 1 + np.exp( -x ) )

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

def get_coupling_matrix(J):
    """
    For given 1-4th NN coupling strength J, return 5 x 5 matrix to be convolved with spins
    """

    a = np.array([[4,1,0,1,4]])
    r2 = a + a.T
    W = np.zeros((5,5))
    W[r2 == 1] = J[0]
    W[r2 == 2] = J[1]
    W[r2 == 4] = J[2]
    W[r2 == 5] = J[3]
    return W

def site_nn_correlation(X, W):
    """
    Get spin-spin correlation per site for given coupling matrix W
    """

   # corr_mat = np.zeros_like(X)

    corr_mat = convolve(X, W, mode='same')

    return corr_mat * X

def total_correlation(X):
    """
    Get 1-4th NN spin-spin correlation of the entire system
    """

    #total_corr = [0, 0, 0, 0]
    total_corr = np.zeros(4)
    for n in range(4):
        j = [0, 0, 0, 0]
        j[n] = 1
        W = get_coupling_matrix(j)
        C = site_nn_correlation(X, W)
        total_corr[n] = C.sum()

    return total_corr / 2

def fourth_order_interaction(X):
    """
    Return the energy due to fourth order interaction
    """

    M = np.ones((2,2))
    XM = convolve(X, M, 'valid')
    Q = np.ones_like(XM)
    Q[np.abs(XM) == 2] = -1
    #E4[n] = Q.sum()

    return Q.sum()

def energy(X, J, K):
    W = get_coupling_matrix(J)
    C = site_nn_correlation(X, W)

    # Energy due to second order interactions
    E = -C.sum() / 2 
    # Energy due to fourth order interactions
    E += -K * fourth_order_interaction(X)

    return E

def sample_local(D, N, J, K, burn_in, thin):
    n_sample_steps = burn_in + N * thin

    if isinstance(D, int):
        Dx = Dy = D
    else:
        Dx, Dy = D

    pRand = np.random.random(n_sample_steps)
    dxRand = np.random.randint(Dx, size = (n_sample_steps))
    dyRand = np.random.randint(Dy, size = (n_sample_steps))

    X = np.zeros((N, Dx, Dy))
    x = get_rand_X(D)[0]

    for i in range(n_sample_steps):
        p = pRand[i]
        dx = dxRand[i]
        dy = dyRand[i]
        E0 = energy(x, J, K)
        x[dx, dy] *= -1
        Ep = energy(x, J, K)
        if sigmoid(-(Ep - E0)) < p:
            x[dx, dy] *= -1

        if i >= burn_in and (i - burn_in) % thin == 0:
            print((i - burn_in) // thin )
            X[(i - burn_in) // thin ] = x

    return X

def get_H_eff(Xs, J, K):
    N = len(Xs)
    C1 = np.zeros(N)
    E = np.zeros(N)
    for n, X in enumerate(Xs):
        C1[n] = total_correlation(X)[0]
        E[n] = energy(X, J, K)
    a = np.hstack((np.ones(N)[:, None], -C1[:, None]))
    E0, J1 = np.linalg.lstsq(a, E)[0]
    return E0, J1

def flip_cluster(X0, d_init, J):
    visited = []
    # Initial site
    to_flip = [d_init]
    p = 1 - np.exp(-2 * J)

    X = X0.copy()
    D = X0.shape

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
    if i < D[0] - 1:
        nn.append((i + 1, j))
    if j < D[1] - 1:
        nn.append((i, j + 1))
    return nn

def sample_wolff_effective(E0, J_eff, J, K, D, N, burn_in, thin):

    n_sample_steps = burn_in + N * thin
    if isinstance(D, int):
        Dx = Dy = D
    else:
        Dx, Dy = D

    pRand = np.random.random(n_sample_steps)
    dxRand = np.random.randint(Dx, size = (n_sample_steps))
    dyRand = np.random.randint(Dy, size = (n_sample_steps))

    X = np.zeros((N, Dx, Dy))
    #x = np.random.randint(2, size=(Dx, Dy))
    x = get_rand_X(D)[0]

    delta_J = list(J)
    delta_J[0] -= J_eff

    for i in range(n_sample_steps):
        d = (dxRand[i], dyRand[i])

        xp = flip_cluster(x, d, J_eff) 
        delta_E = energy(xp, delta_J, K) - energy(x, delta_J, K)

        if sigmoid(-delta_E) > pRand[i]:
            x = xp

        if i >= burn_in and (i - burn_in) % thin == 0:
            print((i - burn_in) // thin)
            X[(i - burn_in) // thin] = x
    
    return X

def SLMC_sample(J, K, D, N, burn_in, thin):
    print('Sampling with local updates...')
    X_local = sample_local(D, N, J, K, burn_in, thin)
    E0, J_eff = get_H_eff(X_local, J, K)
    print('Effective parameters : E0 = {}, J_eff = {}'.format(E0, J_eff))
    print('Wolff sampling...')
    X = sample_wolff_effective(E0, J_eff, J, K, D, N, burn_in, thin)
    return X

if __name__ == '__main__':

    N = 20 #Number of samples
    D = (18, 18) #Dimension
    burn_in = int(10 * ( D[0] * D[1] ))
    thin = int(1 * ( D[0] * D[1] ))
    J = [-j for j in [-0.6905666,  -0.15444032,  0.06169692,  0.05076923]]
    K = 0.06480178

    if False:
        X_local = sample_local(D, N, J, K, burn_in, thin)
        np.save('./X_local', X_local)
        print('sample created with local updates')
    else:
        X_local = np.load('./X_local.npy')
    plt.figure()
    plt.imshow(stack_X(X_local))
    plt.show()

    estimator_X_local = MPF_Estimator(X_local)
    print('local parameter estimates : {}'.format(estimator_X_local.learn_jk()))

    E0, J_eff = get_H_eff(X_local, J, K)
    print('effective parameters : E0 = {}, J_eff = {}'.format(E0, J_eff))

    N = 30 #Number of samples
    burn_in = int(1 * ( D[0] * D[1] ))
    thin = int(10 * ( D[0] * D[1] )) 

    if True:
        X = sample_wolff_effective(E0, J_eff, J, K, D, N, burn_in, thin)
        np.save('./X', X)
    else:
        X = np.load('./X.npy')

    print('true parameters : J = {}, K = {} '.format(J, K)) 
    estimator_X = MPF_Estimator(X)
    print('woff_eff parameter estimates : {}'.format(estimator_X.learn_jk()))

    plt.figure()
    plt.imshow(stack_X(X))


    plt.show()

    



