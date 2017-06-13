import numpy as np
import matplotlib.pylab as plt
from scipy.signal import convolve
from scipy import optimize
import time

# Conventions: spins are symmetric: {-1,1}, J has vanishing diagonals, energy is E = 0.5 x.T @ J @ X

np.set_printoptions( precision = 3 )
epsilon = 1E-10
Seed = 8
if Seed:
    np.random.seed(Seed)

def getW(J):
    a = np.array([[4,1,0,1,4]])
    r2 = a + a.T
    W = np.zeros((5,5))
    W[r2 == 1] = J[0]
    W[r2 == 2] = J[1]
    W[r2 == 4] = J[2]
    W[r2 == 5] = J[3]
    return W

def sigmoid( x ):
    return 1 / ( 1 + np.exp( -x ) )

def getRandX(D, N = 1):
    Dx, Dy = D
    X = np.floor(np.random.random( (N, Dx, Dy) ) * 2)
    return 2*X - 1

def getdEK(X):
    N = X.shape[0]
    dE = np.zeros_like(X)
    M = np.ones((2,2))
    for n in range(N):
        XM = convolve(X[n], M, 'valid')
        Q = np.ones_like(XM)
        Q[np.abs(XM) == 2] = -1
        dE[n] = -2 * convolve(Q, M, 'full')

    return dE

def getH(X, J):
    N = X.shape[0]
    W = np.array( getW(J) )
    H = np.zeros_like(X)

    for n in range(N):
        H[n] = convolve(X[n], W, 'same') 

    return H

def sampleX(JK, D, N, burnIn, thin):
    nSampleSteps = burnIn + (N) * thin
    Dx, Dy = D
    J = JK[:4]
    K = JK[4]

    dxRand = np.random.randint(Dx, size = (nSampleSteps))
    dyRand = np.random.randint(Dy, size = (nSampleSteps))
    pRand = np.random.random(nSampleSteps)

    X = np.zeros((N, Dx, Dy))
    x = getRandX(D)

    for i in range(nSampleSteps):
        dx = dxRand[i]
        dy = dyRand[i]
        d = [dx, dy]
        H = getH(x, J)
        dE = (-2*x*H + K * getdEK(x))[0, dx, dy]
        p = sigmoid(dE)
        
        if p > pRand[i]:
            x[:,dx, dy] = x[:,dx, dy]
        else:
            x[:,dx, dy] = -x[:,dx, dy]

        if i >= burnIn and (i - burnIn) % thin == 0:
            X[(i - burnIn) // thin ] = x
    return X

def KdK(X, JK):
    J = JK[:4]
    K = JK[4]

    dE = - 2 * X * getH(X, J) + K * getdEK(X)
    Kdn = np.exp(-0.5 * dE)
    K = Kdn.sum()

    dK = []
    for j in np.eye(4):
        dKdn = X * getH(X, j) * Kdn
        dK.append(dKdn.sum())

    dKdn = -0.5 * getdEK(X) * Kdn
    dK.append(dKdn.sum())
    dK = np.array(dK)

    return K, dK

def learnJ( X ):

    #Get random initial J
    J = 2 * np.random.random(5) - 1

    fdf = lambda j: KdK(X, j)
    minOut = optimize.fmin_l_bfgs_b(fdf, J)
    return np.array(minOut[0])

def stackX(X, ratio = 1.5, pad = 1):
    N, Dx, Dy = X.shape
    W = int(np.ceil(np.sqrt(ratio * N)))
    H = int(np.ceil(N / W))

    if H * W > N:
        X = np.concatenate((X, np.zeros((H * W - N, Dx, Dy))))

    padX = np.pad(X, ((0,0), (pad,pad), (pad,pad)), 'constant', constant_values = 0)
    padX = np.pad(X, ((0,0), (pad,pad), (pad,pad)), 'constant', constant_values = 0)
    rows = []
    for i in range(H):
        rows.append(np.hstack((padX[i*W:(i+1)*W])))
    Xstack = np.vstack(rows)
    return Xstack


#Set parameters
N = 50 #Number of samples
Dx = 10
D = (Dx, Dx) #Dimension of lattice
burnIn = 100 * D[0] * D[1]
thin = 10 * D[0] * D[1]

if True:
    for j0 in np.arange(-10, 10):
        if j0 < 0:
            str0 = 'n' +  str(-j0)
        else:
            str0 = str(j0)
        for j1 in np.arange(-10, 10):
            if j1 < 0:
                str1 = 'n' +  str(-j1)
            else:
                str1 = str(j1)
            fname = 'j12_' + str0 + '_' + str1 + '.npy'
            JK = [j0/10., j1/10., 0, 0, 0]
            X = sampleX(JK, D, N, burnIn, thin)
            print('Saving ' + fname + '...')
            np.save(fname, X) 

JKest = np.zeros((12, 21, 5))

for j0 in np.arange(-10, 1):
    if j0 < 0:
        str0 = 'n' +  str(-j0)
    else:
        str0 = str(j0)
    for j1 in np.arange(-10, 10):
        i = j0 + 10
        j = j1 + 10
        if j1 < 0:
            str1 = 'n' +  str(-j1)
        else:
            str1 = str(j1)
        fname = './data/j12_' + str0 + '_' + str1 + '.npy'
        X = np.load(fname) 
        plt.imshow(stackX(X), cmap = 'gray')
        plt.show()
        JKest[i, j] = learnJ(X)
        print(i,j, JKest[i,j])


