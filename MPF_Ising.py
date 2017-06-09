import numpy as np
import matplotlib.pylab as plt
from scipy.signal import convolve
from scipy import optimize

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

def deltaE(X, J):
    N = X.shape[0]
    W = np.array( [getW(J),]* N )
    H = convolve(X, W, 'same')
    return (-2*X) * H

def sampleX(J, D, N, burnIn, thin):
    nSampleSteps = burnIn + (N) * thin
    Dx, Dy = D

    dxRand = np.random.randint(Dx, size = (nSampleSteps))
    dyRand = np.random.randint(Dy, size = (nSampleSteps))
    pRand = np.random.random(nSampleSteps)

    X = np.zeros((N, Dx, Dy))
    x = getRandX(D)

    for i in range(nSampleSteps):
        dx = dxRand[i]
        dy = dyRand[i]
        d = [dx, dy]
        dE = (deltaE(x, J)[0, dx, dy]) 
        p = sigmoid(dE)
        
        if p > pRand[i]:
            x[:,dx, dy] = x[:,dx, dy]
        else:
            x[:,dx, dy] = -x[:,dx, dy]

        if i >= burnIn and (i - burnIn) % thin == 0:
            X[(i - burnIn) // thin ] = x
    return X

def KdK(X, J):
    if len(X.shape) == 2:
        D, N = X.shape
    else:
        D = X.shape[0]
        N = 1

    J = np.reshape(J, (D, D))
    dE = deltaE(X, J)
    Kdn = np.exp(-0.5 * dE)
    K = Kdn.sum()

    dK = - ((1 - 2*X) * Kdn) @ X.T
    dK = 0.5 * (dK + dK.T)
    dK = dK - np.diag(dK.diagonal())

    dK = dK.reshape(-1)

    return K, dK

def learnJ( X ):
    D = np.shape(X)[0]

    #Get random initial J
    J = getRandJ(D)
    J = np.array(np.reshape(J, (-1)))

    fdf = lambda j: KdK(X, j)
    minOut = optimize.fmin_l_bfgs_b(fdf, J)
    print (minOut[1:])
    return np.reshape(minOut[0], (D, D))

#Set parameters
N = 4 #Number of samples
D = (5, 5) #Dimension of lattice
burnIn = 100 * D[0] * D[1]
thin = 10 * D[0] * D[1]

J = [-0.1, 0, 0, 0]
X = sampleX(J, D, N, burnIn, thin)
print(X)
