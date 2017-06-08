import numpy as np
import matplotlib.pylab as plt
from scipy import optimize

# Conventions: spins are binary {0,1}, J has vanishing diagonals, energy is E = 0.5 x.T @ J @ X

np.set_printoptions( precision = 3 )
epsilon = 1E-10
Seed = 8
if Seed:
    np.random.seed(Seed)

def sigmoid( x ):
    return 1 / ( 1 + np.exp( -x ) )

def getRandX(D, N = 1):
    X = np.floor(np.random.random( (D,N) ) * 2)
    if N == 1:
        X = np.reshape(X,-1)
    return X

def getRandJ(D, zeroDiag = True):
    J = np.random.random( (D,D) ) - 0.5
    J = 0.5 * (J + J.T)
    J = J - np.diag( np.diag(J) )
    return J
'''
def deltaE(X, J, d):
    return np.dot(
            (1-2*X)[d],
            np.dot(J, X)[d]
            )
'''
def deltaE(X, J):
    return (1-2*X) * (J@X)

def sampleX(J, N, burnIn, thin):
    nSampleSteps = burnIn + (N) * thin
    D = np.shape(J)[0]

    dRand = np.random.randint(D, size = nSampleSteps)
    pRand = np.random.random(nSampleSteps)

    X = np.zeros((D, N))
    x = getRandX(D)

    for i in range(nSampleSteps):
        d = dRand[i]
        dE = (deltaE(x, J)[d])

        p = sigmoid(dE)
        if p > pRand[i]:
            x[d] = x[d]
        else:
            x[d] = 1-x[d]

        if i >= burnIn and (i - burnIn) % thin == 0:
            X[:,(i - burnIn) // thin ] = x
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
N = 10000 #Number of samples
D = 5 #Number of dimensions
burnIn = 100 * D
thin = 10 * D

J = getRandJ(D) 
print(J)
X = sampleX(J, N, burnIn, thin)

print(learnJ(X))
