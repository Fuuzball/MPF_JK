from __future__ import division
import numpy as np
import matplotlib.pylab as plt
import json
from scipy.signal import convolve
from scipy import optimize
import time
import os

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

def plotPhase(Xs, index = 0):
    plt.figure(figsize = (20,20))
    plt.imshow(
            stackX(Xs[:,index,:,:], 1),
            cmap = 'gray'
            )

    plt.show()

def loadSample(dirStr): 
    paramFile = dirStr + 'param.json'
    sampleFile = dirStr + 'sample.npy'

    with open(paramFile) as f:
        data = json.load(f) 
    JKList = np.array(data['JKList'])
    #Xs = np.array(data['samples'])
    Xs = np.load(sampleFile)

    return Xs, JKList

def JKestSweep(Xs, fileName = None):
    JKest = [] 
    for X in Xs:
        JKest.append(learnJ(X))

    JKest = np.array(JKest)
    if fileName:
        np.save(fileName, JKest)
    return JKest

def JKSweep(N, D, burnIn, thin, JKList, dirStr = None):
    try:
        os.mkdir(dirStr)
    except FileExistsError:
        while True:
            res = input('Directory already exsists, write sample into same directory anyway? Y/N:')
            if res == 'y' or res == 'Y':
                break
            if res == 'n' or res == 'N':
                return


    params = {
        'N' : N,
        'Dx' : Dx,
        'Dy' : Dy,
        'burnIn' : burnIn,
        'thin' : thin,
        } 

    samples = []

    times = []
    totaliters = len(JKList)
    iters = 0
    tStart = time.time()

    for jk in JKList:
        t0 = time.time()
        #samples[j1, j2] = sampleX(JK, D, N, burnIn, thin)
        samples.append(sampleX(jk, D, N, burnIn, thin))
        iters += 1
        times.append(time.time() - t0)
        meanT = sum(times) / len(times)
        tLeft = (totaliters - iters) * meanT

        print ('Finished sampling at ', jk, 'approx. time remaining: ', time.strftime("%H:%M:%S", time.gmtime(tLeft)), '. Time elapsed: ', time.strftime("%H:%M:%S", time.gmtime(time.time() - tStart))) 

    data = {} 
    data['params'] = params 
    data['JKList'] = JKList 
    #data['samples'] = samples
    samples = np.array(samples)

    if dirStr:
        paramFile = dirStr + 'param.json'
        sampleFile = dirStr + 'sample.npy'
        with open(paramFile, 'w') as f:
            json.dump(data, f) 
        np.save(sampleFile, samples) 

    return data, samples

def plotError(JKList, JKest, j1, j2, jX, jY):
    JKerr = JKest - JKList 
    toPlot = [
            JKList[:, j1], JKest[:, j1], JKerr[:, j1],
            JKList[:, j2], JKest[:, j2], JKerr[:, j2],
            ]
    cm = 'RdBu'

    plt.figure(figsize= (18,12))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        aMax = np.abs(toPlot[i]).max()
        plt.imshow(toPlot[i].reshape((jX, jY)), cmap = cm, vmin = -aMax, vmax = aMax)
        plt.colorbar()

    plt.show()

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

if __name__ == '__main__':

    #Set parameters
    N = 500 #Number of samples
    Dx, Dy = 10, 10
    D = (Dx, Dy) #Dimension of lattice
    burnIn = 100 * D[0] * D[1]
    thin = 10 * D[0] * D[1]

    N = 60 #Number of samples
    D = (10, 10) #Dimension
    burnIn = 100 * D[0] * D[1]
    thin = 10 * D[0] * D[1]
    JK = (0, 0, 0, 0, 0.2)

    X = sampleX(JK, D, N, burnIn, thin)
    plt.imshow(stackX(X))
    plt.show()

