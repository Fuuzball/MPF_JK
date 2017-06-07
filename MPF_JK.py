import numpy as np
import matplotlib.pylab as plt

np.random.seed(5)

def E(W, S):
    (nX, nY) = S.shape

    E = 0
    for i in range(nX):
        for j in range(nY):
            E += S[i,j] * H(W, S, (i,j))

    return E

def H(JK, S, ij):
    i, j = ij
    # Pad arrays with zero 
    SPad = np.pad(S, 2, 'constant', constant_values = 0)
    # Return NNs of S[i,j]
    SNN = SPad[i:i+1+4, j:j+1+4]

    J1, J2, J3, J4, K = JK
    WJ = np.zeros((5,5))
    a = np.array([[4,1,0,1,4]])
    dist = a + a.T
    WJ[dist == 1] = J1
    WJ[dist == 2] = J2
    WJ[dist == 4] = J3
    WJ[dist == 5] = J4
    WJ *= 0.5
    HJ = (SNN * WJ).sum()

    HK = 0
    for di, dj in [[-1,-1], [-1,1], [1,-1], [1,1]]:
        HK += K * SNN[2 + di, 2] * SNN[2, 2 + dj] * SNN[2 + di, 2 + dj]

    return HJ + HK


S = np.random.randint(2, size=(5,5))
JK = [1,2,0,0,0]
print (S)
print (E(JK, S))



