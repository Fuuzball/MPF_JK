import numpy as np
import matplotlib.pylab as plt

def E(W, S):
    """ Returns the energy of spins (S) given couplings (W) """
    (J1, J2, J3, J4, K) = W 
    (nX, nY) = S.shape

    E = 0
    for i in range(nX):
        for j in range(nY):
            for (di, dj) in [[0,1], [1,0]]:
                try:
                    E += J1 * S[i,j] * S[i + di, j + dj]
                except:
                    pass

            for (di, dj) in [[1,1], [1,-1]]:
                try:
                    E += J2 * S[i,j] * S[i + di, j + dj]
                except:
                    pass

            for (di, dj) in [[0,2], [2,0]]:
                try:
                    E += J3 * S[i,j] * S[i + di, j + dj]
                except:
                    pass

            for (di, dj) in [[-1,-2], [-1,2], [-2,-1], [2,-1]]:
                try:
                    E += J4 * S[i,j] * S[i + di, j + dj]
                except:
                    pass

            try:
                E += K * S[i, j] * S[i+1, j] * S[i, j+1] * S[i+1, j+1]
            except:
                pass
    return E

def Hij(W, S, ij):
    i, j = ij
    (J1, J2, J3, J4, K) = W 
    (nX, nY) = S.shape

    H = 0
    for (di, dj) in [[0,1], [1,0], [0,-1], [-1,0]]:
        try:
            H += 0.5 * J1 *  S[i + di, j + dj]
        except:
            pass

    for (di, dj) in [[-1,1], [1,-1], [-1,-1], [1,1]]:
        try:
            H += 0.5 * J2 * S[i + di, j + dj]
        except:
            pass

    for (di, dj) in [[0,2], [2,0], [0,-2], [-2,0]]:
        try:
            H += 0.5 * J3 * S[i + di, j + dj]
        except:
            pass

    for (di, dj) in [[-1,-2], [-1,2], [-2,-1], [2,-1]]:
        try:
            H += 0.5 * J4 * S[i + di, j + dj]
        except:
            pass

    for (di, dj) in [[-1,1], [1,-1], [-1,-1], [1,1]]:
        try:
            H += 0.25 * K * S[i+di, j] * S[i, j+dj] * S[i+di, j+dj]
        except:
            pass
    return H

def E2(W, S):
    (nX, nY) = S.shape

    E = 0
    for i in range(nX):
        for j in range(nY):
            E += S[i,j] * Hij(W, S, (i,j))

    return E

S = np.random.randint(2, size=(5,5))
JK = [1,0,0,0,0]
print (S)
print (E(JK, S))
print (E2(JK, S))

print (np.array([[Hij(JK, S, (i,j)) for j in range(5)] for i in range(5)]))

