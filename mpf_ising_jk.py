import numpy as np
from scipy.signal import convolve
from scipy import optimize

class MPF_Estimator(object):
# Conventions: spins are symmetric: {-1,1}, energy is E = -0.5 x.T @ J @ X



    def __init__(self, X, JK0=None, fit_params=[1, 1, 1, 1, 1]):
        self.X = X
        self.JK_init = JK0
        self.corr_second = [None, None, None, None]
        self.fit_params = fit_params
        for n in range(4):
            J = [0, 0, 0, 0]
            J[n] = 1
            self.corr_second[n] = self._H(J)
        self.corr_fourth = self._get_fourth_order()

    def _get_fourth_order(self):
        dE = np.zeros_like(self.X)
        M = np.ones((2,2))
        for n, x in enumerate(self.X):
            XM = convolve(x, M, 'valid')
            Q = np.ones_like(XM)
            Q[np.abs(XM) == 2] = -1
            dE[n] = 2 * convolve(Q, M, 'full')

        return dE

    def _W(self, J):
        a = np.array([[4,1,0,1,4]])
        r2 = a + a.T
        W = np.zeros((5,5))
        W[r2 == 1] = J[0]
        W[r2 == 2] = J[1]
        W[r2 == 4] = J[2]
        W[r2 == 5] = J[3]

        return W
            
    def _H(self, J):
        W = self._W(J)
        H = np.zeros_like(self.X)

        for n, x in enumerate(self.X):
            H[n] = convolve(x, W, 'same')

        return H / 2

    def total_second_corr(self):
        C = []
        for n in range(4):
            C.append( self.corr_second[n].sum(axis=(1,2)) )

        return C

    def energy(self, J, K):
        E = -self._H(J).sum(axis=(1,2))
        M = np.ones((2,2))
        for n, x in enumerate(self.X):
            XM = convolve(x, M, 'valid')
            Q = np.ones_like(XM)
            Q[np.abs(XM) == 2] = -1
            E[n] -= K * Q.sum()
        return E

    def K_dK(self, JK):
        J = JK[:4]
        K = JK[4]

        dE = 0

        # Adding contributions due to second order
        for n in range(4):
            dE += 2 * J[n] * self.X * self.corr_second[n] * self.fit_params[n]

        # Adding contributions due to fourth order
        dE += K * self.corr_fourth * self.fit_params[4]

        Kdn = np.exp(-0.5 * dE)
        K = Kdn.sum()

        dK = []
        for C in self.corr_second:
            #dKdn = X * getH(X, j) * Kdn
            #dKdn = -0.5 * self.X *  C * Kdn
            dKdn = - self.X *  C * Kdn
            dK.append(dKdn.sum())

        dKdn = -0.5 * self.corr_fourth * Kdn
        dK.append(dKdn.sum())
        dK = np.array(dK)

        return K, dK

    def learn_jk(self):
        if self.JK_init is None:
            JK = 2 * np.random.random(5) - 1

        min_out = optimize.fmin_l_bfgs_b(self.K_dK, JK)
        estimate = min_out[0]

        param_names = ['J1', 'J2', 'J3', 'J4', 'K']
        output=[]
        params=[]

        for n in range(5):
            if self.fit_params[n]:
                params.append(param_names[n])
                output.append(estimate[n])

        print('Parameters fitted :', params)
        return np.array(output)



if __name__ == '__main__':
    D = 3
    X = np.ones((10, D, D))
    mpf = MPF_Estimator(X)
    print(mpf.energy([1,0,0,0], 1))
