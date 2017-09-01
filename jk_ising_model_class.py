
class JKIsingModel(object):

    def __init__(self, X, J = (0, 0, 0, 0), K = 0):
        self.X = X
        self.J = J
        self.K = K

        self.n_samples, self.D1, self.D2 = X.shape

    def get_coupling_matrix(self, J):
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

    def site_nn_correlation(self, W):
        """
        Get spin-spin correlation per site for given coupling matrix W
        """

        corr_mat = np.zeros_like(self.X)

        for i, x in enumerate(self.X):
            corr_mat[i] = convolve(x, W,  mode='same')
        
        return corr_mat * self.X

    def total_correlation(self):
        """
        Get 1-4th NN spin-spin correlation of the entire system
        """

        total_corr = np.zeros((self.n_samples, 4))
        for n in range(4):
            j = [0, 0, 0, 0]
            j[n] = 1
            W = self.get_coupling_matrix(j)
            C = self.site_nn_correlation(W)
            total_corr[:, n] = C.sum(axis=(1,2))

        return total_corr / 2

    def fourth_order_interaction(self):
        """
        Return the energy due to fourth order interaction
        """

        E4 = np.zeros(self.n_samples)
        M = np.ones((2,2))
        for n in range(self.n_samples):
            XM = convolve(self.X[n], M, 'valid')
            Q = np.ones_like(XM)
            Q[np.abs(XM) == 2] = -1
            E4[n] = Q.sum()

        return E4

    def energy(self):
        W = self.get_coupling_matrix(self.J)
        C = self.site_nn_correlation(W)


        # Energy due to second order interactions
        E = -C.sum(axis=(1,2)) / 2 
        # Energy due to fourth order interactions
        E += -self. K * self.fourth_order_interaction()

        return E

    def propose_flip(self, dx, dy, p):
        assert dx < self.D1
        assert dy < self.D2

        E0 = self.energy()
        self.X[:, dx, dy] *= -1
        dE = self.energy() - E0

        # Reject proposal if fails check
        fails = sigmoid(-dE) < p
        self.X[fails, dx, dy] *= -1

    def get_H_eff(self, J1_only=True):
        C1 = self.total_correlation()[:, 0]
        a = np.hstack((np.ones(N)[:, None], -C1[:, None]))
        E = self.energy()
        E0, J1 = np.linalg.lstsq(a, E)[0]
        return E0, J1

    def get_sub_array(self, dx, dy):
        a = np.zeros((self.n_samples, 5, 5))
        for i in range(5):
            for j in range(5):
                x = dx + i - 2
                y = dy + j - 2
                if (0 <= x < self.D1) and (0 <= y < self.D2):
                    a[:, i, j] = self.X[:, x, y]
        return a
