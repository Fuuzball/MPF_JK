import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from scipy import optimize
import time

from mpf_spin_glass import MPF_Glass_JK as MPF_JK
from mpf_spin_glass import MPF_Glass as MPF_GlassDirect

def get_rand_J(D):

    """
        Return random symmetric D x D matrix J with vanishing diagonals
    """

    J = np.random.random((D, D))
    J = 0.5 * (J + J.T)
    J = J - np.diag(np.diagonal(J))
    return J

def torch_double_var(npy_arry, grad):
    return Variable(torch.from_numpy(npy_arry).double(), requires_grad=grad)

class MPF_Glass(object):

    def __init__(self, X):
        self.N, self.D = X.shape
        #Convert to float
        self.X = Variable(torch.from_numpy(X).type(torch.DoubleTensor), requires_grad=False)
        # Indices for the upper triangle (not including diagonals)

    def flatten_params(self, J, b):
        return np.hstack((J.flatten(), b))

    def unflatten_params(self, theta):
        J = theta[:-self.D].reshape((self.D, self.D))
        b = theta[-self.D:]
        return J, b

    def dE_glass(self, J, b):
        if not torch.equal(J, J.t()):
            print("Warning: J is not symmetric")

        # Enforce the matrix to be symmetric and have vanishing diagonals
        mask = Variable((torch.ones((D,D)) - torch.eye(D)).double(), requires_grad=False)
        J_sym = 0.5 * (J.t() + J) * mask
        dE = 2 * self.X * (self.X.mm(J_sym)) - 2 * self.X * b[None, :]

        return dE

    def get_dE(self, theta):
        D = self.D
        assert len(theta) == D**2 + D, "The number of parameters is incorrect"
        J = theta[:-D].view(D, D)
        b = theta[-D:]
        return self.dE_glass(J, b)

    def K_dK(self, theta_npy_arr):
        theta = torch_double_var(theta_npy_arr, True)
        # Assign values
        dE = self.get_dE(theta)
        Knd = torch.exp(-0.5 * dE)
        K = Knd.sum()

        # Get gradient
        K.backward()
        dK = theta.grad.data.numpy()

        # Convert to numpy scalar and arrays
        K = K.data.numpy()[0]
        return K, dK

    def learn(self):
        """
        Returns parameters estimated through MPF
        """

        # Initial parameters
        theta = np.zeros(self.num_params) 

        min_out = optimize.fmin_l_bfgs_b(self.K_dK, theta)
        estimate = min_out[0] 
        return estimate

class MPF_Glass_Fourth(MPF_Glass):

    def __init__(self, X, shape=None, M=None):
        super().__init__(X)

        if shape == None:
            W = int(np.sqrt(self.D))
            H = int(self.D / W)
            self.shape = W, H
        X_2d = X.reshape((self.N, *self.shape))
        self.X_2d = Variable(torch.from_numpy(X_2d).double(), requires_grad=False)

        # Default matrix is 2 x 2 all ones
        if M == None:
            M = torch.ones(2,2).double()
        else:
            M = torch.from_numpy(M).double()

        self.interaction_order = (M == 1).sum()

        self.M = Variable(M, requires_grad=True)

        XM = F.conv2d(
                self.X_2d[:, None, :, :], self.M[None, None, :, :]
                )

        self.Q = 1 - 2*(XM.abs() == 2).double()

    def dE4(self, K):
        Q_pad = (F.pad(self.Q, (1, 1, 1, 1)))
        return -2 * F.conv2d(Q_pad, self.M[None, None, :, :])

    def dE_HLE(self, K, M = None):

        """
            Calculates the energy due to higher-order local interactions

            Args:
                M (np.array): A Binary (0, 1) matrix of aribtrary size M_H x M_W which sets the sites relevant for interaction.
                                The number of ones in M, N_M is the order of interaction.
                K (np.array): A (H - M_H + 1) x (W - M_W + 1) pytorch tensor denoting the coupling strength

            Returns energy difference due to flipping bits of X_2d

        """
        
        # Default to fourth order interaction (JK model)
        if M == None:
            M = np.ones((2, 2))

        # Get shape of M and determine the order of interaction
        M_H, M_W = M.shape
        # Order of interaction
        N_M = (M == 1).sum().astype(float)

        # Flipped to do real convolution (as opposed to xcorrelation)
        M_flipped = M[::-1, ::-1].copy()

        # Turn M into pytorch Variable
        M = torch_double_var(M, False)
        M_flipped = torch_double_var(M_flipped, False)

        # Make convolution to count number of spin+ - spin-
        XM = F.conv2d(
                self.X_2d[:, None, :, :], self.M[None, None, :, :]
                )

        # Get number of spin-
        N_minus = -(XM - N_M) / 2

        # Even spin- : 1 odd spin- : -1
        Q = 2 * (N_minus % 2 == 0).double() - 1

        # Energy contribution to interaction at each position
        #K = Variable(torch.from_numpy(K), requires_grad=True)
        print(K)
        print('-'*20)
        print(Q)
        E = K * Q
        print(E)
        E_padded = F.pad(E, (M_W-1, M_W-1, M_H-1, M_H-1))
        print(M_flipped)
        dE = F.conv2d(
                E_padded, M_flipped[None, None, :, :]
                )
        print(dE)


if __name__ == '__main__':
    D = 9
    N = 3
    np.random.seed(15)
    X = np.random.randint(2, size=(N, D)) * 2 - 1
    J = get_rand_J(D)
    b = np.zeros(D)

    glass_torch = MPF_Glass(X)
    glass = MPF_GlassDirect(X)
    glass_JK_torch = MPF_Glass_Fourth(X)
    glass_JK = MPF_JK(X)


    #print(glass_JK_torch.dE4(1))
    #print(glass_JK.dE4(1))

    K = np.arange(4).reshape((2,2))
    K = torch_double_var(K, True)
    print(glass_JK_torch.dE_HLE(K))
    #print(glass_JK_torch.Q)

