import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from scipy import optimize
import time

from mpf_spin_glass import MPF_Glass_JK as MPF_JK

TORCH_DOUBLE = torch.DoubleTensor

def get_rand_J(D):

    """
        Return random symmetric D x D matrix J with vanishing diagonals
    """

    J = np.random.random((D, D))
    J = 0.5 * (J + J.T)
    J = J - np.diag(np.diagonal(J))
    return J

class MPF_Glass(object):

    def __init__(self, X):
        self.N, self.D = X.shape
        #Convert to float
        self.X = Variable(torch.from_numpy(X).type(torch.DoubleTensor), requires_grad=False)
        # Indices for the upper triangle (not including diagonals)
        self.num_params = np.int(self.D * (self.D + 1))

    def flatten_params(self, J, b):
        return np.hstack((J.flatten(), b))

    def unflatten_params(self, theta):
        J = theta[:-self.D].reshape((self.D, self.D))
        b = theta[-self.D:]
        return J, b

    def dE_glass(self, theta_npy_arr):

        theta_tensor = torch.from_numpy(theta_npy_arr).type(TORCH_DOUBLE)
        self.theta = Variable(theta_tensor, requires_grad=True)

        D = self.D
        J_mat = theta_npy_arr[:-D].reshape(D, D)
        if not (np.array_equiv(J_mat, J_mat.T)):
            print('Warning: J is not symmetric')

        J = self.theta[:-D].view(D, D)
        b = self.theta[-D:]
        mask = Variable((torch.ones((D,D)) - torch.eye(D)).double(), requires_grad=False)

        J_sym = 0.5 * (J.t() + J) * mask
        
        dE = 2 * self.X * (self.X.mm(J_sym)) - 2 * self.X * b[None, :]
        return dE

    def get_dE(self, theta):
        return self.dE_glass(theta)

    def K_dK(self, theta):
        # Assign values
        dE = self.get_dE(theta)
        Knd = torch.exp(-0.5 * dE)
        K = Knd.sum()

        # Get gradient
        K.backward()
        dK = self.theta.grad.data.numpy()

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
        self.X_2d = Variable(torch.from_numpy(X_2d).type(TORCH_DOUBLE), requires_grad=False)

        # Default matrix is 2 x 2 all ones
        if M is None:
            M = torch.ones(2,2).type(TORCH_DOUBLE)
        else:
            M = torch.from_numpy(M).type(TORCH_DOUBLE)

        self.M = Variable(M, requires_grad=True)

        XM = F.conv2d(
                self.X_2d[:, None, :, :], self.M[None, None, :, :]
                )

        self.Q = 1 - 2*(XM.abs() == 2).double()

    def dE4(self, K):
        Q_pad = (F.pad(self.Q, (1, 1, 1, 1)))
        return -2 * F.conv2d(Q_pad, self.M[None, None, :, :])

if __name__ == '__main__':
    D = 9
    N = 3
    np.random.seed(15)
    X = np.random.randint(2, size=(N, D)) * 2 - 1
    J = get_rand_J(D)
    b = np.zeros(D)

    glass_torch = MPF_Glass(X)

    glass_JK_torch = MPF_Glass_Fourth(X)
    glass_JK = MPF_JK(X)
    print(glass_JK_torch.dE4(1))
    print(glass_JK.dE4(1))
