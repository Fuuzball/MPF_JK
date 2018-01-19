import torch
from torch.autograd import Variable
from torch import optim
import numpy as np
from scipy import optimize
import time
from mpf_spin_glass import MPF_Glass as MPF

TORCH_DOUBLE = torch.DoubleTensor

def get_rand_J(D):

    """
        Return random symmetric D x D matrix J with vanishing diagonals
    """

    J = np.random.random((D, D))
    J = 0.5 * (J + J.T)
    J = J - np.diag(np.diagonal(J))
    return J

class MPF_Estimator(object):

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

    def get_dE(self, theta_npy_arr):

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

    def learn_jb(self):
        """
        Returns parameters estimated through MPF
        """

        # Initial parameters
        theta = np.zeros(self.num_params) 

        min_out = optimize.fmin_l_bfgs_b(self.K_dK, theta)
        estimate = min_out[0] 
        return estimate

                
if __name__ == '__main__':
    D = 5
    N = 3
    np.random.seed(15)
    X = np.random.randint(2, size=(N, D)) * 2 - 1
    J = get_rand_J(D)
    b = np.zeros(D)


    glass = MPF_Estimator(X)
    glass_no_torch = MPF(X)

    if True:
        t0 = time.time()
        print('-'*20, 'Direct Implimentation', '-'*20)
        params = (glass.learn_jb())
        print(glass.unflatten_params(params))
        print(time.time() - t0)
        t0 = time.time()
        print(glass_no_torch.learn_jb())
        print(time.time() - t0)
