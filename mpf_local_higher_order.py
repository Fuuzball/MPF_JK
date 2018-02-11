import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from scipy import optimize
import time
import logging
import sys

from mpf_spin_glass_direct import MPF_Glass_JK as MPF_JK
from mpf_spin_glass_direct import MPF_Glass as MPF_GlassDirect

logger = logging.getLogger(__name__)


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


def is_ndarray(arr):
    return isinstance(arr, np.ndarray) 


def is_Variable(arr):
    return isinstance(arr, Variable) 


def make_arr_torch(arr, arr_name):
    if is_ndarray(arr):
        return  torch_double_var(arr, False)
    elif is_Variable(arr):
        return arr
    else:
        logger.error(arr_name + 'needs to be either np array or torch variable')
        sys.exit()


class MPF_Glass(object):
    def __init__(self, X):
        self.N, self.D = X.shape
        #Convert to float
        self.X = Variable(torch.from_numpy(X).type(torch.DoubleTensor), requires_grad=False)
        # Indices for the upper triangle (not including diagonals)

        self.num_params = self.D **2 + self.D

    def flatten_params(self, J, b):
        return np.hstack((J.flatten(), b))

    def unflatten_params(self, theta):
        J = theta[:-self.D].reshape((self.D, self.D))
        b = theta[-self.D:]
        return J, b

    def dE_glass(self, J, b):
        logger.debug('Calling dE_glass')
        if not torch.equal(J, J.t()):
            logger.debug('J is not symmetric')

        # Enforce the matrix to be symmetric and have vanishing diagonals
        D = self.D
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


class MPF_Glass_HOLI(MPF_Glass):
    def __init__(self, X, shape=None, M=None):
        logger.info('Instantiating MPF_Glass_HOLI')
        super().__init__(X)

        if shape == None:
            W = int(np.sqrt(self.D))
            H = int(self.D / W)
        else:
            H, W = shape
        self.H = H
        self.W = W
        X_2d = X.reshape((self.N, H, W))
        self.X_2d = Variable(torch.from_numpy(X_2d).double(), requires_grad=False)

        # Default to fourth order interaction (JK model)
        if M is None:
            M = np.ones((2, 2))

        self.M = M

        # Get shape of M and determine the order of interaction
        self.M_H, self.M_W = M.shape
        self.N_M = (M == 1).sum().astype(float)

        # Shape of the parameters to be fit
        self.K_H = H - self.M_H + 1
        self.K_W = W - self.M_W + 1

        self.num_params = self.D**2 + self.D + (self.K_H * self.K_W)

    def flatten_params(self, J, b, K):
        logger.debug('Calling function flatten_params')
        logger.debug('-'*20 + 'J' + '-'*20 + '\n{}'.format(J))
        logger.debug('-'*20 + 'b' + '-'*20 + '\n{}'.format(b))
        logger.debug('-'*20 + 'K' + '-'*20 + '\n{}'.format(K))

        if not (isinstance(J, np.ndarray) or isinstance(J, Variable)):
            logger.error('Parameter J is not passed as either numpy array or torch variable')
            
            sys.exit()



        if isinstance(J, np.ndarray):
            logger.debug('Parameters are numpy arrays')
            assert isinstance(b, np.ndarray) and isinstance(K, np.ndarray), 'All parameters need to be of the same type (i.e. np.ndarray)'
            return np.hstack((J.flatten(), b, K.flatten()))

        if isinstance(J, Variable):
            logger.debug('Parameters are torch variables')
            assert isinstance(b, Variable) and isinstance(K, Variable), 'All parameters need to be of the same type (i.e. pytorch Variable)'
            J_flat = J.view(-1)
            K_flat = K.view(-1)
            theta = Variable(torch.zeros(len(J_flat) + len(b) + len(K_flat)))

            D = self.D
            start = 0
            end = start + D**2
            theta[start:end] = J_flat

            start = end
            end = start + D
            theta[start:end] = b

            start = end
            end = start + self.K_H * self.K_W
            theta[start:end] = K_flat
            assert end == len(theta), 'Input parameters incorrect length. (len(theta) = {} but should be {})'.format(len(theta), end)

            return theta.double()

    def unflatten_params(self, theta):
        D = self.D
        K_H = self.K_H
        K_W = self.K_W

        if isinstance(theta, Variable):
            is_numpy = False
        elif isinstance(theta, np.ndarray):
            is_numpy = True
        else:
            sys.exit('Parameters need to be a numpy array or a pytorch.autograd Variable')


        start = 0
        end = start + D**2
        J_flat = theta[start:end]
        if is_numpy:
            J = J_flat.reshape((D, D))
        else:
            J = J_flat.view((D, D))

        start = end
        end = start + D
        b = theta[start:end]

        start = end
        end = start + K_H * K_W
        K_flat = theta[start:end]
        if is_numpy:
            K = K_flat.reshape((K_H, K_W))
        else:
            K = K_flat.view((K_H, K_W))

        assert end == len(theta), 'Input parameters incorrect length. (len(theta) = {} but should be {})'.format(len(theta), end)
        return J, b, K

    def dE_HOLI(self, K):
        logger.debug('Calling dE_HOLI')
        logger.debug('-'*20 + 'K' + '-'*20 + '\n{}'.format(K))

        """
            Calculates the energy due to higher-order local interactions

            Args:
                M (np.array): A Binary (0, 1) matrix of aribtrary size M_H x M_W which sets the sites relevant for interaction.
                                The number of ones in M, N_M is the order of interaction.
                K (np.array): A (H - M_H + 1) x (W - M_W + 1) pytorch tensor denoting the coupling strength

            Returns energy difference due to flipping bits of X_2d in matrix of shape N x H x W

        """
        
        M_H = self.M_H
        M_W = self.M_W
        N_M = self.N_M
        
        # Flipped to do real convolution (as opposed to xcorrelation)
        M_flipped = self.M[::-1, ::-1].copy()

        # Turn M into pytorch Variable
        M = torch_double_var(self.M, False)
        M_flipped = torch_double_var(M_flipped, False)

        # Make convolution to count number of spin+ - spin-
        XM = F.conv2d(
                self.X_2d[:, None, :, :], M[None, None, :, :]
                )

        # Get number of spin-
        N_minus = -(XM - N_M) / 2

        # Even spin- : 1 odd spin- : -1
        Q = 2 * (N_minus % 2 == 0).double() - 1

        # Energy contribution to interaction at each position
        #K = Variable(torch.from_numpy(K), requires_grad=True)
        E = K * Q

        # Convolve with inverted matrix M to get dE
        E_padded = F.pad(E, (M_W-1, M_W-1, M_H-1, M_H-1))
        dE = -2 * F.conv2d(
                E_padded, M_flipped[None, None, :, :]
                )

        return dE

    def get_dE(self, theta):
        logger.debug('Calling get_dE')
        logger.debug('-'*20 + 'theta' + '-'*20 + '\n{}'.format(theta))
        J, b, K = self.unflatten_params(theta)
        dE = self.dE_glass(J, b)
        dE += self.dE_HOLI(K).view(self.N, -1)
        return dE


class HOLIGlass(object):

    def __init__(self, X, shape_2d=None, M=None):
        self.N, self.D = X.shape
        #Convert to float
        self.X = Variable(torch.from_numpy(X).type(torch.DoubleTensor), requires_grad=False)
        # Indices for the upper triangle (not including diagonals)

        self.num_params = self.D **2 + self.D

        # Default shape to square
        if shape_2d == None:
            W = int(np.sqrt(self.D))
            H = int(self.D / W)
        else:
            H, W = shape
        self.H = H
        self.W = W

        # Default to fourth order interaction (JK model)
        if M is None:
            M = [
                    torch_double_var(np.ones((2, 2)), False)
                    ]
        self.M = M

        # Set number of parameters


        self.k_dims = []
        self.num_params = self.D**2 + self.D # Size of J + b

        for m in self.M:
            m_h, m_w = m.shape
            k_h = H - m_h + 1
            k_w = W - m_w + 1
            self.k_dims.append((k_h, k_w))
            self.num_params += k_h * k_w

        if False:
            # Get shape of M and determine the order of interaction
            self.M_H, self.M_W = M.shape
            self.N_M = (M == 1).sum().astype(float)

            # Shape of the parameters to be fit
            self.K_H = H - self.M_H + 1
            self.K_W = W - self.M_W + 1

            self.num_params = self.D**2 + self.D + (self.K_H * self.K_W)

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, new_X):
        self._X = make_arr_torch(new_X, 'X')

    @property
    def X_2d(self):
        return self.X.view((self.N, self.H, self.W))

    def assert_param_shape(self, param, name, shape):
        if not param.shape == shape:
            logger.error('Parameter {} has shape {} instead correct shape {}'.format(name, param.shape, shape))
            sys.exit()
        else:
            logger.debug('Parameter {} has the correct shape of {}'.format(name, shape))

    def get_random_params(self, req_grad=False):
        D = self.D
        J = np.random.normal(size=(D, D))
        J += J.T
        np.fill_diagonal(J, 0)
        J = torch_double_var(J, req_grad)

        b = np.random.normal(size=D)
        b = torch_double_var(b, req_grad)

        K = []
        for k_hw in self.k_dims:
            K.append(
                torch_double_var(np.random.normal(size=k_hw), req_grad)
                    )

        return J, b, K


    def flatten_params(self, J, b, K):
        logger.debug('Calling function flatten_params')
        logger.debug('-'*20 + 'J' + '-'*20 + '\n{}'.format(J))
        logger.debug('-'*20 + 'b' + '-'*20 + '\n{}'.format(b))
        logger.debug('-'*20 + 'K' + '-'*20 + '\n{}'.format(K))

        # Initialize empty theta
        theta = Variable(torch.zeros(self.num_params))

        # Make all arrays torch variables and assert their shape to be correct
        J = make_arr_torch(J, 'J')
        b = make_arr_torch(b, 'J')

        D = self.D

        self.assert_param_shape(J, 'J', (D, D))
        self.assert_param_shape(b, 'b', (D, ))

        theta[:D**2] = J.view(-1)
        theta[D**2: D**2 + D] = b

        start = D**2 + D
        for n, k in enumerate(K):
            k_h, k_w = self.k_dims[n]
            name = 'k_{}'.format(n)
            K[n] = make_arr_torch(k, name)
            self.assert_param_shape(K[n], name, (k_h, k_w))

            end = start + k_h * k_w
            theta[start:end] = K[n].view(-1)
            start = end

        return theta

        if not (isinstance(J, np.ndarray) or isinstance(J, Variable)):
            logger.error('Parameter J is not passed as either numpy array or torch variable')
            
            sys.exit()

        if isinstance(J, np.ndarray):
            logger.debug('Parameters are numpy arrays')
            assert isinstance(b, np.ndarray) and isinstance(K, np.ndarray), 'All parameters need to be of the same type (i.e. np.ndarray)'
            return np.hstack((J.flatten(), b, K.flatten()))

        if isinstance(J, Variable):
            logger.debug('Parameters are torch variables')
            assert isinstance(b, Variable) and isinstance(K, Variable), 'All parameters need to be of the same type (i.e. pytorch Variable)'
            J_flat = J.view(-1)
            K_flat = K.view(-1)
            theta = Variable(torch.zeros(len(J_flat) + len(b) + len(K_flat)))

            D = self.D
            start = 0
            end = start + D**2
            theta[start:end] = J_flat

            start = end
            end = start + D
            theta[start:end] = b

            start = end
            end = start + self.K_H * self.K_W
            theta[start:end] = K_flat
            assert end == len(theta), 'Input parameters incorrect length. (len(theta) = {} but should be {})'.format(len(theta), end)

            return theta.double()

    def unflatten_params(self, theta):
        D = self.D
        K_H = self.K_H
        K_W = self.K_W

        if isinstance(theta, Variable):
            is_numpy = False
        elif isinstance(theta, np.ndarray):
            is_numpy = True
        else:
            sys.exit('Parameters need to be a numpy array or a pytorch.autograd Variable')


        start = 0
        end = start + D**2
        J_flat = theta[start:end]
        if is_numpy:
            J = J_flat.reshape((D, D))
        else:
            J = J_flat.view((D, D))

        start = end
        end = start + D
        b = theta[start:end]

        start = end
        end = start + K_H * K_W
        K_flat = theta[start:end]
        if is_numpy:
            K = K_flat.reshape((K_H, K_W))
        else:
            K = K_flat.view((K_H, K_W))

        assert end == len(theta), 'Input parameters incorrect length. (len(theta) = {} but should be {})'.format(len(theta), end)
        return J, b, K

    def dE_glass(self, J, b):
        logger.debug('Calling dE_glass')
        if not torch.equal(J, J.t()):
            logger.debug('J is not symmetric')

        # Enforce the matrix to be symmetric and have vanishing diagonals
        D = self.D
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

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    D = 16
    N = 100

    X = np.random.randint(2, size=(N, D)) * 2 - 1

    M1 = np.ones((2,2))
    M2 = np.array(
            [
                [1, 0, 1],
                [0, 1, 0],
                [1, 0, 1]
                ]
            )

    holi = HOLIGlass(X, M=[M1, M2])
    J, b, K = (holi.get_random_params())
    theta = holi.flatten_params(J, b, K)
    print(theta)


if False:
    logging.basicConfig(level=logging.INFO)
    D = 100
    N = 1000
    np.random.seed(15)
    X = np.random.randint(2, size=(N, D)) * 2 - 1
    X = (np.random.random(size=(N, D)) < 0.3) * 2 - 1
    X[:, 0] = 1
    X[:, 2] = 1
    X[:, 11] = 1
    X[:, 20] = 1
    X[:, 22] = 1
    J = get_rand_J(D)
    b = np.zeros(D)

    M = np.array(
            [
                [1, 0, 1],
                [0, 1, 0],
                [1, 0, 1]
                ]
            )

    glass_torch = MPF_Glass(X)
    glass = MPF_GlassDirect(X)
    glass_HOLI_torch = MPF_Glass_HOLI(X, M=M)
    glass_JK = MPF_JK(X)


    #print(glass_JK_torch.dE4(1))
    #print(glass_JK.dE4(1))
    if False:
        K = np.ones((2, 2))
        K = torch_double_var(K, True)
        J = torch_double_var(J, True)
        b = torch_double_var(b, True)
        theta = (glass_JK_torch.learn())
        print(glass_JK_torch.unflatten_params(theta))

        theta = glass_JK.learn_jbk()
        print(theta)


    print('-'*20, 'HOLI', '-'*20)
    theta = (glass_HOLI_torch.learn())
    J, b, K = (glass_HOLI_torch.unflatten_params(theta))
    print('-'*20, 'J',  '-'*20)
    print(J)
    print('-'*20, 'b',  '-'*20)
    print(b)
    print('-'*20, 'K',  '-'*20)
    print(K)
