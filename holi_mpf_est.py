import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from scipy import optimize
import time
import logging
import sys
from collections import OrderedDict

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


def make_arr_torch(arr, arr_name, req_grad=False):
    if is_ndarray(arr):
        return torch_double_var(arr, grad=req_grad)
    elif is_Variable(arr):
        return arr.double()
    else:
        logger.error(arr_name + 'needs to be either np array or torch variable')
        sys.exit()

class HOLIGlass(object):

    def __init__(self, X, shape_2d=None, M=None, params=['J_glass', 'b']):
        logger.info('Initializing HOLIGlass...')
        self.N, self.D = X.shape

        #Convert to float
        #self.X = Variable(torch.from_numpy(X).type(torch.DoubleTensor), requires_grad=False)
        #self.X = make_arr_torch(X, 'X')

        # Default shape to square
        if shape_2d == None:
            W = int(np.sqrt(self.D))
            H = int(self.D / W)
        else:
            H, W = shape
        self.H = H
        self.W = W

        self.X = X

        # Default to fourth order interaction (JK model)
        if M is None:
            M = [
                    np.ones((2, 2))
                    ]
        self.M = M

        # Make 2nd order correlations for local ising model
        self.corr_mats = OrderedDict()
        for name in params:
            if 'j_' in name:
                k = int(name[2:])
                self.corr_mats[name] = self.get_corr_mat(k)

        # Define param shape
        self.param_shape = self.get_param_shape(params)
        # Set number of parameter
        self.num_params = 0
        for shape in self.param_shape.values():
            len = 1
            for d in shape:
                len *= d
            self.num_params += len

    def get_param_shape(self, params):
        param_shape = OrderedDict()
        for name in params:
            if name == 'J_glass':
                param_shape[name] = (self.D, self.D)
            elif name == 'b':
                param_shape[name] = (self.D, )
            elif 'j_' in name:
                param_shape[name] = (1, )
            else:
                logger.error('Parameter name {} is not recognized'.format_map(name))
                sys.exit()

        for n, m in enumerate(self.M):
            m_h, m_w = m.shape
            k_h = self.H - m_h + 1
            k_w = self.W - m_w + 1
            param_shape['k_%d'%n] = (k_h, k_w)

        return param_shape
        
    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, new_X):
        self._X = make_arr_torch(new_X, 'X')
        self._X_2d =  self._X.view((self.N, self.H, self.W))

    @property
    def X_2d(self):
        #return self.X.view((self.N, self.H, self.W))
        return self._X_2d

    def assert_param_shape(self, param, name, shape):
        if not param.shape == shape:
            logger.error('Parameter {} has shape {} instead correct shape {}'.format(name, param.shape, shape))
            sys.exit()
        else:
            logger.debug('Parameter {} has the correct shape of {}'.format(name, shape))

    def get_random_params(self, req_grad=False):
        params = {}

        for p_name in self.param_shape:
            shape = self.param_shape[p_name]
            param = np.random.normal(size=shape)
            if p_name == 'J_glass':
                param += param.T
                np.fill_diagonal(param, 0)
            params[p_name] = torch_double_var(param, req_grad)

        return params

    def flatten_params(self, params):
        logger.debug('Calling function flatten_params')
        #logger.debug('-'*20 + 'J' + '-'*20 + '\n{}'.format(J))

        # Check if number of params are correct
        if len(params) != len(self.param_shape):
            logger.error('Number of input params ({}) is different from number of declared params ({})'.format(len(params), len(self.param_shape)))
            sys.exit()

        params_list = []
        for f_name in params:
            shape = self.param_shape[f_name]
            param = params[f_name]
            self.assert_param_shape(param, f_name, shape)
            params_list.append(param.view(-1))

        return torch.cat(params_list)

    def unflatten_params(self, theta):
        theta = make_arr_torch(theta, 'theta')
        start = 0
        params = {}
        for f_name in self.param_shape:
            shape = self.param_shape[f_name]
            len = 1
            for d in shape:
                len *= d
            end = start + len

            params[f_name] = theta[start:end].view(shape)
            start = end

        return params
            


        D = self.D
        theta = make_arr_torch(theta, 'theta')
        self.assert_param_shape(theta, 'theta', (self.num_params, ))

        J = theta[0:D**2].view((D, D))
        b = theta[D**2:D**2 + D]

        start = D**2 + D
        K = []
        for (k_h, k_w) in self.k_dims:
            end = start + k_h * k_w
            K.append(theta[start:end].view((k_h, k_w)))
            start = end
        return J, b, K

    def get_W(self, J):
        a = torch.Tensor([[4,1,0,1,4]])
        r2 = a + a.t()
        W = torch.zeros((5,5))
        W[r2 == 1] = J[0]
        W[r2 == 2] = J[1]
        W[r2 == 4] = J[2]
        W[r2 == 5] = J[3]
        return Variable(W.double(), requires_grad=False)

    def get_corr_mat(self, k):
        J = [0, 0, 0, 0]
        J[k - 1] = 1
        W = self.get_W(J)
        H = F.conv2d( self.X_2d[:, None, :, :], W[None, None, :, :], padding=2)
        H = H.squeeze(1) # Get rid of the channel dimension
        return H
        dE_2d = - 2 * self.X_2d * H
        return dE_2d.view(self.N, -1)

    def dE_local_ising(self, J):
        W = self.get_W(J)
        H = F.conv2d( self.X_2d[:, None, :, :], W[None, None, :, :], padding=2)
        H = H.squeeze(1) # Get rid of the channel dimension
        dE_2d = - 2 * self.X_2d * H
        return dE_2d.view(self.N, -1)

    def dE_glass(self, J):
        logger.debug('Calling dE_glass')
        if not torch.equal(J, J.t()):
            logger.debug('J is not symmetric')

        # Enforce the matrix to be symmetric and have vanishing diagonals
        D = self.D
        mask = Variable((torch.ones((D,D)) - torch.eye(D)).double(), requires_grad=False)
        J_sym = 0.5 * (J.t() + J) * mask
        dE = 2 * self.X * (self.X.mm(J_sym))
        return dE

    def dE_bias(self, b):
        return -2 * self.X * b[None, :]

    def dE_HOLI(self, K, M):
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
        
        M_H, M_W = M.shape
        N_M = (M == 1).sum().astype(float)
        
        # Flipped to do real convolution (as opposed to xcorrelation)
        M_flipped = M[::-1, ::-1].copy()

        # Turn M into pytorch Variable
        M = torch_double_var(M, False)
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
        params = self.unflatten_params(theta)
        logger.debug('-'*20 + 'params' + '-'*20 + '\n{}'.format(params))

        dE = 0

        # Energy due to bias
        if 'b' in params:
            b = params['b']
            dE += self.dE_bias(b)

        # Energy due to ising glass
        if 'J_glass' in params:
            J = params['J_glass']
            dE += self.dE_glass(J)

        # Energy due to fixed, local, second interaction
        for p in params:
            if 'j_' in p:
                C = self.corr_mats[p]
                j = params[p]
                dE += - 2 * j * self.X * C.view(self.N, -1)

        # Energy due to higher order local interaction
        k_params = [params[p] for p in params if 'k_' in p]
        for k, m in zip(k_params, self.M):
            dE += self.dE_HOLI(k, m).view(self.N, -1)

        return dE

    def K(self, theta):
        theta = make_arr_torch(theta, 'theta', req_grad=True)
        #theta = torch_double_var(theta_npy_arr, True)
        # Assign values
        dE = self.get_dE(theta)
        Knd = torch.exp(-0.5 * dE)
        K = Knd.sum() 
        return K

    def learn(self, unflatten=True, tol=1E-5, max_iter=1000, lr=.1):
        logger.info('Start fitting parameters...')
        t0 = time.time()

        theta = Variable(torch.zeros(self.num_params).double(), requires_grad=True)

        optimizer = optim.LBFGS([theta], lr=.1, max_iter=max_iter, tolerance_grad=tol)

        def f():
            optimizer.zero_grad()
            loss = self.K(theta)
            loss.backward()
            return loss

        optimizer.step(f)
        flat_grad = optimizer._gather_flat_grad()
        abs_grad_sum = flat_grad.abs().sum()
        if abs_grad_sum < tol:
            logger.info(f'Optimization converged')
        else:
            logger.info(f'Optimization did not converge with abs_grad_sum={abs_grad_sum}')

        logger.info('Fitting took {:.4f}s'.format(time.time() - t0))
        if unflatten:
            return self.unflatten_params(theta)
        else:
            return theta

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y:%H:%M:%S',
    level=logging.INFO)
    D = 8**2
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

    holi = HOLIGlass(X, params=['j_1', 'b'])
    params = holi.learn(max_iter=1000)
