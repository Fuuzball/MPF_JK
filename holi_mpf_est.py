import torch as th
from torch import optim
from torch_lbfgs import LBFGS
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from scipy import optimize
import time
import logging
import sys
from collections import OrderedDict

logger = logging.getLogger(__name__)
seed=None
rng = np.random.RandomState(seed=seed)


def get_rand_J(D):
    """
        Return random symmetric D x D matrix J with vanishing diagonals
    """

    J = np.random.random((D, D))
    J = 0.5 * (J + J.T)
    J = J - np.diag(np.diagonal(J))
    return J

def th_double_var(npy_arry, grad=False):
    return Variable(th.from_numpy(npy_arry).double(), requires_grad=grad)

def is_ndarray(arr):
    return isinstance(arr, np.ndarray) 

def is_Variable(arr):
    return isinstance(arr, Variable) 

class HOLIGlass(object):
    def __init__(self, shape, shape_2d=None, M=None, params=['J_glass', 'b'], use_cuda=True):
        """
        Build hopfield network with memories X
        """
        logger.info('Initializing HOLIGlass...')
        self.N, self.D = shape
        self.corr_mats = OrderedDict()
        self.USE_CUDA = use_cuda
        if use_cuda:
            logger.info('Using CUDA')
            th.cuda.init()
        else:
            logger.info('Using CPU')

        # Default to fourth order interaction (JK model)
        if M is None:
            M = [
                    np.ones((2, 2))
                    ]
        self.M = M

        self.need_2d = False

        for name in params:
            if 'j_' in name:
                self.need_2d = True

        if M != []:
            self.need_2d = True

        if self.need_2d:
            # Default shape to square
            if shape_2d == None:
                W = int(np.sqrt(self.D))
                H = int(self.D / W)
            else:
                H, W = shape
            self.H = H
            self.W = W
            self.shape_2d = (H, W)

        if self.need_2d:
            # Make 2nd order correlations for local ising model
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

        if self.num_params > self.N:
            logger.warning(f'The number of parameters {self.num_params} is greater than the size of training set {self.N}')
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
    if False:
        @property
        def X(self):
            return self._X
        @X.setter
        def X(self, new_X):
            self._X = self.to_double_var(new_X, 'X')
            if len(self._X.shape) == 1:
                self._X = self._X[None, :]

            if self.need_2d:
                self._X_2d =  self._X.view((self.N, self.H, self.W))
                self.update_corr_mat()
        @property
        def X_2d(self):
            #return self.X.view((self.N, self.H, self.W))
            return self._X_2d
    @property
    def theta(self):
        return self._theta
    @theta.setter
    def theta(self, new_theta):
        self._theta = new_theta
        self.params = self.unflatten_params(new_theta) 
    def to_double_var(self, X, arr_name='', requires_grad=False):
        if self.USE_CUDA:
            d_type = th.cuda.DoubleTensor
        else:
            d_type = th.DoubleTensor

        if is_ndarray(X):
            return Variable(th.from_numpy(X).type(d_type), requires_grad=requires_grad)
        elif is_Variable(X):
            return X.type(d_type)
        else:
            try:
                return Variable(X.type(d_type), requires_grad=requires_grad)
            except:
                logger.error(f'Variable {arr_name} needs to be either np array or th variable instead of {type(X)}')
                sys.exit()
    def to_numpy(self, X):
        if self.USE_CUDA:
            return X.data.cpu().numpy()
        else:
            return X.data.numpy()
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
            #params[p_name] = th_double_var(param, req_grad)
            params[p_name] = self.to_double_var(param, p_name, req_grad)

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
            params_list.append(self.to_numpy(param.view(-1)))

        return np.hstack(params_list)
    def unflatten_params(self, theta, numpy=False):
        theta = self.to_double_var(theta, 'theta')
        start = 0
        params = {}
        for f_name in self.param_shape:
            shape = self.param_shape[f_name]
            len = 1
            for d in shape:
                len *= d
            end = start + len

            param = theta[start:end].view(shape)
            if numpy:
                if self.USE_CUDA:
                    param = param.data.cpu().numpy()
                else:
                    param = param.data.numpy()

            params[f_name] = param
            start = end

        return params
    def get_W(self, J):
        a = th.Tensor([[4,1,0,1,4]])
        r2 = a + a.t()
        W = th.zeros((5,5))
        W[r2 == 1] = J[0]
        W[r2 == 2] = J[1]
        W[r2 == 4] = J[2]
        W[r2 == 5] = J[3]
        return self.to_double_var(W, arr_name='W', requires_grad=False)
    def get_corr_mat(self, k):
        J = [0, 0, 0, 0]
        J[k - 1] = 1
        W = self.get_W(J)
        H = F.conv2d( self.X_2d[:, None, :, :], W[None, None, :, :], padding=2)
        H = H.squeeze(1) # Get rid of the channel dimension
        return self.X_2d * H
    def update_corr_mat(self):
        for name in self.corr_mats:
            k = int(name[2:])
            self.corr_mats[name] = self.get_corr_mat(k)
    def dE_glass(self, J, X):
        logger.debug('Calling dE_glass')
        if not th.equal(J, J.t()):
            logger.debug('J is not symmetric')

        # Enforce the matrix to be symmetric and have vanishing diagonals
        D = self.D
        #mask = Variable((th.ones((D,D)) - th.eye(D)).double(), requires_grad=False)
        mask = th.ones((D, D)) - th.eye(D)
        mask = self.to_double_var(mask, 'mask')
        J_sym = 0.5 * (J.t() + J) * mask
        dE = 2 * X * (X.mm(J_sym))
        return dE
    def dE_bias(self, b, X):
        return -2 * X * b[None, :]
    def dE_HOLI(self, K, M, X):
        logger.debug('Calling dE_HOLI')
        logger.debug('-'*20 + 'K' + '-'*20 + '\n{}'.format(K))

        """
            Calculates the energy due to higher-order local interactions

            Args:
                M (np.array): A Binary (0, 1) matrix of aribtrary size M_H x M_W which sets the sites relevant for interaction.
                                The number of ones in M, N_M is the order of interaction.
                K (np.array): A (H - M_H + 1) x (W - M_W + 1) pyth tensor denoting the coupling strength

            Returns energy difference due to flipping bits of X_2d in matrix of shape N x H x W

        """
        
        M_H, M_W = M.shape
        N_M = (M == 1).sum().astype(float)
        
        # Flipped to do real convolution (as opposed to xcorrelation)
        M_flipped = M[::-1, ::-1].copy()

        # Turn M into pyth Variable
        #M = th_double_var(M, False)
        M = self.to_double_var(M, 'M', False)
        #M_flipped = th_double_var(M_flipped, False)
        M_flipped = self.to_double_var(M, 'M', False)

        # Make convolution to count number of spin+ - spin-
        X_2d =  X.view((self.N, self.H, self.W))
        XM = F.conv2d(
                X_2d[:, None, :, :], M[None, None, :, :]
                )


        # Get number of spin-
        N_minus = -(XM - N_M) / 2

        # Even spin- : 1 odd spin- : -1
        Q = 2 * (N_minus % 2 == 0).double() - 1

        # Energy contribution to interaction at each position
        #K = Variable(th.from_numpy(K), requires_grad=True)
        E = K * Q

        # Convolve with inverted matrix M to get dE
        E_padded = F.pad(E, (M_W-1, M_W-1, M_H-1, M_H-1))
        dE = -2 * F.conv2d(
                E_padded, M_flipped[None, None, :, :]
                )

        return dE
    def get_dE(self, theta, X, to_numpy=False):
        logger.debug('Calling get_dE')
        params = self.unflatten_params(theta)
        logger.debug('-'*20 + 'params' + '-'*20 + '\n{}'.format(params))

        dE = 0

        # Energy due to bias
        if 'b' in params:
            b = params['b']
            dE += self.dE_bias(b, X)

        # Energy due to ising glass
        if 'J_glass' in params:
            J = params['J_glass']
            dE += self.dE_glass(J, X)

        # Energy due to fixed, local, second interaction
        for p in params:
            if 'j_' in p:
                C = self.corr_mats[p]
                j = params[p]
                dE += j * C.view(self.N, -1)

        # Energy due to higher order local interaction
        k_params = [params[p] for p in params if 'k_' in p]
        for k, m in zip(k_params, self.M):
            dE += self.dE_HOLI(k, m, X).view(self.N, -1)

        if to_numpy:
            return self.to_numpy(dE)
        else:
            return dE
    def K(self, theta, X):
        theta = self.to_double_var(theta, 'theta', requires_grad=True)
        #theta = th_double_var(theta_npy_arr, True)
        # Assign values
        dE = self.get_dE(theta, X)
        Knd = th.exp(-0.5 * dE)
        K = Knd.mean() 
        return K
    def learn(self, X, unflatten=True, theta0=None, params=None):
        logger.info('Start fitting parameters...')
        t0 = time.time()
        X = self.to_double_var(X, 'X')

        if isinstance(theta0, float):
            theta = self.to_double_var(
                        theta0 * self.flatten_params(self.get_random_params()), 'theta', requires_grad=True
                    )
        elif theta0 is not None:
            theta = theta0
        else:
            theta = self.to_double_var(np.zeros(self.num_params), requires_grad=True)

        def f():
            optimizer.zero_grad()
            loss = self.K(theta, X)
            loss.backward()
            return loss

        if params is None:
            logger.info('Optimizing with default parameters')
            optimizer = LBFGS([theta])
            optimizer.step(f)
        else:
            for param in params:
                logger.info('Optimizing with parameters: {}'.format(param))
                optimizer = LBFGS([theta], **param)
                optimizer.step(f)

        logger.info('Fitting took {:.4f}s'.format(time.time() - t0))
        self.theta = theta
        if unflatten:
            return self.params
        else:
            return self.theta
    # Alternative learning algorithms need to be refactored to include X
    def learn_sgd(elf, unflatten=True):
        theta = Variable(th.zeros(self.num_params).double(), requires_grad=True)
        optimizer = optim.SGD([theta], lr =.0001)

        for _ in range(10):
            self.K(theta).backward()
            optimizer.step()

        if unflatten:
            return self.unflatten_params(theta)
        else:
            return theta
    def learn_scipy(self):
        theta = Variable(th.zeros(self.num_params).double(), requires_grad=True)

        def f(theta):
            theta = th_double_var(theta, True)
            theta = self.to_double_var(theta, 'theta', True)
            loss = self.K(theta)
            loss.backward()
            return loss.data.numpy(), theta.grad.data.numpy()

        min_out = optimize.fmin_l_bfgs_b(f, theta)
        estimate = min_out[0]
        self.flat_params = estimate
        self.unflat_params = self.unflatten_params(estimate, True)

        return self.unflatten_params(estimate, True)
    def get_frac_capacity(self, X, noise_p=0):
        """
        Return fraction of memories (X) that are stored as local minima. Noise can be added by setting parameter noise_p
        """
        noise = rng.binomial(1, noise_p, size=(X.shape))

        noise = self.to_double_var(noise)
        X = self.to_double_var(X)
        X_noisy = X + noise

        dE = self.get_dE(self.theta, X_noisy, to_numpy=True)
        frac_min = (dE > 0).all(axis=1).sum()/self.N
        return frac_min


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y:%H:%M:%S',
    level=logging.INFO)
    logging.getLogger('th_lbfgs.py').setLevel(logging.DEBUG)
    D = 10**2
    N = int(1.1E2)


    X = rng.randint(2, size=(N, D)) * 2 - 1
    X = th_double_var(X)

    estimator = HOLIGlass(params=['J_glass', 'b'])
    estimator.learn(X)
    print(estimator.get_frac_capacity(X, 0.00))
    print(estimator.get_frac_capacity(X, 0.01))

    estimator = HOLIGlass(M=[], params=['J_glass', 'b'])
    estimator.learn(X)
    print(estimator.get_frac_capacity(X, 0.00))
    print(estimator.get_frac_capacity(X, 0.01))
