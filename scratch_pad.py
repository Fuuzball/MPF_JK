import numpy as np
import torch
from torch.autograd import Variable
rng = np.random.RandomState()

D = 10**2
N = 1

X = rng.randint(2, size=(N, D)) * 2 - 1

X_t = Variable(torch.from_numpy(X).type(torch.cuda.DoubleTensor))
