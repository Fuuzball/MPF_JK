import torch
from torch.autograd import Variable

J = Variable(torch.Tensor([1, 2, 3]))
print(J[0].shape)
