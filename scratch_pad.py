import numpy as np
import torch
from torch.autograd import Variable

for i in range(3):
    try:
        print(1/(i * (i-1)))
        break
    except:
        print(f'Attempt {i} failed')
else:
    print('asdfasd')
