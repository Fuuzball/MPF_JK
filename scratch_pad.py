import numpy as np

D_list = np.arange(10, 110, 10)
N_list = np.arange(10, 160, 10)
print(np.array(np.meshgrid(D_list, N_list)).T)
