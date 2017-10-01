import numpy as np
from hopfield_network import HopfieldNet
import matplotlib.pylab as plt
np.random.seed(2)

N = 20
D = 100
random_X = np.random.randint(2, size=(N, D)) * 2 - 1

n = 15
p_list = []
p = 0.5
for n in range(200):
    print(n)
    #random_X = np.random.randint(2, size=(n, D)) * 2 - 1
    random_X = (np.random.random((n+1, D)) > p) * 2 - 1
    #random_X[:, -1] = 1
    hopfield = HopfieldNet(random_X)
    hopfield.learn_Jb(method='opr')
    p_list.append( (hopfield.is_local_min()).mean() )

if False:
    p_list_mpf = []
    p = 0.5
    for n in range(200):
        print(n)
        #random_X = np.random.randint(2, size=(n, D)) * 2 - 1
        random_X = (np.random.random((n+1, D)) > p) * 2 - 1
        #random_X[:, -1] = 1
        hopfield = HopfieldNet(random_X)
        hopfield.learn_Jb(method='mpf')
        p_list_mpf.append( (hopfield.is_local_min()).mean() )

plt.plot(p_list)
plt.show()

