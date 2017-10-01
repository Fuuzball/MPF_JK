import numpy as np
from hopfield_network import HopfieldNet, HopfieldNetJK
import matplotlib.pylab as plt
np.random.seed(2)

D = 64

n = 15
p_list = []
p_list_4 = []
p = 0.5
n_list = range(int(1*D), int(2 * D))
n_list = range(100, 110)
for n in n_list:
    print(n)
    #random_X = np.random.randint(2, size=(n, D)) * 2 - 1
    random_X = (np.random.random((n+1, D)) > p) * 2 - 1
    #random_X[:, -1] = 1
    #hopfield = HopfieldNet(random_X)
    #hopfield.learn_Jb(method='mpf')
    hopfield_4 = HopfieldNetJK(random_X)
    hopfield_4.learn_JbK()
    hopfield = HopfieldNet(random_X)
    hopfield.learn_Jb()
    p_list_4.append( (hopfield_4.is_local_min()).mean() )
    p_list.append( (hopfield.is_local_min()).mean() )


#plt.plot(np.arange(D, 2*D), p_list)
plt.plot(n_list, p_list)
plt.plot(n_list, p_list_4)
plt.show()

