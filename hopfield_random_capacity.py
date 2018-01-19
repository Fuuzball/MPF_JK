import numpy as np
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original') 
from hopfield_network import HopfieldNet, HopfieldNetJK, HopfieldNet4
import matplotlib.pylab as plt
np.random.seed(2)

def stack_X(X, ratio = 1.5, pad = 1):
    N, Dx, Dy = X.shape
    W = int(np.ceil(np.sqrt(ratio * N)))
    H = int(np.ceil(N / W))

    if H * W > N:
        X = np.concatenate((X, np.zeros((H * W - N, Dx, Dy))))

    padX = np.pad(X, ((0,0), (pad,pad), (pad,pad)), 'constant', constant_values = 0.5)
    rows = []
    for i in range(H):
        rows.append(np.hstack((padX[i*W:(i+1)*W])))
    Xstack = np.vstack(rows)
    return Xstack

D = 10**2

n = 15
p_list = []
p_list_4 = []
p_list_opr = []
p_list_4_opr = []
p = 0.40
p = 0.3
frac = 0.2

N_max = 0.10 * D**2
N_max = 300
n_N = 30
d_N = np.log(N_max)/ n_N
print(d_N)
s_list = np.arange(0, np.log(N_max), d_N)
n_list = np.array(np.exp(s_list), dtype = int)
n_list = np.arange(1, 200, 5)
print(n_list)
#n_list = range(int(0*D), int(2 * D))
#n_list = range(100, 110)

random_X = (np.random.random((20, D)) > p) * 2 - 1

if False:
    for n in n_list:
        print(n)

        random_X = (np.random.random((n+1, D)) > p) * 2 - 1

        hopfield_opr = HopfieldNet(random_X)
        hopfield_opr.learn_Jb(method='opr')

        hopfield_mpf = HopfieldNet(random_X)
        hopfield_mpf.learn_Jb()

        hopfield_4_opr = HopfieldNet4(random_X)

        hopfield_4 = HopfieldNetJK(random_X)
        hopfield_4.learn_JbK()

        X_corrupt = hopfield_opr.corrupt_memory(frac=frac)
        X_conv_opr = np.zeros_like(random_X)
        X_conv_mpf = np.zeros_like(random_X)
        X_conv_opr_4 = np.zeros_like(random_X)
        X_conv_mpf_4 = np.zeros_like(random_X)

        for n, x in enumerate(X_corrupt):
            X_conv_opr[n] = hopfield_opr.run_network(X0=x, output=False)
            X_conv_mpf[n] = hopfield_mpf.run_network(X0=x, output=False)
            X_conv_opr_4[n] = hopfield_4_opr.run_network(X0=x, output=False)
            X_conv_mpf_4[n] = hopfield_4.run_network(X0=x, output=False)

        p_list.append(np.all(X_conv_mpf == random_X, axis=1).mean())
        p_list_opr.append(np.all(X_conv_opr == random_X, axis=1).mean())
        p_list_4.append(np.all(X_conv_mpf_4 == random_X, axis=1).mean())
        p_list_4_opr.append(np.all(X_conv_opr_4 == random_X, axis=1).mean())



    #plt.plot(np.arange(D, 2*D), p_list)
    print('p = ', p)
    print('frac = ', frac)
    print('--'*20)
    print('4th OPR')
    print(p_list_4_opr)
    print('2nd OPR')
    print(p_list_opr)
    print('2nd MPF')
    print(p_list)
    print('2nd MPF')
    print(p_list_4)
    plt.plot(n_list, p_list)
    plt.plot(n_list, p_list_4)
    plt.plot(n_list, p_list_opr)
    plt.plot(n_list, p_list_4_opr)
    plt.show()

if True:
    for n in n_list:
        print(n)

        mnist_bin = np.array(mnist.data > 128, dtype=np.int8)
        mnist_bin = mnist_bin * 2 -1

        N_mnist, D = mnist_bin.shape 
        idx = np.random.choice(N_mnist, n, replace=False)
        random_X = mnist_bin[idx]

        random_X = (np.random.random((n+1, D)) > p) * 2 - 1

        hopfield_4_opr = HopfieldNet4(random_X)

        hopfield_4 = HopfieldNetJK(random_X)
        hopfield_4.learn_JbK()

        hopfield = HopfieldNet(random_X)
        hopfield.learn_Jb()

        hopfield_opr = HopfieldNet(random_X)
        hopfield_opr.learn_Jb(method='opr')

        p_list_4_opr.append( (hopfield_4_opr.is_local_min()).mean() )
        p_list_4.append( (hopfield_4.is_local_min()).mean() )
        p_list.append( (hopfield.is_local_min()).mean() )
        p_list_opr.append( (hopfield_opr.is_local_min()).mean() )


    #plt.plot(np.arange(D, 2*D), p_list)
    print('4th OPR')
    print(p_list_4_opr)
    print('2nd OPR')
    print(p_list_opr)
    print('2nd MPF')
    print(p_list)
    print('2nd MPF')
    print(p_list_4)
    plt.plot(n_list, p_list)
    plt.plot(n_list, p_list_4)
    plt.plot(n_list, p_list_opr)
    plt.plot(n_list, p_list_4_opr)
    plt.show()

