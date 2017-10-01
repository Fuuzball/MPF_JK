import matplotlib.pylab as plt
from mpf_spin_glass import MPF_Glass
from sklearn.decomposition import PCA
import numpy as np

J = np.load('./J.npy')
b = np.load('./b.npy')
history = np.load('./history.npy')

pca = PCA(n_components=2)
history_pca = pca.fit_transform(history)
#plt.plot(history_pca[:,0], history_pca[:,1])
#plt.show()
XY = np.array(
        np.meshgrid(
            np.arange(-5, 22),
            np.arange(-8, 10)
            )
        )

_, H, W = XY.shape
XY_list = (XY.T).reshape(-1, 2)
mpf = MPF_Glass(
        pca.inverse_transform(XY_list)
        )
XY_energy = mpf.energy(J, b)
XY_energy = (XY_energy.reshape((W, H))).T
plt.imshow(XY_energy)
plt.show()
