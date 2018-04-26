import numpy as np
import matplotlib.pylab as plt
import seaborn as ns
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

data = np.load('./data/random_capacity/OPR_d_4_p_03/frac_min.npy')

print(data.shape)
def get_threshold_line(thres):
    threshold_data = []
    for D_data in data:
        y = np.argmax(D_data < thres)
        threshold_data.append(y)

    return threshold_data



fig, ax = plt.subplots()
im = ax.imshow(data.T, cmap='inferno', origin='lower')
ax.set_title('Training with OPR (d = 4) on random data (p = 0.3)')
ax.set_xlabel('Dimension of network')
ax.set_ylabel('Number of patterns')

cmap = plt.get_cmap('viridis')
ax.plot(get_threshold_line(1), color='0', linewidth=1)
ax.plot(get_threshold_line(.1), color='.9', linewidth=1)
ax.plot(get_threshold_line(.5), color='.5', linewidth=1)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size='10%', pad=0.05)

fig.colorbar(im, cax=cax)
plt.show()
#plt.savefig('./figures/OPR_d_4_p_03.png')


