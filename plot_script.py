import numpy as np
import matplotlib.pylab as plt
import seaborn as ns
from matplotlib import cm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

dir_names = ['OPR_d_4_p_05',
        'OPR_d_4_p_04',
        'OPR_d_4_p_03']
fnames = [ './data/random_capacity/{}/frac_min.npy'.format(d) for d in dir_names ]
opr_datas = [np.load(f) for f in fnames]
data = np.load('./data/random_capacity/OPR_d_4_p_03/frac_min.npy')
#data = np.load('./data/random_capacity/OPR/frac_min.npy')

def get_threshold_line(data, thres):
    threshold_data = []
    for D_data in data:
        where_smaller = D_data < thres
        if where_smaller.sum() > 0:
            y = np.argmax(where_smaller)
            threshold_data.append(y)
        else:
            #threshold_data.append(len(D_data))
            threshold_data.append(None)

    return threshold_data



fig = plt.figure(figsize=(8,4))

ps = ['0.3', '0.4', '0.5']
gs = gridspec.GridSpec(2, 3)
#fig, ax = plt.subplots()
colors = ['green', 'blue', 'red']
for i, data in enumerate(opr_datas):
    #ax = fig.add_subplot(1, 3, i+1)
    ax = plt.subplot(gs[0, i])
    im = ax.imshow(data.T, cmap='gray', origin='lower', aspect='auto')
    #ax.set_title('Training with OPR (d = 4) on random data (p = 0.3)')
    ax.set_title('p = {}'.format(ps[i]))
    ax.set_xlabel('Dimension of network')
    ax.set_ylabel('Number of patterns')
    ax.set_ylim((0,400))

    cmap = plt.get_cmap('viridis')
    ax.plot(get_threshold_line(data, .95), color=colors[i], linewidth=1)
    #ax.plot(get_threshold_line(.1), color='.9', linewidth=1)
    #ax.plot(get_threshold_line(.5), color='.5', linewidth=1)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size='10%', pad=0.05)

    fig.colorbar(im, cax=cax)

ax = plt.subplot(gs[1, :])
ax.plot(get_threshold_line(opr_datas[0], .95), color=colors[0], linewidth=1)
ax.plot(get_threshold_line(opr_datas[1], .95), color=colors[1], linewidth=1)
ax.plot(get_threshold_line(opr_datas[2], .95), color=colors[2], linewidth=1)
ax.plot()

plt.tight_layout()
plt.show()
#plt.savefig('./figures/OPR_d_4_p_03.png')


