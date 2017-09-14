from scipy.io import loadmat
import matplotlib.pylab as plt
import numpy as np
import os

image_fnames = [f for f in os.listdir('./geisler') if '.B1' in f]

img_h = 2844
img_w = 4284
n_files = 50

def write_images_npz(nfiles=n_files):
    image_fnames_s = image_fnames[:nfiles]
    img_n = len(image_fnames_s)
    image_arr = np.zeros((img_n, img_h, img_w), dtype=np.int8)
    for i, f in enumerate(image_fnames_s):
        print(i)
        fname = os.path.join('./geisler', f)
        M = np.array(loadmat(fname)['B'])
        image_arr[i] = 2 * M - 1

    print(image_arr)
    np.savez_compressed('./geisler/images', image_arr)

#write_images_npz()
print('finished writing images')

patch_size = 18

n_h_split = 2844 // patch_size
n_w_split = 4284 // patch_size
n_patches = n_h_split * n_w_split

patches = []

image_arr = np.load('./geisler/images.npz')['arr_0']

if True:
    for i, image in enumerate(image_arr):
        print(i)
        img_patches = [a for m in np.split(image, n_h_split) for a in np.split(m, n_w_split, axis=1)]
        patches.extend(
                [a for m in np.split(image, n_h_split) for a in np.split(m, n_w_split, axis=1)]
                )

    patches_arr = np.array(patches) 
    np.savez_compressed('./geisler/patches', patches_arr) 
