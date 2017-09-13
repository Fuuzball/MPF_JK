from scipy.io import loadmat
import matplotlib.pylab as plt
import numpy as np
import os

image_fnames = [f for f in os.listdir('./geisler') if '.B1' in f]

img_h = 2844
img_w = 4284
n_files = 50

def write_images_npz(nfiles=n_files):
    image_fnames = image_fnames[:nfiles]
    img_n = len(image_fnames)
    image_arr = np.zeros((img_n, img_h, img_w), dtype=np.uint8)
    for i, f in enumerate(image_fnames):
        print(i)
        fname = os.path.join('./geisler', f)
        M = np.array(loadmat(fname)['B'])
        image_arr[i] = M

    print(image_arr)
    np.savez_compressed('./geisler/images', image_arr)

patch_size = 18

n_h_split = 2844 // patch_size
n_w_split = 4284 // patch_size
n_patches = n_h_split * n_w_split

patches = []

image_arr = np.load('./geisler/images.npz')['arr_0']

image = image_arr[0]
img_patches = np.array( 
            [a for m in np.split(image, n_h_split) for a in np.split(m, n_w_split, axis=1)]
        )
print(img_patches.mean())

if False:
    for image in image_arr:
        print('-'*10)
        img_patches = [a for m in np.split(image, n_h_split) for a in np.split(m, n_w_split, axis=1)]
        print(img_patches[2])
        patches.extend(
                [a for m in np.split(image, n_h_split) for a in np.split(m, n_w_split, axis=1)]
                )

    patches_arr = np.array(patches) 
    np.savez_compressed('./geisler/patches', patches_arr) 
