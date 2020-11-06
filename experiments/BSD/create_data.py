import long_range_compare

from long_range_compare.data_paths import get_trendytukan_drive_path, get_abailoni_hci_home_path
import numpy as np
import os

from imageio import imread

dataset_path = os.path.join(get_abailoni_hci_home_path(), "projects/GASP_on_BSD/images")

def compute_affs(image):
    img = imread(os.path.join(dataset_path, image))
    img = img[..., 0] / 255.

    # Decide if to rotate the image:
    rotate = img.shape[0] > img.shape[1]
    if rotate:
        img = np.rot90(img, -1)

    eps = 0.01

    affinities = np.ones((2, int(img.shape[0] / 2), int(img.shape[1] / 2)))

    import scipy.ndimage
    filtered = scipy.ndimage.uniform_filter(img, size=(1, 2), origin=(0, -1))
    l1, l2 = filtered[::2, ::2], filtered[1::2, ::2]
    affinities[0] = np.maximum(l1, l2)

    filtered = scipy.ndimage.uniform_filter(img, size=(2, 1), origin=(-1, 0))
    l1, l2 = filtered[::2, ::2], filtered[::2, 1::2]
    affinities[1] = np.maximum(l1, l2)

    affinities = np.clip(affinities, eps, 1. - eps)
    return affinities, rotate

nb_collected = []
affs_collected = []
rotated_collected = []
for filename in os.listdir(dataset_path):
    if filename.endswith(".ppm"):
        nb_collected.append(int(filename.replace(".jpg_edges.ppm", "")))
        # print(nb_collected[-1])
        affs, rotated = compute_affs(filename)
        affs_collected.append(affs)
        rotated_collected.append(rotated)
        print(affs_collected[-1].shape)

    # offsets = [
    #     [1, 0],
    #     [0, 1]
    # ]
    #
    # def convert_to_weights(edges, bias=0.5):
    #     beta = np.log(bias / (1 - bias))
    #     z = np.log(edges / (1 - edges))
    #     return 1. / (1. + np.exp(-z - beta))
    #
    # convert_to_weights(0.5, 0.9)

from segmfriends.utils import writeHDF5

output_dataset = os.path.join(dataset_path, "affinites.h5")
writeHDF5(np.stack(affs_collected), output_dataset, "affs")
writeHDF5(np.stack(nb_collected), output_dataset, "numbers")
writeHDF5(np.stack(rotated_collected), output_dataset, "rotated")


# filtered = scipy.ndimage.uniform_filter(img, size=(2,1))
# l1, l2, l3, l4 = filtered[::2,:-2:4], filtered[::2, 1:-2:4], filtered[::2, 2::4], filtered[::2, 3::4]
# print(np.where(l3>l4,l3,0).shape)
# check = eps + np.maximum(np.where(l1<l2,l2,0), np.where(l3>l4,l3,0))
# print(check.max())
# print(check.min())
# print(check.shape)
# import imageio
# imageio.imwrite("100007_x.png", biased_affs[0])
# imageio.imwrite("100007_y.png", biased_affs[1])


