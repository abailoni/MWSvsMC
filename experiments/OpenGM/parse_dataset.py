import long_range_compare
import long_range_compare
from long_range_compare.data_paths import get_trendytukan_drive_path, get_abailoni_hci_home_path
import numpy as np
import os

from imageio import imread

from segmfriends.utils import writeHDF5, readHDF5, get_hdf5_inner_paths

dataset_path = os.path.join(get_abailoni_hci_home_path(), "projects/GASP_on_OpenGM/data")

file_path = os.path.join(dataset_path, "knott-3d-150/gm_knott_3d_032.h5")

data = []
print(get_hdf5_inner_paths(file_path, 'gm/header'))
for inner_path in ['ccm_relabeling', 'faces-in-subset', 'regions-in-subset']:
    data.append(readHDF5(file_path, inner_path))
    print(readHDF5(file_path, inner_path).shape)

extra_stuff = ['function-id-16006', 'header', 'numbers-of-states']

for inner_path_gm in ['factors']:
    print(inner_path_gm)
    factors = readHDF5(file_path, "gm/{}".format(inner_path_gm))
    print(readHDF5(file_path, "gm/{}".format(inner_path_gm)).shape)

print(factors.max())
print(factors.min())


# writeHDF5(np.stack(affs_collected), output_dataset, "affs")
# writeHDF5(np.stack(nb_collected), output_dataset, "numbers")
# writeHDF5(np.stack(rotated_collected), output_dataset, "rotated")


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


