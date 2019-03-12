import long_range_compare # Add missing package-paths
from long_range_compare.data_paths import get_trendytukan_drive_path, get_hci_home_path

"""
Align dataset for better prediction
"""


import vigra
import numpy as np
import os
import h5py

import sys
sys.path += [
os.path.join(get_hci_home_path(), "python_libraries/cremi_tools"),]


def get_gt_bounding_box(gt):

    # no-label ids are <0, i.e. the highest numbers in uint64
    fg_indices = np.where(gt == 1)
    return tuple(
        slice(np.min(fg_indices[d]),np.max(fg_indices[d])+1)
        for d in range(3)
    )



# original_pad = ((37, 38), (911, 911), (911, 911))


# FOUND CROP SLICES:
# A+ (slice(36, 163, None), slice(1154, 2753, None), slice(934, 2335, None))
# B+ (slice(36, 163, None), slice(1061, 2802, None), slice(1254, 4009, None))
# C+ (slice(36, 163, None), slice(980, 2443, None), slice(1138, 2569, None))

for sample in ["A+", "B+", "C+"]:

    affs_path = os.path.join(get_trendytukan_drive_path(), "datasets/CREMI/constantin_affs/test_samples/sample{}.h5".format(sample))
    affs_inner_path = "affinities"

    # Load GT mask:
    print("Loading")
    mask_inner_path = "volumes/labels/mask"
    source_path = os.path.join(get_trendytukan_drive_path(),
                                "datasets/CREMI/alignment_experiments/sample_{}_aligned.hdf".format(sample))
    with h5py.File(source_path, 'r') as f:
        mask_GT = f[mask_inner_path][:]

    # print(mask_GT.shape)
    print("Find crop")
    crop_slice = get_gt_bounding_box(mask_GT)
    print(crop_slice)

    # Write affs and mask in target file:
    print("Saving")
    target_path = os.path.join(get_trendytukan_drive_path(),
                             "datasets/CREMI/constantin_affs/test_samples/sample{}_cropped_plus_mask.h5".format(sample))

    with h5py.File(affs_path, 'r') as f:
        affs_crop_slice = (slice(None), ) + crop_slice
        affs = f[affs_inner_path][affs_crop_slice]

    with h5py.File(source_path, 'r') as f:
        raw = f["volumes/raw"][crop_slice]


    with h5py.File(target_path, 'w') as f:
        f[mask_inner_path] = mask_GT[crop_slice]
        # TODO: save crop slice!
        # f["crop_slice"] = crop
        f["affinities"] = affs
        f["volumes/raw"] = raw
