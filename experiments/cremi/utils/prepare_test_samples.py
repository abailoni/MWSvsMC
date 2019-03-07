import long_range_compare # Add missing package-paths
from long_range_compare.data_paths import get_trendytukan_drive_path, get_hci_home_path

"""
Align dataset for better prediction
"""


import vigra
import numpy as np
import os

import sys
sys.path += [
os.path.join(get_hci_home_path(), "python_libraries/cremi_tools"),]


original_pad = ((37, 38), (911, 911), (911, 911))

original_pad = ((37, 38), (911, 911), (911, 911))
slice_original_pad = (slice(37, -38), slice(911, -911), slice(911, -911))
padded_shape = (200, 3072, 3072)

sample = "A"

test_sample_path = os.path.join(get_trendytukan_drive_path(), "datasets/CREMI/constantin_affs/test_samples/sample{}+.h5".format(sample))

raw_file = os.path.join(get_trendytukan_drive_path(), "datasets/CREMI/official_test_samples/sample_{}+_padded_20160601.hdf".format(sample))
out_file = os.path.join(get_trendytukan_drive_path(), "datasets/CREMI/alignment_experiments/sample_{}+_aligned.hdf".format(sample))

import h5py
GT_mask_file = os.path.join(get_trendytukan_drive_path(), "datasets/CREMI/alignment_experiments/sample_{}+_GT_mask.hdf".format(sample))
mask_inner_path = "volumes/labels/mask"
GT_box = np.zeros(padded_shape, dtype="uint32")
GT_box[slice_original_pad] = 1

with h5py.File(GT_mask_file, 'w') as f:
    f[mask_inner_path] = GT_box

from cremi_tools.alignment import realign


GT_mask_file = os.path.join(get_trendytukan_drive_path(), "datasets/CREMI/alignment_experiments/sample_{}+_backaligned.hdf".format(sample))

realign(raw_file,
            "A+",
            out_file,
            labels_file=GT_mask_file,
            labels_key=mask_inner_path)