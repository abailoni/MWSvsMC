import long_range_compare # Add missing package-paths
from long_range_compare.data_paths import get_trendytukan_drive_path, get_hci_home_path

import vigra
import numpy as np
import os

import sys
sys.path += [
os.path.join(get_hci_home_path(), "python_libraries/cremi_tools"),]


"""
Prepare segmentation for submission
"""


original_pad = ((37, 38), (911, 911), (911, 911))

original_pad = ((37, 38), (911, 911), (911, 911))
slice_original_pad = (slice(37, -38), slice(911, -911), slice(911, -911))
padded_shape = (200, 3072, 3072)

sample = "A"


# raw_file = os.path.join(get_trendytukan_drive_path(), "datasets/CREMI/official_test_samples/sample_{}+_padded_20160601.hdf".format(sample))
#
# import h5py
# GT_mask_file = os.path.join(get_trendytukan_drive_path(), "datasets/CREMI/alignment_experiments/sample_{}+_GT_mask.hdf".format(sample))
# mask_inner_path = "volumes/labels/mask"
# GT_box = np.zeros(padded_shape, dtype="uint8")
# GT_box[slice_original_pad] = 1
#
# with h5py.File(GT_mask_file, 'w') as f:
#     f[mask_inner_path] = GT_box
#
from cremi_tools.alignment import backalign_segmentation

# mask_GT_path = os.path.join(get_trendytukan_drive_path(), "datasets/CREMI/alignment_experiments/sample_{}+_aligned.hdf".format(sample))
mask_GT_path = os.path.join(get_hci_home_path(), "sampleA+_gt.h5")
# mask_GT = vigra.readHDF5(mask_GT_path, "volumes/labels/mask")

out_file = os.path.join(get_trendytukan_drive_path(), "datasets/CREMI/alignment_experiments/sample_{}+_backaligned.hdf".format(sample))
backalign_segmentation("A+", mask_GT_path, out_file,
                           key="segmentation",
                           postprocess=False)

