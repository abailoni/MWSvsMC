import long_range_compare # Add missing package-paths
from long_range_compare.data_paths import get_trendytukan_drive_path, get_hci_home_path

"""
Align dataset for better prediction - STEP 1:

This script get the original padded raw data from CREMI test and then:

- create a file sample_{sample}_GT_mask.hdf with a (slightly expanded) box surrounding the GT box
- saves an aligned version of the raw (together with the aligned version of the GT box) in sample_{sample}_aligned.hdf

"""


import vigra
import numpy as np
import os

import sys
sys.path += [
os.path.join(get_hci_home_path(), "python_libraries/cremi_tools"),]


# original_pad = ((37, 38), (911, 911), (911, 911))

# This specify how big is the padding of the padded raw (inside this we have actual GT):
slice_original_pad = (slice(37, -38), slice(911, -911), slice(911, -911))
# -------------
# This specify how much pad we want to predict  
# -------------
# Original pad for GASP paper:
# slice_GT_mask = (slice(36, -37), slice(890, -890), slice(890, -890))

# # With the embedding we need at least a padding of (2, 240, 240) in the original res (big_pad_version)
# slice_GT_mask = (slice(32, -33), slice(580, -580), slice(580, -580))

# This is almost just a box around the raw data (raw_mask)
slice_GT_mask = (slice(5, -5), slice(180, -180), slice(180, -180))

padded_shape = (200, 3072, 3072)

for sample in ["A+", "B+", "C+"]:

    # test_sample_path = os.path.join(get_trendytukan_drive_path(), "datasets/CREMI/constantin_affs/test_samples/sample{}.h5".format(sample))

    raw_file = os.path.join(get_trendytukan_drive_path(), "datasets/CREMI/official_test_samples/sample_{}_padded_20160601.hdf".format(sample))
    out_file = os.path.join(get_trendytukan_drive_path(), "datasets/CREMI/official_test_samples/full_aligned_samples/sample_{}_aligned_plus_raw_mask.hdf".format(sample))

    import h5py
    GT_mask_file = os.path.join(get_trendytukan_drive_path(), "datasets/CREMI/official_test_samples/full_aligned_samples/sample_{}_GT_mask_temp.hdf".format(sample))
    mask_inner_path = "volumes/labels/mask"
    GT_box = np.zeros(padded_shape, dtype="uint32")
    GT_box[slice_GT_mask] = 1

    from segmfriends.utils.various import writeHDF5
    writeHDF5(GT_box, GT_mask_file, mask_inner_path)

    from cremi_tools.alignment import realign


    # GT_mask_file = os.path.join(get_trendytukan_drive_path(), "datasets/CREMI/alignment_experiments/sample_{}_backaligned.hdf".format(sample))

    realign(raw_file,
                sample,
                out_file,
                labels_file=GT_mask_file,
                labels_key=mask_inner_path)

    os.remove(GT_mask_file)
