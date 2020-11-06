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

for sample in ["A", "B", "C"]:

    # test_sample_path = os.path.join(get_trendytukan_drive_path(), "datasets/CREMI/constantin_affs/test_samples/sample{}.h5".format(sample))

    raw_file = os.path.join(get_trendytukan_drive_path(), "datasets/CREMI/padded_data/sample_{}_padded_20160501.hdf".format(sample))
    out_file = os.path.join(get_trendytukan_drive_path(), "datasets/CREMI/padded_data/full_aligned_samples/sample_{}_aligned_plus_raw_mask.hdf".format(sample))

    import h5py
    GT_file = os.path.join(get_trendytukan_drive_path(), "datasets/CREMI/padded_data/sample_{}_padded_20160501.hdf".format(sample))
    GT_inner_path = "volumes/labels/neuron_ids"

    from cremi_tools.alignment import realign

    realign(raw_file,
                sample,
                out_file,
                labels_file=GT_file,
                labels_key=GT_inner_path)

