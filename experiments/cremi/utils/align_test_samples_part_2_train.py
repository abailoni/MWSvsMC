import long_range_compare # Add missing package-paths
from long_range_compare.data_paths import get_trendytukan_drive_path, get_hci_home_path

"""
Align dataset for better prediction - STEP 2

- Load the raw/GT_box aligned in step 1
- Load SOA affinities (also predicted on the same aligned dataset)
- Find crop around (warped) GT box and save only this crop of raw/affs/GT_mask in sample{}_cropped_plus_mask.h5
"""

# FIXME: add blacking out defected slices

from scipy.ndimage import zoom

import vigra
import numpy as np
import os
import h5py

import sys
sys.path += [
os.path.join(get_hci_home_path(), "python_libraries/cremi_tools"),]

from segmfriends.utils.various import readHDF5, writeHDF5


def get_gt_bounding_box(gt):

    # no-label ids are <0, i.e. the highest numbers in uint64
    fg_indices = np.where(gt > 0)
    return list(
        [np.min(fg_indices[d]), np.max(fg_indices[d])+1]
        for d in range(3)
    )



# original_pad = ((37, 38), (911, 911), (911, 911))


# FOUND CROP SLICES:
# A+ (slice(36, 163, None), slice(1154, 2753, None), slice(934, 2335, None))
# B+ (slice(36, 163, None), slice(1061, 2802, None), slice(1254, 4009, None))
# C+ (slice(36, 163, None), slice(980, 2443, None), slice(1138, 2569, None))

# # Big pad blacked out:
funny_artifacts_GT = {"A": [],
              "C": [51, 111]}

padding = [6, 240, 240]

for sample in ["A", "B", "C"]:

    # Load GT mask:
    print("Loading sample ", sample)
    gt_inner_path = "volumes/labels/neuron_ids"
    source_path = os.path.join(get_trendytukan_drive_path(),
                                "datasets/CREMI/padded_data/full_aligned_samples/sample_{}_aligned_plus_raw_mask.hdf".format(sample))
    print("Reading...")
    GT = readHDF5(source_path, gt_inner_path, dtype='uint64')

    print(GT.max())
    print(GT.min())

    # Convert ignore label to zero:
    assert (GT == 0).sum() == 0, "Some label was already set to zero"
    GT[GT == GT.max()] = 0
    if sample in funny_artifacts_GT:
        for slc in funny_artifacts_GT[sample]:
            # TODO: here I should better save a defected slice in the extra slices
            GT[slc] = GT[slc-1]

    # print(mask_GT.shape)
    print("Find crop")
    crop_slice = get_gt_bounding_box(GT)

    crop_slice = tuple(slice(crp[0]-pad, crp[1]+pad) for crp, pad in zip(crop_slice, padding))




    # Write crop_slice to file:
    import csv
    csv_file_path = os.path.join(get_trendytukan_drive_path(),
                             "datasets/CREMI/padded_data/cropped_aligned_samples/sample_{}_crop.csv".format(sample))
    with open(csv_file_path, mode='w') as f:
        employee_writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(3):
            employee_writer.writerow([str(crop_slice[i].start), str(crop_slice[i].stop)])

    print(crop_slice)
    GT = GT[crop_slice]

    # Write affs and mask in target file:
    target_path = os.path.join(get_trendytukan_drive_path(),
                             "datasets/CREMI/padded_data/cropped_aligned_samples/sample_{}.h5".format(sample))
    target_path_2x = os.path.join(get_trendytukan_drive_path(),
                               "datasets/CREMI/padded_data/cropped_aligned_samples/sample_{}_2x.h5".format(sample))


    raw = readHDF5(source_path, "volumes/raw", crop_slice=crop_slice)
    print("Saving...")
    writeHDF5(GT, target_path, gt_inner_path)
    writeHDF5(raw, target_path, "volumes/raw")
    print("Downscaling and saving...")
    writeHDF5(zoom(GT, (1, 0.5, 0.5), order=0), target_path_2x, gt_inner_path)
    writeHDF5(zoom(raw, (1, 0.5, 0.5), order=3), target_path_2x, "volumes/raw")
