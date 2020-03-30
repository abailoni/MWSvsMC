import long_range_compare # Add missing package-paths
from long_range_compare.data_paths import get_trendytukan_drive_path, get_hci_home_path

"""
This is a modified version of part 2 to downscale the whole aligned data without cropping it at all
"""

downscale = True
include_affs = False

from scipy.ndimage import zoom

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

POSTFIX = "_no_crop"

# Weird defects to be blacked out in the uncropped version:
blacked_out = {"A+": [11, 25, 37, 70],
               "B+": [18],
              "C+": [123]}

# original_pad = ((37, 38), (911, 911), (911, 911))


# FOUND CROP SLICES:
# A+ (slice(36, 163, None), slice(1154, 2753, None), slice(934, 2335, None))
# B+ (slice(36, 163, None), slice(1061, 2802, None), slice(1254, 4009, None))
# C+ (slice(36, 163, None), slice(980, 2443, None), slice(1138, 2569, None))

# for sample in ["A+"]:
for sample in ["A+", "B+", "C+"]:

    # Load GT mask:
    print("Loading")
    mask_inner_path = "volumes/labels/mask"
    # source_path_big_pad = os.path.join(get_trendytukan_drive_path(),
    #                             "datasets/CREMI/official_test_samples/full_aligned_samples/sample_{}_aligned_plus_big_pad.hdf".format(sample))
    source_path_raw_mask = os.path.join(get_trendytukan_drive_path(),
                                       "datasets/CREMI/official_test_samples/full_aligned_samples/sample_{}_aligned_plus_raw_mask.hdf".format(
                                           sample))
    # source_path = os.path.join(get_trendytukan_drive_path(),
    #                             "datasets/CREMI/official_test_samples/full_aligned_samples/sample_{}_aligned.hdf".format(sample))
    from segmfriends.utils.various import readHDF5, writeHDF5
    print("Reading...")
    mask_big_pad = readHDF5(source_path_raw_mask, mask_inner_path)
    print("Max big pad: ", mask_big_pad.max())
    # mask_border = mask_big_pad > 10
    mask_big_pad = (mask_big_pad == 1).astype('uint16')


    # print(mask_GT.shape)
    print("Find crop")
    # crop_slice = get_gt_bounding_box(mask_big_pad)

    # Write crop_slice to file:

    # import csv
    # csv_file_path = os.path.join(get_hci_home_path(),
    #                          "datasets/CREMI/official_test_samples/cropped_aligned_samples/sample{}_cropped{}.csv".format(sample, POSTFIX))
    # with open(csv_file_path, mode='w') as f:
    #     employee_writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     for i in range(3):
    #         employee_writer.writerow([0, str(mask_big_pad.shape[i])])

    # print(crop_slice)

    # Write affs and mask in target file:
    print("Saving...")
    # target_path_old = os.path.join(get_hci_home_path(),
    #                            "datasets/CREMI/official_test_samples/cropped_aligned_samples/sample{}_cropped{}.h5".format(
    #                                sample, POSTFIX))
    target_path = os.path.join(get_hci_home_path(),
                             "datasets/CREMI/official_test_samples/cropped_aligned_samples/sample{}_cropped{}.h5".format(sample, POSTFIX))

    # if include_affs:
    #     affs_path = os.path.join(get_trendytukan_drive_path(), "datasets/CREMI/constantin_affs/test_samples/sample{}.h5".format(sample))
    #     affs_inner_path = "affinities"
    #     affs = readHDF5(affs_path, affs_inner_path, crop_slice=(slice(None), ) + crop_slice)
    #     writeHDF5(affs, target_path, "volumes/affinities")

    # raw = readHDF5(source_path, "volumes/raw")
    # # raw = readHDF5(target_path_old, "volumes/raw_2x")
    # if sample in blacked_out:
    #     for blk in blacked_out[sample]:
    #         print("blacking out ", blk)
    #         raw[blk] = 0


    # mask_gt = readHDF5(source_path, mask_inner_path, dtype="uint16", crop_slice=crop_slice)
    # writeHDF5(raw, target_path, "volumes/raw")
    # writeHDF5(mask_big_pad.astype('uint16'), target_path, "volumes/labels/mask_gt")
    # writeHDF5(mask_gt, target_path, "volumes/labels/mask_gt")

    if downscale:
        writeHDF5(zoom(mask_big_pad, (1, 0.5, 0.5), order=0), target_path, "volumes/labels/mask_raw_2x")
        # writeHDF5(zoom(raw, (1, 0.5, 0.5), order=3), target_path, "volumes/raw_2x")
        # writeHDF5(raw, target_path, "volumes/raw_2x")
