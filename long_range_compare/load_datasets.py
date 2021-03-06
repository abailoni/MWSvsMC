import vigra
import numpy as np
import os

import h5py

from segmfriends.utils import yaml2dict, parse_data_slice
from .data_paths import get_hci_home_path, get_trendytukan_drive_path
from segmfriends.io.load import parse_offsets

def get_dataset_data(dataset='CREMI', sample=None, crop_slice_str=None, run_connected_components=True,
                     affs_path="SOA"):
    # FIXME: get rid of this hack
    assert dataset in ['ISBI', 'CREMI', 'CREMI-emb'], "Only Cremi and ISBI datasets are supported"
    if crop_slice_str is not None:
        crop_slice = tuple(parse_data_slice(crop_slice_str))
        assert len(
            crop_slice) == 4, "The passed slice should have 4 dimensions (including the offset dim. for affinities)"
    else:
        crop_slice = tuple([slice(None) for _ in range(4)])

    home_path = get_hci_home_path()

    if dataset in ['CREMI', 'CREMI-emb']:
        # -----------------
        # Load CREMI dataset:
        # -----------------
        if sample in ["A", "B", "C"]:
            cremi_path = os.path.join(home_path, "datasets/cremi/SOA_affinities/")
            dt_path = os.path.join(cremi_path, "sample{}_train.h5".format(sample))
            # FIXME: what is this shit?
            inner_path_GT = "segmentations/groundtruth_fixed_OLD" if sample == "B" else "segmentations/groundtruth_fixed"
            # inner_path_GT = "segmentations/groundtruth_fixed"
            inner_path_raw = "raw_old" if sample == "B" else "raw"
            inner_path_affs = "predictions/full_affs"
        elif sample in ["A+", "B+", "C+"]:
            dt_path = os.path.join(get_trendytukan_drive_path(),
                             "datasets/CREMI/constantin_affs/test_samples/sample{}_cropped_plus_mask.h5".format(sample))
            inner_path_affs = "affinities"
            inner_path_GT = "volumes/labels/mask"
            inner_path_raw = 'volumes/raw'
        else:
            raise ValueError
        with h5py.File(dt_path, 'r') as f:
            GT = f[inner_path_GT][crop_slice[1:]]
            if affs_path == "SOA":
                affs = f[inner_path_affs][crop_slice]
                # FIXME: convert to float32 and invert
                # affs = 1. - affs.astype('float32') / 255.
            else:
                # FIXME: generalize the path/sample...
                with h5py.File(os.path.join(affs_path, "predictions_sample_{}.h5".format(sample)), 'r') as f_affs:
                    affs = f_affs['data'][crop_slice].astype(np.float32)
            if sample in ["A+", "B+", "C+"]:
                # FIXME: clean mask
                ignore_mask_border = GT > np.uint64(-10)
                GT[ignore_mask_border] = 0
                # FIXME: fix defected slices GT box
                if sample == "A+":
                    GT[52] = 1
                elif sample == "C+":
                    GT[75] = 1
                    GT[15] = 1

            # # ##################
            # assert dataset == "CREMI"
            # raw = f[inner_path_raw][crop_slice[1:]]
            # path = os.path.join(get_hci_home_path(),
            #                     "datasets/cremi/tmp_cropped_train_data/compressed/sample_{}.hdf".format(sample))
            # print(path)
            # print("Raw type: ", raw.dtype)
            # vigra.writeHDF5(affs.astype('float16'), path, "affinities", compression='gzip')
            # print("affs wrote")
            # vigra.writeHDF5(GT.astype('uint32'), path, "gt", compression='gzip')
            # vigra.writeHDF5(raw, path, "raw", compression='gzip')
            # # ########################

    elif dataset == 'ISBI':
        # -----------------
        # Load ISBI dataset:
        # -----------------
        isbi_dir_path = os.path.join(get_trendytukan_drive_path(), "datasets/ISBI/MWS_affs/")
        GT = None
        if sample == "test_ISBI":
            filename = "isbi_test_offsetsV4_3d_meantda_damws2deval_final.h5"
            GT = vigra.readHDF5(os.path.join(isbi_dir_path, "mst_isbi_test_offsetsV4_3d_meantda_damws2deval_final.h5"), 'data')
            # GT = np.transpose(GT, (2, 1, 0))[crop_slice[1:]]
        elif sample == "train_ISBI":
            filename = "isbi_train_offsetsV4_3d_meantda_damws2deval_final.h5"
            GT = vigra.readHDF5(os.path.join(isbi_dir_path, "../gt_cleaned.h5"), 'data')
            GT = np.transpose(GT, (2, 1, 0))[crop_slice[1:]]
        else:
            raise NotImplementedError()
        affs = 1 - vigra.readHDF5(
            os.path.join(isbi_dir_path, filename), 'data')[
            crop_slice]
        # raw = io.imread(os.path.join(isbi_path, "train-volume.tif"))
        # raw = np.array(raw)[crop_slice[1:]]
        # gt_3D = vigra.readHDF5(os.path.join(isbi_path, "gt_mc3d.h5"), 'data')
    else:
        raise NotImplementedError

    if crop_slice_str is not None and run_connected_components:
        GT = vigra.analysis.labelVolumeWithBackground(GT.astype('uint32'))

    return affs, GT


def get_dataset_offsets(dataset='CREMI'):
    # TODO: I know, it's crap code..
    if dataset == "CREMI":
        offset_file = 'SOA_offsets.json'
    elif dataset == "ISBI":
        offset_file = 'offsets_MWS.json'
    elif dataset == 'CREMI-emb':
        offset_file = 'offsets_embeddings.json'
    offset_file = os.path.join(get_hci_home_path(), 'pyCharm_projects/hc_segmentation/experiments/postprocessing/cremi/offsets/', offset_file)
    return parse_offsets(offset_file)



def get_GMIS_dataset(type='val', partial=False):
    some_img = ['frankfurt_000001_016273_leftImg8bit', 'frankfurt_000001_013016_leftImg8bit', 'frankfurt_000001_030310_leftImg8bit', 'frankfurt_000001_080830_leftImg8bit', 'frankfurt_000001_028335_leftImg8bit', 'frankfurt_000001_042384_leftImg8bit', 'frankfurt_000001_064798_leftImg8bit', 'frankfurt_000001_034047_leftImg8bit', 'frankfurt_000000_013067_leftImg8bit', 'frankfurt_000001_010444_leftImg8bit', 'lindau_000024_000019_leftImg8bit', 'lindau_000034_000019_leftImg8bit', 'lindau_000026_000019_leftImg8bit', 'lindau_000044_000019_leftImg8bit', 'lindau_000011_000019_leftImg8bit', 'lindau_000042_000019_leftImg8bit', 'lindau_000018_000019_leftImg8bit', 'lindau_000027_000019_leftImg8bit', 'lindau_000036_000019_leftImg8bit', 'lindau_000030_000019_leftImg8bit', 'munster_000001_000019_leftImg8bit', 'munster_000080_000019_leftImg8bit', 'munster_000069_000019_leftImg8bit', 'munster_000155_000019_leftImg8bit', 'munster_000100_000019_leftImg8bit', 'munster_000046_000019_leftImg8bit', 'munster_000167_000019_leftImg8bit', 'munster_000153_000019_leftImg8bit', 'munster_000112_000019_leftImg8bit', 'munster_000030_000019_leftImg8bit']

    some_img = ['frankfurt_000001_020693_leftImg8bit']
    # some_img = ['frankfurt_000001_016273_leftImg8bit']
    # some_img = ['frankfurt_000001_016273_leftImg8bit0_01']
    # "000001_020693"

    if type == "val":
        root_dir = os.path.join(get_trendytukan_drive_path(), "GMIS_predictions/{}/temp_ram".format(type))
    elif type == "test":
        root_dir = os.path.join(get_trendytukan_drive_path(), "../quadxeon5_scratch/projects/GASP_cityscapes/test/temp_ram")
    else:
        root_dir = os.path.join(get_hci_home_path(), "GMIS_predictions/{}/temp_ram".format(type))
    all_file_paths = []
    for subdir, dirs, files in os.walk(root_dir):
        for dir in dirs:
            for _, _, subdir_files in os.walk(os.path.join(subdir, dir)):
                for i, filename in enumerate(subdir_files):
                    if not filename.endswith("input.h5") or filename.startswith("."):
                        continue
                    if partial:
                        check = False
                        for accepted_img in some_img:
                            if filename.startswith(accepted_img):
                                check = True
                                break
                        if not check:
                            continue
                    all_file_paths.append(os.path.join(subdir, dir, filename))
    return all_file_paths


CREMI_crop_slices = {
    "A": [    ":, 0: 31,:1300, -1300:",
            ":, 31: 62,:1300, -1300:",
            ":, 62: 93, 25: 1325,:1300",
            ":, 93: 124, -1300:,:1300",
            ":, :,:,:",
              ":,6:20, 120:-120, 120:-120",
              ":,8:-6, 146:-120, 146:-146"
              ],
    "B": [
            ":, 0: 31, 50: 1350, 200: 1500",
            ":, 31: 62, 20: 1320, 400: 1700",
            ":, 62: 93, 90: 1390, 580: 1880",
            ":, 93: 124, -1300:, 740: 2040",
            ":, :, 90:, 580: 1900",
            ":, 0:90, 90:1320, 580: 1500",
            # ":,13:17,110:560,270:720"
            ":,33:38,:1050,:1050",
        ":,6:20, 120:-120, 120:-120",
        ":,8:-6, 146:-120, 146:-146"

    ],
    "C": [
            ":, 0: 31, -1300:,:1300",
            ":, 31: 62, 150: 1450, 95: 1395",
            ":, 62: 93, 70: 1370, 125: 1425",
            ":, 93: 124,:1300, -1300:",
            ":, :, 70:1450, 95:1425",
        ":,6:20, 120:-120, 120:-120",
        ":,8:-6, 146:-120, 146:-146"

    ],
    "A+": [":, :, :, :"],
    "B+": [":, :, :, :"],
    "C+": [":, :, :, :"],
    "train_ISBI": [":, :, :, :"],
    "test_ISBI": [":, :, :, :"],
}

CREMI_sub_crops_slices = [":,2:, 100:600, 100:600",
                    ":,2:, 100:600, 600:1100",
                    ":,2:, 600:1100, 600:1100",
                    ":,2:, 600:1100, 100:600",
                    ":,2:, 100:1100, 100:1100",
                    ":,:, 200:500, 200:500",
                    # ":,:, 200:500, 200:500",
                    ":,:, :, :",
                    ":,14:15, 300:900, 300:900",
                    ":,:30, 200:500, 200:500",
                    ]

# crops_padded_volumes = {
# "A+": (slice(36, 163, None), slice(1154, 2753, None), slice(934, 2335, None)),
# "B+": (slice(36, 163, None), slice(1061, 2802, None), slice(1254, 4009, None)),
# "C+": (slice(36, 163, None), slice(980, 2443, None), slice(1138, 2569, None))
# }
crops_padded_volumes = {
"A+": (slice(36, 163, None), slice(1154, 2753, None), slice(934, 2426, None)),
"B+": (slice(36, 163, None), slice(1055, 2802, None), slice(1254, 4105, None)),
"C+": (slice(36, 163, None), slice(979, 2450, None), slice(1138, 2659, None))
}


shape_padded_aligned_datasets = {
    "A+": (200, 3727, 3505),
    "B+": (200, 3832, 5455),
    "C+": (200, 3465, 3668)
}
