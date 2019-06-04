import long_range_compare # Add missing package-paths

import vigra
import numpy as np
import os
import argparse
from multiprocessing.pool import ThreadPool
from itertools import repeat

from long_range_compare.data_paths import get_hci_home_path, get_trendytukan_drive_path
from long_range_compare import cremi_utils as cremi_utils
from long_range_compare import cremi_experiments as cremi_experiments

from segmfriends.utils.various import starmap_with_kwargs

from segmfriends import vis as vis_utils

import h5py

import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sample = "B+"
    # main_folder = "projects/agglo_cluster_compare/FullTestSamples/out_segms/"

    # mask_realigned = vigra.readHDF5("/mnt/localdata0/abailoni/datasets/CREMI/alignment_experiments/sample_{}_backaligned.hdf".format(sample), "volumes/labels/mask")
    dt_path = os.path.join(get_trendytukan_drive_path(), "datasets/CREMI/constantin_affs/test_samples/sample{}_cropped_plus_mask.h5".format(
        sample))
    with h5py.File(dt_path, 'r') as f:
        #     GT_mask = f["volumes/labels/mask"][:]
        raw = f["volumes/raw"][40:50, 400:750, 400:750]
        affs = 1. - f["affinities"][1:2, 40:50, 400:750, 400:750].astype('float32') / 255.

    mod_affs = cremi_utils.add_opensimplex_noise_to_affs(affs, scale_factor=8, mod="merge-biased", seed=223)
    mod_affs_split = cremi_utils.add_opensimplex_noise_to_affs(affs, scale_factor=8, mod="split-biased", seed=223)

    # for SLICE in range(10):

    SLICE = 2

    for k in range(4):
        # f, axes = plt.subplots(ncols=4, nrows=1, figsize=(12, 4))
        f, axes = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))

        for a in f.get_axes():
            a.axis('off')

        if k ==0:
            axes.matshow(raw[SLICE], cmap='gray', interpolation='none')
        if k == 1:
            axes.matshow(affs[0, SLICE], cmap=plt.get_cmap('seismic'), interpolation='none')

        if k == 2:
            axes.matshow(mod_affs[0, SLICE], cmap=plt.get_cmap('seismic'), interpolation='none')

        if k == 3:
            axes.matshow(mod_affs_split[0, SLICE], cmap=plt.get_cmap('seismic'), interpolation='none')

        plt.tight_layout()
        plot_dir = os.path.join(get_trendytukan_drive_path(), "projects/agglo_cluster_compare/noise_plots")
        f.savefig(os.path.join(plot_dir,
                               'noisy_affs_compare_2_{}.pdf'.format(k)),
                  format='pdf')



