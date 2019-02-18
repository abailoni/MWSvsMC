# Add missing package-paths
from long_range_compare.data_paths  import get_hci_home_path, get_trendytukan_drive_path


import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import vigra
import nifty as nf
import nifty.graph.agglo as nagglo
import numpy as np
import os

from skimage import io
import argparse
import time
import yaml
import json
import os

import segmfriends.vis as vis_utils
from segmfriends.utils.config_utils import adapt_configs_to_model, recursive_dict_update, return_recursive_key_in_dict

from long_range_compare.load_datasets import CREMI_crop_slices, CREMI_sub_crops_slices


import matplotlib.animation as manimation




root_path = os.path.join(get_hci_home_path(), "GEC_comparison_longRangeGraph")

# file_names = ["9054435_B_sum_False", "9575382_B_max_False"]
file_names = ["998353761_B_sum_False", "229028997_B_max_False", "839260953_B_mean_False"]

SELECTED_OFFSET = 1
SELECTED_SLICE = 0
NB_FRAMES = 100

for filename in file_names:
    ucm_path = os.path.join(root_path, "UCM", filename + ".h5")
    config_path = os.path.join(root_path, filename + ".json")

    assert len(filename.split("_")) == 4

    ID, sample, agglo_type, _ = filename.split("_")
    with open(config_path, 'rb') as f:
        config_dict = json.load(f)

    UCM = vigra.readHDF5(ucm_path, 'merge_times')[SELECTED_OFFSET, SELECTED_SLICE]

    mask_1 = UCM == -15
    nb_nodes = UCM.max()
    mask_2 = UCM == nb_nodes

    nb_iterations = (UCM * np.logical_not(mask_2)).max()
    nb_iterations_per_frame = int(nb_iterations / NB_FRAMES)

    masked_UCM = UCM.copy()
    masked_UCM[mask_1] = nb_nodes

    print("Done")

    # Prepare video:
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie UCM', artist='alberto.bailoni@iwr.uni-heidelberg.de',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)

    from segmfriends import vis as vis

    fig, ax = plt.subplots(ncols=1, nrows=1,
                         figsize=(7, 7))

    data = np.ones_like(masked_UCM) * nb_iterations
    img = ax.matshow(data, cmap=plt.get_cmap('binary'), vmin=0, vmax=nb_iterations,
                         interpolation='none')

    # vis.plot_output_affin(ax, affinities, nb_offset=16, z_slice=0)

    plt.subplots_adjust(wspace=0, hspace=0)


    with writer.saving(fig, "UCM_{}.mp4".format(config_dict['agglo_type']), 100):
        for i in range(NB_FRAMES):
            frame_data = masked_UCM.copy()
            frame_data[frame_data > i*nb_iterations_per_frame] = nb_iterations
            frame_data[frame_data < i*nb_iterations_per_frame] = 0
            img.set_data(frame_data)
            writer.grab_frame()