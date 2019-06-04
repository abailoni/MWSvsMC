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



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="PlotUCM")  #DebugExp
    parser.add_argument('--project_directory', default="projects/agglo_cluster_compare",  type=str)
    # parser.add_argument('--project_directory', default="projects/agglo_cluster_compare/quadxeon5_results",  type=str)
    # parser.add_argument('--project_directory', default="../quadxeon5_scratch/projects/agglo_cluster_compare",  type=str)
    # TODO: option to pass some other fixed kwargs and overwrite it...?

    args = parser.parse_args()

    exp_name = args.exp_name
    project_dir = os.path.join(get_trendytukan_drive_path(), args.project_directory)
    # project_dir = os.path.join(get_hci_home_path(), "../quadxeon5_scratch", args.project_directory)

    sample = "C"
    # str_crop_slice = "1:2,:30,150:1150,300:1300" #B
    str_crop_slice = "1:2,2:31,-1200:-200,100:1100"  # C
    from segmfriends.utils.various import parse_data_slice
    import h5py
    import vigra


    slc = tuple(parse_data_slice(str_crop_slice))
    dt_path = os.path.join(get_hci_home_path(), "datasets/cremi/SOA_affinities/sample{}_train.h5".format(sample))
    inner_path_GT = "segmentations/groundtruth_fixed_OLD" if sample == "B" else "segmentations/groundtruth_fixed"
    inner_path_raw = "raw_old" if sample == "B" else "raw"
    inner_path_affs = "predictions/full_affs"

    with h5py.File(dt_path, 'r') as f:
        GT = f[inner_path_GT][slc[1:]]
        raw = f[inner_path_raw][slc[1:]].astype('float32') / 255.
        affs = f[inner_path_affs][slc]

    segm_dir = "/mnt/localdata0/abailoni/projects/agglo_cluster_compare/cropTrainSamples/out_segms/"
    segms = {}
    segms['mean_F'] = vigra.readHDF5(segm_dir + "914711833_C_mean_False.h5", "segm_WS").astype('uint16')
    segms['mean_T'] = vigra.readHDF5(segm_dir + "805074224_C_mean_True.h5", "segm_WS").astype('uint16')
    segms['max_T'] = vigra.readHDF5(segm_dir + "121007305_C_max_True.h5", "segm_WS").astype('uint16')
    segms['min_F'] = vigra.readHDF5(segm_dir + "585436522_C_min_False.h5", "segm_WS").astype('uint16')
    segms['min_T'] = vigra.readHDF5(segm_dir + "344991648_C_min_True.h5", "segm_WS").astype('uint16')
    segms['sum_T'] = vigra.readHDF5(segm_dir + "639799137_C_sum_True.h5", "segm_WS").astype('uint16')
    segms['sum_T'] = vigra.readHDF5(segm_dir + "559488549_C_sum_True.h5", "segm_WS").astype('uint16')
    segms['max_F'] = vigra.readHDF5(segm_dir + "728923330_C_max_False.h5", "segm_WS").astype('uint16')
    segms['sum_F'] = vigra.readHDF5(segm_dir + "162721234_C_sum_False.h5", "segm_WS").astype('uint16')
    segms['mean_T'] = vigra.readHDF5(segm_dir + "498995203_C_mean_True.h5", "segm_WS").astype('uint16')
    segms['MWS'] = vigra.readHDF5(segm_dir + "696342370_C_MutexWatershed_False.h5", "segm_WS").astype('uint16')
    segms['mean_F'] = vigra.readHDF5(segm_dir + "274988907_C_mean_False.h5", "segm_WS").astype('uint16')
    segms['sum_F'] = vigra.readHDF5(segm_dir + "128235279_C_sum_False.h5", "segm_WS").astype('uint16')

    # fixed_kargs = {
    #     "experiment_name": exp_name,
    #     "project_directory": project_dir,
    #     "configs_dir_path": os.path.join(get_hci_home_path(), "pyCharm_projects/longRangeAgglo/experiments/cremi/configs")
    # }
    #
    # # Select experiment and plot results:
    # experiment = cremi_experiments.get_experiment_by_name(exp_name)(fixed_kwargs=fixed_kargs)
    # experiment.make_plots(project_dir)
    #

    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import numpy as np


    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=500):
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap


    # ----------------------------------

    slc_str = {
        'mean_F': ":,10:,:840,60:900",
        'mean_T': ":,15:,180:860,180:860",
        'max_T': ":,14:,-800:,:800",
        'min_F': ":,14:,300:600,300:600",
        'min_T': ":,14:,300:600,300:600",
        'sum_T': ":,16:,150:,150:",
        'max_F': ":,14:,300:600,300:600",
        'sum_F': ":,14:,:,:",
        'MWS': ":,14:,30:860,180:1010",
    }

    mask_segms = {
        'mean_F': [168], # 285 25
        'mean_T': [194, 195, 348, 28, 327, 300, 337, 292, 322, 260, 340],
        'max_T': [320, 279, 280, 249, 136, 135, 25, 210, 30, 28],
        'min_F' : [],
        'min_T' : [],
        'sum_T': [171, 197, 24, 497, 367, 436, 506, 164, 159],
        'max_F': [2],
        'sum_F': [23],  # [23,167,189],
        'MWS': [26, 120, 204, 208, 264],
    }


    def mask_multiple_labels(array, list_labels, invert=False,
                             apply_mask=True):
        array = array.copy()
        all_masks = []
        for label in list_labels:
            all_masks.append(array == label)
        if invert:
            array[np.array(all_masks).astype('uint16').sum(axis=0) != 0] = 0
        else:
            array[np.array(all_masks).astype('uint16').sum(axis=0) == 0] = 0
        if apply_mask:
            array = np.ma.masked_where(array == 0, array)
        return array

    # ----------------------------------
    import segmfriends.vis as vis_utils
    import matplotlib
    import matplotlib.pyplot as plt

    for curr_agglo in mask_segms:
    # curr_agglo = "MWS"

        fig, ax = plt.subplots(ncols=1, nrows=1,
                               figsize=(7, 7))
        for a in fig.get_axes():
            a.axis('off')

        slc_plt = tuple(parse_data_slice(slc_str[curr_agglo]))
        z_slice = slc_plt[1].start

        invert_check = True if curr_agglo == 'min' else False

        selected_segm = segms[curr_agglo][slc_plt[1:]]
        if curr_agglo in mask_segms:
            print(mask_segms[curr_agglo])
            selected_segm_masked = mask_multiple_labels(selected_segm, mask_segms[curr_agglo], invert=invert_check)
        else:
            selected_segm_masked = selected_segm

        cax = ax.matshow(raw[slc_plt[1:]][0], cmap='gray', alpha=0.98)
        cax = ax.matshow(affs[slc_plt][0, 0], cmap='gray', alpha=0.2)

        if 'min' not in curr_agglo:
            MAX_LABEL = selected_segm.max()
            ax.matshow(mask_multiple_labels(selected_segm, mask_segms[curr_agglo], invert=not invert_check)[0]+100, cmap='Greens', alpha=0.45, interpolation='none', vmin=0, vmax=MAX_LABEL+100)

            ax.matshow(mask_multiple_labels(selected_segm, mask_segms[curr_agglo], invert=invert_check)[0], cmap=truncate_colormap(plt.get_cmap('autumn'), maxval=0.8),
                       alpha=0.55, interpolation='none')

            vis_utils.plot_segm(ax, selected_segm, z_slice=0, background=None, highlight_boundaries=True,
                            alpha_labels=0.,
                            )

            vis_utils.plot_segm(ax, selected_segm_masked, z_slice=0, background=None, highlight_boundaries=True, alpha_labels=0.5,
                                mask_value=0
                                )
        else:
            ax.matshow(selected_segm[0],
                   cmap=truncate_colormap(plt.get_cmap('autumn'), maxval=0.8),
                   alpha=0.45, interpolation='none')


            vis_utils.plot_segm(ax, selected_segm_masked, z_slice=0, background=None, highlight_boundaries=True,
                        alpha_labels=0.25,
                        mask_value=0
                        )
            vis_utils.plot_segm(ax, selected_segm, z_slice=0, background=None, highlight_boundaries=True,alpha_boundary=0.7,
                        alpha_labels=0.,
                        )


        # plt.show()
        plot_dir = os.path.join(get_trendytukan_drive_path(), "projects/agglo_cluster_compare/cropTrainSamples/plots/")
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()
        fig.savefig(os.path.join(plot_dir, '{}_NEW.pdf'.format(curr_agglo)), format='pdf')
