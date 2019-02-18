import sys

sys.path += ["/home/abailoni_local/hci_home/python_libraries/nifty/python",
"/home/abailoni_local/hci_home/python_libraries/cremi_python",
"/home/abailoni_local/hci_home/pyCharm_projects/inferno",
"/home/abailoni_local/hci_home/pyCharm_projects/constrained_mst",
"/home/abailoni_local/hci_home/pyCharm_projects/neuro-skunkworks",
"/home/abailoni_local/hci_home/pyCharm_projects/segmfriends",
"/home/abailoni_local/hci_home/pyCharm_projects/hc_segmentation",
"/home/abailoni_local/hci_home/pyCharm_projects/neurofire",]

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

from skunkworks.metrics.cremi_score import cremi_score
import segmfriends.vis as vis_utils

if __name__ == '__main__':
    # -----------------
    # Load data:
    # -----------------
    root_path = "/export/home/abailoni/supervised_projs/divisiveMWS"
    # dataset_path = os.path.join(root_path, "cremi-dataset-crop")
    dataset_path = "/net/hciserver03/storage/abailoni/datasets/ISBI/"
    plots_path = os.path.join("/net/hciserver03/storage/abailoni/greedy_edge_contr/plots")
    save_path = os.path.join(root_path, "outputs")

    # Import data:
    affinities = vigra.readHDF5(
        os.path.join(dataset_path, "isbi_results_MWS/isbi_train_offsetsV4_3d_meantda_damws2deval_final.h5"), 'data')
    raw = io.imread(os.path.join(dataset_path, "train-volume.tif"))
    raw = np.array(raw)
    gt = vigra.readHDF5(os.path.join(dataset_path, "gt_mc3d.h5"), 'data')
    gt_2D = vigra.readHDF5(os.path.join(dataset_path, "gt_cleaned.h5"), 'data')
    gt_2D = np.transpose(gt_2D, (2, 1, 0))

    MWS_results = vigra.readHDF5(
        os.path.join(dataset_path, "isbi_results_MWS/mst_isbi_train_offsetsV4_3d_meantda_damws2deval_final.h5"), 'data')
    evals = cremi_score(gt, MWS_results, border_threshold=None, return_all_scores=True)
    print("MWS results:", evals)

    result_file = os.path.join('/home/abailoni_local/', 'generalized_GED_comparison.json')
    if os.path.exists(result_file):
        with open(result_file, 'rb') as f:
            result_dict = json.load(f)
    else:
        raise FileNotFoundError()

    # for agglo_type in result_dict:
    #     for non_link in result_dict[agglo_type]:
    #         sub_dict = result_dict[agglo_type][non_link]
    #         probs = []
    #         values = []
    #         error_bars = []
    #         for edge_prob in sub_dict:
    #             probs.append(float(edge_prob))
    #             multiple_values = []
    #             for ID in sub_dict[edge_prob]:
    #                 multiple_values.append(sub_dict[edge_prob][ID]["score_WS"]["adapted-rand"])
    #
    #
    #             multiple_values = np.array(multiple_values)
    #             values.append(multiple_values.mean())
    #             error_bars.append(multiple_values.std())
    #         error_bars = np.array(error_bars)
    #         values = np.array(values)
    #         probs = np.array(probs)
    #         ax.errorbar(probs, values, yerr=error_bars, fmt='.', label="{} {}".format(agglo_type, non_link))
    #
    # ax.set_xticks(np.arange(0, 1, step=0.1))
    # ax.legend(loc='upper left')
    #
    # f.savefig(os.path.join('/home/abailoni_local/', 'comparison_locAttr_ARAND_WS.pdf'), format='pdf')

    root_dir = '/home/abailoni_local/GEC_comparison_kept'
    for subdir, dirs, files in os.walk(root_dir):
        for i, filename in enumerate(files):
            aggl_type = filename.split("_")[1]
            prob = filename.split("_")[2]
            if aggl_type != "sum" or prob != "1.0":
                continue

            print(filename)

            # segm = MWS_results.astype('uint32')
            segm = vigra.readHDF5(os.path.join(root_dir, filename), 'segm_WS')

            evals = cremi_score(gt, segm, border_threshold=None, return_all_scores=True)
            print("3D and WS:", evals)

            segm_2D = np.empty_like(segm)
            max_label = 0
            for z in range(segm.shape[0]):
                segm_2D[z] = segm[z] + max_label
                max_label += segm[z].max() + 1

            segm_2D = vigra.analysis.labelVolume(segm_2D)
            evals = cremi_score(gt_2D, segm_2D, border_threshold=None, return_all_scores=True)
            print("2D and WS:", evals)
            break









