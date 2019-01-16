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

import segmfriends.vis as vis_utils

if __name__ == '__main__':
    result_file_2D = os.path.join('/home/abailoni_local/', 'generalized_GED_comparison_2D.json')
    if os.path.exists(result_file_2D):
        with open(result_file_2D, 'rb') as f:
            result_dict_2D = json.load(f)
    else:
        raise FileNotFoundError()

    result_file = os.path.join('/home/abailoni_local/', 'generalized_GED_comparison.json')
    if os.path.exists(result_file):
        with open(result_file, 'rb') as f:
            result_dict = json.load(f)
    else:
        raise FileNotFoundError()


    ncols, nrows = 1, 1
    f, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(7, 7))

    for agglo_type in result_dict:
        if agglo_type != "max":
            continue
        for non_link in result_dict[agglo_type]:
            sub_dict = result_dict[agglo_type][non_link]
            probs = []
            values = []
            error_bars = []
            for edge_prob in sub_dict:
                if float(edge_prob) < 0.05:
                    continue
                multiple_values = []
                for ID in sub_dict[edge_prob]:
                    if ID not in result_dict_2D:
                        continue
                    multiple_values.append(result_dict_2D[ID]["scores_2D_WS"]["adapted-rand"])
                    # multiple_values.append(result_dict_2D[ID]["runtime"])
                    # multiple_values.append(result_dict_2D[ID]["energy"])
                if len(multiple_values) == 0:
                    continue
                probs.append(float(edge_prob))
                multiple_values = np.array(multiple_values)
                values.append(multiple_values.mean())
                error_bars.append(multiple_values.std())
            if len(probs) == 0:
                continue
            error_bars = np.array(error_bars)
            values = np.array(values)
            probs = np.array(probs)
            ax.errorbar(probs, values, yerr=error_bars, fmt='.', label="{} {}".format(agglo_type, non_link))

    ax.set_xticks(np.arange(0, 1, step=0.1))
    ax.legend(loc='upper left')

    f.savefig(os.path.join('/home/abailoni_local/', 'comparison_ARAND_max_WS.pdf'), format='pdf')

    # root_dir = '/home/abailoni_local/GEC_comparison_kept'
    # for subdir, dirs, files in os.walk(root_dir):
    #     for i, filename in enumerate(files):
    #         segm = vigra.readHDF5(os.path.join(root_dir, filename), 'segm_WS')
    #
    #         f, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(7, 7))
    #         vis_utils.plot_segm(ax, segm, z_slice=0)
    #         f.savefig(os.path.join('/home/abailoni_local/', filename+'.pdf'), format='pdf')
    #         if i > 5:
    #             break


