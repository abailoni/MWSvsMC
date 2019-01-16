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
from segmfriends.utils.config_utils import adapt_configs_to_model, recursive_dict_update

if __name__ == '__main__':
    results_collected = {}
    root_dir = '/home/abailoni_local/hci_home/GEC_comparison_local'
    for subdir, dirs, files in os.walk(root_dir):
        for i, filename in enumerate(files):
            if not filename.endswith(".json") or filename.startswith("."):
                continue
            outputs = filename.split("_")
            ID, agglo_type, _ = filename.split("_")
            result_file = os.path.join(root_dir, filename)
            with open(result_file, 'rb') as f:
                result_dict = json.load(f)
            edge_prob = result_dict["edge_prob"]
            non_link = result_dict["non_link"]

            new_results = {}

            new_results[agglo_type] = {}
            new_results[agglo_type][str(non_link)] = {}
            new_results[agglo_type][str(non_link)][edge_prob] = {}
            new_results[agglo_type][str(non_link)][edge_prob][ID] = {'energy': result_dict["energy"],
                                                                    'score': result_dict["score"],
                                                                     'score_WS': result_dict["score_WS"],
                                                                     'runtime': result_dict["runtime"]}
            results_collected = recursive_dict_update(new_results, results_collected)

    ncols, nrows = 1, 1
    f, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(7, 7))

    for agglo_type in results_collected:
        if agglo_type != "max":
            continue
        for non_link in results_collected[agglo_type]:
            sub_dict = results_collected[agglo_type][non_link]
            probs = []
            values = []
            error_bars = []
            for edge_prob in sub_dict:
                multiple_values = []
                for ID in sub_dict[edge_prob]:
                    multiple_values.append(sub_dict[edge_prob][ID]["score_WS"]["adapted-rand"])
                    # multiple_values.append(sub_dict[edge_prob][ID]["runtime"])
                    # multiple_values.append(sub_dict[edge_prob][ID]["energy"])
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

    f.savefig(os.path.join('/home/abailoni_local/', 'comparison_local_ARAND_WS.pdf'), format='pdf')

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


