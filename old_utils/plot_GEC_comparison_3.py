import sys

sys.path += ["/home/abailoni_local/hci_home/python_libraries/nifty/python",
"/home/abailoni_local/hci_home/python_libraries/cremi_python",
"/home/abailoni_local/hci_home/python_libraries/affogato/python",
"/home/abailoni_local/hci_home/pyCharm_projects/inferno",
"/home/abailoni_local/hci_home/pyCharm_projects/constrained_mst",
"/home/abailoni_local/hci_home/pyCharm_projects/neuro-skunkworks",
"/home/abailoni_local/hci_home/pyCharm_projects/segmfriends",
"/home/abailoni_local/hci_home/pyCharm_projects/hc_segmentation",
"/home/abailoni_local/hci_home/pyCharm_projects/neurofire",]

sys.path += ["/home/abailoni/hci_home/python_libraries/nifty/python",
"/home/abailoni/hci_home/python_libraries/cremi_python",
"/home/abailoni/hci_home/python_libraries/affogato/python",
"/home/abailoni/hci_home/pyCharm_projects/inferno",
"/home/abailoni/hci_home/pyCharm_projects/constrained_mst",
"/home/abailoni/hci_home/pyCharm_projects/neuro-skunkworks",
"/home/abailoni/hci_home/pyCharm_projects/segmfriends",
"/home/abailoni/hci_home/pyCharm_projects/hc_segmentation",
"/home/abailoni/hci_home/pyCharm_projects/neurofire",]


sys.path += [
"/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation",
"/net/hciserver03/storage/abailoni/python_libraries/affogato/python",
]

from segmfriends.io.save import get_hci_home_path
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
from segmfriends.utils.config_utils import adapt_configs_to_model, recursive_dict_update

if __name__ == '__main__':

    crop_slices = {
        "A": [":, 0: 31,:1300, -1300:",
              ":, 31: 62,:1300, -1300:",
              ":, 62: 93, 25: 1325,:1300",
              ":, 93: 124, -1300:,:1300"
              ],
        "B": [
            ":, 0: 31, 50: 1350, 200: 1500",
            ":, 31: 62, 20: 1320, 400: 1700",
            ":, 62: 93, 90: 1390, 580: 1880",
            ":, 93: 124, -1300:, 740: 2040",
        ],
        "C": [
            ":, 0: 31, -1300:,:1300",
            ":, 31: 62, 150: 1450, 95: 1395",
            ":, 62: 93, 70: 1370, 125: 1425",
            ":, 93: 124,:1300, -1300:",
        ]
    }

    sub_crops_slices = [":,2:, 100:600, 100:600",
                        ":,2:, 100:600, 600:1100",
                        ":,2:, 600:1100, 600:1100",
                        ":,2:, 600:1100, 100:600",
                        ":,2:, 100:1100, 100:1100",
                        ]




    for SAMPLE in ["A", "B", "C"]:
        results_collected = {}
        root_dir = os.path.join(get_hci_home_path(), 'GEC_comparison_longRangeGraph')
        for subdir, dirs, files in os.walk(root_dir):
            for i, filename in enumerate(files):
                if not filename.endswith(".json") or filename.startswith("."):
                    continue
                outputs = filename.split("_")
                ID, sample, agglo_type, _ = filename.split("_")
                result_file = os.path.join(root_dir, filename)
                with open(result_file, 'rb') as f:
                    result_dict = json.load(f)

                accepted_crop = crop_slices[SAMPLE][1]
                accepted_sub_crop = sub_crops_slices[4]
                # agglo_type != "max" or
                if  result_dict["crop"]!=accepted_crop or \
                                result_dict["subcrop"]!=accepted_sub_crop or  result_dict["edge_prob"]<0.0001:
                    continue

                # if result_dict["score_WS"]["adapted-rand"] > 0.1:
                #     os.remove(result_file)
                #     continue

                edge_prob = result_dict["edge_prob"]
                non_link = result_dict["non_link"]
                local_attraction = result_dict["local_attraction"]

                new_results = {}

                new_results[agglo_type] = {}
                new_results[agglo_type][str(non_link)] = {}
                new_results[agglo_type][str(non_link)][str(local_attraction)] = {}
                new_results[agglo_type][str(non_link)][str(local_attraction)][edge_prob] = {}
                new_results[agglo_type][str(non_link)][str(local_attraction)][edge_prob][ID] = {
                    'energy': result_dict["energy"],
                    'score': result_dict["score"],
                     'score_WS': result_dict["score_WS"],
                     'runtime': result_dict["runtime"]
                }
                results_collected = recursive_dict_update(new_results, results_collected)

        keys = ["vi-split", "vi-merge", "adapted-rand"]
        # for key in keys:
        #     ncols, nrows = 1, 1
        #     f, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(7, 7))
        #
        #     for agglo_type in results_collected:
        #         for non_link in results_collected[agglo_type]:
        #             for local_attraction in results_collected[agglo_type][non_link]:
        #                 sub_dict = results_collected[agglo_type][non_link][local_attraction]
        #                 probs = []
        #                 values = []
        #                 error_bars = []
        #                 for edge_prob in sub_dict:
        #                     multiple_values = []
        #                     for ID in sub_dict[edge_prob]:
        #                         # multiple_values.append(sub_dict[edge_prob][ID]["score_WS"]["vi-split"])
        #                         # multiple_values.append(sub_dict[edge_prob][ID]["score_WS"]["vi-merge"])
        #                         multiple_values.append(sub_dict[edge_prob][ID]["score_WS"][key])
        #                         # multiple_values.append(sub_dict[edge_prob][ID]["runtime"])
        #                         # multiple_values.append(sub_dict[edge_prob][ID]["energy"])
        #                     if len(multiple_values) == 0:
        #                         continue
        #                     probs.append(float(edge_prob))
        #                     multiple_values = np.array(multiple_values)
        #                     values.append(multiple_values.mean())
        #                     error_bars.append(multiple_values.std())
        #                 if len(probs) == 0:
        #                     continue
        #                 error_bars = np.array(error_bars)
        #                 values = np.array(values)
        #                 probs = np.array(probs)
        #                 ax.errorbar(probs, values, yerr=error_bars, fmt='.', label="{} NL:{}; Loc:{}".format(agglo_type, non_link, local_attraction))
        #
        #     ax.set_xticks(np.arange(0, 1, step=0.1))
        #     ax.legend(loc='upper left')
        #
        #     f.savefig(os.path.join(get_hci_home_path(), 'GEC_plots', 'comparison_{}_WS_{}_big.pdf'.format(key, SAMPLE)), format='pdf')

        for key in keys:
            ncols, nrows = 1, 1

            edge_prob = 0.02

            f, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(7, 7))

            scores_local_attr = ([] , [])
            scores_starnd_weights = ([] , [])
            label_names = []

            for agglo_type in ['max', 'mean', 'sum']:
                for non_link in results_collected[agglo_type]:
                    if eval(non_link):
                        label_names.append("Constr. {}".format(agglo_type))
                    else:
                        label_names.append("{}".format(agglo_type))
                    for local_attraction in ['False', 'True']:
                        if local_attraction not in results_collected[agglo_type][non_link]:
                            mean = np.nan
                            err = np.nan
                        else:
                            sub_dict = results_collected[agglo_type][non_link][local_attraction]
                            values = []
                            error_bars = []
                            multiple_values = []

                            if edge_prob not in sub_dict:
                                mean = np.nan
                                err = np.nan
                            else:
                                for ID in sub_dict[edge_prob]:
                                    # multiple_values.append(sub_dict[edge_prob][ID]["score_WS"]["vi-split"])
                                    # multiple_values.append(sub_dict[edge_prob][ID]["score_WS"]["vi-merge"])
                                    multiple_values.append(sub_dict[edge_prob][ID]["score_WS"][key])
                                    # multiple_values.append(sub_dict[edge_prob][ID]["runtime"])
                                    # multiple_values.append(sub_dict[edge_prob][ID]["energy"])
                                if len(multiple_values) == 0:
                                    continue
                                multiple_values = np.array(multiple_values)
                                mean = multiple_values.mean()
                                err = multiple_values.std()
                        if eval(local_attraction):
                            scores_local_attr[0].append(mean)
                            scores_local_attr[1].append(err)
                        else:
                            scores_starnd_weights[0].append(mean)
                            scores_starnd_weights[1].append(err)

            ind = np.arange(len(scores_local_attr[0]))  # the x locations for the groups
            width = 0.35  # the width of the bars

            max_error = np.nanmax(np.concatenate((np.array(scores_local_attr[1]), np.array(scores_starnd_weights[1]))))
            max_value = np.nanmax(np.concatenate((np.array(scores_local_attr[0]), np.array(scores_starnd_weights[0]))))
            min_value = np.nanmin(np.concatenate((np.array(scores_local_attr[0]), np.array(scores_starnd_weights[0]))))


            rects1 = ax.bar(ind - width / 2, scores_local_attr[0], width, yerr=scores_local_attr[1],
                            color='SkyBlue', label='Attractive local edges')
            rects2 = ax.bar(ind + width / 2, scores_starnd_weights[0], width, yerr=scores_starnd_weights[1],
                            color='IndianRed', label='Standard weights')

            ax.set_xticks(ind)
            ax.set_ylim([min_value-max_error, max_value+max_error])
            ax.set_xticklabels(tuple(label_names))
            ax.legend(loc='upper left')



            f.savefig(
                os.path.join(get_hci_home_path(), 'GEC_plots', 'comparison_{}_WS_{}_{}_big.pdf'.format(key, edge_prob,SAMPLE)),
                format='pdf')

                # f.savefig(os.path.join(get_hci_home_path(), 'GEC_plots', 'comparison_VIm_WS.pdf'), format='pdf')
    # f.savefig(os.path.join(get_hci_home_path(), 'GEC_plots', 'comparison_VIs_WS.pdf'), format='pdf')
    # f.savefig(os.path.join(get_hci_home_path(), 'GEC_plots', 'comparison_runtime_WS.pdf'), format='pdf')
    #
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


