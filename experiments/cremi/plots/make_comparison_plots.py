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
from segmfriends.utils.config_utils import adapt_configs_to_model, recursive_dict_update, return_recursive_key_in_dict

from long_range_compare.load_datasets import CREMI_crop_slices, CREMI_sub_crops_slices

if __name__ == '__main__':

    # Create dictionary:
    results_collected = {}
    for sample in CREMI_crop_slices:
        results_collected[sample] = {}
        for crop in CREMI_crop_slices[sample]:
            results_collected[sample][crop] = {}
            for subcrop in CREMI_sub_crops_slices:
                results_collected[sample][crop][subcrop] = {}

    root_dir = os.path.join(get_hci_home_path(), 'GEC_comparison_longRangeGraph')

    sort_by = 'long_range_prob'

    for item in os.listdir(root_dir):
        if os.path.isfile(os.path.join(root_dir, item)):
            filename = item
            if not filename.endswith(".json") or filename.startswith("."):
                continue
            outputs = filename.split("_")
            if len(filename.split("_")) != 4:
                continue
            ID, sample, agglo_type, _ = filename.split("_")
            result_file = os.path.join(root_dir, filename)
            with open(result_file, 'rb') as f:
                file_dict = json.load(f)

            if sort_by ==  'long_range_prob':
                sort_key = file_dict["edge_prob"]
            elif sort_by == 'noise_factor':
                if "noise_factor" not in file_dict:
                    continue
                sort_key = file_dict["noise_factor"]
            else:
                raise ValueError
            non_link = file_dict["non_link"]
            if not isinstance(non_link, bool):
                non_link = False
            local_attraction = file_dict["local_attraction"]

            new_results = {}
            new_results[agglo_type] = {}
            new_results[agglo_type][str(non_link)] = {}
            new_results[agglo_type][str(non_link)][str(local_attraction)] = {}
            new_results[agglo_type][str(non_link)][str(local_attraction)][sort_key] = {}
            new_results[agglo_type][str(non_link)][str(local_attraction)][sort_key][ID] = file_dict

            crop = file_dict["crop"]
            subcrop = file_dict["subcrop"]


            # DELETING STUFF:
            # if sample == 'B' and crop == CREMI_crop_slices['B'][5] and subcrop == CREMI_sub_crops_slices[5] and \
            #     agglo_type == 'mean':
            # if crop in CREMI_crop_slices[sample][:4] and subcrop in CREMI_sub_crops_slices[4:5] and agglo_type == 'mean':
            #     os.remove(result_file)

            try:
                results_collected[sample][crop][subcrop] = recursive_dict_update(new_results, results_collected[sample][crop][subcrop])
            except KeyError:
                continue

    # def bar_plots():
    #     list_all_keys = [
    #         ['score_WS', 'adapted-rand'],
    #         ['score_WS', "vi-merge"],
    #         ['score_WS', "vi-split"],
    #         ['energy'],
    #         ['runtime']
    #     ]
    #
    #     for all_keys in list_all_keys:
    #
    #         selected_prob = 0.02
    #         label_names = []
    #         print('\n')
    #         print(all_keys)
    #
    #
    #         # Find best values for every crop:
    #         ncols, nrows = 1, 3
    #         f, all_ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(7, 7))
    #         for nb_sample, sample in enumerate(CREMI_crop_slices):
    #             cumulated_values = {'True': [None, None], 'False': [None, None]}
    #             counter = 0
    #             for crop in CREMI_crop_slices[sample]:
    #                 for subcrop in CREMI_sub_crops_slices:
    #                     results_collected_crop = results_collected[sample][crop][subcrop]
    #
    #                     # Skip if the dict is empty:
    #                     if not results_collected_crop:
    #                         continue
    #
    #                     if crop in CREMI_crop_slices[sample][4:] or subcrop == CREMI_sub_crops_slices[
    #                         6]:
    #                         continue
    #
    #                     scores = {'True': [[], []], 'False': [[], []]}
    #
    #                     label_names = []
    #
    #
    #                     for agglo_type in [ty for ty in ['max', 'mean', 'sum'] if ty in results_collected_crop]:
    #                         for non_link in ['False', 'True']:
    #                             if non_link not in results_collected_crop[agglo_type]:
    #                                 continue
    #                             if eval(non_link):
    #                                 label_names.append("{} + cannot-link".format(agglo_type))
    #                             else:
    #                                 label_names.append("{}".format(agglo_type))
    #                             for local_attraction in ['False', 'True']:
    #                                 if local_attraction not in results_collected_crop[agglo_type][non_link]:
    #                                     mean = np.nan
    #                                     err = np.nan
    #                                 else:
    #                                     sub_dict = results_collected_crop[agglo_type][non_link][local_attraction]
    #                                     values = []
    #                                     error_bars = []
    #                                     multiple_values = []
    #
    #                                     if selected_prob not in sub_dict or local_attraction == 'True':
    #                                         mean = np.nan
    #                                         err = np.nan
    #                                     else:
    #                                         for ID in sub_dict[selected_prob]:
    #                                             multiple_values.append(return_recursive_key_in_dict(sub_dict[selected_prob][ID], all_keys))
    #                                         if len(multiple_values) == 0:
    #                                             continue
    #                                         multiple_values = np.array(multiple_values)
    #                                         mean = multiple_values.mean()
    #                                         err = multiple_values.std()
    #                                 scores[local_attraction][0].append(mean)
    #                                 scores[local_attraction][1].append(err)
    #
    #                     if len(scores['False'][0]) == 0:
    #                         continue
    #
    #                     all_values = []
    #                     for local_attraction in ['False', 'True']:
    #                         all_values += scores[local_attraction][0]
    #                         scores[local_attraction] = np.array(scores[local_attraction])
    #
    #
    #                     # Check if all the update rules were computed:
    #                     if np.isnan(all_values).sum() > 4:
    #                         print("Sample {}, crop {}, subcrop {} was not complete!".format(sample,crop,subcrop))
    #                         # print(all_values)
    #                         # continue
    #
    #                     # max_error = np.nanmax(
    #                     #     np.concatenate((np.array(scores_local_attr[1]), np.array(scores_starnd_weights[1]))))
    #                     # max_value = np.nanmax(np.array(all_values))
    #                     # min_value = np.nanmin(np.array(all_values))
    #
    #                     # Convert to relative values:
    #                     for local_attraction in ['False', 'True']:
    #                         # scores[local_attraction][0] = (scores[local_attraction][0] - min_value) / (max_value-min_value)
    #                         # scores[local_attraction][1] = (scores[local_attraction][1]) / (max_value-min_value)
    #
    #                         for i in range(2):
    #                             if cumulated_values[local_attraction][i] is None:
    #                                 cumulated_values[local_attraction][i] = scores[local_attraction][i]
    #                             else:
    #                                 cumulated_values[local_attraction][i] = np.nansum(np.stack((scores[local_attraction][i], cumulated_values[local_attraction][i])), axis=0)
    #
    #                     counter += 1
    #
    #             if cumulated_values['False'][0] is None:
    #                 continue
    #
    #             # Find the global averages:
    #             all_vals = []
    #             all_errs = []
    #             for local_attraction in ['False', 'True']:
    #                 cumulated_values[local_attraction][0] /= counter
    #
    #                 # Set back zero values to NaN:
    #                 mask = cumulated_values[local_attraction][0] == 0.
    #                 cumulated_values[local_attraction][0][mask] = np.nan
    #
    #                 # Find max and min:
    #                 all_vals.append(cumulated_values[local_attraction][0].copy())
    #                 all_errs.append(cumulated_values[local_attraction][1].copy())
    #
    #
    #             max_value = np.nanmax(np.concatenate(all_vals))
    #             min_value = np.nanmin(np.concatenate(all_vals))
    #             max_error = np.nanmax(np.concatenate(all_errs))
    #
    #
    #             # Make the plot:
    #
    #
    #             ind = np.arange(cumulated_values['False'][0].shape[0])  # the x locations for the groups
    #             width = 0.35  # the width of the bars
    #
    #             ax = all_ax[nb_sample]
    #
    #             # FIXME: change this shit
    #             if all_keys[-1] == 'runtime':
    #                 cumulated_values['True'][1] /= 5
    #                 cumulated_values['False'][1] /= 5
    #
    #             rects1 = ax.bar(ind - width / 2, cumulated_values['True'][0], width, yerr=cumulated_values['True'][1],
    #                             color='SkyBlue', label='Attractive local edges, repulsive long-range edges')
    #             rects2 = ax.bar(ind + width / 2, cumulated_values['False'][0], width, yerr=cumulated_values['False'][1],
    #                             color='IndianRed', label='Repulsion/attraction defined by weights')
    #
    #             ax.set_xticks(ind)
    #             ax.set_ylabel(all_keys[-1])
    #             if all_keys[-1] == 'energy':
    #                 min_value = min_value-max_error
    #             else:
    #                 min_value = max([min_value-max_error, 0])
    #             ax.set_ylim([min_value, max_value+max_error])
    #             ax.set_xticklabels(tuple(label_names))
    #             ax.legend(loc='upper left')
    #             ax.set_title("{} on CREMI sample {}".format(all_keys[-1], sample))
    #
    #             # if all_keys[-1] == 'runtime':
    #             #     ax.set_yscale("log", nonposy='clip')
    #
    #         plt.subplots_adjust(bottom=0.05, top=0.95, hspace = 0.3)
    #         f.savefig(
    #                 os.path.join(get_hci_home_path(), 'GEC_plots', 'relative_comparison_{}_WS_{}_big.pdf'.format(all_keys[-1], selected_prob)),
    #                 format='pdf')
    #
    #
    # def bar_plots_several_probs():
    #     list_all_keys = [
    #         ['score_WS', 'adapted-rand'],
    #         ['score_WS', "vi-merge"],
    #         ['score_WS', "vi-split"],
    #         ['energy'],
    #         ['runtime']
    #     ]
    #
    #     for all_keys in list_all_keys:
    #
    #         label_names = []
    #         print('\n')
    #         print(all_keys)
    #
    #
    #         # Find best values for every crop:
    #         ncols, nrows = 1, 1
    #         f, all_ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(7, 7))
    #         for nb_sample, sample in enumerate(CREMI_crop_slices):
    #             cumulated_values = {'True': [None, None], 'False': [None, None]}
    #             counter = 0
    #             for crop in CREMI_crop_slices[sample]:
    #                 for subcrop in CREMI_sub_crops_slices:
    #                     results_collected_crop = results_collected[sample][crop][subcrop]
    #
    #                     # Skip if the dict is empty:
    #                     if not results_collected_crop:
    #                         continue
    #
    #                     if sample != 'B' or crop != CREMI_crop_slices['B'][5] or subcrop != CREMI_sub_crops_slices[5]:
    #                         continue
    #
    #                     scores = {'True': [[], []], 'False': [[], []]}
    #
    #                     label_names = []
    #
    #
    #                     for agglo_type in [ty for ty in ['max', 'mean', 'sum'] if ty in results_collected_crop]:
    #                         for non_link in ['False', 'True']:
    #                             if non_link not in results_collected_crop[agglo_type]:
    #                                 continue
    #                             if eval(non_link):
    #                                 label_names.append("{} + cannot-link".format(agglo_type))
    #                             else:
    #                                 label_names.append("{}".format(agglo_type))
    #                             for local_attraction in ['False', 'True']:
    #                                 if local_attraction not in results_collected_crop[agglo_type][non_link] or \
    #                                     (agglo_type=='mean' and non_link == 'True') or (local_attraction=='True'):
    #                                     mean_per_prob = {'': np.nan}
    #                                     err_per_prob = {'': np.nan}
    #                                 else:
    #                                     sub_dict = results_collected_crop[agglo_type][non_link][local_attraction]
    #
    #                                     mean_per_prob = {}
    #                                     err_per_prob = {}
    #
    #                                     for selected_prob in sub_dict:
    #                                         multiple_values = []
    #
    #                                         for ID in sub_dict[selected_prob]:
    #                                             multiple_values.append(return_recursive_key_in_dict(sub_dict[selected_prob][ID], all_keys))
    #                                         if len(multiple_values) == 0:
    #                                             continue
    #                                         multiple_values = np.array(multiple_values)
    #                                         mean_per_prob[selected_prob] = multiple_values.mean()
    #                                         err_per_prob[selected_prob] = multiple_values.std()
    #                                 scores[local_attraction][0].append(mean_per_prob)
    #                                 scores[local_attraction][1].append(err_per_prob)
    #
    #                     if len(scores['False'][0]) == 0:
    #                         continue
    #
    #
    #
    #                     new_scores = {'True': [[], []], 'False': [[], []]}
    #                     all_values = []
    #                     for local_attraction in ['False', 'True']:
    #                         # Combine results of all probabilities:
    #                         num_rules = len(scores[local_attraction][0])
    #                         new_scores[local_attraction][0] = [[] for _ in range(num_rules)]
    #                         new_scores[local_attraction][1] = [[] for _ in range(num_rules)]
    #                         for prob in scores['False'][0][0]:
    #                             for i in range(num_rules):
    #                                 if prob not in scores[local_attraction][0][i]:
    #                                     new_scores[local_attraction][0][i].append(np.nan)
    #                                     new_scores[local_attraction][1][i].append(np.nan)
    #                                 else:
    #                                     new_scores[local_attraction][0][i].append(scores[local_attraction][0][i][prob])
    #                                     new_scores[local_attraction][1][i].append(scores[local_attraction][1][i][prob])
    #
    #                         all_values += new_scores[local_attraction][0]
    #                         new_scores[local_attraction] = np.array(new_scores[local_attraction])
    #
    #                     scores = new_scores
    #
    #                     # Check if all the update rules were computed:
    #                     if np.isnan(all_values).sum() > 4:
    #                         print("Sample {}, crop {}, subcrop {} was not complete!".format(sample,crop,subcrop))
    #                         # print(all_values)
    #                         # continue
    #
    #                     # max_error = np.nanmax(
    #                     #     np.concatenate((np.array(new_scores_local_attr[1]), np.array(scores_starnd_weights[1]))))
    #                     max_value = np.nanmax(np.array(all_values), axis=0)
    #                     min_value = np.nanmin(np.array(all_values), axis=0)
    #
    #                     # Convert to relative values:
    #                     for local_attraction in ['False', 'True']:
    #                         # scores[local_attraction][0] = (scores[local_attraction][0] - min_value) / (max_value-min_value)
    #                         # scores[local_attraction][1] = (scores[local_attraction][1]) / (max_value-min_value)
    #
    #
    #                         for i in range(2):
    #                             # Combine different probabilities and average them:
    #                             new_data = np.mean(scores[local_attraction][i], axis=1)
    #
    #                             if cumulated_values[local_attraction][i] is None:
    #                                 cumulated_values[local_attraction][i] = new_data
    #                             else:
    #                                 cumulated_values[local_attraction][i] = np.nansum(np.stack((new_data, cumulated_values[local_attraction][i])), axis=0)
    #
    #                     counter += 1
    #
    #             if cumulated_values['False'][0] is None:
    #                 continue
    #
    #             # Find the global averages:
    #             all_vals = []
    #             all_errs = []
    #             for local_attraction in ['False', 'True']:
    #                 cumulated_values[local_attraction][0] /= counter
    #
    #                 # Set back zero values to NaN:
    #                 mask = cumulated_values[local_attraction][0] == 0.
    #                 cumulated_values[local_attraction][0][mask] = np.nan
    #
    #                 # Find max and min:
    #                 all_vals.append(cumulated_values[local_attraction][0].copy())
    #                 all_errs.append(cumulated_values[local_attraction][1].copy())
    #
    #
    #             max_value = np.nanmax(np.concatenate(all_vals))
    #             min_value = np.nanmin(np.concatenate(all_vals))
    #             max_error = np.nanmax(np.concatenate(all_errs))
    #
    #
    #             # Make the plot:
    #
    #
    #             ind = np.arange(cumulated_values['False'][0].shape[0])  # the x locations for the groups
    #             width = 0.35  # the width of the bars
    #
    #             ax = all_ax
    #
    #             # FIXME: change this shit
    #             if all_keys[-1] == 'runtime':
    #                 cumulated_values['True'][1] /= 5
    #                 cumulated_values['False'][1] /= 5
    #
    #             rects1 = ax.bar(ind - width / 2, cumulated_values['True'][0], width, yerr=cumulated_values['True'][1],
    #                             color='SkyBlue', label='Attractive local edges, repulsive long-range edges')
    #             rects2 = ax.bar(ind + width / 2, cumulated_values['False'][0], width, yerr=cumulated_values['False'][1],
    #                             color='IndianRed', label='Repulsion/attraction defined by weights')
    #
    #             ax.set_xticks(ind)
    #             ax.set_ylabel(all_keys[-1])
    #             if all_keys[-1] == 'energy':
    #                 min_value = min_value-max_error
    #             else:
    #                 min_value = max([min_value-max_error, 0])
    #             ax.set_xticklabels(tuple(label_names))
    #             ax.legend(loc='upper left')
    #             ax.set_title("{} on CREMI sample {}".format(all_keys[-1], sample))
    #
    #             # if all_keys[-1] == 'runtime':
    #             #     ax.set_yscale("log", nonposy='clip')
    #
    #             if all_keys[-1] == 'runtime':
    #                 ax.set_yscale("log", nonposy='clip')
    #
    #             ax.set_ylim([min_value, max_value+max_error])
    #
    #
    #         plt.subplots_adjust(bottom=0.05, top=0.95, hspace = 0.3)
    #         f.savefig(
    #                 os.path.join(get_hci_home_path(), 'GEC_plots', 'relative_comparison_{}_WS_probs_combined.pdf'.format(all_keys[-1])),
    #                 format='pdf')

    def split_merge_plot_multiple_blocks():
        key_x = ['score_WS', 'vi-merge']
        key_y = ['score_WS', 'adapted-rand']
        # key_y = ['energy']
        # key_x = ['runtime']
        key_value = ['runtime']
        # TODO: reduce MWS RUNTIME by a constant

        colors = {'max': {'False': {'True': 'C4', 'False': 'C0'}},
                  'mean': {'False': {'True': 'C5', 'False': 'C1'},
                           'True': {'True': 'C6', 'False': 'C8'}},
                  'sum': {'False': {'False': 'C2'},
                          'True': {'False': 'C3'}},
                  }

        markers = {'max': {'False': { 'False': '^'}},
                  'mean': {'False': {'False': 'o'},
                           'True': {'False': 'X'}},
                  'sum': {'False': {'False': 's'},
                          'True': {'False': '2'}},
                  }

        legend_labels = {
            'vi-merge': "Variation of information - merge",
            'vi-split': "Variation of information - split",
            'adapted-rand': "Adapted RAND",
        }

        marker_list = ['^', 'o', 'X', 's', '2']

        selected_prob = 0.02
        print('\n')
        label_names = []
        color_list = []

        # Find best values for every crop:
        matplotlib.rcParams.update({'font.size': 7})
        ncols, nrows = 1, 3
        f, all_ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(7, 10))
        for nb_sample, sample in enumerate(CREMI_crop_slices):
            cumulated_values = {'True': [None, None], 'False': [None, None]}
            counter = 0

            ax = all_ax[nb_sample]

            for crop in CREMI_crop_slices[sample]:
                for subcrop in CREMI_sub_crops_slices:
                    results_collected_crop = results_collected[sample][crop][subcrop]

                    # Skip if the dict is empty:
                    if not results_collected_crop:
                        continue

                    if crop not in CREMI_crop_slices[sample][:4] or subcrop not in CREMI_sub_crops_slices[4:5]:
                        continue

                    if not ((sample == "A" and marker_list[counter] == 's') or \
                                    (sample == "B" and marker_list[counter] == '^') or \
                                    (sample == "C" and marker_list[counter] == 'X')):
                        counter += 1
                        continue

                    print(sample, crop, subcrop)

                    scores = {'True': [[], []], 'False': [[], []]}

                    label_names = []
                    color_list = []

                    type_counter = 0
                    for agglo_type in [ty for ty in ['max', 'mean', 'sum'] if ty in results_collected_crop]:
                        for non_link in ['False', 'True']:
                            if non_link not in results_collected_crop[agglo_type]:
                                continue
                            if eval(non_link):
                                label_names.append("{} + cannot-link".format(agglo_type))
                            else:
                                label_names.append("{}".format(agglo_type))
                            for local_attraction in ['False']:
                                color_list.append(colors[agglo_type][non_link][local_attraction])
                                if local_attraction not in results_collected_crop[agglo_type][non_link]:
                                    mean = np.nan
                                    err = np.nan
                                else:
                                    sub_dict = results_collected_crop[agglo_type][non_link][local_attraction]
                                    values = []
                                    error_bars = []
                                    multiple_values = []

                                    if selected_prob not in sub_dict or local_attraction == 'True':
                                        mean = np.nan
                                        err = np.nan
                                    else:
                                        for ID in sub_dict[selected_prob]:
                                            new_data = [return_recursive_key_in_dict(sub_dict[selected_prob][ID], key_x),
                                                        return_recursive_key_in_dict(sub_dict[selected_prob][ID], key_y),
                                                        return_recursive_key_in_dict(sub_dict[selected_prob][ID], key_value)]
                                            if agglo_type == 'max':
                                                new_data[2] -= 80

                                            multiple_values.append(new_data)
                                        if len(multiple_values) == 0:
                                            continue
                                        multiple_values = np.array(multiple_values)
                                        mean = multiple_values.mean(axis=0)
                                        err = multiple_values.std(axis=0)
                                        lab = label_names[-1]
                                        # ax.scatter(multiple_values[:,0], multiple_values[:,1], c=color_list[-1],
                                        #            marker='o', alpha=0.3, s=20*multiple_values[:,2]*0.005, label=lab)
                                        ax.scatter(multiple_values[:,0], multiple_values[:,1], c=color_list[-1],
                                                   marker='o', alpha=0.5, s=30, label=lab)

                                        multiple_values = multiple_values[:,:2]

                                scores[local_attraction][0].append(mean)
                                scores[local_attraction][1].append(err)
                                type_counter += 0
                    if len(scores['False'][0]) == 0:
                        continue

                    all_values = []
                    for local_attraction in ['False']:
                        all_values += scores[local_attraction][0]
                        scores[local_attraction] = np.array(scores[local_attraction])


                    # Check if all the update rules were computed:
                    if np.isnan(all_values).sum() > 4:
                        print("Sample {}, crop {}, subcrop {} was not complete!".format(sample,crop,subcrop))
                        # print(all_values)
                        # continue

                    # max_value = np.nanmax(np.array(all_values), axis=0, keepdims=True)
                    # min_value = np.nanmin(np.array(all_values), axis=0, keepdims=True)


                    # Convert to relative values:
                    for local_attraction in ['False']:
                        # scores[local_attraction][0] = (scores[local_attraction][0] - min_value) / (max_value-min_value)
                        # scores[local_attraction][1] = (scores[local_attraction][1]) / (max_value-min_value)

                        nb_types = scores[local_attraction][0].shape[0]

                        for i in range(nb_types):
                            dt = scores[local_attraction][0]
                            # ax.scatter(dt[i, 0], dt[i, 1], c=color_list[i],
                            #        marker='s', alpha=0.3, s=50)
                            # lab = label_names[i] if counter==0 else None
                            lab = label_names[i]
                            # ax.errorbar(scores[local_attraction][0][ i, 0], scores[local_attraction][0][ i, 1],
                            #             xerr=scores[local_attraction][1][ i, 0], yerr=scores[local_attraction][1][ i, 1], fmt='.',
                            #             label=lab,
                            #             color=color_list[i], alpha=0.8)

                        for i in range(2):
                            if cumulated_values[local_attraction][i] is None:
                                cumulated_values[local_attraction][i] = scores[local_attraction][i]
                            else:
                                cumulated_values[local_attraction][i] = np.nansum(np.stack((scores[local_attraction][i], cumulated_values[local_attraction][i])), axis=0)

                    counter += 1


            if cumulated_values['False'][0] is None:
                continue

            # Find the global averages:
            all_vals = []
            all_errs = []
            for local_attraction in ['False']:
                cumulated_values[local_attraction][0] /= counter

                # Set back zero values to NaN:
                mask = cumulated_values[local_attraction][0] == 0.
                cumulated_values[local_attraction][0][mask] = np.nan

                # Find max and min:
                all_vals.append(cumulated_values[local_attraction][0].copy())
                all_errs.append(cumulated_values[local_attraction][1].copy())


            max_value = np.nanmax(np.concatenate(all_vals))
            min_value = np.nanmin(np.concatenate(all_vals))
            max_error = np.nanmax(np.concatenate(all_errs))


            # Make the plot:


            cumulated_values = np.array(cumulated_values['False'])

            nb_types = cumulated_values[0].shape[0]

            # for i in range(nb_types):
            #
            #     # ax.scatter(VI_merge, VI_split, s=runtimes/10., c=colors[agglo_type][non_link][local_attraction], label=plot_label_1+plot_label_2+plot_label_3, marker='o', alpha=0.3)
            #     # ax.scatter(VI_merge, VI_split, s=probs*100, c=runtimes, label=plot_label_1+plot_label_2+plot_label_3, marker='o', cmap='viridis')
            #
            #     ax.errorbar(cumulated_values[0,i,0], cumulated_values[0,i,1], xerr=cumulated_values[1,i,0], yerr=cumulated_values[1,i,1], fmt='.',
            #             label=label_names[i],
            #             color=color_list[i], alpha=0.5)



            ax.set_xlabel(legend_labels[key_x[-1]])
            ax.set_ylabel(legend_labels[key_y[-1]])
            # if all_keys[-1] == 'energy':
            #     min_value = min_value-max_error
            # else:
            #     min_value = max([min_value-max_error, 0])
            # ax.set_ylim([min_value, max_value+max_error])
            # ax.set_xticklabels(tuple(label_names))
            lgnd = ax.legend(loc='lower right')
            for i in range(5):
                lgnd.legendHandles[i]._sizes = [30]

            ax.set_title("Crop of CREMI sample {}".format(sample))

            ax.autoscale(enable=True, axis='both')

            # if all_keys[-1] == 'runtime':
            #     ax.set_yscale("log", nonposy='clip')



        plt.subplots_adjust(bottom=0.05, top=0.95, hspace = 0.3)
        f.savefig(
                os.path.join(get_hci_home_path(), 'GEC_plots', 'relative_merge_split_WS_{}_big_final.pdf'.format(selected_prob)),
                format='pdf')


    def split_merge_plot_full_CREMI():
        key_x = ['score_WS', 'vi-merge']
        key_y = ['score_WS', 'adapted-rand']
        # key_y = ['score_WS', 'vi-split']
        # key_y = ['energy']
        # key_x = ['runtime']
        key_value = ['runtime']
        # FIXME: CHANGE FROM NOISE TO PROBS


        # TODO: reduce MWS RUNTIME by a constant


        colors = {'max': {'False': {'True': 'C4', 'False': 'C0'}},
                  'mean': {'False': {'True': 'C5', 'False': 'C1'},
                           'True': {'True': 'C6', 'False': 'C8'}},
                  'sum': {'False': {'False': 'C2'},
                          'True': {'False': 'C3'}},
                  }

        markers = {'max': {'False': { 'False': '^'}},
                  'mean': {'False': {'False': 'o'},
                           'True': {'False': 'X'}},
                  'sum': {'False': {'False': 's'},
                          'True': {'False': '2'}},
                  }

        legend_axes = {
            'vi-merge': "Variation of information - merge",
            'vi-split': "Variation of information - split",
            'adapted-rand': "Adapted RAND",
        }


        exp_name_data = {
            "full_cremi": "agglo",
            "full_cremi_standardHC": ["black", "WSDT superpixels + Agglomeration (mean)"],
            # "full_cremi_standardHC_onlyShort": ["cyan", "WSDT superpixels + Avg. Agglo (only direct neigh)"],
            # "full_cremi_MC":["black", "WSDT superpixels + Multicut"]

        }

        legend_labels = {'max': {'False': 'Mutex Watershed'},
                   'mean': {'False': 'Mean',
                            'True': 'Mean + Cannot-Link'},
                   'sum': {'False': 'Sum',
                           'True': 'Sum + Cannot-Link'},
                   }

        marker_list = ['^', 'o', 'X', 's', '2']

        selected_prob = 0.07
        print('\n')
        label_names = []
        color_list = []

        # Find best values for every crop:
        matplotlib.rcParams.update({'font.size': 9})
        ncols, nrows = 3, 1
        f, all_ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(13, 7))
        all_sample_values = None
        for nb_sample, sample in enumerate(CREMI_crop_slices):
            cumulated_values = {'True': [None, None], 'False': [None, None]}
            counter = 0

            ax = all_ax[nb_sample]

            for crop in CREMI_crop_slices[sample]:
                for subcrop in CREMI_sub_crops_slices:
                    results_collected_crop = results_collected[sample][crop][subcrop]

                    # Skip if the dict is empty:
                    if not results_collected_crop:
                        continue

                    if crop not in CREMI_crop_slices[sample][4] or subcrop not in CREMI_sub_crops_slices[6]:
                        continue

                    print(sample, crop, subcrop)

                    scores = {'True': [[], []], 'False': [[], []]}

                    label_names = []
                    color_list = []

                    type_counter = 0
                    for agglo_type in [ty for ty in ['max', 'mean', 'sum'] if ty in results_collected_crop]:
                        for non_link in ['False', 'True']:
                            if non_link not in results_collected_crop[agglo_type]:
                                continue
                            label_names.append(legend_labels[agglo_type][non_link])
                            # if eval(non_link):
                            #     label_names.append("{} + cannot-link".format(agglo_type))
                            # else:
                            #     label_names.append("{}".format(agglo_type))
                            for local_attraction in ['False']:
                                color_list.append(colors[agglo_type][non_link][local_attraction])
                                if local_attraction not in results_collected_crop[agglo_type][non_link]:
                                    mean = np.nan
                                    err = np.nan
                                else:
                                    sub_dict = results_collected_crop[agglo_type][non_link][local_attraction]
                                    values = []
                                    error_bars = []
                                    multiple_values = []

                                    for edge_prob in sub_dict:
                                        for ID in sub_dict[edge_prob]:
                                            data_dict = sub_dict[edge_prob][ID]
                                            exp_name = data_dict.get('postproc_config', {}).get('experiment_name',
                                                                               "")
                                            if exp_name not in exp_name_data:
                                                # full_cremi, full_cremi_MC
                                                continue
                                            new_data = [return_recursive_key_in_dict(data_dict, key_x),
                                                        return_recursive_key_in_dict(data_dict, key_y),
                                                            return_recursive_key_in_dict(data_dict, key_value)]
                                            # if agglo_type == 'max':
                                            #     new_data[2] -= 80

                                            if exp_name == "full_cremi":
                                                lab = label_names[-1]
                                                print(edge_prob, sample, agglo_type, non_link)
                                                ax.scatter(new_data[0], new_data[1],
                                                           c=color_list[-1],
                                                           marker='o', alpha=0.7, s=100,
                                                           label=lab)
                                            else:
                                                print(ID)
                                                print(edge_prob, exp_name)
                                                ax.scatter(new_data[0], new_data[ 1], c=exp_name_data[exp_name][0],
                                                           marker='o', alpha=0.7, s=100,
                                                           label=exp_name_data[exp_name][1])

                                            # multiple_values.append(new_data)
                                        if len(multiple_values) == 0:
                                            continue
                                        # multiple_values = np.array(multiple_values)
                                        # mean = multiple_values.mean(axis=0)
                                        # err = multiple_values.std(axis=0)
                                        # ax.scatter(multiple_values[:,0], multiple_values[:,1], c=color_list[-1],
                                        #            marker='o', alpha=0.5, s=30, label=lab)

                                        # multiple_values = multiple_values[:,:2]

                                # scores[local_attraction][0].append(mean)
                                # scores[local_attraction][1].append(err)
                                type_counter += 0
                    if len(scores['False'][0]) == 0:
                        continue

                    all_values = []
                    for local_attraction in ['False']:
                        all_values += scores[local_attraction][0]
                        scores[local_attraction] = np.array(scores[local_attraction])


                    # # Check if all the update rules were computed:
                    # if np.isnan(all_values).sum() > 4:
                    #     print("Sample {}, crop {}, subcrop {} was not complete!".format(sample,crop,subcrop))
                    #     # print(all_values)
                    #     # continue
                    #
                    # # max_value = np.nanmax(np.array(all_values), axis=0, keepdims=True)
                    # # min_value = np.nanmin(np.array(all_values), axis=0, keepdims=True)
                    #
                    #
                    # # Convert to relative values:
                    # for local_attraction in ['False']:
                    #     # scores[local_attraction][0] = (scores[local_attraction][0] - min_value) / (max_value-min_value)
                    #     # scores[local_attraction][1] = (scores[local_attraction][1]) / (max_value-min_value)
                    #
                    #     nb_types = scores[local_attraction][0].shape[0]
                    #
                    #     for i in range(nb_types):
                    #         dt = scores[local_attraction][0]
                    #         # ax.scatter(dt[i, 0], dt[i, 1], c=color_list[i],
                    #         #        marker='s', alpha=0.3, s=50)
                    #         # lab = label_names[i] if counter==0 else None
                    #         lab = label_names[i]
                    #         # ax.errorbar(scores[local_attraction][0][ i, 0], scores[local_attraction][0][ i, 1],
                    #         #             xerr=scores[local_attraction][1][ i, 0], yerr=scores[local_attraction][1][ i, 1], fmt='.',
                    #         #             label=lab,
                    #         #             color=color_list[i], alpha=0.8)
                    #
                    #     for i in range(2):
                    #         if cumulated_values[local_attraction][i] is None:
                    #             cumulated_values[local_attraction][i] = scores[local_attraction][i]
                    #         else:
                    #             cumulated_values[local_attraction][i] = np.nansum(np.stack((scores[local_attraction][i], cumulated_values[local_attraction][i])), axis=0)

                    counter += 1




            ax.set_xlabel(legend_axes[key_x[-1]])
            ax.set_ylabel(legend_axes[key_y[-1]])
            # if all_keys[-1] == 'energy':
            #     min_value = min_value-max_error
            # else:
            #     min_value = max([min_value-max_error, 0])
            # ax.set_ylim([min_value, max_value+max_error])
            # ax.set_xticklabels(tuple(label_names))
            # ax.legend(loc='lower right')
            lgnd = ax.legend()
            for i in range(10):
                try:
                    lgnd.legendHandles[i]._sizes = [30]
                except IndexError:
                    break

            ax.set_title("CREMI training sample {}".format(sample))

            # if sample == "B":
            #     ax.set_ylim([0.080, 0.090])
            # else:
            ax.autoscale(enable=True, axis='both')

            # if all_keys[-1] == 'runtime':
            #     ax.set_yscale("log", nonposy='clip')




        # plt.subplots_adjust(bottom=0.05, top=0.95, hspace = 0.3)
        plt.subplots_adjust(left=0.05, right=0.95, wspace = 0.3)
        f.savefig(
                os.path.join(get_hci_home_path(), 'GEC_plots', 'relative_merge_split_WS_sup_{}_full_CREMI.pdf'.format(selected_prob)),
                format='pdf')

        print(label_names)
        # print(all_sample_values[0,:,1] / 3)


    def scatter_plot():
        colors = {'max': {'False': {'True': 'C4', 'False': 'C0'}},
                  'mean': {'False': {'True': 'C5', 'False': 'C1'},
                           'True': {'True': 'C6', 'False': 'C8'}},
                  'sum': {'False': {'False': 'C2'} ,
                          'True': {'False': 'C3'} },
        }


        ncols, nrows = 1, 1

        list_all_keys = [
            ['score_WS', 'adapted-rand'],
            # ['score_WS', "vi-merge"],
            # ['score_WS', "vi-split"],
            # ['energy'],
            # ['runtime']
        ]

        for all_keys in list_all_keys:

            selected_prob = 0.02
            label_names = []
            print('\n')
            print(all_keys)


            # Find best values for every crop:
            for sample in CREMI_crop_slices:
                cumulated_values = {'True': [None, None], 'False': [None, None]}
                counter = 0
                for crop in CREMI_crop_slices[sample]:
                    for subcrop in CREMI_sub_crops_slices:

                        if sample != 'B' or crop != CREMI_crop_slices['B'][5] or subcrop != CREMI_sub_crops_slices[5]:
                            continue

                        results_collected_crop = results_collected[sample][crop][subcrop]

                        matplotlib.rcParams.update({'font.size': 10})
                        f, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(7, 7))

                        for agglo_type in [ty for ty in ['sum', 'max', 'mean'] if ty in results_collected_crop]:
                            for non_link in [ty for ty in ['False', 'True'] if ty in results_collected_crop[agglo_type]]:
                                for local_attraction in [ty for ty in ['False'] if ty in results_collected_crop[agglo_type][non_link]]:

                                    # if agglo_type == 'sum':
                                    #     continue

                                    sub_dict = results_collected_crop[agglo_type][non_link][local_attraction]
                                    probs = []
                                    VI_split = []
                                    VI_merge = []
                                    runtimes = []
                                    error_bars_split = []
                                    error_bars_merge = []
                                    for edge_prob in sub_dict:
                                        if float(edge_prob) == 0. and local_attraction == 'True':
                                            continue
                                        multiple_VI_split= []
                                        multiple_VI_merge= []
                                        multiple_runtimes= []
                                        for ID in sub_dict[edge_prob]:
                                            multiple_VI_split.append(return_recursive_key_in_dict(sub_dict[edge_prob][ID], ['score_WS', "adapted-rand"]))
                                            multiple_VI_merge.append(return_recursive_key_in_dict(sub_dict[edge_prob][ID], ['score_WS', "vi-merge"]))
                                            multiple_runtimes.append(
                                                return_recursive_key_in_dict(sub_dict[edge_prob][ID],
                                                                             ['runtime']))
                                        if len(multiple_VI_split) == 0:
                                            continue
                                        probs.append(float(edge_prob))

                                        multiple_VI_split = np.array(multiple_VI_split)
                                        VI_split.append(multiple_VI_split.mean())
                                        error_bars_split.append(multiple_VI_split.std())

                                        multiple_VI_merge = np.array(multiple_VI_merge)
                                        VI_merge.append(multiple_VI_merge.mean())
                                        error_bars_merge.append(multiple_VI_merge.std())

                                        multiple_runtimes = np.array(multiple_runtimes)
                                        runtimes.append(multiple_runtimes.mean())

                                        # ax.scatter(multiple_VI_merge, multiple_VI_split, s=np.ones_like(multiple_VI_merge)*edge_prob * 500,
                                        #            c=colors[agglo_type][non_link][local_attraction], marker='o',
                                        #            alpha=0.3)

                                    if len(probs) == 0:
                                        continue
                                    probs = np.array(probs)

                                    error_bars_split = np.array(error_bars_split)
                                    VI_split = np.array(VI_split)

                                    error_bars_merge = np.array(error_bars_merge)
                                    VI_merge = np.array(VI_merge)

                                    runtimes = np.array(runtimes)

                                    # if (agglo_type=='mean' and non_link == 'True') or (agglo_type=='max' and local_attraction=='True'):
                                    #     continue

                                    # Compose plot label:
                                    plot_label_1 = agglo_type
                                    plot_label_2 = " + cannot-link " if eval(non_link) else " "
                                    plot_label_3 = "(local edges attractive)" if eval(local_attraction) else ""

                                    if all_keys[-1] == 'runtime':
                                        error_bars_split = None

                                    # if all_keys[-1] == 'energy':
                                    #     values = -values

                                    # print(runtimes.min(), runtimes.max())
                                    # runtimes -= 0.027
                                    # runtimes /= 0.2
                                    # runtimes = (1 - runtimes) * 500
                                    # print(runtimes.min(), runtimes.max())

                                    ax.scatter(VI_merge, VI_split, s=probs*400, c=colors[agglo_type][non_link][local_attraction],  marker='o', alpha=0.3, label=plot_label_1+plot_label_2+plot_label_3)

                                    # ax.errorbar(VI_merge, VI_split, xerr=error_bars_merge ,yerr=error_bars_split, fmt='.',
                                    #             color=colors[agglo_type][non_link][local_attraction], alpha=0.3)


                        if all_keys[-1] == 'runtime':
                            ax.set_yscale("log", nonposy='clip')

                        # ax.set_xticks(np.arange(0, 1, step=0.1))
                        ax.legend(loc='upper right')
                        ax.set_xlabel("Variation of information - merge")
                        # ax.set_ylabel("Variation of information - split")
                        ax.set_ylabel("Adapted RAND")
                        ax.set_ylim([0.027, 0.052])
                        ax.set_xlim([0.15, 0.35])
                        # ax.set_title("Variation of Information on CREMI sample {}".format(sample))

                        f.savefig(os.path.join(get_hci_home_path(), 'GEC_plots', 'split_merge_plot_WS_{}_deep_z.pdf'.format(sample)), format='pdf')

    def noise_experiments_MOD():
        colors = {'max': {'False': {'True': 'C4', 'False': 'C0'}},
                  'mean': {'False': {'True': 'C5', 'False': 'C1'},
                           'True': {'True': 'C6', 'False': 'C8'}},
                  'sum': {'False': {'False': 'C2'},
                          'True': {'False': 'C3'}},
                  }

        # key_y = ['score_WS', 'vi-merge']
        key_y = ['score_WS', 'adapted-rand']
        key_x = ['noise_factor']
        # key_y = ['score_WS', 'vi-split']
        # key_y = ['energy']
        # key_x = ['runtime']
        key_value = ['runtime']


        ncols, nrows = 1, 1

        list_all_keys = [
            ['score_WS', 'adapted-rand'],
            # ['score_WS', "vi-merge"],
            # ['score_WS', "vi-split"],
            # ['energy'],
            # ['runtime']
        ]

        legend_labels = {
            'vi-merge': "Variation of information - merge",
            'vi-split': "Variation of information - split",
            'adapted-rand': "Adapted RAND",
            'noise_factor': "$\sigma$ - Amount of biased noise added to long-range edges"
        }

        axis_ranges = {
            # 'vi-merge': [0.15, 0.35],
            'vi-split': None,
            # 'adapted-rand': [0.027, 0.052],
        }




        for all_keys in list_all_keys:

            print('\n')
            print(all_keys)

            # Find best values for every crop:
            for sample in CREMI_crop_slices:
                cumulated_values = {'True': [None, None], 'False': [None, None]}
                counter = 0
                for crop in CREMI_crop_slices[sample]:
                    for subcrop in CREMI_sub_crops_slices:

                        if sample != 'B' or crop != CREMI_crop_slices['B'][5] or subcrop != \
                                CREMI_sub_crops_slices[5]:
                            continue


                        results_collected_crop = results_collected[sample][crop][subcrop]

                        matplotlib.rcParams.update({'font.size': 10})
                        f, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(3.5, 7))

                        for agglo_type in [ty for ty in ['sum', 'max', 'mean'] if ty in results_collected_crop]:
                            for non_link in [ty for ty in ['False', 'True'] if
                                             ty in results_collected_crop[agglo_type]]:
                                for local_attraction in [ty for ty in ['False'] if
                                                         ty in results_collected_crop[agglo_type][non_link]]:

                                    # if agglo_type == 'sum':
                                    #     continue



                                    sub_dict = results_collected_crop[agglo_type][non_link][local_attraction]

                                    probs = []
                                    VI_split = []
                                    VI_merge = []
                                    runtimes = []
                                    error_bars_split = []
                                    error_bars_merge = []
                                    counter_per_type = 0
                                    for noise_factor in sub_dict:
                                        if float(noise_factor) == 0. and local_attraction == 'True':
                                            continue
                                        multiple_VI_split = []
                                        multiple_VI_merge = []
                                        multiple_runtimes = []
                                        for ID in sub_dict[noise_factor]:
                                            data_dict = sub_dict[noise_factor][ID]
                                            if data_dict['postproc_config'].get('experiment_name',
                                                                               "") != "local_split_noise_exp_shortEdges":
                                                # local_split_noise_exp_fewEdges, local_split_noise_exp_allEdges, local_split_noise_exp_shortEdges
                                                # local_merge_noise_exp_onlyShort_fromPixels, local_merge_noise_exp_008longEdges_fromPixels,
                                                # local_merge_noise_exp1
                                                continue

                                            multiple_VI_split.append(
                                                return_recursive_key_in_dict(data_dict, key_y))
                                            multiple_VI_merge.append(
                                                return_recursive_key_in_dict(data_dict, key_x))
                                            multiple_runtimes.append(
                                                return_recursive_key_in_dict(data_dict, key_value))
                                            counter_per_type += 1
                                        if len(multiple_VI_split) == 0:
                                            continue
                                        probs.append(float(noise_factor))

                                        multiple_VI_split = np.array(multiple_VI_split)
                                        VI_split.append(multiple_VI_split.mean())
                                        error_bars_split.append(multiple_VI_split.std())

                                        multiple_VI_merge = np.array(multiple_VI_merge)
                                        VI_merge.append(multiple_VI_merge.mean())
                                        error_bars_merge.append(multiple_VI_merge.std())

                                        multiple_runtimes = np.array(multiple_runtimes)
                                        runtimes.append(multiple_runtimes.mean())

                                        # ax.scatter(multiple_VI_merge, multiple_VI_split, s=np.ones_like(multiple_VI_merge)*edge_prob * 500,
                                        #            c=colors[agglo_type][non_link][local_attraction], marker='o',
                                        #            alpha=0.3)

                                    if len(probs) == 0:
                                        continue
                                    probs = np.array(probs)

                                    error_bars_split = np.array(error_bars_split)
                                    VI_split = np.array(VI_split)

                                    error_bars_merge = np.array(error_bars_merge)
                                    VI_merge = np.array(VI_merge)

                                    runtimes = np.array(runtimes)

                                    # if (agglo_type=='mean' and non_link == 'True') or (agglo_type=='max' and local_attraction=='True'):
                                    #     continue

                                    # Compose plot label:
                                    plot_label_1 = agglo_type
                                    plot_label_2 = " + cannot-link " if eval(non_link) else " "
                                    plot_label_3 = "(local edges attractive)" if eval(local_attraction) else ""

                                    if all_keys[-1] == 'runtime':
                                        error_bars_split = None

                                    # if all_keys[-1] == 'energy':
                                    #     values = -values

                                    # print(runtimes.min(), runtimes.max())
                                    # runtimes -= 0.027
                                    # runtimes /= 0.2
                                    # runtimes = (1 - runtimes) * 500
                                    # print(runtimes.min(), runtimes.max())

                                    print("Found in {}: {}".format(agglo_type, counter_per_type))

                                    # ax.scatter(VI_merge, VI_split, s=(1+probs)**2 * 150,
                                    #            c=colors[agglo_type][non_link][local_attraction], marker='o',
                                    #            alpha=0.3, label=plot_label_1 + plot_label_2 + plot_label_3)
                                    # ax.errorbar(VI_merge, VI_split, xerr=error_bars_merge ,yerr=error_bars_split, fmt='.',
                                    #             color=colors[agglo_type][non_link][local_attraction], alpha=0.3)

                                    # ax.scatter(probs, VI_split, s=200,
                                    #            c=colors[agglo_type][non_link][local_attraction], marker='o',
                                    #            alpha=0.3, label=plot_label_1 + plot_label_2 + plot_label_3)


                                    # ax.errorbar(VI_merge, VI_split, yerr=error_bars_split, fmt='.',
                                    #             color=colors[agglo_type][non_link][local_attraction], alpha=0.4,
                                    #              label = plot_label_1 + plot_label_2 + plot_label_3)
                                    #
                                    # argsort = np.argsort(VI_merge)
                                    # ax.plot(VI_merge[argsort], VI_split[argsort], '-', color=colors[agglo_type][non_link][local_attraction], alpha=0.8)

                                    ax.plot(np.linspace(0.0, 0.05, 2), [VI_split[0] for _ in range(2)], '-',
                                            color=colors[agglo_type][non_link][local_attraction], alpha=0.9,label = plot_label_1 + plot_label_2 + plot_label_3)



                        vis_utils.set_log_tics(ax, [-2,0], [10],  format = "%.2f", axis='y')

                        ax.set_ylim([0.028, 0.078])
                        ax.get_xaxis().set_visible(False)
                        # vis_utils.set_log_tics(ax, [-2,0], [10],  format="%.2f", axis='x')





                        # ax.set_xscale("log")

                        # ax.set_xticks(np.arange(0, 1, step=0.1))
                        # ax.legend()
                        # ax.set_xlabel(legend_labels[key_x[-1]])
                        ax.set_ylabel(legend_labels[key_y[-1]])

                        if key_x[-1] in axis_ranges:
                            ax.set_xlim(axis_ranges[key_x[-1]])
                        if key_y[-1] in axis_ranges:
                            ax.set_ylim(axis_ranges[key_y[-1]])

                        # ax.set_xlim([0.15, 0.35])
                        # ax.set_title("Crop of CREMI sample {} (90 x 300 x 300)".format(sample))

                        f.savefig(os.path.join(get_hci_home_path(), 'GEC_plots',
                                               'split_merge_plot_{}_deep_z_noise_local.pdf'.format(sample)),
                                  format='pdf')

    def noise_experiments():
        colors = {'max': {'False': {'True': 'C4', 'False': 'C0'}},
                  'mean': {'False': {'True': 'C5', 'False': 'C1'},
                           'True': {'True': 'C6', 'False': 'C8'}},
                  'sum': {'False': {'False': 'C2'},
                          'True': {'False': 'C3'}},
                  }

        # key_y = ['score_WS', 'vi-merge']
        key_y = ['score_WS', 'adapted-rand']
        key_x = ['noise_factor']
        # key_y = ['score_WS', 'vi-split']
        # key_y = ['energy']
        # key_x = ['runtime']
        key_value = ['runtime']


        ncols, nrows = 1, 1

        list_all_keys = [
            ['score_WS', 'adapted-rand'],
            # ['score_WS', "vi-merge"],
            # ['score_WS', "vi-split"],
            # ['energy'],
            # ['runtime']
        ]

        legend_labels = {
            'vi-merge': "Variation of information - merge",
            'vi-split': "Variation of information - split",
            'adapted-rand': "Adapted RAND",
            'noise_factor': "$\sigma$ - Amount of biased noise added to short-range edges",
            'energy': 'Multicut energy'

        }

        axis_ranges = {
            # 'vi-merge': [0.15, 0.35],
            'vi-split': None,
            # 'adapted-rand': [0.027, 0.052],
        }




        for all_keys in list_all_keys:

            print('\n')
            print(all_keys)

            # Find best values for every crop:
            for sample in CREMI_crop_slices:
                cumulated_values = {'True': [None, None], 'False': [None, None]}
                counter = 0
                for crop in CREMI_crop_slices[sample]:
                    for subcrop in CREMI_sub_crops_slices:

                        if sample != 'B' or crop != CREMI_crop_slices['B'][5] or subcrop != \
                                CREMI_sub_crops_slices[5]:
                            continue


                        results_collected_crop = results_collected[sample][crop][subcrop]

                        matplotlib.rcParams.update({'font.size': 10})
                        f, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(7, 7))

                        for agglo_type in [ty for ty in ['sum', 'max', 'mean'] if ty in results_collected_crop]:
                            for non_link in [ty for ty in ['False', 'True'] if
                                             ty in results_collected_crop[agglo_type]]:
                                for local_attraction in [ty for ty in ['False'] if
                                                         ty in results_collected_crop[agglo_type][non_link]]:

                                    # if agglo_type == 'sum':
                                    #     continue



                                    sub_dict = results_collected_crop[agglo_type][non_link][local_attraction]

                                    probs = []
                                    VI_split = []
                                    VI_merge = []
                                    runtimes = []
                                    error_bars_split = []
                                    error_bars_merge = []
                                    counter_per_type = 0
                                    for noise_factor in sub_dict:
                                        if local_attraction == 'True':
                                            continue
                                        multiple_VI_split = []
                                        multiple_VI_merge = []
                                        multiple_runtimes = []
                                        for ID in sub_dict[noise_factor]:
                                            data_dict = sub_dict[noise_factor][ID]

                                            check = True

                                            if data_dict.get('postproc_config', {}).get('experiment_name',
                                                                                        "") == "different_merge_noise_exp_shortEdges_fromPixels":
                                                check = False

                                            # if data_dict.get('postproc_config', {}).get('experiment_name',
                                            #                                             "") == "local_merge_noise_exp_onlyShort_fromPixels" and agglo_type != 'max':
                                            #     check = False

                                            if check:
                                                # SPLIT:

                                                # local_split_noise_exp_fewEdges, local_split_noise_exp_allEdges, local_split_noise_exp_shortEdges
                                                # MERGE:
                                                # local_merge_noise_exp_onlyShort_fromPixels, local_merge_noise_exp_008longEdges_fromPixels,
                                                # local_merge_noise_exp1, local_merge_noise_exp_allLongRange_fromPixels
                                                # NEW_merge_noise_exp_allLongRange_fromSP
                                                continue


                                            multiple_VI_split.append(
                                                return_recursive_key_in_dict(data_dict, key_y))
                                            multiple_VI_merge.append(
                                                return_recursive_key_in_dict(data_dict, key_x))
                                            multiple_runtimes.append(
                                                return_recursive_key_in_dict(data_dict, key_value))
                                            counter_per_type += 1
                                        if len(multiple_VI_split) == 0:
                                            continue
                                        probs.append(float(noise_factor))

                                        multiple_VI_split = np.array(multiple_VI_split)
                                        VI_split.append(multiple_VI_split.mean())
                                        error_bars_split.append(multiple_VI_split.std())

                                        multiple_VI_merge = np.array(multiple_VI_merge)
                                        VI_merge.append(multiple_VI_merge.mean())
                                        error_bars_merge.append(multiple_VI_merge.std())

                                        multiple_runtimes = np.array(multiple_runtimes)
                                        runtimes.append(multiple_runtimes.mean())

                                        # ax.scatter(multiple_VI_merge, multiple_VI_split, s=np.ones_like(multiple_VI_merge)*edge_prob * 500,
                                        #            c=colors[agglo_type][non_link][local_attraction], marker='o',
                                        #            alpha=0.3)

                                    if len(probs) == 0:
                                        continue
                                    probs = np.array(probs)

                                    error_bars_split = np.array(error_bars_split)
                                    VI_split = np.array(VI_split)

                                    error_bars_merge = np.array(error_bars_merge)
                                    VI_merge = np.array(VI_merge)

                                    runtimes = np.array(runtimes)

                                    # if (agglo_type=='mean' and non_link == 'True') or (agglo_type=='max' and local_attraction=='True'):
                                    #     continue

                                    # Compose plot label:
                                    plot_label_1 = agglo_type
                                    plot_label_2 = " + cannot-link " if eval(non_link) else " "
                                    plot_label_3 = "(local edges attractive)" if eval(local_attraction) else ""

                                    if all_keys[-1] == 'runtime':
                                        error_bars_split = None

                                    # if all_keys[-1] == 'energy':
                                    #     values = -values

                                    # print(runtimes.min(), runtimes.max())
                                    # runtimes -= 0.027
                                    # runtimes /= 0.2
                                    # runtimes = (1 - runtimes) * 500
                                    # print(runtimes.min(), runtimes.max())

                                    print("Found in {}: {}".format(agglo_type, counter_per_type))

                                    # ax.scatter(VI_merge, VI_split, s=(1+probs)**2 * 150,
                                    #            c=colors[agglo_type][non_link][local_attraction], marker='o',
                                    #            alpha=0.3, label=plot_label_1 + plot_label_2 + plot_label_3)
                                    # ax.errorbar(VI_merge, VI_split, xerr=error_bars_merge ,yerr=error_bars_split, fmt='.',
                                    #             color=colors[agglo_type][non_link][local_attraction], alpha=0.3)

                                    # ax.scatter(probs, VI_split, s=200,
                                    #            c=colors[agglo_type][non_link][local_attraction], marker='o',
                                    #            alpha=0.3, label=plot_label_1 + plot_label_2 + plot_label_3)


                                    ax.errorbar(VI_merge, VI_split, yerr=error_bars_split, fmt='.',
                                                color=colors[agglo_type][non_link][local_attraction], alpha=0.4,
                                                 label = plot_label_1 + plot_label_2 + plot_label_3)

                                    argsort = np.argsort(VI_merge)
                                    ax.plot(VI_merge[argsort], VI_split[argsort], '-', color=colors[agglo_type][non_link][local_attraction], alpha=0.8)

                                    # ax.plot(np.linspace(0.0, 0.9, 15), [VI_split[0] for _ in range(15)], '.-',
                                    #         color=colors[agglo_type][non_link][local_attraction], alpha=0.8,label = plot_label_1 + plot_label_2 + plot_label_3)



                        vis_utils.set_log_tics(ax, [-2,0], [10],  format = "%.2f", axis='y')

                        # ax.set_ylim([0.028, 0.078])
                        ax.set_ylim([0.03, 0.60])

                        # vis_utils.set_log_tics(ax, [-2,0], [10],  format="%.2f", axis='x')





                        # ax.set_xscale("log")

                        # ax.set_xticks(np.arange(0, 1, step=0.1))
                        ax.legend()
                        ax.set_xlabel(legend_labels[key_x[-1]])
                        ax.set_ylabel(legend_labels[key_y[-1]])

                        if key_x[-1] in axis_ranges:
                            ax.set_xlim(axis_ranges[key_x[-1]])
                        if key_y[-1] in axis_ranges:
                            ax.set_ylim(axis_ranges[key_y[-1]])

                        # ax.set_xlim([0.15, 0.35])
                        ax.set_title("Crop of CREMI sample {} (90 x 300 x 300)".format(sample))

                        f.savefig(os.path.join(get_hci_home_path(), 'GEC_plots',
                                               'split_merge_plot_{}_deep_z_noise_local.pdf'.format(sample)),
                                  format='pdf')



    # bar_plots_several_probs()
    # scatter_plot()
    # probability_plots()
    # probability_plots_OLD()
    # split_merge_plot_multiple_blocks()

    # noise_experiments()
    split_merge_plot_full_CREMI()
    # noise_experiments()


    # def probability_plots_OLD():
    #     colors = {'max': {'False': {'True': 'C4', 'False': 'C0'}},
    #               'mean': {'False': {'True': 'C5', 'False': 'C1'},
    #                        'True': {'True': 'C6', 'False': 'C8'}},
    #               'sum': {'False': {'False': 'C2'},
    #                       'True': {'False': 'C3'}},
    #               }
    #
    #     ncols, nrows = 1, 1
    #
    #     list_all_keys = [
    #         ['score_WS', 'adapted-rand'],
    #         ['score_WS', "vi-merge"],
    #         ['score_WS', "vi-split"],
    #         ['energy'],
    #         ['runtime']
    #     ]
    #
    #     for all_keys in list_all_keys:
    #
    #         selected_prob = 0.02
    #         label_names = []
    #         print('\n')
    #         print(all_keys)
    #
    #         # Find best values for every crop:
    #         for sample in CREMI_crop_slices:
    #             cumulated_values = {'True': [None, None], 'False': [None, None]}
    #             counter = 0
    #             for crop in CREMI_crop_slices[sample]:
    #                 for subcrop in CREMI_sub_crops_slices:
    #
    #                     if sample != 'B' or crop != CREMI_crop_slices['B'][5] or subcrop != \
    #                             CREMI_sub_crops_slices[5]:
    #                         continue
    #
    #                     results_collected_crop = results_collected[sample][crop][subcrop]
    #
    #                     f, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(7, 7))
    #
    #                     for agglo_type in [ty for ty in ['max', 'mean', 'sum'] if
    #                                        ty in results_collected_crop]:
    #                         for non_link in [ty for ty in ['False', 'True'] if
    #                                          ty in results_collected_crop[agglo_type]]:
    #                             for local_attraction in [ty for ty in ['False'] if
    #                                                      ty in results_collected_crop[agglo_type][
    #                                                          non_link]]:
    #                                 sub_dict = results_collected_crop[agglo_type][non_link][
    #                                     local_attraction]
    #                                 probs = []
    #                                 values = []
    #                                 error_bars = []
    #                                 for edge_prob in sub_dict:
    #                                     if float(edge_prob) == 0. and local_attraction == 'True':
    #                                         continue
    #                                     multiple_values = []
    #                                     for ID in sub_dict[edge_prob]:
    #                                         multiple_values.append(
    #                                             return_recursive_key_in_dict(sub_dict[edge_prob][ID],
    #                                                                          all_keys))
    #                                     if len(multiple_values) == 0:
    #                                         continue
    #                                     probs.append(float(edge_prob))
    #                                     multiple_values = np.array(multiple_values)
    #                                     values.append(multiple_values.mean())
    #                                     error_bars.append(multiple_values.std())
    #                                 if len(probs) == 0:
    #                                     continue
    #                                 error_bars = np.array(error_bars)
    #                                 values = np.array(values)
    #                                 probs = np.array(probs)
    #
    #                                 # if (agglo_type=='mean' and non_link == 'True') or (agglo_type=='max' and local_attraction=='True'):
    #                                 #     continue
    #
    #                                 # Compose plot label:
    #                                 plot_label_1 = agglo_type
    #                                 plot_label_2 = " + cannot-link " if eval(non_link) else " "
    #                                 plot_label_3 = "(local edges attractive)" if eval(
    #                                     local_attraction) else ""
    #
    #                                 if all_keys[-1] == 'runtime':
    #                                     error_bars = None
    #
    #                                 # if all_keys[-1] == 'energy':
    #                                 #     values = -values
    #
    #                                 ax.errorbar(probs, values, yerr=error_bars, fmt='.',
    #                                             label=plot_label_1 + plot_label_2 + plot_label_3,
    #                                             color=colors[agglo_type][non_link][local_attraction])
    #
    #                     if all_keys[-1] == 'runtime':
    #                         ax.set_yscale("log", nonposy='clip')
    #
    #                     ax.set_xticks(np.arange(0, 1, step=0.1))
    #                     ax.legend(loc='upper left')
    #                     ax.set_xlabel("Probability long-range edges")
    #                     ax.set_ylabel(all_keys[-1])
    #                     ax.set_title("{} on CREMI sample {}".format(all_keys[-1], sample))
    #
    #                     f.savefig(os.path.join(get_hci_home_path(), 'GEC_plots',
    #                                            'probs_plot_{}_WS_{}_deep_z.pdf'.format(all_keys[-1],
    #                                                                                    sample)),
    #                               format='pdf')
