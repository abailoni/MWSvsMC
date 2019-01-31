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

from compareMCandMWS.load_datasets import CREMI_crop_slices, CREMI_sub_crops_slices

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

    for subdir, dirs, files in os.walk(root_dir):
        for i, filename in enumerate(files):
            if not filename.endswith(".json") or filename.startswith("."):
                continue
            outputs = filename.split("_")
            if len(filename.split("_")) != 4:
                continue
            ID, sample, agglo_type, _ = filename.split("_")
            result_file = os.path.join(root_dir, filename)
            with open(result_file, 'rb') as f:
                file_dict = json.load(f)

            edge_prob = file_dict["edge_prob"]
            non_link = file_dict["non_link"]
            local_attraction = file_dict["local_attraction"]

            new_results = {}
            new_results[agglo_type] = {}
            new_results[agglo_type][str(non_link)] = {}
            new_results[agglo_type][str(non_link)][str(local_attraction)] = {}
            new_results[agglo_type][str(non_link)][str(local_attraction)][edge_prob] = {}
            new_results[agglo_type][str(non_link)][str(local_attraction)][edge_prob][ID] = file_dict

            crop = file_dict["crop"]
            subcrop = file_dict["subcrop"]
            try:
                results_collected[sample][crop][subcrop] = recursive_dict_update(new_results, results_collected[sample][crop][subcrop])
            except KeyError:
                continue

    def bar_plots():
        list_all_keys = [
            ['score_WS', 'adapted-rand'],
            ['score_WS', "vi-merge"],
            ['score_WS', "vi-split"],
            ['energy'],
            ['runtime']
        ]

        for all_keys in list_all_keys:

            selected_prob = 0.02
            label_names = []
            print('\n')
            print(all_keys)


            # Find best values for every crop:
            ncols, nrows = 1, 3
            f, all_ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(7, 7))
            for nb_sample, sample in enumerate(CREMI_crop_slices):
                cumulated_values = {'True': [None, None], 'False': [None, None]}
                counter = 0
                for crop in CREMI_crop_slices[sample]:
                    for subcrop in CREMI_sub_crops_slices:
                        results_collected_crop = results_collected[sample][crop][subcrop]

                        # Skip if the dict is empty:
                        if not results_collected_crop:
                            continue

                        if crop == CREMI_crop_slices[sample][4] or subcrop == CREMI_sub_crops_slices[
                            6]:
                            continue

                        scores = {'True': [[], []], 'False': [[], []]}

                        label_names = []


                        for agglo_type in [ty for ty in ['max', 'mean', 'sum'] if ty in results_collected_crop]:
                            for non_link in ['False', 'True']:
                                if non_link not in results_collected_crop[agglo_type]:
                                    continue
                                if eval(non_link):
                                    label_names.append("{} + cannot-link".format(agglo_type))
                                else:
                                    label_names.append("{}".format(agglo_type))
                                for local_attraction in ['False', 'True']:
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
                                                multiple_values.append(return_recursive_key_in_dict(sub_dict[selected_prob][ID], all_keys))
                                            if len(multiple_values) == 0:
                                                continue
                                            multiple_values = np.array(multiple_values)
                                            mean = multiple_values.mean()
                                            err = multiple_values.std()
                                    scores[local_attraction][0].append(mean)
                                    scores[local_attraction][1].append(err)

                        if len(scores['False'][0]) == 0:
                            continue

                        all_values = []
                        for local_attraction in ['False', 'True']:
                            all_values += scores[local_attraction][0]
                            scores[local_attraction] = np.array(scores[local_attraction])


                        # Check if all the update rules were computed:
                        if np.isnan(all_values).sum() > 4:
                            print("Sample {}, crop {}, subcrop {} was not complete!".format(sample,crop,subcrop))
                            # print(all_values)
                            # continue

                        # max_error = np.nanmax(
                        #     np.concatenate((np.array(scores_local_attr[1]), np.array(scores_starnd_weights[1]))))
                        # max_value = np.nanmax(np.array(all_values))
                        # min_value = np.nanmin(np.array(all_values))

                        # Convert to relative values:
                        for local_attraction in ['False', 'True']:
                            # scores[local_attraction][0] = (scores[local_attraction][0] - min_value) / (max_value-min_value)
                            # scores[local_attraction][1] = (scores[local_attraction][1]) / (max_value-min_value)

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
                for local_attraction in ['False', 'True']:
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


                ind = np.arange(cumulated_values['False'][0].shape[0])  # the x locations for the groups
                width = 0.35  # the width of the bars

                ax = all_ax[nb_sample]

                # FIXME: change this shit
                if all_keys[-1] == 'runtime':
                    cumulated_values['True'][1] /= 5
                    cumulated_values['False'][1] /= 5

                rects1 = ax.bar(ind - width / 2, cumulated_values['True'][0], width, yerr=cumulated_values['True'][1],
                                color='SkyBlue', label='Attractive local edges, repulsive long-range edges')
                rects2 = ax.bar(ind + width / 2, cumulated_values['False'][0], width, yerr=cumulated_values['False'][1],
                                color='IndianRed', label='Repulsion/attraction defined by weights')

                ax.set_xticks(ind)
                ax.set_ylabel(all_keys[-1])
                if all_keys[-1] == 'energy':
                    min_value = min_value-max_error
                else:
                    min_value = max([min_value-max_error, 0])
                ax.set_ylim([min_value, max_value+max_error])
                ax.set_xticklabels(tuple(label_names))
                ax.legend(loc='upper left')
                ax.set_title("{} on CREMI sample {}".format(all_keys[-1], sample))

                # if all_keys[-1] == 'runtime':
                #     ax.set_yscale("log", nonposy='clip')

            plt.subplots_adjust(bottom=0.05, top=0.95, hspace = 0.3)
            f.savefig(
                    os.path.join(get_hci_home_path(), 'GEC_plots', 'relative_comparison_{}_WS_{}_big.pdf'.format(all_keys[-1], selected_prob)),
                    format='pdf')


    def bar_plots_several_probs():
        list_all_keys = [
            ['score_WS', 'adapted-rand'],
            ['score_WS', "vi-merge"],
            ['score_WS', "vi-split"],
            ['energy'],
            ['runtime']
        ]

        for all_keys in list_all_keys:

            label_names = []
            print('\n')
            print(all_keys)


            # Find best values for every crop:
            ncols, nrows = 1, 1
            f, all_ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(7, 7))
            for nb_sample, sample in enumerate(CREMI_crop_slices):
                cumulated_values = {'True': [None, None], 'False': [None, None]}
                counter = 0
                for crop in CREMI_crop_slices[sample]:
                    for subcrop in CREMI_sub_crops_slices:
                        results_collected_crop = results_collected[sample][crop][subcrop]

                        # Skip if the dict is empty:
                        if not results_collected_crop:
                            continue

                        if sample != 'B' or crop != CREMI_crop_slices['B'][5] or subcrop != CREMI_sub_crops_slices[5]:
                            continue

                        scores = {'True': [[], []], 'False': [[], []]}

                        label_names = []


                        for agglo_type in [ty for ty in ['max', 'mean', 'sum'] if ty in results_collected_crop]:
                            for non_link in ['False', 'True']:
                                if non_link not in results_collected_crop[agglo_type]:
                                    continue
                                if eval(non_link):
                                    label_names.append("{} + cannot-link".format(agglo_type))
                                else:
                                    label_names.append("{}".format(agglo_type))
                                for local_attraction in ['False', 'True']:
                                    if local_attraction not in results_collected_crop[agglo_type][non_link] or \
                                        (agglo_type=='mean' and non_link == 'True') or (local_attraction=='True'):
                                        mean_per_prob = {'': np.nan}
                                        err_per_prob = {'': np.nan}
                                    else:
                                        sub_dict = results_collected_crop[agglo_type][non_link][local_attraction]

                                        mean_per_prob = {}
                                        err_per_prob = {}

                                        for selected_prob in sub_dict:
                                            multiple_values = []

                                            for ID in sub_dict[selected_prob]:
                                                multiple_values.append(return_recursive_key_in_dict(sub_dict[selected_prob][ID], all_keys))
                                            if len(multiple_values) == 0:
                                                continue
                                            multiple_values = np.array(multiple_values)
                                            mean_per_prob[selected_prob] = multiple_values.mean()
                                            err_per_prob[selected_prob] = multiple_values.std()
                                    scores[local_attraction][0].append(mean_per_prob)
                                    scores[local_attraction][1].append(err_per_prob)

                        if len(scores['False'][0]) == 0:
                            continue



                        new_scores = {'True': [[], []], 'False': [[], []]}
                        all_values = []
                        for local_attraction in ['False', 'True']:
                            # Combine results of all probabilities:
                            num_rules = len(scores[local_attraction][0])
                            new_scores[local_attraction][0] = [[] for _ in range(num_rules)]
                            new_scores[local_attraction][1] = [[] for _ in range(num_rules)]
                            for prob in scores['False'][0][0]:
                                for i in range(num_rules):
                                    if prob not in scores[local_attraction][0][i]:
                                        new_scores[local_attraction][0][i].append(np.nan)
                                        new_scores[local_attraction][1][i].append(np.nan)
                                    else:
                                        new_scores[local_attraction][0][i].append(scores[local_attraction][0][i][prob])
                                        new_scores[local_attraction][1][i].append(scores[local_attraction][1][i][prob])

                            all_values += new_scores[local_attraction][0]
                            new_scores[local_attraction] = np.array(new_scores[local_attraction])

                        scores = new_scores

                        # Check if all the update rules were computed:
                        if np.isnan(all_values).sum() > 4:
                            print("Sample {}, crop {}, subcrop {} was not complete!".format(sample,crop,subcrop))
                            # print(all_values)
                            # continue

                        # max_error = np.nanmax(
                        #     np.concatenate((np.array(new_scores_local_attr[1]), np.array(scores_starnd_weights[1]))))
                        max_value = np.nanmax(np.array(all_values), axis=0)
                        min_value = np.nanmin(np.array(all_values), axis=0)

                        # Convert to relative values:
                        for local_attraction in ['False', 'True']:
                            # scores[local_attraction][0] = (scores[local_attraction][0] - min_value) / (max_value-min_value)
                            # scores[local_attraction][1] = (scores[local_attraction][1]) / (max_value-min_value)


                            for i in range(2):
                                # Combine different probabilities and average them:
                                new_data = np.mean(scores[local_attraction][i], axis=1)

                                if cumulated_values[local_attraction][i] is None:
                                    cumulated_values[local_attraction][i] = new_data
                                else:
                                    cumulated_values[local_attraction][i] = np.nansum(np.stack((new_data, cumulated_values[local_attraction][i])), axis=0)

                        counter += 1

                if cumulated_values['False'][0] is None:
                    continue

                # Find the global averages:
                all_vals = []
                all_errs = []
                for local_attraction in ['False', 'True']:
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


                ind = np.arange(cumulated_values['False'][0].shape[0])  # the x locations for the groups
                width = 0.35  # the width of the bars

                ax = all_ax

                # FIXME: change this shit
                if all_keys[-1] == 'runtime':
                    cumulated_values['True'][1] /= 5
                    cumulated_values['False'][1] /= 5

                rects1 = ax.bar(ind - width / 2, cumulated_values['True'][0], width, yerr=cumulated_values['True'][1],
                                color='SkyBlue', label='Attractive local edges, repulsive long-range edges')
                rects2 = ax.bar(ind + width / 2, cumulated_values['False'][0], width, yerr=cumulated_values['False'][1],
                                color='IndianRed', label='Repulsion/attraction defined by weights')

                ax.set_xticks(ind)
                ax.set_ylabel(all_keys[-1])
                if all_keys[-1] == 'energy':
                    min_value = min_value-max_error
                else:
                    min_value = max([min_value-max_error, 0])
                ax.set_xticklabels(tuple(label_names))
                ax.legend(loc='upper left')
                ax.set_title("{} on CREMI sample {}".format(all_keys[-1], sample))

                # if all_keys[-1] == 'runtime':
                #     ax.set_yscale("log", nonposy='clip')

                if all_keys[-1] == 'runtime':
                    ax.set_yscale("log", nonposy='clip')

                ax.set_ylim([min_value, max_value+max_error])


            plt.subplots_adjust(bottom=0.05, top=0.95, hspace = 0.3)
            f.savefig(
                    os.path.join(get_hci_home_path(), 'GEC_plots', 'relative_comparison_{}_WS_probs_combined.pdf'.format(all_keys[-1])),
                    format='pdf')




    def probability_plots():
        colors = {'max': {'False': {'True': 'C4', 'False': 'C0'}},
                  'mean': {'False': {'True': 'C5', 'False': 'C1'},
                           'True': {'True': 'C6', 'False': 'C8'}},
                  'sum': {'False': {'False': 'C2'} ,
                          'True': {'False': 'C3'} },
        }


        ncols, nrows = 1, 1

        list_all_keys = [
            ['score_WS', 'adapted-rand'],
            ['score_WS', "vi-merge"],
            ['score_WS', "vi-split"],
            ['energy'],
            ['runtime']
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

                        f, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(7, 7))

                        for agglo_type in [ty for ty in ['max', 'mean', 'sum'] if ty in results_collected_crop]:
                            for non_link in [ty for ty in ['False', 'True'] if ty in results_collected_crop[agglo_type]]:
                                for local_attraction in [ty for ty in ['False', 'True'] if ty in results_collected_crop[agglo_type][non_link]]:
                                    sub_dict = results_collected_crop[agglo_type][non_link][local_attraction]
                                    probs = []
                                    values = []
                                    error_bars = []
                                    for edge_prob in sub_dict:
                                        if float(edge_prob) == 0. and local_attraction == 'True':
                                            continue
                                        multiple_values = []
                                        for ID in sub_dict[edge_prob]:
                                            multiple_values.append(return_recursive_key_in_dict(sub_dict[edge_prob][ID], all_keys))
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

                                    # if (agglo_type=='mean' and non_link == 'True') or (agglo_type=='max' and local_attraction=='True'):
                                    #     continue

                                    # Compose plot label:
                                    plot_label_1 = agglo_type
                                    plot_label_2 = " + cannot-link " if eval(non_link) else " "
                                    plot_label_3 = "(local edges attractive)" if eval(local_attraction) else ""

                                    if all_keys[-1] == 'runtime':
                                        error_bars = None

                                    # if all_keys[-1] == 'energy':
                                    #     values = -values

                                    ax.errorbar(probs, values, yerr=error_bars, fmt='.', label=plot_label_1+plot_label_2+plot_label_3,
                                                color=colors[agglo_type][non_link][local_attraction])


                        if all_keys[-1] == 'runtime':
                            ax.set_yscale("log", nonposy='clip')

                        ax.set_xticks(np.arange(0, 1, step=0.1))
                        ax.legend(loc='upper left')
                        ax.set_xlabel("Probability long-range edges")
                        ax.set_ylabel(all_keys[-1])
                        ax.set_title("{} on CREMI sample {}".format(all_keys[-1], sample))

                        f.savefig(os.path.join(get_hci_home_path(), 'GEC_plots', 'probs_plot_{}_WS_{}_deep_z.pdf'.format(all_keys[-1], sample)), format='pdf')


    # bar_plots_several_probs()
    probability_plots()


