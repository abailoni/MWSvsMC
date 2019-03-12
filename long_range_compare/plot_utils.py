import os
from .load_datasets import CREMI_sub_crops_slices, CREMI_crop_slices
import matplotlib
import matplotlib.pyplot as plt
from segmfriends.utils.config_utils import return_recursive_key_in_dict
from segmfriends.utils.various import check_dir_and_create
import numpy as np
from .data_paths import get_hci_home_path

def scatter_plot(results_collected, project_directory, exp_name):
    colors = {'max': {'False': {'True': 'C4', 'False': 'C0'}},
              'mean': {'False': {'True': 'C5', 'False': 'C1'},
                       'True': {'True': 'C6', 'False': 'C8'}},
              'sum': {'False': {'False': 'C2'},
                      'True': {'False': 'C3'}},
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
                                for noise_factor in sub_dict:
                                    multiple_VI_split = []
                                    multiple_VI_merge = []
                                    multiple_runtimes = []
                                    for ID in sub_dict[noise_factor]:
                                        if sub_dict[noise_factor][ID]["edge_prob"] != 1.:
                                            continue
                                        multiple_VI_split.append(return_recursive_key_in_dict(sub_dict[noise_factor][ID],
                                                                                              ['score_WS',
                                                                                               "adapted-rand"]))
                                        multiple_VI_merge.append(return_recursive_key_in_dict(sub_dict[noise_factor][ID],
                                                                                              ['score_WS', "vi-merge"]))
                                        multiple_runtimes.append(
                                            return_recursive_key_in_dict(sub_dict[noise_factor][ID],
                                                                         ['runtime']))
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

                                    # ax.scatter(multiple_VI_merge, multiple_VI_split, s=np.ones_like(multiple_VI_merge)*noise_factor * 500,
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

                                ax.scatter(VI_merge, VI_split, s=probs*probs*probs ,
                                           c=colors[agglo_type][non_link][local_attraction], marker='o', alpha=0.3,
                                           label=plot_label_1 + plot_label_2 + plot_label_3)

                                # ax.errorbar(VI_merge, VI_split, xerr=error_bars_merge ,yerr=error_bars_split, fmt='.',
                                #             color=colors[agglo_type][non_link][local_attraction], alpha=0.3)

                    if all_keys[-1] == 'runtime':
                        ax.set_yscale("log", nonposy='clip')
                        # ax.set_yscale("log", nonposy='clip')
                    ax.set_yscale("log", nonposy='clip')
                    import segmfriends.vis as vis_utils
                    # vis_utils.set_log_tics(ax, [-2, 0], [10], format="%.2f", axis='y')
                    # vis_utils.set_log_tics(ax, [-2, 0], [10], format="%.2f", axis='x')

                    # ax.set_xticks(np.arange(0, 1, step=0.1))
                    ax.legend(loc='upper right')
                    ax.set_xlabel("Variation of information - merge")
                    # ax.set_ylabel("Variation of information - split")
                    ax.set_ylabel("Adapted RAND")
                    # ax.set_ylim([0.027, 0.052])
                    # ax.set_xlim([0.15, 0.35])
                    # ax.set_title("Variation of Information on CREMI sample {}".format(sample))

                    plot_dir = os.path.join(project_directory, exp_name, "plots")
                    check_dir_and_create(plot_dir)
                    f.savefig(os.path.join(plot_dir,
                                           'noise_plot_{}_deep_z_scatter_plots.pdf'.format(sample)),
                              format='pdf')
