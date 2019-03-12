from . import cremi_utils as cremi_utils
import numpy as np
import os
import json
import matplotlib
import matplotlib.pyplot as plt

from .cremi_utils import CREMI_crop_slices, CREMI_sub_crops_slices
from segmfriends.utils.config_utils import adapt_configs_to_model, recursive_dict_update, return_recursive_key_in_dict
from segmfriends.utils.various import check_dir_and_create
import segmfriends.vis as vis_utils

def get_experiment_by_name(name):
    assert name in globals(), "Experiment not found."
    return globals().get(name)


class CremiExperiment(object):
    def __init__(self, fixed_kwargs=None):
        if fixed_kwargs is None:
            self.fixed_kwargs = {}
        else:
            assert isinstance(fixed_kwargs, dict)
            self.fixed_kwargs = fixed_kwargs

        self.kwargs_to_be_iterated = {}

    def get_cremi_kwargs_iter(self, crop_iter, subcrop_iter,
                              init_kwargs_iter=None, nb_iterations=1):
        """
        CROPS:    Deep-z: 5     MC: 4   All: 0:4
        SUBCROPS: Deep-z: 5     MC: 6  All: 4 Tiny: 5
        """
        return cremi_utils.get_kwargs_iter(self.fixed_kwargs, self.kwargs_to_be_iterated,
                                           crop_iter=crop_iter, subcrop_iter=subcrop_iter,
                                           init_kwargs_iter=init_kwargs_iter, nb_iterations=nb_iterations)

    def get_list_of_runs(self, path):
        IDs, configs, json_files = [], [], []
        for item in os.listdir(path):
            if os.path.isfile(os.path.join(path, item)):
                filename = item
                if not filename.endswith(".json") or filename.startswith("."):
                    continue
                # outputs = filename.split("_")
                if len(filename.split("_")) != 4:
                    continue
                ID, sample, agglo_type, _ = filename.split("_")
                result_file = os.path.join(path, filename)
                json_files.append(filename)
                with open(result_file, 'rb') as f:
                    file_dict = json.load(f)
                configs.append(file_dict)
                IDs.append(ID)
        return configs, json_files

    def get_plot_data(self, path, sort_by = 'long_range_prob'):
        """
        :param sort_by: 'noise_factor' or 'long_range_prob'
        """
        # TODO: use experiment path

        # Create dictionary:
        results_collected = {}
        for sample in CREMI_crop_slices:
            results_collected[sample] = {}
            for crop in CREMI_crop_slices[sample]:
                results_collected[sample][crop] = {}
                for subcrop in CREMI_sub_crops_slices:
                    results_collected[sample][crop][subcrop] = {}

        for item in os.listdir(path):
            if os.path.isfile(os.path.join(path, item)):
                filename = item
                if not filename.endswith(".json") or filename.startswith("."):
                    continue
                outputs = filename.split("_")
                if len(filename.split("_")) != 4:
                    continue
                ID, sample, agglo_type, _ = filename.split("_")
                result_file = os.path.join(path, filename)
                with open(result_file, 'rb') as f:
                    file_dict = json.load(f)

                if sort_by == 'long_range_prob':
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
                    results_collected[sample][crop][subcrop] = recursive_dict_update(new_results,
                                                                                     results_collected[sample][crop][
                                                                                         subcrop])
                except KeyError:
                    continue
        return results_collected

class DebugExp(CremiExperiment):
    def __init__(self, *super_args, **super_kwargs):
        super(DebugExp, self).__init__(*super_args, **super_kwargs)

        self.fixed_kwargs.update({
            "dataset": "CREMI",
            "from_superpixels": False,
            "use_multicut": False,
            "save_segm": True,
            "WS_growing": True,
            "edge_prob": 0.0,
            "sample": "B",
            "experiment_name": "debug_exp",
            "local_attraction": False,
            "additional_model_keys": [],
            "save_UCM": False
        })

        self.kwargs_to_be_iterated.update({
            "noise_factor": [0.],
            'agglo': ["MEAN"]
        })

    def get_data(self, kwargs_iter=None):
        nb_threads_pool = 1
        nb_iterations = 1

        kwargs_iter = self.get_cremi_kwargs_iter(crop_iter=range(0, 1), subcrop_iter=range(5, 6),
                                                 init_kwargs_iter=kwargs_iter, nb_iterations=nb_iterations)

        return kwargs_iter, nb_threads_pool


class FullTestSamples(CremiExperiment):
    def __init__(self, *super_args, **super_kwargs):
        super(FullTestSamples, self).__init__(*super_args, **super_kwargs)

        self.fixed_kwargs.update({
            "dataset": "CREMI",
            "from_superpixels": True,
            "use_multicut": False,
            "save_segm": True,
            "WS_growing": False,
            "edge_prob": 0.1,
            # "sample": "B",
            "experiment_name": "FullTestSamples",
            "local_attraction": False,
            "additional_model_keys": ["debug_postproc"],
            "compute_scores": False,
            "save_UCM": False,
            "noise_factor": 0.
        })

        self.kwargs_to_be_iterated.update({
            # 'agglo': ["MEAN", "MutexWatershed", "MEAN_constr", "GAEC", "greedyFixation"],
            # 'agglo': ["MEAN_constr", "GAEC", "greedyFixation"],
            'agglo': ["MEAN"],
            # 'sample': ["B+"]
            'sample': ["A+", "C+", "B+"]
        })

    def get_data(self, kwargs_iter=None):
        nb_threads_pool = 1
        nb_iterations = 1

        kwargs_iter = self.get_cremi_kwargs_iter(crop_iter=range(0, 1), subcrop_iter=range(6, 7),
                                                 init_kwargs_iter=kwargs_iter, nb_iterations=nb_iterations)

        return kwargs_iter, nb_threads_pool

    def prepare_submission(self, project_directory):
        exp_name = self.fixed_kwargs['experiment_name']
        scores_path = os.path.join(project_directory, exp_name, "scores")
        results_collected = self.get_plot_data(scores_path, sort_by="long_range_prob")

        import sys
        import vigra
        from .data_paths import get_hci_home_path
        from . import load_datasets
        sys.path += [
            os.path.join(get_hci_home_path(), "python_libraries/cremi_tools"), ]

        from cremi_tools.alignment import backalign_segmentation
        from cremi_tools.alignment.backalign import bounding_boxes

        from cremi import Annotations, Volume
        from cremi.io import CremiFile

        for sample in ["A+", "B+", "C+"]:
            results_collected_crop = results_collected[sample][CREMI_crop_slices[sample][0]][CREMI_sub_crops_slices[6]]

            for agglo_type in [ty for ty in ['mean'] if ty in results_collected_crop]:
                for non_link in [ty for ty in ['False'] if
                                 ty in results_collected_crop[agglo_type]]:
                    for local_attraction in [ty for ty in ['False'] if
                                             ty in results_collected_crop[agglo_type][non_link]]:
                        sub_dict = results_collected_crop[agglo_type][non_link][local_attraction]

                        # Check long-range prob:
                        if 0.1 not in sub_dict:
                            continue

                        IDs = list(sub_dict[0.1].keys())
                        assert len(IDs) == 1
                        ID = IDs[0]

                        config_dict = sub_dict[0.1][ID]
                        sample = config_dict["sample"]

                        print("aligning sample ", sample)

                        # Load segm from file:
                        filename = "{}_{}_{}_{}.h5".format(ID, sample, config_dict["agglo_type"], config_dict["non_link"])
                        segm_path = os.path.join(project_directory, exp_name, "out_segms", filename)
                        segm = vigra.readHDF5(segm_path, "segm_WS")

                        # Add padding to bring it back in the shape of the padded-aligned volumes:
                        crop = load_datasets.crops_padded_volumes[sample]
                        orig_shape = load_datasets.shape_padded_aligned_datasets[sample]
                        padding = tuple( (slc.start, shp - slc.stop) for slc, shp in zip(crop, orig_shape) )
                        padded_segm = np.pad(segm, pad_width=padding, mode="constant")

                        # Apply Constantin crop and then backalign:
                        cropped_segm = padded_segm[bounding_boxes[sample]]
                        tmp_file = segm_path.replace(".h5", "_submission_temp.hdf")
                        backalign_segmentation(sample, cropped_segm, tmp_file,
                                               key="temp_data",
                                               postprocess=False)

                        # Create a CREMI-style file ready to submit:
                        final_submission_path = segm_path.replace(".h5", "_submission.hdf")
                        file = CremiFile(final_submission_path, "w")

                        # Write volumes representing the neuron and synaptic cleft segmentation.
                        backaligned_segm = vigra.readHDF5(tmp_file, "temp_data")
                        neuron_ids = Volume(backaligned_segm.astype('uint64'), resolution=(40.0, 4.0, 4.0),
                                            comment="SP-CNN-submission")

                        file.write_neuron_ids(neuron_ids)
                        file.close()

                        os.remove(tmp_file)



class FullTrainSamples(CremiExperiment):
    def __init__(self, *super_args, **super_kwargs):
        super(FullTrainSamples, self).__init__(*super_args, **super_kwargs)

        self.fixed_kwargs.update({
            "dataset": "CREMI",
            "from_superpixels": True,
            "use_multicut": False,
            "save_segm": True,
            "WS_growing": False,
            "edge_prob": 0.1,
            # "sample": "B",
            "experiment_name": "FullTrainSamples",
            "local_attraction": False,
            "additional_model_keys": ["debug_postproc"],
            "compute_scores": True,
            "save_UCM": False,
            "noise_factor": 0.
        })

        self.kwargs_to_be_iterated.update({
            'agglo': ["MEAN", "MutexWatershed", "MEAN_constr", "GAEC", "greedyFixation"],
            # 'agglo': ["MEAN_constr", "GAEC", "greedyFixation"],
            'sample': ["C", "A", "B"],
            # 'sample': ["B+", "A+", "C+"]
        })

    def get_data(self, kwargs_iter=None):
        nb_threads_pool = 1
        nb_iterations = 1

        kwargs_iter = self.get_cremi_kwargs_iter(crop_iter=range(4, 5), subcrop_iter=range(6, 7),
                                                 init_kwargs_iter=kwargs_iter, nb_iterations=nb_iterations)

        return kwargs_iter, nb_threads_pool



class CropTrainSamples(CremiExperiment):
    """
    Used for MC energy comparison...?
    """
    def __init__(self, *super_args, **super_kwargs):
        super(CropTrainSamples, self).__init__(*super_args, **super_kwargs)

        self.fixed_kwargs.update({
            "dataset": "CREMI",
            "from_superpixels": False,
            "use_multicut": False,
            "save_segm": False,
            "WS_growing": True,
            "edge_prob": 0.23,
            # "sample": "B",
            "experiment_name": "cropTrainSamples",
            "local_attraction": False,
            # "additional_model_keys": ["debug_postproc"],
            "compute_scores": True,
            "save_UCM": False,
            "noise_factor": 0.
        })


        self.kwargs_to_be_iterated.update({
            'agglo': ["SingleLinkage", "SingleLinkagePlusCLC", "CompleteLinkage", "CompleteLinkagePlusCLC",
                      "MEAN", "MutexWatershed", "MEAN_constr", "GAEC", "greedyFixation", "GAEC_noLogCosts", "MEAN_constr_logCosts",
                      "MEAN_logCosts", "greedyFixation_noLogCosts"],
            # 'agglo': ["SingleLinkage"],
            # 'agglo': ["MutexWatershed"],
            # 'agglo': ["MEAN_constr", "GAEC", "greedyFixation"],
            'sample': ["C"],
            # "additional_model_keys": ["debug_postproc"],
            # 'sample': ["B+", "A+", "C+"]
        })
    #     TODO: crops! agglo, from_superpixels, edge_prob, check and merge_edge, WS_grow

    def get_data(self, kwargs_iter=None):
        nb_threads_pool = 4
        nb_iterations = 1

        kwargs_iter = self.get_cremi_kwargs_iter(crop_iter=range(0, 1), subcrop_iter=range(4, 5), #4
                                                 init_kwargs_iter=kwargs_iter, nb_iterations=nb_iterations)

        return kwargs_iter, nb_threads_pool



class NoiseExperiment(CremiExperiment):
    def __init__(self, *super_args, **super_kwargs):
        super(NoiseExperiment, self).__init__(*super_args, **super_kwargs)

        self.fixed_kwargs.update({
            "dataset": "CREMI",
            "from_superpixels": True,
            "use_multicut": False,
            "save_segm": False,
            "WS_growing": True,
            "edge_prob": 0.1,
            "sample": "B",
            "experiment_name": "simplexNoiseMergeBiasedAllEdges",
            "local_attraction": False,
            "additional_model_keys": ["noise_sups"],
            "compute_scores": True,
            "save_UCM": False,
            # "noise_factor": 0.
        })
        # TODO: crop, delete, noise factor!
        self.kwargs_to_be_iterated.update({
            'agglo': ["MEAN", "MutexWatershed", "MEAN_constr", "GAEC", "greedyFixation"],
            # 'agglo': ["MEAN"],
            # "noise_factor": np.concatenate((np.linspace(2., 4.5, 5), np.linspace(4.5, 10., 15)))
            "noise_factor": [4.0]
            # 'sample': ["B"]
            # 'sample': ["B+", "A+", "C+"]
        })

    def get_data(self, kwargs_iter=None):
        nb_threads_pool = 15
        nb_iterations = 1

        kwargs_iter = self.get_cremi_kwargs_iter(crop_iter=range(5, 6), subcrop_iter=range(5, 6),
                                                 init_kwargs_iter=kwargs_iter, nb_iterations=nb_iterations)

        return kwargs_iter, nb_threads_pool


    def make_plots(self, project_directory):
        exp_name = self.fixed_kwargs['experiment_name']
        scores_path = os.path.join(project_directory, exp_name, "scores")
        results_collected = self.get_plot_data(scores_path, sort_by="noise_factor")

        colors = {'MutexWatershed': {'False': {'True': 'C4', 'False': 'C0'}},
                  'mean': {'False': {'True': 'C5', 'False': 'C1'},
                           'True': {'True': 'C6', 'False': 'C8'}},
                  'sum': {'False': {'False': 'C2'},
                          'True': {'False': 'C3'}},
                  }

        # key_y = ['score_WS', 'vi-merge']
        # key_y = ['score_WS', 'adapted-rand']
        key_x = ['noise_factor']
        # key_y = ['score_WS', 'vi-split']
        key_y = ['energy']
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

                        for agglo_type in [ty for ty in ['sum', 'MutexWatershed', 'mean'] if ty in results_collected_crop]:
                            for non_link in [ty for ty in ['False', 'True'] if
                                             ty in results_collected_crop[agglo_type]]:
                                for local_attraction in [ty for ty in ['False'] if
                                                         ty in results_collected_crop[agglo_type][non_link]]:

                                    sub_dict = results_collected_crop[agglo_type][non_link][local_attraction]

                                    probs = []
                                    VI_split = []
                                    VI_merge = []
                                    runtimes = []
                                    error_bars_split = []
                                    error_bars_merge = []
                                    counter_per_type = 0
                                    for noise_factor in sub_dict:
                                        multiple_VI_split = []
                                        multiple_VI_merge = []
                                        multiple_runtimes = []
                                        for ID in sub_dict[noise_factor]:
                                            data_dict = sub_dict[noise_factor][ID]

                                            if data_dict["edge_prob"] != 0.1:
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
                                                label=plot_label_1 + plot_label_2 + plot_label_3)

                                    argsort = np.argsort(VI_merge)
                                    ax.plot(VI_merge[argsort], VI_split[argsort], '-',
                                            color=colors[agglo_type][non_link][local_attraction], alpha=0.8)

                                    # ax.plot(np.linspace(0.0, 0.9, 15), [VI_split[0] for _ in range(15)], '.-',
                                    #         color=colors[agglo_type][non_link][local_attraction], alpha=0.8,label = plot_label_1 + plot_label_2 + plot_label_3)

                        # vis_utils.set_log_tics(ax, [-2, 0], [10], format="%.2f", axis='y')

                        # ax.set_ylim([0.028, 0.078])
                        # ax.set_ylim([0.03, 0.60])

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

                        plot_dir = os.path.join(project_directory, exp_name, "plots")
                        check_dir_and_create(plot_dir)

                        f.savefig(os.path.join(plot_dir,
                                               'noise_plot_{}_deep_z_noise_local.pdf'.format(sample)),
                                  format='pdf')




class NoiseExperimentSplit(CremiExperiment):
    def __init__(self, *super_args, **super_kwargs):
        super(NoiseExperimentSplit, self).__init__(*super_args, **super_kwargs)

        self.fixed_kwargs.update({
            "dataset": "CREMI",
            "from_superpixels": True,
            "use_multicut": False,
            "save_segm": True,
            "WS_growing": True,
            "edge_prob": 0.,
            "sample": "B",
            "experiment_name": "simplexNoiseSplitBiasedAllEdges",
            "local_attraction": False,
            "additional_model_keys": ["noise_sups"],
            "compute_scores": True,
            "save_UCM": False,
            # "noise_factor": 0.
        })
        # TODO: agglos, noise factor! Save segm, save noisy affs
        self.kwargs_to_be_iterated.update({
            # 'agglo': ["MEAN", "MutexWatershed", "MEAN_constr", "GAEC", "greedyFixation"],
            'agglo': ["MEAN"],
            # "noise_factor": np.concatenate((np.linspace(2., 4.5, 5), np.linspace(4.5, 10., 15)))
            "noise_factor": [8.0]
            # 'sample': ["B"]
            # 'sample': ["B+", "A+", "C+"]
        })

    def get_data(self, kwargs_iter=None):
        nb_threads_pool = 7
        nb_iterations = 2

        kwargs_iter = self.get_cremi_kwargs_iter(crop_iter=range(5, 6), subcrop_iter=range(5, 6),
                                                 init_kwargs_iter=kwargs_iter, nb_iterations=nb_iterations)

        return kwargs_iter, nb_threads_pool


    def make_scatter_plots(self, project_directory):
        exp_name = self.fixed_kwargs['experiment_name']
        scores_path = os.path.join(project_directory, exp_name, "scores")
        results_collected = self.get_plot_data(scores_path, sort_by="noise_factor")

        from .plot_utils import scatter_plot
        scatter_plot(results_collected, project_directory, exp_name)



    def make_plots(self, project_directory):
        exp_name = self.fixed_kwargs['experiment_name']
        scores_path = os.path.join(project_directory, exp_name, "scores")
        results_collected = self.get_plot_data(scores_path, sort_by="noise_factor")

        colors = {'MutexWatershed': {'False': {'True': 'C4', 'False': 'C0'}},
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

                        for agglo_type in [ty for ty in ['sum', 'MutexWatershed', 'mean'] if ty in results_collected_crop]:
                            for non_link in [ty for ty in ['False', 'True'] if
                                             ty in results_collected_crop[agglo_type]]:
                                for local_attraction in [ty for ty in ['False'] if
                                                         ty in results_collected_crop[agglo_type][non_link]]:

                                    sub_dict = results_collected_crop[agglo_type][non_link][local_attraction]

                                    probs = []
                                    VI_split = []
                                    VI_merge = []
                                    runtimes = []
                                    error_bars_split = []
                                    error_bars_merge = []
                                    counter_per_type = 0
                                    for noise_factor in sub_dict:
                                        multiple_VI_split = []
                                        multiple_VI_merge = []
                                        multiple_runtimes = []
                                        for ID in sub_dict[noise_factor]:
                                            data_dict = sub_dict[noise_factor][ID]

                                            if data_dict["edge_prob"] != 0.:
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
                                                label=plot_label_1 + plot_label_2 + plot_label_3)

                                    argsort = np.argsort(VI_merge)
                                    ax.plot(VI_merge[argsort], VI_split[argsort], '-',
                                            color=colors[agglo_type][non_link][local_attraction], alpha=0.8)

                                    # ax.plot(np.linspace(0.0, 0.9, 15), [VI_split[0] for _ in range(15)], '.-',
                                    #         color=colors[agglo_type][non_link][local_attraction], alpha=0.8,label = plot_label_1 + plot_label_2 + plot_label_3)

                        vis_utils.set_log_tics(ax, [-2, 0], [10], format="%.2f", axis='y')

                        # ax.set_ylim([0.028, 0.078])
                        # ax.set_ylim([0.03, 0.60])

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

                        plot_dir = os.path.join(project_directory, exp_name, "plots")
                        check_dir_and_create(plot_dir)

                        f.savefig(os.path.join(plot_dir,
                                               'noise_plot_{}_deep_z_noise_local.pdf'.format(sample)),
                                  format='pdf')

