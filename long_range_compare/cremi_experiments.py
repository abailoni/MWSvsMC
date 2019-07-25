from . import cremi_utils as cremi_utils
import numpy as np
import os
import json
import matplotlib
import matplotlib.pyplot as plt

from .cremi_utils import CREMI_crop_slices, CREMI_sub_crops_slices
from segmfriends.utils.config_utils import adapt_configs_to_model, recursive_dict_update, return_recursive_key_in_dict
from segmfriends.utils.various import check_dir_and_create
from segmfriends.utils import various as segm_utils
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
                              init_kwargs_iter=None, nb_iterations=1, noise_mod='split-biased'):
        """
        CROPS:    Deep-z: 5     MC: 4   All: 0:4
        SUBCROPS: Deep-z: 5     MC: 6  All: 4 Tiny: 5
        """
        return cremi_utils.get_kwargs_iter(self.fixed_kwargs, self.kwargs_to_be_iterated,
                                           crop_iter=crop_iter, subcrop_iter=subcrop_iter,
                                           init_kwargs_iter=init_kwargs_iter, nb_iterations=nb_iterations, noise_mod=noise_mod)

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
            "from_superpixels": True,
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
            'agglo': ["SingleLinkage"]
        })

    def get_data(self, kwargs_iter=None):
        nb_threads_pool = 1
        nb_iterations = 1

        kwargs_iter = self.get_cremi_kwargs_iter(crop_iter=range(0, 1), subcrop_iter=range(5, 6),
                                                 init_kwargs_iter=kwargs_iter, nb_iterations=nb_iterations)

        return kwargs_iter, nb_threads_pool

class BlockWiseExp(CremiExperiment):
    def __init__(self, *super_args, **super_kwargs):
        super(BlockWiseExp, self).__init__(*super_args, **super_kwargs)

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
            "experiment_name": "FullTestSamplesEmbed",
            "local_attraction": False,
            "additional_model_keys": ["debug_postproc", "simple_WSDT"],
            # "additional_model_keys": ["debug_postproc", "embeddings_agglo"],
            "compute_scores": False,
            "save_UCM": False,
            "noise_factor": 0.
        })

        self.kwargs_to_be_iterated.update({
            # 'agglo': ["MEAN", "MutexWatershed", "MEAN_constr", "GAEC", "greedyFixation"],
            # 'agglo': ["MEAN_constr", "GAEC", "greedyFixation"],
            'agglo': ["MEAN"],
            'sample': ["A+", "B+", "C+"]
            # 'sample': ["A+"]
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

        for sample in ["B+"]:
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

                        for ID in sub_dict[0.1]:
                            config_dict = sub_dict[0.1][ID]
                            sample = config_dict["sample"]



                            # Load segm from file:
                            filename = "{}_{}_{}_{}.h5".format(ID, sample, config_dict["agglo_type"], config_dict["non_link"])
                            segm_path = os.path.join(project_directory, exp_name, "out_segms", filename)
                            if not os.path.exists(segm_path):
                                continue
                            print("aligning sample ", filename)
                            key = "segm_WS" if "segm_WS" in segm_utils.getHDF5datasets(segm_path) else "segm"
                            segm = segm_utils.readHDF5(segm_path, key)

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
                            backaligned_segm = segm_utils.readHDF5(tmp_file, "temp_data")
                            neuron_ids = Volume(backaligned_segm.astype('uint64'), resolution=(40.0, 4.0, 4.0),
                                                comment="Stacked-CNN-submission")

                            file.write_neuron_ids(neuron_ids)
                            file.close()

                            os.remove(tmp_file)



class FullTrainSamples(CremiExperiment):
    def __init__(self, *super_args, **super_kwargs):
        super(FullTrainSamples, self).__init__(*super_args, **super_kwargs)

        self.fixed_kwargs.update({
            "dataset": "CREMI",
            "from_superpixels": False,
            "use_multicut": False,
            "save_segm": True,
            "WS_growing": False,
            "edge_prob": 0.1,
            # "sample": "B",
            "experiment_name": "FullTrainSamples_debug",
            "local_attraction": False,
            "additional_model_keys": ["debug_postproc", "simple_WSDT"],
            # "additional_model_keys": ["debug_postproc", "embeddings_agglo"],
            "compute_scores": True,
            "save_UCM": False,
            "noise_factor": 0.
        })

        self.kwargs_to_be_iterated.update({
            'agglo': ["MEAN"],
            # 'agglo': ["MEAN", "MutexWatershed", "MEAN_constr", "GAEC", "greedyFixation"],
            # 'agglo': [ "greedyFixation", "MEAN", "MutexWatershed", "MEAN_constr", "GAEC"],
            # 'agglo': [ "greedyFixation", "MEAN", "MutexWatershed", "MEAN_constr", "GAEC"],
            # 'agglo': ["SingleLinkagePlusCLC", "CompleteLinkage", "CompleteLinkagePlusCLC", "SingleLinkage"],
            'sample': ["A", "B", "C"],
            # 'sample': ["B+", "A+", "C+"]
        })

    def get_data(self, kwargs_iter=None):
        nb_threads_pool = 1
        nb_iterations = 1

        kwargs_iter = self.get_cremi_kwargs_iter(crop_iter=range(0, 1), subcrop_iter=range(5, 6),  # 4, 6
                                                 init_kwargs_iter=kwargs_iter, nb_iterations=nb_iterations)

        return kwargs_iter, nb_threads_pool

    def collect_scores(self, project_directory):
        exp_name = self.fixed_kwargs['experiment_name']
        scores_path = os.path.join(project_directory, exp_name, "scores")
        config_list, json_file_list = self.get_list_of_runs(scores_path)

        def assign_color(value, good_thresh, bad_thresh, nb_flt, best="lowest"):
            if best == "lowest":
                if value < good_thresh:
                    return '{{\color{{ForestGreen}} {num:.{prec}f} }}'.format(prec=nb_flt, num=value)
                if value > good_thresh and value < bad_thresh:
                    return '{{\color{{Orange}} {num:.{prec}f} }}'.format(prec=nb_flt, num=value)
                if value > bad_thresh:
                    return '{{\color{{Red}} {num:.{prec}f} }}'.format(prec=nb_flt, num=value)
            elif best == "highest":
                if value > good_thresh:
                    return '{{\color{{ForestGreen}} {num:.{prec}f} }}'.format(prec=nb_flt, num=value)
                if value < good_thresh and value > bad_thresh:
                    return '{{\color{{Orange}} {num:.{prec}f} }}'.format(prec=nb_flt, num=value)
                if value < bad_thresh:
                    return '{{\color{{Red}} {num:.{prec}f} }}'.format(prec=nb_flt, num=value)
            else:
                raise ValueError

        collected_results = []
        energies, ARAND = [], []
        SEL_PROB = 0.1

        sorting_column_idx = 0

        keys_to_collect = [
            ['score_WS', 'cremi-score'],
            ['score_WS','adapted-rand'],
            ['score_WS','vi-merge'],
            ['score_WS', 'vi-split'],
            ['runtime'],
            # ['energy']
        ]

        nb_flt_digits = [
            3,
            3,
            3,
            3,
            2,
            # 2,
        ]
        nb_formats = [
            'f',
            'f',
            'f',
            'f',
            'e',
            # 'e',
        ]

        label_names = {
            'MutexWatershed' : "Abs Max",
            'mean': "Average",
            "max": "Max",
            "min": "Min",
            "sum": "Sum",
        }


        collected_results = []
        for agglo_type in ['MutexWatershed', 'mean', 'sum', 'min', 'max']:
            for non_link in ['False', 'True']:
                label = label_names[agglo_type]
                if eval(non_link):
                    label += " + CLC"
                new_table_entrance = [label]
                collected_configs = []
                for config, json_file in zip(config_list, json_file_list):
                    if config["agglo_type"] == agglo_type and config["non_link"] == eval(non_link):
                        collected_configs.append(config)
                if len(collected_configs) == 0:
                    continue

                result_matrix = np.empty((len(collected_configs), len(keys_to_collect)) )
                for i, config in enumerate(collected_configs):
                    for j, key in enumerate(keys_to_collect):
                        result_matrix[i,j] = return_recursive_key_in_dict(config, key)

                means, errors = result_matrix.mean(axis=0), result_matrix.std(axis=0)
                for j, key in enumerate(keys_to_collect):
                    # if key[-1] == 'adapted-rand':
                    #     print("Hola")
                    #     means[j] = 1. - means[j]
                    # new_table_entrance.append("{0:.{prec}{type}}".format(means[j],prec=nb_flt_digits[j],
                    #                                                            type=nb_formats[j]) +
                    #                           " $\pm$ {0:.{prec}{type}}".format(errors[j],
                    #                                                                   prec=nb_flt_digits[j],
                    #                                                                 type=nb_formats[j]))
                    if key[-1] == 'adapted-rand':
                        new_table_entrance.append("{0:.{prec}{type}}".format(1.-means[j],prec=nb_flt_digits[j],
                                                                               type=nb_formats[j]))
                    else:
                        new_table_entrance.append("{0:.{prec}{type}}".format(means[j], prec=nb_flt_digits[j],
                                                                             type=nb_formats[j]))


                collected_results.append(new_table_entrance)

        collected_results = np.array(collected_results)
        collected_results = collected_results[collected_results[:,sorting_column_idx+1].argsort()]
        np.savetxt(os.path.join(scores_path, "collected_cremi.csv"), collected_results, delimiter=' & ',
                   fmt='%s',
                   newline=' \\\\\n')



    def make_plots(self, project_directory):
        self.collect_scores(project_directory)


        exp_name = self.fixed_kwargs['experiment_name']
        scores_path = os.path.join(project_directory, exp_name, "scores")
        results_collected = self.get_plot_data(scores_path)

        key_x = ['score_WS', 'vi-merge']
        key_y = ['score_WS', 'adapted-rand']
        # key_y = ['score_WS', 'vi-split']
        # key_y = ['energy']
        # key_x = ['runtime']
        key_value = ['runtime']
        # FIXME: CHANGE FROM NOISE TO PROBS

        # TODO: reduce MWS RUNTIME by a constant

        colors = {'MutexWatershed': {'False': {'True': 'C4', 'False': 'C0'}},
                  'mean': {'False': {'True': 'C5', 'False': 'C1'},
                           'True': {'True': 'C6', 'False': 'C8'}},
                  'sum': {'False': {'False': 'C2'},
                          'True': {'False': 'C3'}},
                  }

        markers = {'MutexWatershed': {'False': {'False': '^'}},
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

        legend_labels = {'MutexWatershed': {'False': 'Mutex Watershed'},
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
        for nb_sample, sample in enumerate(["A", "B", "C"]):
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
                    for agglo_type in [ty for ty in ['MutexWatershed', 'mean', 'sum'] if ty in results_collected_crop]:
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
                                            # exp_name = data_dict.get('postproc_config', {}).get('experiment_name',
                                            #                                                     "")
                                            # if exp_name not in exp_name_data:
                                            #     # full_cremi, full_cremi_MC
                                            #     continue
                                            new_data = [return_recursive_key_in_dict(data_dict, key_x),
                                                        return_recursive_key_in_dict(data_dict, key_y),
                                                        return_recursive_key_in_dict(data_dict, key_value)]
                                            # if agglo_type == 'max':
                                            #     new_data[2] -= 80

                                            # if exp_name == "full_cremi":
                                            lab = label_names[-1]
                                            print(edge_prob, sample, agglo_type, non_link)
                                            ax.scatter(new_data[0], new_data[1],
                                                       c=color_list[-1],
                                                       marker='o', alpha=0.7, s=100,
                                                       label=lab)
                                            # else:
                                            #     print(ID)
                                            #     print(edge_prob, exp_name)
                                            #     ax.scatter(new_data[0], new_data[1], c=exp_name_data[exp_name][0],
                                            #                marker='o', alpha=0.7, s=100,
                                            #                label=exp_name_data[exp_name][1])

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
        plt.subplots_adjust(left=0.05, right=0.95, wspace=0.3)

        plot_dir = os.path.join(project_directory, exp_name, "plots")
        check_dir_and_create(plot_dir)

        # f.suptitle("Crop of CREMI sample {} (90 x 300 x 300)".format(sample))
        f.savefig(os.path.join(plot_dir,
                               'relative_merge_split_WS_sup_{}_full_CREMI.pdf'.format(selected_prob)),
                  format='pdf')


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
            "save_segm": True,
            "WS_growing": True,
            "edge_prob": 0.201,
            # "sample": "B",
            "experiment_name": "cropTrainSamples",
            "local_attraction": False,
            # "additional_model_keys": ["debug_postproc"],
            "compute_scores": True,
            "save_UCM": False,
            "noise_factor": 0.
        })


        self.kwargs_to_be_iterated.update({
            'agglo': [
                      "MEAN",  "greedyFixation_noLogCosts", "MEAN_constr", "GAEC", "greedyFixation", "GAEC_noLogCosts", "SingleLinkagePlusCLC", "CompleteLinkage", "CompleteLinkagePlusCLC", "MEAN_constr_logCosts",
                      "MEAN_logCosts", "SingleLinkage", "MutexWatershed"],
            # 'agglo': ["GAEC"],
            # 'agglo': ["MutexWatershed"],
            # 'agglo': ["MEAN_constr", "GAEC", "greedyFixation"],
            'sample': ["C"],
            # "additional_model_keys": ["debug_postproc"],
            # 'sample': ["B+", "A+", "C+"]
        })
    #     TODO: crops! agglo, from_superpixels, edge_prob, check and merge_edge, WS_grow

    def get_data(self, kwargs_iter=None):
        nb_threads_pool = 1
        nb_iterations = 1

        kwargs_iter = self.get_cremi_kwargs_iter(crop_iter=range(0, 1), subcrop_iter=range(4, 5), #4
                                                 init_kwargs_iter=kwargs_iter, nb_iterations=nb_iterations)

        return kwargs_iter, nb_threads_pool


    def collect_scores(self, project_directory):
        exp_name = self.fixed_kwargs['experiment_name']
        scores_path = os.path.join(project_directory, exp_name, "scores")
        config_list, json_file_list = self.get_list_of_runs(scores_path)

        def assign_color(value, good_thresh, bad_thresh, nb_flt, best="lowest"):
            if best == "lowest":
                if value < good_thresh:
                    return '{{\color{{ForestGreen}} {num:.{prec}f} }}'.format(prec=nb_flt, num=value)
                if value > good_thresh and value < bad_thresh:
                    return '{{\color{{Orange}} {num:.{prec}f} }}'.format(prec=nb_flt, num=value)
                if value > bad_thresh:
                    return '{{\color{{Red}} {num:.{prec}f} }}'.format(prec=nb_flt, num=value)
            elif best == "highest":
                if value > good_thresh:
                    return '{{\color{{ForestGreen}} {num:.{prec}f} }}'.format(prec=nb_flt, num=value)
                if value < good_thresh and value > bad_thresh:
                    return '{{\color{{Orange}} {num:.{prec}f} }}'.format(prec=nb_flt, num=value)
                if value < bad_thresh:
                    return '{{\color{{Red}} {num:.{prec}f} }}'.format(prec=nb_flt, num=value)
            else:
                raise ValueError

        collected_results = []
        energies, ARAND = [], []
        SEL_PROB = 0.23
        for config, json_file in zip(config_list, json_file_list):
            if config["edge_prob"] != SEL_PROB: # 0.232  0.231  0.19  0.23  0.2   0.201   0.0
                print(config["edge_prob"])
                continue
            print(config["sample"], json_file)
            CLC = config["non_link"]
            agglo_type = config["agglo_type"]
            new_table_entrance = [agglo_type, str(CLC)]
            if 'use_log_costs' in config['postproc_config']['generalized_HC_kwargs']['agglomeration_kwargs']['extra_aggl_kwargs']:
                new_table_entrance.append(str(config['postproc_config']['generalized_HC_kwargs']['agglomeration_kwargs']['extra_aggl_kwargs']['use_log_costs']))
            else:
                new_table_entrance.append("")
            energies.append(config['energy'])
            ARAND.append(config['score_WS']['adapted-rand'])
            new_table_entrance.append('{:.0f}'.format(config['energy']))
            # new_table_entrance.append('{:.0f}'.format(config['runtime']))
            new_table_entrance.append(assign_color(config['runtime'], 1000, 5000, 0))
            new_table_entrance.append(assign_color(1.-config['score_WS']['adapted-rand'], 0.9, 0.8, 4, best="highest"))
            new_table_entrance.append(assign_color(config['score_WS']['vi-merge'], 0.3, 0.45, 3))
            new_table_entrance.append(assign_color(config['score_WS']['vi-split'], 0.450, 0.6, 3))
            # new_table_entrance.append('{:.3f}'.format(config['score_WS']['vi-split']))
            collected_results.append(new_table_entrance)
        collected_results = np.array(collected_results)
        collected_results = collected_results[np.array(ARAND).argsort()]
        np.savetxt(os.path.join(scores_path, "collected_{}.csv".format(SEL_PROB)), collected_results, delimiter=' & ', fmt='%s',
                   newline=' \\\\\n')

class NoiseExperiment(CremiExperiment):
    def __init__(self, *super_args, **super_kwargs):
        super(NoiseExperiment, self).__init__(*super_args, **super_kwargs)

        self.fixed_kwargs.update({
            "dataset": "CREMI",
            "from_superpixels": False,
            "use_multicut": False,
            "save_segm": False,
            "WS_growing": True,
            "edge_prob": 0.,
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
            "noise_factor": np.concatenate((np.linspace(2., 4.5, 5), np.linspace(4.5, 10., 15)))
            # "noise_factor": [4.0]
            # 'sample': ["B"]
            # 'sample': ["B+", "A+", "C+"]
        })

    def get_data(self, kwargs_iter=None):
        nb_threads_pool = 25
        nb_iterations = 6

        kwargs_iter = self.get_cremi_kwargs_iter(crop_iter=range(5, 6), subcrop_iter=range(5, 6),
                                                 init_kwargs_iter=kwargs_iter, nb_iterations=nb_iterations,
                                                 noise_mod='merge-biased')

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
        # TODO: changed
        # key_y = ['score_WS', 'vi-merge']
        key_y = ['score_WS', 'adapted-rand']
        key_x = ['noise_factor']
        # key_y = ['score_WS', 'vi-split']
        # key_y = ['energy']
        # key_x = ['runtime']
        key_value = ['energy']


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
            'adapted-rand': "Rand-Score",
            # 'noise_factor': "Noise factor $\mathcal{K}$ - Amount of merge-biased noise added to edge weights",
            'noise_factor': "\\textbf{Under-clustering} noise added to edge weights",
            'energy': 'Difference in Multicut Objective'

        }

        update_rule_names = {
            'sum': "GASP Sum", 'MutexWatershed': "GASP Abs Max", 'mean': "GASP Average"
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

                        from matplotlib import rc
                        # rc('font', **{'family': 'serif', 'serif': ['Times']})
                        ## for Palatino and other serif fonts use:
                        # rc('font',**{'family':'serif','serif':['Palatino']})
                        rc('text', usetex=True)
                        # matplotlib.rcParams['mathtext.fontset'] = 'stix'

                        matplotlib.rcParams.update({'font.size': 12})


                        for k, selected_edge_prob in enumerate([0., 0.1]):
                            f, axes = plt.subplots(ncols=1, nrows=1, figsize=(9, 3.8))

                            results_to_plot = {}
                            for agglo_type in [ty for ty in ['sum', 'MutexWatershed', 'mean'] if ty in results_collected_crop]:
                                for non_link in [ty for ty in ['False', 'True'] if
                                                 ty in results_collected_crop[agglo_type]]:
                                    for local_attraction in [ty for ty in ['False'] if
                                                             ty in results_collected_crop[agglo_type][non_link]]:

                                        sub_dict = results_collected_crop[agglo_type][non_link][local_attraction]

                                        probs = []
                                        VI_split_median = []
                                        VI_merge = []
                                        runtimes = []
                                        runtimes_perc = []
                                        split_max = []
                                        split_min = []
                                        split_q_0_25 = []
                                        split_q_0_75 = []
                                        error_bars_merge = []
                                        counter_per_type = 0
                                        for noise_factor in sub_dict:
                                            multiple_VI_split = []
                                            multiple_VI_merge = []
                                            multiple_runtimes = []
                                            for ID in sub_dict[noise_factor]:
                                                data_dict = sub_dict[noise_factor][ID]

                                                if data_dict["edge_prob"] != selected_edge_prob:
                                                    continue

                                                # ZERO_CONSTANT = 5512152.22595745 if k == 0 else 10854057.255651988

                                                multiple_VI_split.append(
                                                    return_recursive_key_in_dict(data_dict, key_y))
                                                multiple_VI_merge.append(
                                                    return_recursive_key_in_dict(data_dict, key_x))
                                                multiple_runtimes.append(
                                                    return_recursive_key_in_dict(data_dict, key_value))
                                                if key_y[-1] == 'adapted-rand':
                                                    multiple_VI_split[-1] = 1 - multiple_VI_split[-1]

                                                counter_per_type += 1
                                            if len(multiple_VI_split) == 0:
                                                continue
                                            probs.append(float(noise_factor))

                                            multiple_VI_split = np.array(multiple_VI_split)
                                            VI_split_median.append(np.median(multiple_VI_split))
                                            split_max.append(multiple_VI_split.max())
                                            split_min.append(multiple_VI_split.min())
                                            split_q_0_25.append(np.percentile(multiple_VI_split, 25))
                                            split_q_0_75.append(np.percentile(multiple_VI_split, 75))

                                            multiple_VI_merge = np.array(multiple_VI_merge)
                                            VI_merge.append(multiple_VI_merge.mean())
                                            error_bars_merge.append(multiple_VI_merge.std())

                                            multiple_runtimes = np.array(multiple_runtimes)
                                            runtimes.append(multiple_runtimes.mean())
                                            runtimes_perc.append([np.percentile(multiple_runtimes, 25), np.percentile(multiple_runtimes, 75)] )

                                            # ax.scatter(multiple_VI_merge, multiple_VI_split, s=np.ones_like(multiple_VI_merge)*edge_prob * 500,
                                            #            c=colors[agglo_type][non_link][local_attraction], marker='o',
                                            #            alpha=0.3)

                                        if len(probs) == 0:
                                            continue
                                        probs = np.array(probs)

                                        split_max = np.array(split_max)
                                        split_min = np.array(split_min)
                                        VI_split_median = np.array(VI_split_median)

                                        # Set energy to zero:
                                        # if k == 0:
                                        #     VI_split_median +=
                                        # else:
                                        #     VI_split_median += 10854057.255651988

                                        split_q_0_25 = np.array(split_q_0_25)
                                        split_q_0_75 = np.array(split_q_0_75)

                                        error_bars_merge = np.array(error_bars_merge)
                                        VI_merge = np.array(VI_merge)

                                        runtimes = np.array(runtimes)
                                        runtimes_perc = np.array(runtimes_perc)

                                        results_to_plot[agglo_type+non_link] = (VI_split_median, runtimes, VI_merge, split_q_0_25, split_q_0_75, counter_per_type, runtimes_perc)

                                        # if (agglo_type=='mean' and non_link == 'True') or (agglo_type=='max' and local_attraction=='True'):
                                        #     continue


                            # Compute energy zero offsets:
                            all_energies = np.stack([results_to_plot[agg][1] for agg in results_to_plot])
                            all_merge = np.stack([results_to_plot[agg][2] for agg in results_to_plot])
                            for j in range(all_merge.shape[0]):
                                all_energies[j] = all_energies[j,np.argsort(all_merge[j,:])]
                                # all_merge[j] = all_merge[j, np.argsort(all_merge[j, :])]
                            ZERO_OFFSETS = -all_energies.mean(axis=0)
                            # ZERO_OFFSETS = all_energies.min(axis=0)
                            # ZERO_OFFSETS = 1
                            # prova = all_median[:,np.argsort(all_merge,axis=1)]


                            for agglo_type in [ty for ty in ['sum', 'MutexWatershed', 'mean'] if
                                               ty in results_collected_crop]:
                                for non_link in [ty for ty in ['False', 'True'] if
                                                 ty in results_collected_crop[agglo_type]]:
                                    for local_attraction in [ty for ty in ['False'] if
                                                             ty in results_collected_crop[agglo_type][
                                                                 non_link]]:

                                        VI_split_median, energies, VI_merge, split_q_0_25, split_q_0_75, counter_per_type, energies_perc = results_to_plot[agglo_type + non_link]

                                        # Compose plot label:
                                        plot_label_1 = update_rule_names[agglo_type]
                                        plot_label_2 = " + Constraints" if eval(non_link) else " "
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


                                        label = plot_label_1 + plot_label_2 + plot_label_3

                                        argsort = np.argsort(VI_merge)


                                        ax = axes
                                        ax.fill_between(VI_merge[argsort], split_q_0_25[argsort],
                                                        split_q_0_75[argsort],
                                                        alpha=0.32, facecolor=colors[agglo_type][non_link][local_attraction], label=label)
                                        ax.errorbar(VI_merge, VI_split_median,
                                                    # yerr=(VI_split_median - split_min, split_max - VI_split_median),
                                                    fmt='.',
                                                    color=colors[agglo_type][non_link][local_attraction], alpha=0.5,
                                                    )

                                        ax.plot(VI_merge[argsort], VI_split_median[argsort], '-',
                                                color=colors[agglo_type][non_link][local_attraction], alpha=0.8)


                                        # # Select axes with energies:
                                        # ax = axes[1]
                                        #
                                        # ax.errorbar(VI_merge[argsort], energies[argsort] + ZERO_OFFSETS,
                                        #             # yerr=(VI_split_median - split_min, split_max - VI_split_median),
                                        #             fmt='.',
                                        #             color=colors[agglo_type][non_link][local_attraction], alpha=0.5,
                                        #             )
                                        # ax.fill_between(VI_merge[argsort], energies_perc[:,0][argsort]+ ZERO_OFFSETS,
                                        #                 energies_perc[:,1][argsort] + ZERO_OFFSETS,
                                        #                 alpha=0.32,
                                        #                 facecolor=colors[agglo_type][non_link][local_attraction],
                                        #                 label=label)
                                        #
                                        #
                                        # ax.plot(VI_merge[argsort], energies[argsort] + ZERO_OFFSETS, '-',
                                        #         color=colors[agglo_type][non_link][local_attraction], alpha=0.8, label=label)

                                        # title = "No long-range connections ($p_{\mathrm{long}}=0$)" if k == 0 else "10\% long-range connections ($p_{\mathrm{long}}=0.1$)"
                                        # ax.set_title(title)
                                        # props = dict(boxstyle='round', facecolor='white', alpha=0.5)
                                        # ax.text(0.02, 0.95, title, transform=ax.transAxes, fontsize=10, fontweight='bold',
                                        #         verticalalignment='top', bbox=props)

                            # vis_utils.set_log_tics(ax, [-2, 0], [10], format="%.2f", axis='y')
                            # ////////////////
                            # ARAND plot
                            # ////////////////


                            ax = axes
                            ax.set_ylim([0.55, 0.98])
                            ax.set_xlim([2, 10])

                            # vis_utils.set_log_tics(ax, [-2,0], [10],  format="%.2f", axis='x')

                            # ax.set_xscale("log")

                            # ax.set_xticks(np.arange(0, 1, step=0.1))

                            # Reorder legend:
                            handles, labels = ax.get_legend_handles_labels()
                            # Original order: 0:Sum, 1:SumCLC, 2:MWS, 3:Mean, 4:MeanCLC
                            new_ordering = [4,1,3,2,0] if k == 0 else [3,4,1,0,2]

                            handles = [handles[new_indx] for new_indx in new_ordering]
                            labels = [labels[new_indx] for new_indx in new_ordering]

                            ax.legend(handles, labels) #loc="lower left"
                            # if k == 1:
                            ax.set_xlabel(legend_labels[key_x[-1]])
                            ax.set_ylabel(legend_labels[key_y[-1]])

                            if key_x[-1] in axis_ranges:
                                ax.set_xlim(axis_ranges[key_x[-1]])
                            if key_y[-1] in axis_ranges:
                                ax.set_ylim(axis_ranges[key_y[-1]])



                            # # ////////////////
                            # # ENERGY PLOT:
                            # # ////////////////
                            # ax = axes[1]
                            # ylim = [-100000, 1200000.] if k == 1 else [-100000, 500000.]
                            # # ax.set_ylim(ylim)
                            # ax.set_xlim([2, 10])
                            #
                            #
                            # # ax.set_yscale("log")
                            # #
                            # # ax.set_xticks(np.arange(0, 1, step=0.1))
                            #
                            # # Reorder legend:
                            # handles, labels = ax.get_legend_handles_labels()
                            # # Original order: 0:Sum, 1:SumCLC, 2:MWS, 3:Mean, 4:MeanCLC
                            # new_ordering = [4,3,2,0,1] if k == 0 else [4,2,3,1,0]
                            #
                            # handles = [handles[new_indx] for new_indx in new_ordering]
                            # labels = [labels[new_indx] for new_indx in new_ordering]
                            #
                            # ax.legend(handles, labels) #loc="lower left"
                            # ax.set_xlabel(legend_labels[key_x[-1]])
                            # ax.set_ylabel(legend_labels[key_value[-1]])
                            #
                            # if key_x[-1] in axis_ranges:
                            #     ax.set_xlim(axis_ranges[key_x[-1]])
                            # if key_y[-1] in axis_ranges:
                            #     ax.set_ylim(axis_ranges[key_y[-1]])
                            #
                            # # ////////////////
                            # # SAVE PLOT:
                            # # ////////////////
                            #
                            plt.subplots_adjust(bottom=0.15)

                            plot_dir = os.path.join(project_directory, exp_name, "plots")
                            check_dir_and_create(plot_dir)

                            # f.suptitle("Crop of CREMI sample {} (90 x 300 x 300)".format(sample))
                            f.savefig(os.path.join(plot_dir,
                                                   'under_segment_plots_{}.pdf'.format(k)),
                                      format='pdf')


    def make_plots_RAND(self, project_directory):
        exp_name = self.fixed_kwargs['experiment_name']
        scores_path = os.path.join(project_directory, exp_name, "scores")
        results_collected = self.get_plot_data(scores_path, sort_by="noise_factor")

        colors = {'MutexWatershed': {'False': {'True': 'C4', 'False': 'C0'}},
                  'mean': {'False': {'True': 'C5', 'False': 'C1'},
                           'True': {'True': 'C6', 'False': 'C8'}},
                  'sum': {'False': {'False': 'C2'},
                          'True': {'False': 'C3'}},
                  }
        # TODO: changed
        # key_y = ['score_WS', 'vi-merge']
        key_y = ['score_WS', 'adapted-rand']
        key_x = ['noise_factor']
        # key_y = ['score_WS', 'vi-split']
        # key_y = ['energy']
        # key_x = ['runtime']
        key_value = ['runtime']


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
            'adapted-rand': "Rand-Score",
            'noise_factor': "Noise factor $\mathcal{K}$ - Amount of merge-biased noise added to edge weights",
            'energy': 'Multicut energy'

        }

        update_rule_names = {
            'sum': "Sum", 'MutexWatershed': "Absolute Max", 'mean': "Mean"
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

                        from matplotlib import rc
                        # rc('font', **{'family': 'serif', 'serif': ['Times']})
                        ## for Palatino and other serif fonts use:
                        # rc('font',**{'family':'serif','serif':['Palatino']})
                        rc('text', usetex=True)
                        # matplotlib.rcParams['mathtext.fontset'] = 'stix'

                        matplotlib.rcParams.update({'font.size': 10})
                        f, axes = plt.subplots(ncols=1, nrows=2, figsize=(9, 7))

                        for k, selected_edge_prob in enumerate([0., 0.1]):
                            ax = axes[k]
                            for agglo_type in [ty for ty in ['sum', 'MutexWatershed', 'mean'] if ty in results_collected_crop]:
                                for non_link in [ty for ty in ['False', 'True'] if
                                                 ty in results_collected_crop[agglo_type]]:
                                    for local_attraction in [ty for ty in ['False'] if
                                                             ty in results_collected_crop[agglo_type][non_link]]:

                                        sub_dict = results_collected_crop[agglo_type][non_link][local_attraction]

                                        probs = []
                                        VI_split_median = []
                                        VI_merge = []
                                        runtimes = []
                                        split_max = []
                                        split_min = []
                                        split_q_0_25 = []
                                        split_q_0_75 = []
                                        error_bars_merge = []
                                        counter_per_type = 0
                                        for noise_factor in sub_dict:
                                            multiple_VI_split = []
                                            multiple_VI_merge = []
                                            multiple_runtimes = []
                                            for ID in sub_dict[noise_factor]:
                                                data_dict = sub_dict[noise_factor][ID]

                                                if data_dict["edge_prob"] != selected_edge_prob:
                                                    continue

                                                multiple_VI_split.append(
                                                    return_recursive_key_in_dict(data_dict, key_y))
                                                multiple_VI_merge.append(
                                                    return_recursive_key_in_dict(data_dict, key_x))
                                                multiple_runtimes.append(
                                                    return_recursive_key_in_dict(data_dict, key_value))
                                                if key_y[-1] == 'adapted-rand':
                                                    multiple_VI_split[-1] = 1 - multiple_VI_split[-1]

                                                counter_per_type += 1
                                            if len(multiple_VI_split) == 0:
                                                continue
                                            probs.append(float(noise_factor))

                                            multiple_VI_split = np.array(multiple_VI_split)
                                            VI_split_median.append(np.median(multiple_VI_split))
                                            split_max.append(multiple_VI_split.max())
                                            split_min.append(multiple_VI_split.min())
                                            split_q_0_25.append(np.percentile(multiple_VI_split, 25))
                                            split_q_0_75.append(np.percentile(multiple_VI_split, 75))

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

                                        split_max = np.array(split_max)
                                        split_min = np.array(split_min)
                                        VI_split_median = np.array(VI_split_median)
                                        split_q_0_25 = np.array(split_q_0_25)
                                        split_q_0_75 = np.array(split_q_0_75)

                                        error_bars_merge = np.array(error_bars_merge)
                                        VI_merge = np.array(VI_merge)

                                        runtimes = np.array(runtimes)

                                        # if (agglo_type=='mean' and non_link == 'True') or (agglo_type=='max' and local_attraction=='True'):
                                        #     continue

                                        # Compose plot label:
                                        plot_label_1 = update_rule_names[agglo_type]
                                        plot_label_2 = " + Cannot-Link " if eval(non_link) else " "
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


                                        label = plot_label_1 + plot_label_2 + plot_label_3

                                        argsort = np.argsort(VI_merge)
                                        ax.fill_between(VI_merge[argsort], split_q_0_25[argsort],
                                                        split_q_0_75[argsort],
                                                        alpha=0.32, facecolor=colors[agglo_type][non_link][local_attraction], label=label)
                                        ax.errorbar(VI_merge, VI_split_median,
                                                    # yerr=(VI_split_median - split_min, split_max - VI_split_median),
                                                    fmt='.',
                                                    color=colors[agglo_type][non_link][local_attraction], alpha=0.5,
                                                    )

                                        ax.plot(VI_merge[argsort], VI_split_median[argsort], '-',
                                                color=colors[agglo_type][non_link][local_attraction], alpha=0.8)
                                        # ax.plot(VI_merge[argsort], VI_split[argsort] + error_bars_split[argsort]*0.8, '-',
                                        #         color=colors[agglo_type][non_link][local_attraction], alpha=0.7)
                                        # ax.plot(VI_merge[argsort], error_bars_split[argsort],
                                        #         '-',
                                        #         color=colors[agglo_type][non_link][local_attraction], alpha=0.7,
                                        #         label=label)
                                        # ax.plot(VI_merge[argsort], VI_split[argsort],
                                        #         '-',
                                        #         color=colors[agglo_type][non_link][local_attraction], alpha=0.7)
                                        # ax.fill_between(VI_merge[argsort],
                                        #                 error_bars_split[argsort],
                                        #                 VI_split[argsort],
                                        #                 alpha=0.4,
                                        #                 facecolor=colors[agglo_type][non_link][local_attraction])


                                        # ax.plot(np.linspace(0.0, 0.9, 15), [VI_split[0] for _ in range(15)], '.-',
                                        #         color=colors[agglo_type][non_link][local_attraction], alpha=0.8,label = plot_label_1 + plot_label_2 + plot_label_3)

                                        title = "No long-range connections ($p_{\mathrm{long}}=0$)" if k == 0 else "10\% long-range connections ($p_{\mathrm{long}}=0.1$)"
                                        # ax.set_title(title)
                                        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
                                        ax.text(0.7, 0.1, title, transform=ax.transAxes, fontsize=10, fontweight='bold',
                                                verticalalignment='top', bbox=props)

                            # vis_utils.set_log_tics(ax, [-2, 0], [10], format="%.2f", axis='y')

                            # TODO: changed
                            ax.set_ylim([0.55, 0.98])
                            ax.set_xlim([2, 10])

                            # vis_utils.set_log_tics(ax, [-2,0], [10],  format="%.2f", axis='x')





                            # ax.set_xscale("log")

                            # ax.set_xticks(np.arange(0, 1, step=0.1))

                            # Reorder legend:
                            handles, labels = ax.get_legend_handles_labels()
                            # Original order: 0:Sum, 1:SumCLC, 2:MWS, 3:Mean, 4:MeanCLC
                            new_ordering = [4,1,3,2,0] if k == 0 else [3,4,1,0,2]

                            handles = [handles[new_indx] for new_indx in new_ordering]
                            labels = [labels[new_indx] for new_indx in new_ordering]

                            ax.legend(handles, labels) #loc="lower left"
                            if k == 1:
                                ax.set_xlabel(legend_labels[key_x[-1]])
                            ax.set_ylabel(legend_labels[key_y[-1]])

                            if key_x[-1] in axis_ranges:
                                ax.set_xlim(axis_ranges[key_x[-1]])
                            if key_y[-1] in axis_ranges:
                                ax.set_ylim(axis_ranges[key_y[-1]])

                            # ax.set_xlim([0.15, 0.35])

                        plt.subplots_adjust(hspace=0.2)

                        plot_dir = os.path.join(project_directory, exp_name, "plots")
                        check_dir_and_create(plot_dir)

                        # f.suptitle("Crop of CREMI sample {} (90 x 300 x 300)".format(sample))
                        f.savefig(os.path.join(plot_dir,
                                               'noise_plot_{}_deep_z_noise_local.pdf'.format(sample)),
                                  format='pdf')





class NoiseExperimentSplit(CremiExperiment):
    def __init__(self, *super_args, **super_kwargs):
        super(NoiseExperimentSplit, self).__init__(*super_args, **super_kwargs)

        self.fixed_kwargs.update({
            "dataset": "CREMI",
            "from_superpixels": False,
            "use_multicut": False,
            "save_segm": False,
            "WS_growing": True,
            "edge_prob": 0.1,
            "sample": "B",
            "experiment_name": "simplexNoiseSplitBiasedAllEdgesFromPix",
            "local_attraction": False,
            "additional_model_keys": ["noise_sups"],
            "compute_scores": True,
            "save_UCM": False,
            # "noise_factor": 0.
        })
        # TODO: agglos, noise factor! Save segm, save noisy affs
        self.kwargs_to_be_iterated.update({
            'agglo': ["MEAN", "MutexWatershed", "MEAN_constr", "GAEC", "greedyFixation"],
            # 'agglo': ["MEAN"],
            "noise_factor": np.concatenate((np.linspace(2., 4.5, 5), np.linspace(4.5, 10., 15)))
            # "noise_factor": [8.0]
            # 'sample': ["B"]
            # 'sample': ["B+", "A+", "C+"]
        })

    def get_data(self, kwargs_iter=None):
        nb_threads_pool = 28
        nb_iterations = 10

        kwargs_iter = self.get_cremi_kwargs_iter(crop_iter=range(5, 6), subcrop_iter=range(5, 6),
                                                 init_kwargs_iter=kwargs_iter, nb_iterations=nb_iterations,
                                                 noise_mod='split-biased')

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
            'adapted-rand': "Rand-Score",
            'noise_factor': "\\textbf{Over-clustering} noise added to edge weights",
            'energy': 'Multicut energy'

        }

        update_rule_names = {
            'sum': "GASP Sum", 'MutexWatershed': "GASP Abs Max", 'mean': "GASP Average"
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
                        from matplotlib import rc
                        rc('text', usetex=True)
                        matplotlib.rcParams.update({'font.size': 12})
                        # f, axes = plt.subplots(ncols=1, nrows=2, figsize=(9, 7))

                        for k, selected_edge_prob in enumerate([0., 0.1]):
                            f, axes = plt.subplots(ncols=1, nrows=1, figsize=(9, 3.8))

                            ax = axes
                            for agglo_type in [ty for ty in ['sum', 'MutexWatershed', 'mean'] if ty in results_collected_crop]:
                                for non_link in [ty for ty in ['False', 'True'] if
                                                 ty in results_collected_crop[agglo_type]]:
                                    for local_attraction in [ty for ty in ['False'] if
                                                             ty in results_collected_crop[agglo_type][non_link]]:

                                        sub_dict = results_collected_crop[agglo_type][non_link][local_attraction]

                                        probs = []
                                        VI_split_median = []
                                        VI_merge = []
                                        runtimes = []
                                        split_max = []
                                        split_min = []
                                        split_q_0_25 = []
                                        split_q_0_75 = []
                                        error_bars_merge = []
                                        counter_per_type = 0
                                        for noise_factor in sub_dict:
                                            multiple_VI_split = []
                                            multiple_VI_merge = []
                                            multiple_runtimes = []
                                            for ID in sub_dict[noise_factor]:
                                                data_dict = sub_dict[noise_factor][ID]

                                                if data_dict["edge_prob"] != selected_edge_prob:
                                                    continue

                                                multiple_VI_split.append(
                                                    return_recursive_key_in_dict(data_dict, key_y))
                                                multiple_VI_merge.append(
                                                    return_recursive_key_in_dict(data_dict, key_x))
                                                multiple_runtimes.append(
                                                    return_recursive_key_in_dict(data_dict, key_value))
                                                if key_y[-1] == 'adapted-rand':
                                                    multiple_VI_split[-1] = 1 - multiple_VI_split[-1]

                                                counter_per_type += 1
                                            if len(multiple_VI_split) == 0:
                                                continue
                                            probs.append(float(noise_factor))

                                            multiple_VI_split = np.array(multiple_VI_split)
                                            VI_split_median.append(np.median(multiple_VI_split))
                                            split_max.append(multiple_VI_split.max())
                                            split_min.append(multiple_VI_split.min())
                                            split_q_0_25.append(np.percentile(multiple_VI_split, 25))
                                            split_q_0_75.append(np.percentile(multiple_VI_split, 75))

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

                                        split_max = np.array(split_max)
                                        split_min = np.array(split_min)
                                        VI_split_median = np.array(VI_split_median)
                                        split_q_0_25 = np.array(split_q_0_25)
                                        split_q_0_75 = np.array(split_q_0_75)

                                        error_bars_merge = np.array(error_bars_merge)
                                        VI_merge = np.array(VI_merge)

                                        runtimes = np.array(runtimes)

                                        # if (agglo_type=='mean' and non_link == 'True') or (agglo_type=='max' and local_attraction=='True'):
                                        #     continue

                                        # Compose plot label:
                                        plot_label_1 = update_rule_names[agglo_type]
                                        plot_label_2 = " + Constraints" if eval(non_link) else " "
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

                                        print("Found in {} - : {} ({})".format(agglo_type, non_link, counter_per_type, k))


                                        label = plot_label_1 + plot_label_2 + plot_label_3

                                        argsort = np.argsort(VI_merge)
                                        ax.fill_between(VI_merge[argsort], split_q_0_25[argsort],
                                                        split_q_0_75[argsort],
                                                        alpha=0.32, facecolor=colors[agglo_type][non_link][local_attraction], label=label)
                                        ax.errorbar(VI_merge, VI_split_median,
                                                    # yerr=(VI_split_median - split_min, split_max - VI_split_median),
                                                    fmt='.',
                                                    color=colors[agglo_type][non_link][local_attraction], alpha=0.5,
                                                    )

                                        ax.plot(VI_merge[argsort], VI_split_median[argsort], '-',
                                                color=colors[agglo_type][non_link][local_attraction], alpha=0.8)

                                        # ax.plot(VI_merge[argsort], VI_split[argsort] - error_bars_split[argsort]*0.8, '-',
                                        #         color=colors[agglo_type][non_link][local_attraction], alpha=0.7, label=label)
                                        # ax.plot(VI_merge[argsort], VI_split[argsort] + error_bars_split[argsort]*0.8, '-',
                                        #         color=colors[agglo_type][non_link][local_attraction], alpha=0.7)
                                        # ax.plot(VI_merge[argsort], error_bars_split[argsort],
                                        #         '-',
                                        #         color=colors[agglo_type][non_link][local_attraction], alpha=0.7,
                                        #         label=label)
                                        # ax.plot(VI_merge[argsort], VI_split[argsort],
                                        #         '-',
                                        #         color=colors[agglo_type][non_link][local_attraction], alpha=0.7)
                                        # ax.fill_between(VI_merge[argsort],
                                        #                 error_bars_split[argsort],
                                        #                 VI_split[argsort],
                                        #                 alpha=0.4,
                                        #                 facecolor=colors[agglo_type][non_link][local_attraction])


                                        # ax.plot(np.linspace(0.0, 0.9, 15), [VI_split[0] for _ in range(15)], '.-',
                                        #         color=colors[agglo_type][non_link][local_attraction], alpha=0.8,label = plot_label_1 + plot_label_2 + plot_label_3)

                                        # title = "Without long-range connections" if k == 0 else "With 10% long-range connections"
                                        # # ax.set_title(title)
                                        # props = dict(boxstyle='round', facecolor='white', alpha=0.5)
                                        # ax.text(0.6, 0.1, title, transform=ax.transAxes, fontsize=10, fontweight='bold',
                                        #         verticalalignment='top', bbox=props)

                            # vis_utils.set_log_tics(ax, [-2, 0], [10], format="%.2f", axis='y')

                            ax.set_ylim([0.65, 0.98])
                            ax.set_xlim([2, 10])

                            # vis_utils.set_log_tics(ax, [-2,0], [10],  format="%.2f", axis='x')





                            # ax.set_xscale("log")

                            # ax.set_xticks(np.arange(0, 1, step=0.1))

                            # Reorder legend:
                            handles, labels = ax.get_legend_handles_labels()
                            # Original order: 0:Sum, 1:SumCLC, 2:MWS, 3:Mean, 4:MeanCLC
                            new_ordering = [3, 0, 1, 2, 4] if k == 0 else [3, 0, 1, 2, 4]

                            handles = [handles[new_indx] for new_indx in new_ordering]
                            labels = [labels[new_indx] for new_indx in new_ordering]

                            loc = 'best' if k == 1 else "lower left"
                            ax.legend(handles, labels, loc=loc)

                            # if k == 1:
                            ax.set_xlabel(legend_labels[key_x[-1]])
                            ax.set_ylabel(legend_labels[key_y[-1]])

                            if key_x[-1] in axis_ranges:
                                ax.set_xlim(axis_ranges[key_x[-1]])
                            if key_y[-1] in axis_ranges:
                                ax.set_ylim(axis_ranges[key_y[-1]])

                            # ax.set_xlim([0.15, 0.35])

                            plot_dir = os.path.join(project_directory, exp_name, "plots")
                            check_dir_and_create(plot_dir)
                            plt.subplots_adjust(bottom=0.15)

                            # f.suptitle("Crop of CREMI sample {} (90 x 300 x 300)".format(sample))
                            f.savefig(os.path.join(plot_dir,
                                                   'over_segment_plots_{}.pdf'.format(k)),
                                      format='pdf')



class PlotUCM(CremiExperiment):
    def __init__(self, *super_args, **super_kwargs):
        super(PlotUCM, self).__init__(*super_args, **super_kwargs)

        self.fixed_kwargs.update({
            "dataset": "CREMI",
            "from_superpixels": False,
            "use_multicut": False,
            "save_segm": True,
            "WS_growing": True,
            "edge_prob": 1.0,
            "sample": "B",
            "experiment_name": "plotUCM_v2",
            "local_attraction": False,
            # "additional_model_keys": ["thresh000"],
            "compute_scores": True,
            "save_UCM": True,
            "noise_factor": 0.
        })
        # TODO: affs noise, MWS
        self.kwargs_to_be_iterated.update({
            # 'agglo': ["GAEC_noLogCosts"],
            # 'agglo': ["MEAN"],
            'agglo': ["MutexWatershed"],
            # "noise_factor": np.concatenate((np.linspace(2., 4.5, 5), np.linspace(4.5, 10., 15)))
            # "noise_factor": [8.0]
            # 'sample': ["B"]
            # 'sample': ["B+", "A+", "C+"]
        })

    def get_data(self, kwargs_iter=None):
        nb_threads_pool = 1
        nb_iterations = 1

        # ":,13:17,110:560,270:720", ":,:, :, :"
        kwargs_iter = self.get_cremi_kwargs_iter(crop_iter=range(6, 7), subcrop_iter=range(6, 7),
                                                 init_kwargs_iter=kwargs_iter, nb_iterations=nb_iterations,
                                                 noise_mod='split-biased')

        return kwargs_iter, nb_threads_pool


    def make_plots(self, project_directory):
        import vigra
        import h5py
        from segmfriends import vis as vis_utils
        from segmfriends.utils.various import parse_data_slice
        import matplotlib.ticker as ticker

        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        matplotlib.rcParams['font.family'] = 'STIXGeneral'
        matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')


        # Import dataset data (raw, affs): TODO: update get_dts_function
        # from .load_datasets import get_dataset_data
        # affs, GT = get_dataset_data("CREMI", "B", ":,13:14,110:560,270:720", run_connected_components=False)
        sample = "B"
        # str_crop_slice = "1:2,13:14,110:560,270:720"  # B
        str_crop_slice = ":,33:38,:1050,:1050"  # B
        SLICE = 1
        SUB_SLICE = ":,:,:1000,:1000"
        # str_crop_slice = "1:2,2:31,-1200:-200,100:1100" #C
        slc = tuple(parse_data_slice(str_crop_slice))
        sub_slc = tuple(parse_data_slice(SUB_SLICE))

        from long_range_compare.data_paths import get_hci_home_path
        dt_path = os.path.join(get_hci_home_path(), "datasets/cremi/SOA_affinities/sample{}_train.h5".format(sample))
        inner_path_GT = "segmentations/groundtruth_fixed_OLD" if sample == "B" else "segmentations/groundtruth_fixed"
        inner_path_raw = "raw_old" if sample == "B" else "raw"
        inner_path_affs = "predictions/full_affs"


        with h5py.File(dt_path, 'r') as f:
            GT = f[inner_path_GT][slc[1:]]
            raw = f[inner_path_raw][slc[1:]].astype('float32') / 255.
            affs = f[inner_path_affs][slc]

        affs = affs[sub_slc]


        # IMPORT EXPERIMENTS:
        exp_name = self.fixed_kwargs['experiment_name']
        scores_path = os.path.join(project_directory, exp_name, "scores")
        configs, json_files = self.get_list_of_runs(scores_path)

        scores_path = os.path.join(project_directory, exp_name, "UCM")

        first = True
        for config_dict, config_filename in zip(configs, json_files):

            if first:
                # Plot affinities:
                f, axes = plt.subplots(ncols=1, nrows=4, figsize=(14, 28))
                for a in f.get_axes():
                    a.axis('off')

                axes[0].matshow(raw[SLICE], cmap='gray', interpolation='none')
                axes[1].matshow(affs[1,SLICE], cmap='gray_r', interpolation='none')
                axes[2].matshow(affs[10,SLICE], cmap='gray_r', interpolation='none')
                axes[3].matshow(affs[11,SLICE], cmap='gray_r', interpolation='none')

                plt.subplots_adjust(wspace=0, hspace=0)
                # plt.tight_layout()
                # f.savefig(plot_path, format='pdf')
                plot_path = os.path.join(project_directory, exp_name, "plots", "affs.pdf")
                f.savefig(plot_path, format='pdf')
                first = False

            for zoom in range(2):
                sub_slc = tuple(parse_data_slice(SUB_SLICE)) if zoom ==0 else tuple(parse_data_slice(":,:,700:1000,:300"))

                current_raw = raw[sub_slc[1:]]

                # PLOT UCM:

                ucm_path = os.path.join(project_directory, exp_name, "UCM", config_filename.replace(".json",".h5"))

                UCM = vigra.readHDF5(ucm_path, 'merge_times')[sub_slc]

                segm_path = os.path.join(project_directory, exp_name, "out_segms", config_filename.replace(".json", ".h5"))

                segm = vigra.readHDF5(segm_path, 'segm_WS')[sub_slc[1:]]



                mask_1 = UCM == -15
                nb_nodes = UCM.max()
                border_mask = UCM == nb_nodes
                nb_iterations = (UCM * np.logical_not(border_mask)).max()
                print(nb_iterations)

                plotted_UCM = UCM.copy()
                plotted_UCM[border_mask] = nb_iterations

                matplotlib.rcParams.update({'font.size': 15})
                f, axes = plt.subplots(ncols=2, nrows=1, figsize=(14, 7))
                for a in f.get_axes():
                    a.axis('off')

                ax = axes[1]
                # cax = ax.matshow(raw[SLICE], cmap='gray', alpha=1.)
                # cax = ax.matshow(affs[slc][0, 0], cmap='gray', alpha=0.2)
                cax = ax.matshow(plotted_UCM[1,SLICE], cmap=plt.get_cmap('Reds'),
                                 interpolation='none', alpha=0.6)
                # ax.matshow(vis_utils.mask_the_mask(np.logical_not(border_mask).astype('int')[1,SLICE], value_to_mask=1.), cmap='gray',
                #                  interpolation='none', alpha=1.)
                vis_utils.plot_segm(ax, segm, highlight_boundaries=True, alpha_labels=0., z_slice=SLICE,
                                    alpha_boundary=1.)

                # Plot thick border:


                # f.colorbar(cax, ax=ax, orientation='horizontal', extend='both')


                def fmt(x, pos):
                    a, b = '{:.2e}'.format(x).split('e')
                    b = int(b)
                    return r'${} \cdot 10^{{{}}}$'.format(a, b)

                check_dir_and_create(os.path.join(project_directory, exp_name, "plots"))
                plot_path = os.path.join(project_directory, exp_name, "plots", config_filename.replace(".json", "")  +"_{}.pdf".format(zoom))
                # cbar = f.colorbar(cax, ax=ax, orientation='vertical', ticks=[0 ,nb_iterations],format='$k = %d$',fraction=0.04, pad=0.04, extend='both')
                # cbar.solids.set_edgecolor("face")


                # PLOT SEGM:


                # matplotlib.rcParams.update({'font.size': 15})
                # f, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
                # for a in f.get_axes():
                #     a.axis('off')

                ax = axes[0]

                # cax = ax.matshow(raw[0], cmap='gray', alpha=1.)
                # cax = ax.matshow(affs[slc][0, 0], cmap='gray', alpha=0.2)
                vis_utils.plot_segm(ax, segm, background=current_raw,highlight_boundaries=True, z_slice=SLICE, alpha_labels=0.5, alpha_boundary=1.)
                # plot_path = os.path.join(project_directory, exp_name, "plots", config_filename.replace(".json", "_segm.pdf"))
                plt.subplots_adjust(wspace=0, hspace=0)
                # plt.tight_layout()
                # f.savefig(plot_path, format='pdf')

                f.savefig(plot_path, format='pdf')

