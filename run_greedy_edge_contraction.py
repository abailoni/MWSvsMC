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


import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from h5py import highlevel
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
import h5py

import getpass


from compareMCandMWS import utils as utils

from segmfriends.utils.config_utils import adapt_configs_to_model, recursive_dict_update
from segmfriends.utils import yaml2dict, parse_data_slice

from segmfriends.io.save import get_hci_home_path
from segmfriends.algorithms.agglo import GreedyEdgeContractionAgglomeraterFromSuperpixels
from segmfriends.algorithms.WS.WS_growing import SizeThreshAndGrowWithWS
from segmfriends.algorithms.blockwise import BlockWise

from skunkworks.metrics.cremi_score import cremi_score
from long_range_hc.postprocessing.pipelines import get_segmentation_pipeline

from compareMCandMWS.load_datasets import get_dataset_data, get_dataset_offsets, CREMI_crop_slices, CREMI_sub_crops_slices




# Add epsilon to affinities:
def add_epsilon(affs, eps=1e-5):
    p_min = eps
    p_max = 1. - eps
    return (p_max - p_min) * affs + p_min


def combine_crop_slice_str(crop_str, subcrop_str):
    """
    Limited version!!!

    - it does not work with crops like '::2'
    - the subcrop should not have the minus notation

    """
    crop_slc = parse_data_slice(crop_str)
    subcrop_slc = parse_data_slice(subcrop_str)
    assert len(crop_slc) == len(subcrop_slc)

    new_crop_slc = []
    for i in range(len(crop_slc)):
        start = 0 if crop_slc[i].start is None else crop_slc[i].start
        substart = 0 if subcrop_slc[i].start is None else subcrop_slc[i].start
        new_crop_slc.append(slice(start + substart, start ))



def get_segmentation(affinities, GT, dataset, sample, crop_slice, sub_crop_slice, edge_prob, agglo, local_attraction, save_UCM,
                     from_superpixels=False, use_multicut=False):

    offsets = get_dataset_offsets(dataset)
    # affinities, GT = get_dataset_data(dataset, sample, crop_slice, run_connected_components=False)
    # sub_crop = parse_data_slice(sub_crop_slice)
    # affinities = affinities[sub_crop]
    # GT = GT[sub_crop[1:]]
    # GT = vigra.analysis.labelVolumeWithBackground(GT.astype('uint32'))
    # affinities = add_epsilon(affinities)


    configs = {'models': yaml2dict('./experiments/models_config.yml'),
               'postproc': yaml2dict('./experiments/post_proc_config.yml')}
    model_keys = [agglo] if not local_attraction else [agglo, "impose_local_attraction"]
    if from_superpixels:
        if use_multicut:
            model_keys = ["use_fragmenter", 'multicut_exact']
        else:
            model_keys += ["gen_HC_WS"]
    configs = adapt_configs_to_model(model_keys, debug=True, **configs)
    post_proc_config = configs['postproc']
    post_proc_config['generalized_HC_kwargs']['probability_long_range_edges'] = edge_prob
    post_proc_config['generalized_HC_kwargs']['return_UCM'] = save_UCM

    n_threads = post_proc_config.pop('nb_threads')
    invert_affinities = post_proc_config.pop('invert_affinities', False)
    segm_pipeline_type = post_proc_config.pop('segm_pipeline_type', 'gen_HC')

    segmentation_pipeline = get_segmentation_pipeline(
        segm_pipeline_type,
        offsets,
        nb_threads=n_threads,
        invert_affinities=invert_affinities,
        return_fragments=False,
        **post_proc_config
    )

    if post_proc_config.get('use_final_agglomerater', False):
        final_agglomerater = GreedyEdgeContractionAgglomeraterFromSuperpixels(
                        offsets,
                        n_threads=n_threads,
                        invert_affinities=invert_affinities,
                         **post_proc_config['generalized_HC_kwargs']['final_agglomeration_kwargs']
        )
    else:
        final_agglomerater = None


    post_proc_solver = BlockWise(segmentation_pipeline=segmentation_pipeline,
              offsets=offsets,
                                 final_agglomerater=final_agglomerater,
              blockwise=post_proc_config.get('blockwise', False),
              invert_affinities=invert_affinities,
              nb_threads=n_threads,
              return_fragments=False,
              blockwise_config=post_proc_config.get('blockwise_kwargs', {}))




    print("Starting prediction...")
    tick = time.time()
    pred_segm, out_dict = post_proc_solver(affinities)
    MC_energy = out_dict['MC_energy']
    if save_UCM:
        UCM, mergeTimes = out_dict['UCM'], out_dict['mergeTimes']

    if 'agglomeration_data' in out_dict:
        # Make some nice plot! :)
        from segmfriends import vis as vis
        fig, ax = plt.subplots(ncols=1, nrows=2)

        aggl_data = out_dict['agglomeration_data']

        iterations = np.arange(aggl_data.shape[0])
        ax[0].plot(iterations, aggl_data[:,0])
        ax[0].set(ylabel='Maximum size of the segments')

        iterations = np.arange(aggl_data[:].shape[0])
        ax[1].plot(iterations, aggl_data[:, 1])
        ax[1].set(xlabel='iterations', ylabel='Highest cost in PQ')

        # ax.plot(iterations, aggl_data[:,1])
        # affs_repr = np.linalg.norm(affs_repr, axis=0, keepdims=True)

        # ax.imshow(affs_repr, interpolation="none")

        # vis.plot_output_affin(ax, affinities, nb_offset=16, z_slice=0)
        fig.savefig("./iteration_plot_{}.pdf".format(agglo))

        fig, ax = plt.subplots(ncols=1, nrows=2)

        iterations = np.arange(aggl_data[1:].shape[0])
        ax[0].plot(iterations, aggl_data[1:, 2])
        ax[0].set(ylabel='Mean size')
        ax[1].plot(iterations, aggl_data[1:, 3])
        ax[1].set(xlabel='iterations', ylabel='Variance size distribution')
        # ax.plot(iterations, aggl_data[:,1])
        # affs_repr = np.linalg.norm(affs_repr, axis=0, keepdims=True)

        # ax.imshow(affs_repr, interpolation="none")

        # vis.plot_output_affin(ax, affinities, nb_offset=16, z_slice=0)


    comp_time = time.time() - tick
    print("Post-processing took {} s".format(comp_time))
    # print("Pred. sahpe: ", pred_segm.shape)
    # if not use_test_datasets:
    #     print("GT shape: ", gt.shape)
    #     print("Min. GT label: ", gt.min())

    # if post_proc_config.get('stacking_2D', False):
    #     print('2D stacking...')
    #     stacked_pred_segm = np.empty_like(pred_segm)
    #     max_label = 0
    #     for z in range(pred_segm.shape[0]):
    #         slc = vigra.analysis.labelImage(pred_segm[z].astype(np.uint32))
    #         stacked_pred_segm[z] = slc + max_label
    #         max_label += slc.max() + 1
    #     pred_segm = stacked_pred_segm

    # pred_segm_WS = None

    if dataset == "ISBI":
        # Make 2D segmentation:
        # Compute 2D scores:
        segm_2D = np.empty_like(pred_segm)
        max_label = 0
        for z in range(pred_segm.shape[0]):
            segm_2D[z] = pred_segm[z] + max_label
            max_label += pred_segm[z].max() + 1
        pred_segm = vigra.analysis.labelVolume(segm_2D.astype('uint32'))


    # if post_proc_config.get('thresh_segm_size', 0) != 0:
    if from_superpixels:
        pred_segm_WS = pred_segm
    else:
        grow = SizeThreshAndGrowWithWS(post_proc_config['thresh_segm_size'],
                 offsets,
                 hmap_kwargs=post_proc_config['prob_map_kwargs'],
                 apply_WS_growing=True,)
        pred_segm_WS = grow(affinities, pred_segm)


    fig, ax = plt.subplots(ncols=1, nrows=1)
    vis.plot_segm(ax, pred_segm_WS, z_slice=0, )
    fig.savefig("./segm_{}.pdf".format(agglo))


    # SAVING RESULTS:
    evals = cremi_score(GT, pred_segm, border_threshold=None, return_all_scores=True)
    evals_WS = cremi_score(GT, pred_segm_WS, border_threshold=None, return_all_scores=True)
    print("Scores achieved WS: ", evals_WS)
    print("Scores achieved: ", evals)

    ID = str(np.random.randint(1000000000))

    extra_agglo = post_proc_config['generalized_HC_kwargs']['agglomeration_kwargs']['extra_aggl_kwargs']
    if use_multicut:
        agglo_type = "MC_WS"
        non_link = None
    else:
        agglo_type = extra_agglo['update_rule']
        non_link = extra_agglo['add_cannot_link_constraints']

    EXPORT_PATH = os.path.join(get_hci_home_path(), 'GEC_comparison_longRangeGraph')

    result_file = os.path.join(EXPORT_PATH, '{}_{}_{}_{}.json'.format(ID,sample,agglo_type,non_link))

    new_results = {}
    new_results["agglo_type"] = agglo_type
    new_results["dataset"] = dataset
    new_results["sample"] = sample
    new_results["crop"] = crop_slice
    new_results["subcrop"] = sub_crop_slice
    new_results["save_UCM"] = save_UCM
    new_results["local_attraction"] = local_attraction
    new_results["ID"] = ID
    new_results["non_link"] = non_link
    new_results["edge_prob"] = edge_prob
    new_results.update({'energy': np.asscalar(MC_energy), 'score': evals, 'score_WS': evals_WS, 'runtime': comp_time})
    new_results['postproc_config'] = post_proc_config

    if from_superpixels:
        new_results["from_superpixels"] = "DTWS"

    with open(result_file, 'w') as f:
        json.dump(new_results, f, indent=4, sort_keys=True)
        # yaml.dump(result_dict, f)

    if save_UCM:
        UCM_file = os.path.join(EXPORT_PATH, 'UCM', '{}_{}_{}_{}.h5'.format(ID, sample, agglo_type, non_link))
        # vigra.writeHDF5(UCM, UCM_file, 'UCM')
        vigra.writeHDF5(mergeTimes[:3].astype('int64'), UCM_file, 'merge_times')





if __name__ == '__main__':

    # cremi_datasets = {}
    # for sample in ['A', "B", "C"]:
    #     cremi_datasets[sample] = get_dataset_data('CREMI', sample)


    # root_path = "/export/home/abailoni/supervised_projs/divisiveMWS"
    # dataset_path = os.path.join(root_path, "cremi-dataset-crop")
    # dataset_path = "/net/hciserver03/storage/abailoni/datasets/ISBI/"
    # plots_path = os.path.join("/net/hciserver03/storage/abailoni/greedy_edge_contr/plots")
    # save_path = os.path.join(root_path, "outputs")




    all_agglo_type = []
    all_edge_prob = []
    all_samples = []
    all_local_attr = []
    all_crops = []
    all_sub_crops = []
    all_UCM = []
    all_datasets = []
    all_affinites = []
    all_GTs = []

    len_cremi_slices = max([len(CREMI_crop_slices[smpl]) for smpl in CREMI_crop_slices])
    len_cremi_sub_slices = len(CREMI_sub_crops_slices)

    all_affinities_blocks = {
        "A": [[None for _ in range(len_cremi_sub_slices)] for _ in range(len_cremi_slices)],
        "B": [[None for _ in range(len_cremi_sub_slices)] for _ in range(len_cremi_slices)],
        "C" : [[None for _ in range(len_cremi_sub_slices)] for _ in range(len_cremi_slices)], }
    all_GT_blocks = {
        "A": [[None for _ in range(len_cremi_sub_slices)] for _ in range(len_cremi_slices)],
        "B": [[None for _ in range(len_cremi_sub_slices)] for _ in range(len_cremi_slices)],
        "C" : [[None for _ in range(len_cremi_sub_slices)] for _ in range(len_cremi_slices)], }

    tick = time.time()

    print("Loading...")
    # for sample in ["B"]:
    #     for crop in range(4):
    #         for sub_crop in range(4):
    #             affinities, GT = get_dataset_data("CREMI", sample, crop_slices[sample][crop], run_connected_components=False)
    #             sub_crop_slc = parse_data_slice(sub_crops_slices[sub_crop])
    #             affinities = affinities[sub_crop_slc]
    #             GT = GT[sub_crop_slc[1:]]
    #             GT = vigra.analysis.labelVolumeWithBackground(GT.astype('uint32'))
    #             affinities = add_epsilon(affinities)
    #             all_affinities_blocks[sample][crop][sub_crop] = affinities
    #             all_GT_blocks[sample][crop][sub_crop] = GT

    check = False
    for _ in range(1):
        for sample in ["B"]:
            for crop in range(0,1):  # 5       MC: 4
                for sub_crop in range(5,6): # 5     MC: 6
                    if all_affinities_blocks[sample][crop][sub_crop] is None:
                        # Load data:
                        affinities, GT = get_dataset_data("CREMI", sample, CREMI_crop_slices[sample][crop],
                                                          run_connected_components=False)
                        sub_crop_slc = parse_data_slice(CREMI_sub_crops_slices[sub_crop])
                        affinities = affinities[sub_crop_slc]
                        GT = GT[sub_crop_slc[1:]]
                        GT = vigra.analysis.labelVolumeWithBackground(GT.astype('uint32'))
                        affinities = add_epsilon(affinities)
                        all_affinities_blocks[sample][crop][sub_crop] = affinities
                        all_GT_blocks[sample][crop][sub_crop] = GT

                    for local_attr in [False]:
                        for agglo in ['MEAN', 'MEAN_constr', "GAEC", "MAX", 'greedyFixation']:
                        # for agglo in ['MEAN', 'MAX', 'greedyFixation', 'GAEC', 'MEAN_constr']:
                        # for agglo in ['MEAN', 'MAX', 'MEAN_constr']:
                            if local_attr and agglo in ['greedyFixation', 'GAEC']:
                                continue
                            # for edge_prob in np.concatenate((np.linspace(0.0, 0.1, 17), np.linspace(0.11, 0.8, 18))):
                            for edge_prob in [0.05]:
                                all_datasets.append('CREMI')
                                all_samples.append(sample)
                                all_crops.append(CREMI_crop_slices[sample][crop])
                                all_sub_crops.append(CREMI_sub_crops_slices[sub_crop])
                                all_local_attr.append(local_attr)
                                all_agglo_type.append(agglo)
                                all_edge_prob.append(edge_prob)
                                saveUCM = True if edge_prob > 0.0999 and edge_prob < 0.1001 and agglo != 'MAX' else False
                                saveUCM = False
                                if saveUCM and not check:
                                    print("UCM scheduled!")
                                    check = True
                                all_UCM.append(saveUCM)
                                assert all_affinities_blocks[sample][crop][sub_crop] is not None
                                all_affinites.append(all_affinities_blocks[sample][crop][sub_crop])
                                all_GTs.append(all_GT_blocks[sample][crop][sub_crop])

    # for sample in ["C", "A"]:
    #     for crop in range(1,2):
    #         for sub_crop in range(2,3):
    #             if all_affinities_blocks[sample][crop][sub_crop] is None:
    #                 # Load data:
    #                 affinities, GT = get_dataset_data("CREMI", sample, crop_slices[sample][crop],
    #                                                   run_connected_components=False)
    #                 sub_crop_slc = parse_data_slice(sub_crops_slices[sub_crop])
    #                 affinities = affinities[sub_crop_slc]
    #                 GT = GT[sub_crop_slc[1:]]
    #                 GT = vigra.analysis.labelVolumeWithBackground(GT.astype('uint32'))
    #                 affinities = add_epsilon(affinities)
    #                 all_affinities_blocks[sample][crop][sub_crop] = affinities
    #                 all_GT_blocks[sample][crop][sub_crop] = GT
    #
    #             for local_attr in [False, True]:
    #                 for agglo in ['MAX']:
    #                 # for agglo in ['MEAN', 'MAX', 'MEAN_constr', 'greedyFixation', 'GAEC']:
    #                 # for agglo in ['MEAN', 'MAX', 'MEAN_constr']:
    #                     if local_attr and agglo in ['greedyFixation', 'GAEC']:
    #                         continue
    #                     for edge_prob in np.linspace(0.11, 0.7, 15):
    #                         all_datasets.append('CREMI')
    #                         all_samples.append(sample)
    #                         all_crops.append(crop_slices[sample][crop])
    #                         all_sub_crops.append(sub_crops_slices[sub_crop])
    #                         all_local_attr.append(local_attr)
    #                         all_agglo_type.append(agglo)
    #                         all_edge_prob.append(edge_prob)
    #                         saveUCM = True if edge_prob > 0.0999 and edge_prob < 0.1001 and agglo != 'MAX' else False
    #                         if saveUCM and not check:
    #                             print("UCM scheduled!")
    #                             check = True
    #                         all_UCM.append(saveUCM)
    #                         assert all_affinities_blocks[sample][crop][sub_crop] is not None
    #                         all_affinites.append(all_affinities_blocks[sample][crop][sub_crop])
    #                         all_GTs.append(all_GT_blocks[sample][crop][sub_crop])
    print("Loaded dataset in {}s".format(time.time() - tick))

    print("Agglomarations to run: ", len(all_datasets))

    # Multithread:
    from multiprocessing.pool import ThreadPool
    from itertools import repeat
    pool = ThreadPool(processes=1)



    pool.starmap(get_segmentation,
                 zip(all_affinites,
                     all_GTs,
                     all_datasets,
                     all_samples,
                     all_crops,
                     all_sub_crops,
                     all_edge_prob,
                     all_agglo_type,
                     all_local_attr,
                     all_UCM
                     ))

    pool.close()
    pool.join()


