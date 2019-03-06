import long_range_compare # Add missing package-paths


import vigra
import numpy as np
import os

import time
import json



from segmfriends.utils.config_utils import adapt_configs_to_model, recursive_dict_update
from segmfriends.utils import yaml2dict, parse_data_slice

from long_range_compare.data_paths import get_hci_home_path
from segmfriends.algorithms.agglo import GreedyEdgeContractionAgglomeraterFromSuperpixels
from segmfriends.algorithms.WS.WS_growing import SizeThreshAndGrowWithWS
from segmfriends.algorithms.blockwise import BlockWise
from segmfriends.algorithms import get_segmentation_pipeline

from skunkworks.metrics.cremi_score import cremi_score

from long_range_compare.load_datasets import get_dataset_data, get_dataset_offsets, CREMI_crop_slices, CREMI_sub_crops_slices

from long_range_compare.vis_UCM import save_UCM_video


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


def add_smart_noise_to_affs(affinities, scale_factor,
                            mod='merge-bias',
                            target_affs='short',
                            use_GT=False,
                            GT=None,
                            dump_affs=False,
                            dump_path=None):
    if dump_path is None:
        dump_path = os.path.join(get_hci_home_path(), 'affs_plus_local_smart_noise.h5')

    affinities = affinities.copy()
    temp_file = os.path.join(get_hci_home_path(), 'affs_plus_smart_noise.h5')
    # vigra.writeHDF5(GT.astype('uint64'), temp_file, 'GT')

    # from segmfriends import vis as vis
    # fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(7, 4))
    # for a in fig.get_axes():
    #     a.axis('off')
    #
    # vis.plot_output_affin(ax[0], affinities, 1, 15)


    if target_affs == 'short':
        noise_slc = slice(0, 3)
    elif target_affs == 'long':
        noise_slc = slice(3, None)
    else:
        raise ValueError

    # vigra.writeHDF5(affinities[noise_slc], temp_file, 'affs')

    # from segmfriends.transform.segm_to_bound import compute_mask_boundaries
    #
    # true_merges = np.logical_not(compute_mask_boundaries(GT,
    #                             np.array(offsets[noise_slc]),
    #                             compress_channels=False,
    #                             channel_affs=0))

    def apply_median_filter(affs):
        from scipy.ndimage.filters import gaussian_filter, median_filter
        for aff_nb in range(affs.shape[0]):
            for z_slice in range(affs.shape[1]):
                affs[aff_nb, z_slice] = median_filter(affs[aff_nb, z_slice], 7)
        return affs

    if mod == 'merge-bias':
        # ---------------------------------------
        # Increase merges:
        # affinities[noise_slc] += (1 - (1-affinities[noise_slc])**2) *0.15
        # smart_noise_affs = np.absolute(np.random.normal(scale=affinities[noise_slc] * scale_factor, size=affinities[noise_slc].shape))

        smart_noise_affs = np.absolute(
            np.random.normal(scale=(1-affinities[noise_slc]) * scale_factor, size=affinities[noise_slc].shape))


        #
        # smart_noise_affs = np.absolute(np.random.normal(scale=(0.5 - np.abs(0.5 - affinities[noise_slc])) * scale_factor,
        #                                                 size=affinities[noise_slc].shape))

        # smart_GT_noise = np.absolute(np.random.normal(scale= np.abs(affinities[noise_slc] - true_merges.astype('float32')) * noise_factor, size=affinities[noise_slc].shape)) * np.logical_not(true_merges)

        # smart_noise_full = smart_noise_affs + smart_GT_noise
        smart_noise_full = smart_noise_affs
        apply_median_filter(smart_noise_full)
        affinities[noise_slc] += smart_noise_full
        # print("{0:.7f}".format(affinities.min()))
        affinities -= affinities.min()
        # print("{0:.7f}".format(affinities.min()))
    elif mod == 'split-bias':
        # ---------------------------------------
        # Increase splits:
        smart_noise_affs = np.absolute(np.random.normal(scale=(1 - affinities[noise_slc])**2  * scale_factor,
                                                        size=affinities[noise_slc].shape))

        # smart_GT_noise = np.absolute(np.random.normal(scale= np.abs(affinities[noise_slc] - true_merges.astype('float32')) * noise_factor, size=affinities[noise_slc].shape)) * np.logical_not(true_merges)

        # smart_noise_full = smart_noise_affs + smart_GT_noise
        smart_noise_full = smart_noise_affs
        apply_median_filter(smart_noise_full)
        affinities[noise_slc] -= smart_noise_full
    else:
        raise ValueError


    affinities = np.clip(affinities, 0., 1.)

    # # Add back some noise to the extremes:
    # affinities += np.random.normal(scale=0.00001, size=affinities.shape)
    # min_affs, max_affs = affinities.min(), affinities.max()
    # if min_affs < 0:
    #     affinities -= min_affs
    # if max_affs > 1.0:
    #     affinities /= max_affs

    # print("{0:.7f}".format(affinities.min()))
    # vigra.writeHDF5(affinities[noise_slc], temp_file, 'affs_mod')

    #
    # vis.plot_output_affin(ax[1], affinities, 1, 15)
    # # vis.plot_output_affin(ax, affinities, nb_offset=16, z_slice=0)
    # fig.savefig(os.path.join(get_hci_home_path(), "mod_affinities.pdf"))



    return affinities

def get_segmentation(affinities, GT, dataset, sample, crop_slice, sub_crop_slice, edge_prob, agglo, local_attraction, save_UCM,
                     from_superpixels=True, use_multicut=False, add_smart_noise=False, noise_factor=0.,
                     save_segm=False, WS_growing=True, additional_model_keys=None):
    affinities = affinities.copy()

    print(affinities.max(), affinities.min(), affinities.mean())

    offsets = get_dataset_offsets(dataset)

    # # Add back some noise to the extremes:
    # affinities += np.random.normal(scale=0.00001, size=affinities.shape)
    # min_affs, max_affs = affinities.min(), affinities.max()
    # if min_affs < 0:
    #     affinities -= min_affs
    # if max_affs > 1.0:
    #     affinities /= max_affs

    config_path = os.path.join(get_hci_home_path(), "pyCharm_projects/longRangeAgglo/experiments/cremi/configs")

    configs = {'models': yaml2dict(os.path.join(config_path, 'models_config.yml')),
               'postproc': yaml2dict(os.path.join(config_path, 'post_proc_config.yml'))}
    model_keys = [agglo] if not local_attraction else [agglo, "impose_local_attraction"]
    if from_superpixels:
        if use_multicut:
            model_keys = ["use_fragmenter", 'multicut_exact']
        else:
            model_keys += ["gen_HC_DTWS"] #DTWS
    if additional_model_keys is not None:
        model_keys += additional_model_keys
    configs = adapt_configs_to_model(model_keys, debug=True, **configs)
    post_proc_config = configs['postproc']
    post_proc_config['generalized_HC_kwargs']['agglomeration_kwargs']['offsets_probabilities'] = edge_prob
    if use_multicut:
        post_proc_config['multicut_kwargs']['offsets_probabilities'] = edge_prob
    post_proc_config['generalized_HC_kwargs']['agglomeration_kwargs']['return_UCM'] = save_UCM

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

    comp_time = time.time() - tick
    print("Post-processing took {} s".format(comp_time))

    extra_agglo = post_proc_config['generalized_HC_kwargs']['agglomeration_kwargs']['extra_aggl_kwargs']
    if use_multicut:
        agglo_type = "max"
        non_link = False
    else:
        agglo_type = extra_agglo['update_rule']
        non_link = extra_agglo['add_cannot_link_constraints']


    EXPORT_PATH = os.path.join(get_hci_home_path(), 'debug_nifty_agglo')

    # MWS in affogato could make problems... (and we also renormalize indices)
    pred_segm = vigra.analysis.labelVolume(pred_segm.astype('uint32'))

    pred_segm_WS = pred_segm

    export_file = os.path.join(EXPORT_PATH, 'debug_{}_{}_{}_new.h5'.format(sample, agglo_type, non_link))
    export_file = os.path.join(EXPORT_PATH, 'debug_B_mean_True.h5')
    print(export_file)

    # vigra.writeHDF5(pred_segm.astype('uint64'), export_file, 'segm')
    check_segm = vigra.readHDF5(export_file, 'segm')

    debug_check = (check_segm != pred_segm).sum() == 0

    print("\n---------------------")
    print("Debug check:", debug_check, agglo_type, non_link)
    print("---------------------\n")

    from segmfriends import vis as vis
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(7, 4))
    for a in fig.get_axes():
        a.axis('off')

    vis.plot_segm(ax[0], check_segm, 0)
    vis.plot_segm(ax[1], pred_segm, 0)
    ax[0].set_title("ciao")

    fig.savefig(os.path.join(os.path.join(EXPORT_PATH, 'debug_average.pdf')))





nb_threads_pool = 1
from_superpixels = True
use_multicut = False
add_smart_noise = False
save_segm = True
WS_growing = False
additional_model_keys = None

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
    all_noise_factors = []

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


    nb_threads_pool = 1

    def debug():
        global nb_threads_pool
        global from_superpixels
        global use_multicut
        global add_smart_noise
        global save_segm
        global WS_growing
        global additional_model_keys

        nb_threads_pool = 1
        from_superpixels = False
        use_multicut = False
        add_smart_noise = True
        save_segm = False
        WS_growing = True
        LONG_RANGE_EDGE_PROBABILITY = 0.0
        NOISE_FACTORS = [0.]
        # additional_model_keys = ['smart_noise_exp_merge_only_local']
        additional_model_keys = ['different_noise_shortEdges']
        # additional_model_keys = ['smart_noise_exp_merge_allLong_fromSP']

        # TODO: noise, save, crop
        for _ in range(1):
            # Generate noisy affinities:
            print("Generating noisy affs...")
            for crop in range(0, 1):  # Deep-z: 5       MC: 4   All: 0:4
                for sub_crop in range(5, 6):  # Deep-z: 5     MC: 6  All: 4 Tiny: 5
                    temp_file = os.path.join(get_hci_home_path(), 'debug_nifty_agglo/random_affs.h5')


                    # random_affs = np.random.uniform(size=(12, 10, 300, 300))
                    # print("SAVING!", random_affs.shape)
                    # vigra.writeHDF5(random_affs, temp_file, 'data')

                    random_affs = vigra.readHDF5(temp_file, 'data')



            print("Done!")
            check = False
            for sample in ["B"]:
                # crop_range = {"A": range(3,4),
                #               "B": range(0,1),
                #               "C": range(2,3)}

                # for crop in crop_range[sample]:  # Deep-z: 5       MC: 4   All: 0:4
                for crop in range(0, 1):  # Deep-z: 5       MC: 4   All: 0:4
                    for sub_crop in range(5, 6):  # Deep-z: 5     MC: 6  All: 4 Tiny: 5

                        for local_attr in [False]:
                            for agglo in ['MEAN_constr']:
                            # for agglo in ['MAX', 'greedyFixation', 'MEAN_constr', 'GAEC', 'MEAN']:
                                # for agglo in ['MEAN_constr', 'MAX', 'MEAN']:
                                # for agglo in ['MEAN', 'MAX', 'greedyFixation', 'GAEC', 'MEAN_constr']:
                                # for agglo in ['MEAN', 'MAX', 'MEAN_constr']:
                                if local_attr and agglo in ['greedyFixation', 'GAEC']:
                                    continue
                                for noise in NOISE_FACTORS:
                                    # for noise_factor in [0.9]:
                                    all_datasets.append('CREMI')
                                    all_samples.append(sample)
                                    all_crops.append(CREMI_crop_slices[sample][crop])
                                    all_sub_crops.append(CREMI_sub_crops_slices[sub_crop])
                                    all_local_attr.append(local_attr)
                                    all_agglo_type.append(agglo)
                                    all_edge_prob.append(LONG_RANGE_EDGE_PROBABILITY)
                                    # saveUCM = True if edge_prob > 0.0999 and edge_prob < 0.1001 and agglo != 'MAX' else False
                                    saveUCM = False
                                    if saveUCM and not check:
                                        print("UCM scheduled!")
                                        check = True
                                    all_UCM.append(saveUCM)
                                    all_affinites.append(random_affs)
                                    all_GTs.append(None)
                                    all_noise_factors.append(noise)



    from segmfriends.algorithms.agglo.greedy_edge_contr import runGreedyGraphEdgeContraction
    from nifty.graph import UndirectedGraph

    graph = UndirectedGraph(4)

    edges = [
        [0,1],
        [1,2],
        [2,3],
        [3,0],
        [1,3],
        [0,2],
    ]
    signedWeights = np.array([
        7.,
        6.,
        4.,
        5.,
        8.,
        -9.,
    ])

    graph.insertEdges(np.array(edges))

    nodeSeg, out_dict = runGreedyGraphEdgeContraction(graph,signedWeights,
                                  'mean',
                                  add_cannot_link_constraints=True,
                                  edge_sizes=np.ones_like(signedWeights),
                                  )

    print(nodeSeg)
    # print("Loading...")
    # tick = time.time()
    # # full_CREMI()
    # # smart_noise_merge_only_local()
    #
    # # smart_noise_merge_only_local()
    # # smart_noise_split()
    # debug()
    #
    # print("Loaded dataset in {}s".format(time.time() - tick))
    #
    # print("Agglomarations to run: ", len(all_datasets))
    #
    # # Multithread:
    # from multiprocessing.pool import ThreadPool
    # from itertools import repeat
    # pool = ThreadPool(processes=nb_threads_pool)
    #
    #
    #
    # pool.starmap(get_segmentation,
    #              zip(all_affinites,
    #                  all_GTs,
    #                  all_datasets,
    #                  all_samples,
    #                  all_crops,
    #                  all_sub_crops,
    #                  all_edge_prob,
    #                  all_agglo_type,
    #                  all_local_attr,
    #                  all_UCM,
    #                  repeat(from_superpixels),
    #                  repeat(use_multicut),
    #                  repeat(add_smart_noise),
    #                  all_noise_factors,
    #                  repeat(save_segm),
    #                  repeat(WS_growing),
    #                  repeat(additional_model_keys)
    #                  ))
    #
    # pool.close()
    # pool.join()
    #
    #
