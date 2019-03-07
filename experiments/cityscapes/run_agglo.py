# Add missing package-paths
import long_range_compare


import vigra
import numpy as np
import os

import time
import h5py

import matplotlib.pyplot as plt

from segmfriends.utils.config_utils import adapt_configs_to_model, recursive_dict_update
from segmfriends.utils import yaml2dict, parse_data_slice

from long_range_compare.data_paths import get_hci_home_path
from segmfriends.algorithms.agglo import GreedyEdgeContractionAgglomeraterFromSuperpixels
from segmfriends.algorithms.WS.WS_growing import SizeThreshAndGrowWithWS
from segmfriends.algorithms.blockwise import BlockWise

from skunkworks.metrics.cremi_score import cremi_score
from segmfriends.algorithms import get_segmentation_pipeline

from long_range_compare.load_datasets import get_dataset_data, get_dataset_offsets, CREMI_crop_slices, CREMI_sub_crops_slices, get_GMIS_dataset

from long_range_compare import GMIS_utils as GMIS_utils
from long_range_compare.data_paths import get_trendytukan_drive_path

from long_range_compare.GMIS_utils import LogisticRegression

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



def get_segmentation(image_path, input_model_keys, agglo, local_attraction, save_UCM,
                     from_superpixels=False, use_multicut=False):
    edge_prob = 1.
    THRESH = input_model_keys[0]

    inst_out_file = image_path.replace(
        '.input.h5', '.output.h5')
    inst_out_conf_file = image_path.replace(
        '.input.h5', '.inst.confidence.h5')

    # TODO: 1
    # NAME_AGGLO = "orig_affs"
    # THRESH = 'thresh030'
    # NAME_AGGLO = "finetuned_affs"
    # THRESH = 'thresh050'
    NAME_AGGLO = "finetuned_affs_avg"
    # THRESH = 'thresh050'

    # inner_path = agglo + "_avg_retrained_bal_affs_thresh050"
    partial_path = ""
    for key in input_model_keys:
        partial_path += "_{}".format(key)
    inner_path = "{}_{}{}".format(agglo, NAME_AGGLO, partial_path)
    # inner_path = "{}_orig_affs_thresh030".format(agglo)
    # print(inner_path)

    model_keys = [agglo] if not local_attraction else [agglo, "impose_local_attraction"]
    model_keys += input_model_keys

    already_exists = True
    with h5py.File(inst_out_file, 'r') as f:
        if inner_path not in f:
            already_exists = False
    with h5py.File(inst_out_conf_file, 'r') as f:
        if inner_path not in f:
            already_exists = False

    if already_exists:
        pbar.update(1)
        return
    # print(image_path)

    # print("Processing {}...".format(image_path))
    # Load data:
    with h5py.File(image_path, 'r') as f:
        # TODO: 2
        # affinities = f['instance_affinities'][:]
        # affinities = f['finetuned_affs_noAvg'][:]
        affinities = f['finetuned_affs'][:]


        # shape = f['shape'][:]
        # strides = f['offset_ranges'][:]
        # affs_prob = f['instance_affinities'][:]
        # affs_balanced = f['balanced_affs'][:]
        # class_prob = f['semantic_affinities'][:]
        # class_mask = f['semantic_argmax'][:]

    strides = np.array([1, 2, 4, 8, 16, 32], dtype=np.int32)
    offsets = GMIS_utils.get_offsets(strides)

    #
    # # -----------------------------------
    # # Pre-process affinities:
    # # -----------------------------------
    #
    # TODO: 3
    # affinities, foreground_mask_affs = GMIS_utils.combine_affs_and_mask(affinities, class_prob, class_mask, offsets)


    config_path = os.path.join(get_hci_home_path(), "pyCharm_projects/longRangeAgglo/experiments/cityscapes/configs")
    configs = {'models': yaml2dict(os.path.join(config_path, 'models_config.yml')),
               'postproc': yaml2dict(os.path.join(config_path, 'post_proc_config.yml'))}

    if from_superpixels:
        if use_multicut:
            model_keys = ["use_fragmenter", 'multicut_exact']
        else:
            model_keys += ["gen_HC_DTWS"]
    configs = adapt_configs_to_model(model_keys, debug=False, **configs)
    post_proc_config = configs['postproc']
    post_proc_config['generalized_HC_kwargs']['agglomeration_kwargs']['offsets_probabilities'] = edge_prob
    post_proc_config['generalized_HC_kwargs']['agglomeration_kwargs']['return_UCM'] = save_UCM

    # # Add longRange weights:
    # offset_weights = np.ones_like(offsets[:,0])
    # offset_weights[:16] = 35
    # offset_weights[16:32] = 20
    # offset_weights[32:] = 1
    #

    # affs_balanced = GMIS_utils.combine_affs_with_class(affs_balanced, class_prob, refine_bike=True, class_mask=class_mask)
    # affs_balanced = np.expand_dims(affs_balanced.reshape(affs_balanced.shape[0], affs_balanced.shape[1], -1), axis=0)
    # affs_balanced = np.rollaxis(affs_balanced, axis=-1, start=0)
    # affs_balanced *= foreground_mask_affs
    # post_proc_config['generalized_HC_kwargs']['agglomeration_kwargs']['offsets_weights'] = affs_balanced
    # post_proc_config['generalized_HC_kwargs']['agglomeration_kwargs']['extra_aggl_kwargs']['threshold'] = 0.25


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




    # print("Starting prediction...")
    tick = time.time()
    outs = post_proc_solver(affinities)
    if save_UCM:
        pred_segm, MC_energy, UCM, mergeTimes = outs
    else:
        pred_segm, MC_energy = outs
        # pred_segm = outs
        # MC_energy = 0
    comp_time = time.time() - tick
    # print("Post-processing took {} s".format(comp_time))

    # pred_segm *= foreground_mask

    # if post_proc_config.get('thresh_segm_size', 0) != 0:
    if from_superpixels:
        pred_segm_WS = pred_segm
    else:
        grow = SizeThreshAndGrowWithWS(post_proc_config['thresh_segm_size'],
                 offsets,
                 hmap_kwargs=post_proc_config['prob_map_kwargs'],
                 apply_WS_growing=False,
                                       debug=False)
        # pred_segm_WS = vigra.analysis.labelVolumeWithBackground(grow(affinities, pred_segm).astype(np.uint32), neighborhood='indirect')
        pred_segm_WS = grow(affinities, pred_segm)
        pred_segm_WS, _, _ = vigra.analysis.relabelConsecutive(pred_segm_WS)


    # TODO: allow all kinds of merges (not onyl local); skip connected components on the seeds
    # pred_segm_WS *= foreground_mask
    confidence_scores = GMIS_utils.get_confidence_scores(pred_segm_WS, affinities, offsets)


    # inner_path = "MEAN_bk_fixed"
    vigra.writeHDF5(pred_segm_WS[0].astype('uint16'), inst_out_file, inner_path)
    vigra.writeHDF5(confidence_scores, inst_out_conf_file, inner_path)
    vigra.writeHDF5(np.array([MC_energy['MC_energy']]), inst_out_conf_file, "MC_energy/" + inner_path)


    # # # # -------------------------------------------------
    # # # # PLOTTING:
    # # #
    # from segmfriends import vis as vis
    #
    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
    # for a in fig.get_axes():
    #     a.axis('off')
    #
    # # affs_repr = np.linalg.norm(affs_repr, axis=-1)
    # # ax.imshow(affs_repr, interpolation="none")
    #
    # vis.plot_segm(ax, pred_segm_WS, z_slice=0)
    #
    # # fig.savefig(pdf_path)
    # pdf_path = "./segm.pdf"
    # vis.save_plot(fig, os.path.dirname(pdf_path), os.path.basename(pdf_path))
    #
    # for off_stride in [0,8,16,]:
    #     # affs_repr = GMIS_utils.get_affinities_representation(affinities[:off_stride+8], offsets[:off_stride+8])
    #     # # affs_repr = GMIS_utils.get_affinities_representation(affinities[16:32], offsets[16:32])
    #     # affs_repr = np.rollaxis(affs_repr, axis=0, start=4)[0]
    #     # if affs_repr.min() < 0:
    #     #     affs_repr += np.abs(affs_repr.min())
    #     # affs_repr /= affs_repr.max()
    #
    #
    #     fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
    #     for a in fig.get_axes():
    #         a.axis('off')
    #
    #
    #     # affs_repr = np.linalg.norm(affs_repr, axis=-1)
    #     # ax.imshow(affs_repr, interpolation="none")
    #
    #     vis.plot_output_affin(ax, affinities, nb_offset=off_stride+3, z_slice=0)
    #
    #     pdf_path = image_path.replace(
    #         '.input.h5', '.affs_{}.pdf'.format(off_stride))
    #     # fig.savefig(pdf_path)
    #     pdf_path = "./balanced_affs_{}.pdf".format(off_stride)
    #     vis.save_plot(fig, os.path.dirname(pdf_path), os.path.basename(pdf_path))
    #     print(off_stride)
    #
    #
    #


    pbar.update(1)


    # ID = str(np.random.randint(1000000000))
    #
    # extra_agglo = post_proc_config['generalized_HC_kwargs']['agglomeration_kwargs']['extra_aggl_kwargs']
    # if use_multicut:
    #     agglo_type = "MC_WS"
    #     non_link = None
    # else:
    #     agglo_type = extra_agglo['update_rule']
    #     non_link = extra_agglo['add_cannot_link_constraints']
    #
    # EXPORT_PATH = os.path.join(get_hci_home_path(), 'GEC_comparison_longRangeGraph')
    #
    # result_file = os.path.join(EXPORT_PATH, '{}_{}_{}_{}.json'.format(ID,sample,agglo_type,non_link))
    #
    # new_results = {}
    # new_results["agglo_type"] = agglo_type
    # new_results["save_UCM"] = save_UCM
    # new_results["local_attraction"] = local_attraction
    # new_results["ID"] = ID
    # new_results["non_link"] = non_link
    # new_results["edge_prob"] = edge_prob
    # new_results.update({'energy': np.asscalar(MC_energy), 'score': evals, 'score_WS': evals_WS, 'runtime': comp_time})
    # new_results['postproc_config'] = post_proc_config
    #
    # if from_superpixels:
    #     new_results["from_superpixels"] = "DTWS"
    #
    # with open(result_file, 'w') as f:
    #     json.dump(new_results, f, indent=4, sort_keys=True)
    #     # yaml.dump(result_dict, f)
    #
    # if save_UCM:
    #     UCM_file = os.path.join(EXPORT_PATH, 'UCM', '{}_{}_{}_{}.h5'.format(ID, sample, agglo_type, non_link))
    #     # vigra.writeHDF5(UCM, UCM_file, 'UCM')
    #     vigra.writeHDF5(mergeTimes[:3].astype('int64'), UCM_file, 'merge_times')


def pool_initializer(l):
    global lock
    lock = l


if __name__ == '__main__':

    all_images_paths = get_GMIS_dataset(partial=False)

    # global trained_log_regr
    # trained_log_regr = GMIS_utils.LogRegrModel()

    all_paths_to_process = []
    all_agglo_type = []
    all_edge_prob = []
    all_local_attr = []
    all_UCM = []

    check = False
    for _ in range(1):
        for path in all_images_paths:
            for local_attr in [False]:
                for agglo, edge_prob, use_log_costs in [
                    # ["MEAN", "thresh030", False],
                    # ["MEAN", "thresh035", True],
                    # ["MEAN_constr", "thresh035", False],
                    # ["MEAN_constr", "thresh035", True],
                    # ["GAEC", "thresh035", False],
                    # ["GAEC", "thresh035", True],
                    ["MEAN_constr", "thresh030", False],
                    ["MEAN_constr", "thresh025", False],
                    ["MEAN_constr", "thresh020", False],
                    ["MEAN_constr", "thresh040", False],
                ]:
                    # for agglo in ['MEAN']:
                    # for agglo in ['MAX']:
                    # for agglo in ['MEAN_constr']:
                    # for agglo in ['MEAN_constr', 'MEAN', 'MutexWatershed', 'greedyFixation', 'GAEC',
                    #               'CompleteLinkage', 'CompleteLinkagePlusCLC', 'SingleLinkage', 'SingleLinkagePlusCLC']:
                    #     if local_attr and agglo in ['greedyFixation', 'GAEC']:
                    #         continue
                    #     # for edge_prob in np.concatenate((np.linspace(0.0, 0.1, 17), np.linspace(0.11, 0.8, 18))):
                    #     for edge_prob in ['thresh040', 'thresh045', 'thresh035']:
                    edge_prob = [edge_prob, "use_log_costs"] if use_log_costs else [edge_prob, "dont_use_log_costs"]

                    all_paths_to_process.append(path)
                    all_local_attr.append(local_attr)
                    all_agglo_type.append(agglo)
                    all_edge_prob.append(edge_prob)
                    # saveUCM = True if edge_prob > 0.0999 and edge_prob < 0.1001 and agglo != 'MAX' else False
                    saveUCM = False
                    if saveUCM and not check:
                        print("UCM scheduled!")
                        check = True
                    all_UCM.append(saveUCM)


    print("Agglomarations to run: ", len(all_paths_to_process))

    # Multithread:
    from multiprocessing.pool import ThreadPool
    from itertools import repeat
    from multiprocessing import Lock
    l = Lock()
    pool = ThreadPool(initializer=pool_initializer, initargs=(l,),  processes=12)

    from tqdm import tqdm
    #
    pbar = tqdm(total=len(all_paths_to_process))

    pool.starmap(get_segmentation,
                 zip(all_paths_to_process,
                     all_edge_prob,
                     all_agglo_type,
                     all_local_attr,
                     all_UCM
                     ))

    pool.close()
    pool.join()


