# Add missing package-paths
import long_range_compare


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
from PIL import Image



from segmfriends.utils.config_utils import adapt_configs_to_model, recursive_dict_update
from segmfriends.utils import yaml2dict, parse_data_slice

from long_range_compare.data_paths import get_hci_home_path
from segmfriends.algorithms.agglo import GreedyEdgeContractionAgglomeraterFromSuperpixels
from segmfriends.algorithms.WS.WS_growing import SizeThreshAndGrowWithWS
from segmfriends.algorithms.blockwise import BlockWise

from skunkworks.metrics.cremi_score import cremi_score
from long_range_hc.postprocessing.pipelines import get_segmentation_pipeline

from long_range_compare.load_datasets import get_dataset_data, get_dataset_offsets, CREMI_crop_slices, CREMI_sub_crops_slices, get_GMIS_dataset

from long_range_compare import GMIS_utils as GMIS_utils


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



def get_segmentation(image_path, edge_prob, agglo, local_attraction, save_UCM,
                     from_superpixels=False, use_multicut=False):

    # print("Processing {}...".format(image_path))
    # Load data:
    with h5py.File(image_path, 'r') as f:
        shape = f['shape'][:]
        strides = f['strides'][:]
        affs_prob = f['probs'][:]
        class_prob = f['class_probs'][:]
        class_mask = f['semantic_argmax'][:]


    # -----------------------------------
    # Pre-process affinities:
    # -----------------------------------
    strides = np.array([1, 2, 4, 8, 16, 32], dtype=np.int32)
    offsets = GMIS_utils.get_offsets(strides)

    # combined_affs = affs_prob
    combined_affs = GMIS_utils.combine_affs_with_class(affs_prob, class_prob, refine_bike=True, class_mask=class_mask)

    foreground_mask = GMIS_utils.get_foreground_mask(combined_affs)
    # foreground_mask_affs = np.tile(foreground_mask, reps=(combined_affs.shape[2], combined_affs.shape[3], 1, 1))
    # foreground_mask_affs = np.transpose(foreground_mask_affs, (2, 3, 0, 1))

    def compute_real_background_mask(foreground_mask,
                                offsets,
                                compress_channels=False,
                                channel_affs=-1):
        """
        Faster than the nifty version, but does not check the actual connectivity of the segments (no rag is
        built). A non-local edge could be cut, but it could also connect not-neighboring segments.
    b
        It returns a boundary mask (1 on boundaries, 0 otherwise). To get affinities reverse it.

        :param offsets: numpy array
            Example: [ [0,1,0], [0,0,1] ]

        :param return_boundary_affinities:
            if True, the output shape is (len(axes, z, x, y)
            if False, the shape is       (z, x, y)

        :param channel_affs: accepted options are 0 or -1
        """
        # TODO: use the version already implemented in the trasformations and using convolution kernels
        assert foreground_mask.ndim == 3
        ndim = 3

        padding = [[0, 0] for _ in range(3)]
        for ax in range(3):
            padding[ax][1] = offsets[:, ax].max()

        padded_foreground_mask= np.pad(foreground_mask, pad_width=padding, mode='constant', constant_values=True)
        crop_slices = [slice(0, padded_foreground_mask.shape[ax] - padding[ax][1]) for ax in range(3)]

        boundary_mask = []
        for offset in offsets:
            rolled_segm = padded_foreground_mask
            for ax, offset_ax in enumerate(offset):
                if offset_ax != 0:
                    rolled_segm = np.roll(rolled_segm, -offset_ax, axis=ax)
            boundary_mask.append((np.logical_and(padded_foreground_mask, rolled_segm))[crop_slices])

        boundary_affin = np.stack(boundary_mask)

        if compress_channels:
            compressed_mask = np.zeros(foreground_mask.shape[:ndim], dtype=np.int8)
            for ch_nb in range(boundary_affin.shape[0]):
                compressed_mask = np.logical_or(compressed_mask, boundary_affin[ch_nb])
            return compressed_mask

        if channel_affs == 0:
            return boundary_affin
        else:
            assert channel_affs == -1
            return np.transpose(boundary_affin, (1, 2, 3, 0))


    # RE-adjust affinities:
    def rescale_affs(affs, scale):
        p_min = scale
        p_max = 1. - scale
        return (p_max - p_min) * affs + p_min


    # scaling_factors = np.array([0.49, 0.49, 0.49, 0.0, 0.0, 0.0])
    # # scaling_factors = np.array([0., 0., 0., 0., 0., 0.])
    # # bias_factors = np.array([0.0501, 0.0901, 0., 0., 0., 0.0]) # More positive == merge more
    # bias_factors = np.array([-0.011, -0.011, -0., 0., 0., 0.0]) # More positive == merge more
    # only_attractive = [False, False, False, False, False, False]
    # for nb_str in range(strides.shape[0]):
    #     combined_affs[:,:,nb_str] = rescale_affs(combined_affs[:,:,nb_str], scaling_factors[nb_str]) + bias_factors[nb_str]
    #     if only_attractive[nb_str]:
    #         combined_affs[:, :, nb_str][combined_affs[:, :, nb_str] < 0.5] = 0.5

    # Mask affinities with the foreground_mask_affs:
    # combined_affs *= foreground_mask_affs


    # Reshape affinities in the expected nifty-shape:
    affinities = np.expand_dims(combined_affs.reshape(combined_affs.shape[0], combined_affs.shape[1], -1), axis=0)
    affinities = np.rollaxis(affinities, axis=-1, start=0)

    def distort_affs(affs):

        affs = affs.copy()



        def mod_affs(mod, slc, scale_factor=0.1):
            if mod == 'merge-bias':
                # ---------------------------------------
                # Increase merges:
                affs[slc] += (1 - (1 - affs[slc])) * scale_factor
            elif mod == 'split-bias':
                # ---------------------------------------
                # Increase splits:
                affs[slc] -= (1 - affs[slc]) * scale_factor

        slc_short = slice(0, 16)
        slc_middle = slice(16, 32)
        slc_long = slice(32, 48)
        mod_affs('split-bias', slc_short, scale_factor=0.2)
        mod_affs('split-bias', slc_middle, scale_factor=0.08)
        mod_affs('merge-bias', slc_long, scale_factor=0.06)

        affs = np.clip(affs, 0., 1.)

        # Add back some noise to the extremes:
        # affs += np.random.normal(scale=0.00001, size=affs.shape)
        # min_affs, max_affs = affs.min(), affs.max()
        # if min_affs < 0:
        #     affs -= min_affs
        # if max_affs > 1.0:
        #     affs /= max_affs

        return affs

    affinities = distort_affs(affinities)

    # affinities += np.random.normal(scale=0.0001, size=affinities.shape)
    # affinities -= affinities.min()
    # affinities /= affinities.max()


    foreground_mask_affs = compute_real_background_mask(np.expand_dims(foreground_mask, axis=0), offsets, channel_affs=0)

    affinities *= foreground_mask_affs

    # TODO: add noise, add offset_weights and thresh 0.3

    # fake_foreground = np.array([[[1, 1, 1], [1, 0, 1], [1, 1, 1], ]])
    # print(compute_real_background_mask(fake_foreground, offsets, channel_affs=0)[0], offsets[0])


    # # PLOTTING STUFF:
    #
    # if "frankfurt_000001_020693_leftImg8bit1_01" in image_path:
    #     from segmfriends import vis as vis
    #     for off_stride in [0,8,16,24,32,40]:
    #         affs_repr = GMIS_utils.get_affinities_representation(affinities[:off_stride+8], offsets[:off_stride+8])
    #         # affs_repr = GMIS_utils.get_affinities_representation(affinities[16:32], offsets[16:32])
    #         affs_repr = np.rollaxis(affs_repr, axis=0, start=4)[0]
    #         if affs_repr.min() < 0:
    #             affs_repr += np.abs(affs_repr.min())
    #         affs_repr /= affs_repr.max()
    #
    #
    #         fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
    #         for a in fig.get_axes():
    #             a.axis('off')
    #
    #
    #         affs_repr = np.linalg.norm(affs_repr, axis=-1)
    #         ax.imshow(affs_repr, interpolation="none")
    #
    #         # vis.plot_output_affin(ax, affinities, nb_offset=off_stride+3, z_slice=0)
    #
    #         pdf_path = image_path.replace(
    #             '.input.h5', '.affs_{}.pdf'.format(off_stride))
    #         # fig.savefig(pdf_path)
    #         vis.save_plot(fig, os.path.dirname(pdf_path), os.path.basename(pdf_path))
    #         print(off_stride)
    #
    #


    config_path = os.path.join(get_hci_home_path(), "pyCharm_projects/longRangeAgglo/experiments/cityscapes/configs")
    configs = {'models': yaml2dict(os.path.join(config_path, 'models_config.yml')),
               'postproc': yaml2dict(os.path.join(config_path, 'post_proc_config.yml'))}
    model_keys = [agglo] if not local_attraction else [agglo, "impose_local_attraction"]
    # model_keys += ['thresh030']
    if from_superpixels:
        if use_multicut:
            model_keys = ["use_fragmenter", 'multicut_exact']
        else:
            model_keys += ["gen_HC_DTWS"]
    configs = adapt_configs_to_model(model_keys, debug=False, **configs)
    post_proc_config = configs['postproc']
    post_proc_config['generalized_HC_kwargs']['agglomeration_kwargs']['offsets_probabilities'] = edge_prob
    post_proc_config['generalized_HC_kwargs']['agglomeration_kwargs']['return_UCM'] = save_UCM

    # Add longRange weights:
    offset_weights = np.ones_like(offsets[:,0])
    offset_weights[:16] = 35
    offset_weights[16:32] = 20
    offset_weights[32:] = 1

    post_proc_config['generalized_HC_kwargs']['agglomeration_kwargs']['offsets_weights'] = list(offset_weights)
    post_proc_config['generalized_HC_kwargs']['agglomeration_kwargs']['extra_aggl_kwargs']['threshold'] = 0.25


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


    # Save results:
    inst_out_file = image_path.replace(
        '.input.h5', '.output.h5')
    inst_out_conf_file = image_path.replace(
        '.input.h5', '.inst.confidence.h5')
    inst_out_conf_txt_file = image_path.replace(
        '.input.h5', '.inst.confidence.txt')

    # inner_path = agglo + "_" + str(local_attraction)
    inner_path = agglo + "_prove_exp_distortAffs_exp2"
    # inner_path = "MEAN_bk_fixed"
    vigra.writeHDF5(pred_segm_WS[0].astype('uint16'), inst_out_file, inner_path)
    vigra.writeHDF5(confidence_scores, inst_out_conf_file, inner_path)


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





if __name__ == '__main__':

    all_images_paths = get_GMIS_dataset(partial=False)

    all_paths_to_process = []
    all_agglo_type = []
    all_edge_prob = []
    all_local_attr = []
    all_UCM = []

    check = False
    for _ in range(1):
        for path in all_images_paths:
            for local_attr in [False]:
                # for agglo in ['MEAN_constr']:
                # for agglo in ['MAX']:
                for agglo in ['MEAN_constr', 'MEAN']:
                    if local_attr and agglo in ['greedyFixation', 'GAEC']:
                        continue
                    # for edge_prob in np.concatenate((np.linspace(0.0, 0.1, 17), np.linspace(0.11, 0.8, 18))):
                    for edge_prob in [1.]:
                        all_paths_to_process.append(path)
                        all_local_attr.append(local_attr)
                        all_agglo_type.append(agglo)
                        all_edge_prob.append(edge_prob)
                        saveUCM = True if edge_prob > 0.0999 and edge_prob < 0.1001 and agglo != 'MAX' else False
                        saveUCM = False
                        if saveUCM and not check:
                            print("UCM scheduled!")
                            check = True
                        all_UCM.append(saveUCM)

    print("Agglomarations to run: ", len(all_paths_to_process))

    # Multithread:
    from multiprocessing.pool import ThreadPool
    from itertools import repeat
    pool = ThreadPool(processes=12)

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


