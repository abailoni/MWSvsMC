from .load_datasets import get_dataset_data, get_dataset_offsets, CREMI_crop_slices, CREMI_sub_crops_slices
import vigra
import os
import numpy as np

import time
import json
import h5py

from segmfriends.utils.config_utils import adapt_configs_to_model, recursive_dict_update
from segmfriends.utils import yaml2dict, parse_data_slice, check_dir_and_create
from segmfriends.algorithms.agglo import GreedyEdgeContractionAgglomeraterFromSuperpixels
from segmfriends.algorithms.WS.WS_growing import SizeThreshAndGrowWithWS
from segmfriends.algorithms.blockwise import BlockWise
from segmfriends.algorithms import get_segmentation_pipeline

# FIXME: get rid of this skunkwork dependence!
from skunkworks.metrics.cremi_score import cremi_score

from .load_datasets import get_dataset_data, get_dataset_offsets, CREMI_crop_slices, CREMI_sub_crops_slices
from .data_paths import get_hci_home_path
from .vis_UCM import save_UCM_video


def run_clustering(affinities, GT, dataset, sample, crop_slice, sub_crop_slice, agglo,
                     experiment_name, project_directory, configs_dir_path,
                     edge_prob=1.0,
                     local_attraction=False,
                     save_UCM=False,
                     from_superpixels=True, use_multicut=False, noise_factor=0.,
                     save_segm=False, WS_growing=True, additional_model_keys=None, compute_scores=True,
                   mask_used_edges=None):
    # TODO: add experiment folder!
    # TODO: simplify/generalize function and move to segmfriends

    affinities = affinities.copy()

    # # FIXME: delete noise
    # affinities += np.random.normal(scale=1e-2,size=affinities.shape)
    # affinities = np.clip(affinities, 0., 1.)

    offsets = get_dataset_offsets(dataset)

    configs = {'models': yaml2dict(os.path.join(configs_dir_path, 'models_config.yml')),
               'postproc': yaml2dict(os.path.join(configs_dir_path, 'post_proc_config.yml'))}
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
    post_proc_config['generalized_HC_kwargs']['agglomeration_kwargs']['mask_used_edges'] = mask_used_edges

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

    if from_superpixels:
        pred_segm, out_dict = post_proc_solver(affinities, GT != 0)
    else:
        pred_segm, out_dict = post_proc_solver(affinities)
    MC_energy = out_dict['MC_energy']
    if save_UCM:
        UCM, mergeTimes = out_dict['UCM'], out_dict['mergeTimes']


    comp_time = time.time() - tick
    print("Post-processing took {} s".format(comp_time))

    # if 'agglomeration_data' in out_dict:
    #     # Make some nice plot! :)
    #     from segmfriends import vis as vis
    #     fig, ax = plt.subplots(ncols=1, nrows=2)
    #
    #     aggl_data = out_dict['agglomeration_data']
    #
    #     iterations = np.arange(aggl_data.shape[0])
    #     ax[0].plot(iterations, aggl_data[:,0])
    #     ax[0].set(ylabel='Maximum size of the segments')
    #
    #     iterations = np.arange(aggl_data[:].shape[0])
    #     ax[1].plot(iterations, aggl_data[:, 1])
    #     ax[1].set(xlabel='iterations', ylabel='Highest cost in PQ')
    #
    #     # ax.plot(iterations, aggl_data[:,1])
    #     # affs_repr = np.linalg.norm(affs_repr, axis=0, keepdims=True)
    #
    #     # ax.imshow(affs_repr, interpolation="none")
    #
    #     # vis.plot_output_affin(ax, affinities, nb_offset=16, z_slice=0)
    #     fig.savefig("./iteration_plot_{}.pdf".format(agglo))
    #
    #     fig, ax = plt.subplots(ncols=1, nrows=2)
    #
    #     iterations = np.arange(aggl_data[1:].shape[0])
    #     ax[0].plot(iterations, aggl_data[1:, 2])
    #     ax[0].set(ylabel='Mean size')
    #     ax[1].plot(iterations, aggl_data[1:, 3])
    #     ax[1].set(xlabel='iterations', ylabel='Variance size distribution')
    #     # ax.plot(iterations, aggl_data[:,1])
    #     # affs_repr = np.linalg.norm(affs_repr, axis=0, keepdims=True)
    #
    #     # ax.imshow(affs_repr, interpolation="none")
    #
    #     # vis.plot_output_affin(ax, affinities, nb_offset=16, z_slice=0)


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

    ID = str(np.random.randint(1000000000))

    extra_agglo = post_proc_config['generalized_HC_kwargs']['agglomeration_kwargs']['extra_aggl_kwargs']
    if use_multicut:
        raise DeprecationWarning
        agglo_type = "max"
        non_link = False
    else:
        agglo_type = extra_agglo['update_rule']
        non_link = extra_agglo['add_cannot_link_constraints']


    # MWS in affogato could make problems... (and we also renormalize indices)
    pred_segm = vigra.analysis.relabelConsecutive(pred_segm.astype('uint64'))[0]
    if pred_segm.max() > np.uint32(-1):
        print("!!!!!!!!!WARNING!!!!!!!!!! uint32 limit reached!")
    else:
        pred_segm = vigra.analysis.labelVolumeWithBackground(pred_segm.astype('uint32'))

    if post_proc_config.get('thresh_segm_size', 0) != 0 and WS_growing:
        grow = SizeThreshAndGrowWithWS(post_proc_config['thresh_segm_size'],
                 offsets,
                 hmap_kwargs=post_proc_config['prob_map_kwargs'],
                 apply_WS_growing=True,
                 size_of_2d_slices=False)
        pred_segm_WS = grow(affinities, pred_segm)
    else:
        pred_segm_WS = pred_segm

    # fig, ax = plt.subplots(ncols=1, nrows=1)
    # vis.plot_segm(ax, pred_segm_WS, z_slice=0, )
    # fig.savefig("./segm_{}.pdf".format(agglo))


    # ------------------------------
    # SAVING RESULTS:
    # ------------------------------
    experiment_dir_path = os.path.join(project_directory, experiment_name)
    check_dir_and_create(experiment_dir_path)

    # Save config setup:
    new_results = {}

    if mask_used_edges is not None:
        post_proc_config['generalized_HC_kwargs']['agglomeration_kwargs'].pop('mask_used_edges')
    new_results['postproc_config'] = post_proc_config

    # TODO: delete this crap
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
    new_results['from_superpixels'] = from_superpixels
    new_results['use_multicut'] = use_multicut
    new_results['noise_factor'] = noise_factor
    new_results['save_segm'] = save_segm
    new_results['WS_growing'] = WS_growing

    # Save scores:
    if compute_scores:
        evals = cremi_score(GT, pred_segm, border_threshold=None, return_all_scores=True)
        evals_WS = cremi_score(GT, pred_segm_WS, border_threshold=None, return_all_scores=True)
        print("Scores achieved ({} - {} - {}): ".format(agglo_type,non_link, noise_factor), evals_WS)
        new_results.update({'energy': np.asscalar(MC_energy), 'score': evals, 'score_WS': evals_WS, 'runtime': out_dict['runtime']})

    check_dir_and_create(os.path.join(experiment_dir_path, 'scores'))
    result_file = os.path.join(experiment_dir_path, 'scores', '{}_{}_{}_{}.json'.format(ID,sample,agglo_type,non_link))
    with open(result_file, 'w') as f:
        json.dump(new_results, f, indent=4, sort_keys=True)
        # yaml.dump(result_dict, f)

    # Save scores:
    if save_segm:
        check_dir_and_create(os.path.join(experiment_dir_path, 'out_segms'))
        export_file = os.path.join(experiment_dir_path, 'out_segms', '{}_{}_{}_{}.h5'.format(ID, sample, agglo_type, non_link))
        print('{}/out_segms/{}_{}_{}_{}.h5'.format(experiment_name, ID, sample, agglo_type, non_link))
        vigra.writeHDF5(pred_segm_WS.astype('uint64'), export_file, 'segm_WS')
        vigra.writeHDF5(pred_segm.astype('uint64'), export_file, 'segm')

    if save_UCM:
        # TODO: avoid saving to disk
        UCM_folder = os.path.join(experiment_dir_path, 'UCM')
        check_dir_and_create(UCM_folder)
        UCM_h5_file = os.path.join(experiment_dir_path, 'UCM', '{}_{}_{}_{}.h5'.format(ID, sample, agglo_type, non_link))
        # vigra.writeHDF5(UCM, UCM_h5_file, 'UCM')
        vigra.writeHDF5(mergeTimes[:3].astype('int64'), UCM_h5_file, 'merge_times')

        # save_UCM_video('{}_{}_{}_{}'.format(ID, sample, agglo_type, non_link), UCM_folder,
        #                selected_offset=1, selected_slice=0, nb_frames=100,
        #                postfix="allow_merge", final_segm=pred_segm_WS)



def grow_WS(json_filename, config_dict, project_directory, experiment_name):
    if config_dict["WS_growing"]:
        return

    experiment_dir_path = os.path.join(project_directory, experiment_name)
    export_file = os.path.join(experiment_dir_path, 'out_segms', json_filename.replace('.json', '.h5'))
    if not os.path.exists(export_file):
        return

    print(json_filename)

    post_proc_config = config_dict['postproc_config']
    offsets = get_dataset_offsets("CREMI")

    # Load affinities:
    print("Loading ", json_filename)
    affinities, _ = get_dataset_data("CREMI", config_dict["sample"], config_dict["crop"],
                                      run_connected_components=False)
    pred_segm = vigra.readHDF5(export_file, "segm")

    grow = SizeThreshAndGrowWithWS(post_proc_config['thresh_segm_size'],
             offsets,
             hmap_kwargs=post_proc_config['prob_map_kwargs'],
             apply_WS_growing=True,
             size_of_2d_slices=True,
                                   with_background=True)



    print("Computing WS ")
    try:
        pred_segm_WS = grow(affinities, pred_segm)
    except MemoryError:
        print("Memory error on ", json_filename)
        return

    # TODO: add option to compute scores
    # evals_WS = cremi_score(GT, pred_segm_WS, border_threshold=None, return_all_scores=True)
    # print("Scores achieved ({} - {} - {}): ".format(agglo_type, non_link, noise_factor), evals_WS)
    # new_results.update(
    #     {'energy': np.asscalar(MC_energy), 'score': evals, 'score_WS': evals_WS, 'runtime': out_dict['runtime']})

    # ------------------------------
    # SAVING RESULTS:
    # ------------------------------
    # TODO: save_growing True, delete old segm, update scores

    vigra.writeHDF5(pred_segm_WS.astype('uint64'), export_file, 'segm_WS')

    # Save config setup:
    config_dict["WS_growing"] = True

    json_file_path = os.path.join(experiment_dir_path, 'scores', json_filename)
    with open(json_file_path, 'w') as f:
        json.dump(config_dict, f, indent=4, sort_keys=True)


def delete_segm(json_filename, config_dict, project_directory, experiment_name):
    if not config_dict["WS_growing"]:
        return

    experiment_dir_path = os.path.join(project_directory, experiment_name)
    export_file = os.path.join(experiment_dir_path, 'out_segms', json_filename.replace('.json', '.h5'))
    if not os.path.exists(export_file):
        return

    with h5py.File(export_file, 'r') as f:
        keys = [k for k in f]
        if "segm" in f and "segm_WS" in f:
            segm = f["segm_WS"][:]
        else:
            return

    # Change type:
    max_value = segm.max()
    print(json_filename,"Max value: ", max_value)
    if (max_value < np.uint16(-10) and segm.min() >= 0):
        segm = segm.astype('uint16')
    elif (max_value < np.uint32(-10) and segm.min() >= 0):
        segm = segm.astype('uint32')

    with h5py.File(export_file, 'w') as f2:
        f2["segm_WS"] = segm





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


def add_opensimplex_noise_to_affs(affinities, scale_factor,
                            mod='add',
                            target_affs='all',
                            seed=None
                            ):
    affinities = affinities.copy()

    temp_file = os.path.join(get_hci_home_path(), 'affs_plus_opensimplex_noise.h5')
    # vigra.writeHDF5(affinities.astype('float32'), temp_file, 'affs')

    # import matplotlib.pyplot as plt
    # from segmfriends import vis as vis
    # fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(7, 4))
    # for a in fig.get_axes():
    #     a.axis('off')
    # vis.plot_output_affin(ax[0], affinities, 1, 15)



    if target_affs == 'short':
        noise_slc = slice(0, 3)
    elif target_affs == 'long':
        noise_slc = slice(3, None)
    elif target_affs == "all":
        noise_slc = slice(None)
    else:
        raise ValueError


    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    def logit(x, clip=True):
        if clip:
            x = add_epsilon(x)
        return np.log(x / (1. - x))

    def add_epsilon(affs, eps=1e-2):
        p_min = eps
        p_max = 1. - eps
        return (p_max - p_min) * affs + p_min

    # Generate noise:
    from nifty.graph.rag import generate_opensimplex_noise
    shape = affinities[noise_slc].shape

    large_ft_size = np.array((1., 3., 50., 50.))
    large_scale_noise = (generate_opensimplex_noise(shape, seed=seed, features_size=large_ft_size, number_of_threads=8)
                         +1.0) / 2.0
    fine_ft_size = np.array((1., 3., 20., 20.))
    fine_scale_noise = (generate_opensimplex_noise(shape, seed=seed, features_size=fine_ft_size, number_of_threads=8)
                        + 1.0) / 2.0

    # Combine large and fine features:
    # TODO: more or simplify?
    large_scale, fine_scale = 10, 5
    simplex_noise = (large_scale_noise * large_scale + fine_scale_noise * fine_scale) / (large_scale + fine_scale)

    if mod == "merge-biased":
        noisy_affs = sigmoid(logit(affinities[noise_slc]) + scale_factor * logit(np.maximum(simplex_noise, 0.5)))
    elif mod == "split-biased":
        noisy_affs = sigmoid(logit(affinities[noise_slc]) + scale_factor * logit(np.minimum(simplex_noise, 0.5)))
    elif mod == "unbiased":
        noisy_affs = sigmoid(logit(affinities[noise_slc]) + scale_factor * logit(simplex_noise))
    else:
        raise ValueError("Accepted mods are add or subtract")

    affinities[noise_slc] = noisy_affs

    # vis.plot_output_affin(ax[1], affinities, 1, 15)
    # # # vis.plot_output_affin(ax, affinities, nb_offset=16, z_slice=0)
    # fig.savefig(os.path.join(get_hci_home_path(), "perlin_noise.pdf"))

    # vigra.writeHDF5(affinities.astype('float32'), temp_file, 'affs_noisy')

    return affinities





def get_block_data_lists():
    len_cremi_slices = max([len(CREMI_crop_slices[smpl]) for smpl in CREMI_crop_slices])
    len_cremi_sub_slices = len(CREMI_sub_crops_slices)

    affinities_blocks = {
        smpl: [[None for _ in range(len_cremi_sub_slices)] for _ in range(len_cremi_slices)] for smpl in CREMI_crop_slices}
    masks_used_edges_blocks = {
        smpl: [[None for _ in range(len_cremi_sub_slices)] for _ in range(len_cremi_slices)] for smpl in
    CREMI_crop_slices}
    GT_blocks = {
        smpl: [[None for _ in range(len_cremi_sub_slices)] for _ in range(len_cremi_slices)] for smpl in CREMI_crop_slices}

    return affinities_blocks, masks_used_edges_blocks, GT_blocks


def get_kwargs_iter(fixed_kwargs, kwargs_to_be_iterated,
                    crop_iter, subcrop_iter,
                    init_kwargs_iter=None, nb_iterations=1, noise_mod='split-biased'):
    kwargs_iter = init_kwargs_iter if isinstance(init_kwargs_iter, list) else []

    iter_collected = {
        'crop': crop_iter,
        'subcrop': subcrop_iter
    }

    KEYS_TO_ITER = ['sample', 'noise_factor', 'edge_prob', 'agglo', 'local_attraction']
    for key in KEYS_TO_ITER:
        if key in fixed_kwargs:
            iter_collected[key] = [fixed_kwargs[key]]
        elif key in kwargs_to_be_iterated:
            iter_collected[key] = kwargs_to_be_iterated[key]
        else:
            raise ValueError("Iter key {} was not passed!".format(key))

    for _ in range(nb_iterations):
        affinities_blocks, masks_used_edges_blocks, GT_blocks = get_block_data_lists()

        for sample in iter_collected['sample']:
            print("Loading...")
            # ----------------------------------------------------------------------
            # Load data (and possibly add noise or select long range edges):
            # ----------------------------------------------------------------------
            for crop in iter_collected['crop']:
                for sub_crop in iter_collected['subcrop']:
                    # FIXME:
                    noise_seed = np.random.randint(-100000, 100000)

                    # FIXME: actually cremi is the only supported dataset at the moment (mainly for the crops...)
                    affinities, GT = get_dataset_data(fixed_kwargs['dataset'], sample, CREMI_crop_slices[sample][crop],
                                                      run_connected_components=False)
                    # temp_file = os.path.join(get_hci_home_path(), 'affs_plus_smart_noise.h5')

                    sub_crop_slc = parse_data_slice(CREMI_sub_crops_slices[sub_crop])
                    affinities = affinities[sub_crop_slc]
                    GT = GT[sub_crop_slc[1:]]
                    GT = vigra.analysis.labelVolumeWithBackground(GT.astype('uint32'))
                    # affinities = add_epsilon(affinities)

                    # from segmfriends import vis as vis
                    # fig, ax = vis.get_figure(1, 1, hide_axes=True)
                    #
                    # vis.plot_segm(ax, GT, z_slice=15, background=None, mask_value=None, highlight_boundaries=True, plot_label_colors=True)
                    #
                    # fig.savefig(os.path.join(get_hci_home_path(), "GT_affs.pdf"))


                    GT_blocks[sample][crop][sub_crop] = GT
                    affinities_blocks[sample][crop][sub_crop] = {}
                    masks_used_edges_blocks[sample][crop][sub_crop] = {}
                    for long_range_prob in iter_collected['edge_prob']:
                        for noise in iter_collected['noise_factor']:
                            # TODO: this needs to be generalized for more noise options
                            # TODO: generate random probs long range edges
                            # all_affinities_blocks[sample][crop][sub_crop][noise] = np.copy(affinities)
                            # all_affinities_blocks[sample][crop][sub_crop][noise] = vigra.readHDF5(temp_file,
                            #                 '{:.4f}'.format(noise))
                            if noise != 0.:
                                affinities_blocks[sample][crop][sub_crop][noise] = add_opensimplex_noise_to_affs(affinities, noise,
                                                              mod=noise_mod,
                                                              target_affs='all',
                                                              seed=noise_seed
                                                              )
                                # affinities_blocks[sample][crop][sub_crop][noise] = add_smart_noise_to_affs(affinities,
                                #                                                                        scale_factor=noise,
                                #                                                                        mod='merge-bias',
                                #                                                                        target_affs='short')
                            else:
                                affinities_blocks[sample][crop][sub_crop][noise] = affinities
                            # vigra.writeHDF5(all_affinities_blocks[sample][crop][sub_crop][noise], temp_file, '{:.4f}'.format(noise))

                            # Fix already long-range edges that will be in the graph:
                            if long_range_prob < 1.0 and long_range_prob > 0.0:
                                masks_used_edges_blocks[sample][crop][sub_crop][long_range_prob] = np.random.random(affinities.shape) >= long_range_prob


            # ----------------------------------------------------------------------
            # Create iterators:
            # ----------------------------------------------------------------------
            print("Creating pool instances...")
            for crop in iter_collected['crop']:
                for sub_crop in iter_collected['subcrop']:
                    assert affinities_blocks[sample][crop][sub_crop] is not None
                    for local_attr in iter_collected['local_attraction']:
                        for agglo in iter_collected['agglo']:
                            if local_attr and agglo in ['greedyFixation', 'GAEC']:
                                continue
                            for edge_prob in iter_collected['edge_prob']:
                                for noise in iter_collected['noise_factor']:
                                    new_kwargs = {}
                                    new_kwargs.update(fixed_kwargs)

                                    # Update the iterated kwargs only if not already present in the fixed args:
                                    iterated_kwargs = {
                                        'sample': sample,
                                        'noise_factor': noise,
                                        'edge_prob': edge_prob,
                                        'agglo': agglo,
                                        'local_attraction': local_attr,
                                        'crop_slice': CREMI_crop_slices[sample][crop],
                                        'sub_crop_slice': CREMI_sub_crops_slices[sub_crop],
                                        'affinities': affinities_blocks[sample][crop][sub_crop][noise],
                                        'GT': GT_blocks[sample][crop][sub_crop],
                                        'mask_used_edges': masks_used_edges_blocks[sample][crop][sub_crop].get(edge_prob, None)
                                    }
                                    new_kwargs.update({k: v for k, v in iterated_kwargs.items() if k not in new_kwargs})

                                    # TODO: add dynamic UCM option...?
                                    # # saveUCM = True if edge_prob > 0.0999 and edge_prob < 0.1001 and agglo != 'MAX' else False
                                    # saveUCM = False
                                    # if saveUCM and not check:
                                    #     print("UCM scheduled!")
                                    #     check = True

                                    kwargs_iter.append(new_kwargs)
    return kwargs_iter