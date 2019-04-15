import long_range_compare # Add missing package-paths

import vigra
import numpy as np
import os
import argparse
from multiprocessing.pool import ThreadPool
from itertools import repeat

from long_range_compare.data_paths import get_hci_home_path, get_trendytukan_drive_path
from long_range_compare import cremi_utils as cremi_utils
from long_range_compare import cremi_experiments as cremi_experiments

from segmfriends.utils.various import starmap_with_kwargs



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="DebugExp")  #DebugExp
    parser.add_argument('--project_directory', default="projects/agglo_cluster_compare",  type=str)
    # TODO: option to pass some other fixed kwargs and overwrite it...?

    args = parser.parse_args()

    exp_name = args.exp_name

    fixed_kargs = {
        "experiment_name": exp_name,
        "project_directory": os.path.join(get_trendytukan_drive_path(), args.project_directory),
        "configs_dir_path": os.path.join(get_hci_home_path(), "pyCharm_projects/longRangeAgglo/experiments/cremi/configs")
    }

    # Select experiment and load data:
    experiment = cremi_experiments.get_experiment_by_name(exp_name)(fixed_kwargs=fixed_kargs)
    kwargs_iter, nb_threads_pool = experiment.get_data()
    print("Agglomarations to run: ", len(kwargs_iter))

    # Start pool:
    pool = ThreadPool(processes=nb_threads_pool)
    starmap_with_kwargs(pool, cremi_utils.run_clustering, args_iter=repeat([]),
                        kwargs_iter=kwargs_iter)
    pool.close()
    pool.join()
    np.random.uniform()

    #
    # def full_CREMI():
    #     global nb_threads_pool
    #     global from_superpixels
    #     global use_multicut
    #     global add_smart_noise
    #     global save_segm
    #     global WS_growing
    #     global additional_model_keys
    #
    #     nb_threads_pool =  1
    #     from_superpixels = True
    #     use_multicut = False
    #     add_smart_noise = False
    #     save_segm = True
    #     WS_growing = False
    #     additional_model_keys = ['fullCREMIexp']
    #
    #
    #
    #     check = False
    #     for _ in range(1):
    #         for sample in ["C"]:
    #             # crop_range = {"A": range(3,4),
    #             #               "B": range(0,1),
    #             #               "C": range(2,3)}
    #
    #             # for crop in crop_range[sample]:  # Deep-z: 5       MC: 4   All: 0:4
    #             for crop in range(4,5):  # Deep-z: 5       MC: 4   All: 0:4
    #                 for sub_crop in range(6,7): # Deep-z: 5     MC: 6  All: 4 Tiny: 5
    #                     if all_affinities_blocks[sample][crop][sub_crop] is None:
    #                         # Load data:
    #                         affinities, GT = get_dataset_data("CREMI", sample, CREMI_crop_slices[sample][crop],
    #                                                           run_connected_components=False)
    #                         sub_crop_slc = parse_data_slice(CREMI_sub_crops_slices[sub_crop])
    #                         affinities = affinities[sub_crop_slc]
    #                         GT = GT[sub_crop_slc[1:]]
    #                         GT = vigra.analysis.labelVolumeWithBackground(GT.astype('uint32'))
    #                         affinities = add_epsilon(affinities)
    #                         all_affinities_blocks[sample][crop][sub_crop] = affinities
    #                         all_GT_blocks[sample][crop][sub_crop] = GT
    #
    #                     for local_attr in [False]:
    #                         # agglos = ['MEAN'] if sample == "B" else ['MEAN_constr', 'MAX', 'MEAN']
    #                         agglos = ['greedyFixation']
    #                         for agglo in agglos:
    #                         # for agglo in ['MEAN_constr', 'MAX', 'MEAN', 'greedyFixation', 'GAEC']:
    #                         # for agglo in ['MEAN', 'MAX', 'greedyFixation', 'GAEC', 'MEAN_constr']:
    #                         # for agglo in ['MEAN', 'MAX', 'MEAN_constr']:
    #                             if local_attr and agglo in ['greedyFixation', 'GAEC']:
    #                                 continue
    #                             # for edge_prob in np.concatenate((np.linspace(0.0, 0.1, 17), np.linspace(0.11, 0.8, 18))):
    #                             for edge_prob in [0.07]:
    #                                 all_datasets.append('CREMI')
    #                                 all_samples.append(sample)
    #                                 all_crops.append(CREMI_crop_slices[sample][crop])
    #                                 all_sub_crops.append(CREMI_sub_crops_slices[sub_crop])
    #                                 all_local_attr.append(local_attr)
    #                                 all_agglo_type.append(agglo)
    #                                 all_edge_prob.append(edge_prob)
    #                                 # saveUCM = True if edge_prob > 0.0999 and edge_prob < 0.1001 and agglo != 'MAX' else False
    #                                 saveUCM = False
    #                                 if saveUCM and not check:
    #                                     print("UCM scheduled!")
    #                                     check = True
    #                                 all_UCM.append(saveUCM)
    #                                 assert all_affinities_blocks[sample][crop][sub_crop] is not None
    #                                 all_affinites.append(all_affinities_blocks[sample][crop][sub_crop])
    #                                 all_GTs.append(all_GT_blocks[sample][crop][sub_crop])
    #                                 all_noise_factors.append(0.)
    #
    # def full_CREMI_MC():
    #     global nb_threads_pool
    #     global from_superpixels
    #     global use_multicut
    #     global add_smart_noise
    #     global save_segm
    #     global WS_growing
    #     global additional_model_keys
    #
    #     nb_threads_pool =  1
    #     from_superpixels = True
    #     use_multicut = True
    #     add_smart_noise = False
    #     save_segm = True
    #     WS_growing = False
    #     additional_model_keys = ['fullCREMIexp_MC']
    #
    #
    #
    #     check = False
    #     for _ in range(1):
    #         for sample in ["C", "A"]:
    #             # crop_range = {"A": range(3,4),
    #             #               "B": range(0,1),
    #             #               "C": range(2,3)}
    #
    #             # for crop in crop_range[sample]:  # Deep-z: 5       MC: 4   All: 0:4
    #             for crop in range(4,5):  # Deep-z: 5       MC: 4   All: 0:4
    #                 for sub_crop in range(6,7): # Deep-z: 5     MC: 6  All: 4 Tiny: 5
    #                     if all_affinities_blocks[sample][crop][sub_crop] is None:
    #                         # Load data:
    #                         affinities, GT = get_dataset_data("CREMI", sample, CREMI_crop_slices[sample][crop],
    #                                                           run_connected_components=False)
    #                         sub_crop_slc = parse_data_slice(CREMI_sub_crops_slices[sub_crop])
    #                         affinities = affinities[sub_crop_slc]
    #                         GT = GT[sub_crop_slc[1:]]
    #                         GT = vigra.analysis.labelVolumeWithBackground(GT.astype('uint32'))
    #                         affinities = add_epsilon(affinities)
    #                         all_affinities_blocks[sample][crop][sub_crop] = affinities
    #                         all_GT_blocks[sample][crop][sub_crop] = GT
    #
    #                     for local_attr in [False]:
    #                         agglos = ['MEAN']
    #                         for agglo in agglos:
    #                         # for agglo in ['MEAN_constr', 'MAX', 'MEAN', 'greedyFixation', 'GAEC']:
    #                         # for agglo in ['MEAN', 'MAX', 'greedyFixation', 'GAEC', 'MEAN_constr']:
    #                         # for agglo in ['MEAN', 'MAX', 'MEAN_constr']:
    #                             if local_attr and agglo in ['greedyFixation', 'GAEC']:
    #                                 continue
    #                             # for edge_prob in np.concatenate((np.linspace(0.0, 0.1, 17), np.linspace(0.11, 0.8, 18))):
    #                             for edge_prob in [0.0]:
    #                                 all_datasets.append('CREMI')
    #                                 all_samples.append(sample)
    #                                 all_crops.append(CREMI_crop_slices[sample][crop])
    #                                 all_sub_crops.append(CREMI_sub_crops_slices[sub_crop])
    #                                 all_local_attr.append(local_attr)
    #                                 all_agglo_type.append(agglo)
    #                                 all_edge_prob.append(edge_prob)
    #                                 # saveUCM = True if edge_prob > 0.0999 and edge_prob < 0.1001 and agglo != 'MAX' else False
    #                                 saveUCM = False
    #                                 if saveUCM and not check:
    #                                     print("UCM scheduled!")
    #                                     check = True
    #                                 all_UCM.append(saveUCM)
    #                                 assert all_affinities_blocks[sample][crop][sub_crop] is not None
    #                                 all_affinites.append(all_affinities_blocks[sample][crop][sub_crop])
    #                                 all_GTs.append(all_GT_blocks[sample][crop][sub_crop])
    #                                 all_noise_factors.append(0.)
    #
    # def smart_noise_split():
    #     global nb_threads_pool
    #     global from_superpixels
    #     global use_multicut
    #     global add_smart_noise
    #     global save_segm
    #     global WS_growing
    #     global additional_model_keys
    #
    #     nb_threads_pool = 6
    #     from_superpixels = False
    #     use_multicut = False
    #     add_smart_noise = True
    #     save_segm = False
    #     WS_growing = True
    #     LONG_RANGE_EDGE_PROBABILITY = 0.0
    #     # NOISE_FACTORS = np.linspace(0.0, 0.9, 15)
    #     NOISE_FACTORS = [0.0]
    #     additional_model_keys = ['smart_noise_split_onlyShort_fromPix']
    #
    #     for _ in range(1):
    #         # Generate noisy affinities:
    #         print("Generating noisy affs...")
    #         for crop in range(5, 6):  # Deep-z: 5       MC: 4   All: 0:4
    #             for sub_crop in range(5, 6):  # Deep-z: 5     MC: 6  All: 4 Tiny: 5
    #                 # Load data:
    #                 affinities, GT = get_dataset_data("CREMI", "B", CREMI_crop_slices["B"][crop],
    #                                                   run_connected_components=False)
    #                 temp_file = os.path.join(get_hci_home_path(), 'affs_plus_smart_noise.h5')
    #
    #                 sub_crop_slc = parse_data_slice(CREMI_sub_crops_slices[sub_crop])
    #                 affinities = affinities[sub_crop_slc]
    #                 GT = GT[sub_crop_slc[1:]]
    #                 GT = vigra.analysis.labelVolumeWithBackground(GT.astype('uint32'))
    #                 affinities = add_epsilon(affinities)
    #
    #                 all_GT_blocks["B"][crop][sub_crop] = GT
    #                 all_affinities_blocks["B"][crop][sub_crop] = {}
    #                 for noise in NOISE_FACTORS:
    #                     # all_affinities_blocks["B"][crop][sub_crop][noise] = np.copy(affinities)
    #                     # all_affinities_blocks["B"][crop][sub_crop][noise] = vigra.readHDF5(temp_file,
    #                     #                 '{:.4f}'.format(noise))
    #                     all_affinities_blocks["B"][crop][sub_crop][noise] = add_smart_noise_to_affs(affinities,
    #                                                                                                 scale_factor=noise,
    #                                                                                                 mod='split-bias',
    #                                                                                                 target_affs='long')
    #                     # vigra.writeHDF5(all_affinities_blocks["B"][crop][sub_crop][noise], temp_file, '{:.4f}'.format(noise))
    #
    #         print("Done!")
    #         check = False
    #         for sample in ["B"]:
    #             # crop_range = {"A": range(3,4),
    #             #               "B": range(0,1),
    #             #               "C": range(2,3)}
    #
    #             # for crop in crop_range[sample]:  # Deep-z: 5       MC: 4   All: 0:4
    #             for crop in range(5, 6):  # Deep-z: 5       MC: 4   All: 0:4
    #                 for sub_crop in range(5, 6):  # Deep-z: 5     MC: 6  All: 4 Tiny: 5
    #                     if all_affinities_blocks[sample][crop][sub_crop] is None:
    #                         # Load data:
    #                         affinities, GT = get_dataset_data("CREMI", sample, CREMI_crop_slices[sample][crop],
    #                                                           run_connected_components=False)
    #                         sub_crop_slc = parse_data_slice(CREMI_sub_crops_slices[sub_crop])
    #                         affinities = affinities[sub_crop_slc]
    #                         GT = GT[sub_crop_slc[1:]]
    #                         GT = vigra.analysis.labelVolumeWithBackground(GT.astype('uint32'))
    #                         affinities = add_epsilon(affinities)
    #                         all_affinities_blocks[sample][crop][sub_crop] = affinities
    #                         all_GT_blocks[sample][crop][sub_crop] = GT
    #
    #                     for local_attr in [False]:
    #                         # for agglo in ['MAX']:
    #                         for agglo in ['greedyFixation', 'MEAN_constr', 'GAEC', 'MAX', 'MEAN']:
    #                         # for agglo in ['MEAN_constr', 'MAX', 'MEAN']:
    #                             # for agglo in ['MEAN', 'MAX', 'greedyFixation', 'GAEC', 'MEAN_constr']:
    #                             # for agglo in ['MEAN', 'MAX', 'MEAN_constr']:
    #                             if local_attr and agglo in ['greedyFixation', 'GAEC']:
    #                                 continue
    #                             for noise in NOISE_FACTORS:
    #                             # for noise_factor in [0.9]:
    #                                 all_datasets.append('CREMI')
    #                                 all_samples.append(sample)
    #                                 all_crops.append(CREMI_crop_slices[sample][crop])
    #                                 all_sub_crops.append(CREMI_sub_crops_slices[sub_crop])
    #                                 all_local_attr.append(local_attr)
    #                                 all_agglo_type.append(agglo)
    #                                 all_edge_prob.append(LONG_RANGE_EDGE_PROBABILITY)
    #                                 # saveUCM = True if edge_prob > 0.0999 and edge_prob < 0.1001 and agglo != 'MAX' else False
    #                                 saveUCM = False
    #                                 if saveUCM and not check:
    #                                     print("UCM scheduled!")
    #                                     check = True
    #                                 all_UCM.append(saveUCM)
    #                                 assert all_affinities_blocks[sample][crop][sub_crop] is not None
    #                                 all_affinites.append(all_affinities_blocks[sample][crop][sub_crop][noise])
    #                                 all_GTs.append(all_GT_blocks[sample][crop][sub_crop])
    #                                 all_noise_factors.append(noise)
    #
    # def smart_noise_merge():
    #     global nb_threads_pool
    #     global from_superpixels
    #     global use_multicut
    #     global add_smart_noise
    #     global save_segm
    #     global WS_growing
    #     global additional_model_keys
    #
    #     nb_threads_pool = 1
    #     from_superpixels = False
    #     use_multicut = False
    #     add_smart_noise = True
    #     save_segm = False
    #     WS_growing = True
    #     LONG_RANGE_EDGE_PROBABILITY = 1.0
    #     NOISE_FACTORS = np.linspace(0.0, 0.9, 15)
    #     additional_model_keys = ['smart_noise_exp_merge_allLong_fromPx']
    #
    #
    #
    #     for _ in range(1):
    #         # Generate noisy affinities:
    #         print("Generating noisy affs...")
    #         for crop in range(5, 6):  # Deep-z: 5       MC: 4   All: 0:4
    #             for sub_crop in range(5, 6):  # Deep-z: 5     MC: 6  All: 4 Tiny: 5
    #                 affinities, GT = get_dataset_data("CREMI", "B", CREMI_crop_slices["B"][crop],
    #                                                   run_connected_components=False)
    #                 temp_file = os.path.join(get_hci_home_path(), 'affs_plus_smart_noise.h5')
    #
    #                 sub_crop_slc = parse_data_slice(CREMI_sub_crops_slices[sub_crop])
    #                 affinities = affinities[sub_crop_slc]
    #                 GT = GT[sub_crop_slc[1:]]
    #                 GT = vigra.analysis.labelVolumeWithBackground(GT.astype('uint32'))
    #                 affinities = add_epsilon(affinities)
    #
    #                 all_GT_blocks["B"][crop][sub_crop] = GT
    #                 all_affinities_blocks["B"][crop][sub_crop] = {}
    #                 for noise in NOISE_FACTORS:
    #                     # all_affinities_blocks["B"][crop][sub_crop][noise] = np.copy(affinities)
    #                     # all_affinities_blocks["B"][crop][sub_crop][noise] = vigra.readHDF5(temp_file,
    #                     #                 '{:.4f}'.format(noise))
    #                     all_affinities_blocks["B"][crop][sub_crop][noise] = add_smart_noise_to_affs(affinities,
    #                                                                                                 scale_factor=noise,
    #                                                                                                 mod='merge-bias',
    #                                                                                                 target_affs='short')
    #                     # vigra.writeHDF5(all_affinities_blocks["B"][crop][sub_crop][noise], temp_file, '{:.4f}'.format(noise))
    #
    #         print("Done!")
    #         check = False
    #         for sample in ["B"]:
    #             # crop_range = {"A": range(3,4),
    #             #               "B": range(0,1),
    #             #               "C": range(2,3)}
    #
    #             # for crop in crop_range[sample]:  # Deep-z: 5       MC: 4   All: 0:4
    #             for crop in range(5, 6):  # Deep-z: 5       MC: 4   All: 0:4
    #                 for sub_crop in range(5, 6):  # Deep-z: 5     MC: 6  All: 4 Tiny: 5
    #                     if all_affinities_blocks[sample][crop][sub_crop] is None:
    #                         # Load data:
    #                         affinities, GT = get_dataset_data("CREMI", sample, CREMI_crop_slices[sample][crop],
    #                                                           run_connected_components=False)
    #                         sub_crop_slc = parse_data_slice(CREMI_sub_crops_slices[sub_crop])
    #                         affinities = affinities[sub_crop_slc]
    #                         GT = GT[sub_crop_slc[1:]]
    #                         GT = vigra.analysis.labelVolumeWithBackground(GT.astype('uint32'))
    #                         affinities = add_epsilon(affinities)
    #                         all_affinities_blocks[sample][crop][sub_crop] = affinities
    #                         all_GT_blocks[sample][crop][sub_crop] = GT
    #
    #                     for local_attr in [False]:
    #                         for agglo in ['MAX']:
    #                         # for agglo in [ 'GAEC', 'greedyFixation']:
    #                         # for agglo in ['MEAN_constr', 'MAX', 'MEAN']:
    #                             # for agglo in ['MEAN', 'MAX', 'greedyFixation', 'GAEC', 'MEAN_constr']:
    #                             # for agglo in ['MEAN', 'MAX', 'MEAN_constr']:
    #                             if local_attr and agglo in ['greedyFixation', 'GAEC']:
    #                                 continue
    #                             for noise in NOISE_FACTORS:
    #                             # for noise_factor in [0.9]:
    #                                 all_datasets.append('CREMI')
    #                                 all_samples.append(sample)
    #                                 all_crops.append(CREMI_crop_slices[sample][crop])
    #                                 all_sub_crops.append(CREMI_sub_crops_slices[sub_crop])
    #                                 all_local_attr.append(local_attr)
    #                                 all_agglo_type.append(agglo)
    #                                 all_edge_prob.append(LONG_RANGE_EDGE_PROBABILITY)
    #                                 # saveUCM = True if edge_prob > 0.0999 and edge_prob < 0.1001 and agglo != 'MAX' else False
    #                                 saveUCM = False
    #                                 if saveUCM and not check:
    #                                     print("UCM scheduled!")
    #                                     check = True
    #                                 all_UCM.append(saveUCM)
    #                                 assert all_affinities_blocks[sample][crop][sub_crop] is not None
    #                                 all_affinites.append(all_affinities_blocks[sample][crop][sub_crop][noise])
    #                                 all_GTs.append(all_GT_blocks[sample][crop][sub_crop])
    #                                 all_noise_factors.append(noise)
    #
    # def smart_noise_merge_only_local():
    #     global nb_threads_pool
    #     global from_superpixels
    #     global use_multicut
    #     global add_smart_noise
    #     global save_segm
    #     global WS_growing
    #     global additional_model_keys
    #
    #     nb_threads_pool = 12
    #     from_superpixels = False
    #     use_multicut = False
    #     add_smart_noise = True
    #     save_segm = False
    #     WS_growing = True
    #     LONG_RANGE_EDGE_PROBABILITY = 0.0
    #     NOISE_FACTORS = np.linspace(0.0, 0.9, 15)
    #     # additional_model_keys = ['smart_noise_exp_merge_only_local']
    #     additional_model_keys = ['different_noise_shortEdges']
    #     # additional_model_keys = ['smart_noise_exp_merge_allLong_fromSP']
    #
    #     # TODO: noise, save, crop
    #     for _ in range(3):
    #         # Generate noisy affinities:
    #         print("Generating noisy affs...")
    #         for crop in range(5, 6):  # Deep-z: 5       MC: 4   All: 0:4
    #             for sub_crop in range(5, 6):  # Deep-z: 5     MC: 6  All: 4 Tiny: 5
    #                 affinities, GT = get_dataset_data("CREMI", "B", CREMI_crop_slices["B"][crop],
    #                                                   run_connected_components=False)
    #                 temp_file = os.path.join(get_hci_home_path(), 'affs_plus_smart_noise.h5')
    #
    #                 sub_crop_slc = parse_data_slice(CREMI_sub_crops_slices[sub_crop])
    #                 affinities = affinities[sub_crop_slc]
    #                 GT = GT[sub_crop_slc[1:]]
    #                 GT = vigra.analysis.labelVolumeWithBackground(GT.astype('uint32'))
    #                 # affinities = add_epsilon(affinities)
    #
    #                 # from segmfriends import vis as vis
    #                 # fig, ax = vis.get_figure(1, 1, hide_axes=True)
    #                 #
    #                 # vis.plot_segm(ax, GT, z_slice=15, background=None, mask_value=None, highlight_boundaries=True, plot_label_colors=True)
    #                 #
    #                 # fig.savefig(os.path.join(get_hci_home_path(), "GT_affs.pdf"))
    #
    #
    #
    #                 all_GT_blocks["B"][crop][sub_crop] = GT
    #                 all_affinities_blocks["B"][crop][sub_crop] = {}
    #                 for noise in NOISE_FACTORS:
    #                     # all_affinities_blocks["B"][crop][sub_crop][noise] = np.copy(affinities)
    #                     # all_affinities_blocks["B"][crop][sub_crop][noise] = vigra.readHDF5(temp_file,
    #                     #                 '{:.4f}'.format(noise))
    #                     all_affinities_blocks["B"][crop][sub_crop][noise] = add_smart_noise_to_affs(affinities,
    #                                                                                                 scale_factor=noise,
    #                                                                                                 mod='merge-bias',
    #                                                                                                 target_affs='short')
    #                     # vigra.writeHDF5(all_affinities_blocks["B"][crop][sub_crop][noise], temp_file, '{:.4f}'.format(noise))
    #
    #         print("Done!")
    #         check = False
    #         for sample in ["B"]:
    #             # crop_range = {"A": range(3,4),
    #             #               "B": range(0,1),
    #             #               "C": range(2,3)}
    #
    #             # for crop in crop_range[sample]:  # Deep-z: 5       MC: 4   All: 0:4
    #             for crop in range(5, 6):  # Deep-z: 5       MC: 4   All: 0:4
    #                 for sub_crop in range(5, 6):  # Deep-z: 5     MC: 6  All: 4 Tiny: 5
    #                     if all_affinities_blocks[sample][crop][sub_crop] is None:
    #                         # Load data:
    #                         affinities, GT = get_dataset_data("CREMI", sample, CREMI_crop_slices[sample][crop],
    #                                                           run_connected_components=False)
    #                         sub_crop_slc = parse_data_slice(CREMI_sub_crops_slices[sub_crop])
    #                         affinities = affinities[sub_crop_slc]
    #                         GT = GT[sub_crop_slc[1:]]
    #                         GT = vigra.analysis.labelVolumeWithBackground(GT.astype('uint32'))
    #                         affinities = add_epsilon(affinities)
    #                         all_affinities_blocks[sample][crop][sub_crop] = affinities
    #                         all_GT_blocks[sample][crop][sub_crop] = GT
    #
    #                     for local_attr in [False]:
    #                         # for agglo in ['MAX']:
    #                         for agglo in ['MAX', 'greedyFixation', 'MEAN_constr', 'GAEC',  'MEAN']:
    #                         # for agglo in ['MEAN_constr', 'MAX', 'MEAN']:
    #                             # for agglo in ['MEAN', 'MAX', 'greedyFixation', 'GAEC', 'MEAN_constr']:
    #                             # for agglo in ['MEAN', 'MAX', 'MEAN_constr']:
    #                             if local_attr and agglo in ['greedyFixation', 'GAEC']:
    #                                 continue
    #                             for noise in NOISE_FACTORS:
    #                             # for noise_factor in [0.9]:
    #                                 all_datasets.append('CREMI')
    #                                 all_samples.append(sample)
    #                                 all_crops.append(CREMI_crop_slices[sample][crop])
    #                                 all_sub_crops.append(CREMI_sub_crops_slices[sub_crop])
    #                                 all_local_attr.append(local_attr)
    #                                 all_agglo_type.append(agglo)
    #                                 all_edge_prob.append(LONG_RANGE_EDGE_PROBABILITY)
    #                                 # saveUCM = True if edge_prob > 0.0999 and edge_prob < 0.1001 and agglo != 'MAX' else False
    #                                 saveUCM = False
    #                                 if saveUCM and not check:
    #                                     print("UCM scheduled!")
    #                                     check = True
    #                                 all_UCM.append(saveUCM)
    #                                 assert all_affinities_blocks[sample][crop][sub_crop] is not None
    #                                 all_affinites.append(all_affinities_blocks[sample][crop][sub_crop][noise])
    #                                 all_GTs.append(all_GT_blocks[sample][crop][sub_crop])
    #                                 all_noise_factors.append(noise)
    #
