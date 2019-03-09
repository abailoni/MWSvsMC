# Add missing package-paths:
import long_range_compare

# FIXME: outdated spaghetti code, update

import vigra
import os

import time
import json


from segmfriends.utils.config_utils import adapt_configs_to_model, recursive_dict_update
from segmfriends.utils import yaml2dict, parse_data_slice

from long_range_compare.data_paths import get_hci_home_path
from segmfriends.algorithms.WS.WS_growing import SizeThreshAndGrowWithWS

from skunkworks.metrics.cremi_score import cremi_score

from long_range_compare.load_datasets import get_dataset_data, get_dataset_offsets, CREMI_crop_slices, CREMI_sub_crops_slices



def get_segmentation(affinities, GT, json_filename, dataset, run_conn_comp=False):

    offsets = get_dataset_offsets(dataset)

    configs = {'models': yaml2dict('./experiments/models_config.yml'),
               'postproc': yaml2dict('./experiments/post_proc_config.yml')}
    configs = adapt_configs_to_model([], debug=True, **configs)
    post_proc_config = configs['postproc']



    EXPORT_PATH = os.path.join(get_hci_home_path(), 'GEC_comparison_longRangeGraph')

    export_file = os.path.join(EXPORT_PATH, json_filename.replace('.json', '.h5'))
    print(export_file)
    pred_segm = vigra.readHDF5(export_file, 'segm').astype('uint64')

    # Run connected components (for max in affogato):
    if run_conn_comp:
        print("Run connected comp...")
        tick = time.time()
        pred_segm = vigra.analysis.labelVolume(pred_segm.astype('uint32'))
        print("Done in {} s".format(time.time() - tick))
    print("WS growing:")
    tick = time.time()

    grow = SizeThreshAndGrowWithWS(post_proc_config['thresh_segm_size'],
                 offsets,
                 hmap_kwargs=post_proc_config['prob_map_kwargs'],
                 apply_WS_growing=True,
                 size_of_2d_slices=True)
    pred_segm_WS = grow(affinities, pred_segm)
    print("Done in {} s".format(time.time() - tick))

    # SAVING RESULTS:
    # evals = cremi_score(GT, pred_segm, border_threshold=None, return_all_scores=True)

    evals_WS = cremi_score(GT, pred_segm_WS, border_threshold=None, return_all_scores=True)
    print("Scores achieved : ", evals_WS)
    # print("Scores achieved: ", evals)


    result_file = os.path.join(EXPORT_PATH, json_filename)
    with open(result_file, 'r') as f:
        new_results = json.load(f)


    new_results['score_WS'] = evals_WS

    with open(result_file, 'w') as f:
        json.dump(new_results, f, indent=4, sort_keys=True)
        # yaml.dump(result_dict, f)

    vigra.writeHDF5(pred_segm_WS.astype('uint32'), export_file, 'segm_WS')



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


    def full_CREMI():
        global nb_threads_pool
        global from_superpixels
        global use_multicut
        global add_smart_noise
        global save_segm
        global WS_growing
        global additional_model_keys

        nb_threads_pool =  1
        from_superpixels = True
        use_multicut = False
        add_smart_noise = False
        save_segm = True
        WS_growing = False
        additional_model_keys = ['fullCREMIexp']

        check = False
        for _ in range(1):
            for sample in ["C"]:
                # crop_range = {"A": range(3,4),
                #               "B": range(0,1),
                #               "C": range(2,3)}

                # for crop in crop_range[sample]:  # Deep-z: 5       MC: 4   All: 0:4
                for crop in range(4,5):  # Deep-z: 5       MC: 4   All: 0:4
                    for sub_crop in range(6,7): # Deep-z: 5     MC: 6  All: 4 Tiny: 5
                        if all_affinities_blocks[sample][crop][sub_crop] is None:
                            # Load data:
                            affinities, GT = get_dataset_data("CREMI", sample, CREMI_crop_slices[sample][crop],
                                                              run_connected_components=False)
                            sub_crop_slc = parse_data_slice(CREMI_sub_crops_slices[sub_crop])
                            affinities = affinities[sub_crop_slc]
                            GT = GT[sub_crop_slc[1:]]
                            GT = vigra.analysis.labelVolumeWithBackground(GT.astype('uint32'))
                            # affinities = add_epsilon(affinities)
                            all_affinities_blocks[sample][crop][sub_crop] = affinities
                            all_GT_blocks[sample][crop][sub_crop] = GT

                        for local_attr in [False]:
                            agglos = ['MEAN']
                            for agglo in agglos:
                            # for agglo in ['MEAN_constr', 'MAX', 'MEAN', 'greedyFixation', 'GAEC']:
                            # for agglo in ['MEAN', 'MAX', 'greedyFixation', 'GAEC', 'MEAN_constr']:
                            # for agglo in ['MEAN', 'MAX', 'MEAN_constr']:
                                if local_attr and agglo in ['greedyFixation', 'GAEC']:
                                    continue
                                # for edge_prob in np.concatenate((np.linspace(0.0, 0.1, 17), np.linspace(0.11, 0.8, 18))):
                                for edge_prob in [0.07]:
                                    all_datasets.append('CREMI')
                                    all_samples.append(sample)
                                    all_crops.append(CREMI_crop_slices[sample][crop])
                                    all_sub_crops.append(CREMI_sub_crops_slices[sub_crop])
                                    all_local_attr.append(local_attr)
                                    all_agglo_type.append(agglo)
                                    all_edge_prob.append(edge_prob)
                                    # saveUCM = True if edge_prob > 0.0999 and edge_prob < 0.1001 and agglo != 'MAX' else False
                                    saveUCM = False
                                    if saveUCM and not check:
                                        print("UCM scheduled!")
                                        check = True
                                    all_UCM.append(saveUCM)
                                    assert all_affinities_blocks[sample][crop][sub_crop] is not None
                                    all_affinites.append(all_affinities_blocks[sample][crop][sub_crop])
                                    all_GTs.append(all_GT_blocks[sample][crop][sub_crop])
                                    all_noise_factors.append(0.)



    print("Loading...")
    tick = time.time()
    full_CREMI()
    # smart_noise_merge()
    # smart_noise_merge_only_local()
    # smart_noise_split()

    print("Loaded dataset in {}s".format(time.time() - tick))

    print("Agglomarations to run: ", len(all_datasets))


    assert len(all_affinites) == 1
    # get_segmentation(all_affinites[0], all_GTs[0], "942327871_B_sum_True.json", "CREMI", run_conn_comp=True)
    get_segmentation(all_affinites[0], all_GTs[0], "78023428_C_sum_True.json", "CREMI", run_conn_comp=True)
    # get_segmentation(all_affinites[0], all_GTs[0], "696838786_A_sum_False.json", "CREMI", run_conn_comp=True)
    # get_segmentation(all_affinites[0], all_GTs[0], "930025566_A_max_False.json", "CREMI", run_conn_comp=True)
    # get_segmentation(all_affinites[0], all_GTs[0], "921046704_C_max_False.json", "CREMI", run_conn_comp=True)

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


