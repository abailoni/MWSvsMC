import sys

sys.path += ["/home/abailoni_local/hci_home/python_libraries/nifty/python",
"/home/abailoni_local/hci_home/python_libraries/affogato/python",
"/home/abailoni_local/hci_home/python_libraries/cremi_python",
"/home/abailoni_local/hci_home/pyCharm_projects/inferno",
"/home/abailoni_local/hci_home/pyCharm_projects/constrained_mst",
"/home/abailoni_local/hci_home/pyCharm_projects/neuro-skunkworks",
"/home/abailoni_local/hci_home/pyCharm_projects/segmfriends",
"/home/abailoni_local/hci_home/pyCharm_projects/hc_segmentation",
"/home/abailoni_local/hci_home/pyCharm_projects/neurofire",]

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import vigra
import nifty as nf
import nifty.graph.agglo as nagglo
import numpy as np
import os
from affogato.segmentation import compute_mws_segmentation
from affogato.segmentation import compute_mws_segmentation

from skimage import io
import argparse
import time
import yaml
import json



from long_range_compare import utils as utils

from segmfriends.utils.config_utils import adapt_configs_to_model, recursive_dict_update
from segmfriends.utils import yaml2dict
from segmfriends.io.load import parse_offsets
from segmfriends.algorithms.agglo import GreedyEdgeContractionAgglomeraterFromSuperpixels
from segmfriends.algorithms.WS.WS_growing import SizeThreshAndGrowWithWS
from segmfriends.algorithms.blockwise import BlockWise

from skunkworks.metrics.cremi_score import cremi_score
from long_range_hc.postprocessing.pipelines import get_segmentation_pipeline




def get_segmentation(inverted_affinities, offsets, post_proc_config):
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
    outputs = post_proc_solver(inverted_affinities)
    comp_time = time.time() - tick
    print("Post-processing took {} s".format(comp_time))

    return outputs

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
    # if post_proc_config.get('thresh_segm_size', 0) != 0:
    grow = SizeThreshAndGrowWithWS(post_proc_config['thresh_segm_size'],
             offsets,
             hmap_kwargs=post_proc_config['prob_map_kwargs'],
             apply_WS_growing=True,)
    pred_segm_WS = grow(1 - inverted_affinities, pred_segm)


    # SAVING RESULTS:
    evals = cremi_score(gt, pred_segm, border_threshold=None, return_all_scores=True)
    evals_WS = cremi_score(gt, pred_segm_WS, border_threshold=None, return_all_scores=True)
    print("Scores achieved WS: ", evals_WS)
    print("Scores achieved: ", evals)

    ID = str(np.random.randint(10000000))

    extra_agglo = post_proc_config['generalized_HC_kwargs']['agglomeration_kwargs']['extra_aggl_kwargs']
    agglo_type = extra_agglo['update_rule']
    non_link = extra_agglo['add_cannot_link_constraints']
    edge_prob = str(np.asscalar(post_proc_config['generalized_HC_kwargs']['probability_long_range_edges']))


    result_file = os.path.join('/home/abailoni_local/', 'generalized_GED_comparison_local_attraction.json')
    # result_dict = yaml2dict(result_file)
    # result_dict = {} if result_dict is None else result_dict
    if os.path.exists(result_file):
        with open(result_file, 'rb') as f:
            result_dict = json.load(f)
        os.remove(result_file)
    else:
        result_dict = {}



    new_results = {}
    new_results[agglo_type] = {}
    new_results[agglo_type][str(non_link)] = {}
    new_results[agglo_type][str(non_link)][edge_prob] = {}
    new_results[agglo_type][str(non_link)][edge_prob][ID] = {'energy': np.asscalar(MC_energy), 'score': evals, 'score_WS': evals_WS, 'runtime': comp_time}

    file_name = "{}_{}_{}".format(ID, agglo_type, edge_prob)
    # file_path = os.path.join('/net/hciserver03/storage/abailoni/GEC_comparison', file_name)
    file_path = os.path.join('/home/abailoni_local/GEC_comparison_local_attraction', file_name)

    result_dict = recursive_dict_update(new_results, result_dict)

    with open(result_file, 'w') as f:
        json.dump(result_dict, f, indent=4, sort_keys=True)
        # yaml.dump(result_dict, f)

    # Save some data:
    vigra.writeHDF5(pred_segm.astype('uint32'), file_path, 'segm')
    vigra.writeHDF5(pred_segm_WS.astype('uint32'), file_path, 'segm_WS')







if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--crop_slice', default=None)
    # parser.add_argument('--n_threads', default=1, type=int)
    # parser.add_argument('--name_aggl', default=None)
    # parser.add_argument('--model_IDs', nargs='+', default=None, type=str)
    # args = parser.parse_args()

    # models_IDs = args.model_IDs
    #
    # configs = {'models': yaml2dict('./experiments/models_config.yml'),
    #     'postproc': yaml2dict('./experiments/post_proc_config.yml')}
    # if models_IDs is not None:
    #     configs = adapt_configs_to_model(models_IDs, debug=True, **configs)
    #
    # postproc_config = configs['postproc']



    # -----------------
    # Load data:
    # -----------------
    root_path = "/export/home/abailoni/supervised_projs/divisiveMWS"
    # dataset_path = os.path.join(root_path, "cremi-dataset-crop")
    dataset_path = "/net/hciserver03/storage/abailoni/datasets/ISBI/"
    plots_path = os.path.join("/net/hciserver03/storage/abailoni/greedy_edge_contr/plots")
    save_path = os.path.join(root_path, "outputs")

    # Import data:
    cremi_path = "/net/hciserver03/storage/abailoni/datasets/cremi/SOA_predictions_MWS/sampleA_small.h5"
    affinities  = vigra.readHDF5(cremi_path, 'predictions/full_affs').astype('float32') / 255.
    # affinities += np.random.normal(scale=0.0001, size=affinities.shape)

    raw  = vigra.readHDF5(cremi_path, 'raw')
    gt = vigra.readHDF5(cremi_path, 'segmentation/groundtruth')

    # # If this volume is too big, take a crop of it:
    # crop_slice = (slice(None), slice(None, 1), slice(None, 200), slice(None, 200))
    # affinities = affinities[crop_slice]
    # raw = raw[crop_slice[1:]]
    # gt = gt[crop_slice[1:]]




    # Build the graph:
    volume_shape = raw.shape
    print(volume_shape)

    offsets_MWS = 'offsets_MWS.json'
    offsets_SOA = 'SOA_offsets.json'
    offset_file = os.path.join('/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/experiments/postprocessing/cremi/offsets/', offsets_SOA)
    offsets = parse_offsets(offset_file)

    # offsets = np.array([[-1, 0, 0],
    #                     [0, -1, 0],
    #                     [0, 0, -1],
    #                     [-2, 0, 0],
    #                     [-2, 1, 1],
    #                     [-2, -1, 1],
    #                     [-2, 1, -1],
    #                     [0, -2, 0],
    #                     [0, 0, -2],
    #                     [0, -2, -2],
    #                     [0, 2, -2],
    #                     [0, -9, -4],
    #                     [0, -4, -9],
    #                     [0, 4, -9],
    #                     [0, 9, -4],
    #                     [0, -27, 0],
    #                     [0, 0, -27]])
    #
    # # Inverted affinities: 0 means merge, 1 means split
    # affinities = np.random.uniform(0., 1., (17, 1, 2, 3))
    # affinities[:3] = 0.99  # local
    # affinities[3:] = 0.9  # lifted
    #
    # affinities[:3] -= np.abs(np.random.normal(scale=0.001, size=(3, 1, 2, 3)))
    # affinities[3:] += np.abs(np.random.normal(scale=0.001, size=(14, 1, 2, 3)))

    # # Constantin implementation:
    # tick = time.time()
    # divMWS = compute_mws_segmentation(affinities, offsets, 3,
    #                                   randomize_strides=False,
    #                                   strides = [1,1,1],
    #                                   algorithm='kruskal')
    # print("Took ", time.time() - tick)
    # # print(divMWS)
    # #
    #
    # file_path_newMWS = os.path.join('/home/abailoni_local/', "ISBI_results_new_MWS.h5")
    # MWS = vigra.readHDF5(file_path_newMWS, 'segm_newMWS')

    # evals = cremi_score(MWS+1, divMWS+1, border_threshold=None, return_all_scores=True)
    # print(evals)

    # vigra.writeHDF5(divMWS.astype('uint32'), file_path_newMWS, 'segm_newMWS')

    #
    # # # steffen_MWS = vigra.readHDF5(file_path_newMWS, 'segm')
    # # # constatine_MWS = vigra.readHDF5(file_path_newMWS, 'segm_2')
    #
    # # affinities = affinities[:,:10]
    #
    # GEC:
    configs = {'models': yaml2dict('./experiments/models_config.yml'),
               'postproc': yaml2dict('./experiments/post_proc_config.yml')}
    configs = adapt_configs_to_model(['MAX', 'impose_local_attraction'], debug=True, **configs)
    # configs = adapt_configs_to_model(['MEAN'], debug=True, **configs)
    postproc_config = configs['postproc']
    postproc_config['generalized_HC_kwargs']['probability_long_range_edges'] = 1.0
    # postproc_config['generalized_HC_kwargs']['strides'] = np.array([1,12,12])
    print(affinities[:,:31,:650,:650].shape)
    segm_GEC = get_segmentation(1 - affinities[:,:31,:650,:650], offsets, postproc_config)[0]


    file_path_segm = os.path.join('/home/abailoni_local/hci_home', "cremi_A.h5")
    vigra.writeHDF5(segm_GEC.astype('uint32'), file_path_segm, 'GEC_mean')
    #
    # evals = cremi_score(segm_GEC+1, divMWS+1, border_threshold=None, return_all_scores=True)
    # print(evals)

