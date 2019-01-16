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



from compareMCandMWS import utils as utils

from segmfriends.utils.config_utils import adapt_configs_to_model, recursive_dict_update
from segmfriends.utils import yaml2dict
from segmfriends.io.load import parse_offsets
from segmfriends.algorithms.agglo import GreedyEdgeContractionAgglomeraterFromSuperpixels
from segmfriends.algorithms.WS.WS_growing import SizeThreshAndGrowWithWS
from segmfriends.algorithms.blockwise import BlockWise

from skunkworks.metrics.cremi_score import cremi_score
from long_range_hc.postprocessing.pipelines import get_segmentation_pipeline

import mutex_watershed as mws

def run_mws(affinities,
            offsets, stride,
            seperating_channel=2,
            invert_dam_channels=True,
            bias_cut=0.,
            randomize_bounds=True):
    assert len(affinities) == len(offsets), "%s, %i" % (str(affinities.shape), len(offsets))
    affinities_ = np.require(affinities.copy(), requirements='C')
    if invert_dam_channels:
        affinities_[seperating_channel:] *= -1
        affinities_[seperating_channel:] += 1
    affinities_[:seperating_channel] += bias_cut
    sorted_edges = np.argsort(affinities_.ravel())
    # run the mst watershed
    vol_shape = affinities_.shape[1:]
    mst = mws.MutexWatershed(np.array(vol_shape),
                             offsets,
                             seperating_channel,
                             stride)
    if randomize_bounds:
        mst.compute_randomized_bounds()
    mst.repulsive_ucc_mst_cut(sorted_edges, 0)
    segmentation = mst.get_flat_label_image().reshape(vol_shape)
    return segmentation


# Add epsilon to affinities:
def add_epsilon(affs, eps=1e-5):
    p_min = eps
    p_max = 1. - eps
    return (p_max - p_min) * affs + p_min



def get_segmentation(affs, offsets, post_proc_config):
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
    outputs = post_proc_solver(affs)
    comp_time = time.time() - tick
    print("Post-processing took {} s".format(comp_time))

    return outputs







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
    affinities  = 1. - vigra.readHDF5(os.path.join(dataset_path, "isbi_results_MWS/isbi_train_offsetsV4_3d_meantda_damws2deval_final.h5"), 'data')
    raw = io.imread(os.path.join(dataset_path, "train-volume.tif"))
    raw = np.array(raw)
    gt = vigra.readHDF5(os.path.join(dataset_path, "gt_mc3d.h5"), 'data')

    # # If this volume is too big, take a crop of it:
    # crop_slice = (slice(None), slice(None, 1), slice(None, 200), slice(None, 200))
    # affinities = affinities[crop_slice]
    # raw = raw[crop_slice[1:]]
    # gt = gt[crop_slice[1:]]




    # Build the graph:
    volume_shape = raw.shape
    print(volume_shape)

    offset_file = 'offsets_MWS.json'
    offset_file = os.path.join('/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/experiments/postprocessing/cremi/offsets/', offset_file)
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
    # affinities = np.random.uniform(0., 1., (17, 10, 200, 100))
    # affinities[:3] = 0.01  # local
    # affinities[3:] = 0.1  # lifted

    # affinities -= np.abs(np.random.normal(scale=0.001, size=affinities.shape))
    # affinities[3:] += np.abs(np.random.normal(scale=0.001, size=(14, 1, 2, 3)))

    # affinities = affinities[:,:10,:150,:150]
    #
    # affinities += np.random.normal(scale=0.002, size=affinities.shape)
    # affinities = np.clip(affinities, 0., 1.)
    affinities = add_epsilon(affinities)


    # # Constantin implementation:
    tick = time.time()
    divMWS = compute_mws_segmentation(affinities, offsets, 3,
                                      randomize_strides=False,
                                      strides = [1,15,15],
                                      algorithm='kruskal')
    print("Took ", time.time() - tick)
    # print(divMWS)

    # yet_another_mws = run_mws(1-affinities, offsets, [1, 1, 1], seperating_channel=3, randomize_bounds=False)
    # print(yet_another_mws)
    # print(yet_another_mws)
    # #
    #
    # file_path_newMWS = os.path.join('/home/abailoni_local/', "ISBI_results_new_MWS.h5")
    # MWS = vigra.readHDF5(file_path_newMWS, 'segm_newMWS')
    #
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
    postproc_config['generalized_HC_kwargs']['strides'] = [1,15,15]
    print(affinities.shape)
    segm_GEC = get_segmentation(affinities, offsets, postproc_config)[0]

    # file_path_segm = os.path.join('/home/abailoni_local/hci_home', "GEC_wo_small_segments.h5")
    # vigra.writeHDF5(segm_GEC.astype('uint32'), file_path_segm, 'segm')

    # print(segm_GEC)

    evals = cremi_score(segm_GEC+1, divMWS+1, border_threshold=None, return_all_scores=True)
    print("Mine and Const")
    print(evals)
    # evals = cremi_score(yet_another_mws + 1, divMWS + 1, border_threshold=None, return_all_scores=True)
    # print("Const and Steffen")
    # print(evals)

    file_path_segm = os.path.join('/home/abailoni_local/', "ISBI_comparing_MWS.h5")
    vigra.writeHDF5(divMWS.astype('uint32'), file_path_segm, 'costMWS')
    # vigra.writeHDF5(yet_another_mws.astype('uint32'), file_path_segm, 'steffenMWS')
    vigra.writeHDF5(segm_GEC.astype('uint32'), file_path_segm, 'mineMWS')

    # print(segm_GEC)
    # evals = cremi_score(gt, divMWS + 1, border_threshold=None, return_all_scores=True)
    # print(evals)
    # evals = cremi_score(gt, segm_GEC + 1, border_threshold=None, return_all_scores=True)
    # print(evals)

