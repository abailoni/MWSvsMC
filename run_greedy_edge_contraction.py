import sys

sys.path += ["/home/abailoni_local/hci_home/python_libraries/nifty/python",
"/home/abailoni_local/hci_home/pyCharm_projects/inferno",
"/home/abailoni_local/hci_home/pyCharm_projects/constrained_mst",
"/home/abailoni_local/hci_home/pyCharm_projects/neuro-skunkworks",
"/home/abailoni_local/hci_home/pyCharm_projects/segmfriends",
"/home/abailoni_local/hci_home/pyCharm_projects/hc_segmentation",
"/home/abailoni_local/hci_home/pyCharm_projects/neurofire",]

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import nifty as nf
import nifty.graph.agglo as nagglo
import vigra
import numpy as np
import os

from skimage import io
import argparse
import time
import json


from compareMCandMWS import utils as utils

from segmfriends.utils.config_utils import adapt_configs_to_model
from segmfriends.utils import yaml2dict
from skunkworks.metrics.cremi_score import cremi_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_slice', default=None)
    parser.add_argument('--n_threads', default=1, type=int)
    parser.add_argument('--name_aggl', default=None)
    parser.add_argument('--model_IDs', nargs='+', default=None, type=str)
    args = parser.parse_args()

    models_IDs = args.model_IDs

    configs = {'models': yaml2dict('./experiments/models_config.yml'),
        'postproc': {}}
    if models_IDs is not None:
        configs = adapt_configs_to_model(models_IDs, debug=True, **configs)

    postproc_config = configs['postproc']
    aggl_kwargs = postproc_config.get('generalized_HC_kwargs', {}).get('agglomeration_kwargs', {}).get(
        'extra_aggl_kwargs', {})

    root_path = "/export/home/abailoni/supervised_projs/divisiveMWS"
    # dataset_path = os.path.join(root_path, "cremi-dataset-crop")
    dataset_path = "/net/hciserver03/storage/abailoni/datasets/ISBI/"
    plots_path = os.path.join("/net/hciserver03/storage/abailoni/greedy_edge_contr/plots")
    save_path = os.path.join(root_path, "outputs")

    # Import data:
    affinities  = 1 - vigra.readHDF5(os.path.join(dataset_path, "isbi_results_MWS/isbi_train_offsetsV4_3d_meantda_damws2deval_final.h5"), 'data')
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

    # Offsets defines the 3D-connectivity patterns of the edges in the 3D pixel grid graph:
    nb_local_offsets = 3
    local_offsets = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])  # Local edges in three directions
    # Long-range connectivity patterns:
    non_local_offsets = np.array([[-1, -1, -1],
                                  [-1, 1, 1],
                                  [-1, -1, 1],
                                  [-1, 1, -1],
                                  [0, -9, 0],
                                  [0, 0, -9],
                                  [0, -9, -9],
                                  [0, 9, -9],
                                  [0, -9, -4],
                                  [0, -4, -9],
                                  [0, 4, -9],
                                  [0, 9, -4],
                                  [0, -27, 0],
                                  [0, 0, -27]])
    offsets = np.concatenate((local_offsets, non_local_offsets))



    is_local_offset = [True] * 3 + [False] * 14

    # BUILD GRAPH:
    graph = nf.graph.undirectedLongRangeGridGraph(np.array(volume_shape), offsets,
                                                     is_local_offset=np.array(is_local_offset))
    nb_nodes = graph.numberOfNodes
    nb_edges = graph.numberOfEdges

    # Get IDs of the local edges in the graph:
    offset_index = graph.edgeOffsetIndex()
    is_edge_local = np.zeros((nb_edges,), dtype='bool')
    is_edge_local[offset_index < nb_local_offsets] = True


    # These three arrays should be enough for your algorithm:
    # and in principle now if you really want to build the adj. matrix it should be really easy
    uvIds = graph.uvIds()  # returned shape: (nb_edges, 2)
    edge_weights = graph.edgeValues(np.rollaxis(affinities, axis=0, start=4))  # returned shape: (nb_edges, )

    multicut_costs = utils.probs_to_costs(1 - edge_weights)

    if aggl_kwargs.pop('use_log_costs', False):
        signed_weights = multicut_costs
    else:
        signed_weights = edge_weights - 0.5

    # is_edge_attractive = is_edge_local.copy()  # returned shape: (nb_edges, )
    # edge_weights[is_edge_attractive] = 1. - edge_weights[is_edge_attractive]
    print("Number of nodes and edges: ", graph.numberOfNodes, graph.numberOfEdges)


    cluster_policy = nagglo.greedyGraphEdgeContraction(graph, signed_weights,
                                                       is_merge_edge=is_edge_local,
                                                       **aggl_kwargs
                                                       )

    agglomerativeClustering = nagglo.agglomerativeClustering(cluster_policy)
    # agglomerativeClustering.run(**self.extra_runAggl_kwargs)
    tick = time.time()

    from skunkworks.postprocessing.watershed import DamWatershed
    mws = DamWatershed(list(offsets), [1, 1, 1],
                       seperating_channel=3, invert_dam_channels=True,
                       randomize_bounds=False,
                       )

    final_segm = mws(1 - affinities)

    # outputs = agglomerativeClustering.runAndGetMergeTimesAndDendrogramHeight(verbose=False)
    # mergeTimes, UCMap = outputs
    # nodeSeg = agglomerativeClustering.result()

    comp_time = time.time() - tick

    # edge_IDs = graph.mapEdgesIDToImage()
    #
    # # final_UCM = np.squeeze(
    # #     mappings.map_features_to_label_array(edge_IDs, np.expand_dims(mergeTimes, axis=-1)))
    #
    #
    # edge_labels = graph.nodesLabelsToEdgeLabels(nodeSeg)
    # MC_energy = (multicut_costs * edge_labels).sum()
    # print("MC energy: {}".format(MC_energy))
    #
    #
    #
    # # PLOT a slice of the final segmented volume:
    # final_segm = nodeSeg.reshape(volume_shape)
    ncols, nrows = 1, 1
    f, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(7, 7))
    utils.plot_segm(ax, final_segm, z_slice=0, highlight_boundaries=False)
    f.savefig(os.path.join(plots_path, 'segm_{}_eff.png'.format(models_IDs[0])), format='png', dpi=300)

    ncols, nrows = 1, 1
    f, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(7, 7))
    utils.plot_segm(ax, final_segm, z_slice=0, background=raw)
    f.savefig(os.path.join(plots_path, 'segm_{}_2_eff.png'.format(models_IDs[0])), format='png', dpi=300)

    # # PLOT a slice of the final segmented volume:
    # ncols, nrows = 2, 1
    # f, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(7, 7))
    #
    # print(mapped_edge_push[0, :, :, 1:3, 0].min() , mapped_edge_push[0, :, :, 1:3, 0].max())
    # print(mapped_edge_push[0, :, :, 0, 0].min(), mapped_edge_push[0, :, :, 0, 0].max())
    # ax[0].matshow(raw[0], cmap='gray', interpolation='none')
    # ax[1].matshow(affinities[2,0], cmap='gray', interpolation='none')
    # cax = ax[0].matshow(mask_the_mask(mapped_edge_push[0,:,:,1,0]), cmap=plt.get_cmap('cool'), interpolation='none')
    # cax = ax[1].matshow(mask_the_mask(mapped_edge_push[0, :, :,2, 0]), cmap=plt.get_cmap('cool'), interpolation='none')
    # f.savefig(os.path.join(plots_path, 'strength_edges{}_iterations.png'.format(nb_iterations)), format='png',  dpi = 300)

    evals = cremi_score(gt, final_segm, border_threshold=None, return_all_scores=True)
    print("Scores achieved: ", evals)

    eval_file = os.path.join(plots_path, 'scores_{}_eff.json'.format(models_IDs[0]))

    evals['computation_time'] = comp_time
    # evals['MC_energy'] = np.asscalar(MC_energy)
    with open(eval_file, 'w') as f:
        json.dump(evals, f, indent=4, sort_keys=True)


    # Save some data:
    vigra.writeHDF5(final_segm.astype('uint32'), os.path.join(plots_path, "out_{}_eff.h5".format(models_IDs[0])), 'data')

