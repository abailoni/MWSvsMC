import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import nifty as nf
import vigra
import numpy as np
import os

from skimage import io



from compareMCandMWS import utils as utils
from compareMCandMWS.divisiveMWS import DivisiveMWS


if __name__ == '__main__':

    root_path = "/export/home/abailoni/supervised_projs/divisiveMWS"
    # dataset_path = os.path.join(root_path, "cremi-dataset-crop")
    dataset_path = "/net/hciserver03/storage/abailoni/datasets/ISBI/"
    plots_path = os.path.join(root_path, "plots")
    save_path = os.path.join(root_path, "outputs")

    # Import data:
    affinities  = vigra.readHDF5(os.path.join(dataset_path, "isbi_results_MWS/isbi_train_offsetsV4_3d_meantda_damws2deval_final.h5"), 'data')
    raw = io.imread(os.path.join(dataset_path, "train-volume.tif"))
    raw = np.array(raw)
    gt = vigra.readHDF5(os.path.join(dataset_path, "gt_mc3d.h5"), 'data')

    # If this volume is too big, take a crop of it:
    crop_slice = (slice(None), slice(None, 1), slice(None, 200), slice(None, 200))
    affinities = affinities[crop_slice]
    raw = raw[crop_slice[1:]]
    gt = gt[crop_slice[1:]]

    # Build the graph:
    volume_shape = raw.shape

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
    multicut_costs = utils.probs_to_costs(edge_weights)
    is_edge_attractive = is_edge_local.copy()  # returned shape: (nb_edges, )
    edge_weights[is_edge_attractive] = 1. - edge_weights[is_edge_attractive]
    print("Number of nodes and edges: ", graph.numberOfNodes, graph.numberOfEdges)

    # The final edge weights you get here are like the ones described in the MWS paper:
    #  - all weights are positive (here in particular all in the interval [0., 1.0])
    #  - if an edge is attractive, high weight means that it wants to be connected
    #  - if an edge is repulsive, high weight means that it does NOT want to be connected


    clustering = DivisiveMWS(graph, edge_weights, is_edge_attractive)
    nb_iterations = -1
    root_path = "/export/home/abailoni/supervised_projs/divisiveMWS"
    # dataset_path = os.path.join(root_path, "cremi-dataset-crop")
    dataset_path = "/net/hciserver03/storage/abailoni/datasets/ISBI/"
    plots_path = os.path.join(root_path, "plots")
    save_path = os.path.join(root_path, "outputs")

    # Import data:
    affinities  = vigra.readHDF5(os.path.join(dataset_path, "isbi_results_MWS/isbi_train_offsetsV4_3d_meantda_damws2deval_final.h5"), 'data')
    raw = io.imread(os.path.join(dataset_path, "train-volume.tif"))
    raw = np.array(raw)
    gt = vigra.readHDF5(os.path.join(dataset_path, "gt_mc3d.h5"), 'data')

    # If this volume is too big, take a crop of it:
    crop_slice = (slice(None), slice(None, 1), slice(None, 200), slice(None, 200))
    affinities = affinities[crop_slice]
    raw = raw[crop_slice[1:]]
    gt = gt[crop_slice[1:]]

    # Build the graph:
    volume_shape = raw.shape

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
    multicut_costs = utils.probs_to_costs(edge_weights)
    is_edge_attractive = is_edge_local.copy()  # returned shape: (nb_edges, )
    edge_weights[is_edge_attractive] = 1. - edge_weights[is_edge_attractive]
    print("Number of nodes and edges: ", graph.numberOfNodes, graph.numberOfEdges)

    # The final edge weights you get here are like the ones described in the MWS paper:
    #  - all weights are positive (here in particular all in the interval [0., 1.0])
    #  - if an edge is attractive, high weight means that it wants to be connected
    #  - if an edge is repulsive, high weight means that it does NOT want to be connected


    clustering = DivisiveMWS(graph, edge_weights, is_edge_attractive)
    nb_iterations = -1

    final_node_labels, final_edge_labels, edge_push = clustering(nb_iterations=nb_iterations)

    edge_ids = graph.mapEdgesIDToImage()

    from segmfriends.features.mappings import map_features_to_label_array
    from segmfriends.vis import mask_the_mask
    mapped_edge_push = map_features_to_label_array(edge_ids, np.expand_dims(edge_push, axis=1))


    # Multicut energy of the final segmentation:
    energy = (multicut_costs * final_edge_labels).sum()
    print("Final MC energy: {}".format(energy))

    # PLOT a slice of the final segmented volume:
    final_segm = final_node_labels.reshape(volume_shape)
    ncols, nrows = 1, 1
    f, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(7, 7))
    utils.plot_segm(ax, final_segm, z_slice=0, background=raw)
    f.savefig(os.path.join(plots_path, 'divisive_MWS_{}_iterations.pdf'.format(nb_iterations)), format='pdf')

    # PLOT a slice of the final segmented volume:
    ncols, nrows = 2, 1
    f, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(7, 7))

    print(mapped_edge_push[0, :, :, 1:3, 0].min() , mapped_edge_push[0, :, :, 1:3, 0].max())
    print(mapped_edge_push[0, :, :, 0, 0].min(), mapped_edge_push[0, :, :, 0, 0].max())
    ax[0].matshow(raw[0], cmap='gray', interpolation='none')
    ax[1].matshow(affinities[2,0], cmap='gray', interpolation='none')
    cax = ax[0].matshow(mask_the_mask(mapped_edge_push[0,:,:,1,0]), cmap=plt.get_cmap('cool'), interpolation='none')
    cax = ax[1].matshow(mask_the_mask(mapped_edge_push[0, :, :,2, 0]), cmap=plt.get_cmap('cool'), interpolation='none')
    f.savefig(os.path.join(plots_path, 'strength_edges{}_iterations.png'.format(nb_iterations)), format='png',  dpi = 300)


    np.where




    # Save some data:
    vigra.writeHDF5(final_segm.astype('uint32'), os.path.join(save_path, "divisiveMWS_{}_iter.h5".format(nb_iterations)), 'data')

    final_node_labels, final_edge_labels, edge_push = clustering(nb_iterations=nb_iterations)

    edge_ids = graph.mapEdgesIDToImage()

    from segmfriends.features.mappings import map_features_to_label_array
    from segmfriends.vis import mask_the_mask
    mapped_edge_push = map_features_to_label_array(edge_ids, np.expand_dims(edge_push, axis=1))


    # Multicut energy of the final segmentation:
    energy = (multicut_costs * final_edge_labels).sum()
    print("Final MC energy: {}".format(energy))

    # PLOT a slice of the final segmented volume:
    final_segm = final_node_labels.reshape(volume_shape)
    ncols, nrows = 1, 1
    f, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(7, 7))
    utils.plot_segm(ax, final_segm, z_slice=0, background=raw)
    f.savefig(os.path.join(plots_path, 'divisive_MWS_{}_iterations.pdf'.format(nb_iterations)), format='pdf')

    # PLOT a slice of the final segmented volume:
    ncols, nrows = 2, 1
    f, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(7, 7))

    print(mapped_edge_push[0, :, :, 1:3, 0].min() , mapped_edge_push[0, :, :, 1:3, 0].max())
    print(mapped_edge_push[0, :, :, 0, 0].min(), mapped_edge_push[0, :, :, 0, 0].max())
    ax[0].matshow(raw[0], cmap='gray', interpolation='none')
    ax[1].matshow(affinities[2,0], cmap='gray', interpolation='none')
    cax = ax[0].matshow(mask_the_mask(mapped_edge_push[0,:,:,1,0]), cmap=plt.get_cmap('cool'), interpolation='none')
    cax = ax[1].matshow(mask_the_mask(mapped_edge_push[0, :, :,2, 0]), cmap=plt.get_cmap('cool'), interpolation='none')
    f.savefig(os.path.join(plots_path, 'strength_edges{}_iterations.png'.format(nb_iterations)), format='png',  dpi = 300)


    np.where




    # Save some data:
    vigra.writeHDF5(final_segm.astype('uint32'), os.path.join(save_path, "divisiveMWS_{}_iter.h5".format(nb_iterations)), 'data')

