import nifty
import numpy as np
import vigra

from long_range_compare.utils import probs_to_costs

dataset_path = "/net/hciserver03/storage/abailoni/datasets/ISBI/isbi_results_MWS/isbi_train_offsetsV4_3d_meantda_damws2deval_final.h5"


affinities = 1 - vigra.readHDF5(dataset_path,'data')

# If this volume is too big, take a crop of it
affinities = affinities[:, :10, :200, :200]


volume_shape = affinities.shape[1:]

# Offsets defines the 3D-connectivity patterns of the edges in the 3D pixel grid graph:
nb_local_offsets = 3
local_offsets = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]) # Local edges in three directions
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


# Build graph:
graph = nifty.graph.undirectedLongRangeGridGraph(volume_shape, offsets)
nb_nodes = graph.numberOfNodes
nb_edges = graph.numberOfEdges

# Get IDs of the local edges in the graph:
offset_index = graph.edgeOffsetIndex()
is_edge_local = np.zeros((nb_edges,), dtype='bool')
is_edge_local[offset_index < nb_local_offsets] = True



# FINAL GRAPH REPRESENTATION:

# These three arrays should be enough for your algorithm:
# and in principle now if you really want to build the adj. matrix it should be really easy
uvIds = graph.uvIds() # returned shape: (nb_edges, 2)
edge_weights = graph.edgeValues(np.rollaxis(affinities, axis=0, start=4)) # returned shape: (nb_edges, )


weight_setup = 'long-range-repulsions'
if weight_setup == 'long-range-repulsions':
     # OPTION 1: MWS paper setup
     is_edge_repulsive = np.logical_not(is_edge_local) # returned shape: (nb_edges, )
     edge_weights[is_edge_repulsive] = 1. - edge_weights[is_edge_repulsive]
elif weight_setup == 'MC_setup':
     # OPTION 2, MC setup: use log costs and define repulsive/attractive edges accordingly
     edge_weights = probs_to_costs(1-edge_weights, beta=0.5)
     is_edge_repulsive = edge_weights < 0
     edge_weights = np.abs(edge_weights)

# ....
