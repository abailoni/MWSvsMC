import sys

sys.path += ["/home/abailoni_local/hci_home/python_libraries/nifty/python",
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
import nifty as nf
import nifty.graph.agglo as nagglo
import vigra
import numpy as np
import os

from skimage import io
import argparse
import time
import json


from long_range_compare import utils as utils

from segmfriends.utils.config_utils import adapt_configs_to_model
from segmfriends.utils import yaml2dict
from skunkworks.metrics.cremi_score import cremi_score

from memory_profiler import profile

@profile
def test_nifty_memory():
    # image_shape = np.array((120, 1250, 1250))
    image_shape = np.array((10, 500, 500))
    nb_nodes = np.asscalar(np.prod(image_shape))
    nb_edges = 17 * nb_nodes


    # # Simplified graph representation:
    node_representation = np.random.randint(nb_nodes, size=(nb_nodes, 17, 2), dtype='int64')
    edge_representation = np.random.randint(nb_nodes, size=(nb_edges, 2), dtype='int64')

    # # Initialize some UnionFind datastruct:
    # import nifty.ufd as nufd
    # UF = nufd.ufd(nb_nodes)
    # # UF_edges = nufd.ufd(nb_edges)



    # # Build nifty graph:
    # print("Build nifty graph")
    # graph = nf.graph.undirectedGraph(nb_nodes)
    # graph.insertEdges(edge_representation)
    # print("DONE")

    # print("Build my graph")
    # from nifty.graph import lightGraph
    # lightGraph = lightGraph(nb_nodes+1, edge_representation)
    # print("DONE")


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



    # graph = nf.graph.undirectedLongRangeGridGraph(image_shape, offsets,
    #                                                  is_local_offset=np.array(is_local_offset))
    graph = nf.graph.undirectedLongRangeGridGraph(image_shape, offsets,
                                                  is_local_offset=np.array(is_local_offset),
                                                  strides=[1,1,1])


test_nifty_memory()