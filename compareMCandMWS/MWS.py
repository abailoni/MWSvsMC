import numpy as np
import nifty.graph.agglo as nagglo
import time


def MWS(graph, edge_weights, is_attractive_edge):
    assert (edge_weights<0.).sum() == 0, "Edge weights should be all positive!"
    assert edge_weights.shape == is_attractive_edge.shape
    assert edge_weights.shape[0] == graph.numberOfEdges

    mergePrio = edge_weights.copy()
    notMergePrio = edge_weights.copy()
    mergePrio[np.logical_not(is_attractive_edge)] = -1.
    notMergePrio[is_attractive_edge] = -1.

    tick = time.time()
    cluster_policy = nagglo.fixationClusterPolicy(graph=graph,
                                              mergePrios=mergePrio, notMergePrios=notMergePrio,
                                              edgeSizes=np.ones_like(edge_weights), nodeSizes=np.ones(graph.numberOfNodes),
                                              isMergeEdge=is_attractive_edge,
                                              updateRule0=nagglo.updatRule('max'),
                                              updateRule1=nagglo.updatRule('max'),
                                              zeroInit=False,
                                              initSignedWeights=False,
                                              sizeRegularizer=0.,
                                              sizeThreshMin=0.,
                                              sizeThreshMax=300.,
                                              postponeThresholding=False,
                                              costsInPQ=False,
                                              checkForNegCosts=True,
                                              addNonLinkConstraints=False,
                                              threshold=0.5,
                                              )

    agglomerativeClustering = nagglo.agglomerativeClustering(cluster_policy)
    agglomerativeClustering.run(verbose=False)
    final_node_labels = agglomerativeClustering.result()
    tock = time.time()

    return final_node_labels, tock-tick
