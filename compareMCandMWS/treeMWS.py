import numpy as np
import nifty
import nifty.graph.agglo as nagglo
import time

import nifty.ufd as nufd

class MyCallback(nifty.graph.EdgeContractionGraphCallback):
    def __init__(self, uf_edges=None):
        """
        :param uf_edges: Union find structure for edges, optional.
        """
        self.uf_edges = uf_edges
        super(MyCallback, self).__init__()

    def contractEdge(self, edgeToContract):
        pass

    def mergeEdges(self, aliveEdge, deadEdge):
        if self.uf_edges is not None:
            self.uf_edges.merge(aliveEdge, deadEdge)

    def mergeNodes(self, aliveNode, deadNode):
        pass

    def contractEdgeDone(self, contractedEdge):
        pass


def node_labels_from_spanning_forest(graph, spanning_forest):
    """
    :param spanning_forest: array with shape (nb_edges, ) s.t. e=1 if it is in the forest, 0 otherwise.
    :return: node labels of the associated clustering with shape (nb_nodes, )
    """
    contrGraph = nifty.graph.edgeContractionGraph(graph, MyCallback())


    for e in graph.edges():
        if spanning_forest[e] == 1:
            # get the endpoints of e in the original
            u, v = graph.uv(e)

            # get the parent nodes in the contracted graph
            cu = contrGraph.findRepresentativeNode(u)
            cv = contrGraph.findRepresentativeNode(v)

            if (cu != cv):
                # Contract the edge in the graph (and update UF structures):
                ce = contrGraph.findRepresentativeEdge(e)
                contrGraph.contractEdge(ce)
            else:
                raise ValueError("Something went wrong. The passed one was not a spanning forest (cycle encountered)!")

    # Deduce node labels from contracted graph:
    final_node_labels = np.empty(graph.numberOfNodes, dtype='uint64')
    for n in graph.nodes():
        final_node_labels[n] = contrGraph.findRepresentativeNode(n)

    return final_node_labels



class TreeMWS(object):
    def __init__(self, graph, edge_weights, is_attractive_edge):
        """
        The expected edge weights are like the ones described in the MWS paper:
         - all weights are positive (here in particular all in the interval [0., 1.0])
         - if an edge is attractive, high weight means that it wants to be connected
         - if an edge is repulsive, high weight means that it does NOT want to be connected

        """

        assert (edge_weights < 0.).sum() == 0, "Edge weights should be all positive!"
        assert edge_weights.shape == is_attractive_edge.shape
        assert edge_weights.shape[0] == graph.numberOfEdges
        self.nb_nodes = graph.numberOfNodes
        self.nb_edges = graph.numberOfEdges
        self.graph = graph
        self.edge_weights = edge_weights
        self.is_edge_attractive = is_attractive_edge

    def __call__(self, nb_iterations=-1):
        """
        :param nb_iterations: How many repulsive edges we want to enforce.
                If -1, all repulsive constraints are added.
        :return: node and edge labelling of the final clustering (1 if edge is on cut, 0 otherwise)
        """
        # Build UnionFind structures for edges:
        # REMARK: Actually the union find structure for edges and nodes is already implemented in nifty contracted graph
        # (the find operation can be called with contrGraph.findRepresentativeEdge() and contrGraph.findRepresentativeNode()
        # So the following is extra and not needed. I left it for more clarity.
        self.uf_edges = UFedg = nufd.ufd(self.nb_edges)

        # Build contracted graph:
        # the callback
        callback = MyCallback(uf_edges=UFedg)
        self.contrGraph = nifty.graph.edgeContractionGraph(self.graph, callback)

        # Sort edges starting from the highest:
        edge_arg_sort = np.argsort(-self.edge_weights)

        # Build maximum spanning tree of the attractive edges and remember weakest edges:
        weakest_edges = np.empty_like(self.edge_weights, dtype='uint64')
        max_spanning_tree = np.zeros_like(self.edge_weights, dtype='uint8')
        for e in edge_arg_sort:
            # get the endpoints of e in the original
            u, v = self.graph.uv(e)

            # get the parent nodes in the contracted graph
            cu = self.contrGraph.findRepresentativeNode(u)
            cv = self.contrGraph.findRepresentativeNode(v)

            # Avoid cycles in the tree:
            if (cu != cv):
                if self.is_edge_attractive[e]:
                    max_spanning_tree[e] = 1

                # Remember about this weaked edge:
                weakest_edges[self.uf_edges.find(e)] = e

                # Contract the edge in the graph (and update UF structures):
                ce = self.contrGraph.findRepresentativeEdge(e)
                self.contrGraph.contractEdge(ce)

        # This tree (or forest) will represent the final segmentation:
        clustering_forest = max_spanning_tree.copy()


        # Get the final node_labelling from the obtained spanning forest:
        final_node_labels = node_labels_from_spanning_forest(self.graph, clustering_forest)

        final_edge_labels = self.graph.nodesLabelsToEdgeLabels(final_node_labels)

        return final_node_labels, final_edge_labels

