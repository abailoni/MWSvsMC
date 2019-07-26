import numpy as np

from nifty.tools import fromAdjMatrixToEdgeList, fromEdgeListToAdjMatrix

def from_adj_matrix_to_edge_list(sparse_adj_matrix):
    # sparse_adj_matrix.setdiag(np.zeros(sparse_adj_matrix.shape[0], sparse_adj_matrix.dtype))
    nb_edges = sparse_adj_matrix.count_nonzero()
    if not isinstance(sparse_adj_matrix, np.ndarray):
        sparse_adj_matrix = sparse_adj_matrix.toarray()
    # sh = sparse_adj_matrix.shape[0]
    # nb_edges = int((sh*sh - sh) / 2)
    edge_list = np.empty((nb_edges, 3))

    # Set diagonal elements to zero, we don't care about them:
    real_nb_edges = fromAdjMatrixToEdgeList(sparse_adj_matrix, edge_list, 1)
    edge_list = edge_list[:real_nb_edges]
    uvIds = edge_list[:,:2].astype('uint64')
    edge_weights = edge_list[:,2].astype('float32')
    return uvIds, edge_weights


def from_edge_list_to_adj_matrix(uvIds, edge_weights):
    edge_list = np.concatenate((uvIds, np.expand_dims(edge_weights, axis=-1)),
                               axis=1)
    nb_nodes = int(edge_list[:,:2].max()+1)
    adj_matrix = np.zeros((nb_nodes, nb_nodes))
    fromEdgeListToAdjMatrix(adj_matrix, edge_list, 1)
    return adj_matrix
