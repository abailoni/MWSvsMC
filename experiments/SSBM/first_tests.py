from signet.cluster import Cluster
import signet.block_models as signetMdl
from sklearn.metrics import adjusted_rand_score
from scipy import sparse
import numpy as np

# generate a random graph with community structure by the signed stochastic block model


n = 5000    # number of nodes
k = 100      # number of communities
eta = 0.25  # sign flipping probability
p = 0.1   # edge probability


(A_p, A_n), true_assign = signetMdl.SSBM(n = n, k = k, pin=p, etain=eta, values='gaussian')

# initialise the Cluster object with the data (adjacency matrix of positive and negative graph)

from long_range_compare.SSBM_utils import from_adj_matrix_to_edge_list, from_edge_list_to_adj_matrix

A_signed = A_p-A_n
uv_ids, signed_edge_weights = from_adj_matrix_to_edge_list(A_signed)

# adj_dense = from_edge_list_to_adj_matrix(uv_ids, edge_weights)
#
# adj_sparse = sparse.csr_matrix(adj_dense)

# TODO: try dense graph and then add noise!
# TODO: smaller Gaussian!
# TODO: is not symmetrical!!! Graph should be undirected

from GASP.segmentation import run_GASP
from nifty.graph import UndirectedGraph



graph = UndirectedGraph(n)
graph.insertEdges(uv_ids)
nb_edges = graph.numberOfEdges
assert graph.numberOfEdges == uv_ids.shape[0]


# Test connected components:
from nifty.graph import components
components = components(graph)
components.build()
print("Nb. connected components in graph:", np.unique(components.componentLabels()).shape)

# edge_sizes = np.ones_like(signed_edge_weights)
# edge_sizes[signed_edge_weights==0.] = 0.0001

# from segmfriends.algorithms.multicut.multicut import multicut
# node_labels = multicut(graph, n, uv_ids, signed_edge_weights, solver_type="GAEC+kernighanLin")
#
# score_multicut = adjusted_rand_score(node_labels, true_assign)
# print('score MULTICUT', score_multicut)
# _, counts = np.unique(node_labels, return_counts=True)
# print("Numbers of total clusters:", counts.shape[0])
# print("Clusters with more than 50 nodes:", (counts > 50).sum(), "(with sizes {})".format(np.sort(counts)[::-1][:5]))

for update_rule in ["mean", "sum", "abs_max"]:
    # signed_edge_weights = signed_edge_weights - signed_edge_weights.min()
    # additional_kwargs = {'number_of_nodes_to_stop': k}

    print(update_rule, ": ---------------")
    node_labels, runtime = run_GASP(graph, signed_edge_weights,
                                     # edge_sizes=edge_sizes,
                                     linkage_criteria=update_rule,
                                     use_efficient_implementations=False,
                                    # **additional_kwargs
                                    )
    print("DOne in ", runtime)
    labels, counts = np.unique(node_labels,return_counts=True)
    print("Numbers of total clusters:", counts.shape[0])
    print("Clusters with more than 50 nodes:", (counts>50).sum(), "(with sizes {})".format(np.sort(counts)[::-1][:5]))
    score_GASP = adjusted_rand_score(node_labels, true_assign)
    print('score ', update_rule, score_GASP)

print("###############")
print("###############")

# TODO: should I symmetrize it???

# Symmetrize matrices:
grid = np.indices((n, n))
matrix_mask = grid[0] > grid[1]
A_p = matrix_mask * A_p.toarray()
A_n = matrix_mask * A_n.toarray()
A_p = A_p + np.transpose(A_p)
A_n = A_n + np.transpose(A_n)
c = Cluster((sparse.csr_matrix(A_p), sparse.csr_matrix(A_n)))


# calculate the assignments provided by the algorithms you want to analyse

# A_assign = c.spectral_cluster_adjacency(k = k)
import time
tick = time.time()
L_assign = c.spectral_cluster_laplacian(k = k, normalisation='sym')
print("L_sym took", time.time()-tick)

tick = time.time()
SPONGE_assign = c.geproblem_laplacian(k = k, normalisation='additive')
print("SPONGE took", time.time()-tick)
tick = time.time()
SPONGEsym_assign = c.geproblem_laplacian(k = k, normalisation='multiplicative')
print("SPONGEsym took", time.time()-tick)
# compute the recovery score of the algorithms against the SSBM ground truth

# score_A = adjusted_rand_score(A_assign, true_assign)

score_L = adjusted_rand_score(L_assign, true_assign)

score_SPONGE = adjusted_rand_score(SPONGE_assign, true_assign)

score_SPONGEsym = adjusted_rand_score(SPONGEsym_assign, true_assign)

print("####### Score L")
_, counts = np.unique(L_assign, return_counts=True)
print("Numbers of total clusters:", counts.shape[0])
print("Clusters with more than 50 nodes:", (counts > 50).sum(), "(with sizes {})".format(np.sort(counts)[::-1][:5]))
print('score_L: ', score_L)


# print('score_A: ', score_A)
print("####### Score SPONGE")
_, counts = np.unique(SPONGE_assign, return_counts=True)
print("Numbers of total clusters:", counts.shape[0])
print("Clusters with more than 50 nodes:", (counts > 50).sum(), "(with sizes {})".format(np.sort(counts)[::-1][:5]))
print('score_SPONGE: ', score_SPONGE)
print('score_SPONGEsym: ', score_SPONGEsym)
