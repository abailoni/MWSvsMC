import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sys
sys.path.append("/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/")
import nifty
import nifty.graph.rag as nrag
import vigra
import numpy as np
import os
from long_range_hc.criteria.learned_HC.utils.segm_utils import map_features_to_label_array

import mutex_watershed
import nifty.graph.agglo as nagglo
import time
from long_range_compare import utils as utils

import json

from skunkworks.metrics.cremi_score import cremi_score

# Load graph weights
root_path = "/export/home/abailoni/supervised_projs/MWS_vs_MC"
dataset_path = os.path.join(root_path, "cremi-dataset-crop")
# dataset_path = "/net/hciserver03/storage/abailoni/learnedHC/plain_unstruct/pureDICE_wholeTrainingSet/postprocess/inferName_v100k_repAttrHC095_B/"
plots_path = os.path.join(root_path, "plots")
save_path = os.path.join(root_path, "outputs")

# Import data:
uv_IDs = vigra.readHDF5(os.path.join(dataset_path, "edge_data.h5"), 'uv_IDs')
edge_costs = vigra.readHDF5(os.path.join(dataset_path, "edge_data.h5"), 'edge_weights')
segm = vigra.readHDF5(os.path.join(dataset_path, "segmentation.h5"), 'data')
raw = vigra.readHDF5(os.path.join(dataset_path, "raw.h5"), 'data')
# gt = vigra.readHDF5(os.path.join(dataset_path, "gt.h5"), 'data')
rag = nrag.gridRag(segm.astype('uint32'))
print("Number of nodes and edges: ", rag.numberOfNodes, rag.numberOfEdges)

# # Compute edge sizes:
# shape = segm.shape
# fake_data = np.zeros(shape, dtype='float32')
# edge_sizes = nrag.accumulateEdgeMeanAndLength(rag, fake_data)[:, 1]
# print(edge_sizes.min(), edge_sizes.max(), edge_sizes.mean())
# edge_costs = edge_costs * edge_sizes / edge_sizes.max() * 16.
edge_sizes = np.ones_like(edge_costs)

# NOISE:
# noise = np.random.normal(scale=0.001, size=edge_costs.shape)
# vigra.writeHDF5(noise, "/export/home/abailoni/supervised_projs/MWS_vs_MC/results/noise.h5", 'data')
# noise  = vigra.readHDF5("/export/home/abailoni/supervised_projs/MWS_vs_MC/results/noise.h5", 'data')
# edge_costs += noise




# Find mutex-edges:
negative_edges = edge_costs<0.
print((edge_costs==0.).sum())

back_to_probs = False
if back_to_probs:
    # Go back to affinities:
    used_costs = 1 - 1. / (1. + np.exp(edge_costs))
    mergePrio = used_costs.copy()
    notmergePrio = 1 - used_costs.copy()
else:
    mergePrio = edge_costs.copy()
    notmergePrio = -edge_costs.copy()

tick = time.time()
# final_node_labels = mutex_watershed.compute_mws_clustering(rag.numberOfNodes,
#                                                      uv_IDs[1-negative_edges], uv_IDs[negative_edges],
#                                                      edge_costs[1-negative_edges], -edge_costs[negative_edges])


mergePrio[negative_edges] = -1.
notmergePrio[np.logical_not(negative_edges)] = -1.
# notmergePrio[:] = 0.

cluster_policy = nagglo.fixationClusterPolicy(graph=rag,
                                              mergePrios=mergePrio, notMergePrios=notmergePrio,
                                              edgeSizes=np.ones_like(edge_sizes), nodeSizes=np.ones(rag.numberOfNodes),
                                              isMergeEdge=np.ones(rag.numberOfEdges),
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
agglomerativeClustering.run()
final_node_labels = agglomerativeClustering.result()
tock = time.time()


# else:
#     neg_weights = edge_costs < 0.
#     p = 1
#     exp_costs = np.abs(edge_costs)**p
#     exp_costs[neg_weights] *= -1
#     mc_obj = rag.MulticutObjective(graph=rag, weights=exp_costs)
#
#     tick = time.time()
#     log_visitor = mc_obj.loggingVisitor(verbose=False, timeLimitSolver=np.inf, timeLimitTotal=np.inf)
#
#     # solverFactory = mc_obj.multicutIlpFactory()
#     # solver = solverFactory.create(mc_obj)
#     # final_node_labels = solver.optimize(visitor=log_visitor)
#
#
#     # 1. Initialize a warm-up solver and run optimization
#     solverFactory = mc_obj.greedyAdditiveFactory()
#     solver = solverFactory.create(mc_obj)
#     node_labels = solver.optimize(visitor=log_visitor)
#     # 2. Use a second better warm-up solver to get a better solution:
#     log_visitor = mc_obj.loggingVisitor(verbose=True, timeLimitSolver=np.inf, timeLimitTotal=np.inf)
#     solverFactory = mc_obj.kernighanLinFactory()
#     solver = solverFactory.create(mc_obj)
#     new_node_labels = solver.optimize(visitor=log_visitor, nodeLabels=node_labels)
#     # final_node_labels = new_node_labels
#
#     # 4. Run the funsionMuves solver
#     pgen = mc_obj.watershedCcProposals(sigma=1.0, numberOfSeeds=0.1)
#     # pgen = mc_obj.greedyAdditiveProposals(sigma=1.5, weightStopCond=0.0, nodeNumStopCond=-1.0)
#     solverFactory = mc_obj.ccFusionMoveBasedFactory(proposalGenerator=pgen, numberOfIterations=2,
#                                                             stopIfNoImprovement=1, numberOfThreads=1)
#     solver = solverFactory.create(mc_obj)
#     final_node_labels = solver.optimize(visitor=log_visitor, nodeLabels=new_node_labels)
#
#
#     # results = np.stack([log_visitor.iterations(), log_visitor.energies(), log_visitor.runtimes()])
#     # vigra.writeHDF5(results, "/export/home/abailoni/supervised_projs/MWS_vs_MC/results/fusionMove.h5", 'data')
#
#     tock = time.time()
#
#
#     ncols, nrows = 1, 1
#     f, ax = plt.subplots(ncols=ncols, nrows=nrows)
#
#     ax.semilogy(log_visitor.runtimes(), -log_visitor.energies())
#
#     # plt.subplots_adjust(wspace=0, hspace=0)
#     # plt.tight_layout()
#     save_path = "/export/home/abailoni/supervised_projs/MWS_vs_MC/plots"
#     f.savefig(os.path.join(save_path, 'second_try.pdf'), format='pdf')

edge_labels = rag.nodesLabelsToEdgeLabels(final_node_labels)

vigra.writeHDF5(edge_labels, "/export/home/abailoni/supervised_projs/MWS_vs_MC/results/edgeLabelsMWS.h5", 'data')

energy = (edge_costs * edge_labels).sum()
print("Took {}s. Final energy: {}".format(tock-tick, energy))



final_segm = np.squeeze(map_features_to_label_array(segm, np.expand_dims(final_node_labels, axis=-1)))

ncols, nrows = 2, 1
f, ax = plt.subplots(ncols=ncols, nrows=nrows)

utils.plot_segm(ax[0], segm, z_slice=5, background=raw)
utils.plot_segm(ax[1], final_segm, z_slice=5, background=raw)
f.savefig(os.path.join(plots_path, 'aggl.pdf'), format='pdf')

# evals = cremi_score(gt, final_segm, border_threshold=None, return_all_scores=True)
# print("Scores achieved: ", evals)
#
# with open(os.path.join(save_path, 'scores.json'), 'w') as f:
#     json.dump(evals, f, indent=4, sort_keys=True)
#


# plt.subplots_adjust(wspace=0, hspace=0)
# plt.tight_layout()
