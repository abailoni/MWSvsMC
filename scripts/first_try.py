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



# Load graph weights
folder_path = "/export/home/abailoni/supervised_projs/MWS_vs_MC/cremi-dataset-crop"
uv_IDs  = vigra.readHDF5(os.path.join(folder_path, "edge_data.h5"), 'uv_IDs')
edge_costs  = vigra.readHDF5(os.path.join(folder_path, "edge_data.h5"), 'edge_weights')
segm  = vigra.readHDF5(os.path.join(folder_path, "segmentation.h5"), 'data')
raw  = vigra.readHDF5(os.path.join(folder_path, "raw.h5"), 'data')


rag = nrag.gridRag(segm.astype('uint32'))



if False:
    # Go back to affinities:
    affs = 1 - 1. / (1. + np.exp(edge_costs))


    # Find mutex-edges:
    mutex_ids = edge_costs<0.

    tick = time.time()
    # final_node_labels = mutex_watershed.compute_mws_clustering(rag.numberOfNodes,
    #                                                      uv_IDs[1-mutex_ids], uv_IDs[mutex_ids],
    #                                                      edge_costs[1-mutex_ids], -edge_costs[mutex_ids])

    # noise = np.random.normal(scale=0.001, size=edge_costs.shape)
    # edge_costs += noise

    mergePrio = affs.copy()
    notmergePrio = 1 - affs.copy()
    mergePrio[mutex_ids] = -1.
    notmergePrio[1-mutex_ids] = -1.

    cluster_policy = nagglo.fixationClusterPolicy(graph=rag,
                                                  mergePrios=mergePrio, notMergePrios=notmergePrio,
                                                  edgeSizes=np.ones(rag.numberOfEdges), nodeSizes=np.ones(rag.numberOfNodes),
                                                  isMergeEdge=np.ones(rag.numberOfEdges),
                                                  updateRule0=nagglo.updatRule('max'),
                                                  updateRule1=nagglo.updatRule('max'),
                                                  zeroInit=False,
                                                  initSignedWeights=False,
                                                  sizeRegularizer=0.,
                                                  sizeThreshMin=0.,
                                                  sizeThreshMax=300.,
                                                  postponeThresholding=False,
                                                  threshold=0.5,
                                                  )

    agglomerativeClustering = nagglo.agglomerativeClustering(cluster_policy)
    agglomerativeClustering.run()
    final_node_labels = agglomerativeClustering.result()
    tock = time.time()
    pass


else:
    neg_weights = edge_costs < 0.
    p = 1
    exp_costs = np.abs(edge_costs)**p
    exp_costs[neg_weights] *= -1
    mc_obj = rag.MulticutObjective(graph=rag, weights=exp_costs)

    tick = time.time()
    log_visitor = mc_obj.loggingVisitor(verbose=False, timeLimitSolver=np.inf, timeLimitTotal=np.inf)

    # solverFactory = mc_obj.multicutIlpFactory()
    # solver = solverFactory.create(mc_obj)
    # final_node_labels = solver.optimize(visitor=log_visitor)


    # 1. Initialize a warm-up solver and run optimization
    solverFactory = mc_obj.greedyAdditiveFactory()
    solver = solverFactory.create(mc_obj)
    node_labels = solver.optimize(visitor=log_visitor)
    # 2. Use a second better warm-up solver to get a better solution:
    log_visitor = mc_obj.loggingVisitor(verbose=True, timeLimitSolver=np.inf, timeLimitTotal=np.inf)
    solverFactory = mc_obj.kernighanLinFactory()
    solver = solverFactory.create(mc_obj)
    new_node_labels = solver.optimize(visitor=log_visitor, nodeLabels=node_labels)
    # final_node_labels = new_node_labels

    # 4. Run the funsionMuves solver
    mc_obj.watershedProposals(sigma=1.0, seedFraction=0.0)
    pgen = mc_obj.watershedCcProposals(sigma=1.0, numberOfSeeds=0.1)
    # pgen = mc_obj.greedyAdditiveProposals(sigma=1.5, weightStopCond=0.0, nodeNumStopCond=-1.0)
    solverFactory = mc_obj.ccFusionMoveBasedFactory(proposalGenerator=pgen, numberOfIterations=1,
                                                            stopIfNoImprovement=1, numberOfThreads=1)
    solver = solverFactory.create(mc_obj)
    final_node_labels = solver.optimize(visitor=log_visitor, nodeLabels=new_node_labels)


    # results = np.stack([log_visitor.iterations(), log_visitor.energies(), log_visitor.runtimes()])
    # vigra.writeHDF5(results, "/export/home/abailoni/supervised_projs/MWS_vs_MC/results/fusionMove.h5", 'data')

    tock = time.time()


    ncols, nrows = 1, 1
    f, ax = plt.subplots(ncols=ncols, nrows=nrows)

    ax.semilogy(log_visitor.runtimes(), -log_visitor.energies())

    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.tight_layout()
    save_path = "/export/home/abailoni/supervised_projs/MWS_vs_MC/plots"
    f.savefig(os.path.join(save_path, 'second_try.pdf'), format='pdf')

edge_labels = rag.nodesLabelsToEdgeLabels(final_node_labels)

vigra.writeHDF5(edge_labels, "/export/home/abailoni/supervised_projs/MWS_vs_MC/results/edgeLabelsMWS.h5", 'data')

energy = (edge_costs * edge_labels).sum()
print("Took {}s. Final energy: {}".format(tock-tick, energy))


import long_range_hc.trainers.learnedHC.visualization as HCvis

final_segm = np.squeeze(map_features_to_label_array(segm, np.expand_dims(final_node_labels, axis=-1)))

ncols, nrows = 2, 1
f, ax = plt.subplots(ncols=ncols, nrows=nrows)

HCvis.plot_segm(ax[0], segm, z_slice=5, background=raw)
HCvis.plot_segm(ax[1], final_segm, z_slice=5, background=raw)


# plt.subplots_adjust(wspace=0, hspace=0)
# plt.tight_layout()
save_path = "/export/home/abailoni/supervised_projs/MWS_vs_MC/plots"
f.savefig(os.path.join(save_path, 'MWS.pdf'), format='pdf')
