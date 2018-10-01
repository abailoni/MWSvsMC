import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import nifty.graph.rag as nrag
import vigra
import numpy as np
import os


from compareMCandMWS import utils as utils
from compareMCandMWS.multicut_solvers import solve_multicut



if __name__ == '__main__':
    root_path = "/export/home/abailoni/supervised_projs/MWS_vs_MC"
    dataset_path = os.path.join(root_path, "cremi-dataset-crop")
    plots_path = os.path.join(root_path, "plots")
    save_path = os.path.join(root_path, "outputs")

    # Import data:
    uv_IDs  = vigra.readHDF5(os.path.join(dataset_path, "edge_data.h5"), 'uv_IDs')
    edge_costs  = vigra.readHDF5(os.path.join(dataset_path, "edge_data.h5"), 'edge_weights')
    segm  = vigra.readHDF5(os.path.join(dataset_path, "segmentation.h5"), 'data')
    raw  = vigra.readHDF5(os.path.join(dataset_path, "raw.h5"), 'data')
    rag = nrag.gridRag(segm.astype('uint32'))

    # Solve multicut problem:
    outputs = solve_multicut(rag, edge_costs, p=1, solver_type='HC')
    energy, final_node_labels, final_edge_labels, log_visitor, runtime = outputs
    print("Took {}s. Final energy: {}".format(runtime, energy))

    # Plot 1: energy vs runtime
    ncols, nrows = 1, 1
    f, ax = plt.subplots(ncols=ncols, nrows=nrows)
    ax.semilogy(log_visitor.runtimes(), -log_visitor.energies())
    f.savefig(os.path.join(plots_path, 'energy-runtime.pdf'), format='pdf')

    # Plot 2: initial and final segmentation (take a 2D slice of the volume)
    final_segm = np.squeeze(utils.map_features_to_label_array(segm, np.expand_dims(final_node_labels, axis=-1)))
    ncols, nrows = 2, 1
    f, ax = plt.subplots(ncols=ncols, nrows=nrows)
    utils.plot_segm(ax[0], segm, z_slice=5, background=raw)
    utils.plot_segm(ax[1], final_segm, z_slice=5, background=raw)
    f.savefig(os.path.join(plots_path, 'final_segm.pdf'), format='pdf')

    # Save some data:
    vigra.writeHDF5(final_edge_labels, os.path.join(save_path, "finalEdgeLabels.h5"), 'data')

