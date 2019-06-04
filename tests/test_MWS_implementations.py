import sys

sys.path += ["/home/abailoni_local/hci_home/python_libraries/nifty/python",
"/home/abailoni_local/hci_home/python_libraries/affogato/python",
"/net/hciserver03/storage/abailoni/python_libraries/affogato/python",
"/home/abailoni_local/hci_home/python_libraries/cremi_python",
"/home/abailoni_local/hci_home/pyCharm_projects/inferno",
"/home/abailoni_local/hci_home/pyCharm_projects/constrained_mst",
"/home/abailoni_local/hci_home/pyCharm_projects/neuro-skunkworks",
"/home/abailoni_local/hci_home/pyCharm_projects/segmfriends",
"/home/abailoni_local/hci_home/pyCharm_projects/hc_segmentation",
"/home/abailoni_local/hci_home/pyCharm_projects/neurofire",]


import matplotlib
matplotlib.use('Agg')
from h5py import highlevel
import mutex_watershed as mws
from affogato.segmentation import compute_mws_segmentation
import vigra
import os
from skimage import io
import numpy as np
import nifty
import time


from skunkworks.metrics.cremi_score import cremi_score

def run_mws(affinities,
            offsets, stride,
            seperating_channel=2,
            invert_dam_channels=True,
            bias_cut=0.,
            randomize_bounds=True):
    assert len(affinities) == len(offsets), "%s, %i" % (str(affinities.shape), len(offsets))
    affinities_ = np.require(affinities.copy(), requirements='C')
    if invert_dam_channels:
        affinities_[seperating_channel:] *= -1
        affinities_[seperating_channel:] += 1
    affinities_[:seperating_channel] += bias_cut
    sorted_edges = np.argsort(affinities_.ravel())
    # run the mst watershed
    vol_shape = affinities_.shape[1:]
    mst = mws.MutexWatershed(np.array(vol_shape),
                             offsets,
                             seperating_channel,
                             stride)
    if randomize_bounds:
        mst.compute_randomized_bounds()
    mst.repulsive_ucc_mst_cut(sorted_edges, 0)
    segmentation = mst.get_flat_label_image().reshape(vol_shape)
    return segmentation


if __name__ == '__main__':

    # -----------------
    # Load kwargs:
    # -----------------
    root_path = "/export/home/abailoni/supervised_projs/divisiveMWS"
    # dataset_path = os.path.join(root_path, "cremi-dataset-crop")
    dataset_path = "/net/hciserver03/storage/abailoni/datasets/ISBI/"
    plots_path = os.path.join("/net/hciserver03/storage/abailoni/greedy_edge_contr/plots")
    save_path = os.path.join(root_path, "outputs")

    # Import kwargs:
    affinities  = vigra.readHDF5(os.path.join(dataset_path, "isbi_results_MWS/isbi_train_offsetsV4_3d_meantda_damws2deval_final.h5"), 'kwargs')
    # raw = io.imread(os.path.join(dataset_path, "train-volume.tif"))
    # raw = np.array(raw)
    # gt = vigra.readHDF5(os.path.join(dataset_path, "gt_mc3d.h5"), 'kwargs')

    # # If this volume is too big, take a crop of it:
    # crop_slice = (slice(None), slice(None, 1), slice(None, 200), slice(None, 200))
    # affinities = affinities[crop_slice]
    # raw = raw[crop_slice[1:]]
    # gt = gt[crop_slice[1:]]




    # # Build the graph:
    # volume_shape = raw.shape
    # print(volume_shape)

    # offset_file = 'offsets_MWS.json'
    # offset_file = os.path.join('/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/experiments/postprocessing/cremi/offsets/', offset_file)
    # offsets = parse_offsets(offset_file)
  #   offsets = np.array([[-1, 0, 0],
  # [0, -1, 0],
  # [0, 0, -1],
  # [-1, -1, -1],
  # [-1, 1, 1],
  # [-1, -1, 1],
  # [-1, 1, -1],
  # [0, -9, 0],
  # [0, 0, -9],
  # [0, -9, -9],
  # [0, 9, -9],
  # [0, -9, -4],
  # [0, -4, -9],
  # [0, 4, -9],
  # [0, 9, -4],
  # [0, -27, 0],
  # [0, 0, -27]])

    offsets = np.array([[-1, 0, 0],
     [0, -1, 0],
     [0, 0, -1],
     [-2, 0, 0],
     [-2, 1, 1],
     [-2, -1, 1],
     [-2, 1, -1],
     [0, -2, 0],
     [0, 0, -2],
     [0, -2, -2],
     [0, 2, -2],
     [0, -9, -4],
     [0, -4, -9],
     [0, 4, -9],
     [0, 9, -4],
     [0, -27, 0],
     [0, 0, -27]])


    # Inverted affinities: 0 means merge, 1 means split
    affinities = np.random.uniform(0., 1., (17, 1, 2, 3))
    affinities[:3] = 0.99 #local
    affinities[3:] = 0.9 #lifted

    affinities[:3] -= np.abs(np.random.normal(scale=0.001, size=(3, 1, 2, 3)))
    affinities[3:] += np.abs(np.random.normal(scale=0.001, size=(14, 1, 2, 3)))

    tick = time.time()
    segm = run_mws(affinities, offsets, [1,1,1], seperating_channel=3, randomize_bounds=False)
    print(segm)
    print("Took ", time.time() - tick)

    file_path_segm = os.path.join('/home/abailoni_local/', "ISBI_results_new_MWS.h5")
    # vigra.writeHDF5(segm.astype('uint32'), file_path_segm, 'segm')

    file_path = os.path.join('/home/abailoni_local/', "noise.h5")
    vigra.writeHDF5(affinities, file_path, 'kwargs')


    # Constantin implementation:
    tick = time.time()
    labels = compute_mws_segmentation(1 - affinities, offsets, 3,
                                  randomize_strides=False,
                                 algorithm='kruskal')
    print("Took ", time.time() - tick)

  #   labels = compute_mws_segmentation(np.random.uniform(size=(3,1,2,2)), np.array([[-1, 0, 0],
  # [0, -1, 0],
  # [0, 0, -1]]),
  #                                     3,
  #                                     randomize_strides=False,
  #                                     algorithm='kruskal')

    print(labels)
    vigra.writeHDF5(labels.astype('uint32'), file_path_segm, 'segm_dMWS')

    # # Steffen:
    # configs = {'models': yaml2dict('./experiments/models_config.yml'),
    #            'postproc': yaml2dict('./experiments/post_proc_config.yml')}
    # configs = adapt_configs_to_model(['MWS'], debug=True, **configs)
    # postproc_config = configs['postproc']
    # segm_MWS = get_segmentation(affinities, offsets, postproc_config)
    #
    evals = cremi_score(labels+1, segm+1, border_threshold=None, return_all_scores=True)
    print(evals)

