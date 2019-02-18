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
import vigra
import nifty as nf
import nifty.graph.agglo as nagglo
import numpy as np
import os

from skimage import io
import argparse
import time
import yaml
import json

import segmfriends.vis as vis_utils
from skunkworks.metrics.cremi_score import cremi_score
from segmfriends.utils.config_utils import adapt_configs_to_model, recursive_dict_update
from segmfriends.utils import yaml2dict
from segmfriends.io.load import parse_offsets
from segmfriends.algorithms.WS.WS_growing import SizeThreshAndGrowWithWS

if __name__ == '__main__':
    # -----------------
    # Load data:
    # -----------------
    root_path = "/export/home/abailoni/supervised_projs/divisiveMWS"
    # dataset_path = os.path.join(root_path, "cremi-dataset-crop")
    dataset_path = "/net/hciserver03/storage/abailoni/datasets/ISBI/"
    plots_path = os.path.join("/net/hciserver03/storage/abailoni/greedy_edge_contr/plots")
    save_path = os.path.join(root_path, "outputs")

    # Import data:
    inverted_affinities = vigra.readHDF5(
        os.path.join(dataset_path, "isbi_results_MWS/isbi_train_offsetsV4_3d_meantda_damws2deval_final.h5"), 'data')
    raw = io.imread(os.path.join(dataset_path, "train-volume.tif"))
    raw = np.array(raw)
    gt = vigra.readHDF5(os.path.join(dataset_path, "gt_mc3d.h5"), 'data')
    gt_2d = vigra.readHDF5(os.path.join(dataset_path, "gt_cleaned.h5"), 'data')
    gt_2d = np.transpose(gt_2d, (2, 1, 0))

    result_file = os.path.join('/home/abailoni_local/', 'generalized_GED_comparison.json')
    if os.path.exists(result_file):
        with open(result_file, 'rb') as f:
            result_dict = json.load(f)
    else:
        raise FileNotFoundError()





    dict_per_ID = {}

    for agglo_type in result_dict:
        for non_link in result_dict[agglo_type]:
            sub_dict = result_dict[agglo_type][non_link]
            for edge_prob in sub_dict:
                for ID in sub_dict[edge_prob]:
                    dict_per_ID[ID] = sub_dict[edge_prob][ID]
                    dict_per_ID[ID]['edge_prob'] = float(edge_prob)
                    dict_per_ID[ID]['non_link'] = bool(non_link)
                    dict_per_ID[ID]['agglo_type'] = agglo_type


    converted_dict = {}
    root_dir = '/home/abailoni_local/GEC_comparison_kept'
    for subdir, dirs, files in os.walk(root_dir):
        for i, filename in enumerate(files):
            specs = filename.split('_')
            ID, aggl_type, prob = specs[0], specs[1], float(specs[2])
            if ID not in dict_per_ID:
                print(ID, " not found!")
                continue

            non_link = dict_per_ID[ID]['non_link']

            converted_dict[ID] = dict_per_ID[ID]
            segm = vigra.readHDF5(os.path.join(root_dir, filename), 'segm')

            # Compute 2D scores:
            segm_2D = np.empty_like(segm)
            max_label = 0
            for z in range(segm.shape[0]):
                segm_2D[z] = segm[z] + max_label
                max_label += segm[z].max() + 1

            segm_2D = vigra.analysis.labelVolume(segm_2D)
            evals = cremi_score(gt_2d, segm_2D, border_threshold=None, return_all_scores=True)
            print("2D:", evals)
            converted_dict[ID]["scores_2D"] = evals

            # Get rid of tiny segments:
            offset_file = 'offsets_MWS.json'
            offset_file = os.path.join(
                '/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation/experiments/postprocessing/cremi/offsets/',
                offset_file)
            offsets = parse_offsets(offset_file)
            configs = {'models': yaml2dict('./experiments/models_config.yml'),
                       'postproc': yaml2dict('./experiments/post_proc_config.yml')}
            # configs = adapt_configs_to_model([aggl_type], debug=True, **configs)
            postproc_config = configs['postproc']
            grow = SizeThreshAndGrowWithWS(postproc_config['thresh_segm_size'],
                                           offsets,
                                           hmap_kwargs=postproc_config['prob_map_kwargs'],
                                           apply_WS_growing=True, )
            segm_WS = grow(1 - inverted_affinities, segm_2D)
            evals = cremi_score(gt_2d, segm_WS, border_threshold=None, return_all_scores=True)
            print("2D and WS:", evals)
            converted_dict[ID]["scores_2D_WS"] = evals

            if i % 30 == 0:
                new_result_file = os.path.join('/home/abailoni_local/', 'generalized_GED_comparison_2D.json')
                with open(new_result_file, 'w') as f:
                    json.dump(converted_dict, f, indent=4, sort_keys=True)



