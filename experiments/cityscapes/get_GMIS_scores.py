import sys

sys.path += ["/home/abailoni_local/hci_home/python_libraries/nifty/python",
"/home/abailoni_local/hci_home/python_libraries/cremi_python",
"/home/abailoni_local/hci_home/python_libraries/affogato/python",
"/home/abailoni_local/hci_home/pyCharm_projects/inferno",
"/home/abailoni_local/hci_home/pyCharm_projects/MWSvsMC",
"/home/abailoni_local/hci_home/pyCharm_projects/constrained_mst",
"/home/abailoni_local/hci_home/pyCharm_projects/neuro-skunkworks",
"/home/abailoni_local/hci_home/pyCharm_projects/segmfriends",
"/home/abailoni_local/hci_home/pyCharm_projects/hc_segmentation",
"/home/abailoni_local/hci_home/pyCharm_projects/neurofire",]

sys.path += ["/home/abailoni/hci_home/python_libraries/nifty/python",
"/home/abailoni/hci_home/python_libraries/cremi_python",
"/home/abailoni/hci_home/python_libraries/affogato/python",
"/home/abailoni/hci_home/pyCharm_projects/inferno",
"/home/abailoni/hci_home/pyCharm_projects/MWSvsMC",
"/home/abailoni/hci_home/pyCharm_projects/constrained_mst",
"/home/abailoni/hci_home/pyCharm_projects/neuro-skunkworks",
"/home/abailoni/hci_home/pyCharm_projects/segmfriends",
"/home/abailoni/hci_home/pyCharm_projects/hc_segmentation",
"/home/abailoni/hci_home/pyCharm_projects/neurofire",]


sys.path += [
"/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation",
"/net/hciserver03/storage/abailoni/pyCharm_projects/MWSvsMC",
"/net/hciserver03/storage/abailoni/python_libraries/affogato/python",
]


from segmfriends.io.save import get_hci_home_path, get_trendytukan_drive_path
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
import os

import segmfriends.vis as vis_utils
from segmfriends.utils.config_utils import adapt_configs_to_model, recursive_dict_update, return_recursive_key_in_dict

from long_range_compare.load_datasets import CREMI_crop_slices, CREMI_sub_crops_slices

if __name__ == '__main__':

    # Create dictionary:
    results_collected = {}

    results_dir = os.path.join(get_trendytukan_drive_path(), 'datasets/cityscape/data/gtFine_trainvaltest/evaluationResults/eval_out')

    result_matrix = []
    all_agglo_type = [' ']



    for subdir, dirs, files in os.walk(results_dir):
        for i, agglo_type in enumerate(dirs):
            json_file = os.path.join(subdir, agglo_type, 'resultInstanceLevelSemanticLabeling.json')
            if not os.path.exists(json_file):
                continue

            if 'clean' not in agglo_type and 'ORIG' not in agglo_type:
                continue
            with open(json_file, 'rb') as f:
                result_dict = json.load(f)

            scores = [agglo_type.replace('_', ' ')]
            scores.append('{:.4f}'.format(result_dict['averages']['allAp']))
            scores.append('{:.4f}'.format(result_dict['averages']['allAp50%']))

            # for cls in result_dict['instLabels']:
            #     scores.append( '{:.2f}'.format(result_dict['averages']['classes'][cls]['ap']))

            result_matrix.append(np.array(scores))

            labels = result_dict['instLabels']

    # labels = [' ', 'Overall mAP'] + labels
    labels = [' ', 'AP', 'AP50\% ']


    result_matrix = [labels] + result_matrix

    ndarray = np.array(result_matrix)

    np.savetxt(os.path.join(results_dir, "GMIS.csv"), ndarray, delimiter=' & ', fmt='%s', newline=' \\\\\n')


