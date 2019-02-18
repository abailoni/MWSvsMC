import sys

sys.path += ["/home/abailoni_local/hci_home/python_libraries/nifty/python",
"/home/abailoni_local/hci_home/python_libraries/cremi_python",
"/home/abailoni_local/hci_home/python_libraries/affogato/python",
"/home/abailoni_local/hci_home/pyCharm_projects/inferno",
"/home/abailoni_local/hci_home/pyCharm_projects/constrained_mst",
"/home/abailoni_local/hci_home/pyCharm_projects/neuro-skunkworks",
"/home/abailoni_local/hci_home/pyCharm_projects/segmfriends",
"/home/abailoni_local/hci_home/pyCharm_projects/hc_segmentation",
"/home/abailoni_local/hci_home/pyCharm_projects/neurofire",]

sys.path += ["/home/abailoni/hci_home/python_libraries/nifty/python",
"/home/abailoni/hci_home/python_libraries/cremi_python",
"/home/abailoni/hci_home/python_libraries/affogato/python",
"/home/abailoni/hci_home/pyCharm_projects/inferno",
"/home/abailoni/hci_home/pyCharm_projects/constrained_mst",
"/home/abailoni/hci_home/pyCharm_projects/neuro-skunkworks",
"/home/abailoni/hci_home/pyCharm_projects/segmfriends",
"/home/abailoni/hci_home/pyCharm_projects/hc_segmentation",
"/home/abailoni/hci_home/pyCharm_projects/neurofire",]


sys.path += [
"/net/hciserver03/storage/abailoni/pyCharm_projects/hc_segmentation",
"/net/hciserver03/storage/abailoni/python_libraries/affogato/python",
]

from segmfriends.io.save import get_hci_home_path
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
from segmfriends.utils.config_utils import adapt_configs_to_model, recursive_dict_update
import h5py

if __name__ == '__main__':

    home_path = get_hci_home_path()
    all_model_dir = [
        "learnedHC/plain_unstruct/pureDICE_wholeTrainingSet",
        # "learnedHC/new_experiments/SOA_affinities",
        # "learnedHC/new_experiments/radial_offsets",
        # "learnedHC/new_experiments/mutexWS_offsets_noSideLoss",
        # "learnedHC/new_experiments/mutexWS_offsets",

        # "learnedHC/model_090_v2/unstrInitSegm_pureDICE",
        # "learnedHC/model_050_A_v3/pureDICE_wholeDtSet",
        # "learnedHC/splitSpCNN/pureDICE_v2/infer_v100k-HC050/sampleB",

    ]
    for model_dir in all_model_dir:
        root_dir = os.path.join(home_path, model_dir, "Predictions")
        for subdir, dirs, files in os.walk(root_dir):
            for i, filename in enumerate(files):
                if filename.endswith(".h5"):
                    if filename != "prediction_sampleA.h5":
                        continue
                    file_path = os.path.join(root_dir, filename)
                    new_file_path = os.path.join(root_dir, "float8_{}".format(filename))
                    with h5py.File(file_path, 'r') as f_from:
                        with h5py.File(new_file_path, 'w') as f_to:
                            for inner_path in f_from:
                                print("Reading '{}' from '{}'".format(inner_path, file_path))
                                data = f_from[inner_path][:]
                                assert data.max() <= 1.0
                                assert data.min() >= 0.0
                                print("Writing...")
                                # new_data = (data * 255.).astype('uint8')
                                f_to[inner_path] = (data).astype('float16')

                        # print(f["predictions/full_affs"].dtype)
                        # print(f["raw_old"].shape)
                        # print(f["segmentations/groundtruth_fixed_OLD"].shape)
                        # raw = f["raw_old"][crop_slice[1:]].astype('float32') / 255.





